# KVCM × PACE CopyGA 异步化设计

KVCM 以进程内异步任务编排 CopyGA，完成后检查源状态并更新目标 Location，物理 allocation 的生命周期安全依赖 PACE 的 Copy/Delete 互斥契约。Copy 任务不持久化，不在切主或重启后恢复；普通 Location 元数据仍按所选元数据后端的语义存储。

| 项目 | 内容 |
|---|---|
| 默认模式 | `MIGRATION_COPY_SYNC` |
| 异步模式 | `MIGRATION_COPY_ASYNC_REQUIRED`，仅接受显式支持异步的 backend |
| KVCM 职责 | 并发/字节准入、异步提交与完成、源状态检查与目标发布、失败后的单次清理 |
| PACE 安全前提 | Copy 与源/目标 Delete 在 allocation 生命周期上互斥；busy、失败或未确认不能报告删除成功 |

## 1. 目标与设计取舍

KVCache 存在 68/136 MiB 等较大 block；当 Copy RT 达到 200–300 ms 时，同步等待会长时间占用共享 Executor worker 和 DataStorageManager 锁。异步化的目标是释放这些编排资源，为提高 Copy 并发提供条件，物理搬运耗时仍由存储后端决定。

KVCM 不持久化 Copy Guard、operation 阶段、源信息或远端 task IDs，也不通过持久化 pin 阻止源逐出。源可进入普通回收流程，PACE 在真实 I/O 仍访问 allocation 时拒绝物理删除；Copy 完成时，KVCM 检查源的 ID、创建时间和可读状态，再单独更新目标。源检查与目标发布之间的并发变化仍遵循第 4 节的边界。这一分工允许切主或重启时放弃在途迁移，代价是可能丢失已完成但尚未发布的结果，且不提供远端失联任务的自动恢复。

普通 write/read、Provider 执行队列重构、全局 allocation epoch、跨进程任务 exactly-once 和所有 backend 的统一保护改造不属于本设计。同步 Copy 使用同步后端接口及 Storage timeout，MARK 继续引导后续写入；同步与异步 Copy 共用源状态检查、目标发布和元数据 Sync 收尾，因此收尾延迟包含元数据操作。

## 2. 必需的后端安全契约

`SupportsAsyncCopy()` 是 backend 的显式能力声明。启用原生异步不仅需要 submit/query 接口，还要求 PACE 满足以下 allocation 安全契约；这些要求是部署前提，不能视为已经部署验证的能力。

1. Provider 绑定源 allocation 后，源 lease 必须持有到最后一次源访问结束。其间 Delete 返回 busy，并保留 allocation 及映射，不能提前回收或复用给其他 GA。
2. 目标搬运和 commit 使用同样的互斥机制：Copy 先获得 lease 时 Delete 返回 busy；Delete 先关闭访问入口时，迟到的 Copy 不能重新写入或重建已删除目标。
3. HTTP timeout、cancel ACK、pending timeout 和节点注销成功均不等于真实 I/O drain，不能仅凭这些事件释放底层 owner。
4. 明确 GA 列表中存在 busy、失败或未确认项时，删除链必须向 KVCM 返回真实失败，不能因路由缺失或后台清理已提交就报告整体 success。
5. PACE 在 Delete 失败后保留未释放 allocation 的业务引用和定位信息；只有确认对象不存在时，才能以不存在为由返回幂等成功。

该契约的源保护范围从 Provider 绑定 allocation 开始。Meta 接受请求到 Provider 绑定之前，在途请求仍可能遇到源 GA 同址复用；KVCM 发布前的源检查不构成 allocation generation 协议。这个边界不保证 PACE 从未读到复用后的数据，也不覆盖绕过 KVCM 元数据直接 Free 的端到端身份安全。

无响应且无法证明 drain 时，PACE 可能长期保留 pending 与 allocation。这是安全契约允许的异常容量代价，不能用固定超时强制释放；KVCM 不持久化 Copy 任务也不意味着底层可以无条件 Free。

## 3. 运行流程与异步接口

```text
Group 并发/字节准入，登记进程内任务
  → 分配目标，写普通 CLS_WRITING Location
  → CopyAsync 本地接收，返回提交结果
  → backend 协调器 POST → 内存保存 task IDs → 批量 query / cancel
  → 唯一完成回调 → Executor future → MigrationManager
      成功：检查源 ID、create_time 和 SERVING 状态，再单独 CAS 目标为 SERVING
      失败/取消/未知：按目标 Location ID 尝试一次清理；Delete 返回失败仍尝试 CAD，以失败结束请求
```

`CopyAsync` 分为本地接收结果和最终 completion 两个阶段。本地接收不等待远端 Copy 完成；接收成功后，backend 负责恰好一次完成回调。operation 和 task IDs 仅用于进程内跟踪，不写入 CacheLocation，也没有远端提交持久化回调或 `ResumeAsyncCopy` 接口。

Reclaimer 和 Admin 分发路径先准备源信息，再进入批量准入。`ASYNC_REQUIRED` 请求必须提供非空 `src_specs`、`instance_group_name` 和非零字节额度，缺少这些参数时返回 `EC_BADARGS`。直接调用 `MigrationManager::Submit` 的异步分支也遵守此约定；只有同步分支支持通过 `PrepareCopyTask` 补读省略的源信息。

PACE adapter 的本地接收在发送 POST 前完成。本地明确拒绝且没有远端副作用时可直接进入清理；POST 响应丢失或不完整时，不能重发为新的 Copy，而应报告 unknown，再由普通 Delete 请求后端判断是否仍有活跃 I/O。通用接口中的 `acceptance_unknown` 用于表达无法确认接收结果的 backend，不构成安全完成证明。

异步路径的 `DataStorageManager` 在查找 backend 和本地交接时短暂持锁，随后通过完成回调捕获的 `shared_ptr` 保持 backend 存活；远端请求、轮询和完成等待不占用 manager 锁。协调器对单个 operation 的 task IDs 批量查询，并按请求大小分片，不跨 operation 合并查询；协调器需保持有界准入、轮询退避、分阶段 HTTP 预算、取消意图和唯一完成回调。

`ASYNC_REQUIRED` 遇到不支持异步的 backend 直接拒绝，不回退到同步 Copy。完成结果的整批 `status` 必须为 `EC_OK`，item 数量必须与请求一致，且每个 item 都满足 `terminal && safe_to_reuse_dst && outcome == success` 才能报告 Copy 成功；`AsyncCopyItemResult::error` 仅用于诊断。PACE adapter 还需核验 drained 与 server safe，短返回、缺字段、not_found 或 timeout 都不能冒充成功。

异步 Copy 的提交统计与提交事件在本地接收成功后登记，并先于完成结果消费。提交前被拒绝的请求仍须清理目标、释放额度和移除活跃记录，但不计入已提交任务的成功、失败或取消终态，也不发送完成事件。

**结果未知不能发布成功，但可以发起由后端安全契约保护的 Delete。** 发起 Delete 不代表 allocation 已可复用，只有后端确认安全删除才能完成清理。

## 4. 元数据发布与删除

### 4.1 成功发布

Copy 提交时在内存中保存源 Location ID、`create_time`、目标 Location ID 及迁移参数。完成处理属于当前 Leader 的任务生命周期，并通过唯一认领与取消仲裁；取消先取得处理权时，即使远端 Copy 成功也不发布目标、不删除源，而是按目标 Location ID 提交清理。

源和目标必须属于同一个 Instance/block，KVCache 不跨 `instance_id` 复用。完成时先读取源，检查其 ID、`create_time` 和 `SERVING` 状态，再按目标 ID 执行独立的 `WRITING → SERVING` CAS。源检查或目标 CAS 失败时，丢弃结果并按目标 ID 提交清理。源检查与目标更新不是同一次原子操作，目标 CAS 也不比较完整快照；两步之间的源变化以及同 ID 对象换代的竞态不由该流程消除。

CAS 已生效但 Sync 未确认时，以发布确认失败结束迁移，保留源和已发布目标，不执行源删除或清标，并释放进程内任务和 Copy 额度。该失败不计为 backend Copy unknown。

Sync 遵循所选元数据后端的语义。memory-primary 模式下，普通 `SERVING` 更新的备份是 best-effort，Sync 不等于每条目标更新都已持久化，也不提供迁移结果跨重启恢复保证。成功收尾后按 retention 决定是否删除源，源删除按源 Location ID 提交普通删除链。

`DELETE_SOURCE` 不在源删除准入时重新检查目标。目标发布后可以被独立删除，因此不保证删除源时目标仍然存在。

### 4.2 删除准入与失败处理

失败、取消和 unknown 均通过普通删除链尝试一次清理。GC、Reclaimer 和普通删除不依赖 Copy Guard 或反向源 pin，只保留进程内 active target 检查；`WRITING`、`DELETING` 是普通 Location 状态，不是持久化迁移任务。

1. 按 Location ID 读取当前状态，再 CAS 为 `DELETING`；调用方显式提供快照时，沿用普通删除准入的快照比较。
2. 确认元数据变更的 Sync 屏障后，遵守调用方原有的删除宽限时间，再调用物理 Delete，使元数据中的源失效先于源空间释放。Sync 失败时不调用物理 Delete。
3. Delete 返回 busy、超时等失败码或结果数量不符时，记录失败并继续执行该批次的 metadata CAD，最终返回失败。Delete 抛出异常时由 Executor 捕获并以失败结束，不继续执行后续步骤。
4. metadata CAD 按 Location ID 和 `DELETING` 状态删除记录，并按成功删除的元数据扣减账面用量；CAD 失败同样以失败结果结束，不自动重新删除物理数据或重做 CAD。

每个删除请求只执行一轮，失败后完成原 future。Executor 不将其重新放入定时队列，MigrationManager 也不保留失败清理队列或重新提交删除。物理删除阶段处理本次准入的 Location ID，不扩大到同 key 的其他 ID；后端返回错误、短返回或无法确认删除时按失败处理。

future 以失败结束后，上层可释放删除任务的 pending 记录和额度。物理删除失败但 CAD 成功时，Location 元数据和账面用量仍会移除，后端可能留下没有 Location 记录的物理残留；Sync、执行异常或 CAD 失败也可能留下 `DELETING`。账面用量扣减不证明物理空间已经释放，当前普通 GC、Reclaimer 不负责自动接管所有这类残留，本设计不提供后台持续重试或跨重启恢复保证。

### 4.3 CAS 竞争与重复删除边界

显式清理请求可以处理已经处于 `DELETING` 的 Location，但仍须执行 `DELETING → DELETING` CAS 并确认 Sync，不能只凭本地状态授权物理删除。CAS 竞争或备份准入失败需要返回失败，不能因另一路删除开始了就宣告本次清理完成。

普通删除准入保留调用方显式提供 `expected_location_values` 时的完整值条件 CAS；迁移清理不携带 Copy 创建时的完整快照。物理删除阶段和最终 CAD 不比较准入时的完整快照，因此不提供同 ID 换代对象的端到端隔离保证。

KVCM 不因本次清理失败自动重新提交 Delete。后续显式请求或独立任务仍可能重复删除同一 URI；模糊删除结果与底层 GA 身份复用依赖后端 Delete 契约，KVCM 不提供额外的 allocation generation 协议。

## 5. 并发、容量与候选准入

Group 在 SYNC 模式使用 `copy_max_concurrency`，默认 1；在 ASYNC_REQUIRED 模式使用 `async_copy_max_concurrency`，默认 1024。两个并发配置分别保存，均必须大于 0，缺失时各自恢复默认值；异步模式不会回退使用同步并发配置。异步模式还受 `copy_max_inflight_bytes` 限制，默认 1 TiB，显式配置时必须非零。准入先预留额度和进程内任务，再分配目标；源 URI size 缺失、为零、解析失败或求和溢出时拒绝。终态处理唯一认领，重复完成或取消不能重复归还额度。

切换模式只改变后续准入使用的并发上限，不修改另一模式的配置值，不取消或等待已有任务排空。同步与异步任务共用 Group 的总 active 计数；旧模式任务继续按提交时的模式完成，并占用新模式准入看到的名额。切换到较小上限时，已有任务不被中断，后续 Copy 等待总 active 低于上限后才有准入机会；同步任务不占异步字节额度。配置更新不是所有入口的原子切换点，排队中的 Prepare 会读取当前配置。

异步默认窗口以“待回收数据先完成 Copy，再回收源”作为容量规划目标。Copy 的 operation 与字节额度均按 Group 计账；Reclaimer 的删除批次上限按进程计账，待删除字节额度另按 Group × Storage 类型计账，二者分别准入。该规划目标不改变现有迁移触发和 retention 策略。

Copy inflight credit 表示 KVCM 正在跟踪的 Copy，不等于 PACE 的全部物理占用。unknown 收尾可释放 Copy credit，但清理失败残留及 PACE 未 drain 的 pending 仍可能占用容量。清理请求结束后不为失败残留持续持有清理任务。本设计没有独立的 quarantine operation/bytes 容量闸门；持续 busy/unknown 时，需要控制新 Copy 压力并检查未清理残留和 PACE 实际容量。

Copy 根据当前 Location 状态和目标覆盖情况选择源。源必须是指定 Storage 上的 `SERVING` 副本；若目标的 `SERVING/WRITING` Locations 已覆盖某个源的 specs，则继续检查其他仍需迁移的源副本。错误码不被当成源永久失效的证明，也不触发自动删除源。

纯 MARK 与 BOTH 的 MARK 回退仍须通过候选结构校验。已有任务、源缺失或目标已覆盖时不提交 MARK；启用 MARK 去重的路径还会跳过已有标记或去重读取失败的候选。BOTH 在没有 Copy 名额或 Copy 提交被拒时可回退为 MARK，已接受 Copy 的最终失败不会自动转为 MARK。

Reclaimer 的候选数量不等于实际准入数量。有 Group 硬限制时，分发层将候选交给 `BatchSubmit`，由它在锁内结合活跃并发和已占字节完成准入；批内 operation 名额只在两项准入都成功后扣减。大候选因字节额度不足被拒后，后续小候选仍有补位机会，例如剩余 1 个名额和 100 MiB 时，128 MiB 候选被拒后仍可接纳后面的 64 MiB 候选。不带 Group 硬限制的内部调用仍遵守分发层的软上限。

## 6. 停止、重启与管理操作

### 6.1 任务生命周期

切主或 Stop 关闭 Copy 准入，使旧 generation 的 Prepare Job 失效，停止完成结果消费，并丢弃内存 Copy 任务及 future。新 Leader 不接回旧任务或发布其迟到结果，也不扫描元数据恢复迁移；遗留 `WRITING` 由普通维护规则处理。

已提交的删除请求由 Executor 尝试一次；失败或 Executor 停止时以失败结束等待。没有进程内失败清理重试队列，也不存在跨重启的 `DELETING` 清理队列，不承诺遗留半成品自动收敛；普通维护能力需要按具体环境和元数据后端验证。

backend 关闭不能逐任务串行等待完整 Copy。唯一完成回调可以报告 unknown，但不能伪造 drained；KVCM 停止跟踪之后，allocation 仍必须受 PACE 安全契约保护。

### 6.2 在途任务与管理请求

异步 Copy 接入现有的实例生命周期流程。`RemoveInstance` 先关闭该实例的迁移准入并请求取消活跃 Copy，最多等待 5 秒；仍有活跃 Copy 时返回 `EC_TIMEOUT`。排空后，通过 Registry 记录实例移除状态，将 searcher 和 indexer 从运行时查找表中取出，阻止新请求使用旧实例，并等待已取得的引用排空。已准入的删除任务持有同一个 indexer 至本次尝试结束。

`TrimCache` 收集已提交删除批次的结果并执行 `SyncAll`；删除或 Sync 失败时返回错误，全部成功后才清扫残留。实例移除使用取出的 indexer 执行清扫，并在有界的剩余数据检查后调用 `Retire` 清除持久元数据，最后完成 Registry 移除。期间阻止同 ID 重新注册。

已建立的 Registry 移除状态在失败时保留；尚未退休的 indexer 恢复到管理器，供调用方显式再次执行 `RemoveInstance`。若持久元数据已退休而 Registry 收尾失败，再次调用只完成 Registry 收尾。返回非成功后需读取当前状态；实例生命周期管理不恢复 Copy 任务，也不为失败 Delete 建立自动重试队列。

## 7. 配置与超时

| 配置 | 默认 | 含义 |
|---|---:|---|
| copy_execution_mode | SYNC | 同步 / ASYNC_REQUIRED |
| copy_max_concurrency | 1 | SYNC 模式的 Group operation 上限 |
| async_copy_max_concurrency | 1024 | ASYNC_REQUIRED 模式的 Group operation 上限 |
| copy_max_inflight_bytes | 1099511627776（1 TiB） | 异步模式下必须非零 |
| copy_operation_deadline_ms | 600000 | 操作总跟踪预算 |
| copy_poll_initial_interval_ms | 20 | 首次查询间隔 |
| copy_poll_max_interval_ms | 1000 | 轮询退避上限，不保证实际轮询延迟 |
| copy_connect_timeout_ms | 1000 | 建连预算 |
| copy_submit_timeout_ms | 3000 | 单次 submit 总预算 |
| copy_query_timeout_ms | 3000 | 单次 query/cancel 总预算 |

```text
0 < poll_initial <= poll_max < operation_deadline
0 < connect <= submit, query
16 * max(submit, query) <= operation_deadline
```

默认值用于新建配置或未提供相应字段的配置；已有配置中显式保存的额度保持不变。

这些限制约束单次阻塞，不能保证大量任务共享一个协调器时的真实轮询周期。使用默认 1024 并发或调整并发配置时，需要观察协调器 queue wait、查询耗时及超时、实际 Copy 完成延迟，并评估远端 task 保留窗口；队列等待、轮询和正常 Copy P99 之和应留在 operation budget 内。

Storage timeout 控制 Create/Delete/health/同步 Copy，异步 HTTP 请求使用独立预算。降低全局 Storage timeout 不能代替异步预算配置；Provider 的等待超时同样不代表物理 I/O 停止，任何一层超时都不单独授权内存复用。

proto 的 `MigrationConfig` 将字段号 6、7 及名称 `copy_max_quarantine_operations`、`copy_max_quarantine_bytes` 声明为 reserved，不能复用为其他含义；它们不是可用配置项。
