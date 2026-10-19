# KVCM × PACE CopyGA 异步化设计

KVCM 以进程内异步任务编排 CopyGA，通过条件发布保证迁移结果只关联到有效的源和目标，物理 allocation 的生命周期安全依赖 PACE 的 Copy/Delete 互斥契约。Copy 任务不持久化，不在切主或重启后恢复；普通 Location 元数据仍按所选元数据后端的语义存储。

| 项目 | 内容 |
|---|---|
| 默认模式 | `MIGRATION_COPY_SYNC` |
| 异步模式 | `MIGRATION_COPY_ASYNC_REQUIRED`，仅接受显式支持异步的 backend |
| KVCM 职责 | 并发/字节准入、异步提交与完成、源和目标条件发布、失败清理与重试、失败源退避 |
| PACE 安全前提 | Copy 与源/目标 Delete 在 allocation 生命周期上互斥；busy、失败或未确认不能报告删除成功 |

## 1. 目标与设计取舍

KVCache 存在 68/136 MiB 等较大 block；当 Copy RT 达到 200–300 ms 时，同步等待会长时间占用共享 Executor worker 和 DataStorageManager 锁。异步化的目标是释放这些编排资源，为提高 Copy 并发提供条件，物理搬运耗时仍由存储后端决定。

KVCM 不持久化 Copy Guard、operation 阶段、源身份快照或远端 task IDs，也不通过持久化 pin 阻止源逐出。源可进入普通回收流程，PACE 在真实 I/O 仍访问 allocation 时拒绝物理删除；Copy 完成时，KVCM 再通过源和目标的条件更新拒绝过期结果。这一分工允许切主或重启时放弃在途迁移，代价是可能丢失已完成但尚未发布的结果，且不提供远端失联任务的自动恢复。

普通 write/read、Provider 执行队列重构、全局 allocation epoch、跨进程任务 exactly-once 和所有 backend 的统一保护改造不属于本设计。同步 Copy 使用同步后端接口及 Storage timeout，MARK 继续引导后续写入；同步与异步 Copy 共用源/目标条件发布和元数据 Sync 收尾，因此收尾延迟包含元数据操作。

## 2. 必需的后端安全契约

`SupportsAsyncCopy()` 是 backend 的显式能力声明。启用原生异步不仅需要 submit/query 接口，还要求 PACE 满足以下 allocation 安全契约；这些要求是部署前提，不能视为已经部署验证的能力。

1. Provider 绑定源 allocation 后，源 lease 必须持有到最后一次源访问结束。其间 Delete 返回 busy，并保留 allocation 及映射，不能提前回收或复用给其他 GA。
2. 目标搬运和 commit 使用同样的互斥机制：Copy 先获得 lease 时 Delete 返回 busy；Delete 先关闭访问入口时，迟到的 Copy 不能重新写入或重建已删除目标。
3. HTTP timeout、cancel ACK、pending timeout 和节点注销成功均不等于真实 I/O drain，不能仅凭这些事件释放底层 owner。
4. 明确 GA 列表中存在 busy、失败或未确认项时，删除链必须向 KVCM 返回失败或待重试，不能因路由缺失或后台清理已提交就报告整体 success。
5. Delete 失败后必须保留重试所需的业务引用；只有确认对象不存在时，才能以不存在为由返回幂等成功。

该契约的源保护范围从 Provider 绑定 allocation 开始。Meta 接受请求到 Provider 绑定之前，在途请求仍可能遇到源 GA 同址复用；KVCM 依赖第 4 节的条件发布拒绝过期结果。这个边界不保证 PACE 从未读到复用后的数据，也不覆盖绕过 KVCM 元数据直接 Free 的端到端身份安全。

无响应且无法证明 drain 时，PACE 可能长期保留 pending 与 allocation。这是安全契约允许的异常容量代价，不能用固定超时强制释放；KVCM 不持久化 Copy 任务也不意味着底层可以无条件 Free。

## 3. 运行流程与异步接口

```text
Group 并发/字节准入，登记进程内任务
  → 分配目标，写普通 CLS_WRITING Location
  → CopyAsync 本地接收，返回提交结果
  → backend 协调器 POST → 内存保存 task IDs → 批量 query / cancel
  → 唯一完成回调 → Executor future → MigrationManager
      成功：同一 block 条件更新核验 source + target，再发布 SERVING
      失败/取消/未知：精确删除本次目标，busy/失败保留 DELETING 并重试
```

`CopyAsync` 分为本地接收结果和最终 completion 两个阶段。本地接收不等待远端 Copy 完成；接收成功后，backend 负责恰好一次完成回调。operation 和 task IDs 仅用于进程内跟踪，不写入 CacheLocation，也没有远端提交持久化回调或 `ResumeAsyncCopy` 接口。

Reclaimer 和 Admin 分发路径先准备源信息，再进入批量准入。`ASYNC_REQUIRED` 请求必须提供非空 `src_specs`、`instance_group_name` 和非零字节额度，缺少这些参数时返回 `EC_BADARGS`。直接调用 `MigrationManager::Submit` 的异步分支也遵守此约定；只有同步分支支持通过 `PrepareCopyTask` 补读省略的源信息。

PACE adapter 的本地接收在发送 POST 前完成。本地明确拒绝且没有远端副作用时可直接进入清理；POST 响应丢失或不完整时，不能重发为新的 Copy，而应报告 unknown，再由普通 Delete 请求后端判断是否仍有活跃 I/O。通用接口中的 `acceptance_unknown` 用于表达无法确认接收结果的 backend，不构成安全完成证明。

异步路径的 `DataStorageManager` 在核对 backend 身份和完成本地接收时持 shared lock，使本地提交与 Storage 注销互斥；远端请求、轮询和完成等待不持该锁。任务用 `shared_ptr` 保存源、目标 backend 的进程内身份，保持对象存活不代表它仍注册或尚未关闭。协调器对单个 operation 的 task IDs 批量查询，并按请求大小分片，不跨 operation 合并查询；协调器需保持有界准入、轮询退避、分阶段 HTTP 预算、取消意图和唯一完成回调。

`ASYNC_REQUIRED` 遇到不支持异步的 backend 直接拒绝，不回退到同步 Copy。完成结果的整批 `status` 必须为 `EC_OK`，item 数量必须与请求一致，且每个 item 都满足 `terminal && safe_to_reuse_dst && outcome == success` 才能报告 Copy 成功；`AsyncCopyItemResult::error` 仅用于诊断。PACE adapter 还需核验 drained 与 server safe，短返回、缺字段、not_found 或 timeout 都不能冒充成功。

异步 Copy 的提交统计与提交事件在本地接收成功后登记，并先于完成结果消费。提交前被拒绝的请求仍须清理目标、释放额度和移除活跃记录，但不计入已提交任务的成功、失败或取消终态，也不发送完成事件。

**结果未知不能发布成功，但可以发起由后端安全契约保护的 Delete。** 发起 Delete 不代表 allocation 已可复用，只有后端确认安全删除才能完成清理。

## 4. 元数据发布与删除

### 4.1 成功发布

Copy 准入时绑定源和目标 backend 的进程内身份，并保存本次源和目标的 Location 身份及完整快照。完成处理属于当前 Leader 的任务生命周期，并通过唯一认领与取消仲裁；取消先取得处理权时，即使远端 Copy 成功也不发布目标、不删除源，而是清理本次目标。

源和目标必须属于同一个 Instance/block，KVCache 不跨 `instance_id` 复用。发布不能依靠先 Get(source)、再独立 Promote(target)，因为两步之间可能发生源删除。一次 block 级条件 RMW 同时校验：

- 源 Location ID、`create_time`、完整快照和 `SERVING` 状态仍匹配。
- 目标仍是本次创建的 `WRITING` Location，身份和完整值未变化。

全部条件满足才把目标改成 `SERVING`；任一条件不符则丢弃结果，并按本次目标快照清理，不触碰替换后的 Location。CAS 已生效但 Sync 暂未确认时，进程内任务保留完成处理权，只重试 Sync，不能因再次 CAS 不匹配而误删已发布目标。

发布时还在 `DataStorageManager` shared lock 内核对源、目标 backend 仍是本次绑定的对象，并执行条件更新，与注销所需的 unique lock 互斥。Sync 在该锁外等待；Sync 完成后，再次核对 backend 身份并在锁内提交源删除任务。任一 Storage 已注销或被同名新 backend 替换时，Copy 按失败收尾，不再发布目标或提交源删除；已发布的目标也不会被本次 `WRITING` 快照清理误删。

Sync 遵循所选元数据后端的语义。memory-primary 模式下，普通 `SERVING` 更新的备份是 best-effort，Sync 不等于每条目标更新都已持久化，也不提供迁移结果跨重启恢复保证。成功收尾后按 retention 决定是否删除源，源删除同样使用精确快照和普通删除链。

`DELETE_SOURCE` 在源首次转为 `DELETING` 的同一次 block 条件 RMW 中，校验本次已发布目标的完整 `SERVING` 快照。目标已被删除、进入删除状态或被替换时，保留源；这也覆盖目标发布后等待 Sync 的期间。源已成功进入 `DELETING` 后，其物理清理继续遵守精确身份和 Sync 屏障，不再要求目标持续存在，避免目标后续正常淘汰导致源清理永久挂起。该条件保证的是源开始失效时目标仍有效，不阻止之后对目标的独立删除。

### 4.2 删除准入与重试责任

失败、取消和 unknown 均使用普通删除链。GC、Reclaimer 和普通删除不依赖 Copy Guard 或反向源 pin，只保留进程内 active target 检查；`WRITING`、`DELETING` 是普通 Location 状态，不是持久化迁移任务。

1. 按精确 Location 快照将 `WRITING/SERVING` 改为 `DELETING`。
2. 确认元数据变更的 Sync 屏障后才调用物理 Delete，使元数据中的源失效先于源空间释放。
3. Delete busy 或失败时保留 `DELETING` 和进程内清理责任，不提前删除元数据。Executor 通过现有定时队列退避重试，等待不占 worker；原 future 保持 pending，使上层保留删除任务对应的 pending 记录和额度。
4. 重试重新读取原身份，允许本次对象已是 `DELETING`，但必须执行精确的 `DELETING → DELETING` CAS，重新确认备份准入后再 Sync。仅凭本地已有状态和空队列的 Sync，不能授权物理删除。
5. 物理删除确认后才执行精确的 metadata CAD；CAD 失败仍由同一任务保留重试责任。

这套重试用于普通 RemoveCache、Reclaimer、GC 和迁移清理。删除请求入队前绑定 backend 身份，Executor 只跟踪本次准入的精确 Location，不重新扩展到同 key 的新 Location，也不转用同名重建的 backend。Storage 注销后，任务在下一次执行时停止对该 Storage 的清理；同批次中其他 Storage 的清理继续执行。未确认物理删除的 Location 元数据保留，最终 future 以失败结束，并明确标记清理被放弃，不把注销当作删除成功。

任务结束或 Executor 停止时释放其进程内引用，停止以失败结束等待，不代表物理删除成功。无法建立重试责任的首次准入失败会返回错误；异步迁移清理另由 MigrationManager 的进程内队列保留上层责任，直到清理成功或 Executor 确认原 Storage 已注销并放弃相应清理。

### 4.3 CAS 竞争与重复删除边界

首次迁移精确清理也接受同一对象已经是 `DELETING` 的情况，不能把另一路删除尚未完成当成清理成功。如果在首次读取之后、CAS 之前，并发删除只将本地状态改为 `DELETING`，甚至其备份准入失败，CAS mismatch 也不能证明物理删除完成。

精确清理将整批原身份请求交给重试所有者重新读取，包括本批 CAS 成功项和竞争失败项，不能只保留成功子集后就放弃竞争项。再次读取确认原对象已不存在或已被不同身份替换，才可跳过；同 ID 的换代对象不能误删。

同一内存删除任务记录完整返回中已确认 `EC_OK/EC_NOENT` 的 URI。部分物理删除成功或后续 metadata CAD 失败时，不重复删除这些 URI；该记录与 GC 的“查询确认不存在”提示分开，不受 backend 的 GC 删除策略影响。短返回、异常和超时不构成成功证明。模糊删除结果、跨任务重复删除以及底层 GA 身份复用仍依赖后端 Delete 契约，KVCM 不提供额外的 allocation generation 协议。

## 5. 并发、容量与失败源退避

Group 的 `copy_max_concurrency` 限制活跃 Copy operation，默认 64；异步模式同时受 `copy_max_inflight_bytes` 限制，默认 32 GiB，显式配置时必须非零。准入先预留额度和进程内任务，再分配目标；源 URI size 缺失、为零、解析失败或求和溢出时拒绝。终态处理唯一认领，重复完成或取消不能重复归还额度。

Copy inflight credit 表示 KVCM 正在跟踪的 Copy，不等于 PACE 的全部物理占用。unknown 收尾可释放 Copy credit，但其删除重试及 PACE 未 drain 的 pending 仍可能占用容量，清理记录继续保存原 backend 身份和 Location 快照。本设计没有独立的 quarantine operation/bytes 容量闸门；持续 busy/unknown 时，需要控制新 Copy 压力并检查删除积压和 PACE 实际容量。

同步与异步 Copy 共用进程内失败源退避，身份键为：

```text
(instance_id, block_key, source_location_id, source_create_time, target_storage_name)
```

重复失败采用有上限的指数退避，优先选择未失败或失败次数较少的合格副本；所有可选源均在退避期时，不分配 Copy 目标、不提交 Copy。`create_time` 防止同 ID 旧副本的失败记录影响换代后的副本。错误码不被当成永久数据失效证明，也不触发自动删除源。

失败源退避只影响 Copy。纯 MARK 以及 BOTH 的 MARK 回退仍按独立的结构准入和去重规则处理：已有任务、源缺失、目标已覆盖、已有 MARK 或去重读取错误等条件仍需检查。

Reclaimer 的候选数量不等于实际准入数量。有 Group 硬限制时，分发层将候选交给 `BatchSubmit`，由它在锁内结合活跃并发和已占字节完成准入；批内 operation 名额只在两项准入都成功后扣减。大候选因字节额度不足被拒后，后续小候选仍有补位机会，例如剩余 1 个名额和 100 MiB 时，128 MiB 候选被拒后仍可接纳后面的 64 MiB 候选。不带 Group 硬限制的内部调用仍遵守分发层的软上限。

## 6. 停止、重启与管理操作

### 6.1 任务生命周期

切主或 Stop 关闭 Copy 准入，使旧 generation 的 Prepare Job 失效，停止完成结果消费，并丢弃内存 Copy 任务及 future。新 Leader 不接回旧任务或发布其迟到结果，也不扫描元数据恢复迁移；遗留 `WRITING` 由普通维护规则处理。

进程存活且相应组件运行时，Executor 负责已接管删除的失败重试，MigrationManager 清理队列保留上层责任。不存在跨重启的 `DELETING` 清理队列，也不承诺所有遗留半成品自动收敛；普通维护能力需要按具体环境和元数据后端验证。

backend 关闭不能逐任务串行等待完整 Copy。唯一完成回调可以报告 unknown，但不能伪造 drained；KVCM 停止跟踪之后，allocation 仍必须受 PACE 安全契约保护。

### 6.2 在途任务与管理请求

Storage 注销不以 Copy 完成或清理成功为前提。`DataStorageManager` 用现有读写锁串行化本地 Copy 接收、结果发布和注销；普通 Create/Delete 仍在调用 backend 期间持 shared lock。注销可能等待这些正在执行的调用及 backend 的 Close，但不等待远端 Copy completion 或清理 future。backend Close 的实际 I/O 安全仍由第 2 节的契约保证。

Monitor 定期核对活跃 Copy 绑定的 backend 身份，发现源或目标已注销时按失败收尾，无需等待远端结果；迟到结果不能恢复任务。Executor 在已有定时任务的下一次执行中结束相应清理责任，后台重试不会否决 Storage 移除。同名 Storage 重建也不会让旧任务重新获得发布或物理删除权限。清理记录用于保存进程内任务责任，不保证注销后残留数据自动清除。

`RemoveInstance` 先关闭该实例的迁移准入并请求取消活跃 Copy，最多等待 5 秒；仍有活跃 Copy 时返回 `EC_EXIST`。待重试的迁移清理和普通删除任务不单独阻止实例移除。

`TrimCache` 对每个已提交批次最多同步等待 5 秒，超过后返回 `EC_TIMEOUT`，不继续清扫残留，也不取消 Executor 的后台重试。该上限只约束单批等待，不是整个多批 Trim 的总时长上限；它避免持续 busy 使管理请求和切主请求排空永久等待。`RemoveInstance` 按先注销、后 Trim 的顺序执行；本进程确认注销成功后保留待清扫记录，失败或超时后再次调用可继续 Trim，已经接管的删除仍由 Executor 重试。显式 Trim 重试可能与已有删除任务重叠，跨任务重复删除仍遵循第 4.3 节的后端契约。待清扫记录在清扫成功、实例重新注册或主备清理时移除，不提供跨重启恢复。同一实例的 RegisterInstance 与 RemoveInstance 调用互斥，其他实例不等待它的 drain 或 Trim。返回非成功后仍需读取当前注册状态。

## 7. 配置与超时

| 配置 | 默认 | 含义 |
|---|---:|---|
| copy_execution_mode | SYNC | 同步 / ASYNC_REQUIRED |
| copy_max_concurrency | 64 | Group operation 上限 |
| copy_max_inflight_bytes | 34359738368（32 GiB） | 异步模式下必须非零 |
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

这些限制约束单次阻塞，不能保证大量任务共享一个协调器时的真实轮询周期。使用默认 64 并发或调整并发配置时，需要观察协调器 queue wait、查询耗时及超时、实际 Copy 完成延迟，并评估远端 task 保留窗口；队列等待、轮询和正常 Copy P99 之和应留在 operation budget 内。

Storage timeout 控制 Create/Delete/health/同步 Copy，异步 HTTP 请求使用独立预算。降低全局 Storage timeout 不能代替异步预算配置；Provider 的等待超时同样不代表物理 I/O 停止，任何一层超时都不单独授权内存复用。

proto 的 `MigrationConfig` 将字段号 6、7 及名称 `copy_max_quarantine_operations`、`copy_max_quarantine_bytes` 声明为 reserved，不能复用为其他含义；它们不是可用配置项。
