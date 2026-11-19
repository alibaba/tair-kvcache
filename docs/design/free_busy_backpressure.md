# KVCM 水位驱逐的 Free Busy 退避设计

设计评审稿，2026-11-18。水位驱逐按固定 pending 额度提交，额度过小会拖慢回收，过大则可能让繁忙后端的 Free 积压、超时。首版让 Meta 在接不下整批 Free 时立即告知 KVCM。Reclaimer 据此少接新任务，Executor 延后重试被拒的批次，压力解除后逐步恢复准入。

## 1. 本次 PR 的范围

只接入 **CacheReclaimer 水位驱逐发往 Tair mempool 的 Free**。KVCM 的新协议默认开启，由 `kvcm.cache_reclaimer.free_busy_enabled` 统一控制。默认发送协议头并预留重试内存，收到 Meta 明确的整批拒收后才收紧额度、延后重试。显式设为 `false` 时走原 Delete 和原 pending 准入。GC、RemoveCache、Trim、RemoveInstance 和分配/迁移回滚沿用各自的入口。其他入口占满后端时，首版只能减少水位驱逐贡献的负载。

Meta 的 busy 针对一次 HTTP 请求。每个 KVCM 进程按现有的**实例组 × 基础存储类型**记账，起始额度直接取已有 pending 配置。同组同类型的删除共用一份额度，即使目标属于不同 Provider 或不同 Tair 集群也会一起降速。访问同一 Meta 的不同组、DRAM 和 SSD 则分别调整。

pending 从 Executor 接纳任务算到任务收尾，包括读者宽限期、执行 Free 和删除元数据。收紧额度只影响新任务，已经接纳的任务继续执行。一次 Free 可合并多个 Location，因此实际 HTTP 并发和每秒删除量还取决于批大小、宽限时间和回包耗时。

## 2. 把反馈接入两处已有检查

| 现有配置 | 计量内容 | 默认硬上限 |
|---|---|---|
| `kvcm.cache_reclaimer.pending_location_limit_per_group_type` | 该实例组、该基础存储类型未收尾的 Location 数 | 20,000 个 |
| `kvcm.cache_reclaimer.pending_bytes_limit_per_group_type` | 这些 Location 对应的待回收数据量 | 1 TiB |

两项配置继续作为硬上限。首次收到明确 busy 时，Reclaimer 才为对应的实例组和 Tair 基础存储类型记录**当前有效额度**（Location 数、bytes）和冷却截止时间。实际占用直接读现有的 `pending_quota_by_group_type_`。

- **筛选候选。** `FilterLocIDImpl` 发现 LRU 候选时沿用已有的 `eligibility_only` 扫描。实际选取时先走原有硬上限检查，再对受退避影响的类型比较“当前 pending + 本轮已选 + 候选”和两项有效额度。冷却中或超额就跳过该候选。[候选筛选](../../kv_cache_manager/manager/cache_reclaimer.cc)
- **提交与记账。** `SubmitDelReq` 先复核原有组级、进程级硬上限，再用同一规则复核实际批量。软额度拒绝单独处理，`pending_limit_reject_count` 仍只统计原有硬上限。Executor 接纳后照常记账，原任务的 Future 结束时释放一次。重试沿用这份 pending 和原有回收量预测的到期时间（`credit_deadline`）。[现有 pending 记账](cache_reclaimer_async_delete.md)

### 额度怎样收紧和恢复

以下增减幅度是联调起点。直接复用原 DeleteHandler 的单调时钟准入时间 `submitted_at_`。每个受压的组和类型只多记最近一次调额时间，用它区分在新额度下接纳的任务与已经排队的旧任务。[原任务的准入时间](../../kv_cache_manager/manager/cache_reclaimer.h)

| 反馈 | Reclaimer 的处理 |
|---|---|
| 尚未收到 busy | 两项 pending 额度取原配置值。重试内存预算有余量时，按现有流程提交 |
| 本轮准入的任务首次收到明确 busy | 暂停该组、该类型的新准入。两项有效额度分别取“当前有效额度”和“实际 pending 占用”的较小值，再减半，向下取整，最低保留 1 个 Location 和 1 字节。本轮冷却取触发这次收紧的批次的退避间隔，从处理反馈时开始计时。旧任务的回包只影响它自己的重试 |
| 冷却结束 | 当前占用降到有效额度以内后再接新任务。已经接纳的批次沿用原 pending，退避到期后按第 4 节的重试并发上限执行 |
| 冷却后收到实际 Free 成功 | 用同一冷却时长作为恢复观察间隔。观察期内，本轮准入的任务有实际 Free 成功、没有本轮 busy，两项额度最多翻倍一次，最高回到配置值。每次调额更新轮次边界，旧成功不能在后续轮询中重复使用 |

例如首次 busy 时实际 pending 是 80 个 Location、4 GiB，有效额度收至 40 个、2 GiB。原已接纳的 80 个继续处理。即使其中有任务到宽限期结束才首次发 Free，也属于旧轮次。两项额度恢复、旧轮次任务也已收尾时，可清理这份组级状态。

为避免一个大 Location 永远超过临时的待删数据字节额度，冷却结束、该组类型的 pending 和本轮已选都为空时，可单独放行一个满足各项硬上限的 Location。筛选和提交复核共用这条规则，重试记录也要能放进第 4 节的进程内存预算。当前占用归零只清理记账，仍在生效的退避状态保留。

## 3. 只让水位驱逐识别 busy

**灰度顺序。** 混部期间显式关闭 KVCM 的协议开关。旧 follower 转发时可能把新 leader 的 429 写为 501 并保留原正文，默认本地执行的 Meta 不经过这次转发。所有可能服务、转发或接任 leader 的 Meta 升级完成后，逐实例核对实际额度，并通过实际路由确认完整回包。启用自动重试还需满足第 4 节的 GA 身份和时长前提。

Meta 在 `HandleDeleteGAs` 对带协议头的请求先校验完整 GA 列表。空列表或任一 GA 非法都返回参数错误，KVCM 发前逐 URI 校验路径完整解析为非零十进制 uint64，按十进制串和逗号组装 GA 列表，跳过空批。非法 spec 按 `(key, location_id, 原 URI)` 记录并保留所属 Location 的元数据，其他可独立完成的 Location 继续处理。一个 Location 有多个 spec 时，记录各 spec 的处理结果，避免过滤一项后把整个 Location 当作成功 CAD。

对合法列表，Meta 在调用 `FreeGAs` 前按正在处理的 Free 批数和 GA 数检查额度。单批能容纳、已有占用使本批加入后超限时，立即返回 `FREE_BUSY_NOT_ACCEPTED`，整批还未向任何 Provider 下发。若单批自身已超过 GA 硬上限，优先返回 HTTP 400 和 `{"protocol_version":1,"code":"FREE_BATCH_TOO_LARGE","status":"error","accepted":false,"rejection_scope":"whole_request","max_gas":1000}`，其中 1000 为示例。KVCM 校验 HTTP 状态和整份协议字段。这类错误直接按未发出失败收尾，保留受影响的 `DELETING`，记录实际项数和该 Meta 的上限供调整组批或配置。接纳的请求占用额度，处理结束时归还。所有显式列表 Free 都计入负载，只有带协议头的请求参与拒收。已下发的请求即使超时或部分失败，也按实际结果返回。

盘侧的 `pending_keys`、`pending_bytes` 若以后能汇总到 Meta，也可用于上述下发前判断，KVCM 仍只消费同一个 busy 信号。

Meta 可按下列格式返回 HTTP 429，其中 `retry_after_ms` 是 Meta 可选的等待建议，单位为毫秒。

```json
{
  "protocol_version": 1,
  "code": "FREE_BUSY_NOT_ACCEPTED",
  "status": "busy",
  "accepted": false,
  "rejection_scope": "whole_request",
  "retry_after_ms": 50
}
```

**入口与接口。** 由 `CacheReclaimer::SubmitDelReq` 给启用协议的任务加标记，随 `PrepareDeleteTask` 生成的实际任务传下去。GC 也用 `SubmitAsync` 和 `kReclaim`，任务类别不能识别来源。复用 `DataStorageManager` 已有的具体 Tair backend 依赖，只加水位驱逐专用的调用。开源 Tair 桩和内源实现对齐这个入口，内源为该次 HTTP 请求带上 `X-TairMempool-Free-Protocol: 1`，在错误归并前识别约定的 429、协议版本、业务码和整批未接纳保证。返回原删除结果、明确 busy 及可选等待时间，或单批过大及本机 GA 上限。公共 `DataStorageBackend::Delete`、共享错误码和其他后端继续按原接口工作。[现有 Tair 装配](../../kv_cache_manager/data_storage/data_storage_manager.cc) · [公共 Delete 接口](../../kv_cache_manager/data_storage/data_storage_backend.h)

**存储访问。** 首发和重试都在 `DataStorageManager` 现有删除读锁内查找并调用后端。退避等待期间释放锁，重试时在锁内核对仍是原后端实例，避免注销存储时的 `Close` 和 Delete 同时执行。[存储的删除与注销](../../kv_cache_manager/data_storage/data_storage_manager.cc)

**反馈时机。** Executor 在任务共享的线程安全状态中记录明确 busy、建议等待时间和实际 Free 成功。Reclaimer 的 `HandleDelRes` 在已有轮询中取走反馈，Future 就绪时，在移除 DeleteHandler 前取完最后的回包反馈。同一轮询按实例组和基础存储类型合并本轮准入任务的反馈，有 busy 优先收紧。跳过物理删除的任务不计入成功恢复。[Reclaimer 轮询与收尾](../../kv_cache_manager/manager/cache_reclaimer.cc)

## 4. 被拒的原批次如何重试

**重试的安全假设。** Meta 返回 `FREE_BUSY_NOT_ACCEPTED` 时，整批尚未向任何 Provider 下发。KVCM 的重试假定 GA 仍对应原分配。重读元数据后，主动删除仍可能抢先释放同一个 GA。并发删除、Provider 重启或地址复用时的重试安全性，以及允许重发的时长，由双方另行确认。

**保存和投递。** 首次删除沿用 CAS、Sync 和读者宽限期。某个 storage 整批被拒时，只保存该批原始 URI 文本和对应的 `(block_key, location_id)`。其他 storage 继续执行，已完成的 Location 及时 CAD（按条件删除元数据）。重试使用已有的 `SubmitRaw(delay)` 让出 worker，每批最多挂一次，连续 busy 复用同一份记录。原任务的 pending 在所有批次结束、Future 完成后释放一次。[现有延迟执行与删除收尾](../../kv_cache_manager/manager/schedule_plan_executor.cc)

**何时重发。** KVCM 从可选配置 `kvcm.cache_reclaimer.free_busy_initial_backoff_ms` 读取首次本地退避时间，默认 50 ms，有效范围为 1 至 5000 毫秒。同一批连续 busy 时，本地等待从配置值起逐次翻倍，加入抖动后最多 5 s。Meta 提供 `retry_after_ms` 时，实际等待取它与本地等待的较大值。触发调额的这次实际等待，同时决定第 2 节的冷却期和随后等长的观察期。

首版将允许启动重试的总时限取为 30 秒，从原批次首次准备 Free 开始计时。投递、回调和持有存储读锁准备下发时均检查是否到期。已下发的请求按实际回包收尾，联调时需结合后端请求耗时确认 GA 身份的有效窗口。

**重试删除的并发上限。** 每个 KVCM 进程的所有实例组和 Tair 存储共享一组名额，只限制水位驱逐收到明确 busy 后同时执行的重试批数。按 `SchedulePlanExecutor::GetWorkerCount()` 返回的实际 worker 数 N 计算，联调起点取 `max(1, floor(N / 2))`。2 个 worker 最多同时重试 1 批，8 个最多 4 批。首次 Free 和其他任务共用原线程池，无需申请重试名额，也无需按部署规模另配参数。[Executor 的实际 worker 数](../../kv_cache_manager/manager/schedule_plan_executor.h)

延迟回调到期后先检查取消和总截止时间，再尝试取一个名额。取到才开始核对元数据，Free 返回就归还。若整批无需重发，或因读取失败、取消而提前结束，退出时也归还名额。名额用满时，回调用原批次记录错开时间再投递，立即让出 worker。排队和退避期间不占重试名额，原 pending、内存预留和总截止时间继续沿用。等待本地名额不计作 Meta busy，不触发降额或下一档退避。

**重发哪些目标。** 等待期间对原 MetaIndexer 和后端只持有弱引用。重发前核对当前注册的仍是原实例和原后端，通过 `GetLocationsFromPrimary` 按 key 批读。只发送仍处于 `DELETING`、storage 和完整 URI 均与原记录一致的 Location。已消失或已变化的目标直接跳过，剩余为空就不发 HTTP。需要延后 CAD 时复用 `BatchDeleteLocations`，传入本次核对的 Location 期望值。[主存储批读](../../kv_cache_manager/meta/meta_indexer.h) · [按期望值删除](../../kv_cache_manager/manager/meta_searcher.h)

| 该批最后得到的结果 | KVCM 如何收尾 |
|---|---|
| 收到匹配的 400 和 `FREE_BATCH_TOO_LARGE` | 终止原批次的自动重发及降速反馈，保留未释放目标供调整组批或配置 |
| Free 已发出并成功 | 按条件删除对应的 Location 元数据 |
| Free 已发出，但超时、返回无协议的 429/5xx 或部分失败 | 首发和重发都沿用已有错误及 CAD 规则。停止自动重发，记录可能未释放的 GA |
| 持续明确拒收或等待重试名额直到截止，或再次发送前读取、投递失败，原实例或后端失效 | 失败收尾。对仍存在的原目标保留 `DELETING`，记录实例、key、Location、storage 和原因，供与 mempool 核对。GC V1 会跳过这类残留 |
| 关闭协议、降主或停机时仍在等待重发 | 使原任务的重试上下文失效，等待中的批次按取消收尾，沿用未发出目标的记录方式 |

截止时间只限制新的发送。截止前已发出的 Free 仍按回包结果收尾。[现有 CAD 处理](../../kv_cache_manager/manager/schedule_plan_executor.cc) · [GC 扫描范围](cache_garbage_collector.md)

### 重试记录的进程级内存上限

可选配置 `kvcm.cache_reclaimer.free_busy_retry_memory_limit_bytes`，以字节为单位，限制单个 KVCM 进程为 Tair 水位驱逐重试记录预留的内存总量。该进程内的所有实例组和 Tair 存储共用这份预算。常规部署使用默认值，内存余量较小的部署可显式调低。

退避期间保存**完整 URI 文本和受影响的 Location 关联**，重发时再解析当批 URI。预算覆盖这些文本、关联、容器容量和每批的固定开销。

- **准入时预留。** 开关启用时，在把目标标记为 `DELETING` 前按整批都可能被拒的情况原子预留额度。即使尚未发生 busy，进程预算不足也会缩小本轮批次，其余候选保留原状态，留待下轮扫描。单个 Location 自身就超过总预算时，记录所需字节并继续扫描其他候选。
- **发送和重试。** 首次 Free 使用执行时核实的 URI。如果实际批次变大，先补足额度再发。无法补足的未发部分归还其预留，用原任务的 Location 标识在同一总时限内延后复核，到期按未发出收尾。Meta 明确拒收后才留存该批 URI，连续 busy 复用同一份记录和预留。
- **收尾时归还。** 跳过的目标及时退还对应预留。每批结束时，释放这批的重试记录并归还剩余额度一次。原任务的 pending 在 Future 完成时统一释放。

在本机 arm64 macOS / libc++ 上，用项目的 URI 类型测了 10 万条合成记录的堆占用。每批 100 条，每条 URI 长 89 或 108 字节，含 `size`、`node_id`、`media_type`、`range_id`。每个 GA 配一个 64 位 block key 和一个 36 字符的 Location ID。范围覆盖两种 URI 长度以及容器预留、自然扩容两种方式。[URI 对象定义](../../kv_cache_manager/common/standard_uri.h)

| 等待期间的保存方式 | 仅 URI 列表 | 含 Location 关联 |
|---|---:|---:|
| 完整 URI 文本 | 约 12–14 MiB | 约 20–22 MiB |
| 已解析的 `DataStorageUri` 对象 | 约 43–51 MiB | 约 51–59 MiB |

以上只计这批记录持有的堆内存。进程峰值还包含已有 pending、发送时的临时对象和分配器碎片。现有 `pending_bytes_limit_per_group_type` 按 URI 的 `size` 累加后端待删数据量，一个 Location 也可以关联多个 GA。这些配额只能间接约束重试记录。[pending 字节计量](../../kv_cache_manager/manager/cache_reclaimer.cc)

KVCM 为 Tair 新建的 Location ID 通常是 8 字符。上表只作量级参考。初始默认预算取 64 MiB。在 202 容器用 Linux/libstdc++ 模拟每批 100 个 Location、每个 Location 一个 90 字节 URI 和 8 字符 ID，10 万条关联记录的堆增量约 26 MiB。实现按每个 Location 单独成批保守预留，同样样本中 2 万个 Location 需要约 28.5 MiB 预算。联调时可用实际 URI 分布和进程余量复核默认值。[Location ID 的生成](../../kv_cache_manager/manager/meta_searcher.cc)

## 5. 验收

| 场景 | 应看到的结果 |
|---|---|
| 未配置协议开关，或显式设为 `false` | 默认仅 Tair 水位驱逐携带协议头并预留重试预算。预算充足且无明确 busy 时按原 pending 额度提交，旧 Meta 的普通回包沿用原成功、错误及 CAD 规则。显式关闭时走原 Delete，跳过协议头和重试预算预留 |
| 初始退避使用默认值或改为 100 ms | 首次本地等待分别从 50 ms、100 ms 起步并加抖动。如果 Meta 建议等待 200 ms 且长于本地等待，实际重试等待为 200 ms。若本批触发调额，本轮冷却和随后的观察期各为 200 ms |
| 带协议头的列表为空、含非法 GA、超限，或 Provider 已经开始执行但超时 | 空或非法列表无副作用。明确 busy 对应零次 Provider 下发。普通显式 Free 计入负载但仍可执行。已下发的超时与普通 429/5xx 均不会触发重试。用真实 spec 数确认空载时正常大小的批次可被接纳 |
| 空载超大批次，或 URI 中含零、溢出、非法路径 | 超大批次获 400 和专属码，零下发、无原样重试、受影响的 Location 保留；非法 spec 可定位，空批零 HTTP，其他独立 Location 正常完成 |
| 同组多个任务跨越一次调额，读者宽限期大于冷却期 | 首个本轮 busy 只收紧一次。此前准入的任务稍后返回 busy 或成功，都不能再次减半或提前放宽。压力解除后由新准入任务的实际成功逐轮恢复，受软字节额度阻挡的大 Location 可以单独推进 |
| 同一任务的 Location A、B 分属两个 storage，A 成功，B 明确 busy | A 及时 CAD。只重发 B 的原批次，原 pending 最后释放一次。若重发返回普通错误，按既有规则收尾并记录可能未释放的对象 |
| 分别用 2、8 个 worker，让多组重试同时到期，并安排一批已就绪的首次 Free | 同时执行的重试最多分别为 1、4 批。重试名额占满时，空闲 worker 仍可执行首发。等待名额的批次只挂一次，Meta busy 计数和总截止时间不变，名额归还后可继续重试 |
| KVCM 重发前读到 RemoveCache、Trim 或 RemoveInstance 已处理同一目标 | 跳过已删除、已换 URI 或已注销实例的目标，无需对它再执行 Free 或 CAD。剩余为空时零次 HTTP |
| Meta 的建议等待时间超过剩余总时限，或延迟任务排队至截止、取消、停机切主、后端替换 | 停止重发，未发出且仍在的原目标可定位。截止前已发出的请求按实际结果收尾，重试内存预留和原任务 pending 各释放一次 |
| 分别使用默认内存预算和显式调低的字节上限，覆盖尚无 busy 和多组同时 busy | 各组预留总和始终在本进程的生效上限内。放不下的新候选保留原状态，已被拒批次复用原预留。归还后恢复准入，组级及进程级 pending 硬上限始终有效 |

联调用同一份水位驱逐负载对比开关前后的正常回收速度和过载超时数，再混入主动删除观察水位驱逐的让步。评估重试并发上限时，同时看首次 Free 从就绪到 worker 开始执行的排队时间，以及压力解除后积压批次的排空时间。只剩重试时也按同一上限执行，此时排空可能变慢。复用现有的 `pending_location_count`、`pending_delete_bytes` 和回收耗时。新增 `free_busy_reject_count`、`free_busy_retry_attempt_count` 和 `free_busy_reserved_bytes`，分别观测明确拒收、重试尝试次数和进程预留字节数。额度变化记录到实例组和类型，终止日志带上实例、key、Location、storage 和原因。
