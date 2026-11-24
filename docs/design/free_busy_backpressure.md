# Tair Mempool 自适应回收设计

## 1. 目标与范围

水位驱逐、主动 GC 和运维 `TrimCache` 可以同时向同一个 Tair Mempool（PACE）发送 Free。每个来源单独限制并发时，后端承担的是三路之和。额度配少会拖慢回收，配多则可能让繁忙的 Meta 或 Provider 超时。

每个 KVCM 进程在 `SchedulePlanExecutor` 中统一管理这三路回收。**自适应调速**根据目标集群的健康 Provider 数设置在途 Location 和字节额度。Meta 明确拒收时收紧额度，后续 Provider 确认成功后逐步恢复。**退避重试**将整批被拒的 Free 延后重发，等待期间释放 worker。三路共享目标集群的额度，每个来源可单独封顶。外部 `RemoveCache`、注销实例时的清理和写入/迁移回滚沿用普通删除入口。[额度与来源配置](#6-三路删除统一流控)

## 2. 自适应调速如何决定在途额度

候选筛选读取 Executor 当前额度，提交时在同一把锁内复核并占用来源、集群和重试记录预算。每个已接纳的逻辑任务占一个来源请求名额，包括宽限等待、实际 Free 和元数据收尾。Tair Mempool 协议批按 storage 完成时归还对应的 Location 和字节，任务 Future 收尾时归还请求名额及剩余额度。一次 Free 可以包含多个 Location 和更多 GA，配置值计量的是在途工作量。[准入与归还](../../kv_cache_manager/manager/delete_admission.cc)

**按健康 Provider 数起步。** 开启 Free Busy 协议时，PACE 后端复用约每 10 秒一次的健康检查，从同一份 `GET /v1/api/memnode` 回包按节点 ID 去重，只计健康且承担 Provider 角色的节点。共享一个接入地址的存储使用最近一份有效观测，DRAM 和 SSD 上报同一集群时只计一次。Executor 在下次准入时读取缓存，不额外请求 Meta。[PACE 存储适配](../../kv_cache_manager/data_storage/pace_reclaim_storage.cc)

每个健康 Provider 给集群提供 **384 个 Location、20 GiB** 的基准在途额度，两项分别以 `cluster_locations`、`cluster_bytes` 为硬上限。在默认配置下，4 个 Provider 从 1,536 个 Location、80 GiB 起步，40 个从 15,360 个、800 GiB 起步，64 个及以上达到配置上限。这组初值取默认硬上限的 1/64，用于确定首批可放入多少工作。实际每秒完成量还取决于批大小、Executor 并发和后端耗时。

首次尚无观测、观测超过 30 秒时，按 1 个 Provider 起步。短暂查询失败或旧 Meta 未提供完整角色时，沿用仍新鲜的观测。明确观测到 0 个健康 Provider 时，集群一次只放行 1 个 Location 探测。缩容后立即用较小的窗口控制新准入，已接纳任务继续执行。扩容时，正常集群可以立即使用新窗口，处于退避中的集群仍按下表逐轮恢复。

| 集群收到的反馈 | 对新准入的影响 |
|---|---|
| 尚未收到 busy | 使用当前健康 Provider 数对应的基准额度，同时遵守来源上限及重试内存预算 |
| 当前轮次的任务首次收到明确 busy | 两项有效额度分别取当前额度和实际占用的较小值，再减半，向下取整，最低为 1。暂停该集群的新准入一个实际退避间隔 |
| 冷却到期 | 新任务能放进有效额度就接纳。此前已接纳的任务继续处理，被拒的批次沿用原额度重试 |
| 新轮次的任务收到 Provider 确认成功 | 冷却后再观察一个等长间隔。期满有成功反馈，后续准入检查可将有效额度翻倍一次，最高回到最近健康 Provider 数对应的基准额度。期间收到本轮 busy 会清掉该成功反馈并再次收紧 |

例如集群实际在途 80 个 Location、4 GiB，首次 busy 可使额度降至 40 个、2 GiB。每次调额用单调时钟记录轮次边界。此前接纳的任务即使在宽限期后才发出 Free，它的回包也只决定本批重试和收尾。冷却结束且该集群已经排空时，允许一个超过临时字节额度的 Location 单独推进，前提是满足共享、来源和内存的硬上限。

## 3. Tair Mempool Free 如何识别 busy

**灰度顺序。** KVCM 的协议和自动重试共用一个默认开启的开关。Meta 混部期间，或双方尚未确认第 4 节的 GA 身份和重发窗口时，部署须显式设置 `kvcm.cache_reclaimer.free_busy_enabled=false`。旧 follower 转发时可能把新 leader 的 429 写为 501 并保留原正文。启用前，所有可能服务、转发或接任 leader 的 Meta 都要完成升级，并按实际路由核对额度和完整回包。

Meta 在 `HandleDeleteGAs` 对带协议头的请求先校验完整 GA 列表。空列表或任一 GA 非法都返回参数错误，KVCM 发前逐 URI 校验路径完整解析为非零十进制 uint64，按十进制串和逗号组装 GA 列表，跳过空批。非法 spec 按 `(key, location_id, 原 URI)` 记录并保留所属 Location 的元数据，其他可独立完成的 Location 继续处理。一个 Location 有多个 spec 时，记录各 spec 的处理结果，避免过滤一项后把整个 Location 当作成功 CAD。

对合法列表，Meta 在调用 `FreeGAs` 前按正在处理的 Free 批数和 GA 数检查额度。单批能容纳、已有占用使本批加入后超限时，立即返回 `FREE_BUSY_NOT_ACCEPTED`，整批还未向任何 Provider 下发。若单批自身已超过 GA 硬上限，优先返回 HTTP 400 和 `{"protocol_version":1,"code":"FREE_BATCH_TOO_LARGE","status":"error","accepted":false,"rejection_scope":"whole_request","max_gas":1000}`，其中 1000 为示例。KVCM 校验 HTTP 状态和整份协议字段。这类错误直接按未发出失败收尾，保留受影响的 `DELETING`，记录实际项数和该 Meta 的上限供调整组批或配置。接纳的请求占用额度，处理结束时归还。所有显式列表 Free 都计入负载，只有带协议头的请求参与拒收。已下发的请求即使超时或部分失败，也按实际结果返回。

支持盘侧反压的 Meta 还可在下发前检查目标 Provider 上报的 SSD 删除池忙态。如果任一目标繁忙，Meta 使用同一个 `FREE_BUSY_NOT_ACCEPTED` 拒收整批。KVCM 按相同方式退避。

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

**入口与接口。** 三个来源在申请额度时标明来源。开启协议后，Executor 负责组批和重试，PACE 存储适配代码校验 GA 地址并调用具体后端。内源客户端为请求带上 `X-TairMempool-Free-Protocol: 1`，在归并错误前识别协议版本、状态码和整批未接纳保证。只清元数据的 EventReport GC 占用 GC 请求名额，跳过物理 Free。[Executor 入口](../../kv_cache_manager/manager/schedule_plan_executor.cc) · [PACE 存储适配](../../kv_cache_manager/data_storage/pace_reclaim_storage.cc)

**存储访问。** `DataStorageManager::WithRegisteredBackend` 按名称查找后端，回调执行期间持有注册表读锁。PACE 适配代码在回调内发送 Free，重试时先核对注册的仍是首发使用的后端。Free 返回即释放锁，退避期间只保留弱引用。同名存储重新注册后，旧批次会停止重发。[注册表访问](../../kv_cache_manager/data_storage/data_storage_manager.h) · [PACE Free 发送](../../kv_cache_manager/data_storage/pace_reclaim_storage.cc)

**反馈时机。** v1 的 HTTP 200 成功响应带布尔字段 `provider_free_acknowledged`。只要本批至少一个 Provider 对 Free RPC 返回业务成功，该值就是 `true`，包括同批另有 GA 缺路由的情况。整批因缺路由或节点断连被跳过时，该值为 `false`。KVCM 对成功回包照常执行 CAD，只有明确的布尔值 `true` 才计入额度恢复。旧 Meta 缺字段或字段类型异常时，成功收尾并保持当前额度。

Executor 收到回包时，按本次准入记录中的 storage 找到对应集群，在共享锁内立即处理反馈。明确 busy 清除尚未使用的成功反馈，三路的下一次准入都能看到新额度。[调额逻辑](../../kv_cache_manager/manager/delete_admission.cc)

## 4. 被拒的原批次如何重试

**重试的安全假设。** Meta 返回 `FREE_BUSY_NOT_ACCEPTED` 时，整批尚未向任何 Provider 下发。KVCM 的重试假定 GA 仍对应原分配。若 Trim 读页时目标已经是 `DELETING`，或 `RemoveCache` 正好处理该目标，主动删除和原任务仍可能对同一 GA 发 Free。并发删除、Provider 重启或地址复用时的重试安全性，以及允许重发的时长，由双方另行确认。

**保存和投递。** 首次删除在把 Location 标记为 `DELETING` 前检查任务的准入代次，已撤销的排队任务直接结束。CAS 成功后沿用原有 Sync 和读者宽限期。某个 storage 整批被拒时，保存该批原始 URI 文本和对应的 `(block_key, location_id)`。GC 同时保留各 spec 是否按后端策略跳过删除的标记，重试继续跳过已确认缺失的对象。其他 storage 继续执行，已完成的 Location 及时 CAD（按条件删除元数据）。重试使用已有的 `SubmitRaw(delay)` 让出 worker，每批最多挂一次，连续 busy 复用同一份记录。每个 storage 批次独立收尾，原任务在最后一个批次结束时释放来源请求名额。[现有延迟执行与删除收尾](../../kv_cache_manager/manager/schedule_plan_executor.cc)

**何时重发。** KVCM 从可选配置 `kvcm.cache_reclaimer.free_busy_initial_backoff_ms` 读取首次本地退避时间，默认 50 ms，有效范围为 1 至 5000 毫秒。同一批连续 busy 时，本地等待从配置值起逐次翻倍，加入抖动后最多 5 s。Meta 提供 `retry_after_ms` 时，实际等待取它与本地等待的较大值。触发调额的这次实际等待，同时决定第 2 节的冷却期和随后等长的观察期。

允许启动重试的总时限为 30 秒，从原批次首次准备 Free 开始计时。延迟投递和回调使用同一截止时间。取得存储读锁后先检查代次和时限，取到 PACE 客户端并准备启动 HTTP 时再检查一次，覆盖等待连接池的时间。已经启动的请求按实际回包收尾。

**重试删除的并发上限。** 每个 KVCM 进程的所有实例组和 Tair Mempool 存储共享一组名额，限制三路收到明确 busy 后同时执行的重试批数。按 `SchedulePlanExecutor::GetWorkerCount()` 返回的实际 worker 数 N 计算，联调起点取 `max(1, floor(N / 2))`。2 个 worker 最多同时重试 1 批，8 个最多 4 批。首次 Free 和其他任务共用原线程池，无需申请重试名额，也无需按部署规模另配参数。[Executor 的实际 worker 数](../../kv_cache_manager/manager/schedule_plan_executor.h)

延迟回调到期后先检查取消和总截止时间，再尝试取一个名额。取到才开始核对元数据，Free 返回就归还。若整批无需重发，或因读取失败、取消而提前结束，退出时也归还名额。名额用满时，回调用原批次记录错开时间再投递，立即让出 worker。排队和退避期间不占重试名额，原 pending、内存预留和总截止时间继续沿用。等待本地名额不计作 Meta busy，不触发降额或下一档退避。

**重发哪些目标。** 等待期间对原 MetaIndexer 和后端只持有弱引用。重发前核对当前注册的仍是原实例和原后端，通过 `GetLocationsFromPrimary` 按 key 批读。只发送仍处于 `DELETING`、storage 和完整 URI 均与原记录一致的 Location。已消失或已变化的目标直接跳过，剩余为空就不发 HTTP。需要延后 CAD 时复用 `BatchDeleteLocations`，传入本次核对的 Location 期望值。[主存储批读](../../kv_cache_manager/meta/meta_indexer.h) · [按期望值删除](../../kv_cache_manager/manager/meta_searcher.h)

重读遇到超时、I/O 错误或后端通用读取错误时，每个 storage 批次有一次额外的读机会。Executor 归还名额，按初始退避间隔加抖动重新投递，沿用原截止时间。整批读完并核对身份后才发 Free。读取失败不会产生 Meta busy 反馈或增加 Free 重试次数。再次读失败、数据损坏或返回结构异常时保留原目标并收尾。

| 该批最后得到的结果 | KVCM 如何收尾 |
|---|---|
| 收到匹配的 400 和 `FREE_BATCH_TOO_LARGE` | 终止原批次的自动重发及降速反馈，保留未释放目标供调整组批或配置 |
| Free 已发出并成功 | 按条件删除对应的 Location 元数据 |
| Free 已发出，但超时、返回无协议的 429/5xx 或部分失败 | 首发和重发都沿用已有错误及 CAD 规则。停止自动重发，记录可能未释放的 GA |
| 等待重试名额或持续拒收直到截止，或因预算不足、后端只读、缺少客户端、主存储读取最终失败、投递失败、实例或后端失效而停止 | 按未发出收尾。对仍存在的原目标保留 `DELETING`，记录实例、key、Location、storage 和原因，供与 mempool 核对。GC V1 会跳过这类残留 |
| 关闭协议、降主或停机时仍在等待重发 | 使原任务的重试上下文失效，等待中的批次按取消收尾，沿用未发出目标的记录方式 |

截止时间只限制新的发送。截止前已发出的 Free 仍按回包结果收尾。[现有 CAD 处理](../../kv_cache_manager/manager/schedule_plan_executor.cc) · [GC 扫描范围](cache_garbage_collector.md)

### 重试记录的进程级内存上限

可选配置 `kvcm.cache_reclaimer.free_busy_retry_memory_limit_bytes`，以字节为单位，限制单个 KVCM 进程为三路 Tair Mempool 删除重试记录预留的内存总量。该进程内的所有实例组和 Tair Mempool 存储共用这份预算。常规部署使用默认值，内存余量较小的部署可显式调低。

退避期间保存**完整 URI 文本和受影响的 Location 关联**，重发时再解析当批 URI。预算覆盖这些文本、关联、容器容量和每批的固定开销。

- **准入时预留。** 开关启用时，在把目标标记为 `DELETING` 前按整批都可能被拒的情况原子预留额度。Reclaimer 和 GC 跳过本轮放不下的候选，后续扫描可再选。Trim 在调用线程等待可用预算，整页超过预算时按 Location 拆批。单个 Location 放不进总预算时，Trim 返回失败。
- **发送和重试。** 首次 Free 使用执行时核实的完整 URI。如果实际批次变大，先补足额度再发。补额失败时，该 storage 批次立即按未发出收尾，归还预留并保留元数据。只有 Meta 明确拒收后才保存该批 URI，供每次重发时核对。重试剔除已变更的 Location 后立即归还其预算，剩余目标继续退避。
- **收尾时归还。** 跳过的目标及时退还对应预留。每批结束时，释放这批的重试记录并归还剩余额度一次。原任务的 pending 在 Future 完成时统一释放。

在本机 arm64 macOS / libc++ 上，用项目的 URI 类型测了 10 万条合成记录的堆占用。每批 100 条，每条 URI 长 89 或 108 字节，含 `size`、`node_id`、`media_type`、`range_id`。每个 GA 配一个 64 位 block key 和一个 36 字符的 Location ID。范围覆盖两种 URI 长度以及容器预留、自然扩容两种方式。[URI 对象定义](../../kv_cache_manager/common/standard_uri.h)

| 等待期间的保存方式 | 仅 URI 列表 | 含 Location 关联 |
|---|---:|---:|
| 完整 URI 文本 | 约 12–14 MiB | 约 20–22 MiB |
| 已解析的 `DataStorageUri` 对象 | 约 43–51 MiB | 约 51–59 MiB |

以上只计这批记录持有的堆内存。进程峰值还包含已有 pending、发送时的临时对象和分配器碎片。在途字节按 URI 的 `size` 累加后端待删数据量，一个 Location 也可以关联多个 GA。这些配额只能间接约束重试记录。[pending 字节计量](../../kv_cache_manager/manager/delete_admission.cc)

默认预算取 **128 MiB**，按需预留。Linux/libstdc++ 容器中的校准使用 8 字符 Location ID、30 字符实例名，每批 100 个 Location。用 90 字节 URI 时，2 万个单 spec Location 预留 29.1 MiB，20 spec 时预留 **117.5 MiB**。20 spec 的记录实际堆增量约 57.4 MiB。用分层存储集成测试中的 72–77 字节 URI 长度代入同一 20 spec 场景，预留为 103.1–107.1 MiB。128 MiB 可以覆盖上述 2 万个 Location 样本。三路共用预算。在上述 20 spec、90 字节 URI 的场景中，24,576 个 Location 的预留约为 144 MiB，因此即使 Meta 空闲也可能先触及 128 MiB 的进程预算。[Location ID 的生成](../../kv_cache_manager/manager/meta_searcher.cc) · [预留公式](../../kv_cache_manager/manager/free_busy.h)

早期仅水位驱逐的版本在同一 Linux 环境，使用真实删除执行器、本地元数据和始终成功的测试后端做了三轮对照。每轮删除 2 万个 Location、40 万个 GA，使用 8 个 worker 和 1 秒读者宽限期。从开始提交到所有 Future 完成，64 MiB 预算约需 2.17 秒，128 MiB 和关闭协议均约需 1.19 秒。128 MiB 在三轮中都没有等待预算。该对照衡量 KVCM 在此工作负载下的排空时间，线上端到端耗时还取决于 Meta 和 Provider。

## 5. 验收

| 场景 | 应看到的结果 |
|---|---|
| 未配置协议开关，或显式设为 `false` | 默认三路 Tair Mempool 删除携带协议头并预留重试预算，按 Provider 基准额度起步。旧 Meta 的普通回包沿用原成功、错误及 CAD 规则。显式关闭时走普通 Delete，并使用配置的静态准入上限 |
| 健康 Provider 从 4 个增至 40 个再缩容，同一接入地址有两个 storage | 正常集群随最新观测扩缩窗口。缩容后占用超额时等待已有任务收尾。两个 storage 共用最新观测，较旧回包不能扩大或缩小窗口 |
| 退避期间扩容，或健康节点数暂时未知 | 扩容保留当前冷却和已经收紧的额度，只有新轮次的 Provider 确认成功才能逐步放宽。超期观测退回单 Provider 基准，明确的空集群只允许一个 Location 探测 |
| 初始退避使用默认值或改为 100 ms | 首次本地等待分别从 50 ms、100 ms 起步并加抖动。如果 Meta 建议等待 200 ms 且长于本地等待，实际重试等待为 200 ms。若本批触发调额，本轮冷却和随后的观察期各为 200 ms |
| 带协议头的列表为空、含非法 GA、超限，或 Provider 已经开始执行但超时 | 空或非法列表无副作用。明确 busy 对应零次 Provider 下发。普通显式 Free 计入负载但仍可执行。已下发的超时与普通 429/5xx 均不会触发重试。用真实 spec 数确认空载时正常大小的批次可被接纳 |
| 空载超大批次，或 URI 中含零、溢出、非法路径 | 超大批次获 400 和专属码，零下发、无原样重试、受影响的 Location 保留；非法 spec 可定位，空批零 HTTP，其他独立 Location 正常完成 |
| 同一集群多个任务跨越一次调额，读者宽限期大于冷却期 | 首个本轮 busy 只收紧一次。此前准入的任务稍后返回 busy 或成功，都不能再次减半或提前放宽。压力解除后由新准入任务的实际成功逐轮恢复，受软字节额度阻挡的大 Location 可以单独推进 |
| 同一任务的 Location A、B 分属两个 storage，A 成功，B 明确 busy | A 及时 CAD。只重发 B 的原批次，原 pending 最后释放一次。若重发返回普通错误，按既有规则收尾并记录可能未释放的对象 |
| 首发或重发收到 200 success | 照常 CAD。只有 `provider_free_acknowledged=true` 产生成功反馈。全批缺路由、全批断连和旧 Meta 缺字段均保持当前额度 |
| 分别用 2、8 个 worker，让多组重试同时到期，并安排一批已就绪的首次 Free | 同时执行的重试最多分别为 1、4 批。重试名额占满时，空闲 worker 仍可执行首发。等待名额的批次只挂一次，Meta busy 计数和总截止时间不变，名额归还后可继续重试 |
| KVCM 重发前读到 RemoveCache、Trim 或 RemoveInstance 已处理同一目标 | 跳过已删除、已换 URI 或已注销实例的目标，无需对它再执行 Free 或 CAD。剩余为空时零次 HTTP |
| 主存储第一次重读超时，或连续读失败 | 超时后恢复时只重发核对过的目标。连续失败时无新增 Free，按剩余 Location 数记录 `primary_read`，名额和预留均归还 |
| 单个 storage 的首发预算补足失败，或同批两个 Location 中一个在退避期间换了 URI | 补额失败的批次零下发，保留元数据。重试批次在剔除已变更目标后只持有剩余记录的预算 |
| 排队任务在首次 CAS 前被撤销，或重试在连接池等待期间被撤销或到期 | 前者的 Location 保持可读，后者保留 `DELETING`，两者均无新增 HTTP 请求 |
| Meta 的建议等待时间超过剩余总时限，或延迟任务排队至截止、取消、停机切主、后端替换 | 停止重发，未发出且仍在的原目标可定位。截止前已发出的请求按实际结果收尾，重试内存预留和原任务 pending 各释放一次 |
| 分别使用默认内存预算和显式调低的字节上限，覆盖尚无 busy 和多组同时 busy | 各组预留总和始终在本进程的生效上限内。放不下的新候选保留原状态，已被拒批次复用原预留。归还后恢复准入，集群和来源硬上限始终有效 |

三路联调还应覆盖以下场景。

- 同一进程的两个 Trim 和 GC 同时访问同一接入地址的 DRAM、SSD。观察三路合计占用、Trim 共用的请求名额以及一个集群 busy 时另一个集群的进展。两份 Spectrum 配置只改 `timeout` 时应共用额度和冷却，改为另一个 `port` 时可独立准入。
- 一页 Trim 包含 100 个 Location。把有效额度压到 40，确认页内所有部分都收尾后接口才返回。只临时占用了部分额度时，确认能放进完整窗口的批次保持完整。单项超过硬上限时，确认 Trim 报错并跳过最终的残留清理。
- Trim 读到 A、B 后，驱逐任务先把 A 标成 `DELETING`。Trim 跳过 A，只为 B 发 Free。B 删除成功、其余收尾也成功时，Trim 返回成功。若 B 的物理删除失败，Trim 按原规则返回错误。
- 让 Trim 在调用线程等额度后降主，确认等待被唤醒、已接纳任务按取消规则收尾。用普通 `RemoveCache` 对照三路满额时的提交行为。
- 让 GC 占满集群额度，再触发集群冷却。在本地内存元数据上检查两个可删冷 key 的访问时间和 LRU 顺序。额度恢复后两者同轮被采到时，先选访问时间更早的 key。单个 Location 超过硬上限时仍应让位。
- 同一组的最冷三个 key 都在冷却中的集群，后续样本位于健康集群，单轮预算为两个 key。Group LRU 应提交第 4、5 个 key，前三个保留访问时间和顺序。若一个 key 仍有可删的其他 Location，该 key 占一个名额。

原有协议联调用同一份水位驱逐负载对比开关前后的正常回收速度和过载超时数，再混入主动删除观察水位驱逐的让步。评估重试并发上限时，同时看首次 Free 从就绪到 worker 开始执行的排队时间，以及压力解除后积压批次的排空时间。只剩重试时也按同一上限执行，此时排空可能变慢。复用现有的 `pending_location_count`、`pending_delete_bytes` 和回收耗时。`free_busy_reject_count`、`free_busy_retry_attempt_count` 和 `free_busy_reserved_bytes` 分别观测 Meta 的明确拒收、重试尝试次数和进程预留字节数。

- `free_busy_budget_skip_location_count` 统计 Reclaimer 筛选或提交时因内存预留不足被挡住的 Tair Mempool 候选次数。同一 Location 下轮再次被挡会再计一次，这些候选保留原状态。
- `delete_admission.cluster_locations` 和 `delete_admission.cluster_bytes` 分别显示每个 mempool 的在途 Location 数和待删字节数。`cluster` 标签取记账用的 `endpoint/<接入地址>` 或 `storage/<存储名>`，`quota=used` 表示当前占用，`quota=effective` 表示当前有效上限。冷却期间暂停新准入，已接纳的工作仍计入占用。
- `delete_admission.trim_wait_count` 统计实际等待准入的 Trim 批次数，每批首次等待时加一。`delete_admission.trim_wait_us` 累计这些批次等待额度的微秒数，包括降主取消前的等待。两个计数均按进程汇总，观察一轮压测前后的增量，可估算每个受阻批次的平均等待时间。
- `free_busy_retained_location_count{reason}` 统计专用删除路径停止处理、交由日志核对的目标个数。等本地名额到期记 `slot_deadline`，等 Meta busy 退避到期记 `busy_deadline`，准备发送时到期记 `dispatch_deadline`，主存储读失败记 `primary_read`，撤销记 `cancelled`。其余原因包括预算、URI、批次大小、实例或后端变化及本地执行失败。累计值按 Location 计算，当前仍在 `DELETING` 的数量可能随后变化。具体目标查终止日志。

## 6. 三路删除统一流控

Reclaimer、GC 和 `TrimCache` 各自按批提交并标明来源，由 **`SchedulePlanExecutor` 统一准入和调额**。三路竞争目标 mempool 的空闲额度。Executor 同时检查本批能否放进集群共享额度和该来源的剩余额度，接纳后记账一次。来源上限只限定这一路最多能同时占用多少。

| 适用范围 | 计量内容 | 默认策略 |
|---|---|---|
| 单个目标 mempool，三路合计 | 在途 Location 数、待删数据字节数 | 硬上限为 24,576 个 Location、1,280 GiB。开启协议时按第 2 节的 Provider 数和 busy 反馈决定实际窗口 |
| 单个删除来源 | 在途删除请求数、在途 Location 数、待删数据字节数 | 各项可独立封顶，数值 0 表示该项不限 |

来源的在途请求数按 Executor 已接纳、尚未收尾的逻辑删除任务计量。同一任务按 storage 拆批或因 busy 多次重试，在适用的来源请求数上限中只占一个名额，原任务收尾时归还。各批次收尾时归还对应的 Location 和字节额度。

共享硬上限在原每组、每类型 20,000 个 Location 和 1 TiB 的数值上留了余量。每个 KVCM 进程独立记账。多个进程访问同一集群时，通过同一 Meta 的 busy 信号各自调整。

Reclaimer 和 GC 在筛选候选时查询 Executor。额度不足时，两者按各自的扫描方式重访候选。

- Reclaimer 筛选时，如果一个 key 因临时反压而整项落选，且其中有可删的 Tair Mempool Location 单独能放进空窗口，本地元数据保留它的访问时间和 LRU 位置。临时反压包括当前额度不足、重试内存暂满或集群冷却。Group LRU 用本轮样本中下一个最冷的 key 补这个名额，最多提交原定数量。单项超出硬上限或只有非 Tair Mempool Location 被挡住时，沿用原有维护性 touch。[候选筛选](../../kv_cache_manager/manager/cache_reclaimer.cc) · [Group LRU 补位范围](cache_reclaimer_group_lru.md#72-执行和停止)
- GC 在扫描前发现自身额度已满时保留游标。若读完一批后才被目标集群挡住，已跳过的候选随游标前进，可能要等下一轮完整扫描才会重访。[GC 扫描与准入](../../kv_cache_manager/manager/cache_garbage_collector.cc)

实际准入由 Executor 记账一次。来源上限按本进程的逻辑任务统计，混合存储任务也占来源名额。目标位于 Tair Mempool 时，同时占用该集群的共享额度。同一来源跨集群提交时共用该来源的进程上限。

PACE 存储适配代码从配置中提取集群接入地址，供 Executor 记账。优先取 `service_discovery_url`，为空时取 `domain`。计算记账键时忽略 Spectrum 的 `timeout`、`cache_time`、`retry_time` 和 VIPServer 的 `timeout`。仅这些参数不同的存储共用额度和冷却。其余查询参数保留，其中 Spectrum 的 `port` 会覆盖服务发现返回的端口。客户端仍用原配置连接。不同地址别名分别记账，缺少接入地址时按 storage 唯一名记账。[PACE 接入地址](../../kv_cache_manager/data_storage/pace_reclaim_storage.cc) · [集群记账](../../kv_cache_manager/manager/delete_admission.cc)

任一路收到 Meta 的明确 busy 后，Executor 从下一次准入起收紧该集群的额度。此前已接纳、尚未发 Free 的批次按原计划执行。被 Meta 明确拒收的原批次沿用已占额度，按退避时间重试。

### Executor 怎样取下一项任务

三个来源共用同一组 worker。每个空闲 worker 一次取一个已到执行时间的任务，一项物理删除任务可以处理多个 Location。每次取任务时，worker 都按以下顺序检查队列。

1. 先看 Reclaimer 和 GC 共用的 `kReclaim` 队列，队里有就绪任务就取一项。
2. 当 `kReclaim` 没有就绪任务时，再看 Trim 所在的 `kSystem` 队列。

同一队列按可执行时间排序，时间相同则先取先入队的任务。已接纳的批次因 busy 重试时沿用原任务类别，退避到期后参与同样的排序。[任务类别与队内排序](../../kv_cache_manager/manager/schedule_plan_executor.h) · [worker 取任务顺序](../../kv_cache_manager/manager/schedule_plan_executor.cc)

例如，只有一个 worker，GC 的两批物理删除和 Trim 的一批物理删除都已就绪，期间也没有新任务加入。worker 会连续取 GC 的两批，再取 Trim 的一批。多个 worker 可以并行执行已经取出的任务。

### TrimCache 的运维额度

`TrimCache` 用于运维清理，后端有余量时应允许单次 Trim 更快推进。其在途 Location 数和待删字节数默认只受目标集群的共享有效额度约束，部署方仍可按需设置 Trim 来源上限。集群满额或因 busy 收紧时，三路继续竞争可用额度。

原先每次 Trim 调用最多有 `2 × Executor worker 数` 批在途。现在每个 KVCM 进程的所有 Trim 调用**合计最多 32 个已接纳、尚未收尾的批次**。单个 Trim 可以用足这 32 个名额，各批仍须满足目标集群的 Location 数和字节额度。在 Linux 集成环境分别运行单个大 Trim、多个并行 Trim，并与后台回收混跑，用 Trim 完成耗时、Meta busy 次数和超时次数验证 32 批的默认值。[Trim 提交与等待](../../kv_cache_manager/manager/cache_manager.cc)

对于整批能放进当前有效上限、只是部分额度正被占用的情况，Trim 提交方保留当前批，等可用额度足够时再提交。等待期间 Executor 的 worker 继续处理在途任务。整次 Trim 在所有批次完成且元数据收尾成功后返回成功。

**整页超过有效上限时按 Location 拆批。** Trim 每次扫描以 256 个 key 为一页的提示值。假设下一页有 100 个 Location，当前额度是 40 个，字节和其他上限均有余量。额度稳定时，Trim 依次提交 40、40、20 个。额度恢复后，剩余部分按新额度计算。未提交的目标暂存在当前页。单个 Location 超过临时字节额度时适用第 2 节的单独放行规则，超过硬上限时明确报错。已收到 Meta busy 的批次仍按原批重试。

每个子批按页内读到的 Location 申请额度，再按同一份值尝试标记为 `DELETING`。等待期间已消失或已变化的目标直接跳过。Trim 只对成功标记的目标执行删除，按实际删除结果收尾。若整批都跳过且元数据操作正常，这批直接成功。所有批次和元数据同步成功后，按原流程执行实例残留清理。

### 旧配置迁移

配置加载时生成一份生效规则。显式的新配置优先。缺少新值时，只自动转换计量单位和作用范围都等价的旧值。对于明确设置却无法等价转换的旧项，输出迁移提示及新配置的生效值，部署方可以按新口径重新配置。缺省项取新模型默认值。无论配置来自哪种写法，运行时均执行上述 Executor 准入逻辑。

新配置使用前缀 `kvcm.delete_admission.`。下表是没有显式配置时的值，bytes 的配置单位为字节。

| 作用范围 | 请求数 | Location 数 | bytes |
|---|---:|---:|---:|
| 单个 mempool 接入地址 | — | `cluster_locations=24576` | `cluster_bytes=1374389534720`（1,280 GiB） |
| Reclaimer | `reclaimer_requests=1024` | `reclaimer_locations=0` | `reclaimer_bytes=4398046511104`（4 TiB） |
| GC | `gc_requests=64` | `gc_locations=0` | `gc_bytes=0` |
| Trim | `trim_requests=32` | `trim_locations=0` | `trim_bytes=0` |

Reclaimer 的旧 `pending_delete_handler_limit`、`pending_bytes_limit` 分别映射到 `reclaimer_requests`、`reclaimer_bytes`。GC 的旧 `max_inflight_delete_requests` 映射到 `gc_requests`。显式的新来源值优先，包括 0。协议开关沿用 `kvcm.cache_reclaimer.free_busy_enabled`，同时控制三路是否发送协议头、预留重试内存和自动重发。关闭时仍执行静态共享额度与来源上限。

例如，旧的 `kvcm.cache_reclaimer.pending_location_limit_per_group_type=8000` 约束一个实例组、一种基础存储类型的水位驱逐。这个数无法推导出一个 mempool 能承受的三路总量。升级时若希望三路合计最多 8,000 个 Location，就显式把新共享上限设为 8,000。只提供这个旧值的部署会看到迁移提示，Tair Mempool 共享上限取新默认值。非 Tair Mempool 的水位驱逐继续用原组、类型上限，通过同一套准入逻辑执行。[旧配置的计量范围](cache_reclaimer_async_delete.md)

### RemoveCache 走普通 Free

`RemoveCache` 通过原有入口向 Executor 提交删除任务，提交后接口即可返回，后台 worker 调用普通 Free。该来源跳过 KVCM 的共享额度、来源上限和 busy 重试，即使三路已经用完目标集群的额度，`RemoveCache` 仍可提交。[接口文档](../api/meta_service.md#remove-cache) · [删除任务提交与跟踪](../../kv_cache_manager/manager/reclaimer_task_supervisor.cc)

`RemoveInstance` 使用已经摘除的 MetaIndexer 完成实例清理，保留每次调用 `2 × worker 数` 的窗口以及普通 Delete。

Meta 会把 `RemoveCache` 和 `RemoveInstance` 发来的显式 GA 列表计入 Free 负载。大量普通 Free 持续占满后端时，另外三路可能反复收到 busy，回收速度随之下降。
