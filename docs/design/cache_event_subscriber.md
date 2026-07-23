# Cache Event Subscriber 设计

## 目标

在不依赖 RTP-LLM #1198、tair-kvcache #241 或 #236 代码的前提下，从 public `main` 实现相同业务能力：将 RTP-LLM/vLLM 的本地 KVCache 状态可靠同步到 KVCM，并同时支持权威全量恢复和低开销增量更新。

## 数据流

```text
RTP GetCacheStatus v2 / vLLM ZMQ
  → Source.prepare（不提交 cursor）
  → 规范化 BlockRecord
  → KVCM ReportEvent
  → ACK 成功
  → Source.commit（提交 generation/cursor/state）
```

- RTP：changefeed 返回 `generation`、队列 head、明确的分页 `next_cursor`；冷启动、generation 不匹配、history gap 和周期校准返回快照。
- vLLM：通过 sequence 检测丢包并请求 replay。replay 是有限历史，只有从 sequence 0 完整重放或收到 `AllBlocksCleared` 才能建立权威状态。
- vLLM publisher 在已提交正游标后重新出现 sequence 0 时视为新 epoch，立即清空旧候选状态并上报权威快照。
- vLLM 冷启动没有任何 batch 时保持“未就绪”，既不注册节点也不发送 HEARTBEAT；Subscriber 会主动查询 replay，直到 sequence 0 或 `AllBlocksCleared` 建立权威基线，不把空闲误判成引擎故障。
- vLLM block hash 按低 64 bit 统一映射到 KVCM signed-int64 key；这与 vLLM 的 legacy int event 表示一致。按 engine hash 查询 KVCM 的调用方必须复用同一 codec。
- KVCM：`EVENT_BLOCK_SNAPSHOT` 表示该 `instance_id + host_ip_port` 的完整集合。服务端同步清理旧位置后写入新集合；空快照表示清空。同一 instance + host 的快照、增量和异步清理通过分片锁串行化；generation 变化或节点已恢复时旧清理立即中止。

## 存储身份

引擎本地 cache 使用独立的 `ST_EVENT_REPORT` 类型和 `event-report://` URI，不复用 `ST_VINEYARD`。该后端只管理节点注册、心跳、generation 和位置元数据，不提供数据读写；实际 KV 数据仍由 RTP-LLM/vLLM 进程持有。这样 KVCM 返回位置时，调用方不会误选 Vineyard connector。

`ST_EVENT_REPORT` 不计入 KVCM 管理的存储容量，也不会进入主动回收请求。其位置只能由引擎事件、权威快照、`HOST_DOWN` 或心跳超时清理，避免 KVCM 调用一个并不拥有引擎显存的 backend 执行 `Delete`。

部署时需创建一个 `event_report` storage，并将其名字放入 instance group 的 `event_reporting_storage_candidates`。现有 Vineyard 配置和位置协议不受影响。

## 一致性语义

1. STORED 只在引擎真实 cache index 写入成功后产生，REMOVED 只在真实删除后产生。
2. KVCM 只接受已注册且可用节点的缓存变更；节点失活后必须重新注册再重放未确认更新。
3. Subscriber 同一时刻最多保留一个未确认更新；KVCM 不可用时不继续拉取。
4. 增量 ADD/DELETE 是幂等的；多批请求中途失败时，可从旧 cursor 重放。
5. 全量快照是恢复与校准能力，不是稳态传输方式；稳态仅发送变化块。
6. 所有索引都按 `instance_id` 隔离；不同 DP endpoint 必须全部拉取成功后才提交聚合状态。
7. KVCM 的 key 域只有 64 bit；vLLM 的完整 digest 会被截断，部署方需接受相应碰撞模型，并以 `instance_id` 做租户/模型隔离。
8. Source 短暂失败期间暂停 HEARTBEAT；连续失败达到阈值后先上报 `HOST_DOWN` 再退出。RTP Launcher 在可选模式下限频重启 Subscriber，在 required 模式下由 `ProcessManager` 传播失败。
9. 对无法从 ZMQ 空闲与断连中区分存活状态的引擎，可配置 HTTP health URL；连续探测失败会停止 HEARTBEAT 并触发 Subscriber 下线。
10. 冷启动先 `prepare` 权威全量，再执行“节点注册 → 快照 ACK → cursor commit → 启动 HEARTBEAT”；首快照失败时立即 `HOST_DOWN` 取消该次注册，并以同一个未提交快照重新注册重试。

小快照通过单个 `EVENT_BLOCK_SNAPSHOT` 替换；超过单请求上限时采用“空快照清理 + 分批 ADD”。后者在传输期间允许短暂的部分可见，但任一批失败都不会提交 source cursor，重试会再次从清理开始，最终收敛且不会产生超大 HTTP 请求。

RTP 配置必须提供与 `dp_size` 等量且互不重复的 endpoint。vLLM 当前明确限制为单 DP；这与其每个 DP rank 使用独立 ZMQ 端口的发布模型一致，禁止在没有多 publisher 聚合器时静默只消费 rank 0。

## 与原 PR 的关键区别

- 基线直接来自两个仓库 public `main`，没有 cherry-pick、commit 依赖或隐藏的 #236 前置条件。
- 在 main 已有 `EventReportingBackend` 抽象上增加最小 `ST_EVENT_REPORT` 后端，不引入 #236 的大范围 storage 重构。
- 全量和增量是一个协议的两种模式，而不是两套互相独立的上报链路。
- RTP 分页区分 queue head 与 next cursor，避免分页时跳过事件。
- cursor 只在 KVCM ACK 后提交；Manager 故障形成反压，而不是继续堆积无界事件。
- vLLM 历史不足时显式失败，不把残缺 replay 当作完整缓存状态。

## 验收

- 冷启动空/非空快照、增量新增/删除、分页、generation 变化、history gap 均可收敛。
- KVCM 请求失败后 cursor 不推进，恢复后重试不丢事件。
- 空权威快照能清除该 host 的旧位置；其他 host 和其他 instance 不受影响。
- RTP 多 endpoint 任一失败时不提交任何 endpoint 状态。
- 主动回收只选择 KVCM 管理的存储位置，不选择 `ST_EVENT_REPORT` 位置。
