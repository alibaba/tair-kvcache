# ReportEvent 小 block / 大批量性能优化记录

本文记录 `ReportEvent` 在 block size 较小、单次或并发上报 block 数较多时的性能分析、已经实施的
低风险优化、必须保持的并发语义，以及后续继续优化的边界。后续 AI 或开发者应先阅读本文和
[`report_event_snapshot_uri_version.md`](report_event_snapshot_uri_version.md)，不要仅根据某个 RT
指标直接引入并行。

## 1. 现象与指标解释

线上曾观察到 `ReportEvent` 与查询 RT 接近 100ms，同时 `meta_indexer.get_io_time_us` 可占约
80ms，查询侧 `manager.prefix_match_time_us` 也偏高。这里最容易误判的是指标名：

- `meta_indexer.get_io_time_us` 是 metadata backend 调用的墙钟时间，不等于磁盘或 Redis 网络 I/O。
  `storage_type=local` 时完全不经过 Redis，时间来自 sharded LRU lookup、每个 item 的 shared lock、
  location 容器复制/分配、revisit 统计及 CPU/cache miss；
- `storage_type=cached` 才是 local cache 加 persistent backend fallback/recovery；`storage_type=redis`
  才会进入 Redis pipeline、reply 传输和 HSCAN/HMGET。分析监控前必须先确认实例实际 storage type；
- `manager.prefix_match_time_us` 包含 metadata read、location 可见性判断、URI 解析、host 投影和前缀
  归约。其内层指标不能与它相加；
- ReportEvent 原有的节点表独占锁和 lifecycle lease 放大会增加 CPU/排队与 heap allocation。纯 local
  模式观察到 80ms 时，应先查 O(block 数) 的串行读和容器复制，不能归因于不存在的 Redis。

## 2. 已实施的低风险优化

### 2.1 请求内 medium 注册去重

一个 ReportEvent 请求内，相同 reporter/medium 的每个 ADD/DELETE 原来都会调用
`EnsureNodeRegistered`。现在只在该 medium 第一次成功时调用；成功 REGISTER 声明的 medium 也会
写入请求内集合。失败不会缓存，因此后续有序 REGISTER 或下一事件仍可重试，保持
“delta 可以出现在显式 REGISTER 前”的既有语义。
节点是否已经确保与 medium 集合分开记录：fresh reporter 的空 snapshot 即使没有 medium，也必须
调用一次 `EnsureNodeRegistered` 完成懒初始化，不能被“缺少待注册 medium”的快路径跳过。

`EventReportBackend::EnsureNodeRegistered` 对已存在且 medium 已知的节点使用 shared lock 快路径；
只有缺少 medium 时才释放 shared lock、获取 unique lock 并二次检查。节点 map 和 mediums 仍始终
受 `nodes_mutex_` 保护，不能改成无锁读取或用一个原子布尔值替代整个 map 的一致性。

### 2.2 lifecycle lease 从每 key 收敛到每次 metadata mutation 一次

原实现会在 modifier 的每个 key 上查 reporter fence、分配 `shared_lock`；已有 location 的 ADD
经过 block-create 和 targeted-location 两阶段时还会重复一轮。当前 `BatchMergeLocationSpecs` 已按
2.5 合并为一次 targeted RMW，在该次 RMW 第一次进入 modifier 时获取一个 lease，后续 key 复用：

```text
metadata read（可被 lifecycle writer 抢占）
        |
non-blocking lifecycle lease（本次 fused RMW 一次）
        |
本阶段全部 metadata mutation
        |
释放 lease
```

lease 不能在 metadata read 前获取，也不能无条件持有整个 ReportEvent。HOST_DOWN/REGISTER 的
锁序是 `lifecycle -> metadata`；如果旧请求阻塞在 metadata I/O，lifecycle writer 必须能先完成。
旧请求恢复后获取 generation-pinned lease 失败并放弃写入。fused RMW 从读取 key/目标 location 到
upsert 始终处于同一个 metadata shard 临界区，lease 从 read 后持有到 upsert 返回。确定性
HOST_DOWN/重注册竞态测试是这个优化的强制回归项。

### 2.3 GetHostCacheState 的 local 大查询路径

小 block 会放大一次请求的 block key 数。旧查询对每个 key 串行读取 local LRU，并把完整
`unordered_map<location_id, shared_ptr<CacheLocation>>` 克隆到请求结果；随后每个 location 又重复查
instance group、data-storage backend 和 reporter node lock。当前实现做了以下收敛：

1. `MetaLocalBackend::GetLocationValues` 只复制不可变 `CacheLocation` 的 `shared_ptr`，不复制 map node、
   hash bucket 和 location-id 字符串。原有 `GetLocations` 保留给需要按 id 查找的调用方；
2. 仅在“单一 persistent backend 且类型为 `local`”时，把达到阈值的 key 切成连续 chunk 并发读取。
   `cached`、`redis`、dummy 等模式仍执行一次原有 batch 调用，避免放大远端请求或破坏 recovery 语义；
3. 全进程所有 MetaIndexer 共享一个有界 `QueryExecutor`。配置的 worker 数包含 RPC caller，默认 4
   表示 caller + 3 个后台线程，不为每个请求创建线程。队列满时 caller 自己完成剩余 chunk；若 caller
   已完成全部工作，尚未启动的 helper 会取消，不能为了一个排队中的空任务制造队头阻塞。线程池部分
   构造失败时会停止并 join 已创建线程；`ParallelFor` 的分配/入队失败会先阻止 queued helper 再进入
   callback、等待 active helper 退出并返回失败，不能因 `noexcept` 直接终止进程，也不能让 helper 在
   请求返回后继续访问 request-local 引用；
4. 首个有界 metadata range 读取完成后，第一次处理 event-report location 时用 `std::call_once` 为该请求
   抓取 reporter liveness 与 committed-version 快照。每个 backend 只持有一次 `nodes_mutex_` shared lock，
   后续 range 的 `(block, location)` 只读不可变快照；
5. host/spec 投影和候选 host 前缀归约复用同一个有界 executor，输出仍按 host 字典序构造，普通 prefix、
   Mamba、Eagle pop 和 medium filter 的结果语义不变；
6. 普通 prefix 只为首 key 建立排序后的候选 host，后续 key 直接写入按候选编号组织的 packed bitset，
   不再为每个 key 构造 `map<string, set<string>>`，也不保存普通 prefix 根本不需要的 spec name。
   Mamba 仍需 spec 完整性信息，但改用排序的小 vector，避免每个 key 的红黑树 node 分配；
7. GetHostCacheState 专用可见性 checker 在校验 EventReport reporter 状态和 URI 时一并返回已经解析的
   medium/host。host 投影复用该结果，不再对同一 location id 做第二次 split。EventReport URI 的查询侧
   校验直接在不可变字符串上单次扫描，用 `string_view` 比较 generation；不再为每个 spec 拆分
   protocol/host/path、构造 query-param `std::map` 或复制 token。普通 prefix 对一个 EventReport location
   只做一次候选 host 标记；
8. service access log 默认只记录 key count、首末 key、query type、medium count、返回 host 数和最大
   prefix，不再把数千 key 的 protobuf 完整转 JSON 后再 parse 成 DOM。诊断时可临时设置
   `KVCM_GET_HOST_CACHE_STATE_FULL_ACCESS_LOG=true` 恢复完整 request/response，压测时应保持关闭；
9. local metadata read 不再让每个 query worker 对每个 block 直接更新共享 revisit histogram counters。
   每个 128-key chunk 先在本地累计 bucket/count/sum，再按非零 bucket 提交原子增量。Prometheus 最终值与
   逐 key `Observe` 完全一致，但避免十万级原子 RMW 争抢同一组 cache line；这段时间属于
   `meta_indexer.get_io_time_us`；
10. prefix 只把首个 `EC_NOENT` 当作正常终止。首个 miss 之后的 speculative read 结果不影响已经确定的
    前缀；但 miss 之前的 `EC_ERROR`、`EC_MISMATCH` 等硬错误必须原样返回，不能伪装成较短的 cache miss。
    普通 prefix 与 Mamba 路径遵循相同规则。

可见性快照在首个有界 metadata range 读取之后、其 projection 开始时采集。采集前已经可见的 HOST_DOWN
会被当前请求过滤；采集后的 HOST_DOWN 允许当前请求继续看到旧状态，但采集完成后本请求不再变化，下一
请求会重新采集。百万 key 流式查询不能等全部 range 读完再开始 projection，否则必须保留全量 location
或再次扫描 metadata；因此这里明确把快照线性化点放在首 range 与后续并行 range 之间。由于 `available`
是逐 reporter atomic，多个 reporter 与并发 liveness 变化之间不承诺一个全局事务时间点；保证的是
request-stable 结果，避免同一 reporter 在一次长 projection 中前半段 up、后半段 down。

这不是用“原子变量 + 双重检查”替换 node map。`available` 本身可以是 atomic，但 reporter 的存在性、
lifecycle generation、strict flag 与 committed token 必须在同一个受保护快照中一致；只原子化一个布尔值
会产生 host 已换代但仍配旧 token 的组合。`EnsureNodeRegistered` 的 shared-lock fast path + unique-lock
二次检查仍用于写路径，请求级只读快照用于大查询路径，两者解决的问题不同。

相关启动参数如下，修改后需重启：

| 参数 | 默认值 | 约束/含义 |
| --- | ---: | --- |
| `kvcm.meta_query.worker_count` | 4 | 1..64；包含 caller，设为 1 可回退为串行 |
| `kvcm.meta_query.parallel_threshold` | 256 | key/投影元素数小于该值时不并发 |
| `kvcm.meta_query.chunk_size` | 128 | `0 < chunk_size <= threshold` |

默认值是保守起点，不是固定 SLA。调参必须同时看单请求 p50/p99、并发查询 p99、CPU、RPC worker 排队和
ReportEvent RT；worker 并非越多越好。

### 2.4 ReportEvent 折叠、URI 与 Location 拷贝收敛

2026-08-05 的进一步分段和同机 A/B 表明，纯 local 模式下两阶段 RMW 的真实 backend I/O 只占少数，
主要成本是请求内聚合以及 modifier 在 metadata shard 锁内做的 Location/spec 深拷贝。当前实现做了以下
不改变持久化语义的收敛：

1. `LocationSpec` 和 `CacheLocation` 因显式析构函数而没有隐式 move，原来多处看似
   `std::move` 的代码实际退化为深拷贝；`set_location_specs(vector&&)` 也错误地执行了拷贝赋值。现在显式
   提供 `noexcept` move，并真正移动 vector。该修复同时覆盖新建 Location、spec merge、迁移等已有调用点；
2. delta 请求从“三层 `map` + 每个 mutation 的 event vector”改为 block 哈希表以及通常很小的
   location/spec 连续 vector。每个稳定 `(block, location, spec name)` 仍按请求顺序原地覆盖，最后按
   block/location/spec 排序后直接生成 ADD/DELETE task；删除了
   `delta_spec_mutations -> block_to_add/del -> merged_entries -> tasks` 的重复聚合和拷贝链；
3. 重试依赖保留为每个稳定 block/location 的有序 event 引用。只有已经 materialize 且参与最终 ADD 或
   DELETE phase 的事件先接收该 phase 的写错误，随后再按原有规则闭包传播。因此 last-operation-wins、
   admission failure、两 phase 不同错误以及逐 item 返回语义均未因扁平化改变；
4. 协议 URI 在入口完整校验时保存已经解析的 `DataStorageUri`，追加 `s_version` 时复用，不再为同一 URI
   重复 parse；BatchMerge 的版本一致性校验也复用本轮已解析对象，同时仍在 API 边界保留 raw 参数计数，
   duplicate `s_version` 仍会 fail closed；
5. `BatchMergeLocationSpecs` 直接让 caller-owned task 与目标 location id 对齐，不再复制 location id 和
   整组 spec/URI。existing Location 仍做一次必要的 copy-on-write，之后在其小 vector 内按 name 原地
   覆盖、兼容 legacy 重名并恢复字典序；不再构造每 key 的 ordered map 和第二份完整 spec vector；
6. 单 spec/task 的常见路径不创建去重 hash set，多 spec 时 set 保存 `string_view`；storage usage 使用
   一段 flat vector + offsets，不再为每 key 分配 vector。入口 URI 校验时顺带累计新 spec 的 `size`，
   merge 只解析旧 spec，避免写入成功后再次遍历、解析全部新 URI。

这些优化不改变 lifecycle lease、shard lock 或 HOST_DOWN 的锁序。收益来自减少进入和持有 metadata
shard 锁期间的 CPU/分配工作，因此既降低单请求 RT，也缩短并发请求的锁占用窗口；不能把它解释成
“把锁换成原子变量”。

### 2.5 BLOCK_ADD fused targeted RMW

2026-08-06 在独立分支落地了此前刻意延后的 fused 原语。旧的 existing-location ADD 先通过
`ReadModifyWriteBlock(GetLocationIds)` 枚举 block 的全部 location id，再通过
`ReadModifyWriteLocation` 读取目标 location 并 merge/upsert；纯 local 下至少产生两次 metadata read、
一次 upsert 和两轮 RMW 容器。当前路径改为：

```text
GetLocationsWithKeyStatus(key, requested_location_ids)
        |  同时返回 key 是否存在 + 各目标 location 的值/错误
        v
modifier：create 或 copy-on-write merge
        |
同一 shard-lock 临界区内一次 Upsert
```

必须保持以下不变量：

1. `key missing` 与 `key exists but target location missing` 严格区分。只有前者进入
   `put_global_indices`，参与 `max_key_count` 检查并在 upsert 成功后增加 `key_count`；已有 key 新增
   location 不增加 key 数，容量已满时仍允许更新已有 key。若一个 internal upsert batch 同时包含已有
   key 更新和超容量的新 key，只给新 key 返回 `EC_NOSPC`，不能把已有更新连带拒绝；
2. local backend 在一次 LRU lookup/item shared-lock 中返回上述两个层次的状态。返回 `EC_OK` 的 location
   必须非空且 id 与请求一致，否则 indexer fail closed；输出 vector 每次重新初始化，miss 不能泄漏 caller
   复用缓冲区中的旧指针；
3. `MetaStorageBackendManager` 在 cached recovery 模式仅对真正的 cache key miss 回源 persistent；cache
   中已有 key 但缺目标 location 时，cache 状态仍是 authoritative。generic backend fallback 仅对
   “所有目标均 NOENT”的歧义行补一次 `Exists`；纯 local 主路径不走该 fallback；
4. generation-pinned lifecycle lease 在 targeted read 后、modifier 第一次 mutation 前获取一次，并持有到
   upsert 返回；不能提前到 metadata read 前，也不能在 read 与 write 之间释放；
5. 逐 location read/type/modifier/write 错误保持原位，部分成功只更新成功 location 的 storage usage。
   legacy duplicate spec name 仍按原 last-value-wins 规则归一化；malformed backend shape、空/错 id、
   `key missing + target EC_OK` 等矛盾状态全部拒绝写入；
6. 当前实现只是把两个逻辑 RMW 合为一个 targeted read-modify-upsert；local backend 的 upsert 仍会再次
   lookup 并获取 item unique-lock。没有把 backend item 指针或锁暴露给 modifier，也没有引入批内线程。

## 3. 当前明确不做的事情

暂不并行执行 ReportEvent 内的 metadata batch。原因不是并行永远无效，而是 5.10 的同吞吐 A/B 已证明
把单次 L2 batch 控制在约 2k 能显著降低 ReportEvent、Get 和 Heartbeat 尾延迟；服务端再增加 writer
会与优先级更高的查询争夺 CPU、allocator 和 local LRU/item lock，同时扩大 key-count、请求内顺序与
lifecycle fencing 的并发面。若后续仍要并行，必须使用独立有界 executor、按互斥 shard 分组并保留串行
回退，不能创建 request-local thread 或复用查询 executor。

暂不把 local targeted read 与 upsert 进一步合成“持有一个 item unique-lock、在 backend 内执行
modifier”的原语。那会让 manager callback 进入 backend 临界区、扩大锁序与异常安全边界；当前一次
targeted read + 一次 upsert 已消除整轮 block-id 枚举，同时保持 backend API 分层。

GetHostCacheState 也不并发 Redis/cached backend batch，不创建 request-local thread，不复用只有少量
worker 且承载回收/迁移的 `SchedulePlanExecutor`。查询池独立且有界，避免长 metadata 请求饿死系统任务。

## 4. 非 local backend 的扩展边界

当前部署和本轮交付门槛是纯 local metadata；真实 Redis 不在本轮验证范围。通用
`GetLocationsWithKeyStatus` fallback 已保证正确性，但当所有目标 location 均不存在时会额外调用
`Exists`。若未来启用 Redis，应实现原生 pipeline，在一个 round trip 中同时返回：

1. key 是否存在（用于 `key_count/max_key_count`）；
2. 请求指定 location id 的值与逐项错误；
3. 不枚举、不传输无关 location value。

Redis 实现应把 EXISTS 与目标 HMGET 放进同一 pipeline round trip，并单独跑真实 Redis 的新 key、已有
key 缺 location、已有 location、properties-only key、部分 I/O 失败与恢复测试。在完成这些验证前，
不能把本轮纯 local 性能数据外推到 Redis，也不能删除 generic fallback。

## 5. 验证与观测清单

- UT：请求内 512 个跨重复 medium 的 ADD；并发 EnsureNodeRegistered；fused RMW 单 lease 与失败原子性；
  HOST_DOWN、REGISTER、新 snapshot 抢占阻塞 metadata read；同请求 ADD/DELETE 顺序；新 key、已有 key
  缺 location、已有 location、容量已满更新、部分 backend 错误、malformed response shape 与 storage
  usage 精确性。
- 手工容量：`EventReportBenchTest.test_20_large_single_request_delta_scaling` 分别记录 100/1000/5000/20000
  个新 block ADD 与相同 block 再次 ADD 的总 RT、单 event RT，并查询首/中/末 block，不能只看
  HTTP 成功码。
  可在启动 KVCM 后用 `--bench-test test_20_large_single_request_delta_scaling` 单独执行。纯 cache
  部署保持默认 local metadata backend；只有明确验证 Redis 部署形态时才传 `--meta-storage-uri`。
- 线上对比：至少拆分 request parse/fold、node ensure、lifecycle lease wait/fail、RMW lock wait、
  `get_io_time_us`、serialize、enqueue/upsert 和完整 ReportEvent RT；同时观察查询 p50/p99。
- local 路径已经使用目标化 backend read；若 `get_io_time_us` 仍接近总 RT，应先看 LRU/item-lock wait 与
  request key-count 分桶，不能再归因于已删除的全 location-id 枚举。
- 若 backend I/O 已显著下降但 CPU/锁等待仍主导，再评估按互斥 shard 分组的有界并行（建议先从
  2~4 并发开始），并对查询 p99、连接池等待和 Redis CPU 设置回退阈值。

GetHostCacheState 的新增分段指标：

- `meta_searcher.indexer_get_time_us`：整个 MetaIndexer 读取；
- `meta_indexer.get_io_time_us`：其内部 backend 调用墙钟时间；
- `meta_searcher.host_projection_time_us`：location 可见性、URI/host/spec 投影；
- `meta_searcher.host_prefix_reduce_time_us`：按 host 计算普通或 Mamba 前缀；
- `manager.prefix_match_time_us`：上述阶段及少量管理层开销的外层总时间。

UT 覆盖单线程/4-worker 结果对照、缺失 key 和重复 key、普通/Mamba 大于阈值、medium filter、Eagle pop、
HOST_DOWN 发生在 metadata read 期间、快照 instance 隔离、并发读写 local item、executor 队列饱和、
callback 异常、嵌套调用与线程池异常清理。可选 HTTP benchmark：

```bash
python integration_test/meta_service/test_report_event_snapshot.py \
  --host localhost --http_port 56020 --admin_http_port 56040 \
  --instance_id event_report_cluster_0 \
  --bench-test test_21_get_host_cache_state_local_scaling
```

它会对 100/1000/5000/20000 个命中 block（再加一个尾部 miss）记录 20 次串行请求以及 16-way 并发请求的
p50/p99，并逐次校验 host prefix。对照串行基线时分别用
`kvcm.meta_query.worker_count=1` 和默认 `4` 重启服务运行；不要在一次进程内动态改配置。

### 5.1 2026-08-04 本地 Redis 基准记录

在同机 Redis 7.2.5、debug KVCM、真实 meta/admin HTTP 接口下，使用本文新增的 benchmark 得到：

| events/request | 新建 block ADD | 已有 block/location 再次 ADD |
| ---: | ---: | ---: |
| 100 | 7.61ms（0.0761ms/event） | 10.40ms（0.1040ms/event） |
| 1000 | 62.99ms（0.0630ms/event） | 88.10ms（0.0881ms/event） |
| 5000 | 303.59ms（0.0607ms/event） | 420.14ms（0.0840ms/event） |

命令使用独立 Redis metadata backend，并对每档首/中/末 block 做查询校验。该数据只是当前开发机的
可重复基线，不是 SLA；已有 location 明显更慢也印证了两阶段 HSCAN + targeted HMGET 是下一轮
I/O 优化重点。线上判断必须结合 `get_io_time_us`、Redis CPU/网络和查询 p99。

### 5.2 2026-08-04 GetHostCacheState 纯 local 对照

同一台开发机、debug KVCM、真实 meta/admin HTTP 接口、单 reporter/medium 下，分别以
`kvcm.meta_query.worker_count=1` 和默认 `4` 重启进程运行
`test_21_get_host_cache_state_local_scaling`。每个请求包含 N 个连续命中 block 和一个尾部 miss；串行数据
为 3 次 warmup 后 20 次请求，并发数据为 16-way、共 32 次请求。每次响应均校验 prefix：

| blocks | worker=1 串行 p50/p99 | worker=4 串行 p50/p99 | worker=1 16-way p50/p99 | worker=4 16-way p50/p99 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 1.83/1.87ms | 1.87/2.71ms | 10.71/23.03ms | 13.68/23.01ms |
| 1000 | 11.26/11.62ms | 6.25/6.31ms | 14.86/23.20ms | 13.24/21.75ms |
| 5000 | 53.35/55.41ms | 27.79/29.60ms | 68.45/102.07ms | 52.59/70.90ms |

100 block 小于默认 threshold，差异属于噪声且没有并发收益；5000 block 的单请求 p50 下降约 48%，
16-way p99 下降约 31%。这是 debug 单机趋势而非线上 SLA。线上 rollout 仍应先小流量，确认分段 metrics、
CPU 和 ReportEvent p99；若并发度带来负收益，可直接把 worker_count 回退为 1。

### 5.3 2026-08-04 Release/O2 纯 local before/after

为避免把 Debug 构建开销当成线上结论，在同一台开发机上分别构建性能提交的直接父版本
`88e29c1` 和当前版本；两者均使用 Release/O2、真实 meta/admin HTTP、纯 local metadata backend。
当前版本使用默认 `worker_count=4`、`parallel_threshold=256`、`chunk_size=128`。下表选择双方第二轮
完整运行的数据；每个响应仍逐次校验 host prefix：

| blocks | 父版本串行 p50/p99 | 当前串行 p50/p99 | 父版本 16-way p50/p99 | 当前 16-way p50/p99 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.71/0.73ms | 0.68/0.70ms | 5.09/10.27ms | 5.34/11.42ms |
| 1000 | 2.62/2.69ms | 1.80/1.87ms | 9.03/15.78ms | 6.99/11.10ms |
| 5000 | 11.60/11.82ms | 6.95/7.05ms | 29.04/42.10ms | 22.01/28.56ms |
| 20000 | 47.33/51.67ms | 26.02/27.30ms | 142.68/188.26ms | 95.26/123.18ms |

20k 串行 p50 下降约 45%；这说明轻量 local read、host 投影与有界并行都产生了实际收益。高基数
16-way 请求即使优化后仍可能超过 100ms，不应只看全局 RT；至少要按 `request_key_count` 分桶。
一次 5001-key 并发请求的 gauge 快照中，父版本 `prefix_match/get_io/service` 分别为
25.717/9.620/25.750ms；当前版本为 10.441/6.623/10.471ms，且新增 projection/reduce 分别为
3.168/0.184ms。Gauge 只是最后一次观测，不是 percentile，不能与上表混用。

并发度和 chunk 调优结论：

- 低并发下 worker 8/16 可继续降低单请求 RT，但 worker 32 收益已明显递减且并发尾延迟回升；
- 20k isolated 热轮中，worker 4/8/16 的串行 p50 约为 26.02/23.05/21.18ms；16-way
  p50/p99 代表值分别为 95.26/123.18、73.92/115.82、100.24/133.01ms；
- 同时运行 100-thread ReportEvent ADD 与重复 20k GetHost 时，worker 4 与 8 的 ReportEvent 分别为
  1399/1435 QPS、p99 186.93/184.97ms；但第二轮 GetHost 16-way p50/p99 从 worker 4 的
  108.62/184.91ms 恶化到 worker 8 的 132.50/217.67ms；
- chunk 64 在小批次略快但冷并发更抖，chunk 256 在 5000 blocks 略慢，默认 128 更均衡。

因此默认仍保持 worker 4/chunk 128。低并发、CPU 余量充足且更关心单请求 RT 的部署可以显式尝试
worker 8；必须同时观察 ReportEvent p99、GetHost key-count 分桶、CPU 和 executor queue saturation，
不能把本机 isolated 结果直接当作通用默认值。

### 5.4 ReportEvent 高基数结论

本节与 5.5、5.6 保留的是 fused targeted RMW 落地前的历史基线和决策背景；当前实现状态以 2.5 和
5.10 为准，不能再把“尚未合并两阶段”当作现状。

Release/O2、纯 local 下，当前版本单请求 20k BLOCK_ADD 的 create/update 为
196.55/239.51ms，约 9.8/12.0us 每 event，整体近似线性。对比 `feature/event_report_4@b776dd4`，
5000-event 单请求及 100-thread ADD、50-thread mixed throughput 均只相差约 0~4%，属于噪声；当前改动
没有回归 ReportEvent，但也不能宣称改善其高并发尾延迟。

一次 20k existing-location update 的详细观测为：客户端 RT 249.59ms，KVCM service/event timer
约 161ms，block RMW 32.18ms，location RMW 63.38ms，local backend get/upsert I/O 仅
7.33/6.04ms，metadata lock wait 约 1us。也就是说，剩余成本不是 Redis 或全局锁，而是：

1. HTTP/JSON/protobuf 请求解析与响应边界约 89ms；
2. 两阶段 RMW 内逐 key 的 CPU、拷贝和 modifier 约 95ms，其中真实 backend I/O 约 13ms。

这组数据是 5.5 拷贝收敛前的基线；本轮先减少 block-create/targeted-location 两阶段的数据变换，并保持
串行 RMW。若线上 10k~20k 单批 ReportEvent 在 5.5 的收益后仍需显著低于 100ms，再单独评估 local RMW
shard 并行并增加 parse/fold 指标。并行会触及写入原子性、lifecycle fence 和锁序，不能仅凭单请求 RT
开启；上线前仍可通过限制单请求 event 数或分批上报控制尾延迟。

### 5.5 2026-08-05 ReportEvent 拷贝收敛同机 A/B

使用本节 2.4 的性能改动直接父提交 `8f7d5bc` 和当前工作树分别构建 Release/O2 二进制；两个进程使用
独立端口、独立纯 local instance，在同一台机器连续运行
`test_20_large_single_request_delta_scaling` 两轮。下表是两轮平均值，所有请求均校验 committed token 以及
首/中/末 block 的最终 URI：

| events/request | 父提交 create/update | 当前 create/update | create/update 降幅 |
| ---: | ---: | ---: | ---: |
| 100 | 1.84/1.89ms | 1.55/1.56ms | 15.8%/17.7% |
| 1000 | 11.78/13.16ms | 9.41/10.16ms | 20.1%/22.8% |
| 5000 | 54.23/63.01ms | 43.88/48.14ms | 19.1%/23.6% |
| 20000 | 217.19/269.40ms | 171.52/202.26ms | 21.0%/24.9% |

20k 单次请求的两轮原始区间分别为：父提交 create 216.86~217.51ms、update
267.44~271.36ms；当前 create 170.27~172.77ms、update 201.97~202.54ms。趋势随规模增大而扩大，
符合“减少每 event/node 分配与深拷贝”的预期，不是固定开销或单次快样本。

同一当前二进制随后运行 GetHostCacheState local benchmark，100/1000/5000/20000 block 串行 p50 为
0.67/1.78/6.71/25.52ms；20k 的 16-way p50/p99 为 95.65/125.41ms，与 5.3 的优化后基线基本一致，
未观察到通用 move 修复带来的查询回归。以上仍是单机趋势而非 SLA；线上 rollout 应同时观察
ReportEvent key-count 分桶、RMW 两阶段、lock wait、CPU 与 GetHost p99。

### 5.6 2026-08-05 线上 L2 大批次反馈与后续判别

本节记录当时尚未落地 fused targeted RMW 的线上反馈。后文关于“下一步做 fused”的表述是历史判断，
最终实现、风险控制与新数据见 2.5、5.10。

一轮按目标 QPS 持续发送的线上压测中，客户端全程 `fail/drop/skipped=0`，工作队列通常为 0~1；
RSS 峰值约 159.4MB、payload budget 峰值约 129.2MB。停止边界丢弃的一个任务不计入稳态失败。
客户端 L2 构包平均约 3.84ms，而 HTTP 约 217ms；Get 构包约 1.28ms，而 HTTP 约 118ms。因此本轮
瓶颈不在客户端构包、排队或内存预算，HTTP 往返与服务端处理占绝大多数。

L2 延迟随单批 block 数近似线性：batch p50 约 7k 时 RT p50 约 158ms，batch p95 约 34.5k 时
RT p95 约 795ms，两点折算的处理速度都约为 44k blocks/s。大批次期间 Get 出现 200~400ms 尾延迟，
很小的 Heartbeat 也达到约 164ms p99。这些数据足以判断“大 L2 请求的服务端线性工作正在拖慢共享
资源”，但仅凭客户端 HTTP 时间仍不能区分以下来源各占多少：

1. meta HTTP 端口上的所有 API 共用同一组 `coro_http_server` I/O worker，业务 handler 又同步进入
   `MetaServiceImpl`/`CacheManager`；长请求可能造成 worker 占用或 CPU 调度排队；
2. `MetaIndexer` 的 RMW 对每个 batch 在 writer shard lock 内依次完成 backend read、modifier、可选的
   persistent-backend 序列化和 upsert/delete；同 shard 的其他写入会等待。纯 local 的 Get 不获取这把
   writer shard lock，但会和 upsert 争用对应 `MetaMemCacheItem` 的 shared/unique mutex；跨 shard 请求
   仍可能争用 CPU、allocator 和 cache；
3. HTTP body 解析、protobuf/JSON 转换以及响应序列化不在现有 RMW 分段指标内，大 payload 会继续带来
   线性 CPU 和内存带宽成本。

下一轮线上观测应把客户端 HTTP RT 与服务端 `ReportEvent` service timer 对齐，并同时按 batch-size
分桶采集 request parse/fold、block/location RMW、`get_io_time_us`、deserialize/serialize、
`lock_wait_time_us`、upsert 和 HTTP worker queue/active 数。现有 `lock_wait_time_us` 只覆盖 writer shard
mutex，不覆盖 local item mutex；若要验证 Get 被 upsert 阻塞，需另加 item-lock wait 指标。若 HTTP RT
显著大于 service timer，优先查 HTTP worker 排队与 body 编解码；若 service timer 本身接近 HTTP RT，
再依据 RMW/lock/CPU 分段决定是继续减少 Location 处理成本，还是做 shard-aware 并行。不要从 Heartbeat
p99 单独反推出某一把锁有问题。

本节 5.5 的拷贝收敛可把 20k create/update 降低约 21%/25%，属于应先上线验证的常数优化；按本轮
34.5k p95 批次估算，它不足以单独消除 700ms 级尾延迟。发布侧最直接的保护是限制单次 L2 block 数并
拆成较小批次（可先以 2k~5k 做压测起点），同时给 Get/Heartbeat 保留独立的并发或队列预算。若拆批后
服务端总吞吐仍稳定且尾延迟显著下降，再决定是否需要把 ReportEvent 放到独立有界 executor，或按互斥
metadata shard 做 2~4 路有界并行。两种结构性改动都必须保留 request 内 last-operation-wins、两阶段
key-count、lifecycle fence 和逐 item 错误语义，并设置过载回退，不能用无限并行掩盖单批过大。

随后另一轮混合压测得到：L1P5 ADD 为 1.999 QPS、平均/p99/max RT 为
7.56/42/146ms；L2 ADD 为 14.998 QPS、149,969 blocks/s，即平均约 10k blocks/request，平均/p99/max
RT 为 224/841/924ms；Get 为 0.099 QPS、平均 8,984 keys/request，平均/p99 RT 为 119/418ms。L2 的
单请求处理速度约为 44.6k blocks/s，与前一轮约 44k blocks/s 基本相同，进一步确认瓶颈随 block 数线性
增长，而不是压测器吞吐不足；约 3.36 个 L2 请求的平均并发也解释了为何 aggregate blocks/s 高于单请求
速度。

这轮数据发生在 5.5 的本地性能 patch 推送之前；当时该远端开发分支 head 仍为 `637d3e0`，所以不能把
它当作 5.5 优化后的线上结果。应在包含 5.5 commit 的新 head 上用相同流量重跑 before/after，再判断
剩余差距。预期 20%~25% 的 RMW/折叠收益仍不足以完全消除 800ms p99；若 after 仍呈相同斜率，下一项
若本轮只优化 GetHostCacheState，应先上线 2.3 的紧凑投影、摘要 access log，并在 CPU 有余量的环境用
worker 4/8 做 A/B；fused targeted RMW 会改变 ReportEvent 写入原语和 key-count 语义，不应混入这次
低风险查询优化。

`GetHostCacheState` 的完整 access-log JSON 问题已按 2.3 收敛。`MakeBatches` 的 `batch_key_size` 仍是
shard-boundary soft limit，同一 shard 的 keys 不会被硬切；但 writer shard lock 不阻塞纯 local Get。
若线上 `mutex_shard_num=16`，10k keys 的均匀请求约为 625 keys/shard；增加 128/256 的 lock-hold hard
limit 主要改善写写公平性，必须用混合写 p99 验证，不能把它误报成 Get 尾延迟根因。

### 5.7 2026-08-05 GetHostCacheState 紧凑投影与 worker 4/8 A/B

在 2.3 的 packed presence、EventReport 解析复用和摘要 access log 完成后，使用同一 Release/O2
二进制、纯 local metadata、真实 HTTP 接口运行 `test_21_get_host_cache_state_local_scaling`。每档都先
构造连续命中 block，再追加一个尾部 miss；20 次串行与 16-way 请求逐次校验 prefix。结果如下：

| blocks | worker=4 串行 p50/p99 | worker=8 串行 p50/p99 | worker=4 16-way p50/p99 | worker=8 16-way p50/p99 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.56/0.59ms | 0.56/0.59ms | 8.47/17.51ms | 10.84/28.22ms |
| 1000 | 1.32/1.35ms | 1.20/1.23ms | 6.47/10.87ms | 6.49/11.22ms |
| 5000 | 4.54/4.72ms | 3.86/3.89ms | 21.12/40.57ms | 20.18/39.99ms |
| 20000 | 16.07/16.18ms | 13.56/13.63ms | 77.87/101.60ms | 103.57/134.09ms |

对比 5.5 中改动前同一分支的 20k worker=4 基线（串行 25.52ms、16-way 95.65/125.41ms），新实现的
串行 p50 下降约 37%，16-way p50/p99 下降约 19%/19%。worker=8 在低并发 20k 单请求上比 worker=4
再快约 16%，符合“CPU 有余量、Get QPS 很低”的部署条件；但 16-way 20k p99 反而增加约 32%。因此代码
默认值继续保持 4。若线上 Get 约 0.1 QPS 且 CPU 确有余量，可显式配置 worker=8 做小流量 A/B，必须
同时观察 L2 ReportEvent 压力下的 Get p99 和 executor queue；不能仅凭单请求数据修改全局默认值。

### 5.8 2026-08-05 Get 查询 URI 零分配扫描与 histogram 批量提交

在 5.7 的 worker=4 版本上继续检查发现两个与 block 数线性相关、且都位于 GetHostCacheState 的热点：

1. `IsEventReportLocationReadable` 对每个 spec 构造 `DataStorageUri`。解析会复制 URI 的多个 substring，并为
   每个 query param 分配 `std::map` node；随后 `GetParam` 和 `SnapshotUriInfo` 又复制 32-byte token。查询
   实际只需要确认 URI 有合法 scheme、`s_version` 不重复且为 32 位十六进制，并与 request snapshot 中的
   committed token 比较。因此改为一次 `string_view` 扫描，malformed/重复 token 仍然 fail closed；
2. `MetaLocalBackend::GetLocationValues` 对每个命中 key 调用 revisit histogram `Observe`。默认 13 个 bucket
   下，一次 10k-key 查询会产生十万级共享 counter 原子 RMW；4 个 query worker 会争抢同一组 cache line。
   现在每个 executor chunk 先本地聚合，再一次性提交 bucket/count/sum，最终指标值不变。

用当前源码、Release/O2、4096 条变化的典型 EventReport URI、100 万次循环做隔离微基准：

| URI 可见性检查 | 每 spec 分配次数 | 每 spec 累计分配 | 每 spec CPU |
| --- | ---: | ---: | ---: |
| 原完整 `DataStorageUri` parse | 10 | 605.7B | 约 497ns |
| 新 `string_view` 单次扫描 | 0 | 0B | 约 51ns |

新扫描约快 9.7 倍。按 20k specs 估算，仅这一步减少约 12.1MB 短生命周期 allocator 流量和 8.9ms
单核 CPU；这里的 MB 是累计分配流量，不是常驻 RSS。另一个 4-thread、默认 13 buckets、128-key chunk、
每轮 10240 observations 的隔离基准中，逐 key 原子更新每轮约 3.9~4.1ms，chunk 聚合约
0.039~0.041ms，count/sum/buckets 完全一致。该数字只衡量 histogram 自身，不应外推成完整 API 倍数。

随后使用与 5.7 相同的 Release/O2 二进制、纯 local metadata、真实 HTTP benchmark，连续运行三次并取
各项中位数：

| blocks | 本轮 worker=4 串行 p50/p99 | 5.7 串行 p50/p99 | 本轮 worker=4 16-way p50/p99 | 5.7 16-way p50/p99 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.51/0.54ms | 0.56/0.59ms | 6.26/13.51ms | 8.47/17.51ms |
| 1000 | 1.12/1.15ms | 1.32/1.35ms | 5.91/10.48ms | 6.47/10.87ms |
| 5000 | 3.10/3.17ms | 4.54/4.72ms | 10.32/23.56ms | 21.12/40.57ms |
| 20000 | 10.28/10.47ms | 16.07/16.18ms | 23.95/34.25ms | 77.87/101.60ms |

20k 串行 p50/p99 下降约 36%/35%，16-way p50/p99 下降约 69%/66%；三次 20k 串行 p50 为
10.26/10.30/10.28ms，结果稳定。最后一批 20k 并发请求的 gauges 随单请求调度不同落在：
`get_io_time_us=4.7~8.4ms`、`host_projection_time_us=2.2~3.3ms`、外层
`prefix_match_time_us=8.7~12.9ms`。这证明两项分别降低了 metadata read 内共享原子竞争和 read 后 URI
投影；它不改变 ReportEvent 写入语义，也不缓存每 block 的解析对象。

该 benchmark 是空闲服务的 local 对照，不含线上 15 QPS、约 10k blocks/request 的 L2 ReportEvent
混合压力。上线后仍需用同一压测流量重点比较 `meta_indexer.get_io_time_us`、
`meta_searcher.host_projection_time_us` 和 Get p99；若混合负载仍远高于该基线，再依据分段指标检查 local
LRU/item lock，而不是重新增加无界 worker。

### 5.9 2026-08-05 混合压力复核与并行位图组合回归

在当前 Release/O2、纯 local metadata、真实 HTTP 服务上继续做两组隔离验证。第一组持续 75 秒，使用
4 个 reporter 混合 20 QPS ADD（100 blocks/request）、10 QPS DELETE（50 blocks/request）、2 QPS
10k-key Get、5 秒 heartbeat，并在 35 秒周期 snapshot 中主动遗漏 10% 当前数据验证 authoritative cleanup。
最终 1503 ADD、752 DELETE、8 snapshot、151 Get 和 60 heartbeat 全部成功，影子状态逐事件校验无误；
10k Get 客户端 p50/p95/p99 为 5.43/20.35/25.18ms，最大 101.20ms 出现在大 snapshot 并发窗口。

第二组按线上复现参数发送约 15 QPS、约 10k blocks/request 的 L2 ADD，持续 45 秒。单个 Python 进程
同时承担大 JSON 构包、writer、逐 ADD 的 10k-key 正确性查询和独立 reader 时，客户端统计出现约 2.7s
的 Get p99；但三种小请求的延迟同时抬升，说明该值包含本地 GIL/worker 排队。以服务端 access log 作为
业务处理边界重新统计，同一阶段结果为：

| 服务端请求 | count | avg | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| GetHostCacheState（全部） | 753 | 7.43ms | 11.35ms | 12.83ms | 26.00ms |
| GetHostCacheState（独立 10k reader） | 46 | 7.11ms | 10.16ms | 10.28ms | 10.47ms |
| ReportEvent | 707 | 73.77ms | 96.97ms | 117.87ms | 143.41ms |

因此同进程压测器的 HTTP wall time 不能直接当作服务端 Get RT。大写入结束后立即重新运行连续命中
benchmark，20k Get 串行 p50/p99 为 8.84/8.95ms，16-way p50/p99 为 23.39/35.07ms，未观察到
allocator/LRU 经高分配压力后的持续退化。

另外补充一个确定性参考模型 UT，将此前分别覆盖的两个维度叠加：70 个 candidate host 横跨两个
64-bit presence word，384 个 key 触发 metadata query executor 并行路径，每个 host 使用不同的预期
prefix，并同时检查 Eagle-pop 和 medium filter。该用例完成 Release 100 轮、ASAN 20 轮以及完整
MetaSearcher Release/ASAN 回归，结果一致。它主要防止 packed presence 的跨 word 索引、并行 slice 写入
或 prefix reduction 在后续优化中发生静默错位。

### 5.10 2026-08-06 fused targeted RMW、纯 local 回归与同吞吐分批 A/B

本轮工作位于 `codex/report-event-write-performance`，开始前已 fetch 并 rebase 到
`origin/main@ae9cb0dfa593071be47e0a601af05d8159b383b7`；原查询优化分支未改动。部署明确使用纯 local
metadata，因此本节的功能、ASAN 与性能结论只以 local 为交付门槛，不把真实 Redis 结果混入结论。

#### 单请求 before/after

在同一 Release/O2、真实 HTTP、全新 local instance 上运行
`test_20_large_single_request_delta_scaling`。before 是 rebase 后、fused 改动前的基线；after 是最终链接
产物三次独立串行复测的中位数。每档均校验 committed token 和首/中/末 block 的最终 URI：

| events/request | before create/update | after create/update | update 降幅 |
| ---: | ---: | ---: | ---: |
| 100 | 1.57/1.59ms | 1.53/1.40ms | 11.9% |
| 1000 | 9.41/10.16ms | 9.20/8.73ms | 14.1% |
| 5000 | 43.75/48.26ms | 42.34/40.99ms | 15.1% |
| 20000 | 172.00/202.12ms | 169.75/172.09ms | 14.9% |

create 路径没有旧的第二阶段，after 基本持平，说明新 key-status 语义和 flat bookkeeping 没有引入明显
回归；收益集中在已有 location。三轮 20k create/update 区间为 `166.81~191.29ms` /
`170.75~189.44ms`，因此应看中位数和线上分桶，不能拿单次最好值当 SLA。20k 的 KVCM
service/access-log timer 从 before create/update `81.642/113.413ms` 变为两轮 after 平均
`79.28/85.60ms`，约下降 `2.9%/24.5%`。客户端 update 中位数下降约 14.9%，剩余差异主要在 HTTP
JSON/protobuf 边界和 worker 排队，不能继续算作 metadata RMW 收益。

#### 相同约 150k blocks/s 的 batch-size A/B

使用 `tools/scripts/report_event_load.py`、16 reporter、纯 local、真实 HTTP，把总 blocks/s 保持接近，
同时发送 9k-key Get 和 heartbeat。三轮 ADD、Get、heartbeat 均 `failed=0`，每个 ADD 还抽查一个最终 key；
结果如下：

| ADD 形态 | 成功数 / 实际 QPS | ADD avg/p95/p99 | 9k Get avg/p95/p99 | Heartbeat avg/p99 |
| --- | ---: | ---: | ---: | ---: |
| 10k × 15QPS | 317 / 14.41 | 364.49/1268.12/1661.32ms | 271.45/1050.49/1254.13ms（n=5） | 54.66/489.21ms |
| 5k × 30QPS | 457 / 29.60 | 98.84/197.97/313.58ms | 29.56/150.72/152.65ms（n=16） | 12.55/82.22ms |
| 2k × 75QPS | 1149 / 74.62 | 50.04/123.93/174.86ms | 18.71/56.68/99.07ms（n=16） | 14.36/87.62ms |

10k 轮的 Get 只有 5 个样本，不能把其 percentile 当成精确 SLA；但 ADD 样本数足够，三个 API 的尾延迟
又同向变化，足以说明超大 HTTP 请求的长任务、瞬时分配和共享执行资源占用是主要放大器。相同吞吐下，
2k batch 的 ADD p99 比 10k batch 低约 89%，本轮 9k Get p99 也降到约 99ms。当前推荐发布端先以
`2k blocks/request` 为起点做线上 A/B；5k 可作为降低 QPS/请求数的折中。该建议不是服务端硬限制，也
不是无条件的 SLA 保证，仍需按线上 CPU、HTTP worker 数和 URI 大小复测。

这组数据同时否定了“先在 ReportEvent 内继续加线程”的必要性：外部分批已经在不改变写入语义的前提下
显著改善公平性；服务端 writer 并行会抢占 Get 使用的 CPU、allocator 与 local LRU/item lock。若线上
2k 分批后仍不满足，下一轮先补 HTTP parse/serialize/worker queue 和 item-lock wait 指标，再决定是否
做独立有界 writer executor。

#### 本轮验证矩阵

- HTTP 功能：`test_report_event_snapshot.py --skip-bench`，36/36 通过，覆盖 ADD/DELETE/SNAPSHOT、
  last-operation-wins、校验无副作用、部分失败/重试、并发 snapshot/delta、首次 delta 与清理；
- Release UT：纯内存 meta/manager 全包 23/23 通过；`CacheManagerTest` 的 10 个 shard 全部通过；
- 生命周期竞态：HOST_DOWN 拦截已 admission 的 delta、旧 lifecycle 不能跨重注册写入，两项连续 50 轮
  通过。测试 backend 已显式拦截新的 `GetLocationsWithKeyStatus`，不能只 hook 旧 `GetLocations`；
- 容量边界：新 key、已有 key 缺 location、已有 target update、max capacity，以及“已有更新 + 超容量新
  key 位于同一个 internal batch”均通过；最后一种返回 `EC_OK/EC_NOSPC`，`key_count` 与 usage 不漂移；
- ASAN：`meta_indexer_test`、`meta_local_backend_test`、`meta_dummy_backend_test`、
  `meta_storage_backend_manager_test`、`MetaSearcherTest` 全部通过；ReportEvent/HOST_DOWN/并发相关
  `CacheManagerTest` filter 通过；
- 性能与数据正确性：100/1k/5k/20k create/update 单请求通过；10k/5k/2k 同吞吐持续压测全部零失败。

新增原语不改变协议、URI 或持久化 schema，回滚可以整体 revert 本轮 ReportEvent commit，恢复原两阶段
RMW。回滚/后续修改时必须一起处理 `MetaStorageBackend::GetLocationsWithKeyStatus`、manager recovery
路由、`MetaIndexer::ReadModifyWriteTargetLocations` 和 `BatchMergeLocationSpecs`，不能只删除其中一层；
否则最容易出现的是 `key_count` 漂移或 lifecycle fence 窗口被重新打开。

### 5.11 2026-08-06 百万 key 纯 local 查询优化

本轮位于独立分支 `codex/gethost-million-key-performance`，基于 5.10 的 ReportEvent 分支头创建，未继续
修改 ReportEvent 写入流程。目标是让连续全命中的百万 key 查询保持有界内存和可取消性，同时收敛纯
local LRU 上逐 key 加锁的固定成本。实现要点如下：

1. local backend 提供紧凑结果：一个扁平 `shared_ptr<const CacheLocation>` 数组加每 key offset，避免为
   百万 key 创建百万个外层 vector/map。其他 backend 使用兼容 fallback，不改变 Redis/cached recovery；
2. `MetaIndexer::VisitLocationValuesForPrefix` 以连续范围渐进读取。首个范围同步完成，用来建立候选 host；
   后续范围才提交共享有界 executor。普通 prefix 或 Mamba 的所有候选已经终止时，通过原子 stop index
   阻止尚未领取的后缀工作；
3. metadata 读取和 host/spec 投影在 visitor 内融合，不再先保存百万 key 的 location 集合、随后做第二次
   全量遍历。普通查询只保留每 host 的 prefix stop；Mamba 只保留后续 Eagle pop 所需的状态位矩阵；
4. 只有 prefix 内的 `EC_NOENT` 是正常 miss。答案终止位置之前的硬错误仍返回；终止位置之后已经开始的
   speculative 读取即使失败，也不能推翻已经确定的短 prefix；
5. EventReport location id 与 URI 使用只读 `string_view` 解析，并通过透明比较直接查询 snapshot，避免
   对每个 key 复制 reporter medium/host 和 generation token；
6. advanced cache 增加带默认 fallback 的 batch lookup/release，LRU 实现先按内部 shard 分组，每个
   shard 一次持锁完成该批 lookup 或 release。引用计数、重复 key、被 pin 时 erase、LRU reinsertion 和
   capacity eviction 语义与逐 key API 相同；
7. 纯 local metadata 读取范围固定至少为 4096 key，以摊薄默认 1024 个 LRU shard 的锁获取。这个数与
   projection 的 128-key CPU chunk 解耦。首范围仍有界，因此短 prefix 不会扫描完整百万 key 后缀；代价是
   极短 miss 最多多读一个 4096-key 窗口。

使用 Release/O2、纯 local backend、同一进程直接调用 manager 内部查询链路（不含 HTTP JSON、protobuf
响应序列化和网络），最终结果如下。每项先 warmup，再重复采样并报告 p50/平均值：

| case | 100k p50/avg | 500k p50/avg | 1M p50/avg |
| --- | ---: | ---: | ---: |
| metadata only | 7.21/7.34ms | 39.90/39.98ms | 84.73/90.02ms |
| 全命中完整 host 投影 | 9.05/9.04ms | 48.61/48.69ms | 108.76/113.54ms |

同一 benchmark 中，第 1024 个 key 结束 host prefix 的 1M 请求为 p50/avg `0.483/0.497ms`；第 1024 个
key metadata miss 为 `0.438/0.438ms`。批量 LRU 前的同机 1M metadata/all-hit 约为
`132.1/155.1ms`，最终分别下降约 36%/30%。这里不能解读成完整 HTTP API 已保证 100ms：1M metadata
读取已低于 100ms，但全命中投影仍约 109ms p50，HTTP 边界还会额外增加延迟。

可复现命令：

```bash
bazelisk test -c opt //kv_cache_manager/manager/test:GetHostCacheStateBenchmark \
  --test_output=streamed --test_arg=--gtest_also_run_disabled_tests
```

该 benchmark 默认带 `manual` tag 且测试名为 `DISABLED_...`，不会拖慢常规 CI。修改 compact layout、
visitor stop/error 规则、LRU batch 引用管理或 4096 窗口时，必须同时运行 LRU、QueryExecutor、
MetaLocalBackend、MetaIndexer、MetaSearcher 五组回归及本 benchmark；不能只比较全命中吞吐而忽略短 prefix
取消延迟。

### 5.12 package 启动脚本预加载 jemalloc

`package/script/start_server.sh` 默认尝试在启动时预加载 jemalloc，以降低大批量 ReportEvent 与百万 key
查询产生的短生命周期分配对系统 allocator 的压力。它只影响通过 package 启动脚本拉起的进程；直接运行
Bazel 二进制不会自动启用。控制项和降级规则如下：

- `KVCM_USE_JEMALLOC=0` 显式禁用；其他值或未设置时尝试启用；
- `KVCM_JEMALLOC_PATH` 非空时优先尝试该路径；否则/失败后，x86_64 依次检查
  `/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`、`/usr/lib64/libjemalloc.so.2`，aarch64 依次检查
  `/usr/lib/aarch64-linux-gnu/libjemalloc.so.2`、`/usr/lib64/libjemalloc.so.2`；
- 找不到可读库或架构不受支持时打印告警并继续使用默认 allocator，不能因此阻止服务启动；
- 已有 `LD_PRELOAD` 会保留，jemalloc 放在最前；若已经包含同一个解析路径则不重复追加；
- `configure_jemalloc` 当前在 `install_kvcm_ops` 之前执行，因此同一启动脚本中的 pip 子进程也会继承
  `LD_PRELOAD`。后续若只想影响 server，应调整调用顺序并重新验证 package 启动，而不是改变变量作用域后
  假定 `exec` 仍会继承。

这项改动只选择 allocator，不构成性能 SLA。上线应同时比较 Get/ReportEvent p50/p99、RSS、CPU 与 allocator
相关崩溃；回滚可设置 `KVCM_USE_JEMALLOC=0`，无需重新打包。最低提交门禁包括 `bash -n`，以及禁用、默认
探测、自定义路径、已有 preload、缺库/未知架构降级的隔离函数测试。
