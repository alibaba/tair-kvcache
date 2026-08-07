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

### 5.13 2026-08-06 ReportEvent 请求内热路径收敛

本轮位于独立分支 `codex/report-event-hotpath-optimization`，直接基于
`codex/gethost-million-key-performance@6c44b5025f5aeefbfa663cf94c39c915f4966314` 创建；查询优化分支未被
修改。本节只讨论纯 local metadata、Release/O2、直接运行 Bazel 二进制（未预加载 jemalloc）的结果，
before/after 使用相同进程形态，因此 allocator 条件一致。

#### 设计与实现

1. delta fold 使用三个 request-wide 连续数组保存稳定 `(block, location)`、最终 spec mutation 和事件重试
   依赖，并用一个哈希索引定位 location。常见的一 event/一 spec/一 location block 不再创建三组小 vector；
   最后只排序 location 索引并直接构造 ADD/DELETE task。last-operation-wins、ADD 先于 DELETE 的持久化阶段、
   admission failure 和整组重试闭包语义保持不变；
2. 同一请求的 `medium -> location_id` 只构造一次。location 索引中的指针只指向这个 request-owned intern
   map 的 mapped string；`unordered_map` rehash 不会使元素引用失效，且 map 在整个 fold/task 构造阶段都
   存活。后续不能把该指针改为指向临时字符串或可能搬迁元素的 vector；
3. `DeltaMutationGuard` 缓存并返回 request-owned lease，后续事件只读其 snapshot token/generation，不再为
   每个 event 复制 32-byte token。lease 仍由 guard 析构时成对结束，generation-pinned lifecycle fence
   窗口不变；
4. 入口完整解析 URI 时同时保存 `size` 和已经验证的 parsed URI；追加 `s_version` 使用
   `ToUriStringWithExtraParam` 直接生成与 `SetParam + ToUriString` 完全相同的 canonical sorted URI，不再
   clone/mutate 参数 map 或使用 `ostringstream`。`StandardUri` 的 query 解析改用 `string_view`，host/port
   也不再先复制一份中间字符串；
5. ReportEvent 内部 task 把已验证 spec 的总 size 传给 `BatchMergeLocationSpecs`，避免在 shard RMW 前再次
   parse 新 URI。只有该内部路径允许设置 `prevalidated_total_size`；普通 caller 留空后仍执行完整的 URI、
   spec name、duplicate name 和 snapshot-version 一致性校验。这个 hint 不能扩展为对外部输入跳过验证；
6. pure-local targeted read 使用 LRU batch lookup/release，把同 shard 的数千次 lookup mutex 获取合并。
   unconditional Upsert 对唯一 key 同样 batch lookup、原地更新、统一 release，再插入 miss；release 必须在
   插入 miss 前完成，以保留 strict-capacity eviction。公共 backend API 若出现重复 key，会自动退回原逐项
   Upsert，保持“同一 batch 内先创建、后续 partial update 继续 merge”的请求顺序语义；
7. HTTP handler 直接把 request body 的 `string_view` 交给 protobuf JSON parser，删除整份大 body copy。
   parser 在 handler 栈内同步完成，view 不会越过 request body 生命周期。

这些改动没有新增 writer 线程、没有把 manager callback 放进 backend item lock，也没有修改协议或 metadata
schema。Local batch lookup 会同时 pin 一批 handle，但只在本次同步投影/更新期间持有；所有 handle 均通过
`ReleaseBatch` 一次释放。新增 `StandardUri` serializer 用精确输出对照测试锁定 canonical 兼容性，不能仅以
“URI 语义等价”为由改变输出顺序。

#### 纯 local 单请求 before/after

真实 HTTP benchmark 使用 `test_20_large_single_request_delta_scaling`，每档先创建 block，再对相同 location
执行 update，并检查 committed token 和首/中/末 key。before 为本分支创建后立即记录的基线；after 为最终
Release 链接产物三个全新 instance 的串行复测中位数：

| events/request | before create/update | after create/update | create/update 降幅 |
| ---: | ---: | ---: | ---: |
| 100 | 1.53/1.38ms | 1.47/1.34ms | 3.9%/2.9% |
| 1000 | 9.47/9.00ms | 8.21/7.80ms | 13.3%/13.3% |
| 5000 | 43.54/42.57ms | 37.33/35.14ms | 14.3%/17.5% |
| 20000 | 173.87/175.00ms | 145.85/145.87ms | 16.1%/16.6% |

最后一次安全审查和重新链接后的三轮 20k create 为 `145.27~147.59ms`，update 为
`145.32~146.53ms`。
结果仍近似 O(events)，本轮降低了每 event 常数，没有把大 JSON API 变为固定耗时，也不能据此承诺线上
混合负载 p99。

最后一轮 20k existing-location update 的服务端指标为：

| 指标 | 数值 |
| --- | ---: |
| `service.query_rt_us` | 58,473us |
| `meta_searcher.indexer_read_modify_write_location_time_us` | 35,602us |
| `meta_indexer.rmw_get_io_time_us` | 7,061us |
| `meta_indexer.upsert_io_time_us` | 6,715us |
| `meta_indexer.lock_wait_time_us` | 1us |
| `meta_searcher.index_deserialize_time_us` / `index_serialize_time_us` | 0/0us |

同一次请求的 Python 客户端 HTTP wall time 为 145.32ms。两者之间约 87ms 包含客户端 JSON 编码、loopback
HTTP、服务端 protobuf JSON parse/response serialization 及统计边界差异，不能全部归到某一个环节；但可以
确定纯 local 下不是 Redis、location deserialize/serialize 或 metadata shard lock wait 主导。下一项有望
立竿见影的优化应先补 `http_parse_us/http_serialize_us` 并做同 payload 的 gRPC A/B，而不是继续增加
ReportEvent writer 并行度。

#### 本轮验证矩阵与后续门禁

- HTTP 纯内存功能集成 36/36 通过，覆盖 ADD/DELETE/SNAPSHOT、同请求 last-op-wins、部分失败重试闭包、
  snapshot/delta 并发、HostDown/重注册、首次 delta、异常 URI/payload 无副作用；
- Release/O2 的 StandardUri、MetaLocalBackend 和完整 CacheManager 分片测试通过；重复且非相邻 key 的
  Upsert 专项测试验证请求顺序 merge；
- ASAN 下 `StandardUriTest`、`meta_local_backend_test`、`CacheManagerTest`、`SnapshotUriUtilsTest`、
  `MetaSearcherTest`、`ProtoMessageJsonUtilTest` 全部通过；
- 100/1k/5k/20k create/update 三次性能复测全部通过数据校验，没有错误返回或规模拐点。

后续若修改连续数组索引、location-id intern、prevalidated size、batch handle 生命周期或 HTTP body view，
至少重跑上述六个 ASAN 目标和 36 项 HTTP 功能套件。当前没有真实 Redis 验证，不能把本节数据外推到
cached/redis backend；也不要在没有混合 ReportEvent/Get/heartbeat p99 对照前增加内部 writer 并行。

### 5.14 2026-08-06 热路径优化后的第二轮正确性审计

本节记录 `codex/report-event-hotpath-optimization` 在 5.13 提交后的继续审查结果。它不是另一轮并行化，
而是针对批处理顺序、可信输入边界、整数上溢和借用内存生命周期逐层构造反例。后续 AI 不应只重跑 happy
path benchmark 后删除这些保护。

#### 审计中发现并修复的问题

1. `MetaLocalBackend::Upsert` 的初版 batch lookup 会先更新所有 hit，再插入所有 miss。在 strict-capacity
   local cache 中这会改变可观察的请求顺序：`[new key, existing key]` 原本可以先接纳新 key、再原地扩展旧
   key，重排后却可能让新 key 返回 `EC_NOSPC`。最终实现对重复 key 继续逐项处理；全 hit 和全 miss 保留
   batch fast path；混合 hit/miss 按原下标更新/插入并逐个释放已有 handle。全 miss 不再调用只包含空
   handle 的 `ReleaseBatch`，避免一次无意义的 shard 分桶分配；
2. `MergeLocationSpecsTask::prevalidated_total_size` 最初是公开的 `optional<uint64_t>`，任何 caller 都能伪造
   值并绕过 URI/spec/version 校验。现在它是只能由 `CacheManager` 构造的 capability token；普通
   `MetaSearcher` caller 无法创建 token，必须走严格验证。不要为了测试方便重新开放它的构造函数；
3. `StandardUri(const string&)` 忽略 `Parse()` 返回值。旧 parser 遇到非数字端口时会保留 protocol，导致
   `Valid()==true`，随后追加 snapshot version 又把非法端口静默丢掉。现在端口直接在原字符区间上用
   `from_chars` 严格解析，非数字、负数（包括数值等于 0 的 `-0`）、空端口、前导空白和 `+` 均
   fail-closed；保留仓库已有的 `:0`
   兼容语义。authority 中的 `@`/`:` 只在 path/query 之前解释，query value 中的 callback URL、`/` 和邮箱
   地址不再污染 user-info/path；
4. 多 spec 的 `size` 以前直接做 `uint64_t` 加法，恶意或损坏输入可以回绕并破坏 storage usage。单个 ADD
   event 和完整 SNAPSHOT 在获取 reporter generation/write gate 前按整个输入做 checked sum；跨多个 ADD
   event 折叠后若才发生上溢，则不信任预计算 hint，退回 `MetaSearcher` 严格校验并按既有“首次 metadata
   写失败仍复用 generation”的语义返回失败。通用 merge/replace 校验也拒绝上溢，Replace 复用校验得到的
   total，删除一次写阶段重复 URI size 解析。第二轮反例还覆盖“已有 location 的 size 合法、当前 ADD 的
   size 也合法，但两者合并后才上溢”：target-location modifier 在写入前计算 retained + incoming 的 checked
   sum，返回 `EC_BADARGS` 并保持原 metadata/usage 不变；
5. HTTP body 的 `string_view` 解析新增非 NUL 结尾、带前后垃圾 backing string 的边界测试，证明 protobuf
   parser 严格使用 view 长度且不读取尾部。`req.get_body()` 的 view 仍只在同步 handler/parser 栈内使用，
   不能缓存到异步任务或 response 生命周期之后。

#### 新增的模型与反例

- 768 个确定性伪随机 ADD/DELETE event、48 个 block、17 个 medium、4 个 spec name，与独立 map 参考模型
  比较最终 canonical URI、last-op-wins、spec 排序、location 数量、`spec_size` 和总 storage usage。17 个
  medium 会强制 request-owned intern `unordered_map` rehash，用来验证 location-id 指针在 rehash 后仍有效；
- optimized batch Upsert 与逐项 `UpsertForOneKey` 做差分，覆盖全 hit、全 miss、混合、非相邻重复 key，比较
  每项错误码、location、除动态 `BP#lru_time` 外的全部 properties 及内存计数；另用 1 MiB strict capacity
  分别锁定 `[new, existing] -> [OK, OK]` 和 `[existing, new] -> [OK, NOSPC]`；
- 非法端口、单 event size 上溢、跨 event fold 后 size 上溢、已有 spec 与新 spec 合并后上溢、
  snapshot/Replace size 上溢均验证无错误 metadata 写入。跨 event 上溢还验证失败响应保留可复用
  generation，这与已有
  `TestReportEventFirstDeltaMetadataFailureReportsFailureAndReusesGeneration` 契约一致，不应误改成失败即删除
  generation；
- `PrevalidatedTotalSize` 的私有构造保证 ReportEvent 之外的测试和生产 caller 仍走完整校验，而不是仅依赖
  注释约定。

#### 验证结果和环境限制

- Release/O2 完整通过 10 个纯内存目标：`StandardUriTest`、`LruCacheTest`、
  `SnapshotUriUtilsTest`、`EventReportBackendTest`、`meta_local_backend_test`、`meta_indexer_test`、
  `meta_storage_backend_manager_test`、`MetaSearcherTest`、`CacheManagerTest`、
  `ProtoMessageJsonUtilTest`；
- 相同 10 个目标 ASAN 全绿。UBSAN 下 URI、snapshot URI、local backend、MetaSearcher、JSON 边界及定向
  ReportEvent 用例全绿；完整 `CacheManagerTest` 中依赖第三方 `cpp_stub` 改写函数机器码的 3 个测试因
  `external/cpp_stub/stub.h:456` 非对齐写入被 UBSAN 阻止，这发生在 mock 安装阶段，不在本轮生产路径。
  TSAN 在链接 gRPC 时因环境缺失 `/usr/lib64/libtsan.so.0.0.0` 无法启动，不能记录为通过；
- Release 下并发读写、ReportEvent/HostDown 并发、flat fold、容量顺序、折叠/最终合并上溢和写失败传播的
  关键组合重复 50 轮；`CacheManagerTest` 分片合计执行 500 次，另对并发可见性和失败依赖闭包合计执行
  200 次，全部通过；
- 最终源码重新构建的新进程上，真实 HTTP、纯 local metadata 的双类型基础套件 20/20、
  snapshot/并发/失败套件 36/36 全绿。随后 10 秒影子状态混合压力完成 401 次 100-block ADD、198 次最多
  50-block DELETE、201 次 1000-key Get、40 次 heartbeat，所有请求和写后抽样校验零失败；ADD
  p50/p99 `2.50/4.69ms`，Get p50/p99 `2.27/10.24ms`；
- 最终源码无并行编译负载时三轮 20k 单请求 create/update 为 `147.26/145.87ms`、`147.27/146.39ms`、
  `145.51/145.16ms`。一次与两个大 Bazel build 重叠的 `315.50/344.60ms` 已明确标记为环境噪声，不用于
  before/after 结论。性能复测前必须确认没有编译器/linker/其他 benchmark 占用 CPU。

本轮仍只验证 pure local metadata；按用户部署前提没有启动 Redis。local 结果不能证明 cached/redis 语义，
但本轮也没有修改 Redis 专属实现。若以后改变 mixed Upsert 分支、capability token、checked size、URI parser
或 borrowed HTTP body，必须至少重复本节的差分模型、sanitizer 定向用例和 36 项 HTTP 套件。

### 5.15 2026-08-06 TSAN 关闭期审计与写读混合 A/B

本节是在 5.14 之后继续检查 `codex/report-event-hotpath-optimization` 的结果。测试仍只使用 pure local
metadata。系统补装 `libtsan-10.2.1-3.3.alios7.x86_64` 后，5.14 记录的 TSAN 环境限制已经解除；这只是
本机测试依赖，不是仓库或发布包改动。

#### TSAN 发现的两个关闭期问题

1. `EventReportBackend::Close()` 先收集 `LifecycleFence::mutex` 的 `unique_lock`，再清空
   `lifecycle_fences_`。`unique_lock` 不拥有 mutex 对象，原实现可能先析构最后一个 fence，再由 lock vector
   解锁已经释放的 `shared_mutex`。TSAN 在
   `LivenessUnregistersBeforeCleanupAndHeartbeatCannotReviveOldSnapshot` 中稳定报 heap-use-after-free；
2. 仅增加 fence 的强引用仍不够。原 `Close()` 在持有 `lifecycle_fences_mutex_` 时阻塞等待每个 fence 的
   writer lock，能形成真实三线程环：host cleanup 持有 lifecycle read lease 后等待 metadata；已进入的
   metadata RMW 持有 metadata 后查找 lifecycle fence；`Close()` 持有 fence table mutex 后等待 cleanup
   的 lifecycle lease。delta modifier 使用 `try_lock` 只能切断直接的 metadata/lifecycle 两锁环，不能切断
   `Close()` 引入的第三条边；
3. 最终实现只在 `lifecycle_fences_mutex_` 下复制 `shared_ptr<LifecycleFence>`，释放 table mutex 后才等待
   每个 fence，清空节点状态后再短暂获取 table mutex 清表；显式先销毁 lock vector，再销毁强引用。
   `Close()` 在任何阻塞等待期间都不持有 table mutex；
4. 所有可能在 `Close()` 设置 `retired_` 后才返回或新建 fence 的入口均已逐个审计。它们在获得 fence 后、
   修改 node/snapshot 状态前再次检查 `Retired()` 或 `AcceptingReports()`。因此 table snapshot 之后创建的
   fence 只会得到关闭错误，不会越过关闭边界写状态。以后若新增 `GetOrCreateLifecycleFence()` caller，必须
   保留这次二次检查；不能用一次函数入口检查替代。

以后修改这里必须同时保持三个不变量：不在持有 fence table mutex 时等待 reporter fence；保留 fence 的
强引用直到对应 lock 已释放；可能跨过 `Close()` 的 caller 在 fence 内再次检查 retired/available 状态。

#### Sanitizer 与 Release 验证

- TSAN：完整 `EventReportBackendTest`、8 个 ReportEvent/snapshot/HostDown/Get 跨层并发用例、MetaLocal
  batch Upsert/compact read 及 MetaIndexer 多线程/提前终止用例全绿。最初触发 UAF/锁环的 4 个用例最后各
  重复 20 次；EventReportBackend 20/20，分片后的 CacheManager 合计 200/200，无 data race、UAF 或
  lock-order-inversion；
- ASAN+UBSAN：`LruCacheTest`、`StandardUriTest`、`EventReportBackendTest`、`SnapshotUriUtilsTest`、
  `query_executor_test`、`meta_local_backend_test`、`meta_indexer_test`、`MetaSearcherTest`、
  `ProtoMessageJsonUtilTest` 全量通过；`CacheManagerTest` 10 个 shard 全量通过。当前构建会把同一个
  generated protobuf 常量从两个 DSO 注册给 ASAN，因此使用 ASAN 建议的
  `detect_odr_violation=0`；UBSAN 只 suppress `external/cpp_stub/stub.h` 的 alignment 检查，因为该测试库
  本来就通过非对齐机器码写入安装函数 stub，仓库自身路径仍为 `halt_on_error=1`；
- Release/O2：上述 9 个目标加完整 `CacheManagerTest`，共 10 个目标全绿；最终重新链接的真实 HTTP
  进程上，snapshot/并发/失败套件 36/36 全绿；
- 本轮 ASAN 首次运行若不关闭 protobuf ODR 检查，会在测试 main 之前退出；完整 CacheManager UBSAN
  若不 suppress `cpp_stub`，会在 mock 安装阶段退出。这两种已知测试基础设施诊断不能当作生产代码失败，
  也不能直接忽略后宣称 sanitizer 通过，必须按上面的精确范围重跑。

#### ReportEvent 优化对 Get 的 2×2 A/B

为了验证写侧优化不会挤压用户更关心的 Get，使用同一台机器、Release/O2、全新 pure-local instance，比较
父提交 `6c44b5025f5aeefbfa663cf94c39c915f4966314` 与当前实现。每轮 30 秒、8 reporter、15 QPS × 10k
BLOCK_ADD、2 QPS × 10k-key standalone Get、5 秒心跳，所有请求和 shadow-state 校验均零失败。

注意 `report_event_load.py` 会在每个 ADD 后用同一批 10k key 再调用一次 Get 并校验完整 prefix，且 `add`
统计从 ReportEvent 开始一直覆盖到该校验结束。因此这里的 `add` 不是纯 ReportEvent HTTP RT；实际查询
压力约为 15 QPS 的写后 10k-key Get 加 2 QPS standalone Get。这个模型故意比线上 0.1 QPS Get 更苛刻，
适合观察写优化是否伤害查询，但不能用 `add` 数字替代独立 ReportEvent benchmark。

| 版本 | ADD 实际 QPS | ADD avg | ADD p50 | Get 实际 QPS | Get avg | Get p50 | Get p95 | Get p99 | Get max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| before round 1 | 12.60 | 1157.91ms | 489.22ms | 1.70 | 605.96ms | 144.49ms | 1993.64ms | 6175.34ms | 7163.79ms |
| before round 2 | 13.07 | 982.79ms | 406.74ms | 1.74 | 828.08ms | 120.25ms | 3645.72ms | 4322.02ms | 4510.07ms |
| current round 1 | 13.19 | 906.85ms | 314.52ms | 1.78 | 458.93ms | 78.01ms | 2037.09ms | 2939.51ms | 3223.37ms |
| current round 2 | 13.41 | 770.95ms | 343.81ms | 1.81 | 470.10ms | 89.03ms | 1961.27ms | 3176.85ms | 3855.16ms |

按请求数加权，ADD+写后校验平均从 `1071.20ms` 降到 `838.30ms`，下降 21.7%；standalone Get 平均从
`715.23ms` 降到 `464.56ms`，下降 35.0%。两轮 current 的 Get p50 都低于两轮 before，p99 也从
`4.32~6.18s` 收敛到 `2.94~3.18s`。第一组配对的 Get p95 有约 2.2% 反向波动，第二组明显改善；不能从
约 60 个 Get 样本推导精确 percentile SLA，但 2×2 的平均、p50、p99 和完成吞吐一致表明：在固定目标写
负载下，当前 ReportEvent 优化对查询是正向的，没有发现以增加 writer 并行度换吞吐、反而抢占 Get 的情况。

绝对尾延迟仍不合格：10k 大批次加每写一次完整 10k-key 校验会把服务推入排队区。当前结果支持“优化没有
伤害查询”，不支持“150k blocks/s 下 Get 已满足 100ms SLA”。线上仍应优先把 ReportEvent 外部分成
2k/5k block 批次，并按固定 blocks/s 重测 Get p99。

#### 最终混合正确性与日志边界

最终源码另跑 45 秒混合压力：1357 次 100-block ADD、673 次最多 50-block DELETE、24 次周期 SNAPSHOT
（每次主动遗漏 10% 当前 key 验证权威对账）、227 次 5k-key Get、72 次 heartbeat，全部请求和写后
shadow-state 校验零失败。Get avg/p50/p95/p99/max 为
`5.24/3.67/11.70/12.72/86.03ms`；ADD 为 `4.16/2.54/4.33/54.37/237.09ms`，snapshot 平均
`191.59ms`。

压测停止心跳并等待 liveness 清理时，多个 host 会并行扫描同一个 local index。一个 cleanup 的 `Scan`
返回 key 后，另一个 cleanup 可能先删除该 key，使前者的 `GetLocations` 返回包含 `EC_NOENT` 的
`EC_PARTIAL_OK`。当前 `CleanupLocationsByHost` 会立即把任何 `EC_PARTIAL_OK` 记为 failure，所以外层可能
记录 `finished with partial failures`，即使后续逐 location 已把 `EC_NOENT/EC_MISMATCH` 当作幂等成功且没有
底层 delete error。这是并发清理的保守/偏噪声日志，不能仅凭该汇总 warning 判断 metadata 丢失；后续若
收敛日志，应只把非 `EC_NOENT` 的 per-key read error 计为真实 failure，并补 Scan/Get 竞态测试，不要改变
cleanup 的 generation lease 或 conditional-delete 语义。

本节没有启动 Redis，也没有把 local 结果外推到 cached/Redis backend。后续 AI 至少应保留：36 项 HTTP
功能套件、固定 blocks/s 的写读混合 A/B、TSAN 关闭期复现，以及 sanitizer 对第三方 ODR/stub 的精确处理。

### 5.16 2026-08-07 perf/futex 驱动的 pure-local 单 location RMW 优化

本节基于 `codex/mu-main` 的 Release/O2 产物重新采集 CPU 与 futex 栈，目标只覆盖用户线上采用的 pure-local
metadata 模式。结论是：大请求确实会在 metadata shard mutex 上形成可见等待，但等待的根因是锁内做了过多
通用 RMW 构造、local LRU 的逐 key lookup/release 以及 allocator 工作；单纯把 mutex 换成原子变量或增加
writer 线程既不能保护多字段 RMW，也会破坏 key-count、capacity 和同 key 顺序语义。

#### 优化前的证据

- 20k `BLOCK_ADD` create/update 的 HTTP wall time 都约为 `145~147ms`；
- CPU 样本中 HTTP JSON -> protobuf 约占 `34~36%`，`CacheManager::ReportEvent` 约占 `22%`，
  BatchMerge/RMW 约占 `12%`；glibc `malloc/free/_int_malloc/_int_free` 等 allocator 自身累计接近 `29%`；
- 117 个并发 10k update 请求的 paired futex 样本中，metadata shard mutex 有 284 次等待，总计
  `119.31ms`，平均 `420us`，p95 `1.35ms`，p99 `3.65ms`，最大 `4.136ms`；glibc allocator 另有
  1158 次等待，总计 `26.59ms`；local LRU futex 等待可忽略；
- `meta_indexer.get_io_time_us` 在 pure-local 下并不是远端 I/O，它包含 local LRU lookup、item shared lock、
  location 投影以及通用容器构造。不能仅凭指标名把它归因到 Redis。

#### 最终实现

1. `/api/reportEvent` 使用 request-scoped protobuf Arena。request/response 及其嵌套 event/spec 一次性回收，
   不改变 protobuf wire/JSON schema；其他 HTTP API 继续使用原 handler，缩小行为变化范围；
2. 仅当 backend 是**精确的生产 `MetaLocalBackend` 类型**、没有 cache/persistent 双层组合且每个 key 恰好一个
   target location 时，启用扁平 RMW。装饰器、子类、测试 fault backend、cached/Redis 或多 location 请求全部
   回退原 `ReadModifyWriteTargetLocations`，避免绕过它们覆写的审计、故障注入或恢复语义；
3. 快路径使用扁平 `keys/location-id/location/result` 数组，一次 local batch lookup 同时返回“key 是否存在”和
   “target location 是否存在”，随后在同一组 metadata shard locks 内 merge/upsert。这样去掉每 key 的
   `vector<vector<...>>`、location map 构造和第二次 key existence probe，但仍严格区分新 key 与“已有 key
   缺少该 location”，所以 `max_key_count`、`key_count` 与 sibling locations/properties 都保持正确；
4. local backend 对 LRU 使用 `LookupBatch/ReleaseBatch`，并只替换一个 immutable `CacheLocation` shared_ptr。
   所有 request-shaped scratch vector 在取得 metadata shard locks 前 reserve；`ScopedBatchLock` 借用稳定的
   shard-index vector，不再为每个 batch 复制一次；
5. duplicate key 在进入快路径前拒绝，防止同批次把一个新 key 重复计数。非法 location/null pointer、backend
   返回 shape mismatch、key/location 状态矛盾、capacity full、modifier skip/fail 和写失败均有 fail-closed
   处理；容量满时已有 key 的 update 仍可写入，新 key 返回 `EC_NOSPC`；
6. metadata write lease 仍在读到旧值之后、实际 merge 前获取，并持续到 RMW 返回。不要为了缩短锁时间把
   lease 提前到读取之前或在写入前释放，否则会破坏 snapshot generation/leader 切换期间的校验闭包。

#### Release 结果

同机、同配置、无编译负载的真实 HTTP `test_20_large_single_request_delta_scaling`：

| events/request | 本轮基线 create/update | 最终 create/update | 降幅 |
| ---: | ---: | ---: | ---: |
| 100 | 约 1.5/1.4ms | 1.32~1.37/1.17~1.20ms | 小请求主要受固定 HTTP 开销影响 |
| 1000 | 约 9ms | 7.23~7.35/6.72~6.79ms | 约 18%/24% |
| 5000 | 约 43/42ms | 32.03~32.30/30.21~30.42ms | 约 25%/28% |
| 20000 | 约 146/146ms | 128.01~128.11/119.76~121.40ms | 约 12%/17% |

10k existing-location update 的并发墙钟时间为：单请求 `52~54ms`；同 reporter x2/x4/x8 分别
`59.39/57.99/68.48ms`，distinct reporter x2/x4/x8 分别 `55.37/61.21/72.30ms`。x8 总吞吐超过
`1.1M blocks/s`，说明此时继续拆 metadata lock 或增加 writer pool 没有收益证据。

优化后的 futex 采样覆盖 186 个并发大请求，8 个 HTTP worker 没有一次进入 `FUTEX_WAIT`；原来最长的
metadata shard wait 栈已降到内核休眠采样阈值以下。最新 CPU 样本的 flat-RMW 函数只剩约 `0.59%`，主要
自耗时转为 protobuf JSON parser、glibc allocator 与 memmove/memcmp：`malloc 7.68%`、`_int_free 6.04%`、
`_int_malloc 3.52%`、JSON token/string parse `10%+`。生产启动脚本会 preload jemalloc，直接运行 Bazel
binary 的 benchmark 不会，因此这里的 glibc allocator 比例是保守结果。下一轮若继续优化，应先对 HTTP
JSON 与 gRPC 做同 payload A/B；不要在没有新 futex 证据时重写正确性敏感的 RMW 锁协议。

最终重新链接产物又做了一轮 20 秒全进程 trace，并在窗口内完成 8 个 10k-key reporter 的建库和 31 个
existing-location 并发 update。全进程没有任何 `FUTEX_WAIT/FUTEX_WAIT_PRIVATE`；其余 wait-like 事件全部
是后台线程稳定的 50ms/100ms/200ms/1s/5s timed condition wait，不在 ReportEvent 请求栈。最终 x8 same
reporter/distinct reporter 墙钟为 `67.54/73.37ms`，吞吐为 `1.184M/1.090M blocks/s`。

#### 回归门禁

- pure-local HTTP snapshot/并发/失败功能套件 36/36、旧版双类型兼容套件 20/20 通过；
- `meta_local_backend_test` 覆盖 flat read 的 key-status、sibling/property 保留、新建/更新、重复 key 顺序、
  malformed/null/empty；`meta_indexer_test` 覆盖精确 capacity、已有 key 缺 target、modifier skip/no-op、
  duplicate rejection；
- `--nocache_test_results` 下 8 个核心 Release Bazel 目标全部通过；完整 `CacheManagerTest` 10 个 shard 全绿，
  并专门验证继承 `MetaLocalBackend` 的 fault backend 会回退通用虚函数路径；MetaSearcher、backend manager、
  HTTP JSON 等相关目标同时通过；
- 本轮没有 Redis 性能结论，也没有修改 Redis 数据路径。以后改动借用的 location-id 生命周期、exact-type
  guard、batch lock 范围、key-status 或 capacity 逻辑时，必须重跑上述单测、36 项 HTTP 套件、20k scaling
  benchmark 和 paired futex 采样。

### 5.17 2026-08-07 ReportEvent JSON/Arena 与 RMW allocator 收敛

5.16 的锁内 flat RMW 完成后，Release `perf` 显示通用 protobuf JSON 转换仍占 CPU 的约 34%~36%，glibc
allocator 相关符号累计接近 29%。protobuf 3.13 的通用 JSON 路径会先构造中间表示/二进制字符串，再把它解析
进 request；默认 Arena block 只有 256B 起步、最大 8KiB。20k event 请求因此仍有大量短命分配。优化目标是
只收敛 `/api/reportEvent` 的常见 JSON 形态，不改变其他 API、protobuf schema 或异常输入兼容性。

#### JSON 与 request allocator

1. 新增 `ReportEventJsonParser`，使用 RapidJSON 直接把已知 ReportEvent 字段写入最终 protobuf request；request
   位于 handler 的 request-scoped Arena，因此不再经过“JSON -> 临时 protobuf binary -> request”的通用路径；
2. fast parser 支持 protobuf JSON 的 snake_case/lowerCamel 字段名、字符串/数字 enum、全部现有 event oneof、
   map/repeated 字段和未知字段忽略。`null`、未知 enum、重复字段等少见但通用 protobuf parser 可接受的形态会
   返回 fast-path miss，清空 request 后退回 `ProtoMessageJsonUtil`。这个 fallback 是兼容性边界，不能删除；
3. ASCII 请求先用同一轮 64-bit scan 同时检查 high bit 和原始 NUL。大于 32KiB、全 ASCII 且无原始 NUL 的
   body 只做一次连续 mutable copy，再由 RapidJSON in-situ parse：DOM 字符串直接引用/原地 unescape 该 buffer，
   不再先复制到 DOM pool、随后又复制到 protobuf。buffer 保持到 `ParseRequest` 完成，不跨 parser 生命周期；
4. 非 ASCII 请求仍启用 length-aware 严格 UTF-8 校验；带原始 NUL 的 ASCII body 也保留 length-aware parser，
   防止 C-string 提前结束把畸形 trailing bytes 当成合法 JSON。合法 `\u0000`、Unicode escape、quote/backslash/
   newline escape 均由 in-situ 路径正确解码，并与通用 protobuf parser 做语义差分；
5. 大于 32KiB 的 JSON 按 body 大小配置 RapidJSON pool/stack（pool 单块最多 4MiB，stack 64KiB~1MiB），
   protobuf Arena 使用 64KiB start block、1MiB max block；小 heartbeat 保留小块默认行为。response string 预留
   512B，避免常见响应的额外增长；
6. parser 只借用 `req.get_body()` 的同步 `string_view`，不会保存指针。测试覆盖非 NUL 结尾 view、前后垃圾
   backing storage、Unicode、非法 UTF-8、unknown field、数字 enum、oneof 和 fallback 与通用 parser 语义一致。

#### RMW writer 临时分配

1. `Cache::BatchOperationScratch` 保存 LRU batch lookup/release 的 hash、shard offset、cursor 和排序下标；pure-local
   single-location RMW 在进入 metadata shard locks 前一次 reserve，并让读、写以及后续内部 batch 重用。普通
   cache API 和其他 backend 仍走原虚函数语义；
2. key view、handle、location、错误码和 upsert 数组提升到 RMW batch 循环外，按最大 batch 一次 reserve。
   读阶段的错误码 vector 在 modifier 完成后直接承接 writer 返回值，不再另建 request-sized upsert vector；
3. `max_key_count` 满时先把新 key 标记为 `EC_NOSPC`，再原地 compact 已有 key 的 update，删除锁内四个 subset
   vector。已有 key 仍照常更新，新 key 不增加 key count；location handle 在每次 backend 调用结束前全部释放；
4. scratch 快路径仍受 5.16 的 exact `MetaLocalBackend`、无 cache backend、single target location guard 保护。
   cached/Redis、装饰 backend、fault backend 和多 location RMW 不会被静态转换或 allocation fast path 绕过。

#### Release 性能与最终 profile

同机、纯 local、8 HTTP worker、无编译负载，真实 HTTP
`test_20_large_single_request_delta_scaling` 的最终结果：

| events/request | create | update |
| ---: | ---: | ---: |
| 100 | 1.15ms | 1.15ms |
| 1000 | 4.38ms | 4.54ms |
| 5000 | 17.43ms | 18.12ms |
| 20000 | 79.31~82.86ms，五轮均值 80.65ms | 74.21~76.07ms，五轮均值 74.85ms |

只读 DOM 版本与 in-situ 版本在同机交替五轮均值为 `85.51/79.83ms` 与 `80.65/74.85ms`，in-situ 额外降低
`5.7%/6.2%`。相对 5.16 同一基线的 `128.11/121.40ms`，最终 create/update 累计下降约 `37.0%/38.3%`。
8 路同 reporter、每请求 10k existing update 的 1992 个稳态请求为 avg/p50/p95/p99
`42.32/41.91/51.54/55.52ms`，墙钟吞吐 `1.561M blocks/s`。

最终 in-situ `perf record -e cycles:u -F 999 -g --call-graph dwarf` 覆盖 14,272 个用户态样本。ASCII 路径原有的
RapidJSON UTF-8 validation 热点保持为零，`ParseString` 从只读 DOM profile 的 `8.82%` 降到 `2.52%`。
剩余 self CPU 以 `malloc 9.14%`、`_int_free 8.28%`、`_int_malloc 6.31%`、`memmove 5.56%` 为主，主要来自
必须跨请求存活的 URI/LocationSpec、immutable CacheLocation、shared_ptr 和 hash/LRU 节点，而不再是 JSON
临时 string。production 启动脚本还会 preload jemalloc，因此不要仅凭直接运行 Bazel binary 的 glibc self
比例引入无界 thread-local body、全局对象池或 location-id interner。SAX 最多继续消除约 2.5% 的 string parse
self CPU，却会显著扩大 parser 状态机；没有新的端到端收益证据前不建议实施。

system-wide futex trace 在 792 个成功的 10k 请求窗口内，按当前 server PID 过滤后只有 9 次后台
`FUTEX_WAIT_BITSET_PRIVATE|CLOCK_REALTIME` 定时等待和对应 wake；没有请求路径
`FUTEX_WAIT_PRIVATE`。因此本轮没有继续拆 metadata locks 或增加 writer worker。

#### 验证与后续 AI 门禁

- Release 定向 `LruCacheTest`、`meta_local_backend_test`、`meta_indexer_test`、
  `ProtoMessageJsonUtilTest` 全绿；最终真实 HTTP snapshot/并发/失败功能套件 36/36 全绿；
- `bazelisk test --config=release --test_output=errors --nocache_test_results //kv_cache_manager/...` 分析 351 个
  target，发现 107 个 test；106 个通过，唯一未执行的 `SdkBufferCheckUtilTest` 由 BUILD 在无 CUDA/MUSA 平台
  显式标记 incompatible；
- 仓库根 `//...` 会在 analysis 阶段因可选 `//3rdparty/tair_mempool` 引用未声明的 `@tair_mempool` 失败，
  尚未进入任何 test。这是当前 checkout 的可选外部依赖配置限制，不能误报为本轮测试失败或“全仓通过”；
- 后续修改 fast parser 必须保留 generic fallback、非 ASCII 严格校验和 bounded `string_view`；修改 RMW scratch
  必须保证 reserve/clear 在 metadata lock 外、handle 不跨 backend 调用逃逸，并保留 exact-type/capacity/key-count
  门禁。若新 profile 仍以 URI/CacheLocation 为主，应先优化解析结果复用或对象布局，不要回到盲目拆锁。

### 5.18 2026-08-07 body 复用、location-id interning 与 fused LRU handle

5.17 最终 profile 中，10K block 请求体约 1.94MiB，HTTP worker 每次仍要为 in-situ parse 创建 mutable copy；
每个 block 又分别复制相同 reporter/medium location-id。pure-local fused RMW 的读阶段释放 LRU handle 后，写阶段
还会对同一批 key 再做一次 hash/LookupBatch；替换旧 `CacheLocation` 时，最后一个 shared_ptr 的 URI/容器析构也
发生在 metadata shard locks 内。本轮只优化这些已经由 profile 证实的重复工作，不改变 Redis/cached backend。

#### 实现与边界

1. 大 ASCII ReportEvent 使用每 HTTP thread 一个 mutable body buffer。常驻 capacity 上限为 4MiB；超过上限的
   body 使用 request-local buffer，递归/重入解析同样回退本地 buffer。这样 10K 请求不再反复 malloc/free
   约 1.94MiB，同时最多保留 `4MiB * HTTP worker 数`，不能删除该上限；
2. ReportEvent request 内按 medium 创建一个 `shared_ptr<const string>` location-id，并贯通 delta ADD/DELETE、
   snapshot replace 与 MetaSearcher task。`CacheLocation` 用 variant 保存 owned 或 interned id：普通调用方仍保留
   owned string，只有 event path 共享。unordered_map rehash 只移动 shared_ptr，不会改变 pointee 地址，因此
   request fold 可安全使用 pointee identity；序列化和值比较仍通过 `id()`，cache charge 继续按每 location
   保守计算完整 id 大小；
3. delta 的 `(block_key, location-id)` fold 从 node-based unordered_map 改为请求内 power-of-two linear-probing
   table，正常负载不超过 50%，去掉每个 distinct block 的 hash node allocation。最终仍按 block/id 排序，
   last-operation-wins、failure dependency closure 与输出顺序不变；
4. exact `MetaLocalBackend` + pure-local + single-target 快路径的第一次 batch lookup 可把 handle 保留到匹配的
   writer call。writer 使用原 read index 直接 update/insert，跳过第二次 key hash、LRU lookup 和 handle acquire；
   skipped hit 会在新 key admission 前先释放，保持 strict-capacity 顺序。scratch 析构和所有 validation/shape/
   capacity early-return 都兜底释放 handle；其他 backend 仍走原路径；
5. writer 消费新 `CacheLocation` shared_ptr 并 move 进 item map，避免一次无意义的 refcount increment/decrement。
   被替换的旧指针先移入预留好的 retired vector；该 vector 的 guard 在 `ScopedBatchLock` 之后析构，所以 URI、
   spec vector 和 CacheLocation 的最终 free 发生在 metadata locks 外。guard 在 lock 前构造，`continue`/错误分支
   也遵循“先 unlock、后 clear”的 C++ 逆序析构；
6. 单 location item 用一次字符串比较代替 unordered_map hash+比较；多 location item 保留通用 find。针对
   interned-id variant 的 hot copy constructor 显式分派 owned/shared alternative，避免 libstdc++ 通用 variant
   copy 分派。曾尝试直接重建最终 spec vector、跳过旧 LocationSpec 深拷贝，但 20K update 五轮约慢 1ms，
   已撤回，不能在没有新证据时重新引入。

这里有意修正 5.17 的旧门禁：handle 现在可以**只在同一次 exact-local fused RMW 的配对 read/write backend
调用之间**保留；不得越过 metadata shard lock 生命周期、请求或 backend。bounded TLS buffer 与 request-scoped
location-id interning 也已有明确上限/所有权，不等同于无界 thread-local 或全局 interner。

#### Release 性能

同机、pure-local、8 HTTP worker 的 `test_20_large_single_request_delta_scaling`，最终 20K 五轮范围/均值为：

| 路径 | 5.17 均值 | 本轮范围 | 本轮均值 | 进一步降幅 |
| --- | ---: | ---: | ---: | ---: |
| create | 80.65ms | 76.85~84.18ms | 78.40ms（中位数 77.12ms） | 2.8% |
| existing update | 74.85ms | 69.46~71.43ms | 70.46ms（中位数 70.28ms） | 5.9% |

8 路同 reporter、每请求 10K existing update 的长跑吞吐在多轮测试中为 `1.53~1.75M blocks/s`；较短且无
编译干扰的轮次曾达到 `1.82~2.09M blocks/s`。并发结果会明显受同机编译负载和温度影响，因此不把某一轮
最好值或固定提升百分比作为结论。可确定的结构性收益是第二次 LRU lookup/handle acquire 已消除，旧
CacheLocation 的析构也已移出 metadata locks；perf 中 retained-handle writer self 约 0.2%，旧的 delta
unordered_map node lookup 已退出主要热点。发布判断仍应使用隔离机器上的相同负载 A/B。

#### 必须保留的测试门禁

- parser 重复解析测试要覆盖 TLS buffer 复用、escaped NUL/Unicode 与 generic fallback；
- ReportEvent 端到端测试要确认同一请求两个 block 的 `&CacheLocation::id()` 相同，不同 medium 仍隔离；
- retained-handle 单测必须同时覆盖 existing target、existing key/missing target、new key、部分 skip、非法
  read index 和 RAII release；MetaIndexer 的 capacity、duplicate、modifier skip/fail 用例不可删；
- 性能结论只适用于 pure-local。任何把 retained-handle API 扩到 cached/Redis、装饰 backend 或多 target RMW
  的修改，都必须重新证明 recovery、fault injection、capacity 和 lock ordering 语义。

### 5.19 2026-08-07 spec/URI 与 request-scoped ownership 收敛

5.18 后的同机 Release profile 仍显示 `StandardUri::ParseParams/Parse/ToUriStringWithExtraParam` 合计约 3.8%，
单元素 spec/task 容器、旧 URI 深拷贝和 allocator 仍是主要 CPU 来源。RapidJSON ASCII in-situ 路径已经没有
UTF-8 validation 栈；剩余 `ParseString` 是 JSON 字符串扫描和反转义，不能通过关闭合法性校验消除。本轮因此
只收敛已被 profile 证明的 URI、LocationSpec、vector 和 shared ownership 开销，不引入 SAX parser，也不修改
metadata lock、Redis/cached backend 或查询语义。

#### 实现与所有权边界

1. 常见单 spec BLOCK_ADD/SNAPSHOT 的校验结果和 `MergeLocationSpecsTask` 使用 inline optional；只有第二个 spec
   到来时才提升为预留好容量的 vector。纯 ADD 请求也不再创建每 block 的一元素 task vector、空 delete task
   reserve，最终用 flat task vector + offsets 调用 MetaSearcher；通用嵌套接口和多 spec 语义保持不变；
2. canonical URI 使用 allocation-free string_view 扫描，一次得到 `size` 和 `s_version` 的有序插入位置，再直接
   生成最终 URI。只有协议、正数 canonical port、显式 `key=value`、严格递增且无重复参数等条件全部满足时
   才走快路；合法但非 canonical 的历史输入继续回退完整 StandardUri，非法 port、重复内部参数和溢出仍失败；
3. 同一请求的 snapshot token 只做一次严格 ASCII `[0-9A-Fa-f]` 校验。每个已经独立验证过的 spec 随后使用
   prevalidated append；通用 URI helper 仍保留逐次校验。locale `isxdigit` 被显式 ASCII 判断替代，非 ASCII
   字节继续拒绝，不把协议 token 校验与 JSON UTF-8 校验混为一谈；
4. `CacheLocation` 保存一个不序列化的 validated total-size hint。常见“一个旧 spec 被同名新 spec 替换”可直接
   复用旧总大小并构造最终 immutable location，避免再次解析旧 URI、复制后立即销毁旧 URI。反序列化和任何
   mutable spec 访问都会使 hint 失效并安全回退；多 spec、重复历史 name 和溢出仍走严格校验；
5. ReportEvent 的 medium map 是请求内 location-id 的唯一 owner。delta fold、snapshot entry 和同步 MetaSearcher
   task 只借用该 `shared_ptr` 对象，不再为每个 block 做原子 refcount 增减；持久化 CacheLocation 时仍获取正常
   shared ownership。unordered_map rehash 不使 element reference 失效，且所有 borrowed 指针只允许存活到同一
   次同步 Batch 调用返回，禁止缓存、异步投递或跨请求保存；
6. 没有把“消费 task 并移动 spec”扩展成通用 RMW API：modifier 在不同 backend 上可能重试，贸然消费输入会改变
   retry 语义。当前只在已经严格限定的 one-old/one-new/same-name 情况直接构造最终 immutable value，删除旧 URI
   copy 和中间一元素 vector；CacheLocation 对外仍保持 `vector<LocationSpec>`，不扩大查询侧对象模型。

#### Release A/B 与最终 profile

同机、pure-local、8 HTTP worker、同一 Release 构建方式。修改前 20K 五轮 create/update 均值为
`77.01/70.31ms`；最终独立五轮范围为 `66.96~69.57/56.12~57.57ms`，均值 `67.92/57.09ms`，分别降低约
`11.8%/18.8%`。8 路同 reporter、每请求 10K existing update 的 100 轮长跑为 avg/p50/p95/p99
`23.12/21.87/33.89/36.07ms`，吞吐 `2.572M blocks/s`；本轮初始基线为 avg `31.92ms`、吞吐
`1.992M blocks/s`，对应平均 RT 降约 `27.6%`、吞吐升约 `29.1%`。共享机器存在频率和编译扰动，生产判断仍
应在隔离机器做同 payload A/B。

提交前另启一个长生命周期 pure-local Release 实例复验：20K 五轮 create/update 为
`71.16~73.93/57.31~59.44ms`，均值 `73.00/58.41ms`；8x10K、100 轮的两次独立长跑 avg 为
`22.68~27.67ms`、p99 为 `30.33~42.17ms`、吞吐 `2.328~2.973M blocks/s`。create 对共享机频率和
allocator 冷热更敏感，但 update 与并发吞吐均保持相对基线的明确改善，且全程无业务错误。

重放到包含 P2P host-count 配置的新远端基线后，再次从头构建 Release 二进制：20K create/update 为
`67.51/55.96ms`；8x10K、100 轮为 avg/p50/p95/p99 `23.07/22.03/33.16/34.85ms`，吞吐
`2.605M blocks/s`。这组提交前数据说明基线组合没有抵消本轮收益。

最终 perf 中 `StandardUri::ParseParams/Parse/ToUriStringWithExtraParam` 和 locale `isxdigit` 已退出热点列表；
中间版本的 shared_ptr add-ref self 从 `3.50%` 降到最终 `1.53%`。剩余主要是 JSON 必需字符串扫描、glibc
allocator、immutable CacheLocation 写入、LRU hash/lock 和最终 URI copy。production 使用 jemalloc，直接 Bazel
binary 的 glibc 比例仍是保守上界；没有证据支持无界 object pool、全局 location-id interner或放宽 JSON 校验。

#### 验证门禁

- `SnapshotUriUtilsTest` 覆盖 canonical 与 StandardUri 输出等价、非 canonical fallback、无效/重复参数、严格
  ASCII token；`MetaSearcherTest` 覆盖 flat offsets、inline spec、borrowed owner 引用计数和 total-size hint；
- `CacheManagerTest` 10 个 shard 覆盖 ADD/DELETE/SNAPSHOT、last-op-wins、failure closure、capacity 和同请求
  location-id 共享；真实 HTTP snapshot 套件 36/36、双类型兼容套件 20/20 通过；
- reporter lifecycle 定向阻塞回归连续 50/50 通过；全量 `bazel test --config=release
  //kv_cache_manager/...` 共发现 107 个目标，106 个可执行目标全部通过，另 1 个 GPU-only 目标因环境不兼容跳过；
- 修改 canonical parser 时必须维持“无法证明 canonical 就 fallback”的 fail-safe 边界；修改 borrowed id 时
  必须证明 owner 覆盖整个同步 Batch 调用。不得把裸指针写入 backend、队列、cleanup callback 或 response。

### 5.20 2026-08-07 顺序流快路、异常路径按需分配与最终锁审计

5.19 之后用 production-like `LD_PRELOAD=/lib64/libjemalloc.so.2` 重新采样。常见请求是同一 reporter、同一
medium、block key 递增且每个 `(block, location)` 只有一个 event；旧实现仍会为这些已经有序且唯一的数据建立
完整 open-address index、event dependency 数组、排序 permutation 和 ADD/DELETE 两套 failure range。另一个
剩余热点是通用字段名 helper 的函数调用，以及 canonical decimal 字段走 `from_chars`。本轮只增加能保持任意
事件顺序、partial failure 和历史 URI 兼容性的按需快路，没有修改 persistent/cached/Redis backend、查询对象
布局、metadata lock 范围或 lifecycle fencing。

#### 实现与正确性边界

1. `DeltaMutationGuard` 直接保存本 RPC 唯一的 `ReporterSnapshotKey`、可选 lease 和可选 snapshot-in-progress
   failure，不再为每个 delta event 查两张 reporter-key unordered map。一个 ReportEvent request 的 instance、
   host 和 storage 在入口已经固定，因此不存在第二个合法 reporter key；generation adoption 和析构时
   `EndDeltaMutation` 仍只针对成功取得的同一 lease；
2. registration 状态和 interned location-id 合并到 request-scoped medium state。连续相同 medium 用最后一次
   state 指针命中，不重复 hash/probe；location-id 仍按需创建并由 medium state 持有。`NODE_REGISTER` 的全请求
   预扫描改为第一次真正遇到 register 时才执行，纯 ADD/DELETE 不再额外遍历一遍 protobuf events；多 register、
   malformed register、delta-before-register 和 generation 继承语义由原有回归保留；
3. delta fold 对递增唯一 `(block_key, location-id)` 直接 append。只有出现非相邻 duplicate 或逆序 pair 时才
   建立 power-of-two index；只有最终 unique-location 顺序确实非递增时才分配并排序 permutation。任意顺序仍
   使用同一个 last-op-wins fold，随机 768-event reference-model 测试覆盖 48 keys、17 media、4 specs；
4. 每个 location 的第一个 dependency event 内联保存，额外 event 链只在同一 `(block, location)` 第二次出现时
   分配，索引收窄为 `uint32_t`（protobuf event count 上限是 `INT_MAX`）。ADD/DELETE phase failure 以及 admission
   failure 的 retry closure 仍遍历完整逻辑 event 链；
5. 只在实际 materialize ADD/DELETE 后 reserve 对应 phase 数组，ADD-only 不再分配 DELETE capacity，反之亦然。
   每 block 的 24-byte failure range 也被移除：成功路径不写这份数据，只有 backend 返回错误时才在已排序
   location view 中二分定位 block range。新增乱序三 block fault test 验证只标记实际失败 block，前后成功 block
   仍可查询；
6. canonical URI 的 port/size 与 block key 使用严格、overflow-checked 的手写十进制循环，去掉 generic
   `from_chars`。port 继续拒绝 0、符号、前导零和大于 `INT64_MAX`；size 接受完整 `uint64`；block key 继续接受
   signed int64 和 vLLM unsigned uint64 decimal，并保持相同 64-bit pattern。测试覆盖 `INT64_MAX` port、port
   overflow、`UINT64_MAX` size、size overflow，以及 block-key 两侧边界和非法符号；
7. canonical spec 不再内嵌构造重量级 `StandardUri`；只有合法但非 canonical 的兼容输入才按需分配 parser。
   `ValidatedEventLocationSpecs::Push`、`PushReportEventSpec` 改为显式右值入口，最终 URI/LocationSpec 继续 move；
8. JSON parser 的字段名比较改成 compile-time string-literal 长度加 `memcmp`，保留 snake_case、camelCase、
   embedded-NUL 和 exact-length 语义。最终 profile 中原先约 1.58% 的 out-of-line `NameIs` 已完全退出热点；
9. task 构造复用入口已验证过的 `requested_type`，不再对每个 block 重复调用 backend virtual getter。snapshot、
   delta 和 query 仍使用同一 routing decision，不改变 L1P5/L2 隔离。

#### 同机 A/B 与 perf 结论

同一 Release 构建、pure-local、8 HTTP workers、jemalloc、五个 fresh instance 的 20K scaling：

| 路径 | 修改前五轮均值 | 最终五轮范围/均值 | 降幅 |
| --- | ---: | ---: | ---: |
| create | 65.75ms | 58.38~61.89ms / 59.56ms | 约 9.4% |
| existing update | 57.37ms | 49.20~51.99ms / 50.23ms | 约 12.5% |

8 路同 reporter、每请求 10K existing update 在共享机器及 perf instrumentation 下为约
`2.72~3.47M blocks/s`，所有轮次 fail/drop 均为 0。共享机器频率、其他 Bazel 进程和 perf tracing 会造成明显
漂移，因此并发结果只用于排除回退，生产收益应继续做隔离机 paired A/B；串行五轮是本轮更稳定的比较。

最终 `cycles:u` profile 覆盖 57,389 samples、35.92M blocks。主要 self CPU 为 LRU mutex unlock `9.34%`、
`memmove 5.50%`、shared_ptr add/release `5.46%/3.47%`、RapidJSON `ParseString 4.99%`、ReportEvent orchestration
`4.59%`、targeted RMW `3.60%`、mutex lock `3.01%`、`ParseObject 2.67%`、LRU hash find `2.05%` 和 canonical URI
`1.75%`；jemalloc `malloc` 仅 `1.45%`。`from_chars`、out-of-line `NameIs`、eager fallback `StandardUri`、
common-path delta hash-table build 和 success-path failure range 都已退出热点。

锁需要区分“执行 lock/unlock 指令”与“线程实际睡眠”。15 秒 futex enter/exit 配对覆盖 2,392 个 10K 请求：

- `FUTEX_WAIT_PRIVATE` 真正睡眠 17,679 次，总计约 2.879s，单次平均约 163us、最大 4.735ms，折合每请求约
  1.20ms；另有 35,158 次在约 19.1ms 总计内返回 `EAGAIN`；
- CPU call graph 将 mutex 成本定位在 local LRU `LookupBatchWithScratch` 和 retained-handle
  `ReleaseBatchWithScratch`；metadata `pthread_rwlock` 没有进入 futex sleep，因此不是线上 80ms 的来源；
- 尝试按 worker 旋转 LRU shard 遍历起点以打散 convoy，真实吞吐连续下降到 `2.21~2.77M blocks/s`，串行也
  回退，已完整撤销。稳定 shard 顺序的 cache locality 比减少短 wait 更重要；
- retained handle 的 release 当前仍发生在 metadata lock 生命周期内。把它移出锁会改变并发 capacity/LRU
  admission 窗口，不能只为约 1.2ms/request 的可消除上界冒险。若以后重写 refs 为 atomic 或 fused shard
  callback，必须单独证明 eviction、delete、capacity、query 并发和 lock ordering。

剩余 `ParseString` 是 JSON 字符串扫描/反转义，ASCII 路径已没有 UTF-8 validation 栈；继续下降需要 SAX/direct
parser，收益上界约 5% 且兼容性风险明显。shared_ptr/LRU 与最终 URI copy 是下一批结构性候选，但都会触及查询
共享对象或 cache eviction。没有新的隔离机 profile 与正确性模型前，不应继续用全局 interner、无界对象池、
扩大 metadata lock batch 或 lock-free refcount 改写换取小优化。

#### 最终门禁

- 每项结构变化后均运行对应 Release `CacheManagerTest --test_filter=*ReportEvent*`；新增乱序 failure-range 测试；
- `SnapshotUriUtilsTest`、`ProtoMessageJsonUtilTest`、`MetaSearcherTest`、`LruCacheTest`、
  `meta_local_backend_test` 必须保持全绿；
- 提交前必须重新跑 pure-local HTTP snapshot 套件 41/41（36 functional + 5 benchmark）、旧 ReportEvent
  兼容套件 21/21（20 functional + 1 benchmark）、完整 Release
  `//kv_cache_manager/...`（无 GPU 环境预期 106 pass + 1 incompatible）；
- 性能测试必须确认启动进程 maps 中加载 jemalloc。实验性 URI prefix cache、直接重建最终 spec vector 和 LRU
  shard rotation 都已因无收益或回退撤销，后续 AI 不应在没有新的 paired A/B 证据时重复引入。

### 5.21 2026-08-07 收敛复核：allocator、URI 所有权与 item lock 的负向实验

5.20 后又在完全相同的 pure-local Release 配置下做了一轮独立复核。本节的目的不是记录“还能想到什么”，而是
把已经实测无收益的候选、其正确性边界和停止条件固定下来，避免后续仅凭 profile 百分比重复引入更复杂的所有权
或锁协议。下面所有实验都在独立修改后测试、A/B，未达到门槛的实现均已完整撤销；最终生产代码仍是 5.20 的
实现。

#### 更大样本的 clean profile

使用 production-like `LD_PRELOAD=/lib64/libjemalloc.so.2`、8 HTTP workers，对干净 HEAD 采集
`cycles:u -c 100003 -g --call-graph fp`。样本覆盖 198,470 个 samples、792 个 10K-block 请求，共 7.92M
blocks；负载 avg/p50/p95/p99 为 `19.09/18.46/27.00/28.53ms`，吞吐 `3.177M blocks/s`，业务错误为 0。
主要 self CPU 为：

| 热点 | self CPU |
| --- | ---: |
| `memmove` | 10.08% |
| LRU `pthread_mutex_unlock` | 6.22% |
| RapidJSON in-situ `ParseString` | 5.19% |
| `CacheManager::ReportEvent` | 4.73% |
| targeted RMW | 3.63% |
| merge modifier | 3.40% |
| parser orchestration/ASCII scan | 3.08% |
| shared_ptr add/release | 2.81% / 2.21% |
| RapidJSON `ParseObject` | 2.77% |
| LRU hash lookup | 2.20% |
| LRU `pthread_mutex_lock` | 1.92% |
| item rwlock read/write/unlock | 1.68% / 1.63% / 1.59% |
| jemalloc `malloc` | 1.52% |

`memmove` 的主要调用方依次是最终 immutable `LocationSpec` 构造、TLS body copy、protobuf arena string 和
RapidJSON；它不是一个可整体删除的重复 copy。锁的百分比同样主要是成功的 lock/unlock 指令，不等于线程睡眠：
5.20 的 futex trace 已把实际等待上限量化为约 1.20ms/request。

#### 已撤销实验及 A/B

1. **旧版消费 task URI 实验（历史结果；5.24 已用更窄的 prevalidated-only 实现重新验证并保留）。**
   当时为 pure-local、单 location、flat task 增加了显式 consumable
   API，并保留 spec name 供失败映射；create/update、多 location 不消费及错误语义定向 UT 全部通过。但五轮串行
   existing-update 对照为 `50.02ms`，候选为 `50.08ms`，并发也无稳定信号。该 copy 在 profile 中可见，却不是
   当前 wall-time 瓶颈；消费输入还会扩大 backend retry 语义，故完整撤销。5.18 中“直接重建最终 spec vector”
   曾回退约 1ms，这次不同实现得到相同结论。
2. **exact-local RMW 跳过逐 item shared read-lock。** 前提审计确认 MetaIndexer shard lock 覆盖本路径的生产写，
   定向 `MetaLocalBackend/MetaIndexer/MetaSearcher/CacheManager` Release UT 全绿。12 轮 fresh-process A/B 中，baseline
   update `50.01ms`、候选 `50.28ms`；并发吞吐 baseline `3.255M`、候选 `3.165M blocks/s`。把分支移出 key loop
   后仍为 `50.42ms` 对 `50.48ms`，约 1.8% 的并发差异处于机器噪声内。可见的 rwlock 指令没有形成稳定 RT 收益，
   而特殊“外层锁隐含保护”会增加未来 backend 维护风险，故完整撤销。
3. **bounded TLS RapidJSON DOM arena + DOM/stack 共用 allocator。** 实测 DOM：`202,157B` body 使用
   `256,160B`、capacity `404,314B`；`1,010,163B` body 使用 `1,280,160B`、capacity `2,020,326B`，即 DOM
   约为 body 的 1.27 倍。第一版 `vector::resize` 首次清零使 create 从 baseline `60.15ms` 回退到 `62.12ms`，
   虽然同连接 update 一度为 `48.93ms` 对 `50.09ms`。改成对齐但不初始化、最大 6MiB 的 TLS storage 后，七轮
   create/update 仍为 `61.52/50.91ms`，相对 baseline `60.15/50.09ms` 均回退。jemalloc 已能有效复用大块；
   单一 arena 的布局/局部性损失超过 1.52% allocator 上限，故完整撤销且不承担每 worker 额外常驻内存。

#### 最终停止条件与后续边界

- 不实现 SAX/direct JSON parser。剩余 `ParseString` 上界约 5.2%，但必须重新实现 snake/camel aliases、enum
  string/numeric、int64/uint64 JSON 表达、unknown/duplicate field、escaped/raw NUL、Unicode/非法 UTF-8 和 generic
  fallback 的全部兼容面；收益与上线风险不匹配。
- 不启用 adaptive LRU mutex，也不把 item unique-lock跨越 modifier。真实 futex wait 很短，而持锁延长或自旋会
  直接与优先级更高的 GetHostCacheState 争用；“CPU 有余量”不能替代 query p99 的同负载 A/B。
- 不把**持久化 owner**改成裸指针/全局 interner，不原地修改对查询可见的 CacheLocation。add/ref 与析构是
  immutable query snapshot 的所有权成本；只有 5.22 所述“同步调用内、retained handle + shard lock 双重保护”的
  旧值借用是例外，借用结束后的 owner 仍是 `shared_ptr`。
- `EstimateMemUsage`、metrics attach、block-key decimal parse 等单项均低于约 1%；为它们增加 persistent 字段、
  request TLS side channel 或跨层 hint 会增加每 key 内存与协议耦合，不满足“端到端有稳定收益”的门槛。

后续只有在隔离机的新 profile 显示热点结构发生变化时，才应重新打开上述方向。下一阶段若要获取超过噪声的收益，
需要独立设计并验证 cache item/immutable location 的表示或真正的 backend shard callback；这属于新的并发协议，必须
以 Get/ReportEvent 混合负载、eviction/capacity/fault/lifecycle 完整模型作为前置条件。5.22 只完成了 exact-local
同步借用这一条窄路径的证明，不代表可以把相同假设扩展到持久 owner、异步 backend 或调用边界之外。

### 5.22 2026-08-07 最终所有权复核：借用旧 location，保留 immutable query snapshot

5.21 的停止条件之后又用更大的 clean profile 做了源码行和调用栈聚合。结论是不能把全部
`shared_ptr add/release` 都看成同一个问题：新 `CacheLocation` 持有 interned location id 所产生的一次引用是
持久化所有权，不能删除；pure-local fused RMW 从 item map 读取旧 `CacheLocation` 时产生的临时引用，仅用于同步
modifier，生命周期已被 metadata shard lock 和 retained cache handle 覆盖，可以安全消除。本节记录两项先行负向
实验、最终保留实现及其严格边界。

#### 两项已完整撤销的 HTTP/parser 实验

1. **canonical BLOCK_ADD 的 RapidJSON SAX parser。** 实现支持 snake/camel alias、字符串/数字 enum、任意字段顺序，
   遇到 mixed event、未知形状、Unicode 或非 canonical 输入即回退现有 DOM parser，相关 JSON UT 全绿。SAX 确实
   让 DOM `ParseObject`/`Document::String` 退出 profile，但状态机和逐 token protobuf setter 把成本转移到了 SAX
   `ParseString`：七轮 20K create/update 为 `60.39/50.26ms`，同机 fresh baseline 为 `60.28/50.25ms`；8 路
   concurrent 10K 为 avg `19.10ms`、`3.508M blocks/s`，baseline 为 `17.96ms`、`3.699M blocks/s`。没有串行
   收益且并发回退，已完整撤销。
2. **thread-local reusable protobuf Arena/request graph。** 对 32KiB~2MiB body 复用 worker-local request message，
   reentrant/oversize 请求回退到 request-scoped arena，并设置 8MiB hard cap。`Clear()` 仍需遍历全部 message tree，
   保留的 repeated message/string capacity 还增加 cache footprint。三轮 concurrent 10K 只有
   `3.01~3.23M blocks/s`，低于同机 baseline `3.699M`；进程 RSS 约 `267MiB`。实现和常驻内存均已撤销。

这两项说明 allocator 百分比不能直接当作可回收 wall time：jemalloc 和 request-scoped protobuf Arena 已能较好
复用大块，跨请求保留对象反而破坏局部性；direct parser 也必须用端到端 A/B 判断，不能只看某个 DOM symbol 消失。

#### 保留实现：RMW 旧值使用同步借用视图

pure-local `ReadModifyWriteSingleTargetLocations` 原来把 item map 中的旧 `shared_ptr<CacheLocation const>` 复制到
batch vector，modifier 构造新 immutable value 时再释放这份临时引用。最终实现改为：

- backend 在 item shared lock 内只返回 `const CacheLocation *`；接口名显式为 borrowed view，并且**没有**
  “不保留 handle”的开关；
- `ScopedBatchLock` 在整个 read/modifier/write 周期持有目标 metadata shards，retained cache handle 保证被 LRU
  eviction 摘除的 `MetaMemCacheItem` 也不会析构，item map 自身继续拥有旧 location；
- modifier 只读旧对象，在栈上构造独立的 `shared_ptr<const CacheLocation>` 新值，成功后直接 move 到 upsert vector；
  不原地修改任何查询可见对象；
- writer 仍在 item unique lock 内原子替换 map entry，旧 owner move 到 `retired_locations`，并在 metadata lock
  释放后析构。并发查询仍先在 item shared lock 内复制旧或新 `shared_ptr`，因此继续获得完整 immutable snapshot；
- generic/Redis/cached backend 仍走原有 owning `shared_ptr` 路径。借用接口只在
  `SupportsSingleLocationRmw()` 已确认 exact pure-local backend 时可达。

第一版曾额外创建一个 request-sized replacement `shared_ptr` 数组；串行虽有约 2% 信号，但 8 路并发回退
4%~8%。最终改成每次 modifier 的栈上新值、成功后直接 move，去掉第二个数组后并发回退消失。后续不得重新引入
双 request-sized location 数组。

#### 正确性门禁与性能证据

- backend/indexer UT 直接记录旧 owner 的 `use_count`，断言 borrowed read 前后不增加；同时覆盖 hit、key exists but
  location miss、key miss、capacity full 时保留 existing-key update、modifier skip、duplicate key、retained handle
  subset validation 和旧值延迟析构；
- `TestGetHostCacheStateConcurrentWithReportEventAndHostDown` 在 Release 下重复 200 个 sharded runs 全绿，覆盖查询
  与连续 immutable replacement、HOST_DOWN 可见性切换并发；定向 local backend、MetaIndexer、CacheManager 测试全绿；
- 最终完整 Release `//kv_cache_manager/...` 为 106 个可执行测试全绿、1 个 GPU-only 测试按预期 skip；最终链接
  HTTP 二进制的 snapshot 套件 36 functional + 5 benchmark 全绿，旧协议/双类型套件 20 functional + 1 benchmark
  全绿。`/proc/<pid>/environ` 和 `maps` 同时确认加载 `/usr/lib64/libjemalloc.so.2`；
- 八组交错顺序的同机 20K paired A/B：create `61.017 -> 60.911ms`（符合预期，new-key 路径基本中性），existing
  update `50.429 -> 49.714ms`，下降约 `1.42%`；
- 三轮固定 800 个 10K 请求的 `perf stat`：cycles 均值 `44.171B -> 41.675B`（约 -5.65%），instructions
  `103.277B -> 102.978B`（约 -0.29%），cache misses `211.54M -> 187.13M`（约 -11.54%）。共享机频率会影响
  cycles，稳定结论是“没有新增指令膨胀，旧 owner 的 cache-line/refcount 流量下降”；
- 最终候选 profile 覆盖 548,846 samples。`shared_ptr::_M_release` self share 从 clean baseline 的 `2.21%` 降到
  `1.20%`。剩余 `_M_add_ref_copy` 主要来自每个新 `CacheLocation` 对 interned location id 的持久 owner，不应按本次
  方法继续删除；
- 8 路 10K update 加 200 次 10K-key GetHostCacheState 的 closed-loop 混合压测无错误。五组候选 Get avg 聚合约
  `10.75ms`，baseline 约 `11.23ms`；p99 范围分别为 `22.13~27.97ms` 与 `14.53~26.84ms`，共享机 tail 有明显
  抖动且两者重叠，因此只下“查询未出现系统性回退”的结论，不宣称 p99 提升。

#### 本轮后的停止线

当前 profile 的大项依次是 LRU mutex、最终 string/memmove、RapidJSON string scan、manager/merge、interned-id owner
和 item rwlock。已有 futex trace 证明真正 sleep 远小于 lock/unlock self CPU；SAX、TLS protobuf graph、DOM pool、
LRU shard rotation 和跳过 item read-lock 均已有负向 A/B。继续消除 interned-id owner 需要 process/instance lifetime
string pool，继续消除 item lock 需要把 modifier 放进 backend critical section，这两者都会扩大查询或回收风险。
在出现新的隔离机 profile 之前，本分支不再接受以 raw global pool、原地可变 CacheLocation、扩大锁范围或无界 TLS
缓存换取低个位数百分比的改动。

### 5.23 2026-08-07 HTTP body 原地解析、SIMD 扫描与 RMW 临时状态收敛

5.22 的 clean profile 继续按 `memmove` 调用方拆分后发现一个此前被总占比掩盖的确定重复工作：
`MutableJsonBufferLease` 把 cinatra 已经完整收进 `std::string` 的 HTTP body 再复制一遍，随后才执行 RapidJSON
in-situ parse。该调用方占当时全部 `memmove` samples 的 `23.44%`，折算约占总 CPU `2.07%`。这与最终
`LocationSpec/URI` 的持久化 copy 不同：后者建立 immutable metadata 所有权，前者只是为获得 mutable buffer
而复制同一请求体，可以在明确 transport 生命周期后删除。

#### 保留实现与兼容边界

1. `GetArenaHandler` 允许为特定请求注册 `char * + size` parser。当前锁定的
   yalantinglibs/cinatra 0.5.5 在 `coro_http_connection` 中用 mutable `std::string body_` 保存完整 Content-Length
   body，`request_.set_body(body_)` 只暴露同步 `string_view`；logger 在 handler 前读取 request，handler 返回后才复用
   connection。因此 ReportEvent 可以在 coroutine 内直接 ParseInsitu，protobuf/DOM 均不保存 body 指针；其他 HTTP
   API 继续使用 immutable、length-aware parser。
2. mutable API 的契约显式要求 `json[size]` 可读且为 `\0`，入口仍做防御检查。小于 32KiB、非 ASCII 或含 raw NUL
   的 body 保留旧 parser；小 heartbeat/register 直接进入旧 parser，不再先做一次随后必然重复的 ASCII scan。
3. 快速 protobuf converter 若遇到 `null`、未知 enum 等少见但 generic protobuf JSON 接受的形状，不能再使用已被
   in-situ 修改的原文。实现从**完整 DOM**序列化一次 normalized JSON，再调用 generic parser；该分配只发生在兼容
   fallback。JSON 语法错误直接返回 bad request，因为 fast/generic parser 都不应接受它。
4. ASCII/raw-NUL 预扫描在 x86 上运行时分派 AVX2（32 bytes/iteration），无 AVX2 时使用 SSE2；AArch64 使用 NEON，
   其他平台保留严格 scalar fallback。非 ASCII 路径仍启用 RapidJSON UTF-8 validation，没有用 SIMD 检测替代编码
   正确性。
5. `BatchMergeLocationSpecsImpl` 不再同时保存 `incoming_task_sizes` 与 `usage_changes`。incoming size 直接初始化对应
   usage slot，modifier 一次调用内读出后写回 final size；exact pure-local 的一 location/key 路径还直接用 key index，
   不分配 offsets。20K 主路径因此减少约 `160KiB + 160KiB` 临时数组和一次 request-shaped allocation；
   multi-location 仍保留 flattened offsets 和完全相同的验证/计量语义。

这里的 transport 假设是严格回退边界：若升级 cinatra 后 body 不再由 mutable、NUL-terminated `std::string` 支撑，
必须删除 specialized parser 或恢复 immutable copy，不能仅依赖 `const_cast` 继续运行。UT 直接覆盖 mutable API 的
escaped quote/backslash/newline、合法 `\u0000`、Unicode escape、raw NUL 拒绝，以及 source 已被修改后 rare fallback
仍与 generic protobuf parser 完全一致。

#### A/B 与最终 profile

- 删除 HTTP TLS body copy 的 8 组交错 20K scaling：create `60.276 -> 59.301ms`（约 `-1.62%`），existing update
  `50.035 -> 49.894ms`（接近中性）；8 路 10K steady throughput 约提高 `1.83%`。create 包含旧 TLS buffer 的首次
  allocation/first-touch，warm update 的收益上界本来就较小。
- SSE2/AVX2 各用 fresh process 做 8 组同长度 payload：`3.258M -> 3.307M blocks/s`，约 `+1.5%`；profile 中
  ASCII scan self share 从约 `2.13%` 降至 `1.69%`。该数据只证明 portable SIMD dispatch 有小幅正收益，不外推
  为整条 ReportEvent 的固定 SLA。
- RMW 临时状态收敛单独做 8 组 paired A/B：20K create/update 均值
  `56.755/49.008 -> 56.392/48.648ms`，约 `-0.64%/-0.73%`；8 路 10K 的平均 RT
  `18.730 -> 18.281ms`，吞吐 `3.238M -> 3.347M blocks/s`（约 `+3.4%`）。共享机频率会放大低个位数差异，
  稳定结论以“减少 320KiB 临时写流量且没有回退”为主。
- 最终 `cycles:u` profile 覆盖 14.32M blocks。TLS body-copy caller 已从 `memmove` call graph 完全退出；剩余
  `memmove` 中约 `72.26%` 来自 merge modifier 建立最终 LocationSpec/URI，约 `11.87%` 来自 protobuf arena string。
  flat self CPU 主要为 LRU mutex unlock `8.91%`、RapidJSON ParseString `5.90%`、memmove `5.67%`、interned-id
  persistent owner add-ref `5.00%`、single-target RMW `4.86%`、manager `4.06%`、mutex lock `2.94%`。热点结构与
  5.22 的所有权/锁结论一致，没有出现新的远端 I/O、serialization 或 futex sleep 路径。

#### 本轮已撤销实验

1. **把 UTF-8 validation 融入一次 RapidJSON parse、删除预扫描。** 定向 UT 通过，但 fresh 20K create/update 从
   `58.78/49.52ms` 回退到 `62.77/52.83ms`，约 6%；branch-heavy codepoint validation 明显慢于 SIMD ASCII scan，
   已撤销。
2. **按 canonical JSON 字段位置直接取 DOM member。** 兼容 fallback 与 UT 均通过，但 8 组吞吐信号仅约 `+1.4%`
   且噪声较大，`memcmp` self share 仍为 `1.62%`，只是把成本移进 `ParseBlockAdd`，已撤销。5.22 已完整测试并撤销
   SAX parser，本轮没有重复引入。
3. **ordered unique key 只做一次 duplicate scan。** 理论上少一遍线性比较，但 10 组 20K A/B 为
   `56.554/48.300 -> 56.270/48.792ms`，existing update 反向约 1%，没有端到端证据，已撤销。
4. **优先访问 `CacheLocation` 的 interned-id variant。** 汇编确实少一个 index branch，但 10 组 ReportEvent
   create/update 为 `55.785/48.578 -> 56.656/48.510ms`；两台 fresh process 的 6 组 20K Get 串行均值
   `13.613 -> 13.880ms`，16-way p50/p99 基本重叠。为避免牺牲 owned-id 的通用路径，已撤销。`std::visit` 版本还
   生成了 out-of-line indirect dispatch，更不应保留。
5. **targeted RMW shard index 改成连续 counting-sort。** 第一版用一张 flat index 表替换 per-shard vector 和
   batch index copy，却在 count/scatter 两遍重复计算 `HashKey`；8 组交错 20K A/B 为 baseline
   `58.454/49.156ms`、candidate `59.020/49.084ms`，create 明确回退。第二版缓存首遍 shard id，定向 Release UT
   通过，并重启两端进程做 10 组交错 A/B；baseline create/update 为 `57.628/48.962ms`，candidate 为
   `57.903/49.062ms`，平均仍分别多 `0.275/0.100ms`。少量 allocator/node 收益被额外的连续 scatter 写流量抵消，
   没有 wall-time 正收益，故代码完整撤销，只保留本记录。
6. **直接 move 每个 shard 已排序的 index vector 到空 batch。** 典型 20K/16-shard 请求中单个 shard 已超过 soft
   batch size，理论上可省掉每 batch 一次 allocation 和约 1250 个 `int32_t` copy，而且不改变 shard/batch 顺序。
   定向 Release UT 通过后，重启 baseline/candidate 做 12 组交错 A/B：candidate 相对 baseline 的 create 平均
   `+0.113ms`、update `-0.037ms`，paired median 为 `-0.025/-0.085ms`，全部处于噪声内。说明这段 index copy
   已不是端到端瓶颈；为避免增加特殊所有权分支，代码完整撤销。

最终源码重新构建后完成以下门禁（均为 pure-local metadata，不依赖 Redis）：

- Release `//kv_cache_manager/...`：106 个可执行测试通过，1 个 GPU-only 测试按预期 skip；
- snapshot HTTP：36 functional + 5 benchmark 全绿；20K scaling 对 create/update 后逐点查询验证最终 URI；
- 旧协议/双 storage type HTTP：20 functional + 1 mixed benchmark 全绿；
- `TestGetHostCacheStateConcurrentWithReportEventAndHostDown` 连续 200 次通过；最终二进制的 8 writer × 50 个
  10K update 与 200 个 10K-key Get 混合压测零错误，write avg/p95/p99 为 `15.89/22.80/28.57ms`，Get
  avg/p50/p95/p99 为 `10.86/7.39/23.20/24.81ms`；
- 真实 HTTP 额外验证 large rare fallback 与 large Unicode 返回 200，截断 JSON 与 raw NUL 返回 400；
- `/proc/<pid>/environ` 与 `/proc/<pid>/maps` 同时确认最终 Release 进程加载
  `/usr/lib64/libjemalloc.so.2`。

因此下一步不应继续围绕低于 1% 的 metrics、decimal parse、variant branch 或 URI copy 做局部改写。若线上新 profile
仍以 LRU lock/unlock 与 persistent owner 为主，能够超过噪声的下一阶段已经属于 LRU ref/list 协议或 metadata
representation 重设计，必须先建立 capacity/eviction、Get/ReportEvent 混合负载和 lifecycle fault 的独立正确性模型，
不能作为本次低风险 hot-path patch 顺手合入。

### 5.24 2026-08-08 对齐 pure-local 分片锁与消费 prevalidated URI

5.23 的最终 profile 中，LRU lock/unlock 仍是最大项，且 memmove 的约 72% 来自把 flat task 中已经拥有的 URI
复制到最终 immutable LocationSpec。本轮没有扩大 backend critical section，也没有改变 Get 的 immutable snapshot
协议，而是分别消除两类可以严格证明为冗余的工作。

#### 保留实现一：MetaIndexer mutex shard 复用真实 local LRU hash seed

此前 MetaIndexer 用固定 HashKey seed 把请求按 16 个 metadata mutex shard 分批，而 pure-local LRU 用自身的
host-specific seed 把同一批 key 分散到默认 1024 个 LRU shard。两个 seed 不同意味着一个 metadata batch 通常又
横跨大部分 LRU shard；两阶段 lookup/release 因而反复执行数千次 LRU mutex lock/unlock。最终实现：

- MetaLocalBackend 只读暴露其实际 Cache::GetHashSeed()；MetaStorageBackendManager 仅在 single pure-local
  backend 下向 MetaIndexer 提供该值；
- MetaIndexer 初始化完成后保存每实例的 mutex hash seed，所有 mutation/RMW batching 统一通过
  GetMutexShardIndex()，不能让同一个 key 在不同操作中使用不同 mutex；
- Redis、cached 和其他 backend 保留原固定 seed 与原 batching 行为。实现没有给 LRU 设置固定 seed，也没有改变
  LRU 的 key→shard/capacity 分布；只是让外层 metadata mutex 使用 LRU 已经选定的 host seed，因此仍保留跨主机
  hash 独立性；
- 测试不再把两个连续 key 写死为“不同 shard”，而是按 indexer 的真实 seed 选择。新增 UT 直接断言 pure-local
  mutex 的低 hash bits 与 LRU 一致；这也修复了更换 seed 后生命周期并发测试自身可能等待闸门造成的假死。

原型 perf 在相同 8-way 10K update 下显示：pthread_mutex_unlock self share 7.93% → 1.99%，
pthread_mutex_lock 2.53% → 0.94%，baseline 可见的 futex wait/wake 在候选中降到 0.05% 报告阈值以下。
动态 seed 最终版的十组交错 20K A/B 为：

- create 57.780 → 56.048ms（约 -3.0%）；
- existing-location update 49.039 → 47.512ms（约 -3.1%），两项均 10/10 轮更快；
- 两组各 2392 个请求的长窗口 8-way 10K update，吞吐平均约
  3.226M → 3.686M blocks/s（约 +14.3%），两组 p95/p99 均下降。

收益在并发下更大，符合“减少锁指令和 cache-line 争用、没有增加单 key 业务逻辑”的预期。该优化不得扩展为
固定全局 LRU seed，也不得让 read/write 选择不同 metadata seed。

#### 保留实现二：只消费 CacheManager-prevalidated flat task 的 URI

ReportEvent 在进入 MetaSearcher 前已经完成 URI 解析、snapshot metadata 校验和 size 汇总，且 flat task 在同步调用
结束后不再重试。旧实现仍把其中的 LocationSpec 复制到新 CacheLocation，长 URI 因而产生一次额外 allocation +
memmove。最终实现将边界收窄为：

- BatchMergeLocationSpecsFlat 接收 mutable task vector，但只有携带
  CacheManager::PrevalidatedTotalSize 的 task 可以转移 URI 所有权；普通/non-prevalidated flat task 和 nested
  通用 API 仍保持完整 copy/retry 语义；
- 转移前复制并在 source task 中恢复 spec name。metadata write 即使失败，CacheManager 仍能用
  (location_id, spec_name) 精确回填每个原始 event；source URI 可以为空，因为失败映射从不读取它；
- allocation、reserve、类型/size/duplicate-name 校验都在消费前完成。最终对象继续是独立 immutable
  shared_ptr<const CacheLocation>，没有原地修改查询可见值；
- 多 spec、legacy duplicate name、new key、existing location、lease failure、capacity/error alignment 均继续走同一
  modifier/write 结果模型。UT 同时断言 prevalidated task 的 URI 被消费、name 保留，non-prevalidated task 不变。

相对“已含动态 shard seed、尚未消费 URI”的二进制，十二组交错 20K A/B 为：

- create 56.289 → 54.570ms（约 -3.05%）；
- existing update 47.674 → 46.950ms（约 -1.52%）；
- 两组长窗口 8-way 10K update 吞吐分别
  2.956M → 3.385M（+14.5%）和 2.951M → 3.290M blocks/s（+11.5%），p99 均下降约 3~4ms。

5.21 记录的旧 consumable API 在当时代码形态和五轮短 A/B 中没有稳定收益，因此被正确撤销。本轮是在后续
HTTP body copy、RMW scratch、borrowed old owner 都已经收敛后重新按调用栈定位，并以现有 prevalidated marker
作为严格消费凭据；历史负向数据保留，后续不应恢复更宽泛的“所有 flat task 都可消费”版本。

#### 本轮撤销实验与剩余停止线

- 尝试在“旧 location 只有一个 spec、incoming 同名一个 spec”时更早折叠 update，跳过通用 old-name/duplicate
  检查。定向 Release MetaSearcher/CacheManager UT 全绿，但 12 轮 fresh-process paired A/B 的 create/update
  仅 -0.102/-0.104ms，update median 为 0ms，没有稳定收益，已完整撤销。
- 不把 persistent interned location-id owner 改成 raw pointer/immortal global pool。剩余
  shared_ptr::_M_add_ref_copy 是每个缓存 CacheLocation 的真实跨线程生命周期所有权；移除它会把收益建立在
  backend/reporter 永不销毁的隐含假设上。
- 不跳过 RapidJSON string/UTF-8 语义。当前 ASCII SIMD 预扫描后，ParseString 是输入字节本身的线性扫描；
  SAX、TLS protobuf graph、DOM pool 和直接 UTF-8 parse 都已有负向 A/B。除非协议改成 protobuf/gRPC 或建立新的
  direct parser 兼容模型，否则不能用放松 JSON/Unicode 校验换性能。
- 不继续增加 ReportEvent worker 或延长 item/LRU 锁。当前优化已从锁次数入手；GetHostCacheState 优先级更高，
  后续任何锁范围/并行度变化都必须先通过 mixed Get p99 门禁。

#### 最终正确性与端到端门禁

全部测试使用 pure-local metadata，不依赖 Redis：

- Release bazel test //kv_cache_manager/...：106 个可执行测试通过，1 个 GPU-only 测试按预期 skip；
- 最终 Release HTTP 二进制：snapshot 36 functional + 5 benchmark 全绿；旧协议/双 storage type
  20 functional + 1 benchmark 全绿；
- snapshot 20K scaling 在同一最终进程中为 create/update 59.25/48.70ms，随后逐点查询校验最终 URI；
  GetHostCacheState 20K local serial p50/p99/avg 为 13.83/13.98/13.85ms；
- 8 writer × 50 个 10K update 与 200 个 10K-key Get 的混合压测零错误，write avg/p95/p99
  14.34/21.11/23.17ms，Get avg/p95/p99 11.08/21.15/22.41ms。共享机数据只作为无功能/查询回退门禁，
  不声明线上固定 SLA。

本轮之后，仍能看到的主要成本是协议 JSON scan、最终 URI 的第一次持久化所有权、interned-id owner 和必要的
immutable object allocation。它们不再是可通过局部 move、换容器或跳过锁安全删除的重复工作。后续若继续优化，
应从新 profile 重新建立证据，优先考虑协议边界或 metadata representation 的独立设计，不要重复 5.20~5.24
已经完整撤销的微优化。

### 5.25 2026-08-10 功能正确性复核与大批量部分失败门禁

本轮不再改动生产逻辑，重点复核 5.24 优化后容易被性能测试遗漏的失败闭包、生命周期并发和 HTTP 大请求解析。
新增或加强以下自动化门禁：

- fast JSON parser 使用完整事件类型矩阵构造大于 40 KiB 的请求，强制进入 HTTP mutable in-situ 路径，并与
  protobuf generic JSON parser 做 message 等价比较；同时断言输入 buffer 确实被原地消费，防止测试误走小请求
  compatibility 路径；
- `max_key_count=1` 下同批更新已有 key 并插入新 key，验证 fused writer 只拒绝新 key、已有 key 更新仍提交，且
  已消费 URI 的 source task 仍能把 `EC_NOSPC` 精确映射到原 event index；
- HTTP 端到端一次提交 512 个 ADD，按固定间隔混入 31 个非法 URI，请求体大于 32 KiB。验证
  `item_results` 与全部 512 个输入严格对齐、合法项可查询、非法项无副作用；随后只重试失败项，验证 generation
  不变、`snapshot_required` 清除且 metadata 最终收敛。

pure-local Release 验证结果：

- snapshot HTTP 功能集（包含上述大批量用例）37/37、legacy/dual-storage HTTP 功能集 20/20 全绿；
- 5 个 reporter lifecycle/query 并发用例各重复 30 次，共 150 次通过；3 个 snapshot atomicity/lease/fused-RMW
  用例各重复 50 次，共 150 次通过；
- `CacheManagerTest` 全量重跑通过；其中一次并行构建压力下已有的 `TestFilterWriteCache_StaleSuffix` 报
  `std::future_error: Broken promise`，该用例隔离重复 100 次以及随后全量重跑均未复现，因此未把偶发结果误判为
  本轮生产缺陷或静默修改无关代码；
- `bazel test --config=release //kv_cache_manager/...` 共 106 个可执行测试通过，1 个 GPU-only 测试按预期跳过。

同一 Release 进程的功能后性能门禁仍保持：20K ReportEvent create/update 为 58.96/50.38ms；20K-key
GetHostCacheState local serial p50/p99/avg 为 14.92/16.03/15.11ms。该共享机数据只证明本轮测试加强没有引入明显
性能回退，不承诺线上固定 SLA。

### 5.26 2026-08-10 parser 差分与跨请求多 reporter 状态机复核

本轮从“快路径必须与原实现语义等价”出发补充差分测试，发现并修复一个此前用例未覆盖的真实兼容问题：
heartbeat 的 `system_status` JSON object 若包含重复 key，protobuf `JsonStringToMessage` 会拒绝请求，而 specialized
parser 原先通过 protobuf map 的 `operator[]` 静默采用最后一个值。这样同一 payload 会因为是否进入快速 parser
而产生不同结果。现在写入 map 前显式检查已存在 key；重复 key 让 fast conversion 失败，再由 generic parser 给出
与原协议完全一致的拒绝结果。该检查只位于低频 heartbeat map 解析，不进入 BLOCK_ADD/DELETE、metadata RMW 或
查询路径，也不改变任何锁范围。

新增 parser 兼容语料同时覆盖 canonical/snake/camel 字段、字符串和数字 enum、known field 为 null、未来 enum、
数字形式的 string 字段、字段 alias 重复、map key 重复、多个 oneof member、unknown nested value、Unicode surrogate
pair/未配对 surrogate、错误字段类型以及 null repeated/oneof entry。每个样例都执行两套比较：

1. immutable `ReportEventJsonParser::FromJson` 与 protobuf generic parser 的成功/失败和 message equality；
2. 追加未知 padding 形成大于 40 KiB 的 body，强制走 HTTP mutable in-situ 路径，再与 generic parser 做相同比较。

ReportEvent 状态模型也从单请求、单 reporter 扩展到跨请求场景：两个 reporter 交替执行 8 轮、每轮 96 个事件，
在 32 个 key、5 个 medium、3 个 spec 上做确定性随机 ADD/DELETE。每轮末尾强制更新两个 reporter 共用 block 的
各自 location，专门验证 fused targeted RMW 只替换目标 `(block_key, location_id)`，不会覆盖同 key 的另一 reporter。
每轮提交后均重新核对：

- 全部物理 location id、spec name、完整 versioned URI 和 location type；
- query-visible URI 必须来自 reference model，空 key 不能产生伪命中；
- 两个 reporter 的 committed generation 独立且跨 delta 保持不变；
- 删除最后一个 spec 后 location/key 收敛，所有 replacement/delete 后的 storage usage 与 reference 精确相等。

pure-local Release 验证结果：

- parser 完整测试进程重复 100 次通过；上述跨请求模型、单请求 flat-fold reference model 和容量部分失败映射分别
  重复 20 次通过；
- 真实 HTTP snapshot 功能集 37/37、legacy/dual-storage 功能集 20/20 全绿；另外直接发送保留重复 JSON member
  的小 heartbeat 和大于 40 KiB heartbeat，均返回 HTTP 400，证明 handler 两条 parser 路径行为一致；
- `bazel test --config=release //kv_cache_manager/... --nocache_test_results`：106 个可执行测试通过，1 个
  GPU-only 测试按预期跳过；
- 同一 Release + jemalloc 进程的 20K ReportEvent create/update 为 55.86/48.20ms；20K-key GetHostCacheState
  local serial p50/p99/avg 为 14.15/14.74/14.17ms，16-way p50/p99 为 35.88/50.53ms。数据表明本轮兼容修复和测试
  扩展没有造成 ADD/Get 性能回退，但仍只作为共享开发机门禁，不是线上 SLA。
