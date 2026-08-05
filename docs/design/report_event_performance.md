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

### 2.2 lifecycle lease 从每 key 收敛到每 RMW 阶段一次

原实现会在 modifier 的每个 key 上查 reporter fence、分配 `shared_lock`；已有 location 的 ADD
经过 block-create 和 targeted-location 两阶段时还会重复一轮。现在每个 RMW 阶段第一次进入
modifier 时获取一次，后续 key 复用同一个 lease：

```text
metadata read（可被 lifecycle writer 抢占）
        |
non-blocking lifecycle lease（每阶段一次）
        |
本阶段全部 metadata mutation
        |
释放 lease
```

lease 不能在 metadata read 前获取，也不能无条件持有整个 ReportEvent。HOST_DOWN/REGISTER 的
锁序是 `lifecycle -> metadata`；如果旧请求阻塞在 metadata I/O，lifecycle writer 必须能先完成。
旧请求恢复后 `try_lock` 失败并放弃写入。BatchMerge 的两个 RMW 阶段之间释放并重新获取，保留同样
的抢占窗口。确定性 HOST_DOWN/重注册竞态测试是这个优化的强制回归项。

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
4. metadata read 完成后，第一次处理 event-report location 时用 `std::call_once` 为该请求抓取 reporter
   liveness 与 committed-version 快照。每个 backend 只持有一次 `nodes_mutex_` shared lock，后续
   `(block, location)` 只读不可变快照；
5. host/spec 投影和候选 host 前缀归约复用同一个有界 executor，输出仍按 host 字典序构造，普通 prefix、
   Mamba、Eagle pop 和 medium filter 的结果语义不变。

可见性快照在 metadata read 之后开始采集。采集前已经可见的 HOST_DOWN 会被当前请求过滤；与采集并发
的 HOST_DOWN 允许当前请求看到前或后的状态，但采集完成后本请求不再变化，下一请求会重新采集。由于
`available` 是逐 reporter atomic，多个 reporter 与并发 liveness 变化之间不承诺一个全局事务时间点；这里
保证的是 request-stable 结果，避免同一 reporter 在一次长 projection 中前半段 up、后半段 down，也把
stale window 放在耗时 metadata read 之后。

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

## 3. 当前明确不做的事情

暂不并行执行 ReportEvent 内的 metadata batch。原因不是并行永远无效，而是当前主要指标已经指向
backend I/O；直接并行可能把排队转移到 Redis、放大连接池竞争并拖高查询 p99，同时会引入
key-count 容量、同 key mutation 顺序和 lifecycle fencing 的新并发面。

暂不把 `BatchMergeLocationSpecs` 的两阶段 RMW 简单合并。第一阶段通过“整个 block 是否存在”维护
`key_count/max_key_count`，第二阶段按目标 location 做 merge。仅使用目标 HMGET 无法区分“key 不存在”
和“key 存在但目标 location 不存在”；直接替换会造成容量计数错误。

GetHostCacheState 也不并发 Redis/cached backend batch，不创建 request-local thread，不复用只有少量
worker 且承载回收/迁移的 `SchedulePlanExecutor`。查询池独立且有界，避免长 metadata 请求饿死系统任务。

## 4. 下一步最值得验证的 I/O 优化

Redis `GetLocationIds` 当前为每批 key 做 EXISTS，再用 HSCAN 枚举 location field；已有目标 location
还要进入 targeted HMGET。若优化后线上 `get_io_time_us` 仍占主导，优先设计一个 backend/indexer
原语，在保持 shard lock 的一次 RMW 中同时返回：

1. key 是否存在（用于 `key_count/max_key_count`）；
2. 请求指定 location id 的值与逐项错误；
3. 不枚举、不传输无关 location value。

Redis 实现应尽量把 EXISTS 与目标 HMGET 放进同一 pipeline round trip；Local/Dummy/Async Redis 和
cached+recover 模式必须具有一致语义。只有补齐新 key、已有 key 缺 location、已有 location、
properties-only key、tombstone、部分 I/O 失败和 max-key-count UT 后，才能替换现有两阶段逻辑。

## 5. 验证与观测清单

- UT：请求内 512 个跨重复 medium 的 ADD；并发 EnsureNodeRegistered；每 RMW 阶段 lease 次数与失败
  原子性；HOST_DOWN、REGISTER、新 snapshot 抢占阻塞 metadata read；同请求 ADD/DELETE 顺序。
- 手工容量：`EventReportBenchTest.test_20_large_single_request_delta_scaling` 分别记录 100/1000/5000/20000
  个新 block ADD 与相同 block 再次 ADD 的总 RT、单 event RT，并查询首/中/末 block，不能只看
  HTTP 成功码。
  可在启动 KVCM 后用 `--bench-test test_20_large_single_request_delta_scaling` 单独执行。纯 cache
  部署保持默认 local metadata backend；只有明确验证 Redis 部署形态时才传 `--meta-storage-uri`。
- 线上对比：至少拆分 request parse/fold、node ensure、lifecycle lease wait/fail、RMW lock wait、
  `get_io_time_us`、serialize、enqueue/upsert 和完整 ReportEvent RT；同时观察查询 p50/p99。
- 若去重后 `get_io_time_us` 仍接近总 RT，下一步应做目标化 backend read，而不是先加线程。
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

Release/O2、纯 local 下，当前版本单请求 20k BLOCK_ADD 的 create/update 为
196.55/239.51ms，约 9.8/12.0us 每 event，整体近似线性。对比 `feature/event_report_4@b776dd4`，
5000-event 单请求及 100-thread ADD、50-thread mixed throughput 均只相差约 0~4%，属于噪声；当前改动
没有回归 ReportEvent，但也不能宣称改善其高并发尾延迟。

一次 20k existing-location update 的详细观测为：客户端 RT 249.59ms，KVCM service/event timer
约 161ms，block RMW 32.18ms，location RMW 63.38ms，local backend get/upsert I/O 仅
7.33/6.04ms，metadata lock wait 约 1us。也就是说，剩余成本不是 Redis 或全局锁，而是：

1. HTTP/JSON/protobuf 请求解析与响应边界约 89ms；
2. 两阶段 RMW 内逐 key 的 CPU、拷贝和 modifier 约 95ms，其中真实 backend I/O 约 13ms。

若线上 10k~20k 单批 ReportEvent 仍需显著低于 100ms，下一轮应单独设计 local RMW shard 并行或减少
block-create/targeted-location 两阶段数据变换，并增加 parse/fold 指标。该改动会触及写入原子性、
lifecycle fence 和锁序，不能作为本轮查询优化的顺手补丁；上线前可先限制单请求 event 数或分批上报。
