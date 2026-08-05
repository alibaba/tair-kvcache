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
5. `BatchMergeLocationSpecs` 的第二阶段只保存原 task 下标，不再复制 location id 和整组 spec/URI。
   existing Location 仍做一次必要的 copy-on-write，之后在其小 vector 内按 name 原地覆盖、兼容 legacy
   重名并恢复字典序；不再构造每 key 的 ordered map 和第二份完整 spec vector。

这些优化没有合并两阶段 RMW，也没有改变 lifecycle lease、shard lock 或 HOST_DOWN 的锁序。收益来自减少
进入和持有 metadata shard 锁期间的 CPU/分配工作，因此既降低单请求 RT，也缩短并发请求的锁占用窗口；
不能把它解释成“把锁换成原子变量”。

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
结构性工作应是一次 shard lock 内同时返回“key 是否存在 + 目标 location value”的 fused targeted RMW，
而不是继续扩大 Get 查询线程数。

代码复核还确认两个次级问题：`GetHostCacheState` 的默认 access log 会先完整 protobuf-to-JSON，再由
access-log builder parse 为 DOM；`MakeBatches` 的 `batch_key_size` 是 shard-boundary soft limit，同一
shard 的 keys 不会被硬切。这两点都值得单独收敛和补指标，但前者在本轮 0.099 Get QPS 下不足以解释
共享压力，后者的 writer shard lock 也不阻塞纯 local Get。若线上 `mutex_shard_num=16`，10k keys 的
均匀请求约为 625 keys/shard；增加 128/256 的 lock-hold hard limit 主要改善写写公平性，必须用混合写
p99 验证，不能把它误报成 Get 尾延迟根因。
