# ReportEvent 小 block / 大批量性能优化记录

本文记录 `ReportEvent` 在 block size 较小、单次或并发上报 block 数较多时的性能分析、已经实施的
低风险优化、必须保持的并发语义，以及后续继续优化的边界。后续 AI 或开发者应先阅读本文和
[`report_event_snapshot_uri_version.md`](report_event_snapshot_uri_version.md)，不要仅根据某个 RT
指标直接引入并行。

## 1. 现象与指标解释

线上曾观察到 `ReportEvent` 与查询 RT 接近 100ms，同时 `meta_indexer.get_io_time_us` 可占约
80ms，查询侧 `manager.prefix_match_time_us` 也偏高。需要分开判断：

- `meta_indexer.get_io_time_us` 表示 metadata backend 的读 I/O；对 Redis 而言，批量 key 越多，
  pipeline 的命令数、reply 大小和 HSCAN/HMGET 成本越高；
- `manager.prefix_match_time_us` 属于查询候选计算，不能用它直接证明 ReportEvent 本地锁慢；
- ReportEvent 原有的节点表独占锁和 lifecycle lease 放大会增加 CPU/排队与 heap allocation，
  但不会解释全部 80ms Redis I/O。优化后仍需按阶段观测，而不是预期一个改动消除所有 RT。

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

## 3. 当前明确不做的事情

暂不并行执行 ReportEvent 内的 metadata batch。原因不是并行永远无效，而是当前主要指标已经指向
backend I/O；直接并行可能把排队转移到 Redis、放大连接池竞争并拖高查询 p99，同时会引入
key-count 容量、同 key mutation 顺序和 lifecycle fencing 的新并发面。

暂不把 `BatchMergeLocationSpecs` 的两阶段 RMW 简单合并。第一阶段通过“整个 block 是否存在”维护
`key_count/max_key_count`，第二阶段按目标 location 做 merge。仅使用目标 HMGET 无法区分“key 不存在”
和“key 存在但目标 location 不存在”；直接替换会造成容量计数错误。

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
- 手工容量：`EventReportBenchTest.test_20_large_single_request_delta_scaling` 分别记录 100/1000/5000
  个新 block ADD 与相同 block 再次 ADD 的总 RT、单 event RT，并查询首/中/末 block，不能只看
  HTTP 成功码。
  可在启动 KVCM 后用 `--bench-test test_20_large_single_request_delta_scaling` 单独执行；评估线上
  metadata I/O 时必须给 `--meta-storage-uri` 传入 Redis，而不是只测 local backend。
- 线上对比：至少拆分 request parse/fold、node ensure、lifecycle lease wait/fail、RMW lock wait、
  `get_io_time_us`、serialize、enqueue/upsert 和完整 ReportEvent RT；同时观察查询 p50/p99。
- 若去重后 `get_io_time_us` 仍接近总 RT，下一步应做目标化 backend read，而不是先加线程。
- 若 backend I/O 已显著下降但 CPU/锁等待仍主导，再评估按互斥 shard 分组的有界并行（建议先从
  2~4 并发开始），并对查询 p99、连接池等待和 Redis CPU 设置回退阈值。

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
