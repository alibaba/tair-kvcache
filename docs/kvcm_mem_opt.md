# KVCM 20 亿 key / 800 GB 内存容量优化分析

## 结论与容量缺口

**当前 R62 对象布局放不下。**在纯新增、每 key 一个 location 的 5000 万 key 测试中，KVCM cgroup 内存净增 **494.2 B/key**。按相同数据形态线性外推，20 亿 key 约需 **988.4 GB**，尚未计入固定开销、恢复与写入峰值、Redis 备份或副本。800 GB 如果指 800 GiB，硬上限换算为约 429.5 B/key，结论仍不变。

| 800 GB 预算口径（十进制） | 每 key 上限 | 相对当前至少节省 |
|---|---:|---:|
| 用满全部内存，零余量 | 400 B | 94.2 B/key |github-opensource/docs/kvcm_mem_opt.md
| 总内存留 20% 余量 | 320 B | 174.2 B/key |
| 总内存留 25% 余量 | 300 B | 194.2 B/key |

考虑固定基线和负载峰值，**研发验收目标宜定为 cgroup 净增约 280–300 B/key**，即比当前再省约 **194–214 B/key**；该目标仅针对一份 KVCM 索引，不包含 Redis 全量备份或第二份内存副本。若真实数据有更多 location、property 或更长 URI，应按真实分布重算，不能直接套用单 location 结果。冷基线 R64 中，10% key 带第二个 location 时净增已达 **568.3 B/key**，外推 20 亿 key 约 **1.137 TB**。

## 实测依据与内存构成

R62 在 `memory_primary=true`、单 location、纯 Block Add、5000 万 key、48 GiB 容器、jemalloc `narenas=48` 下得到：

| 指标 | 5000 万 key 净增 | 平均每 key |
|---|---:|---:|
| KVCM 进程 RSS | 22.927 GiB | 492.4 B |
| KVCM cgroup 内存 | 23.015 GiB | **494.2 B** |
| jemalloc `allocated` | 22.738 GiB | 488.3 B |
| `total_cache_usage` 逻辑 charge | 22.767 GiB | 488.9 B |

jemalloc 主要 size class 的净增为 **80 B 档约 3 个/key、128 B 档约 1 个/key、112 B 档约 1 个/key**，合计约 **480 B/key**。这反映大量常驻小对象；`allocated` 与 cgroup 增量只差约 6 B/key，单靠 `narenas`、purge 或碎片治理无法填补 174–214 B/key 的目标缺口。size class 没有分配栈，不能把每档无条件指认为唯一 C++ 类型。

`total_cache_usage` 是 `MetaMemCacheItem::Size()` 加 LRU 元数据的逻辑计费，不是进程 RSS；容量验收应使用稳定基线到终态的 **cgroup 净增/key**，并分别记录 RSS、jemalloc `allocated/active`、cache charge 和峰值。压测脚本的 `spec_size=1024` 是逻辑 block 大小，不代表 KVCM 为每个 key 分配 1 KiB 数据。

## 优化路线：先减少逐 key 分配

| 优先级 | 方向 | 预期作用与限制 |
|---|---|---|
| P1 | 压缩单 location / 单 spec 表示 | 将当前分散的 `CacheLocation`、`LocationSpec` 容器及字符串改为少量紧凑分配；对重复的 host、medium、URI 建立有生命周期管理的共享字典，在逐 key 记录中保存数值 ID。目标是消除至少一到两个逐 key 小对象分配。收益需用真实 URI/属性分布实测。 |
| P2 | 将逐 key `shared_mutex` 改为 backend 分片锁 | 当前 `MetaMemCacheItem` 落在约 112 B 分配档；若安全地降到约 64 B 档，估计可省约 **48 B/key**。普通读、maintenance、恢复并不统一持有上层 meta-indexer 锁，不能直接删除 item 锁；必须验证锁顺序、并发和热点冲突。 |
| P3 | 压缩固定 8 字节 key 的索引/LRU handle | 通用 LRU handle 约 **80 B/key**，可评估 fixed-key 布局、字段与指针压缩；会触及公共缓存实现，改动与性能风险较高，应在 P1/P2 的实际收益确定后实施。 |

上述收益**不能直接相加**：对象重排会改变 jemalloc size class，且字典可能引入引用计数、生命周期和并发成本。仅省 48 B 的锁或一个 80 B 分配仍不足以安全落入 800 GB；需要组合优化并以 280–300 B/key 的实测结果验收。P1 尤其要覆盖单/多 location 转换、更新、删除、COW、序列化、恢复、字典引用回收与实例隔离，不能只验证 Block Add。

## 若不要求 20 亿 key 全部常驻

若工作集显著小于 20 亿，长期方案是**持久层保存全量索引、内存只保留热 key**。但当前 cached local **不是**可直接缩容的热缓存：`MetaLocalBackend` 使用 `strict_capacity_limit=true`、`no_evict_on_insert=true`；`MetaStorageBackendManager` 在 Running 状态的读命中本地 miss 后不再正常回源。直接调低 local `capacity` 可能导致 `EC_NOSPC`，或让已存在的 key 在本地不可见。

真正有界缓存需要独立设计：全量持久层为 source of truth、Running miss 可批量回源并回填、淘汰与并发写/删除/tombstone 一致、key count 不依赖驻留集合，并重新验证恢复和故障切换。若 800 GB 包括 Redis 全量备份或高可用内存副本，单份 KVCM 达到 280–300 B/key 也**不足以证明整体可容纳**；需要磁盘型后端、分层索引，或增加总内存。关闭 `memory_primary` 只会改变主要内存所在层，不会自动消除全量元数据成本。

## 不改代码的资源量与实施门槛

按 R62 的 48 GiB 容器、写入前 cgroup 基线 2.277 GiB、80% 使用水位和 494.2 B/key 斜率，每容器约容纳 **7850 万同形态 key**。20 亿 key 约需 **26 个同规格容器**，合计 **1248 GiB（约 1.34 TB）容器内存限额**，尚不含 Redis 和副本。分片可以控制单进程规模，但不能降低总的逐 key 内存成本；`max_key_count` 和 LRU `capacity` 也不是 RSS 保护的替代品。

实施前先量化真实业务的 location/key、property/key、URI 长度、有效/无效 key 与热工作集比例。每项优化独立做相同镜像配置、相同数据形态的冷基线 A/B，记录：

1. 稳定终态 `Δcgroup / Δkey`、RSS、jemalloc size class 与逻辑 charge；至少从千万级扩到 5000 万、上亿 key，检查非线性增长与恢复峰值。
2. ReportEvent、读取、删除的成功率、吞吐、CPU、延迟；尤其验证 lock wait/hold、异步 Redis queue、flush 与丢失/超时指标，不以 CPU 打满作为验收条件。
3. 多 location、属性、更新删除、重启恢复及 Redis 不可用场景；确认性能、数据一致性与容量门槛均通过后，再按每容器目标 key 数设置 `max_key_count`、cache capacity 和告警水位。

## 数据与代码依据

- [R62：5000 万单 location key 容量与 jemalloc 报告](kvcm_tr_instance_1_addonly_q200_k500_500s_50m_jemalloc_20260928_r62/KVCM_ADDONLY_50M_MEMORY_JEMALLOC_R62_20260928.md)
- [R64：10% 双 location 冷基线容量报告](kvcm_tr_instance_1_addonly_q200_k500_100s_10m_10pct_duallocation_cold_jemalloc_20261008_r64/KVCM_ADDONLY_10M_10PCT_DUAL_LOCATION_COLD_MEMORY_R64_20261008.md)
- [MetaMemCacheItem 与 location/property 存储](tair-kvcache/kv_cache_manager/meta/meta_local_backend.h)、[CacheLocation 与 LocationSpec](tair-kvcache/kv_cache_manager/meta/cache_location.h)
- [LRU handle](tair-kvcache/kv_cache_manager/common/cache/lru_cache.h)、[local cache 初始化](tair-kvcache/kv_cache_manager/meta/meta_local_backend.cc)、[Running 读路径](tair-kvcache/kv_cache_manager/meta/meta_storage_backend_manager.cc)

以上数字是指定压测形态的容量估算，不是对 20 亿 key 的实测承诺。
