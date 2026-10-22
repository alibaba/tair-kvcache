# KVCM 元数据内存紧凑化设计与实现

本文汇总原 `kvcm_mem_opt.md`、`kvcm_mem_opt_plan.md` 和本设计文档的内容，以当前代码为准描述已实现方案。范围是常驻元数据的 `CacheLocation`、`LocationSpec`、`MetaMemCacheItem` 与 local backend item 锁；不改变 URI 表示、Redis JSON、LRU 并发模型或模块依赖方向。

## 目标与历史基线

R62 的 5000 万 key 纯 Block Add 测试记录了单 key、单 location、单 spec 约 **494.2 B/key** 的 cgroup 净增；R64 的 10% 双 location 冷基线约 **568.3 B/key**。这些是优化前特定负载的数据，不是当前代码的实测值，原始压测报告不在本仓库。按十进制 20 亿 key / 800 GB 换算，零余量上限是 400 B/key，保留 20% 总内存余量则是 320 B/key；还要考虑固定开销、峰值、Redis 备份和副本。不能把单个对象的 `sizeof` 差额直接当作 cgroup 收益或达标证明。

主要链路仍是 `CacheManager / MetaSearcher → MetaIndexer → MetaLocalBackend → MetaMemCacheItem → CacheLocation / LocationSpec`。Redis 读与恢复经原有反序列化入口构造相同类型。一个 `CacheLocation` 可能被查询结果、任务或 COW 快照继续持有，因此 ID 和 spec name 的共享表示都必须拥有其引用对象的生命周期。

## 已实现的表示与同步

| 组件 | 当前实现 | 主要取舍 |
|---|---|---|
| [`CacheLocation`](../../kv_cache_manager/meta/cache_location.h) | ID 统一为 `shared_ptr<const string>`；`spec_size` 与已校验总大小共用 64 位字 | 缩小共享 ID 的对象布局；独立随机 ID 仍需单独分配字符串对象 |
| [`LocationSpec`](../../kv_cache_manager/meta/cache_location.h) | name 为 `shared_ptr<const string>`，URI 仍为独立 `string` | 在线复用配置名称，Redis 读复用名称字典；共享指针有引用计数成本 |
| [`MetaMemCacheItem`](../../kv_cache_manager/meta/meta_local_backend.h) | 单 location、规范 `prev_key` 内联；多 location / 额外属性共用按需创建的 `LocalItemExtra` | 主路径减少常驻字段；冷状态多一次对象分配与间接访问 |
| [`MetaLocalBackend`](../../kv_cache_manager/meta/meta_local_backend.cc) | 按 key 哈希选择 backend 级 `shared_mutex` 分段锁 | 不再逐 item 存锁；不同 key 可能撞锁，并产生固定锁数组开销 |

### Location ID 与大小字段

在线 EventReport 按请求中的 reporter / medium 构造一个不可变 ID 句柄，多个 block key 复用它；同一 location 的拷贝也复用句柄。同步和异步 Redis backend 各有自己的 `LocationIdInterner`：仅 EventReport 类型进入字典，16 个分片各最多保留 4096 个 ID key，总计不超过 65536；弱引用过期后可在同名再次访问时替换，超限的新 ID 使用独立句柄。StartWriteCache 产生的随机 ID 不进入字典，但当前 `set_id(string)` 仍以 `make_shared` 独立分配。恢复是否共享只影响内存表示，不改变 JSON 字段或 location ID 文本。

`packed_sizes_` 的高 16 位记录 spec 个数，低 48 位记录已校验的 location 总大小。总大小达到 `2^48-1` 时使用未知哨兵并回到 URI 解析路径，不截断实际大小；spec 个数超过 65535 时从 `location_specs_.size()` 取得。spec 修改会清除已校验标记。`spec_size` 是 location 的 spec **个数**，并非配置中 `LocationSpecInfo::size` 的字节数。`status` 仍为 32 位枚举，`create_time` 仍为 64 位时间戳，JSON 结构不变。

旧表示将 `string` 与共享指针放进同一个 `variant`；即使 EventReport 选择共享指针，每个 location 仍须为较大的内联 `string` 预留空间。当前布局避免了这项主路径成本，但短随机 ID 会增加一次字符串对象/控制块分配。这是按业务形态权衡的结果，不应在缺少高基数 StartWriteCache 对照数据时宣称所有路径都省内存。

### Spec name 的生命周期

在线 StartWriteCache、查询时的空 spec 补齐和 ReportEvent 复用已取得的 `InstanceInfoConstPtr`，用别名 `shared_ptr` 指向其中的 `LocationSpecInfo::name()`；迁移、替换、删除和 COW 传递既有句柄，而不是按文本重建名称。旧 location 持有旧配置快照，新 location 使用新快照；这依赖实例信息在发布后不原位修改。默认空名称引用静态空字符串，移动后的源 `LocationSpec::name()` 仍有效。

Redis 读/恢复没有相同的配置句柄。两个 Redis backend 的直接读取及共用反序列化入口使用各 backend 的 `LocationSpecNameInterner`；字典最多保留 65536 个名称 key，用弱引用共享存活名称，命中走读锁、未命中走写锁。满额后新名称退化为独立分配；条目数有界不代表名称总字节数有界。普通 `FromJsonString` 和接受 `string` 的构造/设置接口仍可独立解析、分配名称，不能假设每个 `LocationSpec` 都共享同一控制块。

### Item 冷状态与分段锁

`MetaMemCacheItem` 的常见形态是一项内联 location、规范整数或空的 `prev_key`，没有额外 map。第二个 location 或非内联属性出现时才创建 `LocalItemExtra`，其中分别按需持有 location map 和 property map；恢复单 location、删除最后一个额外属性后释放空对象。规范 `prev_key` 保持整数加三态表示，非规范字符串仍走原有属性 map 回退。创建、合并、删除和归一化沿原有 local backend 链路计算 LRU charge delta，不为每次写入全量扫描 map。

item 数据的读写与维护操作使用相同的 `ItemMutex(key)`：`GetShardIndex` 将 key 映射到 `2^num_item_mutex_shard_bits` 把 backend 级读写锁。该参数从 local backend 的 `storage_uri` 读取，默认 16（65536 把），最大 24，和 LRU 的 `num_shard_bits` 独立；默认固定成本约 3.5 MiB/backend。维护回调可能先持有 LRU shard 锁再取 item 锁，因此普通路径必须释放 item 锁后才能调用可能获取 LRU 锁的 `AdjustCharge`、`Release`、`Erase`。锁数组按 backend 创建，不应把高位数配置视为免费。

### 容量计费的口径

`CacheLocation::EstimateMemUsage()` 对 spec vector 按 **capacity** 计缓冲；堆字符串按容量加结尾字节计，短字符串保留原有长度 charge。独立 ID 另计字符串对象及控制块的估算，共享 ID/name 则逐 item 计文本长度，以便 LRU charge 不随其他持有者变化而波动。`MetaMemCacheItem::Size()` 加上 item、location/property 和 `LocalItemExtra` 的估算；LRU 再计 handle 元数据，`MetaLocalBackend::GetMemUsage()` 返回这套累计 **charge**。

charge 不是物理分配或 RSS：短字符串可能重复计字符，共享文本会逐项重复计，而独立 spec name、`CacheLocation` 控制块、map bucket、jemalloc 档位与页开销并未精确摊销。它会影响 local cache 的严格容量准入，但不能直接代替 cgroup 预算。当前 local backend 使用 `strict_capacity_limit=true`、`no_evict_on_insert=true`；不能仅靠调低 `capacity` 就把全量 local 索引变成可自动回源的热缓存。

## 验证与边界

代码回归覆盖对象布局、JSON 往返、共享 ID/name 与移动后的生命周期、字典并发/上限、item map 与属性的创建/释放及 charge delta、同 key/同分段锁的维护并发。相关测试位于 `MetaSearcherTest`、`meta_local_backend_test`、Redis backend、`MigrationManagerTest` 和 `CacheManagerTest`；布局测试限制 `sizeof(CacheLocation) <= 80`、`sizeof(LocationSpec) <= 56`、`sizeof(MetaMemCacheItem) <= 48`，但这些断言不代表实际分配档位或集群内存。

合入与容量验收还需在相同镜像、配置和 key 分布下对照 **cgroup 净增/key、jemalloc allocated/active、LRU charge、恢复峰值、成功率、QPS、p95/p99 和锁等待**。至少覆盖 EventReport 大批量共享 ID/name、高基数随机 StartWriteCache、多 spec、多 location、非规范属性、更新/删除及 Redis 恢复。固定锁数组、独立 ID 分配、共享指针引用计数与 Redis 字典锁都可能使某些形态回退；没有这些 A/B 数据，不能宣称达到 20 亿 key / 800 GB 或完全没有性能劣化。

本轮没有实现 URI 压缩、`LocationSpec` 去虚表、单 spec 内联、定长 key 专用 LRU，也没有改变 local 的全量驻留语义。这些方案会改变现有 `uri()` / `location_specs()` API、JSON 模板或缓存并发行为，须在对应分配与负载证据明确后单独设计。
