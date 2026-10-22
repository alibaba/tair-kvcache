# KVCM 内存优化实施方案：思路 1 与思路 3

> 基于 [kvcm_mem_opt.md](kvcm_mem_opt.md) 的 R62 实测（单 location、单 spec、纯写入）。本轮只实施 CacheLocation 紧凑化的安全子集和 item 分段锁；URI 压缩不在范围内。换算口径：每 key 省 1 B ≈ 20 亿 key 省 2 GB。

## 一句话结论

R62 测得约 494 B/key，20 亿 key 约 988 GB。对象档位估算不能直接相加，也不能当作线上收益承诺。本轮以思路 1a 的共享 ID/安全字段收窄和思路 3 的分段锁降低常驻开销；是否达到 800 GB，必须用同负载 A/B 的 cgroup 净增与吞吐实测判断。思路 1b 的单 spec 内联与 name 字典暂不实施，原因见下文。

## 现状：每个 key 的内存花在哪

一个单 location 的 key 当前有多次小对象分配；下列档位来自 R62 的推断，并非每个分配点都有堆栈归因：

| 对象 | 估算大小 | jemalloc 档位 | 说明 |
|---|---:|---:|---|
| LRU handle | 79 B | 80 B | 通用 LRU 节点，8 字节 key 也按变长 key 设计 |
| `MetaMemCacheItem` | 112 B | 112 B | 其中 `shared_mutex` 独占 **56 B** |
| `CacheLocation` + shared_ptr 控制块 | 104+16 B | 128 B | `id` 用 `variant<string, shared_ptr>` 占 40 B；多个 8 字节字段实际用不满 |
| `vector<LocationSpec>` 的堆缓冲 | 72 B | 80 B | 只有 1 个 spec 也要单独分配一次 |
| URI 字符串堆内存 | ~64–79 B | 80 B | URI 分布需用真实负载再确认 |

关键规律：需核对 `sizeof`、`make_shared` 控制块及实际分配档位；字段总字节数不等于进程净省内存，其他对象和 allocator 碎片也会影响结果。

## 实施范围与取舍

### 思路 1：CacheLocation 紧凑化

本轮 1a：将 `id` 的 `variant<string, shared_ptr>` 改为共享指针，保留既有在线写链路传入的 interned ID；Redis 两种 backend 的反序列化仅对 EventReport location 使用**各自有界、弱引用的 ID 字典**，随机 ID 不进入字典。字典按 ID 哈希分成 16 个独立锁/表，每片最多 4096 个条目，总量不超过 65536；超限的新 ID 回退独立分配，实例/backend 退出时释放字典。`validated_total_size` 与 spec 个数共用一个 64 位字：前者占 48 位，≥ `2^48-1` 时标记为未知并沿用解析 URI 的 64 位路径；后者占 16 位，超过其上界时从 `location_specs` 数量取得实际值。`CacheLocation::spec_size` 是 location 的 spec **个数**，不是实例配置中 `LocationSpecInfo::size` 的字节数（后者才有 `-1` 哨兵）。`status` 保持 32 位枚举：现有 JSON 反序列化允许超出已知状态的整数，收窄底层类型会产生回绕。`create_time` 保持 64 位微秒时间戳。

本轮保持 `location_specs()` / `mutable_location_specs()` 的 `std::vector` API 和 JSON 格式不变。单 spec 内联不是仅改类内字段：`LocationSpec` 自身带虚表，现有调用返回 `vector&` 且部分调用依赖其 `data()`；`name()` 返回 `const string&`，字典必须处理指针逃逸和实例生命周期。未经原型测量和接口调整，直接内联甚至可能把 `CacheLocation` 推入更大的分配档位。因此 1b 暂缓，不计入本轮收益。

本轮布局验收：`sizeof(CacheLocation) <= 80`（当前构建配置），再以 allocator 采样确认对象/控制块是否从 128 B 降到 96 B。共享 ID 对非重复 ID 可能额外分配，必须同时测在线写、Redis 恢复和高基数 ID 场景。`EstimateMemUsage()` 对独立 ID 计入外置字符串及控制块，对共享 ID 保留原有按每 key 计长度的保守口径；若物理/逻辑容量偏差影响淘汰，再依据实测调整。

### 思路 2：URI 公共前缀压缩（范围外）

本轮不实施、不计入收益。须先统计真实 URI 的前缀基数与后缀长度，并解决 `uri()` 返回 `const string&`、共享 `CacheLocation` 指针可能逃逸时的字符串生命周期；不能简单在 backend 读出时临时解压。

### 思路 3：去掉 MetaMemCacheItem 的 mutex

**做什么**：去掉每个 item 的 `shared_mutex`，在每个 local backend 中按 `GetShardIndex(key)` 共用 item 分片锁。独立 URI 参数 `num_item_mutex_shard_bits` 默认 16，即 65536 把 `shared_mutex`，固定开销约 3.5 MiB/backend；不改变 LRU 的 `num_shard_bits`。`sizeof(MetaMemCacheItem) <= 64` 作为布局测试；需根据并发数、实例数和锁等待分布评估默认值。

**预期收益**：若 item 实际由 112 B 档降到 64 B 档，约 48 B/key；需扣除固定锁数组并以进程净增确认。

**两种实现对比**

| | COW（写时复制） | 分段锁（推荐） |
|---|---|---|
| 做法 | 读无锁拿快照，写时整体替换 | backend 里放可配置数量的锁，key 哈希选锁 |
| 省内存 | 需实测 | 预计 48 B/key，默认另有约 3.5 MiB/backend 固定开销 |
| 改动面 | 大：读、写、维护扫描、恢复全部要改并发模型 | 小：替换 `meta_local_backend.cc` 的 item 加锁点 |
| 主要风险 | C++17 原子共享指针和对象回收成本需单独论证 | 不同 key 会撞锁；锁顺序若处理不当会死锁；多实例固定锁成本需实测 |

锁顺序约束：`ApplyToEntryNoTouch` 在持有 LRU shard 锁时再取 item 分段锁；普通写从 `Lookup` 得到 handle 后取分段锁，但必须**释放分段锁后**才调用 `AdjustCharge` / `Release` / `Erase` 等可能获取 LRU 锁的操作。维护读/删与普通读/写必须对同一 key 选同一把锁，任何路径不得持分段锁再等待 LRU 锁。增加同 key 和同 stripe 不同 key 的并发读写/维护回归测试。

**建议：去 mutex，做；用分段锁实现。COW 暂不建议，可作为后续读性能优化再评估。**

## 收益边界

| 方案 | 预计 B/key | 20 亿 key 内存 | 能否满足 800 GB |
|---|---:|---:|---|
| 现状 | 494 | ~988 GB | 否 |
| 思路 3 + 本轮思路 1a 安全子集 | 约 414（只按 48+32 B 估算） | 约 828 GB | 不能据此宣称达标 |
| 思路 1b / 2 | 本轮未实施 | 不计入 | 需要单独原型和实测 |

上表仅是对象档位的条件估算，不是可合入或达成 800 GB 的证明。若 A/B 未达标，应优先分析真实对象分配堆栈、ID 基数、第二 location 比例，再决定是否做 1b 或 URI 压缩。

## 实施顺序与验收

1. 本轮：思路 3 分段锁 + 思路 1a 安全子集。先过单测、并发维护回归、Redis 恢复和 JSON 兼容，再运行 R62 同负载 A/B。
2. 后续仅在需要时：为 1b 建立 `LocationSpec` 和容器布局原型、列出 `vector&` API 替换点并验证生命周期；URI 压缩另案评估，不进入本轮代码。

用 R62 同镜像、同配置、同数据形态做冷基线 A/B，以 **cgroup 净增/key** 验收；同时检查写入/读取吞吐与 p95/p99 延迟、恢复吞吐/峰值内存、高基数 ID、不同 key 锁碰撞、multi-location/属性/删除场景正确性。没有这些数据时，只能报告单测和布局结果，不能报告 800 GB 目标已达成。

## 前提与风险

- 以上为基于对象布局和 jemalloc 档位的估算，需逐期实测确认，不是对 20 亿 key 的承诺。
- 只覆盖**单 location** 主路径。R64 中 10% key 带第二个 location 时已达 568 B/key，多 location 走 `unordered_map`，开销大；若真实业务多 location 比例高，需要再加“少量 location 内联”优化。
- `EstimateMemUsage()` 使用 `sizeof(CacheLocation)`，本轮布局变化会自动反映；独立 ID 的外置分配另行计费，共享 ID 长度仍按每 key 保守计费，需与物理占用一并观测。
- 800 GB 仅指一份 KVCM 内存索引，不含 Redis 全量备份或第二副本。
