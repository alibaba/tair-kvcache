# LiteHit 多容量 LRU 命中率分析需求

状态：implemented（第一阶段）
日期：2026-07-13
适用阶段：第一阶段，仅 full attention

## 1. 背景

当前 optimizer 可以通过 trace replay 分析缓存命中率，但它面向完整优化流程，需要维护多组 cache、驱逐策略和中间统计。当分析目标只是“一段 trace 在多个 LRU 容量下的理论命中率”时，现有方案占用内存较高，且随着容量数量增加，运行时间明显增长。

需要新增一个独立、轻量、精确的 LiteHit 模块，只维护计算最终命中率所需的状态。它既能用于离线 trace 理论分析，也能作为 online optimizer 的基础统计能力。

## 2. 目标

- 给定 full-attention block 访问 trace，一次计算多个 LRU 容量下的命中次数和命中率。
- 支持有限容量和无限容量。
- 支持离线完整 trace 和 online 按请求流式输入。
- 离线与 online 使用同一套核心命中语义，结果可互相对拍。
- 结果必须精确，不使用采样或近似算法。
- 不为每个容量维护一套独立 LRU cache，降低多容量分析的时间和内存开销。

## 3. 使用场景

### 3.1 离线理论分析

分析方提供一段包含请求边界和有序 block key 的 trace，以及一组目标容量。LiteHit 扫描 trace 后返回每个容量的最终命中统计，用于容量规划、理论上界分析和不同 trace 的横向比较。

### 3.2 Online optimizer

online optimizer 按请求持续向 LiteHit 输入有序 block key。LiteHit 累积必要状态，并在查询时返回当前已处理流量在各目标容量下的命中统计。

第一阶段不要求 LiteHit 自身负责 trace 下载、解析、服务发现或指标上报；这些能力由调用方适配。

## 4. 输入模型

### 4.1 访问对象

- 一个访问对象对应一个 full-attention cache block，以 block key 唯一标识。
- 同一个 block key 的所有访问视为同一个缓存对象。
- block key 必须由输入层按前缀链式 hash 生成：请求第 j 个 key 是前 j 个完整 block 全部 token 的 hash，key 相等当且仅当整个 token 前缀相同。若只 hash 本 block 内容，跨请求复用语义不成立。LiteHit 只做 key 相等判断，不校验该契约。
- 由该契约可知同一请求内 key 互不相同；核心对请求内重复 key 的处理属于防御行为。
- 第一阶段假定所有 full-attention block 的 cache charge 相同。

### 4.2 请求

- 输入由按时间排序的请求组成。
- 每个请求包含一个有序的完整 block key 列表和原始 `input_token_len`。
- 初始化时固定 `block_size_tokens`，输入必须满足
  `block_keys.size() == floor(input_token_len / block_size_tokens)`。
- 不足一个完整 block 的尾部 token 不进入 LRU，但保留在命中率分母中。
- 请求之间共享同一 LRU cache 状态。
- 请求边界必须保留，以支持现有 optimizer 的 prefix-hit 统计语义。

### 4.3 容量

- 支持同时指定多个有限容量。
- 支持一个“无限容量”查询项。
- LiteHit 核心内部以可容纳的 block 数表示容量。
- 若调用方使用 bytes 或 GB，调用方或适配层根据固定 block charge 转换为 block 数。
- 容量为 0 时，命中次数必须为 0。
- 重复容量不得改变结果；返回顺序和去重行为由最终 API 约定。

## 5. 命中语义

### 5.1 LRU 状态

- 驱逐策略固定为精确 LRU。
- block 第一次出现时为 cold miss。
- 对于非 cold 访问，若 reuse distance 为 `d`，则容量 `C` 下命中的条件为 `d < C`。
- 每个访问都必须更新全局 LRU 状态，包括请求已经发生 prefix miss 后的后续 block。
- 状态提交按请求内倒序 touch（尾先、头后）：链头永远比链上后块新，全局驱逐牺牲者恒为链叶子，对齐生产 prefix cache 的尾先驱逐行为（vLLM 逆序释放 block、SGLang radix cache 叶子 LRU）。命中评估不受提交顺序影响，仍按请求正序基于请求前快照计算。

### 5.2 Full-attention prefix hit

第一阶段的主统计口径固定为 request prefix hit：

- 从请求第一个 block 开始连续判断。
- 第一个 miss 之前的 block 计为该请求的命中 block。
- 第一个 miss 及其后的 block 不计入该请求的命中数。
- 后续 block 虽不贡献该请求的命中数，仍需更新 LRU 状态。

第一阶段实现采用该口径，以对齐当前 online optimizer。若后续需要“每个 block 独立判断”的理论 cache hit rate，应将其定义为单独的输出口径，不能与 prefix hit 混用。

### 5.3 无限容量

- 无限容量不会发生 capacity miss，但 cold miss 仍然存在。
- 在逐 block 口径下，无限容量命中所有非 cold 访问。
- 在 prefix-hit 口径下，一个请求遇到首个 cold miss 后，该请求后续 block 不再贡献命中数，但仍更新全局状态。

## 6. 输出

LiteHit 对每个目标容量至少返回：

- 容量标识，包括有限 block 容量或无限容量。
- `hit_count`：该请求或累计的 prefix 命中 block 数。
- `hit_tokens = hit_count * block_size_tokens`。
- `input_tokens`：该请求的 `input_token_len`，或累计的 `input_token_len` 之和。
- `hit_rate = hit_tokens / input_tokens`。

还需要返回或可推导以下全局信息：

- 已处理请求数。
- 累计 input token 数。

当 `input_tokens` 为 0 时，命中率返回 `0.0`。

LiteHit 不输出逐 key 命中结果、reuse distance 明细、LRU 栈内容、驱逐次数、hit age、TTL 状态或普通 optimizer 的其他中间结果。

## 7. 功能要求

- `FR-1`：支持批量输入完整 trace 并在扫描结束后查询结果。
- `FR-1a`：离线批量入口通过逐请求 callback 流式交付单条结果，不在核心中保存请求结果历史。
- `FR-2`：支持按请求追加 online 流式输入。
- `FR-3`：一次分析支持多个有限容量和无限容量。
- `FR-4`：相同输入和容量必须得到确定、可重复的结果。
- `FR-5`：离线和 online 在处理相同请求序列后必须返回一致结果。
- `FR-6`：支持重置统计器，使实例可开始一段新的独立分析。
- `FR-7`：计数器和位置索引使用足以覆盖长 trace 的整数类型，避免 32 位溢出。
- `FR-8`：核心统计逻辑可被 standalone 离线工具和 online optimizer 复用，不依赖完整 optimizer manager。

## 8. 性能要求

- 不允许为每个目标容量分别回放完整 trace。
- 不允许为每个目标容量维护一套完整 LRU cache 状态。
- 对 `N` 个 block、`Q` 个请求和 `K` 个容量，核心计算复杂度为
  `O(N * (log U + log K) + Q * K)`；`Q * K` 是返回每请求多容量结果本身的输出成本。
- 空间只与需要追踪的不同 block、Fenwick 有效位置和必要聚合统计相关，不保存原始 trace 或逐容量 block 集合。
- online 模式不得要求预先知道完整流长度；具体动态数据结构由设计阶段确定。
- 第一阶段不设固定吞吐量和内存数值门槛，完成基础实现后使用真实 trace 建立 benchmark 基线。

## 9. 非目标

- 不支持 linear attention、checkpoint 或 Mamba state。
- 不支持 RandomLRU、LeafAwareLRU、TTL、admission policy、prefetch 或多级缓存。
- 不支持 block 大小动态变化、range read 或一个 key 对应多个不同 charge。
- 不负责修改或重放请求时间，不进行性能仿真。
- 不复制完整 optimizer 的配置、可视化和中间结果体系。
- 第一阶段不提供独立公共 CLI 或 Python API；现有 Online Optimizer RPC 只增加必要输入输出字段。

## 10. 验收标准

- 使用朴素的多容量 LRU 模拟器作为 oracle，对相同输入逐容量对拍，所有命中次数完全一致。
- 覆盖 cold miss、立即重复访问、重复对象去重、容量 0、容量 1、多个容量、重复容量、容量超过工作集和无限容量。
- 覆盖 request 内首个 miss 截断，以及截断后的 block 仍影响后续请求 cache 状态。
- 覆盖请求前快照评估和按 key 最后出现位置批量提交与朴素顺序 LRU 的等价性。
- 覆盖尾部 token 留在分母、累计率按总 hit token / 总 input token 计算而非请求率平均。
- 同一 trace 通过离线批量输入和 online 逐请求输入，最终结果完全一致。
- 空 trace 不崩溃、不除零，返回约定结果。
- 长 trace 下计数无溢出。
- benchmark 证明多容量场景不再随 `N * K` 线性回放，并记录相对当前 optimizer 的时间和峰值内存。
- 若核心代码进入 `kv_cache_manager/`，按仓库要求完成外源和内源测试验证。

## 11. 第一阶段 API 决策

1. 主输出采用 request prefix hit；第一阶段不额外输出逐 block 命中率。
2. 命中率分母是原始 `input_token_len`；分子是 `hit_count * block_size_tokens`。
3. LiteHit 核心容量单位为 block；Online Manager 或离线输入适配层负责从 GB/bytes 按固定 block charge 换算。
4. `-1` 表示无限容量，其他负数非法。
5. Online 容量集合在初始化时固定；结果保留输入顺序和重复项。
6. Online Optimizer 的 full-attention `InstanceState` 直接持有 LiteHit；不通过通用
   `CacheIndexer` adapter。linear attention 继续走 legacy indexer。
7. 第一阶段 full-attention + TTL 明确不支持，注册时返回参数错误。
8. 新 TraceQuery 调用显式传 `input_token_len`；旧调用缺省时先从 `token_ids` 推导，仍缺省则兼容假设请求没有不完整尾部，并由 `block_keys.size() * block_size_tokens` 推导。
9. LRU 状态更新固定为请求内倒序提交（见 5.1），不提供提交顺序开关。

## 12. 相关文档

- 算法说明：[README.md](README.md)
- Task：`../.agent/tasks/2026-07-09-litehit-lru-hit-analysis.md`
- Plan：`../.agent/plans/2026-07-09-litehit-lru-hit-analysis.md`
- Session 交接：`../.agent/SESSION_HANDOFF.md`
