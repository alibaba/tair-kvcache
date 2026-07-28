# LiteHit 容量无关事实与多容量 LRU 命中率分析需求

状态：implemented（第一阶段 + Full-attention 事实化改造）
日期：2026-07-13（初版）/ 2026-07-22（事实化改造）
适用阶段：仅 full attention、等 charge block

## 1. 背景

当前 optimizer 可以通过 trace replay 分析缓存命中率，但它面向完整优化流程，需要维护多组 cache、驱逐策略和中间统计。当分析目标只是"一段 trace 在多个 LRU 容量下的理论命中率"时，现有方案占用内存较高，且随着容量数量增加，运行时间明显增长。

初版 LiteHit 在核心内接收固定容量集合并累计逐容量结果。事实化改造进一步把核心与容量彻底解耦：核心一次回放只产出**容量无关的事实**（每请求一条 hit curve），任何容量的命中数由无状态投影器事后推出，Offline 将事实作为可对账账本原子落盘，容量查询变成对账本的事后投影。

## 2. 目标

- 核心回放一次 trace，产出容量无关、可复算的事实（`RequestFact`）；容量不需要在分析开始前给定。
- 所有"容量 → 命中块数"换算收敛到唯一投影入口 `HitCurveProjector`（Online 与 facts query 共用）。
- 支持有限容量、容量 0 和无限容量的事后投影。
- Offline 将事实以 CSV 账本落盘（原子发布、fail-fast），并提供事后容量查询工具。
- Online 逐请求投影到配置容量 slot，仅保留最小累计整数。
- 离线与 online 使用同一套核心与预处理，结果可交叉对账。
- 结果必须精确，不使用采样或近似算法；非契约输入允许悲观（下界）、绝不乐观。
- 不为每个容量维护一套独立 LRU cache。

## 3. 使用场景

### 3.1 离线事实生产与事后容量查询

分析方提供标准格式 trace 与实例配置，Offline runner 一次回放产出 `litehit_facts.csv`；之后可用 facts query 工具对任意容量列表（含重复、0、无限）做事后投影，无需重放 trace。

### 3.2 Online optimizer

online optimizer 按请求持续向 LiteHit 输入有序 block key，逐请求把事实投影到配置容量 slot 并累计整数；查询时返回当前累计命中统计。Online 第一阶段不落盘事实。

## 4. 输入模型

### 4.1 访问对象

- 一个访问对象对应一个 full-attention cache block，以 block key 唯一标识。
- block key 必须满足前缀链式 hash 契约：请求第 j 个 key 是前 j 个完整 block 全部 token 的 hash，key 相等当且仅当整个 token 前缀相同。
- 当输入为逐块独立 hash 时，在 Instance Group 上开启 `enable_prefix_hash`（key 形态跟模型部署走，作用于 group 内全部实例；online 建组 RPC 与 offline 配置共用同一字段），由共享预处理用滚动 hash（与 Python 生产端 `prefix_hash.py` 逐 bit 一致的 Jenkins 64 位变体，uint64 逻辑右移）转换为前缀链式 key。
- 由契约可知同一请求内 key 互不相同；核心对非契约输入施加单调防御（门槛只抬不降），投影悲观、绝不乐观。
- 所有 full-attention block 的 cache charge 相同（等 charge 不变量）；`linear_step != 0` 的实例 Offline 拒绝。

### 4.2 请求

- 输入由按时间排序的请求组成；Offline 对时间戳乱序 fail-fast。
- 每个请求包含有序完整 block key 列表和原始 `input_token_len`。
- 共享预处理 `NormalizeRequest` 校验
  `block_keys.size() == floor(input_token_len / block_size_tokens)`；
  `input_token_len <= 0` 视为缺省并按 key 数推导；违约抛异常。
- 不足一个完整 block 的尾部 token 不进入 LRU，但保留在命中率分母中。
- 请求之间共享同一 LRU 状态；请求边界必须保留。

### 4.3 容量

- 核心不接收容量；容量只出现在投影边界。
- 投影支持有限 block 容量、容量 0 和无限容量（`ProjectInfinite`）。
- 字节/GB 容量在投影边界按每行记录的 `block_bytes` floor 换算（`ProjectBytes`）；GB 使用二进制换算。
- facts query 的 capacity_gb 列表保序，允许重复和 0，负数表示无限容量。

## 5. 命中语义

### 5.1 LRU 状态

- 驱逐策略固定为精确 LRU；block 第一次出现时为 cold miss。
- 非 cold 访问 reuse distance 为 `d` 时，容量 `C` 命中条件为 `d < C`。
- 每个访问都更新全局 LRU 状态，包括 prefix miss 后的后续 block。
- 状态提交按请求内倒序 touch（尾先、头后）：契约下链头永远比链上后块新，驱逐牺牲者恒为叶子，对齐生产 prefix cache（vLLM 逆序释放、SGLang radix cache 叶子 LRU）。命中评估基于请求前只读快照按正序计算。

### 5.2 Full-attention prefix hit 与 hit curve

- 主口径固定为 request prefix hit：首个 miss 前的 block 计命中，之后不计但仍提交。
- 每条请求的事实是 hit curve：`prefix_required[j] = max(r1..rj)` 的单调阶梯，任意容量的命中块数 = 满足 `prefix_required[j] <= C` 的 j 个数。
- 契约 + 倒序提交下门槛严格递增，且只在"插队"点跳变，因此以等差段 RLE 编码：
  `HitCurveSegment{start_required_blocks, run_length}`，段数 = 1 + 插队次数。
- 非契约输入编码时施加 `max(threshold, last_encoded + 1)` 单调防御。

### 5.3 无限容量

- 无限容量无 capacity miss，命中数 = curve 总 run length（`ProjectInfinite`）；cold miss 仍存在。

## 6. 输出

### 6.1 事实（核心输出）

`RequestFact{hit_curve}`：容量无关、可复算。空 curve 表示请求头即 cold。

### 6.2 facts CSV（Offline 账本）

固定文件名 `litehit_facts.csv`，表头：

```text
trace_id,instance_id,timestamp_ns,input_token_len,block_size_tokens,block_bytes,hit_curve
```

`hit_curve` 为带引号 JSON `[[start,run],...]`；`block_bytes` 每行自描述，支持事后以修正 charge 重投影。先写 `.tmp` 再原子 rename 发布。

### 6.3 投影输出

- facts query：JSONL，每请求一行（逐 slot hit_blocks/hit_rates）+ summary 行
  （requests / total_input_tokens / capacity_gb / total_hit_blocks / total_hit_tokens / hit_rates）。
- Online：逐 slot 命中数与 token 命中率
  `hit_rate = hit_blocks * block_size_tokens / input_token_len`；
  累计率由累计整数推导。`input_tokens` 为 0 时命中率为 `0.0`。

LiteHit 不输出逐 key 命中结果、reuse distance 明细、LRU 栈内容、驱逐次数、hit age、TTL 状态或普通 optimizer 的其他中间结果。

## 7. 功能要求

- `FR-1`：Offline 批量回放整份 trace，产出 facts CSV 账本。
- `FR-1a`：facts 落盘 fail-fast（乱序时间戳 / 未知 instance / 长度违约 / 零有效行任一即整体失败）且原子发布，不静默丢行。
- `FR-1b`：提供 facts 事后容量查询工具，投影复用 `HitCurveProjector`。
- `FR-2`：Online 支持按请求追加流式输入并逐请求投影。
- `FR-3`：任意容量（含 0、重复、无限）可事后投影，无需重放。
- `FR-4`：相同输入必须得到确定、可重复的事实与投影结果；Offline 并行度不改变输出字节。
- `FR-5`：Offline facts 投影与 Online 逐请求投影对相同请求序列结果一致（可对账）。
- `FR-6`：支持重置核心状态开始新一段独立分析。
- `FR-7`：计数器和位置索引使用 64 位整数，避免溢出。
- `FR-8`：核心、预处理、投影器可被 Offline 工具和 online optimizer 复用，不依赖完整 optimizer manager。
- `FR-9`：Online / Offline 共享同一预处理（长度校验推导 + 可选 prefix hash），hash 与 Python 生产端逐 bit 一致。

## 8. 性能要求

- 不允许为每个目标容量分别回放 trace 或维护独立 LRU 状态。
- 核心 `ProcessRequest` 为 `O(m log U)`（m 为请求块数，U 为历史 unique block 数）；投影为 `O(段数)`，与容量值无关。
- 事实体积由等差段 RLE 控制：段数 = 1 + 插队次数，与请求长度无关。
- 核心不做基于容量的剪枝（容量事后才知道），保留全部历史 unique key；Fenwick 位置空间通过 compaction 回收废弃位置，由活跃 key 数主导。
- Offline 流水线以有界批窗口（`pipeline_worker_count * 256`）并行预处理、按 instance lane 串行提交、按输入顺序写出，内存有界。
- facts query 内存为 O(容量 slot 数) 累计整数，逐行流式处理。

## 9. 非目标

- 不支持 linear attention / Mamba、checkpoint、混合 charge；混合 charge 的加权门槛属于后续独立任务。
- 不支持 RandomLRU、LeafAwareLRU、TTL、admission policy、prefetch 或多级缓存。
- 不支持 block 大小动态变化、range read 或一个 key 对应多个不同 charge。
- 不负责修改或重放请求时间，不进行性能仿真。
- Online 第一阶段不落盘事实（facts 为 Offline 专属产物）。
- 不复制完整 optimizer 的配置、可视化和中间结果体系。

## 10. 验收标准

- 契约输入（随机树状链 + `ApplyPrefixHash`）下，投影结果与朴素多容量 LRU oracle 在含 0/1/中间值/超工作集/无限的容量集合上完全一致。
- 非契约输入投影 ≤ oracle（悲观下界），无限容量仍精确。
- RLE 形态验证：整链重放单段、插队断段、相邻段不可合并。
- 覆盖 cold miss、立即重复访问、容量 0、字节 floor 换算边界、尾部 token 留在分母。
- Offline facts 投影与 Online 逐请求投影交叉对账一致；并行度 4 与串行输出逐字节相同。
- fail-fast 场景（乱序 / 未知 instance / 长度违约 / 零有效行）整体失败且不发布 facts。
- prefix hash golden vector 与 Python 生产端逐 bit 一致。
- compaction 后所有历史 key 仍可命中（不丢历史）。
- 空 trace 不崩溃、不除零；长 trace 计数无溢出。
- 核心代码位于 `kv_cache_manager/`，按仓库要求完成外源和内源测试验证。

## 11. API 决策记录

1. 主输出为容量无关事实 `RequestFact`；prefix hit 口径不变，分母为原始 `input_token_len`。
2. 核心 `ProcessRequest(block_keys)` 无容量参数；`HitCurveProjector` 是唯一投影入口。
3. hit curve 门槛以 block 为单位；字节换算仅发生在 `ProjectBytes` 边界，依据每行 `block_bytes`（= full location spec group 各 spec.size 之和）。
4. 等差段 RLE 的无损性依赖前缀 hash 契约 + 倒序提交 + 等 charge；非契约输入单调防御（悲观）。
5. facts query 的负容量表示无限；slot 保序、允许重复和 0。
6. Online full-attention `InstanceState` 直接持有 LiteHit；linear attention 继续走 legacy indexer（启用 prefix hash 时仅做 `ApplyPrefixHash`）。
7. full-attention + TTL 明确不支持，注册时返回参数错误。
8. TraceQuery 显式传 `input_token_len`；缺省时先从 `token_ids` 推导，再缺省则按 `block_keys.size() * block_size_tokens` 推导。
9. LRU 状态更新固定为请求内倒序提交，不提供开关。
10. `enable_prefix_hash` 是 `OptimizerInstanceGroup` 字段：online 经建组/更新组 RPC 配置，offline 在配置的 instance_groups 里设置，作用于 group 内全部实例；两端 hash 行为一致。

## 12. 相关文档

- 算法说明：[README.md](README.md)
- 算法可视化：`../docs/lite_hit_algorithm.html`
- 初版 Task/Plan：`../.agent/tasks/2026-07-09-litehit-lru-hit-analysis.md`、`../.agent/plans/2026-07-09-litehit-lru-hit-analysis.md`
- 事实化改造 Task/Plan：`../.agent/tasks/2026-07-21-litehit-full-facts-refactor.md`、`../.agent/plans/2026-07-21-litehit-full-facts-refactor.md`
