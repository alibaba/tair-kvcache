# LiteHit：一次回放产出容量无关事实，任意 LRU 容量事后投影

> 中文 | [English](README.md)

LiteHit 是面向 full-attention KVCache block 的轻量、精确 LRU 命中率分析器。核心只回放一次 trace，为每条请求产出**容量无关的事实**（`RequestFact`，一条等差段 RLE 编码的 hit curve）；任何容量的命中数都由无状态投影器 `HitCurveProjector` 事后从事实推出，核心从不接收容量列表，也不累计任何逐容量结果。

LiteHit 要解决的问题是：

```text
给定一组带请求边界的 block trace，
如何只回放一次 trace，就能对"任意"LRU 容量精确回答：

1. 每条请求的 prefix 命中块数；
2. 整段 trace 的累计 prefix 命中率。

容量不再需要在分析开始前给定。
```

第一阶段只支持以下模型：

- full attention；
- 每个完整 block 的 KVCache charge 相同（等 charge，见 6.4）；
- 精确 LRU，请求内倒序提交；
- 不处理 linear attention / Mamba、不同大小 block、admission、prefetch 和多级缓存策略；TTL 支持为"每组一个固定 TTL"叠加在 LRU 之上（见 §7 TTL 回放），不支持任意 TTL 的查询期扫描。

## 0. 架构总览

```text
                 ┌────────────────────────────────────────────┐
   原始请求      │ 共享预处理 request_preprocess               │
 (keys,len) ───> │  NormalizeRequest：长度校验/推导 +          │
                 │  可选 ApplyPrefixHash（前缀链式 hash）      │
                 └───────────────┬────────────────────────────┘
                                 │ NormalizedRequest
                 ┌───────────────▼────────────────────────────┐
                 │ LiteHit 核心（容量无关）                     │
                 │  ProcessRequest(block_keys) → RequestFact   │
                 │  状态：Fenwick + last_positions             │
                 └───────┬───────────────────────┬────────────┘
                         │ RequestFact           │ RequestFact
            Online 路径  │                       │  Offline 路径
                 ┌───────▼────────┐      ┌───────▼─────────────┐
                 │ HitCurveProjector│    │ facts CSV            │
                 │ 逐 slot 投影 +   │    │ litehit_facts.csv    │
                 │ 累计整数         │    │ (原子发布)           │
                 └────────────────┘      └───────┬─────────────┘
                                                 │ 任意时刻、任意容量
                                         ┌───────▼─────────────┐
                                         │ facts query 工具     │
                                         │ HitCurveProjector    │
                                         └─────────────────────┘
```

三个不可违反的分层约束：

1. **核心容量无关**：`LiteHit::ProcessRequest` 只接收 block key，返回 `RequestFact`；不存在容量参数。
2. **投影唯一入口**：所有"容量 → 命中块数"的换算必须经过 `HitCurveProjector`，Online 与 facts query 共用同一实现，禁止任何组件自行实现边界逻辑。
3. **字节换算只发生在投影边界**：核心与事实全部以 block 为单位；`ProjectBytes` 在投影时用 `block_bytes` 做一次 floor 除法。

---

## 1. 输入模型与共享预处理

### 1.1 每条请求

每条请求向预处理层提供：

```text
block_keys[]            按请求从前到后排列的完整 block key（或原始逐块 hash）
input_token_len         原始请求的 input token 数
```

`NormalizeRequest(block_keys, input_token_len, block_size_tokens, enable_prefix_hash, trace_block_size_tokens = 0)`：

- `trace_block_size_tokens` 是 trace 原生 block 粒度（0 = 与 `block_size_tokens` 相同，不重分块）；长度校验发生在 trace 粒度：
  `block_keys.size() == floor(input_token_len / trace_block_size_tokens)`；
- `input_token_len > 0` 时为权威分母；`input_token_len <= 0` 视为缺省，按 `block_keys.size() * trace_block_size_tokens` 推导；
- 违反约束（含 `block_size_tokens` 不是 `trace_block_size_tokens` 的整数倍——只允许变粗）抛 `std::invalid_argument`（Offline 对此 fail-fast，见第 7 节）。

不足一个完整 block 的尾部 token 不进入 LRU，但保留在命中率分母中。

### 1.1a 重分块（re-blocking）：零重 hash 的粒度变粗

分析粒度粗于 trace 粒度时（`block_size_tokens = k * trace_block_size_tokens`，k > 1），无需重算任何 hash：前缀链式 key 的第 `j*k` 个恰好编码了前 j 个粗 block 的全部 token，因此**先做前缀链（若输入是逐块 hash）、再每 k 个取第 k 个**即得到粗粒度下合法的前缀链式 key，凑不满 k 个细块的尾部丢弃（其 token 留在分母中）。只允许变粗——变细需要 block 内部的 token 信息，trace 中不存在。

### 1.2 block key 契约：前缀链式 hash

full-attention 下 block j 的 KVCache 依赖请求前 j 个 block 的全部 token，因此 key 必须编码整个前缀：

```text
key_j = hash(请求前 j 个完整 block 的全部 token)
```

key 相等当且仅当整个 token 前缀完全相同。合法的 trace 形态是"共享前缀 + 分叉"，例如 `[A, B, C]` 与 `[A, B, D]` 共享前两个 block；`[B, A, C]` 这类重排序列在契约下不可能出现。

当输入是逐块独立 hash 时，在 **Instance Group** 上置 `enable_prefix_hash = true`（key 形态跟模型部署走，group 正是这个粒度；online 建组 RPC 与 offline 配置共用同一字段），预处理用滚动 hash 把它转成前缀链式 key：

```text
PrefixHashNext(prev, raw)：Jenkins 64 位变体，显式 uint64 运算（逻辑右移），
与 Python 生产端 prefix_hash.py::hash_int64_func 逐 bit 一致。
注意：有意不复用 HashUtil::HashIntFunc（有符号右移，负 hash 时结果分歧）。
```

契约的两个推论（第 5 节会用到）：

```text
1. 同一请求内 key 必然互不相同（每个 key 编码严格递增的前缀）。
2. 请求内首个 cold block 之后，后续 key 的前缀都包含分歧点，必然也 cold。
```

---

## 2. 命中语义：prefix hit + 倒序提交

### 2.1 prefix hit

full-attention 请求采用 prefix hit：对某个容量，一条请求的命中块数是从请求第一个 block 开始连续命中的完整 block 数，即首个 miss 的位置。首个 miss 只截断本条请求的命中，不停止 LRU 状态更新——所有完整 block 仍然全部提交，否则后续请求看到的缓存状态会错误。

提交不区分 hit / miss 的物理依据是：真实 prefix cache 里 miss 的 block 会被重算并**写回**缓存，写入本身就是一次 touch——请求结束后无论命中与否，block 都在 MRU 端。于是"LRU 更新"与"命中判定"天然解耦（Mattson 栈算法性质）：命中判定依赖容量，推迟到投影层；LRU 更新不依赖容量，核心无条件把所有 block 提交到位置末尾。这也意味着若 trace 带 output（decode 生成的新 block），同一机制可按**读写分离**扩展：output block 不参与阶段一的命中评估（新生成的块谈不上命中），照常参与阶段二提交进入 LRU，供后续请求命中。

### 2.2 状态提交按请求倒序（尾先、头后）

正序 touch 会让链头最老、最先被驱逐——对 prefix 语义这是**价值倒挂**：没有链头，链上其余 resident block 一个 prefix hit 都贡献不了，却还占着容量。生产 prefix cache 全部选择尾先驱逐：vLLM 按逆序把 block 放回 free 队列，SGLang radix cache 只驱逐 LRU 叶子。

倒序提交配合 1.2 的前缀 hash 契约，给出不变式：

```text
父 key 永远比它的任何 resident 后代更新
    ⇒ 全局 LRU 的驱逐牺牲者永远是叶子
    ⇒ 倒序提交的 LRU 等价于"驱逐最久未用的叶子"
```

且全局顺序仍由访问序列唯一决定、与容量无关——栈包含性质保持（第 3 节）。

实现分两阶段：**阶段一**基于请求到达前的只读 LRU 快照计算 hit curve（首个 miss 前只有 hit，hit 改变顺序但不改变成员集合，快照评估精确）；**阶段二**按"每个 key 的首次出现位置、从请求尾部向头部"批量提交，对含重复 key 的输入也与倒序逐块 touch 等价（契约下请求内无重复 key，该去重是防御行为）。

---

## 3. 为什么一个 LRU 状态可以回答所有容量

LRU 具有栈包含性质。假设当前全局最近访问顺序是：

```text
MRU → [X, A, Y, B] → LRU
```

那么容量 1/2/3/4 的缓存内容分别是这条顺序的前 1/2/3/4 个元素。block `B` 位于第 4 位，则它在容量 1、2、3 下 miss，容量 ≥ 4 命中。因此一次访问只需要知道 block 在全局 LRU 顺序中的深度，就能同时回答所有容量——这正是"容量无关事实"可行的根源。

---

## 4. Reuse distance 与 Fenwick

对一次重复访问：

```text
reuse distance d = 上次访问之后、这次访问之前出现过的不同 block 数
required_capacity = d + 1        （容量 C 命中 ⇔ C >= d + 1）
```

第一次出现的 block 是 cold miss，没有有限的 required_capacity；无限容量也不能命中 cold access。

LiteHit 为每个 block 只保留最新访问位置 `last[key]`，Fenwick 的逻辑数组在"某 block 的最新位置"处为 1、历史旧位置处为 0：

```text
d = Fenwick.sum(i - 1) - Fenwick.sum(prev)     // (prev, i) 内仍为 1 的位置数
随后 Fenwick.add(prev, -1)、Fenwick.add(i, +1)、last[key] = i
```

Fenwick 不是一套缓存，只是全局 LRU 顺序的 order-statistics 表示。当历史废弃位置超过活跃 key 数的一倍加上固定 slack 时，实现会重建位置空间（compaction），使 Fenwick 空间由当前活跃 key 数而不是历史访问总数主导——注意这只回收**位置**，不删除任何 key（见第 9 节）。

---

## 5. RequestFact：等差段 RLE 编码的 hit curve

### 5.1 从逐块门槛到 hit curve

设一条请求各 block 基于快照的最小命中容量是 `required = [r1, ..., rm]`（cold 截断于首个无穷）。要让前 j 个 block 连续命中，容量必须满足前面所有 block：

```text
prefix_required[j] = max(r1, ..., rj)
```

序列 `prefix_required[1..h]`（h 为 cold 截断前的长度）就完整决定了该请求在**一切容量**下的命中块数：

```text
hit_blocks(C) = |{ j : prefix_required[j] <= C }|
```

这条单调阶梯函数就是本请求的 **hit curve**——它就是这条请求的全部事实。

### 5.2 契约下门槛严格递增 ⇒ 等差段 RLE 无损

倒序提交下链上后块的最小命中容量**严格大于**前块（每深一块，快照区间内至少多出它的父 key），因此契约输入的 `prefix_required` 严格递增。更强的结构性质是：链上相邻 block 在全局 LRU 中占据**连续**位置，门槛只在"插队"点（兄弟分支交错进来的位置）跳变。于是把连续 `+1` 的门槛压成一个等差段：

```text
HitCurveSegment { start_required_blocks, run_length }
段内第 j 个 block（0-based）在容量 >= start + j 时成为 prefix hit。
段数 = 1 + 插队次数，与请求长度无关。
```

例如门槛 `[1, 2, 3]`（无人插队的整链重放）编码为一段 `{1, 3}`；门槛 `[1, 2, 4]`（第三块前被插了一队）编码为 `{1, 2}, {4, 1}`。相邻段之间必有至少 1 的门槛空隙，不可再合并。

### 5.3 非契约输入的单调防御

对不满足契约的输入（请求内重复 key 等），门槛可能不严格递增。编码时施加单调防御：

```text
encoded_threshold = max(prefix_required[j], last_encoded + 1)
```

契约输入下该防御恒为 no-op；非契约输入下它只会把门槛**抬高**（悲观、绝不乐观），投影结果是真实命中的下界。

### 5.4 HitCurveProjector

```cpp
ProjectBlocks(fact, capacity_blocks)   // 沿段线性扫描：
                                       // hits += min(run_length, C - start + 1)，直到 start > C
ProjectBytes(fact, capacity_bytes, block_bytes)
                                       // = ProjectBlocks(fact, floor(bytes / block_bytes))
ProjectInfinite(fact)                  // = Σ run_length（无 capacity miss，仅 cold miss）
```

空 curve 表示请求头即 cold，任何容量命中为 0。

---

## 6. 单位与换算

### 6.1 两个大小，用途不同

| 名称 | 单位 | 用途 |
|---|---:|---|
| `block_size_tokens` | token/block | 将命中块数换算成命中 token 数 |
| `block_bytes` | byte/block | 将字节容量换算成 block 容量（仅投影边界） |

### 6.2 token 命中率

```text
hit_tokens = hit_blocks * block_size_tokens
trace_hit_rate = hit_tokens / input_token_len
```

尾部不完整 token 在分母中，因此全命中也未必 100%。累计命中率必须先累加整数分子分母（`Σ hit_tokens / Σ input_token_len`），不能对每条请求命中率做算术平均。

### 6.3 字节容量换算

```text
capacity_blocks = floor(capacity_bytes / block_bytes)
capacity_gb 使用二进制换算：capacity_bytes = capacity_gb * 1024^3
```

`block_bytes` 来自实例注册的 full location spec group 各 spec.size 之和（`size_full_only`）。facts CSV 每行记录 `block_bytes`，事实自描述：即使日后修正 charge 估计，也能对历史事实重投影。

### 6.4 等 charge 不变量（block 单位 RLE 的前提）

hit curve 以 block 为单位、`ProjectBytes` 做单一 floor 除法，**当且仅当**所有参与块 charge 完全相等才精确——full-only 实例满足（每块 charge 恒为 `size_full_only`，是精确值不是平均值）。linear/Mamba 混合实例每块 charge 不等，必须改用 charge 加权门槛，属于后续独立任务；当前 Offline 拒绝 `linear_step != 0` 的实例。

---

## 7. Offline facts 流水线

Offline runner（`lite_hit_main` + `OptimizerLiteHitConfig`）逐批处理标准 trace：

```text
批窗口 = pipeline_worker_count * 256 条
  ├─ 并行预处理（按下标条带分配 worker）：解析 + NormalizeRequest + prefix hash
  ├─ 按 instance 分 lane，lane 内严格按输入顺序串行 ProcessRequest
  └─ 按输入顺序串行写出 facts 行
```

**trace 粒度与重分块**：配置字段 `block_size`（默认 256）声明 trace 的原生 block 粒度；每个 instance 的 `block_size` 是该 lane 的分析粒度，必须是它的整数倍（只允许变粗，违约在 lane 初始化时整体失败），按 1.1a 的采样方式重分块。

**写事件**：`write` trace 事件会被识别并忽略。`get` 提交时视为全部 block 已写回（§2 的物理依据：写回本身就是一次 touch），因此拆分的 `get`/`write` trace 中的 `write` 行对 facts 无影响；delayed write 建模只属于 replay 路径。

**fanout 模式**：`fanout_all_instances = true` 时每条请求广播到全部 lane（各 lane 独立 LRU 状态、独立 facts 行），配合多个不同 `block_size` 的 instance 即可一次回放对同一份 trace 扫多个分析粒度；与 `override_instance_id` 互斥。facts query 的 summary 按 instance 分组输出（每 instance 一行 + 总计一行），fanout 结果直接可读。

**TTL 回放**：instance group 配置 `ttl_seconds != 0` 时，该组 lane 的 `LiteHit` 核心叠加固定 TTL（与 online `TtlCacheIndexerWrapper` 语义一致：块在距上次访问严格小于 TTL 内存活，过期块对任意容量都是 miss 并像冷块一样截断前缀；每次访问（命中或未命中）都刷新 last_access；时间取 trace 时间戳，回放确定性）。年龄沿 LRU 栈单调，过期块不会抬高存活块的复用距离，因此一次回放对"固定 TTL × 任意容量"的联合口径仍然精确，facts 仍是普通 hit curve 行。TTL 是回放期参数，直接取组里的 `ttl_seconds`；要扫多个 TTL 就配多个 group 各带不同 `ttl_seconds`（可配合 fanout 一次回放完成）。

**fail-fast**：时间戳乱序、未知 instance、长度校验失败、全文件零有效行，任一发生即整体失败并给出原因——facts 是全有或全无的对账账本，不允许静默丢行。

**原子发布**：先写 `litehit_facts.csv.tmp`，全部成功后 `rename` 为 `litehit_facts.csv`；读者永远不会看到半成品。

### 7.1 facts CSV 格式

```text
trace_id,instance_id,timestamp_ns,input_token_len,block_size_tokens,block_bytes,hit_curve
```

`hit_curve` 是带引号的 JSON 数组 `[[start_required_blocks, run_length], ...]`；字符串字段按 CSV 规则引用转义。每行独立可解析、自描述、可重投影。

### 7.2 facts query 工具

`lite_hit_facts_query_main`（`RunLiteHitFactsQuery`）对已发布的 facts 做事后容量查询：

```text
输入：facts CSV + capacity_gb 列表（保序、允许重复和 0，负数 = 无限容量）
输出：JSONL，每请求一行（hit_blocks/hit_rates 按 slot）
     + 每个 instance_id 一行 summary（instance_id 字典序）+ 一行总计 summary
     （requests / total_input_tokens / total_hit_blocks / total_hit_tokens / hit_rates）
```

内存只有 O(instance 数 × 容量 slot 数) 个累计整数；任一畸形行使整个查询失败。

---

## 8. Online 集成

Online Optimizer 的 full-attention `InstanceState` 直接持有 `LiteHit`（组配置 `ttl_seconds != 0` 时以墙钟时间叠加固定 TTL，口径与 linear 路径的 `TtlCacheIndexerWrapper` 一致）。每次 TraceQuery：

```text
NormalizeRequest → ProcessRequest → 得到 RequestFact
  ├─ 对每个配置容量 slot：ProjectBlocks(fact, lite_hit_capacity_blocks[i])
  ├─ 理论上界：ProjectInfinite(fact)
  └─ 更新累计整数：total_queries / total_input_tokens / 各 slot total_hits
```

Online 不持久化 facts（facts 落盘当前是 Offline 专属）；`ListInstances` 的命中率从累计整数推导（`total_hits * block_size_tokens / total_input_tokens`）。linear attention 继续走 legacy indexer 路径（启用 prefix hash 时仅做 `ApplyPrefixHash`）。

---

## 9. 复杂度与状态量

设 N = 总 block access 数，U = 历史不同 block 数，Q = 请求数，S = 单请求段数（= 1 + 插队次数）。

```text
ProcessRequest：O(m log U)（m 为请求块数）
ProjectBlocks： O(S)，与容量值无关
核心持久状态： Fenwick + last_positions，O(U)
```

**没有基于容量的剪枝**：容量事后才知道，任何 key 都可能被将来某个大容量查询用到，因此核心保留全部历史 unique key（这是容量无关性的固有代价）。第 4 节的 compaction 只回收废弃位置，不删除 key。`memory_usage_bytes()` / `current_unique_blocks()` 提供观测。

---

## 10. 端到端示例

```text
block_size_tokens = 4，block_bytes = 1024
五条请求（满足前缀 hash 契约，[A,B,C] 与 [A,B,D] 共享前两块，[A,E] 一块后分叉）
```

| # | keys | len | 快照门槛 prefix_required | hit_curve (RLE) |
|---|---|---:|---|---|
| 1 | [A,B,C] | 13 | 全 cold | `[]` |
| 2 | [A,B,D] | 12 | [1, 2]，D cold 截断 | `[[1,2]]` |
| 3 | [A,B,C] | 13 | [1, 2, 4]（D 插队使 C 深度 4） | `[[1,2],[4,1]]` |
| 4 | [A,E] | 8 | [1]，E cold 截断 | `[[1,1]]` |
| 5 | [A,E] | 8 | [1, 2] | `[[1,2]]` |

各请求结束后的 LRU 依次为 `[A,B,C]`、`[A,B,D,C]`、`[A,B,C,D]`、`[A,E,B,C,D]`、`[A,E,B,C,D]`。

事后投影三个容量（2048 B → 2 块，3072 B → 3 块，无限）：

| 容量 | 各请求 hit_blocks | 累计块 | 累计 token | 累计命中率 |
|---:|---|---:|---:|---:|
| 2 块 | 0,2,2,1,2 | 7 | 28 | `28 / 54 = 51.85%` |
| 3 块 | 0,2,2,1,2 | 7 | 28 | `28 / 54 = 51.85%` |
| ∞ | 0,2,3,1,2 | 8 | 32 | `32 / 54 = 59.26%` |

请求 3 的 curve `[[1,2],[4,1]]` 直接读出：容量 2、3 命中 2 块（第二段 start=4 超出），容量 ≥ 4 与无限命中 3 块。对比正序提交（链头最老）：同一 trace 下容量 2 的累计命中会从 7 块跌到 2 块——这正是 2.2 所说的价值倒挂。

---

## 11. 正确性验证

单元测试（`LiteHitTest` / `LiteHitOfflineRunnerTest`）覆盖：

1. **oracle 对拍**：契约输入（随机树状链 + `ApplyPrefixHash`）下，`ProjectBlocks` 与朴素多容量 LRU（快照评估 + 倒序逐块 touch）在容量 {0,1,2,4,9,∞} 上**完全一致**；
2. 非契约输入投影 ≤ oracle（单调防御悲观、绝不乐观），无限容量仍精确；
3. RLE 形态：整链重放单段、插队断段、相邻段不可合并；
4. 投影边界：容量 0、段边界、字节 floor 换算；
5. Offline facts 与 Online 逐请求投影交叉对账一致；并行度 4 与串行输出逐字节相同；
6. fail-fast：乱序时间戳、未知 instance、长度违约、零有效行、分析粒度非 trace 粒度整数倍；
7. prefix hash golden vector 与 Python 生产端逐 bit 一致；
8. compaction 后所有 key 仍可命中（不丢历史）；
9. 重分块：粗粒度采样 key / 尾部丢弃 / 只允许变粗；fanout 一次回放对多 block_size 各产出独立 facts，query summary 按 instance 分组。

---

## 12. 核心结论

```text
多个（任意）容量的 LRU cache
    ↓ LRU 栈包含性质
一条全局最近访问顺序
    ↓ Fenwick / order statistics
每次访问的最小命中容量
    ↓ 请求内 prefix max（契约下严格递增）
等差段 RLE hit curve = 容量无关事实
    ↓ HitCurveProjector（唯一投影入口，字节换算仅在此边界）
任意容量的 prefix hit_blocks
    ↓ block_size_tokens / input_token_len
每条和累计 token 命中率
```

最容易混淆的三点：

```text
1. hit curve 是"事实"，容量是"查询"——核心与容量彻底解耦，
   代价是不能做基于容量的剪枝。
2. block_size_tokens 换 token，block_bytes 换容量，不能互相替代。
3. 等差段 RLE 的无损性依赖前缀 hash 契约 + 倒序提交 + 等 charge；
   混合 charge（linear/Mamba）必须另行设计。
```
