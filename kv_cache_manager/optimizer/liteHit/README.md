# LiteHit：一次回放计算多个 LRU 容量的 Prefix 命中率

LiteHit 是面向 full-attention KVCache block 的轻量、精确 LRU 命中率分析器。它既可以逐请求接收 online 流量，也可以流式回放 offline trace；两种入口共享同一套状态和命中语义。

LiteHit 要解决的问题是：

```text
给定一组带请求边界的 block trace，以及多个缓存容量，
如何只回放一次 trace，就精确得到每个容量下：

1. 每条请求的 prefix 命中率；
2. 整段 trace 的累计 prefix 命中率。
```

第一阶段只支持以下模型：

- full attention；
- 每个完整 block 的 KVCache charge 相同；
- 精确 LRU；
- 容量集合在分析开始时固定；
- 支持有限容量、零容量和无限容量；
- 不处理 linear attention、不同大小 block、TTL、admission、prefetch 和多级缓存策略。

---

## 1. 输入模型

### 1.1 初始化参数

一个 LiteHit 实例对应一个独立的 LRU 状态域，例如一个 `instance_id`。初始化时固定：

```text
block_size_tokens       一个完整 block 包含多少 token
capacity_blocks[]       每个目标容量可以容纳多少个完整 block
```

`capacity_blocks` 的语义为：

```text
0    零容量，永远不能命中
正数 可容纳的完整 block 数
∞    无限容量，不发生 capacity miss，但仍有 cold miss
```

实际接口可以使用专门的无限容量类型；当前 C++ 实现以 `-1` 表示无限容量。

### 1.2 每条请求

每条请求向 LiteHit 提供：

```text
full_block_keys[]       按请求从前到后排列的完整 block key
input_token_len         原始请求的 input token 数
```

输入层必须保证：

```text
full_block_keys.size()
    == floor(input_token_len / block_size_tokens)
```

例如：

```text
block_size_tokens = 16
input_token_len = 50

完整 block 数 = floor(50 / 16) = 3
尾部 token 数 = 50 - 3 * 16 = 2
```

因此 LiteHit 回放 3 个完整 block；剩余 2 个 token 不进入 LRU，但仍属于命中率分母。

### 1.3 block key 契约：前缀链式 hash

full-attention 下 block j 的 KVCache 依赖请求前 j 个 block 的全部 token，因此输入层必须按前缀链式 hash 生成 key：

```text
key_j = hash(请求前 j 个完整 block 的全部 token)
```

即 key 相等当且仅当整个 token 前缀完全相同。如果只 hash 本 block 自身内容，两个前缀不同的请求会在中间 block 上错误“复用”，跨请求命中语义直接失效。

该契约由输入层（Connector / trace 生成方）保证，LiteHit 核心只做 key 相等判断，不做校验。契约带来两个推论：

```text
1. 同一请求内的 key 必然互不相同：
   每个 key 编码严格递增的前缀。
   （实现中按首次出现去重的逻辑只是防御行为。）

2. 请求内首个 cold block 之后，后续 key 的前缀都包含分歧点，
   必然也是 cold —— “cold 截断 prefix”与该契约天然自洽。
   而 capacity miss（key 仍 resident、只是深度超过某容量）之后的
   block 可能仍然 resident，这正是 prefix max 门槛存在的意义。
```

合法的 trace 形态是“共享前缀 + 分叉”，例如 `[A, B, C]` 与 `[A, B, D]` 共享前两个 block；而 `[B, A, C]` 这类重排序列在该契约下不可能出现。

---

## 2. `hit_count` 的准确含义

### 2.1 `hit_count` 不是独立 block 命中数

full-attention 请求采用 prefix hit。对于容量 `C`，一条请求的：

```text
hit_count[C]
```

表示从请求第一个 block 开始连续命中的完整 block 数，也就是首个 miss 的位置。

例如：

```text
请求 block：       [A, B, C, D]
独立命中判断：      hit, hit, miss, hit
```

本条请求的结果是：

```text
hit_count = 2
```

`D` 即使独立判断为 hit，也不能越过 `C` 重新贡献 prefix hit。

> 注：该示例仅定义 prefix 语义本身。在本实现中（前缀 hash 契约 + 第 10 节的倒序提交），链上后块的最小命中容量严格大于前块，首个 miss 之后的 block 独立判断也必然 miss——prefix 口径与逐块独立口径事实上重合。

### 2.2 首个 miss 不停止 LRU 状态更新

首个 miss 只截断本条请求的 `hit_count`。所有完整 block 仍然视为被访问，并全部进入或刷新全局 LRU 状态（提交按请求倒序 touch，见第 10 节）：

```text
A → 更新 LRU
B → 更新 LRU
C → miss，截断本条 prefix hit，同时更新 LRU
D → 不贡献本条 hit_count，但仍更新 LRU
```

否则下一条请求看到的缓存状态会错误。

### 2.3 每条请求和累计 `hit_count`

对第 `q` 条请求、容量 `C`，记：

```text
H[q, C] = 该请求的 prefix hit block 数
```

整段 trace 的累计命中 block 数是：

```text
cumulative_hit_count[C] = Σq H[q, C]
```

它不是把所有请求拼成一个大请求后求一次 prefix，也不是逐 block 独立命中数。

---

## 3. 从 `hit_count` 换算 token 命中率

LiteHit 的 LRU 状态按完整 block 更新，但最终命中率按 token 计算。

对一条请求：

```text
hit_tokens[C] = hit_count[C] * block_size_tokens

trace_hit_rate[C]
    = hit_tokens[C] / input_token_len
```

例如：

```text
block_size_tokens = 16
input_token_len = 50
hit_count = 3
```

则：

```text
hit_tokens = 3 * 16 = 48
trace_hit_rate = 48 / 50 = 96%
```

即使所有 3 个完整 block 都命中，尾部 2 个不完整 token 仍在分母中，因此结果不是 100%。

### 3.1 累计命中率

累计命中率必须先累加整数分子和分母：

```text
cumulative_hit_tokens[C]
    = Σq (H[q, C] * block_size_tokens)

cumulative_input_tokens
    = Σq input_token_len[q]

cumulative_hit_rate[C]
    = cumulative_hit_tokens[C] / cumulative_input_tokens
```

不能对每条请求的 `trace_hit_rate` 做算术平均，因为请求长度可能不同。

---

## 4. 字节容量如何转换成 block 容量

这里有两个容易混淆但用途完全不同的大小：

| 名称 | 单位 | 用途 |
|---|---:|---|
| `block_size_tokens` | token/block | 将 prefix hit block 数换算成命中 token 数 |
| `block_charge_bytes` | byte/block | 将缓存字节容量换算成可容纳的 block 数 |

它们不能互相替代。

### 4.1 固定 charge 模型

第一阶段所有 full-attention block 的 charge 相同。对容量 `capacity_bytes`：

```text
capacity_blocks
    = floor(capacity_bytes / block_charge_bytes)
```

必须向下取整，因为不足一个完整 block 的剩余字节不能再容纳一个 block。

例如：

```text
capacity_bytes = 10,000
block_charge_bytes = 3,000

capacity_blocks = floor(10,000 / 3,000) = 3
剩余 1,000 bytes 不足以容纳第 4 个 block
```

如果：

```text
capacity_bytes < block_charge_bytes
```

则：

```text
capacity_blocks = 0
```

### 4.2 从 GB 配置换算

当前 Online Optimizer 的 `capacity_gb` 使用二进制换算：

```text
capacity_bytes = capacity_gb * 1024 * 1024 * 1024
```

然后：

```text
capacity_blocks
    = floor(
        capacity_gb * 1024^3
        / block_charge_bytes
      )
```

例如：

```text
capacity_gb = 1
block_charge_bytes = 256 KiB = 262,144 bytes

capacity_blocks
    = floor(1,073,741,824 / 262,144)
    = 4096
```

因此这一容量点在 LiteHit 核心中表示为：

```text
C = 4096 blocks
```

### 4.3 `block_charge_bytes` 从哪里来

Online Optimizer 中，full-attention block 的字节 charge 来自实例配置中的 full location spec group：

```text
block_charge_bytes
    = full location spec group 内各 spec.size 的总和
```

容量换算属于 Online/Offline 输入适配层；LiteHit 核心只接收换算后的 `capacity_blocks`。

### 4.4 无限容量

无限容量不参与字节除法，直接作为特殊容量传入：

```text
capacity = ∞
```

无限容量不会发生 capacity miss，但一个从未出现过的 block 仍然是 cold miss。由于采用 prefix hit，一条请求遇到首个 cold block 后，当前及后续 block 都不再贡献该请求的 `hit_count`。

### 4.5 为什么第一阶段不使用 weighted Fenwick

第一阶段 block charge 固定，因此最简单且精确的处理是先把字节容量换算成 block 容量，再运行普通 reuse-distance 算法：

```text
capacity_bytes
    ↓ 除以固定 block_charge_bytes 并向下取整
capacity_blocks
    ↓
普通 Fenwick 中每个活跃 block 记为 1
```

不需要在 Fenwick 中保存字节大小。

如果未来支持不同 block charge，命中条件不能简单写成：

```text
newer_blocks_total_bytes < capacity_bytes
```

因为还必须为当前对象自身预留空间。更准确的条件是：

```text
newer_distinct_blocks_bytes + current_block_bytes
    <= capacity_bytes
```

对象大小变化、range read 和 admission policy 还会引入更多语义。因此可变 charge 属于后续独立设计，不属于第一阶段 LiteHit。

---

## 5. 为什么一个 LRU 状态可以回答所有容量

LRU 具有栈包含性质。假设当前全局最近访问顺序是：

```text
MRU → [X, A, Y, B] → LRU
```

那么：

```text
容量 1：[X]
容量 2：[X, A]
容量 3：[X, A, Y]
容量 4：[X, A, Y, B]
```

所有容量都是同一条全局 LRU 顺序的不同长度前缀。

如果 block `B` 位于第 4 位，那么它：

```text
在容量 1、2、3 下 miss
在容量 4 及以上命中
```

因此一次访问只需要知道 block 在全局 LRU 顺序中的深度，就能同时判断所有容量。

---

## 6. Reuse distance 与最小命中容量

对于一次重复访问：

```text
reuse distance
    = 该 block 上次访问之后、这次访问之前，
      出现过的不同 block 数
```

如果 reuse distance 为 `d`，那么 block 在全局 LRU 中位于第 `d + 1` 位。

容量为 `C` 时：

```text
hit  ⇔ d < C
     ⇔ d + 1 <= C
```

定义：

```text
required_capacity = d + 1
```

则一次访问在所有满足以下条件的容量下命中：

```text
C >= required_capacity
```

第一次出现的 block 是 cold miss，没有有限的 `required_capacity`；无限容量也不能命中 cold access。

---

## 7. Fenwick 如何计算 reuse distance

LiteHit 为每个 block 只保留最新访问位置：

```text
last[key] = key 的最新访问位置
```

Fenwick 的逻辑数组满足：

```text
marker[pos] = 1  pos 是某个 block 的最新位置
marker[pos] = 0  pos 是历史旧位置
```

扫描到位置 `i`，当前 key 上次出现在 `prev`：

```text
d = Fenwick.sum(i - 1) - Fenwick.sum(prev)
```

区间 `(prev, i)` 中仍为 1 的位置数量，就是比当前 key 更新的不同 block 数。

随后更新最新位置：

```text
Fenwick.add(prev, -1)
Fenwick.add(i, +1)
last[key] = i
```

如果 key 是 cold，只添加当前位置。

Fenwick 不是一套缓存，也不是一条额外 LRU 链；它只是全局 LRU 顺序的 order-statistics 表示。

---

## 8. 从独立 block 命中得到 prefix hit

设一条请求各 block 的最小命中容量是：

```text
required = [r1, r2, ..., rm]
```

要让前 `j` 个 block 全部连续命中，容量必须满足前面所有 block，因此：

```text
prefix_required[j]
    = max(r1, r2, ..., rj)
```

第 `j` 个 block 能贡献 prefix hit 的条件是：

```text
C >= prefix_required[j]
```

遇到 cold block 后，该 block 以及请求内后续 block 不再贡献任何容量的 prefix hit，但仍继续更新全局 LRU。

---

## 9. 一次得到所有配置容量

内部将有限容量排序并去重，并把无限容量放在所有有限容量之后。输入顺序和重复容量可以通过映射在输出时恢复。

假设容量是：

```text
[1, 2, 4, ∞]
```

一条请求的 prefix 容量门槛是：

```text
[2, 4, 4]
```

每个门槛 `R` 表示当前 prefix block 对所有 `C >= R` 的容量贡献一次命中。使用容量维度差分：

```text
gain[first capacity >= 2] += 1
gain[first capacity >= 4] += 2
```

得到：

```text
容量： [1, 2, 4, ∞]
gain： [0, 1, 2, 0]
```

沿容量从小到大做前缀和：

```text
hit_count：[0, 1, 3, 3]
```

这样不需要为每次访问逐一更新所有容量，更不需要维护多套 LRU。

---

## 10. 请求前快照与倒序批量提交

对外语义分两半：**命中评估**把所有完整 block 视为按请求顺序访问；**状态提交**把它们按请求倒序 touch（尾先、头后）。内部使用等价的两阶段算法：

```text
阶段一：基于请求到达前的只读 LRU 快照计算 prefix hit
阶段二：按请求倒序 touch，批量提交最终 LRU 状态（链头最新）
```

### 10.1 为什么快照可以计算 prefix hit

对任意容量，在首个 miss 之前只有 hit。hit 会改变顺序，但不会改变缓存成员集合。因此首个 miss 等于请求中第一个不属于请求前缓存快照的 block。

可以先查询请求中每个不同 key 在快照 LRU 中的深度，再按请求顺序计算 prefix maximum。

### 10.2 为什么状态提交按倒序

正序 touch 会让链头最老、最先被驱逐——对 prefix 语义这是**价值倒挂**：没有链头，链上其余 resident block 一个 prefix hit 都贡献不了，却还占着容量。生产 prefix cache 全部选择尾先驱逐：vLLM 按逆序把 block 放回 free 队列，SGLang radix cache 只驱逐 LRU 叶子。

倒序 touch 配合 1.3 的前缀 hash 契约，给出一个不变式：

```text
任何 key 的每次访问都伴随其所有前缀 key（同请求），
且前缀 key 在倒序提交中排在它之后
    ⇒ 父 key 永远比它的任何 resident 后代更新
    ⇒ 全局 LRU 的驱逐牺牲者永远是叶子
```

即：**倒序提交的 LRU 等价于"驱逐最久未用的叶子"**，且全局顺序仍由访问序列唯一决定、与容量无关——栈包含性质保持，多容量单态框架不受影响。

另一个推论：链上后块的最小命中容量严格大于前块（每深一块至少多算它的父 key），请求内门槛单调递增，prefix 口径与逐块独立口径重合（见 2.1 注）。

### 10.3 为什么状态仍与倒序逐块 touch 相同

请求结束后：

- 请求中出现过的不同 key 移动到 LRU 最前面；
- 它们之间按请求内位置排序，越靠近请求头部越靠前（倒序 touch 时链头最后被摸）；
- 未在请求中出现的 key 保持原有相对顺序。

例如：

```text
旧 LRU：[X, A, B, C]
请求：  [A, B, D]

倒序逐块 touch：D → [D, X, A, B, C]
               B → [B, D, X, A, C]
               A → [A, B, D, X, C]
```

批量提交得到相同结果：

```text
新 LRU：[A, B, D, X, C]
```

实现按"每个 key 的首次出现位置、从请求尾部向头部"提交，对含重复 key 的输入也保持与倒序逐块 touch 等价；但在 1.3 的前缀 hash 契约下，请求内不会出现重复 key，该去重只是防御行为。

当前实现采用"请求前快照评估 + 倒序批量提交"。每条请求的 prefix hit 和请求结束后的 LRU 状态都必须与"正序评估 + 倒序逐块 touch"的朴素 LRU 完全一致。

---

## 11. 端到端示例：字节容量、hit count 和 token 命中率

假设：

```text
block_size_tokens = 4 tokens/block
block_charge_bytes = 1024 bytes/block

目标字节容量：2048 bytes、3072 bytes、无限容量
```

换算后：

| 字节容量 | LiteHit block 容量 |
|---:|---:|
| 2048 bytes | 2 blocks |
| 3072 bytes | 3 blocks |
| 无限 | 无限 |

依次处理五条请求。所有请求满足 1.3 的前缀 hash 契约：`A`、`B`、`C`、`D`、`E` 都是前缀链上的 key，`[A, B, C]` 与 `[A, B, D]` 共享前两个 block 后分叉，`[A, E]` 在第一个 block 后分叉。

### 11.1 Request 1

```text
keys = [A, B, C]
input_token_len = 13
```

3 个 block 都是 cold：

```text
hit_count：[0, 0, 0]
hit_rate： [0%, 0%, 0%]
```

倒序提交（touch 顺序 C、B、A，链头 A 最新）后的全局 LRU：

```text
[A, B, C]
```

### 11.2 Request 2：共享前缀后分叉

```text
keys = [A, B, D]
input_token_len = 12
请求前 LRU = [A, B, C]
```

基于请求前快照：

| 请求位置 | Block | 快照 LRU 深度 | 自身最小命中容量 | Prefix 最小容量 |
|---:|---|---:|---:|---:|
| 1 | A | 1 | 1 | 1 |
| 2 | B | 2 | 2 | 2 |
| 3 | D | cold | — | 截断 |

注意倒序提交下链式访问的固有形态：链头永远比链上后块新，门槛沿请求单调递增，且共享前缀被兄弟分支持续保鲜。

| 容量 | `hit_count` | `hit_tokens` | 本条命中率 |
|---:|---:|---:|---:|
| 2 blocks / 2048 bytes | 2 | 8 | `8 / 12 = 66.67%` |
| 3 blocks / 3072 bytes | 2 | 8 | `8 / 12 = 66.67%` |
| 无限 | 2 | 8 | `8 / 12 = 66.67%` |

`D` 是 cold，截断所有容量的 prefix，但仍照常提交。请求结束后：

```text
[A, B, D, C]
```

### 11.3 Request 3：重放原链，尾部因分叉变深

```text
keys = [A, B, C]
input_token_len = 13
请求前 LRU = [A, B, D, C]
```

| 请求位置 | Block | 快照 LRU 深度 | 自身最小命中容量 | Prefix 最小容量 |
|---:|---|---:|---:|---:|
| 1 | A | 1 | 1 | 1 |
| 2 | B | 2 | 2 | 2 |
| 3 | C | 4 | 4 | 4 |

`C` 仍然 resident，但 Request 2 的分叉把 `D` 插到了它前面，深度变为 4——容量 2 和 3 在这里发生 capacity miss，无限容量继续命中：

| 容量 | `hit_count` | `hit_tokens` | 本条命中率 |
|---:|---:|---:|---:|
| 2 blocks | 2 | 8 | `8 / 13 = 61.54%` |
| 3 blocks | 2 | 8 | `8 / 13 = 61.54%` |
| 无限 | 3 | 12 | `12 / 13 = 92.31%` |

请求结束后：

```text
[A, B, C, D]
```

### 11.4 Request 4：短链分叉

```text
keys = [A, E]
input_token_len = 8
请求前 LRU = [A, B, C, D]
```

`A` 深度 1（所有容量命中），`E` cold 截断：

| 容量 | `hit_count` | `hit_tokens` | 本条命中率 |
|---:|---:|---:|---:|
| 2 blocks | 1 | 4 | `4 / 8 = 50%` |
| 3 blocks | 1 | 4 | `4 / 8 = 50%` |
| 无限 | 1 | 4 | `4 / 8 = 50%` |

请求结束后：

```text
[A, E, B, C, D]
```

### 11.5 Request 5：立即重放短链

```text
keys = [A, E]
input_token_len = 8
请求前 LRU = [A, E, B, C, D]
```

`A` 深度 1、`E` 深度 2，prefix 门槛 ≤ 2，所有容量全命中：

| 容量 | `hit_count` | `hit_tokens` | 本条命中率 |
|---:|---:|---:|---:|
| 2 blocks | 2 | 8 | `8 / 8 = 100%` |
| 3 blocks | 2 | 8 | `8 / 8 = 100%` |
| 无限 | 2 | 8 | `8 / 8 = 100%` |

请求结束后 LRU 不变：`[A, E, B, C, D]`。

### 11.6 累计结果

累计 input token：

```text
13 + 12 + 13 + 8 + 8 = 54
```

| 容量 | 累计 `hit_count` | 累计 `hit_tokens` | 累计命中率 |
|---:|---:|---:|---:|
| 2 blocks | 0+2+2+1+2 = 7 | 28 | `28 / 54 = 51.85%` |
| 3 blocks | 0+2+2+1+2 = 7 | 28 | `28 / 54 = 51.85%` |
| 无限 | 0+2+3+1+2 = 8 | 32 | `32 / 54 = 59.26%` |

对比正序提交（链头最老）：同一条 trace 下容量 2 的累计命中会从 7 块跌到 2 块——这正是 10.2 所说的价值倒挂，也是选择倒序提交的直接理由。

---

## 12. Offline 与 Online 的共同核心

Offline 和 Online 都应当逐请求调用同一个状态机：

```text
Offline TraceReader ──┐
                     ├──> LiteHit::ProcessRequest
Online TraceQuery ────┘
```

两端的区别只在适配层：

- Offline 负责读取、排序和规范化 trace，并通过 `Analyze` 的逐请求 callback 流式写出每条结果；
- Online 负责 instance 生命周期、并发锁、容量换算、RPC 和指标上报；
- LiteHit 只负责 LRU 状态、每条命中率和累计命中率。

LiteHit 不保留原始 trace、每条历史结果、逐访问 reuse distance，也不维护每个容量的一套 cache。

---

## 13. 复杂度与状态量

设：

```text
N = 总 block access 数
K = 目标容量数
U = 当前需要追踪的不同 block 数
Q = 请求数
```

Fenwick 的查询和更新为：

```text
O(log U) 或 O(log N)
```

固定容量集合使用二分查找定位第一个满足门槛的容量：

```text
O(log K)
```

输出每条请求的 `K` 个命中率本身需要：

```text
O(Q * K)
```

核心持久状态只包括：

```text
Fenwick
last_position
容量映射和累计命中计数
累计 input token 数
```

不随容量数量维护 `K` 套 LRU 链。Online 长流需要周期性压缩已经失效的历史位置，使 Fenwick 空间由当前有效状态而不是历史总访问数主导。

如果只配置有限容量，掉出最大有限容量的 key 对所有查询容量都已经等价于 miss，当前实现会将其从 Fenwick 和 `last_position` 中删除，因此 `U` 不超过最大有限容量。只有配置了无限容量时，才必须保留所有仍需区分 cold/reuse 的历史 unique key。

---

## 14. 正确性验证

LiteHit 应当使用朴素的多容量 LRU 作为 oracle 对拍。对每个容量分别维护真实 LRU（快照评估 prefix hit + 倒序逐块 touch 更新状态），并验证：

1. 每条请求的 prefix `hit_count` 完全一致；
2. 每条请求结束后的 LRU 状态对后续请求产生相同结果；
3. Offline 批量输入与 Online 逐请求输入结果一致；
4. token 命中率使用 `hit_count * block_size_tokens / input_token_len`；
5. 字节容量始终通过 `floor(capacity_bytes / block_charge_bytes)` 换算；
6. 覆盖容量 0、重复容量、无限容量、cold miss、请求内重复 key 和不完整尾部 token。

---

## 15. 核心结论

LiteHit 的多容量计算建立在三层转换上：

```text
多个容量的 LRU cache
    ↓ LRU 栈包含性质
一条全局最近访问顺序
    ↓ Fenwick / order statistics
每次访问的最小命中容量
    ↓ 请求内 prefix maximum + 容量差分
所有目标容量的 prefix hit_count
    ↓ block_size_tokens / input_token_len
每条和累计 token 命中率
```

其中最容易混淆的两点是：

```text
1. hit_count 是每条请求的连续 prefix hit block 数，
   首个 miss 后的 block 不计入本条 hit_count，但仍更新 LRU。

2. block_size_tokens 用于计算命中 token，
   block_charge_bytes 用于把字节容量换算成 block 容量。
```
