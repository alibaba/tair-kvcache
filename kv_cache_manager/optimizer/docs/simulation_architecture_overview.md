# Optimizer 仿真架构概览

本文用于描述 Optimizer 离线仿真的整体架构，重点面向架构图绘制，不展开具体配置字段。

## 整体定位

Optimizer 是一个离线 trace replay 仿真系统。它通过回放标准化后的请求 trace，模拟 KV cache 在不同缓存层级、容量和策略下的读写行为，并输出理论命中率、容量变化和生命周期分析结果。

整体流程可以抽象为：

```text
标准 Trace
  -> 时间顺序回放
  -> 缓存层级仿真
  -> 命中率与生命周期分析
```

Optimizer 不在线上请求链路中工作，它的核心目标是回答：

```text
在给定 trace、容量和策略下，缓存系统理论上能达到什么命中效果。
```

## 顶层分层

架构图建议按四层绘制：

```text
Trace Input
Replay Engine
Cache Simulation
Analysis Output
```

### Trace Input

输入是一份标准 optimizer trace。trace 描述请求发生的时间、所属推理实例、读写的 block key，以及读请求对应的输入 token 数。

进入仿真后，trace 会被加载、校验，并按时间顺序回放。时间顺序是仿真的基础约束，因为缓存状态依赖历史读写顺序。

### Replay Engine

Replay Engine 是仿真编排层，负责把 trace 请求转换成缓存系统上的读写动作。

当前有三类主要入口：

```text
optimizer_main
  模拟单个 optimizer / KVCM / L3 视角。

multi_infer_replay
  模拟多个推理实例各自的本地缓存，不接共享 L3。

hierarchical_replay_main
  同时模拟推理实例本地缓存和共享 L3 pool。
```

其中 `hierarchical_replay_main` 是完整链路仿真的主入口。

### Cache Simulation

缓存仿真层分为两部分：

```text
Inference Local Cache
Shared L3 Pool
```

`Inference Local Cache` 表示推理实例本地缓存。一个推理实例可以包含多层本地缓存，例如：

```text
L1 HBM
L2 DRAM
```

多个推理实例之间的本地缓存相互独立，各自维护缓存内容、访问时间和驱逐状态。

`Shared L3 Pool` 表示 KVCM / L3 的共享缓存池。多个推理实例可以共同访问同一个 L3 pool，用来模拟跨推理实例的缓存复用。

### Analysis Output

仿真完成后输出分析结果，主要包括：

```text
全局端到端命中率
每个推理实例的本地缓存命中率
共享 L3 pool 的命中率
各层容量变化
block 生命周期
```

## 三种仿真模式

### 单 Optimizer / L3 仿真

该模式只关注 KVCM / L3 自身行为。

适合场景：

```text
只分析 KVCM 自身日志
只评估一个 L3 层或普通多层 optimizer 配置
不需要模拟推理实例本地缓存
```

架构图可以画成：

```text
Trace -> OptimizerManager -> L3 Cache Simulation -> Analysis Output
```

### 多推理实例本地仿真

该模式只模拟多个推理实例本地缓存，不接共享 L3。

适合场景：

```text
只评估推理实例本地 HBM / DRAM 缓存效果
trace 已经能区分请求属于哪个推理实例
希望多个推理实例并行回放以提升分析速度
```

架构图可以画成：

```text
Trace
  -> Multi Infer Replay
  -> Infer Instance A Local Cache
  -> Infer Instance B Local Cache
  -> Infer Instance N Local Cache
  -> Aggregated Output
```

### 分层完整链路仿真

该模式同时模拟推理实例本地缓存和共享 L3 pool，是完整链路分析入口。

适合场景：

```text
评估 engine-local + L3 pool 的整体收益
比较不同 L1/L2/L3 流动策略
分析 L3 池化对多个推理实例的全局影响
```

架构图可以画成：

```text
Trace
  -> HierarchicalReplayManager
  -> Infer Cluster
       -> Infer Instance A: L1 + L2
       -> Infer Instance B: L1 + L2
       -> Infer Instance N: L1 + L2
  -> Shared L3 Pool
  -> Combined Analysis Output
```

## 完整链路中的核心组件

### HierarchicalReplayManager

`HierarchicalReplayManager` 是完整链路仿真的编排器。它负责：

```text
按时间顺序回放请求
决定请求进入哪个推理实例
编排本地缓存和 L3 pool 的读写路径
汇总 Local / Remote / Total 命中结果
```

### Infer Cluster

`Infer Cluster` 表示一组同构推理实例。每个推理实例都有独立的本地缓存层级。

典型结构：

```text
Infer Cluster
  -> Infer Instance A
       -> L1 HBM
       -> L2 DRAM
  -> Infer Instance B
       -> L1 HBM
       -> L2 DRAM
```

图中需要强调：

```text
推理实例之间本地缓存互相独立。
```

### Shared L3 Pool

`Shared L3 Pool` 表示 KVCM / L3 的共享池化缓存。多个推理实例可以访问同一个 L3 pool。

图中需要强调：

```text
L3 是多个推理实例共享的远端缓存层。
```

### KVCM / L3 组织结构

Optimizer 最初就是用来模拟 KVCM 的 L3 缓存管理行为。因此在 L3 这一侧，仿真结构沿用了 KVCM 原本的组织方式：

```text
OptimizerManager
  -> Instance Group
       -> Instance
       -> Storage / Tier
       -> Eviction Policy
       -> RadixTreeIndex
```

这些概念在 L3 侧的含义如下。

`OptimizerManager` 表示一个完整的 KVCM / L3 管理器。它负责管理 L3 里的 instance、缓存索引、驱逐策略和统计输出。

`Instance Group` 表示一组共享配额和存储层级配置的 L3 实例集合。容量配额、存储层、层间策略和驱逐配置都挂在这个层级上。

`Instance` 是 KVCM 内部的缓存隔离单元。KV cache 只能在同一个 instance 内复用，不同 instance 之间不会互相命中。对于 L3 pool 场景，多个推理实例可以映射到同一个 L3 instance，从而形成共享池化；如果映射到不同 L3 instance，则它们在 L3 侧也是隔离的。

`Storage / Tier` 表示 L3 内部的存储层。最简单的 L3 pool 可以只有一层，例如一个共享的 L3 存储池；如果要模拟 L3 内部还有多层介质，也可以继续组织成多个 tier。

`RadixTreeIndex` 是 L3 内部的 block key 索引结构。读请求在 L3 侧会通过它做前缀匹配，写请求会把 block key 插入到对应 instance 的索引中。

`Eviction Policy` 控制 L3 容量不足时如何驱逐，例如 LRU、RandomLRU、TTL 等。驱逐策略作用在 L3 instance 的缓存内容上。

因此，在完整链路图里，Shared L3 Pool 不只是一个抽象的大缓存，它内部可以展开成：

```text
Shared L3 Pool
  -> L3 OptimizerManager
       -> L3 Instance Group
            -> L3 Instance
                 -> RadixTreeIndex
                 -> Eviction Policy
            -> L3 Storage Tier
```

对于最常见的全局池化仿真，可以画成：

```text
Infer Instance A \
Infer Instance B  -> same L3 Instance -> same L3 Storage Pool
Infer Instance N /
```

这个结构表达的是：

```text
多个推理实例本地缓存独立，但它们在远端共享同一个 KVCM / L3 instance。
```

如果需要表达多池隔离，也可以画成：

```text
Infer Cluster A -> L3 Instance A
Infer Cluster B -> L3 Instance B
```

这表示不同集群在 L3 侧不会互相复用缓存。

### Tier Flow Policy

`Tier Flow Policy` 是缓存层级之间的数据流动策略。它控制数据如何在以下边上流动：

```text
L1 -> L2
L2 -> L3
```

架构图里可以把它画成连接缓存层级的策略框，而不是画成具体配置字段。

它表达的能力包括：

```text
写入是否穿透到下一层
上层驱逐后是否流向下一层
下层命中后是否提升回上层
上层命中是否刷新下层访问时间
```

## 读路径

完整链路中的读请求路径：

```text
Request
  -> Infer Local Cache
  -> Shared L3 Pool
  -> Optional Promote
  -> Hit Rate Record
```

语义如下：

```text
1. 请求先进入某个推理实例。
2. 先查询推理实例本地缓存。
3. 本地未命中的部分继续查询共享 L3 pool。
4. 如果 L3 命中，可以根据策略选择是否提升回本地缓存。
5. 最后记录本地命中、L3 命中和总命中。
```

命中统计含义：

```text
LocalHit
  推理实例本地 L1 / L2 命中。

RemoteHit
  共享 L3 pool 命中。

TotalHit
  LocalHit + RemoteHit。
```

## 写路径

完整链路中的写请求路径：

```text
Write Request
  -> Infer Local Cache
  -> Tier Flow Policy
  -> Shared L3 Pool
```

语义如下：

```text
1. 写请求先进入推理实例本地缓存。
2. 本地 L1 / L2 之间按策略流动。
3. 本地缓存到 L3 pool 之间也按策略流动。
4. L3 是否立即写入、延迟写入或只接收驱逐数据，由策略决定。
```

## 输出视角

完整链路会产生三类结果：

```text
Combined Output
  端到端全局结果，展示 LocalHit、RemoteHit 和 TotalHit。

Infer Output
  每个推理实例本地缓存的结果。

L3 Pool Output
  共享 L3 pool 的结果。
```

架构图中可以把输出画成：

```text
Analysis Output
  -> Global Hit Rate
  -> Per Infer Instance Stats
  -> L3 Pool Stats
  -> Lifecycle Analysis
```

## 分析能力

Analysis 层用于把一次回放结果转成可比较的指标，判断缓存层级、容量和策略是否有效。

主要分析视角如下。

```text
命中率分析
  端到端 TokenHitRate，以及 LocalHit / RemoteHit / TotalHit 的贡献拆分。

容量占用分析
  观察每个推理实例、L3 pool、每个 tier 的缓存占用随时间变化。

层间数据流动分析
  统计 L1 -> L2、L2 -> L3、L3 -> L1/L2 promote 的数据流动量，用于评估写入放大、远端写入压力和层间策略成本。

扩缩容影响分析
  比较推理实例数量、本地 HBM/DRAM 容量、L3 pool 容量变化后，对命中率、容量占用和层间流量的影响。

池化收益分析
  对比不接 L3、共享一个 L3 pool、多个 L3 pool 隔离等模式，评估 L3 池化带来的增益。

策略对比分析
  对比 write_through、cascading、selective write、promote、access propagation 和不同驱逐策略的效果。

生命周期分析
  分析 block 写入、访问、提升、驱逐和存活时间，用于解释命中率和驱逐行为。

时间序列分析
  观察冷启动、稳定阶段、业务波峰波谷下的 hit rate、容量和流量变化。
```

这些分析可以直接用于容量规划、策略选择、扩缩容评估和回归验证。

## 推荐架构图

可以按下面结构画一张主图：

```text
┌────────────────────┐
│   Standard Trace    │
└─────────┬──────────┘
          │
          v
┌────────────────────┐
│    Replay Engine    │
│ timestamp replay    │
└─────────┬──────────┘
          │
          v
┌─────────────────────────────────────────┐
│        Hierarchical Simulation           │
│                                         │
│  ┌──────────────┐   ┌──────────────┐   │
│  │ Infer Inst A │   │ Infer Inst B │   │
│  │ L1 + L2      │   │ L1 + L2      │   │
│  └──────┬───────┘   └──────┬───────┘   │
│         │                  │           │
│         └────────┬─────────┘           │
│                  v                     │
│          ┌──────────────┐              │
│          │ Shared L3    │              │
│          │ Pool         │              │
│          └──────────────┘              │
└─────────────────┬──────────────────────┘
                  │
                  v
┌────────────────────┐
│  Analysis Output    │
│ hit rate/lifecycle  │
└────────────────────┘
```

旁边可以补一个策略框：

```text
Tier Flow Policy
  controls data movement between L1, L2 and L3
```

## 一句话总结

Optimizer 通过离线回放标准 trace，模拟推理实例本地缓存和共享 L3 pool 在不同层级流动策略下的读写行为，最终输出端到端命中率、各层缓存状态和生命周期分析结果。
