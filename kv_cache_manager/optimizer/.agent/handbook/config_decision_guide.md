# 配置决策指南

这份文档列出必须由人或权威配置判断的参数。不要从旧实验里静默猜这些值。

## 容量和 KV 大小

| 参数 | 为什么重要 | 从哪里确认 |
|---|---|---|
| `block_size` | 定义 trace key 和 token 换算 | 模型服务配置或实验要求 |
| `bytes_per_token` | 把 GB 容量换算成 block 容量 | 模型 KV cache 公式或实测配置 |
| tier 容量 | 决定驱逐行为和 Pareto 横轴 | 部署配置 |
| 容量范围 | per-pod 与 global 会改变仿真问题 | 拓扑或实验目标 |
| 非 KV 内存预留 | 改变真实可用 cache | 部署负责人或配置 |

配置中的容量单位是 GB。Optimizer 内部按下面公式换算 block 容量：

```text
bytes_per_block = block_size * bytes_per_token
cache_blocks = capacity_GB * 1024^3 / bytes_per_block
```

## 回放模式

| 目标 | 选择 |
|---|---|
| 一个全局理论池 | KVCM/L3-only，单 instance，无限容量 |
| 每个 pod 独立理论 cache | `multi_infer_replay` |
| 本地多层 cache 加共享 L3/pool | `hierarchical_replay_main` |
| P2P remote read | `hierarchical_replay_main` + `p2p_read_flows` |
| 扩缩容影响 | active window；需要清空 cache 时再加 cache drop event |

## 读语义

| 参数 | 选项 | 需要确认什么 |
|---|---|---|
| `query_type` | `prefix_match`、`batch_get` | 读是否在第一个 miss 停止，还是逐 block 独立读取 |
| `engine_read_query_type` | `prefix_match`、`batch_get` | hierarchical replay 中 engine-local 的本地读语义 |
| `block_mask` | 空、bool array、offset | trace 是否已包含进入 KVCM 前的 engine 本地命中 |

## 写入和层间流动

| 参数 | 选项 | 含义 |
|---|---|---|
| `write_mode` | `write_through` | 初始写入时写所有配置 tier |
| `write_mode` | `cascading` | 先写最高层，被驱逐的 block 逐层下沉 |
| `write_mode` | `write_through_selective` | write touch 达到阈值后才写低层 |
| `access_propagation_enabled` | bool | 高层读命中时是否 touch 低层已有副本 |
| `write_propagation_enabled` | bool | 高层 write touch 时是否 touch 低层已有副本 |
| `selective_write_threshold` | 正整数 | selective 下写低层需要的 write touch 次数 |
| promote/fill | 读路径触发 | 低层或 pool 命中后是否回填高层 |

## 在线拓扑

当某个 infer instance 只在某个时间段可被调度或 P2P 读取时，使用 `active_windows`。当某个 instance 的 cache 需要在特定时间被清空时，使用 cache drop event。

Cache drop event JSONL：

```json
{"timestamp_ns": 150, "instance_id": "infer_a"}
```

同一时间戳下，已到期的 pending write 会先 flush，然后执行 drop，再执行普通 trace 事件。

## 结果语义

标准对外命中率是 token hit rate：

```text
HitRate = HitTokens / InputTokens
```

block 计数只用于审计行为、读放大和容量规模。除非用户明确要求，不要把 block hit rate 当作主结论。
