# 外部使用 Optimizer

这份文档用于把 Optimizer 当工具使用，重点是选对入口并保持正确的命中率口径。

## 选择入口

| 场景 | 入口 | 说明 |
|---|---|---|
| 一个 config、一个逻辑池 | `optimizer_run` 或 `optimizer_main` | 无限容量用 `quota_capacity=-1` |
| 每个 pod 独立 cache | `multi_infer_replay` | 每个 pod/infer instance 一个 JSONL，按 token 聚合 |
| engine 多层 cache + storage pool | `hierarchical_replay_main` | storage pool、P2P、active window、cache drop 都必须用它 |
| 容量 Pareto | `tradeoff` | 非 hierarchical 的容量 sweep；达到 99% 理论命中后可提前停止 |
| lifecycle 分析 | `optimizer_run --export-lifecycle` 后接 `analyze_lifecycle` | 大 trace 会明显增加内存 |

## 标准单次回放

1. 转换或校验 trace，确保它是标准 optimizer JSONL。
2. 创建 config，明确 `block_size`、`bytes_per_token`、容量、驱逐策略、trace replay mode 和输出目录。
3. 运行：

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- -c /path/to/config.json
```

需要图时加 `--draw-chart`。只有需要 lifecycle 时才加 `--export-lifecycle`。

关键输出：

- `<output_result_path>/*_hit_rates.csv`
- 开启画图后输出 `<output_result_path>/timeseries/*.png`

最终累计 token 命中率取 CSV 最后一行的 `AccHitRate`。

最小非 hierarchical config 形态：

```json
{
  "trace_file_path": "/path/to/optimizer.jsonl",
  "output_result_path": "/path/to/output",
  "eviction_params": {
    "eviction_mode": 3,
    "eviction_batch_size_per_instance": 100
  },
  "trace_replay": {
    "mode": "read_write",
    "write_delay_ns": 1
  },
  "instance_groups": [
    {
      "group_name": "service_or_instance",
      "quota_capacity": -1,
      "used_percentage": 1.0,
      "ttl_config": {
        "default_block_ttl_seconds": 0,
        "refresh_on_read": true
      },
      "storages": [
        {
          "unique_name": "hbm",
          "storage_type": "pace",
          "band_width_mbps": 0,
          "capacity": -1
        }
      ],
      "instances": [
        {
          "instance_id": "service_or_instance",
          "block_size": 2048,
          "bytes_per_token": 57344,
          "eviction_policy_type": "lru",
          "eviction_policy_params": {
            "sample_rate": 1.0,
            "shard_count": 1,
            "sample_times": 32,
            "eviction_amplification_factor": 1.0
          }
        }
      ]
    }
  ]
}
```

多层配置需要在 `storages[]` 中按高到低列出所有 tier，并为每一对相邻 tier 配一条 `tier_flows[]`。Hierarchical replay 使用 [../../docs/hierarchical_replay.md](../../docs/hierarchical_replay.md) 中的配置骨架，不以普通 `instance_groups[]` 作为主拓扑。

## 每个 pod 独立回放

当 trace 已经带路由信息，并且每个 pod 拥有独立 cache 时，使用这个入口。

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:multi_infer_replay -- \
  --trace-dir /path/to/pod_jsonl \
  --trace-glob "*.jsonl" \
  --output-dir /path/to/output \
  --tiers hbm:411.92,dram:1070.4 \
  --block-size 2048 \
  --bytes-per-token 57344 \
  --eviction-policy lru \
  --tier-flow-config /path/to/tier_flows.json \
  --max-workers 8
```

输出：

- `aggregate/instance_aggregate.csv`
- `aggregate/global_aggregate.csv`
- 设置窗口参数时输出 `aggregate/global_window_hit_rates.csv`

聚合必须使用 token 求和：

```text
global hit rate = sum(HitTokens) / sum(InputTokens)
```

## Hierarchical Replay

当仿真包含 engine-local tier、storage pool、P2P read、scheduler、active window 或 cache drop 时，使用这个入口。

```bash
bazel run //kv_cache_manager/optimizer:hierarchical_replay_main -- /path/to/hierarchical_config.json
```

改配置前先读 [../../docs/hierarchical_replay.md](../../docs/hierarchical_replay.md)。combined 输出包括：

- `hierarchical_hit_rates.csv`
- `hierarchical_read_io.csv`
- `hierarchical_pool_write_io.csv`
- `infer/`
- storage-pool 输出目录

## Pareto

非 hierarchical 的容量规划使用 Pareto。

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
  -c /path/to/config.json \
  --num-points 30 \
  --max-workers 8
```

理论命中率来自无限容量 warmup。95% 和 99% 目标分别表示理论 token hit rate 的 95% 和 99%。

## 必须确认的参数

- `block_size`
- `bytes_per_token`
- HBM/DRAM/L3 容量，以及容量是 per instance 还是 global
- 驱逐策略和 LRU shard/sample 设置
- `query_type`
- write mode、selective threshold、promote、touch 行为
- trace 路由策略
- 扩缩容场景的 active window 或 cache drop
