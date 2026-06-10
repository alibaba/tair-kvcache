# 多推理实例回放 Skill

当每个 pod/infer instance 拥有独立本地 cache，且用户需要 per-instance replay 和 service 级聚合时，使用这个 skill。

## 需要确认的输入

- per-instance trace 文件列表或 trace 目录。
- `block_size`、`bytes_per_token` 和 tier 容量。
- 多层配置的 tier flow 策略。
- 是否限制回放时间范围。
- 时间窗口聚合的窗口大小。
- 最大并发 worker 数。

## 步骤

1. 确保每个输入 JSONL 文件只包含一个 `instance_id`。
2. 按 [../../handbook/trace_schema_and_conversion.md](../../handbook/trace_schema_and_conversion.md) 校验每个 trace。
3. 多层回放需要创建或校验完整的 `tier_flows` JSON array，必须覆盖所有相邻 tier。
4. 运行：

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:multi_infer_replay -- \
  --trace-dir /path/to/instance_traces \
  --trace-glob "*.jsonl" \
  --output-dir /path/to/output \
  --tiers hbm:50,dram:128 \
  --block-size 2048 \
  --bytes-per-token 57344 \
  --eviction-policy lru \
  --tier-flow-config /path/to/tier_flows.json \
  --max-workers 8
```

5. 如果用户需要时间窗口命中率，加 `--window-seconds` 或 `--window-ns`。
6. 只有在 replay 输出已存在且用户明确要求只聚合时，才使用 `--aggregate-only`。

## 校验

- 检查 `multi_infer_replay_summary.csv` 是否有失败 instance。
- 全局结果看 `aggregate/global_aggregate.csv`。
- 窗口结果看 `aggregate/global_window_hit_rates.csv`。
- token hit rate 计算方式是 `sum(HitTokens) / sum(InputTokens)`。

## 回复内容

报告：

- 回放的 trace/pod 数量
- 失败 pod，如有
- 全局 token hit rate
- 聚合 CSV 路径
