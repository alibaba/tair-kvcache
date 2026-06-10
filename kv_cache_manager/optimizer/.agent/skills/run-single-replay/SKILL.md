# 单次 Optimizer 回放 Skill

当用户要求跑一个 optimizer config、计算无限容量命中率、画命中率图或检查单次 replay 输出时，使用这个 skill。

## 需要确认的输入

- config 路径，或足够生成 config 的字段。
- `block_size` 和 `bytes_per_token`。
- 容量，以及是否要用 `-1` 表示无限容量。
- `query_type` 和 trace replay mode。
- 输出目录。使用干净的实验目录。

## 步骤

1. 按需阅读 [../../handbook/external_usage.md](../../handbook/external_usage.md) 和 [../../../docs/strategy_config.md](../../../docs/strategy_config.md)。
2. 校验 trace：
   - JSONL 可解析。
   - 按 `timestamp_ns` 排序。
   - `get/request` 行有正数 `input_len`。
   - `keys.size() <= input_len / block_size`。
3. 校验 config：
   - `trace_file_path` 指向已校验 trace。
   - `output_result_path` 与历史实验隔离。
   - trace 中每个 `instance_id` 都存在于 config。
4. 运行：

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- -c /path/to/config.json
```

只有用户要求画图时才加 `--draw-chart`。只有需要 lifecycle 分析时才加 `--export-lifecycle`。

## 校验

- 读取 `*_hit_rates.csv` 最后一行。
- 相关时报告 `AccHitRate`、`AccInputTokens`、`AccHitTokens`、`AccReadBlocks` 和 `AccHitBlocks`。
- 如果要求画图，确认 `timeseries/` 下的 PNG 路径。

## 回复内容

告诉用户：

- config 路径
- 输出路径
- 最终 token hit rate
- 未完成的校验项，如有
