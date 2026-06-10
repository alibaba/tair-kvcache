# Hierarchical Replay Skill

当实验包含 engine-local tier、storage pool、P2P read、scheduler、active window 或 cache drop 时，使用这个 skill。

## 需要确认的输入

- trace 路径，以及是否保留 `instance_id` 路由。
- `infer_scheduling_strategy`：`preserve_trace`、`round_robin` 或 `prefix_hit`。
- infer clusters 和 `infer_ids`。
- engine tier 和容量。
- storage pool 容量和 pool ID。
- `block_size`、`bytes_per_token`、驱逐策略。
- `engine_read_query_type` 和 storage-pool `query_type`。
- storage-pool write flow：`write_through`、`cascading` 或 `write_through_selective`。
- P2P flow tier 和 touch 行为，如有。
- 如果要分析扩缩容，需要 active window 或 cache drop event 文件。

## 步骤

1. 阅读 [../../../docs/hierarchical_replay.md](../../../docs/hierarchical_replay.md)。
2. 按 [../../handbook/trace_schema_and_conversion.md](../../handbook/trace_schema_and_conversion.md) 校验 trace。
3. 校验 config：
   - 每个 `infer_id` 唯一。
   - 每个 infer cluster 引用已存在的 storage pool。
   - 每个 P2P tier 存在于 cluster tiers。
   - `active_windows` 不与 `infer_active_windows_from_trace` 冲突。
   - cache drop event 文件每行包含 `timestamp_ns` 和 `instance_id`。
4. 运行：

```bash
bazel run //kv_cache_manager/optimizer:hierarchical_replay_main -- /path/to/hierarchical_config.json
```

## 校验

- 检查 `hierarchical_hit_rates.csv` 最后一行 `AccHitRate`。
- 检查 local/peer/remote token 列，解释命中组成。
- 检查 `hierarchical_read_io.csv` 的 peer/pool transfer tokens。
- 检查 `hierarchical_pool_write_io.csv` 的 pool 写入量。

## 回复内容

报告：

- combined token hit rate
- local、peer、remote 贡献
- storage-pool 输出路径
- 是否启用了 active-window/cache-drop 语义
