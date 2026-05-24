# L1/L2 + L3 联动回放

入口：

```bash
bazel run //kv_cache_manager/optimizer:hierarchical_replay_main -- /path/to/hierarchical_replay_config.json
```

回放语义：

- `engine_config` 是推理引擎侧模拟，用同一套 optimizer 组件模拟每个 engine instance 独立的 L1/L2。
- `pool_config` 是 L3 池化模拟，用同一套 optimizer 组件模拟全局池化实例。
- `engine_scheduling_strategy=preserve_trace` 时，标准 trace 的 `instance_id` 表示 engine instance。
- `engine_scheduling_strategy=round_robin` 时，回放前按 get/write pair 轮询分配到 `engine_to_pool` 中的 engine instance；write 跟随前一个 get。
- `engine_to_pool` 显式定义每个 engine instance 对应哪个 L3 pool instance；缺失、重复、未知 instance、block_size 不一致都会初始化失败。
- 多个 engine instance 可以映射到同一个 L3 pool instance；不同 pool instance 之间互相隔离。
- `get` 先查 engine L1/L2，未命中的后缀再查 L3；engine 已命中的前缀只用于穿过 L3 前缀树，不记录 L3 命中，也不刷新 L3 访问时间。
- `write` 同时写入 engine instance 和对应的 L3 pool instance。

输出：

- `output_result_path/hierarchical_hit_rates.csv`：联动后的整体命中率，`HitTokens = (EngineHitBlocks + PoolHitBlocks) * block_size`。
- `engine_config.output_result_path`：engine 侧独立统计。
- `pool_config.output_result_path`：L3 pool 侧独立统计。

配置骨架：

```json
{
  "trace_file_path": "/path/to/standard_trace.jsonl",
  "output_result_path": "/tmp/hierarchical/combined",
  "engine_scheduling_strategy": "preserve_trace",
  "engine_config": { "...": "普通 optimizer config，定义 engine instances 和 L1/L2 storages" },
  "pool_config": { "...": "普通 optimizer config，定义 L3 pool instances 和 L3 storages" },
  "engine_to_pool": [
    {
      "engine_instance_id": "engine_a",
      "pool_instance_id": "model_l3"
    }
  ]
}
```
