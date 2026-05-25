# L1/L2 + L3 联动回放

入口：

```bash
bazel run //kv_cache_manager/optimizer:hierarchical_replay_main -- /path/to/hierarchical_replay_config.json
```

回放语义：

- `engine_config` 是推理引擎侧模拟，用同一套 optimizer 组件模拟每个 engine instance 独立的 L1/L2。
- `pool_config` 是 L3 池化模拟，用同一套 optimizer 组件模拟全局池化实例。
- `engine_config` 中每个 instance group 只能放一个 engine instance，保证 HBM/DRAM 容量不会跨 engine 共享；多个 engine instance 用多个同构 group 表达。
- `pool_config` 可以在一个 instance group 下放多个 L3 instance，用 group 级 storage 表达全局池化容量。
- `engine_scheduling_strategy=preserve_trace` 时，标准 trace 的 `instance_id` 表示 engine instance。
- `engine_scheduling_strategy=round_robin` 时，回放前按 get/write pair 轮询分配到 `engine_to_pool` 中的 engine instance；write 跟随前一个 get。
- `engine_scheduling_strategy=prefix_hit` 时，每个 get 选择当前 L1/L2 前缀命中最长的 engine instance；冷启动或并列时按确定性轮询分配，write 跟随前一个 get。
- `l2_l3_strategy` 定义 engine L2 到 pool L3 的跨层流动策略：
  - `write_mode=write_through`：trace write 同时写 engine 和 L3，保持默认行为。
  - `write_mode=cascading`：trace write 只写 engine；engine 侧被 L1/L2 完全驱逐的前缀写入 L3。
  - `write_mode=write_through_selective`：trace write 只写 engine；engine 命中次数达到 `selective_write_threshold` 后再写入 L3。
  - `access_propagation_enabled=true` 时，engine 命中会刷新 L3 中已有副本的访问时间，但不额外写 L3 读统计。
  - `promote_enabled=true` 时，L3 命中会回填 engine，后续同 engine 访问可直接命中 L1/L2。
- `engine_to_pool` 显式定义每个 engine instance 对应哪个 L3 pool instance；缺失、重复、未知 instance、block_size 不一致都会初始化失败。
- 多个 engine instance 可以映射到同一个 L3 pool instance；不同 pool instance 之间互相隔离。
- `get` 先查 engine L1/L2，未命中的后缀再查 L3；engine 已命中的前缀不记录 L3 命中，只有开启 `access_propagation_enabled` 时才刷新 L3 访问时间。
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
  "l2_l3_strategy": {
    "write_mode": "write_through",
    "access_propagation_enabled": false,
    "promote_enabled": false,
    "selective_write_threshold": 2
  },
  "engine_config": { "...": "普通 optimizer config；每个 engine instance 使用单独 instance group 定义 L1/L2 storages" },
  "pool_config": { "...": "普通 optimizer config，定义 L3 pool instances 和 L3 storages" },
  "engine_to_pool": [
    {
      "engine_instance_id": "engine_a",
      "pool_instance_id": "model_l3"
    }
  ]
}
```
