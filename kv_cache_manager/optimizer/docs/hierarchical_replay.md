# L1/L2 + L3 联动回放

`hierarchical_replay_main` 用于模拟完整的 engine-local + L3 pool 流程。只要需要同时评估推理实例本地 L1/L2 和 KVCM/L3 池化，就使用这个入口；trace 是否已经指定推理实例不影响入口选择。

入口：

```bash
bazel run //kv_cache_manager/optimizer:hierarchical_replay_main -- /path/to/hierarchical_replay_config.json
```

回放语义：

- `infer_clusters` 定义推理实例集合。映射到同一个 L3 pool 的同构推理实例只写一份 model、TTL、tier 列表和层间 flow。
- `infer_clusters[].model` 定义该推理实例集合的 block size、bytes per token 和驱逐策略。
- `infer_clusters[].ttl_config` 定义该推理实例集合的 TTL 行为。
- `infer_clusters[].tiers[]` 定义推理实例本地层；多层时用 `infer_clusters[].tier_flows` 写满相邻层级的流动策略。
- `l2_l3_strategy` 定义推理实例本地最后一层到 L3 pool 的流动策略。这条边跨 engine manager 和 pool manager，不属于单个 optimizer instance group 的 `tier_flows`。
- `pool_config` 直接复用普通 optimizer config，用来描述 KVCM/L3 pool；L3 单层不写 `tier_flows`，L3 多层时在对应 instance group 上写满相邻 `tier_flows`。
- `infer_scheduling_strategy=preserve_trace` 时，标准 trace 的 `instance_id` 表示推理实例。
- `infer_scheduling_strategy=round_robin` 时，回放前按 get/write pair 轮询分配到推理实例；write 跟随前一个 get。
- `infer_scheduling_strategy=prefix_hit` 时，每个 get 选择当前 L1/L2 前缀命中最长的推理实例；冷启动或并列时按确定性轮询分配，write 跟随前一个 get。
- `tier_flows` / `l2_l3_strategy` 的 `write_mode`：
  - `write_mode=write_through`：trace write 同时写 engine 和 L3，保持默认行为。
  - `write_mode=cascading`：trace write 只写 engine；engine 侧被 L1/L2 完全驱逐的前缀写入 L3。
  - `write_mode=write_through_selective`：trace write 只写 engine；engine 命中次数达到 `selective_write_threshold` 后再写入 L3。
  - `access_propagation_enabled=true` 时，engine 命中会刷新 L3 中已有副本的访问时间，但不额外写 L3 读统计。
  - `write_propagation_enabled=true` 时，engine 写触达已有前缀会刷新 L3 中已有副本的访问时间；驱逐下沉不受该参数影响。
  - `promote_enabled=true` 时，L3 命中会回填 engine，后续同 engine 访问可直接命中 L1/L2。
- 每个 `infer_clusters[]` 通过 `pool_instance_id` 显式绑定一个 L3 pool；重复推理实例、未知 pool instance 都会初始化失败。
- 多个推理实例可以映射到同一个 L3 pool instance；不同 pool instance 之间互相隔离。
- `get` 先查 engine L1/L2，未命中的后缀再查 L3；engine 已命中的前缀不重复记录 L3 命中，只有开启 `access_propagation_enabled` 时才刷新 L3 访问时间。
- `write` 总是先写 engine instance；是否写入对应的 L3 pool instance 由 `l2_l3_strategy.write_mode` 决定。
- combined 统计中 `LocalHit*` 表示 engine L1/L2 命中，`RemoteHit*` 表示 L3 pool 命中，`Hit* = LocalHit* + RemoteHit*`。
- `enable_lifecycle_tracking=true` 时，engine manager 和 L3 pool manager 都会导出各自 instance 的 `*_lifecycle.csv`；默认关闭，避免大 trace 下额外内存开销。

输出：

- `output_result_path/hierarchical_hit_rates.csv`：端到端 combined 结果，包含 `LocalHitBlocks` / `RemoteHitBlocks` / `HitBlocks`、对应 token 字段、当前与累计 token hit rate。
- `output_result_path/infer/`：推理侧独立统计，用于分析每个推理实例本地 L1/L2；开启 `enable_lifecycle_tracking` 后也会输出 `*_lifecycle.csv`。
- `pool_config.output_result_path`：L3 pool 侧独立统计，用于分析 KVCM/L3 池化层；开启 `enable_lifecycle_tracking` 后也会输出 `*_lifecycle.csv`。

配置骨架：

```json
{
  "trace_file_path": "/path/to/standard_trace.jsonl",
  "output_result_path": "/tmp/hierarchical",
  "infer_scheduling_strategy": "preserve_trace",
  "enable_lifecycle_tracking": false,
  "infer_clusters": [
    {
      "pool_instance_id": "model_l3",
      "model": {
        "block_size": 1024,
        "bytes_per_token": 93594,
        "eviction_policy_type": "lru",
        "eviction_policy_params": {
          "sample_rate": 1.0,
          "shard_count": 1,
          "sample_times": 32,
          "eviction_amplification_factor": 1.0
        }
      },
      "instance_ids": ["infer_a", "infer_b"],
      "ttl_config": {
        "default_block_ttl_seconds": 0,
        "refresh_on_read": true
      },
      "tier_flows": [
        {
          "from_tier": "hbm",
          "to_tier": "dram",
          "write_mode": "write_through",
          "access_propagation_enabled": false,
          "write_propagation_enabled": false,
          "promote_enabled": true,
          "selective_write_threshold": 2
        }
      ],
      "tiers": [
        {
          "name": "hbm",
          "capacity": 1167
        },
        {"name": "dram", "capacity": 1070.4}
      ]
    }
  ],
  "l2_l3_strategy": {
    "write_mode": "cascading",
    "access_propagation_enabled": false,
    "write_propagation_enabled": false,
    "promote_enabled": true,
    "selective_write_threshold": 2
  },
  "pool_config": {
    "trace_file_path": "/path/to/standard_trace.jsonl",
    "output_result_path": "/tmp/hierarchical/pool",
    "eviction_params": {
      "eviction_mode": 1,
      "eviction_batch_size_per_instance": 100
    },
    "instance_groups": [
      {
        "group_name": "model_l3_group",
        "quota_capacity": 18432,
        "used_percentage": 1.0,
        "ttl_config": {
          "default_block_ttl_seconds": 0,
          "refresh_on_read": true
        },
        "storages": [
          {"unique_name": "l3", "storage_type": "dummy", "band_width_mbps": 0, "priority": 0, "capacity": 18432}
        ],
        "instances": [
          {
            "instance_id": "model_l3",
            "block_size": 1024,
            "bytes_per_token": 93594,
            "instance_group_name": "model_l3_group",
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
}
```
