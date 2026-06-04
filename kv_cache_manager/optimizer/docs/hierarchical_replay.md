# 推理本地多层 + Storage Pool 联动回放

`hierarchical_replay_main` 用于模拟完整的 engine-local + storage pool 流程。只要需要同时评估推理实例本地多层缓存和 KVCM/storage pool 池化，就使用这个入口；trace 是否已经指定推理实例不影响入口选择。

入口：

```bash
bazel run //kv_cache_manager/optimizer:hierarchical_replay_main -- /path/to/hierarchical_replay_config.json
```

回放语义：

- `infer_clusters` 定义推理实例集合。映射到同一个 storage pool 的同构推理实例只写一份 model、TTL、tier 列表和层间 flow。
- `infer_clusters[].model` 定义该推理实例集合的 block size、bytes per token 和驱逐策略。
- `infer_clusters[].engine_read_query_type` 定义 engine-local 本地读语义，必须显式配置为 `prefix_match` 或 `batch_get`。
- `infer_eviction_params` 定义推理侧本地缓存的驱逐模式，`1` 是 group rough，`2` 是 instance rough，`3` 是 instance precise。
- `infer_clusters[].ttl_config` 定义该推理实例集合的 TTL 行为。
- `infer_clusters[].tiers[]` 定义推理实例本地层，数组顺序就是写入顺序；第一项是写入口和最高层。
- `infer_clusters[].tier_flows` 写满本地相邻层级的流动策略。
- `infer_clusters[].storage_pool_flow` 定义该推理实例集群本地最后一层到 storage pool 的流动策略。这条边跨 engine manager 和 storage pool manager，不属于单个 optimizer instance group 的 `tier_flows`。
- `infer_clusters[].p2p_read_flows` 可选定义同一个推理集群内的 peer read。P2P 位于 engine-local read 和 storage pool read 之间，命中计入 `PeerHit`，设计细节见 `docs/p2p_read.md`。
- `storage_pool` 描述 KVCM/storage pool；当前只支持单层 hash pool，不再复用普通 optimizer 的 `instance_groups/storages` 配置形态。
- `infer_scheduling_strategy=preserve_trace` 时，标准 trace 的 `instance_id` 表示推理实例。
- `infer_scheduling_strategy=round_robin` 时，回放前按 get/write pair 轮询分配到推理实例；write 跟随前一个 get。
- `infer_scheduling_strategy=prefix_hit` 时，每个 get 选择当前本地前缀命中最长的推理实例；冷启动或并列时按确定性轮询分配，write 跟随前一个 get。
- `tier_flows` / `storage_pool_flow` 的 `write_mode`：
  - `write_mode=write_through`：trace write 先写 engine；多层 engine 只把本次写入实际传到最后一层的 block 写入 storage pool，包括直接写穿到最后一层和 engine 内部驱逐下沉到最后一层的 block；非分层 engine 使用本次写入 shared 层的 block。
  - `write_mode=cascading`：trace write 只写 engine；engine 侧被完全驱逐出去的 block 写入 storage pool。
  - `write_mode=write_through_selective`：trace write 只写 engine；本次写入触达到 engine pool-source 层，且 write touch 次数达到 `selective_write_threshold` 的 block 再写入 storage pool。
  - `local_read_touch_enabled=true` 时，engine 已命中的 block 传入 storage pool 后会刷新 pool 中对应已有副本的读访问时间。
  - `shadow_write_touch_enabled=true` 时，engine 写 pool 遇到已有副本会刷新 pool 冷热；write-through、cascading 和 selective 写入 pool 都使用该配置。
  - storage pool 命中会固定回填 engine，后续同 engine 访问可直接命中本地缓存。
- 每个 `infer_clusters[]` 通过 `storage_pool_id` 显式绑定一个 storage pool；重复推理实例、未知 storage pool 都会初始化失败。
- 多个推理实例可以映射到同一个 storage pool；不同 storage pool 之间互相隔离。
- `get` 先按 `engine_read_query_type` 查 engine 本地缓存，然后将完整 keys 传入 storage pool。trace 的 `query_type` 不会改写 engine-local 语义；它只决定 storage pool 侧使用 `prefix_match` 还是 `batch_get`。`query_type=prefix_match` 时，storage pool 按 prefix 语义推进，engine 已命中的 index 会跳过 pool remote hit 计数但允许 prefix 继续向后匹配；遇到 engine 和 storage pool 都未命中的 block 才停止。`query_type=batch_get` 时，storage pool 也逐 block 独立查询，engine 已命中的 index 不会重复计成 pool remote hit。
- `write` 总是先写 engine instance；是否写入对应的 storage pool 由当前 infer cluster 的 `storage_pool_flow.write_mode` 决定。
- write-through/cascading/selective 写入 storage pool 时只写真实进入 pool 的 block key。
- combined 统计中 `LocalHit*` 表示 engine 本地命中，`PeerHit*` 表示同集群 peer 命中，`RemoteHit*` 表示 storage pool 命中，`Hit* = LocalHit* + PeerHit* + RemoteHit*`。未开启 P2P 时 `PeerHit*` 为 0。
- `enable_lifecycle_tracking=true` 时，engine manager 和 storage pool manager 都会导出各自 instance 的 `*_lifecycle.csv`；默认关闭，避免大 trace 下额外内存开销。

输出：

- `output_result_path/hierarchical_hit_rates.csv`：端到端 combined 结果，包含 `LocalHitBlocks` / `PeerHitBlocks` / `RemoteHitBlocks` / `HitBlocks`、对应 token 字段、当前与累计 token hit rate。
- `output_result_path/infer/`：推理侧独立统计，用于分析每个推理实例本地缓存；开启 `enable_lifecycle_tracking` 后也会输出 `*_lifecycle.csv`。
- `storage_pool.output_result_path`：storage pool 侧独立统计，用于分析 KVCM/storage pool 池化层；开启 `enable_lifecycle_tracking` 后也会输出 `*_lifecycle.csv`。

配置骨架：

```json
{
  "trace_file_path": "/path/to/standard_trace.jsonl",
  "output_result_path": "/tmp/hierarchical",
  "infer_scheduling_strategy": "preserve_trace",
  "enable_lifecycle_tracking": false,
  "infer_eviction_params": {
    "eviction_mode": 3,
    "eviction_batch_size_per_instance": 100
  },
  "infer_clusters": [
    {
      "storage_pool_id": "model_l3",
      "engine_read_query_type": "batch_get",
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
      "infer_ids": ["infer_a", "infer_b"],
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
          "selective_write_threshold": 2
        }
      ],
      "storage_pool_flow": {
        "write_mode": "cascading",
        "local_read_touch_enabled": false,
        "shadow_write_touch_enabled": false,
        "selective_write_threshold": 2
      },
      "p2p_read_flows": [
        {
          "tier": "dram",
          "peer_read_touch_enabled": true
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
  "storage_pool": {
    "output_result_path": "/tmp/hierarchical/pool",
    "storage_name": "l3",
    "capacity": 18432,
    "eviction_params": {
      "eviction_mode": 1,
      "eviction_batch_size_per_instance": 100
    },
    "ttl_config": {
      "default_block_ttl_seconds": 0,
      "refresh_on_read": true
    },
    "pools": [
      {
        "pool_id": "model_l3",
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
        }
      }
    ]
  }
}
```
