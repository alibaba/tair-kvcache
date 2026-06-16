# Optimizer C++ 内核参数映射说明

## 参数传入链路

```
CLI/脚本参数                     Python config builder                   C++ 实例
──────────────────────────────────────────────────────────────────────────────────
--page-size 256            → scheduler_config.page_size             → OptInstanceConfig.block_size_
--kv-bytes-per-token 46080 → kv_bytes_per_token                     → OptInstanceConfig.bytes_per_token_
--hbm-capacity 80          → platform_config.hbm_capacity_gb        → OptTierConfig("hbm").capacity_
--mem-capacity 340         → platform_config.memory_capacity_gb     → OptTierConfig("dram").capacity_
--num-p-instances 200      → p_instance_ids = ["P0"..."P199"]       → 200个 OptInstanceGroupConfig
```

## 对应 C++ 实例和字段

| 仿真参数 | JSON 路径 | C++ 类 | C++ 字段 | 用途 |
|----------|-----------|--------|----------|------|
| `block_size=256` | `infer_clusters[0].model.block_size` | `OptInstanceConfig` | `block_size_ = 256` | 每个 block 包含 256 个 token |
| `bytes_per_token=46080` | `infer_clusters[0].model.bytes_per_token` | `OptInstanceConfig` | `bytes_per_token_ = 46080` | 每个 token 的 KV cache 字节数 |
| `bytes_per_block=11796480` | 无（派生值） | `OptInstanceConfig` | `bytes_per_block()` = `block_size_ * bytes_per_token_` | 淘汰时计算存储占用 |
| `num_instances=200` | `infer_clusters[0].infer_ids` | `OptInstanceGroupConfig` x 200 | 每个 engine 一个 group | 200 个独立的缓存引擎 |
| `hbm=80.0 GB` | `infer_clusters[0].tiers[0].capacity` | `OptTierConfig` | `capacity_` = 80*2^30 bytes | HBM tier 容量上限 |
| `dram=340.0 GB` | `infer_clusters[0].tiers[1].capacity` | `OptTierConfig` | `capacity_` = 340*2^30 bytes | DRAM tier 容量上限 |

## 容量到 block 数的转换

```
eviction 判断: current_used_bytes > tier_capacity_bytes ?

current_used_bytes = 当前 tier 中的 block 数 * bytes_per_block (11,796,480)

HBM 最大 block 数 = 85,899,345,920 / 11,796,480 ≈ 7,281 blocks = 1,863,936 tokens
DRAM 最大 block 数 = 365,072,220,160 / 11,796,480 ≈ 30,948 blocks = 7,922,688 tokens
```

## C++ 对象组织结构

```
HierarchicalReplayManager
├── engine_manager_: OptimizerManager (管理所有引擎的 RadixTree)
│   └── 200 个 OptInstanceGroupConfig，每个:
│       ├── group_name = "P0" ~ "P199"
│       ├── quota_capacity = GbToBytes(80+340) = 450,971,566,080 bytes (总配额)
│       ├── storages[0] = OptTierConfig{name="hbm", capacity=85,899,345,920}
│       ├── storages[1] = OptTierConfig{name="dram", capacity=365,072,220,160}
│       └── instances[0] = OptInstanceConfig{
│               block_size=256,
│               bytes_per_token=46080,
│               bytes_per_block()=11,796,480
│           }
├── p2p_tracker_: TierGlobalTracker
│   └── holders_["infer_cluster_0\x1fhbm"][block_key] → {"P0","P3",...}
├── storage_pool_manager_: HashStoragePoolManager
└── engine_block_size_["P0"~"P199"] = 256  (用于 hit_blocks → token 转换)
```

## 关键源码位置

| 逻辑 | 文件 | 行号/函数 |
|------|------|-----------|
| JSON→Config 解析 | `config/hierarchical_replay_config.cc` | `HierarchicalInferClusterConfig::FromRapidValue` |
| Config→Optimizer 初始化 | `config/hierarchical_replay_config.cc:293` | `BuildInternalConfig()` |
| tier capacity GB→bytes | `config/hierarchical_replay_config.cc:26` | `BuildOptimizerTier()` → `GbToBytes(capacity)` |
| bytes_per_block 计算 | `config/instance_config.h:23` | `bytes_per_block() = bytes_per_token_ * block_size_` |
| 淘汰容量判断 | `manager/eviction_manager.cc:431` | `GetExcessUsage()` → `capacity = storages[tier_idx].capacity()` |
| block 占用计算 | `manager/eviction_manager.cc:419` | `num_blocks * bytes_per_block` |
| Python config 生成 | `schedule_simulator/.../hierarchical_config_builder.py` | `build_hierarchical_config()` |

## 参数一致性要求

1. **所有 200 个 engine 共享相同的** `block_size`、`bytes_per_token`、`tier capacity`
2. 每个 engine 独立管理自己的 RadixTree 和 LRU 淘汰
3. P2P tracker 在 `"hbm"` tier 上追踪所有 engine 的 block 位置
4. `bytes_per_token` 必须是 TP 分片后单卡的值（如 TP=8 时为全量的 1/8）
5. `hbm capacity` 必须是扣除模型权重、框架预留后的实际可用值
