# KVCacheManager Optimizer

> [中文](README_zh.md) | English

## Overview

KVCacheManager Optimizer is a standalone cache optimization analysis module. It replays trace data to simulate cache read/write operations, evaluates the impact of different eviction policies and configurations on cache hit rate, and provides parameter optimization capabilities for the KVCacheManager main program.

Core capabilities include:
- Trace data replay and simulation
- Simulation and comparison of multiple eviction policies
- Real-time statistics and analysis of cache hit rate
- Visualization of the Radix Tree index structure
- LiteHit: capacity-independent hit-rate analysis for full-attention scenarios (a single replay produces facts, arbitrary LRU capacities are projected after the fact, with re-blocking and multi block-size fanout supported). See [liteHit/README.md](liteHit/README.md)

## Motivation

In large language model (LLM) inference scenarios, KV Cache management is critical to system performance. Different eviction policies, cache capacity configurations, and storage tier settings significantly affect cache hit rate and overall inference efficiency. The design motivations of the Optimizer module include:

1. **Policy evaluation**: Before deploying to production, evaluate the effectiveness of different eviction policies through trace replay and select the optimal configuration
2. **Parameter tuning**: Analyze cache access patterns and provide optimization suggestions for eviction policy parameters (such as sample rate, TTL, etc.)
3. **Performance prediction**: Predict cache hit rate under different capacity configurations to assist resource planning
4. **Problem diagnosis**: Help understand cache behavior and performance bottlenecks through Radix Tree visualization and detailed statistics

## Features and Architecture

### Core Features

- **Multiple eviction policies**: Supports eviction algorithms such as LRU, RandomLRU, LeafAwareLRU, and TTL
- **TTL expiration mechanism**: Currently adopts V1 semantics; only `POLICY_TTL` performs physical cleanup of TTL-expired entries
- **Tiered storage**: Supports multi-tier storage configuration; currently the feature is not fully complete
- **Trace replay**: Supports multiple trace formats such as Publisher Log and Qwen Bailian
- **Detailed statistics**: Provides detailed statistics such as hit rate and cache usage
- **Flexible configuration**: Flexibly configure instances, storage, and policies through JSON configuration files
- **Visual analysis**: Supports Radix Tree visualization and hit-rate chart generation

Standard strategy configuration, multi-instance replay, trace schema, and hit-rate metrics are described in [docs/strategy_config.md](docs/strategy_config.md). In the standard edition, `HitRate` uniformly denotes the overall token hit rate, i.e. `HitTokens / InputTokens`; local/remote serve only as a diagnostic split between the trace `block_mask` and the optimizer's simulated hits, and are not used as a standard conclusion dimension. Python entry points that pass an optimizer config uniformly use `output_result_path` from the config; `multi_instance_replay` does not read the full config and uses an explicit `--output-dir`. A standard `get` trace must contain `input_len`; when only request-level logs are available externally, `type=request` can be used, and the optimizer will internally schedule delayed writes according to `trace_replay.write_delay_ns`; already-split `get` / `write` traces are still supported (replay path only; LiteHit facts replay recognizes and ignores `write` events, and a `get` submission is treated as write-back complete).

### Architecture Design

```
OptimizerManager (core coordinator)
    ├── OptEvictionManager (eviction manager)
    ├── OptIndexerManager (indexer manager)
    └── OptimizerRunner (trace executor)
        ↓
    ├── Eviction Policies
    ├── RadixTreeIndex (index)
    └── Trace Converter
        ↓
    HitAnalysis (result analysis)
```

### 在线订阅 KVCM 事件

`online_optimizer_server_main` 可以主动发现 KVCM，并通过 KVCM 现有的 Meta gRPC 端口订阅缓存读取事件。KVCM 不需要知道 Optimizer 地址，也不新增事件端口；Optimizer 是 gRPC 客户端，调用 `OptimizerEventStreamService.SubscribeEvents`，KVCM 通过 response stream 写入 `TraceQueryRequest`。

KVCM 侧先在原有 server 配置中启用 optimizer publisher：

```properties
kvcm.event.event_publishers_configs={"log":{"enable":true,"queue_size":10000},"optimizer":{"enable":true,"queue_size":100000,"max_subscribers":4,"subscriber_queue_size":10000}}
```

Optimizer 侧在现有 JSON 配置中加入订阅配置，不使用额外配置文件：

```json
{
    "kvcm_event_subscription": {
        "enable": true,
        "service_discovery_url": "static://127.0.0.1:6381",
        "consumer_id": "online-optimizer",
        "discovery_refresh_interval_ms": 5000
    }
}
```

`service_discovery_url` 指向 KVCM 的 `kvcm.service.rpc_port`，支持通用服务发现 URL（如 `static://`、`vipserver://`、`spectrum://`）。发现结果只作为 seed：Optimizer 周期调用任一健康 seed 的 `MetaService.GetClusterInfo` 获取当前 Leader，并且只向 Leader 维持一条 `SubscribeEvents` stream。切主后下一次刷新会先同步新 Leader 的配置，再关闭旧 stream、连接新 Leader；断流期间会自动重连。

`OptimizerEventStreamService.GetConfiguration` 与事件流共用 KVCM 的 Meta gRPC 端口。Optimizer 启动时及每次服务发现刷新时拉取一次 Instance Group / Instance 快照，先创建缺失的 Group，再注册缺失的 Instance；收到未知 `instance_id` 时还会立即唤醒一次配置刷新。因此 KVCM 新增实例后不需要再提前调用 Optimizer 的注册 API。当前同步只添加新配置，不删除或热更新已经存在的 Optimizer 配置。

自动注册采用能从 KVCM 配置直接确定的口径：Instance Group 的 quota byte 数转换成一个 GiB 容量点，使用当前 Optimizer 支持的 LRU，并开启 prefix hash；Instance 按 full-only 注册，优先合并名称以 `full` / `FULL` 开头的 location spec group，没有时使用全部 spec。当前在线 indexer 不支持 shared group quota，因此同组各 Instance 分别按完整 Group quota 模拟。KVCM 没有 `linear_step` 等 Optimizer 专属参数，因此这里不猜测 linear/mamba 周期。

订阅器固定使用两个线程：一个 supervisor 线程负责服务发现、Leader 查询和配置同步，一个 stream 线程负责读取当前 Leader 的事件；收到事件后直接调用 `OnlineOptimizerManager`，不增加额外事件队列。未知 Instance 的首条事件会记录并丢弃，配置刷新完成后的后续事件正常进入统计。事件时间戳用于 LiteHit 和线性 indexer 的 TTL 判定，旧客户端未设置时间戳时仍回退到 Optimizer 本机墙钟。

在线 full-attention 实例还会输出 `mrc` gauge（Prometheus 名称默认为 `kvcm_optimizer_mrc`，标签为 `instance_id`，单位 byte）。它表示最近一个 `metrics_report_interval_ms` 上报周期内，达到理论无限容量命中数 95% 所需的最小 LRU 容量；每次上报会原子取走并清空仅供 MRC 使用的 hit curve，不影响查询数、命中率等累计指标。该值直接聚合 LiteHit 产生的容量无关 hit curve，不依赖预先配置的离散容量点；周期内尚无理论可命中 block 时值为 0。


### Eviction Policies

**LRU (Least Recently Used)**
- Maintains a doubly linked list recording the access order of blocks; the most recently accessed block is at the head of the list, and the least recently accessed block is at the tail

**RandomLRU**
- Combines random sampling with the LRU policy: randomly samples a number of blocks from the cache and evicts the least recently accessed one among them

**LeafAwareLRU**
- Adds leaf-node awareness on top of LRU, preferentially evicting blocks within leaf nodes

**TTL (Time-To-Live)**
- Two-phase eviction: first clears all TTL-expired blocks; if capacity still exceeds the limit, falls back to evicting from the oldest by `last_access_time`
- `fallback_on_pressure` (default true): when disabled, degrades to pure TTL, only reclaiming expired blocks with no capacity fallback

#### TTL Expiration Mechanism (V1)

Current implementation semantics:

- Only instances with `eviction_policy_type: "ttl"` perform physical TTL expiration cleanup before reads/writes.
- Under non-TTL policies (`lru` / `random_lru` / `leaf_aware_lru`), TTL is treated as nonexistent (neither pre-cleanup nor logical expiration judgment is performed).

| Usage mode | Configuration | Eviction behavior |
|---|---|---|
| Pure LRU capacity management | `eviction_policy_type: "lru"` | LRU evicts by access time |
| TTL priority + capacity fallback | `eviction_policy_type: "ttl"` + `fallback_on_pressure: true` | First clears all expired blocks; if insufficient, falls back by list tail |
| Pure time eviction (no fallback) | `eviction_policy_type: "ttl"` + `fallback_on_pressure: false` | Only clears expired blocks; ignores capacity shortage |

**TTL behavior rules**:
- `default_block_ttl_seconds`: configured at the instance group level; `0` means TTL is disabled at the group level.
- `ttl_refresh_on_read`: controls whether a read hit refreshes the TTL anchor; default `true` (Sliding TTL); when `false`, reads do not extend lifetime.
- When TTL is disabled at the group level, request-level `ttl_seconds > 0` does not take effect (the write path forcibly disables TTL).
- On write, `ttl_anchor_time` is reset to the write time, and TTL starts counting from that anchor.
- On read (`PrefixQuery`), `last_access_time` is always refreshed; `ttl_anchor_time` is refreshed only when `ttl_refresh_on_read=true`.
- Before a read/write request is executed, a physical cleanup of TTL-expired blocks is performed first, and `CleanEvictedBlocks` uniformly handles node cleanup.
- After a write completes, `CheckAndEvict` handles capacity eviction; when `fallback_on_pressure=false`, capacity eviction is skipped.
- `POLICY_TTL` expiration cleanup has been optimized to incremental reclamation based on a min-heap of expiration times, avoiding a full list scan on every request.

**TTL lifecycle statistics notes (important)**:
- Block lifecycle statistics in TTL mode (`birth_time_us/death_time_us/lifespan_us`) only reflect the time points at which the system actually processes events, not the real expiration moments.
- Because the current implementation uses lazy eviction after write, `death_time_us` records the time when the block is cleaned up/processed by the system, not strictly `last_access_time + ttl`.
- Therefore, lifecycle data in TTL scenarios should only be used for relative trend analysis, and is not recommended for precise duration estimation or precise cross-policy comparison.

**Per-request TTL override** (via the `WriteCache` API):

| `ttl_seconds` value | Meaning |
|---|---|
| `0` (default) | Use the group's `default_block_ttl_seconds` |
| `-1` | Disable TTL; the block never expires |
| `>0` | Custom TTL (seconds); if the group has disabled TTL (`default_block_ttl_seconds=0`), this value is ignored |

### Trace Types

```
OptimizerSchemaTrace (base class)
    ├── GetLocationSchemaTrace (read operation)
    └── WriteCacheSchemaTrace (write operation)
```

**Supported Trace Formats**
- **Publisher Log**: KVCacheManager Event Publisher log, distinguishing read and write requests
- **Qwen Bailian**: Qwen Bailian open-source dataset format; forcibly distinguishes read and write requests after conversion

## Quick Start

### Step 1: Convert trace to standard format

```bash
cd tools/trace_converter

# Install dependencies (first time only)
pip install -r requirements.txt

# Convert trace
python trace_converter.py \
    -i /path/to/your_trace.jsonl \
    -o /path/to/optimizer_trace.jsonl \
    -f qwen_bailian \
    --mode optimizer
```

### Step 2: Build the Optimizer

```bash
bazel build \
    //kv_cache_manager/optimizer:optimizer_main \
    //kv_cache_manager/optimizer/analysis/script:optimizer_run
```

### Step 3: Create a configuration file

Create a JSON configuration file. Below is a minimal usable configuration for non-tiered LRU; the `instance_id` in the trace must match the `instances[].instance_id` here.

```json
{
    "trace_file_path": "/path/to/optimizer_trace.jsonl",
    "output_result_path": "/path/to/output/result/",
    "eviction_params": {
        "eviction_mode": 1,
        "eviction_batch_size_per_instance": 100
    },
    "instance_groups": [
        {
            "group_name": "instance_group_01",
            "quota_capacity": 12000,
            "used_percentage": 1.0,
            "tier_strategy": {
                "hierarchical_eviction_enabled": false,
                "write_mode": "write_through",
                "access_propagation_enabled": true,
                "promote_enabled": true,
                "selective_write_threshold": 2
            },
            "default_block_ttl_seconds": 0,
            "ttl_refresh_on_read": true,
            "storages": [],
            "instances": [
                {
                    "instance_id": "instance-a",
                    "block_size": 16,
                    "bytes_per_token": 512,
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


### Step 4: Run the Optimizer

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json
```

After running, the following is generated in the directory specified by `output_result_path`:

- `{instance_id}_hit_rates.csv` - hit-rate data for each instance

To draw a hit-rate time-series chart, add `--draw-chart`:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json \
    --draw-chart
```

Quick reference for common entry points:

| Need | Recommended entry point |
|---|---|
| Single replay, output `*_hit_rates.csv` | `optimizer_run -c config.json` |
| Single replay with time-series chart | `optimizer_run -c config.json --draw-chart` |
| Infinite-capacity theoretical hit rate | Set `quota_capacity` or tier `capacity` to `-1` in config, then run `optimizer_run` |
| Non-tiered capacity Pareto | `tradeoff -c config.json --num-points 30` |
| Multi eviction-policy Pareto | `tradeoff -c config.json --eviction-policies lru random_lru leaf_aware_lru ttl` |
| Per-pod/instance independent cache replay | `multi_instance_replay --trace-dir ... --output-dir ...` |
| Export lifecycle CSV | `optimizer_run -c config.json --export-lifecycle` |
| Analyze lifecycle charts | `analyze_lifecycle -i <lifecycle.csv or dir>` |
| Export/visualize RadixTree | `export_tree -c config.json --show-hot-paths` |

### Visual Analysis

The Optimizer module provides a variety of visual analysis tools for analyzing cache performance, hit rate, and Radix Tree structure.

#### Hit Rate Over Time Chart

Run the optimizer to analyze the trace and draw multi-instance cache analysis charts, showing the sum of storage capacity across all instances and their respective hit rates over time.

**Note**: The `trace_file_path` in the configuration file must be a standard-format trace file.

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json \
    --draw-chart
```

**Output**: `{output_result_path}/timeseries/multi_instance_cache_analysis.png`

#### Radix Tree Visualization

Export and visualize the prefix tree structure, counting and displaying hot nodes and the prefix paths of their associated nodes.
See `analysis/script/run/export_tree.py` for specific configuration.

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:export_tree -- \
    -c /path/to/config.json
```

**Output**:
- `{output_result_path}/radix_tree/{instance_id}_radix_tree.json` - Radix Tree export data
- `{output_result_path}/radix_tree/{instance_id}_radix_tree.png` - Radix Tree visualization chart

#### Pareto Capacity-Hit-Rate Curve Analysis

The unified entry point is `tradeoff`. It first uses infinite-capacity warmup to obtain the theoretical hit rate and maximum cache size, then replays the optimizer per capacity point and draws the Pareto curve of capacity versus hit rate.

> **Scope of applicability**: Tradeoff only applies to non-tiered mode. In tiered mode, the capacity scan only modifies `quota_capacity` and does not modify each tier's `storages[i].capacity`, so it cannot represent the real tiered capacity tradeoff.

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c /path/to/config.json \
    --num-points 30 \
    --min-capacity-ratio 1e-4 \
    --hit-rate-type total \
    --max-workers 4
```

**Parameter description**:
- `-c, --config` - configuration file path
- `--num-points` - maximum number of capacity sampling points to run per policy (default 30); stops early after reaching 99% of the theoretical hit rate
- `--min-capacity-ratio` - minimum capacity point relative threshold (default `1e-4`)
- `--hit-rate-type` - hit-rate type: total/local/remote/all (default total)
- `--max-workers` - maximum number of parallel execution threads (default 4)
- `--plot-title` - override the chart title

**Output**: `{output_result_path}/pareto/pareto_curve_{hit_rate_type}.png`

#### Multi-Policy Comparison Analysis

Compare multiple eviction policies via `--eviction-policies`. Each policy warms up independently, computes its theoretical hit rate independently, and independently stops after reaching 99% of the theoretical hit rate.

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c /path/to/config.json \
    --eviction-policies lru random_lru leaf_aware_lru ttl \
    --num-points 30 \
    --hit-rate-type total \
    --max-workers 4
```

**Parameter description**:
- `-c, --config` - configuration file path
- `--eviction-policies` - list of eviction policies to compare (default lru random_lru leaf_aware_lru ttl)
- `--num-points` - maximum number of capacity sampling points to run per policy (default 30); stops early after reaching 99% of the theoretical hit rate
- `--hit-rate-type` - hit-rate type: total/local/remote/all (default total)
- `--max-workers` - maximum number of parallel execution threads (default 4)

**Output**: `{output_result_path}/pareto/multi_policy_{hit_rate_type}.png`

Chart rules: the X axis is `Capacity (GB)`, the Y axis is `HitRate (%)`; the curve starts from `(0 GB, 0%)` and only the rising segment is drawn; the 95% and 99% theoretical hit rates are marked with interpolation intersections, dashed lines, and labels. If a capacity point shows a decrease, it is removed from the chart and printed in the log, while the original CSV is retained.

### Python Interface Example

```python
from kv_cache_manager.optimizer import OptimizerConfigLoader, OptimizerLoader, OptimizerManager

# Load configuration
config_loader = OptimizerConfigLoader()
config = config_loader.Load("/path/to/config.json")

# Create the optimizer
optimizer = OptimizerManager(config)
optimizer.Init()

# Run
optimizer.DirectRun()

# Analyze results
optimizer.AnalyzeResults()

# Single read/write operations (instance_id must be specified)
write_res = optimizer.WriteCache("instance_id", "trace_001", timestamp, block_ids)
read_res = optimizer.GetCacheLocation("instance_id", "trace_002", timestamp, block_ids, block_mask,
                                      input_len=real_prompt_tokens)

# Specify TTL on write (optional)
write_res = optimizer.WriteCache("instance_id", "trace_003", timestamp, block_ids,
                                  ttl_seconds=0)     # Use group default
write_res = optimizer.WriteCache("instance_id", "trace_004", timestamp, block_ids,
                                  ttl_seconds=-1)    # Disable TTL, never expires
write_res = optimizer.WriteCache("instance_id", "trace_005", timestamp, block_ids,
                                  ttl_seconds=300)   # Custom 300 seconds

# Clear cache (keep statistics)
optimizer.ClearCache("instance_id")        # Clear the specified instance
optimizer.ClearAllCaches()                 # Clear all instances

# Clear cache and reset statistics
optimizer.ClearCacheAndResetStats("instance_id")  # Clear the specified instance and reset statistics
optimizer.ClearAllCachesAndResetStats()           # Clear all instances and reset statistics
```

### Configuration Parameter Description

| Parameter | Description |
|------|------|
| eviction_mode | Eviction mode: 1=GROUP_ROUGH, 2=INSTANCE_ROUGH, 3=INSTANCE_PRECISE |
| eviction_policy_type | Eviction policy type: lru, random_lru, leaf_aware_lru, ttl |
| quota_capacity | Total group capacity in non-tiered mode, in GB; `-1` means infinite capacity, no capacity eviction triggered |
| storages[].capacity | Capacity of a single tier in tiered mode, in GB; `-1` means infinite capacity for that tier |
| tier_strategy | Multi-tier read/write policy package, including hierarchical eviction switch, write mode, read access propagation, promote and selective write thresholds; top-level fields are the default policy for all adjacent tier edges |
| tier_strategy.write_mode | Write/sink mode: `write_through`, `cascading`, `write_through_selective` |
| tier_strategy.access_propagation_enabled | Whether to refresh the access time of lower-tier copies when a read hits an upper-tier copy; it is not a write mode |
| tier_strategy.tier_flows | Policy overrides for adjacent tier edges; edges not overridden inherit the `tier_strategy` default policy |
| bytes_per_token | KV size per token; Python analysis scripts use it together with `block_size` to convert block capacity to GB |
| default_block_ttl_seconds | Default TTL (seconds) at the instance group level; 0 = disable TTL |
| ttl_refresh_on_read | TTL refresh switch at the instance group level: true=read refreshes, false=fixed window |
| fallback_on_pressure | TTL policy parameter: whether to fall back to LRU when expiration is insufficient (default true) |

## Use Cases

The following cases all assume the input is already a standard optimizer JSONL trace. Standard conclusions use token hit rate, i.e. `AccHitRate = AccHitTokens / AccInputTokens` in the last row of the CSV. The JSON in the examples mostly shows only one item in `instance_groups[]`; a complete config still requires the top-level `trace_file_path`, `output_result_path`, and `eviction_params`.

### Case 1: Global Pooled Infinite-Capacity Theoretical Hit Rate

Used to answer "if capacity were infinite, how much could the current trace theoretically hit at most". In a non-tiered global pooled configuration, set `quota_capacity` to `-1`:

```json
{
    "group_name": "global_pool",
    "quota_capacity": -1,
    "used_percentage": 1.0,
    "tier_strategy": {
        "hierarchical_eviction_enabled": false,
        "write_mode": "write_through",
        "access_propagation_enabled": true,
        "promote_enabled": true,
        "selective_write_threshold": 2
    },
    "storages": [],
    "instances": [
        {
            "instance_id": "global",
            "block_size": 256,
            "bytes_per_token": 512,
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
```

Run:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/global_pool_unlimited.json
```

For results, look at `AccHitRate`, `AccInputTokens`, and `AccHitTokens` in the last row of `{output_result_path}/global_hit_rates.csv`.
If the trace uses real pod names as `instance_id`, you need to change `instances[].instance_id` in the example above to the corresponding names.

### Case 2: Single-Tier Finite-Capacity Replay

Used to evaluate the actual token hit rate under a given capacity. Non-tiered mode only requires setting `quota_capacity`, in GB:

```json
{
    "group_name": "finite_pool",
    "quota_capacity": 742.18,
    "used_percentage": 1.0,
    "tier_strategy": {
        "hierarchical_eviction_enabled": false,
        "write_mode": "write_through",
        "access_propagation_enabled": true,
        "promote_enabled": true,
        "selective_write_threshold": 2
    },
    "storages": []
}
```

`block_size * bytes_per_token` determines the KV capacity corresponding to one block; Pareto charts and capacity output all depend on this conversion.

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/finite_pool.json \
    --draw-chart
```

Output:

- `{output_result_path}/*_hit_rates.csv`
- `{output_result_path}/timeseries/multi_instance_cache_analysis.png`

### Case 3: HBM + DRAM Tiered Replay, write-through, reads do not update lower tier

Used to simulate online multi-tier caching. When `hierarchical_eviction_enabled=true`, each tier evicts independently according to `storages[].capacity`; in this case `quota_capacity` is only a reserved field and is not used for per-tier eviction.

```json
{
    "group_name": "tiered_pool",
    "quota_capacity": 1,
    "used_percentage": 1.0,
    "tier_strategy": {
        "hierarchical_eviction_enabled": true,
        "write_mode": "write_through",
        "access_propagation_enabled": false,
        "promote_enabled": true,
        "selective_write_threshold": 2
    },
    "storages": [
        {
            "unique_name": "hbm",
            "storage_type": "hbm",
            "band_width_mbps": 20000,
            "priority": 0,
            "capacity": 1167
        },
        {
            "unique_name": "dram",
            "storage_type": "dram",
            "band_width_mbps": 20000,
            "priority": 1,
            "capacity": 1070.4
        }
    ]
}
```

Run:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/tiered_hbm_dram.json \
    --draw-chart
```

For tiered results, look at `Tier<N>(name)_HitTokens`, `AccTier<N>(name)_HitRate`, and `Tier<N>(name)_BlockNum`; the chart is at `{output_result_path}/timeseries/per_tier_timeseries.png`.

### Case 4: Three-Tier HBM + DRAM + L3, first two tiers write-through, DRAM to L3 cascading

When the policy differs per edge, use `tier_flows` to override the default policy:

```json
{
    "tier_strategy": {
        "hierarchical_eviction_enabled": true,
        "write_mode": "write_through",
        "access_propagation_enabled": false,
        "promote_enabled": true,
        "selective_write_threshold": 2,
        "tier_flows": [
            {
                "from_tier": "hbm",
                "to_tier": "dram",
                "write_mode": "write_through",
                "access_propagation_enabled": false,
                "promote_enabled": true
            },
            {
                "from_tier": "dram",
                "to_tier": "l3",
                "write_mode": "cascading",
                "access_propagation_enabled": false,
                "promote_enabled": true
            }
        ]
    },
    "storages": [
        {"unique_name": "hbm", "storage_type": "hbm", "band_width_mbps": 20000, "priority": 0, "capacity": 1167},
        {"unique_name": "dram", "storage_type": "dram", "band_width_mbps": 20000, "priority": 1, "capacity": 960},
        {"unique_name": "l3", "storage_type": "ssd", "band_width_mbps": 20000, "priority": 2, "capacity": 2048}
    ]
}
```

The top-level fields of `tier_strategy` are the default edge policy; `tier_flows` only overrides the specified adjacent edges.

### Case 5: Per-Pod Independent Cache Replay with Global Hit-Rate Aggregation

When each online pod/instance caches independently, use `multi_instance_replay`. Each JSONL file in the input directory can only contain one `instance_id`.

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:multi_instance_replay -- \
    --trace-dir /path/to/pod_traces \
    --trace-glob "*.jsonl" \
    --output-dir /path/to/pod_replay_output \
    --l1-capacity 349.52 \
    --l2-capacity 614.4 \
    --block-size 1024 \
    --bytes-per-token 163840 \
    --eviction-policy lru \
    --default-tier-write-mode write_through \
    --disable-tier-access-propagation \
    --enable-promote \
    --max-workers 16 \
    --window-seconds 60
```

Output:

- `<output_dir>/<instance_id>_hit_rates.csv`: replay result for each pod
- `<output_dir>/aggregate/instance_aggregate.csv`: summary per pod
- `<output_dir>/aggregate/global_aggregate.csv`: overall hit rate after aggregating all pods
- `<output_dir>/aggregate/global_window_hit_rates.csv`: window-level overall hit rate

The current `multi_instance_replay` CLI directly supports two tiers, L1/L2. When L3 or more complex topologies are needed, use a full optimizer config or extend the script.

### Case 6: Capacity Pareto Chart

Non-tiered capacity scanning uses `tradeoff`. The script first runs an infinite-capacity warmup to obtain the theoretical hit rate, then generates capacity points; it stops after reaching 99% of the theoretical hit rate.

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c /path/to/global_pool_config.json \
    --num-points 30 \
    --min-capacity-ratio 1e-4 \
    --hit-rate-type total \
    --max-workers 8 \
    --plot-title "service-a Pareto"
```

Multi-policy comparison:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c /path/to/global_pool_config.json \
    --eviction-policies lru random_lru leaf_aware_lru ttl \
    --num-points 30 \
    --hit-rate-type total \
    --max-workers 8
```

Output:

- `{output_result_path}/pareto/pareto_curve_total.png`
- `{output_result_path}/pareto/multi_policy_total.png`

The chart marks the capacities corresponding to 95%/99% theoretical hit rate; decreasing points are only removed from the chart, and the original CSV is not modified.

### Case 7: Block Lifecycle Analysis

First export the lifecycle CSV during replay:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json \
    --export-lifecycle
```

Then generate lifecycle statistics and charts:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:analyze_lifecycle -- \
    -i /path/to/output_result_path
```

To print statistics only, without drawing charts:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:analyze_lifecycle -- \
    -i /path/to/output_result_path \
    --stats-only
```

### Case 8: RadixTree Hot Path Troubleshooting

Used to inspect hot prefixes, leaf nodes, and cached/evicted blocks:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:export_tree -- \
    -c /path/to/config.json \
    --show-hot-paths \
    --hot-nodes 50 \
    --show-blocks
```

Output:

- `{output_result_path}/radix_tree/{instance_id}_radix_tree.json`
- `{output_result_path}/radix_tree/{instance_id}_hot_paths.png`

### Case 9: Result Reading Metrics

Standard reports only look at token hit rate:

| Metric | Description |
|---|---|
| `InputTokens` / `AccInputTokens` | Current / cumulative input token count |
| `HitTokens` / `AccHitTokens` | Current / cumulative hit token count |
| `HitRate` | Current request token hit rate |
| `AccHitRate` | Cumulative token hit rate; the main conclusion looks at this column |
| `ReadBlocks` / `HitBlocks` | Block-level diagnostic fields, not used as the final hit-rate metric |
| `Tier<N>(name)_HitTokens` | Tiered hit token count |
| `AccTier<N>(name)_HitRate` | Tiered cumulative token hit rate |


## Trace Input Format

### Overview

The Optimizer only accepts standard-format trace files. Use a standalone Python tool to convert various trace formats to the standard format.

### Standard Format

The Optimizer supports two standard trace types:

- **GetLocationSchemaTrace**: read operation (prefill phase)
- **WriteCacheSchemaTrace**: write operation (decode phase)

**Recommendation**: Unify all Optimizer inputs to the Get+Write format to preserve precise read/write timing.
A standard `get` trace must contain `input_len`; `keys` can only contain complete block keys — trailing tokens that do not fill a full block are not written into `keys`, but are still counted into `InputTokens`.

---

### Conversion Tool

Use a standalone Python tool to convert various formats (no bazel required):

```bash
# Enter the tool directory
cd tools/trace_converter

# Install dependencies (first time only)
pip install -r requirements.txt

# Convert trace to Optimizer format
python trace_converter.py \
    -i your_trace.log \
    -o optimizer_trace.jsonl \
    -f <format> \
    --mode optimizer

```

**Supported formats**:
- `publisher_log`: KVCacheManager Event Publisher log
- `qwen_bailian`: Qwen Bailian open-source dataset
- `text`: text conversation (requires specifying --tokenizer-path)

**Auto-discovery of Converters**:

The system automatically scans and discovers all available converters:

```bash
# Use a built-in converter
python3 trace_converter.py -i input.jsonl -o output.jsonl -f qwen_bailian

# Use a custom converter
python3 trace_converter.py -i input.jsonl -o output.jsonl -f custom \
    --converter-module /path/to/custom_converter.py
```

See: [Trace Converter documentation](tools/trace_converter/README.md)

---

### Configuration File

**New configuration**:
```json
{
    "trace_file_path": "/path/to/optimizer_trace.jsonl",
    "output_result_path": "/path/to/output",
    "eviction_params": {
        "eviction_mode": 1,
        "eviction_batch_size_per_instance": 100
    },
    "instance_groups": [
        {
            "group_name": "instance_group_01",
            "quota_capacity": 12000,
            "used_percentage": 1.0,
            "tier_strategy": {
                "hierarchical_eviction_enabled": false,
                "write_mode": "write_through",
                "access_propagation_enabled": true,
                "promote_enabled": true,
                "selective_write_threshold": 2
            },
            "default_block_ttl_seconds": 0,
            "storages": [],
            "instances": [
                {
                    "instance_id": "instance",
                    "block_size": 16,
                    "bytes_per_token": 512,
                    "eviction_policy_type": "lru"
                }
            ]
        }
    ]
}
```

**Note**:
- `trace_file_path` must be a standard-format file

---

### Usage Example

Complete workflow:

```bash
# Step 1: Convert trace
cd tools/trace_converter
python trace_converter.py \
    -i /path/to/qwen_trace.jsonl \
    -o /path/to/optimizer_trace.jsonl \
    -f qwen_bailian \
    --mode optimizer

# Step 2: Run the Optimizer
cd ../..
bazel run //kv_cache_manager/optimizer:optimizer_main -- /path/to/config.json
```

---

### Adding a Custom Trace Converter

If you need to support a new trace format, add a new converter in the Python tool:

1. **Create a Converter class**: create a new file in any directory
   ```python
   from converters.base import BaseConverter
   
   class MyCustomConverter(BaseConverter):
       def convert(self, input_file: str, output_file: str) -> int:
           # Implement conversion logic
           pass
   ```

2. **Usage - Option 1 (auto-scan directory)**:
   ```bash
   # Place the converter file in the specified directory; automatically discovers all classes inheriting BaseConverter
   python trace_converter.py -i input.log -o output.jsonl -f my_custom \
       --converter-dir /path/to/your/converters
   ```

3. **Usage - Option 2 (explicitly register file)**:
   ```bash
   # Directly specify the converter file and class name
   python trace_converter.py -i input.log -o output.jsonl -f my_custom \
       --converter-module /path/to/my_custom_converter.py:MyCustomConverter
   ```

**No need to modify the `trace_converter.py` source code** — the Converter automatically infers the format name from the class name:
- `MyCustomConverter` → `my_custom`
- `QwenBailianConverter` → `qwen_bailian`

---

### Related Documentation

- [Trace Converter tool documentation](tools/trace_converter/README.md) - detailed description of the Python conversion tool
