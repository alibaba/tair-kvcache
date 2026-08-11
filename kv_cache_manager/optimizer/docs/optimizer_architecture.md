# KVCacheManager Optimizer Architecture Document

> [中文](optimizer_architecture_zh.md) | English

## Table of Contents

1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [Core Modules](#core-modules)
4. [Configuration System](#configuration-system)
5. [Eviction Policies](#eviction-policies)
6. [Index System](#index-system)
7. [Trace Processing](#trace-processing)
8. [Result Analysis](#result-analysis)
9. [Visual Analysis](#visual-analysis)
10. [Usage Instructions](#usage-instructions)
11. [Extension Guide](#extension-guide)

---

## Overview

KVCacheManager Optimizer is a standalone cache optimization analysis module. It replays trace data to simulate cache read/write operations, evaluates the impact of different eviction policies and configurations on cache hit rate, and provides parameter optimization capabilities for the KVCacheManager main program.

### Main Features

- **Multiple eviction policies**: Supports eviction algorithms such as LRU, RandomLRU, LeafAwareLRU, and TTL
- **Tiered storage**: Supports multi-tier storage configuration; currently the feature is not fully complete
- **Trace replay**: Supports multiple trace formats such as Publisher Log and Qwen Bailian
- **Read/write separation**: Supports read/write separation mode and combined mode
- **Detailed statistics**: Provides detailed statistics such as hit rate and cache usage
- **Flexible configuration**: Flexibly configure instances, storage, and policies through JSON configuration files
- **Visual analysis**: Supports Radix Tree visualization, hit-rate charts, and Trade-off curve analysis
- **Online optimization service**: Supports registering online instances via service interfaces, real-time TraceQuery statistics of hit rate and capacity usage
- **Lightweight multi-capacity analysis**: LiteHit uses reuse distance to precisely compute hit rates for multiple full-attention LRU capacities (including infinite capacity) in one pass, supporting both offline request sequences and online streaming requests

---

## Architecture Design

### Overall Architecture

The Optimizer currently contains two execution paths: offline trace replay and online optimization service:

- The offline replay path uses `OptimizerManager`, `OptIndexerManager`, `OptEvictionManager`, and `RadixTreeIndex`, depending on replay configuration, trace loader/converter, and data storage types.
- The online service path uses `OnlineOptimizerManager`, `LiteHit`/`LiteHitMamba`, and service/protobuf interfaces, depending on online instance group/instance configuration and registry. `InstanceState` holds exactly one analyzer per instance: LiteHit for full attention, LiteHitMamba for linear attention.

The two paths share the hit-rate modeling code within the optimizer module, but the runtime, configuration targets, and service boundaries remain independent, avoiding the online service introducing offline replay/data storage dependencies.

#### Offline Replay

```
main.cc (program entry)
    ↓
OptimizerManager (core coordinator)
    ├── OptEvictionManager (eviction manager)
    ├── OptIndexerManager (indexer manager)
    └── OptimizerRunner (trace executor)
        ↓
    ├── Eviction Policies
    │   ├── LRU
    │   ├── RandomLRU
    │   └── LeafAwareLRU
    ├── RadixTreeIndex (index)
    └── Trace Converter
        ↓
    HitAnalysis (result analysis)
        ↓
    Visualization Tools
```

#### Online Service

```
OptimizerServiceImpl (HTTP/gRPC interface)
    ↓
OnlineOptimizerManager (online runtime coordinator)
    ├── OptimizerRegistryManager (instance group/instance persistence)
    └── InstanceState (instance_id-isolated runtime state)
        ↓
    ├── LiteHit (full-attention multi-capacity LRU, with optional fixed TTL)
    └── LiteHitMamba (linear attention: byte-weighted LRU over Full blocks + Mamba checkpoints)
```

### Directory Structure

```
kv_cache_manager/optimizer/
├── manager/              # core management layer
│   ├── optimizer_manager.h/cc       # main coordinator
│   ├── optimizer_runner.h/cc        # trace executor
│   ├── eviction_manager.h/cc        # eviction manager
│   ├── indexer_manager.h/cc         # indexer manager
│   ├── optimizer_loader.h/cc        # trace loader
│   ├── lite_hit_offline_runner.h/cc # offline LiteHit runner: drives OnlineOptimizerManager in-process
│   └── online_runtime/              # online runtime
│       └── online_optimizer_manager.h/cc # online instance registration, TraceQuery, and statistics
├── index/                # index layer
│   └── radix_tree_index.h/cc        # offline Radix tree index
├── liteHit/              # lightweight multi-capacity hit-rate core
│   ├── lite_hit.h/cc                # full-attention multi-capacity LRU hit-rate analyzer (with optional fixed TTL)
│   ├── lite_hit_mamba.h/cc          # linear attention: Full + Mamba checkpoint hit-rate analyzer
│   ├── weighted_lru_pool.h/cc       # byte-weighted LRU pool (typed keys: Full / Mamba)
│   └── dynamic_fenwick_tree.h/cc    # order-statistics Fenwick for reuse-distance
├── eviction_policy/      # eviction policy layer
│   ├── base.h                   # policy base class
│   ├── common_structure.h       # common data structures
│   ├── lru.h/cc                 # LRU policy
│   ├── random_lru.h/cc          # RandomLRU policy
│   ├── leaf_aware_lru.h/cc      # LeafAwareLRU policy
│   └── policy_factory.h/cc      # policy factory
├── trace_converter/      # trace conversion layer
│   ├── optimizer_schema_trace.h  # trace definitions
│   ├── base_converter.h          # converter base class
│   ├── publisher_log_converter.h/cc  # Publisher Log converter
│   ├── qwen_bailian_converter.h/cc    # Qwen Bailian converter
│   ├── converter_factory.h/cc    # converter factory
│   └── trace_util.h              # trace utilities
├── config/               # configuration layer
│   ├── optimizer_config.h/cc     # top-level config
│   ├── replay_instance_group_config.h/cc # trace replay/tier simulation instance group config
│   ├── replay_instance_config.h/cc      # trace replay instance config
│   ├── optimizer_instance_group.h/cc    # online optimization instance group config
│   ├── optimizer_instance_info.h/cc     # online optimization instance config
│   ├── optimizer_registry_manager.h/cc  # online instance registry
│   ├── tier_config.h/cc          # storage tier config
│   ├── eviction_config.h         # eviction policy parameters
│   └── types.h                   # type definitions
├── service/              # online service implementation
│   ├── optimizer_service_impl.h/cc    # service interface adapter layer
│   └── metrics/                   # online metrics reporting
├── analysis/             # analysis layer
│   ├── result_structure.h        # result structure definitions
│   ├── result_analysis.h/cc      # hit-rate analysis
│   └── script/                   # analysis scripts
│       ├── run/
│       │   ├── optimizer_run.py          # single replay + optional time-series chart
│       │   ├── tradeoff.py               # Pareto curve, unified entry for single/multi-policy
│       │   ├── export_tree.py            # RadixTree export + visualization
│       │   ├── analyze_lifecycle.py      # block lifecycle statistics
│       │   └── multi_instance_replay.py  # multi-instance parallel replay + aggregation
│       ├── plot/
│       │   ├── hit_rate_plot.py          # hit-rate time-series chart
│       │   ├── radix_tree_plot.py        # RadixTree plotting
│       │   └── lifecycle_plot.py         # lifecycle CDF/histogram
│       └── utils/
│           ├── optimizer_runner.py       # optimizer run wrapper
│           ├── csv_loader.py             # CSV loading + capacity point generation
│           └── plot_utils.py             # Pareto/per-tier plotting utilities
├── pybind/               # Python bindings
│   └── py_optimizer_binding.cc   # Python interface
├── main.cc               # offline replay program entry (optimizer_main)
├── lite_hit_main.cc      # offline LiteHit multi-capacity analysis entry (lite_hit_main)
└── optimizer_startup_config_load.json  # config example
```

The online service protocol is defined in `kv_cache_manager/protocol/protobuf/optimizer_service.proto`, and is converted by the service layer into optimizer online config/runtime objects.

A full-attention TraceQuery only puts complete blocks into `block_keys`, and passes the original input length including trailing tokens via `input_token_len`. The Online Manager uses the fixed byte charge of the full location spec group to floor `capacity_gb` to block capacity, then hands the same request to LiteHit; both per-request and cumulative hit rates are `prefix_hit_blocks * block_size_tokens / input_tokens`. When old clients lack the length, the compatibility assumption is that there are no trailing tokens. When a full-attention group config has `ttl_seconds != 0`, a fixed TTL is layered on top of LiteHit with wall-clock time (a strict deadline: a block expires once its age reaches the TTL, refreshed on every access).

Linear attention (`linear_step > 0`) runs on LiteHitMamba: a Full block and a Mamba checkpoint are two independent objects with independent recency, the capacity axis is **total bytes** rather than a block count, and the hit semantics is "resume from a checkpoint". It carries no time axis yet, so a group combining `linear_step > 0` with `ttl_seconds > 0` is rejected at registration (the offline runner applies the same policy).

---

## Core Modules

### 1. OptimizerManager

**Responsibility**: Core coordinator; initializes all subcomponents, manages instance group and instance configuration, and provides the public API interface.

**Main interfaces**:
```cpp
class OptimizerManager {
public:
    OptimizerManager(const OptimizerConfig &config);
    bool Init();
    void DirectRun();
    WriteCacheRes WriteCache(...);
    GetCacheLocationRes GetCacheLocation(...);
    void AnalyzeResults();
    std::unordered_map<std::string, RadixTreeExport> ExportRadixTrees() const;
};
```

### 2. OptimizerRunner

**Responsibility**: Executes trace replay and simulation, handles two trace types (GetLocationSchemaTrace, WriteCacheSchemaTrace), and supports read/write separation mode.

### 3. OptEvictionManager

**Responsibility**: Manages cross-instance eviction policies, supporting three eviction modes:
- `EVICTION_MODE_GROUP_ROUGH` - group-level coarse-grained eviction
- `EVICTION_MODE_INSTANCE_ROUGH` - instance-level coarse-grained eviction
- `EVICTION_MODE_INSTANCE_PRECISE` - instance-level precise eviction

**Main interfaces**:
```cpp
class OptEvictionManager {
public:
    bool Init(const EvictionConfig &eviction_config);
    std::shared_ptr<EvictionPolicy> CreateAndRegisterEvictionPolicy(...);
    std::unordered_map<std::string, std::vector<BlockEntry *>> EvictByMode(...);
    size_t GetCurrentGroupUsage(...) const;
    size_t GetCurrentInstanceUsage(...) const;
};
```

### 4. OptIndexerManager

**Responsibility**: Manages RadixTreeIndex instances, creates an indexer for each instance, and supports multi-tier storage configuration.

**Main interfaces**:
```cpp
class OptIndexerManager {
public:
    bool CreateOptIndexer(...);
    std::shared_ptr<RadixTreeIndex> GetOptIndexer(...) const;
    void RegisterInstanceGroups(...);
    void RegisterInstances(...);
    bool CheckAndEvict(...);
    size_t GetCurrentInstanceUsage(...) const;
};
```

### 5. OptimizerLoader

**Responsibility**: Loads and converts trace files, sorts traces by timestamp, and exports converted traces to files.

---

## Configuration System

### Configuration Hierarchy

```
OptimizerConfig (top-level config)
    ├── trace_file_path (trace file path)
    ├── output_result_path (output path)
    ├── eviction_params (eviction parameters)
    │   ├── eviction_mode (eviction mode)
    │   └── eviction_batch_size_per_instance (eviction batch size)
    └── instance_groups[] (instance group array)
        ├── group_name (group name)
        ├── quota_capacity (quota capacity)
        ├── used_percentage (used percentage)
        ├── tier_strategy (multi-tier read/write policy)
        │   ├── hierarchical_eviction_enabled (hierarchical eviction)
        │   ├── write_mode (multi-tier write mode)
        │   ├── access_propagation_enabled (read access propagation)
        │   ├── promote_enabled (lower-tier hit backfill to higher tier)
        │   └── selective_write_threshold (selective down-write threshold)
        ├── storages[] (storage tier array)
        └── instances[] (instance array)
            ├── instance_id (instance ID)
            ├── block_size (block size)
            ├── eviction_policy_type (eviction policy type)
            └── eviction_policy_params (eviction policy parameters)
```

### Configuration File Example

```json
{
    "trace_file_path": "/path/to/trace/file.jsonl",
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
            "storages": [
                {
                    "unique_name": "pace_00",
                    "storage_type": "pace",
                    "band_width_mbps": 20000,
                    "priority": 0,
                    "capacity": 100000
                }
            ],
            "instances": [
                {
                    "instance_id": "instance",
                    "block_size": 16,
                    "bytes_per_token": 512,
                    "eviction_policy_type": "random_lru",
                    "eviction_policy_params": {
                        "sample_rate": 0.1
                    }
                }
            ]
        }
    ]
}
```

### Configuration Parameter Description

| Parameter | Description |
|------|------|
| trace_file_path | trace file path |
| output_result_path | result output directory; all config-driven entry points use this directory, and `export_tree` outputs to `radix_tree/` under it |
| eviction_mode | eviction mode: 1=GROUP_ROUGH, 2=INSTANCE_ROUGH, 3=INSTANCE_PRECISE |
| eviction_batch_size_per_instance | batch size for coarse-grained eviction |
| group_name | unique identifier of the instance group |
| quota_capacity | total capacity of the group (GB); `-1` means infinite capacity, no capacity eviction triggered |
| used_percentage | percentage of quota actually used |
| instance_id | unique identifier of the instance |
| block_size | number of tokens contained in each block |
| bytes_per_token | KV size per token; Python analysis scripts use it together with `block_size` to convert block capacity to GB |
| eviction_policy_type | eviction policy type: lru, random_lru, leaf_aware_lru |

---

## Eviction Policies

### Eviction Policy Interface

```cpp
class EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;
    virtual size_t size() const = 0;
    virtual void OnBlockWritten(BlockEntry *block) = 0;
    virtual void OnNodeWritten(std::vector<BlockEntry *> &blocks) = 0;
    virtual void OnBlockAccessed(BlockEntry *block, int64_t timestamp) = 0;
    virtual std::vector<BlockEntry *> EvictBlocks(size_t num_blocks) = 0;
    virtual std::string name() const = 0;
    virtual void set_name(const std::string &name) = 0;
};
```

### LRU Policy

**Principle**: Maintains a doubly linked list recording the access order of blocks; the most recently accessed block is at the head of the list, and the least recently accessed block is at the tail. During eviction, blocks are removed from the tail of the list.

**Time complexity**:
- `OnBlockAccessed()`: O(1)
- `OnBlockWritten()`: O(1)
- `EvictBlocks()`: O(n)

### RandomLRU Policy

**Principle**: Combines random sampling with the LRU policy; randomly samples a certain proportion of blocks from the cache and selects the least recently accessed block for eviction.

**Time complexity**:
- `OnBlockAccessed()`: O(1)
- `OnBlockWritten()`: O(1)
- `EvictBlocks()`: O(m log m), where m is the number of samples

### LeafAwareLRU Policy

**Principle**: Adds leaf-node awareness on top of LRU, preferentially evicting blocks within leaf nodes to improve cache efficiency.

**Implementation characteristics**:
- Maintains an independent LRU linked list of leaf nodes
- Tracks all blocks within leaf nodes
- During eviction, preferentially selects the least recently accessed block from the leaf-node list

### TTL Semantics Phased Description (V1 / V2)

The current implementation adopts **V1 semantics** (already implemented):

- Only `POLICY_TTL` instances perform "physical cleanup of TTL-expired entries before reads/writes"
- Under non-TTL policies (`lru` / `random_lru` / `leaf_aware_lru`), TTL is treated as nonexistent
  - No pre-TTL physical cleanup is performed
  - No TTL logical expiration judgment is performed either
- Under `POLICY_TTL`, it is guaranteed that expired blocks are cleaned up first before entering the read/write flow
- `default_block_ttl_seconds = 0` means TTL is disabled at the group level:
  - The write path parses TTL as "never expires"
  - If `fallback_on_pressure = false`, no TTL expiration eviction occurs, and no capacity fallback eviction is triggered
  - If `fallback_on_pressure = true`, only under capacity pressure does it fall back to list-tail eviction (behavior equivalent to LRU)
- The expiration cleanup implementation of `POLICY_TTL` has been optimized from "full list scan before every read/write" to "min-heap of minimum expiration times with incremental reclamation":
  - Expiration events are written on write/access
  - During cleanup, only due events are popped, no full scan
  - Version numbers are used for lazy invalidation, avoiding repeated processing of expiration events
  - Significantly reduces replay latency without changing TTL semantics

The subsequent **V2 design** (documentation plan only, not implemented for now):

- Unified semantics: all policies support pre-expiration cleanup
- Fine-grained performance optimization for expiration scanning (e.g. next_expire_ts / min-heap, etc.)
- Achieve cross-policy consistent TTL lifecycle without changing the semantics of capacity eviction policies

---

## Index System

### RadixTreeIndex Overview

**Responsibility**: A data structure based on the prefix tree (Radix Tree), supporting efficient prefix matching queries, and managing cache insertion, query, and eviction.

**Core operations**:
1. `InsertOnly()` - insert blocks only, no query
2. `PrefixQuery()` - prefix matching query
3. `ExportForVisualization()` - export the prefix tree for visualization

### Radix Tree Data Structure

```cpp
struct RadixTreeNode {
    std::vector<std::unique_ptr<BlockEntry>> blocks;  // contiguous block segment
    NodeStat stat;  // node statistics
    RadixTreeNode *parent = nullptr;
    std::unordered_map<int64_t, std::unique_ptr<RadixTreeNode>> children;
    bool isLeaf() const { return children.empty(); }
};

struct NodeStat {
    size_t access_count = 0;
    int64_t last_access_time = 0;
    int64_t ttl = 250000;  // default TTL is 250000 microseconds
};

struct BlockEntry {
    int64_t key;
    LocationStatMap location_map;
    int64_t writing_time = -1;
    int64_t last_access_time = -1;
    size_t access_count = 0;
    RadixTreeNode *owner_node = nullptr;
};
```

### Radix Tree Visualization

Supports exporting the Radix Tree structure for visual analysis, which can display:
- Node access count
- Last access time
- Number of blocks in the node
- Number of cached blocks
- Node hierarchy relationships

---

## Trace Processing

### Trace Type Definitions

**Inheritance relationships**:
```
OptimizerSchemaTrace (base class)
    ├── GetLocationSchemaTrace (read operation)
    ├── RequestSchemaTrace (request-level operation, internally schedules read and delayed write)
    └── WriteCacheSchemaTrace (write operation)
```

### Trace Converters

**Supported formats**:
- **Publisher Log**: converts KVCacheManager Event Publisher logs, distinguishing read and write requests
- **Qwen Bailian**: converts the Qwen Bailian dataset format, outputting read/write separated traces

**Conversion flow**:
1. Select the converter based on the configuration file
2. Parse the log file and convert to standard Trace; a standard `get` must carry `input_len`
3. Sort traces by timestamp
4. Assign unique Trace IDs

Standard traces accept JSONL with explicit `type=get/write/request`. `request` is used for scenarios where externally only request-level records exist; the optimizer schedules delayed writes internally according to `trace_replay.write_delay_ns`. `get/request.keys` can only contain complete block keys; trailing tokens that do not fill a block are not written into `keys`, but are still counted into the token hit-rate denominator via `input_len`. Inputs missing `input_len`, with illegal timestamps, or with `keys` exceeding `input_len / block_size` will fail before replay.

---

## Result Analysis

### Result Structure

```cpp
struct ReadRecord {
    int64_t timestamp_ns;
    // local = existing local hits brought in by trace block_mask; remote = hits from the optimizer simulation layer
    size_t remote_read_blocks;
    size_t remote_hit_blocks;
    size_t local_read_blocks;
    size_t local_hit_blocks;
    size_t current_cache_blocks;
    size_t input_tokens;
    size_t block_size_tokens;
    std::vector<size_t> per_tier_hit_blocks;
    std::vector<std::string> tier_names;
    std::vector<size_t> per_tier_blocks;
    std::vector<size_t> blocks_per_instance;
    std::string trace_id;
};
```

### CSV Output Format

**File name**: `{instance_id}_hit_rates.csv`

**Main columns**:
- `TimestampNs` - timestamp (nanoseconds)
- `CachedBlocks` - number of cached blocks for the instance corresponding to the current CSV
- `CachedBlocksAllInstances` - total number of cached blocks across all instances within the same optimizer process
- `ReadBlocks` / `HitBlocks` - number of blocks read and hit by the current request
- `LocalHitBlocks` / `RemoteHitBlocks` - diagnostic fields: existing local hits brought in by trace `block_mask` / hits from the optimizer simulation layer
- `InputTokens` / `HitTokens` - input token count and hit token count of the current request
- `HitRate` - current token hit rate, `HitTokens / InputTokens`
- `LocalHitTokens` / `RemoteHitTokens` - diagnostic fields: local / optimizer simulation layer hit token count
- `AccReadBlocks` / `AccHitBlocks` - cumulative read and hit block count
- `AccHitRate` - cumulative token hit rate, `AccHitTokens / AccInputTokens`
- `AccLocalHitRate` / `AccRemoteHitRate` - diagnostic fields, not used as the main metric of standard analysis
- `Tier<N>(name)_HitTokens` / `Tier<N>(name)_HitRate` / `AccTier<N>(name)_HitRate` - tiered hit token metrics

Standard analysis directly computes the overall `HitRate` from request input, and does not treat local/remote as independent conclusion dimensions. local/remote are only used for compatibility when the optimizer acts as a standalone L3 simulation combined with HiSim, or when directly analyzing KVCacheManager event logs with existing local hit information.

---

## Visual Analysis

### 1. Hit Rate Over Time Chart

**Script**: `optimizer_run.py --draw-chart`; the plotting implementation is in `plot/hit_rate_plot.py`

**Function**: Draws multi-instance cache analysis charts, showing the sum of storage capacity across all instances and their respective hit rates over time.

**How to run**:
```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json \
    --draw-chart
```

**Output**:
- `{output_result_path}/timeseries/multi_instance_cache_analysis.png`
- Tiered configurations additionally output `{output_result_path}/timeseries/per_tier_timeseries.png`

**Chart content**:
- Top chart: cumulative hit rate over time
- Bottom chart: current trace hit rate over time

### 2. Radix Tree Visualization

**Script**: `export_tree.py`; the plotting implementation is in `plot/radix_tree_plot.py`

**Function**: Exports and visualizes the prefix tree structure, counting and displaying hot nodes and the prefix paths of their associated nodes.

**How to run**:
```bash
bazel run //kv_cache_manager/optimizer/analysis/script:export_tree -- \
    -c /path/to/config.json
```

**Output**:
- `{output_result_path}/radix_tree/{instance_id}_radix_tree.json` - Radix Tree export data
- `{output_result_path}/radix_tree/{instance_id}_radix_tree.png` - Radix Tree visualization chart
- `{output_result_path}/radix_tree/{instance_id}_hot_paths.png` - hot path chart (when `--show-hot-paths` is passed)

**Visualization content**:
- Node access count
- Last access time
- Number of blocks in the node
- Number of cached blocks
- Node hierarchy relationships

### 3. Pareto Capacity-Hit-Rate Curve Analysis

**Script**: `tradeoff.py`

**Function**: Replays the optimizer under different capacity configurations and draws the Pareto curve of capacity versus token hit rate. When `--eviction-policies` is not given, the policy in the config file is used; when multiple policies are given, a multi-policy comparison chart is generated.

> **Scope of applicability**: Trade-off analysis only applies to non-tiered mode. In tiered mode (`tier_strategy.hierarchical_eviction_enabled=true`), the capacity scan only modifies `quota_capacity` and does not affect each tier's independent `storages[i].capacity`, so it cannot produce meaningful capacity-performance tradeoff results.

**Execution flow**:

1. Run the full trace with infinite-capacity warmup. In implementation, `quota_capacity` is written as `-1`, and the C++ optimizer treats negative capacity as infinite capacity.
2. Read the theoretical hit rate and maximum cache block count from the warmup.
3. Generate up to `--num-points` capacity points based on the maximum cache block count with an exponential distribution, and filter out overly small points with `--min-capacity-ratio`.
4. Run the optimizer for each capacity point; stop scanning larger capacities after reaching 99% of the theoretical hit rate.
5. When plotting, add the `(0 GB, 0%)` starting point and only keep the rising segment of hit rate. Descending points are removed from the chart, and `Drop descending Pareto point ...` is printed to stdout; the original CSV is retained.

**How to run**:
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
- `--hit-rate-type` - hit-rate type; standard analysis uses total; local/remote/all are only diagnostic splits (default total)
- `--max-workers` - maximum number of parallel execution threads (default 4)
- `--plot-title` - override the chart title

**Output**: `{output_result_path}/pareto/pareto_curve_{hit_rate_type}.png`

### 4. Multi-Policy Comparison Analysis

**Script**: `tradeoff.py --eviction-policies ...`

**Function**: Compares the performance of multiple eviction policies under different capacity configurations. All instances use the same given policy for one round of replay; each policy warms up independently, computes its theoretical hit rate independently, and stops early independently.

> **Scope of applicability**: Same as single-policy Trade-off analysis, only applies to non-tiered mode.

**How to run**:
```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff -- \
    -c /path/to/config.json \
    --eviction-policies lru random_lru leaf_aware_lru \
    --num-points 30 \
    --hit-rate-type total \
    --max-workers 4
```

**Parameter description**:
- `-c, --config` - configuration file path
- `--eviction-policies` - list of eviction policies to compare (default lru random_lru leaf_aware_lru)
- `--num-points` - maximum number of capacity sampling points to run per policy (default 30); stops early after reaching 99% of the theoretical hit rate
- `--hit-rate-type` - hit-rate type; standard analysis uses total; local/remote/all are only diagnostic splits (default total)
- `--max-workers` - maximum number of parallel execution threads (default 4)

**Output**: `{output_result_path}/pareto/multi_policy_{hit_rate_type}.png`

Chart rules: the X axis is `Capacity (GB)`, the Y axis is `HitRate (%)`; the 95%/99% theoretical hit rates are marked with capacity and hit rate after linear interpolation between adjacent sweep points. `--skip-run` only plots from existing `csv_results` without re-warming up, so the theoretical 95%/99% annotations are not generated.

---

## Usage Instructions

### Build

```bash
bazel build //kv_cache_manager/optimizer:optimizer_main
```

### Run the Optimizer

**Method: run the binary directly**

```bash
bazel run //kv_cache_manager/optimizer:optimizer_main -- /path/to/config.json
```

### Offline LiteHit Multi-Capacity Analysis

A standalone entry point `lite_hit_main`, parallel to and non-interfering with the replay above. It replays the read requests of a standard trace into the LiteHit core in arrival order, producing **capacity-independent** per-request facts CSV (`litehit_facts.csv`, written to `.tmp` first then atomically renamed); the hit rate is not computed during replay, but is obtained by a second CLI `lite_hit_facts_query_main` projecting onto the facts with an arbitrary capacity list (JSONL per-request results + per-instance / overall summary). The same facts can be repeatedly queried with different capacities without replaying the trace.

The trace reuses the shared `StandardTraceLoader`: the same standard-format trace can be fed to both liteHit offline analysis and directly to replay. Replay is fixed to streaming (`StreamFromFile`, memory O(1)), requiring the trace to be ordered by `timestamp_ns`. **Write events are recognized and ignored**: when a read request is submitted, all blocks are treated as written back (write-back itself is a touch), so split `write` traces have no side effect on liteHit — they are only meaningful for the replay path.

```bash
bazel build //kv_cache_manager/optimizer:lite_hit_main //kv_cache_manager/optimizer:lite_hit_facts_query_main
bazel run //kv_cache_manager/optimizer:lite_hit_main -- /path/to/lite_hit_config.json
bazel run //kv_cache_manager/optimizer:lite_hit_facts_query_main -- <facts_csv> <capacity_gb_list> <output_jsonl>
```

The config `OptimizerLiteHitConfig` reuses the online `OptimizerInstanceGroup` / `OptimizerInstanceInfo` objects (full-attention instances require `linear_step=0`):

| Field | Meaning |
|------|------|
| trace_file_path | standard-format trace file path (shares the same trace as replay) |
| output_result_path | facts CSV output directory |
| instance_groups | list of online `OptimizerInstanceGroup` (including eviction_policy / enable_prefix_hash, etc.) |
| instances | list of online `OptimizerInstanceInfo` (including block_size / linear_step / location spec, etc.) |
| block_size | token granularity of the trace (default 256); instance block_size must be an integer multiple of it (re-blocking only coarsens) |
| override_instance_id | route all requests to the specified instance (must be in instances) |
| fanout_all_instances | broadcast each request to all configured instances (mutually exclusive with override), used to compare multiple block_size in one replay |
| pipeline_worker_count | replay pipeline parallelism (default 1) |

Algorithm details, facts field definitions, projection metrics, and the verification checklist are in [liteHit/README.md](../liteHit/README.md).

### Python Interface

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
```

### Output Files

After running, the following is generated in the directory specified by `output_result_path`:
- `{instance_id}_hit_rates.csv` - hit-rate data for each instance

---

## Extension Guide

### Adding a New Eviction Policy

1. Create a new policy file in `kv_cache_manager/optimizer/eviction_policy/`, inheriting the `EvictionPolicy` base class
2. Add a new policy type enum value in `kv_cache_manager/optimizer/config/types.h`
3. Add a new parameter type in `kv_cache_manager/optimizer/config/eviction_config.h`
4. Add new policy creation logic in `kv_cache_manager/optimizer/eviction_policy/policy_factory.cc`
5. Add the new source file in the BUILD file

### Adding a New Trace Converter

1. Create a new converter file in `kv_cache_manager/optimizer/trace_converter/`, inheriting the `BaseConverter` base class
2. Add a new trace type enum value in `kv_cache_manager/optimizer/config/types.h`
3. Add new converter creation logic in `kv_cache_manager/optimizer/trace_converter/converter_factory.cc`
4. Add the new source file in the BUILD file

### Adding a New Analysis Metric

1. Add a new statistics field in `kv_cache_manager/optimizer/analysis/result_structure.h`
2. Add new analysis logic in `kv_cache_manager/optimizer/analysis/result_analysis.cc`
3. Implement a new export function to output custom metrics

---

## Appendix

### File Index

**Core managers**:
- optimizer_manager.h/cc - main coordinator
- optimizer_runner.h/cc - trace executor
- eviction_manager.h/cc - eviction manager
- indexer_manager.h/cc - indexer manager
- optimizer_loader.h/cc - trace loader

**Index layer**:
- radix_tree_index.h/cc - Radix tree index

**Eviction policies**:
- base.h - policy base class
- common_structure.h - common data structures
- lru.h/cc - LRU policy
- random_lru.h/cc - RandomLRU policy
- leaf_aware_lru.h/cc - LeafAwareLRU policy
- policy_factory.h/cc - policy factory

**Trace conversion**:
- optimizer_schema_trace.h - trace definitions
- base_converter.h - converter base class
- publisher_log_converter.h/cc - Publisher Log converter
- qwen_bailian_converter.h/cc - Qwen Bailian converter
- converter_factory.h/cc - converter factory
- trace_util.h - trace utilities

**Configuration**:
- optimizer_config.h/cc - top-level config
- replay_instance_group_config.h/cc - trace replay/tier simulation instance group config
- replay_instance_config.h/cc - trace replay instance config
- tier_config.h/cc - storage tier config
- eviction_config.h - eviction policy parameters
- types.h - type definitions
- optimizer_config_loader.h/cc - config loader

**Analysis**:
- result_structure.h - result structure definitions
- result_analysis.h/cc - hit-rate analysis
- script/run/optimizer_run.py - single replay + optional time-series chart
- script/run/tradeoff.py - Pareto curve, unified entry for single/multi-policy
- script/run/export_tree.py - Radix Tree export + visualization
- script/run/analyze_lifecycle.py - block lifecycle analysis
- script/run/multi_instance_replay.py - multi-instance parallel replay + aggregation
- script/plot/hit_rate_plot.py - hit-rate time-series chart
- script/plot/radix_tree_plot.py - Radix Tree plotting
- script/utils/optimizer_runner.py - optimizer run wrapper

**Python bindings**:
- pybind/py_optimizer_binding.cc - Python interface

### Glossary

| Term | Description |
|------|------|
| Eviction Policy | The algorithm that selects which blocks are removed when the cache is full |
| LRU | Least Recently Used algorithm |
| RandomLRU | A hybrid algorithm combining random sampling and LRU |
| LeafAwareLRU | A leaf-node-aware LRU algorithm |
| Radix Tree | A tree data structure for efficient prefix matching |
| Trace | Data recording the sequence of system operations |
| Instance | An independent instance of the cache system |
| Instance Group | A collection of instances sharing resources |
| Block | The basic unit of the cache |
| Hit Rate | The ratio of cache hits to total accesses |
| Prefix Match | Finding keys that share the same prefix |
| Read/Write Separation | Processing read and write operations separately |
| Trade-off Curve | The tradeoff curve between capacity and hit rate |



---
