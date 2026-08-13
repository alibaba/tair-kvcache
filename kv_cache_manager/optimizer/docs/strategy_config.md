# Optimizer Standard Strategy Configuration Guide

> [中文](strategy_config_zh.md) | English

This document defines the strategy configuration, multi-instance replay entry point, and hit-rate metrics of the standard edition optimizer. Subsequent new experimental scripts should reuse the field semantics here, avoiding introducing another set of configuration naming in scripts.

## Metric Definitions

The standard edition exposes only one hit-rate metric externally:

```text
HitRate = HitTokens / InputTokens
AccHitRate = AccHitTokens / AccInputTokens
```

Related conventions:

- `HitRate`, `AccHitRate`, `LocalHitRate`, `RemoteHitRate`, and `Tier*_HitRate` are all token hit rates.
- The standard hit rate is computed only by token, but the CSV retains block read/hit counts for verifying read amplification, hit scale, and capacity behavior.
- The standard analysis does not treat local/remote as independent conclusion dimensions. `local` only denotes existing local hit blocks brought in by the trace `block_mask`, e.g. when the optimizer acts as a standalone L3 simulation combined with HiSim, or when directly analyzing KVCacheManager event logs, the local hit block keys already contained in the log; `remote` denotes hits contributed by the optimizer simulation layer. The current standard report directly computes the overall `HitRate` from request input, without relying on the local/remote split.
- A standard `get` trace must contain `input_len`; `InputTokens` directly uses `input_len`.
- Old `get` traces missing `input_len` are no longer compatible. Logs from other sources must first be converted to the optimizer schema.
- `keys` only contains complete block keys; trailing tokens that do not fill a block are not written into `keys`, but are still counted into `InputTokens`. For example, when `block_size=16` and `input_len=33`, `keys` contains at most 2 complete blocks, and the trailing 1 token only enters the denominator.

Core columns of the standard `*_hit_rates.csv`:

| Column | Description |
|---|---|
| `TimestampNs` | trace timestamp, in ns |
| `CachedBlocks` | number of cached blocks for the instance corresponding to the current CSV |
| `CachedBlocksAllInstances` | total number of cached blocks across all instances within the same optimizer process |
| `ReadBlocks` / `HitBlocks` | number of blocks read / hit by the current request |
| `LocalHitBlocks` / `RemoteHitBlocks` | diagnostic fields: existing local hits brought in by trace `block_mask` / hits from the optimizer simulation layer |
| `InputTokens` / `HitTokens` | input token count / hit token count of the current request |
| `LocalHitTokens` / `RemoteHitTokens` | diagnostic fields: local / optimizer simulation layer hit token count |
| `HitRate` | overall token hit rate of the current request |
| `LocalHitRate` / `RemoteHitRate` | diagnostic fields, not used as the main metric of standard analysis |
| `AccReadBlocks` / `AccHitBlocks` | cumulative read / hit block count |
| `AccInputTokens` / `AccHitTokens` | cumulative input token count / cumulative hit token count |
| `AccLocalHitTokens` / `AccRemoteHitTokens` | diagnostic fields: cumulative local / optimizer simulation layer hit token count |
| `AccHitRate` | cumulative overall token hit rate |
| `AccLocalHitRate` / `AccRemoteHitRate` | diagnostic fields, not used as the main metric of standard analysis |
| `AccWriteBlocks` | cumulative written block count up to the current time |
| `Tier<N>(name)_HitTokens` | hit token count of the current request in a certain tier |
| `Tier<N>(name)_HitRate` / `AccTier<N>(name)_HitRate` | current / cumulative tier token hit rate |
| `Tier<N>(name)_BlockNum` | number of cached blocks in the current tier |

## Top-Level Configuration

```json
{
  "trace_file_path": "/path/to/optimizer_trace.jsonl",
  "output_result_path": "/path/to/output",
  "eviction_params": {
    "eviction_mode": 3,
    "eviction_batch_size_per_instance": 100
  },
  "trace_replay": {
    "write_delay_ns": 1
  },
  "instance_groups": []
}
```

`output_result_path` is the result output directory for all config-driven entry points, including `optimizer_main`, `optimizer_run`, `tradeoff`, and `export_tree`. `export_tree` writes to `radix_tree/` under that directory. `multi_instance_replay` does not read the full optimizer config and requires the output directory to be explicitly specified via `--output-dir`.

### eviction_params

| Field | Type | Default | Description |
|---|---:|---:|---|
| `eviction_mode` | int | required | `1`=`GROUP_ROUGH`, `2`=`INSTANCE_ROUGH`, `3`=`INSTANCE_PRECISE` |
| `eviction_batch_size_per_instance` | int | required | maximum number of blocks evicted per instance per round. Must be greater than 0 in rough mode |

It is recommended to use `eviction_mode=3` for standard experiments, because it truncates the last eviction round by the remaining excess capacity, making capacity points more stable.

### trace_replay

| Field | Type | Default | Description |
|---|---:|---:|---|
| `write_delay_ns` | int64 | `1` | internal write delay for `type=request` traces. During replay, the read is executed first at `timestamp_ns`, then the write is scheduled at `timestamp_ns + write_delay_ns`. Must be greater than 0 |

## Standard Trace Schema

The optimizer replay input only accepts JSONL, with one standard trace per line. When fields are incomplete it fails directly, without doing old-format inference.

`timestamp_ns`, `get.input_len`, `block_mask` offset, and `ttl_us` must fall within the `int64_t` range:

```text
[-9223372036854775808, 9223372036854775807]
```

`keys` supports JSON signed/unsigned integers. Unsigned 64-bit values exceeding `INT64_MAX=9223372036854775807` are stably mapped to the internal `int64_t` via two's complement, e.g. `9223372036854775808 -> -9223372036854775808`, `18446744073709551615 -> -1`.

Get trace:

```json
{
  "type": "get",
  "instance_id": "instance-a",
  "trace_id": "trace_instance-a_1000",
  "timestamp_ns": 1000,
  "keys": [101, 102, 103],
  "input_len": 33,
  "query_type": "prefix_match",
  "block_mask": [],
  "sw_size": 0,
  "location_spec_names": []
}
```

Request trace:

When the external trace only has request-level records and is not explicitly split into read and write rows, use `request`. It is equivalent to:

- executing a `get` at `timestamp_ns`
- executing a `write` at `timestamp_ns + trace_replay.write_delay_ns`

```json
{
  "type": "request",
  "instance_id": "instance-a",
  "trace_id": "trace_instance-a_1000",
  "timestamp_ns": 1000,
  "keys": [101, 102, 103],
  "input_len": 33,
  "query_type": "prefix_match",
  "block_mask": [],
  "sw_size": 0,
  "location_spec_names": [],
  "ttl_us": 0
}
```

Write trace:

```json
{
  "type": "write",
  "instance_id": "instance-a",
  "trace_id": "trace_instance-a_1001",
  "timestamp_ns": 1001,
  "keys": [101, 102, 103],
  "ttl_us": 0
}
```

Required fields:

| Field | Type | Description |
|---|---|---|
| `type` | string | can only be `get`, `write`, or `request` |
| `instance_id` | string | non-empty, must match an instance in the config |
| `timestamp_ns` | int64 | ns timestamp, must be a positive integer; `timestamp_us` is no longer accepted |
| `keys` | int64/uint64 array | list of block keys, may be empty |

Optional common fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `trace_id` | string | empty string | request identifier, used for debugging and template analysis |

`get`-specific fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `input_len` | int64 | required | input token count, must be a positive integer; `InputTokens` directly uses this value |
| `query_type` | string | `prefix_match` | currently only `prefix_match` is supported; other values are skipped |
| `block_mask` | bool array or non-negative int64 | empty array | local hit blocks already known by the trace. The array form marks per block; the integer form denotes the number of local hit blocks from the prefix start |
| `sw_size` | int32 | 0 | sliding window parameter, usually 0 for the current standard prefix matching |
| `location_spec_names` | string array | empty array | compatibility field, usually empty for standard analysis |

`write`-specific fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `ttl_us` | int64 | 0 | request-level TTL, in microseconds; `0` uses the group default TTL, `-1` denotes disabling TTL |

`request` fields:

`request` reuses all fields of `get`, and additionally supports `write.ttl_us`. The write internally generated by `request` uses the same set of `keys` and that `ttl_us`.

`write` only reads `type`, `instance_id`, `trace_id`, `timestamp_ns`, `keys`, and `ttl_us`; other fields including `input_len` and `block_mask` are ignored. `block_mask` is only used to mark local hit blocks already known by the trace, and is not the grouping basis of the standard report. When directly analyzing request input, an empty array can usually be passed, in which case the overall `HitRate` is still computed as `HitTokens / InputTokens`.

Old-format or invalid input will fail, including:

- Missing `type`, `instance_id`, `timestamp_ns`, `keys`, or `get` / `request` traces missing `input_len`.
- `get/request.input_len <= 0`, `timestamp_ns <= 0`, or `keys` not being an array.
- `get/request.keys.size() > input_len / block_size`. The `keys` of a standard trace can only contain complete blocks; writing a trailing key that does not fill a block into the trace is not allowed.
- Using `timestamp_us` but without `timestamp_ns`.
- Non-integers present in `keys`.
- Non-bool values present in the `block_mask` array, or offset being negative / exceeding `INT64_MAX`.
- Legacy dialog-style traces that only have `query_type` / `block_mask` / decode metadata but no explicit `type=get/write`.

## Instance Group Configuration

```json
{
  "group_name": "instance-a",
  "quota_capacity": 178,
  "used_percentage": 1.0,
  "tier_strategy": {
    "hierarchical_eviction_enabled": true,
    "write_mode": "write_through",
    "access_propagation_enabled": true,
    "promote_enabled": true,
    "selective_write_threshold": 2,
    "tier_flows": []
  },
  "default_block_ttl_seconds": 0,
  "ttl_refresh_on_read": true,
  "storages": [],
  "instances": []
}
```

| Field | Type | Default | Standard Semantics |
|---|---:|---:|---|
| `group_name` | string | required | instance group name. In multi-instance replay usually equal to the instance's `instance_id` |
| `quota_capacity` | number | required | total group capacity, in GB. Non-tiered mode evicts by this field; `-1` means infinite capacity, no capacity eviction triggered, mainly used for global pooled theoretical hits and Pareto warmup |
| `used_percentage` | number | required | capacity watermark ratio; the actual threshold is capacity × used_percentage |
| `tier_strategy` | object | required | multi-tier read/write policy package, see table below |
| `default_block_ttl_seconds` | int | `0` | group default TTL in seconds; `0` means TTL disabled at the group level |
| `ttl_refresh_on_read` | bool | `true` | whether a read hit refreshes the TTL anchor under the TTL policy |
| `storages` | array | required | tier list, sorted by `priority` ascending |
| `instances` | array | required | list of optimizer instances under this group |

### tier_strategy

The top-level fields of `tier_strategy` are the default policy for all adjacent tier edges; `tier_flows` is only used to override specific adjacent edges. If all inter-tier policies are consistent, you can configure only the top-level fields and omit `tier_flows`; they are not two sets of duplicate configuration.

| Field | Type | Default | Standard Semantics |
|---|---:|---:|---|
| `hierarchical_eviction_enabled` | bool | required | whether to enable tiered capacity and tiered eviction; when `false`, all tiers share one `shared` policy and the `quota_capacity` quota |
| `write_mode` | string | `write_through` | multi-tier write and inter-tier flow policy |
| `access_propagation_enabled` | bool | `true` | when a read hits a higher-priority tier, whether to refresh the access time of subsequent tiers holding copies; `false` means only the hit tier is refreshed |
| `promote_enabled` | bool | `true` | whether to copy back layer by layer to the higher-priority tiers passed through after a lower-tier hit |
| `selective_write_threshold` | int | `2` | under `write_through_selective`, copy to the next tier after the hit tier's access count reaches this threshold; must be a positive integer |
| `tier_flows` | array | `[]` | policy overrides for adjacent tier edges. Unoverridden edges inherit the default policy of `tier_strategy` |

### write_mode

| Value | Write Behavior | Eviction Behavior | Applicable Scenario |
|---|---|---|---|
| `write_through` | writes land in all tiers simultaneously | each tier evicts independently by its own capacity | baseline, full replicas, multi-tier independent hit-rate analysis |
| `cascading` | writes land only in tier 0 | blocks evicted from tier i are demoted to tier i+1; discarded after eviction from the last tier | HBM→DRAM→SSD progressive sinking |
| `write_through_selective` | initially lands only in tier 0 | copy to the next tier after the hit tier's access count reaches `tier_strategy.selective_write_threshold` | control lower-tier write amplification, only let hot blocks sink |

`tier_strategy.write_mode` only accepts the three values in the table above; other values cause config parsing to fail.
`access_propagation_enabled` is not a kind of `write_mode`, but an independent switch for whether to refresh the access time of lower-tier copies after a read hit.

### tier_flows

`tier_flows` is used to override the read/write flow policy between adjacent tiers. Each flow must reference two adjacent `unique_name`s in `storages`; cross-tier skipping is not supported, and configuring the same edge twice is not allowed.
At config load time, edges are validated after sorting by `storages[*].priority`. Unknown tiers, non-adjacent edges, duplicate edges, duplicate `unique_name`s, or duplicate `priority`s all directly raise an error and refuse to load.

```json
{
  "tier_strategy": {
    "hierarchical_eviction_enabled": true,
    "write_mode": "write_through",
    "access_propagation_enabled": true,
    "promote_enabled": true,
    "selective_write_threshold": 2,
    "tier_flows": [
      {
        "from_tier": "hbm",
        "to_tier": "dram",
        "write_mode": "write_through",
        "access_propagation_enabled": false
      },
      {
        "from_tier": "dram",
        "to_tier": "ssd",
        "write_mode": "cascading",
        "promote_enabled": false
      }
    ]
  }
}
```

Fields of a single flow:

| Field | Type | Default | Standard Semantics |
|---|---:|---:|---|
| `from_tier` | string | required | the higher-priority tier name of the edge, must equal some `storages[i].unique_name` |
| `to_tier` | string | required | the lower-priority tier name of the edge, must equal the adjacent `storages[i+1].unique_name` |
| `write_mode` | string | inherit default | the write/eviction sinking policy of this edge |
| `access_propagation_enabled` | bool | inherit default | after a hit on the higher tier, whether to refresh the lower-tier access time across this edge |
| `promote_enabled` | bool | inherit default | after a lower-tier hit, whether to allow backfilling to the higher tier across this edge |
| `selective_write_threshold` | int | inherit default | the down-write threshold when this edge uses `write_through_selective` |

### access_propagation_enabled

This switch is orthogonal to `tier_strategy.write_mode`:

- `true`: when a block has copies in multiple tiers, after a read hits the highest-priority tier, the access time of subsequent copy tiers is also refreshed. This is the default behavior.
- `false`: only the actually hit highest-priority tier is refreshed; the access time of lower-tier copies is not refreshed. Suitable for evaluating the independent lower-tier hot/cold decay in multi-copy scenarios after write-through or cascading/promote.

When `promote_enabled=true`, a lower-priority tier hit triggers layer-by-layer copying to higher-priority tiers. For example, an L3 hit fills in L2 and L1, an L2 hit only fills in L1, and there is no extra writing to even lower tiers. The copy action goes through capacity checks and may immediately trigger eviction of the corresponding tier.

## Storage Configuration

```json
{
  "unique_name": "hbm",
  "storage_type": "pace",
  "band_width_mbps": 20000,
  "priority": 0,
  "capacity": 50
}
```

| Field | Type | Description |
|---|---:|---|
| `unique_name` | string | tier name, which enters the `Tier<N>(name)_*` columns of the CSV |
| `storage_type` | string | storage type label, currently mainly used for config records |
| `band_width_mbps` | number | bandwidth label, currently mainly used for analysis records |
| `priority` | int | tier priority; smaller means closer to the compute side |
| `capacity` | number | tier capacity, in GB; `-1` means infinite capacity for that tier, no capacity eviction triggered for that tier |

## Instance Configuration

```json
{
  "instance_id": "instance-a",
  "block_size": 16,
  "bytes_per_token": 512,
  "eviction_policy_type": "lru",
  "eviction_policy_params": {}
}
```

| Field | Type | Description |
|---|---:|---|
| `instance_id` | string | instance ID in the trace, must match the `instance_id` within trace rows |
| `block_size` | int | number of tokens per block. Token hit rate uses it to convert hit blocks to hit tokens |
| `bytes_per_token` | int | KV size per token. `bytes_per_block = block_size * bytes_per_token` |
| `eviction_policy_type` | string | `lru`, `random_lru`, `leaf_aware_lru`, `ttl` |
| `eviction_policy_params` | object | policy parameters, see below |

## eviction_policy_params

### lru / leaf_aware_lru

```json
{
  "sample_rate": 1.0,
  "shard_count": 1,
  "sample_times": 32,
  "eviction_amplification_factor": 1.0
}
```

| Field | Description |
|---|---|
| `sample_rate` | sampling ratio. `1.0` means complete LRU |
| `shard_count` | number of LRU shards |
| `sample_times` | number of samples per round |
| `eviction_amplification_factor` | eviction amplification factor |

### random_lru

```json
{
  "sample_rate": 1.0
}
```

`random_lru` only requires `sample_rate`, used to control the sampling range.

### ttl

```json
{
  "fallback_on_pressure": true
}
```

| Field | Description |
|---|---|
| `fallback_on_pressure=true` | first clear TTL-expired blocks; if capacity still exceeds the limit, fall back to LRU |
| `fallback_on_pressure=false` | pure TTL, only clear expired blocks; capacity pressure does not trigger LRU fallback |

TTL is only executed when `eviction_policy_type="ttl"`. Non-TTL policies ignore the expiration cleanup semantics of `default_block_ttl_seconds` and `ttl_refresh_on_read`.

## Standard Multi-Instance Replay

The standard edition retains `multi_instance_replay` and no longer uses the multi-machine scheduler as the default replay entry point.
The complete script parameters are subject to [analysis/script/README.md](../analysis/script/README.md); here we give a standard replay configuration example and output conventions.
`multi_instance_replay` does not read the full optimizer config; it generates a single-instance config for each pod/instance based on CLI parameters, then runs the optimizer in parallel. The current CLI directly supports two tiers of capacity, L1/L2; when L3 or more complex tier topologies are needed, you need to use a full optimizer config to run a single replay, or extend the config generation logic of this script.

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:multi_instance_replay -- \
  --trace-dir /path/to/instance_traces \
  --trace-glob "*.jsonl" \
  --output-dir /path/to/output \
  --bucket-name bucket-a \
  --l1-capacity 50 \
  --l2-capacity 128 \
  --block-size 16 \
  --bytes-per-token 512 \
  --eviction-policy lru \
  --default-tier-write-mode write_through \
  --max-workers 32
```

Output:

- The output root directory is `--output-dir`.
- `configs/<instance_id>.json`: the generated config for each instance.
- `<instance_id>_hit_rates.csv`: the standard token hit-rate time series for each instance.
- `aggregate/instance_aggregate.csv`: the aggregated result for each instance.
- `aggregate/global_aggregate.csv`: the global result after summarizing all instances.
- `aggregate/global_window_hit_rates.csv`: the window result generated when `--window-ns` or `--window-seconds` is specified.

The aggregated `HitRate` of multi-instance replay is still token hit rate, computed as the sum of `HitTokens` across all instances divided by the sum of `InputTokens`.
`--bucket-name` only writes into the `Bucket` column of the aggregate CSV, used to mark the experiment source; `--trace-glob` and `--recursive` only take effect when using `--trace-dir` to scan input files.
`--default-tier-write-mode`, `--tier-flow-config`, `--enable/disable-tier-access-propagation`, `--enable/disable-promote`, and `--selective-write-threshold` are written into the `tier_strategy` of the generated config, with semantics consistent with the above.
