# LiteHit Usage Skill

Use this skill when the analysis target is a full-attention prefix cache and any of the following holds:

- The capacity assumption changes repeatedly (capacity is projected after the fact, without replaying the trace);
- You need to sweep multiple block sizes over the same trace in a single replay;
- You need an exact LRU prefix hit rate (not sampled, not approximate);
- You need an online real-time hit rate (gRPC/HTTP service).

## Required Trace

LiteHit reuses the optimizer standard trace (JSONL, one record per line). Only read-request fields are needed; `write` events are recognized and ignored (a `get` submission is treated as all blocks written back):

| Field | Requirement |
|---|---|
| `type` | `get` or `request` (`write` rows are ignored) |
| `instance_id` | non-empty; must match an instance in the config (or be routed via `override_instance_id` / `fanout`) |
| `timestamp_ns` | positive integer ns timestamp; the trace must be ordered by it (streaming replay assumption) |
| `keys` | list of complete block keys, in one of two forms (see below) |
| `input_len` | positive integer, original input token count; trailing tokens that do not fill a block stay out of `keys` but count in the denominator |

The key form is one of:

- **Prefix chained key**: `key_j = hash(all tokens of the first j complete blocks of the request)`; two keys are equal iff the entire token prefix is identical;
- **Per-block raw hash**: enable `enable_prefix_hash` on the Instance Group, and preprocessing converts it into a prefix chained key via a rolling hash.

Hard constraint: `len(keys) <= input_len // block_size`. For trace conversion and validation see [../../tools/trace_converter/README.md](../../tools/trace_converter/README.md); for the full schema see [../../docs/strategy_config.md](../../docs/strategy_config.md).

## Inputs to Confirm

- The trace's native block granularity (config `block_size`, default 256 token/block).
- Each instance's analysis granularity (instance `block_size`, which must be an integer multiple of the trace granularity; only coarsening is allowed).
- Key form: prefix chained key or per-block raw hash (enable `enable_prefix_hash` for the latter).
- Routing: by trace `instance_id`, `override_instance_id` to aggregate into one service view, or `fanout_all_instances` to broadcast and sweep granularity (mutually exclusive with override).
- location spec size (determines `block_bytes`, the anchor for capacity GB → block conversion).
- The capacity list to query (GB; may contain duplicates and 0; negative = infinite; offline does not need it up front).

## Offline Two-Step Flow

```bash
# Step 1: replay to produce facts (config has no capacity)
bazel run //kv_cache_manager/optimizer:lite_hit_main -- /path/to/lite_hit_config.json
# -> <output_result_path>/litehit_facts.csv (atomic publish, fail-fast all-or-nothing)

# Step 2: project arbitrary capacities after the fact, repeatable (arg order: facts_csv output_log capacity_gb...)
bazel run //kv_cache_manager/optimizer:lite_hit_facts_query_main -- \
  /path/to/litehit_facts.csv /path/to/result.jsonl 10 50 100 -1
```

Multi block-size sweep: configure N instances with different `block_size` plus `fanout_all_instances: true`; a single replay produces independent facts per lane; the query summary is grouped by instance (one row per instance + one total row).

## Online Service

```bash
bazel run //kv_cache_manager/optimizer:online_optimizer_server_main -- /path/to/server_config.json
```

On the engine side, use `client/` (Python gRPC/HTTP SDK): create group (capacity slots, `enable_prefix_hash`) → register instances → per-request TraceQuery → `ListInstances` for the cumulative hit rate.

## Validation

- The trace is ordered by `timestamp_ns`; `get/request` carry a positive `input_len`.
- facts row count = number of valid read requests × (number of lanes under fanout).
- query summary cumulative hit rate = `Σ hit_tokens / Σ input_tokens` (token metric, trailing tokens in the denominator).
- Theoretical upper bound (infinite-capacity projection) ≥ all finite-capacity results.

## Reply Content

Report:

- facts CSV path and row count
- cumulative token hit rate per capacity (and per instance/block size)
- theoretical upper-bound hit rate
- any unfinished validation items

Semantic details: [../../liteHit/README.md](../../liteHit/README.md).
