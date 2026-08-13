# Getting Started with the Optimizer Skill

Use this skill when a user is new to the optimizer and needs to set up the environment from scratch, build it, and run the first replay.

## Required Components

| Component | Purpose | How to get it |
|---|---|---|
| bazelisk | Build the C++ optimizer binaries | Install [bazelisk](https://github.com/bazelbuild/bazelisk); the repo pins the Bazel version via `.bazeliskrc` |
| Python 3 + pip | Run the trace converter and analysis scripts | System-provided |
| trace data | Replay input | Convert external logs to standard JSONL via the trace converter |
| optimizer config | Describes instance groups / instances / capacity / policy | Hand-written JSON, see below |

For full build details (Bazel cache, optional backends `--config=mooncake/vcns`), see [../../../../docs/develop/README.md](../../../../docs/develop/README.md).

## Steps

1. **Set up the environment**: confirm `bazelisk --version` works. All bazel commands must run inside the `github-opensource/` directory.

2. **Build the binaries**:

```bash
# Offline replay main + analysis script entry
bazel build \
    //kv_cache_manager/optimizer:optimizer_main \
    //kv_cache_manager/optimizer/analysis/script:optimizer_run

# LiteHit multi-capacity analysis (optional, full-attention scenarios)
bazel build \
    //kv_cache_manager/optimizer:lite_hit_main \
    //kv_cache_manager/optimizer:lite_hit_facts_query_main
```

3. **Prepare the trace**: use the standalone Python tool to convert external logs to standard JSONL (no bazel needed):

```bash
cd kv_cache_manager/optimizer/tools/trace_converter
pip install -r requirements.txt
python trace_converter.py \
    -i /path/to/your_trace.log \
    -o /path/to/optimizer_trace.jsonl \
    -f publisher_log \
    --mode optimizer
```

Supported formats and fields: [../../tools/trace_converter/README.md](../../tools/trace_converter/README.md).

4. **Write the config**: a minimal usable config needs top-level `trace_file_path` / `output_result_path` / `eviction_params`, plus `instance_groups[]` (where `instances[].instance_id` must match the `instance_id` in the trace). Templates and field semantics: the "Quick Start" section of [../../README.md](../../README.md) and [../../docs/strategy_config.md](../../docs/strategy_config.md).

5. **Run the first replay**:

```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json
```

Results land under `output_result_path` as `{instance_id}_hit_rates.csv`; the last row's `AccHitRate` is the cumulative token hit rate. Add `--draw-chart` for a time-series chart.

6. **Going further**:
   - Exact capacity-independent hit rate for full-attention → [liteHit skill](../liteHit/SKILL.md)
   - Capacity Pareto / multi-policy comparison, RadixTree visualization, lifecycle analysis → [../../analysis/script/README.md](../../analysis/script/README.md)

## Validation

- `bazel build` produces the binaries with no compile errors.
- Every trace line parses as JSON, is ordered by `timestamp_ns`, and `get/request` carry a positive `input_len`.
- Every trace `instance_id` has a corresponding instance in the config.
- The replay produces `*_hit_rates.csv` and the last row's `AccHitRate` is in `[0, 1]`.

## Reply Content

Report:

- the built binaries / targets
- trace and config paths
- output directory and final token hit rate
- any unfinished validation items

Architecture and module overview: [../../docs/optimizer_architecture.md](../../docs/optimizer_architecture.md).
