# LLM Multi-Instance Prefill Simulation

## When to use

- Evaluate multi-instance prefill scheduling (TTFT/throughput) under different routing strategies
- Analyze KV cache hit rate with real prefix matching (via tair-kvcache Optimizer)
- Compare iteration-level vs request-level scheduling
- Train and integrate new time predictors for different models/hardware
- Run what-if analysis: node count, routing policy, cache write strategy, P2P

## One-click simulation script

```bash
cd /sgl-workspace/claude_workspace/schedule_simulator

# Basic: random requests, request-level, 5 P instances
python3 scripts/run_simulation.py \
  --num-prompts 200 --request-level --ms-per-token 0.1 --num-p-instances 5

# With real data + hierarchical cache
python3 scripts/run_simulation.py \
  --dataset path/to/enriched.jsonl \
  --request-level --predictor-pkl path/to/predictor.pkl \
  --num-p-instances 10 --enable-hierarchical --routing round_robin \
  --output-dir results/

# Full parameter list
python3 scripts/run_simulation.py --help
```

### Key script parameters

| Parameter | Default | Description |
|---|---|---|
| `--dataset` | None | JSONL input (with optional `block_ids` field) |
| `--num-prompts` | 100 | Number of requests |
| `--num-p-instances` | 1 | Number of prefill instances |
| `--routing` | round_robin | random/round_robin/power_of_two/cache_aware |
| `--request-level` | False | Enable request-level scheduling |
| `--predictor-pkl` | None | Lookup table predictor (.pkl) |
| `--ms-per-token` | None | Constant predictor (ms per uncached token) |
| `--enable-hierarchical` | False | Enable Optimizer cache integration |
| `--enable-p2p` | True | Enable P2P cross-instance cache reads |
| `--write-policy` | write_through | write_through/write_back/write_through_selective |
| `--output-dir` | ./sim_results | Output directory for results |

### Output files

- `simulation_summary.json` — TTFT/TPOT/throughput/queue_wait + cache hit ratios
- `per_request.csv` — Per-request: ttft_ms, queue_wait_ms, engine/peer/pool_hit
- `per_iteration.csv` — Per-iteration: pod, latency_ms, batch composition

## How to integrate a new time predictor

### Option A: Request-level predictor (for models without iteration-level data)

```python
from schedule_simulator.infer_time_predictor import RequestLevelTimePredictor

# Method 1: Lookup table from pkl
predictor = RequestLevelTimePredictor(lookup_table_path="/path/to/model.pkl")
# pkl format: {"train_table": {(lo, hi): median_ms, ...}, "bins": [...]}

# Method 2: Constant rate
predictor = RequestLevelTimePredictor(constant_ms_per_token=0.1)

# Method 3: Custom function
def my_predict(uncached_tokens, cached_tokens):
    return (uncached_tokens * 0.05 + 10) / 1000.0  # return seconds
predictor = RequestLevelTimePredictor(predict_fn=my_predict)
```

Then pass to runner:
```python
runner = DisaggBenchmarkRunner(
    ...,
    p_scheduler_config=SchedulerConfig("model", scenario="disagg_prefill",
                                        request_level_scheduling=True),
    infer_time_predictor=predictor,
)
```

### Option B: Iteration-level predictor (for models with batch-level data)

Inherit `InferTimePredictor` and implement `predict_infer_time(batch)`:

```python
from schedule_simulator.infer_time_predictor.base import InferTimePredictor, ScheduleBatch

class MyIterPredictor(InferTimePredictor):
    def __init__(self, model, hw, config, my_model_path=None):
        super().__init__(model, hw, config)
        # Load your model

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        # batch.reqs: list of ScheduleRequest(input_length, past_kv_length)
        # Return latency in SECONDS for this single iteration
        total_tokens = sum(r.input_length for r in batch.reqs)
        return total_tokens * 0.0001  # example
```

Requires model registered in `kunlun_commons` (for `ModelInfo`/`AcceleratorInfo`).

### Option C: Using existing predictors

| Predictor | Requirements | Use case |
|---|---|---|
| `LLMPerfTimePredictor` | Model in kunlun_commons | Theoretical Roofline (iter-level) |
| `StepBenchmarkTimePredictor` | Step benchmark CSV | Curve-fitted from real data (iter-level) |
| `RequestLevelTimePredictor` | Lookup table or function | Request-level, no iter data needed |

## How to train a new predictor

### From request-level data (e.g., `first_latency_ms`)

```python
import joblib, numpy as np

# 1. Load data, filter queue_ms=0, remove outliers
# 2. Build bins by uncached tokens
bins = [(0,256),(256,1024),(1024,4096),...,(200000,999999)]
table = {}
for lo, hi in bins:
    subset = [r["first_latency_ms"] for r in data if lo <= r["uncached"] < hi]
    if len(subset) >= 5:
        table[(lo, hi)] = float(np.median(subset))

# 3. Save
joblib.dump({"train_table": table, "bins": bins, "version": "v1"}, "predictor.pkl")
```

### Key findings from Qwen3.6-Plus training (applicable to other models)

1. **Filter queue_ms=0** — 95%+ requests have no queue; filtering removes noise
2. **cached_input_tokens has <2% effect on compute time** — use uncached as sole feature
3. **Latency grows sub-linearly with uncached** (GPU utilization effect)
4. **CV (noise) is 0.4-0.96 within bins** — irreducible noise from batch concurrency
5. **Best achievable MAPE: ~38-40%** with request-level data (iteration-level data would be ~5%)

## JSONL input format

```json
{"timestamp": 1780722000040, "input_length": 586, "output_length": 1,
 "block_ids": [435730272841023862, 3709909376369622945, ...],
 "instance_id": "pod-name",
 "device_cache_hit_length": 0}
```

- `timestamp` (required): epoch milliseconds
- `input_length` (required): total input tokens
- `output_length` (required): output tokens (use 1 for prefill-only)
- `block_ids` (optional): real block hash IDs for prefix matching
- `instance_id` (optional): original pod assignment
- `device/host/disk_cache_hit_length` (optional): pre-annotated cache hits

## Converting enriched timeline data to JSONL

```python
import io, json, zstandard
dctx = zstandard.ZstdDecompressor()
with open("timeline.jsonl.zst", "rb") as fh:
    with dctx.stream_reader(fh) as sr:
        with open("output.jsonl", "w") as out:
            for line in io.TextIOWrapper(sr, encoding="utf-8"):
                rec = json.loads(line)
                pf = rec.get("prefill")
                if not pf: continue
                block_ids = [int(b, 16) % (2**63) for b in rec.get("input_block_hash_ids", [])]
                out.write(json.dumps({
                    "timestamp": rec["timestamp"] * 1000,
                    "input_length": rec["input_length"],
                    "output_length": 1,
                    "block_ids": block_ids,
                    "instance_id": rec.get("pods", [""])[0],
                }) + "\n")
```

## Architecture overview

```
Input → BenchmarkEmulator → DisaggBenchmarkRunner (routing)
  → SGLangScheduleEmulator ×N
    ├── iteration mode: get_batch → predict_iter → process_result (loop)
    └── request mode: predict_request_time → complete (one-shot)
  → tree_cache (PrefixCache / HiRadixCache / HierarchicalCacheAdapter)
    → HierarchicalReplayManager (C++ pybind)
      ├── OptimizerManager (engine local RadixTree)
      ├── TierGlobalTracker (P2P peer read)
      └── HashStoragePoolManager (shared storage pool)
  → Output: TTFT/throughput/cache_hit + CSV export
```

## Running tests

```bash
cd /sgl-workspace/claude_workspace/schedule_simulator
PYTHONPATH=/sgl-workspace/claude_workspace/tair-kvcache/bazel-bin/kv_cache_manager/optimizer/pybind:$PYTHONPATH \
  python -m pytest tests/ -v
# 123 tests, ~7 seconds
```

## Related documents

| Document | Content |
|---|---|
| `project_overview.md` (= README.md) | Full architecture, quick start, dev guide |
| `all_changes_summary.md` | File-level change list |
| `qwen36_predictor_analysis.md` | Predictor training: 6 attempts + findings |
| `integration_accuracy_findings.md` | Cache hit accuracy validation |
| `schedule_simulator_vs_optimizer_comparison.md` | Two-system overlap/difference analysis |
