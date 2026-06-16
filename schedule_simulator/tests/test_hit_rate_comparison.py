"""
Test: Optimizer standalone vs integrated E2E hit rate comparison.
Uses identical config (pods, routing, capacity, P2P) for both paths.
Measures the impact of batch scheduling on cache hit rate.
"""
import os, sys, json, random
import numpy as np
import pytest

KVCM_SO_DIR = "/sgl-workspace/claude_workspace/tair-kvcache/bazel-bin/kv_cache_manager/optimizer/pybind"
if KVCM_SO_DIR not in sys.path:
    sys.path.insert(0, KVCM_SO_DIR)
try:
    import kvcm_py_optimizer as kvcm
    HAS_KVCM = True
except ImportError:
    HAS_KVCM = False

from schedule_simulator.schedule_emulator.types import *
from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner
from schedule_simulator.schedule_emulator.base import GlobalValues
from schedule_simulator.schedule_emulator.hierarchical_config_builder import build_hierarchical_config
from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter

ENRICHED_INPUT = os.path.join(os.path.dirname(__file__), "assets/glm5_sample/glm5_enriched_input.jsonl")

pytestmark = pytest.mark.skipif(not HAS_KVCM, reason="kvcm_py_optimizer not available")


def _load_enriched(n=100):
    recs = []
    with open(ENRICHED_INPUT) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs[:n]


def _build_config(pods, output_dir, enable_p2p=True):
    return build_hierarchical_config(
        SchedulerConfig("Qwen2.5-3B", hicache_storage_backend="hf3fs", hicache_read_query_type="prefix_match"),
        PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0,
                        memory_read_bandwidth_gb=16.0, memory_capacity_gb=64.0, peer_read_bandwidth_gb=10.0),
        p_instance_ids=pods, output_dir=output_dir,
        storage_pool_capacity_gb=2.0, enable_p2p=enable_p2p,
    )


def _run_standalone(records, pods, enable_p2p=True, tag="standalone"):
    """Run Optimizer standalone with sequential read-then-write per request."""
    cfg = _build_config(pods, "/tmp/hitcmp_%s" % tag, enable_p2p=enable_p2p)
    loader = kvcm.HierarchicalReplayConfigLoader()
    assert loader.load(cfg)
    mgr = kvcm.HierarchicalReplayManager(loader.config())
    assert mgr.Init()

    engine_hit, peer_hit, pool_hit, total_blocks = 0, 0, 0, 0
    per_req = []

    for idx, r in enumerate(records):
        pod = pods[idx % len(pods)]
        block_ids = r["block_ids"]
        ts_ns = int(r["timestamp"] * 1e6)
        input_len = r["input_length"]

        # Drop partial tail blocks (must match E2E behavior)
        block_size = 1
        max_full_blocks = input_len // block_size
        if max_full_blocks == 0 and block_ids:
            max_full_blocks = 1  # backward compatibility
        full_block_ids = block_ids[:max_full_blocks]

        res = mgr.GetCacheLocation(pod, "r%d" % idx, ts_ns, full_block_ids, input_len)
        eh, ph, sh = res.engine_hit_length, res.peer_hit_length, res.storage_pool_hit_length
        engine_hit += eh; peer_hit += ph; pool_hit += sh
        total_blocks += len(full_block_ids)
        per_req.append({"engine": eh, "peer": ph, "pool": sh, "blocks": len(full_block_ids)})
        mgr.WriteCache(pod, "w%d" % idx, ts_ns + 1, full_block_ids)

    return {
        "engine_hit": engine_hit, "peer_hit": peer_hit, "pool_hit": pool_hit,
        "total_hit": engine_hit + peer_hit + pool_hit,
        "total_blocks": total_blocks,
        "hit_ratio": (engine_hit + peer_hit + pool_hit) / max(total_blocks, 1),
        "per_req": per_req,
    }


def _run_e2e(records, pods, enable_p2p=True, tag="e2e"):
    """Run integrated E2E via DisaggBenchmarkRunner."""
    random.seed(42); np.random.seed(42)

    # Write enriched JSONL
    tmp = "/tmp/hitcmp_%s_input.jsonl" % tag
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps({
                "timestamp": r["timestamp"],  # already in ms from enriched data
                "input_length": r["input_length"],
                "output_length": 1,
                "device_cache_hit_length": 0,
                "block_ids": r["block_ids"],
            }) + "\n")

    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(dataset_path=tmp, num_prompts=len(records), disable_tqdm=True),
        p_scheduler_config=SchedulerConfig(
            "Qwen2.5-3B", scenario="disagg_prefill", chunked_prefill_size=8192,
            hicache_storage_backend="hf3fs",
        ),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(
            device="H20", disk_read_bandwidth_gb=2.0,
            memory_read_bandwidth_gb=16.0, memory_capacity_gb=64.0, peer_read_bandwidth_gb=10.0,
        ),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(
            p_policy=RoutingPolicy.ROUND_ROBIN,
            d_policy=RoutingPolicy.ROUND_ROBIN,
            worker_startup_check_interval=0.01,
        ),
        num_p_instance=len(pods), num_d_instance=0,
        enable_hierarchical=True,
        enable_p2p=enable_p2p,
        hierarchical_output_dir="/tmp/hitcmp_%s_out" % tag,
    )
    metrics = runner.run_benchmark_emulation()
    hier = runner.get_hierarchical_metrics()

    return {
        "engine_hit": hier["total_engine_hit_blocks"],
        "peer_hit": hier["total_peer_hit_blocks"],
        "pool_hit": hier["total_pool_hit_blocks"],
        "total_hit": hier["total_engine_hit_blocks"] + hier["total_peer_hit_blocks"] + hier["total_pool_hit_blocks"],
        "total_blocks": hier["total_blocks_queried"],
        "hit_ratio": hier["block_hit_ratio"],
        "ttft_ms": metrics["mean_ttft_ms"],
        "completed": metrics["completed"],
    }


def _print_comparison(sa, e2e, label):
    print("\n" + "=" * 70)
    print("  %s" % label)
    print("=" * 70)
    print("%-25s %15s %15s %10s" % ("Metric", "Standalone", "E2E Integrated", "Diff"))
    print("-" * 70)
    for key in ["engine_hit", "peer_hit", "pool_hit", "total_hit", "total_blocks"]:
        sv, iv = sa[key], e2e[key]
        diff = iv - sv
        print("%-25s %15d %15d %+10d" % (key, sv, iv, diff))
    print("%-25s %14.2f%% %14.2f%% %+9.2f%%" % (
        "hit_ratio", sa["hit_ratio"]*100, e2e["hit_ratio"]*100,
        (e2e["hit_ratio"]-sa["hit_ratio"])*100))
    if "ttft_ms" in e2e:
        print("%-25s %15s %14.0fms" % ("TTFT mean", "N/A", e2e["ttft_ms"]))
    print("=" * 70)


# ===========================================================================
# Test 1: 5 pods, RoundRobin, P2P enabled — the standard config
# ===========================================================================

def test_comparison_5pods_rr_p2p():
    records = _load_enriched(100)
    pods = ["P0", "P1", "P2", "P3", "P4"]

    sa = _run_standalone(records, pods, enable_p2p=True, tag="5p_rr_p2p_sa")
    e2e = _run_e2e(records, pods, enable_p2p=True, tag="5p_rr_p2p_e2e")

    _print_comparison(sa, e2e, "5 Pods / RoundRobin / P2P Enabled")

    assert e2e["completed"] == 100
    assert sa["total_blocks"] == e2e["total_blocks"], "Same blocks queried"
    # E2E should have fewer hits due to batch scheduling
    assert e2e["total_hit"] <= sa["total_hit"], (
        "E2E should have <= standalone hits (batch delays writes), got e2e=%d > sa=%d" % (e2e["total_hit"], sa["total_hit"]))
    assert sa["total_hit"] > 0, "Standalone should have hits"


# ===========================================================================
# Test 2: 5 pods, RoundRobin, P2P disabled — isolate cross-node effect
# ===========================================================================

def test_comparison_5pods_rr_no_p2p():
    records = _load_enriched(100)
    pods = ["P0", "P1", "P2", "P3", "P4"]

    sa = _run_standalone(records, pods, enable_p2p=False, tag="5p_rr_nop2p_sa")
    e2e = _run_e2e(records, pods, enable_p2p=False, tag="5p_rr_nop2p_e2e")

    _print_comparison(sa, e2e, "5 Pods / RoundRobin / P2P Disabled")

    assert sa["peer_hit"] == 0, "No P2P means no peer hits in standalone"
    assert e2e["peer_hit"] == 0, "No P2P means no peer hits in E2E"
    assert e2e["total_hit"] <= sa["total_hit"]


# ===========================================================================
# Test 3: Single pod — no cross-node effects, purest comparison
# ===========================================================================

def test_comparison_1pod():
    records = _load_enriched(100)
    pods = ["P0"]

    sa = _run_standalone(records, pods, enable_p2p=False, tag="1p_sa")
    e2e = _run_e2e(records, pods, enable_p2p=False, tag="1p_e2e")

    _print_comparison(sa, e2e, "1 Pod / Single Instance")

    assert sa["peer_hit"] == 0
    assert e2e["peer_hit"] == 0
    assert sa["pool_hit"] == 0
    assert e2e["pool_hit"] == 0
    # Single pod: all hits are engine-local
    assert sa["total_hit"] == sa["engine_hit"]
    assert e2e["total_hit"] == e2e["engine_hit"]
    # E2E batch effect: fewer engine hits
    assert e2e["engine_hit"] <= sa["engine_hit"]


# ===========================================================================
# Test 4: 10 pods — more dilution, lower per-node hit rate
# ===========================================================================

def test_comparison_10pods_rr_p2p():
    records = _load_enriched(100)
    pods = ["P%d" % i for i in range(10)]

    sa = _run_standalone(records, pods, enable_p2p=True, tag="10p_rr_p2p_sa")
    e2e = _run_e2e(records, pods, enable_p2p=True, tag="10p_rr_p2p_e2e")

    _print_comparison(sa, e2e, "10 Pods / RoundRobin / P2P Enabled")

    assert e2e["total_hit"] <= sa["total_hit"]
    # More pods = more dilution, so hit rate should be lower than 5-pod case
    # (standalone already accounts for this)


# ===========================================================================
# Test 5: Quantify batch scheduling impact
# ===========================================================================

def test_batch_scheduling_impact():
    """Measure the exact hit rate gap caused by batch scheduling."""
    records = _load_enriched(100)
    pods = ["P0", "P1", "P2", "P3", "P4"]

    sa = _run_standalone(records, pods, enable_p2p=True, tag="impact_sa")
    e2e = _run_e2e(records, pods, enable_p2p=True, tag="impact_e2e")

    gap = sa["hit_ratio"] - e2e["hit_ratio"]
    relative_drop = gap / max(sa["hit_ratio"], 1e-9) * 100

    print("\n=== Batch Scheduling Impact ===")
    print("Standalone hit rate: %.2f%%" % (sa["hit_ratio"] * 100))
    print("E2E hit rate:        %.2f%%" % (e2e["hit_ratio"] * 100))
    print("Absolute gap:        %.2f pp" % (gap * 100))
    print("Relative drop:       %.1f%%" % relative_drop)
    print("E2E TTFT:            %.0f ms" % e2e["ttft_ms"])

    # The gap should be non-negative (batch scheduling can only reduce hits, not increase)
    assert gap >= 0, "E2E should not have more hits than standalone"
    # Both should have some hits with real block_ids
    assert sa["total_hit"] > 0


if __name__ == "__main__":
    test_comparison_5pods_rr_p2p()
    test_comparison_5pods_rr_no_p2p()
    test_comparison_1pod()
    test_comparison_10pods_rr_p2p()
    test_batch_scheduling_impact()
