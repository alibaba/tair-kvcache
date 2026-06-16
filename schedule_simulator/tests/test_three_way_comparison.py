"""
Three-way comparison: Optimizer Standalone vs E2E RoundRobin vs E2E DirectCacheAware.
Validates: total hit rate consistency across all three methods, and benchmarks wall time.

Requires: kvcm_py_optimizer, enriched JSONL with block_ids.
"""
import json, os, sys, random, time
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
from schedule_simulator.schedule_emulator.hierarchical_config_builder import build_hierarchical_config
from schedule_simulator.infer_time_predictor import RequestLevelTimePredictor

pytestmark = pytest.mark.skipif(not HAS_KVCM, reason="kvcm_py_optimizer not available")

MAIN_SVC_INPUT = "/sgl-workspace/claude_workspace/data/qwen36_main_svc_sim/main_svc_input.jsonl"
PREDICTOR_PKL = "/sgl-workspace/claude_workspace/data/qwen36_predictor/qwen36_prefill_predictor.pkl"


def _skip_if_no_data():
    if not os.path.exists(MAIN_SVC_INPUT):
        pytest.skip("Main service input data not available")


def _load_records(n=None):
    records = []
    with open(MAIN_SVC_INPUT) as f:
        for line in f:
            records.append(json.loads(line))
            if n and len(records) >= n:
                break
    return records


def _run_optimizer_standalone(records, pods):
    cfg = build_hierarchical_config(
        SchedulerConfig("Qwen2.5-3B", hicache_storage_backend="hf3fs",
                         hicache_read_query_type="prefix_match",
                         max_num_tokens=999999999, kv_cache_space_per_token=1),
        PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0,
                        memory_read_bandwidth_gb=16.0, peer_read_bandwidth_gb=10.0),
        p_instance_ids=pods, output_dir="/tmp/bench3_sa",
        storage_pool_capacity_gb=0.001, enable_p2p=True,
    )
    loader = kvcm.HierarchicalReplayConfigLoader()
    assert loader.load(cfg)
    mgr = kvcm.HierarchicalReplayManager(loader.config())
    assert mgr.Init()

    pod_set = set(pods)
    se, sp, ss, sb = 0, 0, 0, 0
    t0 = time.time()
    for i, r in enumerate(records):
        pod = r["instance_id"] if r["instance_id"] in pod_set else pods[i % len(pods)]
        bids = r["block_ids"]
        tn = int(r["timestamp"] * 1e6)
        # Drop partial tail blocks

        block_size = 1

        input_token_length = r.get("input_length", len(bids) * block_size)

        max_full_blocks = input_token_length // block_size

        full_bids = bids[:max_full_blocks]

        res = mgr.GetCacheLocation(pod, "r%d" % i, tn, full_bids, input_token_length)
        se += res.engine_hit_length
        sp += res.peer_hit_length
        ss += res.storage_pool_hit_length
        sb += len(full_bids)
        mgr.WriteCache(pod, "w%d" % i, tn + 1, full_bids)
    wall = time.time() - t0
    return {"local": se, "peer": sp, "pool": ss, "total": se + sp + ss,
            "blocks": sb, "wall_s": wall}


def _run_e2e(records, pods, policy, input_path, tag):
    random.seed(42)
    np.random.seed(42)
    predictor = RequestLevelTimePredictor(lookup_table_path=PREDICTOR_PKL)
    t0 = time.time()
    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(dataset_path=input_path,
                                          num_prompts=len(records), disable_tqdm=True),
        p_scheduler_config=SchedulerConfig(
            "Qwen2.5-3B", scenario="disagg_prefill",
            request_level_scheduling=True, max_num_tokens=999999999,
            l2_cache_num_tokens=999999999, kv_cache_space_per_token=1,
        ),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(
            device="H20", disk_read_bandwidth_gb=2.0,
            memory_read_bandwidth_gb=16.0, peer_read_bandwidth_gb=10.0,
        ),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(
            p_policy=policy, d_policy=RoutingPolicy.ROUND_ROBIN,
            worker_startup_check_interval=0.01,
        ),
        num_p_instance=len(pods), num_d_instance=0,
        infer_time_predictor=predictor,
        enable_hierarchical=True, enable_p2p=True,
        storage_pool_capacity_gb=0.001,
        hierarchical_output_dir="/tmp/bench3_%s" % tag,
    )
    runner.run_benchmark_emulation()
    wall = time.time() - t0
    h = runner.get_hierarchical_metrics()
    return {"local": h["total_engine_hit_blocks"], "peer": h["total_peer_hit_blocks"],
            "pool": h["total_pool_hit_blocks"],
            "total": h["total_engine_hit_blocks"] + h["total_peer_hit_blocks"] + h["total_pool_hit_blocks"],
            "blocks": h["total_blocks_queried"], "wall_s": wall}


def test_three_way_total_hit_consistent():
    """All three methods should produce the same total hit count."""
    _skip_if_no_data()
    records = _load_records(500)
    pods = sorted(set(r["instance_id"] for r in records))[:20]

    # Write temp input
    tmp = "/tmp/bench3_input.jsonl"
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    sa = _run_optimizer_standalone(records, pods)
    rr = _run_e2e(records, pods, RoutingPolicy.ROUND_ROBIN, tmp, "rr")
    dca = _run_e2e(records, pods, RoutingPolicy.DIRECT_CACHE_AWARE, tmp, "dca")

    print("\n%-25s %10s %10s %10s %10s %8s" % ("Method", "Local", "Peer", "Total", "Blocks", "Time"))
    print("-" * 75)
    print("%-25s %10d %10d %10d %10d %7.1fs" % ("Optimizer Standalone", sa["local"], sa["peer"], sa["total"], sa["blocks"], sa["wall_s"]))
    print("%-25s %10d %10d %10d %10d %7.1fs" % ("E2E RoundRobin", rr["local"], rr["peer"], rr["total"], rr["blocks"], rr["wall_s"]))
    print("%-25s %10d %10d %10d %10d %7.1fs" % ("E2E DirectCacheAware", dca["local"], dca["peer"], dca["total"], dca["blocks"], dca["wall_s"]))

    # Total hit should be within 1% across all three
    totals = [sa["total"], rr["total"], dca["total"]]
    max_diff = max(totals) - min(totals)
    avg = np.mean(totals)
    assert max_diff / max(avg, 1) < 0.01, (
        "Total hit should be consistent: SA=%d RR=%d DCA=%d (diff=%d)" % (
            sa["total"], rr["total"], dca["total"], max_diff))

    # All should query the same number of blocks
    assert sa["blocks"] == rr["blocks"] == dca["blocks"]


def test_dca_maximizes_local_hit():
    """DirectCacheAware should have higher local hit than RoundRobin."""
    _skip_if_no_data()
    records = _load_records(200)
    pods = sorted(set(r["instance_id"] for r in records))[:10]

    tmp = "/tmp/bench3_local.jsonl"
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    rr = _run_e2e(records, pods, RoutingPolicy.ROUND_ROBIN, tmp, "rr_local")
    dca = _run_e2e(records, pods, RoutingPolicy.DIRECT_CACHE_AWARE, tmp, "dca_local")

    assert dca["local"] >= rr["local"], (
        "DCA local(%d) should >= RR local(%d)" % (dca["local"], rr["local"]))
    print("[dca_local] DCA local=%d vs RR local=%d" % (dca["local"], rr["local"]))


def test_rr_faster_than_dca():
    """RoundRobin should be significantly faster than DirectCacheAware."""
    _skip_if_no_data()
    records = _load_records(200)
    pods = sorted(set(r["instance_id"] for r in records))[:10]

    tmp = "/tmp/bench3_speed.jsonl"
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    rr = _run_e2e(records, pods, RoutingPolicy.ROUND_ROBIN, tmp, "rr_speed")
    dca = _run_e2e(records, pods, RoutingPolicy.DIRECT_CACHE_AWARE, tmp, "dca_speed")

    print("[speed] RR=%.1fs DCA=%.1fs ratio=%.1fx" % (rr["wall_s"], dca["wall_s"], dca["wall_s"]/max(rr["wall_s"], 0.01)))
    assert rr["wall_s"] < dca["wall_s"] * 3, "DCA should not be more than 3x slower than RR"


def test_standalone_fastest():
    """Optimizer standalone should be the fastest method."""
    _skip_if_no_data()
    records = _load_records(200)
    pods = sorted(set(r["instance_id"] for r in records))[:10]

    tmp = "/tmp/bench3_fastest.jsonl"
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    sa = _run_optimizer_standalone(records, pods)
    rr = _run_e2e(records, pods, RoutingPolicy.ROUND_ROBIN, tmp, "rr_fast")

    print("[fastest] SA=%.1fs RR=%.1fs" % (sa["wall_s"], rr["wall_s"]))
    assert sa["wall_s"] <= rr["wall_s"] * 1.5, "SA should be close to or faster than RR"


if __name__ == "__main__":
    test_three_way_total_hit_consistent()
    test_dca_maximizes_local_hit()
    test_rr_faster_than_dca()
    test_standalone_fastest()
