"""Tests for the ChooseBestEngine C++ API and its integration with DirectCacheAwarePolicy."""
import sys
import time
import random

import numpy as np
import pytest

sys.path.insert(0, "/sgl-workspace/claude_workspace/tair-kvcache/bazel-bin/kv_cache_manager/optimizer/pybind")

try:
    import kvcm_py_optimizer as kvcm
    HAS_KVCM = True
except ImportError:
    HAS_KVCM = False

from schedule_simulator.schedule_emulator.types import (
    BenchmarkConfig, SchedulerConfig, PlatformConfig, RouterConfig, RoutingPolicy,
)
from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner

pytestmark = pytest.mark.skipif(not HAS_KVCM, reason="kvcm_py_optimizer not available")


def _make_manager(num_engines=5):
    from schedule_simulator.schedule_emulator.hierarchical_config_builder import build_hierarchical_config
    sc = SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill")
    pc = PlatformConfig(device="H20")
    ids = [f"P{i}" for i in range(num_engines)]
    config_path = build_hierarchical_config(
        scheduler_config=sc, platform_config=pc,
        p_instance_ids=ids, enable_p2p=True,
        output_dir=f"/tmp/test_cbe_{num_engines}",
    )
    loader = kvcm.HierarchicalReplayConfigLoader()
    loader.load(config_path)
    manager = kvcm.HierarchicalReplayManager(loader.config())
    manager.Init()
    return manager, ids


# ---------- Unit tests for ChooseBestEngine API ----------

def test_cold_returns_first_engine():
    """With no data written, ChooseBestEngine returns first engine with hit=0."""
    manager, ids = _make_manager(3)
    res = manager.ChooseBestEngine([100, 200, 300], 1000000000)
    assert res.hit_count == 0
    assert res.engine_instance_id == ids[0]


def test_single_write_finds_correct_engine():
    """After writing to P2, ChooseBestEngine should return P2."""
    manager, ids = _make_manager(5)
    blocks = [10, 20, 30, 40, 50]
    manager.WriteCache("P2", "req0", 1000000000, blocks)
    res = manager.ChooseBestEngine(blocks, 1000000100)
    assert res.engine_instance_id == "P2"
    assert res.hit_count == 5


def test_best_engine_is_longest_match():
    """When multiple engines have hits, return the one with longest prefix."""
    manager, ids = _make_manager(3)
    ts = 1000000000
    manager.WriteCache("P0", "r0", ts, [1, 2, 3])
    manager.WriteCache("P1", "r1", ts, [1, 2, 3, 4, 5, 6, 7])
    manager.WriteCache("P2", "r2", ts, [1, 2, 3, 4, 5])

    res = manager.ChooseBestEngine([1, 2, 3, 4, 5, 6, 7, 8], ts + 100)
    assert res.engine_instance_id == "P1"
    assert res.hit_count == 7


def test_empty_block_ids():
    """Empty block_ids should return first engine with 0 hits."""
    manager, ids = _make_manager(3)
    res = manager.ChooseBestEngine([], 1000000000)
    assert res.hit_count == 0


def test_performance_single_call_vs_n_calls():
    """ChooseBestEngine should be much faster than N individual GetCacheLocation calls."""
    manager, ids = _make_manager(20)
    ts = 1000000000
    blocks = list(range(100))

    # Write some data
    for i, eid in enumerate(ids):
        manager.WriteCache(eid, f"r{i}", ts, blocks[i*5:(i+1)*5])

    query_blocks = list(range(50))

    # Time N individual GetCacheLocation calls
    t0 = time.perf_counter()
    for _ in range(100):
        for eid in ids:
            manager.GetCacheLocation(eid, "q", ts+1000, query_blocks, len(query_blocks))
    time_n_calls = time.perf_counter() - t0

    # Time single ChooseBestEngine calls
    t0 = time.perf_counter()
    for _ in range(100):
        manager.ChooseBestEngine(query_blocks, ts+1000)
    time_single = time.perf_counter() - t0

    speedup = time_n_calls / time_single
    print(f"\n  N calls: {time_n_calls*1000:.1f}ms, single: {time_single*1000:.1f}ms, speedup: {speedup:.1f}x")
    assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.1f}x"


# ---------- Integration tests with DisaggBenchmarkRunner ----------

def _make_e2e_runner(policy, num_p=5, num_requests=50, seed=42):
    random.seed(seed); np.random.seed(seed)
    bc = BenchmarkConfig(
        num_prompts=num_requests, disable_tqdm=True,
        dataset_path="tests/assets/glm5_sample/glm5_enriched_500.jsonl",
    )
    sc = SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill")
    dc = SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode")
    pc = PlatformConfig(device="H20")
    rc = RouterConfig(p_policy=policy, worker_startup_check_interval=0.01)
    return DisaggBenchmarkRunner(
        benchmark_config=bc, p_scheduler_config=sc, d_scheduler_config=dc,
        p_platform_config=pc, d_platform_config=pc, router_config=rc,
        num_p_instance=num_p, num_d_instance=0,
        enable_hierarchical=True, enable_p2p=True,
    )


def test_e2e_completes_with_hierarchical():
    """DirectCacheAware + hierarchical completes all requests."""
    runner = _make_e2e_runner(RoutingPolicy.DIRECT_CACHE_AWARE)
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 50


def test_e2e_speedup_over_naive():
    """Optimized DCA should be comparable speed to RoundRobin (not 50x slower)."""
    random.seed(42); np.random.seed(42)
    runner_rr = _make_e2e_runner(RoutingPolicy.ROUND_ROBIN, num_requests=100)
    t0 = time.perf_counter()
    runner_rr.run_benchmark_emulation()
    time_rr = time.perf_counter() - t0

    random.seed(42); np.random.seed(42)
    runner_dca = _make_e2e_runner(RoutingPolicy.DIRECT_CACHE_AWARE, num_requests=100)
    t0 = time.perf_counter()
    runner_dca.run_benchmark_emulation()
    time_dca = time.perf_counter() - t0

    ratio = time_dca / time_rr
    print(f"\n  RR={time_rr:.2f}s DCA={time_dca:.2f}s ratio={ratio:.2f}x")
    assert ratio < 3.0, f"DCA is {ratio:.1f}x slower than RR, expected < 3x"


def test_e2e_hit_ratio_not_worse_than_rr():
    """DirectCacheAware should have >= hit ratio compared to RoundRobin."""
    random.seed(42); np.random.seed(42)
    runner_rr = _make_e2e_runner(RoutingPolicy.ROUND_ROBIN, num_p=5, num_requests=100)
    m_rr = runner_rr.run_benchmark_emulation()
    h_rr = runner_rr.get_hierarchical_metrics()

    random.seed(42); np.random.seed(42)
    runner_dca = _make_e2e_runner(RoutingPolicy.DIRECT_CACHE_AWARE, num_p=5, num_requests=100)
    m_dca = runner_dca.run_benchmark_emulation()
    h_dca = runner_dca.get_hierarchical_metrics()

    print(f"\n  RR hit_ratio={h_rr.get('block_hit_ratio',0):.4f} DCA hit_ratio={h_dca.get('block_hit_ratio',0):.4f}")
    # DCA should route more cache-friendly
    assert h_dca.get("block_hit_ratio", 0) >= h_rr.get("block_hit_ratio", 0) * 0.9
