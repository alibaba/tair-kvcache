"""Tests for the pre-cache-aware-scheduler probabilistic routing strategy."""
import math
import random
import sys

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


# ---------- Scoring formula unit tests ----------

def _compute_score(p_i, l_i, l_bar, wp=30.0, wl=10.0):
    """Replicate the scoring formula from pre-cache-aware-scheduler."""
    exponent = wp * p_i + wl * (l_bar - l_i)
    return math.pow(2, exponent)


def test_scoring_formula_correctness():
    """Verify the scoring formula produces expected values."""
    # Case 1: Full cache hit (p=1.0), average load
    score_hit = _compute_score(p_i=1.0, l_i=0.5, l_bar=0.5)
    score_miss = _compute_score(p_i=0.0, l_i=0.5, l_bar=0.5)
    # 2^30 vs 2^0 = 1
    assert score_hit == pytest.approx(2**30, rel=1e-6)
    assert score_miss == pytest.approx(1.0, rel=1e-6)

    # Case 2: No hit but low load
    score_low_load = _compute_score(p_i=0.0, l_i=0.0, l_bar=0.5)
    # exponent = 0 + 10*(0.5-0) = 5 → 2^5 = 32
    assert score_low_load == pytest.approx(32.0, rel=1e-6)

    # Case 3: No hit, high load
    score_high_load = _compute_score(p_i=0.0, l_i=1.0, l_bar=0.5)
    # exponent = 0 + 10*(0.5-1.0) = -5 → 2^(-5) ≈ 0.03125
    assert score_high_load == pytest.approx(1.0 / 32.0, rel=1e-6)


def test_high_hit_dominates_load():
    """Cache hit weight (30) should dominate load weight (10)."""
    # Engine A: 100% hit, highest load
    score_a = _compute_score(p_i=1.0, l_i=1.0, l_bar=0.5)
    # Engine B: 0% hit, lowest load
    score_b = _compute_score(p_i=0.0, l_i=0.0, l_bar=0.5)

    # A: 2^(30*1 + 10*(0.5-1.0)) = 2^(30-5) = 2^25
    # B: 2^(0 + 10*(0.5-0)) = 2^5 = 32
    assert score_a > score_b * 1000  # A should be WAY higher


def test_low_load_preferred_on_tie():
    """When cache hit is the same, lower load should get higher score."""
    score_low = _compute_score(p_i=0.5, l_i=0.2, l_bar=0.5)
    score_high = _compute_score(p_i=0.5, l_i=0.8, l_bar=0.5)
    assert score_low > score_high


# ---------- Integration tests ----------

def _make_e2e_runner(topk_routing=True, num_p=5, num_requests=50, seed=42, **kwargs):
    random.seed(seed)
    np.random.seed(seed)
    bc = BenchmarkConfig(
        num_prompts=num_requests, disable_tqdm=True,
        dataset_path="tests/assets/glm5_sample/glm5_enriched_500.jsonl",
    )
    sc = SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill")
    dc = SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode")
    pc = PlatformConfig(device="H20")
    rc = RouterConfig(
        p_policy=RoutingPolicy.DIRECT_CACHE_AWARE,
        worker_startup_check_interval=0.01,
        topk_routing=topk_routing,
        **kwargs,
    )
    return DisaggBenchmarkRunner(
        benchmark_config=bc, p_scheduler_config=sc, d_scheduler_config=dc,
        p_platform_config=pc, d_platform_config=pc, router_config=rc,
        num_p_instance=num_p, num_d_instance=0,
        enable_hierarchical=True, enable_p2p=True,
    )


def test_e2e_completes_all_requests():
    """Probabilistic routing should complete all requests."""
    runner = _make_e2e_runner(topk_routing=True, num_requests=50)
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 50


def test_topk_routing_disabled_backward_compat():
    """When topk_routing=False, behavior should be the same as before."""
    runner = _make_e2e_runner(topk_routing=False, num_requests=30)
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 30


def test_high_hit_gets_more_routing():
    """Engine with high cache hit should receive more requests over many iterations."""
    # Use a small setup: 3 engines, write lots of data to P0, query same prefix
    from schedule_simulator.schedule_emulator.hierarchical_config_builder import build_hierarchical_config
    sc = SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill")
    pc = PlatformConfig(device="H20")
    ids = ["P0", "P1", "P2"]
    config_path = build_hierarchical_config(
        scheduler_config=sc, platform_config=pc,
        p_instance_ids=ids, enable_p2p=True,
        output_dir="/tmp/test_prob_routing",
    )
    loader = kvcm.HierarchicalReplayConfigLoader()
    loader.load(config_path)
    manager = kvcm.HierarchicalReplayManager(loader.config())
    manager.Init()

    # Write heavy prefix to P0, light to P1, none to P2
    ts = 1000000000
    manager.WriteCache("P0", "r0", ts, list(range(1, 51)))  # 50 blocks
    manager.WriteCache("P1", "r1", ts, list(range(1, 11)))  # 10 blocks

    query = list(range(1, 51))

    # Simulate many routing decisions using the scoring formula
    random.seed(42)
    route_counts = {"P0": 0, "P1": 0, "P2": 0}
    N = 1000
    for _ in range(N):
        results = manager.ChooseTopKEngines(query, ts + 100, 0)
        hit_map = {r.engine_instance_id: r.hit_count for r in results}

        page_size = 1
        input_len = 50
        lmax = 40
        loads = [0, 0, 0]  # All idle
        l_bar = 0.0

        scores = []
        for i, eid in enumerate(ids):
            hit_count = hit_map.get(eid, 0)
            p_i = min((hit_count * page_size) / input_len, 1.0)
            exponent = 30.0 * p_i + 10.0 * (l_bar - loads[i] / lmax)
            score = math.pow(2, exponent)
            scores.append((eid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        K = max(math.ceil(math.sqrt(3)), 5)
        topk = scores[:K]

        total_score = sum(s[1] for s in topk)
        r = random.random() * total_score
        cumulative = 0.0
        chosen = topk[0][0]
        for eid, sc_val in topk:
            cumulative += sc_val
            if cumulative >= r:
                chosen = eid
                break
        route_counts[chosen] += 1

    # P0 should get the vast majority (it has 100% hit → 2^30 score)
    assert route_counts["P0"] > N * 0.9, f"P0 should get >90% but got {route_counts['P0']/N*100:.1f}%"
    print(f"\n  Route distribution: P0={route_counts['P0']}, P1={route_counts['P1']}, P2={route_counts['P2']}")


def test_load_balance_effect():
    """When one node is heavily loaded, routing should shift away from it."""
    scores_idle = _compute_score(p_i=1.0, l_i=0.0, l_bar=0.5)
    scores_loaded = _compute_score(p_i=1.0, l_i=1.0, l_bar=0.5)

    # Even with same cache hit, loaded node should have much lower score
    ratio = scores_idle / scores_loaded
    # 2^(30+5) / 2^(30-5) = 2^10 = 1024
    assert ratio == pytest.approx(1024.0, rel=1e-6)
