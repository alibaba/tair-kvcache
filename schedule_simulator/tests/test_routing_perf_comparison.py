"""Performance comparison: DirectCacheAware vs CacheAware routing overhead."""
import random
import time

import numpy as np

from schedule_simulator.schedule_emulator.types import (
    BenchmarkConfig,
    SchedulerConfig,
    PlatformConfig,
    RouterConfig,
    RoutingPolicy,
)
from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner


def _make_runner(policy, num_p, num_requests, seed, use_real_token=False, dataset_path=None):
    random.seed(seed)
    np.random.seed(seed)
    bc = BenchmarkConfig(
        num_prompts=num_requests,
        min_input_length=500,
        max_input_length=2000,
        min_output_length=1,
        max_output_length=2,
        request_rate=float("inf"),
        disable_tqdm=True,
        dataset_path=dataset_path,
    )
    sc = SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill")
    dc = SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode")
    pc = PlatformConfig(device="H20")
    rc = RouterConfig(
        p_policy=policy,
        d_policy=RoutingPolicy.ROUND_ROBIN,
        worker_startup_check_interval=0.01,
    )
    return DisaggBenchmarkRunner(
        benchmark_config=bc,
        p_scheduler_config=sc,
        d_scheduler_config=dc,
        p_platform_config=pc,
        d_platform_config=pc,
        router_config=rc,
        num_p_instance=num_p,
        num_d_instance=0,
        use_real_token=use_real_token,
    )


def _benchmark_policy(policy, num_p, num_requests, seed, use_real_token=False,
                      dataset_path=None, warmup=1, repeats=3):
    """Run the simulation multiple times and return timing stats."""
    # Warmup
    for _ in range(warmup):
        r = _make_runner(policy, num_p, num_requests, seed, use_real_token, dataset_path)
        r.run_benchmark_emulation()

    times = []
    for _ in range(repeats):
        r = _make_runner(policy, num_p, num_requests, seed, use_real_token, dataset_path)
        t0 = time.perf_counter()
        metrics = r.run_benchmark_emulation()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        assert metrics["completed"] == num_requests

    return {
        "mean_s": np.mean(times),
        "std_s": np.std(times),
        "min_s": np.min(times),
        "max_s": np.max(times),
        "times": times,
    }


# ---------- Test: Overhead comparison (statistical mode, no real tokens) ----------

def test_overhead_statistical_mode():
    """Compare wall-clock time: direct_cache_aware vs cache_aware in statistical mode.
    Without real tokens, DirectCacheAware always returns 0 from prefix query,
    so overhead is minimal (just the hasattr check per scheduler).
    """
    NUM_P = 10
    NUM_REQ = 200
    SEED = 42

    stats_cache_aware = _benchmark_policy(
        RoutingPolicy.CACHE_AWARE, NUM_P, NUM_REQ, SEED, use_real_token=False,
    )
    stats_direct = _benchmark_policy(
        RoutingPolicy.DIRECT_CACHE_AWARE, NUM_P, NUM_REQ, SEED, use_real_token=False,
    )

    print(f"\n{'='*60}")
    print(f"Statistical mode: {NUM_REQ} requests, {NUM_P} P nodes")
    print(f"{'='*60}")
    print(f"CacheAware:       mean={stats_cache_aware['mean_s']*1000:.1f}ms  "
          f"std={stats_cache_aware['std_s']*1000:.1f}ms  "
          f"runs={stats_cache_aware['times']}")
    print(f"DirectCacheAware: mean={stats_direct['mean_s']*1000:.1f}ms  "
          f"std={stats_direct['std_s']*1000:.1f}ms  "
          f"runs={stats_direct['times']}")

    ratio = stats_direct["mean_s"] / stats_cache_aware["mean_s"]
    print(f"Ratio (direct/approx): {ratio:.2f}x")
    print(f"{'='*60}")

    # DirectCacheAware should not be more than 2x slower in stat mode
    # (it does less work since no tree maintenance)
    assert ratio < 2.0, f"DirectCacheAware is {ratio:.1f}x slower, expected < 2x"


# ---------- Test: Overhead comparison (real token mode with enriched data) ----------

def test_overhead_real_token_mode():
    """Compare wall-clock time: direct_cache_aware vs cache_aware in real-token mode.
    DirectCacheAware queries N trees per request; CacheAware queries 1 router tree.
    """
    NUM_P = 5
    NUM_REQ = 100
    SEED = 42
    DATASET = "tests/assets/glm5_sample/glm5_enriched_input.jsonl"

    stats_cache_aware = _benchmark_policy(
        RoutingPolicy.CACHE_AWARE, NUM_P, NUM_REQ, SEED,
        use_real_token=True, dataset_path=DATASET,
    )
    stats_direct = _benchmark_policy(
        RoutingPolicy.DIRECT_CACHE_AWARE, NUM_P, NUM_REQ, SEED,
        use_real_token=True, dataset_path=DATASET,
    )

    print(f"\n{'='*60}")
    print(f"Real-token mode: {NUM_REQ} requests, {NUM_P} P nodes, enriched data")
    print(f"{'='*60}")
    print(f"CacheAware:       mean={stats_cache_aware['mean_s']*1000:.1f}ms  "
          f"std={stats_cache_aware['std_s']*1000:.1f}ms  "
          f"runs={stats_cache_aware['times']}")
    print(f"DirectCacheAware: mean={stats_direct['mean_s']*1000:.1f}ms  "
          f"std={stats_direct['std_s']*1000:.1f}ms  "
          f"runs={stats_direct['times']}")

    ratio = stats_direct["mean_s"] / stats_cache_aware["mean_s"]
    print(f"Ratio (direct/approx): {ratio:.2f}x")
    print(f"{'='*60}")

    # In real-token mode, DirectCacheAware queries N trees vs 1 router tree
    # Still should be under 3x overhead since tree queries are O(L) in-memory
    assert ratio < 3.0, f"DirectCacheAware is {ratio:.1f}x slower, expected < 3x"


# ---------- Test: Scaling with number of P instances ----------

def test_overhead_scaling_with_instances():
    """Verify that DirectCacheAware overhead scales linearly (not worse) with N."""
    NUM_REQ = 100
    SEED = 42
    DATASET = "tests/assets/glm5_sample/glm5_enriched_input.jsonl"

    results = {}
    for num_p in [2, 5, 10, 20]:
        stats = _benchmark_policy(
            RoutingPolicy.DIRECT_CACHE_AWARE, num_p, NUM_REQ, SEED,
            use_real_token=True, dataset_path=DATASET,
            warmup=1, repeats=3,
        )
        results[num_p] = stats["mean_s"]

    print(f"\n{'='*60}")
    print(f"DirectCacheAware scaling: {NUM_REQ} requests, real-token mode")
    print(f"{'='*60}")
    for num_p, t in results.items():
        print(f"  N={num_p:2d} nodes: {t*1000:.1f}ms")

    # Time at N=20 should not be more than 15x time at N=2
    # (linear would be 10x; allowing some constant overhead)
    scaling_ratio = results[20] / results[2]
    print(f"Scaling ratio (N=20 / N=2): {scaling_ratio:.2f}x")
    print(f"{'='*60}")

    assert scaling_ratio < 15.0, f"Scaling {scaling_ratio:.1f}x exceeds 15x limit"
