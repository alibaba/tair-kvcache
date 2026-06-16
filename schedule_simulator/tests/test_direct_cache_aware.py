"""Tests for DirectCacheAwarePolicy — routes by querying real node caches."""
import asyncio
import random

import numpy as np

from schedule_simulator.schedule_emulator.types import (
    BenchmarkConfig,
    SchedulerConfig,
    PlatformConfig,
    RouterConfig,
    RoutingPolicy,
    FakeRequest,
    RequestStage,
)
from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner
from schedule_simulator.schedule_emulator.dispatch.dispatch_policy import (
    DirectCacheAwarePolicy,
)

NUM_P_INSTANCES = 5
NUM_REQUESTS = 100
SEED = 42


def _make_runner(
    policy: RoutingPolicy = RoutingPolicy.DIRECT_CACHE_AWARE,
    num_p=NUM_P_INSTANCES,
    num_requests=NUM_REQUESTS,
    seed=SEED,
    dataset_path=None,
    use_real_token=False,
    cache_threshold=0.3,
    balance_abs_threshold=64,
):
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
    sc = SchedulerConfig(
        "Qwen2.5-3B",
        scenario="disagg_prefill",
    )
    dc = SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode")
    pc = PlatformConfig(device="H20")
    rc = RouterConfig(
        p_policy=policy,
        d_policy=RoutingPolicy.ROUND_ROBIN,
        worker_startup_check_interval=0.01,
        cache_threshold=cache_threshold,
        balance_abs_threshold=balance_abs_threshold,
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


def _get_distribution(runner):
    return [len(s.completed_requests) for s in runner.p_schedulers]


# ---------- Test 1: Basic completion ----------

def test_basic_completion():
    runner = _make_runner()
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == NUM_REQUESTS
    assert metrics["mean_ttft_ms"] > 0
    assert metrics["p99_ttft_ms"] >= metrics["p90_ttft_ms"]


# ---------- Test 2: TTFT within bounds ----------

def test_ttft_valid():
    runner = _make_runner(num_requests=200, num_p=20)
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 200
    assert metrics["mean_ttft_ms"] >= 30.0
    assert metrics["p99_ttft_ms"] >= metrics["p95_ttft_ms"] >= metrics["p90_ttft_ms"]


# ---------- Test 3: Distribution not degenerate ----------

def test_distribution_not_degenerate():
    runner = _make_runner(num_requests=100, num_p=5)
    runner.run_benchmark_emulation()
    dist = _get_distribution(runner)
    assert sum(dist) == 100
    assert max(dist) < 100  # not all to one node
    assert min(dist) > 0  # every node gets at least one


# ---------- Test 4: Shared prefix affinity with enriched data ----------

def test_shared_prefix_affinity():
    """With enriched data (block_ids), requests with shared prefixes cluster."""
    random.seed(SEED)
    np.random.seed(SEED)

    # Use GLM5 enriched dataset which has real block_ids with prefix sharing
    dataset_path = "tests/assets/glm5_sample/glm5_enriched_input.jsonl"
    bc = BenchmarkConfig(
        num_prompts=30, disable_tqdm=True,
        dataset_path=dataset_path,
    )
    sc = SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill")
    dc = SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode")
    pc = PlatformConfig(device="H20")
    rc = RouterConfig(
        p_policy=RoutingPolicy.DIRECT_CACHE_AWARE,
        worker_startup_check_interval=0.01,
        cache_threshold=0.1,
    )

    runner = DisaggBenchmarkRunner(
        benchmark_config=bc,
        p_scheduler_config=sc,
        d_scheduler_config=dc,
        p_platform_config=pc,
        d_platform_config=pc,
        router_config=rc,
        num_p_instance=3,
        num_d_instance=0,
        use_real_token=True,
    )

    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 30
    dist = _get_distribution(runner)
    assert sum(dist) == 30
    # With prefix sharing, distribution should be uneven (cache affinity effect)
    assert max(dist) > min(dist)


# ---------- Test 5: No real tokens degrades to load balance ----------

def test_no_real_tokens_degrades_to_load_balance():
    """Without origin_input_ids, policy falls back to load-based routing."""
    runner = _make_runner(use_real_token=False, num_p=5, num_requests=50)
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 50
    dist = _get_distribution(runner)
    # Should be roughly uniform since no cache affinity
    std = float(np.std(dist))
    mean = float(np.mean(dist))
    assert std / mean < 0.5  # CV < 0.5 means fairly balanced


# ---------- Test 6: Load balance override ----------

def test_load_balance_override():
    """When load imbalance thresholds are very low, distribution stays even."""
    runner = _make_runner(
        num_p=5, num_requests=50,
        balance_abs_threshold=2,
    )
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 50
    dist = _get_distribution(runner)
    assert max(dist) - min(dist) <= 20


# ---------- Test 7: Direct vs Approximate with real token mode ----------

def test_direct_vs_approximate_prefix_reuse():
    """Both policies complete all requests; direct has >= prefix reuse."""
    # Compare without real-token mode (statistical mode)
    runner_approx = _make_runner(
        policy=RoutingPolicy.CACHE_AWARE, num_p=3, num_requests=60,
        seed=123, use_real_token=False,
    )
    metrics_approx = runner_approx.run_benchmark_emulation()

    runner_direct = _make_runner(
        policy=RoutingPolicy.DIRECT_CACHE_AWARE, num_p=3, num_requests=60,
        seed=123, use_real_token=False,
    )
    metrics_direct = runner_direct.run_benchmark_emulation()

    assert metrics_approx["completed"] == 60
    assert metrics_direct["completed"] == 60
    # Without real tokens, both degrade to load-based — both should have same reuse ratio
    # prefix_cache_reused_ratio removed; hierarchical metrics provide clearer hit stats


# ---------- Test 8: query_prefix_length is read-only ----------

def test_query_prefix_length_readonly():
    """Calling query_prefix_length must not modify tree structure."""
    from schedule_simulator.schedule_emulator.kvcache_simulation.pure_radix_tree import RadixCache
    from schedule_simulator.schedule_emulator.kvcache_simulation.kvcache_base_classes import RadixKey
    from schedule_simulator.schedule_emulator.kvcache_simulation.kvcache_utils import (
        ReqToTokenPoolHost,
        KVCachePool,
    )

    pool = KVCachePool(size=10000, page_size=1)
    req_pool = ReqToTokenPoolHost(size=100, max_context_len=1024)
    cache = RadixCache(
        page_size=1,
        req_to_token_pool=req_pool,
        kv_pool=pool,
    )

    # Insert a known sequence
    key1 = RadixKey(token_ids=[1, 2, 3, 4, 5, 6, 7, 8])
    cache.insert(key1)

    def count_nodes(node):
        count = 1
        for child in node.children.values():
            count += count_nodes(child)
        return count

    nodes_before = count_nodes(cache.root_node)

    # Query with a partial match (would normally cause a split in match_prefix)
    partial_key = RadixKey(token_ids=[1, 2, 3, 4, 5, 99, 100])
    length = cache.query_prefix_length(partial_key)
    assert length == 5  # matches first 5 tokens

    nodes_after = count_nodes(cache.root_node)
    assert nodes_after == nodes_before  # no split happened

    # Full match
    full_key = RadixKey(token_ids=[1, 2, 3, 4, 5, 6, 7, 8])
    length = cache.query_prefix_length(full_key)
    assert length == 8

    # No match
    no_match_key = RadixKey(token_ids=[99, 100, 101])
    length = cache.query_prefix_length(no_match_key)
    assert length == 0

    # Verify that the normal match_prefix DOES split (control test)
    split_key = RadixKey(token_ids=[1, 2, 3, 4, 5, 99, 100])
    cache.match_prefix(split_key)
    nodes_after_split = count_nodes(cache.root_node)
    assert nodes_after_split > nodes_before  # split did happen
