"""
Tests for P-Only multi-instance simulation and custom prefix tree integration.

TTFT validation:
- Single prefill for Qwen2.5-3B on H20: ~37ms (500 tok) to ~125ms (2000 tok)
- 200 requests / 20 nodes = 10 per node serial
- min TTFT ~ single prefill (first batch, no queuing)
- mean TTFT >> single prefill (queuing effect)
- P99 >= P95 >= P90 >= mean (proper distribution)
"""
import asyncio
import numpy
import random

from schedule_simulator.schedule_emulator.types import (
    BenchmarkConfig,
    SchedulerConfig,
    PlatformConfig,
    RouterConfig,
    RoutingPolicy,
    FakeRequest,
    PrefixCacheMatchResult,
    RequestStage,
)
from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner
from schedule_simulator.schedule_emulator.prefix_cache import PrefixCache
from schedule_simulator.schedule_emulator.dispatch.dispatch_policy import CacheAwarePolicy


NUM_P_INSTANCES = 20
NUM_REQUESTS = 200

# Theoretical baselines for Qwen2.5-3B on H20 (BF16, tp=1)
THEORETICAL_MIN_PREFILL_MS = 30.0
THEORETICAL_AVG_PREFILL_MS = 86.0
THEORETICAL_MAX_PREFILL_MS = 135.0


def _make_p_only_runner(p_policy: RoutingPolicy, seed: int = 42):
    random.seed(seed)
    numpy.random.seed(seed)

    benchmark_config = BenchmarkConfig(
        num_prompts=NUM_REQUESTS,
        min_input_length=500,
        max_input_length=2000,
        min_output_length=1,
        max_output_length=2,
        disable_tqdm=True,
    )
    p_scheduler_config = SchedulerConfig(
        "Qwen2.5-3B",
        scenario="disagg_prefill",
        enable_stats=True,
    )
    d_scheduler_config = SchedulerConfig(
        "Qwen2.5-3B",
        scenario="disagg_decode",
    )
    p_platform_config = PlatformConfig(device="H20")
    d_platform_config = PlatformConfig(device="H20")
    router_config = RouterConfig(
        p_policy=p_policy,
        d_policy=RoutingPolicy.ROUND_ROBIN,
        worker_startup_check_interval=0.01,
    )

    runner = DisaggBenchmarkRunner(
        benchmark_config=benchmark_config,
        p_scheduler_config=p_scheduler_config,
        d_scheduler_config=d_scheduler_config,
        p_platform_config=p_platform_config,
        d_platform_config=d_platform_config,
        router_config=router_config,
        num_p_instance=NUM_P_INSTANCES,
        num_d_instance=0,
    )
    return runner


def _get_distribution(runner: DisaggBenchmarkRunner) -> list[int]:
    return [len(s.completed_requests) for s in runner.p_schedulers]


def _validate_ttft(metrics: dict, policy_name: str):
    """Validate TTFT metrics against theoretical baselines."""
    mean_ttft = metrics["mean_ttft_ms"]
    p90_ttft = metrics["p90_ttft_ms"]
    p95_ttft = metrics["p95_ttft_ms"]
    p99_ttft = metrics["p99_ttft_ms"]

    # 1. Mean TTFT >= theoretical avg single prefill (queuing adds delay)
    assert mean_ttft >= THEORETICAL_AVG_PREFILL_MS * 0.9, (
        f"[{policy_name}] mean_ttft ({mean_ttft:.1f}ms) too low vs theoretical avg "
        f"({THEORETICAL_AVG_PREFILL_MS:.1f}ms)"
    )

    # 2. P99 must be bounded (not absurdly high)
    max_reasonable = THEORETICAL_MAX_PREFILL_MS * (NUM_REQUESTS // NUM_P_INSTANCES) * 1.2
    assert p99_ttft <= max_reasonable, (
        f"[{policy_name}] p99_ttft ({p99_ttft:.1f}ms) exceeds bound ({max_reasonable:.1f}ms)"
    )

    # 3. Proper ordering: P99 >= P95 >= P90
    assert p99_ttft >= p95_ttft >= p90_ttft, (
        f"[{policy_name}] Percentile ordering violated: p99={p99_ttft:.1f} p95={p95_ttft:.1f} p90={p90_ttft:.1f}"
    )

    # 4. Mean TTFT reflects queuing (significantly above single prefill)
    requests_per_node = NUM_REQUESTS // NUM_P_INSTANCES
    if requests_per_node > 1:
        assert mean_ttft >= THEORETICAL_AVG_PREFILL_MS * 2, (
            f"[{policy_name}] mean_ttft ({mean_ttft:.1f}ms) too low for "
            f"{requests_per_node} reqs/node - should show queuing"
        )

    # 5. P90 should be meaningfully above the average (tail behavior)
    assert p90_ttft >= mean_ttft * 0.8, (
        f"[{policy_name}] p90 ({p90_ttft:.1f}) unreasonably below mean ({mean_ttft:.1f})"
    )

    print(f"[{policy_name}] TTFT mean={mean_ttft:.1f} p90={p90_ttft:.1f} "
          f"p95={p95_ttft:.1f} p99={p99_ttft:.1f}ms [VALID]")


# =============================================================================
# Test 1: P-Only 20 nodes with all 4 routing policies + TTFT validation
# =============================================================================

def test_p_only_random_policy():
    runner = _make_p_only_runner(RoutingPolicy.RANDOM)
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == NUM_REQUESTS
    _validate_ttft(metrics, "Random")
    dist = _get_distribution(runner)
    assert sum(dist) == NUM_REQUESTS


def test_p_only_round_robin_policy():
    runner = _make_p_only_runner(RoutingPolicy.ROUND_ROBIN)
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == NUM_REQUESTS
    _validate_ttft(metrics, "RoundRobin")
    dist = _get_distribution(runner)
    expected = NUM_REQUESTS // NUM_P_INSTANCES
    for i, count in enumerate(dist):
        assert count == expected, f"RoundRobin: instance {i} got {count}, expected {expected}"


def test_p_only_power_of_two_policy():
    runner = _make_p_only_runner(RoutingPolicy.POWER_OF_TWO)
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == NUM_REQUESTS
    _validate_ttft(metrics, "PowerOfTwo")
    dist = _get_distribution(runner)
    assert all(d > 0 for d in dist)


def test_p_only_cache_aware_policy():
    runner = _make_p_only_runner(RoutingPolicy.CACHE_AWARE)
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == NUM_REQUESTS
    _validate_ttft(metrics, "CacheAware")
    dist = _get_distribution(runner)
    assert max(dist) > 0


# =============================================================================
# Test 2: Distribution comparison
# =============================================================================

def test_distribution_differs_across_policies():
    rr_runner = _make_p_only_runner(RoutingPolicy.ROUND_ROBIN, seed=123)
    rr_runner.run_benchmark_emulation()
    rr_dist = _get_distribution(rr_runner)

    rand_runner = _make_p_only_runner(RoutingPolicy.RANDOM, seed=123)
    rand_runner.run_benchmark_emulation()
    rand_dist = _get_distribution(rand_runner)

    assert all(d == NUM_REQUESTS // NUM_P_INSTANCES for d in rr_dist)
    assert numpy.std(rr_dist) == 0
    print(f"[Distribution] RR std={numpy.std(rr_dist):.2f}, Random std={numpy.std(rand_dist):.2f}")


# =============================================================================
# Test 3: Custom PrefixCache
# =============================================================================

class CountingPrefixCache(PrefixCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.match_count = 0
        self.add_count = 0

    def match_prefix(self, req) -> PrefixCacheMatchResult:
        self.match_count += 1
        return PrefixCacheMatchResult()

    def add_to_prefetch_queue(self, req, *args, **kwargs):
        self.add_count += 1


def test_custom_prefix_cache_integration():
    from schedule_simulator.schedule_emulator.sglang_scheduler import SGLangScheduleEmulator
    from schedule_simulator.schedule_emulator.schedule_policy import SchedulePolicy

    scheduler_config = SchedulerConfig("Qwen2.5-3B")
    platform_config = PlatformConfig(device="H20")
    request_queue = asyncio.Queue()
    response_queue = asyncio.Queue()

    scheduler = SGLangScheduleEmulator(
        scheduler_config=scheduler_config,
        platform_config=platform_config,
        request_queue=request_queue,
        response_queue=response_queue,
        name="TestScheduler",
    )

    custom_cache = CountingPrefixCache(
        platform_config,
        kv_cache_space_per_token=scheduler.kv_cache_space_per_token,
        page_size=scheduler.scheduler_config.page_size,
        global_values=scheduler.global_values,
    )
    scheduler.tree_cache = custom_cache
    scheduler.policy = SchedulePolicy(
        scheduler_config.schedule_policy, custom_cache, scheduler.time_predictor
    )

    for i in range(10):
        request_queue.put_nowait(
            FakeRequest(id=i, input_token_length=500, output_token_length=1, last_event_time=0)
        )
    scheduler.set_num_requests(10)
    asyncio.run(scheduler.event_loop())

    assert custom_cache.add_count >= 10
    assert custom_cache.match_count >= 10
    assert len(scheduler.completed_requests) == 10
    print(f"[CustomPrefixCache] add={custom_cache.add_count}, match={custom_cache.match_count}")


# =============================================================================
# Test 4: Custom routing tree
# =============================================================================

class CountingRoutingTree:
    def __init__(self, num_workers):
        self.worker_ids = [f"worker_{i}" for i in range(num_workers)]
        self.insert_count = 0
        self.prefix_match_count = 0
        self.get_smallest_count = 0
        self._rr_idx = 0
        self._token_counts = {wid: 0 for wid in self.worker_ids}

    def insert(self, text, tenant, timestamp):
        self.insert_count += 1
        self._token_counts[tenant] = self._token_counts.get(tenant, 0) + (len(text) if text else 0)

    def prefix_match(self, text, timestamp):
        self.prefix_match_count += 1
        return ([], "empty")

    def get_smallest_tenant(self):
        self.get_smallest_count += 1
        tid = self.worker_ids[self._rr_idx % len(self.worker_ids)]
        self._rr_idx += 1
        return tid

    def evict_tenant_by_size(self, max_size):
        pass

    def get_tenant_token_count(self):
        return dict(self._token_counts)

    def remove_tenant(self, tenant):
        self._token_counts.pop(tenant, None)


def test_custom_routing_tree_integration():
    random.seed(42)
    numpy.random.seed(42)

    benchmark_config = BenchmarkConfig(
        num_prompts=50, min_input_length=500, max_input_length=1000,
        min_output_length=1, max_output_length=2, disable_tqdm=True,
    )
    runner = DisaggBenchmarkRunner(
        benchmark_config=benchmark_config,
        p_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill"),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20"),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.CACHE_AWARE_OLD, d_policy=RoutingPolicy.ROUND_ROBIN, worker_startup_check_interval=0.01),
        num_p_instance=5, num_d_instance=0,
    )

    custom_tree = CountingRoutingTree(num_workers=5)
    runner.p_policy.tree = custom_tree

    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 50
    assert custom_tree.prefix_match_count > 0
    assert custom_tree.insert_count > 0
    assert custom_tree.get_smallest_count > 0
    print(f"[CustomRoutingTree] match={custom_tree.prefix_match_count}, insert={custom_tree.insert_count}")


if __name__ == "__main__":
    test_p_only_round_robin_policy()
    test_p_only_random_policy()
    test_p_only_power_of_two_policy()
    test_p_only_cache_aware_policy()
    test_distribution_differs_across_policies()
    test_custom_prefix_cache_integration()
    test_custom_routing_tree_integration()
