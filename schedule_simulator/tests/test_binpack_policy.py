"""
Tests for BinPackPolicy routing strategy.

Validates that:
1. BinPackPolicy can be instantiated and run simulations
2. BinPackPolicy produces the same results as CacheAwarePolicy (same logic)
3. BinPackPolicy is properly registered in the factory
4. All load balancing behaviors are inherited correctly
5. CLI routing parameter 'bin_pack' is properly supported
"""
import asyncio
import random
import numpy as np
import pytest

from schedule_simulator.schedule_emulator.types import (
    BenchmarkConfig,
    SchedulerConfig,
    PlatformConfig,
    RouterConfig,
    RoutingPolicy,
)
from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner
from schedule_simulator.schedule_emulator.dispatch.dispatch_policy import (
    BinPackPolicy,
    CacheAwarePolicy,
)


def _make_runner(
    num_p: int = 3,
    num_requests: int = 20,
    p_policy: RoutingPolicy = RoutingPolicy.BIN_PACK,
    balance_abs_threshold: int = 8,
    seed: int = 42,
):
    """Create a runner with request-level scheduling and BinPackPolicy."""
    random.seed(seed)
    np.random.seed(seed)

    bc = BenchmarkConfig(
        num_prompts=num_requests,
        min_input_length=500,
        max_input_length=2000,
        min_output_length=1,
        max_output_length=2,
        disable_tqdm=True,
    )
    sc = SchedulerConfig(
        "Qwen2.5-3B",
        scenario="disagg_prefill",
        request_level_scheduling=True,
    )
    pc = PlatformConfig(device="H20")
    rc = RouterConfig(
        p_policy=p_policy,
        balance_abs_threshold=balance_abs_threshold,
    )
    runner = DisaggBenchmarkRunner(
        benchmark_config=bc,
        p_scheduler_config=sc,
        d_scheduler_config=SchedulerConfig(
            "Qwen2.5-3B", scenario="disagg_decode", request_level_scheduling=True
        ),
        p_platform_config=pc,
        d_platform_config=pc,
        router_config=rc,
        num_p_instance=num_p,
        num_d_instance=0,
    )
    return runner


def _get_distribution(runner) -> list:
    """Get per-node completed request counts."""
    return [len(s.completed_requests) for s in runner.p_schedulers]


class TestBinPackPolicyBasic:
    """Basic functional tests for BinPackPolicy."""

    def test_binpack_policy_instantiation(self):
        """BinPackPolicy can be instantiated with correct name."""
        config = RouterConfig(p_policy=RoutingPolicy.BIN_PACK)
        policy = BinPackPolicy(num_schedulers=3, config=config)
        assert policy.name == RoutingPolicy.BIN_PACK
        assert isinstance(policy, CacheAwarePolicy)

    def test_binpack_is_subclass_of_cache_aware(self):
        """BinPackPolicy is a proper subclass of CacheAwarePolicy."""
        assert issubclass(BinPackPolicy, CacheAwarePolicy)

    def test_binpack_enum_value(self):
        """RoutingPolicy.BIN_PACK has correct enum value."""
        assert RoutingPolicy.BIN_PACK.value == "bin_pack"

    def test_binpack_from_string(self):
        """BIN_PACK can be constructed from string."""
        policy = RoutingPolicy("bin_pack")
        assert policy == RoutingPolicy.BIN_PACK

    def test_all_requests_complete(self):
        """Basic sanity: all requests should complete with BinPackPolicy."""
        runner = _make_runner(num_p=3, num_requests=30)
        metrics = runner.run_benchmark_emulation()
        assert metrics["completed"] == 30

    def test_metrics_positive(self):
        """Metrics should have valid positive values."""
        runner = _make_runner(num_p=3, num_requests=30)
        metrics = runner.run_benchmark_emulation()
        assert metrics["mean_ttft_ms"] > 0
        assert metrics["p99_ttft_ms"] >= metrics["p90_ttft_ms"]
        assert metrics["p90_ttft_ms"] > 0


class TestBinPackPolicyLoadBalance:
    """Test load balancing behavior (inherited from CacheAwarePolicy)."""

    def test_distributes_across_nodes(self):
        """Requests should be distributed across multiple nodes."""
        runner = _make_runner(num_p=3, num_requests=30, balance_abs_threshold=4)
        metrics = runner.run_benchmark_emulation()
        dist = _get_distribution(runner)
        assert metrics["completed"] == 30
        # No single node should get ALL requests
        assert max(dist) < 30, f"All requests went to one node: {dist}"
        # At least 2 nodes should have received requests
        nodes_with_work = sum(1 for d in dist if d > 0)
        assert nodes_with_work >= 2, f"Only {nodes_with_work} node(s) used: {dist}"

    def test_many_nodes(self):
        """BinPackPolicy works correctly with many nodes."""
        runner = _make_runner(num_p=10, num_requests=50, balance_abs_threshold=4)
        metrics = runner.run_benchmark_emulation()
        assert metrics["completed"] == 50
        dist = _get_distribution(runner)
        nodes_with_work = sum(1 for d in dist if d > 0)
        assert nodes_with_work >= 2


class TestBinPackPolicyEquivalence:
    """Test that BinPackPolicy produces identical results to CacheAwarePolicy."""

    def test_same_routing_results_as_cache_aware(self):
        """BinPackPolicy and CacheAwarePolicy should produce identical results
        given the same seed and configuration (since logic is identical)."""
        # Run with BinPackPolicy
        random.seed(123)
        np.random.seed(123)
        runner_bp = _make_runner(
            num_p=3, num_requests=30, p_policy=RoutingPolicy.BIN_PACK, seed=123
        )
        metrics_bp = runner_bp.run_benchmark_emulation()
        dist_bp = _get_distribution(runner_bp)

        # Run with CacheAwarePolicy
        random.seed(123)
        np.random.seed(123)
        runner_ca = _make_runner(
            num_p=3, num_requests=30, p_policy=RoutingPolicy.CACHE_AWARE, seed=123
        )
        metrics_ca = runner_ca.run_benchmark_emulation()
        dist_ca = _get_distribution(runner_ca)

        # Results should be identical
        assert metrics_bp["completed"] == metrics_ca["completed"]
        assert dist_bp == dist_ca, (
            f"Distribution mismatch: BinPack={dist_bp}, CacheAware={dist_ca}"
        )
        assert abs(metrics_bp["mean_ttft_ms"] - metrics_ca["mean_ttft_ms"]) < 0.01

    def test_same_routing_with_more_requests(self):
        """Equivalence test with more requests to increase coverage."""
        random.seed(456)
        np.random.seed(456)
        runner_bp = _make_runner(
            num_p=5, num_requests=50, p_policy=RoutingPolicy.BIN_PACK, seed=456
        )
        metrics_bp = runner_bp.run_benchmark_emulation()
        dist_bp = _get_distribution(runner_bp)

        random.seed(456)
        np.random.seed(456)
        runner_ca = _make_runner(
            num_p=5, num_requests=50, p_policy=RoutingPolicy.CACHE_AWARE, seed=456
        )
        metrics_ca = runner_ca.run_benchmark_emulation()
        dist_ca = _get_distribution(runner_ca)

        assert metrics_bp["completed"] == metrics_ca["completed"]
        assert dist_bp == dist_ca


class TestBinPackPolicyFactory:
    """Test factory pattern and configuration."""

    def test_factory_creates_binpack_policy(self):
        """_create_policy should create BinPackPolicy for BIN_PACK enum."""
        runner = _make_runner(num_p=3, num_requests=5)
        # Check that the policy is indeed a BinPackPolicy
        assert isinstance(runner.p_policy, BinPackPolicy)
        assert runner.p_policy.name == RoutingPolicy.BIN_PACK

    def test_router_config_from_string(self):
        """RouterConfig.from_args should parse 'bin_pack' string."""
        rc = RouterConfig.from_string_policy(p_policy_str="bin_pack")
        assert rc.p_policy == RoutingPolicy.BIN_PACK

    def test_policy_with_topk_routing(self):
        """BinPackPolicy should support topk_routing flag."""
        bc = BenchmarkConfig(
            num_prompts=10,
            min_input_length=500,
            max_input_length=2000,
            min_output_length=1,
            max_output_length=2,
            disable_tqdm=True,
        )
        sc = SchedulerConfig(
            "Qwen2.5-3B",
            scenario="disagg_prefill",
            request_level_scheduling=True,
        )
        pc = PlatformConfig(device="H20")
        rc = RouterConfig(
            p_policy=RoutingPolicy.BIN_PACK,
            topk_routing=True,
        )
        runner = DisaggBenchmarkRunner(
            benchmark_config=bc,
            p_scheduler_config=sc,
            d_scheduler_config=SchedulerConfig(
                "Qwen2.5-3B", scenario="disagg_decode", request_level_scheduling=True
            ),
            p_platform_config=pc,
            d_platform_config=pc,
            router_config=rc,
            num_p_instance=3,
            num_d_instance=0,
        )
        metrics = runner.run_benchmark_emulation()
        assert metrics["completed"] == 10


class TestBinPackPolicyHierarchical:
    """Test BinPackPolicy with hierarchical cache (Optimizer C++ tree)."""

    def test_with_hierarchical_enabled(self):
        """BinPackPolicy should work with enable_hierarchical=True."""
        random.seed(42)
        np.random.seed(42)
        bc = BenchmarkConfig(
            num_prompts=20,
            min_input_length=500,
            max_input_length=2000,
            min_output_length=1,
            max_output_length=2,
            disable_tqdm=True,
        )
        sc = SchedulerConfig(
            "Qwen2.5-3B",
            scenario="disagg_prefill",
            request_level_scheduling=True,
        )
        pc = PlatformConfig(device="H20")
        rc = RouterConfig(
            p_policy=RoutingPolicy.BIN_PACK,
            balance_abs_threshold=8,
        )
        runner = DisaggBenchmarkRunner(
            benchmark_config=bc,
            p_scheduler_config=sc,
            d_scheduler_config=SchedulerConfig(
                "Qwen2.5-3B", scenario="disagg_decode", request_level_scheduling=True
            ),
            p_platform_config=pc,
            d_platform_config=pc,
            router_config=rc,
            num_p_instance=3,
            num_d_instance=0,
            enable_hierarchical=True,
            enable_p2p=True,
        )
        metrics = runner.run_benchmark_emulation()
        assert metrics["completed"] == 20
        dist = _get_distribution(runner)
        nodes_with_work = sum(1 for d in dist if d > 0)
        assert nodes_with_work >= 1  # At least one node should have work
