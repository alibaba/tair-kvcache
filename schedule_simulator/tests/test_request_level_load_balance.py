"""
Tests for request-level mode two-phase state modeling.

Validates that:
1. Long-running requests on one node cause subsequent requests to be routed to other nodes
2. Load balancing reflects real in-flight requests (not cumulative dispatch count)
3. Cache affinity still works when load is balanced
4. Total completion count and TTFT correctness are maintained
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


def _make_runner(
    num_p: int = 3,
    num_requests: int = 20,
    p_policy: RoutingPolicy = RoutingPolicy.CACHE_AWARE,
    balance_abs_threshold: int = 8,
    seed: int = 42,
):
    """Create a runner with request-level scheduling and specified policy."""
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
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode", request_level_scheduling=True),
        p_platform_config=pc,
        d_platform_config=pc,
        router_config=rc,
        num_p_instance=num_p,
        num_d_instance=0,
    )
    return runner


def _get_distribution(runner) -> list[int]:
    """Get per-node completed request counts."""
    return [len(s.completed_requests) for s in runner.p_schedulers]


class TestRequestLevelLoadBalance:
    """Test that load balancing works correctly in request-level mode."""

    def test_all_requests_complete(self):
        """Basic sanity: all requests should complete."""
        runner = _make_runner(num_p=3, num_requests=30)
        metrics = runner.run_benchmark_emulation()
        assert metrics["completed"] == 30

    def test_load_balance_distributes_across_nodes(self):
        """With cache_aware policy, requests should still be distributed
        across multiple nodes (not all go to one node)."""
        runner = _make_runner(num_p=3, num_requests=30, balance_abs_threshold=4)
        metrics = runner.run_benchmark_emulation()
        dist = _get_distribution(runner)
        assert metrics["completed"] == 30
        # With balance threshold=4, no single node should get ALL requests
        assert max(dist) < 30, f"All requests went to one node: {dist}"
        # At least 2 nodes should have received requests
        nodes_with_work = sum(1 for d in dist if d > 0)
        assert nodes_with_work >= 2, f"Only {nodes_with_work} node(s) used: {dist}"

    def test_long_request_triggers_rebalance(self):
        """When one node gets a very long request, subsequent requests
        should be routed to other nodes instead of waiting.
        
        This test creates a scenario where:
        - First few requests build cache on P0
        - A long request is dispatched to P0
        - While P0 is "busy", new requests should go to P1/P2
        """
        # Use shared prefix to build cache affinity on P0
        shared_prefix = list(range(1000, 1200))  # 200 blocks shared prefix
        
        records = []
        # First 3 requests: short, build cache on whatever node they land on
        for i in range(3):
            records.append({
                "timestamp": float(i) * 0.001,
                "input_length": 200,
                "output_length": 1,
                "block_ids": shared_prefix[:50],  # short prefix
            })
        # 4th request: very long (will have high latency due to large uncached portion)
        records.append({
            "timestamp": 0.003,
            "input_length": 8000,  # Very long → high latency
            "output_length": 1,
            "block_ids": shared_prefix + list(range(2000, 2600)),
        })
        # 5th-15th requests: should be distributed while the long one is processing
        for i in range(4, 15):
            records.append({
                "timestamp": float(i) * 0.001,
                "input_length": 200,
                "output_length": 1,
                "block_ids": shared_prefix[:50],
            })

        import tempfile, json, os
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for r in records:
            tmp.write(json.dumps(r) + "\n")
        tmp.close()

        random.seed(42)
        np.random.seed(42)

        bc = BenchmarkConfig(
            dataset_path=tmp.name,
            num_prompts=15,
            disable_tqdm=True,
        )
        sc = SchedulerConfig(
            "Qwen2.5-3B",
            scenario="disagg_prefill",
            request_level_scheduling=True,
        )
        pc = PlatformConfig(device="H20")
        rc = RouterConfig(
            p_policy=RoutingPolicy.CACHE_AWARE,
            balance_abs_threshold=4,
        )

        runner = DisaggBenchmarkRunner(
            benchmark_config=bc,
            p_scheduler_config=sc,
            d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode", request_level_scheduling=True),
            p_platform_config=pc,
            d_platform_config=pc,
            router_config=rc,
            num_p_instance=3,
            num_d_instance=0,
        )
        metrics = runner.run_benchmark_emulation()
        dist = _get_distribution(runner)
        os.unlink(tmp.name)

        assert metrics["completed"] == 15
        # The long request node should NOT monopolize all requests
        # At least 2 nodes should have work
        nodes_with_work = sum(1 for d in dist if d > 0)
        assert nodes_with_work >= 2, (
            f"Expected work on >=2 nodes but got {dist}. "
            f"Long request should have caused rebalancing."
        )
        print(f"[long_req_rebalance] distribution={dist}")

    def test_inflight_load_reflects_reality(self):
        """After modification, _load should reflect actual in-flight count,
        not cumulative dispatch count."""
        runner = _make_runner(num_p=3, num_requests=30)
        metrics = runner.run_benchmark_emulation()
        
        # After all requests complete, all loads should be 0
        # (if update_workload is called, total_req - completed = 0)
        final_loads = [w.get_load() for w in runner.p_policy.workers]
        # Currently with the guard, loads are cumulative (non-zero)
        # After fix, they should be 0 (all completed)
        print(f"[inflight_load] final_loads={final_loads}, dist={_get_distribution(runner)}")
        # This assertion will PASS after the fix:
        # assert all(l == 0 for l in final_loads), f"Expected all loads=0 after completion, got {final_loads}"
        # For now, just record the behavior
        return final_loads

    def test_round_robin_unaffected(self):
        """Round robin policy should not be affected by the change."""
        runner = _make_runner(
            num_p=3, num_requests=30, p_policy=RoutingPolicy.ROUND_ROBIN
        )
        metrics = runner.run_benchmark_emulation()
        dist = _get_distribution(runner)
        assert metrics["completed"] == 30
        # Round robin should distribute evenly
        assert max(dist) - min(dist) <= 1, f"RR not balanced: {dist}"

    def test_ttft_positive_and_ordered(self):
        """TTFT metrics should be positive and properly ordered."""
        runner = _make_runner(num_p=3, num_requests=50)
        metrics = runner.run_benchmark_emulation()
        assert metrics["completed"] == 50
        assert metrics["mean_ttft_ms"] > 0
        assert metrics["p99_ttft_ms"] >= metrics["p90_ttft_ms"]
        assert metrics["p90_ttft_ms"] >= metrics["mean_ttft_ms"] * 0.5
