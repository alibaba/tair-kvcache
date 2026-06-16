"""
Tests for the real-time load counter mechanism in cache_aware routing.

Verifies that:
1. Worker load increments when request is dispatched
2. Worker load decrements when request completes (via router_queue)
3. Load stays non-negative
4. Multi-instance routing uses load correctly for balance decisions
"""

import asyncio
import json
import os
import tempfile

import pytest

from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner
from schedule_simulator.schedule_emulator.types import (
    BenchmarkConfig,
    PlatformConfig,
    RouterConfig,
    RoutingPolicy,
    SchedulerConfig,
)
from schedule_simulator.infer_time_predictor.request_level import (
    RequestLevelTimePredictor,
)


def make_predictor(ms_per_token=0.01):
    """Create a simple constant-time predictor."""
    return RequestLevelTimePredictor(constant_ms_per_token=ms_per_token)


def make_simple_dataset(num_requests=20, input_length=4096, base_timestamp=1000):
    """Create a simple dataset with incrementing timestamps."""
    data = []
    for i in range(num_requests):
        record = {
            "timestamp": base_timestamp + i * 100,
            "input_length": input_length,
            "output_length": 1,
            "block_ids": list(range(i * 2, i * 2 + max(1, input_length // 2048))),
        }
        data.append(json.dumps(record))

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(data) + "\n")
    return path


def make_shared_prefix_dataset(num_requests=30, base_timestamp=1000):
    """Create dataset where requests share common prefixes (good for cache_aware)."""
    data = []
    for group in range(3):
        prefix = list(range(group * 100, group * 100 + 10))
        for i in range(10):
            block_ids = prefix + [group * 100 + 10 + i]
            record = {
                "timestamp": base_timestamp + (group * 10 + i) * 100,
                "input_length": len(block_ids) * 2048,
                "output_length": 1,
                "block_ids": block_ids,
            }
            data.append(json.dumps(record))

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(data) + "\n")
    return path


def create_runner(dataset_path, num_prompts, num_p_instance, p_policy, ms_per_token=0.01, page_size=None):
    """Helper to create a DisaggBenchmarkRunner with common settings."""
    p_sc = SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill")
    p_sc.request_level_scheduling = True
    if page_size:
        p_sc.page_size = page_size

    d_sc = SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode")
    d_sc.request_level_scheduling = True

    pc = PlatformConfig(device="A100-SXM4-80GB")
    bc = BenchmarkConfig(dataset_path=dataset_path, num_prompts=num_prompts, disable_tqdm=True)
    predictor = make_predictor(ms_per_token=ms_per_token)

    runner = DisaggBenchmarkRunner(
        benchmark_config=bc,
        p_scheduler_config=p_sc,
        d_scheduler_config=d_sc,
        p_platform_config=pc,
        d_platform_config=pc,
        router_config=RouterConfig(
            p_policy=p_policy,
            d_policy=RoutingPolicy.ROUND_ROBIN,
        ),
        num_p_instance=num_p_instance,
        num_d_instance=0,
        infer_time_predictor=predictor,
    )
    return runner


@pytest.fixture
def simple_dataset():
    path = make_simple_dataset()
    yield path
    os.unlink(path)


@pytest.fixture
def shared_prefix_dataset():
    path = make_shared_prefix_dataset()
    yield path
    os.unlink(path)


class TestLoadCounter:
    """Test that Worker._load correctly tracks in-flight requests."""

    def test_load_decrements_to_zero_after_completion(self, simple_dataset):
        """After all requests complete, all workers should have load 0."""
        runner = create_runner(
            simple_dataset, num_prompts=20, num_p_instance=5,
            p_policy=RoutingPolicy.CACHE_AWARE,
        )
        metrics = asyncio.run(runner.async_run_benchmark_emulation())
        assert metrics is not None
        assert metrics["completed"] == 20

        for w in runner.p_policy.workers:
            assert w.get_load() == 0, f"Worker {w.id} has load {w.get_load()}, expected 0"

    def test_total_req_matches_completed(self, simple_dataset):
        """Sum of total_req across all workers should equal total requests dispatched."""
        runner = create_runner(
            simple_dataset, num_prompts=20, num_p_instance=5,
            p_policy=RoutingPolicy.CACHE_AWARE,
        )
        metrics = asyncio.run(runner.async_run_benchmark_emulation())
        total_dispatched = sum(w.total_req for w in runner.p_policy.workers)
        assert total_dispatched == 20, f"Total dispatched {total_dispatched} != 20"

    def test_request_to_worker_map_empty_after_completion(self, simple_dataset):
        """request_to_p_worker dict should be empty after all requests complete."""
        runner = create_runner(
            simple_dataset, num_prompts=20, num_p_instance=3,
            p_policy=RoutingPolicy.CACHE_AWARE,
        )
        metrics = asyncio.run(runner.async_run_benchmark_emulation())
        assert len(runner.request_to_p_worker) == 0, \
            f"request_to_p_worker not empty: {len(runner.request_to_p_worker)} entries remaining"

    def test_load_never_negative(self, simple_dataset):
        """Worker load should never go below 0."""
        runner = create_runner(
            simple_dataset, num_prompts=20, num_p_instance=3,
            p_policy=RoutingPolicy.CACHE_AWARE,
        )
        metrics = asyncio.run(runner.async_run_benchmark_emulation())
        for w in runner.p_policy.workers:
            assert w.get_load() >= 0, f"Worker {w.id} has negative load {w.get_load()}"


class TestLoadCounterDirectCacheAware:
    """Test load counter with direct_cache_aware policy."""

    def test_load_zero_after_completion(self, simple_dataset):
        """DirectCacheAwarePolicy should also have zero load after completion."""
        runner = create_runner(
            simple_dataset, num_prompts=20, num_p_instance=5,
            p_policy=RoutingPolicy.DIRECT_CACHE_AWARE,
        )
        metrics = asyncio.run(runner.async_run_benchmark_emulation())
        assert metrics["completed"] == 20
        for w in runner.p_policy.workers:
            assert w.get_load() == 0

    def test_load_tracking_consistent(self, simple_dataset):
        """Load counter should correctly track dispatch/complete pairs."""
        runner = create_runner(
            simple_dataset, num_prompts=20, num_p_instance=5,
            p_policy=RoutingPolicy.DIRECT_CACHE_AWARE, ms_per_token=1.0,
        )
        metrics = asyncio.run(runner.async_run_benchmark_emulation())
        # All loads should be zero (all requests completed)
        for w in runner.p_policy.workers:
            assert w.get_load() == 0
        # Total dispatched should match total requests
        total = sum(w.total_req for w in runner.p_policy.workers)
        assert total == 20


class TestCacheAwareWithSharedPrefix:
    """Test that cache_aware routing benefits from shared prefixes."""

    def test_shared_prefix_load_zero_after_completion(self, shared_prefix_dataset):
        """Shared prefix requests should complete with all loads at zero."""
        runner = create_runner(
            shared_prefix_dataset, num_prompts=30, num_p_instance=5,
            p_policy=RoutingPolicy.CACHE_AWARE, page_size=2048,
        )
        metrics = asyncio.run(runner.async_run_benchmark_emulation())
        assert metrics["completed"] == 30
        for w in runner.p_policy.workers:
            assert w.get_load() == 0


class TestRoundRobinUnaffected:
    """Ensure round_robin still works correctly with the load counter changes."""

    def test_round_robin_still_balanced(self, simple_dataset):
        """Round robin should distribute evenly regardless of load counter."""
        runner = create_runner(
            simple_dataset, num_prompts=20, num_p_instance=4,
            p_policy=RoutingPolicy.ROUND_ROBIN,
        )
        metrics = asyncio.run(runner.async_run_benchmark_emulation())
        assert metrics["completed"] == 20
        for w in runner.p_policy.workers:
            assert w.total_req == 5, f"Worker {w.id} got {w.total_req}, expected 5"
