"""
Tests for HiCache write policies and prefetch strategies.

Write policies (SimHiRadixCache, real token mode):
- write_back: only write L1->L2 on eviction, _inc_hit_count is no-op
- write_through: write L1->L2 on first hit (threshold=1)
- write_through_selective: write L1->L2 on second hit (threshold=2)

Prefetch strategies (HiRadixCache, statistical mode):
- best_effort: never block, schedule immediately
- wait_complete: block until all prefetch completes
- timeout: block until complete or timeout
"""
import numpy
import random

from schedule_simulator.schedule_emulator.types import (
    BenchmarkConfig,
    SchedulerConfig,
    PlatformConfig,
)
from schedule_simulator.schedule_emulator.run import BenchmarkRunner


DATASET_PATH = "/sgl-workspace/claude_workspace/schedule_simulator/tests/assets/dataset/multiturn_requests.jsonl"


# =============================================================================
# Helper: create runner with real token mode for write policy tests
# =============================================================================

def _make_real_token_runner(write_policy: str):
    random.seed(42)
    numpy.random.seed(42)

    benchmark_config = BenchmarkConfig(
        dataset_path=DATASET_PATH,
        num_prompts=20,
        disable_tqdm=True,
    )
    scheduler_config = SchedulerConfig(
        "Qwen2.5-3B",
        chunked_prefill_size=8192,
        hicache_storage_backend="hf3fs",
        hicache_write_policy=write_policy,
        schedule_policy="fcfs",
    )
    platform_config = PlatformConfig(
        device="H20",
        memory_read_bandwidth_gb=64 / 4,
        disk_read_bandwidth_gb=15 / 8,
    )

    runner = BenchmarkRunner(
        benchmark_config=benchmark_config,
        scheduler_config=scheduler_config,
        platform_config=platform_config,
        use_real_token=True,
    )
    return runner


def _count_backed_up_nodes(root_node) -> int:
    count = 0
    def walk(node):
        nonlocal count
        if node.backuped:
            count += 1
        for child in node.children.values():
            walk(child)
    walk(root_node)
    return count


def _get_max_hit_count(root_node) -> int:
    max_hc = 0
    def walk(node):
        nonlocal max_hc
        if node.hit_count > max_hc:
            max_hc = node.hit_count
        for child in node.children.values():
            walk(child)
    walk(root_node)
    return max_hc


# =============================================================================
# Write Policy Tests
# =============================================================================

def test_write_through():
    """write_through: threshold=1, writes on first hit."""
    runner = _make_real_token_runner("write_through")
    cache = runner.scheduler_emulator.tree_cache

    assert cache.cache_controller.write_policy == "write_through"
    assert cache.write_through_threshold == 1

    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] > 0

    backed_up = _count_backed_up_nodes(cache.root_node)
    max_hc = _get_max_hit_count(cache.root_node)
    assert backed_up > 0, "write_through should backup nodes on first hit"
    assert max_hc >= 1, "hit_count should be incremented"
    print(f"[write_through] completed={metrics['completed']}, backed_up={backed_up}, max_hit_count={max_hc}")


def test_write_through_selective():
    """write_through_selective: threshold=2, writes on second hit."""
    runner = _make_real_token_runner("write_through_selective")
    cache = runner.scheduler_emulator.tree_cache

    assert cache.cache_controller.write_policy == "write_through_selective"
    assert cache.write_through_threshold == 2

    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] > 0

    backed_up = _count_backed_up_nodes(cache.root_node)
    print(f"[write_through_selective] completed={metrics['completed']}, backed_up={backed_up}")


def test_write_back():
    """write_back: _inc_hit_count is no-op. hit_count stays 0. Backup only on eviction."""
    runner = _make_real_token_runner("write_back")
    cache = runner.scheduler_emulator.tree_cache

    assert cache.cache_controller.write_policy == "write_back"

    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] > 0

    max_hc = _get_max_hit_count(cache.root_node)
    assert max_hc == 0, f"write_back should never increment hit_count, got max={max_hc}"
    print(f"[write_back] completed={metrics['completed']}, max_hit_count={max_hc}")


def test_write_policies_ordering():
    """write_through should backup >= write_through_selective >= write_back (proactive backups only)."""
    runner_wt = _make_real_token_runner("write_through")
    runner_wt.run_benchmark_emulation()
    wt_backed = _count_backed_up_nodes(runner_wt.scheduler_emulator.tree_cache.root_node)

    runner_wts = _make_real_token_runner("write_through_selective")
    runner_wts.run_benchmark_emulation()
    wts_backed = _count_backed_up_nodes(runner_wts.scheduler_emulator.tree_cache.root_node)

    print(f"[ordering] write_through={wt_backed}, selective={wts_backed}")
    assert wt_backed >= wts_backed, (
        f"write_through ({wt_backed}) should backup >= write_through_selective ({wts_backed})"
    )


def test_write_policy_config_passthrough():
    """Verify SchedulerConfig.hicache_write_policy propagates to SimHiRadixCache correctly."""
    for policy in ["write_through", "write_through_selective", "write_back"]:
        runner = _make_real_token_runner(policy)
        cache = runner.scheduler_emulator.tree_cache
        assert cache.cache_controller.write_policy == policy, (
            f"Config '{policy}' did not propagate: got '{cache.cache_controller.write_policy}'"
        )
        if policy == "write_through":
            assert cache.write_through_threshold == 1
        elif policy == "write_through_selective":
            assert cache.write_through_threshold == 2
    print("[config_passthrough] All 3 write policies correctly propagated from SchedulerConfig")


# =============================================================================
# Prefetch Strategy Tests (statistical mode, HiRadixCache)
# =============================================================================

def _make_stat_mode_runner(prefetch_policy: str):
    random.seed(42)
    numpy.random.seed(42)

    benchmark_config = BenchmarkConfig(
        num_prompts=50,
        min_input_length=500,
        max_input_length=1500,
        min_output_length=100,
        max_output_length=300,
        min_prefix_disk_hit_rate=0.3,
        max_prefix_disk_hit_rate=0.5,
        min_prefix_host_hit_rate=0.1,
        max_prefix_host_hit_rate=0.2,
        disable_tqdm=True,
    )
    scheduler_config = SchedulerConfig(
        "Qwen2.5-3B",
        chunked_prefill_size=4096,
        hicache_storage_backend="hf3fs",
        hicache_storage_prefetch_policy=prefetch_policy,
        schedule_policy="fcfs",
        max_running_requests=10,
    )
    platform_config = PlatformConfig(
        device="H20",
        memory_read_bandwidth_gb=64 / 4,
        disk_read_bandwidth_gb=15 / 8,
    )

    runner = BenchmarkRunner(
        benchmark_config=benchmark_config,
        scheduler_config=scheduler_config,
        platform_config=platform_config,
    )
    return runner


def test_prefetch_best_effort():
    """best_effort: never blocks, requests scheduled immediately."""
    runner = _make_stat_mode_runner("best_effort")
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 50
    assert metrics["mean_ttft_ms"] > 0
    print(f"[best_effort] TTFT mean={metrics['mean_ttft_ms']:.1f}ms p99={metrics['p99_ttft_ms']:.1f}ms")


def test_prefetch_wait_complete():
    """wait_complete: blocks until all disk->host prefetch finishes."""
    runner = _make_stat_mode_runner("wait_complete")
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 50
    assert metrics["mean_ttft_ms"] > 0
    print(f"[wait_complete] TTFT mean={metrics['mean_ttft_ms']:.1f}ms p99={metrics['p99_ttft_ms']:.1f}ms")


def test_prefetch_timeout():
    """timeout: blocks until prefetch completes or timeout is hit."""
    runner = _make_stat_mode_runner("timeout")
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 50
    assert metrics["mean_ttft_ms"] > 0
    print(f"[timeout] TTFT mean={metrics['mean_ttft_ms']:.1f}ms p99={metrics['p99_ttft_ms']:.1f}ms")


def test_prefetch_strategies_differ():
    """Different prefetch strategies should produce measurably different TTFT characteristics."""
    be_runner = _make_stat_mode_runner("best_effort")
    be_metrics = be_runner.run_benchmark_emulation()

    wc_runner = _make_stat_mode_runner("wait_complete")
    wc_metrics = wc_runner.run_benchmark_emulation()

    to_runner = _make_stat_mode_runner("timeout")
    to_metrics = to_runner.run_benchmark_emulation()

    print(f"[compare] best_effort={be_metrics['mean_ttft_ms']:.1f}ms, "
          f"timeout={to_metrics['mean_ttft_ms']:.1f}ms, "
          f"wait_complete={wc_metrics['mean_ttft_ms']:.1f}ms")

    # wait_complete and timeout should produce same result (timeout > actual prefetch time in this config)
    assert abs(wc_metrics["mean_ttft_ms"] - to_metrics["mean_ttft_ms"]) < 1.0, (
        "timeout and wait_complete should behave similarly when timeout > prefetch duration"
    )
    # best_effort vs wait_complete should differ (different scheduling behavior)
    assert be_metrics["mean_ttft_ms"] != wc_metrics["mean_ttft_ms"], (
        "best_effort and wait_complete should produce different TTFT"
    )


if __name__ == "__main__":
    print("=== Write Policy Tests ===")
    test_write_through()
    test_write_through_selective()
    test_write_back()
    test_write_policies_ordering()
    test_write_policy_config_passthrough()
    print("\n=== Prefetch Strategy Tests ===")
    test_prefetch_best_effort()
    test_prefetch_wait_complete()
    test_prefetch_timeout()
    test_prefetch_strategies_differ()
