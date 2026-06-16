import time
import os
import numpy
import random

from schedule_simulator.schedule_emulator.types import (
    BenchmarkConfig,
    SchedulerConfig,
    PlatformConfig,
)
from schedule_simulator.schedule_emulator.run import BenchmarkRunner


random.seed(0)
numpy.random.seed(0)

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def test_random_request_emulation():
    """Test basic emulation with randomly generated requests."""
    benchmark_config = BenchmarkConfig(
        num_prompts=50,
        min_input_length=900,
        max_input_length=1100,
        min_output_length=250,
        max_output_length=350,
        max_concurrency=20,
        min_prefix_disk_hit_rate=0.3,
        max_prefix_disk_hit_rate=0.5,
        min_prefix_host_hit_rate=0.2,
        max_prefix_host_hit_rate=0.3,
        disable_tqdm=True,
    )
    scheduler_config = SchedulerConfig(
        "Qwen2.5-3B",
        chunked_prefill_size=1000,
        hicache_storage_backend="hf3fs",
        schedule_policy="fcfs",
        enable_stats=True,
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

    start = time.time()
    metrics = runner.run_benchmark_emulation(benchmark_config)
    elapsed = time.time() - start

    assert len(metrics) != 0, "metrics should not be empty"
    assert metrics["completed"] == 50, f"expected 50 completed, got {metrics['completed']}"
    assert metrics["mean_ttft_ms"] > 0, "TTFT should be positive"
    assert metrics["mean_tpot_ms"] > 0, "TPOT should be positive"
    assert metrics["output_throughput"] > 0, "output throughput should be positive"

    iter_stats = runner.get_iteration_stats()
    assert len(iter_stats) > 0, "should have iteration stats"
    assert all(
        len(s.request_stats) <= scheduler_config.max_running_requests
        for s in iter_stats
    ), "running requests should not exceed max_running_requests"

    response_results = runner.get_response_results()
    assert len(response_results) == 50

    print(f"Random emulation: {elapsed:.2f}s, throughput={metrics['output_throughput']:.1f} tok/s")


def test_dataset_file_emulation():
    """Test emulation loading requests from JSONL dataset file."""
    dataset_path = os.path.join(ASSETS_DIR, "dataset", "prefix_cache_requests.jsonl")
    assert os.path.exists(dataset_path), f"test dataset not found: {dataset_path}"

    benchmark_config = BenchmarkConfig(
        dataset_path=dataset_path,
        num_prompts=2,
        disable_tqdm=True,
    )
    scheduler_config = SchedulerConfig(
        "Qwen2.5-3B",
        chunked_prefill_size=1000,
        hicache_storage_backend="hf3fs",
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
    )
    metrics = runner.run_benchmark_emulation(benchmark_config)

    assert len(metrics) != 0, "metrics should not be empty"
    assert metrics["completed"] == 2
    print(f"Dataset emulation: completed={metrics['completed']}, throughput={metrics['output_throughput']:.1f} tok/s")


def test_runner_reuse():
    """Test running multiple benchmarks with a single runner to verify state reset."""
    benchmark_config = BenchmarkConfig(
        num_prompts=20,
        min_input_length=500,
        max_input_length=600,
        min_output_length=100,
        max_output_length=150,
        disable_tqdm=True,
    )
    scheduler_config = SchedulerConfig("Qwen2.5-3B")
    platform_config = PlatformConfig(device="H20")

    runner = BenchmarkRunner(
        benchmark_config=benchmark_config,
        scheduler_config=scheduler_config,
        platform_config=platform_config,
    )

    for i in range(2):
        metrics = runner.run_benchmark_emulation(benchmark_config)
        assert metrics["completed"] == 20, f"run {i}: expected 20 completed"

    print("Runner reuse: 2 consecutive runs passed")


if __name__ == "__main__":
    test_random_request_emulation()
    test_dataset_file_emulation()
    test_runner_reuse()
