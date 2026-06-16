import numpy
import random

from schedule_simulator.schedule_emulator.types import (
    BenchmarkConfig,
    SchedulerConfig,
    PlatformConfig,
    RouterConfig,
    RoutingPolicy,
)
from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner


random.seed(0)
numpy.random.seed(0)


def test_disagg_emulation():
    """Test prefill-decode disaggregated simulation with multiple instances."""
    benchmark_config = BenchmarkConfig(
        num_prompts=50,
        min_input_length=900,
        max_input_length=1100,
        min_output_length=250,
        max_output_length=350,
        max_concurrency=20,
        disable_tqdm=True,
    )

    p_scheduler_config = SchedulerConfig(
        "Qwen2.5-3B",
        enable_stats=True,
        scenario="disagg_prefill",
    )
    d_scheduler_config = SchedulerConfig(
        "Qwen2.5-3B",
        enable_stats=True,
        scenario="disagg_decode",
    )
    p_platform_config = PlatformConfig(
        device="A100-SXM4-80GB",
        memory_read_bandwidth_gb=64 / 4,
        disk_read_bandwidth_gb=15 / 8,
    )
    d_platform_config = PlatformConfig(
        device="H20",
        memory_read_bandwidth_gb=64 / 4,
        disk_read_bandwidth_gb=15 / 8,
    )
    router_config = RouterConfig(
        p_policy=RoutingPolicy.ROUND_ROBIN,
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
        num_p_instance=2,
        num_d_instance=3,
    )

    metrics = runner.run_benchmark_emulation()
    assert len(metrics) != 0, "metrics should not be empty"
    print(f"Disagg emulation: completed={metrics.get('completed')}, throughput={metrics.get('output_throughput', 0):.1f} tok/s")

    reps = runner.get_response_results()
    assert len(reps) != 0, "should have response results"

    stats = runner.get_request_cache_fetch_stats()
    assert len(stats) != 0 and len(stats[0]) != 0, "should have cache fetch stats"

    print(f"Disagg: {len(reps)} responses, {len(stats)} instance stat groups")


if __name__ == "__main__":
    test_disagg_emulation()
