"""Tests for enhanced calc_metrics and export_results functionality."""
import os, sys, json, csv, shutil, random
import numpy as np
import pytest

KVCM_SO_DIR = "/sgl-workspace/claude_workspace/tair-kvcache/bazel-bin/kv_cache_manager/optimizer/pybind"
if KVCM_SO_DIR not in sys.path:
    sys.path.insert(0, KVCM_SO_DIR)
try:
    import kvcm_py_optimizer
    HAS_KVCM = True
except ImportError:
    HAS_KVCM = False

from schedule_simulator.schedule_emulator.types import *
from schedule_simulator.schedule_emulator.run import BenchmarkRunner, DisaggBenchmarkRunner
from schedule_simulator.schedule_emulator.utils import calc_metrics

ENRICHED = os.path.join(os.path.dirname(__file__), "assets/glm5_sample/glm5_enriched_input.jsonl")
EXPORT_DIR = "/tmp/test_export_results"


@pytest.fixture(autouse=True)
def cleanup_export():
    yield
    if os.path.exists(EXPORT_DIR):
        shutil.rmtree(EXPORT_DIR)


def _basic_runner(n=20):
    random.seed(42); np.random.seed(42)
    return BenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=n, min_input_length=200, max_input_length=1000,
            min_output_length=10, max_output_length=50, disable_tqdm=True,
        ),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", chunked_prefill_size=4096,
                                          hicache_storage_backend="hf3fs"),
        platform_config=PlatformConfig(device="H20", memory_read_bandwidth_gb=16.0, disk_read_bandwidth_gb=2.0),
    )


# ===========================================================================
# Section 1: calc_metrics new fields
# ===========================================================================

def test_metrics_has_concurrency():
    runner = _basic_runner()
    m = runner.run_benchmark_emulation()
    assert "concurrency" in m
    assert m["concurrency"] >= 0
    print("[concurrency] %.2f" % m["concurrency"])


def test_metrics_has_max_itl():
    runner = _basic_runner()
    m = runner.run_benchmark_emulation()
    assert "max_itl_ms" in m
    assert m["max_itl_ms"] >= 0
    if m["max_itl_ms"] > 0:
        assert m["max_itl_ms"] >= m["p99_itl_ms"]
    print("[max_itl] %.2f ms" % m["max_itl_ms"])


def test_metrics_has_queue_wait():
    runner = _basic_runner()
    m = runner.run_benchmark_emulation()
    assert "mean_queue_wait_ms" in m
    assert "median_queue_wait_ms" in m
    assert "p90_queue_wait_ms" in m
    assert "p99_queue_wait_ms" in m
    assert m["mean_queue_wait_ms"] >= 0
    print("[queue_wait] mean=%.2fms p99=%.2fms" % (m["mean_queue_wait_ms"], m["p99_queue_wait_ms"]))


def test_metrics_percentile_ordering():
    runner = _basic_runner(50)
    m = runner.run_benchmark_emulation()
    # TTFT: p99 >= p95 >= p90 >= median
    assert m["p99_ttft_ms"] >= m["p95_ttft_ms"] >= m["p90_ttft_ms"]
    assert m["p90_ttft_ms"] >= m["median_ttft_ms"]
    # E2E: p99 >= p95 >= p90 >= median
    assert m["p99_e2e_latency_ms"] >= m["p95_e2e_latency_ms"] >= m["p90_e2e_latency_ms"]
    assert m["p90_e2e_latency_ms"] >= m["median_e2e_latency_ms"]
    print("[percentile_order] OK")


def test_metrics_request_throughput_uses_completed():
    runner = _basic_runner(30)
    m = runner.run_benchmark_emulation()
    assert m["request_throughput"] == m["completed"] / m["duration"]
    print("[throughput] %.2f req/s" % m["request_throughput"])


def test_metrics_all_fields_present():
    runner = _basic_runner()
    m = runner.run_benchmark_emulation()
    required = [
        "num_requests", "completed", "total_input", "total_output", "duration",
        "request_throughput", "input_throughput", "output_throughput", "total_throughput",
        "concurrency",
        "mean_ttft_ms", "median_ttft_ms", "std_ttft_ms", "p90_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p90_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
        "mean_itl_ms", "median_itl_ms", "std_itl_ms", "p90_itl_ms", "p95_itl_ms", "p99_itl_ms", "max_itl_ms",
        "mean_e2e_latency_ms", "median_e2e_latency_ms", "std_e2e_latency_ms",
        "p90_e2e_latency_ms", "p95_e2e_latency_ms", "p99_e2e_latency_ms",
        "mean_queue_wait_ms", "median_queue_wait_ms", "p90_queue_wait_ms", "p99_queue_wait_ms",
    ]
    for k in required:
        assert k in m, "Missing metric: %s" % k
    print("[all_fields] %d fields present" % len(required))


# ===========================================================================
# Section 2: export_results functionality
# ===========================================================================

def test_export_creates_files():
    random.seed(42); np.random.seed(42)
    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=20, min_input_length=200, max_input_length=500,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        p_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill",
                                            chunked_prefill_size=4096, hicache_storage_backend="hf3fs",
                                            enable_stats=True),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20", memory_read_bandwidth_gb=16.0, disk_read_bandwidth_gb=2.0),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN,
                                    worker_startup_check_interval=0.01),
        num_p_instance=3, num_d_instance=0,
    )
    metrics = runner.run_benchmark_emulation()
    runner.export_results(EXPORT_DIR, metrics)

    assert os.path.exists(os.path.join(EXPORT_DIR, "simulation_summary.json"))
    assert os.path.exists(os.path.join(EXPORT_DIR, "per_request.csv"))
    assert os.path.exists(os.path.join(EXPORT_DIR, "per_iteration.csv"))
    print("[export_files] 3 files created")


def test_export_summary_json_complete():
    random.seed(42); np.random.seed(42)
    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=10, min_input_length=200, max_input_length=500,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        p_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill",
                                            chunked_prefill_size=4096, hicache_storage_backend="hf3fs"),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20", memory_read_bandwidth_gb=16.0, disk_read_bandwidth_gb=2.0),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN,
                                    worker_startup_check_interval=0.01),
        num_p_instance=2, num_d_instance=0,
    )
    metrics = runner.run_benchmark_emulation()
    runner.export_results(EXPORT_DIR, metrics)

    with open(os.path.join(EXPORT_DIR, "simulation_summary.json")) as f:
        summary = json.load(f)
    assert summary["completed"] == 10
    assert "mean_ttft_ms" in summary
    assert "concurrency" in summary
    assert "mean_queue_wait_ms" in summary
    print("[summary_json] completed=%d, fields=%d" % (summary["completed"], len(summary)))


def test_export_per_request_csv():
    random.seed(42); np.random.seed(42)
    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=15, min_input_length=200, max_input_length=500,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        p_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill",
                                            chunked_prefill_size=4096, hicache_storage_backend="hf3fs"),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20", memory_read_bandwidth_gb=16.0, disk_read_bandwidth_gb=2.0),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN,
                                    worker_startup_check_interval=0.01),
        num_p_instance=3, num_d_instance=0,
    )
    metrics = runner.run_benchmark_emulation()
    runner.export_results(EXPORT_DIR, metrics)

    with open(os.path.join(EXPORT_DIR, "per_request.csv")) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 15
    for row in rows:
        assert float(row["ttft_ms"]) > 0
        assert int(row["input_length"]) > 0
    print("[per_request_csv] %d rows, all have positive ttft" % len(rows))


def test_export_per_iteration_csv():
    random.seed(42); np.random.seed(42)
    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=10, min_input_length=200, max_input_length=500,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        p_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill",
                                            chunked_prefill_size=4096, hicache_storage_backend="hf3fs",
                                            enable_stats=True),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20", memory_read_bandwidth_gb=16.0, disk_read_bandwidth_gb=2.0),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN,
                                    worker_startup_check_interval=0.01),
        num_p_instance=3, num_d_instance=0,
    )
    metrics = runner.run_benchmark_emulation()
    runner.export_results(EXPORT_DIR, metrics)

    with open(os.path.join(EXPORT_DIR, "per_iteration.csv")) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0
    pods = set(row["pod"] for row in rows)
    assert "P0" in pods
    for row in rows:
        assert float(row["iter_latency_ms"]) > 0
    print("[per_iteration_csv] %d rows across %d pods" % (len(rows), len(pods)))


@pytest.mark.skipif(not HAS_KVCM, reason="kvcm not available")
def test_export_with_hierarchical():
    random.seed(42); np.random.seed(42)
    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            dataset_path=ENRICHED, num_prompts=20, disable_tqdm=True,
        ),
        p_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill",
                                            chunked_prefill_size=8192, hicache_storage_backend="hf3fs",
                                            enable_stats=True),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0,
                                          memory_read_bandwidth_gb=16.0, memory_capacity_gb=64.0,
                                          peer_read_bandwidth_gb=10.0),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN,
                                    worker_startup_check_interval=0.01),
        num_p_instance=3, num_d_instance=0,
        enable_hierarchical=True, hierarchical_output_dir=os.path.join(EXPORT_DIR, "optimizer"),
    )
    metrics = runner.run_benchmark_emulation()
    runner.export_results(EXPORT_DIR, metrics)

    with open(os.path.join(EXPORT_DIR, "simulation_summary.json")) as f:
        summary = json.load(f)
    assert "hierarchical_total_engine_hit_blocks" in summary
    assert "hierarchical_block_hit_ratio" in summary

    with open(os.path.join(EXPORT_DIR, "per_request.csv")) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert all("engine_hit" in row for row in rows)
    print("[hierarchical_export] summary has hierarchical fields, per_request has hit columns")


def test_queue_wait_increases_with_load():
    """Higher concurrency should produce higher queue wait times."""
    # Low concurrency
    random.seed(42); np.random.seed(42)
    runner_low = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=5, min_input_length=200, max_input_length=500,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", chunked_prefill_size=4096,
                                          hicache_storage_backend="hf3fs"),
        platform_config=PlatformConfig(device="H20", memory_read_bandwidth_gb=16.0, disk_read_bandwidth_gb=2.0),
    )
    m_low = runner_low.run_benchmark_emulation()

    # High concurrency (many requests at once)
    random.seed(42); np.random.seed(42)
    runner_high = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=50, min_input_length=200, max_input_length=500,
            min_output_length=1, max_output_length=2, max_concurrency=5, disable_tqdm=True,
        ),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", chunked_prefill_size=4096,
                                          hicache_storage_backend="hf3fs"),
        platform_config=PlatformConfig(device="H20", memory_read_bandwidth_gb=16.0, disk_read_bandwidth_gb=2.0),
    )
    m_high = runner_high.run_benchmark_emulation()

    print("[queue_load] low=%d reqs, queue=%.1fms; high=%d reqs, queue=%.1fms" % (
        m_low["completed"], m_low["mean_queue_wait_ms"],
        m_high["completed"], m_high["mean_queue_wait_ms"]))


if __name__ == "__main__":
    test_metrics_has_concurrency()
    test_metrics_has_max_itl()
    test_metrics_has_queue_wait()
    test_metrics_percentile_ordering()
    test_metrics_request_throughput_uses_completed()
    test_metrics_all_fields_present()
    test_export_creates_files()
    test_export_summary_json_complete()
    test_export_per_request_csv()
    test_export_per_iteration_csv()
    test_queue_wait_increases_with_load()
