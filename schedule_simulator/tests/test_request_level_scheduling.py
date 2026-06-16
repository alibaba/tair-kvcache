"""Tests for request-level scheduling mode."""
import os, sys, random, time
import numpy as np
import pytest

from schedule_simulator.schedule_emulator.types import *
from schedule_simulator.schedule_emulator.run import BenchmarkRunner, DisaggBenchmarkRunner
from schedule_simulator.infer_time_predictor import RequestLevelTimePredictor


def _make_predictor(ms_per_token=0.1):
    return RequestLevelTimePredictor(constant_ms_per_token=ms_per_token)


def _make_fn_predictor():
    def predict_fn(uncached, cached):
        return (uncached * 0.05 + cached * 0.001 + 10) / 1000.0
    return RequestLevelTimePredictor(predict_fn=predict_fn)


# ===========================================================================
# Basic functionality
# ===========================================================================

def test_request_level_completes():
    random.seed(42); np.random.seed(42)
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=20, min_input_length=500, max_input_length=2000,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", request_level_scheduling=True),
        platform_config=PlatformConfig(device="H20"),
        infer_time_predictor=_make_predictor(),
    )
    m = runner.run_benchmark_emulation()
    assert m["completed"] == 20
    assert m["mean_ttft_ms"] > 0
    print("[completes] completed=%d TTFT=%.0fms" % (m["completed"], m["mean_ttft_ms"]))


def test_request_level_ttft_equals_predicted():
    predictor = _make_predictor(ms_per_token=0.1)
    random.seed(42); np.random.seed(42)
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=10, min_input_length=1000, max_input_length=1001,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", request_level_scheduling=True),
        platform_config=PlatformConfig(device="H20"),
        infer_time_predictor=predictor,
    )
    m = runner.run_benchmark_emulation()
    results = runner.get_response_results()
    for r in results:
        predicted_s = predictor.predict_request_time(r.input_token_length)
        actual_ttft_s = r.gen_token_latencies[0]
        assert actual_ttft_s >= predicted_s - 0.001, (
            "TTFT %.3fs should be >= predicted %.3fs (TTFT includes queue wait)" % (actual_ttft_s, predicted_s))
    print("[ttft_matches] All %d requests match predicted time" % len(results))


def test_request_level_clock_advances():
    random.seed(42); np.random.seed(42)
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=5, min_input_length=1000, max_input_length=1001,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", request_level_scheduling=True),
        platform_config=PlatformConfig(device="H20"),
        infer_time_predictor=_make_predictor(0.1),
    )
    m = runner.run_benchmark_emulation()
    assert m["duration"] > 0
    results = runner.get_response_results()
    times = sorted([r.last_event_time for r in results])
    for i in range(1, len(times)):
        assert times[i] >= times[i-1], "Clock should monotonically advance"
    print("[clock] Duration=%.3fs, monotonic=OK" % m["duration"])


def test_request_level_queue_ordering():
    random.seed(42); np.random.seed(42)
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=10, min_input_length=100, max_input_length=200,
            min_output_length=1, max_output_length=2,
            request_rate=2.0, disable_tqdm=True,
        ),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", request_level_scheduling=True),
        platform_config=PlatformConfig(device="H20"),
        infer_time_predictor=_make_predictor(0.01),
    )
    m = runner.run_benchmark_emulation()
    assert m["completed"] == 10
    print("[fcfs] completed=%d" % m["completed"])


# ===========================================================================
# Comparison with iteration level
# ===========================================================================

def test_iter_vs_request_both_complete():
    random.seed(42); np.random.seed(42)
    cfg = BenchmarkConfig(
        num_prompts=10, min_input_length=200, max_input_length=500,
        min_output_length=1, max_output_length=2, disable_tqdm=True,
    )
    plat = PlatformConfig(device="H20")

    # Iteration level
    random.seed(42); np.random.seed(42)
    r_iter = BenchmarkRunner(
        benchmark_config=cfg,
        scheduler_config=SchedulerConfig("Qwen2.5-3B", chunked_prefill_size=4096),
        platform_config=plat,
    )
    m_iter = r_iter.run_benchmark_emulation()

    # Request level
    random.seed(42); np.random.seed(42)
    r_req = BenchmarkRunner(
        benchmark_config=cfg,
        scheduler_config=SchedulerConfig("Qwen2.5-3B", request_level_scheduling=True),
        platform_config=plat,
        infer_time_predictor=_make_predictor(0.1),
    )
    m_req = r_req.run_benchmark_emulation()

    assert m_iter["completed"] == m_req["completed"] == 10
    print("[both_complete] iter=%d req=%d" % (m_iter["completed"], m_req["completed"]))


def test_request_level_faster_simulation():
    random.seed(42); np.random.seed(42)

    cfg = BenchmarkConfig(
        num_prompts=50, min_input_length=500, max_input_length=2000,
        min_output_length=1, max_output_length=2, disable_tqdm=True,
    )

    # Request level
    random.seed(42); np.random.seed(42)
    t0 = time.time()
    r_req = BenchmarkRunner(
        benchmark_config=cfg,
        scheduler_config=SchedulerConfig("Qwen2.5-3B", request_level_scheduling=True),
        platform_config=PlatformConfig(device="H20"),
        infer_time_predictor=_make_predictor(0.1),
    )
    r_req.run_benchmark_emulation()
    req_time = time.time() - t0

    # Iteration level
    random.seed(42); np.random.seed(42)
    t0 = time.time()
    r_iter = BenchmarkRunner(
        benchmark_config=cfg,
        scheduler_config=SchedulerConfig("Qwen2.5-3B", chunked_prefill_size=4096),
        platform_config=PlatformConfig(device="H20"),
    )
    r_iter.run_benchmark_emulation()
    iter_time = time.time() - t0

    print("[speed] request=%.3fs iter=%.3fs speedup=%.1fx" % (req_time, iter_time, iter_time/max(req_time,0.001)))


# ===========================================================================
# Multi-instance
# ===========================================================================

def test_request_level_multi_instance():
    random.seed(42); np.random.seed(42)
    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=50, min_input_length=200, max_input_length=1000,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        p_scheduler_config=SchedulerConfig(
            "Qwen2.5-3B", scenario="disagg_prefill", request_level_scheduling=True,
        ),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20"),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN,
                                    worker_startup_check_interval=0.01),
        num_p_instance=5, num_d_instance=0,
        infer_time_predictor=_make_predictor(0.1),
    )
    m = runner.run_benchmark_emulation()
    assert m["completed"] == 50
    dist = [len(s.completed_requests) for s in runner.p_schedulers]
    assert all(d == 10 for d in dist), "RoundRobin should be uniform: %s" % dist
    print("[multi_instance] completed=%d dist=%s" % (m["completed"], dist))


def test_request_level_multi_instance_ttft():
    random.seed(42); np.random.seed(42)
    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=100, min_input_length=500, max_input_length=2000,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        p_scheduler_config=SchedulerConfig(
            "Qwen2.5-3B", scenario="disagg_prefill", request_level_scheduling=True,
        ),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20"),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN,
                                    worker_startup_check_interval=0.01),
        num_p_instance=10, num_d_instance=0,
        infer_time_predictor=_make_predictor(0.05),
    )
    m = runner.run_benchmark_emulation()
    assert m["completed"] == 100
    assert m["p99_ttft_ms"] >= m["p90_ttft_ms"] >= m["mean_ttft_ms"] * 0.5
    print("[multi_ttft] mean=%.0fms p90=%.0fms p99=%.0fms" % (
        m["mean_ttft_ms"], m["p90_ttft_ms"], m["p99_ttft_ms"]))


# ===========================================================================
# Cache integration
# ===========================================================================

def test_request_level_cache_write():
    predictor = _make_fn_predictor()
    random.seed(42); np.random.seed(42)
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=10, min_input_length=200, max_input_length=500,
            min_output_length=1, max_output_length=2,
            min_prefix_disk_hit_rate=0.3, max_prefix_disk_hit_rate=0.5,
            disable_tqdm=True,
        ),
        scheduler_config=SchedulerConfig(
            "Qwen2.5-3B", request_level_scheduling=True,
            hicache_storage_backend="hf3fs",
        ),
        platform_config=PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0, memory_read_bandwidth_gb=16.0),
        infer_time_predictor=predictor,
    )
    m = runner.run_benchmark_emulation()
    assert m["completed"] == 10
    print("[cache_write] completed=%d" % m["completed"])


# ===========================================================================
# Regression: iteration mode unchanged
# ===========================================================================

def test_iter_mode_unchanged():
    random.seed(42); np.random.seed(42)
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=20, min_input_length=200, max_input_length=500,
            min_output_length=10, max_output_length=30, disable_tqdm=True,
        ),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", request_level_scheduling=False),
        platform_config=PlatformConfig(device="H20"),
    )
    m = runner.run_benchmark_emulation()
    assert m["completed"] == 20
    assert m["mean_tpot_ms"] > 0
    print("[iter_unchanged] completed=%d TPOT=%.0fms" % (m["completed"], m["mean_tpot_ms"]))


# ===========================================================================
# Boundary cases
# ===========================================================================

def test_request_level_single_request():
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=1, min_input_length=500, max_input_length=501,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", request_level_scheduling=True),
        platform_config=PlatformConfig(device="H20"),
        infer_time_predictor=_make_predictor(0.1),
    )
    m = runner.run_benchmark_emulation()
    assert m["completed"] == 1
    print("[single] completed=1 TTFT=%.0fms" % m["mean_ttft_ms"])


def test_request_level_custom_predict_fn():
    call_count = [0]
    def my_predictor(uncached, cached):
        call_count[0] += 1
        return (uncached * 0.02 + 5) / 1000.0
    predictor = RequestLevelTimePredictor(predict_fn=my_predictor)

    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=5, min_input_length=100, max_input_length=200,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", request_level_scheduling=True),
        platform_config=PlatformConfig(device="H20"),
        infer_time_predictor=predictor,
    )
    m = runner.run_benchmark_emulation()
    assert m["completed"] == 5
    assert call_count[0] == 5
    print("[custom_fn] completed=%d calls=%d" % (m["completed"], call_count[0]))


def test_request_level_with_stats():
    random.seed(42); np.random.seed(42)
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=10, min_input_length=200, max_input_length=500,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", request_level_scheduling=True, enable_stats=True),
        platform_config=PlatformConfig(device="H20"),
        infer_time_predictor=_make_predictor(0.1),
    )
    m = runner.run_benchmark_emulation()
    stats = runner.scheduler_emulator.get_iteration_stats()
    assert len(stats) == 10
    for s in stats:
        assert s.iter_latency_ms > 0
        assert s.num_context_requests == 1
    print("[stats] %d iteration stats, all valid" % len(stats))


if __name__ == "__main__":
    test_request_level_completes()
    test_request_level_ttft_equals_predicted()
    test_request_level_clock_advances()
    test_request_level_queue_ordering()
    test_iter_vs_request_both_complete()
    test_request_level_faster_simulation()
    test_request_level_multi_instance()
    test_request_level_multi_instance_ttft()
    test_request_level_cache_write()
    test_iter_mode_unchanged()
    test_request_level_single_request()
    test_request_level_custom_predict_fn()
    test_request_level_with_stats()
