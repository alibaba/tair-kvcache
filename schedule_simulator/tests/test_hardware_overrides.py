"""Tests for hardware parameter overrides (kv_cache_space_per_token, max_num_tokens, l2_cache_num_tokens)."""
import random
import numpy as np
import pytest

from schedule_simulator.schedule_emulator.types import *
from schedule_simulator.schedule_emulator.run import BenchmarkRunner, DisaggBenchmarkRunner
from schedule_simulator.infer_time_predictor import RequestLevelTimePredictor


def _predictor():
    return RequestLevelTimePredictor(constant_ms_per_token=0.1)


# ===========================================================================
# kv_cache_space_per_token override
# ===========================================================================

def test_kv_bytes_override_used():
    """When kv_cache_space_per_token is set, skip ModelInfo derivation."""
    random.seed(42); np.random.seed(42)
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(num_prompts=5, min_input_length=100, max_input_length=200,
                                          min_output_length=1, max_output_length=2, disable_tqdm=True),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", request_level_scheduling=True,
                                          kv_cache_space_per_token=92770),
        platform_config=PlatformConfig(device="H20"),
        infer_time_predictor=_predictor(),
    )
    assert runner.scheduler_emulator.kv_cache_space_per_token == 92770
    m = runner.run_benchmark_emulation()
    assert m["completed"] == 5
    print("[kv_override] kv_space=%d completed=%d" % (runner.scheduler_emulator.kv_cache_space_per_token, m["completed"]))


def test_kv_bytes_default_from_model():
    """Without override, kv_cache_space_per_token is derived from ModelInfo."""
    random.seed(42); np.random.seed(42)
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(num_prompts=5, min_input_length=100, max_input_length=200,
                                          min_output_length=1, max_output_length=2, disable_tqdm=True),
        scheduler_config=SchedulerConfig("Qwen2.5-3B"),
        platform_config=PlatformConfig(device="H20"),
    )
    assert runner.scheduler_emulator.kv_cache_space_per_token > 0
    assert runner.scheduler_emulator.kv_cache_space_per_token != 92770
    print("[kv_default] kv_space=%d (from ModelInfo)" % runner.scheduler_emulator.kv_cache_space_per_token)


def test_different_kv_bytes_different_capacity():
    """Different kv_cache_space_per_token values are correctly stored."""
    random.seed(42); np.random.seed(42)
    r1 = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(num_prompts=5, min_input_length=100, max_input_length=200,
                                          min_output_length=1, max_output_length=2, disable_tqdm=True),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", request_level_scheduling=True,
                                          kv_cache_space_per_token=10000, max_num_tokens=200000),
        platform_config=PlatformConfig(device="H20"),
        infer_time_predictor=_predictor(),
    )
    random.seed(42); np.random.seed(42)
    r2 = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(num_prompts=5, min_input_length=100, max_input_length=200,
                                          min_output_length=1, max_output_length=2, disable_tqdm=True),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", request_level_scheduling=True,
                                          kv_cache_space_per_token=100000, max_num_tokens=50000),
        platform_config=PlatformConfig(device="H20"),
        infer_time_predictor=_predictor(),
    )
    assert r1.scheduler_emulator.kv_cache_space_per_token == 10000
    assert r2.scheduler_emulator.kv_cache_space_per_token == 100000
    assert r1.scheduler_emulator.max_num_tokens > r2.scheduler_emulator.max_num_tokens
    print("[kv_capacity] small_kv L1=%d, large_kv L1=%d" % (
        r1.scheduler_emulator.max_num_tokens, r2.scheduler_emulator.max_num_tokens))


# ===========================================================================
# max_num_tokens override (L1 cache capacity)
# ===========================================================================

def test_max_tokens_override():
    """Directly set L1 KV cache capacity."""
    random.seed(42); np.random.seed(42)
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(num_prompts=10, min_input_length=100, max_input_length=200,
                                          min_output_length=1, max_output_length=2, disable_tqdm=True),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", request_level_scheduling=True,
                                          max_num_tokens=50000),
        platform_config=PlatformConfig(device="H20"),
        infer_time_predictor=_predictor(),
    )
    assert runner.scheduler_emulator.max_num_tokens == 50000
    m = runner.run_benchmark_emulation()
    assert m["completed"] == 10
    print("[max_tokens] L1=%d completed=%d" % (runner.scheduler_emulator.max_num_tokens, m["completed"]))


def test_max_tokens_default_derived():
    """Without override, max_num_tokens is derived from HBM capacity."""
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(num_prompts=1, min_input_length=100, max_input_length=200,
                                          min_output_length=1, max_output_length=2, disable_tqdm=True),
        scheduler_config=SchedulerConfig("Qwen2.5-3B"),
        platform_config=PlatformConfig(device="H20"),
    )
    assert runner.scheduler_emulator.max_num_tokens > 0
    print("[max_tokens_default] L1=%d (derived)" % runner.scheduler_emulator.max_num_tokens)


def test_small_l1_causes_capacity_pressure():
    """Very small L1 should still work but may limit batch size."""
    random.seed(42); np.random.seed(42)
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(num_prompts=5, min_input_length=100, max_input_length=200,
                                          min_output_length=1, max_output_length=2, disable_tqdm=True),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", request_level_scheduling=True,
                                          max_num_tokens=1000),
        platform_config=PlatformConfig(device="H20"),
        infer_time_predictor=_predictor(),
    )
    m = runner.run_benchmark_emulation()
    assert m["completed"] == 5
    print("[small_l1] L1=1000 tokens, completed=%d" % m["completed"])


# ===========================================================================
# l2_cache_num_tokens override
# ===========================================================================

def test_l2_override():
    """l2_cache_num_tokens is correctly passed to SchedulerConfig."""
    sc = SchedulerConfig("Qwen2.5-3B", max_num_tokens=10000, l2_cache_num_tokens=50000)
    assert sc.l2_cache_num_tokens == 50000
    assert sc.max_num_tokens == 10000
    # Verify it works in a simple run
    random.seed(42); np.random.seed(42)
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(num_prompts=5, min_input_length=100, max_input_length=200,
                                          min_output_length=1, max_output_length=2, disable_tqdm=True),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", request_level_scheduling=True,
                                          max_num_tokens=10000, l2_cache_num_tokens=50000),
        platform_config=PlatformConfig(device="H20"),
        infer_time_predictor=_predictor(),
    )
    m = runner.run_benchmark_emulation()
    assert m["completed"] == 5
    print("[l2_override] L2=50000 configured, completed=%d" % m["completed"])


# ===========================================================================
# Combined: skip ModelInfo entirely with all overrides
# ===========================================================================

def test_all_overrides_skip_model_derivation():
    """With all hardware params set, no ModelInfo derivation needed."""
    random.seed(42); np.random.seed(42)
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(num_prompts=10, min_input_length=100, max_input_length=500,
                                          min_output_length=1, max_output_length=2, disable_tqdm=True),
        scheduler_config=SchedulerConfig(
            "Qwen2.5-3B",
            request_level_scheduling=True,
            kv_cache_space_per_token=92770,
            max_num_tokens=100000,
            mem_fraction_static=0.85,
            chunked_prefill_size=8192,
        ),
        platform_config=PlatformConfig(device="H20"),
        infer_time_predictor=_predictor(),
    )
    assert runner.scheduler_emulator.kv_cache_space_per_token == 92770
    assert runner.scheduler_emulator.max_num_tokens == 100000
    m = runner.run_benchmark_emulation()
    assert m["completed"] == 10
    print("[all_overrides] kv=%d L1=%d completed=%d" % (92770, 100000, m["completed"]))


# ===========================================================================
# Multi-instance with hardware overrides
# ===========================================================================

def test_multi_instance_with_overrides():
    """Hardware overrides work in multi-instance mode."""
    random.seed(42); np.random.seed(42)
    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(num_prompts=20, min_input_length=100, max_input_length=500,
                                          min_output_length=1, max_output_length=2, disable_tqdm=True),
        p_scheduler_config=SchedulerConfig(
            "Qwen2.5-3B", scenario="disagg_prefill",
            request_level_scheduling=True,
            kv_cache_space_per_token=92770,
            max_num_tokens=50000,
        ),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20"),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN,
                                    worker_startup_check_interval=0.01),
        num_p_instance=3, num_d_instance=0,
        infer_time_predictor=_predictor(),
    )
    for s in runner.p_schedulers:
        assert s.kv_cache_space_per_token == 92770
        assert s.max_num_tokens == 50000
    m = runner.run_benchmark_emulation()
    assert m["completed"] == 20
    print("[multi_override] 3 P nodes, all with custom kv=%d L1=%d" % (92770, 50000))


# ===========================================================================
# Bandwidth parameters (already in PlatformConfig, verify they work)
# ===========================================================================

def test_bandwidth_params_affect_prefetch():
    """Different bandwidth settings should produce different prefetch latencies."""
    random.seed(42); np.random.seed(42)
    r_fast = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(num_prompts=10, min_input_length=500, max_input_length=1000,
                                          min_output_length=1, max_output_length=2,
                                          min_prefix_disk_hit_rate=0.5, max_prefix_disk_hit_rate=0.5,
                                          disable_tqdm=True),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", hicache_storage_backend="hf3fs", chunked_prefill_size=8192),
        platform_config=PlatformConfig(device="H20", disk_read_bandwidth_gb=100.0, memory_read_bandwidth_gb=100.0),
    )
    m_fast = r_fast.run_benchmark_emulation()

    random.seed(42); np.random.seed(42)
    r_slow = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(num_prompts=10, min_input_length=500, max_input_length=1000,
                                          min_output_length=1, max_output_length=2,
                                          min_prefix_disk_hit_rate=0.5, max_prefix_disk_hit_rate=0.5,
                                          disable_tqdm=True),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", hicache_storage_backend="hf3fs", chunked_prefill_size=8192),
        platform_config=PlatformConfig(device="H20", disk_read_bandwidth_gb=0.1, memory_read_bandwidth_gb=0.1),
    )
    m_slow = r_slow.run_benchmark_emulation()

    assert m_fast["mean_ttft_ms"] <= m_slow["mean_ttft_ms"]
    print("[bandwidth] fast_bw TTFT=%.0fms, slow_bw TTFT=%.0fms" % (m_fast["mean_ttft_ms"], m_slow["mean_ttft_ms"]))


# ===========================================================================
# Regression: None values use defaults
# ===========================================================================

def test_none_overrides_use_defaults():
    """All override fields as None should produce same results as before."""
    random.seed(42); np.random.seed(42)
    r1 = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(num_prompts=5, min_input_length=100, max_input_length=200,
                                          min_output_length=1, max_output_length=2, disable_tqdm=True),
        scheduler_config=SchedulerConfig("Qwen2.5-3B",
                                          kv_cache_space_per_token=None, max_num_tokens=None, l2_cache_num_tokens=None),
        platform_config=PlatformConfig(device="H20"),
    )
    random.seed(42); np.random.seed(42)
    r2 = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(num_prompts=5, min_input_length=100, max_input_length=200,
                                          min_output_length=1, max_output_length=2, disable_tqdm=True),
        scheduler_config=SchedulerConfig("Qwen2.5-3B"),
        platform_config=PlatformConfig(device="H20"),
    )
    assert r1.scheduler_emulator.kv_cache_space_per_token == r2.scheduler_emulator.kv_cache_space_per_token
    assert r1.scheduler_emulator.max_num_tokens == r2.scheduler_emulator.max_num_tokens
    print("[none_defaults] kv=%d max_tokens=%d (identical)" % (
        r1.scheduler_emulator.kv_cache_space_per_token, r1.scheduler_emulator.max_num_tokens))


if __name__ == "__main__":
    test_kv_bytes_override_used()
    test_kv_bytes_default_from_model()
    test_different_kv_bytes_different_capacity()
    test_max_tokens_override()
    test_max_tokens_default_derived()
    test_small_l1_causes_capacity_pressure()
    test_l2_override()
    test_all_overrides_skip_model_derivation()
    test_multi_instance_with_overrides()
    test_bandwidth_params_affect_prefetch()
    test_none_overrides_use_defaults()
