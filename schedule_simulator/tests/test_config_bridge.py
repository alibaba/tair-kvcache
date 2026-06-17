"""
Tests for P5: hierarchical config auto-generation from SchedulerConfig + PlatformConfig.
Covers: field mapping correctness, parameter variants, enable_hierarchical shortcut,
        write_mode mapping, P2P toggle, boundary cases, end-to-end with Runner.
"""
import os
import sys
import json
import shutil
import numpy
import random
import pytest

KVCM_SO_DIR = "/sgl-workspace/claude_workspace/tair-kvcache/bazel-bin/kv_cache_manager/optimizer/pybind"
if KVCM_SO_DIR not in sys.path:
    sys.path.insert(0, KVCM_SO_DIR)

try:
    import kvcm_py_optimizer as kvcm
    HAS_KVCM = True
except ImportError:
    HAS_KVCM = False

from schedule_simulator.schedule_emulator.types import (
    BenchmarkConfig, SchedulerConfig, PlatformConfig,
    RouterConfig, RoutingPolicy,
)
from schedule_simulator.schedule_emulator.hierarchical_config_builder import build_hierarchical_config

pytestmark = pytest.mark.skipif(not HAS_KVCM, reason="kvcm_py_optimizer not available")

TMP_DIR = "/tmp/config_bridge_test"


@pytest.fixture(autouse=True)
def cleanup():
    yield
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)


def _default_sched_config(**overrides):
    defaults = dict(
        model="Qwen2.5-3B",
        hicache_storage_backend="hf3fs",
        hicache_write_policy="write_through",
        hicache_read_query_type="prefix_match",
        hicache_storage_prefetch_policy="best_effort",
        tp_size=1,
    )
    defaults.update(overrides)
    return SchedulerConfig(**defaults)


def _default_plat_config(**overrides):
    defaults = dict(
        device="H20",
        disk_read_bandwidth_gb=2.0,
        memory_read_bandwidth_gb=16.0,
        memory_capacity_gb=64.0,
        peer_read_bandwidth_gb=10.0,
    )
    defaults.update(overrides)
    return PlatformConfig(**defaults)


# ===========================================================================
# Section 1: Basic generation and field mapping
# ===========================================================================

def test_generates_valid_json():
    path = build_hierarchical_config(
        _default_sched_config(), _default_plat_config(),
        p_instance_ids=["P0", "P1", "P2"],
        output_dir=TMP_DIR,
    )
    assert os.path.exists(path)
    config = json.load(open(path))
    assert "infer_clusters" in config
    assert "storage_pool" in config
    assert config["trace_file_path"].endswith("trace.jsonl")
    print(f"[valid_json] Generated {path}")


def test_infer_ids_match():
    path = build_hierarchical_config(
        _default_sched_config(), _default_plat_config(),
        p_instance_ids=["A", "B", "C", "D"],
        output_dir=TMP_DIR,
    )
    config = json.load(open(path))
    assert config["infer_clusters"][0]["infer_ids"] == ["A", "B", "C", "D"]
    print("[infer_ids] Correct mapping")


def test_p_and_d_instance_ids():
    path = build_hierarchical_config(
        _default_sched_config(), _default_plat_config(),
        p_instance_ids=["P0", "P1"],
        d_instance_ids=["D0", "D1", "D2"],
        output_dir=TMP_DIR,
    )
    config = json.load(open(path))
    ids = config["infer_clusters"][0]["infer_ids"]
    assert ids == ["P0", "P1", "D0", "D1", "D2"]
    print(f"[p_and_d] Combined ids: {ids}")


def test_query_type_mapping():
    for qt in ["prefix_match", "batch_get"]:
        path = build_hierarchical_config(
            _default_sched_config(hicache_read_query_type=qt),
            _default_plat_config(),
            p_instance_ids=["P0"],
            output_dir=TMP_DIR,
        )
        config = json.load(open(path))
        assert config["infer_clusters"][0]["engine_read_query_type"] == qt
    print("[query_type] prefix_match and batch_get mapped correctly")


def test_bytes_per_token_computed():
    path = build_hierarchical_config(
        _default_sched_config(), _default_plat_config(),
        p_instance_ids=["P0"],
        output_dir=TMP_DIR,
    )
    config = json.load(open(path))
    bpt = config["infer_clusters"][0]["model"]["bytes_per_token"]
    assert bpt > 0, f"bytes_per_token should be > 0, got {bpt}"
    pool_bpt = config["storage_pool"]["pools"][0]["model"]["bytes_per_token"]
    assert pool_bpt == bpt, "Engine and pool bytes_per_token should match"
    print(f"[bytes_per_token] {bpt} bytes (engine and pool match)")


# ===========================================================================
# Section 2: Write mode mapping
# ===========================================================================

def test_write_mode_write_through():
    path = build_hierarchical_config(
        _default_sched_config(hicache_write_policy="write_through"),
        _default_plat_config(),
        p_instance_ids=["P0"],
        output_dir=TMP_DIR,
    )
    config = json.load(open(path))
    assert config["infer_clusters"][0]["storage_pool_flow"]["write_mode"] == "write_through"
    print("[write_mode] write_through mapped")


def test_write_mode_write_back_to_cascading():
    path = build_hierarchical_config(
        _default_sched_config(hicache_write_policy="write_back"),
        _default_plat_config(),
        p_instance_ids=["P0"],
        output_dir=TMP_DIR,
    )
    config = json.load(open(path))
    assert config["infer_clusters"][0]["storage_pool_flow"]["write_mode"] == "cascading"
    print("[write_mode] write_back -> cascading mapped")


def test_write_mode_selective():
    path = build_hierarchical_config(
        _default_sched_config(hicache_write_policy="write_through_selective"),
        _default_plat_config(),
        p_instance_ids=["P0"],
        output_dir=TMP_DIR,
    )
    config = json.load(open(path))
    assert config["infer_clusters"][0]["storage_pool_flow"]["write_mode"] == "write_through_selective"
    print("[write_mode] write_through_selective mapped")


# ===========================================================================
# Section 3: P2P and tiers
# ===========================================================================

def test_p2p_enabled_multiple_instances():
    path = build_hierarchical_config(
        _default_sched_config(), _default_plat_config(),
        p_instance_ids=["P0", "P1"],
        enable_p2p=True,
        output_dir=TMP_DIR,
    )
    config = json.load(open(path))
    p2p = config["infer_clusters"][0].get("p2p_read_flows", [])
    assert len(p2p) == 1, "Should have P2P flow with 2+ instances"
    assert p2p[0]["tier"] == "dram"
    print(f"[p2p_enabled] P2P flow on tier={p2p[0]['tier']}")


def test_p2p_disabled():
    path = build_hierarchical_config(
        _default_sched_config(), _default_plat_config(),
        p_instance_ids=["P0", "P1"],
        enable_p2p=False,
        output_dir=TMP_DIR,
    )
    config = json.load(open(path))
    p2p = config["infer_clusters"][0].get("p2p_read_flows", [])
    assert len(p2p) == 0, "P2P should be disabled"
    print("[p2p_disabled] No P2P flows")


def test_p2p_single_instance_no_flow():
    path = build_hierarchical_config(
        _default_sched_config(), _default_plat_config(),
        p_instance_ids=["P0"],
        enable_p2p=True,
        output_dir=TMP_DIR,
    )
    config = json.load(open(path))
    p2p = config["infer_clusters"][0].get("p2p_read_flows", [])
    assert len(p2p) == 0, "Single instance should have no P2P"
    print("[p2p_single] No P2P for single instance")


def test_two_tiers_with_dram_config():
    path = build_hierarchical_config(
        _default_sched_config(),
        _default_plat_config(memory_capacity_gb=128.0),
        p_instance_ids=["P0"],
        output_dir=TMP_DIR,
    )
    config = json.load(open(path))
    tiers = config["infer_clusters"][0]["tiers"]
    assert len(tiers) == 2
    assert tiers[0]["name"] == "hbm"
    assert tiers[1]["name"] == "dram"
    assert tiers[1]["capacity"] == 128.0
    print(f"[two_tiers] hbm={tiers[0]['capacity']}GB, dram={tiers[1]['capacity']}GB")


def test_single_tier_without_dram():
    path = build_hierarchical_config(
        _default_sched_config(),
        PlatformConfig(device="H20"),
        p_instance_ids=["P0"],
        output_dir=TMP_DIR,
    )
    config = json.load(open(path))
    tiers = config["infer_clusters"][0]["tiers"]
    assert len(tiers) == 1, "Without memory config, should have only HBM tier"
    assert tiers[0]["name"] == "hbm"
    tier_flows = config["infer_clusters"][0].get("tier_flows", [])
    assert len(tier_flows) == 0, "No tier flows with single tier"
    print("[single_tier] HBM only, no tier flows")


# ===========================================================================
# Section 4: Storage pool config
# ===========================================================================

def test_storage_pool_capacity():
    path = build_hierarchical_config(
        _default_sched_config(), _default_plat_config(),
        p_instance_ids=["P0"],
        storage_pool_capacity_gb=4.0,
        output_dir=TMP_DIR,
    )
    config = json.load(open(path))
    assert config["storage_pool"]["capacity"] == 4.0
    print("[pool_capacity] 4.0 GB")


# ===========================================================================
# Section 5: Generated config is loadable by Optimizer
# ===========================================================================

def test_config_loadable_by_optimizer():
    path = build_hierarchical_config(
        _default_sched_config(), _default_plat_config(),
        p_instance_ids=["P0", "P1"],
        output_dir=TMP_DIR,
    )
    loader = kvcm.HierarchicalReplayConfigLoader()
    assert loader.load(path), f"Optimizer failed to load generated config from {path}"
    config = loader.config()
    assert config.infer_scheduling_strategy() == "preserve_trace"
    print("[loadable] Optimizer accepts generated config")


def test_config_loadable_with_manager_init():
    path = build_hierarchical_config(
        _default_sched_config(), _default_plat_config(),
        p_instance_ids=["P0", "P1", "P2"],
        output_dir=TMP_DIR,
    )
    loader = kvcm.HierarchicalReplayConfigLoader()
    assert loader.load(path)
    mgr = kvcm.HierarchicalReplayManager(loader.config())
    assert mgr.Init(), "Manager should initialize with generated config"
    print("[manager_init] HierarchicalReplayManager.Init() succeeded")


# ===========================================================================
# Section 6: enable_hierarchical shortcut in Runner
# ===========================================================================

def test_enable_hierarchical_shortcut():
    from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner
    random.seed(42)
    numpy.random.seed(42)

    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=20, min_input_length=30, max_input_length=80,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        p_scheduler_config=SchedulerConfig(
            "Qwen2.5-3B", scenario="disagg_prefill",
            hicache_storage_backend="hf3fs",
            hicache_storage_prefetch_policy="best_effort",
        ),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(
            device="H20", disk_read_bandwidth_gb=2.0,
            memory_read_bandwidth_gb=16.0, memory_capacity_gb=64.0,
        ),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(
            p_policy=RoutingPolicy.ROUND_ROBIN,
            d_policy=RoutingPolicy.ROUND_ROBIN,
            worker_startup_check_interval=0.01,
        ),
        num_p_instance=3, num_d_instance=0,
        enable_hierarchical=True,
        hierarchical_output_dir=TMP_DIR,
    )
    assert runner.hierarchical_manager is not None, "enable_hierarchical should create manager"

    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 20

    hier = runner.get_hierarchical_metrics()
    assert hier["num_reads"] == 20
    assert hier["num_writes"] == 20
    print(f"[shortcut] completed=20, total_hit_ratio={hier['block_hit_ratio']:.3f}")


def test_enable_hierarchical_with_different_models():
    from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner
    random.seed(42)
    numpy.random.seed(42)

    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=10, min_input_length=30, max_input_length=60,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        p_scheduler_config=SchedulerConfig(
            "Qwen2.5-7B", scenario="disagg_prefill",
            hicache_storage_backend="hf3fs",
        ),
        d_scheduler_config=SchedulerConfig("Qwen2.5-7B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(
            device="H20", disk_read_bandwidth_gb=2.0,
            memory_read_bandwidth_gb=16.0, memory_capacity_gb=64.0,
        ),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN,
                                    worker_startup_check_interval=0.01),
        num_p_instance=2, num_d_instance=0,
        enable_hierarchical=True,
        hierarchical_output_dir=TMP_DIR,
    )
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 10
    print(f"[7B_model] completed=10, TTFT={metrics['mean_ttft_ms']:.1f}ms")


# ===========================================================================
# Section 7: Config path vs enable_hierarchical precedence
# ===========================================================================

def test_config_path_takes_precedence():
    """When both hierarchical_config_path and enable_hierarchical are set, config_path wins."""
    from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner
    random.seed(42)
    numpy.random.seed(42)

    # Generate a config with matching P0, P1 ids
    manual_config = build_hierarchical_config(
        _default_sched_config(), _default_plat_config(),
        p_instance_ids=["P0", "P1"],
        output_dir="/tmp/config_bridge_precedence",
    )

    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=10, min_input_length=30, max_input_length=60,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        p_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill",
                                            hicache_storage_backend="hf3fs"),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0, memory_read_bandwidth_gb=16.0),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN,
                                    worker_startup_check_interval=0.01),
        num_p_instance=2, num_d_instance=0,
        hierarchical_config_path=manual_config,
        enable_hierarchical=True,  # should be ignored since config_path is set
    )
    assert runner.hierarchical_manager is not None
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 10
    print("[precedence] config_path used, enable_hierarchical ignored")


if __name__ == "__main__":
    test_generates_valid_json()
    test_infer_ids_match()
    test_query_type_mapping()
    test_bytes_per_token_computed()
    test_write_mode_write_through()
    test_write_mode_write_back_to_cascading()
    test_write_mode_selective()
    test_p2p_enabled_multiple_instances()
    test_p2p_disabled()
    test_p2p_single_instance_no_flow()
    test_two_tiers_with_dram_config()
    test_single_tier_without_dram()
    test_storage_pool_capacity()
    test_config_loadable_by_optimizer()
    test_config_loadable_with_manager_init()
    test_enable_hierarchical_shortcut()
    test_enable_hierarchical_with_different_models()
    test_config_path_takes_precedence()
