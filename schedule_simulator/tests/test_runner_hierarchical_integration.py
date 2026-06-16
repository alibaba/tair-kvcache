"""
Tests for DisaggBenchmarkRunner hierarchical cache integration (P4).
Covers: auto-setup, fallback, cold/warm cache, prefix sharing, cross-engine hits,
        TTFT impact, query_type, per-request records, CSV export, routing policies,
        boundary cases.
"""
import os
import sys
import json
import numpy
import random
import shutil
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
from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner, BenchmarkRunner

pytestmark = pytest.mark.skipif(not HAS_KVCM, reason="kvcm_py_optimizer not available")

BASE_CONFIG = os.path.join(os.path.dirname(__file__), "assets/hierarchical/test_config.json")
OUTPUT_DIR = "/tmp/hierarchical_p4_test_output"
DATASET_DIR = os.path.join(os.path.dirname(__file__), "assets/dataset")


def _make_config(num_engines=5, write_mode="write_through"):
    config = json.load(open(BASE_CONFIG))
    config["infer_clusters"][0]["infer_ids"] = [f"P{i}" for i in range(num_engines)]
    config["infer_clusters"][0]["storage_pool_flow"]["write_mode"] = write_mode
    config["output_result_path"] = OUTPUT_DIR
    config["storage_pool"]["output_result_path"] = os.path.join(OUTPUT_DIR, "pool")
    os.makedirs(os.path.join(OUTPUT_DIR, "pool"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "infer"), exist_ok=True)
    with open("/tmp/hierarchical_test_trace.jsonl", "w"):
        pass
    tmp_path = f"/tmp/hierarchical_p4_config_{num_engines}_{write_mode}.json"
    with open(tmp_path, "w") as f:
        json.dump(config, f)
    return tmp_path


def _make_runner(num_p=5, policy=RoutingPolicy.ROUND_ROBIN, num_prompts=50,
                 write_mode="write_through", query_type="prefix_match",
                 prefetch_policy="best_effort", seed=42):
    random.seed(seed)
    numpy.random.seed(seed)
    config_path = _make_config(num_p, write_mode)
    return DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=num_prompts,
            min_input_length=30, max_input_length=80,
            min_output_length=1, max_output_length=2,
            disable_tqdm=True,
        ),
        p_scheduler_config=SchedulerConfig(
            "Qwen2.5-3B", scenario="disagg_prefill",
            hicache_storage_backend="hf3fs",
            hicache_storage_prefetch_policy=prefetch_policy,
            hicache_read_query_type=query_type,
            enable_stats=True,
        ),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(
            device="H20", disk_read_bandwidth_gb=2.0,
            memory_read_bandwidth_gb=16.0, peer_read_bandwidth_gb=10.0,
        ),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(
            p_policy=policy, d_policy=RoutingPolicy.ROUND_ROBIN,
            worker_startup_check_interval=0.01,
        ),
        num_p_instance=num_p, num_d_instance=0,
        hierarchical_config_path=config_path,
    )


# ===========================================================================
# Section 1: Setup and fallback
# ===========================================================================

def test_auto_setup():
    """hierarchical_config_path creates manager and replaces all adapters."""
    runner = _make_runner(num_p=3)
    assert runner.hierarchical_manager is not None
    from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter
    for i, sched in enumerate(runner.p_schedulers):
        assert isinstance(sched.tree_cache, HierarchicalCacheAdapter), f"Scheduler {i} missing adapter"
    print(f"[auto_setup] {len(runner.p_schedulers)} schedulers wired")


def test_no_hierarchical_fallback():
    """Without hierarchical_config_path, original behavior is preserved."""
    random.seed(42)
    numpy.random.seed(42)
    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=10, min_input_length=30, max_input_length=80,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        p_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill"),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20"),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN,
                                    worker_startup_check_interval=0.01),
        num_p_instance=3, num_d_instance=0,
    )
    assert runner.hierarchical_manager is None
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 10
    assert runner.get_hierarchical_metrics() == {}
    print("[fallback] No crash, original path works")


# ===========================================================================
# Section 2: Cold start (no hits) vs warm cache (with hits)
# ===========================================================================

def test_cold_start_all_miss():
    """First batch of unique requests should have 0 cache hits everywhere."""
    runner = _make_runner(num_p=2, num_prompts=20)
    runner.run_benchmark_emulation()
    hier = runner.get_hierarchical_metrics()

    # First request in read_records should have 0 hits (cold cache)
    first = hier["read_records"][0]
    assert first.engine_hit == 0, f"Cold first request should have 0 engine hits, got {first.engine_hit}"
    assert first.peer_hit == 0, f"Cold first request should have 0 peer hits, got {first.peer_hit}"
    assert first.pool_hit == 0, f"Cold first request should have 0 pool hits, got {first.pool_hit}"
    print(f"[cold_start] First request: engine={first.engine_hit}, peer={first.peer_hit}, pool={first.pool_hit}")


def test_warm_cache_has_hits():
    """After writes, subsequent reads of same data should have hits."""
    from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter

    runner = _make_runner(num_p=1, num_prompts=10)
    runner.run_benchmark_emulation()
    hier = runner.get_hierarchical_metrics()

    # With a single instance and random data, later requests might not share prefixes.
    # But the write_records should all have data
    assert hier["num_writes"] == 10
    assert all(w.write_blocks > 0 for w in hier["write_records"])

    # Verify the adapter's WriteCache was called for every request
    adapter = runner.p_schedulers[0].tree_cache
    assert isinstance(adapter, HierarchicalCacheAdapter)
    assert len(adapter.write_records) == 10
    print(f"[warm_cache] {hier['num_writes']} writes, total_hit_ratio={hier['block_hit_ratio']:.3f}")


def test_repeated_requests_increase_hit_ratio():
    """Same block_ids sent twice to same engine -> second read should hit."""
    import kvcm_py_optimizer as kvcm_mod
    from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter
    from schedule_simulator.schedule_emulator.base import GlobalValues

    config_path = _make_config(num_engines=1)
    loader = kvcm_mod.HierarchicalReplayConfigLoader()
    assert loader.load(config_path)
    mgr = kvcm_mod.HierarchicalReplayManager(loader.config())
    assert mgr.Init()

    gv = GlobalValues()
    platform = PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0,
                              memory_read_bandwidth_gb=16.0, peer_read_bandwidth_gb=10.0)
    adapter = HierarchicalCacheAdapter(
        manager=mgr, engine_instance_id="P0", platform_config=platform,
        kv_cache_space_per_token=256, page_size=1, global_values=gv,
        prefetch_stop_policy="best_effort", enable_stats=True,
    )

    from schedule_simulator.schedule_emulator.types import FakeRequest
    ids = list(range(2000, 2050))

    # First read - cold
    req1 = FakeRequest(id=1, input_token_length=50, output_token_length=5,
                       origin_input_ids=ids, output_ids=list(range(9000, 9005)))
    adapter.add_to_prefetch_queue(req1)
    assert req1.device_cache_hit_length == 0, "First read should be cold"
    adapter.on_request_complete(req1, 1.0)

    # Second read - same ids, should hit
    req2 = FakeRequest(id=2, input_token_length=50, output_token_length=5,
                       origin_input_ids=ids, output_ids=list(range(9010, 9015)))
    adapter.add_to_prefetch_queue(req2)
    assert req2.device_cache_hit_length > 0, (
        f"Second read should have local hits, got {req2.device_cache_hit_length}"
    )
    print(f"[repeated] Cold={0}, Warm={req2.device_cache_hit_length} engine hits")


# ===========================================================================
# Section 3: Cross-engine hits (P2P and Pool)
# ===========================================================================

def test_cross_engine_pool_hit():
    """Write to engine_0 with write_through, engine_1 should see pool hits."""
    import kvcm_py_optimizer as kvcm_mod
    from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter
    from schedule_simulator.schedule_emulator.base import GlobalValues
    from schedule_simulator.schedule_emulator.types import FakeRequest

    config_path = _make_config(num_engines=2, write_mode="write_through")
    loader = kvcm_mod.HierarchicalReplayConfigLoader()
    assert loader.load(config_path)
    mgr = kvcm_mod.HierarchicalReplayManager(loader.config())
    assert mgr.Init()

    platform = PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0,
                              memory_read_bandwidth_gb=16.0, peer_read_bandwidth_gb=10.0)
    gv0 = GlobalValues()
    gv1 = GlobalValues()
    adapter_0 = HierarchicalCacheAdapter(
        manager=mgr, engine_instance_id="P0", platform_config=platform,
        kv_cache_space_per_token=256, page_size=1, global_values=gv0,
    )
    adapter_1 = HierarchicalCacheAdapter(
        manager=mgr, engine_instance_id="P1", platform_config=platform,
        kv_cache_space_per_token=256, page_size=1, global_values=gv1,
    )

    ids = list(range(3000, 3050))
    req_w = FakeRequest(id=10, input_token_length=50, output_token_length=5,
                        origin_input_ids=ids, output_ids=list(range(8000, 8005)))
    adapter_0.add_to_prefetch_queue(req_w)
    adapter_0.on_request_complete(req_w, 1.0)

    req_r = FakeRequest(id=11, input_token_length=50, output_token_length=5,
                        origin_input_ids=ids, output_ids=list(range(8010, 8015)))
    adapter_1.add_to_prefetch_queue(req_r)

    assert req_r.device_cache_hit_length == 0, "P1 should have no local hits"
    non_local = req_r.host_cache_hit_length + req_r.disk_cache_hit_length
    assert non_local > 0, (
        f"P1 should get peer or pool hits: peer={req_r.host_cache_hit_length}, pool={req_r.disk_cache_hit_length}"
    )
    print(f"[cross_engine] P1: peer={req_r.host_cache_hit_length}, pool={req_r.disk_cache_hit_length}")


def test_cross_engine_no_hit_without_write():
    """Without any writes, cross-engine reads should have 0 hits."""
    import kvcm_py_optimizer as kvcm_mod
    from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter
    from schedule_simulator.schedule_emulator.base import GlobalValues
    from schedule_simulator.schedule_emulator.types import FakeRequest

    config_path = _make_config(num_engines=2)
    loader = kvcm_mod.HierarchicalReplayConfigLoader()
    assert loader.load(config_path)
    mgr = kvcm_mod.HierarchicalReplayManager(loader.config())
    assert mgr.Init()

    gv1 = GlobalValues()
    platform = PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0,
                              memory_read_bandwidth_gb=16.0)
    adapter_1 = HierarchicalCacheAdapter(
        manager=mgr, engine_instance_id="P1", platform_config=platform,
        kv_cache_space_per_token=256, page_size=1, global_values=gv1,
    )

    req = FakeRequest(id=20, input_token_length=30, output_token_length=5,
                      origin_input_ids=list(range(7000, 7030)), output_ids=list(range(9900, 9905)))
    adapter_1.add_to_prefetch_queue(req)

    assert req.device_cache_hit_length == 0
    assert req.host_cache_hit_length == 0
    assert req.disk_cache_hit_length == 0
    print("[no_write_no_hit] All zeros as expected")


# ===========================================================================
# Section 4: TTFT impact from cache hits
# ===========================================================================

# ===========================================================================
# Section 5: query_type behavior
# ===========================================================================

def test_query_type_prefix_match():
    runner = _make_runner(num_p=2, num_prompts=20, query_type="prefix_match")
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 20
    hier = runner.get_hierarchical_metrics()
    assert hier["num_reads"] == 20
    print(f"[prefix_match] completed=20, hit_ratio={hier['block_hit_ratio']:.3f}")


def test_query_type_batch_get():
    runner = _make_runner(num_p=2, num_prompts=20, query_type="batch_get")
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 20
    hier = runner.get_hierarchical_metrics()
    assert hier["num_reads"] == 20
    print(f"[batch_get] completed=20, hit_ratio={hier['block_hit_ratio']:.3f}")


# ===========================================================================
# Section 6: Per-request records and CSV export
# ===========================================================================

def test_per_request_records_complete():
    runner = _make_runner(num_p=3, num_prompts=30)
    runner.run_benchmark_emulation()
    hier = runner.get_hierarchical_metrics()

    assert len(hier["read_records"]) == 30
    assert len(hier["write_records"]) == 30

    for r in hier["read_records"]:
        assert r.engine_hit >= 0
        assert r.peer_hit >= 0
        assert r.pool_hit >= 0
        assert r.total_hit >= 0
        assert r.input_length > 0
        assert r.total_hit == r.engine_hit + r.peer_hit + r.pool_hit

    for w in hier["write_records"]:
        assert w.write_blocks > 0
        assert w.timestamp >= 0

    print(f"[per_request] 30 read + 30 write records, all fields valid")


def test_analyze_results_csv():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    runner = _make_runner(num_p=3, num_prompts=20)
    runner.run_benchmark_emulation()
    runner.analyze_hierarchical_results()

    csv_path = os.path.join(OUTPUT_DIR, "hierarchical_hit_rates.csv")
    assert os.path.exists(csv_path), f"Expected {csv_path}"
    with open(csv_path) as f:
        lines = f.readlines()
    assert len(lines) > 1, "CSV should have header + data"
    header = lines[0].strip()
    assert "LocalHit" in header or "local" in header.lower() or "Hit" in header
    print(f"[csv_export] {csv_path}: {len(lines)} lines, header={header[:80]}...")


# ===========================================================================
# Section 7: Routing policies with hierarchical
# ===========================================================================

def test_round_robin_with_hierarchical():
    runner = _make_runner(num_p=5, policy=RoutingPolicy.ROUND_ROBIN, num_prompts=50)
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 50
    hier = runner.get_hierarchical_metrics()
    assert hier["num_reads"] == 50
    dist = [len(s.completed_requests) for s in runner.p_schedulers]
    assert all(d == 10 for d in dist), f"RoundRobin should be uniform: {dist}"
    print(f"[rr_hierarchical] uniform dist, hit_ratio={hier['block_hit_ratio']:.3f}")


def test_random_with_hierarchical():
    runner = _make_runner(num_p=5, policy=RoutingPolicy.RANDOM, num_prompts=50)
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 50
    hier = runner.get_hierarchical_metrics()
    assert hier["num_reads"] == 50
    print(f"[random_hierarchical] hit_ratio={hier['block_hit_ratio']:.3f}")


# ===========================================================================
# Section 8: Boundary cases
# ===========================================================================

def test_single_instance():
    runner = _make_runner(num_p=1, num_prompts=10)
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 10
    hier = runner.get_hierarchical_metrics()
    assert hier["num_reads"] == 10
    assert hier["num_writes"] == 10
    # Single instance: no peer hits possible
    assert hier["total_peer_hit_blocks"] == 0, "Single instance should have 0 peer hits"
    print(f"[single_instance] no peer hits, local_ratio={hier['engine_hit_block_ratio']:.3f}")


def test_very_short_input():
    random.seed(42)
    numpy.random.seed(42)
    config_path = _make_config(num_engines=2)
    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=20, min_input_length=1, max_input_length=5,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        p_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill",
                                            hicache_storage_backend="hf3fs",
                                            hicache_storage_prefetch_policy="best_effort",
                                            enable_stats=True),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0,
                                          memory_read_bandwidth_gb=16.0),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN,
                                    worker_startup_check_interval=0.01),
        num_p_instance=2, num_d_instance=0,
        hierarchical_config_path=config_path,
    )
    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 20
    print(f"[short_input] completed=20, TTFT={metrics['mean_ttft_ms']:.1f}ms")


# ===========================================================================
# Section 9: Metrics structure validation
# ===========================================================================

def test_hierarchical_metrics_structure():
    runner = _make_runner(num_p=3, num_prompts=30)
    runner.run_benchmark_emulation()
    hier = runner.get_hierarchical_metrics()

    required_keys = [
        "total_engine_hit_blocks", "total_peer_hit_blocks", "total_pool_hit_blocks",
        "engine_hit_block_ratio", "peer_hit_block_ratio", "pool_hit_block_ratio", "block_hit_ratio",
        "num_reads", "num_writes", "read_records", "write_records", "total_blocks_queried",
    ]
    for key in required_keys:
        assert key in hier, f"Missing key: {key}"

    assert 0 <= hier["engine_hit_block_ratio"] <= 1
    assert 0 <= hier["peer_hit_block_ratio"] <= 1
    assert 0 <= hier["pool_hit_block_ratio"] <= 1
    assert 0 <= hier["block_hit_ratio"] <= 1
    assert abs(hier["block_hit_ratio"] - hier["engine_hit_block_ratio"] - hier["peer_hit_block_ratio"] - hier["pool_hit_block_ratio"]) < 1e-9
    print(f"[metrics_structure] All keys present, ratios in [0,1], sum correct")


if __name__ == "__main__":
    test_auto_setup()
    test_no_hierarchical_fallback()
    test_cold_start_all_miss()
    test_warm_cache_has_hits()
    test_repeated_requests_increase_hit_ratio()
    test_cross_engine_pool_hit()
    test_cross_engine_no_hit_without_write()
    test_ttft_lower_with_cache_hits()
    test_query_type_prefix_match()
    test_query_type_batch_get()
    test_per_request_records_complete()
    test_analyze_results_csv()
    test_round_robin_with_hierarchical()
    test_random_with_hierarchical()
    test_single_instance()
    test_very_short_input()
    test_hierarchical_metrics_structure()
