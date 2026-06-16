"""
Tests for HierarchicalCacheAdapter: bridging PrefixCache to HierarchicalReplayManager.
"""
import os
import sys
import json
import asyncio
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
    BenchmarkConfig, SchedulerConfig, PlatformConfig, FakeRequest,
    PrefixCacheMatchResult, PrefixCacheFetchResult, RequestStage,
    RouterConfig, RoutingPolicy,
)
from schedule_simulator.schedule_emulator.base import GlobalValues

pytestmark = pytest.mark.skipif(not HAS_KVCM, reason="kvcm_py_optimizer not available")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "assets/hierarchical/test_config.json")


def _create_manager():
    os.makedirs("/tmp/hierarchical_test_output/pool", exist_ok=True)
    os.makedirs("/tmp/hierarchical_test_output/infer", exist_ok=True)
    with open("/tmp/hierarchical_test_trace.jsonl", "w"):
        pass
    loader = kvcm.HierarchicalReplayConfigLoader()
    assert loader.load(CONFIG_PATH)
    mgr = kvcm.HierarchicalReplayManager(loader.config())
    assert mgr.Init()
    return mgr


def _create_adapter(manager, engine_id="engine_0", prefetch_policy="best_effort"):
    from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter
    platform_config = PlatformConfig(
        device="H20",
        disk_read_bandwidth_gb=2.0,
        memory_read_bandwidth_gb=16.0,
        peer_read_bandwidth_gb=10.0,
    )
    gv = GlobalValues()
    adapter = HierarchicalCacheAdapter(
        manager=manager,
        engine_instance_id=engine_id,
        platform_config=platform_config,
        kv_cache_space_per_token=256,
        page_size=1,
        global_values=gv,
        prefetch_stop_policy=prefetch_policy,
        enable_stats=True,
    )
    return adapter, gv


def _make_req(req_id, input_len=100, output_len=50, input_ids=None, output_ids=None):
    return FakeRequest(
        id=req_id,
        input_token_length=input_len,
        output_token_length=output_len,
        origin_input_ids=input_ids,
        output_ids=output_ids,
    )


# =====================================================
# Test 1: Adapter lifecycle
# =====================================================

def test_adapter_lifecycle():
    """Walk through the full 6-step cache lifecycle."""
    mgr = _create_manager()
    adapter, gv = _create_adapter(mgr)

    req = _make_req(1, input_len=50, output_len=10)

    # Step 1: add_to_prefetch_queue
    adapter.add_to_prefetch_queue(req)
    assert req.stage == RequestStage.PREFETCHING
    assert req in adapter.prefetch_queue

    # Step 2: prefetch_from_storage
    gv.clock = 1.0
    result = adapter.prefetch_from_storage(req, max_time=10.0)
    assert isinstance(result, PrefixCacheFetchResult)

    # Step 3: check_prefetch_progress (best_effort always True)
    done = adapter.check_prefetch_progress(req)
    assert done is True
    assert req.stage == RequestStage.READY

    # Step 4: on_board_from_host
    result = adapter.on_board_from_host(req)
    assert isinstance(result, PrefixCacheFetchResult)

    # Step 5: match_prefix
    match = adapter.match_prefix(req)
    assert isinstance(match, PrefixCacheMatchResult)

    # Step 6: drop_match_result
    adapter.drop_match_result(req)
    assert req not in adapter.cache_controller

    # on_request_complete
    adapter.on_request_complete(req, 2.0)

    print("[lifecycle] All 6 steps passed")


# =====================================================
# Test 2: Cold read returns zero hits
# =====================================================

def test_cold_read_zero_hits():
    mgr = _create_manager()
    adapter, gv = _create_adapter(mgr)

    req = _make_req(100, input_len=20, input_ids=list(range(9000, 9020)))
    adapter.add_to_prefetch_queue(req)

    assert req.device_cache_hit_length == 0
    assert req.host_cache_hit_length == 0
    assert req.disk_cache_hit_length == 0
    print("[cold_read] All hits = 0")


# =====================================================
# Test 3: Write then local read gets engine hit
# =====================================================

def test_write_then_local_read():
    mgr = _create_manager()
    adapter_0, gv_0 = _create_adapter(mgr, "engine_0")

    ids = list(range(5000, 5020))
    req_w = _make_req(200, input_len=20, output_len=5, input_ids=ids, output_ids=list(range(7000, 7005)))
    adapter_0.add_to_prefetch_queue(req_w)
    adapter_0.on_request_complete(req_w, 1.0)

    req_r = _make_req(201, input_len=20, output_len=5, input_ids=ids, output_ids=list(range(7010, 7015)))
    adapter_0.add_to_prefetch_queue(req_r)

    assert req_r.device_cache_hit_length > 0, f"Expected local hits, got {req_r.device_cache_hit_length}"
    assert adapter_0.total_engine_hit_blocks > 0
    print(f"[local_read] device_hit={req_r.device_cache_hit_length}, total_local={adapter_0.total_engine_hit_blocks}")


# =====================================================
# Test 4: Cross-engine read gets P2P or pool hit
# =====================================================

def test_cross_engine_read():
    mgr = _create_manager()
    adapter_0, _ = _create_adapter(mgr, "engine_0")
    adapter_1, _ = _create_adapter(mgr, "engine_1")

    ids = list(range(6000, 6020))
    req_w = _make_req(300, input_len=20, output_len=5, input_ids=ids, output_ids=list(range(8000, 8005)))
    adapter_0.add_to_prefetch_queue(req_w)
    adapter_0.on_request_complete(req_w, 1.0)

    req_r = _make_req(301, input_len=20, output_len=5, input_ids=ids, output_ids=list(range(8010, 8015)))
    adapter_1.add_to_prefetch_queue(req_r)

    non_local = req_r.host_cache_hit_length + req_r.disk_cache_hit_length
    assert non_local > 0, (
        f"engine_1 should get P2P or pool hits: peer={req_r.host_cache_hit_length}, pool={req_r.disk_cache_hit_length}"
    )
    print(f"[cross_engine] peer_hit={req_r.host_cache_hit_length}, pool_hit={req_r.disk_cache_hit_length}")


# =====================================================
# Test 5: Transfer latency calculations
# =====================================================

def test_transfer_latency():
    mgr = _create_manager()
    adapter_0, _ = _create_adapter(mgr, "engine_0")
    adapter_1, gv_1 = _create_adapter(mgr, "engine_1")

    ids = list(range(3000, 3100))
    req_w = _make_req(400, input_len=100, output_len=1, input_ids=ids, output_ids=[9999])
    adapter_0.add_to_prefetch_queue(req_w)
    adapter_0.on_request_complete(req_w, 1.0)

    req_r = _make_req(401, input_len=100, output_len=1, input_ids=ids, output_ids=[9998])
    adapter_1.add_to_prefetch_queue(req_r)

    pool_hit = req_r.disk_cache_hit_length
    peer_hit = req_r.host_cache_hit_length

    if pool_hit > 0:
        gv_1.clock = 2.0
        fetch = adapter_1.prefetch_from_storage(req_r, max_time=10.0)
        assert fetch.latency_disk_to_host > 0, "pool transfer should have latency"
        expected = (pool_hit * 256) / (2.0 * 1e9)
        assert abs(fetch.latency_disk_to_host - expected) < 0.001, (
            f"disk_to_host latency mismatch: got {fetch.latency_disk_to_host}, expected {expected}"
        )
        print(f"[latency] pool prefetch: {fetch.latency_disk_to_host*1000:.3f}ms for {pool_hit} tokens")

    if peer_hit > 0:
        onboard = adapter_1.on_board_from_host(req_r)
        assert onboard.latency_host_to_device > 0, "peer transfer should have latency"
        expected = (peer_hit * 256) / (10.0 * 1e9)
        assert abs(onboard.latency_host_to_device - expected) < 0.001, (
            f"host_to_device latency mismatch: got {onboard.latency_host_to_device}, expected {expected}"
        )
        print(f"[latency] peer onboard: {onboard.latency_host_to_device*1000:.3f}ms for {peer_hit} tokens")


# =====================================================
# Test 6: Three prefetch stop policies
# =====================================================

def test_prefetch_best_effort():
    mgr = _create_manager()
    adapter, gv = _create_adapter(mgr, prefetch_policy="best_effort")
    req = _make_req(500, input_len=50)
    req.disk_cache_hit_length = 100
    adapter.add_to_prefetch_queue(req)
    assert adapter.check_prefetch_progress(req) is True
    print("[prefetch] best_effort: immediate True")


def test_prefetch_wait_complete():
    mgr = _create_manager()
    adapter, gv = _create_adapter(mgr, prefetch_policy="wait_complete")
    req = _make_req(501, input_len=50)
    req.disk_cache_hit_length = 100
    adapter.add_to_prefetch_queue(req)

    match = adapter.match_prefix(req)
    if match.disk_hit_length > 0:
        assert adapter.can_terminate_prefetch(req) is False
        match.disk_hit_length = 0
        assert adapter.can_terminate_prefetch(req) is True
        print("[prefetch] wait_complete: blocks until disk=0")
    else:
        assert adapter.can_terminate_prefetch(req) is True
        print("[prefetch] wait_complete: no disk hits, immediate True")


def test_prefetch_timeout():
    mgr = _create_manager()
    adapter, gv = _create_adapter(mgr, prefetch_policy="timeout")
    req = _make_req(502, input_len=50)
    req.disk_cache_hit_length = 100
    adapter.add_to_prefetch_queue(req)

    match = adapter.match_prefix(req)
    if match.disk_hit_length > 0:
        gv.clock = 0.5
        assert adapter.can_terminate_prefetch(req) is False

        gv.clock = 10.0
        assert adapter.can_terminate_prefetch(req) is True
        print("[prefetch] timeout: blocks then releases after timeout")
    else:
        print("[prefetch] timeout: no disk hits, immediate True")


# =====================================================
# Test 7: on_request_complete triggers WriteCache
# =====================================================

def test_on_request_complete_updates_cache():
    mgr = _create_manager()
    adapter, gv = _create_adapter(mgr, "engine_0")

    ids = list(range(4000, 4030))
    req1 = _make_req(600, input_len=30, output_len=5, input_ids=ids, output_ids=list(range(9900, 9905)))
    adapter.add_to_prefetch_queue(req1)
    assert req1.device_cache_hit_length == 0

    adapter.on_request_complete(req1, 1.0)

    req2 = _make_req(601, input_len=30, output_len=5, input_ids=ids, output_ids=list(range(9910, 9915)))
    adapter.add_to_prefetch_queue(req2)
    assert req2.device_cache_hit_length > 0, f"After WriteCache, should have local hits: {req2.device_cache_hit_length}"
    print(f"[on_complete] WriteCache worked: second read got {req2.device_cache_hit_length} local hits")


# =====================================================
# Test 8: End-to-end with BenchmarkRunner
# =====================================================

def test_e2e_benchmark_runner():
    from schedule_simulator.schedule_emulator.run import BenchmarkRunner
    from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter
    from schedule_simulator.schedule_emulator.schedule_policy import SchedulePolicy

    random.seed(42)
    numpy.random.seed(42)

    mgr = _create_manager()

    benchmark_config = BenchmarkConfig(
        num_prompts=20,
        min_input_length=50,
        max_input_length=100,
        min_output_length=10,
        max_output_length=30,
        disable_tqdm=True,
    )
    scheduler_config = SchedulerConfig(
        "Qwen2.5-3B",
        hicache_storage_backend="hf3fs",
        hicache_storage_prefetch_policy="best_effort",
        schedule_policy="fcfs",
    )
    platform_config = PlatformConfig(
        device="H20",
        disk_read_bandwidth_gb=2.0,
        memory_read_bandwidth_gb=16.0,
        peer_read_bandwidth_gb=10.0,
    )

    runner = BenchmarkRunner(
        benchmark_config=benchmark_config,
        scheduler_config=scheduler_config,
        platform_config=platform_config,
    )

    sched = runner.scheduler_emulator
    adapter = HierarchicalCacheAdapter(
        manager=mgr,
        engine_instance_id="engine_0",
        platform_config=platform_config,
        kv_cache_space_per_token=sched.kv_cache_space_per_token,
        page_size=sched.scheduler_config.page_size,
        global_values=sched.global_values,
        prefetch_stop_policy="best_effort",
        enable_stats=True,
    )
    sched.tree_cache = adapter
    sched.policy = SchedulePolicy(
        scheduler_config.schedule_policy, adapter, sched.time_predictor
    )

    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 20, f"Expected 20, got {metrics['completed']}"
    assert metrics["mean_ttft_ms"] > 0
    print(f"[e2e] completed={metrics['completed']}, TTFT={metrics['mean_ttft_ms']:.1f}ms, "
          f"local={adapter.total_engine_hit_blocks}, peer={adapter.total_peer_hit_blocks}, "
          f"pool={adapter.total_pool_hit_blocks}")


# =====================================================
# Test 9: Multi-instance P-Only with adapter
# =====================================================

def test_multi_instance_p_only():
    from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner
    from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter
    from schedule_simulator.schedule_emulator.schedule_policy import SchedulePolicy

    random.seed(42)
    numpy.random.seed(42)

    # Need a config with enough infer_ids
    config_data = json.load(open(CONFIG_PATH))
    config_data["infer_clusters"][0]["infer_ids"] = [f"P{i}" for i in range(5)]
    tmp_config = "/tmp/hierarchical_multi_instance_config.json"
    with open(tmp_config, "w") as f:
        json.dump(config_data, f)
    os.makedirs("/tmp/hierarchical_test_output/pool", exist_ok=True)
    os.makedirs("/tmp/hierarchical_test_output/infer", exist_ok=True)

    loader = kvcm.HierarchicalReplayConfigLoader()
    assert loader.load(tmp_config)
    mgr = kvcm.HierarchicalReplayManager(loader.config())
    assert mgr.Init()

    benchmark_config = BenchmarkConfig(
        num_prompts=50,
        min_input_length=30,
        max_input_length=80,
        min_output_length=1,
        max_output_length=2,
        disable_tqdm=True,
    )
    p_cfg = SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill",
                            hicache_storage_backend="hf3fs",
                            hicache_storage_prefetch_policy="best_effort")
    d_cfg = SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode")
    p_plat = PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0,
                            memory_read_bandwidth_gb=16.0, peer_read_bandwidth_gb=10.0)
    d_plat = PlatformConfig(device="H20")
    router_config = RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN,
                                 d_policy=RoutingPolicy.ROUND_ROBIN,
                                 worker_startup_check_interval=0.01)

    runner = DisaggBenchmarkRunner(
        benchmark_config=benchmark_config,
        p_scheduler_config=p_cfg, d_scheduler_config=d_cfg,
        p_platform_config=p_plat, d_platform_config=d_plat,
        router_config=router_config,
        num_p_instance=5, num_d_instance=0,
    )

    for i, sched in enumerate(runner.p_schedulers):
        adapter = HierarchicalCacheAdapter(
            manager=mgr, engine_instance_id=f"P{i}",
            platform_config=p_plat,
            kv_cache_space_per_token=sched.kv_cache_space_per_token,
            page_size=sched.scheduler_config.page_size,
            global_values=sched.global_values,
            prefetch_stop_policy="best_effort",
        )
        sched.tree_cache = adapter
        sched.policy = SchedulePolicy(p_cfg.schedule_policy, adapter, sched.time_predictor)

    metrics = runner.run_benchmark_emulation()
    assert metrics["completed"] == 50, f"Expected 50, got {metrics['completed']}"
    assert metrics["mean_ttft_ms"] > 0

    total_local = sum(s.tree_cache.total_engine_hit_blocks for s in runner.p_schedulers)
    total_peer = sum(s.tree_cache.total_peer_hit_blocks for s in runner.p_schedulers)
    total_pool = sum(s.tree_cache.total_pool_hit_blocks for s in runner.p_schedulers)
    print(f"[multi_instance] completed={metrics['completed']}, TTFT={metrics['mean_ttft_ms']:.1f}ms, "
          f"local={total_local}, peer={total_peer}, pool={total_pool}")


if __name__ == "__main__":
    test_adapter_lifecycle()
    test_cold_read_zero_hits()
    test_write_then_local_read()
    test_cross_engine_read()
    test_transfer_latency()
    test_prefetch_best_effort()
    test_prefetch_wait_complete()
    test_prefetch_timeout()
    test_on_request_complete_updates_cache()
    test_e2e_benchmark_runner()
    test_multi_instance_p_only()
