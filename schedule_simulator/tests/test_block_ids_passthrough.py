"""
Tests for block_ids passthrough in statistical mode.
Validates: field parsing, FakeRequest population, adapter uses real block_ids,
           hierarchical Runner produces real prefix hits with enriched data,
           comparison with/without block_ids, standalone consistency.
"""
import os, sys, json, random
import numpy as np
import pytest

KVCM_SO_DIR = "/sgl-workspace/claude_workspace/tair-kvcache/bazel-bin/kv_cache_manager/optimizer/pybind"
if KVCM_SO_DIR not in sys.path:
    sys.path.insert(0, KVCM_SO_DIR)
try:
    import kvcm_py_optimizer as kvcm
    HAS_KVCM = True
except ImportError:
    HAS_KVCM = False

from schedule_simulator.schedule_emulator.types import *
from schedule_simulator.schedule_emulator.run import BenchmarkRunner, DisaggBenchmarkRunner
from schedule_simulator.schedule_emulator.base import GlobalValues

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "assets/glm5_sample")
ENRICHED_INPUT = os.path.join(SAMPLE_DIR, "glm5_enriched_input.jsonl")
PLAIN_INPUT = os.path.join(SAMPLE_DIR, "glm5_plain_input.jsonl")

pytestmark = pytest.mark.skipif(not HAS_KVCM, reason="kvcm_py_optimizer not available")


# ===========================================================================
# Test 1: block_ids field is parsed and populated in FakeRequest
# ===========================================================================

def test_block_ids_parsed_into_origin_input_ids():
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(dataset_path=ENRICHED_INPUT, num_prompts=10, disable_tqdm=True),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", chunked_prefill_size=8192, hicache_storage_backend="hf3fs"),
        platform_config=PlatformConfig(device="H20", memory_read_bandwidth_gb=16.0, disk_read_bandwidth_gb=2.0),
    )
    runner.run_benchmark_emulation()
    results = runner.get_response_results()
    for r in results:
        assert r.origin_input_ids is not None, "origin_input_ids should be populated from block_ids"
        assert len(r.origin_input_ids) > 0, "block_ids should not be empty"
    print("[parsed] All %d requests have origin_input_ids from block_ids" % len(results))


# ===========================================================================
# Test 2: Without block_ids field, origin_input_ids stays None
# ===========================================================================

def test_no_block_ids_stays_none():
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(dataset_path=PLAIN_INPUT, num_prompts=10, disable_tqdm=True),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", chunked_prefill_size=8192, hicache_storage_backend="hf3fs"),
        platform_config=PlatformConfig(device="H20", memory_read_bandwidth_gb=16.0, disk_read_bandwidth_gb=2.0),
    )
    runner.run_benchmark_emulation()
    results = runner.get_response_results()
    for r in results:
        assert r.origin_input_ids is None, "Without block_ids, origin_input_ids should be None"
    print("[no_block_ids] All %d requests have None origin_input_ids" % len(results))


# ===========================================================================
# Test 3: Adapter uses real block_ids when available
# ===========================================================================

def test_adapter_uses_real_block_ids():
    from schedule_simulator.schedule_emulator.hierarchical_config_builder import build_hierarchical_config
    from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter

    cfg = build_hierarchical_config(
        SchedulerConfig("Qwen2.5-3B", hicache_storage_backend="hf3fs", hicache_read_query_type="prefix_match"),
        PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0, memory_read_bandwidth_gb=16.0, memory_capacity_gb=64.0),
        p_instance_ids=["E0"], output_dir="/tmp/block_ids_test_adapter",
        storage_pool_capacity_gb=2.0,
    )
    loader = kvcm.HierarchicalReplayConfigLoader(); assert loader.load(cfg)
    mgr = kvcm.HierarchicalReplayManager(loader.config()); assert mgr.Init()

    gv = GlobalValues()
    adapter = HierarchicalCacheAdapter(
        manager=mgr, engine_instance_id="E0",
        platform_config=PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0, memory_read_bandwidth_gb=16.0),
        kv_cache_space_per_token=256, page_size=1, global_values=gv,
    )

    real_ids = [111, 222, 333, 444, 555]
    req_with = FakeRequest(id=1, input_token_length=5, output_token_length=1, origin_input_ids=real_ids, output_ids=[0])
    req_without = FakeRequest(id=2, input_token_length=5, output_token_length=1, output_ids=[0])

    # Adapter should use real_ids for req_with, synthetic for req_without
    ids_with = adapter._req_to_block_ids(req_with)
    ids_without = adapter._req_to_block_ids(req_without)

    assert ids_with == real_ids, "Should use real block_ids"
    assert ids_without[0] == 2000000, "Should use synthetic block_ids when origin_input_ids is None"
    print("[adapter] real ids=%s, synthetic ids=%s..." % (ids_with[:3], ids_without[:3]))


# ===========================================================================
# Test 4: E2E Runner with enriched data produces real prefix hits
# ===========================================================================

def test_e2e_enriched_has_real_hits():
    random.seed(42); np.random.seed(42)
    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(dataset_path=ENRICHED_INPUT, num_prompts=100, disable_tqdm=True),
        p_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill",
            chunked_prefill_size=8192, hicache_storage_backend="hf3fs"),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0,
            memory_read_bandwidth_gb=16.0, memory_capacity_gb=64.0, peer_read_bandwidth_gb=10.0),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN,
            worker_startup_check_interval=0.01),
        num_p_instance=5, num_d_instance=0,
        enable_hierarchical=True, hierarchical_output_dir="/tmp/block_ids_e2e",
    )
    m = runner.run_benchmark_emulation()
    h = runner.get_hierarchical_metrics()

    assert m["completed"] == 100
    total_hit = h["total_engine_hit_blocks"] + h["total_peer_hit_blocks"] + h["total_pool_hit_blocks"]
    assert total_hit > 0, "Enriched data should produce real prefix hits, got 0"
    assert h["total_blocks_queried"] > 0
    assert h["block_hit_ratio"] > 0, "Block hit ratio should be > 0"

    print("[e2e_enriched] local=%d peer=%d pool=%d total=%d blocks_queried=%d block_hit_ratio=%.3f" % (
        h["total_engine_hit_blocks"], h["total_peer_hit_blocks"], h["total_pool_hit_blocks"],
        total_hit, h["total_blocks_queried"], h["block_hit_ratio"]))


# ===========================================================================
# Test 5: Without block_ids, E2E Runner produces zero hits (synthetic ids)
# ===========================================================================

def test_e2e_plain_has_zero_hits():
    random.seed(42); np.random.seed(42)
    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(dataset_path=PLAIN_INPUT, num_prompts=50, disable_tqdm=True),
        p_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill",
            chunked_prefill_size=8192, hicache_storage_backend="hf3fs"),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0,
            memory_read_bandwidth_gb=16.0, memory_capacity_gb=64.0),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN,
            worker_startup_check_interval=0.01),
        num_p_instance=3, num_d_instance=0,
        enable_hierarchical=True, hierarchical_output_dir="/tmp/block_ids_plain",
    )
    m = runner.run_benchmark_emulation()
    h = runner.get_hierarchical_metrics()

    assert m["completed"] == 50
    total_hit = h["total_engine_hit_blocks"] + h["total_peer_hit_blocks"] + h["total_pool_hit_blocks"]
    assert total_hit == 0, "Without real block_ids, should have 0 hits, got %d" % total_hit
    print("[e2e_plain] total_hit=0 as expected (synthetic block_ids)")


# ===========================================================================
# Test 6: Enriched E2E results match standalone Optimizer
# ===========================================================================

def test_e2e_matches_standalone():
    from schedule_simulator.schedule_emulator.hierarchical_config_builder import build_hierarchical_config
    from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter

    # Load enriched records
    records = []
    with open(ENRICHED_INPUT) as f:
        for line in f:
            records.append(json.loads(line))
    records = records[:50]

    pc = PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0,
                         memory_read_bandwidth_gb=16.0, memory_capacity_gb=64.0, peer_read_bandwidth_gb=10.0)
    sc = SchedulerConfig("Qwen2.5-3B", hicache_storage_backend="hf3fs", hicache_read_query_type="prefix_match")
    pods = ["P0", "P1", "P2", "P3", "P4"]

    # Standalone
    cfg_s = build_hierarchical_config(sc, pc, p_instance_ids=pods,
        output_dir="/tmp/block_ids_standalone", storage_pool_capacity_gb=2.0, enable_p2p=True)
    ls = kvcm.HierarchicalReplayConfigLoader(); assert ls.load(cfg_s)
    ms = kvcm.HierarchicalReplayManager(ls.config()); assert ms.Init()

    # Integrated
    cfg_i = build_hierarchical_config(sc, pc, p_instance_ids=pods,
        output_dir="/tmp/block_ids_integrated", storage_pool_capacity_gb=2.0, enable_p2p=True)
    li = kvcm.HierarchicalReplayConfigLoader(); assert li.load(cfg_i)
    mi = kvcm.HierarchicalReplayManager(li.config()); assert mi.Init()

    ads, gvs = {}, {}
    for p in pods:
        gvs[p] = GlobalValues()
        ads[p] = HierarchicalCacheAdapter(manager=mi, engine_instance_id=p, platform_config=pc,
            kv_cache_space_per_token=256, page_size=1, global_values=gvs[p], read_query_type="prefix_match")

    s_hits, i_hits = [], []
    for idx, r in enumerate(records):
        pod = pods[idx % len(pods)]
        block_ids = r["block_ids"]
        input_len = r["input_length"]
        ts_ns = int(r["timestamp"] * 1e6)

        # Drop partial tail blocks
        block_size = 1
        max_full_blocks = input_len // block_size
        if max_full_blocks == 0 and block_ids:
            max_full_blocks = 1  # backward compatibility
        full_block_ids = block_ids[:max_full_blocks]

        res = ms.GetCacheLocation(pod, "r%d"%idx, ts_ns, full_block_ids, input_len)
        s_hits.append(res.engine_hit_length + res.peer_hit_length + res.storage_pool_hit_length)
        ms.WriteCache(pod, "w%d"%idx, ts_ns+1, full_block_ids)

        req = FakeRequest(id=idx, input_token_length=input_len, output_token_length=1,
                          origin_input_ids=full_block_ids, output_ids=[0])
        gvs[pod].clock = r["timestamp"] / 1000
        ads[pod].add_to_prefetch_queue(req)
        i_hits.append(req.device_cache_hit_length + req.host_cache_hit_length + req.disk_cache_hit_length)
        ads[pod].on_request_complete(req, gvs[pod].clock + 0.001)

    matches = sum(1 for s, i in zip(s_hits, i_hits) if s == i)
    assert matches == len(s_hits), "Expected 100%% match, got %d/%d" % (matches, len(s_hits))
    print("[consistency] %d/%d match, standalone=%d integrated=%d" % (matches, len(s_hits), sum(s_hits), sum(i_hits)))


# ===========================================================================
# Test 7: Block-level hit metrics are correct
# ===========================================================================

def test_block_level_metrics():
    random.seed(42); np.random.seed(42)
    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(dataset_path=ENRICHED_INPUT, num_prompts=50, disable_tqdm=True),
        p_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill",
            chunked_prefill_size=8192, hicache_storage_backend="hf3fs"),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0,
            memory_read_bandwidth_gb=16.0, memory_capacity_gb=64.0, peer_read_bandwidth_gb=10.0),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN,
            worker_startup_check_interval=0.01),
        num_p_instance=3, num_d_instance=0,
        enable_hierarchical=True, hierarchical_output_dir="/tmp/block_metrics_test",
    )
    runner.run_benchmark_emulation()
    h = runner.get_hierarchical_metrics()

    assert "total_blocks_queried" in h
    assert "total_blocks_hit" in h
    assert "block_hit_ratio" in h
    assert h["total_blocks_queried"] > 0
    assert h["total_blocks_hit"] >= 0
    assert 0 <= h["block_hit_ratio"] <= 1.0
    assert h["total_blocks_hit"] == h["total_engine_hit_blocks"] + h["total_peer_hit_blocks"] + h["total_pool_hit_blocks"]
    print("[block_metrics] queried=%d hit=%d ratio=%.3f" % (h["total_blocks_queried"], h["total_blocks_hit"], h["block_hit_ratio"]))


# ===========================================================================
# Test 8: instance_id field is parsed (extra_key)
# ===========================================================================

def test_instance_id_parsed():
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(dataset_path=ENRICHED_INPUT, num_prompts=5, disable_tqdm=True),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", chunked_prefill_size=8192, hicache_storage_backend="hf3fs"),
        platform_config=PlatformConfig(device="H20", memory_read_bandwidth_gb=16.0, disk_read_bandwidth_gb=2.0),
    )
    runner.run_benchmark_emulation()
    results = runner.get_response_results()
    has_instance = sum(1 for r in results if r.extra_key is not None)
    assert has_instance > 0, "instance_id should be parsed into extra_key"
    print("[instance_id] %d/%d requests have extra_key from instance_id" % (has_instance, len(results)))


# ===========================================================================
# Test 9: Per-request records include block counts
# ===========================================================================

def test_per_request_records_have_num_blocks():
    random.seed(42); np.random.seed(42)
    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(dataset_path=ENRICHED_INPUT, num_prompts=20, disable_tqdm=True),
        p_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill",
            chunked_prefill_size=8192, hicache_storage_backend="hf3fs"),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0,
            memory_read_bandwidth_gb=16.0, memory_capacity_gb=64.0),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN,
            worker_startup_check_interval=0.01),
        num_p_instance=2, num_d_instance=0,
        enable_hierarchical=True, hierarchical_output_dir="/tmp/block_records_test",
    )
    runner.run_benchmark_emulation()
    h = runner.get_hierarchical_metrics()

    for r in h["read_records"]:
        assert r.num_blocks > 0, "num_blocks should be > 0"
        assert r.num_blocks <= r.input_length, "num_blocks should be <= input_length (tokens)"
    print("[per_request] All %d records have valid num_blocks" % len(h["read_records"]))


if __name__ == "__main__":
    test_block_ids_parsed_into_origin_input_ids()
    test_no_block_ids_stays_none()
    test_adapter_uses_real_block_ids()
    test_e2e_enriched_has_real_hits()
    test_e2e_plain_has_zero_hits()
    test_e2e_matches_standalone()
    test_block_level_metrics()
    test_instance_id_parsed()
    test_per_request_records_have_num_blocks()
