"""P6: Accuracy validation with GLM5 enriched data (real block_hash_ids)."""
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

ENRICHED = "/sgl-workspace/claude_workspace/data/glm5-model-prefill-88e7_sample200.jsonl"
STAT_INPUT = os.path.join(os.path.dirname(__file__), "assets/glm5_sample/glm5_sim_input.jsonl")

def _load_enriched(n=100):
    recs = []
    with open(ENRICHED) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs[:n]

def _to_block_ids(hashes):
    return [int(b, 16) % (2**63) for b in hashes]


def test_path1_stat_mode():
    runner = BenchmarkRunner(
        benchmark_config=BenchmarkConfig(dataset_path=STAT_INPUT, num_prompts=20, disable_tqdm=True),
        scheduler_config=SchedulerConfig("Qwen2.5-3B", chunked_prefill_size=8192, hicache_storage_backend="hf3fs"),
        platform_config=PlatformConfig(device="H20", memory_read_bandwidth_gb=16.0, disk_read_bandwidth_gb=2.0),
    )
    m = runner.run_benchmark_emulation()
    assert m["completed"] == 20
    print("[path1] TTFT=%.0fms" % m["mean_ttft_ms"])


@pytest.mark.skipif(not HAS_KVCM, reason="no kvcm")
def test_optimizer_standalone_real_hits():
    from schedule_simulator.schedule_emulator.hierarchical_config_builder import build_hierarchical_config
    recs = _load_enriched(100)
    pods = sorted(set(r["pods"][0] for r in recs))[:10]
    cfg = build_hierarchical_config(
        SchedulerConfig("Qwen2.5-3B", hicache_storage_backend="hf3fs", hicache_read_query_type="prefix_match"),
        PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0, memory_read_bandwidth_gb=16.0, memory_capacity_gb=64.0),
        p_instance_ids=pods, output_dir="/tmp/glm5_p6_sa", storage_pool_capacity_gb=2.0, enable_p2p=True)
    loader = kvcm.HierarchicalReplayConfigLoader(); assert loader.load(cfg)
    mgr = kvcm.HierarchicalReplayManager(loader.config()); assert mgr.Init()
    ps = set(pods); te, tp, ts, tb = 0, 0, 0, 0
    for i, r in enumerate(recs):
        pod = r["pods"][0] if r["pods"][0] in ps else pods[i % len(pods)]
        bids = _to_block_ids(r["input_block_hash_ids"]); tn = int(r["timestamp"] * 1e9)
        res = mgr.GetCacheLocation(pod, "r%d"%i, tn, bids, len(bids))  # input_len in tokens
        te += res.engine_hit_length; tp += res.peer_hit_length; ts += res.storage_pool_hit_length; tb += len(bids)
        mgr.WriteCache(pod, "w%d"%i, tn+1, bids)
    total = te + tp + ts
    assert total > 0, "Should have real prefix hits"
    print("[standalone] engine=%d peer=%d pool=%d total=%d/%d (%.1f%%)" % (te, tp, ts, total, tb, total/tb*100))


@pytest.mark.skipif(not HAS_KVCM, reason="no kvcm")
def test_standalone_vs_integrated_100pct():
    from schedule_simulator.schedule_emulator.hierarchical_config_builder import build_hierarchical_config
    from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter
    recs = _load_enriched(100)
    pods = sorted(set(r["pods"][0] for r in recs))[:10]; ps = set(pods)
    pc = PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0, memory_read_bandwidth_gb=16.0,
                         memory_capacity_gb=64.0, peer_read_bandwidth_gb=10.0)
    sc = SchedulerConfig("Qwen2.5-3B", hicache_storage_backend="hf3fs", hicache_read_query_type="prefix_match")
    cs = build_hierarchical_config(sc, pc, p_instance_ids=pods, output_dir="/tmp/glm5_p6_cs", storage_pool_capacity_gb=2.0, enable_p2p=True)
    ci = build_hierarchical_config(sc, pc, p_instance_ids=pods, output_dir="/tmp/glm5_p6_ci", storage_pool_capacity_gb=2.0, enable_p2p=True)
    ls = kvcm.HierarchicalReplayConfigLoader(); assert ls.load(cs)
    ms = kvcm.HierarchicalReplayManager(ls.config()); assert ms.Init()
    li = kvcm.HierarchicalReplayConfigLoader(); assert li.load(ci)
    mi = kvcm.HierarchicalReplayManager(li.config()); assert mi.Init()
    ads, gvs = {}, {}
    for p in pods:
        gvs[p] = GlobalValues()
        ads[p] = HierarchicalCacheAdapter(manager=mi, engine_instance_id=p, platform_config=pc,
            kv_cache_space_per_token=256, page_size=1, global_values=gvs[p], read_query_type="prefix_match")
    sh, ih = [], []
    for i, r in enumerate(recs):
        pod = r["pods"][0] if r["pods"][0] in ps else pods[i % len(pods)]
        bids = _to_block_ids(r["input_block_hash_ids"])
        input_len = r.get("input_length", len(bids) * 256)  # Use actual input_length
        tn = int(r["timestamp"] * 1e9)
        
        # Drop partial tail blocks
        block_size = 1
        max_full_blocks = input_len // block_size
        if max_full_blocks == 0 and bids:
            max_full_blocks = 1  # backward compatibility
        full_bids = bids[:max_full_blocks]
        
        res = ms.GetCacheLocation(pod, "r%d"%i, tn, full_bids, input_len)
        sv = res.engine_hit_length + res.peer_hit_length + res.storage_pool_hit_length; sh.append(sv)
        ms.WriteCache(pod, "w%d"%i, tn+1, full_bids)
        req = FakeRequest(id=i, input_token_length=input_len, output_token_length=1, origin_input_ids=full_bids, output_ids=[0])
        gvs[pod].clock = r["timestamp"]; ads[pod].add_to_prefetch_queue(req)
        iv = req.device_cache_hit_length + req.host_cache_hit_length + req.disk_cache_hit_length; ih.append(iv)
        ads[pod].on_request_complete(req, r["timestamp"] + 0.001)
    matches = sum(1 for s, i in zip(sh, ih) if s == i)
    assert matches == len(sh), "Expected 100%% match, got %d/%d" % (matches, len(sh))
    assert sum(sh) == sum(ih)
    print("[consistency] %d/%d match, standalone=%d integrated=%d" % (matches, len(sh), sum(sh), sum(ih)))


@pytest.mark.skipif(not HAS_KVCM, reason="no kvcm")
def test_e2e_runner_with_enriched():
    recs = _load_enriched(50)
    tmp = "/tmp/glm5_p6_e2e.jsonl"
    with open(tmp, "w") as f:
        for r in recs:
            bids = _to_block_ids(r["input_block_hash_ids"])
            f.write(json.dumps({"timestamp": r["timestamp"]*1000, "input_length": len(bids), "output_length": 1, "device_cache_hit_length": 0}) + "\n")
    random.seed(42); np.random.seed(42)
    runner = DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(dataset_path=tmp, num_prompts=50, disable_tqdm=True),
        p_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill", chunked_prefill_size=8192, hicache_storage_backend="hf3fs"),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0, memory_read_bandwidth_gb=16.0, memory_capacity_gb=64.0),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN, d_policy=RoutingPolicy.ROUND_ROBIN, worker_startup_check_interval=0.01),
        num_p_instance=3, num_d_instance=0,
        enable_hierarchical=True, hierarchical_output_dir="/tmp/glm5_p6_e2e_out",
    )
    m = runner.run_benchmark_emulation()
    h = runner.get_hierarchical_metrics()
    assert m["completed"] == 50 and h["num_reads"] == 50
    print("[e2e] completed=%d TTFT=%.0fms local=%.3f peer=%.3f pool=%.3f" % (m["completed"], m["mean_ttft_ms"], h["engine_hit_block_ratio"], h["peer_hit_block_ratio"], h["pool_hit_block_ratio"]))


def test_enriched_has_reuse():
    from collections import Counter
    recs = _load_enriched(200)
    c = Counter()
    for r in recs: c.update(r["input_block_hash_ids"])
    reuse = 1 - len(c) / sum(c.values())
    assert reuse > 0.1
    print("[reuse] %.1f%%" % (reuse*100))
