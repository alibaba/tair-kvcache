"""
Tests for P2P transfer in request-level scheduling mode.
Verifies that peer_hit from GetCacheLocation is properly used to:
1. Reduce uncached tokens (skip prefill for peer-cached data)
2. Include P2P transfer latency (peer_read_bandwidth) in total request time
"""
import os
import sys
import json
import random
import asyncio
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

from schedule_simulator.schedule_emulator.types import (
    BenchmarkConfig, SchedulerConfig, PlatformConfig,
    RouterConfig, RoutingPolicy, FakeRequest,
    PrefixCacheFetchResult,
)
from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner
from schedule_simulator.schedule_emulator.base import GlobalValues

pytestmark = pytest.mark.skipif(not HAS_KVCM, reason="kvcm_py_optimizer not available")

CONFIG_TEMPLATE = os.path.join(os.path.dirname(__file__), "assets/hierarchical/test_config.json")


# ===========================================================================
# Helpers
# ===========================================================================

def _build_config(num_engines=2, page_size=1, capacity_gb=0.25):
    """Build hierarchical config JSON for testing P2P."""
    config = json.load(open(CONFIG_TEMPLATE))
    config["infer_clusters"][0]["infer_ids"] = [f"P{i}" for i in range(num_engines)]
    config["infer_clusters"][0]["model"]["block_size"] = page_size
    config["infer_clusters"][0]["model"]["bytes_per_token"] = 1
    config["infer_clusters"][0]["tiers"][0]["capacity"] = capacity_gb
    config["infer_clusters"][0]["tiers"][1]["capacity"] = capacity_gb * 2
    os.makedirs("/tmp/p2p_test_output/pool", exist_ok=True)
    config["output_result_path"] = "/tmp/p2p_test_output"
    config["storage_pool"]["output_result_path"] = "/tmp/p2p_test_output/pool"
    tmp_path = "/tmp/p2p_test_config.json"
    with open(tmp_path, "w") as f:
        json.dump(config, f)
    return tmp_path


def _make_adapter(num_engines=2, page_size=1, peer_bw_gb=10.0, kv_bytes=256):
    """Create HierarchicalCacheAdapter instances sharing a manager (for unit tests)."""
    from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter

    config_path = _build_config(num_engines=num_engines, page_size=page_size)
    loader = kvcm.HierarchicalReplayConfigLoader()
    assert loader.load(config_path)
    mgr = kvcm.HierarchicalReplayManager(loader.config())
    assert mgr.Init()

    gv = GlobalValues()
    platform = PlatformConfig(
        device="H20",
        disk_read_bandwidth_gb=2.0,
        memory_read_bandwidth_gb=16.0,
        peer_read_bandwidth_gb=peer_bw_gb,
    )

    adapters = []
    for i in range(num_engines):
        adapter = HierarchicalCacheAdapter(
            manager=mgr,
            engine_instance_id=f"P{i}",
            platform_config=platform,
            kv_cache_space_per_token=kv_bytes,
            page_size=page_size,
            global_values=gv,
            prefetch_stop_policy="best_effort",
            enable_stats=True,
        )
        adapters.append(adapter)

    return adapters, gv, mgr


# ===========================================================================
# Unit tests: HierarchicalCacheAdapter.on_board_from_host behavior
# ===========================================================================

class TestOnBoardFromHost:
    """Verify on_board_from_host correctly transfers peer data to device."""

    def test_peer_hit_transfers_to_device(self):
        """After write on engine0, read on engine1 should get peer_hit;
        on_board_from_host should move host_hit -> device_hit."""
        adapters, gv, mgr = _make_adapter(num_engines=2, page_size=1, kv_bytes=256)
        adapter0, adapter1 = adapters

        block_ids = list(range(100, 200))  # 100 blocks

        # Write on engine0
        req_w = FakeRequest(id=1, input_token_length=100, output_token_length=1,
                            origin_input_ids=block_ids, output_ids=[9999])
        adapter0.add_to_prefetch_queue(req_w)
        adapter0.on_request_complete(req_w, 1.0)

        # Read on engine1 - should get peer_hit
        gv.clock = 2.0
        req_r = FakeRequest(id=2, input_token_length=100, output_token_length=1,
                            origin_input_ids=block_ids, output_ids=[9998])
        adapter1.add_to_prefetch_queue(req_r)

        assert req_r.host_cache_hit_length > 0, \
            f"Expected peer_hit > 0, got {req_r.host_cache_hit_length}"
        assert req_r.device_cache_hit_length == 0, \
            "Data is on peer, not local engine"

        # Before on_board_from_host - save values (match_prefix returns cached mutable object)
        match_before = adapter1.match_prefix(req_r)
        host_hit_before = match_before.host_hit_length
        device_hit_before = match_before.device_hit_length
        assert host_hit_before > 0
        assert device_hit_before == 0

        # Perform P2P transfer
        fetch_result = adapter1.on_board_from_host(req_r)

        # After on_board_from_host (same cached object is mutated)
        match_after = adapter1.match_prefix(req_r)
        assert match_after.device_hit_length > 0, \
            "After P2P transfer, device_hit should include transferred peer data"
        assert match_after.host_hit_length == 0, \
            "After transfer, host_hit should be 0"
        assert match_after.device_hit_length == host_hit_before, \
            f"All peer data should transfer: device_hit={match_after.device_hit_length} vs host_before={host_hit_before}"

        print(f"[peer_transfer] host_hit={match_before.host_hit_length} -> "
              f"device_hit={match_after.device_hit_length}")

    def test_p2p_latency_calculated(self):
        """P2P transfer latency should be tokens * kv_bytes / peer_bw."""
        peer_bw_gb = 5.0  # 5 GB/s = 5e9 bytes/s
        kv_bytes = 1000   # 1000 bytes per token
        adapters, gv, mgr = _make_adapter(
            num_engines=2, page_size=1,
            peer_bw_gb=peer_bw_gb, kv_bytes=kv_bytes,
        )
        adapter0, adapter1 = adapters

        block_ids = list(range(200, 300))  # 100 blocks -> 100 tokens

        # Write on engine0
        req_w = FakeRequest(id=1, input_token_length=100, output_token_length=1,
                            origin_input_ids=block_ids, output_ids=[9999])
        adapter0.add_to_prefetch_queue(req_w)
        adapter0.on_request_complete(req_w, 1.0)

        # Read on engine1
        gv.clock = 2.0
        req_r = FakeRequest(id=2, input_token_length=100, output_token_length=1,
                            origin_input_ids=block_ids, output_ids=[9998])
        adapter1.add_to_prefetch_queue(req_r)

        fetch_result = adapter1.on_board_from_host(req_r)

        # Expected latency: tokens * kv_bytes / bw
        peer_hit_tokens = req_r.host_cache_hit_length * 1  # page_size=1
        expected_latency = (peer_hit_tokens * kv_bytes) / (peer_bw_gb * 1e9)
        assert fetch_result.latency_host_to_device > 0, \
            "P2P transfer should have non-zero latency"
        assert abs(fetch_result.latency_host_to_device - expected_latency) < 1e-12, \
            f"Expected latency={expected_latency}, got {fetch_result.latency_host_to_device}"

        print(f"[p2p_latency] {peer_hit_tokens} tokens, latency={fetch_result.latency_host_to_device:.9f}s")

    def test_no_peer_hit_no_transfer(self):
        """When there is no peer data, on_board_from_host returns zero latency."""
        adapters, gv, mgr = _make_adapter(num_engines=2, page_size=1)
        adapter1 = adapters[1]

        # Cold read - no data written anywhere
        block_ids = list(range(500, 550))
        req = FakeRequest(id=10, input_token_length=50, output_token_length=1,
                          origin_input_ids=block_ids, output_ids=[7777])
        adapter1.add_to_prefetch_queue(req)
        assert req.host_cache_hit_length == 0
        assert req.device_cache_hit_length == 0

        fetch_result = adapter1.on_board_from_host(req)
        assert fetch_result.latency_host_to_device == 0
        assert fetch_result.fetched_tokens == 0
        print("[no_peer] No P2P data, zero latency")

    def test_local_engine_hit_no_p2p_needed(self):
        """Data on local engine should be device_hit, no P2P transfer needed."""
        adapters, gv, mgr = _make_adapter(num_engines=2, page_size=1)
        adapter0 = adapters[0]

        block_ids = list(range(600, 650))

        # Write on engine0
        req_w = FakeRequest(id=20, input_token_length=50, output_token_length=1,
                            origin_input_ids=block_ids, output_ids=[8888])
        adapter0.add_to_prefetch_queue(req_w)
        adapter0.on_request_complete(req_w, 1.0)

        # Read on same engine0 -> engine_hit (device), no peer_hit
        gv.clock = 2.0
        req_r = FakeRequest(id=21, input_token_length=50, output_token_length=1,
                            origin_input_ids=block_ids, output_ids=[8887])
        adapter0.add_to_prefetch_queue(req_r)

        assert req_r.device_cache_hit_length > 0, "Should hit local engine cache"
        assert req_r.host_cache_hit_length == 0, "No peer data needed"

        fetch_result = adapter0.on_board_from_host(req_r)
        assert fetch_result.latency_host_to_device == 0
        assert fetch_result.fetched_tokens == 0

        match_result = adapter0.match_prefix(req_r)
        assert match_result.device_hit_length > 0
        print(f"[local_hit] device_hit={match_result.device_hit_length}, no P2P needed")


# ===========================================================================
# Integration tests: _run_request_level with P2P
# ===========================================================================

class TestRequestLevelP2P:
    """Integration tests for P2P in _run_request_level."""

    def _make_dataset(self, records, path="/tmp/p2p_test_dataset.jsonl"):
        """Create a dataset file with block_ids."""
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    def _make_runner(self, num_p=2, peer_bw=10.0, page_size=1,
                     dataset_path=None, num_prompts=10):
        """Create a DisaggBenchmarkRunner with request_level + hierarchical + P2P."""
        random.seed(42)
        np.random.seed(42)
        config = BenchmarkConfig(
            dataset_path=dataset_path,
            num_prompts=num_prompts,
            min_input_length=50, max_input_length=100,
            min_output_length=1, max_output_length=2,
            disable_tqdm=True,
        )
        return DisaggBenchmarkRunner(
            benchmark_config=config,
            p_scheduler_config=SchedulerConfig(
                "Qwen2.5-3B", scenario="disagg_prefill",
                request_level_scheduling=True,
                max_num_tokens=999999999,
                page_size=page_size,
                enable_stats=True,
            ),
            d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
            p_platform_config=PlatformConfig(
                device="H20",
                disk_read_bandwidth_gb=2.0,
                memory_read_bandwidth_gb=16.0,
                peer_read_bandwidth_gb=peer_bw,
            ),
            d_platform_config=PlatformConfig(device="H20"),
            router_config=RouterConfig(
                p_policy=RoutingPolicy.ROUND_ROBIN,
                d_policy=RoutingPolicy.ROUND_ROBIN,
                worker_startup_check_interval=0.01,
            ),
            num_p_instance=num_p, num_d_instance=0,
            enable_hierarchical=True, enable_p2p=True,
            storage_pool_capacity_gb=0.001,
            hierarchical_output_dir="/tmp/p2p_test_hier_output",
        )

    def test_p2p_reduces_uncached_tokens(self):
        """Requests routed to engine with peer data should have fewer uncached tokens."""
        # Create dataset: all requests share same block_ids prefix
        shared_prefix = list(range(1000, 1050))  # 50 blocks shared
        records = []
        for i in range(10):
            records.append({
                # Timestamps spaced 1s apart (>> prefill latency ~0.2s)
                # so that each request arrives after the previous completes,
                # allowing WriteCache to be visible to subsequent GetCacheLocation.
                "timestamp": float(i) * 1000,
                "input_length": 50,
                "output_length": 1,
                "block_ids": shared_prefix,
            })
        dataset_path = self._make_dataset(records)

        runner = self._make_runner(num_p=2, dataset_path=dataset_path, num_prompts=10)
        metrics = runner.run_benchmark_emulation()
        assert metrics["completed"] == 10

        # With round-robin on 2 instances, requests alternate:
        # Engine P0 gets requests 0,2,4,6,8 and P1 gets 1,3,5,7,9
        # After req 0 writes on P0, req 1 on P1 should get peer_hit from P0
        # After req 1 writes on P1, req 2 on P0 should get engine_hit (from self)
        # The key test: peer_hit requests should have reused_tokens > 0
        hier = runner.get_hierarchical_metrics()
        total_peer_hit = hier.get("total_peer_hit_blocks", 0)
        assert total_peer_hit > 0, \
            f"With shared prefixes on 2 engines, should have peer hits. Got {total_peer_hit}"

        # Check that requests with peer_hit have final_reused_tokens > 0
        all_completed = []
        for sched in runner.p_schedulers:
            all_completed.extend(sched.completed_requests)

        reused = [r.final_reused_tokens for r in all_completed if r.final_reused_tokens > 0]
        assert len(reused) > 0, \
            "Some requests should have reused tokens from P2P transfer"
        print(f"[p2p_reduces_uncached] peer_hit_blocks={total_peer_hit}, "
              f"requests_with_reuse={len(reused)}/{len(all_completed)}")

    def test_p2p_bandwidth_no_effect_on_ttft(self):
        """P2P bandwidth should NOT affect TTFT (Optimizer handles P2P internally)."""
        shared_prefix = list(range(3000, 3050))
        records = []
        for i in range(4):
            records.append({
                "timestamp": float(i) * 10,
                "input_length": 50,
                "output_length": 1,
                "block_ids": shared_prefix,
            })
        dataset_path = self._make_dataset(records)

        # Fast P2P: 100 GB/s
        runner_fast = self._make_runner(
            num_p=2, peer_bw=100.0, dataset_path=dataset_path, num_prompts=4
        )
        metrics_fast = runner_fast.run_benchmark_emulation()

        # Slow P2P: 1 GB/s
        runner_slow = self._make_runner(
            num_p=2, peer_bw=1.0, dataset_path=dataset_path, num_prompts=4
        )
        metrics_slow = runner_slow.run_benchmark_emulation()

        assert metrics_fast["completed"] == 4
        assert metrics_slow["completed"] == 4

        # Slow bandwidth should result in higher mean_ttft if there are peer hits
        hier_fast = runner_fast.get_hierarchical_metrics()
        hier_slow = runner_slow.get_hierarchical_metrics()

        if hier_fast.get("total_peer_hit_blocks", 0) > 0:
            # Mean TTFT with slow P2P should be >= fast P2P
            assert metrics_slow["mean_ttft_ms"] >= metrics_fast["mean_ttft_ms"], \
                (f"Slow P2P ({metrics_slow['mean_ttft_ms']:.1f}ms) should be >= "
                 f"fast P2P ({metrics_fast['mean_ttft_ms']:.1f}ms)")
            print(f"[bw_comparison] fast_ttft={metrics_fast['mean_ttft_ms']:.1f}ms, "
                  f"slow_ttft={metrics_slow['mean_ttft_ms']:.1f}ms")
        else:
            print("[bw_comparison] No peer hits in this run, skipping comparison")

    def test_no_p2p_when_local_hit(self):
        """Requests hitting local engine cache should have zero P2P latency."""
        # Use 1 instance so all requests hit local cache after first write
        shared_prefix = list(range(4000, 4050))
        records = []
        for i in range(4):
            records.append({
                "timestamp": float(i) * 10,
                "input_length": 50,
                "output_length": 1,
                "block_ids": shared_prefix,
            })
        dataset_path = self._make_dataset(records)

        runner = self._make_runner(
            num_p=1, peer_bw=0.001, dataset_path=dataset_path, num_prompts=4
        )
        metrics = runner.run_benchmark_emulation()
        assert metrics["completed"] == 4

        hier = runner.get_hierarchical_metrics()
        # With 1 instance, all hits should be engine_hit (local), no peer_hit
        assert hier.get("total_peer_hit_blocks", 0) == 0, \
            "Single instance should have no peer hits"
        assert hier.get("total_engine_hit_blocks", 0) > 0, \
            "Should have local engine hits after first write"
        print(f"[local_only] engine_hit={hier["total_engine_hit_blocks"]}, "
              f"peer_hit={hier["total_peer_hit_blocks"]}")

    def test_cold_cache_no_p2p_overhead(self):
        """With all unique data, no P2P transfer should happen."""
        records = []
        for i in range(6):
            # Each request has unique block_ids
            records.append({
                "timestamp": float(i) * 5,
                "input_length": 50,
                "output_length": 1,
                "block_ids": list(range(i * 1000, i * 1000 + 50)),
            })
        dataset_path = self._make_dataset(records)

        runner = self._make_runner(num_p=2, dataset_path=dataset_path, num_prompts=6)
        metrics = runner.run_benchmark_emulation()
        assert metrics["completed"] == 6

        hier = runner.get_hierarchical_metrics()
        assert hier.get("total_peer_hit_blocks", 0) == 0, \
            "Unique data should have no peer hits"
        assert hier.get("total_engine_hit_blocks", 0) == 0, \
            "Unique data should have no engine hits (each request only appears once)"

        # All requests should have zero reused tokens
        all_completed = []
        for sched in runner.p_schedulers:
            all_completed.extend(sched.completed_requests)
        for req in all_completed:
            assert req.final_reused_tokens == 0, \
                f"Cold cache request {req.id} should have 0 reused tokens"
        print("[cold_cache] No P2P overhead with unique data")


# ===========================================================================
# Direct scheduler unit test (bypass runner)
# ===========================================================================

class TestSchedulerP2PDirect:
    """Direct unit test: _run_request_level sums engine_hit + peer_hit as cached.
    Optimizer internally handles P2P via FillEngineFromHitIndices, no extra latency needed."""

    def test_peer_hit_directly_reduces_uncached(self):
        """Verify scheduler sums device_hit + host_hit without extra P2P latency."""
        from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter
        from schedule_simulator.schedule_emulator.sglang_scheduler import SGLangScheduleEmulator
        import asyncio

        adapters, gv, mgr = _make_adapter(num_engines=2, page_size=1, peer_bw_gb=10.0, kv_bytes=1000)
        adapter0, adapter1 = adapters

        # Pre-populate: write blocks on engine0
        block_ids = list(range(7000, 7100))  # 100 blocks
        req_w = FakeRequest(id=100, input_token_length=100, output_token_length=1,
                            origin_input_ids=block_ids, output_ids=[5555])
        adapter0.add_to_prefetch_queue(req_w)
        adapter0.on_request_complete(req_w, 1.0)

        # Now read on engine1 via scheduler
        gv.clock = 5.0
        req_q = asyncio.Queue()
        resp_q = asyncio.Queue()

        scheduler = SGLangScheduleEmulator(
            scheduler_config=SchedulerConfig(
                "Qwen2.5-3B", request_level_scheduling=True,
                max_num_tokens=999999999, enable_stats=True,
            ),
            platform_config=PlatformConfig(
                device="H20", peer_read_bandwidth_gb=10.0,
                memory_read_bandwidth_gb=16.0,
            ),
            request_queue=req_q,
            response_queue=resp_q,
        )
        # Replace tree_cache with our adapter
        scheduler.tree_cache = adapter1
        scheduler.global_values = gv

        # Inject request into waiting queue
        req_r = FakeRequest(id=101, input_token_length=100, output_token_length=1,
                            origin_input_ids=block_ids, output_ids=[5554])
        req_r.last_event_time = 4.0
        req_r.queue_time_start = 5.0
        scheduler.waiting_queue.append(req_r)

        # Call add_to_prefetch_queue (GetCacheLocation internally fills P2P blocks)
        adapter1.add_to_prefetch_queue(req_r)
        peer_hit_blocks = req_r.host_cache_hit_length
        assert peer_hit_blocks > 0, \
            f"Should have peer_hit, got {peer_hit_blocks}"

        # Run one iteration of _run_request_level
        scheduler._run_request_level()

        # Peer-hit should be counted as reused (no extra transfer step needed)
        assert req_r.final_reused_tokens == peer_hit_blocks * 1, \
            f"Reused tokens should equal peer_hit * page_size, got {req_r.final_reused_tokens}"

        # Clock should advance by ONLY inference latency (no P2P transfer time)
        total_time = gv.clock - 5.0
        assert total_time > 0, "Clock should advance"
        # With 100 tokens cached, uncached is near 0, so latency should be minimal
        print(f"[direct_p2p] reused={req_r.final_reused_tokens}, "
              f"peer_hit_blocks={peer_hit_blocks}, total_time={total_time:.6f}s")

    def test_p2p_bandwidth_has_no_effect_on_request_level(self):
        """P2P bandwidth should NOT affect request-level timing (Optimizer handles P2P)."""
        from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter
        from schedule_simulator.schedule_emulator.sglang_scheduler import SGLangScheduleEmulator
        import asyncio

        def _run_with_bw(peer_bw_gb):
            adapters, gv, mgr = _make_adapter(num_engines=2, page_size=1,
                                              peer_bw_gb=peer_bw_gb, kv_bytes=1000)
            adapter0, adapter1 = adapters
            block_ids = list(range(8000, 8100))
            req_w = FakeRequest(id=200, input_token_length=100, output_token_length=1,
                                origin_input_ids=block_ids, output_ids=[4444])
            adapter0.add_to_prefetch_queue(req_w)
            adapter0.on_request_complete(req_w, 1.0)

            gv.clock = 5.0
            req_q = asyncio.Queue()
            resp_q = asyncio.Queue()
            scheduler = SGLangScheduleEmulator(
                scheduler_config=SchedulerConfig(
                    "Qwen2.5-3B", request_level_scheduling=True,
                    max_num_tokens=999999999,
                ),
                platform_config=PlatformConfig(
                    device="H20", peer_read_bandwidth_gb=peer_bw_gb,
                    memory_read_bandwidth_gb=16.0,
                ),
                request_queue=req_q, response_queue=resp_q,
            )
            scheduler.tree_cache = adapter1
            scheduler.global_values = gv

            req_r = FakeRequest(id=201, input_token_length=100, output_token_length=1,
                                origin_input_ids=block_ids, output_ids=[4443])
            req_r.last_event_time = 4.0
            req_r.queue_time_start = 5.0
            scheduler.waiting_queue.append(req_r)
            adapter1.add_to_prefetch_queue(req_r)
            scheduler._run_request_level()
            return gv.clock - 5.0

        # Different bandwidths should produce identical latency
        time_fast = _run_with_bw(100.0)
        time_slow = _run_with_bw(0.001)
        assert abs(time_fast - time_slow) < 1e-12, \
            f"P2P bandwidth should not affect timing: fast={time_fast}, slow={time_slow}"
        print(f"[bw_no_effect] fast_bw_time={time_fast:.9f}, slow_bw_time={time_slow:.9f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
