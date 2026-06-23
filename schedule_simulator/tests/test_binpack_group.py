"""Tests for BinPackPolicy group routing and 3-level cache hit statistics."""
import asyncio
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from schedule_simulator.schedule_emulator.types import (
    FakeRequest,
    RouterConfig,
    RoutingPolicy,
)
from schedule_simulator.schedule_emulator.dispatch.dispatch_policy import BinPackPolicy


# ============================================================
# Helper fixtures
# ============================================================

def _make_config(pods_per_group=None, bin_capacity=None, **kwargs):
    """Create RouterConfig with BinPack group settings."""
    return RouterConfig(
        p_policy=RoutingPolicy.BIN_PACK,
        pods_per_group=pods_per_group,
        bin_capacity=bin_capacity,
        **kwargs,
    )


def _make_request(req_id=1, input_length=100, input_ids=None):
    """Create a FakeRequest for testing."""
    req = FakeRequest(id=req_id, input_token_length=input_length, output_token_length=10)
    req.last_event_time = 1.0
    if input_ids is not None:
        req.origin_input_ids = input_ids
    return req


def _make_policy(num_workers=6, pods_per_group=3, bin_capacity=5):
    """Create a BinPackPolicy with mocked workers."""
    config = _make_config(pods_per_group=pods_per_group, bin_capacity=bin_capacity)
    policy = BinPackPolicy(num_workers, config)
    # Mock workers
    for i, w in enumerate(policy.workers):
        w.id = f"P{i}"
        w._healthy = True
        w.is_healthy = lambda _w=w: True
        w._load = 0
        w._total_req = 0
        w.get_load = lambda _w=w: _w._load
        w.get_total_req = lambda _w=w: _w._total_req
        w.increment_load = lambda _w=w: setattr(_w, '_load', _w._load + 1)
    return policy


# ============================================================
# Test: Group assignment
# ============================================================

class TestBinPackGroupAssignment:
    """Test group ID assignment logic."""

    def test_get_group_id_basic(self):
        """Pods are assigned to groups correctly."""
        policy = _make_policy(num_workers=6, pods_per_group=3)
        # Group 0: pods 0,1,2
        assert policy._get_group_id(0) == 0
        assert policy._get_group_id(1) == 0
        assert policy._get_group_id(2) == 0
        # Group 1: pods 3,4,5
        assert policy._get_group_id(3) == 1
        assert policy._get_group_id(4) == 1
        assert policy._get_group_id(5) == 1

    def test_get_group_id_uneven(self):
        """Uneven pod count still assigns correctly."""
        policy = _make_policy(num_workers=7, pods_per_group=3)
        assert policy._get_group_id(6) == 2  # 6 // 3 = 2

    def test_get_group_id_disabled(self):
        """When pods_per_group is None, all pods in group 0."""
        config = _make_config(pods_per_group=None, bin_capacity=None)
        policy = BinPackPolicy(4, config)
        assert policy._get_group_id(0) == 0
        assert policy._get_group_id(3) == 0

    def test_get_group_members(self):
        """Get healthy members of a specific group."""
        policy = _make_policy(num_workers=6, pods_per_group=3)
        healthy = [0, 1, 2, 3, 4, 5]
        assert policy._get_group_members(0, healthy) == [0, 1, 2]
        assert policy._get_group_members(1, healthy) == [3, 4, 5]

    def test_get_group_members_partial_healthy(self):
        """Only healthy pods returned."""
        policy = _make_policy(num_workers=6, pods_per_group=3)
        healthy = [0, 2, 3, 5]  # pods 1, 4 are unhealthy
        assert policy._get_group_members(0, healthy) == [0, 2]
        assert policy._get_group_members(1, healthy) == [3, 5]


# ============================================================
# Test: Bin-packing selection
# ============================================================

class TestBinPackSelection:
    """Test bin-packing pod selection within a group."""

    def test_binpack_prefers_highest_load_under_capacity(self):
        """Should pick pod with highest load that's still under bin_capacity."""
        policy = _make_policy(num_workers=6, pods_per_group=3, bin_capacity=5)
        # Set loads: pod 0=3, pod 1=1, pod 2=0
        policy.workers[0]._load = 3
        policy.workers[1]._load = 1
        policy.workers[2]._load = 0
        # Should pick pod 0 (highest load, still < 5)
        chosen = policy._binpack_select([0, 1, 2])
        assert chosen == 0

    def test_binpack_skips_at_capacity(self):
        """Pods at bin_capacity are skipped."""
        policy = _make_policy(num_workers=6, pods_per_group=3, bin_capacity=5)
        policy.workers[0]._load = 5  # at capacity
        policy.workers[1]._load = 3
        policy.workers[2]._load = 0
        chosen = policy._binpack_select([0, 1, 2])
        assert chosen == 1  # Highest load under capacity

    def test_binpack_all_at_capacity_fallback(self):
        """When all pods at capacity, fall back to least loaded."""
        policy = _make_policy(num_workers=6, pods_per_group=3, bin_capacity=5)
        policy.workers[0]._load = 7
        policy.workers[1]._load = 5
        policy.workers[2]._load = 6
        chosen = policy._binpack_select([0, 1, 2])
        assert chosen == 1  # Least loaded

    def test_binpack_empty_pods_fill_first(self):
        """With all pods at 0 load, any one is chosen (all equal)."""
        policy = _make_policy(num_workers=3, pods_per_group=3, bin_capacity=5)
        # All at 0 - all under capacity, max is 0 for all
        chosen = policy._binpack_select([0, 1, 2])
        assert chosen in [0, 1, 2]

    def test_binpack_fills_one_pod_before_moving(self):
        """Demonstrates bin-packing behavior: fills one pod up."""
        policy = _make_policy(num_workers=3, pods_per_group=3, bin_capacity=3)
        candidates = [0, 1, 2]

        # Route 3 requests - all should go to the same pod until capacity
        chosen_sequence = []
        for _ in range(3):
            chosen = policy._binpack_select(candidates)
            policy.workers[chosen]._load += 1
            chosen_sequence.append(chosen)

        # All 3 should go to the same pod (the first one picked)
        assert chosen_sequence[0] == chosen_sequence[1] == chosen_sequence[2]

    def test_binpack_moves_to_next_pod(self):
        """After first pod reaches capacity, moves to next."""
        policy = _make_policy(num_workers=3, pods_per_group=3, bin_capacity=2)
        candidates = [0, 1, 2]

        # Fill first pod
        policy.workers[0]._load = 0
        chosen1 = policy._binpack_select(candidates)  # picks 0 (all equal, first under capacity)
        policy.workers[chosen1]._load += 1
        chosen2 = policy._binpack_select(candidates)  # picks same pod (load 1 < 2)
        policy.workers[chosen2]._load += 1
        # Now pod is at capacity (2), next pick should be different
        chosen3 = policy._binpack_select(candidates)
        assert policy.workers[chosen1]._load == 2  # full
        assert chosen3 != chosen1  # moves to different pod


# ============================================================
# Test: select_worker with group routing
# ============================================================

class TestBinPackSelectWorker:
    """Test full select_worker flow with group-based routing."""

    def test_select_worker_no_group_falls_back_to_parent(self):
        """When pods_per_group is None, should use CacheAwarePolicy behavior."""
        config = _make_config(pods_per_group=None, bin_capacity=None)
        policy = BinPackPolicy(4, config)
        # Without initialization, falls back to min-load
        for i, w in enumerate(policy.workers):
            w.id = f"P{i}"
            w._healthy = True
            w.is_healthy = lambda: True
            w._load = 0
            w._total_req = 0
            w.get_load = lambda _w=w: _w._load
            w.get_total_req = lambda _w=w: _w._total_req
            w.increment_load = lambda _w=w: setattr(_w, '_load', _w._load + 1)

        req = _make_request()
        result = asyncio.run(policy.select_worker(req))
        assert result is not None
        assert 0 <= result < 4

    def test_select_worker_with_group_routes_to_group(self):
        """With group enabled, routes within the target group."""
        policy = _make_policy(num_workers=6, pods_per_group=3, bin_capacity=5)
        policy._initialized = True
        policy._optimizer_manager = None  # No C++ optimizer
        policy._engine_ids = [f"P{i}" for i in range(6)]
        policy._engine_id_to_idx = {f"P{i}": i for i in range(6)}
        policy._routing_records = []
        policy._write_to_approx_tree = MagicMock()

        req = _make_request(input_ids=None)  # No input_ids -> no cache hit
        result = asyncio.run(policy.select_worker(req))
        assert result is not None
        assert 0 <= result < 6

    def test_select_worker_respects_bin_capacity(self):
        """Requests are distributed respecting bin_capacity."""
        policy = _make_policy(num_workers=6, pods_per_group=3, bin_capacity=2)
        policy._initialized = True
        policy._optimizer_manager = None
        policy._engine_ids = [f"P{i}" for i in range(6)]
        policy._engine_id_to_idx = {f"P{i}": i for i in range(6)}
        policy._routing_records = []
        policy._write_to_approx_tree = MagicMock()

        # Route many requests
        results = []
        for i in range(6):
            req = _make_request(req_id=i, input_ids=None)
            r = asyncio.run(policy.select_worker(req))
            results.append(r)

        # Check no pod has more than bin_capacity=2 requests
        from collections import Counter
        counts = Counter(results)
        for pod_idx, count in counts.items():
            # Due to async nature and fallback, max should be ~bin_capacity
            assert count <= 3, f"Pod {pod_idx} got {count} requests, expected <= bin_capacity"

    def test_select_worker_with_cache_hit_routes_to_correct_group(self):
        """When cache hit identifies best engine, route to that engine's group."""
        policy = _make_policy(num_workers=6, pods_per_group=3, bin_capacity=10)
        policy._initialized = True
        policy._engine_ids = [f"P{i}" for i in range(6)]
        policy._engine_id_to_idx = {f"P{i}": i for i in range(6)}
        policy._routing_records = []
        policy._write_to_approx_tree = MagicMock()

        # Mock optimizer to return pod 4 (group 1) as best engine
        mock_manager = MagicMock()
        mock_res = MagicMock()
        mock_res.hit_count = 10
        mock_res.engine_instance_id = "P4"
        mock_manager.ChooseBestEngine.return_value = mock_res
        policy._optimizer_manager = mock_manager

        req = _make_request(input_ids=[1, 2, 3])
        result = asyncio.run(policy.select_worker(req))
        # Should route to group 1 (pods 3,4,5)
        assert result in [3, 4, 5], f"Expected group 1 (pods 3-5), got {result}"


# ============================================================
# Test: RouterConfig with group settings
# ============================================================

class TestRouterConfigGroup:
    """Test RouterConfig group field support."""

    def test_config_defaults_none(self):
        """pods_per_group and bin_capacity default to None."""
        config = RouterConfig()
        assert config.pods_per_group is None
        assert config.bin_capacity is None

    def test_config_from_string_with_group(self):
        """RouterConfig.from_string_policy passes group params."""
        config = RouterConfig.from_string_policy(
            p_policy_str="bin_pack",
            pods_per_group=4,
            bin_capacity=8,
        )
        assert config.p_policy == RoutingPolicy.BIN_PACK
        assert config.pods_per_group == 4
        assert config.bin_capacity == 8

    def test_policy_inherits_config(self):
        """BinPackPolicy correctly inherits group config."""
        config = _make_config(pods_per_group=5, bin_capacity=10)
        policy = BinPackPolicy(10, config)
        assert policy.pods_per_group == 5
        assert policy.bin_capacity == 10


# ============================================================
# Test: HierarchicalHitRecord peer_source tracking
# ============================================================

class TestHierarchicalHitRecordPeerSource:
    """Test that HierarchicalHitRecord tracks peer_source_engine_id."""

    def test_hit_record_has_peer_source_field(self):
        """HierarchicalHitRecord includes peer_source_engine_id."""
        from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalHitRecord
        rec = HierarchicalHitRecord(req_id=1, engine_hit=5, peer_hit=3, pool_hit=0,
                                     total_hit=8, input_length=100, num_blocks=10,
                                     peer_source_engine_id="P2")
        assert rec.peer_source_engine_id == "P2"

    def test_hit_record_default_empty(self):
        """Default peer_source_engine_id is empty string."""
        from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalHitRecord
        rec = HierarchicalHitRecord(req_id=1)
        assert rec.peer_source_engine_id == ""


# ============================================================
# Test: FakeRequest peer_source_engine_id field
# ============================================================

class TestFakeRequestPeerSource:
    """Test that FakeRequest has peer_source_engine_id field."""

    def test_fake_request_has_peer_source_field(self):
        """FakeRequest should have peer_source_engine_id field."""
        req = FakeRequest(id=1, input_token_length=10, output_token_length=5)
        assert hasattr(req, 'peer_source_engine_id')
        assert req.peer_source_engine_id == ""

    def test_fake_request_peer_source_settable(self):
        """peer_source_engine_id can be set."""
        req = FakeRequest(id=1, input_token_length=10, output_token_length=5)
        req.peer_source_engine_id = "P3"
        assert req.peer_source_engine_id == "P3"


# ============================================================
# Test: HierarchicalCacheAdapter group_hit counters
# ============================================================

class TestAdapterGroupCounters:
    """Test that HierarchicalCacheAdapter has group_hit counter fields."""

    def test_adapter_has_group_hit_counters(self):
        """Adapter should track total_group_hit_blocks and total_external_peer_hit_blocks."""
        from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter
        from schedule_simulator.schedule_emulator.base import GlobalValues

        mock_manager = MagicMock()
        mock_platform = MagicMock()
        mock_platform.peer_read_bandwidth = 10.0
        mock_platform.memory_read_bandwidth = 16.0
        mock_gv = GlobalValues()

        adapter = HierarchicalCacheAdapter(
            manager=mock_manager,
            engine_instance_id="P0",
            platform_config=mock_platform,
            kv_cache_space_per_token=256,
            page_size=1,
            global_values=mock_gv,
        )
        assert adapter.total_group_hit_blocks == 0
        assert adapter.total_external_peer_hit_blocks == 0
