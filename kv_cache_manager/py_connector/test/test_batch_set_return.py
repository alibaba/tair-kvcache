"""Unit tests for _batch_set return value: 1:1 positional mapping with input keys."""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

# ── Mock unavailable modules before importing connector ──────────────
_mock_metrics = types.ModuleType("sglang.srt.metrics")
_mock_collector = types.ModuleType("sglang.srt.metrics.collector")
_mock_collector.StorageMetrics = MagicMock
_mock_metrics.collector = _mock_collector
sys.modules.setdefault("sglang.srt.metrics", _mock_metrics)
sys.modules.setdefault("sglang.srt.metrics.collector", _mock_collector)

_mock_pybind = types.ModuleType("kv_cache_manager.client.pybind")
_mock_kvcm = MagicMock()
_mock_kvcm.ClientErrorCode.ER_OK = 0
_mock_pybind.kvcm_py_client = _mock_kvcm
sys.modules["kv_cache_manager.client.pybind"] = _mock_pybind

_mock_version = types.ModuleType("kv_cache_manager.py_connector.common._version_info")
_mock_version.FULL_VERSION = "0.0.0-test"
_mock_version.GIT_COMMIT = "test"
_mock_version.BUILD_TIME = "test"
sys.modules.setdefault("kv_cache_manager.py_connector.common._version_info", _mock_version)

from kv_cache_manager.py_connector.sglang.connector import HiCacheKVCM  # noqa: E402


def _make_obj(cls):
    """Create an instance of cls without calling __init__."""
    return cls.__new__(cls)


def _build_connector(*, tp_rank=0, tp_world_size=1, kv_factor=2,
                     instance_id="test", location_spec_name="tp_0",
                     location_spec_size=4096, write_timeout_seconds=30):
    """Build a HiCacheKVCM with attributes set manually (bypass __init__)."""
    obj = _make_obj(HiCacheKVCM)
    obj.tp_rank = tp_rank
    obj.tp_world_size = tp_world_size
    obj.kv_factor = kv_factor
    obj.instance_id = instance_id
    obj.location_spec_name = location_spec_name
    obj.location_spec_size = location_spec_size
    obj.write_timeout_seconds = write_timeout_seconds
    obj.backup_pgs = []
    obj.backup_bandwidth = []
    obj._manager_client = MagicMock()
    obj.transfer_client = MagicMock()
    obj.mem_pool_host = MagicMock()
    obj.has_mamba = False
    obj.has_indexer = False
    obj.registered_pools = {}
    return obj


class TestBatchSetReturnValue(unittest.TestCase):
    """Verify _batch_set returns a list aligned 1:1 with input keys."""

    def _setup_connector(self, *, start_write_result, save_ok=True):
        """Helper: wire up mocks and return the connector."""
        c = _build_connector()

        c._manager_client.start_write_cache.return_value = start_write_result

        # mem_pool_host stubs – return enough ptrs/sizes
        n_blocks = 20
        c.mem_pool_host.get_page_buffer_meta.return_value = (
            list(range(n_blocks * c.kv_factor)),
            [c.location_spec_size] * (n_blocks * c.kv_factor),
        )

        # transfer_client.SaveKvCaches returns (error_code, ...)
        err_code = _mock_kvcm.ClientErrorCode.ER_OK if save_ok else 999
        c.transfer_client.SaveKvCaches.return_value = (err_code,)

        return c

    # ── Case 1: all keys already cached (unmatched == 0) ─────────────
    def test_all_cached_returns_all_true(self):
        """When startwrite says nothing needs writing, every position is True."""
        keys = ["k0", "k1", "k2", "k3", "k4"]

        # block_mask offset == len(prefix) + len(keys) → nothing to save
        result_from_manager = {
            "locations": [],
            "write_session_id": "ws-1",
            "block_mask": {"offset": 5},  # offset == len_prefix(0) + len_new(5)
        }
        c = self._setup_connector(start_write_result=result_from_manager)

        result = c._batch_set(keys, torch.zeros(5), trace_id="t1")

        self.assertEqual(len(result), len(keys))
        self.assertEqual(result, [True, True, True, True, True])

    # ── Case 2: partial write needed, write succeeds ─────────────────
    def test_partial_write_success(self):
        """Keys 0,1 already cached; keys 2,3,4 need writing and succeed."""
        keys = ["k0", "k1", "k2", "k3", "k4"]

        # offset=2 means indices [2,3,4] need saving (relative: [2,3,4])
        locations = [
            {"location_specs": [{"name": "tp_0", "uri": f"uri_{i}"}]}
            for i in range(3)
        ]
        result_from_manager = {
            "locations": locations,
            "write_session_id": "ws-2",
            "block_mask": {"offset": 2},  # first 2 (prefix=0, so key 0,1) cached
        }
        c = self._setup_connector(start_write_result=result_from_manager,
                                  save_ok=True)

        result = c._batch_set(keys, torch.zeros(5), trace_id="t2")

        self.assertEqual(len(result), len(keys))
        # key 0,1 → True (no write needed), key 2,3,4 → True (write succeeded)
        self.assertEqual(result, [True, True, True, True, True])

    # ── Case 3: partial write needed, write fails ────────────────────
    def test_partial_write_failure(self):
        """Keys 0,1 already cached; keys 2,3,4 need writing but fail."""
        keys = ["k0", "k1", "k2", "k3", "k4"]

        locations = [
            {"location_specs": [{"name": "tp_0", "uri": f"uri_{i}"}]}
            for i in range(3)
        ]
        result_from_manager = {
            "locations": locations,
            "write_session_id": "ws-3",
            "block_mask": {"offset": 2},
        }
        c = self._setup_connector(start_write_result=result_from_manager,
                                  save_ok=False)

        result = c._batch_set(keys, torch.zeros(5), trace_id="t3")

        self.assertEqual(len(result), len(keys))
        # key 0,1 → True (no write needed), key 2,3,4 → False (write failed)
        self.assertEqual(result, [True, True, False, False, False])

    # ── Case 4: all keys need writing, write succeeds ────────────────
    def test_all_need_write_success(self):
        """All keys need writing and succeed."""
        keys = ["k0", "k1", "k2"]

        locations = [
            {"location_specs": [{"name": "tp_0", "uri": f"uri_{i}"}]}
            for i in range(3)
        ]
        result_from_manager = {
            "locations": locations,
            "write_session_id": "ws-4",
            "block_mask": {"offset": 0},
        }
        c = self._setup_connector(start_write_result=result_from_manager,
                                  save_ok=True)

        result = c._batch_set(keys, torch.zeros(3), trace_id="t4")

        self.assertEqual(len(result), len(keys))
        self.assertEqual(result, [True, True, True])

    # ── Case 5: all keys need writing, write fails ───────────────────
    def test_all_need_write_failure(self):
        """All keys need writing but fail."""
        keys = ["k0", "k1", "k2"]

        locations = [
            {"location_specs": [{"name": "tp_0", "uri": f"uri_{i}"}]}
            for i in range(3)
        ]
        result_from_manager = {
            "locations": locations,
            "write_session_id": "ws-5",
            "block_mask": {"offset": 0},
        }
        c = self._setup_connector(start_write_result=result_from_manager,
                                  save_ok=False)

        result = c._batch_set(keys, torch.zeros(3), trace_id="t5")

        self.assertEqual(len(result), len(keys))
        self.assertEqual(result, [False, False, False])

    # ── Case 6: with prefix_keys, partial cached ─────────────────────
    def test_with_prefix_keys(self):
        """prefix_keys present; some new keys cached, some need writing."""
        keys = ["k3", "k4", "k5", "k6"]

        # prefix_keys = ["p0", "p1", "p2"], len_prefix=3
        # block_mask offset=5 means save from index 5 onward
        # relative save_indices = [5-3, 6-3] = [2, 3]
        locations = [
            {"location_specs": [{"name": "tp_0", "uri": f"uri_{i}"}]}
            for i in range(2)
        ]
        result_from_manager = {
            "locations": locations,
            "write_session_id": "ws-6",
            "block_mask": {"offset": 5},
        }
        c = self._setup_connector(start_write_result=result_from_manager,
                                  save_ok=True)

        extra_info = MagicMock()
        extra_info.prefix_keys = ["p0", "p1", "p2"]

        result = c._batch_set(keys, torch.zeros(4), trace_id="t6",
                              extra_info=extra_info)

        self.assertEqual(len(result), len(keys))
        # key 0,1 (k3,k4) → True (no write needed)
        # key 2,3 (k5,k6) → True (write succeeded)
        self.assertEqual(result, [True, True, True, True])

    # ── Case 7: bool_masks with non-contiguous save_indices ──────────
    def test_bool_masks_non_contiguous(self):
        """bool_masks with gaps: only specific positions need writing."""
        keys = ["k0", "k1", "k2", "k3", "k4"]

        # bool_masks: True=cached, False=need write
        # indices [0,1,2,3,4] → masks [False, True, False, True, False]
        # save_indices (need write): [0, 2, 4]
        locations = [
            {"location_specs": [{"name": "tp_0", "uri": f"uri_{i}"}]}
            for i in range(3)
        ]
        result_from_manager = {
            "locations": locations,
            "write_session_id": "ws-7",
            "block_mask": {
                "bool_masks": {
                    "values": [False, True, False, True, False]
                }
            },
        }
        c = self._setup_connector(start_write_result=result_from_manager,
                                  save_ok=True)

        result = c._batch_set(keys, torch.zeros(5), trace_id="t7")

        self.assertEqual(len(result), len(keys))
        # index 0 → write succeeded (True)
        # index 1 → cached (True)
        # index 2 → write succeeded (True)
        # index 3 → cached (True)
        # index 4 → write succeeded (True)
        self.assertEqual(result, [True, True, True, True, True])

    # ── Case 8: bool_masks non-contiguous, write fails ───────────────
    def test_bool_masks_non_contiguous_failure(self):
        """bool_masks with gaps, write fails."""
        keys = ["k0", "k1", "k2", "k3", "k4"]

        locations = [
            {"location_specs": [{"name": "tp_0", "uri": f"uri_{i}"}]}
            for i in range(3)
        ]
        result_from_manager = {
            "locations": locations,
            "write_session_id": "ws-8",
            "block_mask": {
                "bool_masks": {
                    "values": [False, True, False, True, False]
                }
            },
        }
        c = self._setup_connector(start_write_result=result_from_manager,
                                  save_ok=False)

        result = c._batch_set(keys, torch.zeros(5), trace_id="t8")

        self.assertEqual(len(result), len(keys))
        # index 0 → write failed (False)
        # index 1 → cached (True)
        # index 2 → write failed (False)
        # index 3 → cached (True)
        # index 4 → write failed (False)
        self.assertEqual(result, [False, True, False, True, False])

    # ── Case 9: offset behind prefix → best-effort write succeeds ───
    def test_offset_behind_prefix_best_effort_success(self):
        """When offset < len_prefix, skip prefix blocks and write new blocks."""
        keys = ["k3", "k4", "k5"]

        # prefix_keys = ["p0", "p1", "p2"], len_prefix=3
        # offset=1 → blocks [1..5] need writing:
        #   p1, p2 (prefix, can't write) + k3, k4, k5 (new, can write)
        # prefix_write_count=2, save_indices=[0,1,2]
        # locations: 5 entries (2 prefix + 3 new)
        locations = [
            {"location_specs": [{"name": "tp_0", "uri": f"uri_{i}"}]}
            for i in range(5)
        ]
        result_from_manager = {
            "locations": locations,
            "write_session_id": "ws-9",
            "block_mask": {"offset": 1},
        }
        c = self._setup_connector(start_write_result=result_from_manager,
                                  save_ok=True)

        extra_info = MagicMock()
        extra_info.prefix_keys = ["p0", "p1", "p2"]

        result = c._batch_set(keys, torch.zeros(3), trace_id="t9",
                              extra_info=extra_info)

        self.assertEqual(len(result), len(keys))
        # All new blocks written successfully
        self.assertEqual(result, [True, True, True])

        # Verify finish_write_cache was called with prefix blocks marked as failed
        call_args = c._manager_client.finish_write_cache.call_args[0][0]
        finish_mask = call_args["success_blocks"]["bool_masks"]["values"]
        # First 2 entries (prefix) = False, last 3 (new) = True
        self.assertEqual(finish_mask, [False, False, True, True, True])

    # ── Case 10: offset behind prefix → best-effort write fails ───
    def test_offset_behind_prefix_best_effort_failure(self):
        """When offset < len_prefix, skip prefix; new blocks fail → all False."""
        keys = ["k3", "k4", "k5"]

        locations = [
            {"location_specs": [{"name": "tp_0", "uri": f"uri_{i}"}]}
            for i in range(5)
        ]
        result_from_manager = {
            "locations": locations,
            "write_session_id": "ws-10",
            "block_mask": {"offset": 1},
        }
        c = self._setup_connector(start_write_result=result_from_manager,
                                  save_ok=False)

        extra_info = MagicMock()
        extra_info.prefix_keys = ["p0", "p1", "p2"]

        result = c._batch_set(keys, torch.zeros(3), trace_id="t10",
                              extra_info=extra_info)

        self.assertEqual(len(result), len(keys))
        # New blocks write failed
        self.assertEqual(result, [False, False, False])

        # Verify finish_write_cache: prefix=False, new=False
        call_args = c._manager_client.finish_write_cache.call_args[0][0]
        finish_mask = call_args["success_blocks"]["bool_masks"]["values"]
        self.assertEqual(finish_mask, [False, False, False, False, False])

    # ── Case 11: bool_masks prefix not cached → best-effort succeeds ─
    def test_bool_masks_prefix_not_cached_best_effort(self):
        """When prefix blocks not fully cached in bool_masks, write new blocks."""
        keys = ["k2", "k3", "k4"]

        # prefix_keys = ["p0", "p1"], len_prefix=2
        # bool_masks: [True, False, False, True, False]
        #   p0:True p1:False(prefix,skip) k2:False(write) k3:True(cached) k4:False(write)
        # prefix_write_count=1, save_indices=[0,2] (k2 and k4)
        # locations: 3 entries (1 prefix + 2 new)
        locations = [
            {"location_specs": [{"name": "tp_0", "uri": f"uri_{i}"}]}
            for i in range(3)
        ]
        result_from_manager = {
            "locations": locations,
            "write_session_id": "ws-11",
            "block_mask": {
                "bool_masks": {
                    "values": [True, False, False, True, False]
                }
            },
        }
        c = self._setup_connector(start_write_result=result_from_manager,
                                  save_ok=True)

        extra_info = MagicMock()
        extra_info.prefix_keys = ["p0", "p1"]

        result = c._batch_set(keys, torch.zeros(3), trace_id="t11",
                              extra_info=extra_info)

        self.assertEqual(len(result), len(keys))
        # k2: write succeeded, k3: cached, k4: write succeeded
        self.assertEqual(result, [True, True, True])

        # Verify finish_write_cache: prefix p1=False, k2=True, k4=True
        call_args = c._manager_client.finish_write_cache.call_args[0][0]
        finish_mask = call_args["success_blocks"]["bool_masks"]["values"]
        self.assertEqual(finish_mask, [False, True, True])

    # ── Case 12: all new cached but prefix needs write → all True ────
    def test_all_new_cached_but_prefix_needs_write(self):
        """All new blocks cached, only prefix needs writing → return all True."""
        keys = ["k2", "k3"]

        # prefix_keys = ["p0", "p1"], len_prefix=2
        # bool_masks: [True, False, True, True]
        #   p0:True p1:False(prefix,skip) k2:True(cached) k3:True(cached)
        # prefix_write_count=1, save_indices=[] (all new cached)
        # locations: 1 entry (for p1 only)
        locations = [
            {"location_specs": [{"name": "tp_0", "uri": "uri_0"}]}
        ]
        result_from_manager = {
            "locations": locations,
            "write_session_id": "ws-12",
            "block_mask": {
                "bool_masks": {
                    "values": [True, False, True, True]
                }
            },
        }
        c = self._setup_connector(start_write_result=result_from_manager)

        extra_info = MagicMock()
        extra_info.prefix_keys = ["p0", "p1"]

        result = c._batch_set(keys, torch.zeros(2), trace_id="t12",
                              extra_info=extra_info)

        self.assertEqual(len(result), len(keys))
        # All new blocks are cached → True
        self.assertEqual(result, [True, True])

        # Verify finish_write_cache marks prefix location as failed
        call_args = c._manager_client.finish_write_cache.call_args[0][0]
        finish_mask = call_args["success_blocks"]["bool_masks"]["values"]
        self.assertEqual(finish_mask, [False])

    # ── Case 13: incomplete bool_masks (shorter than expected) → all False
    def test_incomplete_bool_masks_returns_all_false(self):
        """When bool_masks is shorter than len_prefix + len_new → all False."""
        keys = ["k0", "k1", "k2"]

        # 3 keys, no prefix → expected bool_masks length = 3, but only 1 given
        result_from_manager = {
            "locations": [],
            "write_session_id": "ws-13",
            "block_mask": {
                "bool_masks": {
                    "values": [True]
                }
            },
        }
        c = self._setup_connector(start_write_result=result_from_manager)

        result = c._batch_set(keys, torch.zeros(3), trace_id="t13")

        self.assertEqual(len(result), len(keys))
        self.assertEqual(result, [False, False, False])

    # ── Case 14: empty bool_masks → all False ────────────────────────
    def test_empty_bool_masks_returns_all_false(self):
        """When bool_masks is empty → all False (all([]) would be True)."""
        keys = ["k0", "k1"]

        result_from_manager = {
            "locations": [],
            "write_session_id": "ws-14",
            "block_mask": {
                "bool_masks": {
                    "values": []
                }
            },
        }
        c = self._setup_connector(start_write_result=result_from_manager)

        result = c._batch_set(keys, torch.zeros(2), trace_id="t14")

        self.assertEqual(len(result), len(keys))
        self.assertEqual(result, [False, False])

    # ── Case 15: per-block best-effort with partial local data ─────────
    def test_per_block_best_effort_partial_local_data(self):
        """When local rank has fewer blocks than save_indices expects,
        only blocks with local data are written; others return False."""
        keys = ["k0", "k1", "k2", "k3", "k4"]

        # All 5 keys need writing (offset=0), 5 locations
        locations = [
            {"location_specs": [{"name": "tp_0", "uri": f"uri_{i}"}]}
            for i in range(5)
        ]
        result_from_manager = {
            "locations": locations,
            "write_session_id": "ws-15",
            "block_mask": {"offset": 0},
        }
        c = _build_connector()
        c._manager_client.start_write_cache.return_value = result_from_manager

        # Only 3 blocks worth of buffer data (kv_factor=2, so 6 entries)
        # This simulates a rank that has fewer local blocks available.
        # local_block_count = 6 // 2 = 3, so only save_indices [0,1,2] are valid.
        c.mem_pool_host.get_page_buffer_meta.return_value = (
            list(range(6)),           # 6 ptrs → 3 blocks
            [c.location_spec_size] * 6,
        )

        # SaveKvCaches succeeds for the 3 valid blocks
        c.transfer_client.SaveKvCaches.return_value = (
            _mock_kvcm.ClientErrorCode.ER_OK,
        )

        result = c._batch_set(keys, torch.zeros(5), trace_id="t15")

        self.assertEqual(len(result), len(keys))
        # Blocks 0,1,2 have local data → True; blocks 3,4 do not → False
        self.assertEqual(result, [True, True, True, False, False])

        # Verify SaveKvCaches was called with only 3 URIs (valid blocks)
        save_call_args = c.transfer_client.SaveKvCaches.call_args[0]
        uris_arg = save_call_args[0]
        self.assertEqual(len(uris_arg), 3)

        # Verify finish_write_cache mask reflects per-block success
        call_args = c._manager_client.finish_write_cache.call_args[0][0]
        finish_mask = call_args["success_blocks"]["bool_masks"]["values"]
        self.assertEqual(finish_mask, [True, True, True, False, False])

    # ── Case 16: per-block best-effort, transfer fails for valid blocks ─
    def test_per_block_best_effort_transfer_fails(self):
        """When local rank has partial data but transfer fails, all return False."""
        keys = ["k0", "k1", "k2", "k3"]

        # All 4 keys need writing (offset=0)
        locations = [
            {"location_specs": [{"name": "tp_0", "uri": f"uri_{i}"}]}
            for i in range(4)
        ]
        result_from_manager = {
            "locations": locations,
            "write_session_id": "ws-16",
            "block_mask": {"offset": 0},
        }
        c = _build_connector()
        c._manager_client.start_write_cache.return_value = result_from_manager

        # Only 2 blocks of local data (kv_factor=2, so 4 entries)
        # local_block_count = 4 // 2 = 2, save_indices [0,1] valid, [2,3] invalid
        c.mem_pool_host.get_page_buffer_meta.return_value = (
            list(range(4)),
            [c.location_spec_size] * 4,
        )

        # SaveKvCaches FAILS
        c.transfer_client.SaveKvCaches.return_value = (999,)

        result = c._batch_set(keys, torch.zeros(4), trace_id="t16")

        self.assertEqual(len(result), len(keys))
        # Blocks 0,1 had local data but transfer failed → False
        # Blocks 2,3 had no local data → False
        self.assertEqual(result, [False, False, False, False])

        # Verify finish_write_cache: all False
        call_args = c._manager_client.finish_write_cache.call_args[0][0]
        finish_mask = call_args["success_blocks"]["bool_masks"]["values"]
        self.assertEqual(finish_mask, [False, False, False, False])

    # ── Case 17: per-block best-effort with prefix + partial local data ─
    def test_per_block_best_effort_prefix_plus_partial(self):
        """Prefix blocks skipped AND local rank has partial data for new blocks."""
        keys = ["k2", "k3", "k4", "k5"]

        # prefix_keys = ["p0", "p1"], len_prefix=2
        # offset=1 → blocks [1..5] need writing:
        #   p1 (prefix, can't write) + k2, k3, k4, k5 (new, can write)
        # prefix_write_count=1, save_indices=[0,1,2,3]
        # locations: 5 entries (1 prefix + 4 new)
        locations = [
            {"location_specs": [{"name": "tp_0", "uri": f"uri_{i}"}]}
            for i in range(5)
        ]
        result_from_manager = {
            "locations": locations,
            "write_session_id": "ws-17",
            "block_mask": {"offset": 1},
        }
        c = _build_connector()
        c._manager_client.start_write_cache.return_value = result_from_manager

        # Only 2 blocks of local data (kv_factor=2, so 4 entries)
        # local_block_count = 4 // 2 = 2
        # save_indices = [0,1,2,3], valid: [0<2, 1<2, 2<2, 3<2] = [T, T, F, F]
        c.mem_pool_host.get_page_buffer_meta.return_value = (
            list(range(4)),
            [c.location_spec_size] * 4,
        )

        # SaveKvCaches succeeds for the 2 valid blocks
        c.transfer_client.SaveKvCaches.return_value = (
            _mock_kvcm.ClientErrorCode.ER_OK,
        )

        extra_info = MagicMock()
        extra_info.prefix_keys = ["p0", "p1"]

        result = c._batch_set(keys, torch.zeros(4), trace_id="t17",
                              extra_info=extra_info)

        self.assertEqual(len(result), len(keys))
        # k2(idx0): local data + success → True
        # k3(idx1): local data + success → True
        # k4(idx2): no local data → False
        # k5(idx3): no local data → False
        self.assertEqual(result, [True, True, False, False])

        # Verify finish_write_cache: prefix=False, block0=True, block1=True,
        # block2=False, block3=False
        call_args = c._manager_client.finish_write_cache.call_args[0][0]
        finish_mask = call_args["success_blocks"]["bool_masks"]["values"]
        self.assertEqual(finish_mask, [False, True, True, False, False])

    # ── Case 18: batch_set_v1 exception returns all False ─────────────
    def test_batch_set_v1_exception(self):
        """batch_set_v1 catches exceptions and returns all False."""
        c = _build_connector()
        c._manager_client.start_write_cache.side_effect = RuntimeError("boom")

        result = c.batch_set_v1(["k0", "k1", "k2"], torch.zeros(3))

        self.assertEqual(result, [False, False, False])


class TestSkipTransfer(unittest.TestCase):
    """Verify _batch_set returns all False when input diverges across TP ranks.

    These tests simulate a non-rank-0 worker whose local block_keys differ
    from rank 0's.  The broadcast delivers rank 0's result + hash, and the
    hash mismatch sets skip_transfer = True.
    """

    @staticmethod
    def _make_rank1_connector():
        """Build a rank-1 connector (tp_world_size=2)."""
        c = _build_connector(tp_rank=1, tp_world_size=2)
        c.storage_tp_group = MagicMock()
        return c

    @staticmethod
    def _broadcast_side_effect(result, len_prefix, len_new, input_hash):
        """Return a side_effect that fills the recv list with rank 0's data."""
        def _side_effect(tensor_list, src, group):
            tensor_list[0] = result
            tensor_list[1] = len_prefix
            tensor_list[2] = len_new
            tensor_list[3] = input_hash
        return _side_effect

    @staticmethod
    def _rank0_hash(keys_raw, len_prefix, len_new):
        """Compute the hash rank 0 would broadcast for the given raw keys."""
        c_tmp = _build_connector()
        int_keys = [c_tmp._sha256_to_int64(k) for k in keys_raw]
        return hash((len_prefix, len_new, *int_keys))

    # ── Case 19: skip_transfer + unmatched == 0 → all False ──────────
    def test_skip_transfer_all_cached(self):
        """Input diverges but all blocks cached (unmatched==0) → all False."""
        c = self._make_rank1_connector()

        # Rank 1's local keys (different from rank 0's)
        local_keys = ["local_k0", "local_k1", "local_k2"]

        # Rank 0 had different keys, all cached
        rank0_len_prefix, rank0_len_new = 0, 3
        rank0_hash = self._rank0_hash(
            ["rank0_k0", "rank0_k1", "rank0_k2"],
            rank0_len_prefix, rank0_len_new,
        )
        rank0_result = {
            "locations": [],
            "write_session_id": "ws-19",
            "block_mask": {"offset": 3},  # all cached
        }

        bcast = self._broadcast_side_effect(
            rank0_result, rank0_len_prefix, rank0_len_new, rank0_hash,
        )
        with patch("torch.distributed.broadcast_object_list", side_effect=bcast):
            result = c._batch_set(local_keys, torch.zeros(3), trace_id="t19")

        self.assertEqual(len(result), 3)
        self.assertEqual(result, [False, False, False])

    # ── Case 20: skip_transfer + unmatched > 0 → all False ───────────
    def test_skip_transfer_partial_write(self):
        """Input diverges, some blocks need writing → all False."""
        c = self._make_rank1_connector()

        local_keys = ["local_k0", "local_k1", "local_k2", "local_k3"]

        # Rank 0 had different keys, first 2 cached, last 2 need writing
        rank0_len_prefix, rank0_len_new = 0, 4
        rank0_hash = self._rank0_hash(
            ["rank0_k0", "rank0_k1", "rank0_k2", "rank0_k3"],
            rank0_len_prefix, rank0_len_new,
        )
        locations = [
            {"location_specs": [{"name": "tp_0", "uri": f"uri_{i}"}]}
            for i in range(2)
        ]
        rank0_result = {
            "locations": locations,
            "write_session_id": "ws-20",
            "block_mask": {"offset": 2},  # indices [2,3] need writing
        }

        bcast = self._broadcast_side_effect(
            rank0_result, rank0_len_prefix, rank0_len_new, rank0_hash,
        )

        # all_reduce is a no-op (keeps all-zero flags)
        with patch("torch.distributed.broadcast_object_list", side_effect=bcast), \
             patch("torch.distributed.all_reduce"):
            result = c._batch_set(local_keys, torch.zeros(4), trace_id="t20")

        self.assertEqual(len(result), 4)
        self.assertEqual(result, [False, False, False, False])

    # ── Case 21: skip_transfer + prefix + unmatched > 0 → all False ──
    def test_skip_transfer_with_prefix(self):
        """Input diverges (including prefix), blocks need writing → all False."""
        c = self._make_rank1_connector()

        local_keys = ["local_k3", "local_k4", "local_k5"]
        local_extra = MagicMock()
        local_extra.prefix_keys = ["local_p0", "local_p1", "local_p2"]

        # Rank 0 had different prefix+keys
        rank0_len_prefix, rank0_len_new = 3, 3
        rank0_hash = self._rank0_hash(
            ["rank0_p0", "rank0_p1", "rank0_p2", "rank0_k3", "rank0_k4", "rank0_k5"],
            rank0_len_prefix, rank0_len_new,
        )
        # offset=1 < len_prefix=3 → prefix best-effort path, all 5 blocks in locations
        locations = [
            {"location_specs": [{"name": "tp_0", "uri": f"uri_{i}"}]}
            for i in range(5)
        ]
        rank0_result = {
            "locations": locations,
            "write_session_id": "ws-21",
            "block_mask": {"offset": 1},
        }

        bcast = self._broadcast_side_effect(
            rank0_result, rank0_len_prefix, rank0_len_new, rank0_hash,
        )

        with patch("torch.distributed.broadcast_object_list", side_effect=bcast), \
             patch("torch.distributed.all_reduce"):
            result = c._batch_set(local_keys, torch.zeros(3), trace_id="t21",
                                  extra_info=local_extra)

        self.assertEqual(len(result), 3)
        self.assertEqual(result, [False, False, False])


if __name__ == "__main__":
    unittest.main()
