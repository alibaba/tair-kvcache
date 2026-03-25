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

    # ── Case 9: inconsistent offset (offset < len_prefix) → all False ─
    def test_inconsistent_offset_returns_all_false(self):
        """When offset < len_prefix, manager state is inconsistent → all False."""
        keys = ["k3", "k4", "k5"]

        # prefix_keys = ["p0", "p1", "p2"], len_prefix=3
        # offset=1 < len_prefix=3 → inconsistent
        result_from_manager = {
            "locations": [],
            "write_session_id": "ws-9",
            "block_mask": {"offset": 1},
        }
        c = self._setup_connector(start_write_result=result_from_manager)

        extra_info = MagicMock()
        extra_info.prefix_keys = ["p0", "p1", "p2"]

        result = c._batch_set(keys, torch.zeros(3), trace_id="t9",
                              extra_info=extra_info)

        self.assertEqual(len(result), len(keys))
        self.assertEqual(result, [False, False, False])

    # ── Case 10: inconsistent bool_masks (prefix not cached) → all False
    def test_inconsistent_bool_masks_returns_all_false(self):
        """When prefix blocks not fully cached in bool_masks → all False."""
        keys = ["k2", "k3", "k4"]

        # prefix_keys = ["p0", "p1"], len_prefix=2
        # bool_masks[:2] = [True, False] → not all True → inconsistent
        result_from_manager = {
            "locations": [],
            "write_session_id": "ws-10",
            "block_mask": {
                "bool_masks": {
                    "values": [True, False, False, True, False]
                }
            },
        }
        c = self._setup_connector(start_write_result=result_from_manager)

        extra_info = MagicMock()
        extra_info.prefix_keys = ["p0", "p1"]

        result = c._batch_set(keys, torch.zeros(3), trace_id="t10",
                              extra_info=extra_info)

        self.assertEqual(len(result), len(keys))
        self.assertEqual(result, [False, False, False])

    # ── Case 11: incomplete bool_masks (shorter than expected) → all False
    def test_incomplete_bool_masks_returns_all_false(self):
        """When bool_masks is shorter than len_prefix + len_new → all False."""
        keys = ["k0", "k1", "k2"]

        # 3 keys, no prefix → expected bool_masks length = 3, but only 1 given
        result_from_manager = {
            "locations": [],
            "write_session_id": "ws-11",
            "block_mask": {
                "bool_masks": {
                    "values": [True]
                }
            },
        }
        c = self._setup_connector(start_write_result=result_from_manager)

        result = c._batch_set(keys, torch.zeros(3), trace_id="t11")

        self.assertEqual(len(result), len(keys))
        self.assertEqual(result, [False, False, False])

    # ── Case 12: empty bool_masks → all False ────────────────────────
    def test_empty_bool_masks_returns_all_false(self):
        """When bool_masks is empty → all False (all([]) would be True)."""
        keys = ["k0", "k1"]

        result_from_manager = {
            "locations": [],
            "write_session_id": "ws-12",
            "block_mask": {
                "bool_masks": {
                    "values": []
                }
            },
        }
        c = self._setup_connector(start_write_result=result_from_manager)

        result = c._batch_set(keys, torch.zeros(2), trace_id="t12")

        self.assertEqual(len(result), len(keys))
        self.assertEqual(result, [False, False])

    # ── Case 13: batch_set_v1 exception returns all False ─────────────
    def test_batch_set_v1_exception(self):
        """batch_set_v1 catches exceptions and returns all False."""
        c = _build_connector()
        c._manager_client.start_write_cache.side_effect = RuntimeError("boom")

        result = c.batch_set_v1(["k0", "k1", "k2"], torch.zeros(3))

        self.assertEqual(result, [False, False, False])


if __name__ == "__main__":
    unittest.main()
