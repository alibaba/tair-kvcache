"""Deterministic (CPU-only) tests for extra-pool handling in HiCacheKVCM.

The GPU integration scripts (test.py / test_linear.py / test_indexer.py) need a
manager process and a CUDA device, so they cannot statically pin down the
zero-copy layout contract.  These tests do: a fake host pool drives the real
``batch_set_v2`` / ``batch_get_v2`` / ``batch_exists_v2`` code paths with a
mocked manager and transfer client.

Covered:

* the IOVs a pool emits per page are taken from the pool's own
  ``get_page_buffer_meta()`` output.  sglang v0.5.16+ drops the mamba temporal
  IOV for conv-only models (no SSM state), so a fixed ``1 + len(conv_buffer)``
  formula silently shifts every page's data;
* a transfer whose host page count does not match its key count is rejected
  instead of being mapped page-by-page onto the wrong keys;
* ``batch_exists_v2`` reports 0 hit pages for pools the connector does not
  manage, instead of implying the caller can fetch them (see
  ``UNMANAGED_POOL`` for the enum member used; sglang < v0.5.12 has no
  ``PoolName.SWA``).

Prerequisites: the connector needs a sglang runtime that can actually import
its backend (``sgl_kernel`` and friends), and ``kv_cache_manager`` must be
importable -- either from an installed wheel or from the repository root on
``sys.path``.  The compiled kvcm pybind client and the build-generated
``_version_info`` are stubbed below, so no wheel is needed for those.

    PYTHONPATH=$PWD python kv_cache_manager/py_connector/sglang/test_extra_pools.py
"""

import sys
import types
import unittest
from typing import Any
from unittest.mock import MagicMock

import torch


# ── Modules the connector imports but this test does not exercise ──────
# Same technique as kv_cache_manager/py_connector/test/test_batch_set_return.py:
# the compiled pybind client and the build-generated version module must be
# importable before the connector module can be loaded.  Iov/BlockBuffer are
# plain value objects here (the real ones are pybind classes); a MagicMock
# would hand the same instance out of every call and hide layout bugs.
class _FakeIov:
    def __init__(self) -> None:
        self.type: Any = None
        self.base = 0
        self.size = 0
        self.ignore = False


class _FakeBlockBuffer:
    def __init__(self) -> None:
        self.iovs: list = []


_mock_pybind: Any = types.ModuleType("kv_cache_manager.client.pybind")
_mock_kvcm = MagicMock()
_mock_kvcm.ClientErrorCode.ER_OK = 0
_mock_kvcm.Iov = _FakeIov
_mock_kvcm.BlockBuffer = _FakeBlockBuffer
_mock_kvcm.MemoryType.CPU = 0
_mock_pybind.kvcm_py_client = _mock_kvcm
sys.modules["kv_cache_manager.client.pybind"] = _mock_pybind

_mock_version: Any = types.ModuleType(
    "kv_cache_manager.py_connector.common._version_info"
)
_mock_version.FULL_VERSION = "0.0.0-test"
_mock_version.GIT_COMMIT = "test"
_mock_version.BUILD_TIME = "test"
sys.modules.setdefault(
    "kv_cache_manager.py_connector.common._version_info", _mock_version
)

from sglang.srt.mem_cache.hicache_storage import (  # noqa: E402
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)

from kv_cache_manager.py_connector.sglang.connector import HiCacheKVCM  # noqa: E402

# Synthetic buffer addresses: the connector only forwards pointers, so the
# tests can assert exact (base, size) tuples.
PAGE_STRIDE = 0x1000
TEMPORAL_BYTES = 0x100
CONV_BYTES = 0x80
INDEXER_PAGE_BYTES = 0x200

# A pool the connector does not manage: PoolName.SWA is the realistic case
# (sglang >= v0.5.12), MAMBA is the stand-in on older versions where the enum
# member does not exist yet.  It has to stay an enum member -- enum
# hashes differ from plain string hashes, so a bare "swa" would not find the
# connector's per-pool result entries.
UNMANAGED_POOL = getattr(PoolName, "SWA", PoolName.MAMBA)


def expected_mamba_page_iovs(
    page: int, *, temporal_state: bool, conv_buffers: int
) -> list[tuple[int, int]]:
    """IOVs the fake mamba pool hands out for one page, in page order.

    Mirrors sglang's MambaPoolHost.get_page_buffer_meta(): the temporal
    component only exists while the model carries an SSM state.
    """
    base = 0x1000_0000 + page * PAGE_STRIDE
    iovs = [(base, TEMPORAL_BYTES)] if temporal_state else []
    for conv in range(conv_buffers):
        iovs.append((base + (conv + 1) * CONV_BYTES, CONV_BYTES))
    return iovs


class FakeMambaPoolHost:
    """Stand-in for sglang's MambaPoolHost (page_size == 1)."""

    def __init__(self, *, temporal_state: bool, conv_buffers: int = 2) -> None:
        self.page_size = 1
        self.dtype = torch.bfloat16
        self.temporal_state = temporal_state
        # The real MambaPoolHost exposes these two internals; the connector is
        # not allowed to read them, it must follow get_page_buffer_meta().
        self.conv_buffer = [torch.zeros(1) for _ in range(conv_buffers)]
        self.temporal_state_elem_size = 1 if temporal_state else 0

    def get_size_per_token(self) -> int:
        return TEMPORAL_BYTES + len(self.conv_buffer) * CONV_BYTES

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        size_list = []
        for page in indices.tolist():
            base = 0x1000_0000 + page * PAGE_STRIDE
            if self.temporal_state:
                ptr_list.append(base)
                size_list.append(TEMPORAL_BYTES)
            for conv in range(len(self.conv_buffer)):
                ptr_list.append(base + (conv + 1) * CONV_BYTES)
                size_list.append(CONV_BYTES)
        return ptr_list, size_list


class FakeIndexerPoolHost:
    """Stand-in for sglang's DSA indexer host pool.

    One IOV per page while ``host_indices`` are token-granular, i.e. it covers
    ``page_size`` indices per page -- the same shape the real pool exposes.
    """

    def __init__(self, *, page_size: int = 64) -> None:
        self.page_size = page_size
        self.dtype = torch.uint8

    def get_size_per_token(self) -> int:
        return INDEXER_PAGE_BYTES // self.page_size

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        size_list = []
        for i in range(0, len(indices), self.page_size):
            page = int(indices[i]) // self.page_size
            ptr_list.append(0x2000_0000 + page * INDEXER_PAGE_BYTES)
            size_list.append(INDEXER_PAGE_BYTES)
        return ptr_list, size_list


def _build_connector(
    *, pools: dict, has_mamba: bool = False, has_indexer: bool = False
):
    """Build a HiCacheKVCM with only the attributes the v2 paths read."""
    connector = HiCacheKVCM.__new__(HiCacheKVCM)
    connector.tp_rank = 0
    connector.tp_world_size = 1
    connector.is_mla_model = False
    connector.instance_id = "test-instance"
    connector.write_timeout_seconds = 30
    connector.location_spec_name = "tp_0"
    connector.location_spec_size = 4096
    connector.mamba_location_spec_name = "tp_0_linear"
    connector.mamba_spec_size = 512
    connector.indexer_location_spec_name = "tp_0_indexer"
    connector.indexer_spec_size = 256
    connector.has_mamba = has_mamba
    connector.has_indexer = has_indexer
    connector.registered_pools = pools
    connector.prefetch_pgs = []
    connector.backup_pgs = []
    connector.prefetch_bandwidth = []
    connector.backup_bandwidth = []
    connector._manager_client = MagicMock()
    connector.transfer_client = MagicMock()
    er_ok = _mock_kvcm.ClientErrorCode.ER_OK
    # SaveKvCaches returns (error code, uris) -- callers read [0];
    # LoadKvCaches returns the error code itself.
    connector.transfer_client.SaveKvCaches.return_value = (er_ok,)
    connector.transfer_client.LoadKvCaches.return_value = er_ok
    # These tests drive the data path of an attached connector; registration
    # and the (deferred) Manager handshake are covered by
    # sglang/test_pool_registration.py.
    connector._client_ready = True
    return connector


def _locations(spec_name: str, present: list[int], count: int) -> list[dict]:
    """Manager locations: only ``present`` pages carry a URI for spec_name."""
    return [
        {"location_specs": [{"name": spec_name, "uri": f"uri_{i}"}]}
        if i in present
        else {"location_specs": []}
        for i in range(count)
    ]


def _buffer_iovs(buffers) -> list[list[tuple[int, int]]]:
    return [[(iov.base, iov.size) for iov in buffer.iovs] for buffer in buffers]


class TestExtraPoolIovLayout(unittest.TestCase):
    """The per-page IOV count must follow get_page_buffer_meta()."""

    def test_conv_only_mamba_set_writes_one_buffer_per_page(self):
        pool = FakeMambaPoolHost(temporal_state=False, conv_buffers=2)
        connector = _build_connector(pools={PoolName.MAMBA: pool}, has_mamba=True)
        num_pages = 3
        connector._manager_client.start_write_cache.return_value = {
            "locations": _locations("tp_0_linear", [0, 1, 2], num_pages),
            "write_session_id": "ws-conv-only",
            "block_mask": {"offset": 0},
        }
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.arange(num_pages),
            keys=[f"block-{i}" for i in range(num_pages)],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        result = connector.batch_set_v2([transfer])

        self.assertEqual(result[PoolName.MAMBA], [True] * num_pages)
        uris, buffers = connector.transfer_client.SaveKvCaches.call_args.args
        self.assertEqual(uris, [f"uri_{i}" for i in range(num_pages)])
        self.assertEqual(
            _buffer_iovs(buffers),
            [
                expected_mamba_page_iovs(page, temporal_state=False, conv_buffers=2)
                for page in range(num_pages)
            ],
        )

    def test_ssm_mamba_set_keeps_temporal_iov(self):
        """Control: models with an SSM state still get temporal + conv IOVs."""
        pool = FakeMambaPoolHost(temporal_state=True, conv_buffers=2)
        connector = _build_connector(pools={PoolName.MAMBA: pool}, has_mamba=True)
        num_pages = 2
        connector._manager_client.start_write_cache.return_value = {
            "locations": _locations("tp_0_linear", [0, 1], num_pages),
            "write_session_id": "ws-ssm",
            "block_mask": {"offset": 0},
        }
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.arange(num_pages),
            keys=[f"block-{i}" for i in range(num_pages)],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        result = connector.batch_set_v2([transfer])

        self.assertEqual(result[PoolName.MAMBA], [True] * num_pages)
        _uris, buffers = connector.transfer_client.SaveKvCaches.call_args.args
        self.assertEqual(
            _buffer_iovs(buffers),
            [
                expected_mamba_page_iovs(page, temporal_state=True, conv_buffers=2)
                for page in range(num_pages)
            ],
        )

    def test_conv_only_mamba_get_maps_only_hit_pages(self):
        pool = FakeMambaPoolHost(temporal_state=False, conv_buffers=2)
        connector = _build_connector(pools={PoolName.MAMBA: pool}, has_mamba=True)
        num_pages = 4
        connector._manager_client.get_cache_location.return_value = {
            "locations": _locations("tp_0_linear", [1, 3], num_pages)
        }
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.arange(num_pages),
            keys=[f"block-{i}" for i in range(num_pages)],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        result = connector.batch_get_v2([transfer])

        self.assertEqual(result[PoolName.MAMBA], [False, True, False, True])
        uris, buffers = connector.transfer_client.LoadKvCaches.call_args.args
        self.assertEqual(uris, ["uri_1", "uri_3"])
        self.assertEqual(
            _buffer_iovs(buffers),
            [
                expected_mamba_page_iovs(page, temporal_state=False, conv_buffers=2)
                for page in (1, 3)
            ],
        )

    def test_partial_mamba_set_writes_only_missing_pages(self):
        """block_mask.offset > 0: only the not-yet-cached pages are written."""
        pool = FakeMambaPoolHost(temporal_state=False, conv_buffers=2)
        connector = _build_connector(pools={PoolName.MAMBA: pool}, has_mamba=True)
        num_pages = 5
        # Pages 0/1 are already cached; the manager only returns locations for
        # the 3 pages it wants written.
        connector._manager_client.start_write_cache.return_value = {
            "locations": _locations("tp_0_linear", [0, 1, 2], 3),
            "write_session_id": "ws-partial",
            "block_mask": {"offset": 2},
        }
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.arange(num_pages),
            keys=[f"block-{i}" for i in range(num_pages)],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        result = connector.batch_set_v2([transfer])

        self.assertEqual(result[PoolName.MAMBA], [True, True, True, True, True])
        uris, buffers = connector.transfer_client.SaveKvCaches.call_args.args
        self.assertEqual(uris, ["uri_0", "uri_1", "uri_2"])
        self.assertEqual(
            _buffer_iovs(buffers),
            [
                expected_mamba_page_iovs(page, temporal_state=False, conv_buffers=2)
                for page in (2, 3, 4)
            ],
        )

    def test_indexer_set_uses_token_granular_host_indices(self):
        """Control: a pool with page_size > 1 still yields one IOV per page."""
        page_size = 64
        pool = FakeIndexerPoolHost(page_size=page_size)
        connector = _build_connector(pools={PoolName.INDEXER: pool}, has_indexer=True)
        num_pages = 2
        connector._manager_client.start_write_cache.return_value = {
            "locations": _locations("tp_0_indexer", [0, 1], num_pages),
            "write_session_id": "ws-indexer",
            "block_mask": {"offset": 0},
        }
        transfer = PoolTransfer(
            name=PoolName.INDEXER,
            # token-granular indices, page_size of them per page
            host_indices=torch.arange(num_pages * page_size),
            keys=[f"block-{i}" for i in range(num_pages)],
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )

        result = connector.batch_set_v2([transfer])

        self.assertEqual(result[PoolName.INDEXER], [True, True])
        uris, buffers = connector.transfer_client.SaveKvCaches.call_args.args
        self.assertEqual(uris, ["uri_0", "uri_1"])
        self.assertEqual(
            _buffer_iovs(buffers),
            [
                [(0x2000_0000 + page * INDEXER_PAGE_BYTES, INDEXER_PAGE_BYTES)]
                for page in range(num_pages)
            ],
        )


class TestTransferShapeGuard(unittest.TestCase):
    """A transfer whose page count differs from its key count must fail loudly.

    Page positions are mapped back to key positions, so a transfer that
    carries more pages than keys used to be accepted silently: pages 0..N-1
    were taken as "the pages of keys 0..N-1".  The guard turns that into the
    usual "transfer failed" outcome instead.
    """

    def test_extra_pages_are_rejected_on_set(self):
        pool = FakeMambaPoolHost(temporal_state=False, conv_buffers=2)
        connector = _build_connector(pools={PoolName.MAMBA: pool}, has_mamba=True)
        num_keys = 2
        connector._manager_client.start_write_cache.return_value = {
            "locations": _locations("tp_0_linear", [0, 1], num_keys),
            "write_session_id": "ws-extra-pages",
            "block_mask": {"offset": 0},
        }
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.arange(4),  # 4 pages for 2 keys
            keys=[f"block-{i}" for i in range(num_keys)],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        with self.assertLogs("kv_cache_manager.py_connector.sglang.connector", "ERROR"):
            result = connector.batch_set_v2([transfer])

        self.assertEqual(result[PoolName.MAMBA], [False] * num_keys)
        connector.transfer_client.SaveKvCaches.assert_not_called()

    def test_extra_pages_are_rejected_on_get(self):
        pool = FakeMambaPoolHost(temporal_state=False, conv_buffers=2)
        connector = _build_connector(pools={PoolName.MAMBA: pool}, has_mamba=True)
        num_keys = 2
        connector._manager_client.get_cache_location.return_value = {
            "locations": _locations("tp_0_linear", [0, 1], num_keys)
        }
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.arange(4),  # 4 pages for 2 keys
            keys=[f"block-{i}" for i in range(num_keys)],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        with self.assertLogs("kv_cache_manager.py_connector.sglang.connector", "ERROR"):
            result = connector.batch_get_v2([transfer])

        self.assertEqual(result[PoolName.MAMBA], [False] * num_keys)
        connector.transfer_client.LoadKvCaches.assert_not_called()


class TestUnmanagedPoolPolicy(unittest.TestCase):
    """batch_exists_v2 must not claim hits for pools the connector ignores."""

    def setUp(self):
        # The report is deduplicated per process; start each test loud.
        HiCacheKVCM._reported_pools.clear()

    def test_unmanaged_pool_reports_zero_hits_and_warns_once(self):
        connector = _build_connector(pools={})
        num_pages = 3
        keys = [f"block-{i}" for i in range(num_pages)]
        connector._manager_client.get_cache_location.return_value = {
            "locations": _locations("tp_0", [0, 1, 2], num_pages)
        }
        transfer = PoolTransfer(
            name=UNMANAGED_POOL,
            host_indices=torch.arange(num_pages),
            keys=keys,
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )

        with self.assertLogs("kv_cache_manager.py_connector.sglang.connector") as logs:
            result = connector.batch_exists_v2(keys, [transfer])

        self.assertEqual(len(logs.records), 1)
        self.assertEqual(logs.records[0].levelname, "WARNING")
        self.assertIn(str(UNMANAGED_POOL), logs.records[0].getMessage())
        # The KV pages are cached, but without the side pool the prefix is
        # unusable: report the KV count per pool, and 0 usable pages overall.
        self.assertEqual(result.extra_pool_hit_pages[PoolName.KV], num_pages)
        self.assertEqual(result.extra_pool_hit_pages[UNMANAGED_POOL], 0)
        self.assertEqual(result.kv_hit_pages, 0)

        # Second query: same conservative answer, no duplicate warning.
        with self.assertNoLogs("kv_cache_manager.py_connector.sglang.connector"):
            result = connector.batch_exists_v2(keys, [transfer])
        self.assertEqual(result.kv_hit_pages, 0)


if __name__ == "__main__":
    unittest.main()
