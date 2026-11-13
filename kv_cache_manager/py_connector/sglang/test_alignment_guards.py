"""Zero-copy data path: uris and buffers must stay aligned without ``assert``.

``LoadKvCaches``/``SaveKvCaches`` pair the uri list with the buffer list by
position, so a mismatch moves every page from the gap onward into another
block's slot -- the read path reports a hit for bytes that belong to a
different block, the write path commits them to storage.  The guards are the
ones ``python -O`` strips when they are written as ``assert``, so this file
runs under both interpreters and asserts the conservative answer: a miss (and a
session aborted with all-False), with both lengths in the log.

    PYTHONPATH=$PWD python    kv_cache_manager/py_connector/sglang/test_alignment_guards.py
    PYTHONPATH=$PWD python -O kv_cache_manager/py_connector/sglang/test_alignment_guards.py

Prerequisites are the two stubs below (the compiled pybind client and the
build-generated version module), the same technique as
``test_extra_pools.py``: no wheel and no CUDA device needed.
"""

import sys
import types
import unittest
from typing import Any
from unittest.mock import MagicMock

import torch


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

LOG = "kv_cache_manager.py_connector.sglang.connector"

PAGE_SIZE = 64  # tokens per page, i.e. the connector's block size
PAGE_BYTES = 0x1000
KV_FACTOR = 2  # the KV host pool hands out a K and a V IOV per page
ER_OK = _mock_kvcm.ClientErrorCode.ER_OK


class FakeKvHostPool:
    """Stand-in for sglang's MHATokenToKVPoolHost: ``kv_factor`` IOVs per page."""

    page_size = PAGE_SIZE

    def get_size_per_token(self) -> int:
        return PAGE_BYTES // PAGE_SIZE

    def get_page_buffer_meta(self, indices):
        ptr_list, size_list = [], []
        for page in range(len(indices) // PAGE_SIZE):
            for component in range(KV_FACTOR):
                base = 0x1000_0000 + (page * KV_FACTOR + component) * PAGE_BYTES
                ptr_list.append(base)
                size_list.append(PAGE_BYTES)
        return ptr_list, size_list


class FakeSidecarPool:
    """Stand-in for a page-granular side pool (Mamba): one IOV per page.

    ``temporal_first_page_only`` models an SSM state that only the first page
    carries, i.e. a flat IOV list that cannot be cut into equal pages -- the
    shape ``_page_buffer_meta`` has to refuse.
    """

    page_size = 1

    def __init__(self, *, temporal_first_page_only: bool = False) -> None:
        self.temporal_first_page_only = temporal_first_page_only

    def get_size_per_token(self) -> int:
        return PAGE_BYTES

    def get_page_buffer_meta(self, indices):
        ptr_list, size_list = [], []
        if self.temporal_first_page_only:
            ptr_list.append(0x3000_0000)
            size_list.append(PAGE_BYTES)
        for page in indices.tolist():
            ptr_list.append(0x2000_0000 + int(page) * PAGE_BYTES)
            size_list.append(PAGE_BYTES)
        return ptr_list, size_list


def _build_connector(*, pools: dict):
    """A HiCacheKVCM carrying the attributes both data paths read.

    The spec table is resolved the way ``_init_kvcm_client`` resolves it, so the
    data path sees the spec names this pool set produces (the anchor takes
    ``tp_0`` alone and ``tp_0_full`` next to a sidecar pool).
    """
    connector = HiCacheKVCM.__new__(HiCacheKVCM)
    connector.tp_rank = 0
    connector.tp_size = 1
    connector.tp_world_size = 1
    connector.kv_factor = KV_FACTOR
    connector.block_size = PAGE_SIZE
    connector.is_mla_model = False
    connector.instance_id = "test-instance"
    connector.write_timeout_seconds = 30
    connector.mem_pool_host = FakeKvHostPool()
    connector.registered_pools = {PoolName.KV: connector.mem_pool_host, **pools}
    connector.pool_specs = connector._resolve_pool_specs()
    anchor = connector.pool_specs[PoolName.KV]
    connector.location_spec_name = anchor.name
    connector.location_spec_size = anchor.size
    connector.prefetch_pgs = []
    connector.backup_pgs = []
    connector.prefetch_bandwidth = []
    connector.backup_bandwidth = []
    connector._manager_client = MagicMock()
    connector.transfer_client = MagicMock()
    # SaveKvCaches returns (error code, ...) -- callers read [0];
    # LoadKvCaches returns the error code itself.
    connector.transfer_client.SaveKvCaches.return_value = (ER_OK,)
    connector.transfer_client.LoadKvCaches.return_value = ER_OK
    connector._client_ready = True
    connector._closed = False
    return connector


def _locations(spec_name: str, present: list, count: int) -> list:
    """Manager locations: only the ``present`` ones carry a uri for spec_name."""
    return [
        {"location_specs": [{"name": spec_name, "uri": f"uri_{i}"}]}
        if i in present
        else {"location_specs": []}
        for i in range(count)
    ]


def _transfer(pool_name: Any, num_pages: int) -> PoolTransfer:
    return PoolTransfer(
        name=pool_name,
        host_indices=torch.arange(num_pages),
        keys=[f"block-{i}" for i in range(num_pages)],
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )


class TestKvAnchorReadPath(unittest.TestCase):
    """``batch_get_v1``: a location without the KV spec must not shift pages."""

    def test_a_location_without_the_kv_spec_is_refused(self):
        # Page 0's KV write never committed (its location carries a sidecar
        # spec only), pages 1..3 are real KV blocks: 3 uris against 4 host
        # pages, i.e. every page from the gap onward would get the next one's
        # bytes.
        connector = _build_connector(pools={})
        num_pages = 4
        other = {"location_specs": [{"name": "tp_0_linear", "uri": "mamba-0"}]}
        locations = [other] + _locations(connector.location_spec_name, [1, 2, 3], 4)[1:]
        connector._manager_client.get_cache_location.return_value = {
            "locations": locations
        }
        keys = [f"block-{i}" for i in range(num_pages)]

        with self.assertLogs(LOG, "ERROR") as logs:
            result = connector.batch_get_v1(keys, torch.arange(num_pages * PAGE_SIZE))

        self.assertEqual(result, [False] * num_pages)
        connector.transfer_client.LoadKvCaches.assert_not_called()
        message = "\n".join(logs.output)
        self.assertIn("3 uris for 4 buffers", message)
        self.assertIn(connector.location_spec_name, message)


class TestKvAnchorWritePath(unittest.TestCase):
    """``batch_set_v1``: the same two guards on the way into storage."""

    def test_a_location_without_the_kv_spec_is_refused(self):
        connector = _build_connector(pools={})
        num_pages = 3
        connector._manager_client.start_write_cache.return_value = {
            "locations": _locations(connector.location_spec_name, [0, 2], num_pages),
            "write_session_id": "ws-hole",
            "block_mask": {"offset": 0},
        }
        keys = [f"block-{i}" for i in range(num_pages)]

        with self.assertLogs(LOG, "ERROR") as logs:
            result = connector.batch_set_v1(keys, torch.arange(num_pages * PAGE_SIZE))

        self.assertEqual(result, [False] * num_pages)
        connector.transfer_client.SaveKvCaches.assert_not_called()
        finish = connector._manager_client.finish_write_cache.call_args.args[0]
        self.assertEqual(
            finish["success_blocks"]["bool_masks"]["values"], [False] * num_pages
        )
        self.assertIn("2 uris for 3 buffers", "\n".join(logs.output))

    def test_more_blocks_than_locations_aborts_the_session(self):
        # The mask wants 3 new blocks written, the session handed out 2
        # locations: writing by position would put uri_0/uri_1 into the slots
        # of blocks 0/1 and report all three committed.
        connector = _build_connector(pools={})
        num_pages = 3
        connector._manager_client.start_write_cache.return_value = {
            "locations": _locations(connector.location_spec_name, [0, 1], 2),
            "write_session_id": "ws-short",
            "block_mask": {"offset": 0},
        }
        keys = [f"block-{i}" for i in range(num_pages)]

        with self.assertLogs(LOG, "WARNING") as logs:
            result = connector.batch_set_v1(keys, torch.arange(num_pages * PAGE_SIZE))

        self.assertEqual(result, [False] * num_pages)
        connector.transfer_client.SaveKvCaches.assert_not_called()
        finish = connector._manager_client.finish_write_cache.call_args.args[0]
        # The session is closed over the locations it did hand out.
        self.assertEqual(
            finish["success_blocks"]["bool_masks"]["values"], [False, False]
        )
        message = "\n".join(logs.output)
        self.assertIn("3 new blocks + 0 prefix blocks", message)
        self.assertIn("2 locations", message)


class TestSidecarReadPath(unittest.TestCase):
    """``batch_get_v2``: a Manager answer longer than the query is refused."""

    def test_more_locations_than_pages_is_refused(self):
        # 5 locations for 4 host pages: the 5th uri has no buffer, so the two
        # lists cannot be paired by position.
        connector = _build_connector(pools={PoolName.MAMBA: FakeSidecarPool()})
        num_pages = 4
        spec_name = connector._get_extra_pool_spec_name(PoolName.MAMBA)
        connector._manager_client.get_cache_location.return_value = {
            "locations": _locations(spec_name, [0, 1, 2, 3, 4], 5)
        }

        with self.assertLogs(LOG, "ERROR") as logs:
            result = connector.batch_get_v2([_transfer(PoolName.MAMBA, num_pages)])

        self.assertEqual(result[PoolName.MAMBA], [False] * num_pages)
        connector.transfer_client.LoadKvCaches.assert_not_called()
        self.assertIn("5 uris for 4 buffers", "\n".join(logs.output))


class TestSidecarWritePath(unittest.TestCase):
    """``batch_set_v2``: a hole in the locations must not shift the pages."""

    def test_a_location_without_the_sidecar_spec_is_refused(self):
        connector = _build_connector(pools={PoolName.MAMBA: FakeSidecarPool()})
        num_pages = 3
        spec_name = connector._get_extra_pool_spec_name(PoolName.MAMBA)
        # The middle location carries no mamba uri: the KV write committed, the
        # sidecar session did not.
        locations = _locations(spec_name, [0, 2], num_pages)
        connector._manager_client.start_write_cache.return_value = {
            "locations": locations,
            "write_session_id": "ws-hole",
            "block_mask": {"offset": 0},
        }

        with self.assertLogs(LOG, "ERROR") as logs:
            result = connector.batch_set_v2([_transfer(PoolName.MAMBA, num_pages)])

        self.assertEqual(result[PoolName.MAMBA], [False] * num_pages)
        connector.transfer_client.SaveKvCaches.assert_not_called()
        finish = connector._manager_client.finish_write_cache.call_args.args[0]
        self.assertEqual(
            finish["success_blocks"]["bool_masks"]["values"], [False] * num_pages
        )
        self.assertIn("2 uris for 3 buffers", "\n".join(logs.output))

    def test_more_blocks_than_locations_aborts_the_session(self):
        connector = _build_connector(pools={PoolName.MAMBA: FakeSidecarPool()})
        num_pages = 3
        spec_name = connector._get_extra_pool_spec_name(PoolName.MAMBA)
        connector._manager_client.start_write_cache.return_value = {
            "locations": _locations(spec_name, [0, 1], 2),
            "write_session_id": "ws-short",
            "block_mask": {"offset": 0},
        }

        with self.assertLogs(LOG, "WARNING") as logs:
            result = connector.batch_set_v2([_transfer(PoolName.MAMBA, num_pages)])

        self.assertEqual(result[PoolName.MAMBA], [False] * num_pages)
        connector.transfer_client.SaveKvCaches.assert_not_called()
        finish = connector._manager_client.finish_write_cache.call_args.args[0]
        self.assertEqual(
            finish["success_blocks"]["bool_masks"]["values"], [False, False]
        )
        self.assertIn("2 locations", "\n".join(logs.output))


class TestPageAndIovShape(unittest.TestCase):
    """The page count and the IOV count per page are contract, not a hint."""

    def test_a_page_count_that_differs_from_the_key_count_is_refused(self):
        connector = _build_connector(pools={PoolName.MAMBA: FakeSidecarPool()})
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.arange(3),  # 3 host pages
            keys=[f"block-{i}" for i in range(4)],  # for 4 keys
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        with self.assertRaisesRegex(ValueError, "3 host pages .* 4 keys"):
            connector._page_buffer_meta(
                connector.registered_pools[PoolName.MAMBA],
                transfer,
                transfer.host_indices,
            )

    def test_iovs_that_do_not_tile_the_pages_are_refused(self):
        # 2 host pages but 3 IOVs: the flat list cannot be cut into pages, so
        # the component count of every page would be a guess.
        connector = _build_connector(
            pools={PoolName.MAMBA: FakeSidecarPool(temporal_first_page_only=True)}
        )
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.arange(2),
            keys=[f"block-{i}" for i in range(2)],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        with self.assertRaisesRegex(ValueError, "3 IOVs for 2 host indices"):
            connector._page_buffer_meta(
                connector.registered_pools[PoolName.MAMBA],
                transfer,
                transfer.host_indices,
            )


if __name__ == "__main__":
    unittest.main()
