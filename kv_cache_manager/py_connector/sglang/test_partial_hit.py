"""Deterministic (CPU-only) tests for a v1 GET that matches less than it was asked for.

``batch_get_v1`` queries a prefix match and then reads the pages the Manager
pointed at.  The Manager answers about a prefix of the keys, and it may answer
about fewer keys than the query carried -- what ``batch_exists`` matched can be
evicted before ``batch_get`` reads it.  A page the Manager did not point at was
never written by the transfer, so reporting it as a hit hands the caller the
bytes of the previous round in that host slot: silent corruption, not a miss.
A longer answer than the query is the other impossible shape, and is refused
rather than truncated.

The host pool is a fake, but the bytes are real: the fake transfer client moves
them with memmove, and every unmatched page is left holding a sentinel.

Prerequisites: a sglang runtime that can import its own backend (``sgl_kernel``
and friends) and ``kv_cache_manager`` importable from the repository root on
``sys.path`` -- the compiled kvcm pybind client and the build-generated
``_version_info`` are stubbed below, so no wheel is needed for them.

    PYTHONPATH=$PWD python kv_cache_manager/py_connector/sglang/test_partial_hit.py
"""

import ctypes
import sys
import types
import unittest
from contextlib import contextmanager
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import torch


# ── Modules the connector imports but this test does not exercise ──────
# Same technique as test_extra_pools.py and test_mla.py: the compiled pybind
# client and the build-generated version module must be importable before the
# connector module can be loaded.  Iov/BlockBuffer are plain value objects here
# (the connector writes base/size into them), so MagicMock cannot stand in.
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

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig  # noqa: E402
from sglang.srt.mem_cache.memory_pool_host import HostKVCache  # noqa: E402

from kv_cache_manager.py_connector.sglang import connector as connector_module  # noqa: E402
from kv_cache_manager.py_connector.sglang.connector import HiCacheKVCM  # noqa: E402

PAGE_SIZE = 64
PAGE_BYTES = 1024
NUM_PAGES = 4
# What an untouched host page holds; a read that reports a page it did not write
# hands these bytes to the device.
SENTINEL = 0xEE


class FakeKvHostPool(HostKVCache):
    """MHA host pool: two IOVs per page, the shape ``kv_factor == 2`` slices."""

    def __init__(self, *, page_bytes: int = PAGE_BYTES) -> None:
        self.page_size = PAGE_SIZE
        self.page_num = NUM_PAGES
        self.page_bytes = page_bytes
        self.dtype = torch.bfloat16
        self._buffer = (ctypes.c_ubyte * (self.page_num * page_bytes))()
        self._base = ctypes.addressof(self._buffer)

    def get_size_per_token(self) -> int:
        return self.page_bytes // self.page_size

    def get_page_buffer_meta(self, indices):
        """Token-granular indices in, one (ptr, size) per K/V half per page."""
        ptr_list = []
        size_list = []
        values = indices.tolist()
        assert len(values) % self.page_size == 0
        for start in range(0, len(values), self.page_size):
            page = values[start] // self.page_size
            for half in range(2):
                ptr_list.append(
                    self._base + page * self.page_bytes + half * self.page_bytes // 2
                )
                size_list.append(self.page_bytes // 2)
        return ptr_list, size_list

    def fill(self, page: int, value: int) -> None:
        ctypes.memset(self._base + page * self.page_bytes, value, self.page_bytes)

    def page(self, page: int) -> bytes:
        return ctypes.string_at(self._base + page * self.page_bytes, self.page_bytes)

    # HostKVCache ABC hooks that need real device pools; unused here.
    def init_kv_buffer(self) -> Any:
        raise NotImplementedError

    def get_data_page(self, index: Any, flat: bool = True) -> Any:
        raise NotImplementedError

    def get_dummy_flat_data_page(self) -> Any:
        raise NotImplementedError

    def set_from_flat_data_page(self, index: Any, data_page: Any) -> None:
        raise NotImplementedError

    def load_to_device_per_layer(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def backup_from_device_all_layer(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class FakeByteStore:
    """Byte-level stand-in for the pybind TransferClient."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.loaded_uris: list[str] = []
        self.load_error: Any = _mock_kvcm.ClientErrorCode.ER_OK

    def LoadKvCaches(self, uris, buffers):
        for uri, buffer in zip(uris, buffers):
            self.loaded_uris.append(uri)
            blob = self.blobs[uri]
            offset = 0
            for iov in buffer.iovs:
                src = ctypes.create_string_buffer(blob[offset : offset + iov.size])
                ctypes.memmove(iov.base, ctypes.addressof(src), iov.size)
                offset += iov.size
        return self.load_error


@contextmanager
def _patched_parallel_context():
    """Neutralize the collective parts of _init_parallel_context."""
    tp_group = types.SimpleNamespace(cpu_group=None)
    with (
        mock.patch.object(connector_module, "get_tp_group", lambda: tp_group),
        mock.patch.object(connector_module, "get_attn_tp_group", lambda: tp_group),
        mock.patch.object(connector_module, "is_dp_attention_enabled", lambda: False),
        mock.patch.object(torch.distributed, "get_world_size", lambda group=None: 1),
    ):
        yield


class TestPartialHit(unittest.TestCase):
    def setUp(self) -> None:
        _mock_kvcm.reset_mock()
        HiCacheKVCM._reported_pools.clear()

    def _connector(self, pool: FakeKvHostPool, store: FakeByteStore) -> tuple[Any, Any]:
        """A flat-model rank whose Manager is mocked and whose SDK is ``store``."""
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="test-model",
            extra_config={
                "manager_uri": "http://127.0.0.1:6382",
                "instance_group": "default",
                "instance_id": "partial-hit-test",
            },
        )
        manager = MagicMock()
        manager.register_instance.return_value = {"storage_configs": "[]"}
        _mock_kvcm.TransferClient.Create.return_value = store

        connector = HiCacheKVCM(config, {})
        connector._manager_client = manager
        connector.register_mem_pool_host(pool)
        connector._ensure_client()
        return connector, manager

    def _locations(self, connector: Any, pages: int) -> dict:
        """A prefix match that pointed at the first ``pages`` keys."""
        return {
            "locations": [
                {
                    "location_specs": [
                        {"name": connector.location_spec_name, "uri": f"uri-{page}"}
                    ]
                }
                for page in range(pages)
            ]
        }

    def _read(
        self, matched: int, *, load_error: Any = None
    ) -> tuple[FakeKvHostPool, FakeByteStore, list, Any]:
        """Read NUM_PAGES keys from a Manager that matches ``matched`` of them."""
        pool = FakeKvHostPool()
        store = FakeByteStore()
        if load_error is not None:
            store.load_error = load_error
        for page in range(NUM_PAGES):
            pool.fill(page, SENTINEL)
            store.blobs[f"uri-{page}"] = bytes([0x10 + page]) * PAGE_BYTES
        with _patched_parallel_context():
            connector, manager = self._connector(pool, store)
            manager.get_cache_location.return_value = self._locations(
                connector, matched
            )
            keys = [f"block-{page}" for page in range(NUM_PAGES)]
            host_indices = torch.arange(NUM_PAGES * PAGE_SIZE)
            result = connector.batch_get_v1(keys, host_indices)
        return pool, store, result, connector

    def test_only_the_pages_the_transfer_filled_are_hits(self) -> None:
        """Two of four keys were still cached: the other two slots stay untouched."""
        pool, store, result, connector = self._read(matched=2)

        self.assertEqual(result, [True, True, False, False])
        self.assertEqual(store.loaded_uris, ["uri-0", "uri-1"])
        for page in range(2):
            self.assertEqual(pool.page(page), bytes([0x10 + page]) * PAGE_BYTES)
        for page in range(2, NUM_PAGES):
            self.assertEqual(
                pool.page(page),
                bytes([SENTINEL]) * PAGE_BYTES,
                f"page {page} was reported as a hit but never written",
            )
        # Only the pages that moved count as prefetched.
        self.assertEqual(connector.prefetch_pgs, [2])

    def test_a_full_match_is_all_hits(self) -> None:
        pool, store, result, _ = self._read(matched=NUM_PAGES)

        self.assertEqual(result, [True] * NUM_PAGES)
        self.assertEqual(
            store.loaded_uris, [f"uri-{page}" for page in range(NUM_PAGES)]
        )
        for page in range(NUM_PAGES):
            self.assertEqual(pool.page(page), bytes([0x10 + page]) * PAGE_BYTES)

    def test_no_match_is_all_misses_and_moves_nothing(self) -> None:
        pool, store, result, connector = self._read(matched=0)

        self.assertEqual(result, [False] * NUM_PAGES)
        self.assertEqual(store.loaded_uris, [])
        self.assertEqual(connector.prefetch_pgs, [])
        for page in range(NUM_PAGES):
            self.assertEqual(pool.page(page), bytes([SENTINEL]) * PAGE_BYTES)

    def test_a_failed_transfer_reports_no_hit(self) -> None:
        """A partial match does not turn the transfer's own failure into hits."""
        _pool, store, result, connector = self._read(matched=2, load_error=-1)

        self.assertEqual(result, [False] * NUM_PAGES)
        # The matched pages were still attempted, and stay in the counters.
        self.assertEqual(store.loaded_uris, ["uri-0", "uri-1"])
        self.assertEqual(connector.prefetch_pgs, [2])

    def test_more_locations_than_queried_keys_is_refused(self) -> None:
        """A longer answer than the query cannot be mapped onto host slots."""
        pool = FakeKvHostPool()
        store = FakeByteStore()
        with _patched_parallel_context():
            connector, manager = self._connector(pool, store)
            manager.get_cache_location.return_value = self._locations(connector, 3)
            with self.assertRaises(RuntimeError) as raised:
                connector._batch_get(
                    [f"block-{page}" for page in range(2)],
                    torch.arange(2 * PAGE_SIZE),
                    "trace",
                )

            message = str(raised.exception)
            self.assertIn("3 locations", message)
            self.assertIn("2 queried", message)

            # The data path entry point answers a conservative miss instead of
            # letting the refusal escape into the engine loop.
            with self.assertLogs(connector_module.__name__, level="ERROR"):
                self.assertEqual(
                    connector.batch_get_v1(
                        [f"block-{page}" for page in range(2)],
                        torch.arange(2 * PAGE_SIZE),
                    ),
                    [False, False],
                )

        self.assertEqual(store.loaded_uris, [])


if __name__ == "__main__":
    unittest.main()
