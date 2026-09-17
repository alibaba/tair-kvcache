"""Deterministic (CPU-only) tests for the MLA zero-copy path.

MLA models are far too large to serve on a development box, and their v1 data
path differs from MHA in ways no other unit test pins down:

* one IOV per page (``kv_factor == 1``) instead of a K/V pair -- a wrong
  factor shifts every page, exactly the class of bug fixed for Mamba pools;
* the KV spec belongs to rank 0 and every rank reads it (MLA KV is replicated,
  rank 0 writes it);
* a non-rank-0 write must return all-False without touching storage.

The host pool is a fake, but the bytes are real: the fake transfer client moves
them with memmove, so "write, clear, read back" is compared byte for byte
instead of by pointer arithmetic alone.

Prerequisites: a sglang runtime that can import its own backend (``sgl_kernel``
and friends) and ``kv_cache_manager`` importable from the repository root on
``sys.path`` -- the compiled kvcm pybind client and the build-generated
``_version_info`` are stubbed below, so no wheel is needed for them.  This is a
runtime environment, not the one ``ty check`` uses; the ``attn_cp_*`` config
fields make v0.5.11 the oldest usable release.

    PYTHONPATH=$PWD python kv_cache_manager/py_connector/sglang/test_mla.py
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
# Same technique as test_extra_pools.py: the compiled pybind client and the
# build-generated version module must be importable before the connector
# module can be loaded.  Iov/BlockBuffer are plain value objects here (the
# connector writes base/size into them), so MagicMock cannot stand in.
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
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache  # noqa: E402

from kv_cache_manager.py_connector.sglang import connector as connector_module  # noqa: E402
from kv_cache_manager.py_connector.sglang.connector import HiCacheKVCM  # noqa: E402

PAGE_SIZE = 64
PAGE_BYTES = 1024
NUM_PAGES = 4


class FakeMlaHostPool(HostKVCache):
    """MLA host pool: one contiguous slot per page, one IOV per page."""

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
        """Token-granular indices in, one (ptr, size) per page out."""
        ptr_list = []
        size_list = []
        assert len(indices) % self.page_size == 0
        values = indices.tolist()
        for index in range(0, len(values), self.page_size):
            page = values[index] // self.page_size
            ptr_list.append(self._base + page * self.page_bytes)
            size_list.append(self.page_bytes)
        return ptr_list, size_list

    def fill(self, page: int, value: int) -> None:
        ctypes.memset(self._base + page * self.page_bytes, value, self.page_bytes)

    def page_base(self, page: int) -> int:
        """Address of a page's first byte, for IOV offset assertions."""
        return self._base + page * self.page_bytes

    def page(self, page: int) -> bytes:
        return ctypes.string_at(self.page_base(page), self.page_bytes)

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
        self.saved_uris: list[str] = []
        self.saved_buffers: list = []
        self.loaded_uris: list[str] = []

    def SaveKvCaches(self, uris, buffers):
        for uri, buffer in zip(uris, buffers):
            self.saved_uris.append(uri)
            self.blobs[uri] = b"".join(
                ctypes.string_at(iov.base, iov.size) for iov in buffer.iovs
            )
        self.saved_buffers = list(buffers)
        return (_mock_kvcm.ClientErrorCode.ER_OK,)

    def LoadKvCaches(self, uris, buffers):
        for uri, buffer in zip(uris, buffers):
            self.loaded_uris.append(uri)
            blob = self.blobs[uri]
            offset = 0
            for iov in buffer.iovs:
                src = ctypes.create_string_buffer(blob[offset : offset + iov.size])
                ctypes.memmove(iov.base, ctypes.addressof(src), iov.size)
                offset += iov.size
        return _mock_kvcm.ClientErrorCode.ER_OK


@contextmanager
def _patched_parallel_context():
    """Neutralize the collective parts of _init_parallel_context.

    ``new_group`` needs a real default process group; the MLA semantics under
    test (one IOV per page, rank-0 ownership) do not.
    """
    tp_group = types.SimpleNamespace(cpu_group=None)
    with (
        mock.patch.object(connector_module, "get_tp_group", lambda: tp_group),
        mock.patch.object(connector_module, "get_attn_tp_group", lambda: tp_group),
        mock.patch.object(connector_module, "is_dp_attention_enabled", lambda: False),
        mock.patch.object(torch.distributed, "get_world_size", lambda group=None: 1),
    ):
        yield


def _mla_connector(
    pool: FakeMlaHostPool, store: FakeByteStore, *, tp_rank: int
) -> tuple[HiCacheKVCM, Any]:
    """A rank of an MLA model with KVCM's Manager and SDK mocked out."""
    config = HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test-mla-model",
        extra_config={
            "manager_uri": "http://127.0.0.1:6382",
            "instance_group": "default",
            "instance_id": "mla-test",
        },
    )
    manager = MagicMock()
    manager.register_instance.return_value = {"storage_configs": "[]"}
    _mock_kvcm.TransferClient.Create.return_value = store

    connector = HiCacheKVCM(config, {})
    connector._manager_client = manager
    connector.register_mem_pool_host(pool)  # ty: ignore[invalid-argument-type]
    return connector, manager


def _locations(spec_uris: list[tuple[str, str]]) -> list[dict]:
    """One location per page, as the Manager returns them."""
    return [{"location_specs": [{"name": name, "uri": uri}]} for name, uri in spec_uris]


class TestMlaDataPath(unittest.TestCase):
    def test_write_then_read_back_is_byte_exact(self) -> None:
        pool = FakeMlaHostPool()
        store = FakeByteStore()
        for page in range(NUM_PAGES):
            pool.fill(page, 0x10 + page)

        with _patched_parallel_context():
            connector, manager = _mla_connector(pool, store, tp_rank=0)
            keys = [f"block-{page}" for page in range(NUM_PAGES)]
            host_indices = torch.arange(NUM_PAGES * PAGE_SIZE)
            manager.start_write_cache.return_value = {
                "locations": _locations(
                    [("tp_0", f"uri-{page}") for page in range(NUM_PAGES)]
                ),
                "write_session_id": "ws-mla",
                "block_mask": {"offset": 0},
            }

            self.assertEqual(
                connector.batch_set_v1(keys, host_indices), [True] * NUM_PAGES
            )

            # MLA replicates the KV cache: kv_factor is 1 (no K/V split), so
            # each page is exactly one IOV of the full page size.
            self.assertEqual(connector.kv_factor, 1)
            self.assertEqual(
                store.saved_uris, [f"uri-{page}" for page in range(NUM_PAGES)]
            )
            self.assertEqual(
                [
                    [(iov.base, iov.size) for iov in buffer.iovs]
                    for buffer in store.saved_buffers
                ],
                [[(pool.page_base(page), PAGE_BYTES)] for page in range(NUM_PAGES)],
            )
            for page in range(NUM_PAGES):
                self.assertEqual(
                    store.blobs[f"uri-{page}"],
                    bytes([0x10 + page]) * PAGE_BYTES,
                    f"page {page} was not written from its own buffer",
                )

            # Wipe the host pages, then read them back from the same URIs.
            for page in range(NUM_PAGES):
                pool.fill(page, 0)
            manager.get_cache_location.return_value = {
                "locations": _locations(
                    [("tp_0", f"uri-{page}") for page in range(NUM_PAGES)]
                )
            }
            self.assertEqual(
                connector.batch_get_v1(keys, host_indices), [True] * NUM_PAGES
            )

        self.assertEqual(
            store.loaded_uris, [f"uri-{page}" for page in range(NUM_PAGES)]
        )
        for page in range(NUM_PAGES):
            self.assertEqual(pool.page(page), bytes([0x10 + page]) * PAGE_BYTES)

    def test_non_rank0_write_is_refused_without_touching_storage(self) -> None:
        """MLA KV is written by rank 0 only; other ranks keep their data."""
        pool = FakeMlaHostPool()
        store = FakeByteStore()

        with _patched_parallel_context():
            connector, manager = _mla_connector(pool, store, tp_rank=1)
            manager.start_write_cache.return_value = {
                "locations": _locations([("tp_0", "uri-0")]),
                "write_session_id": "ws-mla-rank1",
                "block_mask": {"offset": 0},
            }

            result = connector.batch_set_v1(["block-0"], torch.arange(PAGE_SIZE))

        self.assertEqual(result, [False])
        self.assertEqual(store.saved_uris, [])
        self.assertIsNone(manager.start_write_cache.call_args)

    def test_every_rank_reads_the_rank0_spec(self) -> None:
        pool = FakeMlaHostPool()
        store = FakeByteStore()
        store.blobs["uri-rank0"] = b"\x2a" * PAGE_BYTES

        with _patched_parallel_context():
            connector, manager = _mla_connector(pool, store, tp_rank=1)
            manager.get_cache_location.return_value = {
                "locations": [
                    {
                        "location_specs": [
                            # A rank-1 spec exists but is not the one MLA KV
                            # lives in; reading it would return other data.
                            {"name": "tp_1", "uri": "uri-rank1"},
                            {"name": "tp_0", "uri": "uri-rank0"},
                        ]
                    }
                ]
            }
            result = connector.batch_get_v1(["block-0"], torch.arange(PAGE_SIZE))
            # Every rank reads the spec rank 0 wrote; there is no tp_{rank}
            # spec for MLA.
            self.assertEqual(connector.location_spec_name, "tp_0")

        self.assertEqual(result, [True])
        self.assertEqual(store.loaded_uris, ["uri-rank0"])
        self.assertEqual(pool.page(0), b"\x2a" * PAGE_BYTES)

    def test_non_rank0_v2_write_is_refused_without_touching_storage(self) -> None:
        """The v2 write path refuses MLA writes on non-rank-0 the same way.

        It has to be a *refusal*, not a crash: no write session, no bytes
        moved, and the caller still gets an all-False answer.
        """
        pool = FakeMlaHostPool()
        store = FakeByteStore()

        with _patched_parallel_context():
            connector, manager = _mla_connector(pool, store, tp_rank=1)
            connector.register_mem_host_pool_v2(FakeMlaHostPool(), PoolName.MAMBA)
            transfer = PoolTransfer(
                name=PoolName.MAMBA,
                host_indices=torch.arange(PAGE_SIZE),
                keys=["block-0"],
            )

            with self.assertLogs(connector_module.__name__, level="WARNING") as logs:
                result = connector.batch_set_v2([transfer])

        self.assertEqual(result, {PoolName.MAMBA: [False]})
        self.assertEqual(store.saved_uris, [])
        self.assertIsNone(manager.start_write_cache.call_args)
        messages = [record.getMessage() for record in logs.records]
        self.assertTrue(
            any("non-rank-0" in message for message in messages),
            "the refusal must say why the write was skipped",
        )
        self.assertFalse(
            any(record.levelname == "ERROR" for record in logs.records),
            "a refused MLA write is not a failure",
        )


if __name__ == "__main__":
    unittest.main()
