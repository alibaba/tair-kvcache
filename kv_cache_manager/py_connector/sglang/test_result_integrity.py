"""Deterministic (CPU-only) tests for what the connector reports as done.

Two reports have to agree with the work behind them:

* ``batch_exists`` counts the pages a prefix match returned.  The Manager counts
  a block as a hit once *any* spec group registered it, so a block written only
  by a sidecar session (KV write failed, sidecar session of the same block
  succeeded -- the two write sessions are independent) comes back as a location
  that carries no KV spec for this rank.  Counting it promises a page the
  transfer cannot fill: the whole v1 GET used to be dropped on the
  ``len(uris) == len(buffers)`` assertion, i.e. a reported 1..n hits and nothing
  moved.
* ``batch_set_v2`` reports a block as written once its write session is
  committed.  A failing ``finish_write_cache`` leaves the session uncommitted, so
  the bytes never reach storage; reporting the write anyway hands the caller a
  success for blocks the Manager will answer as misses.

The fakes move real bytes through ``ctypes``, so a page the connector claims to
have filled but did not is visible as a sentinel.

Prerequisites: a sglang runtime that can import its own backend (``sgl_kernel``
and friends) and ``kv_cache_manager`` importable from the repository root on
``sys.path`` -- the compiled kvcm pybind client and the build-generated
``_version_info`` are stubbed below, so no wheel is needed for them.

    PYTHONPATH=$PWD python kv_cache_manager/py_connector/sglang/test_result_integrity.py
"""

import ctypes
import sys
import types
import unittest
from typing import Any
from unittest.mock import MagicMock

import torch


# ── Modules the connector imports but this test does not exercise ──────
# Same technique as test_extra_pools.py: Iov/BlockBuffer are plain value objects
# here (the connector writes base/size into them), so MagicMock cannot stand in.
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

from kv_cache_manager.py_connector.sglang import connector as connector_module  # noqa: E402
from kv_cache_manager.py_connector.sglang.connector import HiCacheKVCM  # noqa: E402

PAGE_SIZE = 64
PAGE_BYTES = 1024
MAMBA_PAGE_BYTES = 0x200
NUM_PAGES = 3
# What an untouched host page holds; a page reported as a hit but never written
# hands these bytes to the device.
SENTINEL = 0xEE
# A spec name this rank does not own -- the shape a block carries when a sidecar
# session committed it but the KV session of the same block did not.
OTHER_SPEC = "tp_0_linear"

# What a Manager location carries for one page of the query.
KV = "kv"  # this rank's KV spec: readable
SIDECAR = "sidecar"  # only OTHER_SPEC, as the Manager saw the block
NOTHING = "nothing"  # registered, empty spec list
UNSPECIFIED = "unspecified"  # registered, no "location_specs" key at all


class FakeKvPoolHost:
    """MHA host pool: two IOVs per page, the shape ``kv_factor == 2`` slices."""

    def __init__(self, *, page_size: int = PAGE_SIZE, page_bytes: int = PAGE_BYTES):
        self.page_size = page_size
        self.page_bytes = page_bytes
        self.dtype = torch.bfloat16
        self._buffer = (ctypes.c_ubyte * (NUM_PAGES * page_bytes))()
        self._base = ctypes.addressof(self._buffer)

    def get_size_per_token(self) -> int:
        return self.page_bytes // self.page_size

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        size_list = []
        values = indices.tolist()
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


class FakeMambaPoolHost:
    """Stand-in for sglang's MambaPoolHost (page_size == 1)."""

    def __init__(self) -> None:
        self.page_size = 1
        self.dtype = torch.bfloat16
        self.conv_buffer = [torch.zeros(1) for _ in range(2)]

    def get_size_per_token(self) -> int:
        return MAMBA_PAGE_BYTES

    def get_page_buffer_meta(self, indices):
        ptr_list = [0x1000_0000 + 0x1000 * int(page) for page in indices.tolist()]
        return ptr_list, [MAMBA_PAGE_BYTES] * len(ptr_list)


class FakeByteStore:
    """Byte-level stand-in for the pybind TransferClient."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.loaded: list[str] = []
        self.saved: list[list[str]] = []
        self.load_error: Any = _mock_kvcm.ClientErrorCode.ER_OK

    def LoadKvCaches(self, uris, buffers):
        for uri, buffer in zip(uris, buffers):
            self.loaded.append(uri)
            if self.load_error != _mock_kvcm.ClientErrorCode.ER_OK:
                continue
            blob = self.blobs[uri]
            offset = 0
            for iov in buffer.iovs:
                src = ctypes.create_string_buffer(blob[offset : offset + iov.size])
                ctypes.memmove(iov.base, ctypes.addressof(src), iov.size)
                offset += iov.size
        return self.load_error

    def SaveKvCaches(self, uris, buffers):
        self.saved.append(list(uris))
        return (_mock_kvcm.ClientErrorCode.ER_OK,)


def _build_connector(*, pools: dict | None = None) -> Any:
    """Build a HiCacheKVCM with only the attributes the data paths read.

    A KV anchor is registered alongside ``pools`` and the spec table is resolved
    the way ``_init_kvcm_client`` resolves it, so the paths under test see the
    spec names of a real instance.
    """
    kv_pool = FakeKvPoolHost()
    connector = HiCacheKVCM.__new__(HiCacheKVCM)
    connector.tp_rank = 0
    connector.tp_size = 1
    connector.tp_world_size = 1
    connector.block_size = PAGE_SIZE
    connector.is_mla_model = False
    connector.kv_factor = 2
    connector.instance_id = "result-integrity-test"
    connector.write_timeout_seconds = 30
    connector.registered_pools = {PoolName.KV: kv_pool, **(pools or {})}
    connector.pool_specs = connector._resolve_pool_specs()
    anchor = connector.pool_specs[PoolName.KV]
    connector.location_spec_name = anchor.name
    connector.location_spec_size = anchor.size
    connector.mem_pool_host = kv_pool
    connector.prefetch_pgs = []
    connector.backup_pgs = []
    connector.prefetch_bandwidth = []
    connector.backup_bandwidth = []
    connector._manager_client = MagicMock()
    connector.transfer_client = FakeByteStore()
    # Registration is covered by sglang/test_pool_registration.py.
    connector._client_ready = True
    return connector


def _manager_answer(connector: Any, markers: list[str]) -> dict:
    """A prefix-match answer where each page carries what ``markers`` says."""
    locations = []
    for index, marker in enumerate(markers):
        if marker == UNSPECIFIED:
            locations.append({})
            continue
        specs = []
        if marker == KV:
            specs.append({"name": connector.location_spec_name, "uri": f"uri-{index}"})
        elif marker == SIDECAR:
            specs.append({"name": OTHER_SPEC, "uri": f"lin-{index}"})
        locations.append({"location_specs": specs})
    return {"locations": locations}


class TestV1HitCount(unittest.TestCase):
    """``batch_exists`` must not promise a page ``batch_get_v1`` cannot serve."""

    def _read(self, markers: list[str], *, load_error: Any = None):
        connector = _build_connector()
        if load_error is not None:
            connector.transfer_client.load_error = load_error
        for page in range(NUM_PAGES):
            connector.mem_pool_host.fill(page, SENTINEL)
            connector.transfer_client.blobs[f"uri-{page}"] = bytes([0x10 + page]) * (
                PAGE_BYTES
            )
        connector._manager_client.get_cache_location.return_value = _manager_answer(
            connector, markers
        )
        keys = [f"block-{page}" for page in range(NUM_PAGES)]
        exists = connector.batch_exists(keys)
        served = connector.batch_get_v1(keys, torch.arange(NUM_PAGES * PAGE_SIZE))
        return connector, exists, served

    def test_a_sidecar_only_page_is_not_a_kv_hit(self) -> None:
        """A gap in the prefix ends the run: the page after it is not filled."""
        connector, exists, served = self._read([KV, SIDECAR, KV])

        self.assertEqual(exists, 1)
        self.assertEqual(served, [True, False, False])
        # The readable page moved; the page behind the gap kept its bytes even
        # though the Manager did point at a location for it.
        self.assertEqual(connector.transfer_client.loaded, ["uri-0"])
        self.assertEqual(connector.mem_pool_host.page(0), bytes([0x10]) * PAGE_BYTES)
        for page in range(1, NUM_PAGES):
            self.assertEqual(
                connector.mem_pool_host.page(page),
                bytes([SENTINEL]) * PAGE_BYTES,
                f"page {page} was reported as a hit but never written",
            )
        self.assertEqual(connector.prefetch_pgs, [1])

    def test_exists_counts_exactly_what_a_successful_get_serves(self) -> None:
        cases = {
            "every page readable": [KV, KV, KV],
            "gap in the middle": [KV, SIDECAR, KV],
            "gap first": [SIDECAR, KV, KV],
            "location with an empty spec list": [KV, NOTHING, KV],
            "location without the specs key": [KV, UNSPECIFIED, KV],
            "no readable page": [SIDECAR, SIDECAR, SIDECAR],
            "empty answer": [],
        }
        for name, markers in cases.items():
            with self.subTest(name):
                connector, exists, served = self._read(markers)
                hits = next(
                    (i for i, marker in enumerate(markers) if marker != KV),
                    len(markers),
                )

                self.assertEqual(len(served), NUM_PAGES)
                self.assertEqual(exists, sum(served), "exists and get disagree")
                self.assertEqual(exists, hits)
                self.assertEqual(
                    connector.transfer_client.loaded,
                    [f"uri-{page}" for page in range(hits)],
                )

    def test_a_failed_transfer_is_not_a_hit(self) -> None:
        """The readable pages are attempted, and none of them counts as served."""
        with self.assertLogs(connector_module.__name__, level="ERROR"):
            connector, exists, served = self._read([KV, KV, KV], load_error=-1)

        self.assertEqual(exists, NUM_PAGES)
        self.assertEqual(served, [False] * NUM_PAGES)
        self.assertEqual(
            connector.transfer_client.loaded, [f"uri-{page}" for page in range(3)]
        )
        for page in range(NUM_PAGES):
            self.assertEqual(
                connector.mem_pool_host.page(page), bytes([SENTINEL]) * PAGE_BYTES
            )


class TestV2HitCount(unittest.TestCase):
    """The v2 answer is the same count read by the same rule."""

    def test_both_read_paths_count_the_same_pages(self) -> None:
        connector = _build_connector()
        connector._manager_client.get_cache_location.return_value = _manager_answer(
            connector, [KV, SIDECAR, KV]
        )
        keys = [f"block-{page}" for page in range(NUM_PAGES)]

        result = connector.batch_exists_v2(keys)

        self.assertEqual(result.kv_hit_pages, 1)
        self.assertEqual(connector.batch_exists(keys), 1)


class TestV2Commit(unittest.TestCase):
    """A write is reported as done only once its session is committed."""

    def _write(self, connector: Any) -> dict:
        connector._manager_client.start_write_cache.return_value = {
            "locations": [
                {"location_specs": [{"name": OTHER_SPEC, "uri": f"lin-{i}"}]}
                for i in range(NUM_PAGES)
            ],
            "write_session_id": "ws-1",
            "block_mask": {"offset": 0},
        }
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.arange(NUM_PAGES),
            keys=[f"block-{i}" for i in range(NUM_PAGES)],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        return connector.batch_set_v2([transfer])

    def test_a_failed_commit_is_not_a_write(self) -> None:
        connector = _build_connector(pools={PoolName.MAMBA: FakeMambaPoolHost()})
        connector._manager_client.finish_write_cache.side_effect = RuntimeError(
            "manager said no"
        )

        with self.assertLogs(connector_module.__name__, level="ERROR") as logs:
            result = self._write(connector)

        # The bytes were transferred, but nothing of them is in storage.
        self.assertEqual(connector.transfer_client.saved, [["lin-0", "lin-1", "lin-2"]])
        connector._manager_client.finish_write_cache.assert_called_once()
        self.assertEqual(result[PoolName.MAMBA], [False] * NUM_PAGES)
        messages = "\n".join(logs.output)
        self.assertIn("finish_write_cache failed", messages)
        self.assertIn("pool=mamba", messages)

    def test_a_committed_write_is_a_write(self) -> None:
        connector = _build_connector(pools={PoolName.MAMBA: FakeMambaPoolHost()})

        result = self._write(connector)

        self.assertEqual(result[PoolName.MAMBA], [True] * NUM_PAGES)
        _request = connector._manager_client.finish_write_cache.call_args.args[0]
        self.assertEqual(
            _request["success_blocks"]["bool_masks"]["values"], [True] * NUM_PAGES
        )


if __name__ == "__main__":
    unittest.main()
