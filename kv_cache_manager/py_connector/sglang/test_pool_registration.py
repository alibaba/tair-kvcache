"""Deterministic (CPU-only) tests for the connector's pool registration model.

sglang changed how it hands host pools to a HiCache storage backend in
v0.5.19: the v1 hook now carries only the KV anchor pool, and every pool
(Mamba/Indexer included) arrives through ``register_mem_host_pool_v2``.  Up to
v0.5.18 a hybrid stack passed the whole ``HostPoolGroup`` to v1 instead, and the
connector used to unpack its ``entries``.

These tests pin the resulting contract without a GPU, a Manager or a
process group:

* the registration burst (v1 anchor, then one v2 call per pool) is what the
  connection to the Manager is built from, and both upstream orders must
  produce the *identical* ``register_instance`` payload;
* ``HostPoolGroup.entries`` is not a source of pools any more -- the KV pool of
  a hybrid group is the one v2 registers;
* the Manager registration and the SDK client are created on the first storage
  call (never before the pool set is final) and exactly once;
* a pool registered after that point is reported once and degraded to misses.

* a pool registered after initialization is not recorded and stays unusable,
  even when it reuses the name of a pool whose spec is already registered.

Prerequisites: a sglang runtime that can import its own backend (``sgl_kernel``
and friends) and ``kv_cache_manager`` importable from the repository root on
``sys.path`` -- the compiled kvcm pybind client and the build-generated
``_version_info`` are stubbed below, so no wheel is needed for them.  This is a
runtime environment, not the one ``ty check`` uses.  Enum members that do not
exist on older releases (``SWA`` since v0.5.12, ``DRAFT`` since v0.5.13) are
resolved with ``getattr``; the file itself needs sglang >= v0.5.11, whose
``HiCacheStorageConfig`` carries the ``attn_cp_*`` fields.

    PYTHONPATH=$PWD python kv_cache_manager/py_connector/sglang/test_pool_registration.py
"""

import json
import sys
import threading
import types
import unittest
import weakref
from contextlib import contextmanager
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import torch

# ── Modules the connector imports but this test does not exercise ──────
# Same technique as test_extra_pools.py: the compiled pybind client and the
# build-generated version module must be importable before the connector
# module can be loaded.
_mock_pybind: Any = types.ModuleType("kv_cache_manager.client.pybind")
_mock_kvcm = MagicMock()
_mock_kvcm.ClientErrorCode.ER_OK = 0
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
KV_BYTES_PER_TOKEN = 512
MAMBA_BYTES_PER_TOKEN = 96
# Never used by a correct connector: reading it means pools were taken from
# HostPoolGroup.entries.
SENTINEL_BYTES_PER_TOKEN = 7
# A pool the connector does not manage: PoolName.SWA is the realistic case
# (sglang >= v0.5.12), MAMBA is the stand-in on older versions where the enum
# member does not exist yet.  It stays an enum member -- enum hashes differ
# from plain string hashes, so a bare "swa" would not find the connector's
# per-pool result entries.
UNMANAGED_POOL = getattr(PoolName, "SWA", PoolName.MAMBA)
# Same idea for the never-registered pool of the read/write-path test; DRAFT
# exists since v0.5.13 (MAMBA is unmanaged here: the connector has no v2 pool).
UNMANAGED_DRAFT = getattr(PoolName, "DRAFT", PoolName.MAMBA)


class FakeHostPool(HostKVCache):
    """A sglang host pool without CUDA: only what the registration path reads.

    The ABC's transfer hooks need real device pools, so they are stubbed out.
    Subclassing for real keeps the connector's ``isinstance`` check (and with
    it the "a group is not a host pool" distinction) meaningful.
    """

    def __init__(self, *, bytes_per_token: int, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.bytes_per_token = bytes_per_token
        self.page_size = PAGE_SIZE

    def get_size_per_token(self) -> int:
        return self.bytes_per_token

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


class FakeHostPoolGroup:
    """Shape of the <= 0.5.18 hybrid v1 argument.

    ``entries`` holds a sentinel that no correct connector may use: since
    v0.5.19 the pool set comes from the v2 hook, and on old versions the KV
    entry arrives through v2 as well.
    """

    def __init__(self) -> None:
        sentinel = FakeHostPool(
            bytes_per_token=SENTINEL_BYTES_PER_TOKEN, dtype=torch.bfloat16
        )
        self.entries = [
            types.SimpleNamespace(name=PoolName.KV, host_pool=sentinel),
            types.SimpleNamespace(name=PoolName.MAMBA, host_pool=sentinel),
        ]
        self.page_size = PAGE_SIZE


class FakeLogicalHostPool:
    """Shape of sglang's ``LogicalHostPool`` (v0.5.19 DeepSeek-V4 compressed KV).

    The pool is a pure page anchor: it is not a ``HostKVCache`` and has no
    ``get_size_per_token``, so no location spec can be built from it.  The
    connector must refuse it once instead of failing on every storage call.
    """

    def __init__(self) -> None:
        self.page_size = PAGE_SIZE
        self.dtype = torch.uint8


@contextmanager
def _patched_parallel_context():
    """Neutralize the collective parts of _init_parallel_context.

    ``new_group`` needs every rank of a real default process group, which a
    CPU-only test does not have.  ``tp_size`` (a storage config field) stays
    configurable, so spec naming for several ranks is still covered.
    """
    tp_group = types.SimpleNamespace(cpu_group=None)
    with (
        mock.patch.object(connector_module, "get_tp_group", lambda: tp_group),
        mock.patch.object(connector_module, "get_attn_tp_group", lambda: tp_group),
        mock.patch.object(connector_module, "is_dp_attention_enabled", lambda: False),
        mock.patch.object(torch.distributed, "get_world_size", lambda group=None: 1),
    ):
        yield


def _storage_config(**overrides: Any) -> HiCacheStorageConfig:
    args: dict[str, Any] = {
        "tp_rank": 0,
        "tp_size": 2,
        "pp_rank": 0,
        "pp_size": 1,
        "attn_cp_rank": 0,
        "attn_cp_size": 1,
        "is_mla_model": False,
        "enable_storage_metrics": False,
        "is_page_first_layout": True,
        "model_name": "test-model",
        "extra_config": {
            "manager_uri": "http://127.0.0.1:6382",
            "instance_group": "default",
            "instance_id": "registration-test",
        },
    }
    args.update(overrides)
    return HiCacheStorageConfig(**args)


def _connector(**overrides: Any) -> HiCacheKVCM:
    """Build a connector whose Manager is mocked out."""
    connector = HiCacheKVCM(_storage_config(**overrides), {})
    manager = MagicMock()
    manager.register_instance.return_value = {"storage_configs": "[]"}
    manager.get_cache_location.return_value = {"locations": []}
    connector._manager_client = manager
    return connector


def _manager(connector: HiCacheKVCM) -> Any:
    """The mocked Manager client of a connector built by ``_connector()``."""
    return connector._manager_client


def _register_instance_payload(connector: HiCacheKVCM) -> dict:
    """The register_instance request, minus its per-call trace id.

    Reading it is what a storage call does once the pool set is final, so the
    helper drives that call itself.
    """
    connector.batch_exists(["block-0"])
    call = _manager(connector).register_instance.call_args
    assert call is not None, "register_instance was never called"
    payload = dict(call.args[0])
    payload.pop("trace_id")
    return payload


class TestRegistrationOrders(unittest.TestCase):
    """Both upstream orders must yield the same instance registration."""

    def setUp(self) -> None:
        _mock_kvcm.reset_mock()
        HiCacheKVCM._reported_pools.clear()

    def _kv_pool(self) -> FakeHostPool:
        return FakeHostPool(bytes_per_token=KV_BYTES_PER_TOKEN, dtype=torch.bfloat16)

    def _mamba_pool(self) -> FakeHostPool:
        return FakeHostPool(bytes_per_token=MAMBA_BYTES_PER_TOKEN, dtype=torch.float32)

    def _expected_payload(self) -> dict:
        kv_size = KV_BYTES_PER_TOKEN * PAGE_SIZE
        linear_size = MAMBA_BYTES_PER_TOKEN
        return {
            "instance_group": "default",
            "instance_id": "registration-test",
            "model_deployment": {
                "model_name": "test-model",
                "tp_size": 2,
                "dp_size": 1,
                "pp_size": 1,
                "use_mla": False,
                "dtype": "bfloat16",
            },
            "block_size": PAGE_SIZE,
            "location_spec_infos": [
                {"name": "tp_0_full", "size": kv_size},
                {"name": "tp_1_full", "size": kv_size},
                {"name": "tp_0_linear", "size": linear_size},
                {"name": "tp_1_linear", "size": linear_size},
            ],
            "location_spec_groups": [
                {"name": "Full", "spec_names": ["tp_0_full", "tp_1_full"]},
                {"name": "Linear", "spec_names": ["tp_0_linear", "tp_1_linear"]},
            ],
        }

    def test_both_registration_orders_produce_the_same_payload(self) -> None:
        """<= 0.5.18 (v1 group + v2s) and >= 0.5.19 (v1 anchor + v2s) agree."""
        with _patched_parallel_context():
            legacy = _connector()
            legacy.register_mem_pool_host(FakeHostPoolGroup())  # ty: ignore[invalid-argument-type]
            legacy.register_mem_host_pool_v2(self._kv_pool(), PoolName.KV)
            legacy.register_mem_host_pool_v2(self._mamba_pool(), PoolName.MAMBA)

            current = _connector()
            current.register_mem_pool_host(self._kv_pool())
            current.register_mem_host_pool_v2(self._kv_pool(), PoolName.KV)
            current.register_mem_host_pool_v2(self._mamba_pool(), PoolName.MAMBA)

        legacy_payload = _register_instance_payload(legacy)
        current_payload = _register_instance_payload(current)

        self.assertEqual(legacy_payload, current_payload)
        self.assertEqual(legacy_payload, self._expected_payload())

    def test_transfer_client_uses_the_same_spec_table(self) -> None:
        with _patched_parallel_context():
            connector = _connector()
            connector.register_mem_pool_host(self._kv_pool())
            connector.register_mem_host_pool_v2(self._mamba_pool(), PoolName.MAMBA)
            connector.batch_exists(["block-0"])

        client_config = json.loads(_mock_kvcm.TransferClient.Create.call_args.args[0])
        self.assertEqual(
            client_config["location_spec_infos"],
            {
                "tp_0_full": KV_BYTES_PER_TOKEN * PAGE_SIZE,
                "tp_0_linear": MAMBA_BYTES_PER_TOKEN,
            },
        )
        self.assertEqual(client_config["block_size"], PAGE_SIZE)

    def test_group_entries_are_not_used_as_pools(self) -> None:
        """A hybrid group alone (no v2, no KV) fails loudly, not silently."""
        with _patched_parallel_context():
            connector = _connector()
            connector.register_mem_pool_host(FakeHostPoolGroup())  # ty: ignore[invalid-argument-type]

            self.assertNotIn(PoolName.KV, connector.registered_pools)
            with self.assertLogs(connector_module.__name__, level="ERROR"):
                self.assertEqual(connector.batch_exists(["block-0"]), 0)
            self.assertFalse(connector._client_ready)

    def test_flat_model_keeps_the_legacy_spec_name(self) -> None:
        """No sidecars: names stay ``tp_{rank}`` and no spec groups are sent."""
        with _patched_parallel_context():
            connector = _connector()
            connector.register_mem_pool_host(self._kv_pool())
            connector.batch_exists(["block-0"])

        payload = _register_instance_payload(connector)
        self.assertEqual(
            [spec["name"] for spec in payload["location_spec_infos"]],
            ["tp_0", "tp_1"],
        )
        # Older Managers do not know the field; it must stay absent.
        self.assertNotIn("location_spec_groups", payload)

    def test_mla_uses_rank0_spec_on_every_rank(self) -> None:
        """MLA: replicated KV, rank 0 owns every spec, kv_factor stays 1."""
        for tp_rank in (0, 1):
            with _patched_parallel_context():
                connector = _connector(tp_rank=tp_rank, is_mla_model=True)
                connector.register_mem_pool_host(self._kv_pool())
                connector.batch_exists(["block-0"])

            self.assertEqual(connector.kv_factor, 1)
            self.assertEqual(connector.location_spec_name, "tp_0")


class TestInitializationTiming(unittest.TestCase):
    """The Manager is only contacted once the pool set is final."""

    def setUp(self) -> None:
        _mock_kvcm.reset_mock()
        HiCacheKVCM._reported_pools.clear()

    def test_registration_is_deferred_to_the_first_storage_call(self) -> None:
        with _patched_parallel_context():
            connector = _connector()
            connector.register_mem_pool_host(
                FakeHostPool(bytes_per_token=KV_BYTES_PER_TOKEN, dtype=torch.bfloat16)
            )
            self.assertFalse(connector._client_ready)
            self.assertIsNone(
                _manager(connector).register_instance.call_args,
                "registration must not run from a registration hook",
            )

            connector.batch_exists(["block-0"])
            connector.batch_exists(["block-1"])
            connector.batch_get_v1(["block-0"], torch.arange(PAGE_SIZE))

        self.assertTrue(connector._client_ready)
        self.assertEqual(
            _manager(connector).register_instance.call_count,
            1,
            "register_instance must run exactly once (it is not idempotent "
            "for a changed pool set)",
        )

    def test_pool_registered_after_initialization_is_degraded(self) -> None:
        with _patched_parallel_context():
            connector = _connector()
            connector.register_mem_pool_host(
                FakeHostPool(bytes_per_token=KV_BYTES_PER_TOKEN, dtype=torch.bfloat16)
            )
            connector.batch_exists(["block-0"])

            swa_pool = FakeHostPool(bytes_per_token=256, dtype=torch.bfloat16)
            with self.assertLogs(connector_module.__name__, level="ERROR") as logs:
                connector.register_mem_host_pool_v2(swa_pool, UNMANAGED_POOL)
            # Never recorded -- ``registered_pools`` holds only pools whose
            # spec was frozen into the Manager registration -- and marked
            # unusable for the data path.
            self.assertNotIn(UNMANAGED_POOL, connector.registered_pools)
            self.assertIn(UNMANAGED_POOL, connector._late_pools)
            self.assertEqual(len(logs.records), 1)
            self.assertIn(str(UNMANAGED_POOL), logs.records[0].getMessage())

            transfer = PoolTransfer(
                name=UNMANAGED_POOL,
                host_indices=torch.arange(2),
                keys=["block-0", "block-1"],
            )
            with self.assertNoLogs(connector_module.__name__, level="ERROR"):
                connector.register_mem_host_pool_v2(swa_pool, UNMANAGED_POOL)
            self.assertEqual(
                connector.batch_get_v2([transfer]), {UNMANAGED_POOL: [False, False]}
            )
            self.assertEqual(
                connector.batch_set_v2([transfer]), {UNMANAGED_POOL: [False, False]}
            )

    def test_late_pool_reusing_a_managed_name_is_not_used(self) -> None:
        """A late pool with a managed name must not inherit that pool's spec.

        Its host indices point into its own memory, not into the memory the
        registered ``tp_*_linear`` URIs were written from, so pairing them
        would silently move and read the wrong bytes.
        """
        with _patched_parallel_context():
            connector = _connector()
            kv_pool = FakeHostPool(
                bytes_per_token=KV_BYTES_PER_TOKEN, dtype=torch.bfloat16
            )
            mamba_pool = FakeHostPool(
                bytes_per_token=MAMBA_BYTES_PER_TOKEN, dtype=torch.float32
            )
            connector.register_mem_pool_host(kv_pool)
            connector.register_mem_host_pool_v2(mamba_pool, PoolName.MAMBA)
            connector.batch_exists(["block-0"])

            late_pool = FakeHostPool(
                bytes_per_token=MAMBA_BYTES_PER_TOKEN, dtype=torch.float32
            )
            with self.assertLogs(connector_module.__name__, level="ERROR") as logs:
                connector.register_mem_host_pool_v2(late_pool, PoolName.MAMBA)
            self.assertEqual(len(logs.records), 1)
            self.assertIs(
                connector.registered_pools[PoolName.MAMBA],
                mamba_pool,
                "a late pool must not replace an already-registered one",
            )

            transfer = PoolTransfer(
                name=PoolName.MAMBA,
                host_indices=torch.arange(2),
                keys=["block-0", "block-1"],
            )
            manager = _manager(connector)
            manager.get_cache_location.return_value = {
                # Realistic shapes: two KV pages *and* the registered pool's
                # own spec (tp_0_linear), so "the spec is missing from the
                # Manager" cannot be the reason why the late pool must come
                # back as a miss.
                "locations": [
                    {
                        "location_specs": [
                            {"name": "tp_0_full", "uri": f"uri-{i}"},
                            {"name": "tp_0_linear", "uri": f"lin-{i}"},
                        ]
                    }
                    for i in range(2)
                ]
            }

            with self.assertNoLogs(connector_module.__name__, level="ERROR"):
                exists = connector.batch_exists_v2(["block-0", "block-1"], [transfer])
            self.assertEqual(exists.kv_hit_pages, 0)
            self.assertEqual(exists.extra_pool_hit_pages[PoolName.MAMBA], 0)

            manager.get_cache_location.reset_mock()
            manager.start_write_cache.reset_mock()
            with self.assertNoLogs(connector_module.__name__, level="ERROR"):
                self.assertEqual(
                    connector.batch_get_v2([transfer]),
                    {PoolName.MAMBA: [False, False]},
                )
            self.assertIsNone(
                manager.get_cache_location.call_args, "must not query the Manager"
            )
            with self.assertNoLogs(connector_module.__name__, level="ERROR"):
                self.assertEqual(
                    connector.batch_set_v2([transfer]),
                    {PoolName.MAMBA: [False, False]},
                )
            self.assertIsNone(
                manager.start_write_cache.call_args, "must not open a write session"
            )

    def test_unknown_pool_is_reported_on_the_v2_read_and_write_paths(self) -> None:
        """Never-seen pool: the data path reports it, once per pool name."""
        with _patched_parallel_context():
            connector = _connector()
            connector.register_mem_pool_host(
                FakeHostPool(bytes_per_token=KV_BYTES_PER_TOKEN, dtype=torch.bfloat16)
            )
            connector.batch_exists(["block-0"])

            draft = PoolTransfer(
                name=UNMANAGED_DRAFT,
                host_indices=torch.arange(2),
                keys=["block-0", "block-1"],
            )
            with self.assertLogs(connector_module.__name__, level="WARNING") as logs:
                self.assertEqual(
                    connector.batch_get_v2([draft]),
                    {UNMANAGED_DRAFT: [False, False]},
                )
            self.assertEqual(len(logs.records), 1)
            self.assertIn(str(UNMANAGED_DRAFT), logs.records[0].getMessage())

            with self.assertNoLogs(connector_module.__name__, level="WARNING"):
                self.assertEqual(
                    connector.batch_set_v2([draft]),
                    {UNMANAGED_DRAFT: [False, False]},
                )


class TestV2EntriesLazyInitialization(unittest.TestCase):
    """The v2 entries must trigger the deferred initialization like any other
    data-path entry, and stay conservative when there is no KV pool at all."""

    def setUp(self) -> None:
        _mock_kvcm.reset_mock()
        HiCacheKVCM._reported_pools.clear()

    def _registered_connector(self) -> HiCacheKVCM:
        connector = _connector()
        connector.register_mem_pool_host(
            FakeHostPool(bytes_per_token=KV_BYTES_PER_TOKEN, dtype=torch.bfloat16)
        )
        connector.register_mem_host_pool_v2(
            FakeHostPool(bytes_per_token=MAMBA_BYTES_PER_TOKEN, dtype=torch.float32),
            PoolName.MAMBA,
        )
        return connector

    def _transfer(self) -> PoolTransfer:
        return PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.arange(2),
            keys=["block-0", "block-1"],
        )

    def test_batch_get_v2_initializes_before_the_first_transfer(self) -> None:
        with _patched_parallel_context():
            connector = self._registered_connector()
            self.assertFalse(connector._client_ready)
            result = connector.batch_get_v2([self._transfer()])

        self.assertEqual(result, {PoolName.MAMBA: [False, False]})
        self.assertTrue(connector._client_ready)
        self.assertEqual(_manager(connector).register_instance.call_count, 1)

    def test_batch_set_v2_initializes_before_the_first_transfer(self) -> None:
        with _patched_parallel_context():
            connector = self._registered_connector()
            _manager(connector).start_write_cache.return_value = None
            self.assertFalse(connector._client_ready)
            result = connector.batch_set_v2([self._transfer()])

        self.assertEqual(result, {PoolName.MAMBA: [False, False]})
        self.assertTrue(connector._client_ready)
        self.assertEqual(_manager(connector).register_instance.call_count, 1)

    def test_batch_exists_v2_initializes_before_the_first_query(self) -> None:
        with _patched_parallel_context():
            connector = self._registered_connector()
            self.assertFalse(connector._client_ready)
            result = connector.batch_exists_v2(["block-0"], [self._transfer()])

        self.assertEqual(result.kv_hit_pages, 0)
        self.assertTrue(connector._client_ready)
        self.assertEqual(_manager(connector).register_instance.call_count, 1)

    def test_all_six_entries_stay_conservative_without_a_kv_pool(self) -> None:
        with _patched_parallel_context():
            connector = _connector()
            # Only the v0.5.18-style group arrives: no KV pool at all.
            connector.register_mem_pool_host(FakeHostPoolGroup())  # ty: ignore[invalid-argument-type]
            transfer = self._transfer()
            with self.assertLogs(connector_module.__name__, level="ERROR"):
                self.assertEqual(connector.batch_exists(["block-0"]), 0)
                self.assertEqual(
                    connector.batch_get_v1(["block-0"], torch.arange(PAGE_SIZE)),
                    [False],
                )
                self.assertEqual(
                    connector.batch_set_v1(["block-0"], torch.arange(PAGE_SIZE)),
                    [False],
                )
                self.assertEqual(
                    connector.batch_exists_v2(["block-0"], [transfer]).kv_hit_pages,
                    0,
                )
                self.assertEqual(
                    connector.batch_get_v2([transfer]),
                    {PoolName.MAMBA: [False, False]},
                )
                self.assertEqual(
                    connector.batch_set_v2([transfer]),
                    {PoolName.MAMBA: [False, False]},
                )

        self.assertFalse(connector._client_ready)
        self.assertIsNone(_manager(connector).register_instance.call_args)


class TestUnsupportedKvAnchor(unittest.TestCase):
    """A KV anchor that is not a host pool degrades once, not per call.

    sglang >= 0.5.19 anchors DeepSeek-V4 compressed KV pools on a tensorless
    ``LogicalHostPool``; no location spec can be built for that shape, and the
    operator must learn it from one clear ERROR instead of a flood.
    """

    def setUp(self) -> None:
        _mock_kvcm.reset_mock()
        HiCacheKVCM._reported_pools.clear()

    def test_unsupported_anchor_is_reported_once_then_degraded(self) -> None:
        with _patched_parallel_context():
            connector = _connector()
            anchor = FakeLogicalHostPool()
            connector.register_mem_pool_host(anchor)  # ty: ignore[invalid-argument-type]
            connector.register_mem_host_pool_v2(anchor, PoolName.KV)  # ty: ignore[invalid-argument-type]

            with self.assertLogs(connector_module.__name__, level="ERROR") as logs:
                self.assertEqual(connector.batch_exists(["block-0"]), 0)
            self.assertEqual(len(logs.records), 1)
            self.assertIn("FakeLogicalHostPool", logs.records[0].getMessage())
            self.assertFalse(connector._client_ready)
            self.assertIsNone(
                _manager(connector).register_instance.call_args,
                "an unusable anchor must not be registered with the Manager",
            )

            transfer = PoolTransfer(
                name=PoolName.MAMBA,
                host_indices=torch.arange(2),
                keys=["block-0", "block-1"],
            )
            with self.assertNoLogs(connector_module.__name__, level="ERROR"):
                self.assertEqual(connector.batch_exists(["block-1"]), 0)
                self.assertEqual(
                    connector.batch_get_v1(["block-1"], torch.arange(PAGE_SIZE)),
                    [False],
                )
                self.assertEqual(
                    connector.batch_set_v1(["block-1"], torch.arange(PAGE_SIZE)),
                    [False],
                )
                self.assertEqual(
                    connector.batch_exists_v2(["block-1"]).extra_pool_hit_pages, {}
                )
                self.assertEqual(
                    connector.batch_get_v2([transfer]),
                    {PoolName.MAMBA: [False, False]},
                )
                self.assertEqual(
                    connector.batch_set_v2([transfer]),
                    {PoolName.MAMBA: [False, False]},
                )
            self.assertFalse(connector._client_ready)
            self.assertIsNone(_manager(connector).register_instance.call_args)


class TestClose(unittest.TestCase):
    """detach calls close(); it must release resources and stay closed."""

    def setUp(self) -> None:
        _mock_kvcm.reset_mock()
        HiCacheKVCM._reported_pools.clear()

    def _initialized_connector(self) -> HiCacheKVCM:
        with _patched_parallel_context():
            connector = _connector()
            connector.register_mem_pool_host(
                FakeHostPool(bytes_per_token=KV_BYTES_PER_TOKEN, dtype=torch.bfloat16)
            )
            connector.batch_exists(["block-0"])
        self.assertTrue(connector._client_ready)
        return connector

    def test_close_racing_the_first_initialization_releases_the_client(self) -> None:
        """close() and the one-time init cannot interleave (shared lock).

        Without the lock, an init already past its ``_closed`` check would
        still create the TransferClient after close() had torn everything
        down: the connector would be closed but keep a live SDK client
        forever.  Upstream joins the storage threads before detaching, so
        this ordering is defence in depth rather than the only guard.
        """

        class _Client:
            """Stand-in for the pybind TransferClient (plain, no cycles)."""

        entered = threading.Event()
        release = threading.Event()
        clients: list = []

        def slow_register(request: Any) -> dict:
            entered.set()
            release.wait(timeout=5)
            return {"storage_configs": "[]"}

        def slow_create(*args: Any, **kwargs: Any) -> Any:
            client = _Client()
            clients.append(weakref.ref(client))
            return client

        with _patched_parallel_context():
            connector = _connector()
            connector.register_mem_pool_host(
                FakeHostPool(bytes_per_token=KV_BYTES_PER_TOKEN, dtype=torch.bfloat16)
            )
            _manager(connector).register_instance.side_effect = slow_register
            _mock_kvcm.TransferClient.Create.side_effect = slow_create
            # MagicMock.reset_mock() (the other classes' setUp) does not clear
            # side_effect, and this class runs first: put the shared mock back
            # so later tests still see the default MagicMock client.
            self.addCleanup(
                setattr, _mock_kvcm.TransferClient.Create, "side_effect", None
            )

            storage_call = threading.Thread(
                target=connector.batch_exists, args=(["block-0"],)
            )
            storage_call.start()
            self.assertTrue(entered.wait(timeout=5), "init never started")

            closer = threading.Thread(target=connector.close)
            closer.start()
            self.assertTrue(closer.is_alive(), "close() did not wait for the init")
            release.set()
            storage_call.join(timeout=5)
            closer.join(timeout=5)

        self.assertFalse(storage_call.is_alive() or closer.is_alive())
        self.assertTrue(connector._closed)
        self.assertTrue(connector._client_ready, "the init was already in flight")
        self.assertEqual(len(clients), 1)
        self.assertIsNone(
            connector.transfer_client,
            "close() must release the client created by the racing init",
        )
        self.assertIsNone(clients[0](), "the created SDK client must be freed")

    def test_close_releases_resources_and_is_idempotent(self) -> None:
        connector = self._initialized_connector()
        manager = _manager(connector)
        self.assertIsNotNone(connector.transfer_client)

        connector.close()
        connector.close()  # detach is best-effort and may run twice

        self.assertIsNone(connector.transfer_client)
        self.assertIsNone(connector.init_params)
        manager.close.assert_called()

    def test_storage_calls_after_close_fail_conservatively(self) -> None:
        connector = self._initialized_connector()
        manager = _manager(connector)
        connector.close()
        manager.register_instance.reset_mock()

        with self.assertLogs(connector_module.__name__, level="ERROR") as logs:
            self.assertEqual(connector.batch_exists(["block-1"]), 0)
            self.assertEqual(
                connector.batch_get_v1(["block-1"], torch.arange(PAGE_SIZE)), [False]
            )
            self.assertEqual(
                connector.batch_set_v1(["block-1"], torch.arange(PAGE_SIZE)), [False]
            )

        self.assertEqual(len(logs.records), 3)
        for record in logs.records:
            self.assertIn("closed", record.getMessage())
        self.assertIsNone(
            manager.register_instance.call_args, "close() must not be undone"
        )


class TestLegacyOps(unittest.TestCase):
    """<= 0.5.18 draft/MTP calls the legacy page interface."""

    def setUp(self) -> None:
        HiCacheKVCM._warned_legacy_ops.clear()

    def test_batch_get_reports_misses_once(self) -> None:
        connector = _connector()

        with self.assertLogs(connector_module.__name__, level="WARNING") as logs:
            self.assertEqual(connector.batch_get(["k0", "k1"]), [None, None])
        self.assertEqual(len(logs.records), 1)
        self.assertIn("batch_get", logs.records[0].getMessage())

        with self.assertNoLogs(connector_module.__name__, level="WARNING"):
            self.assertEqual(connector.batch_get(["k2"]), [None])

    def test_batch_set_reports_failure_once(self) -> None:
        connector = _connector()

        with self.assertLogs(connector_module.__name__, level="WARNING") as logs:
            self.assertFalse(connector.batch_set(["k0"]))
        self.assertEqual(len(logs.records), 1)
        self.assertIn("batch_set", logs.records[0].getMessage())

        with self.assertNoLogs(connector_module.__name__, level="WARNING"):
            self.assertFalse(connector.batch_set(["k1"]))


if __name__ == "__main__":
    unittest.main()
