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
  call (never before the pool set is final) and exactly once; sglang that offers
  the duck-typed ``finalize_registration`` hook instead freezes them at the end
  of the attach (draft pools included) and fails the attach when that fails,
  while a release without the hook keeps the lazy timing unchanged;
* a pool registered after that point is reported once and degraded to misses:
  it is not recorded and stays unusable, even when it reuses the name of a pool
  whose spec is already registered;
* the module-level ``_POOL_SPECS`` table holds its invariants (unique pools and
  spec suffixes, exactly one anchor), so "one more row" cannot silently alias
  two pools onto one location spec;
* the whole control plane an instance hands to the Manager and the SDK client
  -- spec names, sizes, groups, iov size, client config -- matches a literal
  golden table for every pool combination, on MLA and non-MLA models, on every
  rank;
* the registered spec set is exactly the set the ranks' data paths use, and a
  pool registers one spec per rank that writes it -- the CPU half of
  "allocated == used"; the half that needs a Manager and a run (was a location
  really written) belongs to the e2e harness.

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

import dataclasses
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

import requests
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
    HiCacheStorage,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache  # noqa: E402

from kv_cache_manager.py_connector.common.manager_client import (  # noqa: E402
    KvCacheManagerHTTPError,
    KvCacheManagerProtocolError,
)
from kv_cache_manager.py_connector.sglang import connector as connector_module  # noqa: E402
from kv_cache_manager.py_connector.sglang.connector import HiCacheKVCM  # noqa: E402

PAGE_SIZE = 64
KV_BYTES_PER_TOKEN = 512
MAMBA_BYTES_PER_TOKEN = 96
# The page size each registered pool's host object reports.  Mamba's host pool is
# page-granular (``MambaPoolHost.page_size == 1``, one slot per page), which is
# exactly why its spec size is its per-token size; every other pool counts tokens.
POOL_PAGE_SIZES = {PoolName.MAMBA: 1}
# Never used by a correct connector: reading it means pools were taken from
# HostPoolGroup.entries.
SENTINEL_BYTES_PER_TOKEN = 7
# A pool the connector does not manage: PoolName.SWA is the realistic case
# (sglang >= v0.5.12) and the plain string with the same text stands in on
# releases without the member.  A derived pool is backed under whatever name it
# was registered with, so both spellings produce the same spec and the same
# per-pool entries; neither is a pool the connector *does* know, which is the
# point -- falling back to MAMBA would re-register the pool the burst already
# registered instead of adding a third one, and on the data path it would build
# two transfers for one pool.
UNMANAGED_POOL: Any = getattr(PoolName, "SWA", "swa")
# Same idea for the never-registered pool of the read/write-path test; DRAFT
# exists since v0.5.13, so its text is the stand-in before that.
UNMANAGED_DRAFT: Any = getattr(PoolName, "DRAFT", "draft")


class FakeHostPool(HostKVCache):
    """A sglang host pool without CUDA: only what the registration path reads.

    The ABC's transfer hooks need real device pools, so they are stubbed out.
    Subclassing for real keeps the connector's ``isinstance`` check (and with
    it the "a group is not a host pool" distinction) meaningful.  ``page_size``
    is the pool's own, which is what its spec size is counted from: a Mamba host
    pool keeps one slot per page (``page_size == 1``), every other pool here
    counts ``PAGE_SIZE`` tokens.
    """

    def __init__(
        self,
        *,
        bytes_per_token: int,
        dtype: torch.dtype,
        page_size: int = PAGE_SIZE,
    ) -> None:
        self.dtype = dtype
        self.bytes_per_token = bytes_per_token
        self.page_size = page_size

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


def _connector_with_pools(
    pools: tuple, *, is_mla: bool, tp_rank: int, tp_size: int = 2
) -> HiCacheKVCM:
    """One rank of one pool combination, registered: the spec set is frozen."""
    with _patched_parallel_context():
        connector = _connector(tp_rank=tp_rank, tp_size=tp_size, is_mla_model=is_mla)
        connector.register_mem_pool_host(
            FakeHostPool(bytes_per_token=KV_BYTES_PER_TOKEN, dtype=torch.bfloat16)
        )
        for name, bytes_per_token in pools:
            connector.register_mem_host_pool_v2(
                FakeHostPool(
                    bytes_per_token=bytes_per_token,
                    dtype=torch.float32,
                    page_size=POOL_PAGE_SIZES.get(name, PAGE_SIZE),
                ),
                name,
            )
        connector.batch_exists(["block-0"])
    return connector


class TestRegistrationOrders(unittest.TestCase):
    """Both upstream orders must yield the same instance registration."""

    def setUp(self) -> None:
        _mock_kvcm.reset_mock()
        HiCacheKVCM._reported_pools.clear()

    def _kv_pool(self) -> FakeHostPool:
        return FakeHostPool(bytes_per_token=KV_BYTES_PER_TOKEN, dtype=torch.bfloat16)

    def _mamba_pool(self) -> FakeHostPool:
        return FakeHostPool(
            bytes_per_token=MAMBA_BYTES_PER_TOKEN, dtype=torch.float32, page_size=1
        )

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
        """MLA: the replicated KV is rank 0's, a rank-sharded Mamba is not.

        sglang writes a replicated MLA KV pool from rank 0 alone, and hands
        the rank-sharded Mamba/KDA pool of a hybrid stack to every rank, each
        owning its own slice.  The two pools therefore disagree on the rank
        whose spec a rank uses, and both stay as chosen here (see
        ``test_tp_init_sync.py`` for the write protocol that goes with it).
        """
        for tp_rank in (0, 1):
            with _patched_parallel_context():
                connector = _connector(tp_rank=tp_rank, is_mla_model=True)
                connector.register_mem_pool_host(self._kv_pool())
                connector.register_mem_host_pool_v2(self._mamba_pool(), PoolName.MAMBA)
                connector.batch_exists(["block-0"])

            self.assertEqual(connector.kv_factor, 1)
            # Hybrid naming adds the suffix once a sidecar exists; the rank is
            # what this test is about.
            self.assertEqual(connector.location_spec_name, "tp_0_full")
            self.assertEqual(
                connector._get_extra_pool_spec_name(PoolName.MAMBA),
                f"tp_{tp_rank}_linear",
                "each rank reads and writes its own Mamba slice",
            )


##################################################
# The pool spec table
##################################################

# Bytes of one block of the fake pools below: KV and Indexer are token-granular
# (times the page size), Mamba is page-granular (one slot per block).
KV_SPEC_BYTES = KV_BYTES_PER_TOKEN * PAGE_SIZE
MAMBA_SPEC_BYTES = MAMBA_BYTES_PER_TOKEN
INDEXER_BYTES_PER_TOKEN = 40
INDEXER_SPEC_BYTES = INDEXER_BYTES_PER_TOKEN * PAGE_SIZE
# A DSA indexer cache can hold more bytes per block than the KV cache (its
# quantized index adds up over layers), which is the case that pins the iov size
# to the largest pool instead of to the KV pool.
BIG_INDEXER_BYTES_PER_TOKEN = 4096
BIG_INDEXER_SPEC_BYTES = BIG_INDEXER_BYTES_PER_TOKEN * PAGE_SIZE

# The negative cases below build their tables from this minimal valid one, not
# from ``_POOL_SPECS``: a later PR adding a row to the real table must not change
# which invariant rejects them (that would hide the one under test).
_BASE_TABLE = (connector_module._KV_SPEC, connector_module._MAMBA_SPEC)
# A pool name no table can contain -- deliberately not a sglang enum member.
_FRESH_POOL = "pool-that-is-not-in-the-table"


def _check_pool_spec_table(table: tuple) -> None:
    """Run the table guard that ``connector`` also runs at import time."""
    connector_module._validate_pool_specs(table)


class TestPoolSpecTable(unittest.TestCase):
    """The spec table is the connector's whole pool knowledge.

    Every later PR adds rows to it, so its invariants are pinned here -- and
    pinned *negatively* too, on the row shapes that would otherwise pass every
    other test in this directory while corrupting data.
    """

    def test_the_table_holds_its_invariants(self) -> None:
        _check_pool_spec_table(connector_module._POOL_SPECS)

    def test_a_sidecar_reusing_a_suffix_is_rejected(self) -> None:
        """P4 writing ``suffix="_linear"`` for SWA must fail here, not at runtime."""
        colliding = connector_module._PoolSpec(
            pool=_FRESH_POOL,
            group="Swa",
            suffix=connector_module._MAMBA_SPEC.suffix,
        )
        with self.assertRaisesRegex(ValueError, "same spec suffix"):
            _check_pool_spec_table((*_BASE_TABLE, colliding))

    def test_a_repeated_pool_row_is_rejected(self) -> None:
        # The suffix differs, so only the pool-uniqueness check can catch it.
        duplicate = dataclasses.replace(connector_module._MAMBA_SPEC, suffix="_kda")
        with self.assertRaisesRegex(ValueError, "more than one row"):
            _check_pool_spec_table((*_BASE_TABLE, duplicate))

    def test_a_second_anchor_is_rejected(self) -> None:
        # A pool, suffix and group nobody else uses, so only the anchor count
        # can reject this table.
        second_anchor = connector_module._PoolSpec(
            pool=_FRESH_POOL, group="Swa", suffix="_swa", anchor=True
        )
        with self.assertRaisesRegex(ValueError, "exactly one anchor"):
            _check_pool_spec_table((*_BASE_TABLE, second_anchor))

    def test_a_missing_anchor_is_rejected(self) -> None:
        without_anchor = tuple(spec for spec in _BASE_TABLE if not spec.anchor)
        with self.assertRaisesRegex(ValueError, "exactly one anchor"):
            _check_pool_spec_table(without_anchor)

    def test_an_anchor_that_is_not_the_kv_pool_is_rejected(self) -> None:
        wrong_anchor = dataclasses.replace(connector_module._KV_SPEC, pool=_FRESH_POOL)
        with self.assertRaisesRegex(ValueError, "the anchor is the KV pool"):
            _check_pool_spec_table((wrong_anchor, connector_module._MAMBA_SPEC))

    def test_a_row_without_a_suffix_is_rejected(self) -> None:
        # Deliberately invalid: the guard must catch what the type checker
        # already refuses.
        no_suffix = connector_module._PoolSpec(
            pool=_FRESH_POOL,
            group="Swa",
            suffix=None,  # ty: ignore[invalid-argument-type]
        )
        with self.assertRaisesRegex(ValueError, "no suffix"):
            _check_pool_spec_table((*_BASE_TABLE, no_suffix))

    def test_a_sidecar_sharing_the_anchor_group_is_rejected(self) -> None:
        same_group = dataclasses.replace(
            connector_module._MAMBA_SPEC, group=connector_module._KV_SPEC.group
        )
        with self.assertRaisesRegex(ValueError, "share a spec group"):
            _check_pool_spec_table((connector_module._KV_SPEC, same_group))

    def test_two_sidecars_sharing_a_group_is_rejected(self) -> None:
        """P4 writing ``group="Linear"`` for SWA: the Manager skips its specs."""
        same_group = connector_module._PoolSpec(
            pool=_FRESH_POOL, group=connector_module._MAMBA_SPEC.group, suffix="_swa"
        )
        with self.assertRaisesRegex(ValueError, "share a spec group"):
            _check_pool_spec_table((*_BASE_TABLE, same_group))

    def test_a_sidecar_with_an_empty_group_is_rejected(self) -> None:
        """An empty group name means "no constraint" to the Manager, so the
        pool would be silently dropped -- it must be rejected at the table."""
        empty = connector_module._PoolSpec(pool=_FRESH_POOL, group="", suffix="_swa")
        with self.assertRaisesRegex(ValueError, "empty spec group"):
            _check_pool_spec_table((*_BASE_TABLE, empty))

    def test_the_anchor_with_an_empty_group_is_rejected(self) -> None:
        empty = dataclasses.replace(connector_module._KV_SPEC, group="")
        with self.assertRaisesRegex(ValueError, "empty spec group"):
            _check_pool_spec_table((empty, connector_module._MAMBA_SPEC))


##################################################
# Control plane golden
##################################################

# What one instance hands to the Manager and to the SDK client, per pool
# combination: the register payload's (name, size) pairs in order, its
# (group, spec_names) pairs in order or None when the field stays absent, and
# the iov size.  Literal on purpose: P2/P3/P4 add pools and change these
# fields, and this is the table that makes "only added, nothing changed"
# checkable without a GPU or a Manager.
#
# This table is the non-MLA layout: every pool registers one spec per TP rank.
# An MLA model registers a replicated pool for rank 0 alone -- ``_MLA_COMBO_GOLDEN``
# carries the differing fields of the same combinations.
#
# What these two tables pin is the exact names, order, sizes and groups (a
# rename or a reorder is a wire change).  What they do not pin is the rule that
# decides how many specs a pool registers: that rule is asserted from the write
# protocol in ``TestAllocationMatchesUsage``, so a new row or a tp size these
# tables do not reach needs no edit here.
_COMBO_GOLDEN: dict[tuple, tuple] = {
    (): (
        [("tp_0", KV_SPEC_BYTES), ("tp_1", KV_SPEC_BYTES)],
        None,
        KV_SPEC_BYTES * 1024,
    ),
    ((PoolName.MAMBA, MAMBA_BYTES_PER_TOKEN),): (
        [
            ("tp_0_full", KV_SPEC_BYTES),
            ("tp_1_full", KV_SPEC_BYTES),
            ("tp_0_linear", MAMBA_SPEC_BYTES),
            ("tp_1_linear", MAMBA_SPEC_BYTES),
        ],
        [
            ("Full", ["tp_0_full", "tp_1_full"]),
            ("Linear", ["tp_0_linear", "tp_1_linear"]),
        ],
        KV_SPEC_BYTES * 1024,
    ),
    ((PoolName.INDEXER, INDEXER_BYTES_PER_TOKEN),): (
        [
            ("tp_0_full", KV_SPEC_BYTES),
            ("tp_1_full", KV_SPEC_BYTES),
            ("tp_0_indexer", INDEXER_SPEC_BYTES),
            ("tp_1_indexer", INDEXER_SPEC_BYTES),
        ],
        [
            ("Full", ["tp_0_full", "tp_1_full"]),
            ("Indexer", ["tp_0_indexer", "tp_1_indexer"]),
        ],
        KV_SPEC_BYTES * 1024,
    ),
    (
        (PoolName.MAMBA, MAMBA_BYTES_PER_TOKEN),
        (PoolName.INDEXER, INDEXER_BYTES_PER_TOKEN),
    ): (
        [
            ("tp_0_full", KV_SPEC_BYTES),
            ("tp_1_full", KV_SPEC_BYTES),
            ("tp_0_linear", MAMBA_SPEC_BYTES),
            ("tp_1_linear", MAMBA_SPEC_BYTES),
            ("tp_0_indexer", INDEXER_SPEC_BYTES),
            ("tp_1_indexer", INDEXER_SPEC_BYTES),
        ],
        [
            ("Full", ["tp_0_full", "tp_1_full"]),
            ("Linear", ["tp_0_linear", "tp_1_linear"]),
            ("Indexer", ["tp_0_indexer", "tp_1_indexer"]),
        ],
        KV_SPEC_BYTES * 1024,
    ),
    ((PoolName.INDEXER, BIG_INDEXER_BYTES_PER_TOKEN),): (
        [
            ("tp_0_full", KV_SPEC_BYTES),
            ("tp_1_full", KV_SPEC_BYTES),
            ("tp_0_indexer", BIG_INDEXER_SPEC_BYTES),
            ("tp_1_indexer", BIG_INDEXER_SPEC_BYTES),
        ],
        [
            ("Full", ["tp_0_full", "tp_1_full"]),
            ("Indexer", ["tp_0_indexer", "tp_1_indexer"]),
        ],
        BIG_INDEXER_SPEC_BYTES * 1024,
    ),
}

# The same combinations on an MLA model: only (infos, groups) differ, so this
# table carries just those two -- the iov size is sized from one block of the
# table pools and does not depend on how many specs are registered.  A
# replicated pool (KV, Indexer) registers rank 0's spec alone and its group
# names that one; the sharded Mamba state keeps one spec per rank.
_MLA_COMBO_GOLDEN: dict[tuple, tuple] = {
    (): (
        [("tp_0", KV_SPEC_BYTES)],
        None,
    ),
    ((PoolName.MAMBA, MAMBA_BYTES_PER_TOKEN),): (
        [
            ("tp_0_full", KV_SPEC_BYTES),
            ("tp_0_linear", MAMBA_SPEC_BYTES),
            ("tp_1_linear", MAMBA_SPEC_BYTES),
        ],
        [
            ("Full", ["tp_0_full"]),
            ("Linear", ["tp_0_linear", "tp_1_linear"]),
        ],
    ),
    ((PoolName.INDEXER, INDEXER_BYTES_PER_TOKEN),): (
        [
            ("tp_0_full", KV_SPEC_BYTES),
            ("tp_0_indexer", INDEXER_SPEC_BYTES),
        ],
        [
            ("Full", ["tp_0_full"]),
            ("Indexer", ["tp_0_indexer"]),
        ],
    ),
    (
        (PoolName.MAMBA, MAMBA_BYTES_PER_TOKEN),
        (PoolName.INDEXER, INDEXER_BYTES_PER_TOKEN),
    ): (
        [
            ("tp_0_full", KV_SPEC_BYTES),
            ("tp_0_linear", MAMBA_SPEC_BYTES),
            ("tp_1_linear", MAMBA_SPEC_BYTES),
            ("tp_0_indexer", INDEXER_SPEC_BYTES),
        ],
        [
            ("Full", ["tp_0_full"]),
            ("Linear", ["tp_0_linear", "tp_1_linear"]),
            ("Indexer", ["tp_0_indexer"]),
        ],
    ),
    ((PoolName.INDEXER, BIG_INDEXER_BYTES_PER_TOKEN),): (
        [
            ("tp_0_full", KV_SPEC_BYTES),
            ("tp_0_indexer", BIG_INDEXER_SPEC_BYTES),
        ],
        [
            ("Full", ["tp_0_full"]),
            ("Indexer", ["tp_0_indexer"]),
        ],
    ),
}

# The SDK client's spec map, i.e. the specs this rank reads and writes: the
# replicated pools (KV, Indexer) are rank 0's on an MLA model, the sharded one
# (Mamba) is always this rank's.  The KV spec leads the map.
_OWN_SPECS: dict[tuple, dict[tuple[bool, int], dict[str, int]]] = {
    (): {
        (False, 0): {"tp_0": KV_SPEC_BYTES},
        (False, 1): {"tp_1": KV_SPEC_BYTES},
        (True, 0): {"tp_0": KV_SPEC_BYTES},
        (True, 1): {"tp_0": KV_SPEC_BYTES},
    },
    ((PoolName.MAMBA, MAMBA_BYTES_PER_TOKEN),): {
        (False, 0): {"tp_0_full": KV_SPEC_BYTES, "tp_0_linear": MAMBA_SPEC_BYTES},
        (False, 1): {"tp_1_full": KV_SPEC_BYTES, "tp_1_linear": MAMBA_SPEC_BYTES},
        (True, 0): {"tp_0_full": KV_SPEC_BYTES, "tp_0_linear": MAMBA_SPEC_BYTES},
        (True, 1): {"tp_0_full": KV_SPEC_BYTES, "tp_1_linear": MAMBA_SPEC_BYTES},
    },
    ((PoolName.INDEXER, INDEXER_BYTES_PER_TOKEN),): {
        (False, 0): {"tp_0_full": KV_SPEC_BYTES, "tp_0_indexer": INDEXER_SPEC_BYTES},
        (False, 1): {"tp_1_full": KV_SPEC_BYTES, "tp_1_indexer": INDEXER_SPEC_BYTES},
        (True, 0): {"tp_0_full": KV_SPEC_BYTES, "tp_0_indexer": INDEXER_SPEC_BYTES},
        (True, 1): {"tp_0_full": KV_SPEC_BYTES, "tp_0_indexer": INDEXER_SPEC_BYTES},
    },
    (
        (PoolName.MAMBA, MAMBA_BYTES_PER_TOKEN),
        (PoolName.INDEXER, INDEXER_BYTES_PER_TOKEN),
    ): {
        (False, 0): {
            "tp_0_full": KV_SPEC_BYTES,
            "tp_0_linear": MAMBA_SPEC_BYTES,
            "tp_0_indexer": INDEXER_SPEC_BYTES,
        },
        (False, 1): {
            "tp_1_full": KV_SPEC_BYTES,
            "tp_1_linear": MAMBA_SPEC_BYTES,
            "tp_1_indexer": INDEXER_SPEC_BYTES,
        },
        (True, 0): {
            "tp_0_full": KV_SPEC_BYTES,
            "tp_0_linear": MAMBA_SPEC_BYTES,
            "tp_0_indexer": INDEXER_SPEC_BYTES,
        },
        (True, 1): {
            "tp_0_full": KV_SPEC_BYTES,
            "tp_1_linear": MAMBA_SPEC_BYTES,
            "tp_0_indexer": INDEXER_SPEC_BYTES,
        },
    },
    ((PoolName.INDEXER, BIG_INDEXER_BYTES_PER_TOKEN),): {
        (False, 0): {
            "tp_0_full": KV_SPEC_BYTES,
            "tp_0_indexer": BIG_INDEXER_SPEC_BYTES,
        },
        (False, 1): {
            "tp_1_full": KV_SPEC_BYTES,
            "tp_1_indexer": BIG_INDEXER_SPEC_BYTES,
        },
        (True, 0): {"tp_0_full": KV_SPEC_BYTES, "tp_0_indexer": BIG_INDEXER_SPEC_BYTES},
        (True, 1): {"tp_0_full": KV_SPEC_BYTES, "tp_0_indexer": BIG_INDEXER_SPEC_BYTES},
    },
}


def _expected_client_config(spec_infos: dict) -> str:
    """The SDK client config string of a control-plane instance.

    Only the spec map varies with the pool combination and the rank; the rest
    is what ``_connector()`` configures (the SDK defaults of an instance that
    sets none of the ``sdk_*`` knobs).
    """
    return json.dumps(
        {
            "instance_group": "default",
            "instance_id": "registration-test",
            "block_size": PAGE_SIZE,
            "sdk_config": {
                "thread_num": 4,
                "queue_size": 1000,
                "sdk_backend_configs": [],
                "timeout_config": {"get_timeout_ms": 15000, "put_timeout_ms": 15000},
            },
            "location_spec_infos": spec_infos,
        }
    )


def _golden_row(pools: tuple, is_mla: bool) -> tuple:
    """(spec infos, spec groups, iov size) an instance of that combination sends.

    A combination without an MLA row raises here on purpose: the two tables have
    to cover the same combinations.
    """
    infos, groups, iov_size = _COMBO_GOLDEN[pools]
    if not is_mla:
        return infos, groups, iov_size
    mla_infos, mla_groups = _MLA_COMBO_GOLDEN[pools]
    return mla_infos, mla_groups, iov_size


class TestControlPlaneGolden(unittest.TestCase):
    """The control plane of an instance, pinned for every pool combination.

    The register payload and the SDK client config are what a Manager and the
    data path are built from; P2/P3/P4 all change them, so their exact content
    is asserted here instead of being compared by hand between branches.
    """

    def setUp(self) -> None:
        _mock_kvcm.reset_mock()
        HiCacheKVCM._reported_pools.clear()

    def _control_plane(
        self, pools: tuple, *, is_mla: bool, tp_rank: int, tp_size: int = 2
    ) -> dict:
        """The control-plane facts one rank of one pool combination hands out."""
        connector = _connector_with_pools(
            pools, is_mla=is_mla, tp_rank=tp_rank, tp_size=tp_size
        )

        payload = _register_instance_payload(connector)
        return {
            "location_spec_infos": payload["location_spec_infos"],
            "location_spec_groups": payload.get("location_spec_groups"),
            "iov_size": connector.iov_size,
            "client_config": _mock_kvcm.TransferClient.Create.call_args.args[0],
            "self_location_spec_name": connector.init_params.self_location_spec_name,
            "location_spec_name": connector.location_spec_name,
            "kv_factor": connector.kv_factor,
        }

    def test_control_plane_is_unchanged_per_pool_combination(self) -> None:
        for pools in _COMBO_GOLDEN:
            for is_mla in (False, True):
                infos, groups, iov_size = _golden_row(pools, is_mla)
                for tp_rank in (0, 1):
                    scenario = f"pools={pools} mla={is_mla} tp_rank={tp_rank}"
                    with self.subTest(scenario=scenario):
                        control_plane = self._control_plane(
                            pools, is_mla=is_mla, tp_rank=tp_rank
                        )
                        self.assertEqual(
                            [
                                (spec["name"], spec["size"])
                                for spec in control_plane["location_spec_infos"]
                            ],
                            infos,
                            f"{scenario}: register payload specs, in order",
                        )
                        if groups is None:
                            # Older Managers do not know the field: a flat
                            # stack must not send it at all, not even empty.
                            self.assertIsNone(
                                control_plane["location_spec_groups"],
                                f"{scenario}: a flat stack must not send spec groups",
                            )
                        else:
                            self.assertEqual(
                                [
                                    (group["name"], group["spec_names"])
                                    for group in control_plane["location_spec_groups"]
                                    or []
                                ],
                                groups,
                                f"{scenario}: register payload spec groups, in order",
                            )
                        self.assertEqual(control_plane["iov_size"], iov_size, scenario)

                        own = _OWN_SPECS[pools][(is_mla, tp_rank)]
                        self.assertEqual(
                            control_plane["client_config"],
                            _expected_client_config(own),
                            f"{scenario}: SDK client config",
                        )
                        # The KV spec leads ``own`` and is the instance's own.
                        self.assertEqual(
                            control_plane["self_location_spec_name"], next(iter(own))
                        )
                        self.assertEqual(
                            control_plane["location_spec_name"], next(iter(own))
                        )
                        self.assertEqual(
                            control_plane["kv_factor"], 1 if is_mla else 2, scenario
                        )

    def test_the_spec_set_follows_the_rank_count(self) -> None:
        """Only a replicated pool of an MLA model registers rank 0's spec alone.

        A non-MLA model shards its KV state across the ranks, so a rank's spec
        that went missing would leave that rank reading rank 0's shard.  Both
        directions are pinned at tp=4, a rank count the golden table above does
        not reach, and rank 3 is a rank neither table looks at either.
        """
        pools = ((PoolName.MAMBA, MAMBA_BYTES_PER_TOKEN),)
        full = [f"tp_{rank}_full" for rank in range(4)]
        linear = [f"tp_{rank}_linear" for rank in range(4)]

        dense = self._control_plane(pools, is_mla=False, tp_rank=3, tp_size=4)
        self.assertEqual(
            [spec["name"] for spec in dense["location_spec_infos"]],
            full + linear,
            "a non-MLA model keeps one spec per rank",
        )
        self.assertEqual(
            dense["location_spec_groups"],
            [
                {"name": "Full", "spec_names": full},
                {"name": "Linear", "spec_names": linear},
            ],
        )
        self.assertEqual(dense["location_spec_name"], "tp_3_full")

        mla = self._control_plane(pools, is_mla=True, tp_rank=3, tp_size=4)
        self.assertEqual(
            [spec["name"] for spec in mla["location_spec_infos"]],
            ["tp_0_full"] + linear,
            "an MLA model registers rank 0's spec for a replicated pool only",
        )
        self.assertEqual(
            mla["location_spec_groups"],
            [
                {"name": "Full", "spec_names": ["tp_0_full"]},
                {"name": "Linear", "spec_names": linear},
            ],
        )
        self.assertEqual(mla["location_spec_name"], "tp_0_full")

    def test_single_rank_flat_model_keeps_the_legacy_spec_name(self) -> None:
        control_plane = self._control_plane((), is_mla=False, tp_rank=0, tp_size=1)

        self.assertEqual(
            [
                (spec["name"], spec["size"])
                for spec in control_plane["location_spec_infos"]
            ],
            [("tp_0", KV_SPEC_BYTES)],
        )
        self.assertIsNone(
            control_plane["location_spec_groups"],
            "a flat stack must not send the field older Managers reject",
        )
        self.assertEqual(control_plane["location_spec_name"], "tp_0")


##################################################
# Allocation == usage
##################################################

# Every pool combination the allocation invariant is held for: flat, a hybrid
# stack with each table sidecar, both sidecars, a pool the table derives, and a
# derived pool beside a sharded one (the anchor's hybrid naming then keys off a
# sidecar no row knows).
_ALLOCATION_COMBOS: tuple = (
    (),
    ((UNMANAGED_POOL, 256),),
    ((PoolName.MAMBA, MAMBA_BYTES_PER_TOKEN),),
    ((PoolName.INDEXER, INDEXER_BYTES_PER_TOKEN),),
    (
        (PoolName.MAMBA, MAMBA_BYTES_PER_TOKEN),
        (PoolName.INDEXER, INDEXER_BYTES_PER_TOKEN),
    ),
    ((PoolName.MAMBA, MAMBA_BYTES_PER_TOKEN), (UNMANAGED_POOL, 256)),
)

# The pools sglang hands to every TP rank, each writing its own slice: written
# down here so the table is checked against this rather than believed.  Every
# other pool is written from rank 0 alone on an MLA model, which is the same
# statement ``_PoolSpec.shared_across_ranks`` carries for the data path.
_SHARDED_POOLS = frozenset({PoolName.MAMBA})


class TestAllocationMatchesUsage(unittest.TestCase):
    """The connector's half of "alloc == used": what it registers is what its
    ranks use.

    Both sides are values the connector produces itself, so this is a
    consistency check, not an observation: whether a location is really written
    is the e2e harness's half of the same invariant (``kvcme2e/alloc_ledger.py``,
    check ``alloc==used``), and nothing on CPU can see it.
    """

    def setUp(self) -> None:
        _mock_kvcm.reset_mock()
        HiCacheKVCM._reported_pools.clear()

    def _ranks(self, pools: tuple, *, is_mla: bool, tp_size: int) -> list:
        """Every rank of one instance, each through the real initialization."""
        return [
            _connector_with_pools(pools, is_mla=is_mla, tp_rank=rank, tp_size=tp_size)
            for rank in range(tp_size)
        ]

    def _in_use(self, connector: HiCacheKVCM) -> dict:
        """The spec each pool's data path is wired to on this rank."""
        return {
            name: (
                connector.location_spec_name
                if row.spec.anchor
                else connector._get_extra_pool_spec_name(name)
            )
            for name, row in connector.pool_specs.items()
        }

    def test_every_rank_registers_the_same_layout(self) -> None:
        """The Manager compares the spec infos against the copy it stored.

        Element by element, so a rank that registers a different layout fails
        its own registration -- one instance has exactly one spec set.
        """
        for pools in _ALLOCATION_COMBOS:
            for is_mla in (False, True):
                for tp_size in (1, 2, 4):
                    scenario = f"pools={pools} mla={is_mla} tp_size={tp_size}"
                    with self.subTest(scenario=scenario):
                        ranks = self._ranks(pools, is_mla=is_mla, tp_size=tp_size)
                        registered = [
                            (spec["name"], spec["size"])
                            for spec in ranks[0].location_spec_infos
                        ]
                        for rank, connector in enumerate(ranks[1:], start=1):
                            self.assertEqual(
                                [
                                    (spec["name"], spec["size"])
                                    for spec in connector.location_spec_infos
                                ],
                                registered,
                                f"{scenario}: rank {rank} registers a different "
                                "layout than rank 0",
                            )

    def test_every_registered_spec_is_owned_by_a_rank(self) -> None:
        """The registered set is the set the ranks' data paths use, and no more.

        A registered spec no rank is wired to is a location nobody writes, and
        the Manager allocates a general area per (key, spec) of the requested
        group: capacity reserved for nothing.  The data path cannot notice -- it
        writes the right bytes into the spec it does own, and every read goes
        through that one.
        """
        for pools in _ALLOCATION_COMBOS:
            for is_mla in (False, True):
                for tp_size in (1, 2, 4):
                    scenario = f"pools={pools} mla={is_mla} tp_size={tp_size}"
                    with self.subTest(scenario=scenario):
                        ranks = self._ranks(pools, is_mla=is_mla, tp_size=tp_size)
                        # A pool that registers nothing is not backed at all,
                        # which is a different failure; hold the pool set first
                        # so the equality below cannot pass by dropping one.
                        self.assertEqual(
                            set(ranks[0].pool_specs),
                            {PoolName.KV} | {name for name, _ in pools},
                            f"{scenario}: the backed pools",
                        )
                        registered = {
                            spec["name"] for spec in ranks[0].location_spec_infos
                        }
                        used = {
                            spec
                            for connector in ranks
                            for spec in self._in_use(connector).values()
                        }
                        self.assertEqual(
                            sorted(registered),
                            sorted(used),
                            f"{scenario}: registered but used by no rank: "
                            f"{sorted(registered - used)}; used by a rank but not "
                            f"registered: {sorted(used - registered)}",
                        )

    def test_registered_spec_count_follows_the_write_protocol(self) -> None:
        """One spec per writing rank -- and the one accepted exception, named.

        A sharded pool keeps one spec per TP rank whatever the engine version
        hands it to (see ``_RegisteredPool``), so on a sglang that calls rank 0
        alone the peer specs are allocated and never written.  That difference
        is asserted here as such instead of being read as the rule; every other
        difference between the registered count and the number of writing ranks
        is a defect.
        """
        sharded_written_by_all = connector_module._sidecars_are_written_by_every_rank()
        for pools in _ALLOCATION_COMBOS:
            for is_mla in (False, True):
                for tp_size in (1, 2, 4):
                    for tp_rank, connector in enumerate(
                        self._ranks(pools, is_mla=is_mla, tp_size=tp_size)
                    ):
                        for name, row in connector.pool_specs.items():
                            scenario = (
                                f"pools={pools} mla={is_mla} tp_size={tp_size} "
                                f"tp_rank={tp_rank} pool={name}"
                            )
                            with self.subTest(scenario=scenario):
                                sharded = name in _SHARDED_POOLS
                                self.assertEqual(
                                    row.spec.shared_across_ranks,
                                    not sharded,
                                    f"{scenario}: the row's sharing mode",
                                )
                                self.assertEqual(
                                    len(row.names),
                                    tp_size if sharded or not is_mla else 1,
                                    f"{scenario}: registered spec count",
                                )
                                # The write path, not the registration, says how
                                # many ranks really write the pool.
                                writers = (
                                    tp_size
                                    if connector._transfer_is_written_by_every_rank(
                                        PoolTransfer(name=name)
                                    )
                                    else 1
                                )
                                # The one accepted difference (see the test
                                # docstring): a sharded pool keeps one spec per
                                # rank whatever the engine version hands it to.
                                stale_peers = (
                                    is_mla
                                    and sharded
                                    and not sharded_written_by_all
                                    and tp_size > 1
                                )
                                self.assertEqual(
                                    len(row.names) == writers,
                                    not stale_peers,
                                    f"{scenario}: {len(row.names)} spec(s) registered "
                                    f"for {writers} writing rank(s)",
                                )
                                # One spec for every rank means each rank has to
                                # read its own; a single spec is rank 0's.
                                self.assertEqual(
                                    row.own_rank,
                                    0 if len(row.names) == 1 else tp_rank,
                                    f"{scenario}: the spec this rank uses",
                                )


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
                bytes_per_token=MAMBA_BYTES_PER_TOKEN, dtype=torch.float32, page_size=1
            )
            connector.register_mem_pool_host(kv_pool)
            connector.register_mem_host_pool_v2(mamba_pool, PoolName.MAMBA)
            connector.batch_exists(["block-0"])

            late_pool = FakeHostPool(
                bytes_per_token=MAMBA_BYTES_PER_TOKEN, dtype=torch.float32, page_size=1
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


class TestFinalizeRegistration(unittest.TestCase):
    """sglang that reports "registration done" freezes the specs right there.

    The hook is duck-typed: sglang that has it calls ``finalize_registration``
    after the last pool of the attach (draft pools included), and a failure
    there must fail the attach instead of leaving every storage call to degrade
    in a background thread.  Upstream sglang has no such callback, so it keeps
    freezing at the first storage call -- the last test here pins that the
    coexistence changes nothing for it.
    """

    def setUp(self) -> None:
        _mock_kvcm.reset_mock()
        HiCacheKVCM._reported_pools.clear()

    def _kv_pool(self) -> FakeHostPool:
        return FakeHostPool(bytes_per_token=KV_BYTES_PER_TOKEN, dtype=torch.bfloat16)

    def _mamba_pool(self) -> FakeHostPool:
        return FakeHostPool(
            bytes_per_token=MAMBA_BYTES_PER_TOKEN, dtype=torch.float32, page_size=1
        )

    def _burst_connector(self) -> HiCacheKVCM:
        """A connector that ran sglang's registration burst, before any I/O."""
        connector = _connector()
        connector.register_mem_pool_host(self._kv_pool())
        connector.register_mem_host_pool_v2(self._mamba_pool(), PoolName.MAMBA)
        return connector

    def test_a_second_instance_still_reports_its_missing_sdk_client(self) -> None:
        """The report names an instance, so the dedup key must carry it.

        sglang can detach and re-attach the backend with a different
        ``instance_id`` at runtime; a process-wide key would swallow the second
        instance's only diagnosis.
        """
        create = _mock_kvcm.TransferClient.Create
        create.return_value = None
        self.addCleanup(setattr, create, "return_value", MagicMock())

        with _patched_parallel_context():
            for instance_id in ("instance-a", "instance-b"):
                connector = _connector(
                    extra_config={
                        "manager_uri": "http://127.0.0.1:6382",
                        "instance_group": "default",
                        "instance_id": instance_id,
                    }
                )
                connector.register_mem_pool_host(self._kv_pool())
                with self.assertLogs(connector_module.__name__, level="ERROR") as logs:
                    with self.assertRaises(connector_module._UnsupportedBackendError):
                        connector.finalize_registration()
                self.assertTrue(
                    any(instance_id in r.getMessage() for r in logs.records),
                    f"{instance_id} was not named in its own report",
                )

    def test_the_hook_is_not_part_of_the_upstream_interface(self) -> None:
        """Duck-typed on purpose: no upstream release declares this method.

        The connector only *implements* the hook; sglang that knows it finds it
        with ``getattr``.  If this fails, upstream has declared a
        ``finalize_registration`` of its own and its contract has to be
        re-read before the connector claims to satisfy it.
        """
        self.assertFalse(hasattr(HiCacheStorage, "finalize_registration"))
        self.assertTrue(callable(HiCacheKVCM.finalize_registration))

    def test_finalize_registration_builds_the_client_once(self) -> None:
        """The hook registers and creates the client; a repeat does neither."""
        with _patched_parallel_context():
            connector = self._burst_connector()
            self.assertFalse(connector._client_ready)
            self.assertIsNone(
                _manager(connector).register_instance.call_args,
                "a registration hook must not register the instance by itself",
            )

            connector.finalize_registration()
            self.assertTrue(connector._client_ready)
            self.assertIsNotNone(connector.transfer_client)

            connector.finalize_registration()
            self.assertFalse(
                connector._tp_init_agreed,
                "the hook is not a collective: it must not claim the TP group "
                "agreed on initialization",
            )

        self.assertEqual(_manager(connector).register_instance.call_count, 1)
        self.assertEqual(_mock_kvcm.TransferClient.Create.call_count, 1)

    def test_finalize_registration_races_the_first_storage_call_once(self) -> None:
        """Both entry points take the same lock: the instance registers once."""
        entered = threading.Event()
        release = threading.Event()
        finalized = threading.Event()

        def slow_register(request: Any) -> dict:
            entered.set()
            release.wait(timeout=5)
            return {"storage_configs": "[]"}

        def finalize() -> None:
            connector.finalize_registration()
            finalized.set()

        with _patched_parallel_context():
            connector = self._burst_connector()
            _manager(connector).register_instance.side_effect = slow_register

            storage_call = threading.Thread(
                target=connector.batch_exists, args=(["block-0"],)
            )
            storage_call.start()
            self.assertTrue(entered.wait(timeout=5), "the storage call never started")

            finalizer = threading.Thread(target=finalize)
            finalizer.start()
            self.assertFalse(
                finalized.is_set(),
                "the hook must wait for the initialization already in flight; "
                "the registration count below is what proves it did not slip "
                "past it into a second registration",
            )
            release.set()
            storage_call.join(timeout=5)
            finalizer.join(timeout=5)

        self.assertTrue(finalized.is_set(), "the finalize thread must complete")
        self.assertFalse(storage_call.is_alive())
        self.assertTrue(connector._client_ready)
        self.assertEqual(_manager(connector).register_instance.call_count, 1)
        self.assertEqual(_mock_kvcm.TransferClient.Create.call_count, 1)

    def test_finalize_registration_covers_every_pool_registered_before_it(self) -> None:
        """Every pool of the attach reaches the registration payload."""
        self.assertNotIn(
            UNMANAGED_POOL,
            {PoolName.KV, PoolName.MAMBA},
            "the burst already registered these; a pool outside them is what "
            "makes the third entry below a third pool rather than an override",
        )
        with _patched_parallel_context():
            connector = self._burst_connector()
            connector.register_mem_host_pool_v2(
                FakeHostPool(bytes_per_token=256, dtype=torch.bfloat16), UNMANAGED_POOL
            )
            connector.finalize_registration()

        payload = dict(_manager(connector).register_instance.call_args.args[0])
        self.assertEqual(
            [spec["name"] for spec in payload["location_spec_infos"]],
            [
                # A sidecar exists, so the anchor carries its hybrid name.
                "tp_0_full",
                "tp_1_full",
                "tp_0_linear",
                "tp_1_linear",
                f"tp_0_{UNMANAGED_POOL}",
                f"tp_1_{UNMANAGED_POOL}",
            ],
        )
        self.assertEqual(
            [group["name"] for group in payload["location_spec_groups"]],
            ["Full", "Linear", f"F{UNMANAGED_POOL}"],
        )
        self.assertEqual(connector._late_pools, frozenset(), "nothing was late")
        self.assertEqual(
            set(connector.pool_specs), {PoolName.KV, PoolName.MAMBA, UNMANAGED_POOL}
        )

    def test_finalize_registration_without_an_anchor_fails_loudly(self) -> None:
        """No KV pool means nothing can be registered: the attach fails."""
        with _patched_parallel_context():
            connector = _connector()
            # Only the v0.5.18-style group arrives, never the KV entry.
            connector.register_mem_pool_host(FakeHostPoolGroup())  # ty: ignore[invalid-argument-type]
            with self.assertLogs(connector_module.__name__, level="ERROR") as logs:
                with self.assertRaisesRegex(RuntimeError, "no KV host pool"):
                    connector.finalize_registration()

        self.assertEqual(len(logs.records), 1)
        self.assertIn("local:", logs.records[0].getMessage())
        self.assertFalse(connector._client_ready)
        self.assertIsNone(_manager(connector).register_instance.call_args)

    def test_finalize_registration_raises_a_permanent_shape_error(self) -> None:
        """An unservable anchor surfaces as the backend error, not a shape bug."""
        with _patched_parallel_context():
            connector = _connector()
            anchor = FakeLogicalHostPool()
            connector.register_mem_pool_host(anchor)  # ty: ignore[invalid-argument-type]
            connector.register_mem_host_pool_v2(anchor, PoolName.KV)  # ty: ignore[invalid-argument-type]

            with self.assertLogs(connector_module.__name__, level="ERROR") as logs:
                with self.assertRaises(
                    connector_module._UnsupportedBackendError
                ) as caught:
                    connector.finalize_registration()

        # What the caller can catch: a RuntimeError, not an AssertionError from
        # a stripped ``assert`` and not a NotImplementedError/AttributeError
        # from an unimplemented interface method.
        self.assertIsInstance(caught.exception, RuntimeError)
        self.assertEqual(len(logs.records), 2)
        self.assertIn("FakeLogicalHostPool", logs.records[0].getMessage())
        self.assertIn("rejected:", logs.records[1].getMessage())
        self.assertFalse(connector._client_ready)
        self.assertIsNone(_manager(connector).register_instance.call_args)

    def test_a_missing_sdk_client_is_a_permanent_failure(self) -> None:
        """``Create`` returning no client must not pass for a working backend."""
        with _patched_parallel_context():
            connector = self._burst_connector()
            # Set the shared mock's return value in place: ``reset_mock()``
            # (the other classes' setUp) keeps return values, and patching the
            # attribute would leave the mock's children unresettable.
            create = _mock_kvcm.TransferClient.Create
            create.return_value = None
            self.addCleanup(setattr, create, "return_value", MagicMock())

            with self.assertLogs(connector_module.__name__, level="ERROR") as logs:
                with self.assertRaises(connector_module._UnsupportedBackendError):
                    connector.finalize_registration()

        # Two distinct reports, not one repeated: what is broken (the SDK
        # client, with the instance and its specs), then that the attach fails.
        self.assertEqual(len(logs.records), 2)
        self.assertIn(
            "TransferClient.Create returned no client", logs.records[0].getMessage()
        )
        self.assertIn("registration-test", logs.records[0].getMessage())
        self.assertIn(
            "rejected: the KVCM transfer runtime could not be created",
            logs.records[1].getMessage(),
        )
        self.assertNotIn(
            "pool or spec shape",
            logs.records[1].getMessage(),
            "the reason of a missing SDK client is not a pool shape",
        )
        self.assertIsNone(connector.transfer_client)
        self.assertFalse(connector._client_ready)

    def test_a_missing_sdk_client_is_reported_once_on_the_lazy_path(self) -> None:
        """Without the hook the first storage call must report, not stay silent.

        Upstream sglang never calls ``finalize_registration``, so the first
        data-path call is what discovers the broken SDK.  The callers answer a
        conservative result without a log of their own, so the report has to
        happen where the failure is detected -- once for the instance, not once
        per call.
        """
        with _patched_parallel_context():
            connector = self._burst_connector()
            create = _mock_kvcm.TransferClient.Create
            create.return_value = None
            self.addCleanup(setattr, create, "return_value", MagicMock())

            with self.assertLogs(connector_module.__name__, level="ERROR") as logs:
                self.assertEqual(connector.batch_exists(["block-0"]), 0)
            self.assertEqual(len(logs.records), 1)
            message = logs.records[0].getMessage()
            self.assertIn("TransferClient.Create returned no client", message)
            self.assertIn("registration-test", message)
            self.assertIn("tp_0_linear", message, "the specs locate the failure")
            self.assertFalse(connector._client_ready)

            with self.assertNoLogs(connector_module.__name__, level="ERROR"):
                self.assertEqual(connector.batch_exists(["block-1"]), 0)
                self.assertEqual(
                    connector.batch_get_v1(["block-1"], torch.arange(PAGE_SIZE)),
                    [False],
                )
                self.assertEqual(connector.batch_exists_v2(["block-1"]).kv_hit_pages, 0)

    def test_an_unusable_anchor_does_not_suppress_the_sdk_report(self) -> None:
        """The two instance-wide reports keep keys of their own.

        ``_reported_pools`` deduplicates by key and is process-wide, so a shared
        key would silence the second cause -- exactly the failure this reporting
        exists to prevent.  Both orders are checked: neither report may depend on
        which one the process saw first.
        """

        def anchor_failure() -> list:
            connector = _connector()
            anchor = FakeLogicalHostPool()
            connector.register_mem_pool_host(anchor)  # ty: ignore[invalid-argument-type]
            connector.register_mem_host_pool_v2(anchor, PoolName.KV)  # ty: ignore[invalid-argument-type]
            with self.assertLogs(connector_module.__name__, level="ERROR") as logs:
                self.assertEqual(connector.batch_exists(["block-0"]), 0)
            return logs.records

        def sdk_failure() -> list:
            connector = self._burst_connector()
            # Set in place: ``reset_mock()`` keeps return values, and patching
            # the attribute would leave the mock's children unresettable.
            create = _mock_kvcm.TransferClient.Create
            create.return_value = None
            self.addCleanup(setattr, create, "return_value", MagicMock())
            with self.assertLogs(connector_module.__name__, level="ERROR") as logs:
                self.assertEqual(connector.batch_exists(["block-0"]), 0)
            return logs.records

        for order, first, second in (
            ("anchor first", anchor_failure, sdk_failure),
            ("sdk first", sdk_failure, anchor_failure),
        ):
            with self.subTest(order=order):
                # A fresh process as far as the process-wide report set goes.
                HiCacheKVCM._reported_pools.clear()
                with _patched_parallel_context():
                    messages = [record.getMessage() for record in first() + second()]
                self.assertEqual(len(messages), 2, messages)
                self.assertTrue(
                    any("not a host pool" in message for message in messages),
                    messages,
                )
                self.assertTrue(
                    any(
                        "has no KVCM transfer client" in message for message in messages
                    ),
                    messages,
                )

    def test_finalize_registration_keeps_the_manager_error_class(self) -> None:
        """Both kinds fail startup, but the type still says what went wrong."""
        cases = [
            (
                "unreachable",
                requests.ConnectionError("connection refused"),
                "unjudged: the Manager is unreachable",
            ),
            (
                "http-503",
                KvCacheManagerHTTPError(
                    "status code 503",
                    response=types.SimpleNamespace(status_code=503),
                ),
                "unjudged: the Manager failed with HTTP 503",
            ),
            (
                "http-400",
                KvCacheManagerHTTPError(
                    "status code 400", response=types.SimpleNamespace(status_code=400)
                ),
                "refused: the Manager rejected the request (HTTP 400)",
            ),
            (
                "protocol",
                KvCacheManagerProtocolError("missing a valid 'header' field"),
                "unjudged: the Manager did not decide on the registration",
            ),
        ]
        for name, error, label in cases:
            with self.subTest(name=name, label=label):
                with _patched_parallel_context():
                    connector = self._burst_connector()
                    _manager(connector).register_instance.side_effect = error
                    with self.assertLogs(
                        connector_module.__name__, level="ERROR"
                    ) as logs:
                        with self.assertRaises(type(error)) as caught:
                            connector.finalize_registration()

                self.assertIs(
                    caught.exception, error, "the Manager error must not be wrapped"
                )
                self.assertIn(label, logs.records[-1].getMessage())
                self.assertFalse(connector._client_ready)

    def test_a_pool_registered_after_finalize_registration_is_degraded(self) -> None:
        """Late means late: one report, misses, and the frozen pools unaffected."""
        with _patched_parallel_context():
            connector = self._burst_connector()
            connector.finalize_registration()

            late_pool = FakeHostPool(bytes_per_token=256, dtype=torch.bfloat16)
            with self.assertLogs(connector_module.__name__, level="WARNING") as logs:
                connector.register_mem_host_pool_v2(late_pool, UNMANAGED_POOL)
            self.assertEqual(len(logs.records), 1)
            self.assertIn(str(UNMANAGED_POOL), logs.records[0].getMessage())
            self.assertNotIn(UNMANAGED_POOL, connector.registered_pools)
            self.assertIn(UNMANAGED_POOL, connector._late_pools)

            manager = _manager(connector)
            manager.get_cache_location.return_value = {
                # Two KV pages that also carry the registered pool's spec, so
                # "the spec is missing from the Manager" cannot be the reason
                # why the late pool must come back as a miss.
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
            mamba = PoolTransfer(
                name=PoolName.MAMBA,
                host_indices=torch.arange(2),
                keys=["block-0", "block-1"],
            )
            late = PoolTransfer(
                name=UNMANAGED_POOL,
                host_indices=torch.arange(2),
                keys=["block-0", "block-1"],
            )
            self.assertNotEqual(
                mamba.name,
                late.name,
                "two transfers for one pool would share a hit-page entry, so "
                "the two assertions below would contradict each other",
            )
            with self.assertNoLogs(connector_module.__name__, level="WARNING"):
                exists = connector.batch_exists_v2(
                    ["block-0", "block-1"], [mamba, late]
                )
            self.assertEqual(exists.extra_pool_hit_pages[PoolName.KV], 2)
            self.assertEqual(exists.extra_pool_hit_pages[PoolName.MAMBA], 2)
            self.assertEqual(exists.extra_pool_hit_pages[UNMANAGED_POOL], 0)
            # The engine may load no page: the late pool's pages are not in
            # storage, so the prefix is clamped to the pool minimum.
            self.assertEqual(exists.kv_hit_pages, 0)

            manager.get_cache_location.reset_mock()
            with self.assertNoLogs(connector_module.__name__, level="WARNING"):
                self.assertEqual(
                    connector.batch_get_v2([late]), {UNMANAGED_POOL: [False, False]}
                )
            self.assertIsNone(
                manager.get_cache_location.call_args,
                "a late pool must not query the Manager for its spec",
            )

    def test_without_the_hook_the_specs_freeze_at_the_first_storage_call(self) -> None:
        """Upstream sglang never calls the hook, so it keeps today's timing."""
        with _patched_parallel_context():
            connector = _connector()
            connector.register_mem_pool_host(self._kv_pool())
            # Registered after the attach but before any storage call: the lazy
            # path freezes late enough to include it.
            connector.register_mem_host_pool_v2(self._mamba_pool(), PoolName.MAMBA)
            self.assertFalse(connector._client_ready)
            self.assertIsNone(_manager(connector).register_instance.call_args)

            connector.batch_exists(["block-0"])
            self.assertTrue(connector._client_ready)
            self.assertIn(PoolName.MAMBA, connector.pool_specs)
            self.assertEqual(connector._late_pools, frozenset(), "nothing was late")
            self.assertEqual(_manager(connector).register_instance.call_count, 1)

            late_pool = FakeHostPool(bytes_per_token=256, dtype=torch.bfloat16)
            with self.assertLogs(connector_module.__name__, level="WARNING") as logs:
                connector.register_mem_host_pool_v2(late_pool, UNMANAGED_POOL)
            self.assertEqual(len(logs.records), 1)
            self.assertIn(UNMANAGED_POOL, connector._late_pools)

    def test_finalize_registration_after_close_fails(self) -> None:
        """A retired backend must not register itself back into life."""
        with _patched_parallel_context():
            connector = self._burst_connector()
            connector.finalize_registration()
            connector.close()

            with self.assertLogs(connector_module.__name__, level="ERROR") as logs:
                with self.assertRaisesRegex(RuntimeError, "was closed"):
                    connector.finalize_registration()

        self.assertIn("local:", logs.records[-1].getMessage())
        self.assertEqual(_manager(connector).register_instance.call_count, 1)
        self.assertIsNone(connector.transfer_client)


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
            FakeHostPool(
                bytes_per_token=MAMBA_BYTES_PER_TOKEN, dtype=torch.float32, page_size=1
            ),
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
