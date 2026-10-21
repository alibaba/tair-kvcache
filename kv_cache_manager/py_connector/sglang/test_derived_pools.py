"""Deterministic (CPU-only) tests for pools the connector derives a spec for.

sglang hands every host pool of a hybrid stack to the storage backend, and a new
engine release may add a pool the connector's table has never heard of.  An
unbacked pool used to hold the whole prefix down to zero pages (the caller takes
the minimum over the pools), so a pool outside ``_POOL_SPECS`` is now derived
from itself: its own spec name, its own per-block size, its own spec group.

Covered:

* a pool outside the table is registered after the anchor, with ``tp_{rank}_name``
  and a group the Manager can classify, and the anchor switches to its hybrid
  name because a sidecar exists;
* its per-block size is counted from the pool's own page size, not from the
  anchor's block size;
* a write and a read go through that pool's own spec and its own page buffers;
* ``batch_exists_v2`` counts it by the transfer's own ``hit_policy``, and a mixed
  stack takes the minimum boundary;
* a sidecar that rides another pool's slots (``host_indices`` empty,
  ``indices_from_pool`` set) uses its source's indices, and answers a miss
  instead of raising when the source is not in the call;
* a derived name that would share the anchor's spec is refused (the pool falls
  back to misses) instead of aliasing two pools onto one spec;
* a registered name that is not a host pool is not derived at all.

Prerequisites: a sglang runtime that can import its backend and
``kv_cache_manager`` importable from the repository root on ``sys.path`` -- the
compiled kvcm pybind client and the build-generated ``_version_info`` are stubbed
below, so no wheel is needed for those.

    PYTHONPATH=$PWD python kv_cache_manager/py_connector/sglang/test_derived_pools.py
"""

import json
import sys
import types
import unittest
from contextlib import contextmanager
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import torch


# ── Modules the connector imports but this test does not exercise ──────
# Same technique as test_extra_pools.py / test_pool_registration.py.
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
    PoolHitPolicy,
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
KV_PAGE_BYTES = 4096
# A pool the table does not know.  PoolName.SWA is the realistic case (sglang
# >= v0.5.12); the plain string keeps the file independent of that enum.
DERIVED_POOL: Any = getattr(PoolName, "SWA", "swa")
# A registered name that is not a host pool (sglang's tensorless page anchor).
NON_HOST_POOL: Any = "no-host-pool"
# Two names of a pool pair no static row describes: a pool with its own slots and
# a sidecar that rides them.
SOURCE_POOL: Any = "source-pool"
SIDECAR_POOL: Any = "sidecar-pool"


class FakeHostPool(HostKVCache):
    """A sglang host pool without CUDA: spec metadata plus page buffers.

    ``get_page_buffer_meta`` hands out one IOV per page, which is the shape a
    single-component sidecar (indexer, plain KV sidecar) has; the per-block spec
    size is ``page_bytes`` by construction, so the tests can compare the two.
    """

    def __init__(
        self,
        *,
        page_size: int = PAGE_SIZE,
        page_bytes: int = KV_PAGE_BYTES,
        base: int = 0x1000_0000,
    ) -> None:
        self.page_size = page_size
        self.page_bytes = page_bytes
        self.base = base
        self.dtype = torch.bfloat16

    def get_size_per_token(self) -> int:
        return self.page_bytes // self.page_size

    def get_page_buffer_meta(self, indices: torch.Tensor) -> tuple[list, list]:
        # Host indices are token-granular: page_size of them address one page,
        # and one IOV covers a whole page.
        values = indices.tolist()
        pages = [
            int(values[offset]) // self.page_size
            for offset in range(0, len(values), self.page_size)
        ]
        return (
            [self.base + page * self.page_bytes for page in pages],
            [self.page_bytes] * len(pages),
        )

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


class FakeNonHostPool:
    """Shape of sglang's ``LogicalHostPool``: no tensor, no per-token size."""

    page_size = PAGE_SIZE
    dtype = torch.uint8


@contextmanager
def _patched_parallel_context():
    """Neutralize the collective parts of ``_init_parallel_context``."""
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
        "tp_size": 1,
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
            "instance_id": "derived-pool-test",
        },
    }
    args.update(overrides)
    return HiCacheStorageConfig(**args)


# One active HF3FS entry: the shared IOV area is only allocated by that backend,
# so the capacity diagnostics must be exercised with it present.
_HF3FS_STORAGE_CONFIGS = json.dumps(
    [
        {
            "type": "hf3fs",
            "is_available": True,
            "storage_spec": {"mountpoint": "/mnt/hf3fs", "root_dir": "/kvcm"},
        }
    ]
)


def _attached_connector(*, hf3fs: bool = False, **overrides: Any) -> HiCacheKVCM:
    """A real connector with a mocked Manager and SDK client."""
    _mock_kvcm.reset_mock()
    connector = HiCacheKVCM(_storage_config(**overrides), {})
    connector._manager_client = MagicMock()
    connector._manager_client.register_instance.return_value = {
        "storage_configs": _HF3FS_STORAGE_CONFIGS if hf3fs else "[]"
    }
    _manager(connector).get_cache_location.return_value = {"locations": []}
    return connector


def _manager(connector: HiCacheKVCM) -> Any:
    """The mocked Manager client of a connector built here."""
    return connector._manager_client


def _initialized_connector(
    *, pools: dict, extra_config: Any = None, hf3fs: bool = False
) -> HiCacheKVCM:
    """A connector initialized through the real ``_init_kvcm_client``.

    The pool set is the KV anchor plus ``pools``; the Manager and the SDK client
    are mocks, so the control plane (spec infos, groups, iov size, fingerprint)
    is what a deployment would compute.
    """
    config = {
        "manager_uri": "http://127.0.0.1:6382",
        "instance_group": "default",
        "instance_id": "derived-pool-test",
        **(extra_config or {}),
    }
    with _patched_parallel_context():
        connector = _attached_connector(extra_config=config, hf3fs=hf3fs)
        connector.register_mem_pool_host(FakeHostPool())
        for name, pool in pools.items():
            connector.register_mem_host_pool_v2(pool, name)
        connector.batch_exists(["block-0"])
    return connector


def _data_path_connector(*, pools: dict, **overrides: Any) -> HiCacheKVCM:
    """A connector with only the fields the v2 data paths read.

    The pool set is resolved through the real ``_resolve_pool_specs``, so the
    specs, sizes and groups are the ones an attached instance would use.
    """
    connector = HiCacheKVCM.__new__(HiCacheKVCM)
    connector.tp_rank = overrides.get("tp_rank", 0)
    connector.tp_size = overrides.get("tp_size", 1)
    connector.tp_world_size = 1
    connector.is_mla_model = overrides.get("is_mla_model", False)
    connector.block_size = PAGE_SIZE
    connector.instance_id = "derived-pool-test"
    connector.write_timeout_seconds = 30
    connector.registered_pools = {PoolName.KV: FakeHostPool(), **pools}
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
    er_ok = _mock_kvcm.ClientErrorCode.ER_OK
    connector.transfer_client.SaveKvCaches.return_value = (er_ok,)
    connector.transfer_client.LoadKvCaches.return_value = er_ok
    connector._client_ready = True
    return connector


def _refusal_reports(logs: Any, pool_name: Any) -> int:
    """How often a pool was reported as not backed.

    Counted by message: the constructor logs its version banner at WARNING too,
    and that is not a pool report.
    """
    return sum(
        1
        for record in logs.records
        if f"pool {pool_name} is not backed" in record.getMessage()
    )


def _gather_this_rank_only(object_list: Any, obj: Any) -> None:
    """This rank was handed the transfers; its peer was handed none."""
    if object_list is not None:
        object_list[:] = [list(obj or []), []]


def _fake_gather_all_ranks(
    object_list: Any = None, obj: Any = None, **kwargs: Any
) -> None:
    """Every rank of the group was handed the same transfers as this one."""
    if object_list is not None:
        object_list[:] = [list(obj or [])] * len(object_list)


def _iov_reports(logs: Any) -> list[tuple[str, str]]:
    """(level, message) of every report about a pool's share of the iov area."""
    return [
        (record.levelname, record.getMessage())
        for record in logs.records
        if "iov area" in record.getMessage()
    ]


def _locations(spec_names: dict, count: int) -> list[dict]:
    """Manager locations: page i carries every spec named for it."""
    return [
        {
            "location_specs": [
                {"name": spec, "uri": f"uri_{spec}_{i}"}
                for spec, pages in spec_names.items()
                if i in pages
            ]
        }
        for i in range(count)
    ]


def _buffer_iovs(buffers: list) -> list[list[tuple[int, int]]]:
    return [[(iov.base, iov.size) for iov in buffer.iovs] for buffer in buffers]


class TestDerivedPoolSpec(unittest.TestCase):
    """A pool outside ``_POOL_SPECS`` is backed by a row derived from itself."""

    def setUp(self) -> None:
        HiCacheKVCM._reported_pools.clear()

    def test_unknown_pool_is_derived_and_registered_after_the_anchor(self) -> None:
        with _patched_parallel_context():
            connector = _attached_connector(tp_size=2)
            connector.register_mem_pool_host(FakeHostPool())
            connector.register_mem_host_pool_v2(
                FakeHostPool(page_bytes=3072), DERIVED_POOL
            )
            connector.batch_exists(["block-0"])

        payload = dict(_manager(connector).register_instance.call_args.args[0])
        self.assertEqual(
            payload["location_spec_infos"],
            [
                # A sidecar exists, so the anchor carries its hybrid name and the
                # derived pool follows in this connector's own order: table rows
                # first, then the pools sglang registered.  That order is what the
                # Manager compares element by element (it sorts the *groups* by
                # name, so their order here carries no meaning).
                {"name": "tp_0_full", "size": KV_PAGE_BYTES},
                {"name": "tp_1_full", "size": KV_PAGE_BYTES},
                {"name": f"tp_0_{DERIVED_POOL}", "size": 3072},
                {"name": f"tp_1_{DERIVED_POOL}", "size": 3072},
            ],
        )
        self.assertEqual(
            payload["location_spec_groups"],
            [
                {"name": "Full", "spec_names": ["tp_0_full", "tp_1_full"]},
                {
                    # 'F' == "every page of the prefix must carry this group"
                    # (PoolHitPolicy.ALL_PAGES), the strict reading a
                    # registration cannot verify.
                    "name": f"F{DERIVED_POOL}",
                    "spec_names": [f"tp_0_{DERIVED_POOL}", f"tp_1_{DERIVED_POOL}"],
                },
            ],
        )
        self.assertEqual(connector.location_spec_name, "tp_0_full")
        # The SDK client is told about the same specs, so its transfers can
        # address the derived pool.
        client_config = connector.transfer_client_config
        self.assertIn(f"tp_0_{DERIVED_POOL}", client_config)

    def test_derived_pool_registers_rank0_spec_alone_on_an_mla_model(self) -> None:
        """A derived pool is replicated, so only rank 0's spec is registered."""
        with _patched_parallel_context():
            connector = _attached_connector(tp_rank=1, tp_size=2, is_mla_model=True)
            connector.register_mem_pool_host(FakeHostPool())
            connector.register_mem_host_pool_v2(
                FakeHostPool(page_bytes=3072), DERIVED_POOL
            )
            connector.batch_exists(["block-0"])

        payload = dict(_manager(connector).register_instance.call_args.args[0])
        self.assertEqual(
            [spec["name"] for spec in payload["location_spec_infos"]],
            ["tp_0_full", f"tp_0_{DERIVED_POOL}"],
        )
        self.assertEqual(
            payload["location_spec_groups"],
            [
                {"name": "Full", "spec_names": ["tp_0_full"]},
                {"name": f"F{DERIVED_POOL}", "spec_names": [f"tp_0_{DERIVED_POOL}"]},
            ],
        )
        # Rank 1 reads and writes rank 0's spec too.
        self.assertEqual(
            connector.pool_specs[DERIVED_POOL].name, f"tp_0_{DERIVED_POOL}"
        )
        self.assertEqual(connector.location_spec_name, "tp_0_full")

    def test_spec_size_uses_the_pool_own_page_size(self) -> None:
        """Not the anchor's block size: a sidecar page may be narrower."""
        pool = FakeHostPool(page_size=32, page_bytes=1024)
        connector = _data_path_connector(pools={DERIVED_POOL: pool})

        # 1024 // 32 bytes per slot times its own 32 slots, not block_size 64.
        self.assertEqual(connector.pool_specs[DERIVED_POOL].size, 1024)
        self.assertEqual(connector.location_spec_size, KV_PAGE_BYTES)

    def test_registered_non_host_pool_is_not_derived(self) -> None:
        """A name without a host pool keeps failing soft, as before."""
        connector = _data_path_connector(pools={NON_HOST_POOL: FakeNonHostPool()})

        self.assertNotIn(NON_HOST_POOL, connector.pool_specs)
        _manager(connector).get_cache_location.return_value = {
            "locations": _locations({connector.location_spec_name: {0, 1}}, 2)
        }
        transfer = PoolTransfer(
            name=NON_HOST_POOL,
            host_indices=torch.arange(1),
            keys=["block-0"],
        )
        with self.assertLogs(connector_module.__name__, "WARNING") as logs:
            result = connector.batch_exists_v2(["block-0", "block-1"], [transfer])
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(result.kv_hit_pages, 0)
        self.assertEqual(result.extra_pool_hit_pages[NON_HOST_POOL], 0)

    def test_derived_row_sharing_the_anchor_spec_is_refused(self) -> None:
        """A pool named like the anchor's hybrid suffix must not alias it.

        ``tp_{rank}_full`` is the anchor's name whenever a sidecar exists, so a
        pool literally called ``full`` would be handed the anchor's spec: the
        Manager would see one spec name twice with two sizes.  The derived row
        is dropped instead, that pool falls back to misses, and it is reported
        exactly once -- whatever the number of calls that hit it.
        """
        colliding: Any = "full"
        with self.assertLogs(connector_module.__name__, "WARNING") as logs:
            # The refusal is decided while the pool set is resolved, i.e. here.
            connector = _data_path_connector(pools={colliding: FakeHostPool()})
            self.assertNotIn(colliding, connector.pool_specs)
            self.assertEqual(
                set(connector.pool_specs),
                {PoolName.KV},
                "only the anchor is backed: the colliding name is refused",
            )
            _manager(connector).get_cache_location.return_value = {
                "locations": _locations({connector.location_spec_name: {0, 1}}, 2)
            }
            transfer = PoolTransfer(
                name=colliding,
                host_indices=torch.arange(PAGE_SIZE),
                keys=["block-0"],
            )
            self.assertEqual(connector.batch_set_v2([transfer])[colliding], [False])
            self.assertEqual(
                connector.batch_exists_v2(
                    ["block-0", "block-1"], [transfer]
                ).kv_hit_pages,
                0,
            )

        self.assertEqual(
            _refusal_reports(logs, colliding),
            1,
            "a refused pool is reported once, not once per call",
        )
        connector.transfer_client.SaveKvCaches.assert_not_called()

    def test_derived_row_sharing_a_static_group_is_refused(self) -> None:
        """A derived group is ``"F" + pool name``, so ``ull`` lands on ``Full``.

        Its spec name is free (``tp_0_ull``), so only the group check can catch
        it: two rows in one group make the Manager resolve the name to a single
        group and silently skip the other row's specs.  The refused pool misses,
        and the pools that are backed keep their hits.
        """
        colliding: Any = "ull"
        with self.assertLogs(connector_module.__name__, "WARNING") as logs:
            connector = _initialized_connector(
                pools={DERIVED_POOL: FakeHostPool(), colliding: FakeHostPool()}
            )
            self.assertEqual(
                set(connector.pool_specs),
                {PoolName.KV, DERIVED_POOL},
                "the group collision is what refuses the row; its name is free",
            )
            self.assertEqual(
                [group["name"] for group in connector.location_spec_groups],
                ["Full", f"F{DERIVED_POOL}"],
                "the anchor's group stays unique",
            )
            derived_name = connector.pool_specs[DERIVED_POOL].name
            _manager(connector).get_cache_location.return_value = {
                "locations": _locations(
                    {
                        connector.location_spec_name: {0, 1},
                        derived_name: {0, 1},
                    },
                    2,
                )
            }
            refused = PoolTransfer(
                name=colliding,
                host_indices=torch.arange(PAGE_SIZE),
                keys=["block-0"],
            )
            good = PoolTransfer(
                name=DERIVED_POOL,
                host_indices=torch.arange(2 * PAGE_SIZE),
                keys=["block-0", "block-1"],
            )
            # The refused pool's write never reaches the SDK ...
            self.assertEqual(connector.batch_set_v2([refused])[colliding], [False])
            # ... and it costs nothing but its own pages in a query.
            mixed = connector.batch_exists_v2(["block-0", "block-1"], [good, refused])
            self.assertEqual(mixed.extra_pool_hit_pages[colliding], 0)
            self.assertEqual(mixed.extra_pool_hit_pages[DERIVED_POOL], 2)
            alone = connector.batch_exists_v2(["block-0", "block-1"], [good])
            self.assertEqual(
                alone.kv_hit_pages, 2, "the backed pools keep their own boundary"
            )

        self.assertEqual(_refusal_reports(logs, colliding), 1)
        connector.transfer_client.SaveKvCaches.assert_not_called()

    def test_derived_pool_does_not_grow_the_shared_iov_area(self) -> None:
        """A pool the table does not know must not be paid for by everybody.

        The HF3FS backend allocates ``iov_size`` bytes twice at Init (read and
        write), host-pinned, and a client that cannot be created costs the whole
        instance its cache.  So only the pools this connector has always backed
        size it -- and since the area is allocated one block at a time, a bigger
        derived pool still fits as long as a single one of its blocks does.
        """
        huge_bytes = 64 * KV_PAGE_BYTES
        huge = FakeHostPool(page_bytes=huge_bytes)
        with self.assertLogs(connector_module.__name__, "WARNING") as logs:
            connector = _initialized_connector(pools={DERIVED_POOL: huge}, hf3fs=True)

        self.assertEqual(
            connector.iov_size,
            KV_PAGE_BYTES * 1024,
            "the value an instance without this pool would compute",
        )
        self.assertEqual(
            _iov_reports(logs),
            [],
            "one block fits, so nothing is reported even though the pool is big",
        )

        # Configuring the area above one block keeps it quiet; below one block
        # the pool's transfers can never allocate, which is what gets reported.
        with self.assertLogs(connector_module.__name__, "WARNING") as quiet:
            enlarged = _initialized_connector(
                pools={DERIVED_POOL: huge},
                extra_config={"iov_size_bytes": huge_bytes},
                hf3fs=True,
            )
        self.assertEqual(enlarged.iov_size, huge_bytes)
        self.assertEqual(_iov_reports(quiet), [])

        with self.assertLogs(connector_module.__name__, "WARNING") as logs:
            too_small = _initialized_connector(
                pools={DERIVED_POOL: huge},
                extra_config={"iov_size_bytes": huge_bytes - 1},
                hf3fs=True,
            )
        self.assertEqual(too_small.iov_size, huge_bytes - 1)
        reports = _iov_reports(logs)
        self.assertEqual([level for level, _ in reports], ["ERROR"])
        self.assertIn(f"pool {DERIVED_POOL}", reports[0][1])
        self.assertIn("does not hold even one", reports[0][1])
        self.assertIn(str(huge_bytes), reports[0][1], "the number to configure")

    def test_iov_report_applies_the_sdk_alignment_first(self) -> None:
        """Both sides are rounded up before the area is compared with a block.

        ``Hf3fsSdk::InitIovHandle`` rounds the configured area up to a multiple of
        ``iov_block_size`` and ``Hf3fsMempool::Alloc`` rounds the requested block
        up to its alignment, so a 110-byte block in a 100-byte area both become
        128 bytes and the transfer fits -- reporting "never backed" there would
        send an operator after pinned memory it does not need.
        """
        with self.assertLogs(connector_module.__name__, "DEBUG") as logs:
            connector = _initialized_connector(
                pools={DERIVED_POOL: FakeHostPool(page_bytes=128)},
                extra_config={
                    "iov_size_bytes": 100,
                    "read_iov_block_size": 128,
                    "write_iov_block_size": 128,
                },
                hf3fs=True,
            )
        # Each direction is rounded with its own alignment: 100 and 128 both
        # become one 128-byte write bucket, but the read area (64-byte buckets)
        # also holds the block (two of them), so neither direction is exhausted.
        reports = _iov_reports(logs)
        self.assertFalse(
            [m for _, m in reports if f"pool {DERIVED_POOL} " in m],
            f"128 bytes and a 100-byte area are both 128 after rounding: {reports}",
        )
        self.assertEqual(connector.pool_specs[DERIVED_POOL].size, 128)
        # The KV anchor is a different story: 4096 bytes needs 32 buckets of 128
        # in a 128-byte area, so it really does not fit and is reported.
        self.assertTrue(
            any("pool kv " in m for _, m in reports),
            "the anchor's block genuinely does not fit this area",
        )

    def test_iov_report_checks_each_direction_separately(self) -> None:
        """A 64-byte read bucket cannot hold a 65-byte block, even if write can."""
        with self.assertLogs(connector_module.__name__, "DEBUG") as logs:
            _initialized_connector(
                pools={DERIVED_POOL: FakeHostPool(page_bytes=128)},
                # read: 64-byte buckets round 100 -> 128 and 128 -> 128, so the
                # read area fits; the raw numbers alone would not tell us that
                # the two directions use different alignments.
                extra_config={
                    "iov_size_bytes": 100,
                    "read_iov_block_size": 64,
                    "write_iov_block_size": 128,
                },
                hf3fs=True,
            )
        reports = [m for _, m in _iov_reports(logs) if f"pool {DERIVED_POOL} " in m]
        self.assertEqual(reports, [], "both directions fit after their own rounding")

    def test_no_iov_report_without_an_hf3fs_backend(self) -> None:
        """Only HF3FS allocates the area, so only HF3FS can be short of it."""
        huge = FakeHostPool(page_bytes=64 * KV_PAGE_BYTES)
        with self.assertLogs(connector_module.__name__, "WARNING") as logs:
            connector = _initialized_connector(
                pools={DERIVED_POOL: huge},
                extra_config={"iov_size_bytes": 1},
            )

        self.assertEqual(_iov_reports(logs), [], "no HF3FS backend, no budget")
        self.assertEqual(connector.iov_size, 1)

    def test_iov_report_is_about_fitting_one_block(self) -> None:
        """The area is allocated per block, so one block is the whole budget.

        ``Hf3fsSdk::Get``/``Put`` walk a v2 operation one URI at a time and each
        call allocates that block's bytes from the mempool before freeing them,
        so a call is not limited by how many blocks the area holds -- only by
        whether the largest single block fits.  Everything above one block just
        allows more transfers in flight.
        """
        block = KV_PAGE_BYTES
        cases = [
            # (iov_size_bytes, derived pool block bytes, reported pools)
            (None, block, []),  # the default area holds 1024 of these
            (2 * block, 2 * block, []),  # exactly one block fits
            (2 * block, 3 * block, [DERIVED_POOL]),  # one byte over does not
            (block - 1, block, ["kv", DERIVED_POOL]),  # nothing fits
        ]
        for iov_bytes, page_bytes, reported in cases:
            with self.subTest(page_bytes=page_bytes, iov_bytes=iov_bytes):
                extra_config = (
                    None if iov_bytes is None else {"iov_size_bytes": iov_bytes}
                )
                with self.assertLogs(connector_module.__name__, "DEBUG") as logs:
                    connector = _initialized_connector(
                        pools={DERIVED_POOL: FakeHostPool(page_bytes=page_bytes)},
                        extra_config=extra_config,
                        hf3fs=True,
                    )
                size = connector.pool_specs[DERIVED_POOL].size
                self.assertEqual(size, page_bytes, "the fake's block is exact")
                reports = _iov_reports(logs)
                if not reported:
                    self.assertEqual(reports, [], "one block fits, nothing to say")
                    continue
                self.assertEqual({level for level, _ in reports}, {"ERROR"})
                self.assertEqual(len(reports), len(reported))
                for pool_name in reported:
                    self.assertTrue(
                        any(f"pool {pool_name} " in m for _, m in reports),
                        f"{pool_name} was not reported: {reports}",
                    )
                message = next(m for _, m in reports if f"pool {DERIVED_POOL} " in m)
                self.assertIn("does not hold even one", message)
                self.assertIn(str(page_bytes), message, "the figure to configure")

    def test_register_failure_names_the_instance_id_rule(self) -> None:
        """The pool set is frozen into the instance id, so it cannot be patched.

        ``registerInstance`` accepts a repeat only when every field matches, and
        the Manager has no spec update API: a pool that this connector backs now
        but did not before is a new deployment.  The failure has to say so,
        together with the layout this run tried to register.
        """
        with _patched_parallel_context():
            connector = _attached_connector()
            connector.register_mem_pool_host(FakeHostPool())
            connector.register_mem_host_pool_v2(
                FakeHostPool(page_bytes=3072), DERIVED_POOL
            )
            _manager(connector).register_instance.side_effect = AssertionError(
                "Request to /api/registerInstance failed with error: register "
                "instance failed: instance_id 'derived-pool-test' already exists "
                "with different configuration, mismatched fields: "
                "[location_spec_infos, location_spec_groups]"
            )
            with self.assertLogs(connector_module.__name__, "DEBUG") as logs:
                self.assertEqual(connector.batch_exists(["block-0"]), 0)

        fingerprint = connector.spec_fingerprint
        self.assertEqual(len(fingerprint), 16)
        debug = [r.getMessage() for r in logs.records if r.levelname == "DEBUG"]
        error = [r.getMessage() for r in logs.records if r.levelname == "ERROR"]
        self.assertTrue(
            any(fingerprint in message and "freezes" in message for message in debug),
            "the frozen layout is logged before the registration",
        )
        self.assertTrue(
            any(
                fingerprint in message and "new instance_id" in message
                for message in error
            ),
            f"the failure must name the rule, got {error}",
        )
        self.assertTrue(
            any(
                "already exists with different configuration" in message
                for message in error
            ),
            "the Manager's own diagnosis stays attached",
        )

        # A different pool set is a different layout, which is the whole point.
        with _patched_parallel_context():
            same_set = _attached_connector()
            same_set.register_mem_pool_host(FakeHostPool())
            same_set.register_mem_host_pool_v2(
                FakeHostPool(page_bytes=3072), DERIVED_POOL
            )
            same_set.batch_exists(["block-0"])
        self.assertEqual(same_set.spec_fingerprint, fingerprint)
        with _patched_parallel_context():
            flat = _attached_connector()
            flat.register_mem_pool_host(FakeHostPool())
            flat.batch_exists(["block-0"])
        self.assertNotEqual(flat.spec_fingerprint, fingerprint)

    def test_register_failure_is_reported_once_per_cause(self) -> None:
        """The retry path must not turn one failure into one log per call.

        ``_ensure_client`` retries on every storage call, and the report is
        deduplicated through the process-wide registry the pool reports already
        use -- once for a rejection the Manager made, once for a failure that kept
        it from judging the request at all, and the two say different things.
        """
        permanent = AssertionError(
            "Request to /api/registerInstance failed with error: register instance "
            "failed: instance_id 'derived-pool-test' already exists with different "
            "configuration, mismatched fields: [location_spec_infos]"
        )
        with _patched_parallel_context():
            connector = _attached_connector()
            connector.register_mem_pool_host(FakeHostPool())
            _manager(connector).register_instance.side_effect = permanent
            with self.assertLogs(connector_module.__name__, "WARNING") as logs:
                for _ in range(3):
                    self.assertEqual(connector.batch_exists(["block-0"]), 0)

        rejections = [
            record.getMessage()
            for record in logs.records
            if "rejected the registration" in record.getMessage()
        ]
        self.assertEqual(len(rejections), 1, "one report for one failed cause")
        self.assertIn("new instance_id", rejections[0])

        # A failure the Manager never judged is a different cause: one report of
        # its own, and no claim about the pool set.
        with _patched_parallel_context():
            _manager(connector).register_instance.side_effect = ConnectionError(
                "connection reset by peer"
            )
            with self.assertLogs(connector_module.__name__, "WARNING") as logs:
                for _ in range(3):
                    self.assertEqual(connector.batch_exists(["block-0"]), 0)

        transient = [
            (record.levelname, record.getMessage())
            for record in logs.records
            if "could not be judged" in record.getMessage()
        ]
        self.assertEqual(len(transient), 1)
        self.assertEqual(transient[0][0], "WARNING")
        self.assertNotIn("new instance_id", transient[0][1])
        self.assertIn("connection reset by peer", transient[0][1])

        # A later call of the first cause is still silent: the registry is
        # process-wide, like the pool reports it lives next to.
        with _patched_parallel_context():
            _manager(connector).register_instance.side_effect = permanent
            with self.assertLogs(connector_module.__name__, "DEBUG") as logs:
                self.assertEqual(connector.batch_exists(["block-0"]), 0)
        self.assertEqual(
            [r for r in logs.records if "rejected the registration" in r.getMessage()],
            [],
        )

    def test_zero_byte_pool_is_not_backed_and_does_not_break_the_instance(self) -> None:
        """A pool with no bytes per block must not cost the instance its cache.

        Its spec size would be zero, which neither the Manager nor the SDK
        accepts, and the IOV area cannot hold any of it -- so the pool is simply
        not backed.  The rest of the instance has to keep working: the alternative
        is one malformed pool failing every other pool's caching too.
        """
        empty: Any = "empty-pool"
        with self.assertLogs(connector_module.__name__, "WARNING") as logs:
            connector = _initialized_connector(
                pools={DERIVED_POOL: FakeHostPool(), empty: FakeHostPool(page_bytes=0)}
            )
            self.assertTrue(connector._client_ready, "the KV anchor still works")
            self.assertNotIn(empty, connector.pool_specs)
            self.assertIn(
                DERIVED_POOL, connector.pool_specs, "the healthy pool is kept"
            )
            _manager(connector).get_cache_location.return_value = {
                "locations": _locations(
                    {
                        connector.location_spec_name: {0, 1},
                        connector.pool_specs[DERIVED_POOL].name: {0, 1},
                    },
                    2,
                )
            }
            result = connector.batch_exists_v2(
                ["block-0", "block-1"],
                [
                    PoolTransfer(
                        name=DERIVED_POOL,
                        host_indices=torch.arange(2 * PAGE_SIZE),
                        keys=["block-0", "block-1"],
                    ),
                    PoolTransfer(
                        name=empty,
                        host_indices=torch.arange(PAGE_SIZE),
                        keys=["block-0"],
                    ),
                ],
            )

        reports = [
            r.getMessage() for r in logs.records if "zero bytes" in r.getMessage()
        ]
        self.assertEqual(len(reports), 1)
        self.assertIn(empty, reports[0])
        self.assertEqual(result.extra_pool_hit_pages[empty], 0, "it answers a miss")
        self.assertEqual(
            result.extra_pool_hit_pages[DERIVED_POOL], 2, "the rest is fine"
        )
        # The zero-size pool is also absent from what the Manager and the SDK see.
        self.assertEqual(
            [spec["name"] for spec in connector.location_spec_infos],
            ["tp_0_full", f"tp_0_{DERIVED_POOL}"],
        )

    def test_register_failure_cause_routes_each_error_kind(self) -> None:
        """Four ways to fail must not end up in two piles, let alone one.

        ``KvCacheManagerHTTPError`` and ``KvCacheManagerProtocolError`` inherit
        ``AssertionError`` (``common/manager_client.py``), so the classification
        cannot be ``isinstance(error, AssertionError)``: a 503, a broken envelope
        or a lost leader election would all be told to change ``instance_id``,
        which restarts the cache for nothing.
        """
        shape_error = AssertionError(
            "Request to /api/registerInstance failed with error: register instance "
            "failed: instance_id 'derived-pool-test' already exists with different "
            "configuration, mismatched fields: [location_spec_infos]"
        )
        not_leader = AssertionError(
            "Request to /api/registerInstance failed with error: Server is not leader"
        )
        unavailable = KvCacheManagerHTTPError(
            "Request to /api/registerInstance failed with status code 503",
            response=types.SimpleNamespace(status_code=503),
        )
        bad_request = KvCacheManagerHTTPError(
            "Request to /api/registerInstance failed with status code 400",
            response=types.SimpleNamespace(status_code=400),
        )
        broken_envelope = KvCacheManagerProtocolError(
            "Response from /api/registerInstance missing a valid 'header' field"
        )
        # A business status the Manager returns when persisting the instance
        # fails (``config/registry_manager.cc``): the same bare AssertionError as
        # a rejected layout, but retrying the same registration can recover.
        persist_failed = AssertionError(
            "Request to /api/registerInstance failed with error: register instance "
            "failed: failed to persist instance info to storage backend"
        )
        cases = [
            (shape_error, "ERROR", "new instance_id"),
            (persist_failed, "WARNING", "could not be judged"),
            (bad_request, "ERROR", "refused the registration request"),
            (not_leader, "WARNING", "could not be judged"),
            (unavailable, "WARNING", "could not be judged"),
            (broken_envelope, "WARNING", "could not be judged"),
        ]
        for error, expected_level, expected_text in cases:
            with self.subTest(error=type(error).__name__, text=expected_text):
                HiCacheKVCM._reported_pools.clear()
                with _patched_parallel_context():
                    connector = _attached_connector()
                    connector.register_mem_pool_host(FakeHostPool())
                    _manager(connector).register_instance.side_effect = error
                    with self.assertLogs(connector_module.__name__, "WARNING") as logs:
                        self.assertEqual(connector.batch_exists(["block-0"]), 0)
                markers = (
                    "rejected the registration",
                    "refused the registration request",
                    "could not be judged",
                )
                reports = [
                    (r.levelname, r.getMessage())
                    for r in logs.records
                    if any(marker in r.getMessage() for marker in markers)
                ]
                self.assertEqual(len(reports), 1)
                self.assertEqual(reports[0][0], expected_level)
                self.assertIn(expected_text, reports[0][1])
                if expected_level == "WARNING":
                    self.assertNotIn("new instance_id", reports[0][1])

    def test_register_failure_is_reported_per_instance(self) -> None:
        """Two instances in one process are two deployments.

        sglang can detach and re-attach the backend with another ``instance_id``
        at runtime, and this connector registers on the first storage call -- so
        a silent second failure would be an attach that "succeeded" while every
        block misses.  The same instance failing again stays quiet.
        """
        permanent = AssertionError(
            "Request to /api/registerInstance failed with error: register instance "
            "failed: instance_id 'x' already exists with different configuration"
        )
        messages = []
        for instance_id in ("instance-A", "instance-B"):
            config = {
                "manager_uri": "http://127.0.0.1:6382",
                "instance_group": "default",
                "instance_id": instance_id,
            }
            with _patched_parallel_context():
                connector = _attached_connector(extra_config=config)
                connector.register_mem_pool_host(FakeHostPool())
                _manager(connector).register_instance.side_effect = permanent
                with self.assertLogs(connector_module.__name__, "WARNING") as logs:
                    for _ in range(2):
                        self.assertEqual(connector.batch_exists(["block-0"]), 0)
            rejected = [
                r.getMessage()
                for r in logs.records
                if "rejected the registration" in r.getMessage()
            ]
            self.assertEqual(len(rejected), 1, f"{instance_id}: one report")
            self.assertIn(instance_id, rejected[0])
            messages.append(rejected[0])
        self.assertNotEqual(messages[0], messages[1], "each instance names itself")

    def test_derived_pool_writes_from_rank0_alone_on_an_mla_model(self) -> None:
        """sglang hands every pool but the Mamba state to rank 0 on MLA.

        A derived pool therefore takes the rank-0 protocol: claiming it is
        sharded would leave rank 0 waiting in a broadcast for ranks sglang never
        calls.  On a non-MLA model every rank keeps its own spec.
        """
        dense = _data_path_connector(
            pools={DERIVED_POOL: FakeHostPool()}, tp_rank=1, tp_size=2
        )
        self.assertEqual(
            dense.pool_specs[DERIVED_POOL].name,
            f"tp_1_{DERIVED_POOL}",
            "a non-MLA rank reads and writes its own spec",
        )
        transfer = PoolTransfer(name=DERIVED_POOL, keys=["block-0"])
        self.assertTrue(dense._transfer_is_written_by_every_rank(transfer))

        mla = _data_path_connector(
            pools={DERIVED_POOL: FakeHostPool()},
            tp_rank=1,
            tp_size=2,
            is_mla_model=True,
        )
        self.assertEqual(mla.pool_specs[DERIVED_POOL].name, f"tp_0_{DERIVED_POOL}")
        self.assertFalse(mla._transfer_is_written_by_every_rank(transfer))

    def test_resolving_without_a_registered_pool_yields_no_row(self) -> None:
        """Resolving against an empty pool set stays an empty answer."""
        connector = _data_path_connector(pools={})
        connector.registered_pools = {}

        self.assertEqual(dict(connector._resolve_pool_specs()), {})


class TestDerivedPoolDataPath(unittest.TestCase):
    """Write, read and existence checks of a derived pool."""

    def setUp(self) -> None:
        HiCacheKVCM._reported_pools.clear()

    def test_write_then_read_uses_the_derived_spec(self) -> None:
        connector = _data_path_connector(pools={DERIVED_POOL: FakeHostPool()})
        num_pages = 2
        spec_name = connector.pool_specs[DERIVED_POOL].name
        self.assertEqual(spec_name, f"tp_0_{DERIVED_POOL}")
        keys = [f"block-{i}" for i in range(num_pages)]
        _manager(connector).start_write_cache.return_value = {
            "locations": _locations({spec_name: {0, 1}}, num_pages),
            "write_session_id": "ws-derived",
            "block_mask": {"offset": 0},
        }
        transfer = PoolTransfer(
            name=DERIVED_POOL,
            host_indices=torch.arange(num_pages * PAGE_SIZE),
            keys=keys,
        )

        written = connector.batch_set_v2([transfer])

        self.assertEqual(written[DERIVED_POOL], [True] * num_pages)
        write_request = _manager(connector).start_write_cache.call_args.args[0]
        self.assertEqual(
            write_request["location_spec_group_names"], [f"F{DERIVED_POOL}"] * 2
        )
        uris, buffers = connector.transfer_client.SaveKvCaches.call_args.args
        self.assertEqual(uris, [f"uri_{spec_name}_{i}" for i in range(num_pages)])
        self.assertEqual(
            _buffer_iovs(buffers),
            [[(0x1000_0000 + i * KV_PAGE_BYTES, KV_PAGE_BYTES)] for i in range(2)],
        )

        _manager(connector).get_cache_location.return_value = {
            "locations": _locations({spec_name: {0, 1}}, num_pages)
        }
        read = connector.batch_get_v2([transfer])

        self.assertEqual(read[DERIVED_POOL], [True, True])
        self.assertEqual(connector.transfer_client.LoadKvCaches.call_count, 1)

    def test_exists_counts_a_derived_pool_by_its_hit_policy(self) -> None:
        """A trailing pool only has to hold the boundary page, an all-pages one
        every page of the prefix."""
        connector = _data_path_connector(pools={DERIVED_POOL: FakeHostPool()})
        spec_name = connector.pool_specs[DERIVED_POOL].name
        keys = [f"block-{i}" for i in range(3)]
        # The anchor is cached on all three pages, the derived pool on the first
        # and the last one (a gap on page 1).
        _manager(connector).get_cache_location.return_value = {
            "locations": _locations(
                {connector.location_spec_name: {0, 1, 2}, spec_name: {0, 2}}, 3
            )
        }

        all_pages = PoolTransfer(
            name=DERIVED_POOL,
            host_indices=torch.arange(3),
            keys=keys,
            hit_policy=PoolHitPolicy.ALL_PAGES,
        )
        result = connector.batch_exists_v2(keys, [all_pages])
        self.assertEqual(result.extra_pool_hit_pages[PoolName.KV], 3)
        self.assertEqual(result.extra_pool_hit_pages[DERIVED_POOL], 1)
        self.assertEqual(result.kv_hit_pages, 1, "the first gap ends the prefix")

        trailing = PoolTransfer(
            name=DERIVED_POOL,
            host_indices=torch.arange(1),
            keys=[keys[-1]],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        result = connector.batch_exists_v2(keys, [trailing])
        self.assertEqual(result.extra_pool_hit_pages[DERIVED_POOL], 3)
        self.assertEqual(result.kv_hit_pages, 3)

    def test_mixed_pools_take_the_minimum_boundary(self) -> None:
        mamba = FakeHostPool(page_size=1, page_bytes=96)
        connector = _data_path_connector(
            pools={PoolName.MAMBA: mamba, DERIVED_POOL: FakeHostPool()}
        )
        keys = [f"block-{i}" for i in range(3)]
        derived_name = connector.pool_specs[DERIVED_POOL].name
        _manager(connector).get_cache_location.return_value = {
            "locations": _locations(
                {
                    connector.location_spec_name: {0, 1, 2},
                    connector.pool_specs[PoolName.MAMBA].name: {0, 2},
                    derived_name: {0, 1, 2},
                },
                3,
            )
        }
        transfers = [
            PoolTransfer(
                name=PoolName.MAMBA,
                host_indices=torch.arange(3),
                keys=keys,
                hit_policy=PoolHitPolicy.ALL_PAGES,
            ),
            PoolTransfer(
                name=DERIVED_POOL,
                host_indices=torch.arange(3),
                keys=keys,
                hit_policy=PoolHitPolicy.ALL_PAGES,
            ),
        ]

        result = connector.batch_exists_v2(keys, transfers)

        self.assertEqual(result.extra_pool_hit_pages[PoolName.KV], 3)
        self.assertEqual(result.extra_pool_hit_pages[PoolName.MAMBA], 1)
        self.assertEqual(result.extra_pool_hit_pages[DERIVED_POOL], 3)
        self.assertEqual(result.kv_hit_pages, 1)


class TestSidecarIndices(unittest.TestCase):
    """A sidecar reuses another pool's slots and owns only its bytes."""

    def setUp(self) -> None:
        HiCacheKVCM._reported_pools.clear()

    def _connector(self) -> HiCacheKVCM:
        return _data_path_connector(
            pools={
                SOURCE_POOL: FakeHostPool(base=0x1000_0000),
                SIDECAR_POOL: FakeHostPool(base=0x2000_0000),
            }
        )

    def test_sidecar_takes_its_source_pool_indices(self) -> None:
        connector = self._connector()
        num_pages = 2
        sidecar_name = connector.pool_specs[SIDECAR_POOL].name
        _manager(connector).get_cache_location.return_value = {
            "locations": _locations({sidecar_name: {0, 1}}, num_pages)
        }
        transfers = [
            PoolTransfer(
                name=SOURCE_POOL,
                # Pages 5 and 6 of the source pool, not pages 0 and 1.  Host
                # indices are token-granular, i.e. page_size of them per page.
                host_indices=torch.arange(5 * PAGE_SIZE, (5 + num_pages) * PAGE_SIZE),
                keys=[f"block-{i}" for i in range(num_pages)],
            ),
            PoolTransfer(
                name=SIDECAR_POOL,
                host_indices=None,
                keys=[f"block-{i}" for i in range(num_pages)],
                indices_from_pool=SOURCE_POOL,
            ),
        ]

        result = connector.batch_get_v2(transfers)

        self.assertEqual(result[SIDECAR_POOL], [True, True])
        _uris, buffers = connector.transfer_client.LoadKvCaches.call_args.args
        self.assertEqual(
            _buffer_iovs(buffers),
            [[(0x2000_0000 + page * KV_PAGE_BYTES, KV_PAGE_BYTES)] for page in (5, 6)],
            "the sidecar's own buffers, at the source pool's page offsets",
        )

    def test_a_rank_that_cannot_resolve_indices_joins_the_session_anyway(
        self,
    ) -> None:
        """A peer would wait for this rank forever if it skipped the handshake.

        Whether the indices resolve can differ per rank (the source pool is in
        one rank's transfer list and not the other's), but the session
        broadcast cannot: every rank of a shared transfer has to take part.
        """
        connector = self._connector()
        connector.tp_rank = 1
        connector.tp_world_size = 2
        connector.storage_tp_group = object()
        connector._ensure_client_for_write = lambda transfers: None  # ty: ignore[invalid-assignment]
        transfers = [
            PoolTransfer(
                name=SIDECAR_POOL,
                host_indices=None,
                keys=["block-0"],
                indices_from_pool=PoolName.KV,
            )
        ]

        broadcasts: list = []
        reduces: list = []
        with (
            mock.patch.object(
                torch.distributed, "all_gather_object", _fake_gather_all_ranks
            ),
            mock.patch.object(
                torch.distributed, "all_reduce", lambda *a, **k: reduces.append(k)
            ),
            mock.patch.object(
                torch.distributed,
                "broadcast_object_list",
                lambda object_list=None, **kwargs: broadcasts.append(kwargs),
            ),
        ):
            result = connector.batch_set_v2(transfers)

        self.assertEqual(result[SIDECAR_POOL], [False])
        self.assertEqual(
            len(broadcasts),
            1,
            "the rank skipped the session broadcast; its peer would block on it",
        )
        self.assertEqual(
            len(reduces),
            1,
            "the rank skipped the group's commit reduce; its peer would block on it",
        )
        connector.transfer_client.SaveKvCaches.assert_not_called()

    def test_an_all_cached_session_still_joins_the_commit_reduce(self) -> None:
        """The already-cached branch returns early, but its peers still wait.

        ``StartWriteCache`` can answer "all of it is already in storage", and
        that answer is the same on every rank -- but whether a peer reached this
        branch can differ, so the rank must still contribute to the group's
        reduce before returning.
        """
        connector = _data_path_connector(pools={PoolName.MAMBA: FakeHostPool()})
        connector.tp_rank = 0
        connector.tp_world_size = 2
        connector.storage_tp_group = object()
        connector._ensure_client_for_write = lambda transfers: None  # ty: ignore[invalid-assignment]
        _manager(connector).start_write_cache.return_value = {
            "locations": [],
            "write_session_id": "write-session-1",
            "block_mask": {"offset": 1},
        }
        transfer = PoolTransfer(
            name=PoolName.MAMBA, host_indices=torch.arange(PAGE_SIZE), keys=["block-0"]
        )

        reduces: list = []
        with (
            mock.patch.object(
                torch.distributed, "all_gather_object", _fake_gather_all_ranks
            ),
            mock.patch.object(
                torch.distributed, "all_reduce", lambda *a, **k: reduces.append(k)
            ),
            mock.patch.object(
                torch.distributed, "broadcast_object_list", lambda *a, **k: None
            ),
        ):
            result = connector.batch_set_v2([transfer])

        self.assertEqual(result[PoolName.MAMBA], [True])
        self.assertEqual(
            len(reduces),
            1,
            "the already-cached rank skipped the commit reduce; its peer would block",
        )

    def test_a_rank_zero_only_write_does_not_gather(self) -> None:
        """The rank-0 sidecar protocol has no peers to agree with.

        Before sglang 0.5.17 an MLA sidecar write calls ``batch_set_v2`` on rank
        0 alone, which is why the connector skips the TP agreement for it; a
        schedule gather added unconditionally would block that one rank forever.
        """
        connector = _data_path_connector(
            pools={SIDECAR_POOL: FakeHostPool(base=0x2000_0000)},
            is_mla_model=True,
            tp_rank=0,
            tp_size=2,
        )
        connector.tp_world_size = 2
        connector.storage_tp_group = object()
        connector._write_path_uses_tp_collectives = lambda transfers=None: False  # ty: ignore[invalid-assignment]
        transfers = [
            PoolTransfer(
                name=SIDECAR_POOL, host_indices=torch.arange(PAGE_SIZE), keys=["b0"]
            )
        ]

        gathers: list = []
        with (
            mock.patch.object(
                torch.distributed,
                "all_gather_object",
                lambda *a, **k: gathers.append(k),
            ),
            mock.patch.object(torch.distributed, "all_reduce", lambda *a, **k: None),
            mock.patch.object(
                torch.distributed, "broadcast_object_list", lambda *a, **k: None
            ),
        ):
            connector.batch_set_v2(transfers)

        self.assertEqual(
            gathers, [], "a rank-0-only write must not wait for peers to gather"
        )

    def test_a_transfer_only_this_rank_was_handed_is_never_written(self) -> None:
        """Peers cannot join a session they were never told about.

        A rank can be handed a transfer its peers are not -- a sidecar whose
        source pool exists in this rank's list alone, for instance.  Opening a
        write session for it would leave this rank blocked on a broadcast no
        peer enters, so the group agrees on the set first and this one is not
        written, with one warning saying why.
        """
        connector = _data_path_connector(
            pools={
                SOURCE_POOL: FakeHostPool(base=0x1000_0000),
                SIDECAR_POOL: FakeHostPool(base=0x2000_0000),
            }
        )
        connector.tp_rank = 0
        connector.tp_world_size = 2
        connector.storage_tp_group = object()
        connector._ensure_client_for_write = lambda transfers: None  # ty: ignore[invalid-assignment]
        transfers = [
            PoolTransfer(
                name=SIDECAR_POOL,
                host_indices=torch.arange(PAGE_SIZE),
                keys=["block-0"],
            )
        ]

        broadcasts: list = []
        with (
            mock.patch.object(
                torch.distributed,
                "all_gather_object",
                lambda object_list=None, obj=None, **k: _gather_this_rank_only(
                    object_list, obj
                ),
            ),
            mock.patch.object(
                torch.distributed,
                "broadcast_object_list",
                lambda object_list=None, **kwargs: broadcasts.append(kwargs),
            ),
            mock.patch.object(torch.distributed, "all_reduce", lambda *a, **k: None),
        ):
            with self.assertLogs(connector_module.__name__, "WARNING") as logs:
                result = connector.batch_set_v2(transfers)

        self.assertEqual(result[SIDECAR_POOL], [False])
        self.assertEqual(
            broadcasts, [], "this rank opened a session its peers cannot join"
        )
        connector.transfer_client.SaveKvCaches.assert_not_called()
        self.assertEqual(
            sum("only this rank was handed it" in r.getMessage() for r in logs.records),
            1,
        )

    def test_two_table_pools_do_not_break_the_schedule_agreement(self) -> None:
        """``PoolName`` members are not orderable, and derived names are strings.

        Sorting the local transfer names raw raises ``TypeError`` on two enum
        members, which the outer handler turns into all-``False`` -- a silent
        loss of every multi-sidecar write, single rank included.  The agreed
        sequence carries ``str()`` of each name (see ``_collective_identity``),
        which sorts and compares as one kind for both spellings.
        """
        connector = _data_path_connector(
            pools={
                PoolName.MAMBA: FakeHostPool(base=0x1000_0000),
                PoolName.INDEXER: FakeHostPool(base=0x2000_0000),
                SIDECAR_POOL: FakeHostPool(base=0x3000_0000),
            }
        )
        connector._ensure_client_for_write = lambda transfers: None  # ty: ignore[invalid-assignment]
        transfers = [
            PoolTransfer(name=PoolName.MAMBA, host_indices=None, keys=["b0"]),
            PoolTransfer(name=PoolName.INDEXER, host_indices=None, keys=["b0"]),
            PoolTransfer(name=SIDECAR_POOL, host_indices=None, keys=["b0"]),
        ]

        agreed = connector._agree_collective_transfers(transfers)

        self.assertEqual(agreed, ["indexer", "mamba", SIDECAR_POOL])

    def test_a_derived_pool_is_replicated_on_every_mla_rank(self) -> None:
        """Every rank reads a sidecar, so a replicated row reads rank 0's copy.

        ``_page_transfer_sidecar`` has no MLA gate: non-zero ranks read the
        sidecar too, and the group's hit count is the minimum over them.  A
        derived row is replicated (the registration carries no sharing mode), so
        rank 0's spec is the copy every rank should read.  A pool that really is
        sharded belongs in the table, where its row says so.
        """
        connector = _data_path_connector(
            pools={SIDECAR_POOL: FakeHostPool(base=0x2000_0000)},
            is_mla_model=True,
            tp_rank=1,
            tp_size=2,
        )
        sidecar_name = connector.pool_specs[SIDECAR_POOL].name
        self.assertTrue(
            sidecar_name.startswith("tp_0"),
            f"a derived row is replicated, so it resolves to rank 0: {sidecar_name}",
        )
        _manager(connector).get_cache_location.return_value = {
            # The KV spec has to be there too: a location without it does not
            # count as a hit page, so the sidecar would be capped at zero.
            "locations": _locations({sidecar_name: {0}, "tp_0_full": {0}}, 1)
        }
        transfers = [
            PoolTransfer(
                name=SIDECAR_POOL,
                host_indices=torch.arange(PAGE_SIZE),
                keys=["block-0"],
            )
        ]

        self.assertEqual(connector.batch_get_v2(transfers), {SIDECAR_POOL: [True]})
        connector.transfer_client.LoadKvCaches.assert_called_once()

        result = connector.batch_exists_v2(["block-0"], transfers)
        self.assertEqual(
            result.extra_pool_hit_pages.get(SIDECAR_POOL, 0),
            1,
            "rank 0's copy is the one every rank reads, so it is a hit here too",
        )

    def test_sidecar_without_a_resolvable_source_reports_a_miss(self) -> None:
        """The KV anchor is not part of a v2 call: answer a miss, never raise."""
        connector = self._connector()
        transfers = [
            PoolTransfer(
                name=SIDECAR_POOL,
                host_indices=None,
                keys=["block-0"],
                indices_from_pool=PoolName.KV,
            )
        ]

        self.assertEqual(connector.batch_get_v2(transfers), {SIDECAR_POOL: [False]})
        self.assertEqual(connector.batch_set_v2(transfers), {SIDECAR_POOL: [False]})
        connector.transfer_client.LoadKvCaches.assert_not_called()
        connector.transfer_client.SaveKvCaches.assert_not_called()

    def test_sidecar_source_without_indices_is_not_used(self) -> None:
        """A source that carries no indices itself resolves to nothing."""
        connector = self._connector()
        transfers = [
            PoolTransfer(name=SOURCE_POOL, host_indices=None, keys=["block-0"]),
            PoolTransfer(
                name=SIDECAR_POOL,
                host_indices=None,
                keys=["block-0"],
                indices_from_pool=SOURCE_POOL,
            ),
        ]

        self.assertEqual(connector.batch_get_v2(transfers)[SIDECAR_POOL], [False])
        connector.transfer_client.LoadKvCaches.assert_not_called()


if __name__ == "__main__":
    unittest.main()
