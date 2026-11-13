"""Deterministic (CPU-only) tests for the host pool spans handed to the client.

The host pools sglang allocates with the shm allocator are ordinary anonymous
shared memory to the backend, so every KVCM transfer bounces the block through a
staging copy.  The backend can DMA a registered region directly, and the client
carries the regions in ``InitParams.regist_spans``: this connector assembles the
list before ``TransferClient.Create`` and the backend registers each span in its
own Init.  The connector never calls into the backend, so what can be pinned here
is its half of that contract, against a fake pybind module and *real* shm
mappings:

* one span per segment, not one per pool -- an allocator appends a segment per
  allocation and the pool's ``fd`` is only the first of them -- and the size is
  the mapping's own page-aligned length, not the tensor's ``nbytes``;
* every span keeps its mapping and allocator alive through ``owner``: the backend
  holds raw addresses into them until the client is destroyed, and the mappings
  must outlive it (``close()`` drops the client first, the spans after);
* a pool without shm segments, and a client without the channel (open-source
  build, older wheel), cost that pool or the whole feature only -- nothing
  raises, the other pools are handed over as usual, and the deployment keeps
  today's behaviour;
* the spans are assembled once, from the lazy init, before the client exists, and
  the summary line counts pools, segments, handed spans and skipped pools.

Prerequisites: a sglang runtime that can import its own backend (``sgl_kernel``
and friends) and ``kv_cache_manager`` importable from the repository root on
``sys.path`` -- the compiled kvcm pybind client and the build-generated
``_version_info`` are stubbed below, so no wheel is needed for them.

    PYTHONPATH=$PWD python kv_cache_manager/py_connector/sglang/test_pace_registration.py
"""

import ctypes
import json
import mmap
import os
import sys
import tempfile
import types
import unittest
import weakref
from contextlib import contextmanager
from typing import Any, List
from unittest import mock
from unittest.mock import MagicMock

import torch

# ── Modules the connector imports but this test does not exercise ──────
# Same technique as test_pool_registration.py: the compiled pybind client and
# the build-generated version module must be importable before the connector
# module can be loaded.
_mock_pybind: Any = types.ModuleType("kv_cache_manager.client.pybind")
_mock_kvcm = MagicMock()
_mock_pybind.kvcm_py_client = _mock_kvcm
sys.modules.setdefault("kv_cache_manager.client.pybind", _mock_pybind)

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
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache  # noqa: E402

from kv_cache_manager.py_connector.sglang import connector as connector_module  # noqa: E402
from kv_cache_manager.py_connector.sglang.connector import HiCacheKVCM  # noqa: E402

PAGE_SIZE = 64
KV_BYTES_PER_TOKEN = 512
# The hybrid pool set: the KV anchor plus the two sidecars that are shm-backed
# in that deployment.  Enum members (not strings) so the per-pool log lookups
# and the connector's own keys agree.  ``STEP4_DSA_SUMMARY`` is resolved with
# ``getattr`` like every version-dependent member in these tests: the releases
# with that sidecar have it, the ones the type gate pins do not.
HYBRID_POOLS = (
    PoolName.KV,
    PoolName.SWA,
    getattr(PoolName, "STEP4_DSA_SUMMARY", PoolName.DRAFT),
)


class FakeRegistSpan:
    """The four fields of ``kvcm_py_client.RegistSpan``; the client reads them."""

    def __init__(self) -> None:
        self.base = 0
        self.size = 0
        self.fd = -1
        self.owner = None

    def __repr__(self) -> str:
        return f"RegistSpan(fd={self.fd}, size={self.size}, base={self.base:#x})"


class FakeInitParams:
    """``kvcm_py_client.InitParams`` with the host-span channel."""

    def __init__(self) -> None:
        self.role_type = None
        self.regist_span = None
        self.regist_spans: list = []
        self.self_location_spec_name = ""
        self.storage_configs = ""


class LegacyInitParams:
    """``InitParams`` of an open-source build: no span list to fill in."""

    def __init__(self) -> None:
        self.role_type = None
        self.regist_span = None
        self.self_location_spec_name = ""
        self.storage_configs = ""


def fake_client(*, spans: bool = True) -> Any:
    """A stand-in for the compiled module, with or without the span channel.

    ``spans=False`` is what an open-source build or an older wheel looks like:
    neither ``RegistSpan`` nor ``InitParams.regist_spans`` exists (absent, not
    ``None``), and the connector must find nothing there.  The rest
    (``InitParams``, ``RoleType``, ``TransferClient``) comes from the shared mock
    so that the lazy-init path can still run.
    """
    client: Any = types.ModuleType("kvcm_py_client")
    client.InitParams = FakeInitParams if spans else LegacyInitParams
    client.RoleType = _mock_kvcm.RoleType
    client.TransferClient = _mock_kvcm.TransferClient
    if spans:
        client.RegistSpan = FakeRegistSpan
    return client


@contextmanager
def patched_client(client: Any):
    """Install ``client`` where the connector looks the pybind module up."""
    pybind: Any = types.ModuleType("kv_cache_manager.client.pybind")
    pybind.kvcm_py_client = client
    with (
        mock.patch.object(connector_module, "kvcm_py_client", client),
        mock.patch.dict(sys.modules, {"kv_cache_manager.client.pybind": pybind}),
    ):
        yield client


@contextmanager
def patched_parallel_context():
    """Neutralize the collective parts of ``_init_parallel_context``."""
    tp_group = types.SimpleNamespace(cpu_group=None)
    with (
        mock.patch.object(connector_module, "get_tp_group", lambda: tp_group),
        mock.patch.object(connector_module, "get_attn_tp_group", lambda: tp_group),
        mock.patch.object(connector_module, "is_dp_attention_enabled", lambda: False),
        mock.patch.object(torch.distributed, "get_world_size", lambda group=None: 1),
    ):
        yield


# What the Manager answers with: only a Tair mempool backend holds host spans,
# so the default deployment here has one and the "no consumer" cases swap in a
# storage set that has none.
TAIR_STORAGE = {
    "type": "pace",
    "is_available": True,
    "global_unique_name": "pace-1",
    "storage_spec": {"local_hostname": "test-host"},
}
TAIR_SSD_STORAGE = dict(TAIR_STORAGE, type="pace_ssd", global_unique_name="pace-ssd-1")
HF3FS_STORAGE = {
    "type": "hf3fs",
    "is_available": True,
    "global_unique_name": "hf3fs-1",
    "storage_spec": {"mountpoint": "/mnt/hf3fs", "root_dir": "/kv"},
}
NFS_STORAGE = {
    "type": "file",
    "is_available": True,
    "global_unique_name": "nfs-1",
    "storage_spec": {"root_path": "/mnt/nfs"},
}


def make_connector(
    *, allocator: Any = "shm", storage_configs: Any = None
) -> HiCacheKVCM:
    """A connector with a mocked Manager; its pool set is filled by the test.

    ``storage_configs`` is what the Manager returns for this instance.  It
    defaults to a single Tair mempool backend -- the deployment the host spans
    exist for.
    """
    configs = [TAIR_STORAGE] if storage_configs is None else storage_configs
    extra_config = {
        "manager_uri": "http://127.0.0.1:6382",
        "instance_group": "default",
        "instance_id": "pace-test",
    }
    if allocator is not None:
        extra_config["allocator"] = allocator
    connector = HiCacheKVCM(
        HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="test-model",
            extra_config=extra_config,
        ),
        {},
    )
    connector._manager_client = MagicMock()
    connector._manager_client.register_instance.return_value = {
        "storage_configs": json.dumps(configs)
    }
    connector._manager_client.get_cache_location.return_value = {"locations": []}
    connector.storage_configs = json.dumps(configs)
    return connector


def address_of(mm: Any) -> int:
    """Where an mmap starts, computed the way the connector computes it."""
    return ctypes.addressof(ctypes.c_char.from_buffer(mm))


def summary_line(logs: Any) -> str:
    """The one frozen summary line of a span-assembly run."""
    lines = [
        record.getMessage()
        for record in logs.records
        if record.getMessage().startswith("host zero-copy: pools=")
    ]
    assert len(lines) == 1, f"expected exactly one summary line, got {lines}"
    return lines[0]


def warnings_of(logs: Any) -> List[str]:
    return [r.getMessage() for r in logs.records if r.levelname == "WARNING"]


def _new_shm_fd(size: int) -> int:
    """An shm file descriptor of ``size`` bytes, the way sglang allocates one.

    ``os.memfd_create`` is not in every CPython build (sglang falls back the same
    way), so the fd comes from ``/dev/shm`` when it is missing.
    """
    create = getattr(os, "memfd_create", None)
    if create is None:
        fd, path = tempfile.mkstemp(prefix="kvcm-pace-test-", dir="/dev/shm")
        os.unlink(path)
    else:
        fd = create("kvcm-pace-test")
    os.ftruncate(fd, size)
    return fd


class ShmAllocator:
    """The parts of ``ShmHostTensorAllocator`` the connector reads, on real shm.

    One segment per ``allocate()``, exactly like sglang: every allocation adds an
    fd and a mapping, and ``fd`` is only the first of them.  ``__del__`` closes
    the fds, which is why the connector has to keep these objects alive for as
    long as its client holds addresses into them.
    """

    def __init__(self, *sizes: int) -> None:
        self.fds: list = []
        self.mms: list = []
        for size in sizes:
            fd = _new_shm_fd(size)
            self.fds.append(fd)
            self.mms.append(
                mmap.mmap(
                    fd,
                    size,
                    flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
            )

    @property
    def fd(self) -> Any:
        return self.fds[0] if self.fds else None

    @property
    def mm(self) -> Any:
        return self.mms[0] if self.mms else None

    def __del__(self) -> None:
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds = []


class FakeHostPool(HostKVCache):
    """A host pool carrying an allocator; nothing else is read here."""

    def __init__(self, allocator: Any, *, bytes_per_token: int = KV_BYTES_PER_TOKEN):
        self.allocator = allocator
        self.dtype = torch.bfloat16
        self.page_size = PAGE_SIZE
        self.bytes_per_token = bytes_per_token

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


def attach(connector: HiCacheKVCM, client: Any) -> Any:
    """Run the span assembly the lazy init runs, and return its ``InitParams``."""
    connector.init_params = client.InitParams()
    with patched_client(client):
        connector._attach_host_pool_spans()
    return connector.init_params


class TestHostPoolSpans(unittest.TestCase):
    """What a pool set turns into, and what the summary line counts."""

    def test_every_pool_segment_is_handed_over(self) -> None:
        """Three pools, one segment each: three spans, one per segment."""
        connector = make_connector()
        allocators = {
            pool: ShmAllocator(mmap.PAGESIZE * (i + 1))
            for i, pool in enumerate(HYBRID_POOLS)
        }
        for pool, allocator in allocators.items():
            connector.registered_pools[pool] = FakeHostPool(allocator)

        client = fake_client()
        with self.assertLogs(connector_module.__name__, level="INFO") as logs:
            init_params = attach(connector, client)

        self.assertEqual(
            [
                (span.fd, span.size, span.base, span.owner)
                for span in init_params.regist_spans
            ],
            [
                (
                    allocator.fd,
                    len(allocator.mm),
                    address_of(allocator.mm),
                    (allocator.mm, allocator),
                )
                for allocator in allocators.values()
            ],
        )
        self.assertEqual(init_params.regist_spans, connector._host_pool_spans)
        self.assertEqual(
            summary_line(logs),
            "host zero-copy: pools=3 segments=3 handed=3 skipped_pools=0 "
            "skipped_segments=0 skipped_no_backend=0",
        )

    def test_a_pool_that_allocated_twice_hands_over_both_segments(self) -> None:
        """The allocator appends a segment per allocation; ``fd`` is the first."""
        connector = make_connector()
        allocator = ShmAllocator(mmap.PAGESIZE, mmap.PAGESIZE * 2)
        connector.registered_pools[PoolName.KV] = FakeHostPool(allocator)

        client = fake_client()
        with self.assertLogs(connector_module.__name__, level="INFO") as logs:
            init_params = attach(connector, client)

        self.assertNotEqual(allocator.fds[0], allocator.fds[1])
        self.assertGreater(len(init_params.regist_spans), 1)
        self.assertEqual(
            [(span.fd, span.size, span.base) for span in init_params.regist_spans],
            [
                (fd, len(mm), address_of(mm))
                for fd, mm in zip(allocator.fds, allocator.mms)
            ],
        )
        self.assertEqual(
            summary_line(logs),
            "host zero-copy: pools=1 segments=2 handed=2 skipped_pools=0 "
            "skipped_segments=0 skipped_no_backend=0",
        )

    def test_the_size_is_the_mapping_not_the_tensor(self) -> None:
        """A segment carries the page-aligned mapping length, not ``nbytes``."""
        connector = make_connector()
        allocator = ShmAllocator(3 * mmap.PAGESIZE)
        pool = FakeHostPool(allocator)
        connector.registered_pools[PoolName.KV] = pool

        client = fake_client()
        with self.assertLogs(connector_module.__name__, level="INFO"):
            init_params = attach(connector, client)

        block_bytes = pool.get_size_per_token() * pool.page_size
        span = init_params.regist_spans[0]
        self.assertEqual(span.size, len(allocator.mm))
        self.assertNotEqual(span.size, block_bytes)


class TestHostPoolSpanDegradation(unittest.TestCase):
    """Every skip is local: it costs that pool (or the feature), nothing else."""

    def test_a_pool_without_shm_segments_is_skipped(self) -> None:
        """A non-shm allocator (no ``fds``/``mms``) does not fail the run."""
        connector = make_connector()
        connector.registered_pools[PoolName.KV] = FakeHostPool(ShmAllocator(4096))
        connector.registered_pools[PoolName.SWA] = FakeHostPool(
            types.SimpleNamespace(dtype=torch.bfloat16)
        )
        connector.registered_pools[HYBRID_POOLS[2]] = FakeHostPool(ShmAllocator(8192))

        client = fake_client()
        with self.assertLogs(connector_module.__name__, level="INFO") as logs:
            init_params = attach(connector, client)

        self.assertEqual(len(init_params.regist_spans), 2)
        warnings = warnings_of(logs)
        self.assertEqual(len(warnings), 1)
        self.assertIn(str(PoolName.SWA), warnings[0])
        self.assertEqual(
            summary_line(logs),
            "host zero-copy: pools=2 segments=2 handed=2 skipped_pools=1 "
            "skipped_segments=0 skipped_no_backend=0",
        )

    def test_a_non_shm_backend_hands_over_nothing(self) -> None:
        """The gate is the engine's own key; missing it means no shm pools."""
        for allocator in ("default", None):
            with self.subTest(allocator=allocator):
                connector = make_connector(allocator=allocator)
                connector.registered_pools[PoolName.KV] = FakeHostPool(
                    ShmAllocator(4096)
                )

                client = fake_client()
                with self.assertLogs(
                    connector_module.__name__, level="WARNING"
                ) as logs:
                    init_params = attach(connector, client)

                self.assertEqual(len(logs.records), 1)
                self.assertIn("allocator", logs.records[0].getMessage())
                self.assertIn("shm", logs.records[0].getMessage())
                self.assertIn("no host segment", logs.records[0].getMessage())
                # Today's behaviour: the field is not set, the client is unchanged.
                self.assertEqual(init_params.regist_spans, [])
                self.assertEqual(connector._host_pool_spans, [])

    def test_a_client_without_the_span_channel_is_not_handed_anything(self) -> None:
        """An open-source or older client takes no spans: warn once, move on."""
        connector = make_connector()
        allocator = ShmAllocator(4096)
        connector.registered_pools[PoolName.KV] = FakeHostPool(allocator)

        client = fake_client(spans=False)
        self.assertFalse(hasattr(client, "RegistSpan"))
        with self.assertLogs(connector_module.__name__, level="WARNING") as logs:
            init_params = attach(connector, client)

        self.assertEqual(len(logs.records), 1)
        self.assertIn("host memory spans", logs.records[0].getMessage())
        # Not set at all, not set to None: an older InitParams has no field.
        self.assertFalse(hasattr(init_params, "regist_spans"))
        self.assertEqual(connector._host_pool_spans, [])
        # Nothing was lost: the segments are still the pool's to keep.
        for fd in allocator.fds:
            os.fstat(fd)

    def test_a_segment_that_cannot_be_addressed_costs_only_itself(self) -> None:
        """A mapping ctypes cannot address is reported and left out."""
        connector = make_connector()
        connector.registered_pools[PoolName.KV] = FakeHostPool(ShmAllocator(4096))
        connector.registered_pools[PoolName.SWA] = FakeHostPool(
            types.SimpleNamespace(fds=[-1], mms=[object()])
        )

        client = fake_client()
        with self.assertLogs(connector_module.__name__, level="INFO") as logs:
            init_params = attach(connector, client)

        self.assertEqual(len(init_params.regist_spans), 1)
        warnings = warnings_of(logs)
        self.assertEqual(len(warnings), 1)
        self.assertIn("fd=-1", warnings[0])
        self.assertIn(str(PoolName.SWA), warnings[0])
        self.assertEqual(
            summary_line(logs),
            "host zero-copy: pools=2 segments=2 handed=1 skipped_pools=0 "
            "skipped_segments=1 skipped_no_backend=0",
        )


class TestHostPoolSpanHealth(unittest.TestCase):
    """A segment the backend would refuse is left out, never handed over.

    The backend's Init gives up on the first segment it cannot take and
    ``TransferClient.Create`` then returns no client at all -- that costs every
    pool its storage, not just the broken segment -- so each segment is checked
    on this side first.
    """

    def test_a_closed_fd_is_skipped(self) -> None:
        """A closed fd is not something the backend could dup."""
        connector = make_connector()
        good = ShmAllocator(4096)
        dead = ShmAllocator(8192)
        connector.registered_pools[PoolName.KV] = FakeHostPool(good)
        connector.registered_pools[PoolName.SWA] = FakeHostPool(dead)
        os.close(dead.fds[0])

        client = fake_client()
        with self.assertLogs(connector_module.__name__, level="INFO") as logs:
            init_params = attach(connector, client)

        self.assertEqual([span.fd for span in init_params.regist_spans], [good.fds[0]])
        warnings = warnings_of(logs)
        self.assertEqual(len(warnings), 1)
        self.assertIn(str(PoolName.SWA), warnings[0])
        self.assertIn(f"fd={dead.fds[0]}", warnings[0])
        self.assertIn("Bad file descriptor", warnings[0])
        self.assertEqual(
            summary_line(logs),
            "host zero-copy: pools=2 segments=2 handed=1 skipped_pools=0 "
            "skipped_segments=1 skipped_no_backend=0",
        )

    def test_a_segment_whose_fd_no_longer_matches_the_mapping_is_skipped(
        self,
    ) -> None:
        """The fd's size is what the backend registers; a mismatch is a defect."""
        connector = make_connector()
        allocator = ShmAllocator(8192)
        connector.registered_pools[PoolName.KV] = FakeHostPool(allocator)
        os.ftruncate(allocator.fds[0], 2 * len(allocator.mm))

        client = fake_client()
        with self.assertLogs(connector_module.__name__, level="INFO") as logs:
            init_params = attach(connector, client)

        self.assertEqual(init_params.regist_spans, [])
        warnings = warnings_of(logs)
        self.assertEqual(len(warnings), 1)
        self.assertIn(f"fd={allocator.fds[0]}", warnings[0])
        self.assertIn(f"the mapping {len(allocator.mm)}", warnings[0])
        self.assertEqual(
            summary_line(logs),
            "host zero-copy: pools=1 segments=1 handed=0 skipped_pools=0 "
            "skipped_segments=1 skipped_no_backend=0",
        )

    def test_one_bad_segment_does_not_take_its_sibling(self) -> None:
        """A pool that allocated twice keeps the segment that is still usable."""
        connector = make_connector()
        allocator = ShmAllocator(4096, 8192)
        connector.registered_pools[PoolName.KV] = FakeHostPool(allocator)
        os.close(allocator.fds[1])

        client = fake_client()
        with self.assertLogs(connector_module.__name__, level="INFO") as logs:
            init_params = attach(connector, client)

        self.assertEqual(
            [span.fd for span in init_params.regist_spans], [allocator.fds[0]]
        )
        self.assertEqual(len(warnings_of(logs)), 1)
        self.assertEqual(
            summary_line(logs),
            "host zero-copy: pools=1 segments=2 handed=1 skipped_pools=0 "
            "skipped_segments=1 skipped_no_backend=0",
        )

    def test_every_segment_broken_still_creates_a_client(self) -> None:
        """Nothing usable is an empty list, not a failed initialization."""
        connector = make_connector()
        allocator = ShmAllocator(4096, 8192)
        os.close(allocator.fds[0])
        os.ftruncate(allocator.fds[1], 4 * len(allocator.mms[1]))
        handed: list = []
        create = _mock_kvcm.TransferClient.Create

        def record(client_config: Any, init_params: Any) -> Any:
            handed.append(list(init_params.regist_spans))
            return create(client_config, init_params)

        with (
            patched_client(fake_client()),
            patched_parallel_context(),
            mock.patch.object(_mock_kvcm.TransferClient, "Create", side_effect=record),
        ):
            connector.register_mem_pool_host(FakeHostPool(allocator))
            with self.assertLogs(connector_module.__name__, level="INFO") as logs:
                connector.batch_exists(["block-0"])

        self.assertEqual(handed, [[]], "the client was not created")
        self.assertTrue(connector._client_ready)
        self.assertEqual(len(warnings_of(logs)), 2)
        self.assertEqual(
            summary_line(logs),
            "host zero-copy: pools=1 segments=2 handed=0 skipped_pools=0 "
            "skipped_segments=2 skipped_no_backend=0",
        )
        self.assertEqual(connector._host_pool_spans, [])


class TestStorageTypeGate(unittest.TestCase):
    """Only a Tair mempool backend consumes the spans, so only it gets them.

    Every other backend leaves them untouched, while the client still dups each
    fd and holds it for the instance's lifetime -- so a deployment without a
    mempool backend pays for spans it cannot use, and a dup at the fd limit
    takes the whole storage client down with it.
    """

    def test_the_gate_reads_the_type_of_every_config(self) -> None:
        """The judgement itself, on the JSON shapes the Manager produces."""
        has = connector_module._has_tair_mempool_storage
        for configs, expected in [
            ([TAIR_STORAGE], True),
            ([TAIR_SSD_STORAGE], True),
            ([HF3FS_STORAGE, NFS_STORAGE], False),
            ([NFS_STORAGE, TAIR_STORAGE], True),
            ([], False),
            (json.dumps([TAIR_STORAGE]), True),
            ([{"type": "pace"}, 7, None], True),
            ([{"global_unique_name": "no-type"}], False),
            ("not json", None),
            (None, None),
            (7, None),
        ]:
            with self.subTest(configs=configs):
                self.assertIs(has(configs), expected)

    def test_no_tair_backend_hands_over_nothing(self) -> None:
        """An HF3FS/NFS-only deployment is a normal one: INFO, no warning."""
        connector = make_connector(storage_configs=[HF3FS_STORAGE, NFS_STORAGE])
        connector.registered_pools[PoolName.KV] = FakeHostPool(ShmAllocator(4096))
        connector.registered_pools[PoolName.SWA] = FakeHostPool(ShmAllocator(8192))
        handed: list = []
        create = _mock_kvcm.TransferClient.Create

        def record(client_config: Any, init_params: Any) -> Any:
            handed.append(list(init_params.regist_spans))
            return create(client_config, init_params)

        with (
            patched_client(fake_client()),
            patched_parallel_context(),
            mock.patch.object(_mock_kvcm.TransferClient, "Create", side_effect=record),
        ):
            connector.register_mem_pool_host(FakeHostPool(ShmAllocator(4096)))
            with self.assertLogs(connector_module.__name__, level="INFO") as logs:
                connector.batch_exists(["block-0"])

        self.assertEqual(handed, [[]], "the spans were handed over without a consumer")
        self.assertTrue(connector._client_ready, "the client was not created")
        self.assertEqual(warnings_of(logs), [])
        self.assertEqual(
            summary_line(logs),
            "host zero-copy: pools=2 segments=2 handed=0 skipped_pools=0 "
            "skipped_segments=0 skipped_no_backend=2",
        )
        self.assertEqual(connector._host_pool_spans, [])

    def test_a_tair_backend_keeps_every_segment(self) -> None:
        """The deployment the spans exist for is unchanged."""
        connector = make_connector()
        allocators = {
            pool: ShmAllocator(mmap.PAGESIZE * (i + 1))
            for i, pool in enumerate(HYBRID_POOLS)
        }
        for pool, allocator in allocators.items():
            connector.registered_pools[pool] = FakeHostPool(allocator)

        client = fake_client()
        with self.assertLogs(connector_module.__name__, level="INFO") as logs:
            init_params = attach(connector, client)

        self.assertEqual(
            [span.fd for span in init_params.regist_spans],
            [allocator.fd for allocator in allocators.values()],
        )
        self.assertEqual(
            summary_line(logs),
            "host zero-copy: pools=3 segments=3 handed=3 skipped_pools=0 "
            "skipped_segments=0 skipped_no_backend=0",
        )

    def test_a_mixed_storage_set_still_hands_the_spans_over(self) -> None:
        """The criterion is "a mempool backend exists", not "only one exists"."""
        for configs in (
            [NFS_STORAGE, TAIR_STORAGE],
            [TAIR_SSD_STORAGE, HF3FS_STORAGE],
        ):
            with self.subTest(configs=[c["type"] for c in configs]):
                connector = make_connector(storage_configs=configs)
                allocator = ShmAllocator(4096, 8192)
                connector.registered_pools[PoolName.KV] = FakeHostPool(allocator)

                client = fake_client()
                with self.assertLogs(connector_module.__name__, level="INFO") as logs:
                    init_params = attach(connector, client)

                self.assertEqual(
                    [span.fd for span in init_params.regist_spans], allocator.fds
                )
                self.assertEqual(
                    summary_line(logs),
                    "host zero-copy: pools=1 segments=2 handed=2 skipped_pools=0 "
                    "skipped_segments=0 skipped_no_backend=0",
                )

    def test_an_unreadable_storage_set_hands_over_nothing(self) -> None:
        """Not being able to show a consumer is treated as having none.

        Handing the spans to a client that cannot be shown to consume them can
        only reproduce the bug this gate fixes; the other way round, the feature
        is off and the warning says so.  A list this connector cannot read is
        also one ``SdkWrapper::Init`` rejects outright, so nothing that works
        today is disabled here.
        """
        connector = make_connector()
        connector.storage_configs = "not json"
        connector.registered_pools[PoolName.KV] = FakeHostPool(ShmAllocator(4096))

        client = fake_client()
        with self.assertLogs(connector_module.__name__, level="INFO") as logs:
            init_params = attach(connector, client)

        self.assertEqual(init_params.regist_spans, [])
        warnings = warnings_of(logs)
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("cannot be read", warnings[0])
        self.assertEqual(
            summary_line(logs),
            "host zero-copy: pools=1 segments=1 handed=0 skipped_pools=0 "
            "skipped_segments=0 skipped_no_backend=1",
        )


class TestHostPoolSpanLifetime(unittest.TestCase):
    """The spans outlive the client; nothing else may own the mappings."""

    def _initialized(self) -> tuple:
        """A connector holding the only references to a pool's two mappings."""
        connector = make_connector()
        allocator = ShmAllocator(4096, 8192)
        connector.registered_pools[PoolName.KV] = FakeHostPool(allocator)
        fds = list(allocator.fds)
        mappings = [weakref.ref(mm) for mm in allocator.mms]
        allocator_ref = weakref.ref(allocator)

        client = fake_client()
        with self.assertLogs(connector_module.__name__, level="INFO"):
            attach(connector, client)
        self.assertEqual(len(connector._host_pool_spans), 2)

        # Only the spans' ``owner`` keeps the segments alive now.
        del connector.registered_pools[PoolName.KV]
        del allocator
        return connector, allocator_ref, mappings, fds

    def test_the_owner_keeps_the_segments_alive(self) -> None:
        connector, allocator_ref, mappings, fds = self._initialized()

        self.assertIsNotNone(allocator_ref(), "the segments were not kept alive")
        for reference in mappings:
            self.assertIsNotNone(reference(), "a mapping was collected")
        for fd in fds:
            os.fstat(fd)

    def test_close_drops_the_client_before_the_segments(self) -> None:
        """The spans go after the client: the backend holds addresses into them."""
        connector, allocator_ref, mappings, fds = self._initialized()
        alive_when_client_died: list = []

        class SentinelClient:
            def __del__(self) -> None:
                alive_when_client_died.append(
                    [reference() is not None for reference in mappings]
                )

        connector.transfer_client = SentinelClient()
        with patched_client(fake_client()):
            connector.close()

        self.assertEqual(connector._host_pool_spans, [])
        self.assertEqual(
            alive_when_client_died,
            [[True, True]],
            "the segments were dropped before the client that points at them",
        )
        self.assertIsNone(allocator_ref(), "the segment references outlived close()")
        for reference in mappings:
            self.assertIsNone(reference(), "a mapping outlived close()")
        for fd in fds:
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_close_is_idempotent(self) -> None:
        connector, allocator_ref, _mappings, _fds = self._initialized()

        with patched_client(fake_client()):
            connector.close()
            connector.close()

        self.assertEqual(connector._host_pool_spans, [])
        self.assertIsNone(allocator_ref())


class TestHostPoolSpanWiring(unittest.TestCase):
    """The spans are assembled inside the lazy init, before the client exists."""

    def test_the_lazy_init_hands_the_spans_over_before_create(self) -> None:
        connector = make_connector()
        allocator = ShmAllocator(4096)
        handed: list = []
        create = _mock_kvcm.TransferClient.Create

        def record(client_config: Any, init_params: Any) -> Any:
            handed.append(list(init_params.regist_spans))
            return create(client_config, init_params)

        client = fake_client()
        with (
            patched_client(client),
            patched_parallel_context(),
            mock.patch.object(_mock_kvcm.TransferClient, "Create", side_effect=record),
        ):
            connector.register_mem_pool_host(FakeHostPool(allocator))
            with self.assertLogs(connector_module.__name__, level="INFO") as logs:
                connector.batch_exists(["block-0"])

        self.assertEqual(len(handed), 1, "the client was never created")
        self.assertEqual(len(handed[0]), 1, "the client was created without spans")
        self.assertEqual(
            (handed[0][0].fd, handed[0][0].size, handed[0][0].base),
            (allocator.fd, len(allocator.mm), address_of(allocator.mm)),
        )
        self.assertEqual(
            summary_line(logs),
            "host zero-copy: pools=1 segments=1 handed=1 skipped_pools=0 "
            "skipped_segments=0 skipped_no_backend=0",
        )
        self.assertTrue(connector._client_ready)


if __name__ == "__main__":
    unittest.main()
