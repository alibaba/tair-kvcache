"""Deterministic (CPU-only) tests for the host pool registration with PACE.

The connector hands every host pool's shm segment to the PACE singleton linked
into the process, which is what lets RDMA gather/scatter the pool's DRAM
directly instead of bouncing through a staging copy.  The C++ side of that
passthrough (``register_host_memory`` & co) exists only in builds with the
mempool SDK linked in, so what can be pinned here is the connector's half of the
contract, against a fake pybind module and *real* shm mappings:

* one segment per pool is not enough -- an allocator appends one segment per
  allocation and the pool's ``fd`` is only the first of them, so every
  ``(fd, mm)`` pair is registered and its ``len(mm)`` (the page-aligned mapping,
  not the tensor's ``nbytes``) is what the size argument carries;
* a pool without shm segments, and a client without the passthrough, cost that
  pool (or the whole feature) only: everything else is registered as usual;
* a failing fd is reported with its pool, fd, size, base and return code, and
  neither its sibling segments nor the other pools are affected;
* ``close()`` deregisters every registered fd *before* dropping the segment
  references -- dropping them frees the allocator, whose ``__del__`` closes the
  fds, and a closed fd can no longer be deregistered.  A second ``close()`` is a
  no-op;
* the registration happens once, from the lazy init, after the SDK client
  exists, and it is gated on the same ``"allocator": "shm"`` key the engine
  reads.

Prerequisites: a sglang runtime that can import its own backend (``sgl_kernel``
and friends) and ``kv_cache_manager`` importable from the repository root on
``sys.path`` -- the compiled kvcm pybind client and the build-generated
``_version_info`` are stubbed below, so no wheel is needed for them.

    PYTHONPATH=$PWD python kv_cache_manager/py_connector/sglang/test_pace_registration.py
"""

import ctypes
import mmap
import os
import sys
import tempfile
import types
import unittest
import weakref
from contextlib import contextmanager
from typing import Any
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
# The step-5 pool set: the KV anchor plus the two sidecars that are shm-backed
# in that deployment.  Enum members (not strings) so the per-pool log lookups
# and the connector's own keys agree.  ``STEP4_DSA_SUMMARY`` is resolved with
# ``getattr`` like every version-dependent member in these tests: the internal
# build has it, the releases the type gate pins do not.
STEP5_POOLS = (
    PoolName.KV,
    PoolName.SWA,
    getattr(PoolName, "STEP4_DSA_SUMMARY", PoolName.DRAFT),
)


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
    the fds, which is why the connector has to keep these objects alive until
    it has deregistered the segments.
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


def fake_client(
    *,
    passthrough: bool = True,
    disabled: str = "",
    on_register: Any = None,
    on_deregister: Any = None,
) -> Any:
    """A stand-in for the compiled module, with or without the PACE passthrough.

    The PACE names are absent -- not ``None`` -- when ``passthrough`` is false,
    which is what an open-source build or an older client looks like: the
    connector must find nothing there.  Everything else (``InitParams``,
    ``RoleType``, ``TransferClient``) comes from the shared mock so that the
    lazy-init path can still run.
    """
    client: Any = types.ModuleType("kvcm_py_client")
    client.InitParams = _mock_kvcm.InitParams
    client.RoleType = _mock_kvcm.RoleType
    client.TransferClient = _mock_kvcm.TransferClient
    if not passthrough:
        return client
    client.register_host_memory = MagicMock(
        side_effect=on_register or (lambda fd, size, base: 0)
    )
    client.deregister_host_memory = MagicMock(
        side_effect=on_deregister or (lambda fd: 0)
    )
    client.external_zero_copy_disable_reason = MagicMock(return_value=disabled)
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


def make_connector(*, allocator: Any = "shm") -> HiCacheKVCM:
    """A connector with a mocked Manager; its pool set is filled by the test."""
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
    connector._manager_client.register_instance.return_value = {"storage_configs": "[]"}
    connector._manager_client.get_cache_location.return_value = {"locations": []}
    return connector


def address_of(mm: Any) -> int:
    """Where an mmap starts, computed the way the connector computes it."""
    return ctypes.addressof(ctypes.c_char.from_buffer(mm))


def summary_line(logs: Any) -> str:
    """The one frozen summary line of a registration run."""
    lines = [
        record.getMessage()
        for record in logs.records
        if record.getMessage().startswith("PACE zero-copy: pools=")
    ]
    assert len(lines) == 1, f"expected exactly one summary line, got {lines}"
    return lines[0]


class TestPaceRegistration(unittest.TestCase):
    """What a successful registration hands to PACE, and what it logs."""

    def test_every_pool_segment_is_registered(self) -> None:
        """Three pools, one segment each: one call per segment, counts match."""
        connector = make_connector()
        allocators = {
            pool: ShmAllocator(mmap.PAGESIZE * (i + 1))
            for i, pool in enumerate(STEP5_POOLS)
        }
        for pool, allocator in allocators.items():
            connector.registered_pools[pool] = FakeHostPool(allocator)

        client = fake_client()
        with patched_client(client):
            with self.assertLogs(connector_module.__name__, level="INFO") as logs:
                connector._register_pace_host_pools()

        expected = [
            mock.call(allocator.fd, mmap.PAGESIZE * (i + 1), address_of(allocator.mm))
            for i, allocator in enumerate(allocators.values())
        ]
        self.assertEqual(client.register_host_memory.call_args_list, expected)
        self.assertEqual(
            [region[:2] for region in connector._pace_regions],
            [(allocator.fd, allocator.mm) for allocator in allocators.values()],
        )
        self.assertEqual(
            summary_line(logs),
            "PACE zero-copy: pools=3 regions=3 registered=3 skipped_pools=0 disabled=-",
        )

    def test_a_pool_that_allocated_twice_registers_both_segments(self) -> None:
        """The allocator appends one segment per allocation; ``fd`` is the first."""
        connector = make_connector()
        allocator = ShmAllocator(mmap.PAGESIZE, mmap.PAGESIZE * 2)
        connector.registered_pools[PoolName.KV] = FakeHostPool(allocator)

        client = fake_client()
        with patched_client(client):
            with self.assertLogs(connector_module.__name__, level="INFO") as logs:
                connector._register_pace_host_pools()

        self.assertEqual(
            client.register_host_memory.call_args_list,
            [
                mock.call(fd, len(mm), address_of(mm))
                for fd, mm in zip(allocator.fds, allocator.mms)
            ],
        )
        self.assertNotEqual(allocator.fds[0], allocator.fds[1])
        self.assertEqual(len(connector._pace_regions), 2)
        self.assertEqual(
            summary_line(logs),
            "PACE zero-copy: pools=1 regions=2 registered=2 skipped_pools=0 disabled=-",
        )

    def test_registered_but_disabled_is_reported(self) -> None:
        """PACE saying zero-copy is off is the acceptance criterion: say it."""
        connector = make_connector()
        connector.registered_pools[PoolName.KV] = FakeHostPool(ShmAllocator(4096))

        client = fake_client(disabled="external zero-copy disabled by env")
        with patched_client(client):
            with self.assertLogs(connector_module.__name__, level="INFO") as logs:
                connector._register_pace_host_pools()

        self.assertEqual(
            summary_line(logs),
            "PACE zero-copy: pools=1 regions=1 registered=1 skipped_pools=0 "
            "disabled=external zero-copy disabled by env",
        )
        warnings = [r.getMessage() for r in logs.records if r.levelname == "WARNING"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("external zero-copy unavailable", warnings[0])
        self.assertIn("external zero-copy disabled by env", warnings[0])


class TestPaceDegradation(unittest.TestCase):
    """Every skip is local: it costs that pool (or the feature), nothing else."""

    def test_a_pool_without_shm_segments_is_skipped(self) -> None:
        """A non-shm allocator (no ``fds``/``mms``) does not fail the run."""
        connector = make_connector()
        connector.registered_pools[PoolName.KV] = FakeHostPool(ShmAllocator(4096))
        connector.registered_pools[PoolName.SWA] = FakeHostPool(
            types.SimpleNamespace(dtype=torch.bfloat16)
        )
        connector.registered_pools[STEP5_POOLS[2]] = FakeHostPool(ShmAllocator(8192))

        client = fake_client()
        with patched_client(client):
            with self.assertLogs(connector_module.__name__, level="INFO") as logs:
                connector._register_pace_host_pools()

        self.assertEqual(client.register_host_memory.call_count, 2)
        warnings = [r.getMessage() for r in logs.records if r.levelname == "WARNING"]
        self.assertEqual(len(warnings), 1)
        self.assertIn(str(PoolName.SWA), warnings[0])
        self.assertEqual(
            summary_line(logs),
            "PACE zero-copy: pools=2 regions=2 registered=2 skipped_pools=1 disabled=-",
        )

    def test_a_non_shm_backend_skips_the_whole_feature(self) -> None:
        """The gate is the engine's own key; missing it means no shm pools."""
        for allocator in ("default", None):
            with self.subTest(allocator=allocator):
                connector = make_connector(allocator=allocator)
                connector.registered_pools[PoolName.KV] = FakeHostPool(
                    ShmAllocator(4096)
                )

                client = fake_client()
                with patched_client(client):
                    with self.assertLogs(
                        connector_module.__name__, level="WARNING"
                    ) as logs:
                        connector._register_pace_host_pools()

                self.assertEqual(client.register_host_memory.call_count, 0)
                self.assertEqual(len(logs.records), 1)
                self.assertIn("allocator", logs.records[0].getMessage())
                self.assertIn("shm", logs.records[0].getMessage())
                self.assertEqual(connector._pace_regions, [])

    def test_a_client_without_the_passthrough_is_skipped(self) -> None:
        """An open-source or older client has no such name: warn once, move on."""
        connector = make_connector()
        allocator = ShmAllocator(4096)
        connector.registered_pools[PoolName.KV] = FakeHostPool(allocator)

        client = fake_client(passthrough=False)
        self.assertIsNone(getattr(client, "register_host_memory", None))
        with patched_client(client):
            with self.assertLogs(connector_module.__name__, level="WARNING") as logs:
                connector._register_pace_host_pools()

        self.assertEqual(len(logs.records), 1)
        self.assertIn("PACE passthrough", logs.records[0].getMessage())
        self.assertEqual(connector._pace_regions, [])
        # Nothing was lost: the segments are still the pool's to keep.
        for fd in allocator.fds:
            os.fstat(fd)

    def test_a_failing_segment_is_located_and_the_rest_proceeds(self) -> None:
        """A non-zero return code names the fd and costs only that segment."""
        connector = make_connector()
        good = ShmAllocator(4096)
        bad = ShmAllocator(8192)
        connector.registered_pools[PoolName.KV] = FakeHostPool(good)
        connector.registered_pools[PoolName.SWA] = FakeHostPool(bad)
        failing_fd = bad.fds[0]

        client = fake_client(
            on_register=lambda fd, size, base: 22 if fd == failing_fd else 0
        )
        with patched_client(client):
            with self.assertLogs(connector_module.__name__, level="INFO") as logs:
                connector._register_pace_host_pools()

        warnings = [r.getMessage() for r in logs.records if r.levelname == "WARNING"]
        self.assertEqual(len(warnings), 1)
        for part in (str(PoolName.SWA), f"fd={failing_fd}", "size=8192", "ret=22"):
            self.assertIn(part, warnings[0])
        self.assertIn(f"base={address_of(bad.mm):#x}", warnings[0])
        self.assertEqual(
            [region[0] for region in connector._pace_regions], [good.fds[0]]
        )
        self.assertEqual(
            summary_line(logs),
            "PACE zero-copy: pools=2 regions=2 registered=1 skipped_pools=0 disabled=-",
        )

        # close() deregisters the registered segments only, and does not raise
        # for the one PACE refused.
        with patched_client(fake_client()):
            connector.close()
        self.assertEqual(connector._pace_regions, [])

    def test_an_unregisterable_mapping_is_reported_not_raised(self) -> None:
        """A mapping ctypes cannot address costs that segment, nothing more."""
        connector = make_connector()
        connector.registered_pools[PoolName.KV] = FakeHostPool(
            types.SimpleNamespace(fds=[-1], mms=[object()])
        )

        client = fake_client()
        with patched_client(client):
            with self.assertLogs(connector_module.__name__, level="INFO") as logs:
                connector._register_pace_host_pools()

        self.assertEqual(client.register_host_memory.call_count, 0)
        warnings = [r.getMessage() for r in logs.records if r.levelname == "WARNING"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("fd=-1", warnings[0])
        self.assertEqual(connector._pace_regions, [])


class TestPaceRetry(unittest.TestCase):
    """A retried initialization must not hand the same segment to PACE twice."""

    def test_registering_twice_does_not_register_again(self) -> None:
        connector = make_connector()
        connector.registered_pools[PoolName.KV] = FakeHostPool(ShmAllocator(4096, 8192))

        client = fake_client()
        with patched_client(client):
            connector._register_pace_host_pools()
            regions = [region[:2] for region in connector._pace_regions]
            connector._register_pace_host_pools()

        self.assertEqual(client.register_host_memory.call_count, len(regions))
        self.assertEqual([region[:2] for region in connector._pace_regions], regions)

    def test_an_unreadable_state_keeps_the_registration_tearable(self) -> None:
        """A raising state getter is not a lost registration: keep it tearable."""
        connector = make_connector()
        allocator = ShmAllocator(4096, 8192)
        connector.registered_pools[PoolName.KV] = FakeHostPool(allocator)

        client = fake_client()
        client.external_zero_copy_disable_reason = MagicMock(
            side_effect=RuntimeError("state unavailable")
        )
        with patched_client(client):
            with self.assertLogs(connector_module.__name__, level="INFO") as logs:
                connector._register_pace_host_pools()
                # What a retried initialization does: nothing left to register.
                connector._register_pace_host_pools()

        self.assertEqual(client.register_host_memory.call_count, 2)
        self.assertEqual(len(connector._pace_regions), 2)
        self.assertEqual(
            summary_line(logs),
            "PACE zero-copy: pools=1 regions=2 registered=2 skipped_pools=0 "
            "disabled=unknown (state unavailable)",
        )
        warnings = [r.getMessage() for r in logs.records if r.levelname == "WARNING"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("could not be read", warnings[0])

        deregistered: list = []
        with patched_client(
            fake_client(on_deregister=lambda fd: deregistered.append(fd) or 0)
        ):
            connector.close()

        self.assertEqual(deregistered, allocator.fds)
        self.assertEqual(connector._pace_regions, [])


class TestPaceTeardown(unittest.TestCase):
    """close() must deregister before the fds can be closed under it."""

    def _initialized(self) -> tuple:
        connector = make_connector()
        allocator = ShmAllocator(4096, 8192)
        connector.registered_pools[PoolName.KV] = FakeHostPool(allocator)
        fds = list(allocator.fds)
        alive = weakref.ref(allocator)

        client = fake_client()
        with patched_client(client):
            connector._register_pace_host_pools()
        self.assertEqual(len(connector._pace_regions), 2)

        # Only the connector's own references keep the segments alive now; the
        # weakref is how the test sees them go.
        del connector.registered_pools[PoolName.KV]
        return connector, alive, fds

    def test_close_deregisters_before_dropping_the_segments(self) -> None:
        connector, alive, fds = self._initialized()
        open_at_deregister: list = []

        def record(fd: int) -> int:
            try:
                os.fstat(fd)
                open_at_deregister.append(True)
            except OSError:
                open_at_deregister.append(False)
            return 0

        with patched_client(fake_client(on_deregister=record)):
            connector.close()

        self.assertEqual(
            open_at_deregister, [True] * len(fds), "deregistered a closed fd"
        )
        self.assertIsNone(alive(), "the segment references outlived deregistration")
        self.assertEqual(connector._pace_regions, [])
        for fd in fds:
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_close_is_idempotent(self) -> None:
        connector, _alive, fds = self._initialized()
        client = fake_client()

        with patched_client(client):
            connector.close()
            connector.close()

        self.assertEqual(
            [call.args[0] for call in client.deregister_host_memory.call_args_list],
            fds,
        )
        self.assertEqual(connector._pace_regions, [])

    def test_a_failing_deregistration_does_not_raise(self) -> None:
        """close() is best-effort: PACE refusing is a warning, not an exception."""
        connector, _alive, fds = self._initialized()

        client = fake_client(on_deregister=lambda fd: 9)
        with patched_client(client):
            with self.assertLogs(connector_module.__name__, level="WARNING") as logs:
                connector.close()

        self.assertEqual(client.deregister_host_memory.call_count, len(fds))
        self.assertEqual(len(logs.records), len(fds))
        self.assertEqual(connector._pace_regions, [])


class TestPaceWiring(unittest.TestCase):
    """The registration is part of the lazy init, after the client exists."""

    def test_the_lazy_init_registers_after_the_client_exists(self) -> None:
        connector = make_connector()
        allocator = ShmAllocator(4096)
        observed: list = []

        def on_register(fd: int, size: int, base: int) -> int:
            observed.append(connector.transfer_client)
            return 0

        client = fake_client(on_register=on_register)
        with patched_client(client), patched_parallel_context():
            connector.register_mem_pool_host(FakeHostPool(allocator))
            with self.assertLogs(connector_module.__name__, level="INFO") as logs:
                connector.batch_exists(["block-0"])

        self.assertEqual(
            len(observed), 1, "the registration never ran during the lazy init"
        )
        self.assertIsNotNone(observed[0], "registered before the SDK client existed")
        self.assertEqual(client.register_host_memory.call_count, 1)
        self.assertEqual(client.register_host_memory.call_args.args[0], allocator.fd)
        self.assertEqual(
            summary_line(logs),
            "PACE zero-copy: pools=1 regions=1 registered=1 skipped_pools=0 disabled=-",
        )
        self.assertTrue(connector._client_ready)


if __name__ == "__main__":
    unittest.main()
