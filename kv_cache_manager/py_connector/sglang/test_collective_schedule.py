"""Deterministic (CPU-only) tests for the order the write collectives run in.

``batch_set_v2`` opens a write session per transfer and folds a commit flag per
transfer, and both are collectives over the TP group: ranks have to enter them
for the same transfers and in the same order, and a rank that skips one leaves
its peers waiting forever.  Agreeing on the *set* of transfer names is not
enough -- it says nothing about the order the ranks iterate them in, nor about
which rank is able to write a given transfer.

Covered:

* two ranks handed the same transfers in opposite orders run the collectives in
  one agreed sequence, and no rank writes one pool's bytes into another pool's
  session;
* a rank with no keys for a transfer still enters its collectives and
  contributes "wrote nothing", so its peers finish with a miss instead of
  blocking; the transfer filters nothing out of the sequence, only out of the
  data;
* whether a write runs collectives at all cannot depend on the transfers the
  call was handed: a rank handed no transfers for the call its peers were handed
  a sharded transfer for still enters the rendezvous, and the sequence drops the
  pool no rank reported instead of waiting for a peer that is not in the call;
* a transfer only rank 0 receives (the MLA rank-0 protocol, and the pre-0.5.17
  sidecar protocol) enters no collective at all -- the agreement must not turn
  that path into a group rendezvous;
* one pool listed twice (a ``PoolName`` member and its value text) stays one
  schedule entry and one session instead of dropping out of the sequence.

Each rank runs on its own thread against a fake group whose collectives need
both ranks, so a rank that skips one shows up as a broken barrier and a
conservative answer rather than as a test that quietly passes.

Prerequisites: the connector needs a sglang runtime that can actually import its
backend (``sgl_kernel`` and friends), and ``kv_cache_manager`` must be
importable -- either from an installed wheel or from the repository root on
``sys.path``.  The compiled kvcm pybind client and the build-generated
``_version_info`` are stubbed below, so no wheel is needed for those.

    PYTHONPATH=$PWD python kv_cache_manager/py_connector/sglang/test_collective_schedule.py
"""

import sys
import threading
import types
import unittest
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Mapping, Optional
from unittest import mock
from unittest.mock import MagicMock

import torch


# ── Modules the connector imports but this test does not exercise ──────
# Same technique as test_extra_pools.py / test_derived_pools.py.
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
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache  # noqa: E402

from kv_cache_manager.py_connector.sglang import connector as connector_module  # noqa: E402
from kv_cache_manager.py_connector.sglang.connector import HiCacheKVCM  # noqa: E402

PAGE_SIZE = 64
PAGE_BYTES = 4096
PAGE_LIMIT = 8
PAGES = 3
KEYS = [f"block-{i}" for i in range(PAGES)]

# The two sidecar pools of a hybrid stack, in the two shapes that matter here:
# MAMBA is a table row (sharded on MLA, a plain sidecar otherwise), SWA a pool
# the connector derives a spec for.  Neither is the KV anchor, which a v2 call
# never carries (the v1 hook writes it).
MAMBA: Any = PoolName.MAMBA
SWA: Any = getattr(PoolName, "SWA", "swa")
# A replicated table row: written by rank 0 alone on MLA, i.e. the rank-0
# protocol the schedule must not turn into a group rendezvous.
INDEXER: Any = PoolName.INDEXER

# One distinct base address per pool, so a saved buffer says which pool it
# belongs to.  The KV anchor has one too (it is registered on every instance,
# even though a v2 call never carries it).  "mamba-str" is the derived pool whose
# *text* is also "mamba": two pools, two bases.
POOL_BASES = {
    "kv": 0x0100_0000,
    str(MAMBA): 0x1000_0000,
    "mamba-str": 0x1800_0000,
    str(SWA): 0x2000_0000,
    str(INDEXER): 0x3000_0000,
}


class FakeHostPool(HostKVCache):
    """A sglang host pool without CUDA: spec metadata plus page buffers.

    One IOV per page, and its base address identifies the pool -- the only way
    the test can tell whose bytes a rank handed to a session.
    """

    def __init__(self, *, label: str, page_bytes: int = PAGE_BYTES) -> None:
        self.label = label
        self.base = POOL_BASES[label]
        self.page_size = PAGE_SIZE
        self.page_bytes = page_bytes
        self.dtype = torch.bfloat16

    def get_size_per_token(self) -> int:
        return self.page_bytes // self.page_size

    def get_page_buffer_meta(self, indices: torch.Tensor) -> tuple[list, list]:
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


def _pool_label(pool: Any) -> str:
    return str(pool)


class FakeGroup:
    """The TP group of a run: two ranks, one thread each, and the collectives.

    Every collective blocks until all ranks arrive, like gloo's do.  A rank that
    skips one therefore does not pass silently: its peer waits ``timeout`` and
    then raises, which is exactly the failure the connector turns into its
    conservative all-``False`` answer.
    """

    def __init__(self, world_size: int = 2, timeout: float = 5.0) -> None:
        self.world_size = world_size
        self.timeout = timeout
        self._barrier = threading.Barrier(world_size, timeout=timeout)
        self._lock = threading.Lock()
        self._thread_rank = threading.local()
        self._reports: Dict[int, Any] = {}
        self._flags: Dict[int, int] = {}
        self._payload: Any = None
        # (op, detail) per rank, in the order that rank entered them.
        self.ops: Dict[int, List[tuple]] = {rank: [] for rank in range(world_size)}
        # (pools the session's URIs belong to, pools the buffers belong to).
        self.saves: Dict[int, List[tuple]] = {rank: [] for rank in range(world_size)}

    # -- the ranks --------------------------------------------------─
    def _rank(self) -> int:
        rank = getattr(self._thread_rank, "value", None)
        assert rank is not None, "a collective ran outside a rank thread"
        return rank

    def run(
        self, rank: int, work: Callable[[], Any]
    ) -> tuple[threading.Thread, Dict[str, Any]]:
        """Run one rank's work on its own thread, tagged for the fakes."""
        outcome: Dict[str, Any] = {}

        def target() -> None:
            self._thread_rank.value = rank
            try:
                outcome["result"] = work()
            except BaseException as error:  # noqa: BLE001 - asserted by the test
                outcome["error"] = error

        thread = threading.Thread(target=target, name=f"tp-rank-{rank}", daemon=True)
        thread.start()
        return thread, outcome

    @contextmanager
    def patched(self):
        """Route torch's TP collectives into this group."""
        with (
            mock.patch.object(
                torch.distributed, "all_gather_object", self.all_gather_object
            ),
            mock.patch.object(
                torch.distributed, "broadcast_object_list", self.broadcast_object_list
            ),
            mock.patch.object(torch.distributed, "all_reduce", self.all_reduce),
        ):
            yield

    # -- the collectives --------------------------------------------─
    def all_gather_object(
        self, object_list: Any = None, obj: Any = None, group: Any = None
    ) -> None:
        rank = self._rank()
        with self._lock:
            self._reports[rank] = list(obj or [])
            self.ops[rank].append(("gather", _reported_names(obj)))
        self._barrier.wait()
        gathered = [self._reports[other] for other in range(self.world_size)]
        if object_list is not None:
            object_list[:] = [list(report) for report in gathered]

    def broadcast_object_list(
        self, object_list: Any = None, group_src: int = 0, group: Any = None
    ) -> None:
        rank = self._rank()
        if rank == group_src:
            self._payload = object_list[0]
        self._barrier.wait()
        payload = self._payload
        with self._lock:
            self.ops[rank].append(("broadcast", _session_of(payload)))
        if object_list is not None:
            object_list[:] = [payload]

    def all_reduce(self, tensor: Any, op: Any = None, group: Any = None) -> None:
        rank = self._rank()
        with self._lock:
            self._flags[rank] = int(tensor.item())
        self._barrier.wait()
        folded = min(self._flags[other] for other in range(self.world_size))
        with self._lock:
            self.ops[rank].append(("reduce", folded))
        tensor.fill_(folded)

    # -- what a rank handed to the SDK ------------------------------─
    def note_save(self, rank: int, uris: List[str], buffers: List[Any]) -> tuple:
        with self._lock:
            self.saves[rank].append((_pools_of_uris(uris), _pools_of_buffers(buffers)))
        return (_mock_kvcm.ClientErrorCode.ER_OK,)


def _session_of(payload: Any) -> Optional[str]:
    """The write session a broadcast carried, or None for "no session"."""
    return None if payload is None else payload["write_session_id"]


def _reported_names(report: Any) -> tuple:
    """The pools a rank reported to the agreement, as readable ``kind:text``.

    Read off the report without assuming its exact shape, so a schedule test
    fails on the collective order it cares about rather than on the payload.
    """
    names = []
    for entry in report or []:
        identity = entry[0] if isinstance(entry, (tuple, list)) else entry
        names.append(
            ":".join(str(part) for part in identity)
            if isinstance(identity, (tuple, list))
            else str(identity)
        )
    return tuple(names)


def _pools_of_uris(uris: List[str]) -> frozenset:
    """Which pools the URIs of a session belong to (``uri-<pool>-<spec>-<page>``)."""
    return frozenset(uri.split("-")[1] for uri in uris)


def _pools_of_buffers(buffers: List[Any]) -> frozenset:
    """Which pools' host memory these block buffers point into."""
    labels = set()
    for buffer in buffers:
        for iov in buffer.iovs:
            for label, base in POOL_BASES.items():
                if base <= iov.base < base + PAGE_BYTES * PAGE_LIMIT:
                    labels.add(label)
                    break
            else:
                raise AssertionError(f"IOV outside every fake pool: {iov.base:#x}")
    return frozenset(labels)


def _spec_index(
    connector: HiCacheKVCM, labels: Mapping[Any, str]
) -> Mapping[str, tuple]:
    """Spec group -> (pool label, the spec names of every rank) for ``labels``.

    A write session is opened per spec group, and the Manager's answer carries
    the spec of *every* rank (each rank picks its own out of it), so a session
    is attributable to one pool -- which is what makes a cross-pool pairing
    visible in the assertions below.  The label is passed in rather than taken
    from the name: two pools may share the same text (see the identity test).
    """
    index = {}
    for pool, label in labels.items():
        registered = connector.pool_specs[pool]
        index[registered.spec.group] = (label, registered.names)
    return index


def _start_write_cache(index: Mapping[str, tuple]) -> Callable[[dict], dict]:
    """A ``start_write_cache`` that opens one session per requested group."""

    def start(request: dict) -> dict:
        group = request["location_spec_group_names"][0]
        pool, spec_names = index[group]
        return {
            "locations": [
                {
                    "location_specs": [
                        {"name": spec, "uri": f"uri-{pool}-{spec}-{page}"}
                        for spec in spec_names
                    ]
                }
                for page in range(len(request["block_keys"]))
            ],
            "write_session_id": f"session-{pool}",
            "block_mask": {"offset": 0},
        }

    return start


def _connector(
    *,
    pools: Dict[Any, Any],
    world: FakeGroup,
    tp_rank: int = 0,
    is_mla_model: bool = False,
    tp_init_agreed: bool = True,
) -> HiCacheKVCM:
    """A connector with only the fields the v2 write path read.

    The pool set is resolved through the real ``_resolve_pool_specs``, so the
    specs, sizes and groups are the ones an attached instance would use.
    ``_tp_init_agreed`` is set by default because the initialization agreement
    (one ``all_reduce`` per call until it succeeds) is covered by
    sglang/test_tp_init_sync.py -- then every reduce belongs to a transfer; the
    rendezvous test turns it off to watch the *first* collective of a write, the
    one a rank reaching it alone waits in.
    """
    connector = HiCacheKVCM.__new__(HiCacheKVCM)
    connector.tp_rank = tp_rank
    connector.tp_size = world.world_size
    connector.tp_world_size = world.world_size
    connector.is_mla_model = is_mla_model
    connector.block_size = PAGE_SIZE
    connector.instance_id = "collective-schedule-test"
    connector.write_timeout_seconds = 30
    connector.registered_pools = {PoolName.KV: FakeHostPool(label="kv"), **pools}
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
    connector.transfer_client.LoadKvCaches.return_value = (
        _mock_kvcm.ClientErrorCode.ER_OK
    )
    connector.transfer_client.SaveKvCaches.side_effect = lambda uris, buffers: (
        world.note_save(tp_rank, uris, buffers)
    )
    # The group handle is opaque to the connector; the collectives are patched.
    connector.storage_tp_group = world
    connector._client_ready = True
    connector._tp_init_agreed = tp_init_agreed
    return connector


def _manager(connector: HiCacheKVCM) -> Any:
    """The mocked Manager client of a connector built here."""
    return connector._manager_client


def _transfers(order: List[Any], *, empty: Optional[List[Any]] = None) -> List[Any]:
    """One ``PoolTransfer`` per pool, in ``order``; ``empty`` ones carry no keys."""
    empty = empty if empty is not None else []
    return [
        PoolTransfer(
            name=pool,
            host_indices=None if pool in empty else torch.arange(PAGE_SIZE * PAGES),
            keys=[] if pool in empty else list(KEYS),
        )
        for pool in order
    ]


def _run_ranks(world: FakeGroup, workloads: Mapping[int, Callable[[], Any]]) -> Dict:
    """Run each rank on its own thread and return ``{rank: result}``.

    A rank that is still inside a collective after its peer finished never
    returns; that is a failure of the code under test, so it is reported here
    instead of hanging the test run.
    """
    threads: Dict[int, threading.Thread] = {}
    outcomes: Dict[int, Dict[str, Any]] = {}
    for rank, work in workloads.items():
        threads[rank], outcomes[rank] = world.run(rank, work)
    for rank, thread in threads.items():
        thread.join(world.timeout + 5)
        if thread.is_alive():
            raise AssertionError(
                f"tp rank {rank} never returned: it is waiting in a collective "
                "its peer skipped"
            )
    errors = {
        rank: outcome["error"]
        for rank, outcome in outcomes.items()
        if "error" in outcome
    }
    if errors:
        raise AssertionError(f"a tp rank raised: {errors}")
    return {rank: outcome["result"] for rank, outcome in outcomes.items()}


class TestAgreedSequence(unittest.TestCase):
    """The collectives run in one sequence, and only where they are wanted."""

    def setUp(self) -> None:
        HiCacheKVCM._reported_pools.clear()

    def test_opposite_local_orders_run_one_agreed_sequence(self) -> None:
        """Ranks handed the transfers in opposite orders still pair them up.

        The intersection of the two name sets is the same set either way, but
        each rank iterated its own list: rank 0 would open the mamba session
        while rank 1 took that broadcast for the swa pool and handed the swa
        bytes to it -- the two ranks would write each other's sessions.  The
        group agrees on one sequence instead, so the n-th broadcast is the same
        transfer everywhere.
        """
        world = FakeGroup()
        pools = {
            MAMBA: FakeHostPool(label=str(MAMBA)),
            SWA: FakeHostPool(label=str(SWA)),
        }
        rank0 = _connector(pools=pools, world=world, tp_rank=0)
        rank1 = _connector(pools=pools, world=world, tp_rank=1)
        index = _spec_index(rank0, {MAMBA: "mamba", SWA: "swa"})
        _manager(rank0).start_write_cache.side_effect = _start_write_cache(index)

        with world.patched():
            results = _run_ranks(
                world,
                {
                    # Opposite local orders, and the agreed sequence is the
                    # sorted one -- which here is rank 0's.
                    0: lambda: rank0.batch_set_v2(_transfers([MAMBA, SWA])),
                    1: lambda: rank1.batch_set_v2(_transfers([SWA, MAMBA])),
                },
            )

        agreed = [
            (frozenset({_pool_label(pool)}), frozenset({_pool_label(pool)}))
            for pool in (MAMBA, SWA)
        ]
        self.assertEqual(world.saves[0], agreed, "rank 0's own order is the agreed one")
        self.assertEqual(
            world.saves[1],
            agreed,
            "rank 1 followed the agreed sequence instead of its own list",
        )
        self.assertEqual(
            [detail for op, detail in world.ops[0] if op == "broadcast"],
            [f"session-{_pool_label(pool)}" for pool in (MAMBA, SWA)],
        )
        self.assertEqual(
            world.ops[0],
            world.ops[1],
            "the ranks entered different collectives, or in a different order",
        )
        for rank in (0, 1):
            self.assertEqual(
                results[rank],
                {MAMBA: [True] * PAGES, SWA: [True] * PAGES},
                "every page of both pools was written on both ranks",
            )

    def test_a_rank_without_keys_still_joins_the_collectives(self) -> None:
        """An empty transfer is "wrote nothing", not a reason to skip a peer.

        ``batch_set_v2`` promises every rank reaches both collectives of a
        cooperative transfer, but a rank with no keys used to return before the
        session handshake: its peers then waited on a broadcast nobody would
        join.  It joins now with nothing to write, the group's minimum makes the
        blocks a miss, and no rank is left waiting.
        """
        world = FakeGroup()
        pools = {
            MAMBA: FakeHostPool(label=str(MAMBA)),
            SWA: FakeHostPool(label=str(SWA)),
        }
        rank0 = _connector(pools=pools, world=world, tp_rank=0)
        rank1 = _connector(pools=pools, world=world, tp_rank=1)
        _manager(rank0).start_write_cache.side_effect = _start_write_cache(
            _spec_index(rank0, {MAMBA: "mamba", SWA: "swa"})
        )

        with self.assertNoLogs(connector_module.__name__, level="ERROR"):
            with world.patched():
                results = _run_ranks(
                    world,
                    {
                        0: lambda: rank0.batch_set_v2(_transfers([MAMBA, SWA])),
                        # The swa transfer carries no keys on rank 1.
                        1: lambda: rank1.batch_set_v2(
                            _transfers([MAMBA, SWA], empty=[SWA])
                        ),
                    },
                )

        self.assertEqual(
            [op for op, _detail in world.ops[1]],
            ["gather", "broadcast", "reduce", "broadcast", "reduce"],
            "the rank with no keys skipped the collectives of that transfer",
        )
        self.assertEqual(
            world.ops[0], world.ops[1], "the ranks entered different collectives"
        )
        self.assertEqual(
            results[0][MAMBA], [True] * PAGES, "the other pool is unaffected"
        )
        self.assertEqual(results[1][MAMBA], [True] * PAGES)
        self.assertEqual(
            results[1][SWA], [], "a transfer without keys has nothing to report"
        )
        self.assertEqual(
            results[0][SWA],
            [False] * PAGES,
            "the group committed a pool only one rank wrote nothing of",
        )
        self.assertEqual(
            world.saves[1],
            [(frozenset({_pool_label(MAMBA)}), frozenset({_pool_label(MAMBA)}))],
            "the rank with no keys for swa wrote no data for it",
        )
        self.assertEqual(
            world.saves[0],
            [
                (frozenset({_pool_label(pool)}), frozenset({_pool_label(pool)}))
                for pool in (MAMBA, SWA)
            ],
            "rank 0 wrote its own slice; the group's commit is what refuses it",
        )

    def test_the_rank_zero_protocol_enters_no_collective(self) -> None:
        """A transfer only rank 0 receives must not become a rendezvous.

        On MLA sglang's own ``should_backup`` filters a replicated pool to rank
        0, which then writes it alone.  Gathering here -- or opening the
        transfer as if it were cooperative -- would wait for a peer that is
        never called, which is why the agreed sequence keeps that path out.
        """
        world = FakeGroup()
        rank0 = _connector(
            pools={INDEXER: FakeHostPool(label=str(INDEXER))},
            world=world,
            tp_rank=0,
            is_mla_model=True,
        )
        _manager(rank0).start_write_cache.side_effect = _start_write_cache(
            _spec_index(rank0, {INDEXER: "indexer"})
        )

        results = rank0.batch_set_v2(_transfers([INDEXER]))

        self.assertEqual(results, {INDEXER: [True] * PAGES})
        self.assertEqual(
            world.ops[0], [], "the rank-0 protocol entered a TP collective"
        )
        self.assertEqual(
            world.saves[0],
            [(frozenset({_pool_label(INDEXER)}), frozenset({_pool_label(INDEXER)}))],
        )

    def test_an_empty_transfer_list_still_enters_the_rendezvous(self) -> None:
        """Whether a write runs collectives cannot depend on its own list.

        A rank with no pages to back up is handed no transfers for the same call
        its peers are handed a sharded one, and judging the write from that list
        split the group: the rank with the transfer waited in the initialization
        ``all_reduce`` for a peer that answered "no collectives here" and left.
        The gate now reads only instance-wide facts, so both ranks answer the
        same and both enter; the agreed sequence then drops the transfer no rank
        reported, and each side reports a miss instead of waiting.
        """
        world = FakeGroup()
        rank0 = _connector(
            pools={MAMBA: FakeHostPool(label=str(MAMBA))},
            world=world,
            tp_rank=0,
            is_mla_model=True,
            tp_init_agreed=False,
        )
        rank1 = _connector(
            pools={MAMBA: FakeHostPool(label=str(MAMBA))},
            world=world,
            tp_rank=1,
            is_mla_model=True,
            tp_init_agreed=False,
        )
        # The gate itself, before any collective: one rank has the transfer, the
        # other has nothing, and both have to answer the same.
        self.assertTrue(rank0._write_path_uses_tp_collectives(_transfers([MAMBA])))
        self.assertTrue(rank1._write_path_uses_tp_collectives([]))

        with self.assertNoLogs(connector_module.__name__, level="ERROR"):
            with world.patched():
                results = _run_ranks(
                    world,
                    {
                        0: lambda: rank0.batch_set_v2(_transfers([MAMBA])),
                        1: lambda: rank1.batch_set_v2([]),
                    },
                )

        self.assertEqual(
            [op for op, _detail in world.ops[0]],
            ["reduce", "gather"],
            "the rank with the transfer skipped the rendezvous",
        )
        self.assertEqual(
            [op for op, _detail in world.ops[1]],
            ["reduce", "gather"],
            "the rank with no transfers skipped the rendezvous its peer waits in",
        )
        self.assertEqual(
            results[0],
            {MAMBA: [False] * PAGES},
            "no peer reported the sharded transfer, so nothing is written",
        )
        self.assertEqual(results[1], {})
        self.assertEqual(world.saves[0], [], "a session nobody joins was opened")

    def test_the_old_sidecar_protocol_runs_no_collective(self) -> None:
        """Before sglang 0.5.17 only rank 0 is called for sidecars.

        The other half of the gate: a build whose sidecars are handed to rank 0
        alone must not rendezvous, or rank 0 waits for ranks that are never
        called.  The v1 KV write keeps its own answer as well (rank-0 only on
        MLA, every rank otherwise).
        """
        world = FakeGroup()
        rank0 = _connector(
            pools={MAMBA: FakeHostPool(label=str(MAMBA))},
            world=world,
            tp_rank=0,
            is_mla_model=True,
        )
        transfer = _transfers([MAMBA])
        _manager(rank0).start_write_cache.side_effect = _start_write_cache(
            _spec_index(rank0, {MAMBA: "mamba"})
        )

        with mock.patch.object(
            connector_module, "_sidecars_are_written_by_every_rank", lambda: False
        ):
            self.assertFalse(rank0._write_path_uses_tp_collectives(transfer))
            results = rank0.batch_set_v2(transfer)

        self.assertEqual(results, {MAMBA: [True] * PAGES})
        self.assertEqual(
            world.ops[0], [], "the rank-0 protocol entered a TP collective"
        )
        self.assertEqual(
            world.saves[0],
            [(frozenset({_pool_label(MAMBA)}), frozenset({_pool_label(MAMBA)}))],
        )
        # The v1 KV write (`pool_transfers is None`) is rank-0 only on MLA and
        # group-wide otherwise; neither changed.
        self.assertFalse(rank0._write_path_uses_tp_collectives(None))
        self.assertTrue(
            _connector(
                pools={MAMBA: FakeHostPool(label=str(MAMBA))},
                world=world,
                is_mla_model=False,
            )._write_path_uses_tp_collectives(None)
        )

    def test_the_two_spellings_of_one_pool_stay_one_schedule_entry(self) -> None:
        """A pool named twice is still one pool, and it keeps its entry.

        ``PoolName`` is a ``str`` enum, so ``PoolName.MAMBA`` and the text
        ``"mamba"`` are one dictionary key: ``registered_pools`` holds one pool,
        ``_resolve_pool_specs`` derives no second row for the text (a table row
        already claims it), and sglang itself reads a result back as
        ``results.get(name)`` then ``results.get(name.value)``.  Counting the
        gathered reports per *occurrence* therefore broke the pool: a rank that
        listed it under both spellings counted twice, no entry reached "every
        rank reported it", and the pool silently dropped out of the sequence --
        written by nobody.  Counting per rank keeps it, once, with one session.
        """
        world = FakeGroup()
        pool = FakeHostPool(label="mamba")
        # One registration, under either spelling: the same dictionary slot.
        pools = {MAMBA: pool}
        rank0 = _connector(pools=pools, world=world, tp_rank=0)
        rank1 = _connector(pools=pools, world=world, tp_rank=1)
        self.assertEqual(len(rank0.registered_pools), 2, "KV anchor plus mamba")
        self.assertIs(rank0.pool_specs["mamba"], rank0.pool_specs[MAMBA])
        _manager(rank0).start_write_cache.side_effect = _start_write_cache(
            _spec_index(rank0, {MAMBA: "mamba"})
        )
        # Both spellings in one call, on both ranks.
        order = [MAMBA, "mamba"]

        with self.assertNoLogs(connector_module.__name__, level="ERROR"):
            with world.patched():
                results = _run_ranks(
                    world,
                    {
                        0: lambda: rank0.batch_set_v2(_transfers(order)),
                        1: lambda: rank1.batch_set_v2(_transfers(order)),
                    },
                )

        self.assertEqual(
            sorted({detail for op, detail in world.ops[0] if op == "broadcast"}),
            ["session-mamba"],
            "one pool, one session",
        )
        self.assertEqual(world.ops[0], world.ops[1])
        for rank in (0, 1):
            self.assertEqual(
                world.saves[rank],
                [(frozenset({"mamba"}), frozenset({"mamba"}))],
                "the pool was written, not dropped from the sequence",
            )
            # ``results`` is keyed by the pool identity too, so the two
            # spellings share one entry -- which is why upstream can read it
            # back with either.
            self.assertEqual(results[rank][MAMBA], [True] * PAGES)
            self.assertEqual(results[rank]["mamba"], [True] * PAGES)


if __name__ == "__main__":
    unittest.main()
