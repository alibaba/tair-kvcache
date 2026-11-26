"""The parallel layouts the connector can serve, and how it addresses them.

A location spec is named by TP rank alone (``tp_{rank}``), so one instance
serves exactly one TP group, and every collective of that group has to be
addressed to *that* group:

* the write collectives are addressed to the group's own rank 0 (``group_src``,
  not the global ``src``).  With attention DP the group is one replica's TP
  ranks (``[2, 3]``, say), global rank 0 is not a member, and a broadcast from
  it raises ``Global rank 0 is not part of group`` -- which the write paths
  catch into an all-miss answer, i.e. every replica but the first stops writing
  to storage;
* the gloo group is joined with ``use_local_synchronization``, so the disjoint
  groups of other attention-DP replicas do not wait for each other's
  rendezvous;
* pipeline and context parallelism are refused at construction: their shards
  would write different bytes into the spec name of one cache.

Both collective properties are about ranks outside the local group, so
``TestDisjointAttentionTpGroups`` runs two disjoint attention-TP groups in four
spawned processes; the single-process tests pin what can be pinned there (the
``new_group`` call, and the group position the broadcasts rely on).

Prerequisites are ``test_tp_init_sync.py``'s: a sglang runtime that can import
its own backend, a torch build with gloo and ``multiprocessing`` spawn, and
``kv_cache_manager`` importable from the repository root on ``sys.path`` -- the
compiled kvcm pybind client and the build-generated ``_version_info`` are
stubbed below, so no wheel is needed for them.

    PYTHONPATH=$PWD python kv_cache_manager/py_connector/sglang/test_parallel_layout.py
"""

import json
import multiprocessing
import os
import sys
import tempfile
import time
import types
import unittest
from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple
from unittest import mock
from unittest.mock import MagicMock

import torch

# ── Modules the connector imports but this test does not exercise ──────
# Same technique as test_tp_init_sync.py: the compiled pybind client and the
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

from kv_cache_manager.py_connector.sglang import connector as connector_module  # noqa: E402

from kv_cache_manager.py_connector.sglang.connector import HiCacheKVCM  # noqa: E402

PAGE_SIZE = 64
KV_BYTES_PER_TOKEN = 128
MAMBA_BYTES_PER_TOKEN = 96
# Two IOVs (K and V) per page: an MHA pool, so the v1 write path slices by page.
KV_FACTOR = 2
KV_IOV_BYTES = KV_BYTES_PER_TOKEN * PAGE_SIZE // KV_FACTOR
# One v1 write call: four new blocks, the Manager asks for the trailing two.
WRITE_KEYS = ["block-0", "block-1", "block-2", "block-3"]
BLOCKS_TO_WRITE = 2
# One v2 write call: a rank-sharded sidecar, one page per key.
MAMBA_KEYS = ["mamba-0", "mamba-1"]


def _storage_config(**overrides: Any) -> HiCacheStorageConfig:
    """A real sglang storage config (same shape as the other test files')."""
    args: dict[str, Any] = {
        "tp_rank": 1,
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
            "instance_id": "parallel-layout-test",
        },
    }
    args.update(overrides)
    return HiCacheStorageConfig(**args)


@contextmanager
def _patched_group_creation(
    ranks: List[int], my_group_rank: int
) -> Iterator[List[Tuple[Any, dict]]]:
    """Run ``_init_parallel_context`` against a scripted attention-TP group.

    Yields the ledger of ``new_group`` calls, each ``(ranks, kwargs)``: the
    library call is what a single process can pin, because the property that
    depends on it -- two replicas not waiting for each other -- needs both
    replicas in different processes (``TestDisjointAttentionTpGroups``).
    ``my_group_rank`` is the position this process occupies in ``ranks``.
    """
    creations: List[Tuple[Any, dict]] = []
    tp_group = types.SimpleNamespace(cpu_group=object())

    def new_group(group_ranks: Any = None, **kwargs: Any) -> Any:
        creations.append((list(group_ranks), kwargs))
        # A handle only: every rank query below is scripted.
        return object()

    with (
        mock.patch.object(connector_module, "get_tp_group", lambda: tp_group),
        mock.patch.object(connector_module, "get_attn_tp_group", lambda: tp_group),
        mock.patch.object(connector_module, "is_dp_attention_enabled", lambda: True),
        mock.patch.object(
            torch.distributed, "get_world_size", lambda group=None: len(ranks)
        ),
        mock.patch.object(
            torch.distributed,
            "get_process_group_ranks",
            lambda group=None: list(ranks),
        ),
        mock.patch.object(
            torch.distributed, "get_rank", lambda group=None: my_group_rank
        ),
        mock.patch.object(torch.distributed, "new_group", new_group),
    ):
        yield creations


class TestPipelineAndContextParallelAreRefused(unittest.TestCase):
    """A spec name has no PP/CP shard, so those layouts cannot be served.

    Two pipeline stages hold different layers of one model and two CP ranks
    hold different slices of one sequence, so both would write different bytes
    into the same ``tp_{rank}`` spec, and the Manager has no shard dimension to
    keep them apart.  Nothing downstream can detect that: silent storage
    corruption is what has to be refused, and it is refused before anything
    registers.
    """

    def test_pp_size_two_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            HiCacheKVCM(_storage_config(pp_rank=1, pp_size=2), {})
        self.assertIn("pp_rank=1/2", str(caught.exception))

    def test_attn_cp_size_two_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            HiCacheKVCM(_storage_config(attn_cp_rank=1, attn_cp_size=2), {})
        message = str(caught.exception)
        self.assertIn("attn_cp_rank=1/2", message)
        self.assertIn("pp_rank=0/1", message, "the message names every shard")

    def test_one_shard_of_each_is_served(self) -> None:
        connector = HiCacheKVCM(_storage_config(), {})
        self.assertEqual(connector.pp_size, 1)

    def test_a_release_without_context_parallelism_is_served(self) -> None:
        """The CP pair arrived after this connector's oldest supported release.

        sglang v0.5.10's ``HiCacheStorageConfig`` has no ``attn_cp_*``, so a
        config without those two fields reports no CP shard: one shard, not a
        layout to refuse.
        """
        config = _storage_config()
        for field in ("attn_cp_rank", "attn_cp_size"):
            delattr(config, field)
        connector = HiCacheKVCM(config, {})
        self.assertEqual(connector.pp_size, 1)


class TestTheStorageGroupIsJoinedLocally(unittest.TestCase):
    """The connector's own group: made with the right call, at the right ranks.

    ``use_local_synchronization=True`` is what the write collectives need: the
    group of one attention-DP replica is disjoint from every other replica's,
    and the default (a rendezvous of the whole default group, under one shared
    group name) makes those replicas wait for each other and share store keys.
    """

    def test_the_group_is_created_with_local_synchronization(self) -> None:
        connector = HiCacheKVCM(_storage_config(tp_rank=1, tp_size=2), {})

        with _patched_group_creation([2, 3], my_group_rank=1) as creations:
            connector._init_parallel_context()

        self.assertEqual(
            creations,
            [([2, 3], {"backend": "gloo", "use_local_synchronization": True})],
        )

    def test_a_group_position_that_disagrees_with_tp_rank_is_refused(self) -> None:
        """The write broadcasts address the group's rank 0, whoever that is.

        They do so by group rank (``group_src=0``), which is the rank the write
        paths call ``tp_rank == 0`` only while this rank's position in the
        group *is* its ``tp_rank``.  A deployment that breaks that would
        broadcast from a rank that never opens a write session, so it is
        refused here instead of hanging in the middle of a write.
        """
        connector = HiCacheKVCM(_storage_config(tp_rank=0, tp_size=2), {})

        with (
            _patched_group_creation([2, 3], my_group_rank=1),
            self.assertRaises(RuntimeError) as caught,
        ):
            connector._init_parallel_context()

        self.assertIn("tp_rank=0", str(caught.exception))

    def test_a_single_tp_rank_creates_no_group(self) -> None:
        connector = HiCacheKVCM(_storage_config(tp_rank=0, tp_size=1), {})

        with _patched_group_creation([0], my_group_rank=0) as creations:
            connector._init_parallel_context()

        self.assertEqual(creations, [], "one rank has nobody to make a group for")


# ── The data path of the spawned ranks ─────────────────────────────────
#
# The four processes below run the connector's real write paths over a real
# gloo group.  Only the Manager and the transfer client are stubs: what these
# tests measure is the group's collectives, and both write paths have to reach
# the end of the session for that.

# Two attention-DP replicas: each pair is one replica's attention-TP group.
_ATTENTION_TP_GROUPS = ((0, 1), (2, 3))
_WORLD_SIZE = 4
# The first replica's ranks create their group this late, while the second
# replica's already has (see _run_replica_rank).
_LATE_REPLICA_SECONDS = 2.0


class _FakeKvPool:
    """The KV anchor as the v1 write path reads it."""

    page_size = PAGE_SIZE
    dtype = torch.bfloat16

    def get_size_per_token(self) -> int:
        return KV_BYTES_PER_TOKEN

    def get_page_buffer_meta(self, indices: torch.Tensor) -> tuple[list, list]:
        slots = len(indices) * KV_FACTOR
        return (
            [0x1000_0000 + i * KV_IOV_BYTES for i in range(slots)],
            [KV_IOV_BYTES] * slots,
        )


class _FakeMambaPool:
    """The rank-sharded sidecar as the v2 write path reads it: a page is a slot."""

    page_size = 1
    dtype = torch.bfloat16

    def get_size_per_token(self) -> int:
        return MAMBA_BYTES_PER_TOKEN

    def get_page_buffer_meta(self, indices: torch.Tensor) -> tuple[list, list]:
        return (
            [0x2000_0000 + int(index) for index in indices.tolist()],
            [MAMBA_BYTES_PER_TOKEN] * len(indices),
        )


class _FakeManager:
    """The write-session calls of a Manager, answering for the session's group.

    A session covers one location-spec group, so its answer carries every TP
    rank's spec of that pool and each rank reads its own out of it.
    """

    def __init__(self, connector: HiCacheKVCM) -> None:
        self._connector = connector
        self.sessions: List[dict] = []

    def _spec_names(self, group_names: Sequence[str]) -> List[str]:
        """The specs one location of a session over ``group_names`` carries."""
        connector = self._connector
        if not group_names:
            return [connector.location_spec_name]
        if group_names[0] == connector._get_extra_pool_spec_group(PoolName.MAMBA):
            namer = connector._tp_rank_to_linear_spec_name
        else:
            namer = connector._tp_rank_to_spec_name
        return [namer(rank) for rank in range(connector.tp_size)]

    def start_write_cache(self, request: dict) -> dict:
        self.sessions.append(request)
        spec_names = self._spec_names(request["location_spec_group_names"])
        blocks = min(BLOCKS_TO_WRITE, len(request["block_keys"]))
        return {
            "locations": [
                {
                    "location_specs": [
                        {"name": name, "uri": f"{name}-{i}"} for name in spec_names
                    ]
                }
                for i in range(blocks)
            ],
            "write_session_id": "write-session",
            "block_mask": {"offset": len(request["block_keys"]) - blocks},
        }

    def finish_write_cache(self, request: dict) -> dict:
        return {}


class _FakeTransferClient:
    """``SaveKvCaches`` as the write paths read it, plus the URIs it was given.

    The call takes a moment: the connector derives a transfer bandwidth from its
    elapsed time, so a stub faster than the clock divides by zero -- no real
    transfer is that fast.
    """

    def __init__(self) -> None:
        self.saved: List[List[str]] = []

    def SaveKvCaches(self, uris: List[str], buffers: Any) -> Any:
        self.saved.append(list(uris))
        time.sleep(0.001)
        return (_mock_kvcm.ClientErrorCode.ER_OK,)


class _StubClientConnector(HiCacheKVCM):
    """A connector whose Manager handshake is stubbed, whose data path is not."""

    def _ensure_client(self) -> None:
        """The one step replaced: registering with a Manager needs a Manager."""
        self._client_ready = True


@contextmanager
def _patched_attention_tp_group(cpu_group: Any) -> Iterator[None]:
    """Make the connector resolve this rank's attention-TP group.

    sglang builds that group in its parallel state; the spawned rank builds it
    itself, so only the lookup is replaced.  ``new_group``, the ranks and the
    leader the connector derives are its own code.
    """
    tp_group = types.SimpleNamespace(cpu_group=cpu_group)
    with (
        mock.patch.object(connector_module, "get_tp_group", lambda: tp_group),
        mock.patch.object(connector_module, "get_attn_tp_group", lambda: tp_group),
        mock.patch.object(connector_module, "is_dp_attention_enabled", lambda: True),
    ):
        yield


def _build_connector(
    rank: int, group_ranks: Sequence[int], cpu_group: Any
) -> HiCacheKVCM:
    """A connector on this rank's real attention-TP group and stubbed I/O."""
    connector = _StubClientConnector(
        _storage_config(
            tp_rank=list(group_ranks).index(rank), tp_size=len(group_ranks)
        ),
        {},
    )
    with _patched_attention_tp_group(cpu_group):
        connector._init_parallel_context()

    connector.mem_pool_host = _FakeKvPool()
    connector.registered_pools[PoolName.KV] = connector.mem_pool_host
    connector.registered_pools[PoolName.MAMBA] = _FakeMambaPool()
    connector.pool_specs = connector._resolve_pool_specs()
    connector.location_spec_name = connector._tp_rank_to_spec_name(connector.tp_rank)
    connector.location_spec_size = KV_BYTES_PER_TOKEN * PAGE_SIZE
    connector.kv_factor = KV_FACTOR
    connector.block_size = PAGE_SIZE
    connector.write_timeout_seconds = 5
    connector.transfer_client = _FakeTransferClient()
    connector._manager_client = _FakeManager(connector)  # ty: ignore[invalid-assignment]
    return connector


def _host_indices() -> torch.Tensor:
    return torch.arange(len(WRITE_KEYS) * PAGE_SIZE)


def _tp_rank_of(rank: int) -> int:
    """The TP rank of a global rank: its position inside its replica's group."""
    replica = next(group for group in _ATTENTION_TP_GROUPS if rank in group)
    return replica.index(rank)


def _mamba_transfer() -> PoolTransfer:
    return PoolTransfer(
        name=PoolName.MAMBA,
        host_indices=torch.arange(len(MAMBA_KEYS)),
        keys=MAMBA_KEYS,
    )


_PHASE_READY = "ready"
_PHASE_V1 = "v1"
_PHASE_V2 = "v2"
_PHASE_SECOND_GROUP = "second-group"
_PHASE_FAILED = "failed"
# The spawned ranks are ready (a spawned import of torch and sglang is the slow
# part), then one write call is milliseconds: a rank that has not answered long
# after that is waiting in a collective nobody will join.
_STARTUP_DEADLINE_SECONDS = 120.0
_WRITE_DEADLINE_SECONDS = 20.0


def _record_path(out_dir: str, rank: int) -> str:
    return os.path.join(out_dir, f"rank{rank}.jsonl")


def _record(
    out_dir: str,
    rank: int,
    phase: str,
    result: Any = None,
    seconds: Optional[float] = None,
    uris: Optional[List[List[str]]] = None,
    error: Optional[str] = None,
) -> None:
    record = {
        "phase": phase,
        "result": result,
        "seconds": seconds,
        "uris": uris,
        "error": error,
    }
    with open(_record_path(out_dir, rank), "a") as handle:
        handle.write(json.dumps(record))
        handle.write("\n")


def _records(out_dir: str, rank: int) -> List[dict]:
    """Every complete record a rank has written so far."""
    try:
        with open(_record_path(out_dir, rank)) as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return []
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:  # a torn tail line, ignore it
            continue
    return records


def _record_of(out_dir: str, rank: int, phase: str) -> Optional[dict]:
    for record in _records(out_dir, rank):
        if record["phase"] == phase:
            return record
    return None


def _required_record(out_dir: str, rank: int, phase: str) -> dict:
    """The record of a phase that must have happened by now."""
    record = _record_of(out_dir, rank, phase)
    if record is None:
        raise AssertionError(f"rank {rank} has not written its {phase} record")
    return record


def _run_replica_rank(rank: int, init_file: str, out_dir: str) -> None:
    """One rank of one attention-DP replica (spawned by the parent)."""
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    try:
        group_ranks = next(
            replica for replica in _ATTENTION_TP_GROUPS if rank in replica
        )
        # The group sglang's parallel state would build for this replica.
        cpu_group = torch.distributed.new_group(
            list(group_ranks), backend="gloo", use_local_synchronization=True
        )
        connector = _build_connector(rank, group_ranks, cpu_group)
        _record(out_dir, rank, _PHASE_READY)

        v1 = connector.batch_set_v1(WRITE_KEYS, _host_indices())
        _record(out_dir, rank, _PHASE_V1, v1, uris=connector.transfer_client.saved)
        written = len(connector.transfer_client.saved)
        v2 = connector.batch_set_v2([_mamba_transfer()])
        _record(
            out_dir,
            rank,
            _PHASE_V2,
            v2,
            uris=connector.transfer_client.saved[written:],
        )

        # A second connector on the same group, created while the other replica
        # is still asleep: joining a group locally is what keeps this rank off
        # the other replica's rendezvous.
        if group_ranks == _ATTENTION_TP_GROUPS[0]:
            time.sleep(_LATE_REPLICA_SECONDS)
        started = time.monotonic()
        _build_connector(rank, group_ranks, cpu_group)
        _record(out_dir, rank, _PHASE_SECOND_GROUP, seconds=time.monotonic() - started)
    except BaseException as e:
        # The parent asserts on these records, so it has to see the reason.
        _record(out_dir, rank, _PHASE_FAILED, error=f"{type(e).__name__}: {e}")
        raise
    finally:
        torch.distributed.destroy_process_group()


def _gloo_available() -> bool:
    try:
        return bool(torch.distributed.is_available()) and bool(
            torch.distributed.is_gloo_available()
        )
    except Exception:
        return False


class TestDisjointAttentionTpGroups(unittest.TestCase):
    """Two attention-DP replicas, each with its own attention-TP group.

    Replica ``[0, 1]`` and replica ``[2, 3]`` are separate sglang engines, so
    neither is in the other's group and neither may address the other's ranks.
    """

    def test_each_replica_broadcasts_from_its_own_leader(self) -> None:
        if not _gloo_available():
            self.skipTest(
                "this torch build has no gloo, so no real process group can be "
                "created for the two replicas"
            )
        with tempfile.TemporaryDirectory(prefix="attn-dp-") as out_dir:
            init_file = os.path.join(out_dir, "gloo-init-file")
            context = multiprocessing.get_context("spawn")
            procs = [
                context.Process(
                    target=_run_replica_rank,
                    args=(rank, init_file, out_dir),
                    name=f"attn-dp-rank{rank}",
                )
                for rank in range(_WORLD_SIZE)
            ]
            for proc in procs:
                proc.start()
            try:
                self._assert_both_replicas_write(out_dir, procs)
                self._assert_a_late_replica_does_not_hold_the_other_back(out_dir, procs)
            finally:
                for proc in procs:
                    if proc.is_alive():
                        proc.terminate()  # our own child, send to it only
                for proc in procs:
                    proc.join(timeout=10)
                for proc in procs:
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=10)

    def _wait_until(self, predicate: Callable[[], bool], deadline: float) -> bool:
        """Poll (the children write from other processes) until the deadline."""
        end = time.monotonic() + deadline
        while time.monotonic() < end:
            if predicate():
                return True
            time.sleep(0.05)
        return predicate()

    def _diagnosis(self, out_dir: str, procs: Sequence[Any]) -> str:
        return " | ".join(
            [f"rank{rank}: {_records(out_dir, rank)!r}" for rank in range(_WORLD_SIZE)]
            + [f"exit codes: {[proc.exitcode for proc in procs]!r}"]
        )

    def _assert_both_replicas_write(self, out_dir: str, procs: Sequence[Any]) -> None:
        """Every rank of every replica sees its leader's session and writes.

        With the write collectives addressed to global rank 0 the second
        replica's ranks raise ``Global rank 0 is not part of group``, the write
        paths answer all-miss, and this is where it shows: those ranks report
        ``False`` for blocks the first replica writes fine.
        """
        ready = self._wait_until(
            lambda: all(
                _record_of(out_dir, rank, _PHASE_READY) for rank in range(_WORLD_SIZE)
            ),
            _STARTUP_DEADLINE_SECONDS,
        )
        self.assertTrue(
            ready, f"not every rank came up: {self._diagnosis(out_dir, procs)}"
        )
        answered = self._wait_until(
            lambda: all(
                _record_of(out_dir, rank, phase)
                for rank in range(_WORLD_SIZE)
                for phase in (_PHASE_V1, _PHASE_V2)
            ),
            _WRITE_DEADLINE_SECONDS,
        )
        self.assertTrue(
            answered,
            "a rank never returned from its write: one rank is waiting in a "
            f"collective its peers never entered -- {self._diagnosis(out_dir, procs)}",
        )

        for rank in range(_WORLD_SIZE):
            local = _tp_rank_of(rank)
            v1 = _required_record(out_dir, rank, _PHASE_V1)
            self.assertEqual(
                v1["result"],
                [True] * len(WRITE_KEYS),
                f"rank {rank} reported a miss for a write its replica performed",
            )
            self.assertEqual(
                v1["uris"],
                [[f"tp_{local}_full-0", f"tp_{local}_full-1"]],
                f"rank {rank} did not send a write of its own KV spec",
            )
            v2 = _required_record(out_dir, rank, _PHASE_V2)
            self.assertEqual(
                v2["result"],
                {PoolName.MAMBA.value: [True, True]},
                f"rank {rank} lost the sidecar session of its replica",
            )
            self.assertEqual(
                v2["uris"],
                [[f"tp_{local}_linear-0", f"tp_{local}_linear-1"]],
                f"rank {rank} did not send a write of its own sidecar spec",
            )

    def _assert_a_late_replica_does_not_hold_the_other_back(
        self, out_dir: str, procs: Sequence[Any]
    ) -> None:
        """The disjoint group of a sleeping replica must not block this one.

        Replica ``[0, 1]`` creates its second group ``_LATE_REPLICA_SECONDS``
        late, replica ``[2, 3]`` creates its own straight away and is done long
        before that.  A default (world-wide) ``new_group`` gives both replicas'
        creations one store prefix, so what it costs depends on how they
        interleave; the deterministic pin for the call itself is
        ``test_the_group_is_created_with_local_synchronization``.
        """
        created = self._wait_until(
            lambda: all(
                _record_of(out_dir, rank, _PHASE_SECOND_GROUP)
                for rank in range(_WORLD_SIZE)
            ),
            _LATE_REPLICA_SECONDS + _WRITE_DEADLINE_SECONDS,
        )
        self.assertTrue(
            created,
            "a rank never finished creating its group: "
            f"{self._diagnosis(out_dir, procs)}",
        )
        for rank in (2, 3):
            seconds = _required_record(out_dir, rank, _PHASE_SECOND_GROUP)["seconds"]
            self.assertLess(
                seconds,
                _LATE_REPLICA_SECONDS / 2,
                f"rank {rank} waited {seconds:.1f}s for a group whose members "
                f"are not in it",
            )


if __name__ == "__main__":
    unittest.main()
