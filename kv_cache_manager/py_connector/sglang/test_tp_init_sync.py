"""Deterministic tests for the TP-group agreement in front of the write paths.

The connector's write paths (``batch_set_v1`` -> ``_batch_set``, and
``batch_set_v2``) are the only callers of its TP collectives, so they must
never let one rank return while the others wait for it: a rank that could not
create its client used to answer conservatively on its own and leave its peers
blocked in ``broadcast_object_list`` / ``all_reduce`` forever.  These tests pin
the fix, which publishes the rank-local initialization outcome to the group and
lets the whole group decide:

* a local failure, or a peer failure, stops *both* write paths without opening
  a write session and without entering any further collective;
* a round that reported a failure is retried on the next call, and a round that
  saw the whole group ready lets the write path proceed exactly as before;
* once the group has agreed, no extra collective is paid on later calls;
* nothing is added on paths that run no collective at all: single rank, MLA
  (rank 0 only) and a missing process group, which is a failure rather than an
  implicit collective on torch's default group.

``TestTwoProcessGlooExperiment`` runs the same thing on a real gloo group in
two spawned processes, with a control phase that uses the pre-fix wiring and
shows the rank that initialized blocking in the broadcast while its peer is
already gone.

Prerequisites: a sglang runtime that can import its own backend (``sgl_kernel``
and friends) and ``kv_cache_manager`` importable from the repository root on
``sys.path`` -- the compiled kvcm pybind client and the build-generated
``_version_info`` are stubbed below, so no wheel is needed for them.  The
two-process test needs a ``torch`` build with gloo and ``multiprocessing``
spawn; it skips itself when gloo is unavailable.

    PYTHONPATH=$PWD python kv_cache_manager/py_connector/sglang/test_tp_init_sync.py
"""

import json
import multiprocessing
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional
from unittest import mock
from unittest.mock import MagicMock

import torch

# ── Modules the connector imports but this test does not exercise ──────
# Same technique as test_pool_registration.py: the compiled pybind client and
# the build-generated version module must be importable before the connector
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
# One write call: four new blocks, the Manager asks for the last two by default.
WRITE_KEYS = ["block-0", "block-1", "block-2", "block-3"]
BLOCKS_TO_WRITE = 2
# One Mamba page is one slot: the pool is page-granular.
MAMBA_BYTES_PER_TOKEN = 96
# The process group of the single-process tests.  Only its identity matters:
# the collectives themselves are scripted below.
GROUP = object()


class _FakeMambaPool:
    """Only what the v2 write path reads from a Mamba host pool."""

    def __init__(self) -> None:
        self.page_size = 1

    def get_page_buffer_meta(self, indices: torch.Tensor) -> tuple[list, list]:
        return (
            [0x1000_0000 + i for i in indices.tolist()],
            [MAMBA_BYTES_PER_TOKEN] * len(indices),
        )


class ScriptedConnector(HiCacheKVCM):
    """A write-path connector with a scripted initialization outcome.

    ``__init__`` is bypassed on purpose (same technique as
    ``test/test_batch_set_return.py``): only the fields the write entry points
    read are set.  The real initialization -- Manager handshake, SDK client,
    pool registration -- is covered by ``test_pool_registration.py``; the tests
    here observe the agreement that runs in front of the write path, so the
    local outcome is scripted and the Manager/SDK/transfer client are mocks
    that only have to be *reachable* and observable.
    """

    def __init__(
        self,
        *,
        local_init_ok: bool,
        script: "_CollectiveScript",
        tp_world_size: int = 2,
        tp_rank: int = 1,
        group: Any = GROUP,
        is_mla_model: bool = False,
        with_mamba: bool = False,
        blocks_to_write: int = BLOCKS_TO_WRITE,
        on_write_session: Optional[Callable[[], None]] = None,
    ) -> None:
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        self.is_mla_model = is_mla_model
        self.storage_tp_group = group
        self.kv_factor = 2
        self.instance_id = "tp-init-sync-test"
        self.location_spec_size = 4096
        self.write_timeout_seconds = 5
        self.has_mamba = with_mamba
        self.has_indexer = False
        self.registered_pools = {}
        # Same names as _init_kvcm_client would pick: the MLA KV pool is
        # replicated, so every rank uses rank 0's KV spec, while the Mamba pool
        # is rank-sharded and every rank uses its own linear spec.
        kv_spec_rank = 0 if is_mla_model else tp_rank
        self.location_spec_name = self._tp_rank_to_spec_name(kv_spec_rank)
        if with_mamba:
            self.mamba_spec_size = MAMBA_BYTES_PER_TOKEN
            self.mamba_location_spec_name = self._tp_rank_to_linear_spec_name(tp_rank)
            self.registered_pools[PoolName.MAMBA] = _FakeMambaPool()
        self.backup_pgs = []
        self.backup_bandwidth = []
        self.prefetch_pgs = []
        self.prefetch_bandwidth = []
        self._init_lock = threading.Lock()
        self._client_ready = False
        self._closed = False
        self._tp_init_agreed = False
        self.local_init_ok = local_init_ok
        self.init_attempts = 0
        self.blocks_to_write = blocks_to_write
        self.on_write_session = on_write_session
        # The ledger this connector reports its agreement region into (see
        # _ensure_client_for_write) and that plays the peer's part.
        self.script = script

        self._manager_client = MagicMock()
        self.transfer_client = MagicMock()
        self.mem_pool_host = MagicMock()
        # Four blocks of two IOVs each, enough for every index the Manager can
        # ask for; the write path's own logic is covered elsewhere.
        self.mem_pool_host.get_page_buffer_meta.return_value = (
            list(range(4 * self.kv_factor)),
            [self.location_spec_size] * (4 * self.kv_factor),
        )
        self._manager_client.start_write_cache.side_effect = self._start_write_cache
        # The connector compares the load result by value and the save result by
        # its first element, so the stubs differ in shape.
        self.transfer_client.SaveKvCaches.return_value = (
            _mock_kvcm.ClientErrorCode.ER_OK,
        )
        self.transfer_client.LoadKvCaches.return_value = (
            _mock_kvcm.ClientErrorCode.ER_OK
        )

    def _ensure_client(self) -> None:
        self.init_attempts += 1
        if not self.local_init_ok:
            raise RuntimeError("simulated local initialization failure")
        self._client_ready = True

    def _ensure_client_for_write(
        self, pool_transfers: Optional[List[PoolTransfer]] = None
    ) -> None:
        """The production gate, with the agreement region marked for the fixture.

        Everything the connector reduces while this method runs *is* the
        agreement; everything reduced after it belongs to the write path.  That
        is what makes the fixture's counting exact instead of inferred: the
        write paths reduce tensors of their own whose shape says nothing about
        which side they are on (``_batch_set`` reduces one flag per block, so a
        single-block write is a one-element tensor too, and ``batch_set_v2``
        reduces a 0-dim flag).  See the tests for both cases.
        """
        self.script.in_agreement = True
        try:
            super()._ensure_client_for_write(pool_transfers)
        finally:
            self.script.in_agreement = False

    def session_specs(self, group_names: Any) -> list:
        """The spec names a Manager lists for a session over ``group_names``.

        A group session covers every rank of the group, and the connector picks
        its own name out of the list; a session without a group name is the
        instance's own spec.
        """
        if not group_names:
            return [self.location_spec_name]
        group = group_names[0]
        if group == self._get_kv_spec_group():
            return [
                self._tp_rank_to_spec_name(rank) for rank in range(self.tp_world_size)
            ]
        for pool in (PoolName.MAMBA, PoolName.INDEXER):
            if group == self._get_extra_pool_spec_group(pool):
                namer = (
                    self._tp_rank_to_linear_spec_name
                    if pool == PoolName.MAMBA
                    else self._tp_rank_to_indexer_spec_name
                )
                return [namer(rank) for rank in range(self.tp_world_size)]
        raise AssertionError(f"fixture does not know spec group {group!r}")

    def manager_session(self, group_names: Any, block_count: int) -> dict:
        """A Manager's answer for a session over ``block_count`` requested blocks.

        As for a real write-through session, only the trailing
        ``blocks_to_write`` blocks need data, so that is how many locations come
        back and the mask says where they start.
        """
        specs = self.session_specs(group_names)
        return {
            "locations": [
                {
                    "location_specs": [
                        {"name": name, "uri": f"{name}-{i}"} for name in specs
                    ]
                }
                for i in range(self.blocks_to_write)
            ],
            "write_session_id": "write-session-1",
            "block_mask": {"offset": block_count - self.blocks_to_write},
        }

    def peer_session(self) -> dict:
        """The session rank 0 opened, as one of its peers receives it."""
        group = self._get_extra_pool_spec_group(PoolName.MAMBA)
        return self.manager_session([group], self.blocks_to_write)

    def _start_write_cache(self, request: Any) -> dict:
        """The session this rank opens for its own write."""
        if self.on_write_session is not None:
            self.on_write_session()
        return self.manager_session(
            request["location_spec_group_names"], len(request["block_keys"])
        )


class _CollectiveScript:
    """Records this rank's TP collectives and plays the peer's part.

    Which ``all_reduce`` is the agreement is a *position*, not a shape: it is
    the one the connector runs inside ``_ensure_client_for_write``, and the
    fixture learns that from the connector itself (``in_agreement``, set by
    ``ScriptedConnector`` around the real gate).  Guessing from the tensor
    would be ambiguous -- the write paths reduce tensors of their own that can
    be one element (``_batch_set`` with a single block to write) or 0-dim
    (``batch_set_v2``'s flag) -- so both shapes are covered by tests that
    assert they are counted on the write-path side.
    """

    def __init__(self, *, peer_ready: bool, peer_wrote: bool = True) -> None:
        # Mutable: a test can let the peer recover between two calls.
        self.peer_ready = peer_ready
        self.peer_wrote = peer_wrote
        self.in_agreement = False
        # What a peer's broadcast delivers: the write session rank 0 opened,
        # as this rank receives it (set by _write_pair).
        self.peer_session: Optional[Callable[[], dict]] = None
        # Agreement rounds: (operator, shape, dtype, group).
        self.agreements: list = []
        # The write path's own collectives, i.e. what must not be reached when
        # the group agreed that it is not ready to write.
        self.broadcasts: list = []
        self.write_reduces: list = []


@contextmanager
def _patched_tp_collectives(script: _CollectiveScript) -> Iterator[_CollectiveScript]:
    """Run one rank's view of the group's collectives, scripted and recorded.

    The agreement reduce folds the peer's readiness in with MIN, exactly like
    the real reduce.  The write path's own collectives are recorded instead --
    a real peer would block there, so reaching them is what the tests assert
    on.  Classification comes from ``script.in_agreement``.  A one-element
    broadcast is rank 0's write session, so it is delivered like a real peer
    would receive it; the four-element one is the v1 session and stays empty,
    which is what the v1 tests want to observe.
    """

    def fake_all_reduce(tensor: Any, op: Any = None, group: Any = None) -> None:
        record = (op, tuple(tensor.shape), tensor.dtype, group)
        if script.in_agreement:
            script.agreements.append(record)
            if not script.peer_ready:
                tensor.fill_(0)
            return
        script.write_reduces.append(record)
        if not script.peer_wrote:
            tensor.fill_(0)

    def fake_broadcast(object_list: Any = None, **kwargs: Any) -> None:
        script.broadcasts.append(kwargs.get("group"))
        if (
            script.peer_session is not None
            and isinstance(object_list, list)
            and len(object_list) == 1
        ):
            object_list[0] = script.peer_session()

    with (
        mock.patch.object(torch.distributed, "all_reduce", fake_all_reduce),
        mock.patch.object(torch.distributed, "broadcast_object_list", fake_broadcast),
    ):
        yield script


def _write_pair(
    *, peer_ready: bool, peer_wrote: bool = True, **connector: Any
) -> tuple[ScriptedConnector, _CollectiveScript]:
    """A connector and the scripted TP group it will agree with."""
    script = _CollectiveScript(peer_ready=peer_ready, peer_wrote=peer_wrote)
    instance = ScriptedConnector(script=script, **connector)
    script.peer_session = instance.peer_session
    return instance, script


def _transfer() -> PoolTransfer:
    return PoolTransfer(
        name=PoolName.MAMBA,
        host_indices=torch.arange(2),
        keys=WRITE_KEYS[:2],
    )


def _host_indices() -> torch.Tensor:
    return torch.arange(len(WRITE_KEYS) * PAGE_SIZE)


def _write_sessions(connector: ScriptedConnector) -> Any:
    """The mocked Manager's ``start_write_cache`` (Any: it is a Mock)."""
    manager: Any = connector._manager_client
    return manager.start_write_cache


def _write_commits(connector: ScriptedConnector) -> Any:
    """The mocked Manager's ``finish_write_cache`` (Any: it is a Mock)."""
    manager: Any = connector._manager_client
    return manager.finish_write_cache


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
            "instance_id": "tp-init-sync-test",
        },
    }
    args.update(overrides)
    return HiCacheStorageConfig(**args)


class TestWritePathsAgreeOnInitialization(unittest.TestCase):
    """Both write paths publish the local outcome before any collective."""

    _ROUND = (
        torch.distributed.ReduceOp.MIN,
        (1,),
        torch.int32,
        GROUP,
    )

    def _assert_rounds(self, script: _CollectiveScript, calls: int) -> None:
        """One agreement per write call, on the write path's own group.

        A round that reported a failure is not an agreement, so every call
        publishes again (including the second write path in the same test).
        """
        self.assertEqual(script.agreements, [self._ROUND] * calls)

    def test_local_failure_stops_both_write_paths_without_a_collective(self) -> None:
        """The failing rank must join the agreement instead of returning.

        With the pre-fix wiring this rank answered by itself; the peers went on
        into the write collectives and waited for it forever.
        """
        for peer_ready in (False, True):
            with self.subTest(peer_ready=peer_ready):
                connector, script = _write_pair(
                    local_init_ok=False, peer_ready=peer_ready
                )

                with _patched_tp_collectives(script):
                    v1_result = connector.batch_set_v1(WRITE_KEYS, _host_indices())
                    v2_result = connector.batch_set_v2([_transfer()])

                self.assertEqual(v1_result, [False] * len(WRITE_KEYS))
                self.assertEqual(v2_result, {PoolName.MAMBA: [False, False]})
                self._assert_rounds(script, calls=2)
                self.assertEqual(script.broadcasts, [])
                self.assertEqual(script.write_reduces, [])
                self.assertFalse(connector._client_ready)
                self.assertFalse(connector._tp_init_agreed)
                self.assertEqual(connector.init_attempts, 2, "both paths retried")
                self.assertIsNone(
                    _write_sessions(connector).call_args,
                    "no write session may be opened when the group is not ready",
                )

    def test_peer_failure_stops_a_ready_rank_without_a_collective(self) -> None:
        """A rank that initialized fine also refuses when a peer did not."""
        connector, script = _write_pair(local_init_ok=True, peer_ready=False)

        with _patched_tp_collectives(script):
            v1_result = connector.batch_set_v1(WRITE_KEYS, _host_indices())
            v2_result = connector.batch_set_v2([_transfer()])

        self.assertEqual(v1_result, [False] * len(WRITE_KEYS))
        self.assertEqual(v2_result, {PoolName.MAMBA: [False, False]})
        self._assert_rounds(script, calls=2)
        self.assertEqual(script.broadcasts, [])
        self.assertTrue(connector._client_ready)
        self.assertFalse(
            connector._tp_init_agreed,
            "a failed round must not be remembered as an agreement",
        )
        self.assertIsNone(
            _write_sessions(connector).call_args,
            "no write session may be opened when the group is not ready",
        )

    def test_a_failed_round_is_retried_and_recovers(self) -> None:
        """Nothing is sticky: the next call initializes and agrees again."""
        connector, script = _write_pair(local_init_ok=False, peer_ready=True, tp_rank=0)

        with _patched_tp_collectives(script):
            self.assertEqual(
                connector.batch_set_v1(WRITE_KEYS, _host_indices()),
                [False] * len(WRITE_KEYS),
            )
            self.assertEqual(connector.init_attempts, 1)

            # The transient failure is over: this rank and its peer are ready.
            connector.local_init_ok = True
            recovered = connector.batch_set_v1(WRITE_KEYS, _host_indices())

        self.assertEqual(recovered, [True] * len(WRITE_KEYS))
        self.assertEqual(connector.init_attempts, 2)
        self._assert_rounds(script, calls=2)
        self.assertTrue(connector._tp_init_agreed)
        self.assertIsNotNone(
            _write_sessions(connector).call_args,
            "the recovered call must write like any healthy call",
        )

    def test_a_ready_group_agrees_once_and_then_writes_as_before(self) -> None:
        """Whole group ready: the write path runs, and the agreement is paid once."""
        connector, script = _write_pair(local_init_ok=True, peer_ready=True, tp_rank=0)

        with _patched_tp_collectives(script):
            first = connector.batch_set_v1(WRITE_KEYS, _host_indices())
            second = connector.batch_set_v2([])

        self.assertEqual(first, [True] * len(WRITE_KEYS))
        self.assertEqual(second, {})
        self.assertTrue(connector._tp_init_agreed)
        self.assertEqual(
            len(script.agreements),
            1,
            "the steady state must not pay a collective per write call",
        )
        self.assertEqual(len(script.broadcasts), 1, "rank 0's own broadcast only")
        self.assertEqual(
            [shape for _, shape, _, _ in script.write_reduces],
            [(len(WRITE_KEYS) - BLOCKS_TO_WRITE,)],
            "the write path's per-block flags, one per block it wrote",
        )
        self.assertEqual(connector.init_attempts, 1)

    def test_a_one_element_write_reduce_is_not_the_agreement(self) -> None:
        """The write path may reduce one element too; that is not the agreement.

        ``_batch_set`` reduces one flag per block, so a write session with a
        single block hands the group a ``(1,)`` tensor -- the same shape and
        dtype as the readiness vector.  Counting by tensor size (or shape)
        would report two agreements here and hide a missing agreement behind a
        write-path reduce, which is why the fixture classifies by the region
        the connector runs it in (``ScriptedConnector`` marks it).
        """
        connector, script = _write_pair(
            local_init_ok=True, peer_ready=True, tp_rank=0, blocks_to_write=1
        )

        with _patched_tp_collectives(script):
            result = connector.batch_set_v1(WRITE_KEYS, _host_indices())

        self.assertEqual(result, [True] * len(WRITE_KEYS))
        self._assert_rounds(script, calls=1)
        self.assertEqual(
            [shape for _, shape, _, _ in script.write_reduces],
            [(1,)],
            "the single-block flags belong to the write path",
        )
        self.assertEqual(len(script.broadcasts), 1)

    def test_the_v2_write_path_writes_once_the_group_agreed(self) -> None:
        """The gate must not swallow the v2 path: real session, real reduce.

        A group that agreed has to reach the per-transfer code unchanged:
        StartWriteCache, SaveKvCaches, the per-transfer broadcast and the flag
        reduce -- and the agreement is not paid again by the next call.
        """
        connector, script = _write_pair(
            local_init_ok=True, peer_ready=True, tp_rank=0, with_mamba=True
        )

        with _patched_tp_collectives(script):
            first = connector.batch_set_v2([_transfer()])
            second = connector.batch_set_v2([_transfer()])

        expected = {PoolName.MAMBA: [True, True]}
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual(
            _write_sessions(connector).call_count,
            2,
            "one write session per v2 call",
        )
        self.assertEqual(
            len(script.agreements), 1, "an agreed group pays the round once"
        )
        self.assertEqual(
            [shape for _, shape, _, _ in script.write_reduces],
            [(), ()],
            "the v2 flag is a 0-dim tensor, once per transfer",
        )
        self.assertEqual(len(script.broadcasts), 2, "one per transfer on rank 0")


class TestRankShardedSidecarsAcrossTheGroup(unittest.TestCase):
    """A rank-sharded sidecar is written by every rank, MLA included.

    sglang hands each TP rank its own Mamba/KDA shard and expects all of them
    to write it, even when the primary MLA KV is replicated and only rank 0
    writes that.  The connector used to drop the call on non-zero MLA ranks,
    which left storage holding rank 0's shard while the group committed the
    blocks.
    """

    def _probe(self, value: bool) -> Any:
        return mock.patch.object(
            connector_module, "_sidecars_are_written_by_every_rank", lambda: value
        )

    def test_the_probe_follows_the_hybrid_controller_shape(self) -> None:
        """The version probe reads the class shape, not a guessed version.

        v0.5.17 gave ``HybridCacheController`` its own ``backup_thread_func``
        (the base one still gates on ``backup_skip``), and that override is what
        makes a non-zero MLA rank reach ``batch_set_v2`` at all -- the one
        signal the connector needs to decide whether its peers will arrive.  An
        unknown shape must answer "no", because that protocol is the safe one.
        """
        probe = connector_module._sidecars_are_written_by_every_rank
        module = types.ModuleType(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller"
        )

        class WithoutTheOverride:
            """The shape up to v0.5.16."""

        class WithTheOverride:
            """The shape since v0.5.17."""

            def backup_thread_func(self) -> None:
                raise NotImplementedError

        with mock.patch.dict(sys.modules, {module.__name__: module}):
            for hybrid_controller, expected in (
                (WithoutTheOverride, False),
                (WithTheOverride, True),
            ):
                with self.subTest(shape=hybrid_controller.__name__):
                    setattr(module, "HybridCacheController", hybrid_controller)
                    probe.cache_clear()
                    self.assertEqual(probe(), expected)
        probe.cache_clear()

    def test_mla_sharded_sidecar_is_written_by_every_rank(self) -> None:
        """Codex P1: a non-zero MLA rank writes its own shard, collectively."""
        connector, script = _write_pair(
            local_init_ok=True,
            peer_ready=True,
            tp_rank=1,
            with_mamba=True,
            is_mla_model=True,
        )

        with self._probe(True), _patched_tp_collectives(script):
            result = connector.batch_set_v2([_transfer()])

        self.assertEqual(result, {PoolName.MAMBA: [True, True]})
        self.assertTrue(connector._tp_init_agreed)
        self.assertEqual(len(script.agreements), 1, "the write joins the group")
        self.assertEqual(script.broadcasts, [GROUP], "rank 0's session is waited for")
        self.assertIsNone(
            _write_sessions(connector).call_args,
            "rank 0 opens the session for the group",
        )
        self.assertIsNone(
            _write_commits(connector).call_args,
            "rank 0 commits the group's blocks",
        )
        uris = connector.transfer_client.SaveKvCaches.call_args.args[0]
        self.assertEqual(
            uris,
            ["tp_1_linear-0", "tp_1_linear-1"],
            "the rank writes its own slice of the sharded pool",
        )
        self.assertEqual(
            [shape for _, shape, _, _ in script.write_reduces],
            [()],
            "the per-block commit waits for every rank's flag",
        )

    def test_mla_sharded_sidecar_without_peer_writes_is_refused(self) -> None:
        """Releases that write only rank 0's shard get no collective.

        Before v0.5.17 a non-zero MLA rank never reached ``batch_set_v2``, so
        waiting for it would hang; the shard it never wrote is reported as a
        miss instead of loading rank 0's shard into its pool.
        """
        connector, script = _write_pair(
            local_init_ok=True,
            peer_ready=True,
            tp_rank=1,
            with_mamba=True,
            is_mla_model=True,
        )

        with self._probe(False), _patched_tp_collectives(script):
            with self.assertLogs(connector_module.__name__, "WARNING") as logs:
                result = connector.batch_set_v2([_transfer()])

        self.assertEqual(result, {PoolName.MAMBA: [False, False]})
        self.assertIn("does not write from this rank", logs.records[0].getMessage())
        self.assertEqual(script.agreements, [], "no peer would join the round")
        self.assertEqual(script.broadcasts, [])
        self.assertEqual(script.write_reduces, [])
        self.assertIsNone(connector.transfer_client.SaveKvCaches.call_args)

    def test_mla_sharded_sidecar_read_uses_its_own_spec(self) -> None:
        """The read side matches the write side: a rank loads its own shard.

        Every rank reads the sidecar (``_page_transfer_sidecar`` has no MLA
        gate) and the group's hit count is the minimum over the ranks, so a
        rank must read the shard it wrote, not a peer's.
        """
        connector, _ = _write_pair(
            local_init_ok=True,
            peer_ready=True,
            tp_rank=1,
            with_mamba=True,
            is_mla_model=True,
        )
        manager: Any = connector._manager_client
        manager.get_cache_location.return_value = {
            "locations": [
                {
                    "location_specs": [
                        {"name": "tp_0_linear", "uri": f"tp_0_linear-{i}"},
                        {"name": "tp_1_linear", "uri": f"tp_1_linear-{i}"},
                    ]
                }
                for i in range(2)
            ]
        }

        result = connector.batch_get_v2([_transfer()])

        self.assertEqual(result, {PoolName.MAMBA: [True, True]})
        self.assertEqual(
            connector.transfer_client.LoadKvCaches.call_args.args[0],
            ["tp_1_linear-0", "tp_1_linear-1"],
            "a rank reads its own slice of the sharded pool",
        )

    def test_mla_kv_write_stays_rank0_only(self) -> None:
        """The replicated MLA KV keeps its rank-0-only write path."""
        connector, script = _write_pair(
            local_init_ok=True,
            peer_ready=True,
            tp_rank=1,
            with_mamba=True,
            is_mla_model=True,
        )

        with self._probe(True), _patched_tp_collectives(script):
            with self.assertLogs(connector_module.__name__, "WARNING"):
                result = connector.batch_set_v1(WRITE_KEYS, _host_indices())

        self.assertEqual(result, [False] * len(WRITE_KEYS))
        self.assertEqual(
            script.agreements,
            [],
            "a replicated KV write runs rank-0 only, with no peers to agree with",
        )
        self.assertEqual(script.broadcasts, [])

    def test_non_mla_rank_writes_its_own_sidecar_through_the_session(self) -> None:
        """Non-MLA: every rank writes its own spec, as it always did."""
        connector, script = _write_pair(
            local_init_ok=True, peer_ready=True, tp_rank=1, with_mamba=True
        )

        with _patched_tp_collectives(script):
            result = connector.batch_set_v2([_transfer()])

        self.assertEqual(result, {PoolName.MAMBA: [True, True]})
        self.assertEqual(len(script.agreements), 1)
        self.assertEqual(script.broadcasts, [GROUP])
        self.assertEqual(
            connector.transfer_client.SaveKvCaches.call_args.args[0],
            ["tp_1_linear-0", "tp_1_linear-1"],
        )


class TestPathsThatDoNotJoinTheGroup(unittest.TestCase):
    """No collective is added where the write paths run none themselves."""

    def test_single_rank_write_is_unchanged(self) -> None:
        connector, script = _write_pair(
            local_init_ok=True, peer_ready=True, tp_world_size=1, tp_rank=0, group=None
        )

        with _patched_tp_collectives(script):
            result = connector.batch_set_v1(WRITE_KEYS, _host_indices())

        self.assertEqual(result, [True] * len(WRITE_KEYS))
        self.assertEqual(script.agreements, [])
        self.assertEqual(script.broadcasts, [])
        self.assertEqual(script.write_reduces, [])

    def test_single_rank_failure_stays_local(self) -> None:
        connector, script = _write_pair(
            local_init_ok=False,
            peer_ready=True,
            tp_world_size=1,
            tp_rank=0,
            group=None,
        )

        with _patched_tp_collectives(script):
            result = connector.batch_set_v1(WRITE_KEYS, _host_indices())

        self.assertEqual(result, [False] * len(WRITE_KEYS))
        self.assertEqual(script.agreements, [])
        self.assertEqual(script.broadcasts, [])
        self.assertEqual(connector.init_attempts, 1)

    def test_mla_write_does_not_join_the_group(self) -> None:
        """MLA writes from rank 0 only and uses no collective at all."""
        for tp_rank, expected in (
            (0, [True] * len(WRITE_KEYS)),
            (1, [False] * len(WRITE_KEYS)),
        ):
            with self.subTest(tp_rank=tp_rank):
                connector, script = _write_pair(
                    local_init_ok=True,
                    peer_ready=True,
                    tp_rank=tp_rank,
                    is_mla_model=True,
                )
                with _patched_tp_collectives(script):
                    result = connector.batch_set_v1(WRITE_KEYS, _host_indices())

                self.assertEqual(result, expected)
                self.assertEqual(connector.init_attempts, 1)
                self.assertEqual(script.agreements, [])
                self.assertEqual(script.broadcasts, [])
                self.assertEqual(script.write_reduces, [])

    def test_mla_is_known_before_the_client_exists(self) -> None:
        """The write paths read the model kind before initialization runs.

        The decision "an MLA write never joins the TP group" is taken on the
        first write call, i.e. while the class default (False) would still be
        in place if the real ``__init__`` did not record the storage config.
        """
        connector = HiCacheKVCM(_storage_config(is_mla_model=True), {})
        connector.tp_world_size = 2  # as _init_parallel_context() resolves it

        self.assertTrue(connector.is_mla_model)
        self.assertFalse(connector._client_ready)
        self.assertFalse(connector._write_path_uses_tp_collectives())

    def test_missing_group_is_a_failure_not_a_default_group_collective(self) -> None:
        """``group=None`` would mean torch's default group: refuse instead."""
        connector, script = _write_pair(local_init_ok=True, peer_ready=True, group=None)

        with _patched_tp_collectives(script):
            with self.assertLogs(connector_module.__name__, "ERROR") as logs:
                result = connector.batch_set_v1(WRITE_KEYS, _host_indices())

        self.assertEqual(result, [False] * len(WRITE_KEYS))
        self.assertIn("process group", logs.records[0].getMessage())
        self.assertEqual(script.agreements, [])
        self.assertEqual(script.broadcasts, [])
        self.assertFalse(connector._tp_init_agreed)

    def test_closed_connector_fails_without_a_collective(self) -> None:
        """close() still wins over the initialization, agreement included."""
        connector, script = _write_pair(local_init_ok=True, peer_ready=True)
        connector._closed = True

        with _patched_tp_collectives(script):
            with self.assertLogs(connector_module.__name__, "ERROR") as logs:
                result = connector.batch_set_v1(WRITE_KEYS, _host_indices())

        self.assertEqual(result, [False] * len(WRITE_KEYS))
        self.assertIn("closed", logs.records[0].getMessage())
        self.assertEqual(script.agreements, [])
        self.assertEqual(script.broadcasts, [])
        self.assertEqual(connector.init_attempts, 0)


# ── Two-process gloo experiment ────────────────────────────────────────
#
# The in-process tests script the collectives; this one runs the real thing.
# Both ranks build the same connector and differ only in rank 1's local
# initialization.  Each rank first calls the fixed entry point and then -- as a
# control -- the pre-fix wiring, reproduced verbatim from the previous commit:
# rank 1 returns at once while rank 0 opens a write session and blocks in the
# broadcast that has nobody left to answer.  The parent process observes both.

_PHASE_FIXED = "fixed"
_PHASE_PRE_FIX = "pre-fix"
# Long enough for the fixed phase (sub-second) to prove "returns in bounded
# time", short enough not to slow the suite down.
_PHASE_DEADLINE_SECONDS = 6.0
_STARTUP_DEADLINE_SECONDS = 120.0
# Rank 1 outlives the observation window: a rank that exits would unblock its
# peer's broadcast and hide the hang this control phase demonstrates.
_PEER_KEEPALIVE_SECONDS = 60.0


def _legacy_batch_set_v1(connector: HiCacheKVCM) -> list:
    """The pre-fix ``batch_set_v1``: local init, then straight into the write path.

    Copied from ``connector.py`` before the agreement was added, so the control
    phase of the experiment measures the old wiring and nothing else.
    """
    trace_id = connector._get_trace_id()
    try:
        connector._ensure_client()
        return connector._batch_set(
            keys=WRITE_KEYS, host_indices=_host_indices(), trace_id=trace_id
        )
    except Exception as e:
        print(f"pre-fix batch_set_v1 failed: {e}", file=sys.stderr)
        return [False] * len(WRITE_KEYS)


def _record_path(out_dir: str, rank: int) -> str:
    return os.path.join(out_dir, f"rank{rank}.jsonl")


def _marker_path(out_dir: str, rank: int, phase: str) -> str:
    """Marker for "this rank opened a write session in this phase"."""
    return os.path.join(out_dir, f"rank{rank}.{phase}.write-session")


def _records(out_dir: str, rank: int) -> list:
    """Every complete JSON record a rank has written so far."""
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


def _run_experiment_rank(rank: int, init_file: str, out_dir: str) -> None:
    """One rank of the two-process experiment (spawned by the parent)."""
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        group = torch.distributed.new_group([0, 1], backend="gloo")
        # A rank walks through the phases quickly; the marker carries the phase
        # so that phase 1's evidence cannot be confused with phase 2's.
        phase = {"current": _PHASE_FIXED}
        # The child uses the real collectives, so its marker script is only
        # there to satisfy the fixture (nothing is counted through it).
        connector = ScriptedConnector(
            local_init_ok=(rank == 0),
            script=_CollectiveScript(peer_ready=True),
            tp_rank=rank,
            group=group,
            on_write_session=lambda: open(
                _marker_path(out_dir, rank, phase["current"]), "w"
            ).close(),
        )

        started = time.monotonic()
        result = connector.batch_set_v1(WRITE_KEYS, _host_indices())
        _record(out_dir, rank, _PHASE_FIXED, result, time.monotonic() - started)

        started = time.monotonic()
        phase["current"] = _PHASE_PRE_FIX
        result = _legacy_batch_set_v1(connector)
        _record(out_dir, rank, _PHASE_PRE_FIX, result, time.monotonic() - started)

        if rank == 1:
            # Keep the group alive while rank 0 blocks in the pre-fix phase.
            time.sleep(_PEER_KEEPALIVE_SECONDS)
    finally:
        torch.distributed.destroy_process_group()


def _record(out_dir: str, rank: int, phase: str, result: Any, seconds: float) -> None:
    with open(_record_path(out_dir, rank), "a") as handle:
        handle.write(
            json.dumps({"phase": phase, "result": list(result), "seconds": seconds})
            + "\n"
        )


def _gloo_available() -> bool:
    try:
        return bool(torch.distributed.is_available()) and bool(
            torch.distributed.is_gloo_available()
        )
    except Exception:
        return False


class TestTwoProcessGlooExperiment(unittest.TestCase):
    """The P1 hang and its fix, on a real two-process gloo group."""

    def _wait_until(self, predicate: Callable[[], bool], deadline: float) -> bool:
        """Poll (the children write from other processes) until the deadline."""
        end = time.monotonic() + deadline
        while time.monotonic() < end:
            if predicate():
                return True
            time.sleep(0.05)
        return predicate()

    def test_fixed_entry_returns_while_the_pre_fix_wiring_blocks(self) -> None:
        if not _gloo_available():
            self.skipTest(
                "this torch build has no gloo, so the two-process experiment "
                "cannot create a real process group"
            )
        with tempfile.TemporaryDirectory(prefix="tp-init-sync-") as out_dir:
            init_file = os.path.join(out_dir, "gloo-init-file")
            context = multiprocessing.get_context("spawn")
            procs = [
                context.Process(
                    target=_run_experiment_rank,
                    args=(rank, init_file, out_dir),
                    name=f"tp-init-sync-rank{rank}",
                )
                for rank in (0, 1)
            ]
            for proc in procs:
                proc.start()
            try:
                self._assert_fixed_phase(out_dir)
                self._assert_pre_fix_phase(out_dir)
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

    def _assert_fixed_phase(self, out_dir: str) -> None:
        returned = self._wait_until(
            lambda: all(len(_records(out_dir, rank)) >= 1 for rank in (0, 1)),
            _STARTUP_DEADLINE_SECONDS,
        )
        self.assertTrue(
            returned,
            f"the fixed entry did not let both ranks return: "
            f"rank0={_records(out_dir, 0)!r} rank1={_records(out_dir, 1)!r}",
        )
        for rank in (0, 1):
            record = _records(out_dir, rank)[0]
            self.assertEqual(record["phase"], _PHASE_FIXED)
            self.assertEqual(record["result"], [False] * len(WRITE_KEYS))
            self.assertLess(
                record["seconds"],
                _PHASE_DEADLINE_SECONDS,
                "the agreement must answer in bounded time",
            )
        self.assertFalse(
            os.path.exists(_marker_path(out_dir, 0, _PHASE_FIXED))
            or os.path.exists(_marker_path(out_dir, 1, _PHASE_FIXED)),
            "no rank may open a write session when the group is not ready",
        )

    def _assert_pre_fix_phase(self, out_dir: str) -> None:
        """Control: rank 1 is gone and rank 0 blocks in the broadcast."""
        self.assertTrue(
            self._wait_until(
                lambda: len(_records(out_dir, 1)) >= 2, _PHASE_DEADLINE_SECONDS
            ),
            "rank 1 must return from the pre-fix call exactly as it used to",
        )
        self.assertEqual(_records(out_dir, 1)[1]["result"], [False] * len(WRITE_KEYS))
        self.assertTrue(
            self._wait_until(
                lambda: os.path.exists(_marker_path(out_dir, 0, _PHASE_PRE_FIX)),
                _PHASE_DEADLINE_SECONDS,
            ),
            "rank 0 must reach the write session: that is the hang's doorstep",
        )
        self.assertFalse(
            self._wait_until(
                lambda: len(_records(out_dir, 0)) >= 2, _PHASE_DEADLINE_SECONDS
            ),
            "pre-fix wiring: rank 0 must still be blocked in the broadcast "
            "that rank 1 will never reach",
        )


if __name__ == "__main__":
    unittest.main()
