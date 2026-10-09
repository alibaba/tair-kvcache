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
from typing import Any, Callable, Iterator, Optional
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
# One write call: four new blocks, the Manager asks for the last two.
WRITE_KEYS = ["block-0", "block-1", "block-2", "block-3"]
BLOCK_MASK_OFFSET = 2
# The process group of the single-process tests.  Only its identity matters:
# the collectives themselves are scripted below.
GROUP = object()


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
        tp_world_size: int = 2,
        tp_rank: int = 1,
        group: Any = GROUP,
        is_mla_model: bool = False,
        on_write_session: Optional[Callable[[], None]] = None,
    ) -> None:
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        self.is_mla_model = is_mla_model
        self.storage_tp_group = group
        self.kv_factor = 2
        self.instance_id = "tp-init-sync-test"
        self.location_spec_name = f"tp_{tp_rank}"
        self.location_spec_size = 4096
        self.write_timeout_seconds = 5
        self.has_mamba = False
        self.has_indexer = False
        self.registered_pools = {}
        self.backup_pgs = []
        self.backup_bandwidth = []
        self._init_lock = threading.Lock()
        self._client_ready = False
        self._closed = False
        self._tp_init_agreed = False
        self.local_init_ok = local_init_ok
        self.init_attempts = 0
        self.on_write_session = on_write_session

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
        self.transfer_client.SaveKvCaches.return_value = (
            _mock_kvcm.ClientErrorCode.ER_OK,
        )

    def _ensure_client(self) -> None:
        self.init_attempts += 1
        if not self.local_init_ok:
            raise RuntimeError("simulated local initialization failure")
        self._client_ready = True

    def _start_write_cache(self, request: Any) -> dict:
        """What a Manager answers for ``WRITE_KEYS``; the last two need writing."""
        if self.on_write_session is not None:
            self.on_write_session()
        return {
            "locations": [
                {
                    "location_specs": [
                        {"name": self.location_spec_name, "uri": f"uri-{i}"}
                    ]
                }
                for i in range(BLOCK_MASK_OFFSET)
            ],
            "write_session_id": "write-session-1",
            "block_mask": {"offset": BLOCK_MASK_OFFSET},
        }


class _CollectiveScript:
    """Records this rank's TP collectives and plays the peer's part."""

    def __init__(self, *, peer_ready: bool, peer_wrote: bool = True) -> None:
        # Mutable: a test can let the peer recover between two calls.
        self.peer_ready = peer_ready
        self.peer_wrote = peer_wrote
        # The agreement rounds: (operator, shape, group).
        self.agreements: list = []
        # The write path's own collectives, i.e. what must not be reached when
        # the group agreed that it is not ready to write.
        self.broadcasts: list = []
        self.reduces: list = []


@contextmanager
def _patched_tp_collectives(script: _CollectiveScript) -> Iterator[_CollectiveScript]:
    """Run one rank's view of the group's collectives, scripted and recorded.

    The one-element ``all_reduce`` is the agreement under test: the tensor
    starts as this rank's own outcome and the peer's outcome is folded in with
    MIN, exactly like the real reduce.  Everything else on the group belongs to
    the write path -- a real peer would block there, so reaching it is recorded
    and left for the test to assert on.
    """

    def fake_all_reduce(tensor: Any, op: Any = None, group: Any = None) -> None:
        if tensor.numel() == 1:
            script.agreements.append((op, tuple(tensor.shape), group))
            if not script.peer_ready:
                tensor.fill_(0)
            return
        script.reduces.append((op, group))
        if not script.peer_wrote:
            tensor.fill_(0)

    def fake_broadcast(*args: Any, **kwargs: Any) -> None:
        script.broadcasts.append(kwargs.get("group"))

    with (
        mock.patch.object(torch.distributed, "all_reduce", fake_all_reduce),
        mock.patch.object(torch.distributed, "broadcast_object_list", fake_broadcast),
    ):
        yield script


def _write_connector(**kwargs: Any) -> ScriptedConnector:
    return ScriptedConnector(**kwargs)


def _transfer() -> PoolTransfer:
    return PoolTransfer(
        name=PoolName.MAMBA,
        host_indices=torch.arange(2),
        keys=WRITE_KEYS[:2],
    )


def _host_indices() -> torch.Tensor:
    return torch.arange(len(WRITE_KEYS) * PAGE_SIZE)


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

    _ROUND = (torch.distributed.ReduceOp.MIN, (1,), GROUP)

    def _write_sessions(self, connector: ScriptedConnector) -> Any:
        """The mocked Manager's ``start_write_cache`` (Any: it is a Mock)."""
        manager: Any = connector._manager_client
        return manager.start_write_cache

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
                connector = _write_connector(local_init_ok=False)
                script = _CollectiveScript(peer_ready=peer_ready)

                with _patched_tp_collectives(script):
                    v1_result = connector.batch_set_v1(WRITE_KEYS, _host_indices())
                    v2_result = connector.batch_set_v2([_transfer()])

                self.assertEqual(v1_result, [False] * len(WRITE_KEYS))
                self.assertEqual(v2_result, {PoolName.MAMBA: [False, False]})
                self._assert_rounds(script, calls=2)
                self.assertEqual(script.broadcasts, [])
                self.assertEqual(script.reduces, [])
                self.assertFalse(connector._client_ready)
                self.assertFalse(connector._tp_init_agreed)
                self.assertEqual(connector.init_attempts, 2, "both paths retried")
                self.assertIsNone(
                    self._write_sessions(connector).call_args,
                    "no write session may be opened when the group is not ready",
                )

    def test_peer_failure_stops_a_ready_rank_without_a_collective(self) -> None:
        """A rank that initialized fine also refuses when a peer did not."""
        connector = _write_connector(local_init_ok=True)
        script = _CollectiveScript(peer_ready=False)

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
            self._write_sessions(connector).call_args,
            "no write session may be opened when the group is not ready",
        )

    def test_a_failed_round_is_retried_and_recovers(self) -> None:
        """Nothing is sticky: the next call initializes and agrees again."""
        connector = _write_connector(local_init_ok=False, tp_rank=0)
        script = _CollectiveScript(peer_ready=True)

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
            self._write_sessions(connector).call_args,
            "the recovered call must write like any healthy call",
        )

    def test_a_ready_group_agrees_once_and_then_writes_as_before(self) -> None:
        """Whole group ready: the write path runs, and the agreement is paid once."""
        connector = _write_connector(local_init_ok=True, tp_rank=0)
        script = _CollectiveScript(peer_ready=True)

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
        self.assertEqual(len(script.reduces), 1, "the write path's per-block reduce")
        self.assertEqual(connector.init_attempts, 1)


class TestPathsThatDoNotJoinTheGroup(unittest.TestCase):
    """No collective is added where the write paths run none themselves."""

    def test_single_rank_write_is_unchanged(self) -> None:
        connector = _write_connector(
            local_init_ok=True, tp_world_size=1, tp_rank=0, group=None
        )
        script = _CollectiveScript(peer_ready=True)

        with _patched_tp_collectives(script):
            result = connector.batch_set_v1(WRITE_KEYS, _host_indices())

        self.assertEqual(result, [True] * len(WRITE_KEYS))
        self.assertEqual(script.agreements, [])
        self.assertEqual(script.broadcasts, [])
        self.assertEqual(script.reduces, [])

    def test_single_rank_failure_stays_local(self) -> None:
        connector = _write_connector(
            local_init_ok=False, tp_world_size=1, tp_rank=0, group=None
        )
        script = _CollectiveScript(peer_ready=True)

        with _patched_tp_collectives(script):
            result = connector.batch_set_v1(WRITE_KEYS, _host_indices())

        self.assertEqual(result, [False] * len(WRITE_KEYS))
        self.assertEqual(script.agreements, [])
        self.assertEqual(script.broadcasts, [])
        self.assertEqual(connector.init_attempts, 1)

    def test_mla_write_does_not_join_the_group(self) -> None:
        """MLA writes from rank 0 only and uses no collective at all."""
        script = _CollectiveScript(peer_ready=True)
        with _patched_tp_collectives(script):
            for tp_rank, expected in (
                (0, [True] * len(WRITE_KEYS)),
                (1, [False] * len(WRITE_KEYS)),
            ):
                with self.subTest(tp_rank=tp_rank):
                    connector = _write_connector(
                        local_init_ok=True,
                        tp_rank=tp_rank,
                        is_mla_model=True,
                    )
                    result = connector.batch_set_v1(WRITE_KEYS, _host_indices())
                    self.assertEqual(result, expected)
                    self.assertEqual(connector.init_attempts, 1)

        self.assertEqual(script.agreements, [])
        self.assertEqual(script.broadcasts, [])
        self.assertEqual(script.reduces, [])

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
        connector = _write_connector(local_init_ok=True, group=None)
        script = _CollectiveScript(peer_ready=True)

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
        connector = _write_connector(local_init_ok=True)
        connector._closed = True
        script = _CollectiveScript(peer_ready=True)

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
        connector = ScriptedConnector(
            local_init_ok=(rank == 0),
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
