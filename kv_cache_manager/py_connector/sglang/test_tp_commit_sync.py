"""CPU regressions for TP commit outcomes and failed storage groups.

Run from the repository root with a compatible SGLang and CPU-capable torch:
    PYTHONPATH=$PWD python kv_cache_manager/py_connector/sglang/test_tp_commit_sync.py

The SDK and Manager are faked by the existing initialization fixture. The
backup loop calls the production entry point for up to two batches and stops
on a failed page. Both a blocking thread group and a real two-process Gloo
group exercise the first-commit timeout with another batch still pending.
"""

import copy
import multiprocessing
import os
import tempfile
import threading
import unittest
from datetime import timedelta
from unittest import mock

import torch

from test_tp_init_sync import (
    ScriptedConnector,
    WRITE_KEYS,
    _CollectiveScript,
    _host_indices,
    _transfer,
)


class WriteGroup:
    """Blocking collectives with separate arrival and consumption barriers."""

    def __init__(self):
        self.barrier = threading.Barrier(2, timeout=3)
        self.local = threading.local()
        self.values = {}
        self.ops = {0: [], 1: []}
        self.payload = None
        self.fail_at = None

    def broadcast(self, values, *, group_src, group):
        rank = self.local.rank
        assert group is self and group_src == 0
        operation = (
            "finish" if isinstance(values[0], bool) or len(values) == 1 else "start"
        )
        # A receiver's empty one-element list is also a finish broadcast in v1.
        if len(values) == 4:
            operation = "start"
        self.ops[rank].append(operation)
        if self.fail_at == operation:
            raise RuntimeError("injected collective failure")
        if rank == 0:
            self.payload = copy.deepcopy(values)
        self.barrier.wait()
        values[:] = copy.deepcopy(self.payload)
        self.barrier.wait()

    def reduce(self, values, *, op, group):
        rank = self.local.rank
        assert group is self and op == torch.distributed.ReduceOp.MIN
        operation = "ready" if not self.ops[rank] else "flags"
        self.ops[rank].append(operation)
        if self.fail_at == operation:
            raise RuntimeError("injected collective failure")
        self.values[rank] = values.clone()
        self.barrier.wait()
        values.copy_(torch.minimum(self.values[0], self.values[1]))
        self.barrier.wait()

    def gather(self, values, local, *, group):
        rank = self.local.rank
        assert group is self
        self.ops[rank].append("schedule")
        if self.fail_at == "schedule":
            raise RuntimeError("injected collective failure")
        self.values[rank] = copy.deepcopy(local)
        self.barrier.wait()
        values[:] = [copy.deepcopy(self.values[i]) for i in range(2)]
        self.barrier.wait()

    def patched(self):
        from contextlib import ExitStack

        stack = ExitStack()
        for name, callback in (
            ("broadcast_object_list", self.broadcast),
            ("all_reduce", self.reduce),
            ("all_gather_object", self.gather),
        ):
            stack.enter_context(mock.patch.object(torch.distributed, name, callback))
        return stack

    def run(self, work):
        results = [None, None]
        errors = []

        def target(rank):
            self.local.rank = rank
            try:
                results[rank] = work(rank)
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=target, args=(rank,), daemon=True)
            for rank in range(2)
        ]
        with self.patched():
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            if any(thread.is_alive() for thread in threads):
                self.barrier.abort()
                raise AssertionError("a rank did not finish its collective sequence")
        if errors:
            raise errors[0]
        return results


def connector_for(rank, group, *, blocks_to_write=2, **kwargs):
    return ScriptedConnector(
        local_init_ok=True,
        script=_CollectiveScript(peer_ready=True),
        tp_rank=rank,
        group=group,
        blocks_to_write=blocks_to_write,
        with_mamba=kwargs.pop("with_mamba", True),
        **kwargs,
    )


def backup(connector):
    results = []
    for _ in range(2):
        result = connector.batch_set_v1(WRITE_KEYS, _host_indices())
        results.append(result)
        if not all(result):
            break
    return results


class TestCommitOutcome(unittest.TestCase):
    def pair(self, *, blocks_to_write=2):
        group = WriteGroup()
        return group, [
            connector_for(rank, group, blocks_to_write=blocks_to_write)
            for rank in range(2)
        ]

    def test_first_commit_timeout_stops_both_ranks_before_next_batch(self):
        for blocks_to_write in (0, 2, 4):
            with self.subTest(blocks_to_write=blocks_to_write):
                group, connectors = self.pair(blocks_to_write=blocks_to_write)
                connectors[
                    0
                ]._manager_client.finish_write_cache.side_effect = TimeoutError(
                    "commit response timed out"
                )
                results = group.run(lambda rank: backup(connectors[rank]))
                self.assertEqual(results, [[[False] * 4]] * 2)
                self.assertEqual(group.ops[0], group.ops[1])
                self.assertEqual(group.ops[0][-1], "finish")
                self.assertEqual(
                    connectors[0]._manager_client.start_write_cache.call_count, 1
                )
                self.assertEqual(
                    connectors[0]._manager_client.finish_write_cache.call_count, 1
                )
                self.assertEqual(
                    connectors[1]._manager_client.finish_write_cache.call_count, 0
                )
                self.assertFalse(any(c._storage_group_failed for c in connectors))

    def test_success_runs_both_batches(self):
        for blocks_to_write in (0, 2, 4):
            with self.subTest(blocks_to_write=blocks_to_write):
                group, connectors = self.pair(blocks_to_write=blocks_to_write)
                self.assertEqual(
                    group.run(lambda rank: backup(connectors[rank])),
                    [[[True] * 4] * 2] * 2,
                )
                self.assertEqual(group.ops[0], group.ops[1])
                self.assertEqual(
                    connectors[0]._manager_client.start_write_cache.call_count, 2
                )
                self.assertEqual(
                    connectors[0]._manager_client.finish_write_cache.call_count, 2
                )

    def test_manager_timeout_does_not_disable_next_backup(self):
        group, connectors = self.pair()
        connectors[0]._manager_client.finish_write_cache.side_effect = [
            TimeoutError("commit response timed out"),
            None,
            None,
        ]
        self.assertEqual(
            group.run(lambda rank: backup(connectors[rank])), [[[False] * 4]] * 2
        )
        self.assertEqual(
            group.run(lambda rank: backup(connectors[rank])), [[[True] * 4] * 2] * 2
        )

    def test_one_rank_transfer_failure_keeps_per_block_minimum(self):
        group, connectors = self.pair()
        connectors[1].transfer_client.SaveKvCaches.side_effect = RuntimeError(
            "transfer failed"
        )
        self.assertEqual(
            group.run(lambda rank: backup(connectors[rank])),
            [[[True, True, False, False]]] * 2,
        )
        request = connectors[0]._manager_client.finish_write_cache.call_args.args[0]
        self.assertEqual(
            request["success_blocks"]["bool_masks"]["values"], [False, False]
        )

    def test_start_failure_stops_without_a_commit_broadcast(self):
        group, connectors = self.pair()
        connectors[0]._manager_client.start_write_cache.side_effect = TimeoutError(
            "start response timed out"
        )
        self.assertEqual(
            group.run(lambda rank: backup(connectors[rank])), [[[False] * 4]] * 2
        )
        self.assertEqual(group.ops, {0: ["ready", "start"], 1: ["ready", "start"]})

    def test_single_rank_and_mla_timeout_stays_local(self):
        for world_size, mla in ((1, False), (2, True)):
            for blocks in (0, 2):
                with self.subTest(world_size=world_size, mla=mla, blocks=blocks):
                    connector = connector_for(
                        0,
                        None,
                        tp_world_size=world_size,
                        is_mla_model=mla,
                        blocks_to_write=blocks,
                    )
                    connector._manager_client.finish_write_cache.side_effect = (
                        TimeoutError("commit response timed out")
                    )
                    with mock.patch.object(
                        torch.distributed, "broadcast_object_list"
                    ) as broadcast:
                        self.assertEqual(backup(connector), [[False] * 4])
                    broadcast.assert_not_called()


class TestFailedStorageGroup(unittest.TestCase):
    def test_collective_failure_rejects_later_v1_and_v2_before_start(self):
        for phase in ("ready", "start", "flags", "finish", "schedule"):
            with self.subTest(phase=phase):
                group = WriteGroup()
                connectors = [
                    connector_for(rank, group, with_mamba=True) for rank in range(2)
                ]
                group.fail_at = phase
                if phase == "schedule":
                    results = group.run(
                        lambda rank: connectors[rank].batch_set_v2([_transfer()])
                    )
                    self.assertTrue(
                        all(
                            result[_transfer().name] == [False, False]
                            for result in results
                        )
                    )
                else:
                    self.assertEqual(
                        group.run(lambda rank: backup(connectors[rank])),
                        [[[False] * 4]] * 2,
                    )
                counts = [
                    c._manager_client.start_write_cache.call_count for c in connectors
                ]
                operations = copy.deepcopy(group.ops)
                group.fail_at = None
                for connector in connectors:
                    self.assertTrue(connector._storage_group_failed)
                    self.assertEqual(backup(connector), [[False] * 4])
                    self.assertEqual(
                        connector.batch_set_v2([_transfer()])[_transfer().name],
                        [False, False],
                    )
                    # The guard applies even to a path that would run rank-0 only.
                    connector.is_mla_model = True
                    self.assertEqual(backup(connector), [[False] * 4])
                    callback = mock.Mock()
                    with self.assertRaisesRegex(RuntimeError, "re-attach"):
                        connector._storage_collective(callback)
                    callback.assert_not_called()
                self.assertEqual(
                    [
                        c._manager_client.start_write_cache.call_count
                        for c in connectors
                    ],
                    counts,
                )
                self.assertEqual(group.ops, operations)


def _gloo_rank(rank, init_file, output):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=5),
    )
    try:
        connector = connector_for(rank, torch.distributed.group.WORLD)
        connector._manager_client.finish_write_cache.side_effect = TimeoutError(
            "commit response timed out"
        )
        failed = backup(connector)
        connector._manager_client.finish_write_cache.side_effect = None
        recovered = backup(connector)
        output.put((rank, failed, recovered, connector._storage_group_failed))
    finally:
        torch.distributed.destroy_process_group()


@unittest.skipUnless(torch.distributed.is_gloo_available(), "torch has no Gloo backend")
class TestRealGlooCommit(unittest.TestCase):
    def test_timeout_with_next_batch_then_recovery(self):
        context = multiprocessing.get_context("spawn")
        output = context.Queue()
        with tempfile.TemporaryDirectory(prefix="tp-commit-") as directory:
            processes = [
                context.Process(
                    target=_gloo_rank,
                    args=(rank, os.path.join(directory, "init"), output),
                )
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            try:
                records = sorted(output.get(timeout=60) for _ in processes)
                for rank, failed, recovered, group_failed in records:
                    self.assertEqual(
                        failed, [[False] * 4], f"rank {rank} entered the next batch"
                    )
                    self.assertEqual(recovered, [[True] * 4] * 2)
                    self.assertFalse(group_failed)
                for process in processes:
                    process.join(timeout=10)
                    self.assertEqual(process.exitcode, 0)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=10)
                output.close()
                output.join_thread()


if __name__ == "__main__":
    unittest.main()
