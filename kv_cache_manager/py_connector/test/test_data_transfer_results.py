"""Unit tests for MultiResult flattening and the save/load done callbacks.

The done callbacks decode a flat result list whose layout is an implicit
contract with ``_submit_group_tasks``: tasks are submitted group-major
(group0's blocks, then group1's blocks, ...), so a manager block's success is
the stride-AND ``flat[i % num_blocks]``. These tests pin that contract with
hand-computed expectations.
"""

import threading
import unittest
from unittest.mock import MagicMock

from kv_cache_manager.py_connector.test import vllm_stubs  # noqa: F401 (stubs)
from kv_cache_manager.py_connector.vllm.data_transfer import (
    DataTransferManager, MultiResult, _PinnedBudget)
from kv_cache_manager.py_connector.common.tp_coordinator import (
    CoordinateMsgSerializer)


class TestMultiResult(unittest.TestCase):
    def test_flatten_in_submission_order(self):
        got = []
        mr = MultiResult(3, got.extend)
        mr.submit_result(0, [True, False])
        mr.submit_result(1, [False])
        mr.submit_result(2, [True, True, True])
        self.assertEqual(got, [True, False, False, True, True, True])

    def test_out_of_order_submit(self):
        got = []
        mr = MultiResult(3, got.extend)
        mr.submit_result(2, ["c"])
        mr.submit_result(0, ["a"])
        self.assertEqual(got, [])  # callback must not fire early
        mr.submit_result(1, ["b"])
        self.assertEqual(got, ["a", "b", "c"])

    def test_duplicate_submit_asserts(self):
        mr = MultiResult(2, lambda flat: None)
        mr.submit_result(0, [True])
        with self.assertRaises(AssertionError):
            mr.submit_result(0, [True])

    def test_concurrent_submit(self):
        n = 64
        results = []
        done = threading.Event()

        def cb(flat):
            results.append(flat)
            done.set()

        mr = MultiResult(n, cb)
        barrier = threading.Barrier(n)

        def worker(i):
            barrier.wait()
            mr.submit_result(i, [i])

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertTrue(done.wait(timeout=5))
        self.assertEqual(len(results), 1)  # callback fires exactly once
        self.assertEqual(results[0], list(range(n)))


def _make_dtm():
    """DataTransferManager with only the state the callbacks touch."""
    dtm = DataTransferManager.__new__(DataTransferManager)
    dtm._coordinator_client = MagicMock()
    dtm._pinned_budget = _PinnedBudget(1 << 30)
    return dtm


def _sent_event(dtm):
    (payload,), _ = dtm._coordinator_client.send.call_args
    return CoordinateMsgSerializer.loads(payload).content


class TestSaveDoneCallback(unittest.TestCase):
    def test_multi_group_stride_and(self):
        # 3 blocks x 2 groups, flat = group0[b0,b1,b2] + group1[b0,b1,b2].
        # Block b is saved only if both groups succeeded for b.
        dtm = _make_dtm()
        cb = dtm.create_save_done_callback("req", 0, "sess", num_blocks=3)
        cb([True, True, False,   # group 0
            True, False, True])  # group 1
        evt = _sent_event(dtm)
        self.assertEqual(evt.type, "SendBlockFinishedEvent")
        self.assertEqual(evt.write_session_id, "sess")
        self.assertEqual(evt.is_success_list, [True, False, False])

    def test_single_group_passthrough(self):
        dtm = _make_dtm()
        cb = dtm.create_save_done_callback("req", 1, "sess", num_blocks=2)
        cb([False, True])
        self.assertEqual(_sent_event(dtm).is_success_list, [False, True])


class TestLoadDoneCallback(unittest.TestCase):
    def test_multi_group_failure_merge(self):
        dtm = _make_dtm()
        cb = dtm.create_load_done_callback(
            "req", 0, epoch=7, block_ids=[10, 20, 30], num_blocks=3)
        cb([True, False, True,    # group 0
            True, True, False])   # group 1
        evt = _sent_event(dtm)
        self.assertEqual(evt.type, "LoadBlockFinishedEvent")
        self.assertEqual(evt.epoch, 7)
        # blocks 1 and 2 each failed in one group -> report their table ids.
        self.assertEqual(evt.failed_block_idxs, [20, 30])

    def test_all_success_reports_empty(self):
        dtm = _make_dtm()
        cb = dtm.create_load_done_callback(
            "req", 0, epoch=0, block_ids=[10, 20], num_blocks=2)
        cb([True, True, True, True])
        self.assertEqual(_sent_event(dtm).failed_block_idxs, [])

    def test_report_failures_false_hybrid(self):
        # Hybrid models cannot report invalid block ids to vLLM: the failure
        # must be swallowed (empty failed list) but the finished event still sent.
        dtm = _make_dtm()
        cb = dtm.create_load_done_callback(
            "req", 0, epoch=1, block_ids=[], num_blocks=2, report_failures=False)
        cb([False, True])
        evt = _sent_event(dtm)
        self.assertEqual(evt.type, "LoadBlockFinishedEvent")
        self.assertEqual(evt.failed_block_idxs, [])


class TestNullStateBlocks(unittest.TestCase):
    """Mamba 'align' mode: null (id 0) state targets carry no state by design
    (vLLM materializes states only at segment boundaries), so save/load must
    treat them as vacuous successes -- failing them would stride-AND whole
    manager blocks out of the manager's prefix chain and kill multi-block
    hybrid caching. The all-null path takes no GPU work, so it runs on CPU."""

    @staticmethod
    def _state_group():
        from kv_cache_manager.py_connector.common.types import TransferGroup
        return TransferGroup(
            group_idx=0, spec_name="tp0_g0", is_attention=False,
            layer_names=["m0"], block_size=528, per_block_bytes=1024,
            kernel_block_size=528)

    def test_save_all_null_blocks_vacuously_succeed(self):
        dtm = _make_dtm()
        results = {}
        mr = MultiResult(1, lambda flat: results.setdefault("flat", flat))
        dtm.save_task(mr, 0, self._state_group(),
                      remote_uris=["u0", "u1"],
                      block_token_indices=None,
                      block_ids=[0, 0],
                      ready_event=None)
        self.assertEqual(results["flat"], [True, True])

    def test_load_all_null_blocks_vacuously_succeed(self):
        dtm = _make_dtm()
        results = {}
        mr = MultiResult(1, lambda flat: results.setdefault("flat", flat))
        dtm.load_task(mr, 0, self._state_group(),
                      remote_uris=["u0", "u1", "u2"],
                      block_token_indices=None,
                      block_ids=[0, 0, 0])
        self.assertEqual(results["flat"], [True, True, True])


class TestPinnedBudget(unittest.TestCase):
    """The staging budget must bound *concurrent* pinned bytes: tasks over the
    watermark wait instead of allocating, and an oversized request may only
    run alone (no deadlock)."""

    def test_peak_usage_never_exceeds_capacity(self):
        capacity = 4
        budget = _PinnedBudget(capacity)
        peak = [0]
        used = [0]
        lock = threading.Lock()
        start = threading.Barrier(8)

        def worker():
            start.wait()
            for _ in range(25):
                budget.acquire(2)
                with lock:
                    used[0] += 2
                    peak[0] = max(peak[0], used[0])
                with lock:
                    used[0] -= 2
                budget.release(2)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertLessEqual(peak[0], capacity)
        self.assertGreater(peak[0], 0)

    def test_acquire_blocks_until_release(self):
        budget = _PinnedBudget(10)
        budget.acquire(8)
        acquired = threading.Event()

        def second():
            budget.acquire(8)  # 8 + 8 > 10: must wait
            acquired.set()

        t = threading.Thread(target=second)
        t.start()
        self.assertFalse(acquired.wait(timeout=0.2))
        budget.release(8)
        self.assertTrue(acquired.wait(timeout=5))
        budget.release(8)
        t.join()

    def test_oversized_request_runs_alone(self):
        # A single request larger than the capacity must not deadlock: it
        # proceeds once the budget is otherwise idle.
        budget = _PinnedBudget(4)
        budget.acquire(100)
        # While it holds the budget, nothing else may enter.
        entered = threading.Event()
        t = threading.Thread(target=lambda: (budget.acquire(1), entered.set()))
        t.start()
        self.assertFalse(entered.wait(timeout=0.2))
        budget.release(100)
        self.assertTrue(entered.wait(timeout=5))
        budget.release(1)
        t.join()

    def test_default_capacity_derivation(self):
        # No explicit cap: 2 x per-task peak x groups, floored at 1 GiB.
        from types import SimpleNamespace
        from kv_cache_manager.py_connector.common.types import (
            KVCacheInfo, TransferGroup)

        def capacity(per_block_bytes_list, cap=0, per_task=128):
            groups = [TransferGroup(
                group_idx=i, spec_name=f"tp0_g{i}", is_attention=True,
                layer_names=[f"l{i}"], block_size=16, per_block_bytes=pb)
                for i, pb in enumerate(per_block_bytes_list)]
            info = KVCacheInfo(tp_rank=0, world_size=1, groups=groups,
                               device="cpu", dtype=None)
            cfg = SimpleNamespace(staging_buffer_max_bytes=cap,
                                  block_per_save_task=per_task,
                                  block_per_load_task=per_task)
            return DataTransferManager._staging_capacity_bytes(info, cfg)

        # Small blocks: floor of 1 GiB wins.
        self.assertEqual(capacity([1024]), 1 << 30)
        # Large blocks: 2 * (32 MiB * 128 blocks) * 2 groups.
        self.assertEqual(capacity([32 << 20, 16 << 20]),
                         2 * (32 << 20) * 128 * 2)
        # Explicit cap wins.
        self.assertEqual(capacity([32 << 20], cap=123456789), 123456789)

    def test_load_task_acquires_and_releases_stage_bytes(self):
        # The transfer path must hold the budget for exactly the staged bytes
        # and release it even on failure (LoadKvCaches returns an error here).
        from kv_cache_manager.py_connector.common.types import TransferGroup
        dtm = _make_dtm()
        dtm._transfer_client = MagicMock()
        dtm._transfer_client.LoadKvCaches.return_value = "ERR"
        calls = []
        dtm._pinned_budget = MagicMock()
        dtm._pinned_budget.acquire.side_effect = lambda b: calls.append(("acq", b))
        dtm._pinned_budget.release.side_effect = lambda b: calls.append(("rel", b))
        group = TransferGroup(
            group_idx=0, spec_name="tp0_g0", is_attention=False,
            layer_names=["m0"], block_size=528, per_block_bytes=1024)
        results = {}
        mr = MultiResult(1, lambda flat: results.setdefault("flat", flat))
        dtm.load_task(mr, 0, group, remote_uris=["u0", "u1"],
                      block_token_indices=None, block_ids=[3, 4])
        self.assertEqual(calls, [("acq", 2 * 1024), ("rel", 2 * 1024)])
        self.assertEqual(results["flat"], [False, False])


if __name__ == "__main__":
    unittest.main()
