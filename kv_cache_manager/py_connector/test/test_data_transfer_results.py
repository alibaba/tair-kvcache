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
    DataTransferManager, MultiResult)
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


if __name__ == "__main__":
    unittest.main()
