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


class TestNullStateBlocks(unittest.TestCase):
    """Mamba 'align' mode: vLLM materializes a recurrent state only at segment
    boundaries, so a state group's null (id 0) target carries no state.

    The connector must never turn that absence into a *success*: the block would
    be published as fully cached and a later, shorter request would resume from
    a state URI nobody ever wrote. Instead the state group abstains for such a
    block (None = "no data of mine here"), and the block's verdict is decided by
    the groups that did carry data -- the missing state is expressed to the
    manager through the block's spec coverage (see
    v1_connector._spec_groups). These paths do no GPU work, so they run on CPU.
    """

    @staticmethod
    def _state_group():
        from kv_cache_manager.py_connector.common.types import TransferGroup
        return TransferGroup(
            group_idx=0, spec_name="tp0_g0", is_attention=False,
            layer_names=["m0"], block_size=528, per_block_bytes=1024,
            kernel_block_size=528)

    @staticmethod
    def _attn_group():
        from kv_cache_manager.py_connector.common.types import TransferGroup
        return TransferGroup(
            group_idx=1, spec_name="tp0_g1", is_attention=True,
            layer_names=["a0"], block_size=528, per_block_bytes=1024,
            kernel_block_size=528)

    def _run(self, method, **kwargs):
        dtm = _make_dtm()
        results = {}
        mr = MultiResult(1, lambda flat: results.setdefault("flat", flat))
        getattr(dtm, method)(mr, 0, self._state_group(), **kwargs)
        return results["flat"]

    def test_save_null_state_blocks_abstain_not_succeed(self):
        # No state and (consistently) no location for it: the group abstains.
        flat = self._run("save_task",
                         remote_uris=[None, None],
                         block_token_indices=None,
                         block_ids=[0, 0],
                         ready_event=None)
        self.assertEqual(flat, [None, None])

    def test_save_null_state_with_location_fails(self):
        # The manager allocated a state location but vLLM has no state to put
        # there: publishing it would advertise unwritten bytes.
        flat = self._run("save_task",
                         remote_uris=["u0", None],
                         block_token_indices=None,
                         block_ids=[0, 0],
                         ready_event=None)
        self.assertEqual(flat, [False, None])

    def test_save_real_state_without_location_fails(self):
        # A state exists but was not announced: it cannot be published.
        flat = self._run("save_task",
                         remote_uris=[None],
                         block_token_indices=None,
                         block_ids=[7],
                         ready_event=None)
        self.assertEqual(flat, [False])

    def test_load_null_state_targets_abstain(self):
        # vLLM does not need a state for these blocks (only the block ending
        # the reused prefix does), whatever the manager published.
        flat = self._run("load_task",
                         remote_uris=["u0", "u1", "u2"],
                         block_token_indices=None,
                         block_ids=[0, 0, 0])
        self.assertEqual(flat, [None, None, None])

    def test_load_real_target_without_location_fails(self):
        # vLLM needs this state but nothing was published for it: the request
        # must not run on an unwritten state.
        flat = self._run("load_task",
                         remote_uris=[None],
                         block_token_indices=None,
                         block_ids=[9])
        self.assertEqual(flat, [False])

    def test_attention_block_without_location_fails(self):
        # Attention KV is never sparse: a missing location is a failure, and
        # the block must be kept out of the staging batch so the remaining
        # buffers stay aligned with the URI list.
        dtm = _make_dtm()
        skipped, failed = dtm._save_dispositions(
            self._attn_group(), remote_uris=["u0", None, "u2"],
            block_ids=None, n=3)
        self.assertEqual((skipped, failed), (set(), {1}))

    def test_save_dispositions_state_group(self):
        dtm = _make_dtm()
        # block0: no state, nothing published -> abstain
        # block1: state + location      -> transfer
        # block2: no state but published -> fail
        # block3: state but unpublished  -> fail
        skipped, failed = dtm._save_dispositions(
            self._state_group(), remote_uris=[None, "u1", "u2", None],
            block_ids=[0, 5, 0, 6], n=4)
        self.assertEqual(skipped, {0})
        self.assertEqual(failed, {2, 3})


class TestTaskCrashReporting(unittest.TestCase):
    """A task that dies mid-transfer must still report, all-failed.

    submit_task drops the future, so an escaping exception is silently
    swallowed: the MultiResult callback never fires, the save session hangs
    (SendBlockFinishedEvent never sent) and -- worse -- a load leaves vLLM
    believing KV it never received under the connector's synchronous-load
    contract. Both tasks wrap their body and report every block as failed."""

    _state_group = staticmethod(TestNullStateBlocks._state_group)

    def _run_crashing(self, method):
        from unittest.mock import patch
        dtm = _make_dtm()
        results = {}
        mr = MultiResult(1, lambda flat: results.setdefault("flat", flat))
        group = self._state_group()
        kwargs = dict(remote_uris=["u0", "u1"],
                      block_token_indices=None,
                      block_ids=[5, 6])
        crash = "_%s_valid_blocks" % method.split("_")[0]
        with patch.object(dtm, crash, side_effect=RuntimeError("boom")):
            if method == "save_task":
                kwargs["ready_event"] = None
            getattr(dtm, method)(mr, 0, group, **kwargs)
        return dtm, results["flat"]

    def test_save_task_crash_reports_all_failed(self):
        dtm, flat = self._run_crashing("save_task")
        self.assertEqual(flat, [False, False])

    def test_load_task_crash_reports_all_failed(self):
        dtm, flat = self._run_crashing("load_task")
        self.assertEqual(flat, [False, False])


class TestAbstainedVerdicts(unittest.TestCase):
    """A block's verdict is the AND over the groups that carried data for it.
    A group that abstained (None) must neither pass nor fail the block, and a
    block no group wrote at all must not be published."""

    def test_save_abstain_does_not_mask_other_group(self):
        dtm = _make_dtm()
        cb = dtm.create_save_done_callback("req", 0, "sess", num_blocks=3)
        cb([None, None, True,    # state group: only block 2 had a state
            True, False, True])  # attention group
        # Block 0 rides on attention alone, block 1 fails there, block 2 both.
        self.assertEqual(_sent_event(dtm).is_success_list, [True, False, True])

    def test_save_all_groups_abstain_is_not_published(self):
        dtm = _make_dtm()
        cb = dtm.create_save_done_callback("req", 0, "sess", num_blocks=2)
        cb([None, None])
        self.assertEqual(_sent_event(dtm).is_success_list, [False, False])

    def test_load_abstain_does_not_mask_other_group(self):
        dtm = _make_dtm()
        cb = dtm.create_load_done_callback(
            "req", 0, epoch=3, block_ids=[10, 20, 30], num_blocks=3)
        cb([None, None, False,   # state group: block 2 needed a state, failed
            True, True, True])   # attention group
        self.assertEqual(_sent_event(dtm).failed_block_idxs, [30])

    def test_load_all_groups_abstain_counts_as_failure(self):
        dtm = _make_dtm()
        cb = dtm.create_load_done_callback(
            "req", 0, epoch=0, block_ids=[11], num_blocks=1)
        cb([None])
        self.assertEqual(_sent_event(dtm).failed_block_idxs, [11])


if __name__ == "__main__":
    unittest.main()
