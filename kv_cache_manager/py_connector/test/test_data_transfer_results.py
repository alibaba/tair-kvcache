"""Unit tests for MultiResult flattening and the save/load done callbacks.

The done callbacks decode a flat result list whose layout is an implicit
contract with ``_submit_group_tasks``: tasks are submitted group-major
(group0's blocks, then group1's blocks, ...), so a manager block's success is
the stride-AND ``flat[i % num_blocks]``. These tests pin that contract with
hand-computed expectations.
"""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import torch

from kv_cache_manager.py_connector.test import vllm_stubs  # noqa: F401 (stubs)
from kv_cache_manager.py_connector.vllm.data_transfer import (
    DataTransferManager, MultiResult)
from kv_cache_manager.py_connector.vllm.transfer_types import KVLayout
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
        from kv_cache_manager.py_connector.vllm.transfer_types import StateTransferGroup
        return StateTransferGroup(
            group_idx=0, spec_name="tp0_g0",
            layer_names=["m0"], block_size=528, per_block_bytes=1024,
            layer_num=1, block_view_tensors=[], page_size_bytes=1024)

    @staticmethod
    def _attn_group():
        from kv_cache_manager.py_connector.vllm.transfer_types import AttentionTransferGroup
        return AttentionTransferGroup(
            group_idx=1, spec_name="tp0_g1",
            layer_names=["a0"], block_size=528, per_block_bytes=1024,
            layer_num=1, kv_layout=KVLayout.PACKED_4D,
            kvcache_ptr_tensor_gpu=None, num_kv_ptrs=1, per_token_dim=8,
            kernel_block_size=528, block_stride=0)

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


class TestStagingPool(unittest.TestCase):
    """_StagingPool: contiguous-run slot management with backpressure.

    The pool stages transfers in pinned host memory only (the kernel reaches
    it directly over PCIe); these tests pin the run bookkeeping: exact fit,
    fragmentation and re-merge on release, blocking acquire, and the capacity
    guard.
    """

    def _pool(self, max_blocks=8, block_bytes=16):
        from kv_cache_manager.py_connector.vllm.data_transfer import _StagingPool
        return _StagingPool(torch.device("cpu"), block_bytes, max_blocks)

    def test_roundtrip_and_merge(self):
        pool = self._pool()
        a = pool.acquire(3)
        b = pool.acquire(5)          # exact fit of the remainder
        self.assertEqual((a, b), (0, 3))
        pool.release(a, 3)
        pool.release(b, 5)           # neighbours must merge back to one run
        self.assertEqual(pool._runs, [[0, 8]])
        self.assertEqual(pool.acquire(8), 0)  # full capacity usable again

    def test_fragmentation_blocks_then_merge_wakes(self):
        pool = self._pool()
        a, b, c = pool.acquire(2), pool.acquire(2), pool.acquire(2)
        self.assertEqual((a, b, c), (0, 2, 4))
        pool.release(a, 2)           # free: [0,2) and [6,8)
        pool.release(c, 2)
        # 5 free blocks in total but no contiguous run of 5: blocks.
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(pool.acquire, 5)
            time.sleep(0.2)
            self.assertFalse(fut.done(), "fragmented pool must block")
            pool.release(b, 2)       # glues [0,8) back together
            self.assertEqual(fut.result(timeout=5), 0)

    def test_blocking_acquire_wakes_on_release(self):
        pool = self._pool(max_blocks=4)
        held = pool.acquire(3)
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(pool.acquire, 3)
            time.sleep(0.2)
            self.assertFalse(fut.done(), "acquire must block while exhausted")
            pool.release(held, 3)
            self.assertEqual(fut.result(timeout=5), 0)

    def test_oversized_acquire_raises(self):
        pool = self._pool(max_blocks=4)
        with self.assertRaises(ValueError):
            pool.acquire(5)

    def test_views_slice_the_same_run(self):
        pool = self._pool(max_blocks=8, block_bytes=16)
        start = pool.acquire(3)
        cpu = pool.cpu_view(start, 3)
        self.assertEqual(cpu.numel(), 48)
        cpu.zero_()
        self.assertTrue(bool((pool._cpu[0:48] == 0).all()))

    def test_pool_allocates_host_memory_only(self):
        """Zero-VRAM regression guard: even for a CUDA device the pool makes
        exactly one allocation, and it is pinned host memory -- no device-side
        mirror may come back."""
        from unittest.mock import patch
        from kv_cache_manager.py_connector.vllm.data_transfer import _StagingPool
        with patch("torch.empty") as empty:
            _StagingPool(torch.device("cuda"), block_bytes=16, max_blocks=8)
        self.assertEqual(empty.call_count, 1,
                         "pool must own exactly one backing allocation")
        kwargs = empty.call_args.kwargs
        self.assertEqual(kwargs.get("device"), "cpu")
        self.assertTrue(kwargs.get("pin_memory"),
                        "pool backing memory must be pinned for zero-copy")


class TestPoolByteCap(unittest.TestCase):
    """_effective_pool_blocks: per-group sizing under the pinned-RAM ceiling.

    Found by the final-validation e2e smoke: a fixed 1024-block count was
    tuned for full-attention blocks (~0.875 MiB each) but hybrid blocks are
    ~17.3 MiB, so four groups pinned ~68 GiB of host RAM and engine start
    died in the pinned allocator. Pinned here: the count is derived per
    group from the byte cap, and one full task batch always fits.
    """

    def _f(self, configured, need, block_bytes, max_bytes):
        from kv_cache_manager.py_connector.vllm.data_transfer import (
            _effective_pool_blocks)
        return _effective_pool_blocks(configured, need, block_bytes, max_bytes)

    def test_full_attention_blocks_keep_the_configured_count(self):
        # 1024 x 0.875 MiB ~= 896 MiB <= 1 GiB cap: unchanged sizing (the
        # validated origin/main concurrency of 8 full tasks in flight).
        self.assertEqual(
            self._f(1024, 128, 917_504, 2**30), 1024)

    def test_hybrid_blocks_cap_by_bytes(self):
        # 17.3 MiB blocks: 1024 would pin ~17.3 GiB per group; a cap that
        # allows more than one task batch caps to cap // block_bytes.
        block_bytes = 17_301_504
        self.assertEqual(
            self._f(1024, 128, block_bytes, 8 * 2**30),
            8 * 2**30 // block_bytes)

    def test_hybrid_default_cap_floors_at_one_task_batch(self):
        # With the 1 GiB default the byte cap alone would allow only 62
        # blocks (< one 128-block task batch); the batch must still fit
        # contiguously, so the effective size is the batch -- the same
        # configuration the staging-removal campaign validated for hybrid.
        self.assertEqual(
            self._f(1024, 128, 17_301_504, 2**30), 128)

    def test_one_task_batch_always_fits(self):
        # Even when the byte cap is smaller than one task batch, the batch
        # must still fit contiguously (contiguity invariant); the caller
        # warns that the cap was exceeded.
        self.assertEqual(
            self._f(1024, 128, 17_301_504, 128 * 1024), 128)

    def test_byte_cap_can_only_shrink_not_grow(self):
        # A configured count below the byte cap is authoritative.
        self.assertEqual(self._f(256, 128, 917_504, 2**30), 256)
