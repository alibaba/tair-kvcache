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
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

# vllm_stubs must be imported before torch: in the open-source CI (no torch
# installed) it registers the MagicMock stand-in that the bare `import torch`
# below then resolves to.
from kv_cache_manager.py_connector.test import vllm_stubs  # noqa: F401 (stubs)

import torch
from kv_cache_manager.py_connector.vllm.data_transfer import (
    DataTransferManager,
    MultiResult,
    StagingShape,
    group_meta_pinned_bytes,
    group_meta_shape,
    plan_staging_pools,
    staging_shape,
    total_pinned_bytes,
)
from kv_cache_manager.py_connector.vllm.transfer_types import KVLayout
from kv_cache_manager.py_connector.vllm.vllm_common import (
    AttentionGroupMeta,
    RowGranularMeta,
)
from kv_cache_manager.py_connector.common.tp_coordinator import CoordinateMsgSerializer


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
        cb(
            [
                True,
                True,
                False,  # group 0
                True,
                False,
                True,
            ]
        )  # group 1
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
            "req", 0, epoch=7, block_ids=[10, 20, 30], num_blocks=3
        )
        cb(
            [
                True,
                False,
                True,  # group 0
                True,
                True,
                False,
            ]
        )  # group 1
        evt = _sent_event(dtm)
        self.assertEqual(evt.type, "LoadBlockFinishedEvent")
        self.assertEqual(evt.epoch, 7)
        # blocks 1 and 2 each failed in one group -> report their table ids.
        self.assertEqual(evt.failed_block_idxs, [20, 30])

    def test_all_success_reports_empty(self):
        dtm = _make_dtm()
        cb = dtm.create_load_done_callback(
            "req", 0, epoch=0, block_ids=[10, 20], num_blocks=2
        )
        cb([True, True, True, True])
        self.assertEqual(_sent_event(dtm).failed_block_idxs, [])

    def test_report_failures_false_hybrid(self):
        # Hybrid models cannot report invalid block ids to vLLM: the failure
        # must be swallowed (empty failed list) but the finished event still sent.
        dtm = _make_dtm()
        cb = dtm.create_load_done_callback(
            "req", 0, epoch=1, block_ids=[], num_blocks=2, report_failures=False
        )
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
    vllm_common.build_spec_groups). These paths do no GPU work, so they run on CPU.
    """

    @staticmethod
    def _state_group():
        from kv_cache_manager.py_connector.vllm.transfer_types import StateTransferGroup

        return StateTransferGroup(
            group_idx=0,
            spec_name="tp0_g0",
            layer_names=["m0"],
            block_size=528,
            per_block_bytes=1024,
            layer_num=1,
            block_view_tensors=[],
            page_size_bytes=1024,
        )

    @staticmethod
    def _attn_group():
        from kv_cache_manager.py_connector.vllm.transfer_types import (
            AttentionTransferGroup,
        )

        return AttentionTransferGroup(
            group_idx=1,
            spec_name="tp0_g1",
            layer_names=["a0"],
            block_size=528,
            per_block_bytes=1024,
            layer_num=1,
            kv_layout=KVLayout.PACKED_4D,
            # No real device pointers in the pure plumbing tests.
            kvcache_ptr_tensor_gpu=None,  # ty: ignore[invalid-argument-type]
            num_kv_ptrs=1,
            per_token_dim=8,
            kernel_block_size=528,
            block_stride=0,
            dtype=torch.bfloat16,
        )

    def _run(self, method, **kwargs):
        dtm = _make_dtm()
        results = {}
        mr = MultiResult(1, lambda flat: results.setdefault("flat", flat))
        getattr(dtm, method)(mr, 0, self._state_group(), **kwargs)
        return results["flat"]

    def test_save_null_state_blocks_abstain_not_succeed(self):
        # No state and (consistently) no location for it: the group abstains.
        flat = self._run(
            "save_task",
            remote_uris=[None, None],
            block_token_indices=None,
            block_ids=[0, 0],
            ready_event=None,
        )
        self.assertEqual(flat, [None, None])

    def test_save_null_state_with_location_fails(self):
        # The manager allocated a state location but vLLM has no state to put
        # there: publishing it would advertise unwritten bytes.
        flat = self._run(
            "save_task",
            remote_uris=["u0", None],
            block_token_indices=None,
            block_ids=[0, 0],
            ready_event=None,
        )
        self.assertEqual(flat, [False, None])

    def test_save_real_state_without_location_fails(self):
        # A state exists but was not announced: it cannot be published.
        flat = self._run(
            "save_task",
            remote_uris=[None],
            block_token_indices=None,
            block_ids=[7],
            ready_event=None,
        )
        self.assertEqual(flat, [False])

    def test_load_null_state_targets_abstain(self):
        # vLLM does not need a state for these blocks (only the block ending
        # the reused prefix does), whatever the manager published.
        flat = self._run(
            "load_task",
            remote_uris=["u0", "u1", "u2"],
            block_token_indices=None,
            block_ids=[0, 0, 0],
        )
        self.assertEqual(flat, [None, None, None])

    def test_load_real_target_without_location_fails(self):
        # vLLM needs this state but nothing was published for it: the request
        # must not run on an unwritten state.
        flat = self._run(
            "load_task", remote_uris=[None], block_token_indices=None, block_ids=[9]
        )
        self.assertEqual(flat, [False])

    def test_attention_block_without_location_fails(self):
        # Attention KV is never sparse: a missing location is a failure, and
        # the block must be kept out of the staging batch so the remaining
        # buffers stay aligned with the URI list.
        dtm = _make_dtm()
        skipped, failed = dtm._save_dispositions(
            self._attn_group(), remote_uris=["u0", None, "u2"], block_ids=None, n=3
        )
        self.assertEqual((skipped, failed), (set(), {1}))

    def test_save_dispositions_state_group(self):
        dtm = _make_dtm()
        # block0: no state, nothing published -> abstain
        # block1: state + location      -> transfer
        # block2: no state but published -> fail
        # block3: state but unpublished  -> fail
        skipped, failed = dtm._save_dispositions(
            self._state_group(),
            remote_uris=[None, "u1", "u2", None],
            block_ids=[0, 5, 0, 6],
            n=4,
        )
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
        kwargs = dict(
            remote_uris=["u0", "u1"], block_token_indices=None, block_ids=[5, 6]
        )
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
        cb(
            [
                None,
                None,
                True,  # state group: only block 2 had a state
                True,
                False,
                True,
            ]
        )  # attention group
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
            "req", 0, epoch=3, block_ids=[10, 20, 30], num_blocks=3
        )
        cb(
            [
                None,
                None,
                False,  # state group: block 2 needed a state, failed
                True,
                True,
                True,
            ]
        )  # attention group
        self.assertEqual(_sent_event(dtm).failed_block_idxs, [30])

    def test_load_all_groups_abstain_is_a_skip_not_a_failure(self):
        # Every group skipped the block because none of them *needed* it: a
        # windowed group's target outside the window (vLLM allocated no slot
        # there on an external hit) or a mamba interior block. Reporting it as
        # an invalid block would make vLLM truncate num_computed_tokens to
        # where the block first appears -- block 0 throws the whole external
        # hit away, and kv_load_failure_policy="fail" fails the request --
        # even though every byte that was needed arrived (review_r1 B1).
        dtm = _make_dtm()
        cb = dtm.create_load_done_callback(
            "req", 0, epoch=0, block_ids=[11], num_blocks=1
        )
        cb([None])
        self.assertEqual(_sent_event(dtm).failed_block_idxs, [])

    def test_load_one_group_failing_still_fails_the_block(self):
        # The same all-skip shape, but one group *reported*: that is a hole in
        # the request's KV and must be named.
        dtm = _make_dtm()
        cb = dtm.create_load_done_callback(
            "req", 0, epoch=0, block_ids=[11], num_blocks=1
        )
        cb([False, None])
        self.assertEqual(_sent_event(dtm).failed_block_idxs, [11])

    def test_load_one_group_succeeding_and_one_failing_fails_the_block(self):
        dtm = _make_dtm()
        cb = dtm.create_load_done_callback(
            "req", 0, epoch=0, block_ids=[11], num_blocks=1
        )
        cb([True, False])
        self.assertEqual(_sent_event(dtm).failed_block_idxs, [11])


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
        b = pool.acquire(5)  # exact fit of the remainder
        self.assertEqual((a, b), (0, 3))
        pool.release(a, 3)
        pool.release(b, 5)  # neighbours must merge back to one run
        self.assertEqual(pool._runs, [[0, 8]])
        self.assertEqual(pool.acquire(8), 0)  # full capacity usable again

    def test_fragmentation_blocks_then_merge_wakes(self):
        pool = self._pool()
        a, b, c = pool.acquire(2), pool.acquire(2), pool.acquire(2)
        self.assertEqual((a, b, c), (0, 2, 4))
        pool.release(a, 2)  # free: [0,2) and [6,8)
        pool.release(c, 2)
        # 5 free blocks in total but no contiguous run of 5: blocks.
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(pool.acquire, 5)
            time.sleep(0.2)
            self.assertFalse(fut.done(), "fragmented pool must block")
            pool.release(b, 2)  # glues [0,8) back together
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

    @unittest.skipIf(
        "torch" in vllm_stubs.STUBBED,
        "needs a real torch tensor (stubbed in the open-source CI)",
    )
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
        mirror may come back. The fake device keeps the contract testable
        wherever torch itself is a stand-in."""
        from types import SimpleNamespace
        from unittest.mock import patch
        from kv_cache_manager.py_connector.vllm.data_transfer import _StagingPool

        with patch("torch.empty") as empty:
            # duck-typed device stand-in
            _StagingPool(
                SimpleNamespace(type="cuda"),  # ty: ignore[invalid-argument-type]
                per_block_bytes=16,
                max_blocks=8,
            )
        self.assertEqual(
            empty.call_count, 1, "pool must own exactly one backing allocation"
        )
        kwargs = empty.call_args.kwargs
        self.assertEqual(kwargs.get("device"), "cpu")
        self.assertTrue(
            kwargs.get("pin_memory"), "pool backing memory must be pinned for zero-copy"
        )


class TestPoolByteCap(unittest.TestCase):
    """_effective_pool_blocks: slots per staging arena under the pinned ceiling.

    Found by the final-validation e2e smoke: a fixed 1024-block count was
    tuned for full-attention blocks (~0.875 MiB each) but hybrid blocks are
    ~17.3 MiB, so four groups pinned ~68 GiB of host RAM and engine start
    died in the pinned allocator. The count is derived from the byte cap, and
    the cap is now a *hard* budget: it shrinks the pool (and the task batches
    riding on it, see DataTransferManager.save_task_blocks) instead of being
    silently breached by the old `max(need, ...)` floor.
    """

    def _f(self, configured, slot_bytes, max_bytes):
        from kv_cache_manager.py_connector.vllm.data_transfer import (
            _effective_pool_blocks,
        )

        return _effective_pool_blocks(configured, slot_bytes, max_bytes)

    def test_full_attention_blocks_keep_the_configured_count(self):
        # 1024 x 0.875 MiB ~= 896 MiB <= 1 GiB cap: unchanged sizing (the
        # validated origin/main concurrency of 8 full tasks in flight).
        self.assertEqual(self._f(1024, 917_504, 2**30), 1024)

    def test_cap_decides_for_large_blocks(self):
        # 17.3 MiB slots: 1024 would pin ~17.3 GiB, so the cap rules.
        self.assertEqual(self._f(1024, 17_301_504, 2**30), 62)

    def test_configured_count_is_an_upper_bound(self):
        self.assertEqual(self._f(256, 917_504, 2**30), 256)

    def test_slot_must_fit_the_cap(self):
        # A cap that cannot pay for one slot is a startup error, not a floor:
        # the old code pinned 128 such slots (= 2x the cap) and warned.
        with self.assertRaises(ValueError) as cm:
            self._f(1024, 17_301_504, 16 * 2**20)
        self.assertIn("staging_pool_max_bytes_per_group", str(cm.exception))

    def test_degenerate_inputs_rejected(self):
        with self.assertRaises(ValueError):
            self._f(0, 1024, 2**30)
        with self.assertRaises(ValueError):
            self._f(1024, 0, 2**30)


# --------------------------------------------------------------------------- #
# Staging arenas: shape keying, pinned totals, sharing
#
# The geometries below are the ones the research pinned down on real configs
# (05 §2.1, data/pin/sa-c/c3_output.txt) and the Qwen3.5-4B e2e logs. The
# numbers are the acceptance numbers of this PR: the arenas must collapse to
# one shape per model family and the pinned total must land on the budget.
# --------------------------------------------------------------------------- #
MBS = 16  # manager block size of the gemma/gpt-oss/qwen e2e configurations
QWEN35_MBS = 528  # Qwen3.5-4B: manager == vLLM scheduler block size
KV_DTYPE = torch.bfloat16  # only ever compared, never interpreted


def _attn_meta(idx: int, layers, page_bytes: int, block_size: int = MBS):
    """An attention GroupMeta whose per_block_bytes is built the way
    vllm_common.parse_groups builds it: per-token bytes * mbs * layers."""
    from kv_cache_manager.py_connector.vllm.vllm_common import AttentionGroupMeta

    return AttentionGroupMeta(
        group_idx=idx,
        layer_names=[f"a{idx}_{i}" for i in range(layers)],
        block_size=block_size,
        per_block_bytes=(page_bytes // block_size) * block_size * layers,
        page_bytes=page_bytes,
    )


def _state_meta(idx: int, layers, page_size_bytes: int, block_size: int):
    from kv_cache_manager.py_connector.vllm.vllm_common import StateGroupMeta

    return StateGroupMeta(
        group_idx=idx,
        layer_names=[f"s{idx}_{i}" for i in range(layers)],
        block_size=block_size,
        per_block_bytes=page_size_bytes * layers,
        page_size_bytes=page_size_bytes,
    )


def _plan(metas, *, dtype=KV_DTYPE, configured_blocks=1024, cap=2**30):
    return plan_staging_pools(
        [group_meta_shape(m, dtype) for m in metas],
        configured_blocks=configured_blocks,
        max_bytes_per_pool=cap,
    )


class TestStagingShapes(unittest.TestCase):
    """One bucket's staging shape, from the worker's and the planner's view."""

    def test_attention_shape_is_per_token_of_one_layer(self):
        from kv_cache_manager.py_connector.vllm.transfer_types import (
            AttentionTransferGroup,
        )

        group = AttentionTransferGroup(
            group_idx=0,
            spec_name="tp0_g0",
            layer_names=["l0"] * 8,
            block_size=528,
            per_block_bytes=17_301_504,
            layer_num=8,
            kv_layout=KVLayout.MLA_3D,
            kvcache_ptr_tensor_gpu=None,  # ty: ignore[invalid-argument-type]
            num_kv_ptrs=8,
            per_token_dim=1024,
            kernel_block_size=528,
            block_stride=0,
            dtype=KV_DTYPE,
        )
        shape = staging_shape(group, QWEN35_MBS, KV_DTYPE)
        self.assertEqual(shape.per_token_bytes, 4096)
        self.assertEqual(shape.per_block_bytes, 17_301_504)
        self.assertIs(shape.dtype, KV_DTYPE)

    def test_worker_and_planner_views_agree(self):
        """The e2e/kvperf gate recomputes the pinned total from GroupMetas; it
        must see exactly the shape the worker built its arena from."""
        from kv_cache_manager.py_connector.vllm.transfer_types import (
            AttentionTransferGroup,
            StateTransferGroup,
        )

        meta = _state_meta(0, 8, 2_162_688, QWEN35_MBS)
        group = StateTransferGroup(
            group_idx=0,
            spec_name="tp0_g0",
            layer_names=list(meta.layer_names),
            block_size=meta.block_size,
            per_block_bytes=meta.per_block_bytes,
            layer_num=8,
            block_view_tensors=[],
            page_size_bytes=meta.page_size_bytes,
        )
        self.assertEqual(
            staging_shape(group, QWEN35_MBS, KV_DTYPE),
            group_meta_shape(meta, KV_DTYPE),
        )
        attn_meta = _attn_meta(3, 8, 2_162_688, QWEN35_MBS)
        attn_group = AttentionTransferGroup(
            group_idx=3,
            spec_name="tp0_g3",
            layer_names=list(attn_meta.layer_names),
            block_size=attn_meta.block_size,
            per_block_bytes=attn_meta.per_block_bytes,
            layer_num=8,
            kv_layout=KVLayout.PACKED_4D,
            kvcache_ptr_tensor_gpu=None,  # ty: ignore[invalid-argument-type]
            num_kv_ptrs=8,
            per_token_dim=1024,
            kernel_block_size=attn_meta.block_size,
            block_stride=0,
            dtype=KV_DTYPE,
        )
        self.assertEqual(
            staging_shape(attn_group, QWEN35_MBS, KV_DTYPE),
            group_meta_shape(attn_meta, KV_DTYPE),
        )


class TestRowGranularStaging(unittest.TestCase):
    """A compressed bucket stages rows, not tokens (PR6).

    The staging view the gather/scatter kernel writes has `mbs // tokens_per_row`
    rows per manager block and each row is the bucket's own row: getting this
    from the manager block size would over-read the tensor by a factor of the
    ratio (and mis-size every slot's sub-view).
    """

    def _group(self, *, tokens_per_row: int, rows_per_page: int, row_bytes: int):
        from kv_cache_manager.py_connector.vllm.transfer_types import (
            AttentionTransferGroup,
        )

        return AttentionTransferGroup(
            group_idx=0,
            spec_name="tp0_g0",
            layer_names=["l0"],
            block_size=rows_per_page * tokens_per_row,
            per_block_bytes=row_bytes * (256 // tokens_per_row),
            layer_num=1,
            kv_layout=KVLayout.MLA_3D,
            kvcache_ptr_tensor_gpu=None,  # ty: ignore[invalid-argument-type]
            num_kv_ptrs=1,
            per_token_dim=row_bytes,
            kernel_block_size=rows_per_page,
            block_stride=0,
            dtype=KV_DTYPE,
            tokens_per_row=tokens_per_row,
        )

    def test_staging_view_rows_follow_the_row_granularity(self):
        import torch

        from kv_cache_manager.py_connector.vllm.data_transfer import (
            DataTransferManager,
        )

        manager = DataTransferManager.__new__(DataTransferManager)
        manager._manager_block_size = 256
        buffer = torch.zeros(256 * 584, dtype=torch.uint8)
        token_granular = self._group(tokens_per_row=1, rows_per_page=64, row_bytes=584)
        compressed = self._group(tokens_per_row=2, rows_per_page=32, row_bytes=584)
        self.assertEqual(256, manager._attn_rows_per_block(token_granular))
        self.assertEqual(128, manager._attn_rows_per_block(compressed))
        view = manager._attn_staging_view(buffer, compressed, batch=1)
        self.assertEqual(view.shape, (1, 1, 128, 584))

    def test_the_pool_key_normalizes_a_row_granular_bucket(self):
        # The arena key stays per-token (the PR2 contract) and a compressed
        # bucket derives it from its rows: 584 B/row over 2 tokens is 292 B per
        # token, which is what a token-granular bucket of the same row shape
        # would key with.
        compressed = self._group(tokens_per_row=2, rows_per_page=32, row_bytes=584)
        shape = staging_shape(compressed, 256, KV_DTYPE)
        self.assertEqual(292, shape.per_token_bytes)
        self.assertEqual(584 * 128, shape.per_block_bytes)
        meta = AttentionGroupMeta(
            group_idx=0,
            layer_names=["l0"],
            block_size=64,
            per_block_bytes=584 * 128,
            page_bytes=32 * 584,
            row_granular=RowGranularMeta(2, 32, 584),
        )
        self.assertEqual(shape, group_meta_shape(meta, KV_DTYPE))


class TestPinnedTotals(unittest.TestCase):
    """GroupMeta list -> staging arenas -> pinned bytes, per research model.

    Column `shapes` is the number of arenas (the old per-group pools are the
    "before" column of the design's reconciliation table), `pinned` the exact
    allocation the engine logs -- the e2e/kvperf gates compare against the very
    same `total_pinned_bytes`/`plan_staging_pools` implementation.
    """

    MODELS = {
        # model: (metas, shapes, pinned bytes, one-line reason)
        "Qwen2.5-7B full attention": (
            [_attn_meta(0, 28, 32_768)],
            1,
            1_024 * 917_504,
            "single bucket: 1024 x 0.875 MiB, unchanged by this PR",
        ),
        "Qwen3.5-4B hybrid": (
            [
                _state_meta(0, 8, 2_162_688, QWEN35_MBS),
                _state_meta(1, 8, 2_162_688, QWEN35_MBS),
                _state_meta(2, 8, 2_162_688, QWEN35_MBS),
                _attn_meta(3, 8, 2_162_688, QWEN35_MBS),
            ],
            1,
            62 * 17_301_504,
            "3 state + 1 attention buckets of one shape: 62 slots, 0.999 GiB",
        ),
        "gemma-3-4b-it": (
            [_attn_meta(i, 5, 65_536) for i in range(5)]
            + [_attn_meta(5, 4, 65_536), _attn_meta(6, 5, 65_536)],
            1,
            1_024 * 327_680,
            "uneven SWA layer counts [5,5,5,5,5,4]+[5]: one arena",
        ),
        "gemma-3-12b-it": (
            [_attn_meta(i, 8, 131_072) for i in range(6)],
            1,
            1_024 * 1_048_576,
            "6 equal buckets",
        ),
        "gemma-3-27b-it": (
            [_attn_meta(i, 9, 131_072) for i in range(4)]
            + [_attn_meta(4, 8, 131_072), _attn_meta(5, 8, 131_072)]
            + [_attn_meta(6, 10, 131_072)],
            1,
            819 * 1_310_720,
            "three per_block tiers collapse onto the 10-layer full bucket",
        ),
        "gemma-2-9b-it": (
            [_attn_meta(0, 21, 131_072), _attn_meta(1, 21, 131_072)],
            1,
            390 * 2_752_512,
            "1:1 pair, 2.625 MiB blocks",
        ),
        "gpt-oss-20b": (
            [_attn_meta(0, 12, 32_768), _attn_meta(1, 12, 32_768)],
            1,
            1_024 * 393_216,
            "1:1 win128 pair",
        ),
        "gpt-oss-120b": (
            [_attn_meta(0, 18, 32_768), _attn_meta(1, 18, 32_768)],
            1,
            1_024 * 589_824,
            "1:1 win128 pair, larger",
        ),
    }

    def test_reconciliation_table(self):
        for name, (metas, shapes, pinned, why) in self.MODELS.items():
            with self.subTest(model=name):
                plans = _plan(metas)
                self.assertEqual(len(plans), shapes, why)
                self.assertEqual(total_pinned_bytes(plans), pinned, why)
                # The one call an e2e/kvperf gate needs: GroupMeta list in,
                # the logged pinned total out.
                self.assertEqual(
                    group_meta_pinned_bytes(
                        metas,
                        KV_DTYPE,
                        configured_blocks=1024,
                        max_bytes_per_pool=2**30,
                    ),
                    pinned,
                    why,
                )

    def test_budgets_of_the_pr_acceptance(self):
        # 04 §3.3 pinned gates: gemma-3-4b <= 0.4 GiB, Qwen3.5-4B <= 1 GiB.
        self.assertLessEqual(
            total_pinned_bytes(_plan(self.MODELS["gemma-3-4b-it"][0])), 0.4 * 2**30
        )
        self.assertLessEqual(
            total_pinned_bytes(_plan(self.MODELS["Qwen3.5-4B hybrid"][0])), 2**30
        )

    def test_single_group_model_keeps_its_pool_parameters(self):
        """Byte-equivalence argument (design D6): one bucket, a cap that pays
        for the configured count -> same slots, same slot size, same task
        batch as the per-group implementation."""
        plans = _plan(self.MODELS["Qwen2.5-7B full attention"][0])
        (plan,) = plans
        self.assertEqual(plan.blocks, 1024)
        self.assertEqual(plan.slot_bytes, 917_504)
        self.assertEqual(plan.members, (0,))

    def test_uneven_layer_counts_share_one_arena(self):
        """M2 regression pin: keying by per_block would take two arenas here
        (0.3125 vs 0.25 MiB blocks) and three for gemma-3-27b."""
        four_b = self.MODELS["gemma-3-4b-it"][0]
        self.assertEqual(len({m.per_block_bytes for m in four_b}), 2)
        self.assertEqual(len(_plan(four_b)), 1)
        twenty_seven = self.MODELS["gemma-3-27b-it"][0]
        self.assertEqual(len({m.per_block_bytes for m in twenty_seven}), 3)
        self.assertEqual(len(_plan(twenty_seven)), 1)

    def test_dtype_is_part_of_the_key(self):
        # Same per-token geometry, different element type: separate arenas (a
        # typed view is per bucket, but the key must not merge row shapes the
        # deployer thinks of as different).
        plans = plan_staging_pools(
            [
                StagingShape(4096, torch.bfloat16, 327_680),
                StagingShape(4096, torch.uint8, 327_680),
            ],
            configured_blocks=1024,
            max_bytes_per_pool=2**30,
        )
        self.assertEqual(len(plans), 2)
        self.assertEqual([plan.members for plan in plans], [(0,), (1,)])

    def test_the_pool_shrinks_before_the_cap_is_breached(self):
        """Qwen3.5's 1 GiB cap buys 62 of its 16.5 MiB slots; the task batches
        must then be 62 blocks (D3), not the configured 128."""
        (plan,) = _plan(self.MODELS["Qwen3.5-4B hybrid"][0])
        self.assertEqual(plan.blocks, 62)
        self.assertLessEqual(plan.pinned_bytes, 2**30)


class TestSharedArena(unittest.TestCase):
    """Several buckets on one arena: the run allocator must keep their byte
    ranges disjoint, hand a small-block member exactly its dense prefix, and
    still block (never fail) when the arena is exhausted.

    Only the *slot* is shared; each member's batch is a dense prefix of the run
    it borrowed, which is what keeps the kernel/SDK "one contiguous piece of
    memory" assumption true (design D2)."""

    def _pool(self, slot_bytes=64, slots=4):
        from kv_cache_manager.py_connector.vllm.data_transfer import _StagingPool

        return _StagingPool(torch.device("cpu"), slot_bytes, slots)

    def test_batches_of_two_members_never_overlap(self):
        pool = self._pool()
        big = pool.acquire(2)  # a full-slot member
        small = pool.acquire(2)  # a half-slot member, same arena
        self.assertNotEqual(big, small)
        self.assertEqual(pool._runs, [])
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(pool.acquire, 1)
            time.sleep(0.2)
            self.assertFalse(fut.done(), "exhausted arena must block, not fail")
            pool.release(small, 2)
            third = fut.result(timeout=5)
            self.assertEqual(third, small)  # the freed run, lowest start
            pool.release(third, 1)
        pool.release(big, 2)
        self.assertEqual(pool._runs, [[0, 4]])  # every slot back, merged

    @unittest.skipIf(
        "torch" in vllm_stubs.STUBBED,
        "needs a real torch tensor (stubbed in the open-source CI)",
    )
    def test_small_block_member_views_only_its_dense_prefix(self):
        pool = self._pool(slot_bytes=64, slots=4)
        pool._cpu.zero_()  # torch.empty hands out garbage; pin *writes* only
        start = pool.acquire(3)
        small = pool.cpu_view(start, 3, 32)
        self.assertEqual(small.numel(), 96)  # 3 x 32, not 3 x 64
        self.assertEqual(small.data_ptr(), pool._cpu.data_ptr() + start * 64)
        small.fill_(7)
        self.assertEqual(int(pool._cpu[start * 64]), 7)
        # Only the borrowed run is written: the slot capacity the member did
        # not claim and the arena's other slots stay untouched, so a
        # neighbour's bytes can never be scribbled on.
        base = start * 64
        unused = pool._cpu[base + 3 * 32 : base + 3 * 64]
        self.assertEqual(int(unused.abs().sum()), 0)
        self.assertEqual(int(pool._cpu[3 * 64 :].abs().sum()), 0)

    def test_oversized_block_size_is_refused(self):
        pool = self._pool(slot_bytes=64)
        with self.assertRaises(ValueError):
            pool.cpu_view(0, 1, 128)


class TestDataTransferManagerPools(unittest.TestCase):
    """Init wiring: one arena per shape, task batches that fit it, and the
    machine-parseable pinned total the e2e gate parses (04 §3.3).

    Fixtures: one shape with three buckets (two attention with different layer
    counts + one state bucket, all 16 B/token/layer) and one bucket of another
    shape (8 B/token/layer)."""

    MBS = 64

    def _dtm(self, groups, **overrides):
        from kv_cache_manager.py_connector.vllm.config import (
            TairKvCacheConnectorExtraConfig,
        )
        from kv_cache_manager.py_connector.vllm.transfer_types import KVCacheInfo
        import kv_cache_manager.py_connector.vllm.data_transfer as dt

        extra = TairKvCacheConnectorExtraConfig(
            manager_uri="http://127.0.0.1:1",
            coordinator_base_port=1,
            instance_group="test",
            instance_id="test",
            **overrides,
        )
        info = KVCacheInfo(
            tp_rank=0,
            world_size=1,
            groups=groups,
            device=torch.device("cpu"),
            dtype=KV_DTYPE,
        )
        with mock.patch.object(dt, "_get_device_module", return_value=MagicMock()):
            return DataTransferManager(info, self.MBS, MagicMock(), MagicMock(), extra)

    def _groups(self):
        from kv_cache_manager.py_connector.vllm.transfer_types import (
            AttentionTransferGroup,
            StateTransferGroup,
        )

        def attn(idx, name, layers, per_block):
            return AttentionTransferGroup(
                group_idx=idx,
                spec_name=name,
                layer_names=[f"{name}_{i}" for i in range(layers)],
                block_size=self.MBS,
                per_block_bytes=per_block,
                layer_num=layers,
                kv_layout=KVLayout.PACKED_4D,
                kvcache_ptr_tensor_gpu=None,  # ty: ignore[invalid-argument-type]
                num_kv_ptrs=layers,
                per_token_dim=8,
                kernel_block_size=self.MBS,
                block_stride=0,
                dtype=KV_DTYPE,
            )

        # 16 B/token/layer (8 dims x 2 B) makes the three buckets of the first
        # shape 4096/2048/1024-byte blocks; the fourth bucket is 8 B/token.
        return [
            attn(0, "tp0_g0", 4, 4096),
            attn(1, "tp0_g1", 2, 2048),
            StateTransferGroup(
                group_idx=2,
                spec_name="tp0_g2",
                layer_names=["s0"],
                block_size=self.MBS,
                per_block_bytes=1024,
                layer_num=1,
                block_view_tensors=[],
                page_size_bytes=1024,
            ),
            attn(3, "tp0_g3", 1, 512),
        ]

    def test_one_arena_per_shape(self):
        groups = self._groups()
        dtm = self._dtm(
            groups,
            staging_pool_blocks=16,
            staging_pool_max_bytes_per_group=16_384,  # 4 x 4096 B slots
        )
        shared = dtm._pools["tp0_g0"]
        # Same per-token geometry, different layer counts (and a state bucket):
        # one arena, sized by the largest member block.
        self.assertIs(dtm._pools["tp0_g1"], shared)
        self.assertIs(dtm._pools["tp0_g2"], shared)
        self.assertIsNot(dtm._pools["tp0_g3"], shared)
        self.assertEqual(shared.block_bytes, 4096)
        self.assertEqual(shared.max_blocks, 4)
        # The other shape is capped by its own slot size: 16 KiB / 512 B.
        self.assertEqual(dtm._pools["tp0_g3"].max_blocks, 16)

    def test_the_reserve_leaves_room_for_a_capture_batch(self):
        # The review's shape: a 62-slot arena with the default 128-block
        # batches used to reserve all 62 slots, so every eager capture timed
        # out and windowed caching was silently disabled. The arena is split in
        # half instead: a step that completes 16 blocks (a chunked prefill)
        # asks for one contiguous run of 16 and must get it, while a whole task
        # batch still starts behind it.
        groups = self._groups()
        dtm = self._dtm(
            groups,
            block_per_save_task=128,
            block_per_load_task=128,
            staging_pool_blocks=62,
        )
        pool = dtm._pools["tp0_g0"]
        self.assertEqual(pool.max_blocks, 62)
        self.assertEqual(pool.reserved_blocks, 31)
        self.assertEqual(dtm.save_task_blocks(groups[0]), 31)
        self.assertEqual(dtm.load_task_blocks(groups[0]), 31)
        del groups
        # A realistic capture batch of 16 blocks fits in one run ...
        capture = pool.acquire(16, keep_reserved=True)
        self.assertIsNotNone(capture)
        # ... and a whole task batch still starts while it holds it.
        task = pool.acquire(31)
        self.assertIsNotNone(task)
        pool.release(capture, 16)
        pool.release(task, 31)
        self.assertEqual(pool.free_blocks, 62)

    def test_task_batch_shrinks_to_the_pool(self):
        groups = self._groups()
        dtm = self._dtm(
            groups,
            block_per_save_task=128,
            block_per_load_task=128,
            staging_pool_blocks=8,
            staging_pool_max_bytes_per_group=3 * 4096,  # 3 slots, not 8
        )
        # The cap buys 3 slots; a task stages its whole batch as one contiguous
        # run, so both directions shrink to that pool's transfer reserve (half
        # the arena: a batch that only fit the shared part could not start
        # while captures hold it).
        self.assertEqual(dtm.save_task_blocks(groups[0]), 1)
        self.assertEqual(dtm.load_task_blocks(groups[0]), 1)
        self.assertEqual(dtm._pools["tp0_g0"].reserved_blocks, 1)
        self.assertEqual(dtm.save_task_blocks(groups[3]), 4)  # other shape

    def test_the_transfer_reserve_is_the_largest_task_batch(self):
        from kv_cache_manager.py_connector.common.logger import logger

        # The eager capture may only borrow above it, so a transfer task can
        # always start and thereby release the captures its own session holds
        # (review_r1 N1): one task's maximum batch per arena, clamped to what
        # actually fits the (possibly capped) pool.
        groups = self._groups()
        dtm = self._dtm(
            groups,
            block_per_save_task=3,
            block_per_load_task=2,
            staging_pool_blocks=16,
            staging_pool_max_bytes_per_group=16_384,  # 4 slots
        )
        shared = dtm._pools["tp0_g0"]
        self.assertEqual(shared.max_blocks, 4)
        # Half the arena goes to captures, and the task batch is capped to the
        # reserve so it can always start.
        self.assertEqual(shared.reserved_blocks, 2)
        self.assertEqual(dtm.save_task_blocks(groups[0]), 2)
        # Never the whole arena: the shared part must keep at least one slot, or
        # a capture could not run at all (it would fail closed, but silently).
        tiny = self._dtm(
            groups,
            block_per_save_task=128,
            block_per_load_task=128,
            staging_pool_blocks=16,
            staging_pool_max_bytes_per_group=4096,  # a single slot
        )
        pool = tiny._pools["tp0_g0"]
        self.assertEqual(pool.max_blocks, 1)
        # A single slot cannot serve a capture and a task at once: it stays with
        # the tasks (captures are off, and the startup log says so).
        self.assertEqual(pool.reserved_blocks, 1)
        self.assertEqual(pool.acquire(1, timeout=0.01, keep_reserved=True), None)
        self.assertEqual(1, tiny.save_task_blocks(groups[0]))
        # The degraded configuration says so at startup instead of leaving it
        # to the per-capture withholding warnings.
        with self.assertLogs(logger, level="INFO") as logs:
            self._dtm(
                groups,
                block_per_save_task=128,
                block_per_load_task=128,
                staging_pool_blocks=16,
                staging_pool_max_bytes_per_group=4096,
            )
        self.assertTrue(
            any("captures are disabled" in line for line in logs.output), logs.output
        )

    def test_configured_batch_below_the_pool_is_untouched(self):
        groups = self._groups()
        dtm = self._dtm(
            groups,
            block_per_save_task=2,
            block_per_load_task=1,
            staging_pool_blocks=8,
            staging_pool_max_bytes_per_group=8 * 4096,  # a pool to spare
        )
        self.assertEqual(dtm.save_task_blocks(groups[0]), 2)
        self.assertEqual(dtm.load_task_blocks(groups[0]), 1)

    def test_total_pinned_line_is_machine_parseable(self):
        import re
        from kv_cache_manager.py_connector.common.logger import logger

        groups = self._groups()
        configured = dict(
            block_per_save_task=128,
            block_per_load_task=128,
            staging_pool_blocks=16,
            staging_pool_max_bytes_per_group=16_384,
        )
        with self.assertLogs(logger, level="INFO") as logs:
            dtm = self._dtm(groups, **configured)
        (total_line,) = [line for line in logs.output if "total pinned" in line]
        match = re.search(
            r"total pinned = ([\d.]+) GiB across (\d+) shapes? / (\d+) groups?"
            r" \((\d+) bytes\)",
            total_line,
        )
        self.assertIsNotNone(match, total_line)
        assert match is not None
        self.assertEqual(int(match.group(2)), 2)
        self.assertEqual(int(match.group(3)), len(groups))
        # The logged total is the one the e2e/kvperf gates recompute from the
        # same shapes with the same implementation.
        expected = total_pinned_bytes(
            plan_staging_pools(
                [staging_shape(g, self.MBS, KV_DTYPE) for g in groups],
                configured_blocks=configured["staging_pool_blocks"],
                max_bytes_per_pool=configured["staging_pool_max_bytes_per_group"],
            )
        )
        self.assertEqual(int(match.group(4)), expected)
        self.assertEqual(
            expected,
            dtm._pools["tp0_g0"].max_blocks * dtm._pools["tp0_g0"].block_bytes
            + dtm._pools["tp0_g3"].max_blocks * dtm._pools["tp0_g3"].block_bytes,
        )


class _FakeStagingBuffer:
    """Records what the staging path does to the pinned host buffer.

    ``view(dtype)`` is the element reinterpretation, ``view(n, ptrs, slots,
    dim)`` the block shape; both are what ``torch.Tensor.view`` sees in
    production, so recording the arguments is enough to pin the contract
    without torch (the CI has none)."""

    def __init__(self) -> None:
        self.dtypes = []
        self.shapes = []

    def view(self, *args):
        if len(args) == 1:
            self.dtypes.append(args[0])
            return self
        self.shapes.append(args)
        return "staging-view"

    def data_ptr(self) -> int:
        return 0


class _FakePool:
    """A staging pool whose one slot is a recording buffer."""

    def __init__(self, block_bytes: int) -> None:
        self.block_bytes = block_bytes
        self.buffer = _FakeStagingBuffer()
        self.released = None

    def acquire(self, n: int) -> int:
        return 0

    def release(self, start: int, n: int) -> None:
        self.released = (start, n)

    def cpu_view(
        self, start: int, n: int, block_bytes: int | None = None
    ) -> _FakeStagingBuffer:
        return self.buffer


def _attn_group(
    dtype=torch.bfloat16,
    spec_name="tp0_g0",
    per_token_dim=656,
    per_block_bytes=41984,
    block_size=64,
    num_kv_ptrs=1,
):
    """One attention bucket; the V3.2 fp8 main geometry by default."""
    from kv_cache_manager.py_connector.vllm.transfer_types import (
        AttentionTransferGroup,
    )

    return AttentionTransferGroup(
        group_idx=0,
        spec_name=spec_name,
        layer_names=["l0"],
        block_size=block_size,
        per_block_bytes=per_block_bytes,
        layer_num=1,
        kv_layout=KVLayout.MLA_3D,
        # No real device pointers in these plumbing tests.
        kvcache_ptr_tensor_gpu=None,  # ty: ignore[invalid-argument-type]
        num_kv_ptrs=num_kv_ptrs,
        per_token_dim=per_token_dim,
        kernel_block_size=64,
        block_stride=0,
        dtype=dtype,
    )


class TestAttentionStagingView(unittest.TestCase):
    """E2: one staging view per bucket, shaped by the bucket's own geometry.

    Mixed element sizes are legal in one instance (V3.2 + auto KV: bf16 main
    next to a uint8 indexer), so the view must take the element type from the
    group -- a global dtype built a wrong-sized view for the other bucket."""

    MBS = 64

    def _dtm(self) -> Any:
        dtm: Any = DataTransferManager.__new__(DataTransferManager)
        dtm._manager_block_size = self.MBS
        return dtm

    def test_view_shape_and_element_type_per_group(self):
        dtm = self._dtm()
        for dtype, itemsize, per_token_dim, per_block_bytes in (
            (torch.bfloat16, 2, 576, 73728),
            (torch.uint8, 1, 656, 41984),
        ):
            with self.subTest(dtype=dtype):
                group = _attn_group(
                    dtype=dtype,
                    per_token_dim=per_token_dim,
                    per_block_bytes=per_block_bytes,
                    block_size=16,
                )
                buffer: Any = _FakeStagingBuffer()
                view = dtm._attn_staging_view(buffer, group, batch=3)
                self.assertEqual(view, "staging-view")
                self.assertEqual(buffer.dtypes, [dtype])
                self.assertEqual(buffer.shapes, [(3, 1, self.MBS, per_token_dim)])
                # The staged batch consumes exactly its per-block bytes.
                self.assertEqual(
                    3 * per_token_dim * self.MBS * itemsize, 3 * per_block_bytes
                )

    def test_split_kv_rows_keep_both_pointers(self):
        # A split K/V bucket stages two rows per manager block; the view must
        # carry them instead of halving the batch.
        group = _attn_group(num_kv_ptrs=2, per_token_dim=512, per_block_bytes=32768)
        buffer: Any = _FakeStagingBuffer()
        self._dtm()._attn_staging_view(buffer, group, batch=2)
        self.assertEqual(buffer.shapes, [(2, 2, self.MBS, 512)])


class TestAttentionTaskWiring(unittest.TestCase):
    """E4: the task hands the kernel the bucket's geometry and element type."""

    MBS = 64

    def _dtm(self, group) -> Any:
        dtm: Any = DataTransferManager.__new__(DataTransferManager)
        dtm._manager_block_size = self.MBS
        dtm._save_stream = MagicMock()
        dtm._load_stream = MagicMock()
        dtm._device_mod = MagicMock()
        dtm._transfer_client = MagicMock()
        dtm._transfer_client.SaveKvCaches.return_value = [0]  # ER_OK
        dtm._transfer_client.LoadKvCaches.return_value = 0  # ER_OK
        dtm._coordinator_client = MagicMock()
        dtm._pools = {group.spec_name: _FakePool(group.per_block_bytes)}
        return dtm

    def _capture(self, method: str, group, n: int = 2):
        import kv_cache_manager.py_connector.vllm.data_transfer as dt

        dtm = self._dtm(group)
        buffer = dtm._pools[group.spec_name].buffer
        results = {}
        mr = MultiResult(1, lambda flat: results.setdefault("flat", flat))
        with mock.patch.object(dt, "batch_gather_scatter_helper") as helper:
            if method == "save_task":
                dtm.save_task(
                    mr,
                    0,
                    group,
                    remote_uris=[f"u{i}" for i in range(n)],
                    block_token_indices=[[i] for i in range(n)],
                    block_ids=None,
                    ready_event=MagicMock(),
                )
                kernel = helper.batch_gather_kv_caches
            else:
                dtm.load_task(
                    mr,
                    0,
                    group,
                    remote_uris=[f"u{i}" for i in range(n)],
                    block_token_indices=[[i] for i in range(n)],
                    block_ids=None,
                )
                kernel = helper.batch_scatter_kv_caches
        return kernel.call_args, buffer, results["flat"]

    def test_save_uses_manager_block_slots(self):
        # The kernel's NUM_TOKENS_PER_BLOCK is one manager block worth of
        # tokens: c == 1, so the slots are tokens.
        group = _attn_group()
        (args, kwargs), buffer, flat = self._capture("save_task", group, n=2)
        self.assertEqual(args[4], self.MBS)
        self.assertEqual(args[5], group.per_token_dim)
        self.assertEqual(kwargs["block_stride"], group.block_stride)
        self.assertEqual(kwargs["local_block_size"], group.kernel_block_size)
        self.assertEqual(buffer.shapes, [(2, 1, self.MBS, group.per_token_dim)])
        self.assertEqual(flat, [True, True])

    def test_load_uses_manager_block_slots(self):
        group = _attn_group()
        (args, kwargs), buffer, flat = self._capture("load_task", group, n=2)
        self.assertEqual(args[4], self.MBS)
        self.assertEqual(args[5], group.per_token_dim)
        self.assertEqual(kwargs["block_stride"], group.block_stride)
        self.assertEqual(kwargs["local_block_size"], group.kernel_block_size)
        self.assertEqual(buffer.shapes, [(2, 1, self.MBS, group.per_token_dim)])
        self.assertEqual(flat, [True, True])

    def test_mixed_element_sizes_use_their_own_dtype(self):
        # The regression this whole path exists for: one instance, two
        # buckets, two element sizes -- each stages with its own dtype.
        bf16 = _attn_group(
            dtype=torch.bfloat16,
            per_token_dim=576,
            per_block_bytes=73728,
            block_size=16,
        )
        self.assertEqual(
            self._capture("save_task", bf16, n=1)[1].dtypes, [torch.bfloat16]
        )
        uint8 = _attn_group(spec_name="tp0_g0_b1", dtype=torch.uint8)
        self.assertEqual(
            self._capture("save_task", uint8, n=1)[1].dtypes, [torch.uint8]
        )
        self.assertEqual(
            self._capture("load_task", uint8, n=1)[1].dtypes, [torch.uint8]
        )


if __name__ == "__main__":
    unittest.main()
