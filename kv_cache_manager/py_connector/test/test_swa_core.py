"""Unit tests for the SWA data plane (PR3).

Three mechanisms are pinned here, all of them keyed on
``AttentionGroupMeta.retention``:

* **eager capture** -- the scheduler asks for the newly completed blocks of
  every windowed group in the step that produced them, and the worker copies
  them into the staging arena before that step ends (vLLM frees them in the
  next schedule);
* **null disposition** -- a block whose windowed slots are already null has no
  bytes to publish (abstain: never the shared zero page), a block whose capture
  was withheld is *withdrawn* (failed: never a location without data), and a
  load never writes into vLLM's null block (rows that would go there are
  aliased onto a real row of the same block);
* **the windowed hit clamp** -- an external match may only end where the window
  region still carries every windowed group's spec, and only the largest such
  end is returned.

The transfers themselves run on CPU tensors / recording doubles: this file
tests the decision logic, the mock harness proves the bytes.
"""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest import mock
from unittest.mock import MagicMock

# vllm_stubs must be imported before torch: in the open-source CI (no torch
# installed) it registers the MagicMock stand-in that the bare `import torch`
# below then resolves to.
from kv_cache_manager.py_connector.test import vllm_stubs  # noqa: F401 (stubs)

import torch
from kv_cache_manager.py_connector.test.vllm_stubs import (
    make_connector,
    make_connector_scheduler,
)
from kv_cache_manager.py_connector.vllm.data_transfer import (
    DataTransferManager,
    MultiResult,
    _StagingPool,
)
from kv_cache_manager.py_connector.vllm.metadata import (
    CaptureRequest,
    LoadRequest,
    SaveRequest,
    TairKvCacheConnectorMetadata,
)
from kv_cache_manager.py_connector.vllm.transfer_types import (
    AttentionTransferGroup,
    KVCacheInfo,
    KVLayout,
    NullBlockRow,
)
from kv_cache_manager.py_connector.vllm.connector_scheduler import RequestLedger
from kv_cache_manager.py_connector.vllm.vllm_common import (
    RETENTION_CHUNKED_LOCAL,
    RETENTION_RSWA_GAP,
    RETENTION_SLIDING,
    AttentionGroupMeta,
    WindowRetention,
    build_spec_groups,
    parse_groups,
    spec_name,
)

MBS = 16
PER_TOKEN_DIM = 8
KV_DTYPE = torch.bfloat16


# --------------------------------------------------------------------------- #
# Fakes / helpers
# --------------------------------------------------------------------------- #
@dataclass
class FakeRequest:
    request_id: str
    prompt_token_ids: list
    output_token_ids: list = field(default_factory=list)
    # vLLM's computed-token count; the save watermark reads it as the pre-step
    # value and the tests advance it where they simulate a step.
    num_computed_tokens: int = 0

    @property
    def num_tokens(self):
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def all_token_ids(self):
        return self.prompt_token_ids + self.output_token_ids


def windowed_group(
    spec_name: str = "tp0_g0",
    *,
    group_idx: int = 0,
    block_size: int = MBS,
    kernel_block_size: int = 0,
    per_token_dim: int = PER_TOKEN_DIM,
    retention: Optional[WindowRetention] = None,
) -> AttentionTransferGroup:
    return AttentionTransferGroup(
        group_idx=group_idx,
        spec_name=spec_name,
        layer_names=[f"{spec_name}_l0"],
        block_size=block_size,
        per_block_bytes=block_size * per_token_dim * 2,
        layer_num=1,
        kv_layout=KVLayout.PACKED_4D,
        # No real device pointers in these plumbing tests.
        kvcache_ptr_tensor_gpu=None,  # ty: ignore[invalid-argument-type]
        num_kv_ptrs=1,
        per_token_dim=per_token_dim,
        kernel_block_size=kernel_block_size or block_size,
        block_stride=0,
        dtype=KV_DTYPE,
        retention=retention or WindowRetention(RETENTION_SLIDING, window_tokens=1024),
    )


class _RealClientStub:
    """The pybind client shapes the linker needs, with real objects.

    The stub ring's ``kvcm_py_client`` is a MagicMock whose ``Iov()`` returns
    one shared object, which makes buffer/pointer assertions vacuous.
    """

    class ClientErrorCode:
        ER_OK = 0

    class MemoryType:
        CPU = "cpu"

    class Iov:
        def __init__(self) -> None:
            self.type = None
            self.base = 0
            self.size = 0
            self.ignore = False

    class BlockBuffer:
        def __init__(self) -> None:
            self.iovs: List[Any] = []


class _FakePool:
    """A staging arena stand-in that records the slots it hands out."""

    def __init__(self, block_bytes: int = 1024, max_blocks: int = 8) -> None:
        self.block_bytes = block_bytes
        self.max_blocks = max_blocks
        self.released: List[Any] = []
        self._next = 0

    def acquire(self, n: int, timeout: Optional[float] = None) -> Optional[int]:
        if self._next + n > self.max_blocks:
            return None
        start = self._next
        self._next += n
        return start

    def release(self, start: int, n: int) -> None:
        self.released.append((start, n))

    def slot_base_ptr(self, slot: int) -> int:
        return slot * self.block_bytes


class _RecordingDTM:
    """Stand-in for DataTransferManager's capture surface.

    ``capture_blocks`` never touches a device: the point under test is which
    blocks are captured, which are withheld, and what the plan annotations say.
    """

    def __init__(self, pool: _FakePool, blocks_per_capture: int = 1) -> None:
        self.pool = pool
        self.blocks_per_capture = blocks_per_capture
        self.captures: List[Any] = []
        self.exhausted = False

    def capture_blocks(self, group, block_slots, timeout):
        self.captures.append((group.spec_name, [list(s) for s in block_slots], timeout))
        if self.exhausted:
            return None
        return self.pool.acquire(len(block_slots) * self.blocks_per_capture)

    def pool_of(self, group):
        return self.pool


def worker_with(groups: List[AttentionTransferGroup], dtm: Any = None) -> Any:
    """A bare ConnectorWorker whose transfer groups are the given ones.

    The group metas carry the same retention as the transfer groups, exactly as
    ``_build_attention_group`` copies it from the meta in production: the shared
    coverage predicate reads the metas.
    """
    conn = make_connector(manager_block_size=MBS)
    conn._group_metas = [
        AttentionGroupMeta(
            group_idx=group.group_idx,
            layer_names=list(group.layer_names),
            block_size=group.block_size,
            per_block_bytes=group.per_block_bytes,
            page_bytes=group.block_size * group.per_token_dim * 2,
            retention=group.retention,
        )
        for group in groups
    ]
    conn._num_groups = len(groups)
    conn._kvcache_info = KVCacheInfo(
        tp_rank=0, world_size=1, groups=list(groups), device="cpu", dtype=KV_DTYPE
    )
    conn._data_transfer = dtm
    conn._staged = {}
    conn._in_flight = {}
    return conn


def captured_worker(
    blocks: List[int],
    table: List[int],
    *,
    group: Optional[AttentionTransferGroup] = None,
    pool: Optional[_FakePool] = None,
    exhausted: bool = False,
) -> Any:
    """A worker that has just run one capture request."""
    group = group or windowed_group()
    pool = pool or _FakePool()
    dtm = _RecordingDTM(pool)
    dtm.exhausted = exhausted
    conn = worker_with([group], dtm)
    conn._capture_request(
        CaptureRequest(
            req_id="r0",
            manager_block_idxes=list(blocks),
            all_block_ids=[list(table)],
        )
    )
    return conn


def _k(window: int, extra: int, mbs: int, eagle: bool) -> int:
    """The blocks the clamp must see claimed, re-derived from the design."""
    return -(-(window - 1 + extra) // mbs) + (1 if eagle else 0)


def locations(names_per_block: List[List[str]], tp_size: int = 1) -> List[dict]:
    return [
        {"location_specs": [{"name": n, "uri": f"u_{i}_{n}"} for n in names]}
        for i, names in enumerate(names_per_block)
    ]


# --------------------------------------------------------------------------- #
# Scheduler: capture requests and the windowed hit clamp
# --------------------------------------------------------------------------- #
class TestCaptureRequests(unittest.TestCase):
    """The scheduler asks for a capture only where a window needs it."""

    @staticmethod
    def _empty_output(num_scheduled=None):
        return SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[], resumed_req_ids=set(), new_block_ids=[]
            ),
            num_scheduled_tokens=dict(num_scheduled or {}),
        )

    def _dispatch(self, conn, prompt_len, blocks):
        req = FakeRequest("r0", list(range(prompt_len)))
        conn.update_state_after_alloc(
            req, SimpleNamespace(get_block_ids=lambda: [list(blocks)]), 0
        )
        # A fresh prefill schedules exactly the tokens it holds: the announce
        # and capture watermark is the computed KV, not the allocated span.
        # vLLM advances the request after the step (_update_after_schedule).
        meta = conn.build_connector_meta(
            self._empty_output(num_scheduled={"r0": prompt_len})
        )
        req.num_computed_tokens += prompt_len
        return meta

    def test_full_attention_instance_emits_none(self):
        # The red line: a model without a windowed group must not grow a
        # capture instruction (its blocks never leave HBM while the request
        # lives, so the late gather stays correct and unchanged).
        conn = make_connector_scheduler(manager_block_size=MBS, locations=[])
        meta = self._dispatch(conn, 4 * MBS, [1, 2, 3, 4])
        self.assertEqual([], meta.to_capture_requests)
        self.assertTrue(conn._http_executor.submit.called)  # the save still ran

    def test_windowed_instance_emits_the_new_range_with_the_table(self):
        conn = make_connector_scheduler(manager_block_size=MBS, locations=[], windowed=1)
        meta = self._dispatch(conn, 4 * MBS, [1, 2, 3, 4])
        self.assertEqual(1, len(meta.to_capture_requests))
        capture = meta.to_capture_requests[0]
        self.assertEqual("r0", capture.req_id)
        self.assertEqual([0, 1, 2, 3], capture.manager_block_idxes)
        self.assertEqual([[1, 2, 3, 4]], capture.all_block_ids)

    def test_no_capture_without_new_blocks(self):
        conn = make_connector_scheduler(manager_block_size=MBS, locations=[], windowed=1)
        self._dispatch(conn, 4 * MBS, [1, 2, 3, 4])
        # Same request, same computed prefix: nothing new to capture.
        meta = conn.build_connector_meta(self._empty_output())
        self.assertEqual([], meta.to_capture_requests)

    def test_capture_range_starts_where_the_last_one_ended(self):
        conn = make_connector_scheduler(manager_block_size=MBS, locations=[], windowed=1)
        self._dispatch(conn, 2 * MBS, [1, 2])
        conn._tracked["r0"].vllm_request.prompt_token_ids = list(range(4 * MBS))
        conn._tracked["r0"].block_ids_per_group = [[1, 2, 3, 4]]
        meta = conn.build_connector_meta(
            self._empty_output(num_scheduled={"r0": 2 * MBS})
        )
        conn._tracked["r0"].vllm_request.num_computed_tokens += 2 * MBS
        self.assertEqual([2, 3], meta.to_capture_requests[0].manager_block_idxes)

    def test_chunk_ending_inside_a_block_waits_for_the_rest_of_it(self):
        """A prefill chunk can end inside a manager block, and the allocation
        still covers the whole block. The allocated span is therefore not a
        safe announce watermark: a capture driven by it copies rows the
        forward has not written yet and publishes zero pages as saved KV
        (regression: the eager capture read exactly the tail rows of the
        boundary block)."""
        conn = make_connector_scheduler(manager_block_size=MBS, locations=[], windowed=1)
        # Chunked prefill: a 3-block prompt whose first chunk covers 2 blocks
        # + 3 tokens. Block 2 is allocated (allocation covers whole blocks)
        # but only 3 of its rows are computed. The old formula -- the
        # allocated span capped only by all_token_ids -- announced [0, 1, 2]
        # here and let the eager capture publish the unwritten rows as zero
        # pages; the computed-token watermark announces [0, 1].
        req = FakeRequest("r0", list(range(3 * MBS)))
        conn.update_state_after_alloc(
            req, SimpleNamespace(get_block_ids=lambda: [[1, 2, 3]]), 0
        )
        meta = conn.build_connector_meta(
            self._empty_output(num_scheduled={"r0": 2 * MBS + 3})
        )
        conn._tracked["r0"].vllm_request.num_computed_tokens += 2 * MBS + 3
        self.assertEqual([0, 1], meta.to_capture_requests[0].manager_block_idxes)

        # The rest of the block lands in the next chunk: now it is captured.
        meta = conn.build_connector_meta(
            self._empty_output(num_scheduled={"r0": MBS - 3})
        )
        self.assertEqual([2], meta.to_capture_requests[0].manager_block_idxes)


class TestWindowedSpecGroups(unittest.TestCase):
    """The announcement says which specs a block really carries.

    A windowed block whose window is already gone must not be advertised as
    having windowed KV: the load side only trusts the manager's per-block spec
    coverage, so the announcement (not the write) is what a later hit believes.
    """

    def test_windowed_instance_declares_the_no_window_twin(self):
        conn = make_connector_scheduler(
            manager_block_size=MBS, num_groups=2, windowed=1
        )
        groups = {g["name"]: g["spec_names"] for g in build_spec_groups(conn._group_metas, 1)}
        self.assertEqual(sorted(groups), ["full", "full_no_window"])
        self.assertEqual(groups["full"], ["tp0_g0", "tp0_g1"])
        self.assertEqual(groups["full_no_window"], ["tp0_g1"])

    def test_hybrid_and_windowed_declare_all_four(self):
        conn = make_connector_scheduler(
            manager_block_size=MBS, num_groups=2, windowed=1, num_state_groups=1
        )
        groups = {g["name"]: g["spec_names"] for g in build_spec_groups(conn._group_metas, 1)}
        self.assertEqual(
            sorted(groups), ["attn", "attn_no_window", "full", "full_no_window"]
        )
        self.assertEqual(groups["attn"], ["tp0_g0", "tp0_g1"])
        self.assertEqual(groups["attn_no_window"], ["tp0_g1"])
        self.assertEqual(groups["full_no_window"], ["tp0_g1", "tp0_g2"])

    def test_a_packed_group_with_two_windows_is_refused(self):
        # A packed UniformType group whose members carry different windows
        # (two RSWASpecs, or two SlidingWindow specs that differ only in
        # extra_retained_tokens) cannot be served: vLLM folds the group to one
        # representative sub spec for the scheduler, so the connector could
        # announce and clamp only one of the two buckets -- the other would
        # never be published and its loads would fail. The bucket keys still
        # exist (see TestCaptureDispositions) for the buckets that *are*
        # servable, but this shape is refused at registration.
        from vllm.v1.kv_cache_interface import SlidingWindowSpec, UniformTypeKVCacheSpecs

        def spec(extra):
            return SlidingWindowSpec(
                block_size=MBS,
                sliding_window=1024,
                page_size_bytes=2048,
                num_kv_heads=2,
                head_size=64,
                dtype=torch.bfloat16,
                extra_retained_tokens=extra,
            )

        config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["l0", "l1"],
                    kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(
                        {"l0": spec(0), "l1": spec(8)}
                    ),
                    is_eagle_group=False,
                )
            ]
        )
        with self.assertRaises(NotImplementedError) as ctx:
            parse_groups(config, MBS, calculate_kv_scales=False)
        self.assertIn("folds", str(ctx.exception))
        self.assertIn("windowed", str(ctx.exception))

    def test_pure_windowed_instance_declares_only_the_full_group(self):
        # Nothing to exclude into: every spec is windowed, so a block without
        # window data carries nothing at all and no group can describe it (it
        # is never published, every group abstains for it).
        conn = make_connector_scheduler(
            manager_block_size=MBS, num_groups=1, windowed=1
        )
        groups = {g["name"]: g["spec_names"] for g in build_spec_groups(conn._group_metas, 1)}
        self.assertEqual(sorted(groups), ["full"])
        self.assertIsNone(conn._spec_group_for(True, False))

    def test_window_mask_reads_the_null_entries(self):
        conn = make_connector_scheduler(manager_block_size=MBS, windowed=1)
        ledger = RequestLedger(
            vllm_request=FakeRequest("r0", []),  # ty: ignore[invalid-argument-type]
            # blocks 0 and 2 are real, block 1 was freed (null) ...
            block_ids_per_group=[[7, 0, 9], [1, 2, 3]],
            has_saved_block_num=0,
        )
        self.assertEqual(
            [True, False, True], conn._window_data_mask(ledger, range(3))
        )

    def _announce(self, conn, req_id, blocks, states, windows):
        conn.start_save_kvcache_async(
            req_id, list(range(blocks)), blocks, states, windows
        )
        return conn._manager_client.start_write_cache.call_args.args[0]

    def test_the_announcement_names_the_coverage(self):
        conn = make_connector_scheduler(
            manager_block_size=MBS, num_groups=2, windowed=1
        )
        conn._extra_config = SimpleNamespace(
            instance_id="inst", write_timeout_seconds=1
        )
        conn._manager_client.start_write_cache.return_value = {
            "locations": [],
            "write_session_id": "s",
        }
        request = self._announce(conn, "r0", 3, [True] * 3, [True, False, True])
        self.assertEqual(
            request["location_spec_group_names"],
            ["full", "full_no_window", "full"],
        )

    def test_a_block_without_any_coverage_keeps_the_others_precise(self):
        # A block that carries neither state nor window data has no group of
        # its own. Dropping the whole request to the "all specs" default (what
        # the connector did before) made the manager allocate every spec for
        # every *other* block too, and _save_dispositions then failed the
        # partial-coverage blocks that were perfectly publishable. The empty
        # block is isolated instead: it gets the narrowest declared group (and
        # is failed by its own verdict), the rest keep theirs.
        conn = make_connector_scheduler(
            manager_block_size=MBS, num_groups=1, num_state_groups=1, windowed=1
        )
        conn._extra_config = SimpleNamespace(
            instance_id="inst", write_timeout_seconds=1
        )
        conn._manager_client.start_write_cache.return_value = {
            "locations": [],
            "write_session_id": "s",
        }
        # Every attention spec here is windowed and there is a state group, so
        # the (no state, no window) block carries *nothing*: no declared group
        # describes it. The other three coverages are all declared.
        request = self._announce(
            conn, "r0", 4, [True, False, False, True], [True, True, False, False]
        )
        names = request["location_spec_group_names"]
        self.assertEqual(
            ["full", "attn", conn._narrowest_spec_group, "full_no_window"], names
        )
        # The narrowest declared group here is "attn" (the windowed attention,
        # no state): the empty block still gets *a* name -- the manager needs
        # one per block -- but the others keep their precise coverage instead
        # of the whole request falling back to "all specs".
        self.assertEqual("attn", names[2])

    def test_a_retried_announcement_keeps_the_claimed_window(self):
        # The window slides between two announcements of the same block (a
        # failed session is retried, or preemption rolls the save watermark
        # back). The bytes the first capture copied are still in the worker's
        # arena, so the retry must claim what they cover: announcing the block
        # as window-less would leave captured bytes with no location, and the
        # worker would fail the block and release its only copy.
        conn = make_connector_scheduler(
            manager_block_size=MBS, num_groups=2, windowed=1
        )
        conn._extra_config = SimpleNamespace(
            instance_id="inst", write_timeout_seconds=1
        )
        conn._manager_client.start_write_cache.return_value = {
            "locations": [],
            "write_session_id": "s",
        }
        first = self._announce(conn, "r0", 3, [True] * 3, [True, True, True])
        self.assertEqual(first["location_spec_group_names"], ["full"] * 3)
        retried = self._announce(conn, "r0", 3, [True] * 3, [False, False, False])
        self.assertEqual(
            retried["location_spec_group_names"],
            ["full"] * 3,
            "the retry must claim the window its captured bytes cover",
        )

    def test_a_concurrent_session_keeps_its_captures(self):
        # Two write sessions of one request overlap. Session A announced 16
        # blocks and its response allocated them, so its write task is pending;
        # session B announced 32 and the manager reports blocks 0-15 as
        # "already having a location" -- that is *A's in-flight allocation*,
        # not committed bytes. Releasing A's captures there makes its write
        # fail with "16/16 announced blocks have no captured bytes" and the
        # request's head is lost for good (later sessions see the blocks as
        # existing and never capture them again).
        g0 = windowed_group("tp0_g0", group_idx=0)
        g1 = windowed_group("tp0_g1", group_idx=1)
        pool = _StagingPool(torch.device("cpu"), g0.per_block_bytes, 128)
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {"tp0_g0": pool, "tp0_g1": pool}  # same shape: one pool
        dtm._manager_block_size = MBS
        dtm._device_mod = MagicMock()
        dtm._transfer_client = MagicMock()
        dtm._transfer_client.SaveKvCaches.return_value = [0]
        dtm._extra_config = SimpleNamespace(
            block_per_save_task=128, block_per_load_task=128
        )
        dtm._io_executor = ThreadPoolExecutor(max_workers=2)
        conn = worker_with([g0, g1], dtm)
        conn._pending_saves = {}
        conn._device_mod = MagicMock()
        with mock.patch(
            "kv_cache_manager.py_connector.vllm.data_transfer.batch_gather_scatter_helper"
        ):
            conn._capture_request(
                CaptureRequest(
                    "r0",
                    list(range(32)),
                    all_block_ids=[list(range(7, 39)), list(range(7, 39))],
                )
            )
        plans = []
        real_plan = conn._plan_group_transfers

        def spy(*args, **kwargs):
            result = real_plan(*args, **kwargs)
            if kwargs.get("for_save"):
                plans.append(result)
            return result

        conn._plan_group_transfers = spy

        sched = make_connector_scheduler(
            manager_block_size=MBS, num_groups=1, windowed=1
        )
        sched._extra_config = SimpleNamespace(
            instance_id="inst", write_timeout_seconds=1
        )

        def locations_for(blocks, session):
            return [
                {
                    "block_idxes": [block],
                    "location_specs": [
                        {"name": "tp0_g0", "uri": f"{session}_{block}"},
                        {"name": "tp0_g1", "uri": f"{session}_{block}_b"},
                    ],
                }
                for block in blocks
            ]

        # A: announces 16 blocks, the manager wants all of them written.
        sched._manager_client.start_write_cache.return_value = {
            "locations": locations_for(range(16), "sA"),
            "write_session_id": "sA",
            "block_mask": {"offset": 0},
        }
        sched._record_announcement("r0", 16)
        sched.start_save_kvcache_async(
            "r0", list(range(16 * MBS)), 16, [True] * 16, [True] * 16
        )
        # B: announces 32, and blocks 0-15 come back as existing (A's in-flight
        # allocation); only 16-31 need writing.
        sched._manager_client.start_write_cache.return_value = {
            "locations": locations_for(range(16, 32), "sB"),
            "write_session_id": "sB",
            "block_mask": {"offset": 16},
        }
        sched._record_announcement("r0", 32)
        sched.start_save_kvcache_async(
            "r0", list(range(32 * MBS)), 32, [True] * 32, [True] * 32
        )

        meta = TairKvCacheConnectorMetadata(epoch=0)
        sched._collect_release_captures(meta)
        # Nothing may be released: session A still has to write 0-15.
        self.assertEqual([], meta.to_release_captures)
        # ... and with nothing released, A's write finds all 16 of its blocks.
        meta.add_save_request(
            SaveRequest(
                req_id="r0",
                target_locations=locations_for(range(16), "sA"),
                manager_block_idxes=list(range(16)),
                write_session_id="sA",
                all_block_ids=[list(range(7, 39)), list(range(7, 39))],
            )
        )
        conn.wait_for_save(meta)
        # One call, one plan per bucket: every one of A's 16 blocks must still
        # have its captured bytes.
        per_group = [plan for call in plans for plan in call]
        self.assertEqual(2, len(per_group))
        for plan in per_group:
            self.assertEqual(16, len(plan.staged_slots))
            self.assertNotIn(
                None,
                plan.staged_slots,
                "the concurrent session's release stripped A's captured bytes",
            )
        # A's writes landed and their slots came back; B's 16 blocks (2
        # buckets) are still captured for the session that has not run yet.
        deadline = time.time() + 5
        while pool.free_blocks != 96 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(96, pool.free_blocks)

    def test_an_unanswered_announcement_protects_its_blocks(self):
        # The same guard has to cover a session whose response has not come
        # back yet: its write set is unknowable, so its whole announced range
        # is held back.
        sched = make_connector_scheduler(
            manager_block_size=MBS, num_groups=1, windowed=1
        )
        sched._extra_config = SimpleNamespace(
            instance_id="inst", write_timeout_seconds=1
        )
        sched._manager_client.start_write_cache.return_value = {
            "locations": [
                {
                    "block_idxes": [block],
                    "location_specs": [{"name": "tp0_g0", "uri": "u%d" % block}],
                }
                for block in range(16, 32)
            ],
            "write_session_id": "sB",
            "block_mask": {"offset": 16},
        }
        sched._record_announcement("r0", 16)  # A: still in flight
        sched._record_announcement("r0", 32)  # B: answering now
        sched.start_save_kvcache_async(
            "r0", list(range(32 * MBS)), 32, [True] * 32, [True] * 32
        )
        meta = TairKvCacheConnectorMetadata(epoch=0)
        sched._collect_release_captures(meta)
        self.assertEqual([], meta.to_release_captures)

    def test_existing_blocks_release_their_captures(self):
        # StartWriteCache reporting that the manager already holds a block means
        # this session will not write it: its eager capture has nothing to
        # publish and must give its arena slot back now, not at retirement.
        conn = make_connector_scheduler(
            manager_block_size=MBS, num_groups=2, windowed=1
        )
        conn._extra_config = SimpleNamespace(
            instance_id="inst", write_timeout_seconds=1
        )
        conn._manager_client.start_write_cache.return_value = {
            "locations": [{"block_idxes": [2]}],
            "write_session_id": "s",
            "block_mask": {"offset": 2},
        }
        self._announce(conn, "r0", 4, [True] * 4, [True] * 4)
        self.assertEqual(
            conn.parse_block_mask_to_existing_indices(
                {"block_mask": {"offset": 2}}, 4
            ),
            [0, 1],
        )
        meta = TairKvCacheConnectorMetadata(epoch=0)
        conn._collect_release_captures(meta)
        self.assertEqual(meta.to_release_captures, [("r0", [0, 1])])

    def test_an_all_existing_response_releases_every_announced_block(self):
        conn = make_connector_scheduler(
            manager_block_size=MBS, num_groups=2, windowed=1
        )
        conn._extra_config = SimpleNamespace(
            instance_id="inst", write_timeout_seconds=1
        )
        conn._manager_client.start_write_cache.return_value = {
            "locations": [],
            "write_session_id": "s",
            "block_mask": {"bool_masks": {"values": [True, True, True]}},
        }
        self._announce(conn, "r0", 3, [True] * 3, [True] * 3)
        self.assertEqual(
            conn.parse_block_mask_to_existing_indices(
                {"block_mask": {"bool_masks": {"values": [True, False, True]}}}, 3
            ),
            [0, 2],
        )
        meta = TairKvCacheConnectorMetadata(epoch=0)
        conn._collect_release_captures(meta)
        self.assertEqual(meta.to_release_captures, [("r0", [0, 1, 2])])

    def test_announcement_and_capture_classify_the_same_blocks(self):
        # (two groups: one windowed, one full -- the shape whose announcement
        # can name a coverage at all)
        # The scheduler's window mask and the worker's capture read the same
        # snapshot through the same predicate; a drift between them would mean
        # either a claimed-but-missing block or a captured-but-unclaimed one.
        table = [7, 0, 9, 0, 11]
        conn = make_connector_scheduler(manager_block_size=MBS, windowed=1)
        ledger = RequestLedger(
            vllm_request=FakeRequest("r0", []),  # ty: ignore[invalid-argument-type]
            block_ids_per_group=[list(table)],
            has_saved_block_num=0,
        )
        announced = conn._window_data_mask(ledger, range(len(table)))
        worker = captured_worker(list(range(len(table))), table)
        captured = [
            (group_idx, block) for group_idx, block in worker._staged["r0"]
        ]
        self.assertEqual([0, 2, 4], [block for _g, block in sorted(captured)])
        self.assertEqual(
            [block for block, present in enumerate(announced) if present],
            [block for _g, block in sorted(captured)],
        )


class TestWindowedHitClamp(unittest.TestCase):
    """`_safe_external_prefix` must only end where the window has data."""

    WINDOW_BLOCKS = 4  # window 1024 tokens / mbs 16 -> cdiv(1023, 16) = 64;
    # tests use a small window so the regions stay readable.

    def _conn(self, blocks: int = 8):
        # window_tokens=64 -> k = cdiv(63, 16) = 4 blocks
        return make_connector_scheduler(
            manager_block_size=MBS, locations=[], windowed=1, window_tokens=64
        )

    def _answer(self, conn, locations_, prompt_len):
        conn._location_query_manager.locations = locations_
        conn._location_query_manager.in_flight = False
        req = FakeRequest("r0", list(range(prompt_len)))
        matched, _async = conn.get_num_new_matched_tokens(req, 0)
        return matched

    def test_full_attention_match_is_untouched(self):
        conn = make_connector_scheduler(manager_block_size=MBS, locations=[])
        locs = locations([["tp0_g0"]] * 8)
        self.assertEqual(8 * MBS, self._answer(conn, locs, 20 * MBS))

    def test_complete_window_region_keeps_the_match(self):
        conn = self._conn(8)
        locs = locations([["tp0_g0"]] * 8)
        self.assertEqual(8 * MBS, self._answer(conn, locs, 20 * MBS))

    def test_hole_in_the_window_region_truncates_to_it(self):
        conn = self._conn(8)
        # Block 6 carries no windowed spec: an end at 8 (region [4, 8)) or at 7
        # (region [3, 7)) is unusable, so the match ends at 6.
        locs = locations([["tp0_g0"]] * 6 + [[]] + [["tp0_g0"]])
        self.assertEqual(6 * MBS, self._answer(conn, locs, 20 * MBS))

    def test_hole_far_below_the_window_region_keeps_the_match(self):
        # The constraint is local to the window region: a hole at block 1 does
        # not stop a match ending at block 8 (whose region is [4, 8)).
        conn = self._conn(8)
        locs = locations([["tp0_g0"]] + [[]] + [["tp0_g0"]] * 6)
        self.assertEqual(8 * MBS, self._answer(conn, locs, 20 * MBS))

    def test_hole_at_the_head_is_irrelevant_to_a_long_match(self):
        conn = self._conn(8)
        locs = locations([[]] + [["tp0_g0"]] * 7)
        self.assertEqual(8 * MBS, self._answer(conn, locs, 20 * MBS))

    def test_match_shorter_than_the_window_is_clamped_to_itself(self):
        # Only 2 blocks matched (region [0, 2)): they must all be covered.
        conn = self._conn(2)
        locs = locations([["tp0_g0"]] + [[]])
        self.assertEqual(1 * MBS, self._answer(conn, locs, 20 * MBS))

    def test_extra_retained_tokens_widen_the_region(self):
        # window 64 + 32 retained -> cdiv(95, 16) = 6 blocks required.
        conn = make_connector_scheduler(manager_block_size=MBS, locations=[], windowed=1)
        conn._windowed_metas[0] = AttentionGroupMeta(
            group_idx=0,
            layer_names=["l0"],
            block_size=MBS,
            per_block_bytes=1024,
            page_bytes=MBS * 8,
            retention=WindowRetention(
                RETENTION_SLIDING, window_tokens=64, extra_retained_tokens=32
            ),
        )
        locs = locations([["tp0_g0"]] * 5 + [[]] + [["tp0_g0"]] * 2)
        self.assertEqual(5 * MBS, self._answer(conn, locs, 20 * MBS))
        self.assertEqual(8 * MBS, self._answer(conn, locations([["tp0_g0"]] * 8), 20 * MBS))

    def test_eagle_requires_one_more_block(self):
        # A hole exactly 4 blocks before the end: without EAGLE the region
        # [4, 8) is complete, with EAGLE it is [3, 8) and the hole is inside.
        locs = locations([["tp0_g0"]] * 3 + [[]] + [["tp0_g0"]] * 4)
        plain = make_connector_scheduler(
            manager_block_size=MBS, locations=[], windowed=1, window_tokens=64
        )
        self.assertEqual(8 * MBS, self._answer(plain, locs, 20 * MBS))
        eagle = make_connector_scheduler(
            manager_block_size=MBS,
            locations=[],
            windowed=1,
            window_tokens=64,
            has_eagle_group=True,
        )
        self.assertEqual(3 * MBS, self._answer(eagle, locs, 20 * MBS))

    def test_every_rank_must_carry_the_windowed_spec(self):
        conn = make_connector_scheduler(
            manager_block_size=MBS, locations=[], windowed=1, tp_size=2, window_tokens=64
        )
        complete = [["tp0_g0", "tp1_g0"]] * 4
        self.assertEqual(4 * MBS, self._answer(conn, locations(complete), 20 * MBS))
        # Rank 1's spec missing on the last block: the region [0, 4) is broken,
        # so the match ends at 3 blocks (of which only rank 0's spec is needed
        # on the remaining blocks... they all carry both).
        broken = complete[:-1] + [["tp0_g0"]]
        self.assertEqual(3 * MBS, self._answer(conn, locations(broken), 20 * MBS))

    def test_each_kind_uses_its_own_window(self):
        # sliding / chunked_local: only the window tail must carry data, so a
        # hole exactly `window blocks` before the end is inside the region and
        # the match is cut back to it.
        for kind, window, blocks in (
            (RETENTION_SLIDING, 64, 4),
            (RETENTION_CHUNKED_LOCAL, 64, 4),
        ):
            with self.subTest(kind=kind):
                conn = make_connector_scheduler(
                    manager_block_size=MBS, locations=[], windowed=1
                )
                conn._windowed_metas[0] = AttentionGroupMeta(
                    group_idx=0,
                    layer_names=["l0"],
                    block_size=MBS,
                    per_block_bytes=1024,
                    page_bytes=MBS * 8,
                    retention=WindowRetention(kind, window_tokens=window),
                )
                locs = locations([["tp0_g0"]] * 8)
                locs[8 - blocks] = {"location_specs": []}
                self.assertEqual((8 - blocks) * MBS, self._answer(conn, locs, 20 * MBS))

        # R-SWA: the hit is a full-prefix match (RSWAManager inherits the
        # full-attention manager) and only the gap is masked out, so a hole
        # anywhere in the prefix -- here far below the window tail -- must
        # refuse the whole end: a visible prefill row without data would
        # otherwise read unloaded KV.
        conn = make_connector_scheduler(manager_block_size=MBS, locations=[], windowed=1)
        conn._windowed_metas[0] = AttentionGroupMeta(
            group_idx=0,
            layer_names=["l0"],
            block_size=MBS,
            per_block_bytes=1024,
            page_bytes=MBS * 8,
            retention=WindowRetention(RETENTION_RSWA_GAP, window_tokens=48),
        )
        locs = locations([["tp0_g0"]] * 8)
        locs[1] = {"location_specs": []}  # a mid-prefix hole
        self.assertEqual(1 * MBS, self._answer(conn, locs, 20 * MBS))
        # Without the hole the same geometry serves the whole prefix.
        self.assertEqual(
            8 * MBS, self._answer(conn, locations([["tp0_g0"]] * 8), 20 * MBS)
        )

    def test_the_full_hit_cap_still_applies(self):
        conn = self._conn(8)
        locs = locations([["tp0_g0"]] * 8)
        matched = self._answer(conn, locs, 8 * MBS)
        self.assertLess(matched, 8 * MBS)  # one token stays uncomputed
        self.assertEqual(7 * MBS, matched)

    def test_state_incomplete_end_still_truncates(self):
        # The windowed and hybrid end conditions are evaluated together: the
        # window is complete everywhere, but the state is missing at the end.
        conn = make_connector_scheduler(
            manager_block_size=MBS, locations=[], num_groups=1, num_state_groups=1, windowed=1
        )
        names = ["tp0_g0", "tp0_g1"]
        # The last block has no state: the match ends on the previous block
        # that does -- its window region [3, 7) is complete.
        locs = locations([names] * 7 + [["tp0_g0"]])
        self.assertEqual(7 * MBS, self._answer(conn, locs, 20 * MBS))
        # Same shape with the window broken at block 6: the window constraint
        # wins over the (later) state-complete block.
        locs = locations([names] * 6 + [["tp0_g1"]] + [names])
        self.assertEqual(6 * MBS, self._answer(conn, locs, 20 * MBS))


# --------------------------------------------------------------------------- #
# Worker: capture dispositions
# --------------------------------------------------------------------------- #
class TestCaptureDispositions(unittest.TestCase):
    def test_slot_alignment_matches_the_arena(self):
        # The capture ends up with one gather per block when the member's block
        # is smaller than the arena's slot (a shared arena), and one dense
        # gather otherwise; both leave every block at a slot start, which is
        # what the save task's SDK buffers assume.
        group = windowed_group(per_token_dim=8)  # 16 * 8 * 2 = 256 B per block
        pool = _StagingPool(torch.device("cpu"), 256, 4)
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {group.spec_name: pool}
        dtm._manager_block_size = MBS
        dtm._device_mod = MagicMock()
        import kv_cache_manager.py_connector.vllm.data_transfer as dt

        with mock.patch.object(dt, "batch_gather_scatter_helper") as helper:
            start = dtm.capture_blocks(group, [[0], [1]], timeout=0.1)
        self.assertEqual(0, start)
        self.assertEqual(1, len(helper.batch_gather_kv_caches.call_args_list))

        wide = _StagingPool(torch.device("cpu"), 1024, 4)  # slot > per_block
        dtm._pools = {group.spec_name: wide}
        with mock.patch.object(dt, "batch_gather_scatter_helper") as helper:
            start = dtm.capture_blocks(group, [[0], [1]], timeout=0.1)
        self.assertEqual(0, start)
        self.assertEqual(2, len(helper.batch_gather_kv_caches.call_args_list))
        # Each per-block gather targets its own slot's prefix.
        for offset, call in enumerate(helper.batch_gather_kv_caches.call_args_list):
            view = call.args[1]
            self.assertEqual(1, view.shape[0])

    def test_captured_blocks_are_staged_per_bucket(self):
        conn = captured_worker([0, 1, 2, 3], [7, 8, 9, 10])
        self.assertEqual(1, len(conn._data_transfer.captures))
        name, slots, timeout = conn._data_transfer.captures[0]
        self.assertEqual("tp0_g0", name)
        self.assertEqual(4, len(slots))
        self.assertGreater(timeout, 0)
        self.assertEqual(
            [("tp0_g0", 0), ("tp0_g0", 1), ("tp0_g0", 2), ("tp0_g0", 3)],
            sorted(conn._staged["r0"]),
        )

    def test_null_block_abstains_instead_of_publishing_zero_page(self):
        # Block 1's slots are vLLM's null block (id 0) at capture time: those
        # bytes do not exist, so it is left out of the run (abstain, see
        # TestStagedDispositions for the verdict).
        conn = captured_worker([0, 1], [7, 0])
        self.assertEqual([("tp0_g0", 0)], sorted(conn._staged["r0"]))
        name, slots, _ = conn._data_transfer.captures[0]
        self.assertEqual(1, len(slots))  # only the real block was gathered

    def test_pool_exhaustion_withholds_the_whole_step(self):
        # The arena is held by an in-flight write: the capture waits its
        # deadline, gives up and says so (the caller stops capturing this step).
        group = windowed_group()
        pool = _StagingPool(torch.device("cpu"), group.per_block_bytes, 2)
        self.assertEqual(0, pool.acquire(2))
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {group.spec_name: pool}
        dtm._manager_block_size = MBS
        dtm._device_mod = MagicMock()
        dtm._transfer_client = MagicMock()
        dtm._transfer_client.SaveKvCaches.return_value = [0]
        conn = worker_with([group], dtm)
        self.assertFalse(
            conn._capture_request(
                CaptureRequest("r0", [0, 1], all_block_ids=[[7, 8]])
            )
        )
        self.assertEqual({}, conn._staged.get("r0", {}))
        # The blocks stay announced with no captured bytes: the plan and the
        # task fail them (claimed + absent), so nothing is published.
        plan = conn._plan_group_transfers(
            locations([["tp0_g0"]] * 2), [0, 1], [[7, 8]], "r0", for_save=True
        )
        self.assertEqual([None, None], plan[0].staged_slots)
        verdicts: List[Any] = [None, None]
        dtm.save_task(
            MultiResult(1, lambda flat: verdicts.__setitem__(slice(None), flat)),
            0,
            group,
            plan[0].uris,
            None,
            None,
            plan[0].staged_slots,
        )
        self.assertEqual([False, False], verdicts)
        self.assertEqual(0, pool.free_blocks)  # the write that holds them

    def test_capture_exception_releases_the_runs_it_already_took(self):
        g0 = windowed_group("tp0_g0", group_idx=0)
        g1 = windowed_group("tp0_g1", group_idx=1)
        pool = _FakePool(max_blocks=8)

        class _Raising(_RecordingDTM):
            def capture_blocks(self, group, block_slots, timeout):
                if group.spec_name == "tp0_g1":
                    raise RuntimeError("team died")
                return super().capture_blocks(group, block_slots, timeout)

        dtm = _Raising(pool)
        conn = worker_with([g0, g1], dtm)
        with self.assertRaises(RuntimeError):
            conn._capture_request(
                CaptureRequest("r0", [0, 1], all_block_ids=[[7, 8], [9, 10]])
            )
        # Nothing is left staged: the caller (_capture_blocks) logs and lets the
        # blocks fail closed.
        self.assertEqual([(0, 2)], pool.released)
        self.assertEqual({}, conn._staged.get("r0", {}))

    def test_capture_failure_withholds_and_keeps_serving(self):
        group = windowed_group()
        dtm = _RecordingDTM(_FakePool())
        conn = worker_with([group], dtm)
        with mock.patch.object(
            conn, "_capture_request", side_effect=RuntimeError("boom")
        ):
            conn._capture_blocks(
                SimpleNamespace(
                    to_capture_requests=[
                        CaptureRequest("r0", [0, 1], all_block_ids=[[7, 8]])
                    ]
                )
            )
        # The step's capture budget is spent: the caller stops there.
        self.assertEqual({}, conn._staged.get("r0", {}))

    def test_two_windowed_groups_capture_together(self):
        g0 = windowed_group("tp0_g0", group_idx=0)
        g1 = windowed_group("tp0_g1", group_idx=1)
        dtm = _RecordingDTM(_FakePool(max_blocks=16))
        conn = worker_with([g0, g1], dtm)
        conn._capture_request(
            CaptureRequest("r0", [0, 1], all_block_ids=[[7, 8], [9, 10]])
        )
        self.assertEqual(["tp0_g0", "tp0_g1"], [c[0] for c in dtm.captures])
        self.assertEqual(
            [("tp0_g0", 0), ("tp0_g0", 1), ("tp0_g1", 0), ("tp0_g1", 1)],
            sorted(conn._staged["r0"]),
        )

    def test_two_buckets_of_one_group_keep_their_own_slots(self):
        # One vLLM group, two transfer buckets (a packed group can hold two
        # R-SWA windows or two window retentions): the capture, the plan and
        # the task must key on the bucket, not on the shared group index --
        # keying on the index makes the second bucket's capture overwrite the
        # first's entry, the first plan then takes the *other* bucket's bytes
        # and the second bucket's block fails with no bytes at all.
        g0 = windowed_group("tp0_g0", group_idx=0)
        # A different content dim: another shape, another arena.
        g1 = windowed_group("tp0_g0_b1", group_idx=0, per_token_dim=12)
        pool_a = _StagingPool(torch.device("cpu"), g0.per_block_bytes, 8)
        pool_b = _StagingPool(torch.device("cpu"), g1.per_block_bytes, 8)
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {"tp0_g0": pool_a, "tp0_g0_b1": pool_b}
        dtm._manager_block_size = MBS
        dtm._device_mod = MagicMock()
        dtm._transfer_client = MagicMock()
        dtm._transfer_client.SaveKvCaches.return_value = [0]
        dtm._extra_config = SimpleNamespace(
            block_per_save_task=128, block_per_load_task=128
        )
        dtm._io_executor = ThreadPoolExecutor(max_workers=2)
        conn = worker_with([g0, g1], dtm)
        with mock.patch("kv_cache_manager.py_connector.vllm.data_transfer"
                        ".batch_gather_scatter_helper"):
            conn._capture_request(
                CaptureRequest("r0", [0, 1], all_block_ids=[[7, 8]])
            )
        self.assertEqual(
            [("tp0_g0", 0), ("tp0_g0", 1), ("tp0_g0_b1", 0), ("tp0_g0_b1", 1)],
            sorted(conn._staged["r0"]),
        )
        plans = conn._plan_group_transfers(
            locations([["tp0_g0", "tp0_g0_b1"]] * 2),
            [0, 1],
            [[7, 8]],
            "r0",
            for_save=True,
        )
        self.assertEqual([0, 1], plans[0].staged_slots)
        self.assertEqual([0, 1], plans[1].staged_slots)
        for plan, group in zip(plans, (g0, g1)):
            verdicts = [None, None]
            dtm.save_task(
                MultiResult(1, lambda flat: verdicts.__setitem__(slice(None), flat)),
                0,
                group,
                plan.uris,
                None,
                None,
                plan.staged_slots,
            )
            self.assertEqual([True, True], verdicts)
        # Every slot went back to the arena it came from, exactly once.
        self.assertEqual(8, pool_a.free_blocks)
        self.assertEqual(8, pool_b.free_blocks)

    def test_second_group_exhaustion_releases_the_first_run(self):
        g0 = windowed_group("tp0_g0", group_idx=0)
        g1 = windowed_group("tp0_g1", group_idx=1)
        pool = _FakePool(max_blocks=2)
        dtm = _RecordingDTM(pool)
        conn = worker_with([g0, g1], dtm)
        # First group gets a run of 2, second group cannot: nothing may be held.
        self.assertFalse(
            conn._capture_request(
                CaptureRequest("r0", [0, 1], all_block_ids=[[7, 8], [9, 10]])
            )
        )
        self.assertEqual([(0, 2)], pool.released)
        self.assertEqual({}, conn._staged.get("r0", {}))

    def test_a_repeat_capture_returns_the_slot_it_displaces(self):
        # The scheduler's save watermark can move backwards (preemption shrinks
        # the allocation), so the same block can be captured twice. The repeat
        # overwrites the old entry -- the old slot has no session reference (a
        # session *moves* the entry into _in_flight when it takes it), so it
        # must go back to the arena: overwriting it silently leaks the slot for
        # the rest of the process's life.
        pool = _StagingPool(torch.device("cpu"), 256, 8)
        dtm = _RecordingDTM(pool)
        conn = worker_with([windowed_group()], dtm)
        conn._capture_request(CaptureRequest("r0", [0, 1], all_block_ids=[[7, 8]]))
        self.assertEqual(6, pool.free_blocks)
        self.assertEqual(
            [0, 1], sorted(slot for _key, (_pool, slot) in conn._staged["r0"].items())
        )
        conn._capture_request(CaptureRequest("r0", [0, 1], all_block_ids=[[7, 8]]))
        # The repeat holds its new run and the displaced one is already back:
        # arena use is one run, not two.
        self.assertEqual(6, pool.free_blocks)
        self.assertEqual(
            [2, 3], sorted(slot for _key, (_pool, slot) in conn._staged["r0"].items())
        )
        conn._release_staged("r0")
        self.assertEqual(8, pool.free_blocks)
        self.assertEqual({}, conn._staged.get("r0", {}))

    def test_one_timeout_spends_the_whole_step_budget(self):
        # Every capture of a step waits on the same arena, so the second one
        # would only burn the same deadline again (the engine thread pays the
        # sum): the first timeout ends the step's capturing, and the skipped
        # requests fail closed exactly like a withheld one.
        group = windowed_group()
        pool = _StagingPool(torch.device("cpu"), group.per_block_bytes, 2)
        self.assertEqual(0, pool.acquire(2))  # held by an in-flight write
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {group.spec_name: pool}
        dtm._manager_block_size = MBS
        dtm._device_mod = MagicMock()
        dtm._transfer_client = MagicMock()
        conn = worker_with([group], dtm)
        with mock.patch.object(
            conn._data_transfer, "capture_blocks", wraps=conn._data_transfer.capture_blocks
        ) as capture:
            conn._capture_blocks(
                SimpleNamespace(
                    to_capture_requests=[
                        CaptureRequest("r0", [0], all_block_ids=[[7]]),
                        CaptureRequest("r1", [0], all_block_ids=[[8]]),
                        CaptureRequest("r2", [0], all_block_ids=[[9]]),
                    ]
                )
            )
        # Only the first request tried (and timed out): the other two were
        # skipped with it.
        self.assertEqual(1, capture.call_count)
        self.assertEqual({}, conn._staged)

    def test_wait_for_save_honours_the_release_instruction(self):
        conn = captured_worker([0, 1], [7, 8])
        meta = TairKvCacheConnectorMetadata(epoch=0)
        meta.release_captures("r0", [1])
        conn.wait_for_save(meta)
        self.assertEqual([("tp0_g0", 0)], sorted(conn._staged["r0"]))

    def test_release_captures_returns_only_the_listed_blocks(self):
        conn = captured_worker([0, 1], [7, 8])
        conn._release_captures_for("r0", [0])
        # Block 0's slot went back with the instruction; block 1's capture is
        # still there for the session that will write it.
        self.assertEqual(
            [("tp0_g0", 1)], sorted(conn._staged["r0"]),
        )
        self.assertEqual([(0, 1)], conn._data_transfer.pool.released)

    def test_finish_releases_unwritten_slots(self):
        conn = captured_worker([0, 1], [7, 8])
        conn._release_staged("r0")
        self.assertEqual([(0, 1), (1, 1)], conn._data_transfer.pool.released)
        self.assertNotIn("r0", conn._staged)
        self.assertNotIn("r0", conn._in_flight)


class TestStagedDispositions(unittest.TestCase):
    """The save side's three states, as the plan and the task see them."""

    def _plan(self, conn, table, staged_blocks):
        return conn._plan_group_transfers(
            locations([["tp0_g0"]] * len(table)),
            list(range(len(table))),
            [list(table)],
            "r0",
            for_save=True,
        )

    def test_captured_block_points_at_its_slot(self):
        conn = captured_worker([0, 1], [7, 8])
        plans = self._plan(conn, [7, 8], conn._staged["r0"])
        self.assertEqual([0, 1], plans[0].staged_slots)

    def test_withheld_block_keeps_its_location_and_fails(self):
        # A withheld capture still has an announced location (the scheduler
        # claimed the spec before the arena turned out to be full): the task
        # must fail the block, never publish the claim.
        conn = captured_worker([0, 1], [7, 8], exhausted=True)
        plans = self._plan(conn, [7, 8], {})
        self.assertIsNotNone(plans[0].uris[0])
        self.assertEqual([None, None], plans[0].staged_slots)

    def test_null_at_capture_abstains(self):
        conn = captured_worker([0, 1], [7, 0])
        plans = self._plan(conn, [7, 0], conn._staged["r0"])
        self.assertEqual([0, None], plans[0].staged_slots)

    def test_verdicts_of_the_four_states(self):
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._transfer_client = MagicMock()
        group = windowed_group()
        # 0: claimed + captured -> transfers
        # 1: unclaimed + no bytes -> abstains (the announcement said "no window")
        # 2: claimed + no bytes -> fails (withheld capture)
        # 3: captured + unclaimed -> fails (the two sides disagree)
        skipped, failed = dtm._staged_dispositions(
            group, ["u0", None, "u2", None], [0, None, None, 3], 4
        )
        self.assertEqual({1}, skipped)
        self.assertEqual({2, 3}, failed)


# --------------------------------------------------------------------------- #
# Worker: the load side's null disposition
# --------------------------------------------------------------------------- #
class TestLoadNullDisposition(unittest.TestCase):
    def _plan(self, conn, table, group=None, blocks=None):
        group = group or windowed_group()
        blocks = list(range(len(table))) if blocks is None else blocks
        return conn._plan_group_transfers(
            locations([["tp0_g0"]] * len(blocks)),
            blocks,
            [list(table)],
            "load-req",
        )[0]

    def test_all_null_block_is_skipped(self):
        # mbs 16 (one group block): the whole block targets the null page.
        conn = worker_with([windowed_group()])
        plan = self._plan(conn, [0])
        self.assertEqual([NullBlockRow(rows=list(range(MBS)), source_row=None)],
                         plan.null_rows)

    def test_no_null_rows_means_no_annotation(self):
        conn = worker_with([windowed_group()])
        self.assertEqual([None], self._plan(conn, [5]).null_rows)

    def test_partial_null_aliases_onto_the_first_real_row(self):
        # mbs 64 = 4 group blocks of 16: group blocks 0 and 1 are null, 2 and 3
        # real -> rows 0..31 alias row 32 (the first real token).
        group = windowed_group(block_size=16, kernel_block_size=16)
        conn = make_connector(manager_block_size=64)
        conn._kvcache_info = KVCacheInfo(
            tp_rank=0, world_size=1, groups=[group], device="cpu", dtype=KV_DTYPE
        )
        conn._data_transfer = None
        conn._staged, conn._in_flight = {}, {}
        conn._withheld, conn._capture_scope = {}, {}
        plan = self._plan(conn, [0, 0, 7, 8], group, blocks=[0])
        alias = plan.null_rows[0]
        self.assertEqual(list(range(32)), alias.rows)
        self.assertEqual(32, alias.source_row)
        # Rows 0..31 of the kernel's slot list must point at row 32's slot: the
        # alias is only sound if the data written there is the data that row
        # holds (the task copies it), so the slots have to agree.
        self.assertEqual(plan.token_indices[0][32 - 1], plan.token_indices[0][32])

    def test_a_full_attention_group_is_never_annotated(self):
        group = windowed_group()
        plain = AttentionTransferGroup(
            **{
                **group.__dict__,
                "retention": None,
                "kvcache_ptr_tensor_gpu": None,
            }
        )
        conn = worker_with([plain])
        self.assertIsNone(self._plan(conn, [0], plain).null_rows)


# --------------------------------------------------------------------------- #
# Transfer manager: staged writes and the pool deadline
# --------------------------------------------------------------------------- #
class TestStagedWrites(unittest.TestCase):
    def setUp(self) -> None:
        # The stub ring's kvcm_py_client is a MagicMock whose Iov()/BlockBuffer()
        # return one shared object; the pointer assertions need real ones.
        import kv_cache_manager.py_connector.vllm.data_transfer as dt

        patcher = mock.patch.object(dt, "kvcm_py_client", _RealClientStub)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _dtm(self, pool: _StagingPool, ok: bool = True) -> DataTransferManager:
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {"tp0_g0": pool}
        dtm._manager_block_size = MBS
        dtm._transfer_client = MagicMock()
        dtm._transfer_client.SaveKvCaches.return_value = [0 if ok else 1]
        return dtm

    def test_buffers_point_at_the_captured_slots(self):
        pool = _StagingPool(torch.device("cpu"), 1024, 4)
        start = pool.acquire(2)  # what the eager capture did
        self.assertNotEqual(start, None)
        dtm = self._dtm(pool)
        results = {}
        mr = MultiResult(1, lambda flat: results.setdefault("flat", flat))
        dtm.save_task(
            mr,
            0,
            windowed_group(),
            remote_uris=["u0", "u1"],
            block_token_indices=None,
            block_ids=None,
            staged_slots=[start + 1, start],
        )
        uris, buffers = dtm._transfer_client.SaveKvCaches.call_args.args
        self.assertEqual(["u0", "u1"], uris)
        self.assertEqual(
            [pool.slot_base_ptr(start + 1), pool.slot_base_ptr(start)],
            [b.iovs[0].base for b in buffers],
        )
        self.assertEqual([True, True], results["flat"])
        # The written slots are back in the arena: a 4-slot acquire succeeds.
        self.assertEqual(0, pool.acquire(4, timeout=0.01))

    def test_failed_write_returns_the_slots_once(self):
        pool = _StagingPool(torch.device("cpu"), 1024, 2)
        start = pool.acquire(2)
        dtm = self._dtm(pool, ok=False)
        results = {}
        mr = MultiResult(1, lambda flat: results.setdefault("flat", flat))
        dtm.save_task(
            mr,
            0,
            windowed_group(),
            remote_uris=["u0", "u1"],
            block_token_indices=None,
            block_ids=None,
            staged_slots=[start, start + 1],
        )
        self.assertEqual([False, False], results["flat"])
        # A failed transfer returns its slots too: the block is announced once,
        # so nothing waits for these bytes, and a held slot would keep the
        # arena -- and the accounting -- from ever settling.
        self.assertEqual(0, pool.acquire(2, timeout=0.01))
        pool.release(0, 2)
        self.assertEqual(2, pool.free_blocks)
        # ... exactly once: the pool refuses a second release of the same slots.
        with self.assertRaises(ValueError):
            pool.release(0, 2)

    def test_withheld_block_fails_and_abstained_block_is_none(self):
        pool = _StagingPool(torch.device("cpu"), 1024, 2)
        start = pool.acquire(1)
        dtm = self._dtm(pool)
        results = {}
        mr = MultiResult(1, lambda flat: results.setdefault("flat", flat))
        dtm.save_task(
            mr,
            0,
            windowed_group(),
            # block0 withheld (claimed, no bytes), block1 abstained (unclaimed,
            # nothing to write), block2 written.
            remote_uris=["u0", None, "u2"],
            block_token_indices=None,
            block_ids=None,
            staged_slots=[None, None, start],
        )
        self.assertEqual([False, None, True], results["flat"])
        self.assertIsNotNone(pool.acquire(2, timeout=0.01))


class TestPoolDeadline(unittest.TestCase):
    """The capture's bounded wait: blocking forever would deadlock the engine."""

    def test_acquire_times_out_and_returns_none(self):
        pool = _StagingPool(torch.device("cpu"), 1024, 2)
        self.assertEqual(0, pool.acquire(2))
        started = time.monotonic()
        self.assertIsNone(pool.acquire(1, timeout=0.05))
        self.assertLess(time.monotonic() - started, 5.0)

    def test_acquire_is_woken_by_a_release(self):
        pool = _StagingPool(torch.device("cpu"), 1024, 2)
        pool.acquire(2)
        woke = threading.Event()

        def waiter():
            self.assertEqual(0, pool.acquire(2, timeout=5.0))
            woke.set()

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.05)
        pool.release(0, 2)
        self.assertTrue(woke.wait(5.0))
        thread.join()

    def test_blocking_acquire_is_unchanged(self):
        # The transfer tasks keep the historical contract: no deadline means
        # wait for a run, however long that takes.
        pool = _StagingPool(torch.device("cpu"), 1024, 1)
        pool.acquire(1)
        done = threading.Event()

        def waiter():
            self.assertEqual(0, pool.acquire(1))
            done.set()

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.05)
        pool.release(0, 1)
        self.assertTrue(done.wait(5.0))
        thread.join()




class TestCapturedSlotOwnership(unittest.TestCase):
    """Who owns a captured slot, and when it comes back (review_r1 S1).

    The report's sequence, reproduced exactly: capture -> the session takes the
    block (the plan hands the slot over) -> the request retires in the *same*
    step (vLLM calls wait_for_save before get_finished) -> the write lands. A
    retiring request must not free a slot its session is still reading, and the
    arena must never hand the same slot out twice.
    """

    CAPACITY = 16
    BLOCKS = 4

    def _fixture(self, ok: bool = True):
        # The capture's gather is the one GPU call in this path; the arena
        # bookkeeping under test stays the real thing.
        import kv_cache_manager.py_connector.vllm.data_transfer as dt

        patcher = mock.patch.object(dt, "batch_gather_scatter_helper")
        patcher.start()
        self.addCleanup(patcher.stop)
        group = windowed_group()
        pool = _StagingPool(
            torch.device("cpu"),
            group.per_block_bytes,
            self.CAPACITY,
            reserved_blocks=2,
        )
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {group.spec_name: pool}
        dtm._manager_block_size = MBS
        dtm._device_mod = MagicMock()
        dtm._transfer_client = MagicMock()
        dtm._transfer_client.SaveKvCaches.return_value = [0 if ok else 1]
        conn = worker_with([group], dtm)
        return conn, dtm, pool, group

    def _assert_arena_accounting(self, conn, pool, sessions):
        """The arena's books: free slots + the slots a session still holds.

        ``sessions`` is the number of captured slots a *write task* is holding
        (the test knows them; after a retirement the worker deliberately stops
        tracking them, the pool does not). Anything else the worker lists
        (``_staged``) is held too. Under the S1 bug the pool's free count grew
        past the capacity (a slot in the free list twice) and this is what
        catches it.
        """
        staged = sum(len(entries) for entries in conn._staged.values())
        self.assertLessEqual(
            pool.free_blocks + staged + sessions,
            self.CAPACITY,
            "the arena handed out more slots than it has: a range was freed twice",
        )
        self.assertEqual(self.CAPACITY, pool.free_blocks + staged + sessions)

    def _plan(self, conn, blocks):
        return conn._plan_group_transfers(
            locations([["tp0_g0"]] * len(blocks)),
            blocks,
            [list(range(7, 7 + len(blocks)))],
            "r0",
            for_save=True,
        )[0]

    def test_a_retiring_request_does_not_free_an_in_flight_slot(self):
        conn, dtm, pool, group = self._fixture()
        blocks = list(range(self.BLOCKS))
        conn._capture_request(CaptureRequest("r0", blocks, all_block_ids=[[7, 8, 9, 10]]))
        self._assert_arena_accounting(conn, pool, sessions=0)
        self.assertEqual(self.BLOCKS, len(conn._staged["r0"]))

        plan = self._plan(conn, blocks)
        self.assertEqual(blocks, plan.staged_slots)
        # The session owns the slots now; they are still held.
        self.assertEqual({}, conn._staged["r0"])
        self.assertEqual(self.BLOCKS, len(conn._in_flight["r0"]))
        self.assertEqual(self.CAPACITY - self.BLOCKS, pool.free_blocks)

        # The FinishRequest arrives in the same engine step as the save: the
        # report's accounting sequence, whose bug freed these slots too.
        conn._release_staged("r0")
        self.assertEqual(
            self.CAPACITY - self.BLOCKS,
            pool.free_blocks,
            "a retiring request freed slots its write session is still reading",
        )
        self._assert_arena_accounting(conn, pool, sessions=self.BLOCKS)

        # The write lands: one release, not two.
        done = {}
        dtm.save_task(
            MultiResult(1, lambda flat: done.setdefault("flat", flat)),
            0,
            group,
            plan.uris,
            None,
            None,
            plan.staged_slots,
        )
        self.assertEqual([True] * self.BLOCKS, done["flat"])
        self.assertEqual(self.CAPACITY, pool.free_blocks)
        self._assert_arena_accounting(conn, pool, sessions=0)
        # Everything is free again, and the two halves of the arena (the shared
        # part and the transfer reserve) add up to the capacity: nothing is in
        # the free list twice (the report measured free=1408/1024 here).
        shared = self.CAPACITY - pool.reserved_blocks
        self.assertEqual(0, pool.acquire(shared))
        self.assertEqual(shared, pool.acquire(pool.reserved_blocks))
        self.assertEqual(0, pool.free_blocks)
        pool.release(0, shared)
        pool.release(shared, pool.reserved_blocks)
        self.assertEqual(self.CAPACITY, pool.free_blocks)

    def test_a_failed_write_returns_the_slot_too(self):
        conn, dtm, pool, group = self._fixture(ok=False)
        blocks = list(range(self.BLOCKS))
        conn._capture_request(CaptureRequest("r0", blocks, all_block_ids=[[7, 8, 9, 10]]))
        plan = self._plan(conn, blocks)
        done = {}
        dtm.save_task(
            MultiResult(1, lambda flat: done.setdefault("flat", flat)),
            0,
            group,
            plan.uris,
            None,
            None,
            plan.staged_slots,
        )
        # A block is announced to the manager once (has_saved_block_num advances
        # at dispatch, not at success), so nothing waits for these bytes: the
        # slot goes back even though the transfer failed -- and the block is
        # not published.
        self.assertEqual([False] * self.BLOCKS, done["flat"])
        self.assertEqual(self.CAPACITY, pool.free_blocks)
        self._assert_arena_accounting(conn, pool, sessions=0)

    def test_the_arena_never_holds_a_slot_twice(self):
        # The pool refuses a second release of the same slots: with the
        # ownership move above this cannot happen, and if a future path breaks
        # it the failure is loud instead of a duplicated dispatch.
        pool = _StagingPool(torch.device("cpu"), 64, 4)
        pool.acquire(2)
        pool.release(0, 2)
        with self.assertRaises(ValueError):
            pool.release(0, 2)
        with self.assertRaises(ValueError):
            pool.release(1, 1)
        with self.assertRaises(ValueError):
            pool.release(0, 5)

    def _release_staged_after_plan(self, conn, pool):
        conn._release_staged("r0")
        # Block 1 came back (no session took it); block 0's slot stays with the
        # task that reads it.
        self.assertEqual(self.CAPACITY - 1, pool.free_blocks)
        self._assert_arena_accounting(conn, pool, sessions=1)

    def test_a_raising_sdk_still_returns_the_slots(self):
        # The plan handed these slots to this task, so nothing else can return
        # them: an SDK that raises (a pybind/backend exception, not an error
        # code) must not shrink the arena for the rest of the process's life
        # (review_r2 N1 measured 384 of 1024 slots leaking that way).
        conn, dtm, pool, group = self._fixture()
        blocks = list(range(self.BLOCKS))
        conn._capture_request(CaptureRequest("r0", blocks, all_block_ids=[[7, 8, 9, 10]]))
        plan = self._plan(conn, blocks)
        dtm._transfer_client.SaveKvCaches.side_effect = RuntimeError("pybind blew up")
        with self.assertRaises(RuntimeError):
            dtm._save_staged_blocks(
                group, plan.uris, plan.staged_slots, [0, 1, 2, 3], [None] * 4
            )
        self.assertEqual(self.CAPACITY, pool.free_blocks)
        self._assert_arena_accounting(conn, pool, sessions=0)

        # The same crash through save_task: its single exit turns the escaping
        # exception into all-failed verdicts, and the slots still come back.
        blocks = list(range(self.BLOCKS))
        conn._capture_request(CaptureRequest("r0", blocks, all_block_ids=[[7, 8, 9, 10]]))
        plan = self._plan(conn, blocks)
        done = {}
        dtm.save_task(
            MultiResult(1, lambda flat: done.setdefault("flat", flat)),
            0,
            group,
            plan.uris,
            None,
            None,
            plan.staged_slots,
        )
        self.assertEqual([False] * self.BLOCKS, done["flat"])
        self.assertEqual(self.CAPACITY, pool.free_blocks)
        self._assert_arena_accounting(conn, pool, sessions=0)

    def test_a_failed_contract_returns_its_staged_slots(self):
        # claimed=False + captured=True (the capture holds bytes for a block
        # whose location never claimed this spec) is a contract violation: the
        # block fails *and* the task returns the slot it owns -- failing alone
        # would leave the arena short for the rest of the process's life.
        group = windowed_group()
        pool = _StagingPool(torch.device("cpu"), group.per_block_bytes, 4)
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {group.spec_name: pool}
        dtm._manager_block_size = MBS
        dtm._device_mod = MagicMock()
        dtm._transfer_client = MagicMock()
        dtm._transfer_client.SaveKvCaches.return_value = [0]
        conn = worker_with([group], dtm)
        with mock.patch("kv_cache_manager.py_connector.vllm.data_transfer"
                        ".batch_gather_scatter_helper"):
            conn._capture_request(CaptureRequest("r0", [0], all_block_ids=[[7]]))
        self.assertEqual(3, pool.free_blocks)
        verdicts: List[Any] = [None]
        dtm.save_task(
            MultiResult(1, lambda flat: verdicts.__setitem__(slice(None), flat)),
            0,
            group,
            [None],  # no location claimed this block's spec
            None,
            None,
            [0],
        )
        self.assertEqual([False], verdicts)
        self.assertEqual(4, pool.free_blocks)

    def test_a_second_plan_fails_instead_of_handing_the_slot_out_twice(self):
        conn, dtm, pool, group = self._fixture()
        conn._capture_request(CaptureRequest("r0", [0], all_block_ids=[[7]]))
        first = self._plan(conn, [0])
        self.assertEqual([0], first.staged_slots)
        second = self._plan(conn, [0])
        # A block is announced once, so a repeat plan is a contract violation:
        # the block is failed (no bytes, and the pool keeps its slot once).
        self.assertEqual([None], second.staged_slots)
        self.assertEqual(self.CAPACITY - 1, pool.free_blocks)

    def test_retirement_releases_what_no_session_took(self):
        conn, dtm, pool, group = self._fixture()
        conn._capture_request(CaptureRequest("r0", [0, 1], all_block_ids=[[7, 8]]))
        self._plan(conn, [0])  # only block 0 reaches a session
        self._release_staged_after_plan(conn, pool)


class TestAllNullLoadVerdict(unittest.TestCase):
    """An all-null target block is skipped, not failed (review_r1 B1)."""

    def test_all_null_block_verdict_is_none_not_false(self):
        group = windowed_group()
        pool = _StagingPool(torch.device("cpu"), group.per_block_bytes, 4)
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {group.spec_name: pool}
        dtm._manager_block_size = MBS
        dtm._device_mod = MagicMock()
        dtm._load_stream = MagicMock()
        dtm._transfer_client = MagicMock()
        done = {}
        dtm.load_task(
            MultiResult(1, lambda flat: done.setdefault("flat", flat)),
            0,
            group,
            remote_uris=[None],
            block_token_indices=[[0] * MBS],
            block_ids=None,
            null_rows=[NullBlockRow(rows=list(range(MBS)), source_row=None)],
        )
        # vLLM allocated no slot for this block (it is outside the window), so
        # there was nothing to restore: the group abstains, and a verdict of
        # False here is what made vLLM discard the whole external hit.
        self.assertEqual([None], done["flat"])
        self.assertFalse(dtm._transfer_client.LoadKvCaches.called)
        self.assertEqual(4, pool.free_blocks)


class TestAliasedNullRows(unittest.TestCase):
    """A manager block straddling the window boundary (review_r1 B2).

    The report's geometry: 32-token manager blocks over 16-token group blocks,
    so one block's first 16 rows target vLLM's null page and the rest are real.
    The loader fills the null rows from the block's first real row -- shapes
    included -- and the token indices already point there, so the scatter
    writes the same correct bytes twice.
    """

    def setUp(self) -> None:
        import kv_cache_manager.py_connector.vllm.data_transfer as dt

        patcher = mock.patch.object(dt, "kvcm_py_client", _RealClientStub)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_null_rows_take_the_source_rows_bytes(self):
        group = replace(
            windowed_group(block_size=16, kernel_block_size=16), per_block_bytes=512
        )
        pool = _StagingPool(torch.device("cpu"), group.per_block_bytes, 4)
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {group.spec_name: pool}
        dtm._manager_block_size = 32
        dtm._device_mod = MagicMock()
        dtm._load_stream = MagicMock()
        dtm._transfer_client = MagicMock()

        def fill(uris, buffers):
            base = pool._cpu.data_ptr()
            for i, buffer in enumerate(buffers):
                offset = buffer.iovs[0].base - base
                size = buffer.iovs[0].size
                pool._cpu[offset : offset + size] = (
                    torch.arange(size, dtype=torch.uint8) + i
                )
            return 0

        dtm._transfer_client.LoadKvCaches.side_effect = fill
        import kv_cache_manager.py_connector.vllm.data_transfer as dt

        with mock.patch.object(dt, "batch_gather_scatter_helper") as helper:
            dtm._load_valid_blocks(
                group,
                remote_uris=["u0"],
                block_token_indices=[[0] * 16 + [112 + i for i in range(16)]],
                block_ids=None,
                valid=[0],
                null_rows=[NullBlockRow(rows=list(range(16)), source_row=16)],
            )
        helper.batch_scatter_kv_caches.assert_called_once()
        view = dtm._attn_staging_view(
            pool.cpu_view(0, 1, group.per_block_bytes), group, 1
        )
        source = view[0, :, 16, :].clone()
        for row in range(16):
            self.assertTrue(
                torch.equal(view[0, :, row, :], source),
                f"null row {row} does not carry the source row's bytes",
            )
        self.assertFalse(torch.equal(source, torch.zeros_like(source)))


class TestClampCoversTheNeededWindow(unittest.TestCase):
    """The clamp's requirement is a superset of the tokens a load needs (N4).

    An accepted external match of ``m`` tokens needs KV for [m-W+1, m). Every
    one of those tokens must lie inside a manager block the *announcement*
    claimed (claimed => the windowed group's data was copied, so the load can
    read it). The clamp asks for ``cdiv(W-1+extra, mbs)`` (+1 with EAGLE) such
    blocks, which covers m-W+1 tokens only if that count times mbs is at least
    W-1 -- the check below re-derives it from the returned match, so a clamp
    that asked for too few blocks would fail here. Run over geometries where
    neither block size divides the other, the case the report could not falsify.
    """

    def _accepted(self, mbs, window, extra, table, eagle=False):
        conn = make_connector_scheduler(
            manager_block_size=mbs,
            locations=[],
            windowed=1,
            has_eagle_group=eagle,
        )
        conn._windowed_metas[0] = AttentionGroupMeta(
            group_idx=0,
            layer_names=["l0"],
            block_size=mbs,
            per_block_bytes=1024,
            page_bytes=mbs * 8,
            retention=WindowRetention(
                RETENTION_SLIDING,
                window_tokens=window,
                extra_retained_tokens=extra,
            ),
        )
        conn._location_query_manager.locations = locations(
            [["tp0_g0"] if entry else [] for entry in table]
        )
        conn._location_query_manager.in_flight = False
        req = FakeRequest("r0", list(range(len(table) * mbs + mbs)))
        matched, _async = conn.get_num_new_matched_tokens(req, 0)
        return matched

    def test_eagle_padding_is_a_hash_block_not_a_manager_block(self):
        # preferred_block_size (16) below the scheduler's block size (64): the
        # EAGLE drop moves the resumed window back a whole 64-token block, i.e.
        # four manager blocks. A hole five blocks before the end is inside that
        # region, so the match must stop there; a constant one-block padding
        # (what the clamp asked for before) accepts the whole tail.
        conn = make_connector_scheduler(
            manager_block_size=16,
            vllm_block_size=64,
            locations=[],
            windowed=1,
            has_eagle_group=True,
        )
        conn._windowed_metas[0] = AttentionGroupMeta(
            group_idx=0,
            layer_names=["l0"],
            block_size=16,
            per_block_bytes=1024,
            page_bytes=16 * 8,
            retention=WindowRetention(RETENTION_SLIDING, window_tokens=64),
        )
        blocks = 24
        table = [True] * blocks
        table[blocks - 6] = False  # a hole six blocks before the end
        locs = locations([["tp0_g0"] if present else [] for present in table])
        # cdiv(64 - 1 + 64, 16) = 8 blocks of tail are required, so [16, 24) is
        # the region and the hole at 18 ends the match there.
        self.assertEqual(blocks - 6, conn._windowed_match_end(locs, blocks))

    def test_the_claimed_region_covers_every_needed_token(self):
        cases = [
            (16, 16, 1024, 0, False),
            (16, 8, 1023, 0, False),  # group blocks twice as large
            (32, 16, 1024, 8, False),  # extra retained tokens
            (16, 24, 1000, 0, False),  # neither divides the other
            (24, 16, 999, 3, True),  # neither divides, EAGLE (+1)
            (16, 16, 64, 0, True),
        ]
        for mbs, gbs, window, extra, eagle in cases:
            with self.subTest(mbs=mbs, gbs=gbs, window=window, extra=extra, eagle=eagle):
                blocks = 24
                # The realistic shape: the head was freed (null blocks), the
                # tail carries data -- the claim mask follows the data.
                null_head = list(range(0, 4))
                table = [0 if mb in null_head else 7 + mb for mb in range(blocks)]
                matched = self._accepted(mbs, window, extra, table, eagle)
                if matched == 0:
                    continue
                covered_from = (matched // mbs - _k(window, extra, mbs, eagle)) * mbs
                self.assertLessEqual(
                    covered_from,
                    max(0, matched - window + 1),
                    f"the clamp accepted {matched} tokens whose window starts at "
                    f"{matched - window + 1}, but only claims data from "
                    f"{covered_from}",
                )




class TestPackedGroupsAndStateBlocks(unittest.TestCase):
    """Configurations the connector must refuse or now accepts (registration)."""

    def test_a_packed_windowed_group_is_refused_but_the_folded_view_hides_it(self):
        # The scheduler only sees vLLM's folded view: one representative sub
        # spec per group. A packed group that mixes a windowed member with
        # another member would therefore be announced and clamped for one
        # bucket only -- the other bucket's blocks would never be published and
        # a load of them would fail silently in multi-group instances. The
        # worker (which sees every member) refuses the shape; the probe shows
        # why the scheduler cannot: its view has a single bucket.
        from vllm.v1.kv_cache_interface import SlidingWindowSpec, UniformTypeKVCacheSpecs

        def spec(window):
            return SlidingWindowSpec(
                block_size=MBS,
                sliding_window=window,
                page_size_bytes=2048,
                num_kv_heads=2,
                head_size=64,
                dtype=torch.bfloat16,
            )

        worker_view = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["l0", "l1"],
                    kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(
                        {"l0": spec(1024), "l1": spec(512)}
                    ),
                    is_eagle_group=False,
                )
            ]
        )
        with self.assertRaises(NotImplementedError) as ctx:
            parse_groups(worker_view, MBS, calculate_kv_scales=False)
        self.assertIn("folds", str(ctx.exception))
        # ... what the scheduler gets instead: one representative, one bucket.
        folded = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["l0", "l1"], kv_cache_spec=spec(1024), is_eagle_group=False
                )
            ]
        )
        metas = parse_groups(folded, MBS, calculate_kv_scales=False)
        self.assertEqual(1, len(metas))
        self.assertEqual(1024, metas[0].retention.window_tokens)

    def test_state_blocks_smaller_than_the_manager_block_are_aligned(self):
        # --mamba-block-size below --block-size: a 48-token manager block over
        # 16-token state blocks selects the states ending at 47, 95, 143, i.e.
        # 2, 5, 8 -- one state per manager block, each covering that block's
        # prefix, which is exactly what the transfer path needs.
        from kv_cache_manager.py_connector.vllm.transfer_types import (
            StateTransferGroup,
        )

        conn = make_connector(manager_block_size=48)
        group = StateTransferGroup(
            group_idx=0,
            spec_name="tp0_s0",
            layer_names=["s0"],
            block_size=16,
            per_block_bytes=1024,
            layer_num=1,
            block_view_tensors=[],
            page_size_bytes=1024,
        )
        conn._kvcache_info = KVCacheInfo(
            tp_rank=0, world_size=1, groups=[group], device="cpu", dtype=KV_DTYPE
        )
        table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(
            [2, 5, 8],
            conn._state_block_ids(group, [0, 1, 2], table),
        )
        # Registration accepts it too (the equality assertion would refuse).
        built = conn._build_state_group(
            SimpleNamespace(
                group_idx=0,
                layer_names=["s0"],
                block_size=16,
                per_block_bytes=1024,
                page_size_bytes=1024,
                spec_suffix="",
            ),
            {"s0": torch.zeros(1, 1, 1, 1024, dtype=torch.uint8)},
        )
        self.assertEqual(16, built.block_size)

    def test_state_blocks_larger_than_the_manager_block_are_refused(self):
        # The other direction reproduces one state under several manager keys.
        conn = make_connector(manager_block_size=16)
        from kv_cache_manager.py_connector.vllm.transfer_types import (
            StateTransferGroup,
        )

        group = StateTransferGroup(
            group_idx=0,
            spec_name="tp0_s0",
            layer_names=["s0"],
            block_size=64,
            per_block_bytes=1024,
            layer_num=1,
            block_view_tensors=[],
            page_size_bytes=1024,
        )
        conn._kvcache_info = KVCacheInfo(
            tp_rank=0, world_size=1, groups=[group], device="cpu", dtype=KV_DTYPE
        )
        meta = SimpleNamespace(
            group_idx=0,
            layer_names=["s0"],
            block_size=64,
            per_block_bytes=1024,
            page_size_bytes=1024,
        )
        with self.assertRaises(AssertionError) as ctx:
            conn._build_state_group(meta, {"s0": torch.zeros(1024)})
        self.assertIn("multiple of the state block", str(ctx.exception))


# --------------------------------------------------------------------------- #
# Wiring: the two task-batch call sites (PR2 review N4)
# --------------------------------------------------------------------------- #
class _FakeBatchDTM:
    """A transfer manager whose every surface is recorded."""

    def __init__(self, batch: int) -> None:
        self.batch = batch
        self.saved: List[Any] = []
        self.loaded: List[Any] = []

    def save_task_blocks(self, group) -> int:
        return self.batch

    def load_task_blocks(self, group) -> int:
        return self.batch

    def submit_task(self, func, *args):
        bucket = self.saved if func.__name__ == "save_task" else self.loaded
        bucket.append(args)

    def save_task(self, *args):
        pass

    def load_task(self, *args):
        pass

    def create_save_done_callback(self, *args):
        return lambda flat: None

    def create_load_done_callback(self, *args, **kwargs):
        return lambda flat: None


class TestTaskChunkWiring(unittest.TestCase):
    """Both task loops must take their batch size from the manager.

    The transfer manager shrinks a task to its arena; a call site that
    hardcodes its own batch would silently over-stage (the PR2 review's N4).
    """

    def _worker(self, batch: int):
        group = windowed_group()
        conn = make_connector(manager_block_size=MBS)
        group_plain = AttentionTransferGroup(**{**group.__dict__, "retention": None})
        conn._kvcache_info = KVCacheInfo(
            tp_rank=0,
            world_size=1,
            groups=[group_plain],
            device="cpu",
            dtype=KV_DTYPE,
        )
        conn._data_transfer = _FakeBatchDTM(batch)
        conn._group_metas = [
            AttentionGroupMeta(
                group_idx=group_plain.group_idx,
                layer_names=list(group_plain.layer_names),
                block_size=MBS,
                per_block_bytes=group_plain.per_block_bytes,
                page_bytes=MBS * 8,
            )
        ]
        conn._state_group_idxs = []
        conn._num_groups = 1
        conn._device_mod = MagicMock()
        conn._pending_saves = {}
        conn._staged, conn._in_flight = {}, {}
        conn._withheld, conn._capture_scope = {}, {}
        conn._coordinator_server = MagicMock()
        return conn

    def test_save_uses_the_manager_batch(self):
        for batch in (1, 2):
            with self.subTest(batch=batch):
                conn = self._worker(batch)
                meta = TairKvCacheConnectorMetadata(epoch=0)
                meta.add_save_request(
                    SaveRequest(
                        req_id="r0",
                        target_locations=locations([["tp0_g0"]] * 4),
                        manager_block_idxes=[0, 1, 2, 3],
                        write_session_id="s0",
                        all_block_ids=[[1, 2, 3, 4]],
                    )
                )
                conn.wait_for_save(meta)
                self.assertEqual(4 // batch, len(conn._data_transfer.saved))

    def test_load_uses_the_manager_batch(self):
        for batch in (1, 4):
            with self.subTest(batch=batch):
                conn = self._worker(batch)
                meta = TairKvCacheConnectorMetadata(epoch=0)
                meta.add_load_request(
                    LoadRequest(
                        req_id="r0",
                        manager_block_idxes=[0, 1, 2, 3],
                        need_load_locations=locations([["tp0_g0"]] * 4),
                        all_block_ids=[[1, 2, 3, 4]],
                    )
                )
                conn.start_load_kv(None, meta)
                self.assertEqual(4 // batch, len(conn._data_transfer.loaded))


if __name__ == "__main__":
    unittest.main()
