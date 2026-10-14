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
    install_capture_ledger,
    make_connector,
    make_connector_scheduler,
)
from kv_cache_manager.py_connector.common.tp_coordinator import SaveContext
from kv_cache_manager.py_connector.vllm.connector_worker import (
    CAPTURE_POOL_WAIT_SECONDS,
    _Capture,
    _CaptureLedger,
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


def released_blocks(meta) -> list:
    """The (req_id, blocks) pairs a metadata carries, ignoring the URIs."""
    return [
        (req_id, blocks)
        for req_id, blocks, _uris, _confirmed in meta.to_release_captures
    ]


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
    install_capture_ledger(conn)
    # The verdict query talks to the coordinator (another process in
    # production): a bare worker cannot ask, so pending stays pending -- the
    # tests that care stub it explicitly.
    conn._coordinator_client = MagicMock()
    conn._coordinator_client.ask_verdict.return_value = None
    conn._coordinator_server = MagicMock()
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

    def test_a_packed_windowed_group_with_agreeing_coverage_splits(self):
        # Two sliding members with the *same* window and different page
        # geometry: their coverage is identical, so the one representative sub
        # spec vLLM folds this group to for the scheduler carries the same
        # retention, and the connector serves the group as its two buckets
        # (own wire name, own staging pool, own staged captures -- see
        # TestCaptureDispositions for the bookkeeping). A packed group whose
        # members *disagree* on coverage is the case the folded view cannot
        # describe; that one is refused by the folding guard, not here.
        from vllm.v1.kv_cache_interface import SlidingWindowSpec, UniformTypeKVCacheSpecs

        def spec(head_size):
            return SlidingWindowSpec(
                block_size=MBS,
                sliding_window=1024,
                page_size_bytes=2048,
                num_kv_heads=2,
                head_size=head_size,
                dtype=torch.bfloat16,
            )

        config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["l0", "l1"],
                    kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(
                        {"l0": spec(64), "l1": spec(128)}
                    ),
                    is_eagle_group=False,
                )
            ]
        )
        metas = parse_groups(config, MBS, calculate_kv_scales=False)
        self.assertEqual(2, len(metas))
        self.assertEqual({0}, {meta.group_idx for meta in metas})
        self.assertEqual(1, len({m.retention for m in metas}))
        self.assertEqual(RETENTION_SLIDING, metas[0].retention.kind)
        self.assertEqual(1024, metas[0].retention.window_tokens)
        self.assertEqual(2, len({spec_name(0, meta) for meta in metas}))

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

    def test_a_multi_block_capture_fits_the_shrunk_pool(self):
        # A pool a byte cap shrank to 62 slots holds 128-block task batches, so
        # the plan halves it: 31 slots for transfers, 31 for captures. One step
        # of a chunked prefill completes 16 blocks and the capture asks for
        # them as *one* contiguous run -- with the old one-slot reserve it would
        # time out on every step that completes more than one block, publishing
        # no windowed KV at all.
        group = windowed_group()
        pool = _StagingPool(
            torch.device("cpu"), group.per_block_bytes, 62, reserved_blocks=31
        )
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {group.spec_name: pool}
        dtm._manager_block_size = MBS
        dtm._device_mod = MagicMock()
        dtm._transfer_client = MagicMock()
        dtm._transfer_client.SaveKvCaches.return_value = [0]
        dtm._extra_config = SimpleNamespace(
            block_per_save_task=128, block_per_load_task=128
        )
        dtm._io_executor = ThreadPoolExecutor(max_workers=2)
        conn = worker_with([group], dtm)
        conn._pending_saves = {}
        conn._device_mod = MagicMock()
        with mock.patch(
            "kv_cache_manager.py_connector.vllm.data_transfer.batch_gather_scatter_helper"
        ):
            capture = conn._capture_request(
                CaptureRequest(
                    "r0",
                    list(range(16)),
                    all_block_ids=[list(range(7, 23))],
                )
            )
        self.assertTrue(capture, "the capture timed out on a 16-block step")
        self.assertEqual(16, len(conn._staged["r0"]))
        # And a save task publishes them out of the arena.
        plans = []
        real_plan = conn._plan_group_transfers

        def spy(*args, **kwargs):
            result = real_plan(*args, **kwargs)
            if kwargs.get("for_save"):
                plans.append(result)
            return result

        conn._plan_group_transfers = spy
        meta = TairKvCacheConnectorMetadata(epoch=0)
        meta.add_save_request(
            SaveRequest(
                req_id="r0",
                target_locations=locations([["tp0_g0"]] * 16),
                manager_block_idxes=list(range(16)),
                write_session_id="s0",
                all_block_ids=[list(range(7, 23))],
            )
        )
        conn.wait_for_save(meta)
        self.assertEqual(16, len(plans[0][0].staged_slots))
        self.assertNotIn(None, plans[0][0].staged_slots)
        # The settle hands the session to the verdict recheck, which returns
        # the slots once the coordinator's merged AND says committed.
        conn._captures.settle("r0", 0, {("tp0_g0", b): True for b in range(16)})
        conn._coordinator_client.ask_verdict.return_value = [True] * 16
        conn._verdict_query_down_until = 0.0
        conn.wait_for_save(TairKvCacheConnectorMetadata(epoch=0))
        deadline = time.time() + 5
        while pool.free_blocks != 62 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(62, pool.free_blocks)

    def test_a_stale_block_table_still_saves_staged_window_bytes(self):
        # Preemption + resume: the save session's manager blocks name the
        # original prefix (whose bytes the eager capture holds), while the
        # block table snapshot is the resume's, shorter allocation. A staged
        # (windowed) save reads the arena, not the table, so the plan must be
        # built -- translating those blocks would hit the range check and abort
        # a save that needs nothing from the table.
        group = windowed_group()
        pool = _StagingPool(torch.device("cpu"), group.per_block_bytes, 64)
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {group.spec_name: pool}
        dtm._manager_block_size = MBS
        dtm._device_mod = MagicMock()
        dtm._transfer_client = MagicMock()
        dtm._transfer_client.SaveKvCaches.return_value = [0]
        dtm._extra_config = SimpleNamespace(
            block_per_save_task=128, block_per_load_task=128
        )
        dtm._io_executor = ThreadPoolExecutor(max_workers=2)
        conn = worker_with([group], dtm)
        conn._pending_saves = {}
        conn._device_mod = MagicMock()
        with mock.patch(
            "kv_cache_manager.py_connector.vllm.data_transfer.batch_gather_scatter_helper"
        ):
            conn._capture_request(
                CaptureRequest(
                    "r0", list(range(16)), all_block_ids=[list(range(7, 23))]
                )
            )
        # The resume's table addresses four blocks only.
        plans = conn._plan_group_transfers(
            locations([["tp0_g0"]] * 16),
            list(range(16)),
            [list(range(7, 11))],
            "r0",
            for_save=True,
        )
        self.assertEqual(16, len(plans[0].staged_slots))
        self.assertNotIn(None, plans[0].staged_slots)

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
        self.assertEqual([], released_blocks(meta))
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
        # A's write landed and (one rank) committed: its slots come back,
        # while B's 16 blocks (2 buckets) are still captured for the session
        # that has not run yet.
        for spec in ("tp0_g0", "tp0_g1"):
            conn._captures.settle("r0", 0, {(spec, b): True for b in range(16)})
        conn._coordinator_client.ask_verdict.return_value = [True] * 16
        conn._verdict_query_down_until = 0.0
        conn.wait_for_save(TairKvCacheConnectorMetadata(epoch=0))
        deadline = time.time() + 5
        while pool.free_blocks != 96 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(96, pool.free_blocks)

    def test_a_repeated_target_session_does_not_release_the_other(self):
        # A regressed save watermark can announce the same target twice while
        # the first session is still in its http call. The two sessions need
        # distinct identities: keyed by their range alone, the second would
        # replace the first and then be skipped as "the answering session",
        # and its all-existing response would release captures the first
        # session still has to write.
        sched = make_connector_scheduler(
            manager_block_size=MBS, num_groups=1, windowed=1
        )
        sched._extra_config = SimpleNamespace(
            instance_id="inst", write_timeout_seconds=1
        )
        first = sched._record_announcement("r0", 16)
        second = sched._record_announcement("r0", 16)
        self.assertNotEqual(first, second)
        sched._manager_client.start_write_cache.return_value = {
            "locations": [],
            "write_session_id": "s2",
            "block_mask": {"offset": 16},  # everything already has a location
        }
        sched.start_save_kvcache_async(
            "r0", list(range(16 * MBS)), 16, [True] * 16, [True] * 16, second
        )
        meta = TairKvCacheConnectorMetadata(epoch=0)
        sched._collect_release_captures(meta)
        self.assertEqual([], released_blocks(meta))

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
        self.assertEqual([], released_blocks(meta))

    def test_a_batch_release_waits_for_the_save_plans(self):
        # A release and a session's SaveRequest can ride in one batch (the
        # release was queued for blocks an older session left behind). The
        # plan of this batch must take its captures first: applied earlier, the
        # release would withdraw the bytes the save is about to write, so the
        # save fails the blocks and cannot even keep them for a retry.
        group = windowed_group()
        pool = _StagingPool(torch.device("cpu"), group.per_block_bytes, 32)
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {group.spec_name: pool}
        dtm._manager_block_size = MBS
        dtm._device_mod = MagicMock()
        dtm._transfer_client = MagicMock()
        dtm._transfer_client.SaveKvCaches.return_value = [0]
        dtm._extra_config = SimpleNamespace(
            block_per_save_task=128, block_per_load_task=128
        )
        dtm._io_executor = ThreadPoolExecutor(max_workers=2)
        conn = worker_with([group], dtm)
        conn._pending_saves = {}
        conn._device_mod = MagicMock()

        blocks = list(range(16))
        meta = TairKvCacheConnectorMetadata(epoch=0)
        meta.add_capture_request(
            CaptureRequest("r0", blocks, all_block_ids=[list(range(7, 23))])
        )
        meta.release_captures("r0", blocks)
        meta.add_save_request(
            SaveRequest(
                req_id="r0",
                write_session_id="sA",
                manager_block_idxes=blocks,
                all_block_ids=[list(range(7, 23))],
                target_locations=[
                    {
                        "block_idxes": [block],
                        "location_specs": [{"name": "tp0_g0", "uri": "a%d" % block}],
                    }
                    for block in blocks
                ],
                session_id=7,
            )
        )
        with mock.patch(
            "kv_cache_manager.py_connector.vllm.data_transfer.batch_gather_scatter_helper"
        ):
            conn.wait_for_save(meta)
        # The plan took every block before the release was applied (whether
        # its own commit then released them or not is the settle's business).
        self.assertEqual(
            16,
            len(conn._in_flight.get("r0", {})) + len(conn._staged.get("r0", {})),
        )

    def test_a_handed_over_session_stops_protecting_its_blocks(self):
        # A session whose plan is in the worker's hands no longer protects its
        # blocks: its plan runs before any release of the same step, and a
        # release computed while it was still in flight already excluded them.
        # Without that retirement a long request with repeated preemptions
        # pins the shared arena for its whole life (the blocks it recaptures
        # after resuming can never be released while an old session's plan
        # still "might" write them).
        sched = make_connector_scheduler(
            manager_block_size=MBS, num_groups=1, windowed=1
        )
        sched._extra_config = SimpleNamespace(
            instance_id="inst", write_timeout_seconds=1
        )
        sched._tracked["r0"] = RequestLedger(
            vllm_request=FakeRequest("r0", []),  # ty: ignore[invalid-argument-type]
            block_ids_per_group=[list(range(16))],
            has_saved_block_num=0,
        )
        sched._manager_client.start_write_cache.return_value = {
            "locations": [
                {
                    "block_idxes": [block],
                    "location_specs": [{"name": "tp0_g0", "uri": "a%d" % block}],
                }
                for block in range(16)
            ],
            "write_session_id": "sA",
            "block_mask": {"offset": 0},
        }
        session_a = sched._record_announcement("r0", 16)
        sched.start_save_kvcache_async(
            "r0", list(range(16 * MBS)), 16, [True] * 16, [True] * 16, session_a
        )
        # A's plan reaches the worker (the scheduler loop's flush point).
        meta = TairKvCacheConnectorMetadata(epoch=0)
        sched._collect_save_instructions(meta)
        self.assertEqual(1, len(meta.to_save_requests))

        # A later session gets an all-existing response for A's blocks: those
        # captures are dead and must be released now.
        sched._manager_client.start_write_cache.return_value = {
            "locations": [],
            "write_session_id": "sB",
            "block_mask": {"offset": 16},
        }
        session_b = sched._record_announcement("r0", 32)
        sched.start_save_kvcache_async(
            "r0", list(range(32 * MBS)), 32, [True] * 32, [True] * 32, session_b
        )
        release = TairKvCacheConnectorMetadata(epoch=0)
        sched._collect_release_captures(release)
        # The whole announced prefix is releasable: A's session is in the
        # worker's hands and B writes nothing at all. While A still protected
        # its blocks, 0-15 would be held back and only 16-31 released.
        self.assertEqual([("r0", list(range(32)))], released_blocks(release))

    def test_a_failed_announcement_releases_its_unprotected_captures(self):
        # start_write_cache raising means this session allocates and writes
        # nothing, and the save watermark has already moved past its blocks --
        # an active request does not announce them again, so without a release
        # here every transient manager failure pins its captures until the
        # request retires.
        sched = make_connector_scheduler(
            manager_block_size=MBS, num_groups=1, windowed=1
        )
        sched._extra_config = SimpleNamespace(
            instance_id="inst", write_timeout_seconds=1
        )
        sched._manager_client.start_write_cache.side_effect = RuntimeError("boom")
        session = sched._record_announcement("r0", 16)
        sched.start_save_kvcache_async(
            "r0", list(range(16 * MBS)), 16, [True] * 16, [True] * 16, session
        )
        release = TairKvCacheConnectorMetadata(epoch=0)
        sched._collect_release_captures(release)
        self.assertEqual([("r0", list(range(16)))], released_blocks(release))

        # ... but a concurrent session that may still write those blocks keeps
        # them, exactly like the all-existing path.
        protecting = sched._record_announcement("r0", 16)
        session2 = sched._record_announcement("r0", 16)
        sched.start_save_kvcache_async(
            "r0", list(range(16 * MBS)), 16, [True] * 16, [True] * 16, session2
        )
        self.assertNotEqual(protecting, session2)
        later = TairKvCacheConnectorMetadata(epoch=0)
        sched._collect_release_captures(later)
        self.assertEqual([], released_blocks(later))

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
        self.assertEqual(released_blocks(meta), [("r0", [0, 1])])

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
        self.assertEqual(released_blocks(meta), [("r0", [0, 1, 2])])

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


    def test_a_recaptured_block_promotes_its_remembered_claim(self):
        # The block was first announced after its window had moved on
        # (claimed=False) and the announcement is remembered; preemption later
        # recomputed it and the eager capture holds its bytes. The memory must
        # follow: a retry after this write fails has to announce the windowed
        # spec, otherwise the staged bytes can never be published.
        sched = make_connector_scheduler(
            manager_block_size=MBS, num_groups=1, windowed=1
        )
        sched._window_claims["r0"] = {0: False}
        announced = sched._announced_window_mask("r0", [True])
        self.assertEqual([True], announced)
        self.assertEqual({0: True}, sched._window_claims["r0"])
        # The other direction still holds: a claimed block keeps its coverage
        # when the window moves away.
        self.assertEqual([True], sched._announced_window_mask("r0", [False]))

    def test_window_claims_are_serialized(self):
        # Overlapping start_save_kvcache_async calls for one request run on
        # different http-executor threads: the claim is a read-modify-write,
        # and a lost update can leave "no window" behind bytes that are staged
        # (no retry could ever announce them). One transition must hold the
        # sequence against the others.
        import threading

        sched = make_connector_scheduler(
            manager_block_size=MBS, num_groups=1, windowed=1
        )

        class _SlowClaims(dict):
            def __init__(self) -> None:
                super().__init__()
                self.entered = threading.Event()
                self.proceed = threading.Event()
                self.slow = True

            def get(self, key, *args):
                if self.slow:
                    self.slow = False
                    self.entered.set()
                    self.proceed.wait(timeout=5)
                return super().get(key, *args)

        sched._window_claims = {"r0": _SlowClaims()}
        other_done = threading.Event()

        first = threading.Thread(
            target=lambda: sched._announced_window_mask("r0", [True])
        )
        second = threading.Thread(
            target=lambda: (
                sched._announced_window_mask("r0", [True]),
                other_done.set(),
            )
        )
        first.start()
        self.assertTrue(
            sched._window_claims["r0"].entered.wait(timeout=5),
            "the first update never reached the claim read",
        )
        second.start()
        self.assertFalse(
            other_done.wait(timeout=0.3),
            "a claim update ran while another held the sequence",
        )
        sched._window_claims["r0"].proceed.set()
        first.join(timeout=10)
        second.join(timeout=10)
        self.assertTrue(other_done.is_set())
        self.assertEqual({0: True}, sched._window_claims["r0"])

    def test_a_queued_release_is_rechecked_against_newer_sessions(self):
        # A failed announcement queues a release for its blocks; a session
        # created afterwards re-announces and recaptures the same block. The
        # flush must drop the stale release, or the worker would capture the
        # fresh bytes and immediately discard them.
        sched = make_connector_scheduler(
            manager_block_size=MBS, num_groups=1, windowed=1
        )
        sched._extra_config = SimpleNamespace(
            instance_id="inst", write_timeout_seconds=1
        )
        first = sched._record_announcement("r0", 16)
        sched._remember_session_plan("r0", first, set())
        sched._queue_capture_release("r0", first, 16)
        meta = TairKvCacheConnectorMetadata(epoch=0)
        sched._collect_release_captures(meta)
        self.assertEqual([("r0", list(range(16)))], released_blocks(meta))

        # Now the release is queued and a *newer* session re-announces and
        # recaptures blocks 0-1 before the flush: their captures must survive.
        sched._queue_capture_release("r0", first, 16)
        second = sched._record_announcement("r0", 16)
        sched._remember_session_plan("r0", second, {0, 1})
        stale = TairKvCacheConnectorMetadata(epoch=0)
        sched._collect_release_captures(stale)
        self.assertEqual([("r0", list(range(2, 16)))], released_blocks(stale))


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
        # Every slot went back to the arena it came from, exactly once -- via
        # the capture ledger's settle, which is the only release point.
        conn._captures.settle("r0", 0, {("tp0_g0", b): True for b in (0, 1)})
        conn._captures.settle("r0", 0, {("tp0_g0_b1", b): True for b in (0, 1)})
        # Local success retains (a block is committed only when every rank
        # wrote it): the confirmation releases both arenas' slots (one entry
        # per bucket and block).
        self.assertEqual(4, len(conn._staged["r0"]))
        conn._captures.release_staged("r0", [0, 1])
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
            [0, 1], sorted(entry.slot for entry in conn._staged["r0"].values())
        )
        conn._capture_request(CaptureRequest("r0", [0, 1], all_block_ids=[[7, 8]]))
        # The repeat holds its new run and the displaced one is already back:
        # arena use is one run, not two.
        self.assertEqual(6, pool.free_blocks)
        self.assertEqual(
            [2, 3], sorted(entry.slot for entry in conn._staged["r0"].values())
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
        meta.release_captures("r0", [1], {1: ["uri-1"]})
        conn.wait_for_save(meta)
        self.assertEqual([("tp0_g0", 0)], sorted(conn._staged["r0"]))

    def test_release_captures_returns_only_the_listed_blocks(self):
        conn = captured_worker([0, 1], [7, 8])
        conn._release_captures_for("r0", [0], {0: ["uri-0"]})
        # Block 0's slot went back with the instruction; block 1's capture is
        # still there for the session that will write it.
        self.assertEqual(
            [("tp0_g0", 1)], sorted(conn._staged["r0"]),
        )
        self.assertEqual([(0, 1)], conn._data_transfer.pool.released)

    def test_a_release_without_a_cache_key_is_deferred(self):
        # The manager returns no location for a deduplicated block (its answer
        # only marks it "already"), so the release has no cache key for it and
        # cannot tell whether another request holds an uncommitted allocation
        # under that key. Dropping the bytes could leave that writer -- and any
        # retry of ours -- with nothing, so the release waits for the commit
        # verdict (or retirement) instead.
        conn = captured_worker([0, 1], [7, 8])
        conn._release_captures_for("r0", [0])
        self.assertEqual(
            [("tp0_g0", 0), ("tp0_g0", 1)],
            sorted(conn._staged["r0"]),
            "a keyless release dropped the bytes",
        )
        # ... while a confirmed release needs no key: the block is durable.
        conn._release_captures_for("r0", [0], confirmed=True)
        self.assertEqual([("tp0_g0", 1)], sorted(conn._staged["r0"]))

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
        install_capture_ledger(conn)
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
        # The slots are the capture ledger's to return, not this task's: they
        # are still held here, and the ledger releases them on the session's
        # settle (see the invariant family).
        self.assertEqual(2, pool.free_blocks)
        pool.release(start, 2)
        self.assertEqual(4, pool.free_blocks)

    def test_failed_write_holds_its_slots_for_the_retry(self):
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
        # A failed transfer keeps its slots: the worker hands the bytes back to
        # its staged map, so returning them here would let another capture take
        # them and leave the retry with nothing to publish.
        self.assertIsNone(pool.acquire(1, timeout=0.01))
        # A *raise* is the exception (no result callback will fire for the
        # task), so those slots do come back and cannot leak in flight.
        dtm._transfer_client.SaveKvCaches.side_effect = RuntimeError("sdk")
        dtm.save_task(
            MultiResult(1, lambda flat: None),
            0,
            windowed_group(),
            remote_uris=["u2", "u3"],
            block_token_indices=None,
            block_ids=None,
            staged_slots=[start, start + 1],
        )
        # A failure keeps its slots (the ledger hands the bytes back to the
        # staged map for the retry), so this second task cannot take them --
        # and the release that used to happen here is exactly what the ledger
        # now owns: it happens once, on the session's settle.
        self.assertIsNone(pool.acquire(1, timeout=0.01))
        pool.release(start, 2)
        self.assertEqual(2, pool.free_blocks)
        # ... exactly once: the pool refuses a second release of the same slots.
        with self.assertRaises(ValueError):
            pool.release(start, 2)

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
        # The written block's slot is the ledger's to return (see the
        # invariant family); this task returns nothing, so the arena still
        # holds exactly that one slot.
        self.assertEqual(1, pool.free_blocks)
        second = pool.acquire(1, timeout=0.01)
        self.assertIsNotNone(second)
        pool.release(second, 1)
        pool.release(start, 1)
        self.assertEqual(2, pool.free_blocks)


class TestCaptureSlotInvariant(unittest.TestCase):
    """The capture ledger's invariant -- and the three shapes that broke it.

    At any moment every staging slot belongs to exactly one live entry in
    exactly one of the ledger's two maps, it is released exactly once, and no
    path ever reads a released slot. Each test below pins one of the holes the
    audit found and checks the same invariant, so a regression fails as an
    accounting error rather than as a corrupted transfer.
    """

    CAPACITY = 8
    BLOCKS = 4

    def _fixture(self, ok: bool = True):
        import kv_cache_manager.py_connector.vllm.data_transfer as dt

        patcher = mock.patch.object(dt, "batch_gather_scatter_helper")
        patcher.start()
        self.addCleanup(patcher.stop)
        group = windowed_group()
        pool = _StagingPool(
            torch.device("cpu"), group.per_block_bytes, self.CAPACITY, reserved_blocks=2
        )
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {group.spec_name: pool}
        dtm._manager_block_size = MBS
        dtm._device_mod = MagicMock()
        dtm._transfer_client = MagicMock()
        dtm._transfer_client.SaveKvCaches.return_value = [0 if ok else 1]
        dtm._extra_config = SimpleNamespace(
            block_per_save_task=2, block_per_load_task=2
        )
        dtm._io_executor = ThreadPoolExecutor(max_workers=2)
        conn = worker_with([group], dtm)
        conn._pending_saves = {}
        conn._device_mod = MagicMock()
        return conn, dtm, pool, group

    def _invariant(self, conn, pool, note: str = ""):
        """Every slot belongs to one live entry, and the pool agrees."""
        entries = [
            entry
            for entries in conn._staged.values()
            for entry in entries.values()
        ] + [
            entry
            for entries in conn._in_flight.values()
            for entry in entries.values()
        ]
        slots = [entry.slot for entry in entries]
        self.assertEqual(len(slots), len(set(slots)), f"a slot has two owners {note}")
        for slot in slots:
            self.assertGreaterEqual(slot, 0, note)
            self.assertLess(slot, pool.max_blocks, note)
        self.assertEqual(
            pool.max_blocks,
            pool.free_blocks + len(slots),
            f"a slot is neither free nor owned (released twice or leaked) {note}",
        )

    def _plan(self, conn, blocks, session):
        return conn._plan_group_transfers(
            locations([["tp0_g0"]] * len(blocks)),
            blocks,
            [list(range(7, 7 + len(blocks)))],
            "r0",
            for_save=True,
            session_id=session,
        )[0]

    def _run_session(self, conn, dtm, group, plan, blocks, session):
        done = {}
        done_cb = conn._settle_session_captures(
            SimpleNamespace(
                req_id="r0", manager_block_idxes=list(blocks), session_id=session
            ),
            [plan],
            lambda flat: done.setdefault("flat", flat),
        )
        dtm.save_task(
            MultiResult(1, done_cb),
            0,
            group,
            plan.uris,
            None,
            None,
            plan.staged_slots,
        )
        return done.get("flat")

    def test_an_sdk_exception_is_not_restored_onto_a_released_slot(self):
        # The exception path used to release the slots and report a failure,
        # which the worker's retain wrapper then restored -- onto slots the
        # pool had already handed out again. The ledger owns both ends now: an
        # exception is a failure, the bytes stay staged, and nothing was
        # released behind its back.
        conn, dtm, pool, group = self._fixture()
        blocks = list(range(self.BLOCKS))
        conn._capture_request(CaptureRequest("r0", blocks, all_block_ids=[[7, 8, 9, 10]]))
        plan = self._plan(conn, blocks, session=1)
        dtm._transfer_client.SaveKvCaches.side_effect = RuntimeError("pybind blew up")
        flat = self._run_session(conn, dtm, group, plan, blocks, session=1)
        self.assertEqual([False] * self.BLOCKS, flat)
        self._invariant(conn, pool, "after an SDK exception")
        self.assertEqual(self.BLOCKS, len(conn._staged["r0"]))
        self.assertEqual({}, conn._in_flight.get("r0", {}))

        # The retry publishes from those very slots, and then releases them.
        dtm._transfer_client.SaveKvCaches.side_effect = None
        retry = self._plan(conn, blocks, session=2)
        self.assertNotIn(None, retry.staged_slots)
        self._run_session(conn, dtm, group, retry, blocks, session=2)
        self._invariant(conn, pool, "after the retry")
        # The retry's local success keeps the bytes until the commit is known
        # (see the ledger's settle): the confirmation releases them.
        conn._captures.release_staged("r0", blocks)
        self._invariant(conn, pool, "after the confirmation")
        self.assertEqual(self.CAPACITY, pool.free_blocks)

    def test_a_repeat_capture_keeps_both_sessions_ownership(self):
        # The save watermark moved backwards, so the block was captured again
        # while the first session's write was still in flight. Both owners must
        # survive: the second session's failure may only move its own entry,
        # and the first session's bytes must not be released or overwritten.
        conn, dtm, pool, group = self._fixture()
        conn._capture_request(CaptureRequest("r0", [0], all_block_ids=[[7]]))
        first = self._plan(conn, [0], session=1)
        self.assertIsNotNone(first.staged_slots[0])
        conn._capture_request(CaptureRequest("r0", [0], all_block_ids=[[7]]))
        self._invariant(conn, pool, "after the repeat capture")
        second = self._plan(conn, [0], session=2)
        self.assertIsNotNone(second.staged_slots[0])
        self.assertNotEqual(first.staged_slots[0], second.staged_slots[0])
        self.assertEqual(2, len(conn._in_flight["r0"]))
        self._invariant(conn, pool, "with two sessions in flight")

        # Session 2 fails: its entry goes back to staging, session 1's slot is
        # untouched and still in flight.
        dtm._transfer_client.SaveKvCaches.return_value = [1]
        flat = self._run_session(conn, dtm, group, second, [0], session=2)
        self.assertEqual([False], flat)
        self._invariant(conn, pool, "after session 2 failed")
        # Session 1's record is the one still in flight; session 2's settled
        # entry went back to staging (with its own slot).
        self.assertEqual([(1, "tp0_g0", 0)], list(conn._in_flight["r0"]))
        self.assertEqual([second.staged_slots[0]],
                         [entry.slot for entry in conn._staged["r0"].values()])

    def test_a_retired_request_settles_its_in_flight_slots(self):
        # The request retired before its task settled. The in-flight record has
        # to survive that (the ledger keeps it) so the failure can release the
        # slot: popping it at retirement is what leaked these slots for good.
        conn, dtm, pool, group = self._fixture()
        blocks = list(range(self.BLOCKS))
        conn._capture_request(CaptureRequest("r0", blocks, all_block_ids=[[7, 8, 9, 10]]))
        plan = self._plan(conn, blocks, session=1)
        conn._release_staged("r0")
        self._invariant(conn, pool, "after retirement, task still in flight")

        dtm._transfer_client.SaveKvCaches.return_value = [1]
        flat = self._run_session(conn, dtm, group, plan, blocks, session=1)
        self.assertEqual([False] * self.BLOCKS, flat)
        self._invariant(conn, pool, "after the retired request settled")
        self.assertEqual(self.CAPACITY, pool.free_blocks)
        self.assertEqual({}, conn._staged.get("r0", {}))
        self.assertEqual({}, conn._in_flight.get("r0", {}))

    def test_a_withheld_capture_does_not_hold_the_arena(self):
        # The plain path still holds: an arena too small for this step's runs
        # withholds the blocks (nothing staged, nothing in flight, nothing
        # released), and the invariant is what says so.
        conn, dtm, pool, group = self._fixture()
        # Take the whole capture partition (capacity - reserve): the two-block
        # run cannot fit, so the capture is withheld rather than queued.
        held = pool.acquire(self.CAPACITY - pool.reserved_blocks, keep_reserved=True)
        self.assertIsNotNone(held)
        self.assertFalse(
            conn._capture_request(
                CaptureRequest("r0", [0, 1], all_block_ids=[[7, 8]])
            )
        )
        # The rollback gave back whatever the failed run had taken; releasing
        # the test's own hold leaves the arena whole again.
        pool.release(held, self.CAPACITY - pool.reserved_blocks)
        self._invariant(conn, pool, "after a withheld capture")
        self.assertEqual(self.CAPACITY, pool.free_blocks)



    def test_concurrent_failed_settles_of_one_block_keep_one_owner(self):
        # Two sessions hold the same (spec, block) and both fail at once: each
        # restages its own entry, so the settle must not overwrite the other's
        # (one slot would be held by the pool with no entry left to release
        # it). The barrier lines the two threads up on the decision.
        import threading

        ledger = _CaptureLedger()
        pool = _StagingPool(torch.device("cpu"), 1024, 8)
        first_e = [pool.acquire(1)]
        first = _Capture(pool=pool, slot=first_e[0])
        second = _Capture(pool=pool, slot=pool.acquire(1))
        ledger.stage("r0", ("tp0_g0", 0), first)
        ledger.take("r0", 1, ("tp0_g0", 0))
        ledger.stage("r0", ("tp0_g0", 0), second)
        ledger.take("r0", 2, ("tp0_g0", 0))

        barrier = threading.Barrier(2)
        errors: List[BaseException] = []

        def settle(session: int) -> None:
            try:
                barrier.wait(timeout=5)
                ledger.settle("r0", session, {("tp0_g0", 0): False})
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=settle, args=(session,)) for session in (1, 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual([], errors)
        staged = list(ledger.staged["r0"].values())
        self.assertEqual(1, len(staged), "one of the two settled entries was lost")
        self.assertEqual({}, ledger.in_flight.get("r0", {}))
        self.assertEqual(8, pool.free_blocks + 1, "a slot is neither free nor owned")

    def test_one_transition_holds_the_ledger_against_the_others(self):
        # The lock is what makes an ownership decision atomic: the release a
        # settle performs happens with the ledger held, so a concurrent settle
        # cannot read a half-updated map. A slow release stands in for the
        # window: with the lock the second settle waits it out, without it the
        # second settle runs to completion inside it.
        import threading

        ledger = _CaptureLedger()
        pool = _StagingPool(torch.device("cpu"), 1024, 8)
        first_e = [pool.acquire(1)]
        first = _Capture(pool=pool, slot=first_e[0])
        second = _Capture(pool=pool, slot=pool.acquire(1))
        ledger.stage("r0", ("tp0_g0", 0), first)
        ledger.take("r0", 1, ("tp0_g0", 0))
        ledger.stage("r0", ("tp0_g0", 1), second)
        ledger.take("r0", 2, ("tp0_g0", 1))

        entered = threading.Event()
        proceed = threading.Event()
        real_release = _CaptureLedger._release

        def slow_release(capture):
            # Only the first settler's release is slow: the point is what the
            # *other* transition is allowed to do while it runs.
            if capture.slot == first_e[0]:
                entered.set()
                proceed.wait(timeout=5)
            real_release(capture)

        ledger._release = slow_release  # ty: ignore[method-assign]
        second_done = threading.Event()

        # A retired request's settle releases inside the lock (that is the
        # release path these two transitions contend on).
        ledger.retire("r0")

        def first_settle():
            ledger.settle("r0", 1, {("tp0_g0", 0): True})

        def second_settle():
            ledger.settle("r0", 2, {("tp0_g0", 1): True})
            second_done.set()

        one = threading.Thread(target=first_settle)
        two = threading.Thread(target=second_settle)
        one.start()
        self.assertTrue(entered.wait(timeout=5), "the first settle never released")
        two.start()
        self.assertFalse(
            second_done.wait(timeout=0.3),
            "a settle ran while another transition held the ledger",
        )
        proceed.set()
        one.join(timeout=10)
        two.join(timeout=10)
        self.assertTrue(second_done.is_set())
        self.assertEqual(8, pool.free_blocks)
        self.assertEqual({}, ledger.in_flight.get("r0", {}))

    def test_a_retire_racing_a_failed_settle_leaves_nothing_behind(self):
        # A retirement and a failure callback can run at the same time. Whichever
        # wins, the invariant holds: the request's staged entries are released,
        # and a settle that lands after the retirement releases its slot too.
        import threading

        for _round in range(50):
            ledger = _CaptureLedger()
            pool = _StagingPool(torch.device("cpu"), 1024, 8)
            staged = _Capture(pool=pool, slot=pool.acquire(1))
            flight = _Capture(pool=pool, slot=pool.acquire(1))
            ledger.stage("r0", ("tp0_g0", 0), staged)
            ledger.stage("r0", ("tp0_g0", 1), flight)
            ledger.take("r0", 1, ("tp0_g0", 1))

            barrier = threading.Barrier(2)
            errors: List[BaseException] = []

            def retire() -> None:
                try:
                    barrier.wait(timeout=5)
                    ledger.retire("r0")
                except BaseException as exc:  # pragma: no cover - failure path
                    errors.append(exc)

            def settle() -> None:
                try:
                    barrier.wait(timeout=5)
                    ledger.settle("r0", 1, {("tp0_g0", 1): False})
                except BaseException as exc:  # pragma: no cover - failure path
                    errors.append(exc)

            threads = [threading.Thread(target=retire), threading.Thread(target=settle)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            self.assertEqual([], errors)
            owned = sum(len(entries) for entries in ledger.staged.values()) + sum(
                len(entries) for entries in ledger.in_flight.values()
            )
            self.assertEqual(8, pool.free_blocks + owned, f"round {_round}")

class TestCommitConfirmation(unittest.TestCase):
    """The confirmation loop: local success -> wait -> confirmation -> release.

    The first wave8 attempt retained every locally-successful capture and had
    no working confirmation in the real flow: the gemma-3-4b run filled the
    arena within one prefill burst (10 sessions x 96 blocks > 896 slots) and
    withheld every capture afterwards. These tests pin both ends: the loop
    closes (a single rank confirms itself at the settle, several ranks at the
    coordinator's AND), and a session whose confirmation never arrives must
    not strand its slots past the request's life.
    """

    CAPACITY = 16
    BLOCKS = 4

    def _fixture(self, tp_size: int):
        import kv_cache_manager.py_connector.vllm.data_transfer as dt

        patcher = mock.patch.object(dt, "batch_gather_scatter_helper")
        patcher.start()
        self.addCleanup(patcher.stop)
        group = windowed_group()
        pool = _StagingPool(
            torch.device("cpu"), group.per_block_bytes, self.CAPACITY
        )
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {group.spec_name: pool}
        dtm._manager_block_size = MBS
        dtm._device_mod = MagicMock()
        dtm._transfer_client = MagicMock()
        dtm._transfer_client.SaveKvCaches.return_value = [0]
        dtm._extra_config = SimpleNamespace(
            block_per_save_task=8, block_per_load_task=8
        )
        dtm._io_executor = ThreadPoolExecutor(max_workers=2)
        conn = worker_with([group], dtm)
        conn._pending_saves = {}
        conn._device_mod = MagicMock()
        conn._tp_size = tp_size
        conn._manager_client = MagicMock()
        conn._extra_config = SimpleNamespace(instance_id="test")
        return conn, dtm, pool, group

    def _run(self, conn, dtm, group, session, write_session_id):
        blocks = list(range(self.BLOCKS))
        meta = TairKvCacheConnectorMetadata(epoch=0)
        meta.add_capture_request(
            CaptureRequest("r0", blocks, all_block_ids=[list(range(7, 11))])
        )
        meta.add_save_request(
            SaveRequest(
                req_id="r0",
                write_session_id=write_session_id,
                manager_block_idxes=blocks,
                all_block_ids=[list(range(7, 11))],
                target_locations=[
                    {
                        "block_idxes": [block],
                        "location_specs": [{"name": "tp0_g0", "uri": "u%d" % block}],
                    }
                    for block in blocks
                ],
                session_id=session,
            )
        )
        conn.wait_for_save(meta)
        # The tasks report on the io threads: wait for the settle to land.
        deadline = time.time() + 5
        while conn._in_flight.get("r0") and time.time() < deadline:
            time.sleep(0.01)
        return blocks

    def test_a_non_windowed_group_failure_keeps_the_window_bytes(self):
        # Mixed windowed/full-attention (or windowed/state) instance, one rank:
        # the settle's outcomes only describe the windowed plans, so releasing
        # there would drop a block whose full-attention group failed -- the
        # manager then rejects that block and a retry can never rewrite its
        # window specs (the HBM window moved). The merged, all-group verdict is
        # the only sound basis.
        conn, dtm, pool, group = self._fixture(tp_size=1)
        conn._coordinator_client.ask_verdict.return_value = None
        blocks = self._run(conn, dtm, group, session=1, write_session_id="sA")
        verdict = [True] * self.BLOCKS
        verdict[1] = False
        conn._coordinator_client.ask_verdict.return_value = verdict
        conn._verdict_query_down_until = 0.0
        conn.wait_for_save(TairKvCacheConnectorMetadata(epoch=0))
        self.assertEqual(
            [("tp0_g0", 1)],
            sorted(conn._staged.get("r0", {})),
            "a block failed by another group lost its windowed bytes",
        )

    def test_one_rank_confirms_through_the_recheck(self):
        # A single rank cannot release at the settle either: its `outcomes`
        # only cover the windowed plans, and a full-attention or state group
        # failing would let the manager reject a block whose window bytes are
        # already gone. The coordinator's merged, all-group AND (it runs in
        # this process) is what releases, within a step.
        conn, dtm, pool, group = self._fixture(tp_size=1)
        conn._coordinator_client.ask_verdict.return_value = None
        blocks = self._run(conn, dtm, group, session=1, write_session_id="sA")
        self.assertEqual(self.BLOCKS, len(conn._staged.get("r0", {})))
        conn._coordinator_client.ask_verdict.return_value = [True] * self.BLOCKS
        conn._verdict_query_down_until = 0.0
        conn.wait_for_save(TairKvCacheConnectorMetadata(epoch=0))
        self.assertEqual(
            self.CAPACITY,
            pool.free_blocks,
            "a single rank's confirmed capture was not released",
        )
        self.assertEqual({}, conn._staged.get("r0", {}))
        self.assertEqual({}, conn._in_flight.get("r0", {}))

    def test_the_coordinator_confirms_a_multi_rank_session(self):
        # Two ranks: the local verdict is not enough, and the coordinator's AND
        # (which arrives after every rank reported) is what releases. Until it
        # does, the bytes stay available for the retry a rejection implies.
        conn, dtm, pool, group = self._fixture(tp_size=2)
        blocks = self._run(conn, dtm, group, session=1, write_session_id="sA")
        self.assertEqual(self.BLOCKS, len(conn._staged.get("r0", {})))
        self.assertEqual(self.CAPACITY - self.BLOCKS, pool.free_blocks)

        context = SaveContext(
            locations=[{"block_idxes": [block]} for block in blocks],
            result_per_rank={0: [True] * self.BLOCKS, 1: [True] * self.BLOCKS},
        )
        conn.on_save_finished("sA", context)
        conn._coordinator_client.ask_verdict.return_value = [True] * self.BLOCKS
        conn._verdict_query_down_until = 0.0
        conn.wait_for_save(TairKvCacheConnectorMetadata(epoch=0))
        self.assertEqual({}, conn._staged.get("r0", {}))
        self.assertEqual(self.CAPACITY, pool.free_blocks)

    def test_a_rejected_rank_keeps_the_bytes_for_the_retry(self):
        # rank 1 failed: the block is rejected globally, and rank 0's bytes are
        # the only copy left (its HBM window is gone). Releasing them would make
        # the block permanently unpublishable.
        conn, dtm, pool, group = self._fixture(tp_size=2)
        blocks = self._run(conn, dtm, group, session=1, write_session_id="sA")
        context = SaveContext(
            locations=[{"block_idxes": [block]} for block in blocks],
            result_per_rank={0: [True] * self.BLOCKS, 1: [False] * self.BLOCKS},
        )
        conn.on_save_finished("sA", context)
        conn._coordinator_client.ask_verdict.return_value = [False] * self.BLOCKS
        conn._verdict_query_down_until = 0.0
        conn.wait_for_save(TairKvCacheConnectorMetadata(epoch=0))
        self.assertEqual(self.BLOCKS, len(conn._staged.get("r0", {})))
        self.assertEqual(self.CAPACITY - self.BLOCKS, pool.free_blocks)

        # The retry (the same blocks announced again) reuses them, and this
        # time both ranks confirm: the slots come back. The verdict is per
        # session, so the retry has its own answer.
        conn._coordinator_client.ask_verdict.return_value = [True] * self.BLOCKS
        retry = self._run(conn, dtm, group, session=2, write_session_id="sB")
        context = SaveContext(
            locations=[{"block_idxes": [block]} for block in retry],
            result_per_rank={0: [True] * self.BLOCKS, 1: [True] * self.BLOCKS},
        )
        conn.on_save_finished("sB", context)
        conn.wait_for_save(TairKvCacheConnectorMetadata(epoch=0))
        self.assertEqual(self.CAPACITY, pool.free_blocks)

    def test_a_failed_confirmation_report_keeps_the_bytes(self):
        # finish_write_cache itself failed: the manager may still hold these
        # blocks uncommitted, so nothing is released (the review's "survive a
        # failed finish_write_cache report").
        conn, dtm, pool, group = self._fixture(tp_size=2)
        blocks = self._run(conn, dtm, group, session=1, write_session_id="sA")
        conn._manager_client.finish_write_cache.side_effect = RuntimeError("socket")
        context = SaveContext(
            locations=[{"block_idxes": [block]} for block in blocks],
            result_per_rank={0: [True] * self.BLOCKS, 1: [True] * self.BLOCKS},
        )
        conn.on_save_finished("sA", context)
        conn._coordinator_client.ask_verdict.return_value = [False] * self.BLOCKS
        conn._verdict_query_down_until = 0.0
        conn.wait_for_save(TairKvCacheConnectorMetadata(epoch=0))
        self.assertEqual(self.BLOCKS, len(conn._staged.get("r0", {})))

    def test_an_unconfirmed_session_does_not_strand_its_slots_forever(self):
        # The regression's shape: the confirmation never arrives (no later
        # manager answer, no coordinator verdict). The request's retirement is
        # then the last chance -- the invariant family checks the books after
        # every step, and this one pins that the arena comes back whole.
        conn, dtm, pool, group = self._fixture(tp_size=2)
        self._run(conn, dtm, group, session=1, write_session_id="sA")
        self.assertEqual(self.BLOCKS, len(conn._staged.get("r0", {})))
        conn._release_staged("r0")  # the request retires
        self.assertEqual({}, conn._staged.get("r0", {}))
        self.assertEqual(self.CAPACITY, pool.free_blocks)
        self.assertEqual({}, conn._block_uris.get("r0", {}))


class TestTpVerdictRelay(unittest.TestCase):
    """TP: a rank that is not the coordinator confirms before it releases.

    The coordinator (the cross-rank AND) runs in the tp0 worker's process, and
    the workers are separate processes -- the task even runs the engine core in
    its own. So a rank-1 settle cannot see the verdict; it asks for it. Until
    the answer arrives its captures stay staged (rank 0 may fail and the
    manager then rejects the block for everyone), and it is rechecked on the
    next step's metadata. Releasing on a local success without asking strands
    the other ranks' captures (the tp run: 2545 withheld, 138 empty groups) --
    hence both directions are pinned here.
    """

    CAPACITY = 16
    BLOCKS = 4

    def _worker(self, tp_size: int, tp_rank: int = 1, verdict=False):
        import kv_cache_manager.py_connector.vllm.data_transfer as dt

        patcher = mock.patch.object(dt, "batch_gather_scatter_helper")
        patcher.start()
        self.addCleanup(patcher.stop)
        group = windowed_group()
        pool = _StagingPool(torch.device("cpu"), group.per_block_bytes, self.CAPACITY)
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {group.spec_name: pool}
        dtm._manager_block_size = MBS
        dtm._device_mod = MagicMock()
        dtm._transfer_client = MagicMock()
        dtm._transfer_client.SaveKvCaches.return_value = [0]
        dtm._extra_config = SimpleNamespace(
            block_per_save_task=8, block_per_load_task=8, instance_id="tp"
        )
        dtm._io_executor = ThreadPoolExecutor(max_workers=2)
        conn = worker_with([group], dtm)
        conn._pending_saves = {}
        conn._device_mod = MagicMock()
        conn._tp_size = tp_size
        conn._tp_rank = tp_rank
        conn._manager_client = MagicMock()
        conn._extra_config = SimpleNamespace(
            instance_id="tp", block_per_save_task=8, block_per_load_task=8
        )
        conn._coordinator_client.ask_verdict.return_value = verdict
        conn._verdict_query_down_until = 0.0
        return conn, dtm, pool, group

    def _run(self, conn, dtm, group, session, uris, ok=True):
        blocks = list(range(self.BLOCKS))
        dtm._transfer_client.SaveKvCaches.return_value = [0 if ok else 1]
        meta = TairKvCacheConnectorMetadata(epoch=0)
        meta.add_capture_request(
            CaptureRequest("r0", blocks, all_block_ids=[list(range(7, 11))])
        )
        meta.add_save_request(
            SaveRequest(
                req_id="r0",
                write_session_id="s%d" % session,
                manager_block_idxes=blocks,
                all_block_ids=[list(range(7, 11))],
                target_locations=[
                    {
                        "block_idxes": [block],
                        "location_specs": [{"name": "tp0_g0", "uri": "u%d" % block}],
                    }
                    for block in blocks
                ],
                session_id=session,
            )
        )
        conn.wait_for_save(meta)
        deadline = time.time() + 5
        while conn._in_flight.get("r0") and time.time() < deadline:
            time.sleep(0.01)
        return blocks

    def test_a_pending_verdict_keeps_the_bytes_until_it_is_confirmed(self):
        # The settle asks and is told "not decided": the bytes stay, marked
        # pending. The next step's metadata rechecks and the confirmation
        # releases them -- without this the captures strand and the arena
        # starves (the regression).
        conn, dtm, pool, group = self._worker(tp_size=2, tp_rank=1, verdict=None)
        blocks = self._run(conn, dtm, group, session=1, uris=None or ["u%d" % b for b in range(self.BLOCKS)])
        self.assertEqual(self.BLOCKS, len(conn._staged["r0"]))
        self.assertIn("s1", conn._verdict_pending.get("r0", {}))

        # The coordinator now has the AND: the recheck releases.
        conn._coordinator_client.ask_verdict.return_value = [True] * self.BLOCKS
        conn._verdict_query_down_until = 0.0
        conn.wait_for_save(TairKvCacheConnectorMetadata(epoch=0))
        self.assertEqual({}, conn._staged.get("r0", {}))
        self.assertEqual(self.CAPACITY, pool.free_blocks)
        self.assertEqual({}, conn._verdict_pending.get("r0", {}))

    def test_a_rejected_verdict_keeps_the_bytes_for_the_retry(self):
        # A rank failed: the manager rejects the block for everyone, and this
        # rank's bytes are the retry's only copy.
        conn, dtm, pool, group = self._worker(tp_size=2, tp_rank=1, verdict=None)
        blocks = self._run(conn, dtm, group, session=1, uris=["u%d" % b for b in range(self.BLOCKS)])
        conn._coordinator_client.ask_verdict.return_value = [
            True,
            False,
            True,
            True,
        ]
        conn._verdict_query_down_until = 0.0
        conn.wait_for_save(TairKvCacheConnectorMetadata(epoch=0))
        # The committed blocks are released; the rejected one keeps its bytes
        # (the manager rejected it for every rank, so it is announced again and
        # the retry needs them).
        self.assertEqual(
            [("tp0_g0", 1)],
            sorted(conn._staged["r0"]),
            "a rejected block lost its bytes (or a committed one was kept)",
        )
        self.assertEqual({}, conn._verdict_pending.get("r0", {}))

    def test_a_local_success_is_never_released_without_the_verdict(self):
        # The P1 rule: with several ranks a local success says nothing, and a
        # rank that cannot even ask keeps its bytes (the retirement is then the
        # only release).
        conn, dtm, pool, group = self._worker(tp_size=2, tp_rank=1, verdict=False)
        self._run(conn, dtm, group, session=1, uris=["u%d" % b for b in range(self.BLOCKS)])
        self.assertEqual(self.BLOCKS, len(conn._staged["r0"]))
        conn._release_staged("r0")
        self.assertEqual(self.CAPACITY, pool.free_blocks)

    def _mapping(self, ordered, verdict):
        conn, dtm, pool, group = self._worker(tp_size=2, tp_rank=0, verdict=verdict)
        conn._captures.release_staged = MagicMock(return_value=len(ordered))
        conn._verdict_pending = {"r0": {"s9": list(ordered)}}
        conn._recheck_pending_verdicts()
        released = [
            call.args[1]
            for call in conn._captures.release_staged.call_args_list
            if call.args and call.args[0] == "r0"
        ]
        self.assertEqual({}, conn._verdict_pending)
        return released

    def test_a_verdict_indexes_by_position_not_by_block_number(self):
        # The mask follows the announced range: position i is the session's
        # i-th block. A session that starts at block 16 has nothing in common
        # with it, so an absolute-number lookup (block < len(mask)) leaves every
        # block unmatched and the session keeps its bytes forever -- only
        # sessions starting at block 0, where both spaces coincide, ever
        # released. That is why the run freed one session's 96 slots and
        # stranded the other 17.
        self.assertEqual(
            [[16, 17, 18, 19]],
            self._mapping([16, 17, 18, 19], [True, True, True, True]),
        )

    def test_a_rejected_position_keeps_that_block(self):
        # Positional in both directions: a False at position 3 must keep the
        # session's 4th block (19), not whichever block shares index 3 (3).
        self.assertEqual(
            [[16, 17, 18]],
            self._mapping([16, 17, 18, 19], [True, True, True, False]),
        )

    def test_tp0_releases_when_the_and_arrives(self):
        # tp0 follows the same rule as every other rank: the AND publishes the
        # verdict, and the pending recheck releases. Releasing inside
        # on_save_finished would race this rank's own settle -- the last report
        # can be its own task, whose callback restages the captures afterwards
        # (17 of 19 sessions were lost that way in the tp run).
        conn, dtm, pool, group = self._worker(tp_size=2, tp_rank=0)
        blocks = self._run(
            conn, dtm, group, session=1, uris=["u%d" % b for b in range(self.BLOCKS)]
        )
        self.assertEqual(self.BLOCKS, len(conn._staged["r0"]))
        self.assertIn("s1", conn._verdict_pending.get("r0", {}))
        context = SaveContext(
            locations=[{"block_idxes": [block]} for block in blocks],
            result_per_rank={0: [True] * self.BLOCKS, 1: [True] * self.BLOCKS},
        )
        conn.on_save_finished("s1", context)
        # The AND is published (for the other ranks to query) and nothing is
        # released here.
        conn._coordinator_server.record_verdict.assert_called_with(
            "s1", [True] * self.BLOCKS
        )
        self.assertEqual(self.BLOCKS, len(conn._staged["r0"]))
        # The next step's recheck (tp0 asks its own in-process coordinator).
        conn._coordinator_client.ask_verdict.return_value = [True] * self.BLOCKS
        conn._verdict_query_down_until = 0.0
        conn.wait_for_save(TairKvCacheConnectorMetadata(epoch=0))
        self.assertEqual(self.CAPACITY, pool.free_blocks)

    def test_a_settle_landing_after_the_and_is_still_released(self):
        # The interleaving the tp run hit: the last rank to report is tp0
        # itself, so the AND is computed *before* tp0's settle restages its
        # captures. Nothing may act on the verdict there (it would find an
        # empty staged map and never retry); the pending recheck releases them
        # on the next step.
        conn, dtm, pool, group = self._worker(tp_size=2, tp_rank=0, verdict=None)
        blocks = list(range(self.BLOCKS))
        release = threading.Event()

        def slow_save(uris, buffers):
            release.wait(timeout=5)
            return [0]

        dtm._transfer_client.SaveKvCaches.side_effect = slow_save
        meta = TairKvCacheConnectorMetadata(epoch=0)
        meta.add_capture_request(
            CaptureRequest("r0", blocks, all_block_ids=[list(range(7, 11))])
        )
        meta.add_save_request(
            SaveRequest(
                req_id="r0",
                write_session_id="s1",
                manager_block_idxes=blocks,
                all_block_ids=[list(range(7, 11))],
                target_locations=[
                    {
                        "block_idxes": [block],
                        "location_specs": [{"name": "tp0_g0", "uri": "u%d" % block}],
                    }
                    for block in blocks
                ],
                session_id=1,
            )
        )
        conn.wait_for_save(meta)
        # The tasks are still running: nothing is staged yet, and the AND
        # arrives now (this rank was the last to report).
        self.assertEqual({}, conn._staged.get("r0", {}))
        context = SaveContext(
            locations=[{"block_idxes": [block]} for block in blocks],
            result_per_rank={0: [True] * self.BLOCKS, 1: [True] * self.BLOCKS},
        )
        conn.on_save_finished("s1", context)
        conn._coordinator_server.record_verdict.assert_called_with(
            "s1", [True] * self.BLOCKS
        )

        # Now the task reports: the settle restages the captures and marks the
        # session pending. The immediate ask has no answer yet, so the bytes
        # stay -- the verdict is published, the release is the recheck's job.
        release.set()
        deadline = time.time() + 5
        while conn._in_flight.get("r0") and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.BLOCKS, len(conn._staged["r0"]))

        # One more step: the recheck releases what the AND already decided.
        conn._coordinator_client.ask_verdict.return_value = [True] * self.BLOCKS
        conn._verdict_query_down_until = 0.0
        conn.wait_for_save(TairKvCacheConnectorMetadata(epoch=0))
        self.assertEqual({}, conn._staged.get("r0", {}))
        self.assertEqual(self.CAPACITY, pool.free_blocks)


class TestVerdictQueryHygiene(unittest.TestCase):
    """The query endpoint's cost and its bookkeeping (wave9 review)."""

    def _worker(self, tp_size=2, tp_rank=1):
        conn = worker_with([], MagicMock())
        conn._tp_size = tp_size
        conn._tp_rank = tp_rank
        conn._verdict_pending = {
            "r0": {"s1": [0, 1], "s2": [2, 3], "s3": [4, 5]},
        }
        conn._verdict_query_down_until = 0.0
        return conn

    def test_one_failed_probe_mutes_the_endpoint_for_the_round(self):
        # The engine thread runs this loop every step for every pending
        # session; when the coordinator's query port is unbound or gone, each
        # probe costs a full receive timeout. One failure must close the whole
        # round (and the following ones, for a while) instead of paying that
        # per session.
        conn = self._worker()
        conn._coordinator_client.ask_verdict.return_value = False
        conn._recheck_pending_verdicts()
        self.assertEqual(1, conn._coordinator_client.ask_verdict.call_count)
        # ... and the next step does not even try (the endpoint is muted).
        conn._recheck_pending_verdicts()
        self.assertEqual(1, conn._coordinator_client.ask_verdict.call_count)
        # Nothing was decided, so every session is still pending.
        self.assertEqual(3, len(conn._verdict_pending["r0"]))

    def test_a_settle_landing_during_the_recheck_is_not_lost(self):
        # Settles add to the map on I/O threads while the engine thread removes
        # from it: an entry added between the emptiness test and the top-level
        # pop must survive, or its captures are never rechecked and the arena
        # starves.
        conn = self._worker()
        conn._verdict_pending = {"r0": {"s1": [0, 1]}}

        def ask(session_id, tp_rank=0, timeout_ms=100):
            if session_id == "s1":
                # A settle lands while we are deciding, adding a new session.
                conn._verdict_pending.setdefault("r0", {})["s9"] = [6, 7]
                return [True, True]
            return None

        conn._coordinator_client.ask_verdict.side_effect = ask
        conn._recheck_pending_verdicts()
        self.assertEqual(
            {"s9": [6, 7]},
            conn._verdict_pending.get("r0", {}),
            "the session that landed during the recheck was dropped",
        )

    def test_a_retired_requests_uris_are_dropped_when_its_saves_settle(self):
        # A retired request keeps its URI map while a save is in flight, but
        # the last settle has to clear it: nothing can release or look those
        # keys up afterwards, and every _uris_other_requests_hold scan walks
        # the map.
        conn = worker_with([], MagicMock())
        conn._tp_size = 2
        conn._tp_rank = 1
        conn._block_uris = {"r0": {0: "uri-0"}}
        conn._in_flight["r0"] = {(1, "tp0_g0", 0): None}
        conn._staged.pop("r0", None)
        conn._captures.retire("r0")
        conn._captures.settle("r0", 1, {("tp0_g0", 0): True})
        conn._drop_uris_if_idle("r0")
        self.assertEqual({}, conn._block_uris.get("r0", {}))
        # A request that still owns captures keeps its keys.
        conn._block_uris = {"r1": {0: "uri-0"}}
        conn._in_flight["r1"] = {(1, "tp0_g0", 0): None}
        conn._drop_uris_if_idle("r1")
        self.assertEqual({"r1": {0: "uri-0"}}, conn._block_uris)


class TestVerdictPublication(unittest.TestCase):
    """wave10: a verdict is a fact only once the manager committed."""

    BLOCKS = 4

    def _worker(self):
        conn = worker_with([], MagicMock())
        conn._tp_size = 2
        conn._tp_rank = 0
        conn._manager_client = MagicMock()
        conn._extra_config = SimpleNamespace(instance_id="test")
        conn._coordinator_server = MagicMock()
        return conn

    def _context(self):
        from kv_cache_manager.py_connector.common.tp_coordinator import SaveContext

        return SaveContext(
            locations=[{"block_idxes": [b]} for b in range(self.BLOCKS)],
            result_per_rank={0: [True] * self.BLOCKS, 1: [True] * self.BLOCKS},
        )

    def test_a_failed_report_does_not_publish_the_verdict(self):
        # Every rank wrote, but the report to the manager failed: the block was
        # never committed. Publishing here would let every rank release its
        # only captured copy, and a retry could not republish the windowed
        # specs once the HBM window moved.
        conn = self._worker()
        conn._manager_client.finish_write_cache.side_effect = RuntimeError("socket")
        conn.on_save_finished("s1", self._context())
        conn._coordinator_server.record_verdict.assert_not_called()

        # The retry's report succeeds: now the verdict is published.
        conn._manager_client.finish_write_cache.side_effect = None
        conn.on_save_finished("s2", self._context())
        conn._coordinator_server.record_verdict.assert_called_once_with(
            "s2", [True] * self.BLOCKS
        )

    def test_retirement_cancels_the_sessions_it_can_no_longer_ask_about(self):
        # A fast rank retires the request while slower ranks are still writing
        # it: without telling the coordinator, its entry waits for a query that
        # cannot come, and the table grows without bound under repeated early
        # retirements.
        conn = self._worker()
        conn._verdict_pending = {"r0": {"s1": [0, 1], "s2": [2, 3]}}
        conn._captures.release_staged = MagicMock()
        conn._drop_pending_verdicts("r0")
        self.assertEqual(
            [mock.call("s1", 0), mock.call("s2", 0)],
            conn._coordinator_client.cancel_verdict.call_args_list,
        )
        self.assertEqual({}, conn._verdict_pending)


class TestVerdictRetention(unittest.TestCase):
    """The coordinator keeps a verdict until every rank has consumed it."""

    def _server(self):
        from kv_cache_manager.py_connector.common.tp_coordinator import (
            TpCoordinatorServer,
        )

        server = TpCoordinatorServer.__new__(TpCoordinatorServer)
        server._tp_world_size = 2
        server._verdicts = {}
        server._verdict_acks = {}
        server._verdicts_lock = threading.Lock()
        return server

    def _ask(self, server, session_id, tp_rank):
        return server.answer_query(session_id, tp_rank)

    def test_a_cancelled_verdict_goes_away(self):
        # The retiring rank cannot ask any more; its cancel counts as the
        # answer so the entry does not wait for it forever.
        server = self._server()
        server.record_verdict("s1", [True])
        server.cancel_query("s1", tp_rank=1)
        self.assertEqual([True], self._ask(server, "s1", tp_rank=0))
        self.assertNotIn("s1", server._verdicts)

    def test_the_storage_is_bounded(self):
        # Entries no rank ever asks about (a dead rank) must not grow without
        # bound: the cap evicts the oldest and says so.
        server = self._server()
        for index in range(5000):
            server.record_verdict("s%d" % index, [True])
        self.assertLessEqual(len(server._verdicts), 4096)
        self.assertLessEqual(len(server._verdict_acks), 4096)

    def test_a_verdict_survives_until_every_rank_asked(self):
        # Count-based eviction used to drop the oldest masks regardless of who
        # had consumed them: a delayed rank then received None forever and kept
        # its captures until retirement, starving the arena. Only the
        # acknowledgements of every rank may retire a verdict.
        server = self._server()
        server.record_verdict("s-oldest", [True, True])
        for index in range(2048):
            server.record_verdict("s%d" % index, [True])
        self.assertEqual(
            [True, True],
            self._ask(server, "s-oldest", tp_rank=0),
            "a verdict was evicted before its rank asked",
        )
        # Rank 1 still gets its answer; after that the entry may go.
        self.assertEqual([True, True], self._ask(server, "s-oldest", tp_rank=1))
        self.assertNotIn("s-oldest", server._verdicts)


class TestCaptureFloor(unittest.TestCase):
    """The capture's room survives a burst of in-flight tasks.

    The gemma-3-4b concurrent run (16 requests, 6 windowed buckets, a 1024-slot
    pool: 896 shared + 128 reserve) withheld one 16-block run and then failed
    6 groups x 16/16 announced blocks: the shared part was held by in-flight
    tasks and other requests' captures, and the 50 ms deadline expired. Sizing
    the partition is not enough -- the shared part needs a runtime floor.
    """

    def _shared_pool_dtm(self, groups, pool):
        class _DTM:
            def __init__(self):
                self._pools = {}
                self.captures = []

            def prepare_capture(self, *a, **kw):  # pragma: no cover - unused
                raise AssertionError

            def pool_of(self, group):
                return pool

            def capture_blocks(self, group, block_slots, timeout):
                # The real capture: one contiguous run per group, all held.
                self.captures.append(group.spec_name)
                start = pool.acquire(len(block_slots), timeout=timeout, keep_reserved=True)
                return start

        return _DTM()

    def test_a_task_may_not_take_the_captures_floor(self):
        pool = _StagingPool(
            torch.device("cpu"), 1024, 8, reserved_blocks=3, capture_blocks=4
        )
        # Tasks take the reserve first, then borrow the shared runs from the
        # top -- only while the floor stays free.
        taken = []
        while True:
            start = pool.acquire(3, timeout=0.05)
            if start is None:
                break
            taken.append(start)
        # Only the reserve: the shared part is all floor (5 slots, floor 4),
        # so a task batch may not borrow from it at all.
        self.assertEqual([5], taken)
        floor = pool.acquire(4, timeout=0.01, keep_reserved=True)
        self.assertIsNotNone(floor, "a burst of tasks ate the captures' room")
        pool.release(floor, 4)
        for start in taken:
            pool.release(start, 3)
        self.assertEqual(8, pool.free_blocks)

    def test_the_floor_stays_one_contiguous_run(self):
        # Fragmentation is the other half of the floor: four free slots split
        # into 2 + 2 cannot serve a four-block capture, so a task may not
        # borrow even though the total looks sufficient.
        pool = _StagingPool(
            torch.device("cpu"), 1024, 8, reserved_blocks=2, capture_blocks=4
        )
        # Carve the shared part into 2 + 2 with two live captures (they keep
        # their runs), leaving 2 free at the bottom too.
        # The two free two-slot runs an interleaved release/allocate pattern
        # can leave behind: four shared slots in total, none usable whole.
        # (Exhaust the reserve first: a task takes its own room before it
        # borrows the shared part at all.)
        reserve = pool.acquire(2, timeout=0.01)
        self.assertIsNotNone(reserve)
        pool._runs = [[0, 2], [4, 2]]
        # Taking one of them would leave at most two in one piece, so the
        # borrow must be refused even though four slots are free (the
        # sum-only rule allowed it and the next capture timed out).
        self.assertIsNone(pool.acquire(1, timeout=0.05))
        # A capture needs one run of four: with the pieces it must wait, and
        # once a two-slot run is handed back it goes through.
        self.assertIsNone(pool.acquire(4, timeout=0.05, keep_reserved=True))
        pool._runs = [[0, 4]]
        self.assertEqual(0, pool.acquire(4, timeout=0.05, keep_reserved=True))
        pool.release(0, 4)
        pool.release(reserve, 2)

        # The policy keeps that shape from forming in the first place: with
        # the shared part holding exactly the floor, a live capture takes its
        # four blocks and the borrow cannot shave the floor's last run.
        pool2 = _StagingPool(
            torch.device("cpu"), 1024, 6, reserved_blocks=2, capture_blocks=4
        )
        first = pool2.acquire(4, timeout=0.01, keep_reserved=True)
        self.assertEqual(0, first)
        reserve = pool2.acquire(2, timeout=0.01)  # a task takes its own room
        self.assertIsNotNone(reserve)
        self.assertIsNone(pool2.acquire(1, timeout=0.05))
        pool2.release(reserve, 2)
        pool2.release(first, 4)
        again = pool2.acquire(4, timeout=0.01, keep_reserved=True)
        self.assertIsNotNone(again, "the floor did not come back")
        pool2.release(again, 4)

    def test_one_step_spends_one_deadline_across_requests(self):
        # The budget belongs to the step: two capture requests must not each
        # get a fresh 50 ms (the engine thread pays for both).
        groups = [windowed_group("tp0_g0", group_idx=0)]
        pool = _StagingPool(torch.device("cpu"), 4096, 32, reserved_blocks=8)
        seen: List[float] = []

        class _SlowFirst:
            def pool_of(self, group):
                return pool

            def capture_blocks(self, group, slots, timeout):
                seen.append(timeout)
                if len(seen) == 1:
                    time.sleep(0.3)
                return pool.acquire(len(slots), timeout=timeout, keep_reserved=True)

        conn = worker_with(groups, _SlowFirst())
        meta = TairKvCacheConnectorMetadata(epoch=0)
        for req_id in ("r0", "r1"):
            meta.add_capture_request(
                CaptureRequest(req_id, [0], all_block_ids=[[7]])
            )
        with mock.patch(
            "kv_cache_manager.py_connector.vllm.data_transfer.batch_gather_scatter_helper"
        ):
            conn._capture_blocks(meta)
        self.assertEqual(2, len(seen))
        self.assertLess(seen[1], seen[0])
        conn._release_staged("r0")
        conn._release_staged("r1")

    def test_one_capture_request_spends_one_deadline(self):
        # Six buckets must not each start a fresh 50 ms wait: the budget is a
        # property of the step, and the engine thread is what pays for it.
        groups = [windowed_group(f"tp0_g{i}", group_idx=i) for i in range(3)]
        pool = _StagingPool(torch.device("cpu"), 4096, 32, reserved_blocks=8)
        seen: List[float] = []

        class _SlowFirst:
            def pool_of(self, group):
                return pool

            def capture_blocks(self, group, slots, timeout):
                seen.append(timeout)
                if len(seen) == 1:
                    # Stand in for the SDK/DMA work a first acquisition waits
                    # on: the remaining buckets must see the budget shrink.
                    time.sleep(0.3)
                return pool.acquire(len(slots), timeout=timeout, keep_reserved=True)

        conn = worker_with(groups, _SlowFirst())
        with mock.patch(
            "kv_cache_manager.py_connector.vllm.data_transfer.batch_gather_scatter_helper"
        ):
            self.assertTrue(
                conn._capture_request(
                    CaptureRequest(
                        "r0",
                        list(range(2)),
                        all_block_ids=[[7, 8], [7, 8], [7, 8]],
                    )
                )
            )
        self.assertEqual(3, len(seen))
        self.assertLessEqual(seen[0], CAPTURE_POOL_WAIT_SECONDS)
        self.assertGreater(seen[0], CAPTURE_POOL_WAIT_SECONDS - 0.01)
        self.assertLess(seen[1], seen[0])
        self.assertLessEqual(seen[2], seen[1])
        conn._release_staged("r0")

    def test_a_burst_of_requests_does_not_starve_a_multi_bucket_capture(self):
        # The e2e's shape: six windowed buckets share one arena, each capture
        # request needs 16 contiguous blocks per bucket and holds all six runs,
        # and the pool's floor covers exactly that. In-flight task staging may
        # fill everything above it, but not it.
        groups = [windowed_group(f"tp0_g{i}", group_idx=i) for i in range(6)]
        pool = _StagingPool(
            torch.device("cpu"),
            groups[0].per_block_bytes,
            1024,
            reserved_blocks=128,
            capture_blocks=6 * 16,
        )
        dtm = self._shared_pool_dtm(groups, pool)
        conn = worker_with(groups, dtm)
        with mock.patch(
            "kv_cache_manager.py_connector.vllm.data_transfer.batch_gather_scatter_helper"
        ):
            # A burst of in-flight task staging (loads/saves) borrows the shared
            # part until the floor stops it.
            held = []
            while True:
                start = pool.acquire(63, timeout=0.05)
                if start is None:
                    break
                held.append(start)
            self.assertGreater(len(held), 2, "the burst did not even reach the floor")
            tables = [
                list(range(7 + 16 * i, 7 + 16 * (i + 1))) for i in range(6)
            ]
            self.assertTrue(
                conn._capture_request(
                    CaptureRequest("r0", list(range(16)), all_block_ids=tables)
                ),
                "the capture was starved by in-flight task staging",
            )
            self.assertEqual(6 * 16, len(conn._staged["r0"]))  # 6 buckets x 16
            for start in held:
                pool.release(start, 63)
            conn._release_staged("r0")
        self.assertEqual(1024, pool.free_blocks)

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
        in_flight = sum(len(entries) for entries in conn._in_flight.values())
        self.assertLessEqual(
            pool.free_blocks + staged + in_flight + sessions,
            self.CAPACITY,
            "the arena handed out more slots than it has: a range was freed twice",
        )
        self.assertEqual(
            self.CAPACITY, pool.free_blocks + staged + in_flight + sessions
        )

    def _plan(self, conn, blocks, session=0):
        return conn._plan_group_transfers(
            locations([["tp0_g0"]] * len(blocks)),
            blocks,
            [list(range(7, 7 + len(blocks)))],
            "r0",
            for_save=True,
            session_id=session,
        )[0]

    def _session_callback(self, conn, req_id, blocks, plans, session=0):
        """The worker's per-session result callback: the retain wrapper over a
        sink, exactly as wait_for_save builds it."""
        done = {}
        base = lambda flat: done.setdefault("flat", flat)  # noqa: E731
        save_req = SimpleNamespace(
            req_id=req_id, manager_block_idxes=list(blocks), session_id=session
        )
        return conn._settle_session_captures(save_req, plans, base), done

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
        self._assert_arena_accounting(conn, pool, sessions=0)

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
        self.assertEqual(self.BLOCKS, len(conn._in_flight["r0"]))
        # The session settles: the slots come back exactly once, even though
        # the request retired before the task reported.
        conn._captures.settle("r0", 0, {("tp0_g0", block): True for block in blocks})
        self.assertEqual(pool.max_blocks, pool.free_blocks)
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

    def test_a_failed_write_keeps_the_bytes_for_a_retry(self):
        # The window that produced these bytes has moved on by the time a later
        # session announces the block again, so the capture is the only copy
        # the connector will ever hold: a failed write keeps the slot and the
        # worker restores the entry to its staged map (the callback below is
        # the worker's, so the restore happens here too), and the next plan
        # publishes the block.
        conn, dtm, pool, group = self._fixture(ok=False)
        blocks = list(range(self.BLOCKS))
        conn._capture_request(CaptureRequest("r0", blocks, all_block_ids=[[7, 8, 9, 10]]))
        plan = self._plan(conn, blocks)
        done_cb, done = self._session_callback(conn, "r0", blocks, [plan])
        dtm.save_task(
            MultiResult(1, done_cb),
            0,
            group,
            plan.uris,
            None,
            None,
            plan.staged_slots,
        )
        self.assertEqual([False] * self.BLOCKS, done["flat"])
        # Nothing was published; the settle kept the slots and the bytes are
        # staged again for the retry.
        self.assertEqual(self.BLOCKS, len(conn._staged.get("r0", {})))
        self.assertEqual({}, conn._in_flight.get("r0", {}))
        self.assertEqual(self.CAPACITY - self.BLOCKS, pool.free_blocks)
        self._assert_arena_accounting(conn, pool, sessions=0)

        # The retry session finds them and publishes.
        dtm._transfer_client.SaveKvCaches.return_value = [0]
        retry = self._plan(conn, blocks)
        self.assertNotIn(None, retry.staged_slots)
        dtm.save_task(
            MultiResult(1, self._session_callback(conn, "r0", blocks, [retry])[0]),
            0,
            group,
            retry.uris,
            None,
            None,
            retry.staged_slots,
        )
        # The retry's success is confirmed by the coordinator (one rank, in
        # process): the recheck returns the slots.
        conn._coordinator_client.ask_verdict.return_value = [True] * self.BLOCKS
        conn._verdict_query_down_until = 0.0
        conn.wait_for_save(TairKvCacheConnectorMetadata(epoch=0))
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
        self._assert_arena_accounting(conn, pool, sessions=0)

    def test_a_raising_completion_report_still_settles_the_captures(self):
        # done_cb talks to the coordinator socket, which can fail: reporting
        # must not be able to skip the settle, or the session's slots stay in
        # flight for good (MultiResult never calls a callback twice).
        conn, dtm, pool, group = self._fixture()
        blocks = list(range(self.BLOCKS))
        conn._capture_request(CaptureRequest("r0", blocks, all_block_ids=[[7, 8, 9, 10]]))
        plan = self._plan(conn, blocks, session=1)
        self.assertNotIn(None, plan.staged_slots)

        def raising_report(flat):
            raise RuntimeError("coordinator socket died")

        done_cb = conn._settle_session_captures(
            SimpleNamespace(
                req_id="r0", manager_block_idxes=blocks, session_id=1
            ),
            [plan],
            raising_report,
        )
        dtm._transfer_client.SaveKvCaches.return_value = [1]  # the write fails
        with self.assertRaises(RuntimeError):
            dtm.save_task(
                MultiResult(1, done_cb),
                0,
                group,
                plan.uris,
                None,
                None,
                plan.staged_slots,
            )
        # Settled despite the report raising: the bytes are staged for a
        # retry, nothing is left in flight.
        self.assertEqual({}, conn._in_flight.get("r0", {}))
        self.assertEqual(self.BLOCKS, len(conn._staged["r0"]))

    def test_a_local_success_keeps_the_bytes_until_the_commit(self):
        # A block is committed only when *every* rank wrote it (the
        # coordinator ANDs the ranks into finish_write_cache). This rank's
        # task success therefore keeps its capture: if another rank failed,
        # the manager rejects the block and the retry has to find the windowed
        # bytes still here -- the HBM window is gone, so a later capture
        # abstains and the block could never be published.
        conn, dtm, pool, group = self._fixture()
        conn._tp_size = 2  # two ranks: a local verdict is not the global one
        conn._manager_client = MagicMock()
        conn._extra_config = SimpleNamespace(instance_id="test")
        blocks = list(range(self.BLOCKS))
        conn._capture_request(CaptureRequest("r0", blocks, all_block_ids=[[7, 8, 9, 10]]))
        first = self._plan(conn, blocks, session=1)
        done = {}
        dtm.save_task(
            MultiResult(1, self._session_callback(conn, "r0", blocks, [first], session=1)[0]),
            0,
            group,
            first.uris,
            None,
            None,
            first.staged_slots,
        )
        _ = first
        self.assertEqual(self.BLOCKS, len(conn._staged["r0"]))
        self.assertEqual({}, conn._in_flight.get("r0", {}))
        self._assert_arena_accounting(conn, pool, sessions=0)

        # The other rank failed, so the manager did not commit: the retry
        # session (a new announcement of the same blocks) reuses the very
        # bytes and both ranks succeed this time.
        retained = {entry.slot for entry in conn._staged["r0"].values()}
        self.assertEqual(self.BLOCKS, len(retained))
        retry = self._plan(conn, blocks, session=2)
        self.assertNotIn(None, retry.staged_slots)
        self.assertTrue(
            set(retry.staged_slots) <= retained,
            "the retry did not reuse the retained bytes",
        )
        retry_session = "sB"
        dtm.save_task(
            MultiResult(1, self._session_callback(conn, "r0", blocks, [retry], session=2)[0]),
            0,
            group,
            retry.uris,
            None,
            None,
            retry.staged_slots,
        )
        # Still retained; the coordinator's confirmation (the AND over the
        # ranks) releases them.
        self.assertEqual(self.BLOCKS, len(conn._staged["r0"]))
        confirmed = SaveContext(
            locations=[{"block_idxes": [block]} for block in blocks],
            result_per_rank={0: [True] * self.BLOCKS, 1: [True] * self.BLOCKS},
        )
        conn.on_save_finished(retry_session, confirmed)
        conn._coordinator_client.ask_verdict.return_value = [True] * self.BLOCKS
        conn._verdict_query_down_until = 0.0
        conn.wait_for_save(TairKvCacheConnectorMetadata(epoch=0))
        self.assertEqual(self.CAPACITY, pool.free_blocks)
        self._assert_arena_accounting(conn, pool, sessions=0)

    def test_a_release_skips_bytes_another_request_is_still_writing(self):
        # Two requests share a prefix: the manager answers "existing" for the
        # second one's blocks while the first one's allocations are still
        # uncommitted (duplicate StartWriteCache calls are deduplicated before
        # FinishWriteCache). The release computed from that answer must not
        # drop the second request's captures -- if the first writer then
        # fails, neither request could publish those blocks.
        group = windowed_group()
        pool = _StagingPool(torch.device("cpu"), group.per_block_bytes, 32)
        dtm = DataTransferManager.__new__(DataTransferManager)
        dtm._pools = {group.spec_name: pool}
        dtm._manager_block_size = MBS
        dtm._device_mod = MagicMock()
        dtm._transfer_client = MagicMock()
        dtm._transfer_client.SaveKvCaches.return_value = [0]
        dtm._extra_config = SimpleNamespace(
            block_per_save_task=128, block_per_load_task=128
        )
        dtm._io_executor = ThreadPoolExecutor(max_workers=2)
        conn = worker_with([group], dtm)
        conn._pending_saves = {}
        conn._device_mod = MagicMock()

        blocks = list(range(4))
        shared_uris = ["u%d" % block for block in blocks]

        def save_meta(req_id, session_id):
            meta = TairKvCacheConnectorMetadata(epoch=0)
            meta.add_capture_request(
                CaptureRequest(req_id, blocks, all_block_ids=[list(range(7, 11))])
            )
            meta.add_save_request(
                SaveRequest(
                    req_id=req_id,
                    write_session_id="s%s" % req_id,
                    manager_block_idxes=blocks,
                    all_block_ids=[list(range(7, 11))],
                    target_locations=[
                        {
                            "block_idxes": [block],
                            "location_specs": [
                                {"name": "tp0_g0", "uri": shared_uris[block]}
                            ],
                        }
                        for block in blocks
                    ],
                    session_id=session_id,
                )
            )
            return meta

        with mock.patch(
            "kv_cache_manager.py_connector.vllm.data_transfer.batch_gather_scatter_helper"
        ):
            # r0's session holds the shared URIs (in flight, or already
            # settled into its staged map -- both keep the keys guarded).
            # Its write fails, which is the case that keeps the bytes.
            dtm._transfer_client.SaveKvCaches.return_value = [1]
            conn.wait_for_save(save_meta("r0", 1))
            # ... while r1 only has its captures staged.
            conn._capture_request(
                CaptureRequest("r1", blocks, all_block_ids=[list(range(7, 11))])
            )
            self.assertEqual(len(blocks), len(conn._staged.get("r1", {})))
            held = len(conn._in_flight.get("r0", {})) + len(
                conn._staged.get("r0", {})
            )
            self.assertEqual(len(blocks), held)

            # The release computed for r1 (the manager reported those very
            # cache keys as already cached) must not drop the blocks r0 is
            # still writing.
            conn._release_captures_for(
                "r1",
                blocks,
                {block: [shared_uris[block]] for block in blocks},
            )
            self.assertEqual(
                len(blocks),
                len(conn._staged.get("r1", {})),
                "a release dropped bytes another request is still writing",
            )

            # Once r0 no longer holds any bytes for those keys (its request
            # retired), the same release goes through.
            # r0's tasks report on the io threads; wait for the settle so the
            # retirement below sees staged entries rather than in-flight ones.
            deadline = time.time() + 5
            while conn._in_flight.get("r0") and time.time() < deadline:
                time.sleep(0.01)
            self.assertEqual({}, conn._in_flight.get("r0", {}))
            conn._release_staged("r0")
            conn._release_captures_for(
                "r1",
                blocks,
                {block: [shared_uris[block]] for block in blocks},
            )
            self.assertEqual({}, conn._staged.get("r1", {}))
        conn._release_staged("r1")

    def test_a_raising_sdk_is_a_failure_and_keeps_the_bytes(self):
        # An SDK that raises (a pybind/backend exception, not an error code) is
        # reported like any other failure, and the capture ledger keeps the
        # bytes for a retry: no slot is lost to the exception, and none is
        # released behind the ledger's back (the r2 N1 leak was a release that
        # only the DTM knew about).
        conn, dtm, pool, group = self._fixture()
        blocks = list(range(self.BLOCKS))
        conn._capture_request(CaptureRequest("r0", blocks, all_block_ids=[[7, 8, 9, 10]]))
        plan = self._plan(conn, blocks)
        dtm._transfer_client.SaveKvCaches.side_effect = RuntimeError("pybind blew up")
        ok_mask: List[Any] = [None] * self.BLOCKS
        dtm._save_staged_blocks(
            group, plan.uris, plan.staged_slots, [0, 1, 2, 3], ok_mask
        )
        self.assertEqual([False] * self.BLOCKS, ok_mask)
        # Still in flight (the session has not settled yet) and the arena is
        # untouched.
        self.assertEqual(self.BLOCKS, len(conn._in_flight["r0"]))
        self.assertEqual(self.CAPACITY - self.BLOCKS, pool.free_blocks)
        self._assert_arena_accounting(conn, pool, sessions=0)

        # Through save_task the exception turns into all-failed verdicts, the
        # settle keeps the bytes staged, and nothing was released twice.
        conn._capture_request(CaptureRequest("r0", blocks, all_block_ids=[[7, 8, 9, 10]]))
        plan = self._plan(conn, blocks, session=7)
        dtm._transfer_client.SaveKvCaches.side_effect = None
        dtm._transfer_client.SaveKvCaches.return_value = [0]
        done = {}
        done_cb = self._session_callback(conn, "r0", blocks, [plan], session=7)[0]
        dtm.save_task(
            MultiResult(1, done_cb),
            0,
            group,
            plan.uris,
            None,
            None,
            plan.staged_slots,
        )
        # Session 0 never settled, so its four records are still in flight
        # (part 1's blocks were never written); session 7's own four were
        # taken and its callback wrote them on this single rank, so they are
        # staged and pending the coordinator's AND. Two sessions on the same
        # (spec, block) keys coexist, which is why the in-flight ownership is
        # keyed by session.
        self.assertEqual(self.BLOCKS, len(conn._in_flight.get("r0", {})))
        self.assertEqual({0}, {key[0] for key in conn._in_flight["r0"]})
        self.assertEqual(self.BLOCKS, len(conn._staged.get("r0", {})))
        conn._coordinator_client.ask_verdict.return_value = [True] * self.BLOCKS
        conn._verdict_query_down_until = 0.0
        conn.wait_for_save(TairKvCacheConnectorMetadata(epoch=0))
        self.assertEqual({}, conn._staged.get("r0", {}))
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
        # The slot is *not* returned here: it belongs to the capture ledger,
        # and releasing it in the task would race the settle that keeps the
        # bytes for a retry (or hands the slot to the next capture that the
        # failure's restaged entry would then read). The ledger settles it.
        self.assertEqual(3, pool.free_blocks)
        self.assertEqual([0], [entry.slot for entry in conn._staged["r0"].values()])
        conn._captures.settle("r0", 0, {("tp0_g0", 0): False})
        self.assertEqual(3, pool.free_blocks)  # kept for the retry
        conn._captures.settle("r0", 0, {("tp0_g0", 0): False})  # idempotent
        self.assertEqual(3, pool.free_blocks)
        # The retry session takes the restaged entry and its write lands; the
        # bytes are kept until the commit is confirmed (see settle).
        self.assertIsNotNone(conn._captures.take("r0", 7, ("tp0_g0", 0)))
        conn._captures.settle("r0", 7, {("tp0_g0", 0): True})
        self.assertEqual(3, pool.free_blocks)
        conn._captures.release_staged("r0", [0])
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

    def test_an_undeclared_ring_subclass_is_refused_not_dropped(self):
        # The scratch skip is exact-type matched like every other spec receipt:
        # when the engine declares its transfer projection, a spec it hands to
        # connectors must be *refused* by name (the connector has no transfer
        # layout for it) rather than silently dropped, so an external hit can
        # never resume with that group's state uninitialized. The declared
        # classes -- the compression ring, and the packed wrapper of rings --
        # are still skipped: the engine rebuilds those.
        from vllm.v1.kv_cache_interface import (
            CircularBufferSpec,
            FullAttentionSpec,
            UniformTypeKVCacheSpecs,
        )

        class MyRing(CircularBufferSpec):
            pass

        class MyWrapper(UniformTypeKVCacheSpecs):
            pass

        def ring(cls=CircularBufferSpec):
            return cls(block_size=MBS, page_size_bytes=2048)

        companion = SimpleNamespace(
            layer_names=["l9"],
            kv_cache_spec=FullAttentionSpec(
                block_size=MBS,
                num_kv_heads=2,
                head_size=64,
                dtype=torch.bfloat16,
                page_size_bytes=2048,
            ),
            is_eagle_group=False,
        )

        def config(spec):
            return SimpleNamespace(
                kv_cache_groups=[
                    SimpleNamespace(
                        layer_names=["l0"], kv_cache_spec=spec, is_eagle_group=False
                    ),
                    companion,
                ],
                transfer_group_ids=[0, 1],
            )

        for spec, name in (
            (ring(MyRing), "MyRing"),
            (MyWrapper(MBS, {"l0": ring()}), "MyWrapper"),
        ):
            with self.subTest(declared_class=name):
                with self.assertRaises(NotImplementedError) as ctx:
                    parse_groups(config(spec), MBS, calculate_kv_scales=False)
                self.assertIn(name, str(ctx.exception))
                self.assertIn("exposes it for transfer", str(ctx.exception))

        for spec in (ring(), UniformTypeKVCacheSpecs(MBS, {"l0": ring()})):
            with self.subTest(skipped=type(spec).__name__):
                metas = parse_groups(config(spec), MBS, calculate_kv_scales=False)
                self.assertEqual([1], [m.group_idx for m in metas])

    def _packed(self, specs):
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

        return SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=list(specs),
                    kv_cache_spec=UniformTypeKVCacheSpecs(MBS, dict(specs)),
                    is_eagle_group=False,
                )
            ]
        )

    def test_a_packed_group_with_divergent_windows_is_refused(self):
        # Two sliding members with different windows are two buckets with
        # different coverage: the scheduler sees only the folded representative
        # (one window), so it would emit no capture and no clamp for the other
        # bucket's window (its claimed saves fail), or announce the wrong
        # retention. The connector refuses the divergence where the whole
        # member list is visible -- and the scheduler's own (folded) view is
        # shown to be exactly what it is: one bucket, one window.
        from vllm.v1.kv_cache_interface import SlidingWindowSpec

        def spec(window):
            return SlidingWindowSpec(
                block_size=MBS,
                sliding_window=window,
                page_size_bytes=2048,
                num_kv_heads=2,
                head_size=64,
                dtype=torch.bfloat16,
            )

        with self.assertRaises(NotImplementedError) as ctx:
            parse_groups(
                self._packed({"l0": spec(1024), "l1": spec(512)}),
                MBS,
                calculate_kv_scales=False,
            )
        self.assertIn("different coverage", str(ctx.exception))

        folded = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["l0", "l1"],
                    kv_cache_spec=spec(1024),
                    is_eagle_group=False,
                )
            ]
        )
        metas = parse_groups(folded, MBS, calculate_kv_scales=False)
        self.assertEqual(1, len(metas))
        self.assertEqual(1024, metas[0].retention.window_tokens)

    def test_a_packed_group_mixing_full_and_windowed_is_refused(self):
        # The same divergence in its other shape: a full-attention member
        # carries no window at all, so the folded representative could be the
        # one bucket whose retention the scheduler can see, while the windowed
        # member's blocks need staged bytes nobody announces.
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        full = FullAttentionSpec(
            block_size=MBS,
            num_kv_heads=2,
            head_size=64,
            dtype=torch.bfloat16,
            page_size_bytes=2048,
        )
        windowed = SlidingWindowSpec(
            block_size=MBS,
            sliding_window=1024,
            page_size_bytes=2048,
            num_kv_heads=2,
            head_size=64,
            dtype=torch.bfloat16,
        )
        with self.assertRaises(NotImplementedError) as ctx:
            parse_groups(
                self._packed({"l0": full, "l1": windowed}),
                MBS,
                calculate_kv_scales=False,
            )
        self.assertIn("different coverage", str(ctx.exception))

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
        install_capture_ledger(conn)
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
