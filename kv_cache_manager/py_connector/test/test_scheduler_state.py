"""Unit tests for the connector's scheduler-side logic.

Covers, against fake vLLM SchedulerOutput / Request objects:

* ``get_num_new_matched_tokens`` -- including the full-prompt external hit cap
  (a fully cached prompt must leave >= 1 token to recompute, otherwise vLLM's
  synchronous-load scheduling path asserts ``num_new_tokens > 0``);
* ``parse_block_mask_to_save_indices`` -- ``offset`` and ``bool_masks`` forms;
* ``_parse_groups`` -- full-attention single group, hybrid multi group, eagle
  group skip, unsupported spec error;
* ``build_connector_meta`` -- new request, cached deltas (``new_block_ids``
  None / non-None), preemption resume via both the 0.26 ``resumed_req_ids`` and
  the legacy ``resumed_from_preemption`` interfaces, save-threshold trigger,
  and the two ``request_finished`` paths (saves landed / in-flight).
"""

import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock

from kv_cache_manager.py_connector.test.vllm_stubs import (
    make_connector, ReqState, GroupMeta)
from kv_cache_manager.py_connector.vllm.v1_connector import TairKvCacheConnector
from kv_cache_manager.py_connector.vllm.metadata import SaveRequest


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
@dataclass
class FakeRequest:
    request_id: str
    prompt_token_ids: list
    output_token_ids: list = field(default_factory=list)

    @property
    def num_tokens(self):
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def all_token_ids(self):
        return self.prompt_token_ids + self.output_token_ids


def make_scheduler_connector(mbs=16, vllm_bs=None, locations=None):
    """Connector with the scheduler-side state build_connector_meta needs."""
    conn = make_connector(manager_block_size=mbs, vllm_block_size=vllm_bs)
    conn._epoch = 0
    conn._alive_requests = {}
    conn._waiting_to_load_requests = []
    import threading
    conn._waiting_to_save_requests_lock = threading.Lock()
    conn._waiting_to_save_requests = []
    conn._waiting_to_finish_requests = []
    conn._canceled_save_request_ids_lock = threading.Lock()
    conn._canceled_save_request_ids = []
    conn._http_executor = MagicMock()
    conn._location_query_manager = MagicMock()
    conn._location_query_manager.get_locations_for_query.return_value = (
        True, locations if locations is not None else [])
    return conn


def fake_scheduler_output(new_reqs=(), cached_req_ids=(), num_scheduled=None,
                          new_block_ids=(), resumed_req_ids=frozenset(),
                          legacy_resumed=None):
    """Build a fake SchedulerOutput. legacy_resumed switches the cached-reqs
    container to the pre-0.26 interface (resumed_from_preemption list, no
    resumed_req_ids attribute)."""
    if legacy_resumed is not None:
        cached = SimpleNamespace(
            req_ids=list(cached_req_ids),
            resumed_from_preemption=list(legacy_resumed),
            new_block_ids=list(new_block_ids),
        )
    else:
        cached = SimpleNamespace(
            req_ids=list(cached_req_ids),
            resumed_req_ids=set(resumed_req_ids),
            new_block_ids=list(new_block_ids),
        )
    return SimpleNamespace(
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=dict(num_scheduled or {}),
    )


def make_locations(n):
    return [{"location_specs": [{"name": "tp0_g0", "uri": f"file://blk{i}"}]}
            for i in range(n)]


# --------------------------------------------------------------------------- #
# get_num_new_matched_tokens
# --------------------------------------------------------------------------- #
class TestGetNumNewMatchedTokens(unittest.TestCase):
    MBS = 16

    def _run(self, prompt_len, num_computed, num_locations):
        conn = make_scheduler_connector(
            mbs=self.MBS, locations=make_locations(num_locations))
        req = FakeRequest("r0", list(range(prompt_len)))
        matched, async_load = conn.get_num_new_matched_tokens(req, num_computed)
        return conn, matched, async_load

    def test_partial_hit_no_cap(self):
        conn, matched, async_load = self._run(4 * self.MBS + 5, 0, 4)
        self.assertEqual(matched, 4 * self.MBS)
        self.assertTrue(async_load)  # a pending load is reported as async
        self.assertEqual(conn._waiting_to_load_requests[0].manager_block_idxes,
                         [0, 1, 2, 3])

    def test_full_hit_capped_to_leave_one_token(self):
        # Prompt is exactly 4 manager blocks, all externally cached: the last
        # block must be dropped so vLLM still schedules >= 1 new token.
        conn, matched, _ = self._run(4 * self.MBS, 0, 4)
        self.assertEqual(matched, 3 * self.MBS)
        self.assertEqual(conn._waiting_to_load_requests[0].manager_block_idxes,
                         [0, 1, 2])
        # has_saved_block_num counts only the blocks actually treated as hit.
        self.assertEqual(conn._alive_requests["r0"].has_saved_block_num, 3)

    def test_full_hit_with_local_prefix(self):
        # 2 blocks locally computed + 2 remote = whole prompt -> drop one remote.
        conn, matched, _ = self._run(4 * self.MBS, 2 * self.MBS, 2)
        self.assertEqual(matched, self.MBS)
        self.assertEqual(conn._waiting_to_load_requests[0].manager_block_idxes, [2])

    def test_single_block_full_hit_degrades_to_zero(self):
        conn, matched, async_load = self._run(self.MBS, 0, 1)
        self.assertEqual(matched, 0)
        self.assertFalse(async_load)
        self.assertEqual(conn._waiting_to_load_requests, [])

    def test_no_locations(self):
        conn, matched, async_load = self._run(100, 0, 0)
        self.assertEqual(matched, 0)
        self.assertFalse(async_load)

    def test_requery_after_load_attempt_skips_external(self):
        # A request that already went through an external load (blocks were
        # allocated) and returned to WAITING -- KV load failure with
        # policy=recompute, or preemption -- must not re-match: the manager may
        # still advertise blocks whose storage is gone, and re-matching loops
        # fail -> reschedule forever.
        conn = make_scheduler_connector(mbs=self.MBS, locations=make_locations(2))
        req = FakeRequest("r0", list(range(4 * self.MBS + 5)))
        matched, _ = conn.get_num_new_matched_tokens(req, 0)
        self.assertEqual(matched, 2 * self.MBS)
        # vLLM allocates blocks for the load attempt.
        conn.update_state_after_alloc(
            req, SimpleNamespace(get_block_ids=lambda: [[100, 101, 102]]), matched)
        # Retry: same request re-enters the waiting queue with 0 computed.
        matched2, async2 = conn.get_num_new_matched_tokens(req, 0)
        self.assertEqual(matched2, 0)
        self.assertFalse(async2)
        self.assertEqual(len(conn._waiting_to_load_requests), 1)  # no new load
        state = conn._alive_requests["r0"]
        self.assertEqual(state.remote_matched_token_num, 0)
        self.assertEqual(state.has_saved_block_num, 0)


# --------------------------------------------------------------------------- #
# parse_block_mask_to_save_indices
# --------------------------------------------------------------------------- #
class TestParseBlockMask(unittest.TestCase):
    def setUp(self):
        self.conn = make_connector()

    def test_offset_branch(self):
        resp = {"block_mask": {"offset": 2}}
        self.assertEqual(
            self.conn.parse_block_mask_to_save_indices(resp, 5), [2, 3, 4])

    def test_offset_zero(self):
        resp = {"block_mask": {"offset": 0}}
        self.assertEqual(
            self.conn.parse_block_mask_to_save_indices(resp, 3), [0, 1, 2])

    def test_bool_masks_branch(self):
        resp = {"block_mask": {"bool_masks": {"values": [True, False, True, False]}}}
        self.assertEqual(
            self.conn.parse_block_mask_to_save_indices(resp, 4), [1, 3])

    def test_missing_mask(self):
        self.assertEqual(self.conn.parse_block_mask_to_save_indices({}, 3), [])


# --------------------------------------------------------------------------- #
# _parse_groups
# --------------------------------------------------------------------------- #
class TestParseGroups(unittest.TestCase):
    def _kv_cache_config(self, groups):
        return SimpleNamespace(kv_cache_groups=groups)

    def _attn_group(self, layers, block_size=16, page_size_bytes=32768):
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        return SimpleNamespace(
            layer_names=layers,
            kv_cache_spec=FullAttentionSpec(block_size, page_size_bytes))

    def _mamba_group(self, layers, block_size=528, page_size_bytes=1024):
        from vllm.v1.kv_cache_interface import MambaSpec
        return SimpleNamespace(
            layer_names=layers,
            kv_cache_spec=MambaSpec(block_size, page_size_bytes))

    def test_full_attention_single_group(self):
        conn = make_connector(manager_block_size=32)
        metas = conn._parse_groups(self._kv_cache_config(
            [self._attn_group(["l0", "l1"], block_size=16, page_size_bytes=32768)]))
        self.assertEqual(len(metas), 1)
        m = metas[0]
        self.assertTrue(m.is_attention)
        self.assertEqual(m.group_idx, 0)
        self.assertEqual(m.block_size, 16)
        # per_token = 32768 // 16 = 2048; per_block = 2048 * 32 (manager) * 2 layers
        self.assertEqual(m.per_block_bytes, 2048 * 32 * 2)

    def test_hybrid_multi_group(self):
        conn = make_connector(manager_block_size=528)
        metas = conn._parse_groups(self._kv_cache_config([
            self._mamba_group(["m0", "m1"], page_size_bytes=1000),
            self._mamba_group(["m2"], page_size_bytes=2000),
            self._attn_group(["a0"], block_size=528, page_size_bytes=528 * 64),
        ]))
        self.assertEqual([m.group_idx for m in metas], [0, 1, 2])
        self.assertEqual([m.is_attention for m in metas], [False, False, True])
        self.assertEqual(metas[0].per_block_bytes, 1000 * 2)  # page * layers
        self.assertEqual(metas[1].per_block_bytes, 2000)
        self.assertEqual(metas[2].per_block_bytes, 64 * 528)  # per_token * mbs

    def test_eagle_group_skipped(self):
        conn = make_connector()
        eagle = self._attn_group(["drafter"])
        eagle.is_eagle_group = True
        metas = conn._parse_groups(self._kv_cache_config(
            [eagle, self._attn_group(["a0"])]))
        self.assertEqual(len(metas), 1)
        self.assertEqual(metas[0].layer_names, ["a0"])
        self.assertEqual(metas[0].group_idx, 1)  # group_idx keeps vLLM numbering

    def test_unsupported_spec_raises(self):
        conn = make_connector()
        bad = SimpleNamespace(layer_names=["x"], kv_cache_spec=object())
        with self.assertRaises(NotImplementedError):
            conn._parse_groups(self._kv_cache_config([bad]))

    def test_no_usable_groups_asserts(self):
        conn = make_connector()
        with self.assertRaises(AssertionError):
            conn._parse_groups(self._kv_cache_config([]))


# --------------------------------------------------------------------------- #
# build_connector_meta
# --------------------------------------------------------------------------- #
class TestBuildConnectorMeta(unittest.TestCase):
    MBS = 16

    def _new_request(self, conn, req_id, num_tokens, num_blocks,
                     num_locations=0):
        """Simulate the scheduler flow for a fresh request: query, alloc, then
        one build_connector_meta step."""
        conn._location_query_manager.get_locations_for_query.return_value = (
            True, make_locations(num_locations))
        req = FakeRequest(req_id, list(range(num_tokens)))
        conn.get_num_new_matched_tokens(req, 0)
        block_ids = [list(range(100, 100 + num_blocks))]
        conn.update_state_after_alloc(
            req, SimpleNamespace(get_block_ids=lambda: block_ids), 0)
        out = fake_scheduler_output(
            new_reqs=[SimpleNamespace(req_id=req_id, block_ids=block_ids)])
        return req, conn.build_connector_meta(out)

    def test_new_request_full_state(self):
        conn = make_scheduler_connector(mbs=self.MBS)
        req, meta = self._new_request(conn, "r0", 40, 3)
        self.assertEqual(len(meta.requests), 1)
        delta = meta.requests[0]
        self.assertFalse(delta.is_delta)
        self.assertEqual(delta.new_tokens_ids, list(range(40)))
        self.assertEqual(delta.new_block_ids_per_group, [[100, 101, 102]])
        # 40 tokens / 3 blocks -> min(40, 48)//16 = 2 blocks to save.
        conn._http_executor.submit.assert_called_once()
        args = conn._http_executor.submit.call_args[0]
        self.assertEqual(args[1:], ("r0", list(range(32)), 2))
        self.assertEqual(conn._alive_requests["r0"].has_saved_block_num, 2)

    def test_load_request_emitted_after_alloc(self):
        conn = make_scheduler_connector(mbs=self.MBS)
        req, meta = self._new_request(conn, "r0", 40, 3, num_locations=2)
        self.assertEqual(len(meta.to_load_requests), 1)
        lr = meta.to_load_requests[0]
        self.assertEqual(lr.manager_block_idxes, [0, 1])
        self.assertEqual(lr.all_block_ids, [[100, 101, 102]])
        # Externally hit blocks are not re-saved.
        conn._http_executor.submit.assert_not_called()

    def test_cached_delta_with_and_without_new_blocks(self):
        conn = make_scheduler_connector(mbs=self.MBS)
        req, _ = self._new_request(conn, "r0", 40, 3)
        # Step 2: 8 decode tokens, no new blocks (PR #23262: may be None).
        req.output_token_ids = list(range(1000, 1008))
        out = fake_scheduler_output(
            cached_req_ids=["r0"], num_scheduled={"r0": 8}, new_block_ids=[None])
        meta = conn.build_connector_meta(out)
        delta = meta.requests[0]
        self.assertTrue(delta.is_delta)
        self.assertEqual(delta.new_tokens_ids, list(range(1000, 1008)))
        self.assertEqual(delta.new_block_ids_per_group, [])
        # Step 3: 2 more tokens with a new block -> table grows.
        req.output_token_ids = list(range(1000, 1010))
        out = fake_scheduler_output(
            cached_req_ids=["r0"], num_scheduled={"r0": 2},
            new_block_ids=[[[103]]])
        meta = conn.build_connector_meta(out)
        self.assertEqual(meta.requests[0].new_block_ids_per_group, [[103]])
        self.assertEqual(conn._alive_requests["r0"].block_ids_per_group,
                         [[100, 101, 102, 103]])

    def _preempted_step(self, conn, req, use_legacy):
        kwargs = dict(cached_req_ids=["r0"], num_scheduled={"r0": 0},
                      new_block_ids=[[[200, 201]]])
        if use_legacy:
            kwargs["legacy_resumed"] = [True]
        else:
            kwargs["resumed_req_ids"] = {"r0"}
        return conn.build_connector_meta(fake_scheduler_output(**kwargs))

    def test_resumed_from_preemption_both_interfaces(self):
        for use_legacy in (False, True):
            with self.subTest(legacy=use_legacy):
                conn = make_scheduler_connector(mbs=self.MBS)
                req, _ = self._new_request(conn, "r0", 40, 3)
                meta = self._preempted_step(conn, req, use_legacy)
                delta = meta.requests[0]
                self.assertTrue(delta.resumed_from_preemption)
                # Resume replaces (not extends) the block table.
                self.assertEqual(conn._alive_requests["r0"].block_ids_per_group,
                                 [[200, 201]])

    def test_save_threshold_grows_incrementally(self):
        conn = make_scheduler_connector(mbs=self.MBS)
        req, _ = self._new_request(conn, "r0", 40, 3)  # saved 2 blocks
        conn._http_executor.submit.reset_mock()
        # 8 more tokens -> 48 total, table full at 3 blocks -> third block saves.
        req.output_token_ids = list(range(1000, 1008))
        out = fake_scheduler_output(
            cached_req_ids=["r0"], num_scheduled={"r0": 8}, new_block_ids=[[[103]]])
        conn.build_connector_meta(out)
        args = conn._http_executor.submit.call_args[0]
        self.assertEqual(args[3], 3)  # target_save_num
        self.assertEqual(conn._alive_requests["r0"].has_saved_block_num, 3)

    def test_save_request_drain_and_finish_paths(self):
        conn = make_scheduler_connector(mbs=self.MBS)
        req, _ = self._new_request(conn, "r0", 40, 3)
        state = conn._alive_requests["r0"]
        self.assertEqual(state.scheduled_saving_count, 1)

        # Finish while the save is still in flight: request must stay alive.
        keep, extra = conn.request_finished(req, [])
        self.assertTrue(keep)
        self.assertTrue(state.need_report_after_saving_finished)
        self.assertIn("r0", conn._alive_requests)

        # The async save lands: drained into to_save_requests and, because the
        # request already finished, a FinishRequest is emitted and state dropped.
        with conn._waiting_to_save_requests_lock:
            conn._waiting_to_save_requests.append(
                SaveRequest("r0", make_locations(2), [0, 1], "sess"))
        meta = conn.build_connector_meta(fake_scheduler_output())
        self.assertEqual(len(meta.to_save_requests), 1)
        self.assertEqual([f.req_id for f in meta.to_finish_requests], ["r0"])
        self.assertNotIn("r0", conn._alive_requests)

    def test_request_finished_when_saves_landed(self):
        conn = make_scheduler_connector(mbs=self.MBS)
        req, _ = self._new_request(conn, "r0", 40, 3)
        with conn._waiting_to_save_requests_lock:
            conn._waiting_to_save_requests.append(
                SaveRequest("r0", make_locations(2), [0, 1], "sess"))
        conn.build_connector_meta(fake_scheduler_output())
        keep, extra = conn.request_finished(req, [])
        self.assertTrue(keep)
        self.assertNotIn("r0", conn._alive_requests)
        meta = conn.build_connector_meta(fake_scheduler_output())
        self.assertEqual([f.req_id for f in meta.to_finish_requests], ["r0"])

    def test_canceled_save_unknown_request_no_crash(self):
        # Cancellations arrive from http_executor threads and may race request
        # teardown; an unknown req_id must be skipped, not KeyError.
        conn = make_scheduler_connector(mbs=self.MBS)
        with conn._canceled_save_request_ids_lock:
            conn._canceled_save_request_ids.append("ghost")
        conn.build_connector_meta(fake_scheduler_output())  # must not raise

    def test_canceled_save_finishes_request(self):
        conn = make_scheduler_connector(mbs=self.MBS)
        req, _ = self._new_request(conn, "r0", 40, 3)
        conn.request_finished(req, [])  # save in flight -> delayed finish
        with conn._canceled_save_request_ids_lock:
            conn._canceled_save_request_ids.append("r0")
        meta = conn.build_connector_meta(fake_scheduler_output())
        self.assertEqual([f.req_id for f in meta.to_finish_requests], ["r0"])
        self.assertNotIn("r0", conn._alive_requests)


if __name__ == "__main__":
    unittest.main()
