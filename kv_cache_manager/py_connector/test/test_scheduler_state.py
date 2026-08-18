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
    make_connector, make_scheduler_core, ReqState, GroupMeta)
from kv_cache_manager.py_connector.vllm.vllm_common import (
    AttentionGroupMeta, StateGroupMeta, parse_groups)
from kv_cache_manager.py_connector.vllm.v1_connector import TairKvCacheConnector
from kv_cache_manager.py_connector.vllm.metadata import (
    SaveRequest, LoadRequest, TairKvCacheConnectorMetadata)


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


def make_scheduler_connector(mbs=16, vllm_bs=None, locations=None,
                             num_groups=1, num_state_groups=0, tp_size=1):
    """SchedulerCore with the scheduler-loop state and a mocked query manager."""
    return make_scheduler_core(
        manager_block_size=mbs, vllm_block_size=vllm_bs,
        num_groups=num_groups, num_state_groups=num_state_groups,
        tp_size=tp_size, locations=locations if locations is not None else [])


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

    def test_query_in_flight_answers_none(self):
        # The manager query runs async; until it lands the connector answers
        # None, vLLM re-asks next step instead of blocking the scheduler loop.
        conn = make_scheduler_connector(mbs=self.MBS)
        conn._location_query_manager.get_locations_for_query.return_value = None
        req = FakeRequest("r0", list(range(4 * self.MBS + 5)))
        matched, async_load = conn.get_num_new_matched_tokens(req, 0)
        self.assertIsNone(matched)
        self.assertFalse(async_load)
        self.assertEqual(conn._waiting_to_load_requests, [])
        # Registration happens when the query lands, not while it is pending.
        self.assertNotIn("r0", conn._alive_requests)

    def test_requery_after_load_failure_skips_external(self):
        # A failed load comes back to the scheduler as invalid block ids
        # (update_connector_output); the re-query under kv_load_failure_policy
        # =recompute must not re-match: the manager still advertises the
        # blocks whose bytes are gone, and re-matching loops
        # fail -> reschedule forever.
        conn = make_scheduler_connector(mbs=self.MBS, locations=make_locations(2))
        req = FakeRequest("r0", list(range(4 * self.MBS + 5)))
        matched, _ = conn.get_num_new_matched_tokens(req, 0)
        self.assertEqual(matched, 2 * self.MBS)
        # vLLM allocates blocks for the load attempt; the load then fails.
        conn.update_state_after_alloc(
            req, SimpleNamespace(get_block_ids=lambda: [[100, 101, 102]]), matched)
        conn.update_connector_output(SimpleNamespace(invalid_block_ids={101}))
        self.assertTrue(conn._alive_requests["r0"].load_failed)
        # Retry: same request re-enters the waiting queue with 0 computed.
        matched2, async2 = conn.get_num_new_matched_tokens(req, 0)
        self.assertEqual(matched2, 0)
        self.assertFalse(async2)
        self.assertEqual(len(conn._waiting_to_load_requests), 1)  # no new load
        state = conn._alive_requests["r0"]
        self.assertEqual(state.remote_matched_token_num, 0)
        self.assertEqual(state.has_saved_block_num, 0)

    def test_preempted_requery_keeps_external_match(self):
        # Preemption alone is not a failure: the loaded KV is healthy, only
        # the scheduling position was lost, so the re-query matches again
        # (full-attention models only -- hybrid cannot tell failure from
        # preemption, see test_hybrid_load_attempt_burns_the_match).
        conn = make_scheduler_connector(mbs=self.MBS, locations=make_locations(2))
        req = FakeRequest("r0", list(range(4 * self.MBS + 5)))
        matched, _ = conn.get_num_new_matched_tokens(req, 0)
        conn.update_state_after_alloc(
            req, SimpleNamespace(get_block_ids=lambda: [[100, 101, 102]]), matched)
        conn.update_connector_output(SimpleNamespace(invalid_block_ids=set()))
        matched2, _ = conn.get_num_new_matched_tokens(req, 0)
        self.assertEqual(matched2, matched)  # re-matched, not fast-failed
        self.assertEqual(len(conn._waiting_to_load_requests), 2)

    def test_hybrid_load_attempt_burns_the_match(self):
        # Hybrid load failures cannot be reported to vLLM (single-group
        # invalid-block recovery only), so no explicit signal exists: any
        # allocation for an external hint burns the match conservatively.
        conn = make_scheduler_connector(
            mbs=self.MBS, num_groups=1, num_state_groups=1,
            locations=hybrid_locations([True, True]))
        req = FakeRequest("r0", list(range(4 * self.MBS + 5)))
        matched, _ = conn.get_num_new_matched_tokens(req, 0)
        self.assertEqual(matched, 2 * self.MBS)
        conn.update_state_after_alloc(
            req, SimpleNamespace(get_block_ids=lambda: [[100, 101], [50, 51]]),
            matched)
        # Even with no failure signal at all, the re-query fast-fails.
        matched2, async2 = conn.get_num_new_matched_tokens(req, 0)
        self.assertEqual(matched2, 0)
        self.assertFalse(async2)

    def test_load_attempted_flag_tracks_the_one_external_load(self):
        # load_attempted is explicit state, not inference from other fields:
        # set exactly when vLLM allocates blocks for an external hit. It only
        # burns the re-query for hybrid requests (which get no failure
        # signal); full-attention requests burn theirs through load_failed.
        conn = make_scheduler_connector(mbs=self.MBS, locations=make_locations(2))
        req = FakeRequest("r0", list(range(4 * self.MBS + 5)))
        conn.get_num_new_matched_tokens(req, 0)
        # Query done, but no allocation yet (vLLM may still re-ask us).
        self.assertFalse(conn._alive_requests["r0"].load_attempted)
        # An allocation with no external hit is not a load attempt.
        conn.update_state_after_alloc(
            req, SimpleNamespace(get_block_ids=lambda: [[100]]), 0)
        self.assertFalse(conn._alive_requests["r0"].load_attempted)
        # Blocks allocated for an external hit: the load attempt begins.
        conn.update_state_after_alloc(
            req, SimpleNamespace(get_block_ids=lambda: [[100, 101, 102]]), 2 * self.MBS)
        self.assertTrue(conn._alive_requests["r0"].load_attempted)


# --------------------------------------------------------------------------- #
# Per-block spec coverage (hybrid sparse recurrent state)
# --------------------------------------------------------------------------- #
def hybrid_locations(coverage, tp_size=1, num_attn=1, num_state=1):
    """Manager locations whose per-block spec set encodes ``coverage``:
    True -> every group's spec (state included), False -> attention specs only.
    """
    locs = []
    for complete in coverage:
        names = [f"tp{r}_g{g}" for r in range(tp_size) for g in range(num_attn)]
        if complete:
            names += [f"tp{r}_g{num_attn + g}"
                      for r in range(tp_size) for g in range(num_state)]
        locs.append({"location_specs": [{"name": n, "uri": f"u_{n}"}
                                        for n in names]})
    return locs


class TestSpecGroups(unittest.TestCase):
    """Registration must advertise the two spec groups a hybrid model needs to
    express per-block state sparsity -- and must stay silent for models that
    have no sparsity (byte-identical requests, old-manager compatible)."""

    def test_full_attention_declares_no_groups(self):
        conn = make_scheduler_core(num_groups=1, tp_size=2)
        self.assertEqual(conn._spec_groups(), [])

    def test_hybrid_declares_attn_and_full(self):
        conn = make_scheduler_core(num_groups=1, num_state_groups=2, tp_size=2)
        groups = {g["name"]: g["spec_names"] for g in conn._spec_groups()}
        self.assertEqual(sorted(groups), ["attn", "full"])
        # attn: the attention spec of every rank; full: every group of every rank.
        self.assertEqual(groups["attn"], ["tp0_g0", "tp1_g0"])
        self.assertEqual(groups["full"],
                         ["tp0_g0", "tp0_g1", "tp0_g2",
                          "tp1_g0", "tp1_g1", "tp1_g2"])


class TestStateCompleteMask(unittest.TestCase):
    """Which manager blocks have a materialized recurrent state is read from
    vLLM's block table: a state group pointing at the null block (id 0) has
    none. This mask is what start_write_cache announces per key."""

    def _req(self, tables):
        return ReqState(req_id="r0", token_ids=[], block_ids_per_group=tables,
                        has_saved_block_num=0, local_matched_token_num=0,
                        remote_matched_token_num=0, vllm_request=None)

    def test_full_attention_is_always_complete(self):
        conn = make_scheduler_core(manager_block_size=16, num_groups=1)
        req = self._req([[7, 0, 9]])
        self.assertEqual(conn._state_complete_mask(req, range(3)),
                         [True, True, True])

    def test_null_state_blocks_are_incomplete(self):
        conn = make_scheduler_core(manager_block_size=16, num_groups=1,
                                   num_state_groups=1)
        # State table: blocks 0 and 2 are null (no state), block 1 is real.
        req = self._req([[100, 101, 102], [0, 55, 0]])
        self.assertEqual(conn._state_complete_mask(req, range(3)),
                         [False, True, False])

    def test_all_state_groups_must_have_state(self):
        conn = make_scheduler_core(manager_block_size=16, num_groups=1,
                                   num_state_groups=2)
        # Block 1 has a state in group 1 but not in group 2 -> incomplete.
        req = self._req([[100, 101], [7, 8], [7, 0]])
        self.assertEqual(conn._state_complete_mask(req, range(2)),
                         [True, False])

    def test_short_state_table_is_incomplete(self):
        # A state table that does not reach the block cannot prove a state.
        conn = make_scheduler_core(manager_block_size=16, num_groups=1,
                                   num_state_groups=1)
        req = self._req([[100, 101], [55]])
        self.assertEqual(conn._state_complete_mask(req, range(2)),
                         [True, False])


class TestExternalHitTruncation(unittest.TestCase):
    """A hybrid request can only resume where the recurrent state ends, so an
    external match must be cut back to the last state-complete block. The
    attention KV of a longer prefix is worthless without that state."""

    MBS = 16

    def _matched(self, coverage, prompt_len=None, num_state_groups=1,
                 tp_size=1):
        conn = make_scheduler_connector(
            mbs=self.MBS, num_state_groups=num_state_groups, tp_size=tp_size,
            locations=hybrid_locations(coverage, tp_size=tp_size,
                                       num_state=num_state_groups))
        req = FakeRequest("r0", list(range(prompt_len or
                                          (len(coverage) + 2) * self.MBS)))
        matched, _ = conn.get_num_new_matched_tokens(req, 0)
        return conn, matched

    def test_truncates_to_last_state_complete_block(self):
        conn, matched = self._matched([True, True, False, False])
        self.assertEqual(matched, 2 * self.MBS)
        self.assertEqual(conn._waiting_to_load_requests[0].manager_block_idxes,
                         [0, 1])

    def test_interior_gap_is_kept(self):
        # Only the *end* of the match must carry state; earlier state-less
        # blocks are fine (their state is never read).
        conn, matched = self._matched([True, False, True, False])
        self.assertEqual(matched, 3 * self.MBS)
        self.assertEqual(conn._waiting_to_load_requests[0].manager_block_idxes,
                         [0, 1, 2])

    def test_no_state_anywhere_drops_the_match(self):
        conn, matched = self._matched([False, False, False])
        self.assertEqual(matched, 0)
        self.assertEqual(conn._waiting_to_load_requests, [])

    def test_all_complete_is_untouched(self):
        conn, matched = self._matched([True, True, True])
        self.assertEqual(matched, 3 * self.MBS)

    def test_every_rank_must_have_the_state(self):
        # tp2: block 1 has the state spec of rank 0 only -- rank 1 would read
        # nothing, so the block cannot end the match.
        conn = make_scheduler_connector(mbs=self.MBS, num_state_groups=1,
                                        tp_size=2)
        locs = hybrid_locations([True, True], tp_size=2, num_state=1)
        locs[1]["location_specs"] = [
            s for s in locs[1]["location_specs"] if s["name"] != "tp1_g1"]
        conn._location_query_manager.get_locations_for_query.return_value = locs
        req = FakeRequest("r0", list(range(6 * self.MBS)))
        matched, _ = conn.get_num_new_matched_tokens(req, 0)
        self.assertEqual(matched, self.MBS)

    def test_full_attention_never_truncates(self):
        # Single group: the state specs do not exist, so coverage is uniform
        # and the match must be untouched (no hit-rate regression).
        conn = make_scheduler_connector(mbs=self.MBS, locations=make_locations(3))
        req = FakeRequest("r0", list(range(6 * self.MBS)))
        matched, _ = conn.get_num_new_matched_tokens(req, 0)
        self.assertEqual(matched, 3 * self.MBS)

    def test_truncation_runs_before_the_full_hit_cap(self):
        # Prompt is exactly 3 blocks; coverage allows 3 but the last carries no
        # state -> truncate to 2, and the full-hit cap then has nothing to drop.
        conn, matched = self._matched([True, True, False],
                                      prompt_len=3 * self.MBS)
        self.assertEqual(matched, 2 * self.MBS)

    def test_full_hit_cap_retruncates_to_state_complete(self):
        # The cap drops trailing blocks without looking at their coverage, so
        # it can move the match end onto a state-less block: coverage
        # [True, False, True] truncates to all 3, the cap (prompt == 3 blocks)
        # drops block 2, and the new match end (block 1) has no state. The
        # match must be re-truncated -- loading it would end a hybrid request
        # on a state nobody wrote, unreportably (report_failures=False).
        conn, matched = self._matched([True, False, True],
                                      prompt_len=3 * self.MBS)
        self.assertEqual(matched, self.MBS)
        self.assertEqual(conn._waiting_to_load_requests[0].manager_block_idxes,
                         [0])
        end = max(conn._waiting_to_load_requests[0].manager_block_idxes)
        self.assertTrue(conn._location_covers_states(
            conn._waiting_to_load_requests[0].need_load_locations[end]))


class TestStartWriteCacheSpecGroups(unittest.TestCase):
    """start_write_cache must tell the manager, per key, which specs the block
    will really hold -- that is how "no state here" becomes visible instead of
    being encoded as a successful write."""

    def _conn(self, num_state_groups):
        conn = make_scheduler_connector(mbs=16, num_state_groups=num_state_groups)
        conn._extra_config = SimpleNamespace(
            instance_id="inst", write_timeout_seconds=30)
        conn._manager_client = MagicMock()
        conn._manager_client.start_write_cache.return_value = {
            "locations": [], "write_session_id": "sess"}
        return conn

    def _request_sent(self, conn):
        (req,), _ = conn._manager_client.start_write_cache.call_args
        return req

    def test_hybrid_sends_per_key_group_names(self):
        conn = self._conn(num_state_groups=1)
        conn.start_save_kvcache_async("r0", list(range(48)), 3,
                                      [True, False, True])
        self.assertEqual(self._request_sent(conn)["location_spec_group_names"],
                         ["full", "attn", "full"])

    def test_full_attention_omits_group_names(self):
        conn = self._conn(num_state_groups=0)
        conn.start_save_kvcache_async("r0", list(range(32)), 2, [True, True])
        self.assertNotIn("location_spec_group_names", self._request_sent(conn))

    def test_mask_length_is_checked(self):
        conn = self._conn(num_state_groups=1)
        with self.assertRaises(AssertionError):
            conn.start_save_kvcache_async("r0", list(range(48)), 3, [True])


# --------------------------------------------------------------------------- #
# parse_block_mask_to_save_indices
# --------------------------------------------------------------------------- #
class TestParseBlockMask(unittest.TestCase):
    def setUp(self):
        self.conn = make_scheduler_core(manager_block_size=16)

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

    def _parse(self, groups, mbs):
        return parse_groups(self._kv_cache_config(groups), mbs)

    def _attn_group(self, layers, block_size=16, page_size_bytes=32768,
                    page_size_padded=None):
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        return SimpleNamespace(
            layer_names=layers,
            kv_cache_spec=FullAttentionSpec(block_size, page_size_bytes,
                                            page_size_padded=page_size_padded))

    def _mamba_group(self, layers, block_size=528, page_size_bytes=1024):
        from vllm.v1.kv_cache_interface import MambaSpec
        return SimpleNamespace(
            layer_names=layers,
            kv_cache_spec=MambaSpec(block_size, page_size_bytes))

    def test_full_attention_single_group(self):
        mbs = 32
        metas = self._parse(
            [self._attn_group(["l0", "l1"], block_size=16, page_size_bytes=32768)], mbs)
        self.assertEqual(len(metas), 1)
        m = metas[0]
        self.assertIsInstance(m, AttentionGroupMeta)
        self.assertEqual(m.group_idx, 0)
        self.assertEqual(m.block_size, 16)
        # per_token = 32768 // 16 = 2048; per_block = 2048 * 32 (manager) * 2 layers
        self.assertEqual(m.per_block_bytes, 2048 * 32 * 2)

    def test_hybrid_multi_group(self):
        mbs = 528
        metas = self._parse([
            self._mamba_group(["m0", "m1"], page_size_bytes=1000),
            self._mamba_group(["m2"], page_size_bytes=2000),
            self._attn_group(["a0"], block_size=528, page_size_bytes=528 * 64),
        ], mbs)
        self.assertEqual([m.group_idx for m in metas], [0, 1, 2])
        self.assertEqual([type(m).__name__ for m in metas],
                         ['StateGroupMeta', 'StateGroupMeta', 'AttentionGroupMeta'])
        self.assertEqual(metas[0].per_block_bytes, 1000 * 2)  # page * layers
        self.assertEqual(metas[1].per_block_bytes, 2000)
        self.assertEqual(metas[2].per_block_bytes, 64 * 528)  # per_token * mbs

    def test_eagle_group_skipped(self):
        mbs = 16
        eagle = self._attn_group(["drafter"])
        eagle.is_eagle_group = True
        metas = self._parse(
            [eagle, self._attn_group(["a0"])], mbs)
        self.assertEqual(len(metas), 1)
        self.assertEqual(metas[0].layer_names, ["a0"])
        self.assertEqual(metas[0].group_idx, 1)  # group_idx keeps vLLM numbering

    def test_padded_attention_page_uses_compact_size(self):
        # page_size_padded inflates spec.page_size_bytes with an allocation
        # gap the gather kernel never copies; per_block_bytes must come from
        # the compact real_page_size_bytes.
        mbs = 16
        metas = self._parse(
            [self._attn_group(["l0", "l1"], block_size=16,
                              page_size_bytes=32768, page_size_padded=40960)], mbs)
        # per_token = 32768 // 16 = 2048 (not 40960 // 16 = 2560).
        self.assertEqual(metas[0].per_block_bytes, 2048 * 16 * 2)

    def test_padded_attention_without_compact_size_raises(self):
        # A padded spec that exposes no real_page_size_bytes cannot be sized
        # correctly -- must refuse, not silently over-allocate.
        mbs = 16
        group = self._attn_group(["l0"], block_size=16,
                                 page_size_bytes=32768, page_size_padded=40960)
        del group.kv_cache_spec.real_page_size_bytes
        with self.assertRaises(NotImplementedError):
            self._parse([group], mbs)

    def test_unpadded_attention_without_compact_size_falls_back(self):
        # No padding + no real_page_size_bytes: page_size_bytes is already
        # compact, use it.
        mbs = 16
        group = self._attn_group(["l0"], block_size=16, page_size_bytes=32768)
        del group.kv_cache_spec.real_page_size_bytes
        metas = self._parse([group], mbs)
        self.assertEqual(metas[0].per_block_bytes, 2048 * 16)

    def test_windowed_attention_spec_rejected(self):
        # vLLM can merge SWA / chunked-attention layers into a
        # FullAttentionSpec that keeps sliding_window / attention_chunk_size
        # set; such blocks are not full-prefix KV and must be refused.
        for window_field in ("sliding_window", "attention_chunk_size"):
            with self.subTest(field=window_field):
                mbs = 16
                group = self._attn_group(["l0"])
                setattr(group.kv_cache_spec, window_field, 1024)
                with self.assertRaises(NotImplementedError) as cm:
                    self._parse([group], mbs)
                self.assertIn(window_field, str(cm.exception))

    def test_windowed_fields_none_accepted(self):
        # Real FullAttentionSpec objects carry the fields as None; that is the
        # ordinary full-attention case and must still parse.
        mbs = 16
        group = self._attn_group(["l0"])
        group.kv_cache_spec.sliding_window = None
        group.kv_cache_spec.attention_chunk_size = None
        metas = self._parse([group], mbs)
        self.assertEqual(len(metas), 1)

    def test_unsupported_spec_raises(self):
        mbs = 16
        bad = SimpleNamespace(layer_names=["x"], kv_cache_spec=object())
        with self.assertRaises(NotImplementedError):
            self._parse([bad], mbs)

    def test_no_usable_groups_asserts(self):
        mbs = 16
        with self.assertRaises(AssertionError):
            self._parse([], mbs)


# --------------------------------------------------------------------------- #
# Skipped (EAGLE/MTP drafter) groups: block tables indexed by vLLM group idx
# --------------------------------------------------------------------------- #
class TestSkippedGroupIndexing(unittest.TestCase):
    """When parse_groups skips a group (EAGLE/MTP drafter), its block table is
    still present in block_ids_per_group / all_block_ids at its vLLM group
    index. Consumers must index by GroupMeta.group_idx, never assume the
    transferred groups start at 0 or include every table."""

    MBS = 16

    def _skipped_group0_connector(self):
        """SchedulerCore where vLLM group 0 is a skipped drafter and group 1
        is the transferred attention group."""
        conn = make_scheduler_core(manager_block_size=self.MBS)
        conn._group_metas = [AttentionGroupMeta(
            group_idx=1, layer_names=["a0"],
            block_size=self.MBS, per_block_bytes=0)]
        conn._num_groups = 1
        return conn

    def test_num_allocated_blocks_ignores_skipped_group(self):
        conn = self._skipped_group0_connector()
        req = ReqState(
            req_id="r0", token_ids=list(range(64)),
            # Drafter table (group 0) lags with 1 block; attention has 4.
            block_ids_per_group=[[100], [200, 201, 202, 203]],
            has_saved_block_num=0, local_matched_token_num=0,
            remote_matched_token_num=0, vllm_request=None)
        self.assertEqual(conn._num_allocated_blocks(req), 4)

    def test_num_allocated_blocks_still_mins_transferred_groups(self):
        # Two transferred groups (1 and 2), one skipped drafter (0): min is
        # taken over the transferred ones only.
        conn = make_scheduler_core(manager_block_size=self.MBS)
        conn._group_metas = [
            AttentionGroupMeta(group_idx=1, layer_names=["a0"],
                               block_size=self.MBS, per_block_bytes=0),
            StateGroupMeta(group_idx=2, layer_names=["m0"],
                           block_size=self.MBS, per_block_bytes=0,
                           page_size_bytes=0),
        ]
        conn._num_groups = 2
        req = ReqState(
            req_id="r0", token_ids=[], block_ids_per_group=[[9], [1, 2, 3], [4, 5]],
            has_saved_block_num=0, local_matched_token_num=0,
            remote_matched_token_num=0, vllm_request=None)
        self.assertEqual(conn._num_allocated_blocks(req), 2)

    def test_num_allocated_blocks_empty(self):
        conn = self._skipped_group0_connector()
        req = ReqState(
            req_id="r0", token_ids=[], block_ids_per_group=[],
            has_saved_block_num=0, local_matched_token_num=0,
            remote_matched_token_num=0, vllm_request=None)
        self.assertEqual(conn._num_allocated_blocks(req), 0)

    def test_load_failure_report_uses_transferred_group_table(self):
        # start_load_kv (worker side) reports failed loads against the block
        # table of the single transferred group -- which is group 1 here, not
        # group 0.
        conn = make_connector(manager_block_size=self.MBS)
        conn._group_metas = [AttentionGroupMeta(
            group_idx=1, layer_names=["a0"],
            block_size=self.MBS, per_block_bytes=0)]
        conn._num_groups = 1
        conn._extra_config = SimpleNamespace(block_per_load_task=8)
        conn._data_transfer = MagicMock()
        conn._plan_group_transfers = MagicMock(return_value=None)
        meta = TairKvCacheConnectorMetadata(epoch=0)
        meta.add_load_request(LoadRequest(
            req_id="r0", manager_block_idxes=[0, 1],
            need_load_locations=[{"location_specs": []}] * 2,
            # Group 0 (drafter) has a lagging 1-entry table; indexing it would
            # IndexError / report the wrong block ids.
            all_block_ids=[[999], [10, 11]]))
        conn.start_load_kv(MagicMock(), meta)
        args, kwargs = conn._data_transfer.create_load_done_callback.call_args
        self.assertEqual(args[3], [10, 11])  # report_ids from group 1's table
        self.assertTrue(kwargs["report_failures"])


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
            make_locations(num_locations))
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
        # The per-block state-completeness mask is computed here, in the
        # scheduler loop, and handed to the http thread: it is read off vLLM's
        # block table, which later steps mutate.
        self.assertEqual(args[1:], ("r0", list(range(32)), 2, [True, True]))
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
