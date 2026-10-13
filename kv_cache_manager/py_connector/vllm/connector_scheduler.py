"""Scheduler side of the connector: matching, saving orchestration, finishing.

The scheduler role owns the request ledger (``_tracked``): per-request state
that compensates for information vLLM only provides inside hooks -- the
accumulated block tables (only increments arrive after the first
allocation), the save water-mark and the save-session ledger that decides
when a finished request's blocks may be freed. Entries are created on the
request's first allocation and dropped at retirement; requests that were
never allocated are never tracked.

Two small side tables carry the external-match discipline: ``_load_failed``
(requests whose load came back invalid) and ``_load_attempted`` (requests
that spent an external allocation -- relevant for hybrid models whose load
failures cannot be reported). The worker side lives in connector_worker; both
speak the vllm_common vocabulary.
"""

import threading
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient
from kv_cache_manager.py_connector.common.tp_coordinator import (
    CoordinateMsgSerializer,
    CoordinateMessage,
    SendBlockStartEvent,
    TpCoordinatorClient,
)
from kv_cache_manager.py_connector.vllm.config import TairKvCacheConnectorExtraConfig
from kv_cache_manager.py_connector.vllm.location_query_manager import (
    LocationQueryManager,
)
from kv_cache_manager.py_connector.vllm.metadata import (
    CaptureRequest,
    FinishRequest,
    LoadRequest,
    SaveRequest,
    TairKvCacheConnectorMetadata,
)
from kv_cache_manager.py_connector.vllm.vllm_common import (
    ATTN_ONLY_SPEC_GROUP,
    ALL_SPEC_GROUP,
    RETENTION_RSWA_GAP,
    AttentionGroupMeta,
    GroupMeta,
    StateGroupMeta,
    spec_group_plan,
    spec_name,
    windowed_block_has_data,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks


@dataclass
class RequestLedger:
    """Per-request scheduler state, created on the request's first allocation.

    ``vllm_request`` is the live vLLM Request object; the token stream is
    read from it on demand (all_token_ids) and only its length is tracked
    here. ``has_saved_block_num`` anchors the incremental saves: blocks are
    saved once as the computed prefix crosses manager-block boundaries."""

    vllm_request: "Request"
    # Per kv_cache_group block table, in each group's own block_size units.
    block_ids_per_group: List[List[int]]
    # Manager blocks already saved (or covered by an external hit).
    has_saved_block_num: int
    # Hit accounting reported to vLLM at finish (kv_transfer_params -> the
    # OpenAI response): local = vLLM's own prefix-cache hit when the match
    # hook last answered, remote = the external hit it returned (clamped).
    # Written on the match hook's answer paths (initial / re-ask / burned),
    # read exactly once by _finish_request. Group-agnostic by design: both
    # are token-granular totals.
    local_matched_token_num: int = 0
    remote_matched_token_num: int = 0
    # Save-session ledger: sessions started vs sessions handed to the worker.
    scheduled_saving_count: int = 0
    sent_saving_count: int = 0
    # Set when the request finished while sessions were still in flight.
    need_report_after_saving_finished: bool = False


class ConnectorScheduler:
    """State and hooks for the scheduler-role connector instance."""

    def __init__(
        self,
        extra_config: TairKvCacheConnectorExtraConfig,
        group_metas: List[GroupMeta],
        manager_block_size: int,
        vllm_block_size: int,
        tp_size: int,
        manager_client: KvCacheManagerClient,
        coordinator_client: TpCoordinatorClient,
        has_eagle_group: bool = False,
    ) -> None:
        self._extra_config = extra_config
        self._group_metas = group_metas
        self._num_groups = len(group_metas)
        self._state_group_idxs = [
            m.group_idx for m in group_metas if isinstance(m, StateGroupMeta)
        ]
        # Windowed buckets: the groups whose blocks vLLM frees while the
        # request lives. They drive the eager-capture requests and the hit
        # clamp; an instance without one keeps the pre-PR matching path.
        self._windowed_metas = [
            m
            for m in group_metas
            if isinstance(m, AttentionGroupMeta) and m.retention is not None
        ]
        # vLLM's own windowed hit rule requires one more block when EAGLE/MTP
        # drops the last matched block (_contiguous_blocks_for_hit's
        # drop_eagle_block); the clamp below mirrors it conservatively.
        self._has_eagle_group = has_eagle_group
        # (states complete, window data present) -> the spec group a block is
        # announced under. An instance with nothing sparse declares nothing and
        # sends no group names at all, exactly as before this PR.
        declared, self._spec_group_names = spec_group_plan(group_metas, tp_size)
        # A block with *no* coverage at all (no state and no window data) has
        # no group of its own: the manager needs a name for every announced
        # block, so it gets the narrowest declared one (the fewest specs to
        # allocate for bytes that will never arrive -- the block is failed by
        # the save verdict either way), which keeps every other block's precise
        # group instead of dropping the whole request to the "all specs"
        # default.
        self._narrowest_spec_group = (
            min(declared, key=lambda group: (len(group["spec_names"]), group["name"]))[
                "name"
            ]
            if declared
            else None
        )
        # (req_id, block indices) whose captures no write session will publish
        # (the manager already holds them): queued by the http threads, flushed
        # by the scheduler loop into the metadata the worker reads.
        self._pending_release_captures: List[Tuple[str, List[int]]] = []
        self._pending_release_captures_lock = threading.Lock()
        # Per request, the write sessions that may still consume captured
        # bytes, keyed by a per-request session id: each entry holds the
        # session's announced target and the blocks it will write (``None``
        # while its response is still in flight). Sessions are identified by
        # that id and not by their target because a regressed save watermark
        # can announce the same target twice, and the two sessions still need
        # distinct identities for the guard (see _blocks_other_sessions_write:
        # the manager's block mask reports an *allocation*, which a concurrent
        # session of the same request may still be filling).
        self._announced_sessions: Dict[
            str, Dict[int, Tuple[int, Optional[Set[int]]]]
        ] = {}
        self._announced_sessions_counters: Dict[str, int] = {}
        self._announced_sessions_lock = threading.Lock()
        # Per request, per block: whether the block's bytes were claimed as
        # windowed when it was first announced (see _dispatch_incremental_saves
        # and _announced_window_mask): a repeat announce of the same block --
        # the save watermark moves backwards when preemption shrinks the
        # allocation, and a failed start_write_cache is retried the same way --
        # must claim what the capture actually holds, not what the window looks
        # like by then.
        self._window_claims: Dict[str, Dict[int, bool]] = {}
        self._manager_block_size = manager_block_size
        self._vllm_block_size = vllm_block_size
        self._tp_size = tp_size
        self._manager_client = manager_client
        self._coordinator_client = coordinator_client

        self._epoch = 0
        self._http_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="kvcm_http_"
        )
        self._location_query_manager = LocationQueryManager(
            manager_client,
            self._http_executor,
            extra_config.instance_id,
            extra_config.async_get_cache_location,
            extra_config.location_query_max_age_seconds,
        )

        self._tracked: Dict[str, RequestLedger] = {}
        self._waiting_to_load_requests: List[LoadRequest] = []
        self._waiting_to_save_requests_lock = threading.Lock()
        self._waiting_to_save_requests: List[SaveRequest] = []
        self._waiting_to_finish_requests: List[FinishRequest] = []
        self._canceled_save_request_ids_lock = threading.Lock()
        self._canceled_save_request_ids: List[str] = []
        # External-match discipline; cleaned at retirement.
        self._load_failed: set = set()
        self._load_attempted: set = set()

    def shutdown(self) -> None:
        self._location_query_manager.shutdown()
        self._http_executor.shutdown(wait=False)

    # ------------------------------------------------------------------ #
    # Hybrid state coverage
    # ------------------------------------------------------------------ #
    def _group_block_size(self, group_idx: int) -> int:
        for meta in self._group_metas:
            if meta.group_idx == group_idx:
                return meta.block_size
        raise KeyError(f"no such transferred group: {group_idx}")

    def _state_complete_mask(
        self, ledger: RequestLedger, manager_block_idxes: Sequence[int]
    ) -> List[bool]:
        """Per manager block: does *every* state group hold a real state?

        vLLM's block table is the ground truth: in "align" mode a manager block
        whose state block is the null block (id 0) has no materialized state.
        Attention-only models return all True (nothing can be missing).
        """
        if not self._state_group_idxs:
            return [True] * len(manager_block_idxes)
        mbs = self._manager_block_size
        mask = []
        for mb in manager_block_idxes:
            complete = True
            for group_idx in self._state_group_idxs:
                table = ledger.block_ids_per_group[group_idx]
                # State covers the prefix ending at the block's last token.
                logical = ((mb + 1) * mbs - 1) // self._group_block_size(group_idx)
                if logical >= len(table) or table[logical] == 0:
                    complete = False
                    break
            mask.append(complete)
        return mask

    def _window_data_mask(
        self, ledger: RequestLedger, manager_block_idxes: Sequence[int]
    ) -> List[bool]:
        """Per manager block: does every windowed group still hold real slots?

        The windowed twin of ``_state_complete_mask``, read from the same block
        table in the same loop iteration: vLLM frees a windowed block by writing
        the null block into its table entry, so this predicate says exactly what
        the step's capture can copy (`windowed_block_has_data` is shared with
        the worker) *and* what the manager may be told this block carries. Read
        here rather than at write time on purpose: a later read is the stale
        ledger M1 of the SWA investigation.
        """
        return [
            windowed_block_has_data(
                self._windowed_metas,
                ledger.block_ids_per_group,
                mb,
                self._manager_block_size,
            )
            for mb in manager_block_idxes
        ]

    def _announced_window_mask(
        self, req_id: str, window_data_mask: List[bool]
    ) -> List[bool]:
        """The window coverage to announce, remembering what was claimed.

        A block that was already announced keeps the coverage of that first
        announcement as long as the request lives: its captured bytes sit in
        the worker's arena keyed to it (the window may have moved on since), so
        re-announcing it as window-less would leave bytes the manager was never
        told about -- the save would fail the block and release its only copy.
        """
        claims = self._window_claims.setdefault(req_id, {})
        announced = []
        for block, present in enumerate(window_data_mask):
            claimed = claims.get(block)
            if claimed is None:
                claims[block] = present
            elif claimed and not present:
                logger.info(
                    "req:%s block %d: keeping the windowed coverage claimed by "
                    "its first announcement (the window has moved since; its "
                    "captured bytes are still in the arena)",
                    req_id,
                    block,
                )
                present = claimed
            announced.append(present)
        return announced

    def _spec_group_for(
        self, states_complete: bool, window_present: bool
    ) -> Optional[str]:
        """The announced spec group of one block, or None when this instance
        declares none for that coverage (see ``spec_group_plan``)."""
        return self._spec_group_names.get((states_complete, window_present))

    def _num_allocated_blocks(self, ledger: RequestLedger) -> int:
        """Min allocated block-table length across the *transferred* groups.

        ``block_ids_per_group`` is indexed by the vLLM group index and includes
        groups skipped by ``parse_groups`` (EAGLE/MTP drafters). A drafter's
        block table can lag behind the target model's, so including it in the
        min would permanently understate how many blocks are saveable."""
        if not ledger.block_ids_per_group:
            return 0
        return min(
            len(ledger.block_ids_per_group[meta.group_idx])
            for meta in self._group_metas
        )

    # ------------------------------------------------------------------ #
    # External matching
    # ------------------------------------------------------------------ #
    def _location_covers_states(self, location: dict) -> bool:
        """Does this published block carry every state group's spec for a rank?

        A hybrid block saved without a materialized recurrent state is published
        under the attention-only spec group, so its location simply has no spec
        for the state groups. ``getCacheLocation`` reports the real coverage,
        which is what makes the sparsity visible here.
        """
        names = {spec.get("name") for spec in location.get("location_specs", [])}
        return all(
            spec_name(rank, meta) in names
            for rank in range(self._tp_size)
            for meta in self._group_metas
            if isinstance(meta, StateGroupMeta)
        )

    def _location_covers_windows(self, location: dict) -> bool:
        """Does this published block carry every windowed group's spec per rank?

        The windowed twin of ``_location_covers_states``: an externally matched
        prefix may only end where the window region still has real data, and the
        manager expresses "this block has no data for that group" exactly this
        way -- the block's location does not carry the spec (a block whose
        windowed bytes were never published, e.g. one that was already outside
        the window when its group was saved).
        """
        names = {spec.get("name") for spec in location.get("location_specs", [])}
        return all(
            spec_name(rank, meta) in names
            for rank in range(self._tp_size)
            for meta in self._windowed_metas
        )

    def _window_blocks(self, meta: AttentionGroupMeta) -> int:
        """Blocks of the window a hit end must have data for.

        ``cdiv(window - 1 + extra_retained, mbs)`` mirrors vLLM's own
        ``_contiguous_blocks_for_hit`` (cdiv(window - 1, block_size) plus the
        extra retained trailing tokens), one manager-block at a time; with
        EAGLE/MTP the matched tail loses one block, which vLLM expresses as
        "+1 contiguous block required", so require one more here too.

        Chunked-local is satisfied by the same upper bound: the chunk
        containing the hit end is at most ``attention_chunk_size`` tokens long,
        so its start lies inside the last cdiv(chunk - 1, mbs) blocks (the
        region is at most one block wider than vLLM needs -- a conservative
        over-requirement trims a hit, it never loads a hole).

        R-SWA gets the opposite treatment: ``RSWAManager`` inherits the
        full-attention manager (its ``get_num_skipped_tokens`` is not
        overridden) and its mask keeps the whole prefill segment visible, so a
        hit is a *prefix* whose every visible row must carry data. Requiring
        only the window tail would accept a hit with a hole in the middle --
        and the load would then either fail the block (vLLM truncating or
        failing the request) or, for a multi-group model, let visible rows read
        unloaded KV. Such a meta therefore requires the whole range instead,
        which ``_window_required_from`` expresses as "from block 0".

        The EAGLE/MTP drop is one *hash* block, i.e. vLLM's own block size in
        tokens, not one manager block: when ``preferred_block_size`` makes the
        manager blocks smaller than the scheduler's, the resumed window moves
        back by several manager blocks at once.
        """
        retention = meta.retention
        assert retention is not None
        dropped = self._vllm_block_size if self._has_eagle_group else 0
        return -(
            -(
                retention.window_tokens
                - 1
                + retention.extra_retained_tokens
                + dropped
            )
            // self._manager_block_size
        )

    def _window_required_from(self, meta: AttentionGroupMeta, end: int) -> int:
        """First manager block of ``[.., end)`` this windowed group must cover.

        The window tail for sliding / chunked-local, block 0 for R-SWA (see
        ``_window_blocks``): the caller only has to check that the run starting
        here reaches ``end``.
        """
        assert meta.retention is not None
        if meta.retention.kind == RETENTION_RSWA_GAP:
            return 0
        return max(0, end - self._window_blocks(meta))

    def _windowed_match_end(self, locations: List[dict], limit: int) -> int:
        """Largest block count <= limit a windowed match may end on.

        Two per-block predicates decide: the block's window region
        ``[max(0, e-k), e)`` must carry every windowed group's spec for every
        rank (k = the widest window, see ``_window_blocks``), and the block at
        ``e-1`` must carry every state group's spec -- the same end condition
        the full-attention path applies, evaluated jointly here because both
        depend on where the match ends.

        The bound is *not* the first hole from the end: the window constraint
        is local to the region, so a longer hit whose tail is covered is legal
        (vLLM's own windowed hit may end mid-prefix -- only the tail has to be
        there), and taking the largest such e is what keeps the historian's
        longer hits usable. A hole inside the region pushes the end down to
        the hole, and a block without state steps back one.
        """
        covered = [self._location_covers_windows(loc) for loc in locations]
        # Longest consecutive covered run *starting* at each index, so
        # "all of [s, e) is covered" is O(1) (run[s] >= e - s).
        run = [0] * (len(covered) + 1)
        for i in range(len(covered) - 1, -1, -1):
            run[i] = run[i + 1] + 1 if covered[i] else 0
        end = limit
        while end > 0:
            if not self._location_covers_states(locations[end - 1]):
                # No recurrent state ending this block: the next candidate is
                # one block shorter (the window region moves with it).
                end -= 1
                continue
            # The union of every windowed group's requirement: the widest tail
            # window decides where it starts, and an R-SWA group pins it to
            # block 0 (see _window_required_from).
            start = min(
                self._window_required_from(meta, end) for meta in self._windowed_metas
            )
            if run[start] >= end - start:
                return end
            # A hole inside [start, end): every end in (hole, end] has that
            # hole in its own region (start only grows with end, and R-SWA's
            # region only shrinks), so the largest candidate left is the hole
            # itself -- which excludes it.
            end = start + run[start]
        return 0

    def _external_match_burned(self, req_id: str) -> bool:
        """Has this request lost its option of an external match?

        * single-group block table: load failures are reported to vLLM
          (report_failures=True in the worker's start_load_kv) and come back
          as invalid block ids in update_connector_output -- the explicit
          signal. A mere preemption re-query keeps its match: the loaded KV
          is healthy, only the scheduling position was lost.
        * multi-group block table (hybrid, or a skipped EAGLE/MTP drafter
          group alongside the transferred attention group): vLLM's
          invalid-block recovery unpacks a single group (upstream TODO), so
          failures cannot be reported and no signal ever comes back. The
          request instead gets one conservative shot: any allocation for an
          external hit burns the match, because a failed block cannot be
          told apart from a healthy one afterwards.

        The shape is read from the block-table snapshot recorded at
        allocation time (``ledger.block_ids_per_group``): it mirrors
        kv_cache_manager.get_block_ids, the tuple the recovery path unpacks,
        and its length -- the group count -- is model-static.
        """
        if req_id in self._load_failed:
            return True
        if req_id not in self._load_attempted:
            return False
        ledger = self._tracked.get(req_id)
        if ledger is None or not ledger.block_ids_per_group:
            # Defensive: every load attempt is recorded through
            # update_state_after_alloc, which snapshots the block table.
            # Without a snapshot the recovery shape is unknown; keep the
            # match rather than burn it blindly.
            return False
        return len(ledger.block_ids_per_group) > 1

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> Tuple[Optional[int], bool]:
        """Answer vLLM's per-request question: beyond the ``num_computed_tokens``
        it already has, how many more tokens can the external KV supply?

        A pure query: it never blocks (an in-flight manager query returns
        None and vLLM re-asks next step) and never mutates request state.
        The answer is cached per request; ``update_state_after_alloc``
        consumes it and turns it into the LoadRequest once vLLM allocates
        blocks for the hit.

        Returns ``(external_tokens, load_kv_async)``: ``None`` while the
        query is in flight; ``0`` when there is no hit, or the request's
        match is burned (see ``_external_match_burned``); ``>0`` for a
        block-aligned count clamped to a prefix vLLM can safely resume from
        (``_safe_external_prefix``).
        """
        req_id = request.request_id
        if self._external_match_burned(req_id):
            # vLLM re-asks whenever a request falls back to the waiting queue:
            # preemption, or a failed load under kv_load_failure_policy=
            # recompute. For this request the external match is burned --
            # see _external_match_burned for which signal burned it. Whatever
            # is left, vLLM recomputes locally.
            logger.warning(
                "req:%s re-queried after an external load attempt, skip external match",
                req_id,
            )
            # Hit accounting: this re-ask answers "no external match";
            # the local hit stands, the remote hit is spent.
            ledger = self._tracked.get(req_id)
            if ledger is not None:
                ledger.local_matched_token_num = num_computed_tokens
                ledger.remote_matched_token_num = 0
            return 0, False

        computed_blocks = num_computed_tokens // self._manager_block_size
        need_load_locations = self._location_query_manager.get_locations_for_query(
            request, computed_blocks
        )
        if need_load_locations is None:
            return None, False

        need_load_locations = self._safe_external_prefix(
            req_id, need_load_locations, num_computed_tokens, request.num_tokens
        )
        # Cache the clamped answer: the allocation consumes exactly this.
        self._location_query_manager.store_result(req_id, need_load_locations)
        new_matched_count = len(need_load_locations) * self._manager_block_size
        # Hit accounting: record the answer the request will finish with
        # (local = vLLM's hit when asked, remote = this clamped answer).
        ledger = self._tracked.get(req_id)
        if ledger is None:
            ledger = RequestLedger(
                vllm_request=request,
                block_ids_per_group=[],
                has_saved_block_num=0,
            )
            self._tracked[req_id] = ledger
        ledger.local_matched_token_num = num_computed_tokens
        ledger.remote_matched_token_num = new_matched_count
        logger.info("req:%s matched %d external tokens", req_id, new_matched_count)
        return new_matched_count, new_matched_count > 0

    def _safe_external_prefix(
        self,
        req_id: str,
        locations: List[dict],
        num_computed_tokens: int,
        num_tokens: int,
    ) -> List[dict]:
        """Clamp an external match to the longest prefix vLLM can resume from.

        Three constraints, all only ever trimming the tail, so the answer is
        the longest prefix satisfying all of them at once:

        * the match must end on a state-complete block -- a hybrid request
          resumes from the recurrent state ending the reused prefix, so an
          ending block without one is unloadable however much attention KV
          precedes it;
        * the window region ending a windowed match must carry that group's
          data for every rank (``_window_hit_limit``) -- vLLM only allocates
          the windowed group's blocks inside the window on an external hit, so
          a region without published data would leave visible tokens unwritten;
        * at least one token must stay uncomputed -- logits are not part of
          the KV cache, so the model still needs one token to sample from,
          and vLLM's synchronous-load path asserts num_new_tokens > 0
          (vLLM's own connectors apply the same cap).

        Full-attention models have no state groups and no windowed groups and
        only feel the cap; their matching path is byte-for-byte the pre-PR one.
        """
        # Cap: how many leading blocks may be matched at all.
        limit = len(locations)
        while (
            limit
            and num_computed_tokens + limit * self._manager_block_size >= num_tokens
        ):
            limit -= 1
        if self._windowed_metas:
            # Window constraint, evaluated jointly with the state-complete end
            # condition: both depend on where the match ends.
            windowed_limit = self._windowed_match_end(locations, limit)
            if windowed_limit < limit:
                logger.info(
                    "req:%s truncated external match from %d to %d blocks "
                    "(window region of a windowed group has no published data)",
                    req_id,
                    len(locations),
                    windowed_limit,
                )
                limit = windowed_limit
        # Within that allowance the match may only end on a block carrying
        # the recurrent state: scan for the last one. (With windowed groups the
        # joint scan above already enforced it on the ending block.)
        keep = 0
        for i, location in enumerate(locations[:limit]):
            if self._location_covers_states(location):
                keep = i + 1
        if keep < len(locations):
            logger.info(
                "req:%s truncated external match from %d to %d blocks "
                "(full-hit cap / no recurrent state at the end)",
                req_id,
                len(locations),
                keep,
            )
        return locations[:keep]

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        """First (or re-) allocation: record the ledger and ship the load.

        The block tables arrive here whole, once per allocation; increments
        come later through the scheduling output. When vLLM allocated for an
        external hit, the query the match hook cached is consumed and turned
        into the LoadRequest -- at this point, and only here, both halves of
        its address are known (manager locations + physical slots)."""
        req_id = request.request_id
        ledger = self._tracked.get(req_id)
        if ledger is None:
            ledger = RequestLedger(
                vllm_request=request,
                block_ids_per_group=[],
                has_saved_block_num=0,
            )
            self._tracked[req_id] = ledger
        ledger.block_ids_per_group = [list(b) for b in blocks.get_block_ids()]

        if num_external_tokens <= 0:
            return
        consumed = self._location_query_manager.consume_locations(req_id)
        if consumed is None:
            # Invariant violation: vLLM allocated blocks for an external hit
            # (num_external_tokens > 0) only after the match hook answered
            # positively, and that answer was stored for exactly this
            # consumption. No cached answer here means the match/alloc
            # contract is broken -- vLLM would run the request on KV that
            # was never loaded. Fail closed: a traceback beats silently
            # corrupting the request's output.
            raise RuntimeError(
                f"req {req_id}: allocated for {num_external_tokens} external "
                f"tokens but no cached location query exists; the match hook "
                f"did not answer, or its answer was consumed/invalidated "
                f"before this allocation"
            )
        locations, computed_blocks = consumed
        # Blocks were allocated for an external hit: the request is now
        # spending its external load (burns hybrid re-queries).
        self._load_attempted.add(req_id)
        if not locations:
            return
        total_remote_blocks = computed_blocks + len(locations)
        ledger.has_saved_block_num = total_remote_blocks
        self._waiting_to_load_requests.append(
            LoadRequest(
                req_id=req_id,
                manager_block_idxes=list(range(computed_blocks, total_remote_blocks)),
                need_load_locations=locations,
                all_block_ids=[list(b) for b in ledger.block_ids_per_group],
            )
        )

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        """Consume the worker's step output: mark requests whose external
        load failed, at request granularity.

        vLLM reports invalid blocks, not requests; a block id is matched
        against the block tables recorded in update_state_after_alloc. One
        failed block is enough -- the request recomputes as a whole, which
        blocks to recompute exactly is vLLM's decision.
        """
        invalid = getattr(connector_output, "invalid_block_ids", None)
        if not invalid:
            return
        for req_id, ledger in self._tracked.items():
            if any(
                b in invalid
                for group_ids in ledger.block_ids_per_group
                for b in group_ids
            ):
                self._load_failed.add(req_id)
                logger.warning(
                    "req:%s external load failed (invalid blocks "
                    "reached its block table)",
                    req_id,
                )

    # ------------------------------------------------------------------ #
    # Per-step metadata assembly and saving orchestration
    # ------------------------------------------------------------------ #
    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> TairKvCacheConnectorMetadata:
        """Assemble one engine step's envelope (see TairKvCacheConnectorMetadata).

        Two sources feed the envelope: vLLM's scheduling output for this step
        (block-table increments for continuing requests) and the residue of
        earlier steps' async work (admitted loads, resolved save sessions,
        retirements). Only the last stage depends on the others: save
        settlement may retire a request within this step, so finishes flush
        last. Capture instructions are emitted while this step's block tables
        are live: the worker runs them in the same step's wait_for_save, before
        vLLM's window sliding touches the blocks.
        """
        meta = TairKvCacheConnectorMetadata(self._epoch)
        self._epoch += 1

        self._ingest_scheduled_reqs(scheduler_output)
        self._dispatch_incremental_saves(meta, scheduler_output)
        self._collect_load_instructions(meta)
        self._collect_release_captures(meta)
        self._collect_save_instructions(meta)
        self._collect_finish_instructions(meta)
        return meta

    def _ingest_scheduled_reqs(self, scheduler_output: "SchedulerOutput") -> None:
        """Absorb vLLM's scheduling list into the request ledger.

        The block tables are the scheduler's ledger: vLLM hands them over
        once per first allocation (update_state_after_alloc /
        scheduled_new_reqs), then only as increments (cached_reqs.
        new_block_ids). Accumulating them here is what lets later stages
        translate manager blocks into physical slots for the worker's
        instructions. A preemption resume replaces the whole table
        (re-allocation maps the same logical blocks to fresh slots)."""
        for vllm_req in scheduler_output.scheduled_new_reqs:
            ledger = self._tracked.get(vllm_req.req_id)
            if ledger is None:
                # update_state_after_alloc creates the ledger before vLLM
                # can schedule the request, so this only fires on a broken
                # hook-order contract: log it loudly instead of silently
                # dropping the block table.
                logger.warning(
                    "scheduled new req %s has no ledger (hook-order "
                    "contract broken?); its block table is not recorded",
                    vllm_req.req_id,
                )
                continue
            ledger.block_ids_per_group = [list(b) for b in vllm_req.block_ids]

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached_reqs.req_ids):
            ledger = self._tracked.get(req_id)
            if ledger is None:
                if hasattr(cached_reqs, "resumed_req_ids"):
                    resumed = req_id in cached_reqs.resumed_req_ids
                else:
                    resumed = cached_reqs.resumed_from_preemption[idx]
                logger.warning(
                    "scheduled cached req %s has no ledger (resumed=%s, "
                    "scheduled_tokens=%d); its block increments are not "
                    "recorded",
                    req_id,
                    resumed,
                    scheduler_output.num_scheduled_tokens[req_id],
                )
                continue

            if hasattr(cached_reqs, "resumed_req_ids"):
                resumed = req_id in cached_reqs.resumed_req_ids
            else:
                resumed = cached_reqs.resumed_from_preemption[idx]

            new_block_ids = cached_reqs.new_block_ids[idx]
            if new_block_ids is None:
                # https://github.com/vllm-project/vllm/pull/23262: None when
                # no group got new blocks this step (get_block_ids with
                # allow_none=True). Keep the recorded table as-is: a resumed
                # request with nothing allocated has no fresh slots to record.
                continue
            if resumed:
                ledger.block_ids_per_group = [list(b) for b in new_block_ids]
            else:
                for group_ids, new_ids in zip(
                    ledger.block_ids_per_group, new_block_ids
                ):
                    group_ids.extend(new_ids)

    def _dispatch_incremental_saves(
        self, meta: TairKvCacheConnectorMetadata, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Start async StartWriteCache calls for requests whose computed
        prefix crossed a manager-block boundary since the last step, and ask
        the worker to capture the newly completed blocks of every windowed
        group in this same step.

        This stage only acquires write locations (the HTTP call resolves tens
        of ms later, off the scheduler thread); the data itself moves later,
        gathered by the worker out of HBM -- except for windowed groups, whose
        blocks vLLM frees while the request lives: their bytes are copied into
        the staging arena by the worker's wait_for_save of *this* step (see
        CaptureRequest), so the announce -> session lag never reads HBM. The
        session the http thread produces surfaces in _collect_save_instructions
        on a later step.
        """
        num_scheduled = scheduler_output.num_scheduled_tokens
        for ledger in self._tracked.values():
            # The watermark is the *fully computed, fully keyed* prefix, capped
            # on both sides. Computed KV: vLLM's own num_computed_tokens counts
            # what the request has computed before this step (hits included);
            # adding this step's scheduled tokens gives the prefix whose KV
            # exists once the step's forward completes -- and the worker
            # consumes this metadata after exactly that forward. The allocated
            # block span is *not* that bound: a prefill chunk can end inside a
            # manager block, and the allocation still covers the whole block,
            # so announcing up to it would let a windowed group's eager capture
            # (which reads in that same step) copy rows the forward never wrote
            # and publish zero pages as KV. Key material: during decode
            # all_token_ids lags one token behind (the token scheduled in this
            # step is appended only once sampled), so it stays as the other
            # cap; the block span remains a purely defensive ceiling (every
            # transfer group's table must cover what we announce).
            target_save_num = (
                min(
                    len(ledger.vllm_request.all_token_ids),
                    ledger.vllm_request.num_computed_tokens
                    + num_scheduled.get(ledger.vllm_request.request_id, 0),
                    self._num_allocated_blocks(ledger) * self._vllm_block_size,
                )
                // self._manager_block_size
            )
            if target_save_num > ledger.has_saved_block_num:
                logger.info(
                    "req:%s incremental save: %d -> %d blocks "
                    "(tokens=%d, allocated=%d)",
                    ledger.vllm_request.request_id,
                    ledger.has_saved_block_num,
                    target_save_num,
                    len(ledger.vllm_request.all_token_ids),
                    self._num_allocated_blocks(ledger),
                )
                if self._windowed_metas:
                    # Windowed groups: the blocks just completed are still in
                    # HBM right now and are freed by the *next* schedule step,
                    # so this is the only moment their bytes can be copied.
                    meta.add_capture_request(
                        CaptureRequest(
                            req_id=ledger.vllm_request.request_id,
                            manager_block_idxes=list(
                                range(ledger.has_saved_block_num, target_save_num)
                            ),
                            all_block_ids=[
                                list(b) for b in ledger.block_ids_per_group
                            ],
                        )
                    )
                ledger.scheduled_saving_count += 1
                # Every session announces the whole prefix (the token prefix
                # and the block keys go up to target_save_num), so its range is
                # (0, target_save_num): noted here, in the loop, before the
                # http thread runs -- a release decision made by a concurrent
                # session must already see this one as in flight.
                session_id = self._record_announcement(
                    ledger.vllm_request.request_id, target_save_num
                )
                # Per-block state completeness must be read here, in the
                # scheduler loop: it comes from vLLM's block table, which the
                # http_executor thread would race against later steps.
                self._http_executor.submit(
                    self.start_save_kvcache_async,
                    ledger.vllm_request.request_id,
                    # vLLM types all_token_ids as list[ConstantList | int];
                    # at runtime it only ever holds ints.
                    ledger.vllm_request.all_token_ids[  # ty: ignore[invalid-argument-type]
                        : target_save_num * self._manager_block_size
                    ],
                    target_save_num,
                    self._state_complete_mask(ledger, range(target_save_num)),
                    self._window_data_mask(ledger, range(target_save_num)),
                    session_id,
                )
            ledger.has_saved_block_num = target_save_num

    def _collect_release_captures(self, meta: TairKvCacheConnectorMetadata) -> None:
        """Hand the worker the captures this step's sessions will not write."""
        with self._pending_release_captures_lock:
            pending = self._pending_release_captures
            self._pending_release_captures = []
        for req_id, blocks in pending:
            if blocks:
                meta.release_captures(req_id, blocks)

    def _collect_load_instructions(self, meta: TairKvCacheConnectorMetadata) -> None:
        """Hand the worker the loads whose blocks were allocated.

        LoadRequests are built by update_state_after_alloc, at the moment
        both halves of their address (manager locations + physical slots)
        became known; this stage only moves them into the envelope."""
        for load_req in self._waiting_to_load_requests:
            meta.add_load_request(load_req)
        self._waiting_to_load_requests = []

    def _collect_save_instructions(self, meta: TairKvCacheConnectorMetadata) -> None:
        """Hand the worker the save sessions whose write locations have
        arrived, and settle finished requests whose saving is now complete.

        Canceled sessions (start_write_cache failed on the http thread)
        settle here too: they count as sent, so a request waiting only on
        them can be retired.
        """
        with self._waiting_to_save_requests_lock:
            new_save_reqs = self._waiting_to_save_requests
            self._waiting_to_save_requests = []
        for save_req in new_save_reqs:
            ledger = self._tracked.get(save_req.req_id)
            if ledger is None:
                logger.warning(
                    "request %s is not tracked, skip saving", save_req.req_id
                )
                continue
            # Snapshot the ledger for the worker: the gather translates
            # manager blocks into physical slots through these tables, and
            # the worker keeps no mirror of its own.
            save_req.all_block_ids = [list(b) for b in ledger.block_ids_per_group]
            meta.add_save_request(save_req)
            ledger.sent_saving_count += 1
            if (
                ledger.need_report_after_saving_finished
                and ledger.scheduled_saving_count == ledger.sent_saving_count
            ):
                self._retire_request(save_req.req_id)

        self.handle_canceled_save_req()

    def _collect_finish_instructions(self, meta: TairKvCacheConnectorMetadata) -> None:
        """Flush retirements queued since the last step. Runs last on purpose:
        save settlement above may retire a request within this very step."""
        for finish_req in self._waiting_to_finish_requests:
            meta.add_finish_request(finish_req)
        self._waiting_to_finish_requests = []

    def _retire_request(self, req_id: str) -> None:
        self._window_claims.pop(req_id, None)
        with self._announced_sessions_lock:
            self._announced_sessions.pop(req_id, None)
            self._announced_sessions_counters.pop(req_id, None)
        """The request's save obligations are settled: tell the worker to drop
        its bookkeeping and stop tracking the request ourselves."""
        self._waiting_to_finish_requests.append(FinishRequest(req_id))
        self._tracked.pop(req_id, None)
        self._load_failed.discard(req_id)
        self._load_attempted.discard(req_id)
        self._location_query_manager.invalidate(req_id)

    def start_save_kvcache_async(
        self,
        req_id: str,
        token_ids: List[int],
        target_save_num: int,
        state_complete_mask: List[bool],
        window_data_mask: Optional[List[bool]] = None,
        session_id: Optional[int] = None,
    ) -> None:
        """Ask the manager for write locations for a request's first
        ``target_save_num`` manager blocks.

        ``state_complete_mask[i]`` / ``window_data_mask[i]`` say whether manager
        block i has a materialized recurrent state / real windowed slots. A
        block missing either is announced under the spec group that describes
        what it *does* carry, so the manager allocates (and later advertises)
        only the specs that will really hold data -- absence is never encoded as
        a successful write, and a windowed block whose window is already gone is
        never advertised as if its bytes were there.
        """
        if session_id is None:
            # Direct callers (tests) did not pre-record the announcement; the
            # session still needs an identity of its own.
            session_id = self._record_announcement(req_id, target_save_num)
        request = {
            "trace_id": "%s_%d" % (req_id, self._epoch),
            "instance_id": self._extra_config.instance_id,
            "block_keys": [],
            "token_ids": token_ids,
            "write_timeout_seconds": self._extra_config.write_timeout_seconds,
        }
        if self._state_group_idxs or self._windowed_metas:
            if window_data_mask is None:
                window_data_mask = [True] * target_save_num
            assert len(state_complete_mask) == target_save_num, (
                f"state mask {len(state_complete_mask)} != {target_save_num} blocks"
            )
            assert len(window_data_mask) == target_save_num, (
                f"window mask {len(window_data_mask)} != {target_save_num} blocks"
            )
            announced = self._announced_window_mask(req_id, window_data_mask)
            coverage_names = [
                self._spec_group_for(complete, window_present)
                for complete, window_present in zip(state_complete_mask, announced)
            ]
            empties = sum(name is None for name in coverage_names)
            if empties and self._narrowest_spec_group is not None:
                # A block with no coverage at all: give it the narrowest
                # declared group (it cannot fill even that, so the save verdict
                # fails it and nothing is published) and keep every other
                # block's precise coverage.
                coverage_names = [
                    name if name is not None else self._narrowest_spec_group
                    for name in coverage_names
                ]
                logger.warning(
                    "req:%s: %d/%d blocks carry neither state nor window data; "
                    "announcing them under %s (they will not be published)",
                    req_id,
                    empties,
                    target_save_num,
                    self._narrowest_spec_group,
                )
            if all(name is not None for name in coverage_names):
                request["location_spec_group_names"] = coverage_names
            else:
                # Nothing at all is declared (no state group, no window): the
                # default is every registered spec, which is what the manager
                # assumed before this PR. Sound only because such an instance
                # has nothing sparse to isolate.
                logger.info(
                    "req:%s: %d/%d blocks have no declared spec group; "
                    "announcing all specs",
                    req_id,
                    empties,
                    target_save_num,
                )
            if not all(window_data_mask):
                logger.info(
                    "req:%s saving %d/%d blocks whose window is already gone "
                    "(windowed specs excluded)",
                    req_id,
                    window_data_mask.count(False),
                    target_save_num,
                )
            if not all(state_complete_mask):
                logger.info(
                    "req:%s saving %d/%d blocks without a recurrent "
                    "state (attention specs only)",
                    req_id,
                    state_complete_mask.count(False),
                    target_save_num,
                )
        try:
            response = self._manager_client.start_write_cache(request)
        except Exception as e:
            # No locations were allocated: this session will write nothing, so
            # it protects nothing either.
            self._remember_session_plan(req_id, session_id, ())
            logger.warning("start_write_cache error, skip saving: %s", e)
            with self._canceled_save_request_ids_lock:
                self._canceled_save_request_ids.append(req_id)
            return

        locations = response["locations"]
        write_session_id = response["write_session_id"]
        mask = response.get("block_mask") or {}
        if "bool_masks" in mask:
            values = mask["bool_masks"].get("values", [])
            mask_summary = (
                f"bool_masks offset={mask['bool_masks'].get('offset')} "
                f"total={len(values)} "
                f"existing={sum(1 for v in values if v)}"
            )
        else:
            mask_summary = f"offset={mask.get('offset')}"
        logger.info(
            "req:%s save session %s: block_mask %s locations=%d",
            req_id,
            write_session_id[:8],
            mask_summary,
            len(locations),
        )
        logger.debug(
            "req:%s save session %s: block_mask=%s locations=%d",
            req_id,
            write_session_id[:8],
            response.get("block_mask"),
            len(locations),
        )

        if not locations:
            # Every announced block already has a location: this session writes
            # nothing at all, so release the captures for the blocks no other
            # session of this request is about to write (the same guard as
            # below -- "already has a location" is not "already written").
            self._remember_session_plan(req_id, session_id, ())
            protected = self._blocks_other_sessions_write(req_id, session_id)
            releasable = [b for b in range(target_save_num) if b not in protected]
            with self._pending_release_captures_lock:
                self._pending_release_captures.append((req_id, releasable))
            try:
                self._manager_client.finish_write_cache(
                    {
                        "trace_id": "finish_%s" % write_session_id[:8],
                        "instance_id": self._extra_config.instance_id,
                        "write_session_id": write_session_id,
                        "success_blocks": {"bool_masks": {"offset": 0}},
                    }
                )
            except Exception as e:
                logger.warning(
                    "finish_write_cache failed, session: %s, error: %s",
                    write_session_id,
                    e,
                )
            with self._canceled_save_request_ids_lock:
                self._canceled_save_request_ids.append(req_id)
            return

        need_block_idx = self.parse_block_mask_to_save_indices(
            response, target_save_num
        )
        self._remember_session_plan(req_id, session_id, need_block_idx)
        already = self.parse_block_mask_to_existing_indices(response, target_save_num)
        protected = self._blocks_other_sessions_write(req_id, session_id)
        already = [block for block in already if block not in protected]
        if already:
            # The manager already holds these blocks and no concurrent session
            # of this request is about to write them, so the worker's eager
            # capture for them has nothing to publish and must give its arena
            # slots back now, not when the request retires (a long duplicate
            # request would otherwise pin one slot per already-cached block).
            # The scheduler loop flushes this into the metadata (the http
            # thread has none).
            with self._pending_release_captures_lock:
                self._pending_release_captures.append((req_id, already))
        message = CoordinateMessage(
            time.time(),
            SendBlockStartEvent(
                request_id=req_id,
                write_session_id=write_session_id,
                locations=locations,
            ),
        )
        self._coordinator_client.send(CoordinateMsgSerializer.dumps(message))

        with self._waiting_to_save_requests_lock:
            self._waiting_to_save_requests.append(
                SaveRequest(req_id, locations, need_block_idx, write_session_id)
            )

    def handle_canceled_save_req(self) -> None:
        with self._canceled_save_request_ids_lock:
            canceled = self._canceled_save_request_ids
            self._canceled_save_request_ids = []
        for req_id in canceled:
            # Cancellations come from http_executor threads; the request may
            # already have been finished and removed by the scheduler loop.
            ledger = self._tracked.get(req_id)
            if ledger is None:
                logger.warning("canceled save for unknown request %s, skip", req_id)
                continue
            ledger.sent_saving_count += 1
            if (
                ledger.need_report_after_saving_finished
                and ledger.scheduled_saving_count == ledger.sent_saving_count
            ):
                self._retire_request(req_id)

    def get_finished_count(self) -> int:
        # Only rank0 reports finished requests.
        return 1

    def _record_announcement(self, req_id: str, target_save_num: int) -> int:
        """Note that a write session for [0, target_save_num) is now in flight.

        Returns the session's id. Called from the scheduler loop *before* the
        http thread runs, so a release decision made by a concurrent session
        already sees it -- and a repeated target (a regressed save watermark)
        gets its own entry rather than replacing the session still running."""
        with self._announced_sessions_lock:
            session_id = self._announced_sessions_counters.get(req_id, 0) + 1
            self._announced_sessions_counters[req_id] = session_id
            self._announced_sessions.setdefault(req_id, {})[session_id] = (
                target_save_num,
                None,
            )
        return session_id

    def _blocks_other_sessions_write(self, req_id: str, own_session: int) -> Set[int]:
        """Blocks a concurrent write session of this request may still write.

        The manager's block mask says a block already has a location, not that
        its bytes are committed: a session that has allocated but not written
        shows up as "existing" to the next session. Releasing such a block's
        capture would make that pending write fail (and lose the request's
        head, which is then never captured again), so the blocks of every other
        session of this request are held back: its write set once it answered,
        its whole announced prefix while the response is still in flight.
        """
        protected: Set[int] = set()
        with self._announced_sessions_lock:
            for session_id, (target, plan) in self._announced_sessions.get(
                req_id, {}
            ).items():
                if session_id == own_session:
                    continue  # the session answering right now
                protected |= plan if plan is not None else set(range(target))
        return protected

    def _remember_session_plan(
        self, req_id: str, session_id: int, plan: Iterable[int]
    ) -> None:
        """Record what this session will write (or nothing, if nothing)."""
        with self._announced_sessions_lock:
            sessions = self._announced_sessions.get(req_id)
            if sessions is not None and session_id in sessions:
                target, _plan = sessions[session_id]
                sessions[session_id] = (target, set(plan))

    def parse_block_mask_to_save_indices(
        self, response: dict, target_save_num: int
    ) -> List[int]:
        block_mask = response.get("block_mask", {})
        if "offset" in block_mask:
            return list(range(block_mask["offset"], target_save_num))
        values = block_mask.get("bool_masks", {}).get("values", [])
        return [idx for idx, saved in enumerate(values) if not saved]

    def parse_block_mask_to_existing_indices(
        self, response: dict, target_save_num: int
    ) -> List[int]:
        """The mirror of ``parse_block_mask_to_save_indices``: the blocks the
        manager already holds, which this session will therefore not write."""
        block_mask = response.get("block_mask", {})
        if "offset" in block_mask:
            return list(range(0, min(block_mask["offset"], target_save_num)))
        values = block_mask.get("bool_masks", {}).get("values", [])
        return [
            idx for idx, saved in enumerate(values[:target_save_num]) if saved
        ]

    # ------------------------------------------------------------------ #
    # Request finish
    # ------------------------------------------------------------------ #
    def request_finished_all_groups(
        self, request: "Request", block_ids: Tuple[List[int], ...]
    ) -> Tuple[bool, Optional[dict]]:
        return self._finish_request(request)

    def request_finished(
        self, request: "Request", block_ids: List[int]
    ) -> Tuple[bool, Optional[dict]]:
        return self._finish_request(request)

    def _finish_request(self, request: "Request") -> Tuple[bool, Optional[dict]]:
        req_id = request.request_id
        ledger = self._tracked.get(req_id)
        if ledger is None:
            logger.info("request_finished for untracked request: %s", req_id)
            self._location_query_manager.invalidate(req_id)
            return False, None

        # Hit accounting travels to the client through vLLM's
        # kv_transfer_params (EngineCoreOutput -> RequestOutput -> the
        # OpenAI response); consumers attribute TTFT and per-request hits
        # to local (vLLM prefix cache) vs remote (KVCM) sources with it.
        extra_info = {
            "local_matched_token_num": ledger.local_matched_token_num,
            "remote_matched_token_num": ledger.remote_matched_token_num,
        }

        if ledger.scheduled_saving_count == ledger.sent_saving_count:
            self._retire_request(req_id)
            return True, extra_info

        # Saves still in flight; delay freeing the blocks until they land.
        ledger.need_report_after_saving_finished = True
        return True, extra_info
