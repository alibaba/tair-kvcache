"""Scheduler side of the connector: matching, saving orchestration, finishing.

Owns the authoritative ``_alive_requests`` (the scheduler's view of every
request's token stream and per-group block tables) and answers the vLLM
scheduler hooks. The worker side lives in worker_core; both speak the
vllm_common vocabulary.
"""

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, List, Optional, Tuple

from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.common.tp_coordinator import (
    CoordinateMsgSerializer, CoordinateMessage, SendBlockStartEvent, TpCoordinatorClient)
from kv_cache_manager.py_connector.vllm.location_query_manager import LocationQueryManager
from kv_cache_manager.py_connector.vllm.metadata import (
    FinishRequest, LoadRequest, ReqStateToWorker, SaveRequest, TairKvCacheConnectorMetadata)
from kv_cache_manager.py_connector.vllm.vllm_common import (
    ATTN_SPEC_GROUP, FULL_SPEC_GROUP, GroupMeta, ReqState, build_spec_groups, spec_name)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks


class SchedulerCore:
    """State and hooks for the scheduler-role connector instance."""

    def __init__(self, extra_config, group_metas: List[GroupMeta],
                 manager_block_size: int, vllm_block_size: int, tp_size: int,
                 manager_client, coordinator_client: TpCoordinatorClient):
        self._extra_config = extra_config
        self._group_metas = group_metas
        self._num_groups = len(group_metas)
        self._state_group_idxs = [m.group_idx for m in group_metas if not m.is_attention]
        self._manager_block_size = manager_block_size
        self._vllm_block_size = vllm_block_size
        self._tp_size = tp_size
        self._manager_client = manager_client
        self._coordinator_client = coordinator_client

        self._epoch = 0
        self._http_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kvcm_http_")
        self._location_query_manager = LocationQueryManager(
            manager_client, self._http_executor, extra_config.instance_id,
            extra_config.async_get_cache_location)

        self._alive_requests: dict[str, ReqState] = {}
        self._waiting_to_load_requests: List[LoadRequest] = []
        self._waiting_to_save_requests_lock = threading.Lock()
        self._waiting_to_save_requests: List[SaveRequest] = []
        self._waiting_to_finish_requests: List[FinishRequest] = []
        self._canceled_save_request_ids_lock = threading.Lock()
        self._canceled_save_request_ids: List[str] = []

    def shutdown(self):
        self._location_query_manager.shutdown()
        self._http_executor.shutdown(wait=False)

    # ------------------------------------------------------------------ #
    # Hybrid state coverage
    # ------------------------------------------------------------------ #
    def _spec_groups(self) -> List[dict]:
        """LocationSpecGroups for registration; see vllm_common.build_spec_groups."""
        return build_spec_groups(self._group_metas, self._tp_size)

    def _group_block_size(self, group_idx: int) -> int:
        for meta in self._group_metas:
            if meta.group_idx == group_idx:
                return meta.block_size
        raise KeyError(f"no such transferred group: {group_idx}")

    def _state_complete_mask(self, req: ReqState, manager_block_idxes) -> List[bool]:
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
                table = req.block_ids_per_group[group_idx]
                # State covers the prefix ending at the block's last token.
                logical = ((mb + 1) * mbs - 1) // self._group_block_size(group_idx)
                if logical >= len(table) or table[logical] == 0:
                    complete = False
                    break
            mask.append(complete)
        return mask

    def _num_allocated_blocks(self, req: ReqState) -> int:
        """Min allocated block-table length across the *transferred* groups.

        ``block_ids_per_group`` is indexed by the vLLM group index and includes
        groups skipped by ``parse_groups`` (EAGLE/MTP drafters). A drafter's
        block table can lag behind the target model's, so including it in the
        min would permanently understate how many blocks are saveable."""
        if not req.block_ids_per_group:
            return 0
        return min(len(req.block_ids_per_group[meta.group_idx])
                   for meta in self._group_metas)

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
        return all(spec_name(rank, group_idx) in names
                   for rank in range(self._tp_size)
                   for group_idx in self._state_group_idxs)

    def get_num_new_matched_tokens(self, request: "Request",
                                   num_computed_tokens: int) -> Tuple[Optional[int], bool]:
        """Answer vLLM's per-request question: beyond the ``num_computed_tokens``
        it already has, how many more tokens can the external KV supply?

        Returns ``(external_tokens, load_kv_async)``:

        * ``None`` -- the manager query is still in flight; vLLM will re-ask
          next step (this connector never blocks the scheduler loop on IO).
        * ``0``    -- no external hit (or the request's external match is
          burned, see ``_external_match_burned``).
        * ``>0``   -- a block-aligned count clamped to a prefix vLLM can
          safely resume from (``_safe_external_prefix``); the load itself runs
          on the worker after vLLM allocates blocks for the hit.
        """
        prev = self._alive_requests.get(request.request_id)
        if prev is not None and self._external_match_burned(prev):
            # vLLM re-asks whenever a request falls back to the waiting queue:
            # preemption, or a failed load under kv_load_failure_policy=
            # recompute. For this request the external match is burned --
            # see _external_match_burned for which signal burned it. Whatever
            # is left, vLLM recomputes locally.
            logger.warning("req:%s re-queried after an external load attempt, "
                           "skip external match", request.request_id)
            prev.local_matched_token_num = num_computed_tokens
            prev.remote_matched_token_num = 0
            prev.has_saved_block_num = num_computed_tokens // self._manager_block_size
            return 0, False

        computed_blocks = num_computed_tokens // self._manager_block_size
        need_load_locations = self._location_query_manager.get_locations_for_query(
            request, computed_blocks)
        if need_load_locations is None:
            return None, False

        need_load_locations = self._safe_external_prefix(
            request.request_id, need_load_locations,
            num_computed_tokens, request.num_tokens)
        new_matched_count = len(need_load_locations) * self._manager_block_size
        total_remote_blocks = computed_blocks + len(need_load_locations)
        logger.info("req:%s matched %d external tokens", request.request_id, new_matched_count)

        if new_matched_count:
            self._waiting_to_load_requests.append(LoadRequest(
                req_id=request.request_id,
                manager_block_idxes=list(range(computed_blocks, total_remote_blocks)),
                need_load_locations=need_load_locations,
            ))

        # Every request is registered here -- this is the first hook a request
        # passes through, hit or no hit: build_connector_meta and
        # request_finished index _alive_requests unconditionally, and
        # has_saved_block_num anchors the incremental saves that follow.
        self._alive_requests[request.request_id] = ReqState(
            req_id=request.request_id,
            token_ids=copy.copy(request.prompt_token_ids),
            block_ids_per_group=[],
            has_saved_block_num=total_remote_blocks,
            local_matched_token_num=num_computed_tokens,
            remote_matched_token_num=new_matched_count,
            vllm_request=request,
        )
        return new_matched_count, new_matched_count > 0

    def _external_match_burned(self, prev: ReqState) -> bool:
        """Has this request lost its option of an external match?

        * full-attention (single group): load failures are reported to vLLM
          (report_failures=True in the worker's start_load_kv) and come back
          as invalid block ids in update_connector_output -- the explicit
          signal. A mere preemption re-query keeps its match: the loaded KV
          is healthy, only the scheduling position was lost.
        * hybrid (multi group): vLLM's invalid-block recovery is single-group
          only (upstream TODO), so failures cannot be reported and no signal
          ever comes back. The request instead gets one conservative shot:
          any allocation for an external hit burns the match, because a
          failed block cannot be told apart from a healthy one afterwards.
        """
        if prev.load_failed:
            return True
        return prev.load_attempted and self._num_groups > 1

    def _safe_external_prefix(self, req_id: str, locations: List[dict],
                              num_computed_tokens: int, num_tokens: int) -> List[dict]:
        """Clamp an external match to the longest prefix vLLM can resume from.

        Two constraints, both only ever trimming the tail, so the answer is
        the longest prefix satisfying both at once:

        * the match must end on a state-complete block -- a hybrid request
          resumes from the recurrent state ending the reused prefix, so an
          ending block without one is unloadable however much attention KV
          precedes it;
        * at least one token must stay uncomputed -- logits are not part of
          the KV cache, so the model still needs one token to sample from,
          and vLLM's synchronous-load path asserts num_new_tokens > 0
          (vLLM's own connectors apply the same cap).

        Full-attention models have no state groups and only feel the cap.
        """
        # Cap: how many leading blocks may be matched at all.
        limit = len(locations)
        while limit and num_computed_tokens + limit * self._manager_block_size >= num_tokens:
            limit -= 1
        # Within that allowance the match may only end on a block carrying
        # the recurrent state: scan for the last one.
        keep = 0
        for i, location in enumerate(locations[:limit]):
            if self._location_covers_states(location):
                keep = i + 1
        if keep < len(locations):
            logger.info("req:%s truncated external match from %d to %d blocks "
                        "(full-hit cap / no recurrent state at the end)",
                        req_id, len(locations), keep)
        return locations[:keep]

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        req_state = self._alive_requests.get(request.request_id)
        if req_state is None:
            return
        req_state.block_ids_per_group = [list(b) for b in blocks.get_block_ids()]
        if num_external_tokens:
            # Blocks were allocated for an external hit: the request is now
            # spending its one external load.
            req_state.load_attempted = True

    def update_connector_output(self, connector_output: "KVConnectorOutput"):
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
        for req_id, req in self._alive_requests.items():
            if any(b in invalid
                   for group_ids in req.block_ids_per_group for b in group_ids):
                req.load_failed = True
                logger.warning("req:%s external load failed (invalid blocks "
                               "reached its block table)", req_id)

    # ------------------------------------------------------------------ #
    # Per-step metadata assembly and saving orchestration
    # ------------------------------------------------------------------ #
    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> TairKvCacheConnectorMetadata:
        """Assemble one engine step's envelope (see TairKvCacheConnectorMetadata).

        Two sources feed the envelope: vLLM's scheduling output for this step
        (mirror deltas for newly scheduled and continuing requests) and the
        residue of earlier steps' async work (admitted loads, resolved save
        sessions, retirements). Only the last stage depends on the others:
        save settlement may retire a request within this step, so finishes
        flush last.
        """
        meta = TairKvCacheConnectorMetadata(self._epoch)
        self._epoch += 1

        self._ingest_scheduled_reqs(scheduler_output, meta)
        self._dispatch_incremental_saves()
        self._collect_load_instructions(meta)
        self._collect_save_instructions(meta)
        self._collect_finish_instructions(meta)
        return meta

    def _ingest_scheduled_reqs(self, scheduler_output: "SchedulerOutput",
                               meta: TairKvCacheConnectorMetadata) -> None:
        """Absorb vLLM's scheduling list into the authoritative request table
        and emit mirror deltas for the worker.

        The worker owns a mirrored copy of every request's token stream and
        per-group block tables -- it needs them to translate manager blocks
        into physical slots when it gathers (save) or scatters (load). New
        requests get a full snapshot; continuing ones get increments; a
        preemption resume replaces the whole table (re-allocation maps the
        same logical blocks to fresh physical slots).
        """
        for vllm_req in scheduler_output.scheduled_new_reqs:
            request = self._alive_requests[vllm_req.req_id]
            request.block_ids_per_group = [list(b) for b in vllm_req.block_ids]
            meta.add_req_state_to_worker(ReqStateToWorker(
                req_id=request.req_id,
                has_saved_block_num=request.has_saved_block_num,
                new_tokens_ids=request.token_ids,
                new_block_ids_per_group=request.block_ids_per_group,
                is_delta=False,
            ))

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached_reqs.req_ids):
            request = self._alive_requests[req_id]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_current_tokens = len(request.token_ids)
            new_token_ids = request.vllm_request.all_token_ids[
                num_current_tokens:num_current_tokens + num_new_tokens]
            request.token_ids.extend(new_token_ids)

            delta = ReqStateToWorker(
                req_id=request.req_id,
                has_saved_block_num=request.has_saved_block_num,
                new_tokens_ids=new_token_ids,
            )

            if hasattr(cached_reqs, "resumed_req_ids"):
                resumed = req_id in cached_reqs.resumed_req_ids
            else:
                resumed = cached_reqs.resumed_from_preemption[idx]

            new_block_ids = cached_reqs.new_block_ids[idx]
            if resumed:
                request.block_ids_per_group = [list(b) for b in new_block_ids]
                delta.resumed_from_preemption = True
                delta.new_block_ids_per_group = request.block_ids_per_group
            elif new_block_ids is not None:
                # https://github.com/vllm-project/vllm/pull/23262: may be None
                delta.new_block_ids_per_group = [list(b) for b in new_block_ids]
                for group_ids, new_ids in zip(request.block_ids_per_group,
                                              delta.new_block_ids_per_group):
                    group_ids.extend(new_ids)
            meta.add_req_state_to_worker(delta)

    def _dispatch_incremental_saves(self) -> None:
        """Start async StartWriteCache calls for requests whose computed
        prefix crossed a manager-block boundary since the last step.

        This stage only acquires write locations (the HTTP call resolves tens
        of ms later, off the scheduler thread); the data itself moves later,
        gathered by the worker out of HBM. The session the http thread
        produces surfaces in _collect_save_instructions on a later step.
        """
        for req in self._alive_requests.values():
            target_save_num = min(
                len(req.token_ids),
                self._num_allocated_blocks(req) * self._vllm_block_size) // self._manager_block_size
            if target_save_num > req.has_saved_block_num:
                req.scheduled_saving_count += 1
                # Per-block state completeness must be read here, in the
                # scheduler loop: it comes from vLLM's block table, which the
                # http_executor thread would race against later steps.
                self._http_executor.submit(
                    self.start_save_kvcache_async, req.req_id,
                    req.token_ids[:target_save_num * self._manager_block_size],
                    target_save_num,
                    self._state_complete_mask(req, range(target_save_num)))
            req.has_saved_block_num = target_save_num

    def _collect_load_instructions(self, meta: TairKvCacheConnectorMetadata) -> None:
        """Hand the worker the loads admitted by get_num_new_matched_tokens
        whose blocks have since been allocated (update_state_after_alloc).

        Without an allocation the load cannot be addressed -- the worker would
        not know which slots to scatter into -- so it is dropped; vLLM will
        re-query and re-admit it on a later step.
        """
        for load_req in self._waiting_to_load_requests:
            request = self._alive_requests[load_req.req_id]
            if not request.block_ids_per_group:
                continue
            load_req.all_block_ids = [list(b) for b in request.block_ids_per_group]
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
            req = self._alive_requests.get(save_req.req_id)
            if req is None:
                logger.warning("request %s is not alive, skip saving", save_req.req_id)
                continue
            meta.add_save_request(save_req)
            req.sent_saving_count += 1
            if (req.need_report_after_saving_finished and
                    req.scheduled_saving_count == req.sent_saving_count):
                self._retire_request(req)

        self.handle_canceled_save_req()

    def _collect_finish_instructions(self, meta: TairKvCacheConnectorMetadata) -> None:
        """Flush retirements queued since the last step. Runs last on purpose:
        save settlement above may retire a request within this very step."""
        for finish_req in self._waiting_to_finish_requests:
            meta.add_finish_request(finish_req)
        self._waiting_to_finish_requests = []

    def _retire_request(self, req: ReqState) -> None:
        """The request's save obligations are settled: tell the worker to drop
        its mirror and stop tracking the request ourselves."""
        self._waiting_to_finish_requests.append(FinishRequest(req.req_id))
        self._alive_requests.pop(req.req_id)

    def start_save_kvcache_async(self, req_id, token_ids, target_save_num,
                                 state_complete_mask):
        """Ask the manager for write locations for a request's first
        ``target_save_num`` manager blocks.

        ``state_complete_mask[i]`` says whether manager block i has a
        materialized recurrent state. Blocks without one are announced under
        the attention-only spec group, so the manager allocates (and later
        advertises) only the specs that will really hold data -- absence is
        never encoded as a successful write.
        """
        request = {
            "trace_id": "%s_%d" % (req_id, self._epoch),
            "instance_id": self._extra_config.instance_id,
            "block_keys": [],
            "token_ids": token_ids,
            "write_timeout_seconds": self._extra_config.write_timeout_seconds,
        }
        if self._state_group_idxs:
            assert len(state_complete_mask) == target_save_num, (
                f"state mask {len(state_complete_mask)} != {target_save_num} blocks")
            request["location_spec_group_names"] = [
                FULL_SPEC_GROUP if complete else ATTN_SPEC_GROUP
                for complete in state_complete_mask]
            if not all(state_complete_mask):
                logger.info("req:%s saving %d/%d blocks without a recurrent "
                            "state (attention specs only)", req_id,
                            state_complete_mask.count(False), target_save_num)
        try:
            response = self._manager_client.start_write_cache(request)
        except Exception as e:
            logger.warning("start_write_cache error, skip saving: %s", e)
            with self._canceled_save_request_ids_lock:
                self._canceled_save_request_ids.append(req_id)
            return

        locations = response["locations"]
        write_session_id = response["write_session_id"]

        if not locations:
            try:
                self._manager_client.finish_write_cache({
                    "trace_id": "finish_%s" % write_session_id[:8],
                    "instance_id": self._extra_config.instance_id,
                    "write_session_id": write_session_id,
                    "success_blocks": {"bool_masks": {"offset": 0}},
                })
            except Exception as e:
                logger.warning("finish_write_cache failed, session: %s, error: %s",
                               write_session_id, e)
            with self._canceled_save_request_ids_lock:
                self._canceled_save_request_ids.append(req_id)
            return

        need_block_idx = self.parse_block_mask_to_save_indices(response, target_save_num)
        message = CoordinateMessage(time.time(), SendBlockStartEvent(
            request_id=req_id, write_session_id=write_session_id, locations=locations))
        self._coordinator_client.send(CoordinateMsgSerializer.dumps(message))

        with self._waiting_to_save_requests_lock:
            self._waiting_to_save_requests.append(SaveRequest(
                req_id, locations, need_block_idx, write_session_id))

    def handle_canceled_save_req(self):
        with self._canceled_save_request_ids_lock:
            canceled = self._canceled_save_request_ids
            self._canceled_save_request_ids = []
        for req_id in canceled:
            # Cancellations come from http_executor threads; the request may
            # already have been finished and removed by the scheduler loop.
            req = self._alive_requests.get(req_id)
            if req is None:
                logger.warning("canceled save for unknown request %s, skip", req_id)
                continue
            req.sent_saving_count += 1
            if (req.need_report_after_saving_finished and
                    req.scheduled_saving_count == req.sent_saving_count):
                self._retire_request(req)

    def get_finished_count(self):
        # Only rank0 reports finished requests.
        return 1

    def parse_block_mask_to_save_indices(self, response: dict, target_save_num: int) -> List[int]:
        block_mask = response.get("block_mask", {})
        if "offset" in block_mask:
            return list(range(block_mask["offset"], target_save_num))
        values = block_mask.get("bool_masks", {}).get("values", [])
        return [idx for idx, saved in enumerate(values) if not saved]

    # ------------------------------------------------------------------ #
    # Request finish
    # ------------------------------------------------------------------ #
    def request_finished_all_groups(
            self, request: "Request",
            block_ids: Tuple[List[int], ...]) -> Tuple[bool, Optional[dict]]:
        return self._finish_request(request)

    def request_finished(self, request: "Request",
                         block_ids: List[int]) -> Tuple[bool, Optional[dict]]:
        return self._finish_request(request)

    def _finish_request(self, request: "Request") -> Tuple[bool, Optional[dict]]:
        req = self._alive_requests.get(request.request_id)
        if req is None:
            logger.info("request_finished for unknown request: %s", request.request_id)
            return False, {}

        extra_info = {"local_matched_token_num": req.local_matched_token_num,
                      "remote_matched_token_num": req.remote_matched_token_num}

        if req.scheduled_saving_count == req.sent_saving_count:
            self._retire_request(req)
            return True, extra_info

        # Saves still in flight; delay freeing the blocks until they land.
        req.need_report_after_saving_finished = True
        return True, extra_info
