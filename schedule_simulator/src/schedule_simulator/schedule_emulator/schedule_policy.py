from enum import Enum
from typing import Union

from schedule_simulator.schedule_emulator.prefix_cache import PrefixCache
from schedule_simulator.schedule_emulator.types import (
    FakeRequest,
    RequestStage,
)
from schedule_simulator.infer_time_predictor import (
    InferTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)


class RequestSchedulePolicy(Enum):
    # Scheduling policies that are aware of the tree cache.
    LPM = "lpm"  # longest prefix match
    # Scheduling policies that are used for insight analysis.
    PLG = "plg"  # Insight: prefill latency greedy
    MCR = "mcr"  # Insight: max cache reusing

    # Scheduling policies that are not aware of the tree cache.
    FCFS = "fcfs"  # first come first serve
    LOF = "lof"  # longest output first


class SchedulePolicy:
    CACHE_TO_PREFETCH_THRESHOLD = 32

    def __init__(
        self,
        policy: Union[RequestSchedulePolicy, str],
        tree_cache: PrefixCache,
        time_predictor: InferTimePredictor,
    ):
        if isinstance(policy, str):
            self.policy = RequestSchedulePolicy(policy)
        else:
            self.policy = policy

        self.tree_cache = tree_cache
        self.time_predictor = time_predictor

    def calc_priority(
        self,
        waiting_queue: list[FakeRequest],
        use_real_token: bool = False,
    ) -> None:
        """Sort the waiting queue in-place."""
        if len(waiting_queue) == 0:
            return
        if use_real_token:
            self._sort_with_real_token(waiting_queue)
            return
        if self.policy == RequestSchedulePolicy.FCFS:
            self._sort_by_arriving_timestamp(waiting_queue)
        elif self.policy == RequestSchedulePolicy.LOF:
            self._sort_by_longest_output(waiting_queue)
        elif self.policy == RequestSchedulePolicy.PLG:
            self._sort_by_ttft_greedy(waiting_queue)
        elif self.policy == RequestSchedulePolicy.MCR:
            self._sort_by_max_cache_reused(waiting_queue)
        else:
            raise ValueError("Unsupported policy")

    def _sort_with_real_token(self, waiting_queue: list[FakeRequest]):
        """Reserved for prefetch"""
        waiting_queue.sort(key=lambda req: req.last_event_time)
        return

    def _sort_by_ttft_greedy(
        self,
        waiting_queue: list[FakeRequest],
    ):
        # waiting_queue.sort(key=lambda req: req.last_event_time)

        better_for_cache_reqs = []
        better_for_computation_reqs = []
        last_req_fetch_end = 0
        for req in waiting_queue:
            if req.stage == RequestStage.READY:
                better_for_computation_reqs.append(req)
                continue
            # estimate the execution time
            matched_cache = self.tree_cache.match_prefix(req)
            if matched_cache.disk_hit_length == 0:
                req.stage = RequestStage.READY
                better_for_computation_reqs.append(req)
                continue
            batch = ScheduleBatch(
                reqs=[
                    ScheduleRequest(
                        input_length=matched_cache.disk_hit_length,
                        past_kv_length=0,
                    )
                ],
            )
            infer_time = self.time_predictor.predict_infer_time(batch)
            # Estimate the prefetching time.
            fetch_result = self.tree_cache.estimate_prefetch_from_storage(
                req, max_time=float("+inf")
            )
            if last_req_fetch_end + fetch_result.latency_disk_to_host <= infer_time:
                req.stage = RequestStage.PREFETCHING
                better_for_cache_reqs.append(req)
            else:
                req.stage = RequestStage.READY
                better_for_computation_reqs.append(req)
            last_req_fetch_end += fetch_result.latency_disk_to_host

        waiting_queue.clear()
        waiting_queue.extend(better_for_computation_reqs)
        waiting_queue.extend(better_for_cache_reqs)

    def _sort_by_max_cache_reused(self, waiting_queue: list[FakeRequest]):
        # First sort by last_event_time to ensure consistent ordering
        # waiting_queue.sort(key=lambda req: req.last_event_time)

        not_ready_reqs = []
        while len(waiting_queue) != 0 and waiting_queue[-1].stage != RequestStage.READY:
            not_ready_reqs.append(waiting_queue.pop())
        not_ready_reqs.reverse()

        prefetching_reqs = []
        for req in not_ready_reqs:
            m = self.tree_cache.match_prefix(req)
            if m.disk_hit_length == 0:
                req.stage = RequestStage.READY
                waiting_queue.append(req)
                # remove the request from the prefetching queue.
                self.tree_cache.prefetch_from_storage(req, float("inf"))
            else:
                req.stage = RequestStage.PREFETCHING
                prefetching_reqs.append(req)
        waiting_queue.extend(prefetching_reqs)

    def _sort_by_arriving_timestamp(
        self,
        waiting_queue: list[FakeRequest],
    ) -> None:
        # waiting_queue.sort(key=lambda req: req.last_event_time)
        for req in reversed(waiting_queue):
            # if req.is_idle():  # The frequent calls to req.is_idle() are time-consuming.
            if req.stage == RequestStage.IDLE:
                matched = self.tree_cache.match_prefix(req)
                if matched.disk_hit_length:
                    req.stage = RequestStage.PREFETCHING
                else:
                    req.stage = RequestStage.READY
            else:
                break

    def _sort_by_longest_output(
        self,
        waiting_queue: list[FakeRequest],
    ) -> None:
        """Sorts the waiting queue based on the longest output (max_new_tokens)."""
        waiting_queue.sort(
            key=lambda x: x.output_token_length,
            reverse=True,
        )
        for req in waiting_queue:
            if req.stage == RequestStage.IDLE:
                matched = self.tree_cache.match_prefix(req)
                if matched.disk_hit_length:
                    req.stage = RequestStage.PREFETCHING
                else:
                    req.stage = RequestStage.READY

    def _sort_by_longest_prefix(
        self,
        waiting_queue: list[FakeRequest],
    ):
        raise NotImplementedError("Require token ids to match the prefix.")
