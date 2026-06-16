from schedule_simulator.schedule_emulator.base import ScheduleEmulator
from schedule_simulator.schedule_emulator.types import (
    FakeRequest,
    IterationStats,
    RequestStats,
    RequestStage,
    RequestCacheFetchStats,
)
from schedule_simulator.infer_time_predictor import (
    ScheduleBatch,
    InferTimePredictor,
    LLMPerfTimePredictor,
)
from schedule_simulator.schedule_emulator.schedule_policy import (
    SchedulePolicy,
)
from schedule_simulator.schedule_emulator.prefix_cache import (
    PrefixCache,
    HiRadixCache,
)
from schedule_simulator.schedule_emulator.utils import (
    calc_kv_cache_cell_elems,
    estimate_kv_cache_pool_capacity,
    calc_metrics,
)
from schedule_simulator.schedule_emulator.kvcache_simulation import (
    SimHiRadixCache,
    ReqToTokenPoolHost,
    KVCachePool,
    RadixKey,
)

from kunlun_commons.model_info import ModelInfo
from kunlun_commons.system_info import AcceleratorInfo, DataType

import asyncio
from typing import Optional
import heapq
from kunlun_commons.utils import get_logger

logger = get_logger("schedule_simulator")


class SGLangScheduleEmulator(ScheduleEmulator):
    def __init__(
        self,
        scheduler_config,
        platform_config,
        request_queue,
        response_queue,
        time_predictor: Optional[InferTimePredictor] = None,
        use_real_token: bool = False,
        enable_hierarchical_cache: bool = False,  # add for test case which only use L1 cache
        kvcm_block_size: int = None,
        name: str = "SGLangScheduler",
    ):
        super().__init__(
            scheduler_config,
            platform_config,
            request_queue,
            response_queue,
            name,
        )

        # Keep the request which arrived before current clock.
        self.waiting_queue: list[FakeRequest] = []
        # Keep the request which will arrive in the future.
        self.future_queue: list[FakeRequest] = []
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[])
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        # Init chunked prefill
        self.chunked_req: Optional[FakeRequest] = None
        # Prefix Cache prefetching.
        self.prefetch_queue: list[FakeRequest] = []
        # Keep track of the historical running times for key-value (KV) cache loading.
        self.last_batch_run_timestamp: float = 0
        self.last_batch_run_time: float = 0
        self.last_prefetch_end_time: float = 0
        # Keep all the responded requests that will be used for logging metrics.
        self.completed_requests: list[FakeRequest] = []

        # The event loop will exit once all the expected requests have been sent to the response_queue.
        self.num_requests = (1 << 63) - 1
        if self.scheduler_config.log_metrics_interval is None:
            self.log_metrics_interval = (1 << 63) - 1  # disable
        elif 0 < self.scheduler_config.log_metrics_interval < 1:
            self.log_metrics_interval = max(
                1, self.scheduler_config.log_metrics_interval * self.num_requests
            )
        else:
            self.log_metrics_interval = self.scheduler_config.log_metrics_interval

        if isinstance(self.scheduler_config.model, str):
            self.model = ModelInfo.find_by_model_name(self.scheduler_config.model)
        else:
            self.model = self.scheduler_config.model
        if isinstance(platform_config.device, str):
            self.hw = AcceleratorInfo.find_by_hw_name(platform_config.device)
        else:
            self.hw = platform_config.device

        self.use_real_token = use_real_token
        self.kvcm_block_size = kvcm_block_size
        self.enable_hicache = enable_hierarchical_cache
        if self.scheduler_config.data_type is None:
            self.scheduler_config.data_type = DataType(
                DataType.alias().get(self.model.torch_dtype, "FP16")
            )
        if self.scheduler_config.kv_cache_data_type is None:
            self.scheduler_config.kv_cache_data_type = self.scheduler_config.data_type

        if self.model is None or self.hw is None:
            raise ValueError("Model or hardware not found")

        if time_predictor is None:
            logger.warning(
                "The predictor class is not specified, using LLMPerfTimePredictor."
            )
            time_predictor = LLMPerfTimePredictor(
                self.model, self.hw, self.scheduler_config
            )

        self.time_predictor = time_predictor

        if self.scheduler_config.kv_cache_space_per_token is not None:
            self.kv_cache_space_per_token = self.scheduler_config.kv_cache_space_per_token
        else:
            self.kv_cache_space_per_token = (
                calc_kv_cache_cell_elems(
                    self.model, self.scheduler_config.tp_size, self.scheduler_config.pp_size
                )
                * self.scheduler_config.data_type.bytes
            )

        self.max_num_tokens = 0
        self.rest_num_tokens = 0

        self.pause = None
        # p, d dispatch
        if self.scheduler_config.scenario != "normal":
            self.pause = asyncio.Event()

        self._post_init_()
        self.tree_cache = self._init_tree_cache()
        self.policy: SchedulePolicy = SchedulePolicy(
            self.scheduler_config.schedule_policy, self.tree_cache, self.time_predictor
        )

    def _post_init_(self):
        # adjust the scheduler config

        # page size
        # Ref: https://github.com/sgl-project/sglang/blob/v0.5.2rc2/python/sglang/srt/server_args.py#L531
        # TODO: Adjust the page size based on the attention backend.
        if self.scheduler_config.page_size is None:
            self.scheduler_config.page_size = 1

        # memory fraction
        # Ref: https://github.com/sgl-project/sglang/blob/v0.4.8/python/sglang/srt/server_args.py#L274
        if self.scheduler_config.mem_fraction_static is None:
            parallel_size = (
                self.scheduler_config.tp_size * self.scheduler_config.pp_size
            )
            if self.hw.hbm_capacity_gb < 20:
                # T4, 4080. (chunked_prefill_size 2k, cuda_graph_max_bs 8)
                reserved_mem = 2.8 + parallel_size / 10
            elif self.hw.hbm_capacity_gb < 35:
                # A10, L40, 4090, 5090. (chunked_prefill_size 2k, cuda_graph_max_bs 8)
                reserved_mem = 2.8 + parallel_size / 10
            elif self.hw.hbm_capacity_gb < 90:
                # H100, A100. (chunked_prefill_size 8k, cuda_graph_max_bs 160)
                reserved_mem = 9.5 + parallel_size / 2
            elif self.hw.hbm_capacity_gb < 100:
                # H20. (chunked_prefill_size 8k, cuda_graph_max_bs 256)
                reserved_mem = 12 + parallel_size / 2
            elif self.hw.hbm_capacity_gb < 160:
                # H200. (chunked_prefill_size 8k, cuda_graph_max_bs 256)
                reserved_mem = 12 + parallel_size / 2
            else:
                # B200, MI300. (chunked_prefill_size 16k, cuda_graph_max_bs 512)
                reserved_mem = 32
            # TODO: add more cases here
            # if self.speculative_algorithm is not None:
            #     # draft model and larger cuda graph buffers
            #     reserved_mem += 2
            # if self.enable_dp_attention:
            #     reserved_mem += 4
            self.scheduler_config.mem_fraction_static = round(
                (self.hw.hbm_capacity_gb - reserved_mem) / self.hw.hbm_capacity_gb, 3
            )

        # chunked prefill size
        # Ref: https://github.com/sgl-project/sglang/blob/v0.4.8/python/sglang/srt/server_args.py#L318
        if self.scheduler_config.chunked_prefill_size is None:
            if self.hw.hbm_capacity_gb < 35:
                self.scheduler_config.chunked_prefill_size = 2048
            elif self.hw.hbm_capacity_gb < 160:
                self.scheduler_config.chunked_prefill_size = 8192
            else:
                self.scheduler_config.chunked_prefill_size = 16384

        # max number of tokens (L1 device cache capacity)
        if self.scheduler_config.max_num_tokens is not None:
            self.max_num_tokens = self.scheduler_config.max_num_tokens
        else:
            self.max_num_tokens = estimate_kv_cache_pool_capacity(
                self.model, self.hw, self.scheduler_config
            )
        self.rest_num_tokens = self.max_num_tokens

        logger.debug(f"The max number of tokens is {self.max_num_tokens}.")
        if self.max_num_tokens <= 0:
            raise RuntimeError("There is not enough memory to run the model.")

    def _init_tree_cache(self) -> PrefixCache:
        if self.use_real_token:
            max_num_tokens = (
                self.max_num_tokens
                if not self.kvcm_block_size
                else self.max_num_tokens // self.kvcm_block_size
            )

            # Limit the maximum number of running requests to prevent excessive memory allocation.
            self.req_pool = ReqToTokenPoolHost(
                size=min(self.scheduler_config.max_running_requests, 4096),
                max_context_len=self.scheduler_config.chunked_prefill_size,
            )
            self.kv_pool = KVCachePool(
                size=max_num_tokens, page_size=self.scheduler_config.page_size
            )
            return SimHiRadixCache(
                req_to_token_pool=self.req_pool,
                token_to_kv_pool_allocator=self.kv_pool,
                page_size=self.scheduler_config.page_size,
                hicache_size=(self.scheduler_config.l2_cache_num_tokens
                             if self.scheduler_config.l2_cache_num_tokens is not None
                             else max_num_tokens * 2),
                hicache_write_policy=self.scheduler_config.hicache_write_policy,
                eviction_policy="lru",
                prefetch_queue=self.prefetch_queue,
                hicache_storage_backend=None,  # Reserved for kvcm
                # hicache_storage_prefetch_policy="best_effort",
                hicache_storage_prefetch_policy="wait_complete",
                storage_backend_extra_config=(256, 1, 0.25),
                global_values=self.global_values,
                kv_cache_space_per_token=self.kv_cache_space_per_token,
                platform_config=self.platform_config,
                kvcm_block_size=self.kvcm_block_size,
                is_eagle=False,
                enable_stats=self.scheduler_config.enable_stats,
            )
        elif self.scheduler_config.hicache_storage_backend is None:
            return PrefixCache(
                self.platform_config,
                self.kv_cache_space_per_token,
                self.scheduler_config.page_size,
                self.global_values,
                self.scheduler_config.enable_stats,
            )
        else:
            # TODO: init different backend according the backend args.
            return HiRadixCache(
                self.platform_config,
                self.kv_cache_space_per_token,
                self.scheduler_config.page_size,
                self.global_values,
                self.scheduler_config.hicache_storage_prefetch_policy,
                self.scheduler_config.enable_stats,
            )

    def set_num_requests(self, num_requests: int):
        self.num_requests = num_requests
        if (
            self.scheduler_config.log_metrics_interval is not None
            and 0 < self.scheduler_config.log_metrics_interval < 1
        ):
            self.log_metrics_interval = max(
                1, self.scheduler_config.log_metrics_interval * num_requests
            )
            logger.info(
                f"Metrics Logger is enable, and the logging interval is {self.log_metrics_interval} requests."
            )

    def reset(self):
        self.last_batch_run_time: float = 0
        self.last_prefetch_end_time: float = 0
        self.completed_requests: list[FakeRequest] = []

        self.global_values.reset()
        self.iter_stats.clear()

    async def set_pause(self):
        self.pause.clear()  # 设置 event_loop 暂停

    async def cancel_pause(self):
        self.pause.set()  # 取消 event_loop 暂停

    async def get_load(self, curtime: float) -> int:  # power_of_two
        num_tokens = 0
        for i in self.waiting_queue + self.future_queue + self.running_batch.reqs:
            num_tokens += i.input_token_length + len(i.gen_token_latencies)
        if self.chunked_req:
            num_tokens += self.chunked_req.input_token_length

        is_prefill = self.scheduler_config.scenario == "disagg_prefill"
        if self.last_batch is not None and self.last_batch.reqs:
            if curtime <= self.last_batch_run_timestamp:
                for req in self.last_batch.reqs:
                    if is_prefill:
                        if self.chunked_req and self.chunked_req.id == req.id:
                            continue
                    elif not req.is_finished():
                        continue
                    num_tokens += (
                        req.input_token_length + len(req.gen_token_latencies) - 1
                    )

        return num_tokens

    async def is_idle(self, curtime: float | None = None) -> bool:
        has_active_or_waiting_work = (
            self.chunked_req is not None
            or self.waiting_queue
            or self.future_queue
            or self.running_batch.reqs
        )

        if has_active_or_waiting_work:
            return False

        # 没有传入时间，是查看当前schedule有没有事件可以跑
        if curtime is None:
            if self.request_queue.qsize() > 0:
                return False

        # 传入时间，是关心当前有没有请求在跑
        if curtime is not None and curtime < self.last_batch_run_timestamp:
            return False

        return True

    async def event_loop(self):
        if self.scheduler_config.enable_real_time_request:
            logger.info(
                f"Real-time request mode is enabled. The expected number of requests is {self.num_requests}"
            )
        while True:
            if self.pause is not None:
                await self.pause.wait()
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.global_values.last_batch_run_time = self.last_batch_run_time

            # If any requests have not yet arrived before the current clock,
            # the system must wait for all of them in real time request mode.
            # Caution: If the currency is set to 1 in a normal request, the system may hang or fail.
            if (
                self.scheduler_config.enable_real_time_request
                and not self.is_all_request_arrived()
            ):
                await asyncio.sleep(0)
                continue

            if self.scheduler_config.request_level_scheduling:
                self._run_request_level()
            else:
                batch = self.get_next_batch_to_run()
                self.cur_batch = batch

                start_timestamp = self.global_values.clock
                if batch:
                    self.run_batch(batch)
                    self.last_batch_run_time = self.global_values.clock - start_timestamp
                    self.last_batch_run_timestamp = self.global_values.clock
                    self.last_batch = ScheduleBatch(reqs=list(batch.reqs))
                    self.process_batch_result(batch)
                else:
                    self.adjust_global_clock()

            if self.num_requests <= 0:
                logger.debug(
                    "All requests have been sent to the response queue. The scheduler event loop exits now."
                )
                return

            # Yield control of the event loop strategically; otherwise, the benchmark task may hang.
            await asyncio.sleep(0)


    def _run_request_level(self):
        import heapq as _heapq
        while (
            len(self.future_queue) > 0
            and self.future_queue[0].last_event_time <= self.global_values.clock
        ):
            req = _heapq.heappop(self.future_queue)
            req.queue_time_start = req.last_event_time
            self.waiting_queue.append(req)
            self.tree_cache.add_to_prefetch_queue(req)

        if not self.waiting_queue:
            if self.future_queue:
                self.global_values.clock = max(
                    self.future_queue[0].last_event_time, self.global_values.clock
                )
            return

        req = self.waiting_queue.pop(0)
        req.queue_time_end = self.global_values.clock

        # Optimizer internally handles P2P: GetCacheLocation fills peer-hit blocks
        # into the local engine via FillEngineFromHitIndices. Both engine_hit and
        # peer_hit are already "local" after add_to_prefetch_queue, so sum them directly.
        match_result = self.tree_cache.match_prefix(req)
        cached = match_result.device_hit_length + match_result.host_hit_length
        uncached = req.input_token_length - cached
        req.final_reused_tokens = cached
        req.context_prefill_start = cached
        req.context_prefill_end = req.input_token_length

        if hasattr(self.time_predictor, "predict_request_time"):
            latency = self.time_predictor.predict_request_time(max(uncached, 1), cached)
        else:
            from schedule_simulator.infer_time_predictor.base import ScheduleBatch as SB, ScheduleRequest as SR
            batch = SB(reqs=[SR(input_length=max(uncached, 1), past_kv_length=cached)])
            latency = self.time_predictor.predict_infer_time(batch)

        self.global_values.clock += latency
        self.global_values.iteration += 1

        req.gen_token_latencies.append(self.global_values.clock - req.last_event_time)
        req.last_event_time = self.global_values.clock
        req.stage = RequestStage.COMPLETE
        self.completed_requests.append(req)
        self.num_requests -= 1
        self.response_queue.put_nowait(req)

        self.tree_cache.on_request_complete(req, self.global_values.clock)
        self.tree_cache.drop_match_result(req)

        if self.scheduler_config.enable_stats:
            self.iter_stats.append(
                IterationStats(
                    timestamp=self.global_values.clock,
                    iter=self.global_values.iteration,
                    iter_latency_ms=latency * 1e3,
                    num_context_requests=1,
                    num_ctx_tokens=max(uncached, 1),
                    num_gen_requests=0,
                    request_stats=[RequestStats(
                        req.id, "prefill",
                        context_prefill_start=req.context_prefill_start,
                        context_prefill_end=req.context_prefill_end,
                        num_generated_tokens=1,
                    )],
                    num_waiting_requests=len(self.waiting_queue),
                )
            )

    def adjust_global_clock(self):
        if len(self.waiting_queue) == 0 and len(self.future_queue) == 0:
            return

        if any(req.is_prefetching() for req in self.waiting_queue):
            # How to estimate the prefetching time?
            self.global_values.clock += 0.01
            self.last_batch_run_time = 0.01
            return

        if len(self.waiting_queue) == 0 and len(self.future_queue) != 0:
            self.global_values.clock = max(
                self.future_queue[0].last_event_time, self.global_values.clock
            )
            self.last_batch_run_time = 0

    def run_batch(self, batch: ScheduleBatch):
        if batch.is_empty():
            return

        latency = self.time_predictor.predict_infer_time(batch)
        self.global_values.iteration += 1
        self.global_values.clock += latency

        if self.scheduler_config.enable_stats:
            request_stats = []
            # Generate request statistics before updating the request.
            for req in batch.reqs:
                request_stats.append(
                    RequestStats(
                        req.id,
                        "prefill" if req.is_prefilling() else "decode",
                        context_prefill_start=req.context_prefill_start,
                        context_prefill_end=req.context_prefill_end,
                        num_generated_tokens=len(req.gen_token_latencies),
                    )
                )

            self.iter_stats.append(
                IterationStats(
                    timestamp=self.global_values.clock,
                    iter=self.global_values.iteration,
                    iter_latency_ms=latency * 1e3,
                    num_context_requests=batch.num_ctx_requests,
                    num_ctx_tokens=batch.num_context_tokens,
                    num_gen_requests=batch.num_gen_requests,
                    request_stats=request_stats,
                    num_waiting_requests=len(self.waiting_queue),
                )
            )

        # update the event timestamp of requests
        for req in batch.reqs:
            # shift the next chunked prefill start position.
            if req.context_prefill_end > req.context_prefill_start:
                req.context_prefill_start = req.context_prefill_end
            if req.remaining_prefill_tokens != 0:
                # The request is chunked and not finished.
                # Checking remaining tokens after adjusting chunked prefill positions!!!
                continue
            req.gen_token_latencies.append(
                self.global_values.clock - req.last_event_time
            )
            req.last_event_time = self.global_values.clock

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        new_batch = self.get_new_batch_prefill()

        if new_batch is not None:
            logger.debug(
                f"[{self.name}] Iteration: {self.global_values.iteration}, Prefill Request: {new_batch.batch_size}, "
                f"Future Queue: {len(self.future_queue)}, Waiting Queue: {len(self.waiting_queue)}"
            )
            ret = new_batch
        elif not self.running_batch.is_empty():
            ret = self.running_batch
        else:
            ret = None
        return ret

    def prefix_loadback(
        self,
        req: FakeRequest,
    ):
        if not isinstance(self.tree_cache, SimHiRadixCache):
            print("loadback func only support SimHiRadixCache.")
            return None
        loading_len, last_node = self.tree_cache.init_load_back(req.last_host_node)
        req.prefix_indices_len = req.prefix_indices_len + loading_len
        req.last_node = last_node
        req.last_matched_prefix_len = req.prefix_indices_len

    def match_and_update_req(self, req: FakeRequest, enable_hicache: bool = True):
        if not isinstance(self.tree_cache, SimHiRadixCache):
            logger.error("only support SimHiRadixCache.")
            return None
        match_result = self.tree_cache.match_prefix(
            key=RadixKey(token_ids=req.origin_input_ids, extra_key=None)
        )
        # update req
        req.prefix_indices_len = len(match_result.device_indices)
        req.last_node = match_result.last_device_node
        req.last_host_node = match_result.last_host_node
        req.host_hit_len = match_result.host_hit_length
        req.last_matched_prefix_len = len(match_result.device_indices)
        # loadback if use hicache and hit in L2 cache
        if enable_hicache and req.host_hit_len > 0:
            self.prefix_loadback(req)
        logger.debug(
            f"[match result] {req.id=}, {req.prefix_indices_len=}, {req.host_hit_len=}"
        )

        req_len = len(req.origin_input_ids)
        new_token_len = req_len - req.prefix_indices_len
        # similar to alloc_req_slots() method
        req_pool_idx = self.req_pool.alloc(1)
        req.req_pool_idx = req_pool_idx[0]
        self.req_pool.write(req.req_pool_idx, req_len)
        # similar to alloc_token_slots()/alloc_paged_token_slots_extend()
        if self.kv_pool is not None:
            if self.kv_pool.available_size < new_token_len:
                logger.debug(
                    f"[Evict] prefill stage: req[{req.id}] try to evict {new_token_len}, {self.kv_pool.available_size=}, {self.kv_pool.evictable_size=}"
                )
                self.tree_cache.evict(new_token_len)
                if self.kv_pool.available_size < new_token_len:
                    self.tree_cache.pretty_print()
                    raise MemoryError(
                        f"[Error] prefill stage: req[{req.id}] OOM Error!! No enough space in mem pool"
                    )
            logger.debug(
                f"[Alloc] prefill stage: req[{req.id}] alloc {new_token_len}, {self.kv_pool.available_size=}, {self.kv_pool.evictable_size=}"
            )
            self.kv_pool.alloc(new_token_len)

        # unlock in cache_finished_req
        self.tree_cache.inc_lock_ref(req.last_node)

        if self.scheduler_config.enable_stats:
            fetch_request_stats = self.tree_cache.get_request_fetch_stats()
            if req not in fetch_request_stats:
                fetch_request_stats[req] = RequestCacheFetchStats(
                    req_id=req.id,
                )
            fetch_request_stats[req].actual_prefix_len = req.prefix_indices_len

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        reqs = []

        if self.scheduler_config.chunked_prefill_size != -1:
            batch_remaining_tokens = min(
                self.scheduler_config.chunked_prefill_size,
                self.scheduler_config.max_prefill_tokens,
            )
        else:
            batch_remaining_tokens = self.scheduler_config.max_prefill_tokens

        if self.chunked_req is not None:
            # Check the last chunked request.
            consumed_tokens = min(
                batch_remaining_tokens, self.chunked_req.remaining_prefill_tokens
            )
            # The start position of chunked prefill will be adjusted in run_bacth()
            self.chunked_req.context_prefill_end += consumed_tokens
            batch_remaining_tokens -= consumed_tokens
            reqs.append(self.chunked_req)
            if not self.chunked_req.has_next_chunk():
                self.chunked_req = None
        if batch_remaining_tokens == 0:
            return ScheduleBatch(reqs)

        if self.scheduler_config.schedule_policy not in ["mcr", "plg"]:
            # prefetch_queue in the tree cache is ordered by arrival time.
            prefetch_queue = self.tree_cache.prefetch_queue
        else:
            # The waiting reqs is ordered by the schedule policy.
            prefetch_queue = self.waiting_queue
        for req in prefetch_queue:
            if req.stage == RequestStage.PREFETCHING:
                # KV cache prefetching is performed prior to the last batch execution.
                # Edge case: New requests may arrive after the final prefetching completes.
                remaining_disk_prefetch_time = (
                    self.global_values.clock
                    - self.last_batch_run_time
                    - max(self.last_prefetch_end_time, req.last_event_time)
                )
                if remaining_disk_prefetch_time <= 0:
                    break
                fetch_result = self.tree_cache.prefetch_from_storage(
                    req,
                    remaining_disk_prefetch_time,
                )
                self.last_prefetch_end_time = (
                    max(req.last_event_time, self.last_prefetch_end_time)
                    + fetch_result.latency_disk_to_host
                )

        self.policy.calc_priority(self.waiting_queue, self.use_real_token)

        waiting_reqs = self.waiting_queue.copy()
        self.waiting_queue.clear()
        remaining_host_prefetch_time = self.last_batch_run_time
        for idx, req in enumerate(waiting_reqs):
            if req.input_token_length + req.output_token_length > self.max_num_tokens:
                # The request is overlarge.
                req.stage = RequestStage.FAILED
                self.response_queue.put_nowait(req)
                continue
            if (
                batch_remaining_tokens <= 0
                or self.rest_num_tokens
                < (req.input_token_length + req.output_token_length)
                or len(reqs) + self.running_batch.batch_size
                >= self.scheduler_config.max_running_requests
            ):
                # it hasn't enough tokens in current batch for running, which is limited by the chunk prefill size and the max tokens.
                self.waiting_queue.extend(waiting_reqs[idx:])
                break

            if self.use_real_token and self.enable_hicache:
                # Check if prefetching is complete
                prefetch_done = self.tree_cache.check_prefetch_progress(req)
                if not prefetch_done:
                    # If not complete, return the item to the waiting queue.
                    self.waiting_queue.append(req)
                    continue
                self.match_and_update_req(req)
                latency_host_to_device = 0
                if req.host_hit_len > 0:
                    # L2 -> L1 time cost
                    latency_host_to_device = (
                        req.host_hit_len
                        * self.kv_cache_space_per_token
                        / self.platform_config.memory_read_bandwidth
                    )
                    if self.kvcm_block_size:
                        latency_host_to_device *= self.kvcm_block_size
                    logger.debug(
                        f"[loadback]  req[{req.id}] loadback {req.host_hit_len} tokens, time cost is {latency_host_to_device:.4f}"
                    )
                    remaining_host_prefetch_time -= latency_host_to_device
                    if self.scheduler_config.enable_stats and self.enable_hicache:
                        fetch_request_stats = self.tree_cache.get_request_fetch_stats()
                        fetch_request_stats[
                            req
                        ].latency_host_to_device = latency_host_to_device
                        fetch_request_stats[req].num_token_from_host = (
                            req.host_hit_len
                            if not self.kvcm_block_size
                            else (req.host_hit_len * self.kvcm_block_size)
                        )

                # Parameters related to chunkprefill
                req.context_prefill_start = (
                    req.prefix_indices_len
                    if not self.kvcm_block_size
                    else (req.prefix_indices_len * self.kvcm_block_size)
                )
                req.context_prefill_end = req.context_prefill_start
                # Used for calculating hit rate.
                req.final_reused_tokens = req.context_prefill_start
            elif self.use_real_token and not self.enable_hicache:
                self.match_and_update_req(req, enable_hicache=False)
                # only use L1 cache
                # Parameters related to chunkprefill
                req.context_prefill_start = (
                    req.prefix_indices_len
                    if not self.kvcm_block_size
                    else (req.prefix_indices_len * self.kvcm_block_size)
                )
                req.context_prefill_end = req.context_prefill_start
                # Used for calculating hit rate.
                req.final_reused_tokens = req.context_prefill_start

            else:
                # original prefetch and loadback
                prefetch_done = self.tree_cache.check_prefetch_progress(req)
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    self.waiting_queue.append(req)
                    continue

                if req.stage != RequestStage.READY:
                    self.waiting_queue.append(req)
                    continue

                # allocate the kv cache space for the new request
                self.rest_num_tokens -= req.input_token_length + req.output_token_length

                fetch_result = self.tree_cache.on_board_from_host(req)
                match_result = self.tree_cache.match_prefix(req)
                req.final_reused_tokens = match_result.device_hit_length
                # There is no cost to retrieve data from the device-side cache.
                req.context_prefill_start = match_result.device_hit_length
                req.context_prefill_end = match_result.device_hit_length
                remaining_host_prefetch_time -= fetch_result.latency_host_to_device
                self.tree_cache.drop_match_result(req)

            consumed_tokens = min(batch_remaining_tokens, req.remaining_prefill_tokens)
            req.context_prefill_end += consumed_tokens
            # If a prefill request's input hits all KV caches, the remaining_prefill_tokens is 0.
            batch_remaining_tokens -= max(consumed_tokens, 1)
            # For chunk prefill
            if req.has_next_chunk():
                self.chunked_req = req

            req.stage = RequestStage.PREFILLING
            req.queue_time_end = self.global_values.clock
            reqs.append(req)

        if remaining_host_prefetch_time < 0:
            # Part of the time consumed in transferring the cache from host to device can't be overlapped with last running batch.
            self.global_values.clock += abs(remaining_host_prefetch_time)

        if len(reqs) > 0:
            return ScheduleBatch(reqs=reqs)
        else:
            return None

    def recv_requests(self) -> list[FakeRequest]:
        reqs = []
        while True:
            try:
                req = self.request_queue.get_nowait()
                reqs.append(req)
            except asyncio.QueueEmpty:
                break
        return reqs

    def format_with_scenario(self, req: FakeRequest) -> FakeRequest:
        if self.scheduler_config.scenario == "normal":
            return req
        elif self.scheduler_config.scenario == "disagg_prefill":
            req.output_token_length = 1
        elif self.scheduler_config.scenario == "disagg_decode":
            # Tokens requiring computation will be fetched from prefill instance.
            req.disk_cache_hit_length += req.input_token_length - (
                req.device_cache_hit_length
                + req.host_cache_hit_length
                + req.disk_cache_hit_length
            )
        return req

    def process_input_requests(self, new_reqs: list[FakeRequest]):
        for req in new_reqs:
            heapq.heappush(self.future_queue, self.format_with_scenario(req))

        while (
            len(self.future_queue) > 0
            # For the overlapping schedule, the waiting queue only keeps the requests that arrived before the start of the last batch.
            and self.future_queue[0].last_event_time
            <= self.global_values.last_batch_start
        ):
            req = heapq.heappop(self.future_queue)
            # add blocksize of kvcm
            if self.kvcm_block_size:
                req.block_size = self.kvcm_block_size
            req.queue_time_start = req.last_event_time
            self.waiting_queue.append(req)

            if self.use_real_token and self.enable_hicache:
                self.tree_cache.add_to_prefetch_queue(req, self.global_values.clock)
            else:
                self.tree_cache.add_to_prefetch_queue(req)

        if new_reqs:
            logger.debug(
                f"[{self.name}] Iteration: {self.global_values.iteration}, New Request: {len(new_reqs)}, Future Queue: {len(self.future_queue)}, Waiting Queue: {len(self.waiting_queue)}"
            )

    def is_all_request_arrived(self) -> bool:
        if self.num_requests == (
            len(self.waiting_queue)
            + self.running_batch.batch_size
            + (1 if self.chunked_req else 0)
        ):
            return True
        elif (
            len(self.future_queue) > 0
            and self.future_queue[-1].last_event_time
            > self.global_values.last_batch_start
        ):
            return True
        else:
            return False

    def process_batch_result(self, batch: ScheduleBatch):
        running_reqs = {req.id: req for req in self.running_batch.reqs}
        has_request_complete = False
        for req in batch.reqs:
            if req.is_finished():
                has_request_complete = True
                req.stage = RequestStage.COMPLETE
                self.completed_requests.append(req)
                self.num_requests -= 1
                self.response_queue.put_nowait(req)
                # If the output length is 1, the request might not be added to the running requests.
                if req.id in running_reqs:
                    running_reqs.pop(req.id)
                if self.use_real_token:
                    # free eos token, only support for page_size=1 now
                    self.kv_pool.free(1)
                    logger.debug(f"[insert] req finish, cache finished req[{req.id}]")
                    req.fill_ids = req.origin_input_ids + req.output_ids
                    self.tree_cache.cache_finished_req(req)
                    self.tree_cache.write_backup_storage(
                        req, timestamp=self.global_values.clock
                    )
                else:
                    # Free the kv cache space for the completed request.
                    self.rest_num_tokens += (
                        req.input_token_length + req.output_token_length
                    )
                    self.tree_cache.on_request_complete(req, self.global_values.clock)
            else:
                if req.is_prefilling() and req.remaining_prefill_tokens == 0:
                    req.stage = RequestStage.DECODING
                    # allocate for decode
                    if self.use_real_token:
                        num_decode_token = len(req.output_ids)
                        # After prefill stage, cache the prefill results and update the fill_ids.
                        req.fill_ids = req.origin_input_ids
                        self.tree_cache.cache_unfinished_req(req)
                        if self.kv_pool.available_size < num_decode_token:
                            logger.debug(
                                f"[Evict] decode stage: req[{req.id}] try to evict {num_decode_token}, \
                                  {self.kv_pool.available_size=}, {self.kv_pool.evictable_size=}"
                            )
                            self.tree_cache.evict(num_decode_token)
                            if self.kv_pool.available_size < num_decode_token:
                                self.tree_cache.pretty_print()
                                raise MemoryError(
                                    f"[Error] decode stage: req[{req.id}] OOM Error!! no enough space in mem pool,"
                                    f" available size is {self.kv_pool.available_size}, evictable size is {self.kv_pool.evictable_size}"
                                )
                        self.kv_pool.alloc(num_decode_token)
                        logger.debug(
                            f"[Alloc] decode stage: req[{req.id}] alloc {num_decode_token}, \
                              {self.kv_pool.available_size=}, {self.kv_pool.evictable_size=}"
                        )
                if req != self.chunked_req and req.id not in running_reqs:
                    running_reqs[req.id] = req
        self.running_batch.reqs = list(running_reqs.values())
        if (
            has_request_complete
            and len(self.completed_requests) > 0
            and (len(self.completed_requests) % int(self.log_metrics_interval) == 0)
        ):
            logger.info(calc_metrics(self.completed_requests))
