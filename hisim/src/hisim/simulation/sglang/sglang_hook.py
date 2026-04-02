import torch
import time
from dataclasses import asdict
from collections import defaultdict
import json
import os
import heapq
import importlib

from hisim.utils import get_logger

from hisim.hook import BaseHook
from hisim.simulation.types import (
    MockSimulationMode,
    RequestStats,
)
from hisim.hook.utils import get_obj_from_args
from hisim.utils.json import CustomJsonEncoder
from hisim.simulation.manager import StateManager, ConfigManager, Envs
from hisim.time_predictor import (
    InferTimePredictor,
    FakeRequest,
    ScheduleBatch as HisimScheduleBatch,
)
from hisim.simulation.utils import (
    calc_metrics,
    estimate_kv_cache_pool_capacity,
)
from hisim.simulation.sglang.version import VersionDispatcher


logger = get_logger("hisim")


class C_EngineHook(BaseHook):
    HOOK_CLASS_NAME = "Engine"
    HOOK_MODULE_NAME = "sglang.srt.entrypoints.engine"

    @classmethod
    def hook(cls, target):
        def hook_clear_hicache_storage(self):
            return self.loop.run_until_complete(
                self.tokenizer_manager.clear_hicache_storage()
            )

        target.clear_hicache_storage = hook_clear_hicache_storage


class C_ModelRunnerHook(BaseHook):
    HOOK_CLASS_NAME = "ModelRunner"
    HOOK_MODULE_NAME = "sglang.srt.model_executor.model_runner"

    @classmethod
    def hook(cls, target):
        _version_dispatcher = VersionDispatcher()

        def override_initialize(self, *args, **kwargs):
            class MockModel:
                def forward(self):
                    pass

            self.model = MockModel()

            self.dtype = self.model_config.dtype
            self.kv_cache_dtype = (
                self.dtype
            )  # FIXME: get kv cache dtype from server args

            model = ConfigManager.get_model_info(self.model_config.hf_config.__dict__)
            hw = ConfigManager.get_accelerator_info()
            config = ConfigManager.get_scheduler_config(
                self.server_args.__dict__,
                "sglang",
                self.model_config.hf_config.__dict__,
            )

            assert model is not None and hw is not None and config is not None

            if self.server_args.max_total_tokens is not None:
                self.max_total_num_tokens = self.server_args.max_total_tokens
            else:
                self.max_total_num_tokens = estimate_kv_cache_pool_capacity(
                    model, hw, config
                )

            if hasattr(self, "page_size") and self.page_size > 1:
                self.max_total_num_tokens = (
                    self.max_total_num_tokens // self.page_size * self.page_size
                )

            max_num_reqs = min(
                max(
                    int(self.max_total_num_tokens / model.max_seq_len * 512),
                    2048,
                ),
                4096,
            )
            logger.info(
                f"Model runner initialized with {self.max_total_num_tokens} tokens. Maximum number of requests: {max_num_reqs}"
            )

            model_has_mtp_layers = (
                self.model_config.num_nextn_predict_layers is not None
            )
            model_num_layers = (
                self.model_config.num_nextn_predict_layers
                if self.is_draft_worker and model_has_mtp_layers
                else max(
                    self.model_config.num_hidden_layers,
                    self.model_config.num_attention_layers,
                )
            )
            self.start_layer = getattr(self.model, "start_layer", 0)
            self.end_layer = getattr(self.model, "end_layer", model_num_layers)
            self.num_effective_layers = self.end_layer - self.start_layer

            from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

            self.req_to_token_pool = ReqToTokenPool(
                size=max_num_reqs,
                max_context_len=model.max_seq_len,
                device=self.device,
                enable_memory_saver=False,
            )

            # During simulation, the actual data in kv cache pool is not important since the MHA computation is skipped,
            # so the head_num and head_dim can be set to 1 to reduce the memory usage.
            # And the scheduler only matters about whether the token_to_kv_pool can be allocated enough space for the requests,
            # so the pool's implementation is not important and can be replaced with `MHATokenToKVPool` that only simulates the allocation logic.
            from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

            self.token_to_kv_pool = MHATokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=1,  # Overwrite head_num and head_dim to 1.
                head_dim=1,
                layer_num=self.num_effective_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                enable_alt_stream=False,
            )

            if self.page_size == 1:
                from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

                self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=False,
                )
            else:
                from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

                self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=False,
                )

            # self.init_memory_pool(50)
            self.attn_backend = None
            self.graph_mem_usage = 0
            self.weight_load_mem_usage = 10

            self.max_running_requests = min(
                (
                    self.max_total_num_tokens // 2
                    if self.server_args.max_running_requests is None
                    else self.server_args.max_running_requests
                    // (
                        self.server_args.dp_size
                        if self.server_args.enable_dp_attention
                        else 1
                    )
                ),
                self.req_to_token_pool.size,
            )

            return

        def wrapped_forward_v1(self, *args, **kwargs):
            batch = args[0]
            from sglang.srt.layers.logits_processor import LogitsProcessorOutput

            output = LogitsProcessorOutput(
                next_token_logits=torch.empty(
                    size=(batch.batch_size, self.model_config.vocab_size),
                    device=self.device,
                )
            )

            return output, False

        _version_dispatcher.register_method(
            "forward", ["0.5.6", "0.5.6.post1", "0.5.6.post2"], wrapped_forward_v1
        )

        def wrapped_forward_v2(self, *args, **kwargs):
            from sglang.srt.model_executor.model_runner import ModelRunnerOutput

            output, _ = wrapped_forward_v1(self, *args, **kwargs)
            return ModelRunnerOutput(
                logits_output=output,
                can_run_graph=False,
                expert_distribution_metrics=None,
            )

        _version_dispatcher.register_method(
            "forward", ["0.5.7", "0.5.8", "0.5.8.post1", "0.5.9"], wrapped_forward_v2
        )

        def wrapped_sample(self, *args, **kwargs):
            logits = args[0]
            ids = torch.ones(
                size=(logits.next_token_logits.shape[0],),
                device=self.device,
                dtype=torch.int64,
            )
            return ids

        def wrapped_compute_logprobs_only(*args, **kwargs):
            return None

        target.initialize = override_initialize
        target.forward = _version_dispatcher.get_compat_method("forward")
        target.sample = wrapped_sample
        target.compute_logprobs_only = wrapped_compute_logprobs_only


class C_SchedulerHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler"

    INFERENCE_PREDICTOR: InferTimePredictor = None

    REQUEST_STATS: dict[str, RequestStats] = defaultdict(RequestStats)
    ITERATION_STATS: list[dict] = []
    LAST_CPU_TS: float = 0
    LAST_FLUSH_TS: float = 0
    HISIM_BATCH: HisimScheduleBatch = None

    OVERLAP_SCHEDULE: bool = False

    SIM_MODE = MockSimulationMode(Envs.simulation_mode())
    OFFLINE_RECV_ALL_REQUEST: bool = False
    FUTURE_QUEUE: list[
        tuple[float, int, RequestStats]
    ] = []  # tuple(created time, salt, request)

    SCHEDULE_REQ_STATS = []

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_recv_requests = target.recv_requests
        original_get_new_batch_prefill = target.get_new_batch_prefill
        original_run_batch = target.run_batch
        original_process_batch_result = target.process_batch_result
        original_event_loop_normal = target.event_loop_normal

        def override_event_loop_overlap(self, *args, **kwargs):
            # To reduce the complexity of the simulation, the overlapping schedule is not needed.
            return original_event_loop_normal(self, *args, **kwargs)

        def wrapped_init(self, *args, **kwargs):
            # Disable overlap schedule
            server_args = get_obj_from_args(
                "sglang.srt.server_args.ServerArgs", *args, **kwargs
            )
            C_SchedulerHook.OVERLAP_SCHEDULE = not getattr(
                server_args, "disable_overlap_schedule", False
            )
            setattr(server_args, "disable_overlap_schedule", True)
            logger.debug(
                f"Overlap schedule simulation mode: {C_SchedulerHook.OVERLAP_SCHEDULE}."
            )

            original_init(self, *args, **kwargs)

            try:
                model = ConfigManager.get_model_info(
                    self.model_config.hf_config.__dict__
                )
                hw = ConfigManager.get_accelerator_info()
                sched_config = ConfigManager.get_scheduler_config(
                    self.server_args.__dict__,
                    "sglang",
                    self.model_config.hf_config.__dict__,
                )
                ConfigManager.set_scheduler_config(sched_config)
                ConfigManager.set_model_info(model)

                C_SchedulerHook.INFERENCE_PREDICTOR = (
                    ConfigManager.get_inference_time_predictor(model, hw, sched_config)
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize inference time predictor. Error: {e}"
                )
                raise e

        def wrapped_recv_requests(self, *args, **kwargs) -> list:
            recv_reqs = []

            if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                recv_reqs.extend(original_recv_requests(self, *args, **kwargs))
            elif C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE:
                # Initializing
                if not C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST:
                    gen_requests = []
                    extra_requests = []
                    time.sleep(0.05)  # waiting requests

                    reqs = original_recv_requests(self, *args, **kwargs)

                    for req in reqs:
                        if req.__class__.__name__ == "TokenizedGenerateReqInput":
                            gen_requests.append(req)
                        else:
                            # Such as: /profile_start, /flush_cache, etc.
                            extra_requests.append(req)

                    # Add requests to future queue
                    for req in gen_requests:
                        sim_params = None
                        if req.sampling_params.custom_params is not None:
                            sim_params = req.sampling_params.custom_params.get(
                                "simulation"
                            )
                        if sim_params is None:
                            # There are some warm-up requests when starting the server without --skip-server-warmup.
                            extra_requests.append(req)
                            logger.warning(
                                "Failed to extract the simulation parameters required for simulation from the request. Ignore this warning if the request is a warm-up request."
                            )
                            continue
                        if sim_params.get("queue_start"):
                            logger.debug(
                                "Add request to waiting queue with custom queue start timestamp."
                            )

                        C_SchedulerHook.FUTURE_QUEUE.append(
                            (
                                sim_params.get("queue_start")
                                or sim_params["created_time"],
                                time.time_ns(),  # The request is not comparable, so add the salt to avoid comparison.
                                req,
                            )
                        )

                    if len(C_SchedulerHook.FUTURE_QUEUE) != 0:
                        _, _, gen_req = C_SchedulerHook.FUTURE_QUEUE[-1]
                        total_request = gen_req.sampling_params.custom_params[
                            "simulation"
                        ]["total_request"]

                        if len(C_SchedulerHook.FUTURE_QUEUE) == total_request:
                            C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = True
                            heapq.heapify(C_SchedulerHook.FUTURE_QUEUE)
                            logger.info(
                                "All requests received. Starting simulation now."
                            )
                        else:
                            logger.info(
                                f"Offline simulation mode enabled. {total_request} requests expected in total. Received {len(C_SchedulerHook.FUTURE_QUEUE)} requests so far."
                            )

                    if len(extra_requests) != 0:
                        # Schedule the extra requests immediately.
                        return extra_requests
                else:
                    # Extra requests include: flush request, abort request, etc.
                    recv_reqs.extend(original_recv_requests(self, *args, **kwargs))

                # Process the arrived requests only after all requests have been added to the future queue
                current_timestamp = StateManager.get_global_clock()
                while (
                    C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST
                    and len(C_SchedulerHook.FUTURE_QUEUE) > 0
                ):
                    enqueue_time, _, req = C_SchedulerHook.FUTURE_QUEUE[0]
                    if enqueue_time > current_timestamp:
                        break
                    recv_reqs.append(req)
                    heapq.heappop(C_SchedulerHook.FUTURE_QUEUE)

            now = time.time()
            for req in recv_reqs:
                if req.__class__.__name__ in [
                    "BatchTokenizedGenerateReqInput",
                    "TokenizedGenerateReqInput",
                ]:
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    req_stats.rid = req.rid
                    req_stats.input_length = len(req.input_ids)
                    req_stats.output_length = req.sampling_params.max_new_tokens
                    simulation_args = req.sampling_params.custom_params["simulation"]
                    if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                        if "server_created_time" not in simulation_args:
                            logger.warning(
                                "The request's creation time is missing, which may cause the TTFT to be inaccurate."
                            )
                        req_stats.created_time = simulation_args.get(
                            "server_created_time", now
                        )
                        req_stats.last_event_time = req_stats.created_time
                        req_stats.queue_start = now
                    elif C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE:
                        req_stats.created_time = simulation_args["created_time"]
                        req_stats.last_event_time = req_stats.created_time
                        # Align with the real queue start timestamp if queue_start is not None. For debugging only.
                        queue_start = simulation_args.get("queue_start")
                        if queue_start is not None:
                            StateManager.set_global_clock(queue_start)
                        req_stats.queue_start = StateManager.get_global_clock()

            if recv_reqs and C_SchedulerHook.LAST_CPU_TS == 0:
                C_SchedulerHook.LAST_CPU_TS = time.time()
                StateManager.set_global_clock(0)

            return recv_reqs

        def wrapped_get_new_batch_prefill(self, *args, **kwargs):
            new_batch = original_get_new_batch_prefill(self, *args, **kwargs)
            now = time.time()
            if new_batch is not None:
                for req in new_batch.reqs:
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    req_stats.final_reused_tokens = req.cached_tokens
                    if req_stats.queue_end == -1:
                        if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                            req_stats.queue_end = now
                        else:
                            req_stats.queue_end = StateManager.get_global_clock()
                    else:
                        # Chunked request
                        pass
            elif len(self.running_batch.reqs) == 0 and len(self.waiting_queue) > 0:
                # Prefetching
                StateManager.step_global_clock(0.005)
                StateManager.set_current_inference_dur(0.005)
            else:
                if C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE and (
                    len(C_SchedulerHook.FUTURE_QUEUE) != 0
                    and len(self.running_batch.reqs) == 0
                ):
                    next_created_time, _, req = C_SchedulerHook.FUTURE_QUEUE[0]
                    StateManager.set_global_clock(next_created_time + 1e-6)
            logger.debug(
                f"Get new batch prefill: global iteration={StateManager.get_iteration()}, "
                f"new batch={new_batch.batch_size() if new_batch is not None else 0}, "
                f"waiting queue={len(self.waiting_queue)}"
            )

            return new_batch

        def wrapped_run_batch(self, *args, **kwargs):
            ret = original_run_batch(self, *args, **kwargs)

            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )

            if ret.__class__.__name__ == "GenerationBatchResult":
                hisim_batch = HisimScheduleBatch(reqs=[])
                if batch.forward_mode.is_extend():
                    for req in batch.reqs:
                        hisim_batch.reqs.append(
                            FakeRequest(
                                input_length=req.extend_input_len,
                                past_kv_length=len(req.prefix_indices)
                                + len(req.output_ids),
                            )
                        )
                elif batch.forward_mode.is_decode():
                    for req in batch.reqs:
                        hisim_batch.reqs.append(
                            FakeRequest(
                                input_length=1,
                                past_kv_length=len(req.prefix_indices)
                                + len(req.output_ids),
                            )
                        )

                if not hisim_batch.is_empty():
                    StateManager.inc_iteration()
                    predicted_latency = (
                        C_SchedulerHook.INFERENCE_PREDICTOR.predict_infer_time(
                            hisim_batch
                        )
                    )
                    predicted_latency = float(predicted_latency)

                    forward_latency = 0
                    if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                        now = time.time()
                        time.sleep(abs(predicted_latency))
                        now = time.time()
                        forward_latency = now - C_SchedulerHook.LAST_CPU_TS
                        C_SchedulerHook.LAST_CPU_TS = now
                    else:
                        now = time.time()
                        forward_latency = predicted_latency

                    StateManager.set_current_inference_dur(forward_latency)

                C_SchedulerHook.HISIM_BATCH = hisim_batch

            return ret

        def wrapped_process_batch_result(self, *args, **kwargs):
            ret = original_process_batch_result(self, *args, **kwargs)

            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )
            if batch is not None:
                if len(batch.reqs) == 0:
                    return ret

                hicache_l2_load_dur = StateManager.pop_hicache_l2_load_dur()
                hicache_l2_backup_dur = StateManager.pop_hicache_l2_backup_dur()
                current_inference_dur = StateManager.get_current_inference_dur()

                if C_SchedulerHook.OVERLAP_SCHEDULE:
                    StateManager.step_global_clock(
                        max(
                            hicache_l2_load_dur - StateManager.get_last_inference_dur(),
                            0,
                        )
                    )
                    StateManager.step_global_clock(current_inference_dur)
                    request_response_time = (
                        StateManager.get_global_clock() + hicache_l2_backup_dur
                    )
                else:
                    StateManager.step_global_clock(
                        hicache_l2_load_dur
                        + current_inference_dur
                        + hicache_l2_backup_dur
                    )
                    request_response_time = StateManager.get_global_clock()
                # Request statistics
                for req in batch.reqs:
                    if req.is_chunked == 0:
                        req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                        req_stats.gen_token_latencies.append(
                            request_response_time
                            - req_stats.last_event_time  # queue duration
                        )
                        req_stats.last_event_time = request_response_time
                    else:
                        # Chunked request: nothing to do
                        pass
                # Iteration statistics
                C_SchedulerHook.ITERATION_STATS.append(
                    {
                        "requests": C_SchedulerHook.HISIM_BATCH.request_info(),
                        "forward_latency": current_inference_dur,
                        "l2_load_latency": hicache_l2_load_dur,
                        "l2_backup_latency": hicache_l2_backup_dur,
                    }
                )
            C_SchedulerHook.LAST_CPU_TS = time.time()
            return ret

        def wrapped_profile(self, req, *args, **kwargs):
            stats: list[RequestStats] = []
            for item in C_SchedulerHook.REQUEST_STATS.values():
                if item.rid is not None and item.input_length > 0:
                    stats.append(item)

            stats = sorted(stats, key=lambda req: req.created_time)

            output_dir = Envs.output_dir()
            os.makedirs(output_dir, exist_ok=True)

            if len(stats) > 0:
                # Remove warmup requests.
                if len(stats) > Envs.num_warmup():
                    metrics_stats = stats[Envs.num_warmup() :]
                else:
                    metrics_stats = stats

                min_created_time = metrics_stats[0].created_time
                # Align timestamps
                for item in stats:
                    item.created_time -= min_created_time
                    item.queue_start -= min_created_time
                    item.queue_end -= min_created_time
                    item.last_event_time -= min_created_time

                metrics = calc_metrics(metrics_stats)
                metrics["time_cost"] = time.time() - C_SchedulerHook.LAST_FLUSH_TS

                try:
                    with open(f"{output_dir}/metrics.json", "w") as f:
                        f.write(json.dumps(metrics, cls=CustomJsonEncoder) + "\n")

                    with open(f"{output_dir}/iteration.jsonl", "w") as f:
                        for item in C_SchedulerHook.ITERATION_STATS:
                            f.write(json.dumps(item) + "\n")

                    with open(f"{output_dir}/request.jsonl", "w") as f:
                        for item in stats:
                            f.write(json.dumps(asdict(item)) + "\n")

                    logger.info(f"Simulation results saved to {output_dir}.")

                except Exception as e:
                    logger.error(f"Failed to dump results. Error: {e}")
            else:
                logger.warning("No request statistics available.")

            StateManager.reset()
            C_SchedulerHook.REQUEST_STATS.clear()
            C_SchedulerHook.ITERATION_STATS.clear()
            C_SchedulerHook.LAST_CPU_TS = 0
            C_SchedulerHook.LAST_FLUSH_TS = time.time()
            C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = False

            ProfileReqOutput = getattr(
                importlib.import_module("sglang.srt.managers.io_struct"),
                "ProfileReqOutput",
            )
            result = {
                "total_request": len(stats),
                "output_directory": output_dir,
            }

            return ProfileReqOutput(True, json.dumps(result))

        target.event_loop_overlap = override_event_loop_overlap
        target.__init__ = wrapped_init
        target.recv_requests = wrapped_recv_requests
        target.get_new_batch_prefill = wrapped_get_new_batch_prefill
        target.run_batch = wrapped_run_batch
        target.process_batch_result = wrapped_process_batch_result
        target.profile = wrapped_profile
