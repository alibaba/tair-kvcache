import os
import time
import json
from typing import Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import importlib

import hisim.hook as hisim_hook
from hisim.hook.utils import get_obj_from_args


HISIM_BENCHMARK_OUT_DIR = os.getenv("HISIM_BENCHMARK_OUT_DIR", os.getcwd())


@dataclass
class RequestInfos:
    rid: str = ""
    timestamp: Optional[float] = None  # created time
    output_length: int = 0
    input_length: int = 0
    input_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)
    queue_end: float = 0
    final_prefix_cache_len: int = 0


REQUEST_INFOS: dict[str, RequestInfos] = defaultdict(RequestInfos)
SCHEDULE_INFOS: list[dict] = []


class C_TokenizerManagerHook(hisim_hook.BaseHook):
    HOOK_CLASS_NAME = "TokenizerManager"
    HOOK_MODULE_NAME = "sglang.srt.managers.tokenizer_manager"

    @classmethod
    def hook(cls, target):
        original_send_one_request = target._send_one_request

        def wrapped_send_one_request(self, obj, tokenized_obj, created_time):
            setattr(tokenized_obj, "created_time", created_time)
            return original_send_one_request(self, obj, tokenized_obj, created_time)

        target._send_one_request = wrapped_send_one_request


class C_SglangSchedulerInfoHook(hisim_hook.BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler"

    LAST_PROCESS_RESULT_END: float = 0
    LAST_GET_NEW_BATCH_DUR: float = 0
    CUR_GET_NEW_BATCH_DUR: float = 0

    @classmethod
    def hook(cls, target_class):
        original_recv_requests = target_class.recv_requests
        original_get_new_batch_prefill = target_class.get_new_batch_prefill
        original_process_batch_result = target_class.process_batch_result

        def wrapped_recv_requests(self, *args, **kwargs):
            reqs = original_recv_requests(self, *args, **kwargs)

            recv_time = time.time()
            for req in reqs:
                if not hasattr(req, "rid"):
                    # control request
                    continue
                if req.rid:
                    req_info = REQUEST_INFOS[req.rid]
                    req_info.rid = req.rid
                    req_info.queue_start = recv_time

                    if not hasattr(req, "created_time"):
                        print("The request's created time is missing.")
                    else:
                        req_info.timestamp = getattr(req, "created_time")
            return reqs

        def wrapped_get_new_batch_prefill(self, *args, **kwargs):
            batch = original_get_new_batch_prefill(self, *args, **kwargs)

            if batch is not None and not batch.is_empty():
                prefill_timestamp = time.time()

                for req in batch.reqs:
                    req_info = REQUEST_INFOS[req.rid]
                    if req_info.queue_end == 0:
                        # Ignore the chunked request.
                        req_info.queue_end = prefill_timestamp
                        req_info.final_prefix_cache_len = len(req.prefix_indices)

            return batch

        def wrapped_process_batch_result(self, *args, **kwargs):
            result = original_process_batch_result(self, *args, **kwargs)

            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )

            if batch.reqs is None:
                # dummy first batch while overlap schedule is enabled.
                return result

            for req in batch.reqs:
                req_info = REQUEST_INFOS[req.rid]
                if req.finished():
                    req_info.input_length = len(req.origin_input_ids)
                    req_info.output_length = req.sampling_params.max_new_tokens
                    req_info.input_ids = req.origin_input_ids
                    req_info.output_ids = req.output_ids

            return result

        def wrapped_profile(self, *args, **kwargs):
            if REQUEST_INFOS:
                filename_prefix = f"TP{self.tp_rank}"

                if getattr(self, "dp_size", 1) > 1:
                    filename_prefix += f"-DP{getattr(self, 'dp_rank', 0)}"
                if getattr(self, "pp_size", 1) > 1:
                    filename_prefix += f"-PP{getattr(self, 'pp_rank', 0)}"
                if getattr(self, "moe_ep_size", 1) > 1:
                    filename_prefix += f"-EP{getattr(self, 'moe_ep_rank', 0)}"

                os.makedirs(HISIM_BENCHMARK_OUT_DIR, exist_ok=True)
                out_file = f"{HISIM_BENCHMARK_OUT_DIR}/{filename_prefix}.requests.jsonl"
                with open(out_file, "w") as f:
                    for req_infos in REQUEST_INFOS.values():
                        f.write(json.dumps(asdict(req_infos)) + "\n")
                print(f"[Hisim Collection] Request data has been saved to {out_file}")

            REQUEST_INFOS.clear()
            # There is no need to call the real profiling API.

            ProfileReqOutput = getattr(
                importlib.import_module("sglang.srt.managers.io_struct"),
                "ProfileReqOutput",
            )
            return ProfileReqOutput(True, "Success")

        target_class.recv_requests = wrapped_recv_requests
        target_class.get_new_batch_prefill = wrapped_get_new_batch_prefill
        target_class.process_batch_result = wrapped_process_batch_result
        target_class.profile = wrapped_profile
        return target_class
