from kunlun_commons.model_info import ModelInfo
from kunlun_commons.hardwares.accelerator import AcceleratorInfo
from deepestim.llmperf.analyzers.theoretical_execution_analyzer import (
    TheoreticalExecutionAnalyzer,
)
from deepestim.llmperf.types import InferenceConfig
from schedule_simulator.schedule_emulator.types import (
    FakeRequest,
    SchedulerConfig,
)
import numpy as np


def calc_kv_cache_cell_elems(model_info: ModelInfo, tp_size: int, pp_size: int) -> int:
    # Ref: https://github.com/sgl-project/sglang/blob/v0.4.8/python/sglang/srt/model_executor/model_runner.py#L832
    num_layers = model_info.num_hidden_layers // pp_size
    if model_info.kv_lora_rank != 0:
        return (model_info.kv_lora_rank + model_info.qk_rope_head_dim) * num_layers
    else:
        num_kv_heads = max(model_info.num_key_value_heads // tp_size, 1)
        return num_kv_heads * model_info.head_dim * num_layers * 2


def calc_kv_cache_per_layer_elems(
    model_info: ModelInfo, tp_size: int, pp_size: int
) -> int:
    if model_info.kv_lora_rank != 0:
        return model_info.kv_lora_rank + model_info.qk_rope_head_dim
    else:
        num_kv_heads = max(model_info.num_key_value_heads // tp_size, 1)
        return num_kv_heads * model_info.head_dim * 2


def estimate_kv_cache_pool_capacity(
    model: ModelInfo, device: AcceleratorInfo, scheduler_config: SchedulerConfig
) -> int:
    # Ref: https://github.com/sgl-project/sglang/blob/v0.4.8/python/sglang/srt/model_executor/model_runner.py#L817
    llm_analyzer = TheoreticalExecutionAnalyzer(
        model=model,
        hw=device,
        infer_config=InferenceConfig(
            framework="sglang",
            max_num_seqs=model.max_seq_len,
            dt=scheduler_config.data_type,
            tp=scheduler_config.tp_size,
            dp=scheduler_config.dp_size,
            ep=scheduler_config.ep_size,
            pp=scheduler_config.pp_size,
            num_accelerator_per_machine=1024,  # It doesn't matter, because it only affects the communication calculation.
        ),
    )
    model_parameter_space = llm_analyzer.model_weight_space()
    # Keep some hbm capacity for framework
    framework_reserved_mem_gb = 1.4
    rest_memory = (
        scheduler_config.mem_fraction_static * device.hbm_capacity_gb
        - framework_reserved_mem_gb
    ) * (1 << 30) - model_parameter_space

    kv_cache_space_per_token = (
        calc_kv_cache_cell_elems(
            model, scheduler_config.tp_size, scheduler_config.pp_size
        )
        * scheduler_config.kv_cache_data_type.bytes
    )

    return int(rest_memory / kv_cache_space_per_token)


def calc_metrics(requests: list[FakeRequest]) -> dict:
    ttfts = []
    tpots = []
    itls = []
    e2e_latencies = []
    queue_waits = []
    min_arrival_s = float("inf")
    max_completion_s = 0.0
    total_input = 0
    total_output = 0
    completed = 0
    for req in requests:
        if not req.is_complete():
            continue
        completed += 1
        ttfts.append(req.gen_token_latencies[0])
        if len(req.gen_token_latencies) > 1:
            tpots.append(np.mean(req.gen_token_latencies[1:]))
        itls.extend(req.gen_token_latencies[1:])
        e2e = sum(req.gen_token_latencies)
        e2e_latencies.append(e2e)
        if req.queue_time_start >= 0 and req.queue_time_end >= 0:
            queue_waits.append(req.queue_time_end - req.queue_time_start)
        # last_event_time = completion time (absolute); arrival = completion - e2e
        arrival_s = req.last_event_time - e2e
        min_arrival_s = min(min_arrival_s, arrival_s)
        max_completion_s = max(max_completion_s, req.last_event_time)
        total_input += req.input_token_length
        total_output += req.output_token_length

    total_dur_s = max(max_completion_s - min_arrival_s, 1e-9)
    concurrency = sum(e2e_latencies) / total_dur_s if total_dur_s > 0 else 0

    return {
        "num_requests": len(requests),
        "completed": completed,
        "total_input": total_input,
        "total_output": total_output,
        "duration": total_dur_s,
        "request_throughput": completed / total_dur_s,
        "input_throughput": total_input / total_dur_s,
        "output_throughput": total_output / total_dur_s,
        "total_throughput": (total_input + total_output) / total_dur_s,

        "concurrency": concurrency,
        "mean_ttft_ms": np.mean(ttfts or 0) * 1000,
        "median_ttft_ms": np.median(ttfts or 0) * 1000,
        "std_ttft_ms": np.std(ttfts or 0) * 1000,
        "p90_ttft_ms": np.percentile(ttfts or 0, 90) * 1000,
        "p95_ttft_ms": np.percentile(ttfts or 0, 95) * 1000,
        "p99_ttft_ms": np.percentile(ttfts or 0, 99) * 1000,
        "mean_tpot_ms": np.mean(tpots or 0) * 1000,
        "median_tpot_ms": np.median(tpots or 0) * 1000,
        "std_tpot_ms": np.std(tpots or 0) * 1000,
        "p90_tpot_ms": np.percentile(tpots or 0, 90) * 1000,
        "p95_tpot_ms": np.percentile(tpots or 0, 95) * 1000,
        "p99_tpot_ms": np.percentile(tpots or 0, 99) * 1000,
        "mean_itl_ms": np.mean(itls or 0) * 1000,
        "median_itl_ms": np.median(itls or 0) * 1000,
        "std_itl_ms": np.std(itls or 0) * 1000,
        "p90_itl_ms": np.percentile(itls or 0, 90) * 1000,
        "p95_itl_ms": np.percentile(itls or 0, 95) * 1000,
        "p99_itl_ms": np.percentile(itls or 0, 99) * 1000,
        "max_itl_ms": float(np.max(itls or 0)) * 1000,
        "mean_e2e_latency_ms": np.mean(e2e_latencies) * 1000,
        "median_e2e_latency_ms": np.median(e2e_latencies) * 1000,
        "std_e2e_latency_ms": np.std(e2e_latencies) * 1000,
        "p90_e2e_latency_ms": np.percentile(e2e_latencies or 0, 90) * 1000,
        "p95_e2e_latency_ms": np.percentile(e2e_latencies or 0, 95) * 1000,
        "p99_e2e_latency_ms": np.percentile(e2e_latencies or 0, 99) * 1000,
        "mean_queue_wait_ms": np.mean(queue_waits or 0) * 1000,
        "median_queue_wait_ms": np.median(queue_waits or 0) * 1000,
        "p90_queue_wait_ms": np.percentile(queue_waits or 0, 90) * 1000,
        "p99_queue_wait_ms": np.percentile(queue_waits or 0, 99) * 1000,
        "time_cost": -1,
    }
