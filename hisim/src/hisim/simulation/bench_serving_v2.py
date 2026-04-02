import time
import numpy as np
from typing import List, AsyncGenerator, Optional
import re
import aiohttp
import os
import json
from dataclasses import fields
import argparse
from argparse import ArgumentParser

from sglang import bench_serving
from sglang.bench_serving import DatasetRow, LoRAPathAction, ASYNC_REQUEST_FUNCS
from packaging.version import Version
from sglang.version import __version__

if Version(__version__) < Version("0.5.9"):
    raise RuntimeError(
        f"Current benchmark script supports sglang 0.5.9+ only, current version: {__version__}"
    )

_ORIG_AIOHTTP_REQUEST = None


def install_aiohttp_json_hijack(
    *,
    hijack_url_regex: Optional[str],
) -> None:
    global _ORIG_AIOHTTP_REQUEST
    if _ORIG_AIOHTTP_REQUEST is not None:
        return

    pattern = re.compile(hijack_url_regex) if hijack_url_regex else None
    _ORIG_AIOHTTP_REQUEST = aiohttp.ClientSession._request

    async def _patched_request(self, method, url, **kwargs):
        if pattern.search(url):
            payload = kwargs.get("json", None)
            if isinstance(payload, dict) and "simulation" in payload:
                if "sampling_params" not in payload:
                    payload["sampling_params"] = {}
                if "custom_params" not in payload["sampling_params"]:
                    payload["sampling_params"]["custom_params"] = {}
                payload["sampling_params"]["custom_params"]["simulation"] = payload.pop(
                    "simulation"
                )
                kwargs["json"] = payload

        return await _ORIG_AIOHTTP_REQUEST(self, method, url, **kwargs)

    aiohttp.ClientSession._request = _patched_request


async def override_get_request(
    input_requests: List[DatasetRow],
    request_rate: float,
    use_trace_timestamps: bool = False,
    slowdown_factor: float = 1.0,
    timestamp_scale_s: float = 1000,  # 1000: ms -> s
) -> AsyncGenerator[DatasetRow, None]:

    if use_trace_timestamps:
        print(
            f"Using trace timestamps for request generation with slowdown factor {slowdown_factor}."
        )
        # Sort requests by timestamp for correct replay
        input_requests.sort(key=lambda r: r.timestamp)

        start_time = time.perf_counter()
        trace_start_time = input_requests[0].timestamp if input_requests else 0

        for request in input_requests:
            request.extra_request_body = {
                "simulation": {
                    "created_time": (request.timestamp - trace_start_time)
                    / timestamp_scale_s,
                    "total_request": len(input_requests),
                }
            }

            yield request
    else:
        input_requests_iter = iter(input_requests)
        start_time = 0
        for request in input_requests_iter:
            request.extra_request_body = {
                "simulation": {
                    "created_time": start_time,
                    "total_request": len(input_requests),
                }
            }
            yield request

            if request_rate == float("inf"):
                # If the request rate is infinity, then we don't need to wait.
                continue

            # Sample the request interval from the exponential distribution.
            interval = np.random.exponential(1.0 / request_rate)
            # The next request will be sent after the interval.
            # await asyncio.sleep(interval)
            start_time += interval


original_calculate_metrics = bench_serving.calculate_metrics


def wrapped_calculate_metrics(*args, **simulation_metrics):
    real_metrics, output_lens = original_calculate_metrics(*args, **simulation_metrics)

    out_dir = os.getenv("HISIM_OUTPUT_DIR", "/tmp/hisim/output/")
    metrics_path = os.path.join(out_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"Fail to get simmulation metrics from {out_dir}")
        return real_metrics, output_lens

    with open(metrics_path) as f:
        data = json.load(f)

    metrics_fields = {f.name for f in fields(bench_serving.BenchmarkMetrics)}
    # -1 means invalid value.
    simulation_metrics = {k: data.get(k, -1) for k in metrics_fields}

    return bench_serving.BenchmarkMetrics(**simulation_metrics), output_lens


original_run_benchmark = bench_serving.run_benchmark


def wrapped_run_benchmark(args_: argparse.Namespace):
    # The profile api has been hooked, which used as a trigger of simulation's start / stop.
    args.profile = True
    return original_run_benchmark(args_)


bench_serving.get_request = override_get_request
bench_serving.calculate_metrics = wrapped_calculate_metrics
bench_serving.run_benchmark = wrapped_run_benchmark

install_aiohttp_json_hijack(hijack_url_regex=r"generate$")


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        default="sglang",
        help="Must specify a backend, depending on the LLM Inference Engine.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Default host is 0.0.0.0."
    )
    parser.add_argument(
        "--port",
        type=int,
        help="If not set, the default port is configured according to its default value for different LLM Inference Engines.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=[
            "sharegpt",
            "custom",
            "openai",
            "random",
            "random-ids",
            "generated-shared-prefix",
            "mmmu",
            "image",
            "mooncake",
        ],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default="", help="Path to the dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name or path of the model. If not set, the default model will request /v1/models for conf.",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        help="The name of the model as served by the serving service. If not set, this defaults to the value of --model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer. If not set, using the model conf.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process. Default is 1000.",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length from the ShareGPT dataset.",
    )
    parser.add_argument(
        "--sharegpt-context-len",
        type=int,
        default=None,
        help="The context length of the model for the ShareGPT dataset. Requests longer than the context length will be dropped.",
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random and image dataset.",
    )
    parser.add_argument(
        "--random-output-len",
        default=1024,
        type=int,
        help="Number of output tokens per request, used only for random and image dataset.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random and image dataset.",
    )
    # image dataset args
    parser.add_argument(
        "--image-count",
        type=int,
        default=1,
        help="Number of images per request (only available with the image dataset)",
    )
    parser.add_argument(
        "--image-resolution",
        type=str,
        default="1080p",
        help=(
            "Resolution of images for image dataset. "
            "Supports presets 4k/1080p/720p/360p or custom 'heightxwidth' (e.g., 1080x1920)."
        ),
    )
    parser.add_argument(
        "--random-image-count",
        action="store_true",
        help="Enable Random Image Count",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="jpeg",
        help=("Format of images for image dataset. Supports jpeg and png."),
    )
    parser.add_argument(
        "--image-content",
        type=str,
        default="random",
        help=("Content for images for image dataset. Supports random and blank."),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
    )
    parser.add_argument(
        "--use-trace-timestamps",
        action="store_true",
        help="Use timestamps from the trace file for request scheduling. Only valid for 'mooncake' dataset.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument("--output-file", type=str, help="Output JSONL file name.")
    parser.add_argument(
        "--output-details", action="store_true", help="Output details of benchmarking."
    )
    parser.add_argument(
        "--print-requests",
        action="store_true",
        help="Print requests immediately during benchmarking. Useful to quickly realize issues.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--disable-stream",
        action="store_true",
        help="Disable streaming mode.",
    )
    parser.add_argument(
        "--return-logprob",
        action="store_true",
        help="Return logprob.",
    )
    parser.add_argument(
        "--return-routed-experts",
        action="store_true",
        help="Return routed experts.",
    )
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument(
        "--disable-ignore-eos",
        action="store_true",
        help="Disable ignoring EOS.",
    )
    parser.add_argument(
        "--extra-request-body",
        metavar='{"key1": "value1", "key2": "value2"}',
        type=str,
        help="Append given JSON object to the request payload. You can use this to specify"
        "additional generate params like sampling params.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply chat template",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "SGLANG_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--plot-throughput",
        action="store_true",
        help="Plot throughput and concurrent requests over time. Requires termplotlib and gnuplot.",
    )
    # TODO unify all these
    parser.add_argument(
        "--profile-activities",
        type=str,
        nargs="+",
        default=["CPU", "GPU"],
        choices=["CPU", "GPU", "CUDA_PROFILER"],
    )
    parser.add_argument("--profile-num-steps", type=int, default=None)
    parser.add_argument("--profile-by-stage", action="store_true", default=False)
    parser.add_argument("--profile-stages", nargs="+", default=None)
    parser.add_argument(
        "--profile-output-dir",
        type=str,
        default=None,
        help="Output directory for profile traces.",
    )
    parser.add_argument(
        "--profile-prefix",
        type=str,
        default=None,
        help="Prefix for profile trace filenames.",
    )
    parser.add_argument(
        "--lora-name",
        type=str,
        nargs="*",
        default=None,
        action=LoRAPathAction,
        help="The names of LoRA adapters. You can provide a list of names in the format {name} {name} {name}...",
    )
    parser.add_argument(
        "--lora-request-distribution",
        type=str,
        default="uniform",
        choices=[
            "uniform",
            "distinct",
            "skewed",
        ],
        help="What distribution to sample the LoRA adapters specified in --lora-name. Borrowed from the Punica paper. "
        "'distinct' distribution means selecting a new LoRA adapter for every request. "
        "'skewed' distribution follows the Zipf distribution, where the number of requests "
        "to model i specified in --lora-name is α times the number of requests for model i+1, "
        "where α > 1.",
    )
    parser.add_argument(
        "--lora-zipf-alpha",
        type=float,
        default=1.5,
        help="The parameter to use for the Zipf distribution when --lora-request-distribution='skewed'.",
    )
    parser.add_argument(
        "--prompt-suffix",
        type=str,
        default="",
        help="Suffix applied to the end of all user prompts, followed by assistant prompt suffix.",
    )
    parser.add_argument(
        "--pd-separated",
        action="store_true",
        help="Benchmark PD disaggregation server",
    )

    # Create a mutually exclusive group for profiling URLs
    # In PD separated mode, prefill and decode workers must be profiled separately
    profile_url_group = parser.add_mutually_exclusive_group()
    profile_url_group.add_argument(
        "--profile-prefill-url",
        type=str,
        nargs="*",
        default=None,
        help="URL(s) of the prefill worker(s) for profiling in PD separated mode. "
        "Can specify multiple URLs: --profile-prefill-url http://localhost:30000 http://localhost:30001. "
        "NOTE: Cannot be used together with --profile-decode-url. "
        "In PD separated mode, prefill and decode workers must be profiled separately.",
    )
    profile_url_group.add_argument(
        "--profile-decode-url",
        type=str,
        nargs="*",
        default=None,
        help="URL(s) of the decode worker(s) for profiling in PD separated mode. "
        "Can specify multiple URLs: --profile-decode-url http://localhost:30010 http://localhost:30011. "
        "NOTE: Cannot be used together with --profile-prefill-url. "
        "In PD separated mode, prefill and decode workers must be profiled separately.",
    )
    parser.add_argument(
        "--flush-cache",
        action="store_true",
        help="Flush the cache before running the benchmark",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="Number of warmup requests to run before the benchmark",
    )
    parser.add_argument(
        "--tokenize-prompt",
        action="store_true",
        help="Use integer ids instead of string for inputs. Useful to control prompt lengths accurately",
    )

    group = parser.add_argument_group("generated-shared-prefix dataset arguments")
    group.add_argument(
        "--gsp-num-groups",
        type=int,
        default=64,
        help="Number of system prompt groups for generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-prompts-per-group",
        type=int,
        default=16,
        help="Number of prompts per system prompt group for generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-system-prompt-len",
        type=int,
        default=2048,
        help="Target length in tokens for system prompts in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-question-len",
        type=int,
        default=128,
        help="Target length in tokens for questions in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-output-len",
        type=int,
        default=256,
        help="Target length in tokens for outputs in generated-shared-prefix dataset",
    )
    parser.add_argument(
        "--gsp-range-ratio",
        type=float,
        # WARN: The default 1.0 is for backward compatibility, and is different from the default 0.0 for random dataset
        default=1.0,
        help="Range of sampled ratio of input/output length, used only for gsp dataset.",
    )
    group.add_argument(
        "--gsp-fast-prepare",
        action="store_true",
        help="Speedup preparing by removing statistics computation, which will make some output statistics inaccurate but suitable for pressure tests.",
    )
    group.add_argument(
        "--gsp-send-routing-key",
        action="store_true",
        help="Send routing key in requests via X-SMG-Routing-Key header. Requests with the same prefix share the same routing key.",
    )
    group.add_argument(
        "--gsp-num-turns",
        type=int,
        default=1,
        help="Number of turns for multi-turn conversations. If > 1, each prompt becomes a list of questions sharing the same system prefix.",
    )
    group.add_argument(
        "--gsp-ordered",
        action="store_true",
        help="Keep requests in order without shuffling. By default, requests are shuffled randomly.",
    )
    mooncake_group = parser.add_argument_group("mooncake dataset arguments")
    mooncake_group.add_argument(
        "--mooncake-slowdown-factor",
        type=float,
        default=1.0,
        help="Slowdown factor for replaying the mooncake trace. "
        "A value of 2.0 means the replay is twice as slow. "
        "NOTE: --request-rate is IGNORED in mooncake mode.",
    )
    mooncake_group.add_argument(
        "--mooncake-num-rounds",
        type=int,
        default=1,
        help="Number of conversation rounds for each session in the mooncake dataset. "
        "A value > 1 will enable true multi-turn session benchmarking.",
    )
    mooncake_group.add_argument(
        "--mooncake-workload",
        type=str,
        default="conversation",
        choices=[
            "mooncake",
            "conversation",
            "synthetic",
            "toolagent",
        ],
        help="Underlying workload for the mooncake dataset.",
    )
    parser.add_argument(
        "--tag", type=str, default=None, help="The tag to be dumped to output."
    )
    parser.add_argument(
        "--header",
        type=str,
        nargs="+",
        default=None,
        help="Custom HTTP headers in Key=Value format. Example: --header MyHeader=MY_VALUE MyAnotherHeader=myanothervalue",
    )
    args = parser.parse_args()
    bench_serving.run_benchmark(args)
