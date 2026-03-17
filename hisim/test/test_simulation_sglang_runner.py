import os
import pytest
from pathlib import Path
from env import check_framework, MODEL_PATH
from hisim.dataset import DatasetArgs
from hisim.simulation.types import BenchmarkConfig

os.environ["HISIM_CONFIG_PATH"] = os.path.dirname(__file__) + "/assets/mock/config.json"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"


from hisim.simulation.sglang.sglang_bench import (
    SGLangBenchmarkRunner,
)


@pytest.mark.skipif(
    not check_framework("sglang", device="cpu"), reason="sglang is not installed."
)
@pytest.mark.skipif(
    not check_framework("vllm"),
    reason="The cpu simulation might require vLLM's kernels.",
)
def test_benchmark_sglang():
    from sglang.srt.server_args import ServerArgs  # noqa

    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=MODEL_PATH,
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=True,
        )
    )

    # Test with insight hook dataset
    dataset_args = DatasetArgs(
        name="hisim_collection",
        filepath=str(
            Path(__file__).parent / "assets/dataset/hisim_collection_requests.jsonl"
        ),
    )
    benchmark_config = BenchmarkConfig()

    metrics = runner.benchmark(benchmark_config, dataset_args=dataset_args)
    assert metrics["completed"] == 4
    request_stats = runner.get_request_stats()
    assert request_stats[-1]["created_time"] != 0

    benchmark_config.ignore_request_timestamp = True
    metrics = runner.benchmark(benchmark_config, dataset_args=dataset_args)
    # This dataset has been used previously.
    assert metrics["prefix_cache_reused_ratio"] > 0.7
    request_stats = runner.get_request_stats()
    for req in request_stats:
        # runner with request rate = `float("inf")`
        assert req["created_time"] == 0

    # Test with benchmark config
    benchmark_config = BenchmarkConfig(request_rate=10, ignore_request_timestamp=True)
    dataset_args = DatasetArgs(
        "random_ids",
        num_prompts=10,
        min_input_len=100,
        max_input_len=101,
        min_output_len=1,
        max_output_len=2,
    )
    metrics = runner.benchmark(benchmark_config, dataset_args=dataset_args)
    assert metrics["completed"] == dataset_args.num_prompts
    request_stats = runner.get_request_stats()
    for idx, req in enumerate(request_stats):
        assert idx == 0 or req["created_time"] != 0, (
            "The created time should not be zero due to request_rate equal to 10"
        )
        assert (
            dataset_args.min_input_len
            <= req["input_length"]
            <= dataset_args.max_input_len
        )

    runner.shutdown()


if __name__ == "__main__":
    test_benchmark_sglang()
