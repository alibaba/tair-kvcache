from dataclasses import asdict
import asyncio
import json
import os
from typing import Iterator
import numpy as np
import torch

from hisim.utils.logger import get_logger

import hisim.hook as hisim_hook
from hisim.simulation.sglang import sgl_kernel_hook
from hisim.simulation.sglang import sglang_hook
from hisim.simulation.base.runner import BaseBenchmarkRunner
from hisim.simulation.types import BenchmarkConfig
from hisim.dataset import (
    DatasetArgs,
    get_dataset,
    GenericRequest,
    BaseDataset,
)

# hook the sglang implementation
if not torch.cuda.is_available():
    # CPU Platform
    hisim_hook.install_module_hooks([sgl_kernel_hook.M_SGLangKernelLoadUtilHook])
hisim_hook.install_class_hooks(
    [
        sglang_hook.C_SchedulerHook,
        sglang_hook.C_ModelRunnerHook,
        sglang_hook.C_TokenizerManagerHook,
        sglang_hook.C_StorageBackendFactory,
        sglang_hook.C_HiCacheController,
        sglang_hook.C_HiRadixCacheHook,
    ]
)

HISIM_OUTPUT_DIR = "/tmp/hisim/simulation"
HISIM_METRICS_PATH = f"{HISIM_OUTPUT_DIR}/metrics.json"
os.environ["HISIM_OUTPUT_DIR"] = HISIM_OUTPUT_DIR

if os.getenv("HISIM_SIMULATION_MODE") is None:
    os.environ["HISIM_SIMULATION_MODE"] = "OFFLINE"

# The sglang must be imported after the hook installer
from sglang.srt.entrypoints.engine import Engine  # noqa
from sglang.srt.server_args import ServerArgs  # noqa
from transformers import AutoTokenizer  # noqa


logger = get_logger("hisim")


class SGLangBenchmarkRunner(BaseBenchmarkRunner):
    def __init__(self, server_args: ServerArgs):
        # disable some features which is not necessary for simulation.
        server_args.disable_cuda_graph = True
        # server_args.disable_overlap_schedule = True
        # server_args.__post_init__()  # recall `__post_init__` due to some args had been modified.
        self.server_args = server_args
        self.engine = Engine(**asdict(server_args))

    def flush_cache(self):
        self.engine.flush_cache()

    def reset_storage_cache(self):
        self.engine.clear_hicache_storage()

    def get_request(
        self,
        dataset: BaseDataset,
        ignore_timestamp: bool = False,
        request_rate: float = float("inf"),
    ) -> Iterator[tuple[GenericRequest, dict]]:
        yield_delay = 0
        for req in dataset:
            if ignore_timestamp:
                created_time = yield_delay
                yield_delay += np.random.exponential(1.0 / request_rate)
            else:
                created_time = req.custom_params.get("created_time", 0)

            simulation_params = {
                "total_request": len(dataset),  # include the warmup requests.
                "created_time": created_time,
            }
            yield (req, simulation_params)

    async def async_benchmark(
        self, benchmark_config: BenchmarkConfig, dataset_args: DatasetArgs
    ):
        dataset = get_dataset(
            dataset_args,
            tokenizer=AutoTokenizer.from_pretrained(self.server_args.model_path),
        )

        await self.engine.tokenizer_manager.start_profile(profile_prefix="reset")

        if os.path.exists(HISIM_METRICS_PATH):
            with open(HISIM_METRICS_PATH, "w") as f:
                # clear data
                pass

        tasks = []
        logger.info(f"Created {len(dataset)} request tasks.")
        for req, simulation_params in self.get_request(
            dataset,
            ignore_timestamp=benchmark_config.ignore_request_timestamp,
            request_rate=benchmark_config.request_rate,
        ):
            task = asyncio.create_task(
                self.engine.async_generate(
                    prompt=req.prompt,
                    input_ids=req.token_ids,
                    sampling_params={
                        "ignore_eos": True,
                        "max_new_tokens": req.output_length,
                        "custom_params": {
                            # (tmp) Transfer simulation arguments to the scheduler through the custom_params in sampling_params
                            "simulation": simulation_params
                        },
                    },
                )
            )
            tasks.append(task)

        _ = await asyncio.gather(*tasks)

        # dump result
        await self.engine.tokenizer_manager.start_profile()

        if os.path.exists(HISIM_METRICS_PATH):
            with open(HISIM_METRICS_PATH, "r") as f:
                metrics = json.loads(f.readline())
        else:
            logger.error(
                f"Failed to load metrics from serving backend. The metrics file should be loaded from {HISIM_METRICS_PATH}."
            )
            return None

        return metrics

    def benchmark(
        self,
        benchmark_config: BenchmarkConfig,
        dataset_args: DatasetArgs,
    ):
        return self.engine.loop.run_until_complete(
            self.async_benchmark(benchmark_config, dataset_args)
        )

    def get_iteration_stats(self) -> list[dict]:
        data = []
        file_path = f"{HISIM_OUTPUT_DIR}/iteration.jsonl"
        if os.path.exists(file_path):
            with open(file_path) as f:
                line = f.readline()
                while line:
                    data.append(json.loads(line))
                    line = f.readline()
        else:
            logger.error(f"The iteration statistics data({file_path}) does not exist.")
        return data

    def get_request_stats(self) -> list[dict]:
        data = []
        file_path = f"{HISIM_OUTPUT_DIR}/request.jsonl"
        if os.path.exists(file_path):
            with open(file_path) as f:
                line = f.readline()
                while line:
                    data.append(json.loads(line))
                    line = f.readline()
        else:
            logger.error(f"The request statistics data({file_path}) does not exist.")
        return data

    def shutdown(self):
        logger.info("Attempting to shut down the SGLang backend engine.")
        return self.engine.shutdown()
