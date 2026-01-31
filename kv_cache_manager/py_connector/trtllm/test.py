import logging
logging.basicConfig(level=logging.DEBUG)

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import click
import torch

from tensorrt_llm import LLM, SamplingParams, logger
from tensorrt_llm._torch.pyexecutor.kv_cache_connector import (
    KvCacheConnectorScheduler, KvCacheConnectorWorker, SchedulerOutput)
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig, TorchLlmArgs

from kv_cache_manager.py_connector.trtllm.connector import (
    KVCM_CONFIG_PATH_KEY,
    KVCMKvCacheConnectorLeader,
    KVCMKvCacheConnectorWorker,
)
logger = logging.getLogger(__name__)


def init_kvcm_config():
    kvcm_config_path = os.environ.get(KVCM_CONFIG_PATH_KEY)
    assert kvcm_config_path is not None, f"env {KVCM_CONFIG_PATH_KEY} is needed"

    kvcm_config = {
        "manager_uri": "http://127.0.0.1:6382",
        "instance_group": "default",
        "instance_id": "0",

        "use_mla": False,
        "dtype": "torch.bfloat16",
    }

    with open(kvcm_config_path, "w") as f:
        f.write(json.dumps(kvcm_config))


@click.command()
@click.argument("model", type=str)
def main(model: str):
    sys.path.append(os.path.join(
        os.path.dirname(__file__),
        "..",
    ))

    this_module = __file__[__file__.rfind("/") + 1:__file__.rfind(".py")]

    init_kvcm_config()

    kv_connector_config = KvCacheConnectorConfig(
        connector_module=this_module,
        connector_scheduler_class="KVCMKvCacheConnectorLeader",
        connector_worker_class="KVCMKvCacheConnectorWorker",
    )

    tensor_parallel_size = 2

    llm = LLM(model=model,
              tensor_parallel_size=tensor_parallel_size,
              backend="pytorch",
              cuda_graph_config=None,
              kv_connector_config=kv_connector_config)

    test_text = (
        "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”"
        "Nvidia Corporation is an American technology company headquartered in Santa Clara, California."
        "Founded in 1993 by Jensen Huang, Chris Malachowsky, and Curtis Priem, it develops graphics processing units (GPUs), "
        "system on a chips (SoCs), and application programming interfaces (APIs) for data science, high-performance computing, "
        "and mobile and automotive applications. Tell me about the company.")

    sampling_params = SamplingParams(max_tokens=32)

    output = llm.generate([test_text], sampling_params)
    text0 = output[0].outputs[0].text

    logger.info(f"First output: {text0}")
    logger.info("Loading new LLM instance...")

    del llm

    llm = LLM(model=model,
              tensor_parallel_size=tensor_parallel_size,
              backend="pytorch",
              cuda_graph_config=None,
              kv_connector_config=kv_connector_config)

    output = llm.generate([test_text], sampling_params)
    text1 = output[0].outputs[0].text

    logger.info(f"Second output (using connector cache): {text1}")

    assert text0 == text1


if __name__ == "__main__":
    main()
