import os
import torch
from importlib.metadata import distributions


MODEL_PATH = os.getenv("BENCHMARK_TEST_MODEL_PATH")
if MODEL_PATH is None:
    MODEL_PATH = "Qwen/Qwen3-0.6B"


def check_framework(name: str, device: str = "cuda") -> bool:
    if name == "sglang" and device == "cuda":
        if torch.cuda.get_device_capability() < (7, 5):
            # SGLang only supports sm75 and above.
            return False
    # hook need to be registered before importing module.
    installed_packages = set()
    for dist in distributions():
        if dist.metadata is not None and "Name" in dist.metadata:
            installed_packages.add(dist.metadata["Name"])
    return name in installed_packages
