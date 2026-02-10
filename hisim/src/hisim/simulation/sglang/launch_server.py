import os
import json
import sys
import argparse
import torch
import hisim.hook as hisim_hook
from hisim.simulation.sglang import sgl_kernel_hook, sglang_hook
from hisim.simulation.sim_args import SimulationArgs
from hisim.utils import get_logger


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


logger = get_logger("hisim")


# Ref: https://github.com/sgl-project/sglang/blob/v0.5.6.post2/python/sglang/launch_server.py
if __name__ == "__main__":
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree

    parser = argparse.ArgumentParser()

    g = parser.add_argument_group("sglang")
    ServerArgs.add_cli_args(g)

    g = parser.add_argument_group("simulation")
    SimulationArgs.add_cli_args(g)

    raw_args = parser.parse_args(sys.argv[1:])
    server_args = ServerArgs.from_cli_args(raw_args)
    simulation_args = SimulationArgs.from_cli_args(raw_args)

    config_path = os.getenv("HISIM_CONFIG_PATH")
    if config_path and os.path.exists(config_path):
        logger.info(f"Using config from {config_path}")
    elif simulation_args.config_path:
        os.environ["HISIM_CONFIG_PATH"] = simulation_args.config_path
    else:
        config_path = "/tmp/hisim/config.json"
        logger.info(f"Export config to {config_path}")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(simulation_args.to_dict(), f)
        os.environ["HISIM_CONFIG_PATH"] = config_path

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
