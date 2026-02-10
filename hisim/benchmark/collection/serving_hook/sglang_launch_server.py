import hisim.hook as hisim_hook

from sglang_hook import (
    C_SglangSchedulerInfoHook,
    C_TokenizerManagerHook,
)

hisim_hook.install_class_hooks(
    [
        C_SglangSchedulerInfoHook,
        C_TokenizerManagerHook,
    ]
)


# Ref: https://github.com/sgl-project/sglang/blob/v0.4.8/python/sglang/launch_server.py
if __name__ == "__main__":
    import os
    import sys

    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(sys.argv[1:])

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
