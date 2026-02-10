import hisim.hook as insight_hook

from sglang_hook import (
    C_SglangSchedulerInfoHook,
    C_TokenizerManagerHook,
)


insight_hook.install_class_hooks(
    [
        C_SglangSchedulerInfoHook,
        C_TokenizerManagerHook,
    ]
)
