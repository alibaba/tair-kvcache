from hisim.hook.class_hook_entry import (
    install_class_hooks,
    remove_class_hooks,
)
from hisim.hook.module_hook_entry import (
    install_module_hooks,
    remove_module_hooks,
)
from hisim.hook.base_hook import BaseHook


__all__ = (
    install_class_hooks,
    remove_class_hooks,
    install_module_hooks,
    remove_module_hooks,
    BaseHook,
)
