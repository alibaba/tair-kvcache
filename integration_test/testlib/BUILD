package(default_visibility = ["//integration_test:__subpackages__"])

py_library(
    name = "test_base",
    srcs = [
        "__init__.py",
        "module_base.py",
        "ranged_port_util.py",
        "test_base.py",
        "worker.py",
        "worker_manager.py",
    ],
    data = [
        "//integration_test:kv_cache_manager_install",
    ],
    imports = [".."],
    deps = [],
)
