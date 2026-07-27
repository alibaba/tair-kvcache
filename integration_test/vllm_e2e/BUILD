package(default_visibility = ["//integration_test:__subpackages__"])

# Shared library: orchestration (manager + vLLM + driver + comparison) and the
# verifying connector injected into vLLM via kv_connector_module_path.
py_library(
    name = "e2e_lib",
    srcs = [
        "e2e_lib.py",
        "test_connector.py",
    ],
    imports = ["."],
    tags = ["no-remote-exec"],
)

py_test(
    name = "test_basic",
    srcs = ["test_basic.py"],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    imports = ["."],
    tags = [
        "no-remote-exec",
        "gpu",  # requires 1+ GPU
        "exclusive",  # GPU tests must run serially to avoid CUDA OOM contention
    ],
    timeout = "eternal",
    deps = [":e2e_lib"],
)

py_test(
    name = "test_concurrent",
    srcs = ["test_concurrent.py"],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    imports = ["."],
    tags = [
        "no-remote-exec",
        "gpu",  # requires 1+ GPU
        "exclusive",  # GPU tests must run serially to avoid CUDA OOM contention
    ],
    timeout = "eternal",
    deps = [":e2e_lib"],
)

py_test(
    name = "test_tp",
    srcs = ["test_tp.py"],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    imports = ["."],
    tags = [
        "no-remote-exec",
        "gpu",  # requires 1+ GPU
        "exclusive",  # GPU tests must run serially to avoid CUDA OOM contention
    ],
    timeout = "eternal",
    deps = [":e2e_lib"],
)
