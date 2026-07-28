load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

# Stubs that make v1_connector importable without vLLM / CUDA / the compiled
# kvcm_py_client. Tests import this module before anything under vllm/.
py_library(
    name = "vllm_stubs",
    srcs = [
        "__init__.py",
        "vllm_stubs.py",
    ],
    deps = [
        "//kv_cache_manager/py_connector/vllm:vllm_connector",
    ],
)

py_test(
    name = "test_block_translation",
    srcs = ["test_block_translation.py"],
    tags = ["no-remote-exec"],
    deps = [":vllm_stubs"],
)

py_test(
    name = "test_data_transfer_results",
    srcs = ["test_data_transfer_results.py"],
    tags = ["no-remote-exec"],
    deps = [":vllm_stubs"],
)

py_test(
    name = "test_scheduler_state",
    srcs = ["test_scheduler_state.py"],
    tags = ["no-remote-exec"],
    deps = [":vllm_stubs"],
)
