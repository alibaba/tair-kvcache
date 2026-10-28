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
    name = "test_kv_layouts",
    srcs = ["test_kv_layouts.py"],
    tags = ["no-remote-exec"],
    deps = [":vllm_stubs"],
)

py_test(
    name = "test_scheduler_state",
    srcs = ["test_scheduler_state.py"],
    tags = ["no-remote-exec"],
    deps = [":vllm_stubs"],
)

py_test(
    name = "test_location_query",
    srcs = ["test_location_query.py"],
    tags = ["no-remote-exec"],
    deps = [":vllm_stubs"],
)

py_test(
    name = "test_transfer_backend",
    srcs = ["test_transfer_backend.py"],
    tags = ["no-remote-exec"],
    deps = [":vllm_stubs"],
)

# The checks run in subprocesses that import the connector themselves, so the
# stubs are only needed for the runfiles those children import from.
py_test(
    name = "test_lazy_triton_import",
    srcs = ["test_lazy_triton_import.py"],
    tags = ["no-remote-exec"],
    deps = [":vllm_stubs"],
)
