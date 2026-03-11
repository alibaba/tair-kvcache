package(default_visibility = ["//integration_test:__subpackages__"])

py_library(
    name = "mock_cuda",
    srcs = ["mock_cuda.py"],
)

py_library(
    name = "vllm_connector_cases",
    srcs = ["vllm_connector_cases.py"],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    imports = ["../testlib"],
    deps = [
        ":mock_cuda",
        "//integration_test/testlib:test_base",
        "//kv_cache_manager/py_connector/vllm:vllm_connector",
    ],
)

py_test(
    name = "connector_lifecycle_test",
    srcs = ["connector_lifecycle_test.py"],
    tags = ["no-remote-exec"],
    deps = [
        ":vllm_connector_cases",
    ],
)
