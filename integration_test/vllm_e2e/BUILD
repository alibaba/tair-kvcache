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
    deps = ["@pip_cpu//requests"],
)

py_test(
    name = "test_basic",
    timeout = "eternal",
    srcs = ["test_basic.py"],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    imports = ["."],
    tags = [
        "exclusive",  # GPU tests must run serially to avoid CUDA OOM contention
        "gpu",  # requires 1+ GPU
        "manual",  # needs a GPU machine + vLLM venv + model; see README.md
        "no-remote-exec",
    ],
    deps = [":e2e_lib"],
)

py_test(
    name = "test_concurrent",
    timeout = "eternal",
    srcs = ["test_concurrent.py"],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    imports = ["."],
    tags = [
        "exclusive",  # GPU tests must run serially to avoid CUDA OOM contention
        "gpu",  # requires 1+ GPU
        "manual",  # needs a GPU machine + vLLM venv + model; see README.md
        "no-remote-exec",
    ],
    deps = [":e2e_lib"],
)

py_test(
    name = "test_tp",
    timeout = "eternal",
    srcs = ["test_tp.py"],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    imports = ["."],
    tags = [
        "exclusive",  # GPU tests must run serially to avoid CUDA OOM contention
        "gpu",  # requires 1+ GPU
        "manual",  # needs a GPU machine + vLLM venv + model; see README.md
        "no-remote-exec",
    ],
    deps = [":e2e_lib"],
)

py_test(
    name = "test_full_hit",
    timeout = "eternal",
    srcs = ["test_full_hit.py"],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    imports = ["."],
    tags = [
        "exclusive",  # GPU tests must run serially to avoid CUDA OOM contention
        "gpu",  # requires 1+ GPU
        "manual",  # needs a GPU machine + vLLM venv + model; see README.md
        "no-remote-exec",
    ],
    deps = [":e2e_lib"],
)

py_test(
    name = "test_partial_hit",
    timeout = "eternal",
    srcs = ["test_partial_hit.py"],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    imports = ["."],
    tags = [
        "exclusive",  # GPU tests must run serially to avoid CUDA OOM contention
        "gpu",  # requires 1+ GPU
        "manual",  # needs a GPU machine + vLLM venv + model; see README.md
        "no-remote-exec",
    ],
    deps = [":e2e_lib"],
)

py_test(
    name = "test_load_failure",
    timeout = "eternal",
    srcs = ["test_load_failure.py"],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    imports = ["."],
    tags = [
        "exclusive",  # GPU tests must run serially to avoid CUDA OOM contention
        "gpu",  # requires 1+ GPU
        "manual",  # needs a GPU machine + vLLM venv + model; see README.md
        "no-remote-exec",
    ],
    deps = [":e2e_lib"],
)

py_test(
    name = "test_multi_turn",
    timeout = "eternal",
    srcs = ["test_multi_turn.py"],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    imports = ["."],
    tags = [
        "exclusive",  # GPU tests must run serially to avoid CUDA OOM contention
        "gpu",  # requires 1+ GPU
        "manual",  # needs a GPU machine + vLLM venv + model; see README.md
        "no-remote-exec",
    ],
    deps = [":e2e_lib"],
)

py_test(
    name = "test_cross_request_prefix",
    timeout = "eternal",
    srcs = ["test_cross_request_prefix.py"],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    imports = ["."],
    tags = [
        "exclusive",  # GPU tests must run serially to avoid CUDA OOM contention
        "gpu",  # requires 1+ GPU
        "manual",  # needs a GPU machine + vLLM venv + model; see README.md
        "no-remote-exec",
    ],
    deps = [":e2e_lib"],
)

# Meta-test: injects an off-by-one into the connector's token translation and
# asserts the KV verification FAILS -- proof the harness is not vacuous.
py_test(
    name = "test_mutation",
    timeout = "eternal",
    srcs = ["test_mutation.py"],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    imports = ["."],
    tags = [
        "exclusive",  # GPU tests must run serially to avoid CUDA OOM contention
        "gpu",  # requires 1+ GPU
        "manual",  # needs a GPU machine + vLLM venv + model; see README.md
        "no-remote-exec",
    ],
    deps = [":e2e_lib"],
)

# Wildcards skip "manual" targets; run the whole suite explicitly with
#   bazelisk test //integration_test/vllm_e2e:e2e_tests
test_suite(
    name = "e2e_tests",
    tags = ["manual"],  # keep wildcard builds from expanding into the GPU tests
    tests = [
        ":test_basic",
        ":test_concurrent",
        ":test_cross_request_prefix",
        ":test_full_hit",
        ":test_load_failure",
        ":test_multi_turn",
        ":test_mutation",
        ":test_partial_hit",
        ":test_tp",
    ],
)
