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

# The frozen MLA variant facts (measured on vLLM 0.26.0) shared by the tests
# below; pure data, no torch/vllm dependency.
py_library(
    name = "mla_variant_facts",
    srcs = ["mla_variant_facts.py"],
)

py_test(
    name = "test_mla_variants",
    srcs = ["test_mla_variants.py"],
    tags = ["no-remote-exec"],
    deps = [
        ":mla_variant_facts",
        ":vllm_stubs",
    ],
)

py_test(
    name = "test_role_views",
    srcs = ["test_role_views.py"],
    tags = ["no-remote-exec"],
    deps = [
        ":mla_variant_facts",
        ":vllm_stubs",
    ],
)

# Deliberately NOT dependent on vllm_stubs: this target must see the real
# vLLM (the stub mirror is anchored separately, in test_mla_variants) and
# skips itself when vLLM is unavailable.
py_test(
    name = "test_mla_spec_facts",
    srcs = ["test_mla_spec_facts.py"],
    tags = ["no-remote-exec"],
    deps = [":mla_variant_facts"],
)
