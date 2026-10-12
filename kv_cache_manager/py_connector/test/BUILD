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
    name = "test_kv_layout_preference",
    srcs = ["test_kv_layout_preference.py"],
    tags = ["no-remote-exec"],
    deps = [":vllm_stubs"],
)

# Deliberately NOT dependent on vllm_stubs, like test_spec_gate_real: the point
# of this target is the real class tree of the installed era. Run it with
# --env vllm-0.26.0 and --env vllm-0.31.0 for the dual anchor; it skips itself
# when vLLM or the connector is unavailable.
py_test(
    name = "test_kv_layout_preference_real",
    srcs = ["test_kv_layout_preference_real.py"],
    tags = ["no-remote-exec"],
    deps = ["//kv_cache_manager/py_connector/vllm:vllm_connector"],
)

py_test(
    name = "test_scheduler_state",
    srcs = ["test_scheduler_state.py"],
    tags = ["no-remote-exec"],
    deps = [":vllm_stubs"],
)

# The SWA data plane: retention-driven eager capture, null dispositions, the
# staged write path and the windowed hit clamp (stub ring, no GPU).
py_test(
    name = "test_swa_core",
    srcs = ["test_swa_core.py"],
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

# The spec-receipt table (one row per vLLM spec class + the era map + the
# pre-PR gate transcription), shared by test_spec_gate (stand-in ring) and
# test_spec_gate_real (the installed vLLM). Imports vllm_common lazily through
# the engine adapter, so it needs no vLLM at import time.
py_library(
    name = "spec_gate_facts",
    srcs = ["spec_gate_facts.py"],
    deps = ["//kv_cache_manager/py_connector/vllm:vllm_connector"],
)

py_test(
    name = "test_spec_gate",
    srcs = ["test_spec_gate.py"],
    tags = ["no-remote-exec"],
    deps = [
        ":spec_gate_facts",
        ":vllm_stubs",
    ],
)

# Deliberately NOT dependent on vllm_stubs, like test_mla_spec_facts below: the
# point of this target is the real class tree of the installed era. Run it with
# --env vllm-0.26.0 and --env vllm-0.31.0 for the dual anchor; it skips itself
# when vLLM is unavailable.
py_test(
    name = "test_spec_gate_real",
    srcs = ["test_spec_gate_real.py"],
    tags = ["no-remote-exec"],
    deps = [":spec_gate_facts"],
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
