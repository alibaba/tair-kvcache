load("@python_abi//:abi.bzl", "python_abi")
load("@python_platform//:platform.bzl", "python_platform")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

py_library(
    name = "vllm_connector",
    srcs = glob(["*.py"]),
    visibility = ["//visibility:public"],
    deps = [
        "//kv_cache_manager/client/pybind:kvcm_py_client_lib",
        "//kv_cache_manager/py_connector/common",
        "//kv_cache_manager/py_connector/kernel",
    ],
)

py_package(
    name = "vllm_connector_pkg",
    packages = [
        "kv_cache_manager",
        "_solib_k8/_U_S_S3rdparty_S3fs_Chf3fs___U3rdparty_S3fs",
        # Do not include cuda library & libnuma, use the system-installed dynamic libraries.
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":vllm_connector",
    ],
)

py_wheel(
    name = "kvcm_vllm_connector_wheel",
    abi = python_abi(),
    classifiers = [
        "Programming Language :: Python :: 3",
    ],
    #    description_file = "DESCRIPTION.md",
    distribution = "kvcm_vllm_connector",
    platform = select({
        "@bazel_tools//src/conditions:windows_x64": "win_amd64",
        "@bazel_tools//src/conditions:windows_arm64": "win_arm64",
        "@bazel_tools//src/conditions:darwin_x86_64": "macosx_14_0_x86_64",
        "@bazel_tools//src/conditions:darwin_arm64": "macosx_14_0_arm64",
        "@bazel_tools//src/conditions:linux_x86_64": python_platform() + "_x86_64",
        "@bazel_tools//src/conditions:linux_aarch64": python_platform() + "_aarch64",
    }),
    #    extra_distinfo_files = {
    #        "//:LICENSE": "LICENSE",
    #    },
    python_requires = ">=3.9",
    python_tag = python_abi(),
    version = "0.0.1",
    deps = [
        ":vllm_connector_pkg",
    ],
)
