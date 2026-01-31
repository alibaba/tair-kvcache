load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("@python_abi//:abi.bzl", "python_abi")
load("@python_platform//:platform.bzl", "python_platform")


pybind_extension(
    name = "kvcm_py_client",
    srcs = ["py_client_binding.cc"],
    deps = [
        "//kv_cache_manager/client:client",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "kvcm_py_client_lib",
    data = [
        "//kv_cache_manager/client/pybind:kvcm_py_client.so",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
)

py_package(
    name = "kvcm_py_client_lib_pkg",
    visibility = ["//visibility:private"],
    deps = [
        ":kvcm_py_client_lib",
    ],
)

py_wheel(
    name = "kvcm_py_client_lib_wheel",
    abi = python_abi(),
    classifiers = [
        "Programming Language :: Python :: 3",
    ],
#    description_file = "DESCRIPTION.md",
    distribution = "kvcm_py_client",
#    extra_distinfo_files = {
#        "//:LICENSE": "LICENSE",
#    },
    python_requires = ">=3.9",
    python_tag = python_abi(),
    version = "0.0.1",
    deps = [
        ":kvcm_py_client_lib_pkg",
    ],
    platform = select({
        "@bazel_tools//src/conditions:windows_x64": "win_amd64",
        "@bazel_tools//src/conditions:windows_arm64": "win_arm64",
        "@bazel_tools//src/conditions:darwin_x86_64": "macosx_14_0_x86_64",
        "@bazel_tools//src/conditions:darwin_arm64": "macosx_14_0_arm64",
        "@bazel_tools//src/conditions:linux_x86_64": python_platform() + "_x86_64",
        "@bazel_tools//src/conditions:linux_aarch64": python_platform() + "_aarch64",
    }),
)