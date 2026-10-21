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
    ] + select({
        # 与 sdk 侧同一 config_setting：--config=client 同时给出 -DENABLE_TAIR_MEMPOOL 与本 define。
        "//kv_cache_manager/client/src/internal/sdk:enable_tair_mempool": [
            "//3rdparty/tair_mempool",
        ],
        "//conditions:default": [],
    }),
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
    stamp = 1,
    version = "{STABLE_KVCM_VERSION}+{STABLE_BUILD_TIMESTAMP}.{STABLE_GIT_COMMIT}",
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
