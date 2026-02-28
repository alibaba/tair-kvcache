load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "with_cuda",
    values = {"define": "TAIR_MEMPOOL_USE_CUDA=true"},
)

cc_library(
    name = "cpu_lib",
    srcs = [
        "src/address_mgmt/bfc_allocator.cpp",
        "src/address_mgmt/shared_meta.cpp",
        "src/address_mgmt/util.cpp",
        "src/client/data_dumper.cpp",
        "src/client/ga_mapper.cpp",
        "src/client/mps_client.cpp",
        "src/client/request_encoder.cpp",
        "src/common/env_util.cpp",
        "src/cuda/cuda_util.cpp",
        "src/pace_mp_api.cpp",
    ],
    hdrs = glob([
        "include/pace_mp.h",
        "include/pace_mp_meta.h",
        "include/kernel/sm_copy_kernel.h",
    ]),
    copts = [
        "-std=c++20",
        "-fPIC",
        "-g",
        "-Wall",
        "-Wno-unused-variable",
        "-Wno-sign-compare",
        "-Wno-unused-result",
    ] + select({
        ":with_cuda": ["-DCUDA_EB"],
        "//conditions:default": [],
    }),
    includes = ["include"],
    linkstatic = True,
    deps = [
        "@boost//:date_time",
        "@boost//:headers-all",
        "@boost//:headers-base",
        "@boost//:interprocess",
        "@boost//:property_tree",
    ] + select({
        ":with_cuda": ["@local_config_cuda//cuda"],
        "//conditions:default": [],
    }),
    alwayslink = True,
)

cuda_library(
    name = "gpu_lib",
    srcs = ["src/kernel/sm_copy_kernel.cu"],
    copts = [
        "-std=c++20",
        "--compiler-options=-fPIC",
        "--expt-relaxed-constexpr",
        "-g",
    ] + select({
        ":with_cuda": ["-DCUDA_EB"],
        "//conditions:default": [],
    }),
    deps = [":cpu_lib"],
    alwayslink = True,
)

cc_library(
    name = "tair_mempool",
    linkopts = [
        "-L/usr/local/lib64/",
        "-libverbs",
        "-lrdmacm",
    ] + select({
        ":with_cuda": ["-lcuda"],
        "//conditions:default": [],
    }),
    deps = select({
        ":with_cuda": [
            ":cpu_lib",
            ":gpu_lib",
        ],
        "//conditions:default": [":cpu_lib"],
    }),
)
