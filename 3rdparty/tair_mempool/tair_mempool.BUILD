package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

config_setting(
    name = "with_cuda",
    values = {"define": "TAIR_MEMPOOL_USE_CUDA=true"},
)

cc_library(
    name = "cpu_lib",
    srcs = [
        "src/pace_mp_api.cpp",
        "src/sdev_mps_api.cpp",
        "src/address_mgmt/bfc_allocator.cpp",
        "src/address_mgmt/shared_meta.cpp",
        "src/address_mgmt/util.cpp",
    ],
    hdrs = glob([
        "include/pace_mp.h",
        "include/pace_mp_meta.h",
        "include/sdev_mps.h",
        "include/sdev_mps_meta.h",
        "include/kernel/sm_copy_kernel.h",
    ]),
    includes = ["include"],
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
    deps = [
        "@boost//:headers-base",
        "@boost//:headers-all",
        "@boost//:interprocess",
        "@boost//:date_time",
        "@boost//:property_tree",
    ]+ select({
        ":with_cuda": ["@local_config_cuda//cuda:cuda"],
        "//conditions:default": [],
    }),
    linkstatic = True,
    alwayslink = True,  
)

cuda_library(
    name = "gpu_lib",
    srcs = ["src/kernel/sm_copy_kernel.cu"],
    deps = [":cpu_lib"],  
    copts = [
        "-std=c++20",
        "--compiler-options=-fPIC",
        "--expt-relaxed-constexpr",
        "-g",
    ]+ select({
        ":with_cuda": ["-DCUDA_EB"],
        "//conditions:default": [],
    }),
    alwayslink = True,
)

cc_library(
    name = "tair_mempool",
    deps = select({
        ":with_cuda": [":gpu_lib", ":cpu_lib"],
        "//conditions:default": [":cpu_lib"], 
    }),
    linkopts = [
        "-L/usr/local/lib64/",
        "-libverbs", "-lrdmacm"
        ] + select({
            ":with_cuda": ["-lcuda"],
            "//conditions:default": [],
        }),
)