package(default_visibility = ["//visibility:public"])

cc_library(
    name = "workload",
    hdrs = ["workload.h"],
    deps = ["//kv_cache_manager/client:client_api"],
)

cc_binary(
    name = "kvcm_stress",
    srcs = ["main.cc"],
    deps = [
        ":workload",
        "//kv_cache_manager/client:client",
        "//kv_cache_manager/client/src/internal/config",
        "//kv_cache_manager/common:logger",
    ],
)

cc_test(
    name = "WorkloadTest",
    srcs = ["workload_test.cc"],
    deps = [
        ":workload",
        "//kv_cache_manager/common:unittest",
    ],
)
