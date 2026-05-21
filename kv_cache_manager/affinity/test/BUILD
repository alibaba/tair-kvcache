package(default_visibility = ["//visibility:private"])

cc_test(
    name = "AffinityTest",
    srcs = [
        "affinity_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/affinity",
        "//kv_cache_manager/affinity:filter_cond",
        "//kv_cache_manager/affinity:node_metrics",
        "//kv_cache_manager/affinity:strategy",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/data_storage:common_define",
        "@rapidjson",
    ],
)
