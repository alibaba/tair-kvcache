package(default_visibility = ["//visibility:public"])

cc_library(
    name = "index",
    srcs = [
        "radix_tree_index.cc",
    ],
    hdrs = [
        "radix_tree_index.h",
    ],
    deps = [
        "//kv_cache_manager/optimizer/eviction_policy",
        "//kv_cache_manager/manager:cache_location",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/optimizer/trace_converter",
        "//kv_cache_manager/optimizer/manager:eviction_manager",
    ],
)
