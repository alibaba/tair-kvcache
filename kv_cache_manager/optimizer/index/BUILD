package(default_visibility = ["//visibility:public"])

cc_library(
    name = "index",
    srcs = [
        "hash_table_index.cc",
        "radix_tree_index.cc",
    ],
    hdrs = [
        "hash_table_index.h",
        "radix_tree_index.h",
    ],
    deps = [
        "//kv_cache_manager/optimizer/eviction_policy",
        "//kv_cache_manager/meta:cache_location",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/optimizer/tier_flow",
        "//kv_cache_manager/optimizer/trace_loader",
        "//kv_cache_manager/optimizer/analysis",
    ],
)
