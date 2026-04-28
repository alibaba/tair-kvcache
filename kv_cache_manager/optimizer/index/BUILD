package(default_visibility = ["//visibility:public"])

cc_library(
    name = "index",
    srcs = [
        "global_registry.cc",
        "radix_tree_index.cc",
    ],
    hdrs = [
        "global_registry.h",
        "radix_tree_index.h",
    ],
    deps = [
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/manager:cache_location",
        "//kv_cache_manager/optimizer/analysis",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/optimizer/eviction_policy",
        "//kv_cache_manager/optimizer/manager:eviction_manager",
        "//kv_cache_manager/optimizer/trace_loader",
    ],
)
