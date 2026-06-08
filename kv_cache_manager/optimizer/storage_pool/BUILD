package(default_visibility = ["//visibility:public"])

cc_library(
    name = "storage_pool",
    srcs = [
        "hash_storage_pool_manager.cc",
    ],
    hdrs = [
        "hash_storage_pool_manager.h",
    ],
    deps = [
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/meta:cache_location",
        "//kv_cache_manager/optimizer/analysis",
        "//kv_cache_manager/optimizer/analysis/tracker",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/optimizer/eviction_policy",
        "//kv_cache_manager/optimizer/index",
    ],
)
