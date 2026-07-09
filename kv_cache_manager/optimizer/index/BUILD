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
        "//kv_cache_manager/meta:cache_location",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/optimizer/trace_loader",
        "//kv_cache_manager/optimizer/analysis",
    ],
)

cc_library(
    name = "cache_indexer",
    srcs = [
        "online/lru_cache_indexer.cc",
    ],
    hdrs = [
        "online/cache_indexer.h",
        "online/lru_cache_indexer.h",
    ],
    deps = [
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common/cache",
    ],
)

cc_library(
    name = "ttl_cache_indexer_wrapper",
    srcs = [
        "online/ttl_cache_indexer_wrapper.cc",
    ],
    hdrs = [
        "online/ttl_cache_indexer_wrapper.h",
    ],
    deps = [
        ":cache_indexer",
        "//kv_cache_manager/common:timestamp_util",
    ],
)

cc_library(
    name = "cache_indexer_factory",
    srcs = [
        "online/cache_indexer_factory.cc",
    ],
    hdrs = [
        "online/cache_indexer_factory.h",
    ],
    deps = [
        ":cache_indexer",
        ":ttl_cache_indexer_wrapper",
        "//kv_cache_manager/common:env_util",
        "//kv_cache_manager/common:logger",
    ],
)
