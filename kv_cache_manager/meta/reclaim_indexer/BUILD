package(default_visibility = ["//visibility:public"])

cc_library(
    name = "reclaim_indexer",
    hdrs = [
        "reclaim_indexer.h",
    ],
    deps = [
        "//kv_cache_manager/meta:types",
    ],
)

cc_library(
    name = "node_lru_reclaim_indexer",
    srcs = [
        "node_lru_reclaim_indexer.cc",
    ],
    hdrs = [
        "node_lru_reclaim_indexer.h",
    ],
    deps = [
        ":reclaim_indexer",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common/cache",
    ],
)

cc_library(
    name = "reclaim_indexer_factory",
    srcs = [
        "reclaim_indexer_factory.cc",
    ],
    hdrs = [
        "reclaim_indexer_factory.h",
    ],
    deps = [
        ":node_lru_reclaim_indexer",
        ":reclaim_indexer",
        "//kv_cache_manager/common",
    ],
)
