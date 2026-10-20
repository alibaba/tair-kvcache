package(default_visibility = ["//visibility:public"])

cc_test(
    name = "NodeLruReclaimIndexerTest",
    srcs = [
        "node_lru_reclaim_indexer_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/meta/reclaim_indexer:node_lru_reclaim_indexer",
    ],
)
