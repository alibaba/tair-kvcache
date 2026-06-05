package(default_visibility = ["//visibility:public"])

cc_library(
    name = "manager",
    srcs = [
        "optimizer_manager.cc",
        "optimizer_runner.cc",
        "hash_storage_pool_manager.cc",
        "hierarchical_replay_manager.cc",
        "infer_engine_scheduler.cc",
        "tier_global_tracker.cc",
        "eviction_manager.cc",
        "optimizer_loader.cc",
        "indexer_manager.cc",
    ],
    hdrs = [
        "optimizer_manager.h",
        "optimizer_runner.h",
        "hash_storage_pool_manager.h",
        "hierarchical_replay_manager.h",
        "infer_engine_scheduler.h",
        "tier_global_tracker.h",
        "eviction_manager.h",
        "optimizer_loader.h",
        "indexer_manager.h",
    ],
    deps = [
        "//kv_cache_manager/optimizer/eviction_policy",
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/optimizer/index",
        "//kv_cache_manager/optimizer/tier_flow",
        "//kv_cache_manager/optimizer/trace_loader",
        "//kv_cache_manager/meta:cache_location",
        "//kv_cache_manager/optimizer/analysis",
        "//kv_cache_manager/optimizer/analysis/tracker",
    ],
)

cc_library(
    name = "eviction_manager",
    srcs = [
        "eviction_manager.cc",
    ],
    hdrs = [
        "eviction_manager.h",
    ],
    deps = [
        "//kv_cache_manager/optimizer/eviction_policy",
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/optimizer/index",
        "//kv_cache_manager/optimizer/tier_flow",
    ],
)
