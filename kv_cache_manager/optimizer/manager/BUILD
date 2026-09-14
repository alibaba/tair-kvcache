package(default_visibility = ["//visibility:public"])

cc_library(
    name = "manager",
    srcs = [
        "optimizer_manager.cc",
        "optimizer_runner.cc",
        "eviction_manager.cc",
        "optimizer_loader.cc",
        "indexer_manager.cc",
    ],
    hdrs = [
        "optimizer_manager.h",
        "optimizer_runner.h",
        "eviction_manager.h",
        "optimizer_loader.h",
        "indexer_manager.h",
    ],
    deps = [
        "//kv_cache_manager/optimizer/eviction_policy",
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/optimizer/index",
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
    ],
)

cc_library(
    name = "online_optimizer_manager",
    srcs = [
        "online_runtime/online_optimizer_manager.cc",
    ],
    hdrs = [
        "online_runtime/online_optimizer_manager.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/optimizer/config:online_optimizer_config",
        "//kv_cache_manager/optimizer/config:optimizer_registry_manager",
        "//kv_cache_manager/optimizer/index:cache_indexer",
        "//kv_cache_manager/optimizer/index:cache_indexer_factory",
        "//kv_cache_manager/optimizer/index:ttl_cache_indexer_wrapper",
        "//kv_cache_manager/optimizer/liteHit:hit_curve",
        "//kv_cache_manager/optimizer/liteHit:lite_hit",
        "//kv_cache_manager/optimizer/liteHit:request_preprocess",
        "//kv_cache_manager/optimizer/metrics:mrc_window",
        "//kv_cache_manager/optimizer:quota_mrc_snapshot",
    ],
)

cc_library(
    name = "lite_hit_offline_runner",
    srcs = [
        "lite_hit_offline_runner.cc",
    ],
    hdrs = [
        "lite_hit_offline_runner.h",
    ],
    deps = [
        ":online_optimizer_manager",
        "//kv_cache_manager/common",
        "//kv_cache_manager/optimizer/config:optimizer_lite_hit_config",
        "//kv_cache_manager/optimizer/config:optimizer_registry_manager",
        "//kv_cache_manager/optimizer/liteHit:facts_csv",
        "//kv_cache_manager/optimizer/liteHit:lite_hit",
        "//kv_cache_manager/optimizer/liteHit:request_preprocess",
        "//kv_cache_manager/optimizer/liteHit:trace_router",
        "//kv_cache_manager/optimizer/trace_loader",
    ],
)
