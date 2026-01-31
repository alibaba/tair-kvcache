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
        "//kv_cache_manager/optimizer/trace_converter",
        "//kv_cache_manager/manager:cache_location",
        "//kv_cache_manager/optimizer/analysis",
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
    ],
)
