package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "optimizer_main",
    srcs = [
        "main.cc",
    ],
    deps = [
        "//kv_cache_manager/optimizer/manager",
        "//kv_cache_manager/optimizer/index",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/optimizer/analysis",
        "//kv_cache_manager/common:logger"
    ],
)

cc_binary(
    name = "hierarchical_replay_main",
    srcs = [
        "hierarchical_replay_main.cc",
    ],
    deps = [
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/optimizer/manager",
    ],
)
cc_library(
    name = "config",
    hdrs = [
        "advanced_cache_reclaimer.h",
        "advanced_data_storage_selector.h",
    ],
    srcs = [

    ],
)
