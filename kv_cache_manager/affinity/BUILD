package(default_visibility = ["//visibility:public"])

cc_library(
    name = "node_metrics",
    hdrs = ["node_metrics.h"],
)

cc_library(
    name = "metric_registry",
    srcs = ["metric_registry.cc"],
    hdrs = ["metric_registry.h"],
    deps = [
        ":node_metrics",
    ],
)

cc_library(
    name = "filter_cond",
    srcs = ["filter_cond.cc"],
    hdrs = ["filter_cond.h"],
    deps = [
        ":metric_registry",
        ":node_metrics",
        "@rapidjson",
    ],
)

cc_library(
    name = "strategy",
    srcs = ["strategy.cc"],
    hdrs = ["strategy.h"],
    deps = [
        ":filter_cond",
        ":metric_registry",
        ":node_metrics",
        "@rapidjson",
    ],
)

cc_library(
    name = "affinity",
    srcs = ["cache_affinity_manager.cc"],
    hdrs = ["cache_affinity_manager.h"],
    deps = [
        ":node_metrics",
        ":strategy",
        "//kv_cache_manager/common",
        "//kv_cache_manager/data_storage:common_define",
    ],
)
