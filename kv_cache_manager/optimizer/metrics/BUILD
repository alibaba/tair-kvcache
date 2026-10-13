package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mrc_window",
    srcs = ["mrc_window.cc"],
    hdrs = ["mrc_window.h"],
    deps = [
        "//kv_cache_manager/optimizer/liteHit:hit_curve",
    ],
)

cc_library(
    name = "optimizer_metrics_collector",
    srcs = ["optimizer_metrics_collector.cc"],
    hdrs = ["optimizer_metrics_collector.h"],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/metrics:metrics_registry",
    ],
)

cc_library(
    name = "optimizer_kmonitor_metrics_reporter",
    srcs = ["optimizer_kmonitor_metrics_reporter.cc"],
    hdrs = ["optimizer_kmonitor_metrics_reporter.h"],
    deps = [
        ":optimizer_metrics_collector",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:env_util",
        "//kv_cache_manager/metrics:kmon_param",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/optimizer/manager:online_optimizer_manager",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ],
)

cc_library(
    name = "optimizer_metrics_reporter",
    srcs = ["optimizer_metrics_reporter.cc"],
    hdrs = ["optimizer_metrics_reporter.h"],
    deps = [
        ":optimizer_metrics_collector",
        ":optimizer_kmonitor_metrics_reporter",
        "//kv_cache_manager/common",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/optimizer/config:optimizer_registry_manager",
        "//kv_cache_manager/optimizer/manager:online_optimizer_manager",
    ],
)
