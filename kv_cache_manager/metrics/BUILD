package(default_visibility = ["//visibility:public"])

cc_library(
    name = "metrics_registry",
    srcs = [
        "metrics_registry.cc",
    ],
    hdrs = [
        "metrics_registry.h",
    ],
)

cc_library(
    name = "metrics_collector",
    srcs = [
        "metrics_collector.cc",
    ],
    hdrs = [
        "metrics_collector.h",
    ],
    deps = [
        ":metrics_registry",
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common:timestamp_util",
    ],
)

cc_library(
    name = "metrics_reporter",
    srcs = [
        "kmon_param.cc",
        "kmonitor_metrics_reporter.cc",
        "local_metrics_reporter.cc",
        "logging_metrics_reporter.cc",
    ],
    hdrs = [
        "dummy_metrics_reporter.h",
        "kmon_param.h",
        "kmonitor_metrics_reporter.h",
        "local_metrics_reporter.h",
        "logging_metrics_reporter.h",
        "metrics_reporter.h",
    ],
    deps = [
        ":metrics_registry",
        ":metrics_collector",
        "//kv_cache_manager/common:utils",
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/manager:cache_manager",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ],
)

cc_library(
    name = "metrics_reporter_factory",
    srcs = [
        "metrics_reporter_factory.cc",
    ],
    hdrs = [
        "metrics_reporter_factory.h",
    ],
    deps = [
        ":metrics_reporter",
        "//kv_cache_manager/common:logger",
    ],
)
