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
cc_library(
    name = "config",
    hdrs = [
        "advanced_cache_reclaimer.h",
        "advanced_data_storage_selector.h",
    ],
    srcs = [

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
    ],
)

cc_library(
    name = "optimizer_metrics_collector",
    srcs = [
        "service/metrics/optimizer_metrics_collector.cc",
    ],
    hdrs = [
        "service/metrics/optimizer_metrics_collector.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/metrics:metrics_registry",
    ],
)

cc_library(
    name = "optimizer_metrics_reporter",
    srcs = [
        "service/metrics/optimizer_metrics_reporter.cc",
    ],
    hdrs = [
        "service/metrics/optimizer_metrics_reporter.h",
    ],
    deps = [
        ":online_optimizer_manager",
        ":optimizer_metrics_collector",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:common_util",
        "//kv_cache_manager/metrics:kmon_param",
        "//kv_cache_manager/metrics:metrics_registry",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ],
)

cc_library(
    name = "optimizer_call_guard",
    srcs = [
        "service/optimizer_call_guard.cc",
    ],
    hdrs = [
        "service/optimizer_call_guard.h",
    ],
    deps = [
        ":optimizer_metrics_collector",
        ":optimizer_metrics_reporter",
        "//kv_cache_manager/common",
        "@rapidjson",
    ],
)

cc_library(
    name = "optimizer_service_impl",
    srcs = [
        "service/optimizer_service_impl.cc",
    ],
    hdrs = [
        "service/optimizer_service_impl.h",
    ],
    deps = [
        ":online_optimizer_manager",
        ":optimizer_call_guard",
        ":optimizer_metrics_collector",
        ":optimizer_metrics_reporter",
        "//kv_cache_manager/common",
        "//kv_cache_manager/optimizer/config:online_optimizer_config",
        "//kv_cache_manager/optimizer/config:optimizer_registry_manager",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
    ],
)

cc_library(
    name = "optimizer_service_grpc",
    srcs = [
        "service/grpc/optimizer_service_grpc.cc",
    ],
    hdrs = [
        "service/grpc/optimizer_service_grpc.h",
    ],
    deps = [
        ":optimizer_metrics_collector",
        ":optimizer_service_impl",
        "//kv_cache_manager/common",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/protocol/protobuf:service_cc_grpc",
        "//kv_cache_manager/service/util:common",
        "@grpc//:grpc++",
    ],
)

cc_library(
    name = "optimizer_service_http",
    srcs = [
        "service/http/optimizer_service_http.cc",
    ],
    hdrs = [
        "service/http/optimizer_service_http.h",
    ],
    copts = [
        "-std=c++20",
        "-fcoroutines",
    ],
    deps = [
        ":optimizer_metrics_collector",
        ":optimizer_service_impl",
        "//kv_cache_manager/common",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/metrics:prometheus_exporter",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
        "//kv_cache_manager/service/http_service",
    ],
)

cc_library(
    name = "online_optimizer_server_config",
    srcs = [
        "service/online_optimizer_server_config.cc",
    ],
    hdrs = [
        "service/online_optimizer_server_config.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:env_util",
        "//kv_cache_manager/common:string_util",
    ],
)

cc_library(
    name = "online_optimizer_server",
    srcs = [
        "service/online_optimizer_server.cc",
    ],
    hdrs = [
        "service/online_optimizer_server.h",
    ],
    copts = [
        "-std=c++20",
        "-fcoroutines",
    ],
    deps = [
        ":online_optimizer_manager",
        ":online_optimizer_server_config",
        ":optimizer_metrics_reporter",
        ":optimizer_service_grpc",
        ":optimizer_service_http",
        ":optimizer_service_impl",
        "//kv_cache_manager/common",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/optimizer/config:optimizer_registry_manager",
        "@grpc//:grpc++",
    ],
)

cc_binary(
    name = "online_optimizer_server_main",
    srcs = [
        "service/online_optimizer_server_main.cc",
    ],
    deps = [
        ":online_optimizer_server",
        "//kv_cache_manager/common",
    ],
)
