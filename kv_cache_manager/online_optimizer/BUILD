package(default_visibility = ["//visibility:public"])

cc_library(
    name = "optimizer_instance_info",
    srcs = [
        "config/optimizer_instance_info.cc",
    ],
    hdrs = [
        "config/optimizer_instance_info.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/config:instance_info",
    ],
)

cc_library(
    name = "optimizer_instance_group",
    srcs = [
        "config/optimizer_instance_group.cc",
    ],
    hdrs = [
        "config/optimizer_instance_group.h",
    ],
    deps = [
        "//kv_cache_manager/common",
    ],
)

cc_library(
    name = "optimizer_registry_manager",
    srcs = [
        "config/optimizer_registry_manager.cc",
    ],
    hdrs = [
        "config/optimizer_registry_manager.h",
    ],
    deps = [
        ":optimizer_instance_group",
        ":optimizer_instance_info",
        "//kv_cache_manager/common",
        "//kv_cache_manager/config:registry_backend",
    ],
)

cc_library(
    name = "cache_indexer",
    srcs = [
        "indexer/bst_cache_indexer.cc",
        "indexer/fenwick_cache_indexer.cc",
        "indexer/lru_cache_indexer.cc",
    ],
    hdrs = [
        "indexer/bst_cache_indexer.h",
        "indexer/cache_indexer.h",
        "indexer/fenwick_cache_indexer.h",
        "indexer/lru_cache_indexer.h",
    ],
    deps = [
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common/cache",
    ],
)

cc_library(
    name = "cache_indexer_factory",
    srcs = [
        "indexer/cache_indexer_factory.cc",
    ],
    hdrs = [
        "indexer/cache_indexer_factory.h",
    ],
    deps = [
        ":cache_indexer",
        ":ttl_cache_indexer_wrapper",
        "//kv_cache_manager/common:env_util",
        "//kv_cache_manager/common:logger",
    ],
)

cc_library(
    name = "ttl_cache_indexer_wrapper",
    srcs = [
        "indexer/ttl_cache_indexer_wrapper.cc",
    ],
    hdrs = [
        "indexer/ttl_cache_indexer_wrapper.h",
    ],
    deps = [
        ":cache_indexer",
        "//kv_cache_manager/common:timestamp_util",
    ],
)

cc_library(
    name = "online_optimizer_manager",
    srcs = [
        "manager/online_optimizer_manager.cc",
    ],
    hdrs = [
        "manager/online_optimizer_manager.h",
    ],
    deps = [
        ":optimizer_instance_group",
        ":optimizer_instance_info",
        ":optimizer_registry_manager",
        ":cache_indexer",
        ":cache_indexer_factory",
        ":ttl_cache_indexer_wrapper",
        "//kv_cache_manager/common",
    ],
)

cc_library(
    name = "optimizer_metrics_collector",
    srcs = [
        "metrics/optimizer_metrics_collector.cc",
    ],
    hdrs = [
        "metrics/optimizer_metrics_collector.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/metrics:metrics_registry",
    ],
)

cc_library(
    name = "optimizer_metrics_reporter",
    srcs = [
        "metrics/optimizer_metrics_reporter.cc",
    ],
    hdrs = [
        "metrics/optimizer_metrics_reporter.h",
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
        "server/optimizer_call_guard.cc",
    ],
    hdrs = [
        "server/optimizer_call_guard.h",
    ],
    deps = [
        ":optimizer_metrics_collector",
        ":optimizer_metrics_reporter",
        "//kv_cache_manager/common",
    ],
)

cc_library(
    name = "optimizer_service_impl",
    srcs = [
        "server/optimizer_service_impl.cc",
    ],
    hdrs = [
        "server/optimizer_service_impl.h",
    ],
    deps = [
        ":online_optimizer_manager",
        ":optimizer_call_guard",
        ":optimizer_instance_group",
        ":optimizer_instance_info",
        ":optimizer_metrics_collector",
        ":optimizer_metrics_reporter",
        ":optimizer_registry_manager",
        "//kv_cache_manager/common",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
    ],
)

cc_library(
    name = "optimizer_service_grpc",
    srcs = [
        "server/grpc/optimizer_service_grpc.cc",
    ],
    hdrs = [
        "server/grpc/optimizer_service_grpc.h",
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
        "server/http/optimizer_service_http.cc",
    ],
    hdrs = [
        "server/http/optimizer_service_http.h",
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
        "server/online_optimizer_server_config.cc",
    ],
    hdrs = [
        "server/online_optimizer_server_config.h",
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
        "server/online_optimizer_server.cc",
    ],
    hdrs = [
        "server/online_optimizer_server.h",
    ],
    copts = [
        "-std=c++20",
        "-fcoroutines",
    ],
    deps = [
        ":online_optimizer_manager",
        ":online_optimizer_server_config",
        ":optimizer_metrics_reporter",
        ":optimizer_registry_manager",
        ":optimizer_service_grpc",
        ":optimizer_service_http",
        ":optimizer_service_impl",
        "//kv_cache_manager/common",
        "//kv_cache_manager/metrics:metrics_registry",
        "@grpc//:grpc++",
    ],
)

cc_binary(
    name = "online_optimizer_server_main",
    srcs = [
        "server/online_optimizer_server_main.cc",
    ],
    deps = [
        ":online_optimizer_server",
        "//kv_cache_manager/common",
    ],
)
