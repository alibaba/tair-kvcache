package(default_visibility = ["//visibility:public"])

cc_library(
    name = "grpc_service",
    srcs = [
        "admin_service_grpc.cc",
        "debug_service_grpc.cc",
        "meta_service_grpc.cc",
    ],
    hdrs = [
        "admin_service_grpc.h",
        "debug_service_grpc.h",
        "meta_service_grpc.h",
    ],
    deps = [
        "//kv_cache_manager/common:request_context",
        "//kv_cache_manager/config",
        "//kv_cache_manager/manager:cache_manager",
        "//kv_cache_manager/protocol/protobuf:service_cc_grpc",
        "//kv_cache_manager/service:admin_service_impl",
        "//kv_cache_manager/service:debug_service_impl",
        "//kv_cache_manager/service:meta_service_impl",
        "//kv_cache_manager/service:meta_service_metrics_base",
    ],
)

cc_library(
    name = "optimizer_event_service_grpc",
    srcs = [
        "optimizer_event_service_grpc.cc",
    ],
    hdrs = [
        "optimizer_event_service_grpc.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/config",
        "//kv_cache_manager/config:leader_elector_base",
        "//kv_cache_manager/event/optimizer_stream:subscription_event_sink",
        "//kv_cache_manager/manager:cache_manager",
        "//kv_cache_manager/protocol/protobuf:service_cc_grpc",
        "//kv_cache_manager/service:grpc_wrapper",
        "//kv_cache_manager/service/util:optimizer_event_proto_util",
    ],
)
