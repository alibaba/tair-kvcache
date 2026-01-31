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
