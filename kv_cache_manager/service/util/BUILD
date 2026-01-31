package(default_visibility = ["//visibility:public"])

config_setting(
    name = "client_only_header",
    values = {"define": "CLIENT_ONLY_HEADER=true"},
)

cc_library(
    name = "proto_message_json_util",
    srcs = ["proto_message_json_util.cc"],
    hdrs = ["proto_message_json_util.h"],
    include_prefix = "service/util",
    deps = [
        ":grpc_wrapper",
        "//kv_cache_manager/common",
    ],
)

cc_library(
    name = "manager_message_proto_util",
    srcs = ["manager_message_proto_util.cc"],
    hdrs = ["manager_message_proto_util.h"],
    deps = [
        ":grpc_wrapper",
        "//kv_cache_manager/common",
        "//kv_cache_manager/config",
        "//kv_cache_manager/config:meta_cache_policy_config",
        "//kv_cache_manager/data_storage",
        "//kv_cache_manager/manager:cache_location",
        "//kv_cache_manager/manager:cache_location_view",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
    ],
)

cc_library(
    name = "service_call_guard",
    srcs = ["service_call_guard.cc"],
    hdrs = ["service_call_guard.h"],
    deps = [
        "//kv_cache_manager/manager:cache_manager",
        "//kv_cache_manager/metrics:metrics_reporter",
    ],
)

cc_library(
    name = "service_access_log",
    srcs = ["service_access_log.cc"],
    hdrs = ["service_access_log.h"],
    deps = [
        "//kv_cache_manager/manager:cache_manager",
        "//kv_cache_manager/metrics:metrics_reporter",
    ],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    deps = [
        "//kv_cache_manager/metrics:metrics_collector",
    ],
)

cc_library(
    name = "fault_injector",
    srcs = [
        "fault_injector.cc"
    ],
    hdrs = [
        "fault_injector.h"
    ],
    deps = [
        "//kv_cache_manager/common:logger",
    ],
    
)

cc_library(
    name = "grpc_wrapper",
    deps = select({
        "client_only_header": [
            "@grpc//:grpc++_public_hdrs",
        ],
        "//conditions:default": [
            "@grpc//:grpc++",
        ],
    }),
)
