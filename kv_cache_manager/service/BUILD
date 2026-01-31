package(default_visibility = ["//visibility:public"])

config_setting(
    name = "client_only_header",
    values = {"define": "CLIENT_ONLY_HEADER=true"},
)

cc_library(
    name = "service",
    srcs = [
        "command_line.cc",
        "server.cc",
        "server_config.cc",
    ],
    hdrs = [
        "command_line.h",
        "server.h",
        "server_config.h",
    ],
    copts = [
        "-std=c++20",
        "-fcoroutines",
    ],
    deps = [
        ":grpc_wrapper",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:loop_thread",
        "//kv_cache_manager/config:distributed_lock_backend",
        "//kv_cache_manager/config:leader_elector_base",
        "//kv_cache_manager/event:event_publisher",
        "//kv_cache_manager/manager:cache_location_view",
        "//kv_cache_manager/manager:cache_manager",
        "//kv_cache_manager/manager:write_location_manager",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/metrics:metrics_reporter",
        "//kv_cache_manager/metrics:metrics_reporter_factory",
        "//kv_cache_manager/service/grpc_service",
        "//kv_cache_manager/service/http_service",
        "//kv_cache_manager/service/util:manager_message_proto_util",
        "//kv_cache_manager/service/util:service_call_guard",
    ],
)

cc_library(
    name = "service_impl_base",
    srcs = [
        "service_impl_base.cc",
    ],
    hdrs = [
        "service_impl_base.h",
    ],
    deps = [
        ":grpc_wrapper",
        "//kv_cache_manager/common",
    ],
)

cc_library(
    name = "meta_service_impl",
    srcs = glob([
        "meta_service_impl.cc",
    ]),
    hdrs = glob([
        "meta_service_impl.h",
    ]),
    deps = [
        ":grpc_wrapper",
        ":service_impl_base",
        "//kv_cache_manager/common",
        "//kv_cache_manager/manager:cache_location_view",
        "//kv_cache_manager/manager:cache_manager",
        "//kv_cache_manager/manager:write_location_manager",
        "//kv_cache_manager/service/util:fault_injector",
        "//kv_cache_manager/service/util:manager_message_proto_util",
        "//kv_cache_manager/service/util:proto_message_json_util",
        "//kv_cache_manager/service/util:service_access_log",
        "//kv_cache_manager/service/util:service_call_guard",
    ],
)

cc_library(
    name = "admin_service_impl",
    srcs = glob([
        "admin_service_impl.cc",
    ]),
    hdrs = glob([
        "admin_service_impl.h",
    ]),
    deps = [
        ":grpc_wrapper",
        ":service_impl_base",
        "//kv_cache_manager/common",
        "//kv_cache_manager/config",
        "//kv_cache_manager/config:leader_elector_base",
        "//kv_cache_manager/data_storage",
        "//kv_cache_manager/manager:cache_manager",
        "//kv_cache_manager/service/util:manager_message_proto_util",
        "//kv_cache_manager/service/util:proto_message_json_util",
        "//kv_cache_manager/service/util:service_access_log",
        "//kv_cache_manager/service/util:service_call_guard",
    ],
)

cc_library(
    name = "debug_service_impl",
    srcs = glob([
        "debug_service_impl.cc",
    ]),
    hdrs = glob([
        "debug_service_impl.h",
    ]),
    deps = [
        ":grpc_wrapper",
        "//kv_cache_manager/common",
        "//kv_cache_manager/config",
        "//kv_cache_manager/data_storage",
        "//kv_cache_manager/manager:cache_manager",
        "//kv_cache_manager/service/util:fault_injector",
        "//kv_cache_manager/service/util:manager_message_proto_util",
        "//kv_cache_manager/service/util:service_access_log",
        "//kv_cache_manager/service/util:service_call_guard",
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

cc_library(
    name = "meta_service_metrics_base",
    srcs = [
        "meta_service_metrics_base.cc",
    ],
    hdrs = [
        "meta_service_metrics_base.h",
    ],
    deps = [
        "//kv_cache_manager/config",
        "//kv_cache_manager/metrics:metrics_collector",
        "//kv_cache_manager/service/util:common",
    ],
)
