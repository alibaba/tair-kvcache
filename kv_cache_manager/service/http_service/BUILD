package(default_visibility = ["//kv_cache_manager/service:__subpackages__"])

cc_library(
    name = "http_service",
    srcs = [
        "admin_service_http.cc",
        "coro_http_service.cc",
        "debug_service_http.cc",
        "meta_service_http.cc",
    ],
    hdrs = [
        "admin_service_http.h",
        "coro_http_service.h",
        "debug_service_http.h",
        "meta_service_http.h",
    ],
    copts = [
        "-std=c++20",
        "-fcoroutines",
    ],
    include_prefix = "service/http_service",
    deps = [
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common:request_context",
        "//kv_cache_manager/config",
        "//kv_cache_manager/data_storage",
        "//kv_cache_manager/data_storage:common_define",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
        "//kv_cache_manager/service:admin_service_impl",
        "//kv_cache_manager/service:debug_service_impl",
        "//kv_cache_manager/service:meta_service_impl",
        "//kv_cache_manager/service:meta_service_metrics_base",
        "//kv_cache_manager/service/util:proto_message_json_util",
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//:protobuf_headers",
        "@yalantinglibs//:ylt",
    ],
)
