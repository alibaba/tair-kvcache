package(default_visibility = ["//visibility:public"])

cc_library(
    name = "meta_route",
    srcs = ["meta_route.cc"],
    hdrs = ["meta_route.h"],
    deps = ["//kv_cache_manager/protocol/protobuf:service_cc_proto", "//kv_cache_manager/routing"],
)

cc_library(
    name = "partition_reporter",
    srcs = ["partition_reporter.cc"],
    hdrs = ["partition_reporter.h"],
    deps = [
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common:service_discovery_factory",
        "//kv_cache_manager/common:string_util",
        "//kv_cache_manager/protocol/protobuf:proxy_control_grpc",
        "//kv_cache_manager/routing",
        "//kv_cache_manager/service:grpc_wrapper",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "meta_forwarder",
    srcs = ["broadcast_result.cc", "meta_forwarder.cc"],
    hdrs = ["broadcast_result.h", "meta_forwarder.h"],
    deps = [
        ":meta_route",
        "@rapidjson",
        "//kv_cache_manager/common:timestamp_util",
        "//kv_cache_manager/protocol/protobuf:service_cc_grpc",
        "//kv_cache_manager/routing",
        "//kv_cache_manager/service:grpc_wrapper",
    ],
)
