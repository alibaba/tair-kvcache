package(default_visibility = ["//kv_cache_manager/client:__subpackages__"])

cc_library(
    name = "grpc_stub",
    srcs = [
        "grpc_stub.cc",
    ],
    hdrs = [
        "grpc_stub.h",
    ],
    deps = [
        ":base_stub",
        "//kv_cache_manager/protocol/protobuf:service_cc_grpc",
    ],
)

cc_library(
    name = "base_stub",
    hdrs = [
        "stub.h",
    ],
    deps = [
        "//kv_cache_manager/client:client_api",
        "//kv_cache_manager/client/src/internal/util",
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/config",
        "//kv_cache_manager/service/util:manager_message_proto_util",
    ],
)
