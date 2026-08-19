package(default_visibility = [
    "//kv_cache_manager/client:__subpackages__",
    "//tools/kvcm_swarm:__pkg__",
])

ASYNC_RPC_COPTS = [
    "-std=c++20",
    "-fcoroutines",
]

cc_library(
    name = "cancellation",
    hdrs = ["cancellation.h"],
)

cc_library(
    name = "core",
    srcs = [
        "api.cc",
        "json_codec.cc",
        "types.cc",
    ],
    hdrs = [
        "api.h",
        "async_rpc_client.h",
        "async_slot.h",
        "json_codec.h",
        "rpc_util.h",
    ],
    copts = ASYNC_RPC_COPTS,
    deps = [
        ":cancellation",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
        "@com_google_protobuf//:protobuf",
        "@yalantinglibs//:ylt",
    ],
)

cc_library(
    name = "grpc_client",
    srcs = ["grpc_client.cc"],
    hdrs = ["grpc_client.h"],
    copts = ASYNC_RPC_COPTS,
    deps = [
        ":core",
        "@grpc//:grpc++",
    ],
)

cc_library(
    name = "http_client",
    srcs = ["http_client.cc"],
    hdrs = ["http_client.h"],
    copts = ASYNC_RPC_COPTS,
    deps = [
        ":core",
        "@yalantinglibs//:ylt",
    ],
)

cc_library(
    name = "async_rpc_client",
    srcs = ["provider.cc"],
    copts = ASYNC_RPC_COPTS,
    deps = [
        ":core",
        ":grpc_client",
        ":http_client",
    ],
)
