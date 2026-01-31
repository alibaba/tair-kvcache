package(default_visibility = ["//visibility:public"])

load("//bazel:arch_select.bzl", "requirement")

requirement([
    "grpcio",
    "protobuf",
])

py_binary(
    name = "kv_cache_manager_rpc_client",
    srcs = glob([
        "*.py",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":grpcio",
        ":protobuf",
        "//kv_cache_manager/protocol/protobuf:admin_service_py_proto",
        "//kv_cache_manager/protocol/protobuf:debug_service_py_proto",
        "//kv_cache_manager/protocol/protobuf:meta_service_py_proto",
    ],
)
