package(default_visibility = ["//visibility:private"])

cc_test(
    name = "OptimizerIntegrationTest",
    srcs = [
        "optimizer_integration_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/optimizer:online_optimizer_server",
        "//kv_cache_manager/protocol/protobuf:service_cc_grpc",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
        "@grpc//:grpc++",
    ],
)

# --- Python Integration Tests ---

py_library(
    name = "optimizer_interface_cases",
    srcs = [
        "optimizer_interface_cases.py",
    ],
    data = [
        "//kv_cache_manager/optimizer:online_optimizer_server_main",
    ],
    deps = [],
)

py_test(
    name = "optimizer_http_interface_test",
    srcs = ["http_interface_test.py"],
    main = "http_interface_test.py",
    data = [
        "//kv_cache_manager/optimizer:online_optimizer_server_main",
    ],
    tags = ["no-remote-exec"],
    deps = [
        ":optimizer_interface_cases",
        "@pip_cpu//requests",
    ],
)

py_test(
    name = "optimizer_grpc_interface_test",
    srcs = ["grpc_interface_test.py"],
    main = "grpc_interface_test.py",
    data = [
        "//kv_cache_manager/optimizer:online_optimizer_server_main",
    ],
    tags = ["no-remote-exec"],
    deps = [
        ":optimizer_interface_cases",
        "//kv_cache_manager/protocol/protobuf:optimizer_service_py_proto",
        "@pip_cpu//grpcio",
        "@pip_cpu//protobuf",
    ],
)
