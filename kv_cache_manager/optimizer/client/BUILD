package(default_visibility = ["//visibility:public"])

py_test(
    name = "OptimizerClientTest",
    srcs = ["client_test.py"],
    main = "client_test.py",
    deps = [":optimizer_client"],
)

py_library(
    name = "optimizer_client",
    srcs = [
        "__init__.py",
        "base.py",
        "grpc_client.py",
        "http_client.py",
    ],
    deps = [
        "//kv_cache_manager/protocol/protobuf:optimizer_service_py_proto",
        "@pip_cpu//grpcio",
        "@pip_cpu//protobuf",
        "@pip_cpu//requests",
    ],
)
