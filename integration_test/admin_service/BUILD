package(default_visibility = ["//integration_test:__subpackages__"])

py_library(
    name = "admin_interface_cases",
    srcs = [
        "admin_interface_cases.py",
    ],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    imports = ["../testlib"],
    deps = [
        "//integration_test/testlib:test_base",
    ],
)

py_test(
    name = "http_interface_test",
    srcs = ["http_interface_test.py"],
    data = [],
    tags = ["no-remote-exec"],
    deps = [
        ":admin_interface_cases",
        "//integration_test/testlib:test_base",
        "@pip_cpu//requests",
    ],
)

py_test(
    name = "grpc_interface_test",
    srcs = ["grpc_interface_test.py"],
    data = [],
    imports = [
        ".",
        "../testlib",
    ],
    tags = ["no-remote-exec"],
    deps = [
        ":admin_interface_cases",
        "//integration_test/testlib:test_base",
        "//kv_cache_manager/protocol/protobuf:admin_service_py_proto",
        "@pip_cpu//grpcio",
        "@pip_cpu//protobuf",
    ],
)
