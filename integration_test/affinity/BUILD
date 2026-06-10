package(default_visibility = ["//integration_test:__subpackages__"])

py_test(
    name = "affinity_replication_test",
    srcs = ["affinity_replication_test.py"],
    data = [],
    imports = [
        ".",
        "../testlib",
    ],
    tags = ["no-remote-exec"],
    deps = [
        "//integration_test/testlib:test_base",
        "//kv_cache_manager/protocol/protobuf:meta_service_py_proto",
        "@pip_cpu//grpcio",
        "@pip_cpu//protobuf",
    ],
)
