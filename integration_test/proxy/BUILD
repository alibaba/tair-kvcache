package(default_visibility = ["//integration_test:__subpackages__"])

py_test(
    name = "partition_proxy_test",
    srcs = ["partition_proxy_test.py"],
    args = ["$(location //kv_cache_manager:kv_cache_manager_bin)"],
    data = ["//kv_cache_manager:kv_cache_manager_bin"],
    tags = ["no-remote-exec"],
    deps = [
        "//kv_cache_manager/protocol/protobuf:meta_service_py_proto",
        "@pip_cpu//grpcio",
        "@pip_cpu//protobuf",
    ],
)
