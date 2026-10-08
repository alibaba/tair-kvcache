package(default_visibility = ["//integration_test:__subpackages__"])

cc_test(
    name = "AffinityPendingContractTest",
    srcs = ["affinity_pending_contract_test.cc"],
    deps = [
        "//kv_cache_manager/affinity",
        "//kv_cache_manager/affinity:local_replica_strategy",
        "//kv_cache_manager/common:unittest",
    ],
)

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
