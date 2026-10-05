package(default_visibility = ["//integration_test:__subpackages__"])

cc_test(
    name = "AffinityPendingContractTest",
    srcs = ["affinity_pending_contract_test.cc"],
    # Explicit acceptance gate for known missing behavior; do not count as passed
    # merely because wildcard CI omits manual targets.
    tags = ["manual"],
    deps = [
        "//kv_cache_manager/affinity",
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
