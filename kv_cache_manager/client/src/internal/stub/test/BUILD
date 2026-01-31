package(default_visibility = ["//visibility:private"])

cc_test(
    name = "GrpcStubTest",
    srcs = [
        "grpc_stub_test.cc",
    ],
    copts = ["-fno-access-control"],
    shard_count = 10,
    deps = [
        "//kv_cache_manager/client/src/internal/stub:grpc_stub",
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config",
        "//kv_cache_manager/manager:cache_manager",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/metrics:metrics_reporter_factory",
        "//kv_cache_manager/service/grpc_service",
    ],
)
