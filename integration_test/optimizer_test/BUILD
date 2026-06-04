package(default_visibility = ["//visibility:private"])

cc_test(
    name = "OptimizerIntegrationTest",
    srcs = [
        "optimizer_integration_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/online_optimizer:online_optimizer_server",
        "//kv_cache_manager/protocol/protobuf:service_cc_grpc",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
        "@grpc//:grpc++",
    ],
)
