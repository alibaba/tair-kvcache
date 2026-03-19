package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "bench_meta_service",
    srcs = ["bench_meta_service.cpp"],
    deps = [
        "//kv_cache_manager/protocol/protobuf:service_cc_grpc",
        "//kv_cache_manager/service:grpc_wrapper",
    ],
    linkopts = ["-pthread"],
)
