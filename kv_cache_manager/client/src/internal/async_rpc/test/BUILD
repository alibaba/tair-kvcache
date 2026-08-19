package(default_visibility = ["//visibility:private"])

cc_test(
    name = "async_rpc_client_test",
    size = "small",
    srcs = ["async_rpc_client_test.cc"],
    copts = [
        "-std=c++20",
        "-fcoroutines",
    ],
    deps = [
        "//kv_cache_manager/client/src/internal/async_rpc:core",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
        "@com_google_googletest//:gtest_main",
    ],
)
