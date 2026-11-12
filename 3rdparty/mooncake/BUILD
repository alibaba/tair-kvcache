exports_files(["client_c.h", "client_c.ccs"])

cc_library(
    name = "mooncake_client_c",
    srcs = ["client_c.cc"],
    hdrs = ["client_c.h"],
    deps = ["@mooncake//:mooncake_store"],
    copts = ["-std=c++20",
        "-g",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        "-fcoroutines",
        "-fPIC"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "mooncake_client_c_test",
    srcs = ["client_c_test.cc"],
    copts = [
        "-std=c++20",
        "-g",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        "-fcoroutines",
        "-fPIC",
    ],
    deps = [
        ":mooncake_client_c",
        "//kv_cache_manager/common:unittest",
        "@mooncake//:mooncake_store",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "mooncake_client_c_integration_test",
    srcs = ["client_c_integration_test.cc"],
    copts = [
        "-std=c++20",
        "-g",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        "-fcoroutines",
        "-fPIC",
    ],
    tags = [
        "manual",
        "external",
    ],
    deps = [
        ":mooncake_client_c",
        "//kv_cache_manager/common:unittest",
        "@mooncake//:mooncake_store",
        "@com_google_googletest//:gtest_main",
    ],
)

# Cross-node put/get smoke test, e.g. over EFA: see header of client_c_smoke.cc.
cc_binary(
    name = "client_c_smoke",
    srcs = ["client_c_smoke.cc"],
    copts = [
        "-std=c++20",
        "-g",
        "-Wall",
        "-Wextra",
        "-Wno-unused-parameter",
        "-fPIC",
    ],
    tags = ["manual"],
    deps = [
        ":mooncake_client_c",
        "@com_github_gflags_gflags//:gflags",
    ],
)
