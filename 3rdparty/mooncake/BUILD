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
