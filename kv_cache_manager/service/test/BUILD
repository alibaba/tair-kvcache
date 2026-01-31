package(default_visibility = ["//visibility:private"])

cc_test(
    name = "CommandLineTest",
    srcs = [
        "command_line_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/service",
    ],
)

cc_test(
    name = "ServerConfigTest",
    srcs = [
        "server_config_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [
        "testdata/server_config_simple.conf",
    ],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/service",
    ],
)
