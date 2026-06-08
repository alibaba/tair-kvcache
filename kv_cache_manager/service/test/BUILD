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

cc_test(
    name = "AdminAuthTokensTest",
    srcs = [
        "admin_auth_tokens_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config",
        "//kv_cache_manager/manager:cache_manager",
        "//kv_cache_manager/manager:startup_config_loader",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/metrics:metrics_reporter",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
        "//kv_cache_manager/service:admin_service_impl",
        "//kv_cache_manager/service/http_service:auth",
    ],
)
