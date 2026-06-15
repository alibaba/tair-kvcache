package(default_visibility = ["//visibility:private"])

cc_test(
    name = "MetaServiceMetricsBaseTest",
    srcs = [
        "meta_service_metrics_base_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/service:meta_service_metrics_base",
    ],
)

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
    name = "AdminServiceRemoveInstanceGroupTest",
    srcs = [
        "admin_service_remove_instance_group_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config",
        "//kv_cache_manager/manager:cache_manager",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/service:admin_service_impl",
    ],
)

cc_test(
    name = "AdminServiceRemoveStorageTest",
    srcs = [
        "admin_service_remove_storage_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config",
        "//kv_cache_manager/data_storage",
        "//kv_cache_manager/manager:cache_manager",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/metrics:metrics_reporter",
        "//kv_cache_manager/service:admin_service_impl",
    ],
)
