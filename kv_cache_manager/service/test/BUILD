package(default_visibility = ["//visibility:private"])

cc_test(
    name = "QuotaPolicyPollerTest",
    srcs = ["quota_policy_poller_test.cc"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/protocol/protobuf:service_cc_grpc",
        "//kv_cache_manager/service:quota_policy_poller",
        "@grpc//:grpc++",
    ],
)

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
    name = "ServerEventPublisherTest",
    srcs = [
        "server_event_publisher_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/event:event_publisher",
        "//kv_cache_manager/event:optimizer_event_publisher",
        "//kv_cache_manager/event/optimizer_stream:subscription_event_sink",
        "//kv_cache_manager/service",
        "//kv_cache_manager/service/grpc_service:optimizer_event_service_grpc",
    ],
)

cc_test(
    name = "OptimizerEventServiceGRpcTest",
    srcs = [
        "optimizer_event_service_grpc_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config",
        "//kv_cache_manager/config:leader_elector_base",
        "//kv_cache_manager/event:event_publisher",
        "//kv_cache_manager/event:event_publishers_config",
        "//kv_cache_manager/event:optimizer_event_publisher",
        "//kv_cache_manager/event/optimizer_stream:subscription_event_sink",
        "//kv_cache_manager/event/spec_events",
        "//kv_cache_manager/manager:cache_manager",
        "//kv_cache_manager/manager:meta_searcher",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/protocol/protobuf:service_cc_grpc",
        "//kv_cache_manager/service:grpc_wrapper",
        "//kv_cache_manager/service",
        "//kv_cache_manager/service/grpc_service:optimizer_event_service_grpc",
    ],
)

cc_test(
    name = "OptimizerEventServiceHttpTest",
    srcs = [
        "optimizer_event_service_http_test.cc",
    ],
    copts = [
        "-fcoroutines",
        "-fno-access-control",
        "-std=c++20",
    ],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config",
        "//kv_cache_manager/event:event_publisher",
        "//kv_cache_manager/event:event_publishers_config",
        "//kv_cache_manager/event:optimizer_event_publisher",
        "//kv_cache_manager/event/optimizer_stream:subscription_event_sink",
        "//kv_cache_manager/manager:cache_manager",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
        "//kv_cache_manager/service:meta_service_impl",
        "//kv_cache_manager/service/http_service",
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

cc_test(
    name = "AdminServiceImplTest",
    srcs = [
        "admin_service_impl_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/manager:cache_manager",
        "//kv_cache_manager/manager:meta_searcher",
        "//kv_cache_manager/manager:migration_manager",
        "//kv_cache_manager/meta",
        "//kv_cache_manager/service:admin_service_impl",
    ],
)
