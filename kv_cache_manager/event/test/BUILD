package(default_visibility = ["//visibility:private"])

cc_test(
    name = "BaseEventTest",
    srcs = [
        "base_event_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/event:event_publisher",
    ],
)

cc_test(
    name = "EventPublisherTest",
    srcs = [
        "event_publisher_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/event:event_publisher",
    ],
)

cc_test(
    name = "LogEventPublisherTest",
    srcs = [
        "log_event_publisher_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/event:event_publisher",
    ],
)

cc_test(
    name = "EventManagerTest",
    srcs = [
        "event_manager_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/event:event_publisher",
    ],
)

cc_test(
    name = "MigrationEventTest",
    srcs = [
        "migration_event_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/event/spec_events:spec_events",
    ],
)

cc_test(
    name = "OptimizerQueryHitEventTest",
    srcs = [
        "optimizer_query_hit_event_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/event/spec_events:spec_events",
    ],
)

cc_test(
    name = "EventPublishersConfigTest",
    srcs = [
        "event_publishers_config_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/event:event_publishers_config",
    ],
)

cc_test(
    name = "SubscriptionEventSinkTest",
    srcs = [
        "subscription_event_sink_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/event:event_publishers_config",
        "//kv_cache_manager/event/optimizer_stream:subscription_event_sink",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
    ],
)

cc_test(
    name = "OptimizerEventPublisherTest",
    srcs = [
        "optimizer_event_publisher_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/event:event_publishers_config",
        "//kv_cache_manager/event:optimizer_event_publisher",
        "//kv_cache_manager/event/spec_events",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
    ],
)
