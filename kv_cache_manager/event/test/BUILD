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