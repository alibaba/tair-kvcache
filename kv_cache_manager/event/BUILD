package(default_visibility = ["//visibility:public"])

cc_library(
    name = "event_publisher",
    srcs = [
        "event_manager.cc",
        "log_event_publisher.cc",
    ],
    hdrs = [
        "event_manager.h",
        "event_publisher.h",
        "log_event_publisher.h",
    ],
    deps = [
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common:string_util",
        ":base_event",
        "//kv_cache_manager/event/spec_events",
    ],
)

cc_library(
    name = "base_event",
    srcs = [],
    hdrs = [
        "base_event.h",
    ],
    deps = [
        "//kv_cache_manager/common:jsonizable",
    ],
)
