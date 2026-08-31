package(default_visibility = ["//visibility:public"])

cc_library(
    name = "event_sink",
    hdrs = [
        "event_sink.h",
    ],
)

cc_library(
    name = "subscription_event_sink",
    srcs = [
        "subscription_event_sink.cc",
    ],
    hdrs = [
        "subscription_event_sink.h",
    ],
    deps = [
        ":event_sink",
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/event:event_publishers_config",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
    ],
)
