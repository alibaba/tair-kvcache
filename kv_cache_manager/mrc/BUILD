package(default_visibility = ["//visibility:public"])

# Observation-only online MRC (theoretical hit rate) components.
# The algorithm layer below depends on nothing but std; keep it that way so
# it stays reusable and trivially testable.
cc_library(
    name = "online_mrc_config",
    hdrs = ["online_mrc_config.h"],
)

cc_library(
    name = "mrc_profiler",
    srcs = [
        "lite_hit_core.cc",
        "mrc_profiler.cc",
    ],
    hdrs = [
        "lite_hit_core.h",
        "mrc_profiler.h",
    ],
)

cc_library(
    name = "hit_curve_projector",
    srcs = ["hit_curve_projector.cc"],
    hdrs = ["hit_curve_projector.h"],
)

# Separate research/reference implementation. The production online path uses
# the request-level exact LiteHit core above, not per-block sampling.
cc_library(
    name = "reuse_distance_tracker",
    srcs = ["reuse_distance_tracker.cc"],
    hdrs = ["reuse_distance_tracker.h"],
)

cc_library(
    name = "mrc_event_consumer",
    srcs = [
        "mrc_event_consumer.cc",
    ],
    hdrs = [
        "mrc_event_consumer.h",
    ],
    deps = [
        ":mrc_profiler",
        ":online_mrc_config",
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common:loop_thread",
        "//kv_cache_manager/event:event_publisher",
        "//kv_cache_manager/event/spec_events",
        "//kv_cache_manager/metrics:metrics_registry",
        "@rapidjson",
    ],
)

cc_library(
    name = "online_mrc_registry",
    srcs = [
        "online_mrc_registry.cc",
        "online_mrc_trace_receiver.cc",
    ],
    hdrs = [
        "online_mrc_registry.h",
        "online_mrc_trace_receiver.h",
    ],
    deps = [
        ":mrc_profiler",
        ":online_mrc_config",
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/metrics:metrics_registry",
        "@rapidjson",
    ],
)

cc_library(
    name = "online_mrc_fact_stream",
    srcs = [
        "kvcm_event_stream_client.cc",
        "online_mrc_fact_registry.cc",
    ],
    hdrs = [
        "kvcm_event_stream_client.h",
        "online_mrc_fact_registry.h",
    ],
    deps = [
        ":online_mrc_config",
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common:service_discovery",
        "//kv_cache_manager/common:service_discovery_factory",
        "//kv_cache_manager/common:string_util",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/optimizer:online_optimizer_manager",
        "//kv_cache_manager/optimizer/config:online_optimizer_config",
        "//kv_cache_manager/optimizer/config:optimizer_registry_manager",
        "//kv_cache_manager/optimizer/liteHit:hit_curve",
        "//kv_cache_manager/optimizer/liteHit:lite_hit",
        "//kv_cache_manager/optimizer/liteHit:request_preprocess",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
    ],
)

cc_library(
    name = "optimizer_trace_forwarder",
    srcs = ["optimizer_trace_forwarder.cc"],
    hdrs = ["optimizer_trace_forwarder.h"],
    deps = [
        ":online_mrc_config",
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common:loop_thread",
        "//kv_cache_manager/common:string_util",
        "//kv_cache_manager/event:event_publisher",
        "//kv_cache_manager/event/spec_events",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/protocol/protobuf:service_cc_grpc",
        "@grpc//:grpc++",
    ],
)

# Offline replay harness: validates the online MRC engine against formal
# LiteHit results on the same trace (P1c acceptance). Not part of any server.
cc_binary(
    name = "mrc_replay_main",
    srcs = [
        "tools/mrc_replay_main.cc",
    ],
    deps = [
        ":mrc_profiler",
    ],
)
