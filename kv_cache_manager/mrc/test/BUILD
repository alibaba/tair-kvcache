package(default_visibility = ["//visibility:private"])

cc_test(
    name = "ReuseDistanceTrackerTest",
    srcs = [
        "reuse_distance_tracker_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/mrc:reuse_distance_tracker",
    ],
)

cc_test(
    name = "MrcProfilerTest",
    srcs = [
        "mrc_profiler_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/mrc:mrc_profiler",
    ],
)

cc_test(
    name = "LiteHitCoreTest",
    srcs = [
        "lite_hit_core_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/mrc:mrc_profiler",
    ],
)

cc_test(
    name = "MrcEventConsumerTest",
    srcs = [
        "mrc_event_consumer_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/event:event_publisher",
        "//kv_cache_manager/event/spec_events",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/mrc:mrc_event_consumer",
    ],
)

cc_test(
    name = "OnlineMrcRegistryTest",
    srcs = ["online_mrc_registry_test.cc"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/mrc:online_mrc_registry",
    ],
)

cc_test(
    name = "OptimizerTraceForwarderTest",
    srcs = ["optimizer_trace_forwarder_test.cc"],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/mrc:optimizer_trace_forwarder",
    ],
)

cc_test(
    name = "OnlineMrcFactStreamTest",
    srcs = ["online_mrc_fact_stream_test.cc"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/metrics:prometheus_exporter",
        "//kv_cache_manager/mrc:online_mrc_fact_stream",
        "//kv_cache_manager/optimizer:online_optimizer_manager",
        "//kv_cache_manager/optimizer/config:optimizer_registry_manager",
        "//kv_cache_manager/optimizer/liteHit:hit_curve",
        "//kv_cache_manager/optimizer/liteHit:lite_hit",
        "//kv_cache_manager/optimizer/liteHit:request_preprocess",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
    ],
)
