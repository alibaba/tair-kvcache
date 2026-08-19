# KVCM Swarm: a standalone C++ client-behavior runner and evidence system.
#
# It only talks to a real KVCM deployment through public APIs. It never links
# kv_cache_manager server internals, and never links the CI fixture or the
# scenario evaluator.
package(default_visibility = ["//visibility:public"])

SWARM_COPTS = [
    "-std=c++20",
    "-fcoroutines",
]

cc_library(
    name = "runtime",
    srcs = [
        "runtime/admission.cc",
        "runtime/clock.cc",
        "runtime/executor.cc",
        "runtime/rng.cc",
    ],
    hdrs = [
        "runtime/admission.h",
        "runtime/clock.h",
        "runtime/executor.h",
        "runtime/rng.h",
        "runtime/sample_spec.h",
        "runtime/stop_token.h",
    ],
    copts = SWARM_COPTS,
    deps = [
        "//kv_cache_manager/client/src/internal/async_rpc:cancellation",
        "@yalantinglibs//:ylt",
    ],
)

cc_library(
    name = "evidence",
    srcs = [
        "evidence/histogram.cc",
        "evidence/json_writer.cc",
        "evidence/observation.cc",
        "evidence/sink.cc",
    ],
    hdrs = [
        "evidence/histogram.h",
        "evidence/json_writer.h",
        "evidence/observation.h",
        "evidence/sink.h",
    ],
    copts = SWARM_COPTS,
    deps = [
        ":runtime",
        "//kv_cache_manager/client/src/internal/async_rpc:core",
        "@rapidjson",
    ],
)

cc_library(
    name = "protocol",
    hdrs = [
        "protocol/api.h",
        "protocol/json_codec.h",
        "protocol/proto_alias.h",
    ],
    copts = SWARM_COPTS,
    deps = [
        "//kv_cache_manager/client/src/internal/async_rpc:core",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "transport",
    srcs = ["transport/transport_provider.cc"],
    hdrs = [
        "transport/call_recorder.h",
        "transport/transport.h",
        "transport/transport_provider.h",
    ],
    copts = SWARM_COPTS,
    deps = [
        ":evidence",
        ":protocol",
        ":runtime",
        "//kv_cache_manager/client/src/internal/async_rpc:async_rpc_client",
    ],
)

cc_library(
    name = "scenario_config",
    srcs = [
        "scenario/config_node.cc",
        "scenario/config_reader.cc",
    ],
    hdrs = [
        "scenario/config_node.h",
        "scenario/config_reader.h",
    ],
    copts = SWARM_COPTS,
    deps = [
        ":evidence",
        ":runtime",
        "@rapidjson",
    ],
)

cc_library(
    name = "behavior_api",
    hdrs = [
        "clients/client_behavior.h",
        "scenario/config.h",
    ],
    copts = SWARM_COPTS,
    deps = [
        ":evidence",
        ":runtime",
        ":scenario_config",
        ":transport",
    ],
)

cc_library(
    name = "health_probe",
    srcs = [
        "clients/health/config.cc",
        "clients/health/health_probe.cc",
    ],
    hdrs = [
        "clients/health/config.h",
        "clients/health/health_probe.h",
    ],
    copts = SWARM_COPTS,
    deps = [
        ":behavior_api",
        ":protocol",
    ],
)

cc_library(
    name = "v6d_model",
    srcs = [
        "clients/v6d/config.cc",
        "clients/v6d/expected_locations.cc",
        "clients/v6d/key_mapper.cc",
        "clients/v6d/local_cache.cc",
        "clients/v6d/workload.cc",
    ],
    hdrs = [
        "clients/v6d/config.h",
        "clients/v6d/expected_locations.h",
        "clients/v6d/key_mapper.h",
        "clients/v6d/local_cache.h",
        "clients/v6d/workload.h",
    ],
    copts = SWARM_COPTS,
    deps = [
        ":behavior_api",
        ":protocol",
    ],
)

cc_library(
    name = "registry",
    srcs = ["clients/registry.cc"],
    hdrs = ["clients/registry.h"],
    copts = SWARM_COPTS,
    deps = [
        ":behavior_api",
        ":health_probe",
        ":protocol",
    ],
)

cc_library(
    name = "scenario_loader",
    srcs = ["scenario/loader.cc"],
    hdrs = ["scenario/loader.h"],
    copts = SWARM_COPTS,
    deps = [
        ":registry",
        ":scenario_config",
    ],
)
