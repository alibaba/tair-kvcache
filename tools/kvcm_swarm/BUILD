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
        "scenario/duration.cc",
        "scenario/json_config.cc",
    ],
    hdrs = [
        "scenario/duration.h",
        "scenario/json_config.h",
    ],
    copts = SWARM_COPTS,
    deps = [
        ":runtime",
        "//kv_cache_manager/common:jsonizable",
        "@rapidjson",
    ],
)

cc_library(
    name = "behavior_api",
    srcs = ["scenario/config.cc"],
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
    name = "v6d_behavior",
    srcs = [
        "clients/v6d/checks.cc",
        "clients/v6d/deployment.cc",
        "clients/v6d/process.cc",
        "clients/v6d/session_manager.cc",
    ],
    hdrs = [
        "clients/v6d/checks.h",
        "clients/v6d/deployment.h",
        "clients/v6d/process.h",
        "clients/v6d/session_manager.h",
    ],
    copts = SWARM_COPTS,
    deps = [
        ":behavior_api",
        ":protocol",
        ":v6d_model",
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
        ":v6d_behavior",
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

cc_library(
    name = "app",
    srcs = [
        "app/preflight.cc",
        "app/run_coordinator.cc",
        "evidence/report.cc",
    ],
    hdrs = [
        "app/preflight.h",
        "app/run_coordinator.h",
        "evidence/report.h",
    ],
    copts = SWARM_COPTS,
    deps = [
        ":registry",
        ":scenario_loader",
        ":v6d_model",
    ],
)

cc_binary(
    name = "kvcm_swarm",
    srcs = ["main.cc"],
    copts = SWARM_COPTS,
    linkopts = ["-pthread"],
    deps = [
        ":app",
        ":registry",
        ":scenario_loader",
    ],
)
