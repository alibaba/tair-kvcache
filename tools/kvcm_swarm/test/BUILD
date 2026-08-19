package(default_visibility = ["//visibility:private"])

SWARM_TEST_COPTS = [
    "-std=c++20",
    "-fcoroutines",
]

cc_test(
    name = "runtime_test",
    size = "medium",
    srcs = ["runtime_test.cc"],
    copts = SWARM_TEST_COPTS,
    deps = [
        "//tools/kvcm_swarm:runtime",
        "//tools/kvcm_swarm:transport",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "workload_test",
    size = "small",
    srcs = ["workload_test.cc"],
    copts = SWARM_TEST_COPTS,
    deps = [
        "//tools/kvcm_swarm:v6d_model",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "local_cache_test",
    size = "small",
    srcs = ["local_cache_test.cc"],
    copts = SWARM_TEST_COPTS,
    deps = [
        "//tools/kvcm_swarm:v6d_model",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "expected_locations_test",
    size = "small",
    srcs = ["expected_locations_test.cc"],
    copts = SWARM_TEST_COPTS,
    deps = [
        "//tools/kvcm_swarm:v6d_model",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "checks_test",
    size = "small",
    srcs = ["checks_test.cc"],
    copts = SWARM_TEST_COPTS,
    deps = [
        "//tools/kvcm_swarm:v6d_behavior",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "v6d_protocol_test",
    size = "small",
    srcs = ["v6d_protocol_test.cc"],
    copts = SWARM_TEST_COPTS,
    deps = [
        "//tools/kvcm_swarm:protocol",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "loader_test",
    size = "small",
    srcs = ["loader_test.cc"],
    copts = SWARM_TEST_COPTS,
    deps = [
        "//tools/kvcm_swarm:scenario_loader",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fake_behavior",
    testonly = True,
    srcs = ["fake_behavior.cc"],
    hdrs = ["fake_behavior.h"],
    copts = SWARM_TEST_COPTS,
    deps = [
        "//tools/kvcm_swarm:behavior_api",
        "//tools/kvcm_swarm:scenario_config",
    ],
)

cc_test(
    name = "registry_test",
    size = "small",
    srcs = ["registry_test.cc"],
    copts = SWARM_TEST_COPTS,
    deps = [
        ":fake_behavior",
        "//tools/kvcm_swarm:registry",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "session_manager_test",
    size = "medium",
    srcs = ["session_manager_test.cc"],
    copts = SWARM_TEST_COPTS,
    deps = [
        "//tools/kvcm_swarm:v6d_behavior",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "report_test",
    size = "small",
    srcs = ["report_test.cc"],
    copts = SWARM_TEST_COPTS,
    deps = [
        "//tools/kvcm_swarm:app",
        "@com_google_googletest//:gtest_main",
        "@rapidjson",
    ],
)
