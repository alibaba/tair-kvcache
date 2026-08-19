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
