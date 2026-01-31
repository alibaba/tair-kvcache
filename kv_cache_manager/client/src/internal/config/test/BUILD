package(default_visibility = ["//visibility:private"])

cc_library(
    name = "config_test_util",
    hdrs = ["config_test_base.h"],
    deps = [
        "//kv_cache_manager/common:unittest",
    ],
    testonly = 1,
)

cc_test(
    name = "ClientConfigTest",
    srcs = [
        "client_config_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = glob(["testdata/**"]),
    deps = [
        ":config_test_util",
        "//kv_cache_manager/client/src/internal/config",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "SdkConfigTest",
    srcs = [
        "sdk_config_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = glob(["testdata/**"]),
    deps = [
        ":config_test_util",
        "//kv_cache_manager/client/src/internal/config",
        "//kv_cache_manager/common:unittest",
    ],
)
