package(default_visibility = ["//visibility:private"])

cc_test(
    name = "EnvUtilTest",
    srcs = [
        "env_util_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:env_util",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "LoggerTest",
    srcs = [
        "logger_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "JsonizableTest",
    srcs = [
        "jsonizable_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:jsonizable",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "LruCacheTest",
    srcs = [
        "lru_cache_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:lru_cache",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "RequestContextTest",
    srcs = [
        "request_context_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:request_context",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "LoopThreadTest",
    srcs = [
        "loop_thread_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:loop_thread",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "StandardUriTest",
    srcs = [
        "standard_uri_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/common:standard_uri",
    ],
)

cc_library(
    name = "redis_client_testlib",
    srcs = [
        "redis_test_base.cc",
    ],
    hdrs = [
        "mock_redis_client.h",
        "redis_test_base.h",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:redis_client",
        "@com_google_googletest//:gtest",
    ],
    visibility = ["//kv_cache_manager/meta/test:__pkg__"],
)

cc_test(
    name = "redis_client_test",
    srcs = [
        "redis_client_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        ":redis_client_testlib",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "redis_client_real_service_test",
    srcs = [
        "redis_client_real_service_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:redis_client",
        "//kv_cache_manager/common:unittest",
    ],
    tags = [
        "manual",
        "redis",
    ],
)
