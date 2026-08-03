package(default_visibility = ["//visibility:private"])

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
    visibility = ["//kv_cache_manager/meta/test:__pkg__"],
    deps = [
        "//kv_cache_manager/common/redis:redis_client",
        "@com_google_googletest//:gtest",
    ],
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
    name = "redis_client_randomkey_batch_num_test",
    srcs = [
        "redis_client_randomkey_batch_num_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        ":redis_client_testlib",
        "//kv_cache_manager/common/redis:redis_client_factory",
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
    tags = [
        "manual",
        "redis",
    ],
    deps = [
        "//kv_cache_manager/common/redis:redis_client_factory",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "RedisClusterExecutorTest",
    srcs = [
        "redis_cluster_executor_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common/redis:redis_cluster_executor",
        "//kv_cache_manager/common:unittest",
    ],
)
