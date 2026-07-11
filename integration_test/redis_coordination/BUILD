package(default_visibility = ["//integration_test/redis_coordination:__subpackages__"])

py_test(
    name = "redis_cluster_isolation_test",
    srcs = ["redis_cluster_isolation_test.py"],
    data = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    tags = [
        "manual",
        "no-remote-exec",
        "redis",
    ],
    deps = [
        "//integration_test/testlib:test_base",
    ],
)
