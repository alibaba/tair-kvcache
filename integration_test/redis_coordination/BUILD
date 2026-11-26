package(default_visibility = ["//integration_test/redis_coordination:__subpackages__"])

cc_binary(
    name = "leader_lock_deletion_test_driver",
    testonly = True,
    srcs = ["leader_lock_deletion_test_driver.cc"],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:redis_client_ext",
        "//kv_cache_manager/common:standard_uri",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config:coordination_backend",
        "//kv_cache_manager/config:leader_elector_base",
    ],
)

py_test(
    name = "redis_leader_recovery_test",
    size = "large",
    srcs = [
        "redis_fault_proxy.py",
        "redis_leader_recovery_test.py",
    ],
    data = [
        ":leader_lock_deletion_test_driver",
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    main = "redis_leader_recovery_test.py",
    tags = [
        "manual",
        "no-remote-exec",
        "redis",
    ],
    deps = ["//integration_test/testlib:test_base"],
)

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
