package(default_visibility = ["//visibility:private"])

cc_test(
    name = "RegistryManagerTest",
    srcs = [
        "registry_manager_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config",
    ],
)

cc_test(
    name = "meta_indexer_config_test",
    srcs = [
        "meta_indexer_config_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config",
    ],
)

cc_test(
    name = "registry_manager_redis_backend_test",
    srcs = [
        "registry_manager_redis_backend_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    tags = [
        "manual",
        "redis",
    ],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config",
    ],
)

cc_test(
    name = "registry_manager_local_backend_test",
    srcs = [
        "registry_manager_local_backend_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config",
    ],
)

cc_test(
    name = "instance_info_test",
    srcs = [
        "instance_info_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config",
    ],
)

cc_library(
    name = "distributed_lock_backend_test_base",
    testonly = True,
    srcs = [],
    hdrs = ["distributed_lock_backend_test_base.h"],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/config:distributed_lock_backend",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "distributed_lock_redis_backend_test",
    srcs = [
        "distributed_lock_redis_backend_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    tags = [
        "manual",
        "redis",
    ],
    deps = [
        ":distributed_lock_backend_test_base",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config:distributed_lock_backend",
    ],
)

cc_test(
    name = "distributed_lock_file_backend_test",
    srcs = [
        "distributed_lock_file_backend_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        ":distributed_lock_backend_test_base",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config:distributed_lock_backend",
    ],
)

cc_test(
    name = "distributed_lock_memory_backend_test",
    srcs = [
        "distributed_lock_memory_backend_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        ":distributed_lock_backend_test_base",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config:distributed_lock_backend",
    ],
)

cc_test(
    name = "leader_elector_test",
    srcs = [
        "leader_elector_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config:distributed_lock_backend",
        "//kv_cache_manager/config:leader_elector_base",
    ],
)
