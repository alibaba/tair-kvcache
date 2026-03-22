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
    name = "coordination_backend_test_base",
    testonly = True,
    srcs = [],
    hdrs = ["coordination_backend_test_base.h"],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/config:coordination_backend",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "coordination_redis_backend_test",
    srcs = [
        "coordination_redis_backend_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    tags = [
        "manual",
        "redis",
    ],
    deps = [
        ":coordination_backend_test_base",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config:coordination_backend",
    ],
)

cc_test(
    name = "coordination_file_backend_test",
    srcs = [
        "coordination_file_backend_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        ":coordination_backend_test_base",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config:coordination_backend",
    ],
)

cc_test(
    name = "coordination_memory_backend_test",
    srcs = [
        "coordination_memory_backend_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        ":coordination_backend_test_base",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config:coordination_backend",
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
        "//kv_cache_manager/config:coordination_backend",
        "//kv_cache_manager/config:leader_elector_base",
    ],
)
