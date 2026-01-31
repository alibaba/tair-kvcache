package(default_visibility = ["//visibility:private"])

cc_test(
    name = "CacheReclaimerTest",
    srcs = [
        "cache_reclaimer_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    linkstatic = True,
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/manager:cache_reclaimer",
        "@cpp_stub",
    ],
)

cc_test(
    name = "DataStorageSelectorTest",
    srcs = [
        "data_storage_selector_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    linkstatic = True,
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/manager:data_storage_selector",
        "@cpp_stub",
    ],
)

cc_test(
    name = "SchedulePlanExecutorTest",
    srcs = [
        "schedule_plan_executor_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/manager:schedule_plan_executor",
    ],
)

cc_test(
    name = "MetaSearcherTest",
    srcs = [
        "meta_searcher_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/manager:cache_location",
        "//kv_cache_manager/manager:meta_searcher",
        "//kv_cache_manager/meta",
    ],
)

cc_test(
    name = "MetaSearcherManagerTest",
    srcs = [
        "meta_searcher_manager_test.cc",
    ],
    copts = [
        "-fno-access-control",
    ],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/manager:meta_searcher",
    ],
)

cc_test(
    name = "CacheManagerTest",
    srcs = [
        "cache_manager_test.cc",
    ],
    copts = [
        "-fno-access-control",
    ],
    data = [],
    shard_count = 10,
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/manager:cache_manager",
    ],
)

cc_test(
    name = "StartupConfigLoaderTest",
    srcs = [
        "startup_config_loader_test.cc",
    ],
    copts = [
        "-fno-access-control",
    ],
    data = [
        "testdata/test_startup_config_load.json",
    ],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/manager:cache_manager",
    ],
)

cc_test(
    name = "MetaSearcherRedisTest",
    srcs = [
        "meta_searcher_redis_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    tags = [
        "manual",
        "redis",
    ],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/manager:cache_location",
        "//kv_cache_manager/manager:meta_searcher",
        "//kv_cache_manager/meta",
    ],
)

cc_test(
    name = "WriteLocationManagerTest",
    srcs = [
        "write_location_manager_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    shard_count = 3,
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/manager:write_location_manager",
    ],
)

cc_test(
    name = "SelectLocationPolicyTest",
    srcs = [
        "selection_location_policy_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:string_util",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/data_storage:common_define",
        "//kv_cache_manager/manager:select_location_policy",
    ],
)
