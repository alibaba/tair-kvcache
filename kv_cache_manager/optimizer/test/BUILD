package(default_visibility = ["//visibility:private"])

cc_test(
    name = "LruEvictionPolicyTest",
    srcs = [
        "lru_eviction_policy_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/optimizer/eviction_policy",
        "//kv_cache_manager/manager:cache_location",
        "//kv_cache_manager/data_storage:common_define",
    ],
)

cc_test(
    name = "RadixTreeIndexTest",
    srcs = [
        "radix_tree_index_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/optimizer/eviction_policy",
        "//kv_cache_manager/optimizer/index",
        "//kv_cache_manager/manager:cache_location",
        "//kv_cache_manager/data_storage:common_define",
    ],
)

cc_test(
    name = "OptEvictionManagerTest",
    srcs = [
        "eviction_manager_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/optimizer/eviction_policy",
        "//kv_cache_manager/optimizer/manager",
        "//kv_cache_manager/manager:cache_location",
        "//kv_cache_manager/data_storage:common_define",
    ],
)

cc_test(
    name = "OptIndexerManagerTest",
    srcs = [
        "indexer_manager_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/optimizer/eviction_policy",
        "//kv_cache_manager/optimizer/index",
        "//kv_cache_manager/optimizer/manager",
        "//kv_cache_manager/manager:cache_location",
        "//kv_cache_manager/data_storage:common_define",
    ],
)

cc_test(
    name = "OptimizerManagerTest",
    srcs = [
        "optimizer_manager_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/optimizer/manager",
        "//kv_cache_manager/manager:cache_location",
        "//kv_cache_manager/data_storage:common_define",
    ],
)

py_binary(
    name = "asan_runner",
    srcs = ["asan_runner.py"],
    visibility = ["//visibility:private"],
)

py_test(
    name = "optimizer_manager_py_test",
    size = "small",
    main = "asan_runner.py",
    srcs = [
        "asan_runner.py",
        "optimizer_manager_py_test.py",
    ],
    deps = [
        "//kv_cache_manager/optimizer/pybind:kvcm_py_optimizer_lib",
    ],
    data = ["testdata/optimizer_startup_config_load.json",
            "testdata/event_publisher.log",
            "lsan_suppressions.txt"],
    args = [
        "$(location lsan_suppressions.txt)",     # → argv[1]
        "$(location optimizer_manager_py_test.py)", # → argv[2]
        # 后续可加 --test_arg="--verbose" 等，会传给真实测试
    ],
)