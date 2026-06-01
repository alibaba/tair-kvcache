package(default_visibility = ["//visibility:private"])

cc_test(
    name = "AffinityTest",
    srcs = [
        "affinity_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/affinity",
        "//kv_cache_manager/affinity:node_metrics",
        "//kv_cache_manager/affinity/pipeline:filter_cond",
        "//kv_cache_manager/affinity/pipeline:candidate_pipeline",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/data_storage:common_define",
        "@rapidjson",
    ],
)

cc_test(
    name = "StrategyFrameworkTest",
    srcs = ["strategy_framework_test.cc"],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/affinity:affinity_strategy",
        "//kv_cache_manager/affinity:strategy_factory",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "LocalReplicaStrategyTest",
    srcs = ["local_replica_strategy_test.cc"],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/affinity:frequency_sketch",
        "//kv_cache_manager/affinity:local_replica_strategy",
        "//kv_cache_manager/affinity/pipeline:candidate_pipeline",
        "//kv_cache_manager/common:unittest",
    ],
)


cc_test(
    name = "CacheAffinityManagerStrategyTest",
    srcs = ["cache_affinity_manager_strategy_test.cc"],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/affinity",
        "//kv_cache_manager/affinity:affinity_strategy",
        "//kv_cache_manager/affinity:local_replica_strategy",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "FrequencySketchTest",
    srcs = ["frequency_sketch_test.cc"],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/affinity:frequency_sketch",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "CacheAffinityManagerIntegrationTest",
    srcs = ["cache_affinity_manager_integration_test.cc"],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/affinity",
        "//kv_cache_manager/affinity:local_replica_strategy",
        "//kv_cache_manager/affinity:node_metrics",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/data_storage:common_define",
    ],
)

cc_test(
    name = "AffinityProbeTest",
    srcs = ["probe_test.cc"],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/affinity",
        "//kv_cache_manager/affinity:local_replica_strategy",
        "//kv_cache_manager/affinity:node_metrics",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/data_storage:common_define",
    ],
)
