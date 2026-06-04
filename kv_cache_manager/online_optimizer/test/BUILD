package(default_visibility = ["//visibility:private"])

cc_test(
    name = "FenwickCacheIndexerTest",
    srcs = [
        "fenwick_cache_indexer_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/online_optimizer:cache_indexer",
    ],
)

cc_test(
    name = "BSTCacheIndexerTest",
    srcs = [
        "bst_cache_indexer_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/online_optimizer:cache_indexer",
    ],
)

cc_test(
    name = "LruCacheIndexerTest",
    srcs = [
        "lru_cache_indexer_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/online_optimizer:cache_indexer",
    ],
)

cc_test(
    name = "TtlCacheIndexerWrapperTest",
    srcs = [
        "ttl_cache_indexer_wrapper_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/online_optimizer:cache_indexer",
        "//kv_cache_manager/online_optimizer:cache_indexer_factory",
        "//kv_cache_manager/online_optimizer:ttl_cache_indexer_wrapper",
    ],
)

cc_test(
    name = "OnlineOptimizerManagerTest",
    srcs = [
        "online_optimizer_manager_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/online_optimizer:online_optimizer_manager",
        "//kv_cache_manager/online_optimizer:optimizer_registry_manager",
    ],
)

cc_test(
    name = "OptimizerConfigTest",
    srcs = [
        "optimizer_config_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/online_optimizer:optimizer_instance_group",
        "//kv_cache_manager/online_optimizer:optimizer_instance_info",
    ],
)

cc_test(
    name = "OptimizerServiceImplTest",
    srcs = [
        "optimizer_service_impl_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/online_optimizer:online_optimizer_manager",
        "//kv_cache_manager/online_optimizer:optimizer_registry_manager",
        "//kv_cache_manager/online_optimizer:optimizer_service_impl",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
    ],
)

cc_test(
    name = "OptimizerMetricsReporterTest",
    srcs = [
        "optimizer_metrics_reporter_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/metrics:metrics_registry",
        "//kv_cache_manager/online_optimizer:online_optimizer_manager",
        "//kv_cache_manager/online_optimizer:optimizer_metrics_collector",
        "//kv_cache_manager/online_optimizer:optimizer_registry_manager",
        "//kv_cache_manager/online_optimizer:optimizer_metrics_reporter",
    ],
)

cc_test(
    name = "OnlineOptimizerServerConfigTest",
    srcs = [
        "online_optimizer_server_config_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:env_util",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/online_optimizer:online_optimizer_server_config",
    ],
)
