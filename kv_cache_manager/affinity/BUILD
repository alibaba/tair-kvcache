package(default_visibility = ["//visibility:public"])

cc_library(
    name = "node_metrics",
    hdrs = ["node_metrics.h"],
)

cc_library(
    name = "affinity",
    srcs = ["cache_affinity_manager.cc"],
    hdrs = ["cache_affinity_manager.h"],
    deps = [
        ":affinity_strategy",
        ":frequency_sketch",
        ":hint_suppressor",
        ":node_metrics",
        ":strategy_factory",
        "//kv_cache_manager/affinity/pipeline:candidate_pipeline",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:affinity_types",
        # Needed for metrics pull loop (SnapshotPerNodeMetrics)
        "//kv_cache_manager/data_storage",
        "//kv_cache_manager/data_storage:common_define",
    ],
)

# AffinityStrategy abstract interface (header-only) + NoopStrategy fallback.
cc_library(
    name = "affinity_strategy",
    hdrs = [
        "affinity_strategy.h",
        "noop_strategy.h",
    ],
    deps = [
        ":node_metrics",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:affinity_types",
        "//kv_cache_manager/data_storage:common_define",
        "//kv_cache_manager/meta:cache_location",
    ],
)

# Per-(caller, key) frequency sketch for read-triggered replication.
cc_library(
    name = "frequency_sketch",
    srcs = ["frequency_sketch.cc"],
    hdrs = ["frequency_sketch.h"],
)

cc_library(
    name = "hint_suppressor",
    srcs = ["hint_suppressor.cc"],
    hdrs = ["hint_suppressor.h"],
    deps = [
        "//kv_cache_manager/common",
    ],
)

# LocalReplicaAffinityStrategy: multi-replica strategy implementing
# ResolveWrite / ResolveRead / ResolveEviction.
cc_library(
    name = "local_replica_strategy",
    srcs = ["local_replica_strategy.cc"],
    hdrs = ["local_replica_strategy.h"],
    deps = [
        ":affinity_strategy",
        ":frequency_sketch",  # ResolveRead 内部需要 FrequencySketch 完整类型
        ":hint_suppressor",
        "//kv_cache_manager/affinity/pipeline:candidate_pipeline",  # v0 5 段算法作 candidate_pipeline 实现
    ],
)

# StrategyFactory: parse JSON config into an AffinityStrategy instance.
cc_library(
    name = "strategy_factory",
    srcs = ["strategy_factory.cc"],
    hdrs = ["strategy_factory.h"],
    deps = [
        ":affinity_strategy",
        ":hint_suppressor",
        ":local_replica_strategy",
        "//kv_cache_manager/affinity/pipeline:candidate_pipeline",  # 解析 5 段流水线
        "//kv_cache_manager/common",
        "@rapidjson",
    ],
)
