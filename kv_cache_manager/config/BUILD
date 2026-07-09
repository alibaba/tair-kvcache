package(default_visibility = ["//visibility:public"])

cc_library(
    name = "leader_elector_base",
    srcs = [
        "leader_elector.cc",
    ],
    hdrs = [
        "leader_elector.h",
        "node_endpoint_info.h",
    ],
    deps = [
        ":coordination_backend",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:loop_thread",
    ],
)

cc_library(
    name = "instance_info",
    srcs = [
        "instance_info.cc",
        "model_deployment.cc",
    ],
    hdrs = [
        "instance_info.h",
        "model_deployment.h",
    ],
    deps = [
        "//kv_cache_manager/common",
    ],
)

cc_library(
    name = "registry_backend",
    srcs = [
        "registry_local_backend.cc",
        "registry_redis_backend.cc",
        "registry_storage_backend_factory.cc",
    ],
    hdrs = [
        "registry_local_backend.h",
        "registry_redis_backend.h",
        "registry_storage_backend.h",
        "registry_storage_backend_factory.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:concurrent_hash_map",
        "//kv_cache_manager/common:redis_client",
    ],
)

cc_library(
    name = "config",
    srcs = [
        "cache_config.cc",
        "cache_reclaim_strategy.cc",
        "data_storage_strategy.cc",
        "instance_group.cc",
        "instance_group_quota.cc",
        "meta_cache_policy_config.cc",
        "meta_indexer_config.cc",
        "meta_storage_backend_config.cc",
        "quota_config.cc",
        "registry_manager.cc",
        "trigger_strategy.cc",
    ],
    hdrs = [
        "account.h",
        "cache_config.h",
        "cache_reclaim_strategy.h",
        "data_storage_strategy.h",
        "instance_group.h",
        "instance_group_quota.h",
        "meta_cache_policy_config.h",
        "meta_indexer_config.h",
        "meta_storage_backend_config.h",
        "quota_config.h",
        "registry_manager.h",
        "trigger_strategy.h",
    ],
    deps = [
        ":instance_info",
        ":registry_backend",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:concurrent_hash_map",
        "//kv_cache_manager/common:redis_client",
        "//kv_cache_manager/data_storage",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
    ],
)

cc_library(
    name = "meta_cache_policy_config",
    srcs = ["meta_cache_policy_config.cc"],
    hdrs = ["meta_cache_policy_config.h"],
    deps = [
        "//kv_cache_manager/common",
    ],
)

cc_library(
    name = "coordination_backend",
    srcs = [
        "coordination_backend_factory.cc",
        "coordination_file_backend.cc",
        "coordination_memory_backend.cc",
        "coordination_redis_backend.cc",
    ],
    hdrs = [
        "coordination_backend.h",
        "coordination_backend_factory.h",
        "coordination_file_backend.h",
        "coordination_memory_backend.h",
        "coordination_redis_backend.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:client_pool",
        "//kv_cache_manager/common:redis_client_ext",
        "//kv_cache_manager/common:standard_uri",
    ],
)
