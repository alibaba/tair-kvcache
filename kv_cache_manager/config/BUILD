package(default_visibility = ["//visibility:public"])

cc_library(
    name = "leader_elector_base",
    srcs = [
        "leader_elector.cc",
    ],
    hdrs = [
        "leader_elector.h",
    ],
    deps = [
        ":distributed_lock_backend",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:loop_thread",
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
        "instance_info.cc",
        "meta_cache_policy_config.cc",
        "meta_indexer_config.cc",
        "meta_storage_backend_config.cc",
        "model_deployment.cc",
        "quota_config.cc",
        "registry_local_backend.cc",
        "registry_manager.cc",
        "registry_redis_backend.cc",
        "registry_storage_backend_factory.cc",
        "trigger_strategy.cc",
    ],
    hdrs = [
        "account.h",
        "cache_config.h",
        "cache_reclaim_strategy.h",
        "data_storage_strategy.h",
        "instance_group.h",
        "instance_group_quota.h",
        "instance_info.h",
        "meta_cache_policy_config.h",
        "meta_indexer_config.h",
        "meta_storage_backend_config.h",
        "model_deployment.h",
        "quota_config.h",
        "registry_local_backend.h",
        "registry_manager.h",
        "registry_redis_backend.h",
        "registry_storage_backend.h",
        "registry_storage_backend_factory.h",
        "trigger_strategy.h",
    ],
    deps = [
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
    name = "distributed_lock_backend",
    srcs = [
        "distributed_lock_backend_factory.cc",
        "distributed_lock_file_backend.cc",
        "distributed_lock_memory_backend.cc",
        "distributed_lock_redis_backend.cc",
    ],
    hdrs = [
        "distributed_lock_backend.h",
        "distributed_lock_backend_factory.h",
        "distributed_lock_file_backend.h",
        "distributed_lock_memory_backend.h",
        "distributed_lock_redis_backend.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:redis_client_ext",
        "//kv_cache_manager/common:standard_uri",
    ],
)
