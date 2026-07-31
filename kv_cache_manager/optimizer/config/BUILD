package(default_visibility = ["//visibility:public"])

cc_library(
    name = "online_optimizer_config",
    srcs = [
        "optimizer_instance_group.cc",
        "optimizer_instance_info.cc",
    ],
    hdrs = [
        "optimizer_instance_group.h",
        "optimizer_instance_info.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:jsonizable",
        "//kv_cache_manager/config:instance_info",
    ],
)

cc_library(
    name = "replay_optimizer_config",
    srcs = [
        "types.cc",
        "optimizer_config.cc",
        "hierarchical_replay_config.cc",
        "hierarchical_replay_config_loader.cc",
        "tier_config.cc",
        "eviction_config.cc",
        "replay_instance_config.cc",
        "replay_instance_group_config.cc",
        "optimizer_config_loader.cc",
    ],
    hdrs = [
        "types.h",
        "optimizer_config.h",
        "hierarchical_replay_config.h",
        "hierarchical_replay_config_loader.h",
        "tier_config.h",
        "eviction_config.h",
        "replay_instance_config.h",
        "replay_instance_group_config.h",
        "insight_simulator_types.h",
        "optimizer_config_loader.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:jsonizable",
        "//kv_cache_manager/data_storage",
        "//kv_cache_manager/optimizer/scheduler:scheduling_strategy",
    ],
)

cc_library(
    name = "optimizer_lite_hit_config",
    srcs = [
        "optimizer_lite_hit_config.cc",
    ],
    hdrs = [
        "optimizer_lite_hit_config.h",
    ],
    deps = [
        ":online_optimizer_config",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:jsonizable",
    ],
)

cc_library(
    name = "config",
    deps = [
        ":online_optimizer_config",
        ":replay_optimizer_config",
    ],
)

cc_library(
    name = "optimizer_registry_manager",
    srcs = [
        "optimizer_registry_manager.cc",
    ],
    hdrs = [
        "optimizer_registry_manager.h",
    ],
    deps = [
        ":online_optimizer_config",
        "//kv_cache_manager/common",
        "//kv_cache_manager/config:registry_backend",
    ],
)
