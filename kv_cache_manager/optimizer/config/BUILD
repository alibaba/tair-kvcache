package(default_visibility = ["//visibility:public"])

cc_library(
    name = "config",
    srcs = [
        "types.cc",
        "optimizer_config.cc",
        "tier_config.cc",
        "eviction_config.cc",
        "replay_instance_config.cc",
        "replay_instance_group_config.cc",
        "optimizer_config_loader.cc",
        "optimizer_instance_group.cc",
        "optimizer_instance_info.cc",
    ],
    hdrs = [
        "types.h",
        "optimizer_config.h",
        "tier_config.h",
        "eviction_config.h",
        "replay_instance_config.h",
        "replay_instance_group_config.h",
        "insight_simulator_types.h",
        "optimizer_config_loader.h",
        "optimizer_instance_info.h",
        "optimizer_instance_group.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:jsonizable",
        "//kv_cache_manager/config:instance_info",
        "//kv_cache_manager/data_storage",
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
        ":config",
        "//kv_cache_manager/common",
        "//kv_cache_manager/config:registry_backend",
    ],
)
