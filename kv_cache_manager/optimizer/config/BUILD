package(default_visibility = ["//visibility:public"])

cc_library(
    name = "config",
    srcs = [
        "types.cc",
        "optimizer_config.cc",
        "tier_config.cc",
        "eviction_config.cc",
        "instance_config.cc",
        "instance_group_config.cc",
        "optimizer_config_loader.cc",
    ],
    hdrs = [
        "types.h",
        "optimizer_config.h",
        "tier_config.h",
        "eviction_config.h",
        "instance_config.h",
        "instance_group_config.h",
        "insight_simulator_types.h",
        "optimizer_config_loader.h",
    ],
    deps = [
        "//kv_cache_manager/common:jsonizable",
        "//kv_cache_manager/data_storage",
    ],
)
