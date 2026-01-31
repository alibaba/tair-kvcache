package(default_visibility = ["//visibility:public"])

cc_library(
    name = "eviction_policy",
    srcs = [
        "lru.cc",
        "random_lru.cc",
        "leaf_aware_lru.cc",
        "policy_factory.cc",
    ],
    hdrs = [
        "base.h",
        "common_structure.h",
        "policy_factory.h",
        "lru.h",
        "random_lru.h",
        "leaf_aware_lru.h",
    ],
    deps = [
        "//kv_cache_manager/manager:cache_location",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/common:logger",
    ],
)
