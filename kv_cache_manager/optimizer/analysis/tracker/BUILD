package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tracker",
    srcs = [
        "block_lifecycle_tracker.cc",
        "hit_rate_tracker.cc",
    ],
    hdrs = [
        "block_lifecycle_tracker.h",
        "hit_rate_tracker.h",
    ],
    deps = [
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/optimizer/analysis",
        "//kv_cache_manager/optimizer/config",
    ],
)
