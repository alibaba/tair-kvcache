package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cache",
    srcs = [
        "cache.cc",
        "sharded_cache.cc",
        "lru_cache.cc",
    ],
    hdrs = [
        "cache.h",
        "advanced_cache.h",
        "sharded_cache.h",
        "lru_cache.h",
    ],
    deps = [
        ":util",
        "//kv_cache_manager/common/hash:hash",
        "//kv_cache_manager/common",
    ],
)

cc_library(
    name = "util",
    srcs = [
    ],
    hdrs = [
        "auto_vector.h",
        "compression_type.h",
        "data_structure.h",
        "math.h",
        "port_posix.h",
    ],
    deps = [
    ],
)
