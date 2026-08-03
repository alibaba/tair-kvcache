package(default_visibility = ["//visibility:public"])

cc_library(
    name = "redis_executor",
    hdrs = [
        "redis_executor.h",
    ],
    deps = [
        "@redis_plus_plus//:redis_plus_plus",
    ],
)

cc_library(
    name = "hiredis_executor",
    srcs = [
        "hiredis_executor.cc",
    ],
    hdrs = [
        "hiredis_executor.h",
    ],
    deps = [
        ":redis_executor",
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common:standard_uri",
        "@hiredis",
    ],
)

cc_library(
    name = "redis_client",
    srcs = [
        "redis_client.cc",
    ],
    hdrs = [
        "redis_client.h",
    ],
    deps = [
        ":redis_executor",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:standard_uri",
        "@hiredis",
    ],
)

cc_library(
    name = "redis_cluster_executor",
    srcs = [
        "redis_cluster_executor.cc",
    ],
    hdrs = [
        "redis_cluster_executor.h",
    ],
    deps = [
        ":redis_executor",
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common:standard_uri",
        "@hiredis",
        "@redis_plus_plus//:redis_plus_plus",
    ],
)

cc_library(
    name = "redis_client_factory",
    srcs = [
        "redis_client_factory.cc",
    ],
    hdrs = [
        "redis_client_factory.h",
    ],
    deps = [
        ":hiredis_executor",
        ":redis_client",
        ":redis_cluster_executor",
        "//kv_cache_manager/common:logger",
    ],
)
