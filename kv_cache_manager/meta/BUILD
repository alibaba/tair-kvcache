package(default_visibility = ["//visibility:public"])

cc_library(
    name = "meta",
    deps = [
        ":meta_indexer_manager",
    ],
)

cc_library(
    name = "meta_indexer_manager",
    srcs = [
        "meta_indexer_manager.cc",
    ],
    hdrs = [
        "meta_indexer_manager.h",
    ],
    deps = [
        ":meta_indexer",
    ],
)

cc_library(
    name = "meta_indexer",
    srcs = [
        "meta_indexer.cc",
        "meta_search_cache.cc",
        "meta_storage_backend_factory.cc",
    ],
    hdrs = [
        "common.h",
        "meta_indexer.h",
        "meta_search_cache.h",
        "meta_storage_backend_factory.h",
    ],
    deps = [
        ":common",
        ":meta_local_backend",
        ":meta_redis_backend",
        ":meta_storage_backend",
        "//kv_cache_manager/config",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:lru_cache",
        "//kv_cache_manager/common:utils",
    ],
)

cc_library(
    name = "common",
    srcs = [
    ],
    hdrs = [
        "common.h",
    ],
    deps = [
    ],
)


cc_library(
    name = "meta_storage_backend",
    srcs = [
    ],
    hdrs = [
        "meta_storage_backend.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/config",
    ],
)

cc_library(
    name = "meta_local_backend",
    srcs = [
        "meta_local_backend.cc",
    ],
    hdrs = [
        "meta_local_backend.h",
    ],
    deps = [
        ":common",
        ":meta_storage_backend",
        "//kv_cache_manager/common:concurrent_hash_map",
        "//kv_cache_manager/common:standard_uri",
        "//kv_cache_manager/common:string_util",
    ],
)

cc_library(
    name = "meta_redis_backend",
    srcs = [
        "meta_redis_backend.cc",
    ],
    hdrs = [
        "meta_redis_backend.h",
    ],
    deps = [
        "//kv_cache_manager/common:redis_client",
        ":common",
        ":meta_storage_backend",
    ],
)

