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
    name = "storage_usage_data",
    srcs = [
        "storage_usage_data.cc",
    ],
    hdrs = [
        "storage_usage_data.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:string_util",
        "//kv_cache_manager/data_storage:common_define",
    ],
)

cc_library(
    name = "meta_indexer",
    srcs = [
        "meta_indexer.cc",
        "meta_storage_backend_factory.cc",
    ],
    hdrs = [
        "common.h",
        "meta_indexer.h",
        "meta_search_cache.h",
        "meta_storage_backend_factory.h",
    ],
    deps = [
        ":cache_location",
        ":common",
        ":meta_search_cache",
        ":meta_storage_backend_manager",
        ":storage_usage_data",
        ":types",
        ":utils",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:utils",
        "//kv_cache_manager/config",
    ],
)

cc_library(
    name = "meta_storage_backend_manager",
    srcs = [
        "meta_storage_backend_factory.cc",
        "meta_storage_backend_manager.cc",
    ],
    hdrs = [
        "meta_storage_backend_factory.h",
        "meta_storage_backend_manager.h",
    ],
    deps = [
        ":cache_location",
        ":common",
        ":meta_async_redis_backend",
        ":meta_dummy_backend",
        ":meta_local_backend",
        ":meta_redis_backend",
        ":meta_storage_backend",
        ":types",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:standard_uri",
        "//kv_cache_manager/common:timestamp_util",
        "//kv_cache_manager/config",
        "//kv_cache_manager/metrics:metrics_collector",
    ],
)

cc_library(
    name = "types",
    hdrs = [
        "types.h",
    ],
    deps = [
        ":cache_location",
        "//kv_cache_manager/common",
    ],
)

cc_library(
    name = "utils",
    hdrs = [
        "utils.h",
    ],
    deps = [
        ":types",
        "//kv_cache_manager/common/hash",
    ],
)

cc_library(
    name = "cache_location",
    srcs = [
        "cache_location.cc",
    ],
    hdrs = [
        "cache_location.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:string_util",
        "//kv_cache_manager/data_storage:common_define",
    ],
)

cc_library(
    name = "meta_search_cache",
    srcs = [
        "meta_search_cache.cc",
    ],
    hdrs = [
        "meta_search_cache.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:lru_cache",
        "//kv_cache_manager/config",
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
        ":types",
        "//kv_cache_manager/common",
        "//kv_cache_manager/config",
    ],
)

cc_library(
    name = "meta_dummy_backend",
    srcs = [
        "meta_dummy_backend.cc",
    ],
    hdrs = [
        "meta_dummy_backend.h",
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
    name = "meta_cache_base_backend",
    srcs = [
    ],
    hdrs = [
        "meta_cache_base_backend.h",
    ],
    deps = [
        ":meta_storage_backend",
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
        ":meta_cache_base_backend",
        ":meta_search_cache",
        "//kv_cache_manager/common:lru_cache",
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
        ":common",
        ":meta_storage_backend",
        "//kv_cache_manager/common:client_pool",
        "//kv_cache_manager/common:redis_client",
    ],
)

cc_library(
    name = "mpsc_write_queue",
    srcs = [
        "mpsc_write_queue.cc",
    ],
    hdrs = [
        "mpsc_write_queue.h",
    ],
    deps = [
        ":types",
    ],
)

cc_library(
    name = "meta_async_redis_backend",
    srcs = [
        "meta_async_redis_backend.cc",
    ],
    hdrs = [
        "meta_async_redis_backend.h",
    ],
    deps = [
        ":common",
        ":meta_storage_backend",
        ":mpsc_write_queue",
        ":utils",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:client_pool",
        "//kv_cache_manager/common:redis_client",
        "//kv_cache_manager/common:standard_uri",
        "//kv_cache_manager/common:string_util",
        "//kv_cache_manager/config",
    ],
)

