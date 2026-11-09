package(default_visibility = ["//visibility:public"])

cc_library(
    name = "simple_kv_cache_item",
    srcs = ["simple_kv_cache_item.cc"],
    hdrs = ["simple_kv_cache_item.h"],
    deps = [
        "//kv_cache_manager/common:lru_cache",
        "//kv_cache_manager/common:standard_uri",
        "//kv_cache_manager/data_storage:snapshot_uri_utils",
        "//kv_cache_manager/meta:types",
    ],
)

cc_library(
    name = "simple_kv_cache_backend",
    srcs = ["simple_kv_cache_backend.cc"],
    hdrs = ["simple_kv_cache_backend.h"],
    deps = [
        ":simple_kv_cache_item",
        "//kv_cache_manager/common:lru_cache",
        "//kv_cache_manager/common:standard_uri",
        "//kv_cache_manager/config",
        "//kv_cache_manager/meta:common",
        "//kv_cache_manager/meta:meta_cache_base_backend",
    ],
)
