package(default_visibility = ["//visibility:public"])

cc_library(
    name = "vcns_hf3fs",
    srcs = [
        "vcns_hf3fs_backend.cc",
    ],
    hdrs = [
        "vcns_hf3fs_backend.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/data_storage:common_define",
    ],
)

cc_library(
    name = "tair_mempool",
    srcs = [
        "tair_mempool_backend.cc",
    ],
    hdrs = [
        "tair_mempool_backend.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/data_storage:common_define",
    ],
)
