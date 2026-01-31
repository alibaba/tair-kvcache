package(default_visibility = ["//visibility:public"])

cc_library(
    name = "data_storage",
    srcs = [
        "data_storage_manager.cc",
        "hf3fs_backend.cc",
        "mooncake_backend.cc",
        "nfs_backend.cc",
    ],
    hdrs = [
        "data_storage_backend.h",
        "data_storage_manager.h",
        "hf3fs_backend.h",
        "mooncake_backend.h",
        "nfs_backend.h",
    ],
    deps = [
        # 不一定用
        # "//3rdparty/3fs:hf3fs",
        "//3rdparty/mooncake:mooncake_client_c",
        "//kv_cache_manager/common/hash",  # for xxph3
        ":common_define",
        # Internal implementations
        "//stub_source/kv_cache_manager/data_storage:vcns_hf3fs",
        "//stub_source/kv_cache_manager/data_storage:tair_mempool",
    ],
)

cc_library(
    name = "common_define",
    srcs = [
        "storage_config.cc",
    ],
    hdrs = [
        "common_define.h",
        "data_storage_backend.h",
        "data_storage_uri.h",
        "storage_config.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:standard_uri",
    ],
)
