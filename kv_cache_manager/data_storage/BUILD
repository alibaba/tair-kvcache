package(default_visibility = ["//visibility:public"])

config_setting(
    name = "enable_mooncake",
    define_values = {"ENABLE_MOONCAKE": "true"},
)

config_setting(
    name = "enable_vcns",
    define_values = {"ENABLE_VCNS": "true"},
)

cc_library(
    name = "data_storage",
    srcs = [
        "data_storage_manager.cc",
        "dummy_backend.cc",
        "event_report_backend.cc",
        "hf3fs_backend.cc",
        "nfs_backend.cc",
    ] + select({
        ":enable_mooncake": ["mooncake_backend.cc"],
        "//conditions:default": [],
    }),
    hdrs = [
        "data_storage_backend.h",
        "data_storage_manager.h",
        "dummy_backend.h",
        "event_report_backend.h",
        "hf3fs_backend.h",
        "nfs_backend.h",
        "snapshot_uri_utils.h",
    ] + select({
        ":enable_mooncake": ["mooncake_backend.h"],
        "//conditions:default": [],
    }),
    defines = select({
        ":enable_mooncake": ["ENABLE_MOONCAKE=1"],
        "//conditions:default": [],
    }) + select({
        ":enable_vcns": ["ENABLE_VCNS=1"],
        "//conditions:default": [],
    }),
    deps = [
        # 不一定用
        # "//3rdparty/3fs:hf3fs",
        "//kv_cache_manager/common/hash",  # for xxph3
        ":common_define",
        # Internal implementations
        "//stub_source/kv_cache_manager/data_storage:tair_mempool",
    ] + select({
        ":enable_mooncake": ["//3rdparty/mooncake:mooncake_client_c"],
        "//conditions:default": [],
    }) + select({
        ":enable_vcns": ["//stub_source/kv_cache_manager/data_storage:vcns_hf3fs"],
        "//conditions:default": [],
    }),
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
        "//kv_cache_manager/common:service_discovery",
        "//kv_cache_manager/common:standard_uri",
    ],
)
