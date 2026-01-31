package(default_visibility = ["//visibility:private"])

cc_test(
    name = "DataStorageManagerTest",
    srcs = [
        "data_storage_manager_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/data_storage",
    ],
)

cc_test(
    name = "NfsBackendTest",
    srcs = [
        "nfs_backend_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/data_storage",
    ],
)

cc_test(
    name = "Hf3fsBackendTest",
    srcs = [
        "hf3fs_backend_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/data_storage",
    ],
)

cc_test(
    name = "StorageConfigTest",
    srcs = [
        "storage_config_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/data_storage",
    ],
)
