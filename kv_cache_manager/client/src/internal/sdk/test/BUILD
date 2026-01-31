package(default_visibility = ["//visibility:private"])

config_setting(
    name = "enable_hf3fs",
    values = {"define": "ENABLE_HF3FS=true"},
)

cc_test(
    name = "LocalFileSdkTest",
    srcs = [
        "local_file_sdk_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/client/src/internal/sdk",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "MooncakeSdkTest",
    srcs = [
        "mooncake_sdk_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    tags = ["manual"],
    deps = [
        "//kv_cache_manager/client/src/internal/sdk",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "SdkWrapperTest",
    srcs = [
        "sdk_wrapper_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/client/src/internal/config",
        "//kv_cache_manager/client/src/internal/sdk",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/data_storage",
    ],
)

cc_test(
    name = "hf3fs_sdk_test",
    srcs = select({
        ":enable_hf3fs": [
            "hf3fs_sdk_test.cc",
        ],
        "//conditions:default": [
            "hf3fs_unittest_empty.cc",
        ],
    }),
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/client/src/internal/sdk",
        "//kv_cache_manager/client/src/internal/sdk/test/mock:mock_hf3fs",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "hf3fs_usrbio_client_test",
    srcs = select({
        ":enable_hf3fs": [
            "hf3fs_usrbio_client_test.cc",
        ],
        "//conditions:default": [
            "hf3fs_unittest_empty.cc",
        ],
    }),
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/client/src/internal/sdk",
        "//kv_cache_manager/client/src/internal/sdk/test/mock:mock_hf3fs",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "hf3fs_mempool_test",
    srcs = select({
        ":enable_hf3fs": [
            "hf3fs_mempool_test.cc",
        ],
        "//conditions:default": [
            "hf3fs_unittest_empty.cc",
        ],
    }),
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/client/src/internal/sdk",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "SdkBufferCheckUtilTest",
    srcs = [
        "sdk_buffer_check_util_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/client/src/internal/sdk:sdk_buffer_check_util",
        "//kv_cache_manager/common:unittest",
    ],
)
