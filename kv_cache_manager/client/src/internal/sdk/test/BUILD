package(default_visibility = ["//visibility:private"])

config_setting(
    name = "enable_hf3fs",
    values = {"define": "ENABLE_HF3FS=true"},
)

config_setting(
    name = "using_cuda",
    values = {"define": "using_cuda=true"},
)

config_setting(
    name = "using_musa",
    values = {"define": "using_musa=true"},
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
    name = "SdkFactoryTest",
    srcs = ["sdk_factory_test.cc"],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/client/src/internal/sdk",
        "//kv_cache_manager/common:unittest",
    ],
)

cc_test(
    name = "SdkTimeoutContractTest",
    srcs = [
        "fake_slow_sdk.h",
        "sdk_timeout_contract_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    # 用例含 sleep（如 500ms 哨兵检查、3s 慢 fake），用 medium 避免小目标超时。
    size = "medium",
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
    # 同 hf3fs_usrbio_client_test：超时/失败路径的设计内泄漏豁免。
    env = select({
        ":enable_hf3fs": {"LSAN_OPTIONS": "suppressions=$(location lsan_suppressions.txt)"},
        "//conditions:default": {},
    }),
    data = ["lsan_suppressions.txt"],
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
    # asan 下，超时/失败路径的 iov/ior、fd 注册按设计泄漏（提交事实判定），
    # LSAN 按该文件豁免，其余泄漏照常报错。
    env = select({
        ":enable_hf3fs": {"LSAN_OPTIONS": "suppressions=$(location lsan_suppressions.txt)"},
        "//conditions:default": {},
    }),
    data = ["lsan_suppressions.txt"],
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
    target_compatible_with = select({
        ":using_cuda": [],
        ":using_musa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//kv_cache_manager/common:unittest",
    ] + select({
        ":using_cuda": ["//kv_cache_manager/client/src/internal/sdk:sdk_buffer_check_util"],
        ":using_musa": ["//kv_cache_manager/client/src/internal/sdk:sdk_buffer_check_util_musa"],
        "//conditions:default": [],
    }),
)
