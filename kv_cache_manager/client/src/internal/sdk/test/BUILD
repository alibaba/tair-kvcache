load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

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

config_setting(
    name = "using_tpu",
    values = {"define": "using_tpu=true"},
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
    name = "LocalFileSdkTpuTest",
    srcs = [
        "local_file_sdk_tpu_test.cc",
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
    name = "TpuClientTest",
    srcs = select({
        ":using_tpu": [
            "tpu_client_test.cc",
        ],
        "//conditions:default": [
            "hf3fs_unittest_empty.cc",
        ],
    }),
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/client/src/internal/sdk",
        "//kv_cache_manager/common:unittest",
    ] + select({
        ":using_tpu": [
            "//3rdparty/libtpu",
        ],
        "//conditions:default": [],
    }),
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

pybind_extension(
    name = "py_tpu_client",
    srcs = select({
        ":using_tpu": [
            "py_tpu_client_binding.cc",
        ],
        "//conditions:default": [
            "hf3fs_unittest_empty.cc",
        ],
    }),
    deps = [
        "//kv_cache_manager/client/src/internal/sdk",
    ] + select({
        ":using_tpu": [
            "//3rdparty/libtpu",
        ],
        "//conditions:default": [],
    }),
)

py_library(
    name = "py_tpu_client_lib",
    data = [
        ":py_tpu_client.so",
    ],
    imports = ["."],
    visibility = ["//visibility:private"],
)

# sh_test 包一层 shell 脚本激活 venv（jax/numpy 不在 Bazel hermetic Python 中）
sh_test(
    name = "py_tpu_client_test",
    srcs = ["run_py_tpu_client_test.sh"],
    data = [
        ":py_tpu_client_lib",
        "py_tpu_client_test.py",
    ],
    tags = ["manual"],
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
