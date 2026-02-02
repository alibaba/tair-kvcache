load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "client_only_header",
    values = {"define": "CLIENT_ONLY_HEADER=true"},
)

config_setting(
    name = "enable_mooncake",
    values = {"define": "ENABLE_MOONCAKE=true"},
)

config_setting(
    name = "enable_hf3fs",
    values = {"define": "ENABLE_HF3FS=true"},
)

config_setting(
    name = "enable_tair_mempool",
    values = {"define": "ENABLE_TAIR_MEMPOOL=true"},
)

config_setting(
    name = "using_cuda",
    values = {"define": "using_cuda=true"},
)

cc_library(
    name = "sdk_common",
    srcs = ["sdk_interface.cc"],
    hdrs = [
        "sdk_interface.h",
        "sdk_type.h",
    ],
    deps = [
        "//kv_cache_manager/client:common",
        "//kv_cache_manager/client/src/internal/config",
        "//kv_cache_manager/data_storage:common_define",
    ],
)

cc_library(
    name = "sdk",
    srcs = [
        "local_file_sdk.cc",
        "sdk_factory.cc",
        "sdk_interface.cc",
        "sdk_wrapper.cc",
    ] + select({
        ":enable_mooncake": [
            "mooncake_sdk.cc",
        ],
        "//conditions:default": [],
    }) + select({
        ":enable_hf3fs": [
            "hf3fs_mempool.cc",
            "hf3fs_sdk.cc",
            "hf3fs_usrbio_client.cc",
        ],
        "//conditions:default": [],
    }),
    hdrs = [
        "local_file_sdk.h",
        "sdk_factory.h",
        "sdk_interface.h",
        "sdk_type.h",
        "sdk_wrapper.h",
    ] + select({
        ":enable_mooncake": [
            "mooncake_sdk.h",
        ],
        "//conditions:default": [],
    }) + select({
        ":enable_hf3fs": [
            "hf3fs_cuda_util.h",
            "hf3fs_mempool.h",
            "hf3fs_sdk.h",
            "hf3fs_usrbio_api.h",
            "hf3fs_usrbio_client.h",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":lock_free_thread_pool",
        "//kv_cache_manager/client:common",
        "//kv_cache_manager/client/src/internal/config",
        "//kv_cache_manager/client/src/internal/util",
        "//kv_cache_manager/common",
        "//kv_cache_manager/data_storage",
        "//kv_cache_manager/data_storage:common_define",
    ] + select({
        ":using_cuda": [
            ":sdk_buffer_check_util",
            "@local_config_cuda//cuda",
        ],
        "//conditions:default": [],
    }) + select({
        ":enable_mooncake": [
            "//3rdparty/mooncake:mooncake_client_c",
        ],
        "//conditions:default": [],
    }) + select({
        ":enable_hf3fs": [
            "//3rdparty/3fs:hf3fs",
        ],
        "//conditions:default": [],
    }) + select({
        ":enable_tair_mempool": [
            "//stub_source/kv_cache_manager/client/src/internal/sdk:tair_mempool_sdk",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "lock_free_thread_pool",
    srcs = [
        "lock_free_thread_pool.cc",
    ],
    hdrs = [
        "lock_free_thread_pool.h",
    ],
    deps = [
        "//kv_cache_manager/client:common",
        "//kv_cache_manager/common:logger",
    ] + select({
        "client_only_header": [
            "@havenask//aios/autil:kvcm_client_patch_autil_header",
        ],
        "//conditions:default": [
            "@havenask//aios/autil:lock_free",
        ],
    }),
)

cuda_library(
    name = "sdk_buffer_check_util",
    srcs = [
        "sdk_buffer_check_util.cu",
    ],
    hdrs = [
        "cuda_util.h",
        "sdk_buffer_check_util.h",
    ],
    copts = [
        "-std=c++20",
        "--compiler-options=-fPIC",
    ],
    target_compatible_with = select({
        ":using_cuda": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//kv_cache_manager/client:common",
        "//kv_cache_manager/common:hash_util",
        "//kv_cache_manager/common:logger",
    ],
)
