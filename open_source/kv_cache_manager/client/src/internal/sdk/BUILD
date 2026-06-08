cc_library(
    name = "tair_mempool_sdk",
    srcs = ["tair_mempool_sdk.cc"],
    hdrs = ["tair_mempool_sdk.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//kv_cache_manager/client/src/internal/sdk:sdk_common",
        "//kv_cache_manager/common",
    ],
)

cc_library(
    name = "tair_mempool_caller_node_provider",
    srcs = ["tair_mempool_caller_node_provider.cc"],
    hdrs = ["tair_mempool_caller_node_provider.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//kv_cache_manager/client:common",
        "//kv_cache_manager/client/src/internal/sdk:caller_node_provider",
        "//kv_cache_manager/common",
        "//kv_cache_manager/data_storage:common_define",
    ],
)
