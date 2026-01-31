package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "kv_cache_manager_client.so",
    srcs = [],
    copts = [],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        ":client",
    ],
)

filegroup(
    name = "include",
    srcs = glob(["include/*.h"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "client",
    deps = [
        ":client_impl",
    ],
    alwayslink = True,
)

########################################################################

cc_library(
    name = "common",
    hdrs = [
        "include/common.h",
    ],
    strip_include_prefix = "include",
    alwayslink = True,
)

cc_library(
    name = "client_api",
    hdrs = [
        "include/manager_client.h",
        "include/meta_client.h",
        "include/rtp_llm_client.h",
        "include/transfer_client.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":common",
    ],
    alwayslink = True,
)

cc_library(
    name = "client_impl",
    srcs = [
        "src/manager_client_impl.cc",
        "src/meta_client_impl.cc",
        "src/rtp_llm_client_impl.cc",
        "src/transfer_client_impl.cc",
    ],
    hdrs = [
        "src/manager_client_impl.h",
        "src/meta_client_impl.h",
        "src/rtp_llm_client_impl.h",
        "src/transfer_client_impl.h",
    ],
    deps = [
        ":client_api",
        "//kv_cache_manager/client/src/internal/config",
        "//kv_cache_manager/client/src/internal/sdk",
        "//kv_cache_manager/client/src/internal/stub:grpc_stub",
        "//kv_cache_manager/common:env_util",
        "//kv_cache_manager/common:string_util",
    ],
    alwayslink = True,
)
