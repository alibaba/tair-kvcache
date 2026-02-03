package(default_visibility = ["//visibility:public"])

config_setting(
    name = "client_only_header",
    values = {"define": "CLIENT_ONLY_HEADER=true"},
)

# TODO(qisa.cb) 目标太杂了，需要归类一下
cc_library(
    name = "common",
    srcs = [
    ],
    hdrs = [
        "common.h",
        "error_code.h",
    ],
    deps = [
        ":jsonizable",
        ":logger",
        ":request_context",
        ":string_util",
    ],
)

cc_library(
    name = "utils",
    deps = [
        ":common_util",
        ":env_util",
        ":hash_util",
        ":string_util",
        ":timestamp_util",
    ],
)

cc_library(
    name = "env_util",
    srcs = [
        "env_util.cc",
    ],
    hdrs = [
        "env_util.h",
    ],
    deps = select({
        "client_only_header": [
            "@havenask//aios/autil:kvcm_client_patch_autil_header",
        ],
        "//conditions:default": [
            "@havenask//aios/autil:string_helper",
        ],
    }),
)

cc_library(
    name = "hash_util",
    hdrs = [
        "hash_util.h",
    ],
)

cc_library(
    name = "logger",
    srcs = [
        "logger.cc",
    ],
    hdrs = [
        "logger.h",
    ],
    deps = [
        ":env_util",
    ] + select({
        "client_only_header": [
            "@havenask//aios/alog:kvcm_client_patch_alog_header",
        ],
        "//conditions:default": [
            "@havenask//aios/alog",
        ],
    }),
)

cc_library(
    name = "jsonizable",
    srcs = [
        "jsonizable.cc",
    ],
    hdrs = [
        "jsonizable.h",
    ],
    deps = [
        ":logger",
        "@rapidjson",
    ],
)

cc_library(
    name = "unittest",
    testonly = True,
    hdrs = [
        "unittest.h",
    ],
    deps = [
        ":logger",
        ":timestamp_util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "common_util",
    srcs = [
        "common_util.cc",
    ],
    hdrs = [
        "common_util.h",
    ],
    deps = [
    ],
)

cc_library(
    name = "string_util",
    srcs = [
    ],
    hdrs = [
        "string_util.h",
    ],
    deps = [
    ],
)

cc_library(
    name = "timestamp_util",
    srcs = [
        "timestamp_util.cc",
    ],
    hdrs = [
        "timestamp_util.h",
    ],
    deps = [
    ],
)

cc_library(
    name = "request_context",
    srcs = [
        "request_context.cc",
    ],
    hdrs = [
        "request_context.h",
    ],
    deps = [
        ":string_util",
        ":tracer",
        "//kv_cache_manager/metrics:metrics_collector",
    ],
)

cc_library(
    name = "concurrent_hash_map",
    srcs = [
    ],
    hdrs = [
        "concurrent_hash_map.h",
    ],
    deps = [
    ],
)

cc_library(
    name = "loop_thread",
    srcs = [
        "loop_thread.cc",
    ],
    hdrs = [
        "loop_thread.h",
    ],
    deps = [
    ],
)

cc_library(
    name = "lru_cache",
    srcs = [
    ],
    hdrs = [
    ],
    deps = [
        "//kv_cache_manager/common/cache",
    ],
)

cc_library(
    name = "standard_uri",
    srcs = [
        "standard_uri.cc",
    ],
    hdrs = [
        "standard_uri.h",
    ],
    deps = [
        ":string_util",
    ],
)

cc_library(
    name = "redis_client",
    srcs = [
        "redis_client.cc",
    ],
    hdrs = [
        "redis_client.h",
    ],
    deps = [
        ":common",
        ":standard_uri",
        "@hiredis",
    ],
)

cc_library(
    name = "redis_client_ext",
    srcs = [
        "redis_client_ext.cc",
    ],
    hdrs = [
        "redis_client_ext.h",
    ],
    deps = [
        ":redis_client",
        "@hiredis",
    ],
)

cc_library(
    name = "tracer",
    srcs = [
    ],
    hdrs = [
        "tracer.h",
    ],
    deps = [
        ":jsonizable",
        ":timestamp_util",
    ],
)

cc_library(
    name = "client_pool",
    hdrs = [
        "client_pool.h",
    ],
)
