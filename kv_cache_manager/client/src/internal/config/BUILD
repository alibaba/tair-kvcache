package(default_visibility = ["//visibility:public"])

cc_library(
    name = "config",
    srcs = [
        "client_config.cc",
        "sdk_config.cc",
    ],
    hdrs = [
        "client_config.h",
        "meta_channel_config.h",
        "sdk_config.h",
    ],
    deps = [
        "//kv_cache_manager/common:jsonizable",
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/config",
    ],
)
