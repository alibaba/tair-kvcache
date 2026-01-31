config_setting(
    name = "client_only_header",
    values = {"define": "CLIENT_ONLY_HEADER=true"},
)

cc_library(
    name = "vipserver",
    srcs = glob([
        "src/**/*.h",
        "src/**/*.hpp",
        "src/**/*.cpp",
    ]),
    hdrs = glob([
        "include/*",
    ]),
    copts = [
        "-Iexternal/vipserver/deps/include",
        "-DEASY_MULTIPLICITY",
    ],
    implementation_deps = [
        "@boost//:date_time",
        "@boost//:headers-base",
        "@boost//:interprocess",
        "@boost//:property_tree",
        "@curl",
        "@jsoncpp_git//:jsoncpp",
        "@kv_cache_manager//3rdparty/easy",
    ] + select({
        "client_only_header": [
            "@havenask//aios/alog:kvcm_client_patch_alog_header",
        ],
        "//conditions:default": [
            "@havenask//aios/alog",
        ],
    }),
    includes = ["src"],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "vipserver_headers",
    hdrs = glob([
        "include/*",
    ]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
