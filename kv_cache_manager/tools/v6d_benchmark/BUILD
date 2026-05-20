package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "v6d_benchmark",
    srcs = [
        "v6d_benchmark_main.cc",
        "v6d_benchmark.cc",
        "v6d_benchmark.h",
        "http_client.cc",
        "http_client.h",
        "metrics_reporter.cc",
        "metrics_reporter.h",
        "result_verifier.h",
        "config.h",
    ],
    deps = [
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common:string_util",
        "//kv_cache_manager/common:jsonizable",
        "//kv_cache_manager/client:common",  # for QueryType in common.h
        "@httplib//:httplib",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ],
    copts = [
        "-std=c++17",
    ],
    linkopts = [
        "-lpthread",
    ],
)
