package(default_visibility = ["//visibility:public"])

cc_library(
    name = "trace_loader",
    srcs = [
        "standard_trace_loader.cc",
        "trace_util.cc",
    ],
    hdrs = [
        "optimizer_schema_trace.h",
        "standard_trace_loader.h",
        "trace_util.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/manager:cache_location",
    ],
)
