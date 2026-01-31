package(default_visibility = ["//visibility:public"])

cc_library(
    name = "trace_converter",
    srcs = [
        "converter_factory.cc",
        "publisher_log_converter.cc",
        "qwen_bailian_converter.cc",
        "optimizer_schema_converter.cc",
        "trace_util.cc",
    ],
    hdrs = [
        "optimizer_schema_trace.h",
        "base_converter.h",
        "converter_factory.h",
        "publisher_log_converter.h",
        "qwen_bailian_converter.h",
        "optimizer_schema_converter.h",
        "trace_util.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/manager:cache_location",
        "//kv_cache_manager/optimizer/config",
        "//kv_cache_manager/manager:hash_util",
    ],
)
