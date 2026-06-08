package(default_visibility = ["//visibility:public"])

cc_library(
    name = "scheduling_strategy",
    srcs = [
        "infer_scheduling_strategy.cc",
    ],
    hdrs = [
        "infer_scheduling_strategy.h",
    ],
)

cc_library(
    name = "scheduler",
    srcs = [
        "infer_active_window.cc",
        "infer_engine_scheduler.cc",
    ],
    hdrs = [
        "infer_active_window.h",
        "infer_engine_scheduler.h",
    ],
    deps = [
        ":scheduling_strategy",
        "//kv_cache_manager/optimizer/trace_loader",
    ],
)
