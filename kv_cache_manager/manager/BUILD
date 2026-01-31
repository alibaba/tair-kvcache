package(default_visibility = ["//visibility:public"])

cc_library(
    name = "hash_util",
    hdrs = [
        "hash_util.h",
    ],
)

cc_library(
    name = "cache_manager",
    srcs = [
        "cache_manager.cc",
    ],
    hdrs = [
        "cache_manager.h",
    ],
    deps = [
        ":cache_location_view",
        ":cache_manager_metrics_recorder",
        ":cache_reclaimer",
        ":data_storage_selector",
        ":hash_util",
        ":meta_searcher",
        ":reclaimer_task_supervisor",
        ":startup_config_loader",
        ":write_location_manager",
        "//kv_cache_manager/common",
        "//kv_cache_manager/config",
        "//kv_cache_manager/event:event_publisher",
        "//kv_cache_manager/metrics:metrics_collector",
        "//kv_cache_manager/metrics:metrics_registry",
    ],
)

cc_library(
    name = "cache_location_view",
    srcs = [
        "cache_location_view.cc",
    ],
    hdrs = [
        "cache_location_view.h",
    ],
    deps = [
        ":cache_location",
    ],
)

cc_library(
    name = "write_location_manager",
    srcs = [
        "write_location_manager.cc",
    ],
    hdrs = [
        "write_location_manager.h",
    ],
    deps = [
        ":cache_location_view",
        "//kv_cache_manager/common:concurrent_hash_map",
        "@havenask//aios/autil:lock_free",
    ],
)

cc_library(
    name = "cache_location",
    srcs = [
        "cache_location.cc",
    ],
    hdrs = [
        "cache_location.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:string_util",
        "//kv_cache_manager/data_storage:common_define",
    ],
)

cc_library(
    name = "cache_reclaimer",
    srcs = ["cache_reclaimer.cc"],
    hdrs = ["cache_reclaimer.h"],
    deps = [
        "//kv_cache_manager/config",
        "//kv_cache_manager/event:event_publisher",
        "//kv_cache_manager/manager:meta_searcher",
        "//kv_cache_manager/manager:schedule_plan_executor",
        "//kv_cache_manager/meta",
        "//kv_cache_manager/metrics:metrics_registry",
    ],
)

cc_library(
    name = "data_storage_selector",
    srcs = ["data_storage_selector.cc"],
    hdrs = ["data_storage_selector.h"],
    deps = [
        "//kv_cache_manager/config",
        "//kv_cache_manager/data_storage",
        "//kv_cache_manager/meta:meta_indexer_manager",
    ],
)

cc_library(
    name = "meta_searcher",
    srcs = [
        "meta_searcher.cc",
        "meta_searcher_manager.cc",
    ],
    hdrs = [
        "meta_searcher.h",
        "meta_searcher_manager.h",
    ],
    deps = [
        ":select_location_policy",
        "//kv_cache_manager/common",
        "//kv_cache_manager/manager:cache_location",
        "//kv_cache_manager/meta",
    ],
)

cc_library(
    name = "schedule_plan_executor",
    srcs = ["schedule_plan_executor.cc"],
    hdrs = ["schedule_plan_executor.h"],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/config",
        "//kv_cache_manager/data_storage",
        "//kv_cache_manager/manager:meta_searcher",
        "//kv_cache_manager/meta",
        "//kv_cache_manager/metrics:metrics_registry",
    ],
)

cc_library(
    name = "reclaimer_task_supervisor",
    srcs = ["reclaimer_task_supervisor.cc"],
    hdrs = ["reclaimer_task_supervisor.h"],
    deps = [
        ":schedule_plan_executor",
        "@havenask//aios/autil:lock_free",
    ],
)

cc_library(
    name = "startup_config_loader",
    srcs = ["startup_config_loader.cc"],
    hdrs = ["startup_config_loader.h"],
    deps = [
        "//kv_cache_manager/config",
        "//kv_cache_manager/data_storage",
    ],
)

cc_library(
    name = "select_location_policy",
    srcs = ["select_location_policy.cc"],
    hdrs = ["select_location_policy.h"],
    deps = [
        ":cache_location",
    ],
)

cc_library(
    name = "cache_manager_metrics_recorder",
    srcs = ["cache_manager_metrics_recorder.cc"],
    hdrs = ["cache_manager_metrics_recorder.h"],
    deps = [
        ":write_location_manager",
        "//kv_cache_manager/config",
        "//kv_cache_manager/meta:meta_indexer_manager",
    ],
)
