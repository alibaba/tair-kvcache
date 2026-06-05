package(default_visibility = ["//visibility:public"])

# Candidate processing pipeline primitives (filter / sort / limit).
# CandidatePipeline is a reusable 5-stage algorithm, not tied to any
# specific path; callers decide where to apply it.

cc_library(
    name = "metric_catalog",
    srcs = ["metric_catalog.cc"],
    hdrs = ["metric_catalog.h"],
    deps = [
        "//kv_cache_manager/affinity:node_metrics",
    ],
)

cc_library(
    name = "filter_cond",
    srcs = ["filter_cond.cc"],
    hdrs = ["filter_cond.h"],
    deps = [
        ":metric_catalog",
        "//kv_cache_manager/affinity:node_metrics",
        "@rapidjson",
    ],
)

cc_library(
    name = "candidate_pipeline",
    srcs = ["candidate_pipeline.cc"],
    hdrs = ["candidate_pipeline.h"],
    deps = [
        ":filter_cond",
        ":metric_catalog",
        "//kv_cache_manager/affinity:node_metrics",
        "//kv_cache_manager/common:affinity_types",
        "@rapidjson",
    ],
)
