package(default_visibility = ["//visibility:public"])

cc_library(
    name = "dynamic_fenwick_tree",
    srcs = ["dynamic_fenwick_tree.cc"],
    hdrs = ["dynamic_fenwick_tree.h"],
)

cc_library(
    name = "hit_curve",
    srcs = ["hit_curve.cc"],
    hdrs = ["hit_curve.h"],
)

cc_library(
    name = "request_preprocess",
    srcs = ["request_preprocess.cc"],
    hdrs = ["request_preprocess.h"],
)

cc_library(
    name = "facts_csv",
    srcs = ["facts_csv.cc"],
    hdrs = ["facts_csv.h"],
    deps = [":hit_curve"],
)

cc_library(
    name = "facts_query",
    srcs = ["facts_query.cc"],
    hdrs = ["facts_query.h"],
    deps = [
        ":facts_csv",
        ":hit_curve",
    ],
)

cc_library(
    name = "trace_router",
    srcs = ["trace_router.cc"],
    hdrs = ["trace_router.h"],
)

cc_library(
    name = "weighted_lru_pool",
    srcs = ["weighted_lru_pool.cc"],
    hdrs = ["weighted_lru_pool.h"],
    deps = [":dynamic_fenwick_tree"],
)

cc_library(
    name = "lite_hit_linear",
    srcs = ["lite_hit_linear.cc"],
    hdrs = ["lite_hit_linear.h"],
    deps = [":weighted_lru_pool"],
)

cc_library(
    name = "lite_hit_ttl",
    srcs = ["lite_hit_ttl.cc"],
    hdrs = ["lite_hit_ttl.h"],
    deps = [
        ":lite_hit",
        ":weighted_lru_pool",
    ],
)

cc_library(
    name = "lite_hit",
    srcs = ["lite_hit.cc"],
    hdrs = ["lite_hit.h"],
    deps = [
        ":hit_curve",
        ":lite_hit_linear",
        ":weighted_lru_pool",
    ],
)
