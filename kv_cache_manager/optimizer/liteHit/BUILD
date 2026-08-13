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
    name = "lite_hit",
    srcs = ["lite_hit.cc"],
    hdrs = ["lite_hit.h"],
    deps = [
        ":dynamic_fenwick_tree",
        ":hit_curve",
    ],
)
