package(default_visibility = ["//visibility:public"])

cc_library(
    name = "dynamic_fenwick_tree",
    srcs = ["dynamic_fenwick_tree.cc"],
    hdrs = ["dynamic_fenwick_tree.h"],
)

cc_library(
    name = "lite_hit",
    srcs = ["lite_hit.cc"],
    hdrs = ["lite_hit.h"],
    deps = [":dynamic_fenwick_tree"],
)
