# MUSA runtime is provided by vendor toolkit and is subject to third-party terms.
licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "musa_headers",
    hdrs = glob(["musa/include/**"]) + ["musa/musa_config.h"],
    includes = [
        ".",
        "musa/include",
    ],
)

cc_library(
    name = "musart",
    srcs = ["musa/lib/libmusart.so"],
    data = ["musa/lib/libmusart.so"],
    linkstatic = 1,
)

cc_library(
    name = "musa",
    deps = [
        ":musa_headers",
        ":musart",
    ],
)
