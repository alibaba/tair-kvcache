load("@pip_cpu//:requirements.bzl", requirement_cpu="requirement")

def requirement(names):
    for name in names:
        native.py_library(
            name = name,
            deps = select({
                "//conditions:default": [requirement_cpu(name)],
            }),
            visibility = ["//visibility:public"],
        )

