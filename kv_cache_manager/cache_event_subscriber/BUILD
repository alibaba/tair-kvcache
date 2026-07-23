load("@rules_python//python:py_library.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "cache_event_subscriber",
    srcs = glob(["*.py"]),
    deps = [
        "//kv_cache_manager/py_connector/common:common",
        "@pip_cpu//grpcio",
        "@pip_cpu//protobuf",
    ],
)
