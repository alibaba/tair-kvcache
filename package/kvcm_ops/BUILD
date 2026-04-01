load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:packaging.bzl", "py_wheel")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "kvcm_ops_lib",
    srcs = glob([
        "kvcm/**/*.py",
        "trace/**/*.py",
        "util/**/*.py",
        "*.py",
    ]),
)

py_wheel(
    name = "kvcm_ops_wheel",
    distribution = "kvcm_ops",
    python_tag = "py3",
    strip_path_prefixes = ["package"],
    version = "0.1.0",
    deps = [":kvcm_ops_lib"],
)
