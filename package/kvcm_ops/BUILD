load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@rules_python//python:packaging.bzl", "py_wheel")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "kvcm_ops_lib",
    srcs = glob([
        "kvcm/**/*.py",
        "trace/**/*.py",
        "util/**/*.py",
        "config_server/**/*.py",
        "*.py",
    ]),
    deps = ["@pip_cpu//requests"],
)

py_wheel(
    name = "kvcm_ops_wheel",
    distribution = "kvcm_ops",
    python_tag = "py3",
    strip_path_prefixes = ["package"],
    version = "0.1.0",
    deps = [":kvcm_ops_lib"],
)

py_test(
    name = "storage_util_test",
    srcs = ["test/storage_util_test.py"],
    deps = [":kvcm_ops_lib"],
)

py_test(
    name = "instance_group_revisit_buckets_test",
    srcs = ["test/instance_group_revisit_buckets_test.py"],
    deps = [":kvcm_ops_lib"],
)

py_test(
    name = "instance_group_fairness_test",
    srcs = ["test/instance_group_fairness_test.py"],
    deps = [":kvcm_ops_lib"],
)
