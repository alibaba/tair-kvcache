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
    requires_file = "//open_source/deps:requirements_kvcm_ops.txt",
    strip_path_prefixes = ["package"],
    version = "0.1.0",
    deps = [":kvcm_ops_lib"],
)

py_test(
    name = "wheel_metadata_test",
    size = "small",
    srcs = ["test/wheel_metadata_test.py"],
    args = [
        "$(location :kvcm_ops_wheel)",
        "$(location //open_source/deps:requirements_kvcm_ops.txt)",
    ],
    data = [
        ":kvcm_ops_wheel",
        "//open_source/deps:requirements_kvcm_ops.txt",
    ],
    main = "test/wheel_metadata_test.py",
)

py_test(
    name = "wheel_install_test",
    size = "medium",
    srcs = ["test/wheel_install_test.py"],
    args = ["$(location :kvcm_ops_wheel)"],
    data = [":kvcm_ops_wheel"],
    main = "test/wheel_install_test.py",
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
