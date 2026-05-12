load("//bazel:version.bzl", "version_info_py")

package(default_visibility = ["//visibility:public"])

version_info_py(name = "gen_version_info")

py_library(
    name = "common",
    srcs = glob(["*.py"]) + [":gen_version_info"],
    deps = [
        "//stub_source/kv_cache_manager/py_connector/common:common",
    ],
)