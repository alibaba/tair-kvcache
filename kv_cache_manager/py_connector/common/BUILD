load("//bazel:version.bzl", "version_info_py")

package(default_visibility = ["//kv_cache_manager/py_connector:__subpackages__"])

version_info_py(name = "gen_version_info")

py_library(
    name = "common",
    srcs = glob(["*.py"]) + [":gen_version_info"],
)