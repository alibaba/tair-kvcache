package(default_visibility = ["//integration_test:__subpackages__"])

load("//bazel:bundle.bzl", "bundle_files", "bundle_tar", "bundle_unpack")

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/**",
        "confs/**",
    ]),
)

bundle_unpack(
    name = "kv_cache_manager_install",
    srcs = [
        "//kv_cache_manager:kv_cache_manager_package",
    ],
    prefix = "install_root",
)
