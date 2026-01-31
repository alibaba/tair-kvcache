package(default_visibility = ["//visibility:public"])

load("//bazel:bundle.bzl", "bundle_files", "bundle_tar")

filegroup(
    name = "default_server_confs",
    srcs = [
        "default_logger_config.conf",
        "default_server_config.conf",
        "default_startup_config.json",
    ],
    visibility = ["//visibility:public"],
)

bundle_files(
    name = "pkg_default_server_confs",
    srcs = [
        ":default_server_confs",
    ],
    prefix = "etc",
)

bundle_tar(
    name = "kv_cache_manager_confs",
    srcs = [
        ":pkg_default_server_confs",
    ],
    extension = "tar.gz",
    tags = [
        "manual",
    ],
    visibility = ["//visibility:public"],
)
