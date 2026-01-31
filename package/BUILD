load("//bazel:bundle.bzl", "bundle_files", "bundle_install", "bundle_tar")

# TODO(qisa.cb) inner open source

bundle_tar(
    name = "kv_cache_manager_server",
    srcs = [
        ":inner_package",
    ],
    extension = "tar.gz",
    visibility = ["//visibility:public"],
)

###############################################################

bundle_files(
    name = "pkg_bin",
    srcs = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    prefix = "bin",
)

bundle_files(
    name = "pkg_etc",
    srcs = [
        "//package/etc:default_server_confs",
    ],
    prefix = "etc",
)

bundle_files(
    name = "pkg_script",
    srcs = [
        "//package/script:start_server_script",
    ],
    prefix = "bin",
)

bundle_files(
    name = "pkg_tools",
    srcs = [
        "//tools/cli_py:kv_cache_manager_rpc_client",
    ],
    prefix = "bin",
)

bundle_files(
    name = "third_party_lib",
    srcs = [
    ],
    prefix = "lib",
)

bundle_files(
    name = "inner_package",
    srcs = [],
    deps = [
        ":pkg_bin",
        ":pkg_etc",
        ":pkg_script",
        ":pkg_tools",
        ":third_party_lib",
        "//package/git_commit:commit_version_file",
    ],
)

bundle_install(
    name = "install",
    srcs = [
        ":inner_package",
    ],
)