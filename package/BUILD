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
        "//package/script:decode_stack_trace_script",
        "//package/script:start_server_script",
        "//package/script:start_server_spectrum_script",
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
    srcs = [
        "//package/kvcm_ops:kvcm_ops_wheel",
    ],
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

###############################################################
# Online Optimizer Server Package
###############################################################

bundle_tar(
    name = "online_optimizer_server",
    srcs = [
        ":optimizer_package",
    ],
    extension = "tar.gz",
    visibility = ["//visibility:public"],
)

bundle_files(
    name = "optimizer_pkg_bin",
    srcs = [
        "//kv_cache_manager/optimizer:online_optimizer_server_main",
    ],
    prefix = "bin",
)

bundle_files(
    name = "optimizer_pkg_etc",
    srcs = [
        "//package/online_optimizer:default_optimizer_confs",
    ],
    prefix = "etc",
)

bundle_files(
    name = "optimizer_pkg_script",
    srcs = [
        "//package/online_optimizer:start_optimizer_server_script",
    ],
    prefix = "bin",
)

bundle_files(
    name = "optimizer_package",
    srcs = [],
    deps = [
        ":optimizer_pkg_bin",
        ":optimizer_pkg_etc",
        ":optimizer_pkg_script",
        "//package/git_commit:commit_version_file",
    ],
)
