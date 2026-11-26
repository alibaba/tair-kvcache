load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")
load("//bazel:arch_select.bzl", "requirement")
load("//bazel:bundle.bzl", "bundle_files")
load("//stub_source/bazel:server_build.bzl", "SERVER_LINK_FEATURES", "SERVER_LINKOPTS")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "main",
    features = SERVER_LINK_FEATURES,
    linkopts = SERVER_LINKOPTS,
    srcs = [
        "main.cpp",
    ],
    deps = [
        "//kv_cache_manager/service",
    ],
)

cc_binary(
    name = "kv_cache_manager_bin",
    features = SERVER_LINK_FEATURES,
    linkopts = SERVER_LINKOPTS,
    srcs = [
        "main.cpp",
    ],
    deps = [
        "//kv_cache_manager/service",
    ],
)

bundle_files(
    name = "kv_cache_manager_package",
    srcs = [
        ":kv_cache_manager_bin",
    ],
    prefix = "usr/local/bin",
    tags = [
        "manual",
        "no-remote-exec",
    ],
    visibility = ["//visibility:public"],
    deps = [],
)

requirement([
    "grpcio-tools",
    "setuptools",
])

refresh_compile_commands(
    name = "refresh_compile_commands",
    #    exclude_headers = "all",
    # exclude_headers = "external",
    # exclude_external_sources = True,
    targets = {
        "//kv_cache_manager/...": "",
        "//integration_test/client_test/...": "",
    },
)
