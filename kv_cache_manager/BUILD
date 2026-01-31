load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")
load("//bazel:arch_select.bzl", "requirement")
load("//bazel:bundle.bzl", "bundle_files")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "main",
    srcs = [
        "main.cpp",
    ],
    deps = [
        "//kv_cache_manager/service",
    ],
)

cc_binary(
    name = "kv_cache_manager_bin",
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
