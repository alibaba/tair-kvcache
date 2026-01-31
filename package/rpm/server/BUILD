load("@rules_pkg//:rpm.bzl", "pkg_rpm")
load("@rules_pkg//pkg:mappings.bzl", "pkg_attributes", "pkg_files")

pkg_rpm(
    name = "kv-cache-manager-server-rpm",
    package_name = "kv-cache-manager",
    srcs = [
        ":rpm_bin",
        ":rpm_commit_info",
        ":rpm_etc",
        ":rpm_script",
        ":rpm_tools",
    ],
    architecture = "x86_64",
    description = """Tair KVCache Manager is a high-performance distributed key-value cache management system with pluggable storage backends. It provides unified storage abstraction and efficient caching for large-scale AI systems. Key features: - Pluggable storage backend support (Hf3fs, Nfs, Local, Mooncake, TairMempool) - gRPC and HTTP service interfaces - Configurable cache management - Efficient metadata handling""",
    group = "Applications/System",
    license = "Alibaba",
    release = "1",
    summary = "Tair KVCache Manager - High-performance distributed key-value cache management system",
    tags = ["manual"],
    url = "https://github.com/alibaba/tair-kvcache",
    version = "1.0.0",
)

###############################################################################

pkg_files(
    name = "rpm_etc",
    srcs = [
        "//package/etc:default_server_confs",
    ],
    attributes = pkg_attributes(
        mode = "0755",
    ),
    prefix = "etc",
)

pkg_files(
    name = "rpm_bin",
    srcs = [
        "//kv_cache_manager:kv_cache_manager_bin",
    ],
    attributes = pkg_attributes(
        mode = "0755",
    ),
    prefix = "bin",
)

pkg_files(
    name = "rpm_script",
    srcs = [
        "//package/script:start_server_script",
    ],
    attributes = pkg_attributes(
        mode = "0755",
    ),
    prefix = "bin",
)

pkg_files(
    name = "rpm_tools",
    srcs = [
        "//tools/cli_py:kv_cache_manager_rpc_client",
    ],
    attributes = pkg_attributes(
        mode = "0755",
    ),
    prefix = "bin",
)

pkg_files(
    name = "rpm_commit_info",
    srcs = [
        "//package/git_commit:commit_version_file",
    ],
    attributes = pkg_attributes(
        mode = "0755",
    ),
    prefix = "etc",
)
