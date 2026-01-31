load("@rules_pkg//:rpm.bzl", "pkg_rpm")
load("@rules_pkg//pkg:mappings.bzl", "pkg_attributes", "pkg_files")

pkg_rpm(
    name = "kv-cache-manager-client-rpm",
    package_name = "kv-cache-manager-client",
    srcs = [
        ":rpm_client_commit_info",
        ":rpm_client_include",
        ":rpm_client_lib",
    ],
    architecture = "x86_64",
    description = """Tair KVCache Manager Client Library provides C++ APIs for interacting with the Tair KVCache Manager service. This library allows applications to load and save KV caches from various storage backends. Key features: - Support for multiple storage backends (Hf3fs, Nfs, Local, Mooncake, TairMempool) - TransferClient API for loading and saving KV caches - MetaClient API for metadata operations - ManagerClient API for cluster management""",
    group = "Development/Libraries",
    license = "Alibaba",
    release = "1",
    summary = "Tair KVCache Manager Client Library - C++ client library for Tair KVCache Manager",
    tags = ["manual"],
    url = "https://github.com/alibaba/tair-kvcache",
    version = "1.0.0",
)

##########################################################################

pkg_files(
    name = "rpm_client_include",
    srcs = [
        "//kv_cache_manager/client:include",
    ],
    attributes = pkg_attributes(
        mode = "0755",
    ),
    prefix = "include/kv_cache_manager",
)

pkg_files(
    name = "rpm_client_lib",
    srcs = [
        "//kv_cache_manager/client:kv_cache_manager_client.so",
    ],
    attributes = pkg_attributes(
        mode = "0755",
    ),
    prefix = "lib",
)

pkg_files(
    name = "rpm_client_commit_info",
    srcs = [
        "//package/git_commit:commit_version_file",
    ],
    attributes = pkg_attributes(
        mode = "0755",
    ),
    prefix = "etc",
)
