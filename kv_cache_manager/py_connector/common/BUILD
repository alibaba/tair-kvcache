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

# Keep the standalone Manager client dependency surface intentionally small.
# In particular, do not pull types.py or tp_coordinator.py into its wheel: those
# modules add torch/attrs/zmq dependencies that the HTTP client does not need.
py_library(
    name = "manager_client",
    srcs = [
        ":gen_version_info",
        "logger.py",
        "manager_client.py",
        "service_discovery.py",
        "service_discovery_factory.py",
        "static_service_discovery.py",
    ],
    deps = [
        "//stub_source/kv_cache_manager/py_connector/common:common",
        "@pip_cpu//requests",
    ],
)
