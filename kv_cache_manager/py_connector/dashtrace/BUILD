load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "kvcm_dashtrace",
    srcs = [
        "__init__.py",
        "_proto/__init__.py",
        "_proto/kserve_minimal_pb2.py",
        "_proto/kserve_minimal_pb2_grpc.py",
        "_proto/kvcm_optimizer_ingest_minimal_pb2.py",
        "grpc_server.py",
        "instance_bootstrap.py",
        "kvcm_shadow_forwarder.py",
        "observed_request.py",
        "request_observer.py",
        "trace_recorder.py",
    ],
)

py_test(
    name = "grpc_server_test",
    srcs = ["test_grpc_server.py"],
    main = "test_grpc_server.py",
    deps = [":kvcm_dashtrace"],
)

py_test(
    name = "instance_bootstrap_test",
    srcs = ["test_instance_bootstrap.py"],
    main = "test_instance_bootstrap.py",
    deps = [":kvcm_dashtrace"],
)

py_test(
    name = "kvcm_shadow_forwarder_test",
    srcs = ["test_kvcm_shadow_forwarder.py"],
    main = "test_kvcm_shadow_forwarder.py",
    deps = [":kvcm_dashtrace"],
)

py_test(
    name = "request_observer_test",
    srcs = ["test_request_observer.py"],
    main = "test_request_observer.py",
    deps = [":kvcm_dashtrace"],
)

py_package(
    name = "kvcm_dashtrace_pkg",
    deps = [":kvcm_dashtrace"],
    packages = ["kv_cache_manager"],
)

py_wheel(
    name = "kvcm_dashtrace_wheel",
    classifiers = ["Programming Language :: Python :: 3"],
    distribution = "kvcm_dashtrace",
    python_requires = ">=3.10",
    python_tag = "py3",
    stamp = 1,
    version = "{STABLE_KVCM_VERSION}+{STABLE_BUILD_TIMESTAMP}.{STABLE_GIT_COMMIT}",
    deps = [":kvcm_dashtrace_pkg"],
)
