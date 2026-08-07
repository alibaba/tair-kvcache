load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "kv_cache_manager_manager_client",
    srcs = ["__init__.py"],
    deps = [
        "//kv_cache_manager/py_connector/common:manager_client",
    ],
)

py_package(
    name = "kv_cache_manager_manager_client_pkg",
    packages = [
        "kv_cache_manager",
        "stub_source",
    ],
    deps = [
        ":kv_cache_manager_manager_client",
    ],
)

py_wheel(
    name = "kv_cache_manager_manager_client_wheel",
    abi = "none",
    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    description_content_type = "text/markdown",
    description_file = "README.md",
    distribution = "tair-kvcache-manager-client",
    homepage = "https://github.com/alibaba/tair-kvcache",
    platform = "any",
    python_requires = ">=3.9",
    python_tag = "py3",
    requires = [
        "requests>=2.31.0,<3",
    ],
    stamp = 1,
    summary = "Python HTTP client for Tair KVCache Manager",
    version = "{STABLE_KVCM_VERSION}+{STABLE_BUILD_TIMESTAMP}.{STABLE_GIT_COMMIT}",
    deps = [
        ":kv_cache_manager_manager_client_pkg",
    ],
)

py_test(
    name = "ManagerClientPublicApiTest",
    size = "small",
    srcs = ["public_api_test.py"],
    main = "public_api_test.py",
    deps = [
        ":kv_cache_manager_manager_client",
    ],
)

py_test(
    name = "ManagerClientWheelContentsTest",
    size = "small",
    srcs = ["wheel_contents_test.py"],
    args = ["$(location :kv_cache_manager_manager_client_wheel)"],
    data = [
        ":kv_cache_manager_manager_client_wheel",
    ],
    main = "wheel_contents_test.py",
)
