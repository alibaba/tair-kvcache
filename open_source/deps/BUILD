package(default_visibility = ["//visibility:public"])

load("@rules_python//python:packaging.bzl", "py_package")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load(":pip.bzl", "PIP_EXTRA_ARGS")

exports_files(["requirements_base.txt"])

py_package(
    name = "extension_package",
    deps = [
    ],
    packages = [
        "kv_cachae_manager"
    ],
)

py_library(
    name = "extension_lib",
    deps = [],
)

compile_pip_requirements(
    name = "requirements_cpu",
    src = "requirements_cpu.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//open_source/deps:requirements_base.txt"],
    requirements_txt = "requirements_lock_cpu.txt",
    tags = ["manual"],
)