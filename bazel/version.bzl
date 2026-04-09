"""Reusable Bazel macros for generating version info files.

Usage in BUILD files:

    # Python version module
    load("//bazel:version.bzl", "version_info_py")
    version_info_py(name = "gen_version_info")

    # C++ version header
    load("//bazel:version.bzl", "version_info_cc")
    version_info_cc(name = "build_version")
"""

def version_info_py(name, **kwargs):
    """Generate a Python _version_info.py module with build metadata.

    The generated file exposes VERSION, GIT_COMMIT, GIT_COMMIT_FULL,
    GIT_REPO, BUILD_DATE, BUILD_TIME, and FULL_VERSION variables.
    """
    native.genrule(
        name = name,
        srcs = ["//bazel:gen_version_info.py"],
        outs = ["_version_info.py"],
        cmd = "python3 $(location //bazel:gen_version_info.py) --format=python" +
              " --stable=bazel-out/stable-status.txt" +
              " --volatile=bazel-out/volatile-status.txt" +
              " --output=$(OUTS)",
        stamp = 1,
        **kwargs
    )

def version_info_cc(name, header_name = "build_version.h", **kwargs):
    """Generate a C++ header with build metadata and wrap it as cc_library.

    The generated header defines KVCM_VERSION, KVCM_GIT_COMMIT,
    KVCM_GIT_COMMIT_FULL, KVCM_GIT_REPO, KVCM_BUILD_DATE,
    KVCM_BUILD_TIME, and KVCM_FULL_VERSION macros.
    """
    gen_name = name + "_gen"
    native.genrule(
        name = gen_name,
        srcs = ["//bazel:gen_version_info.py"],
        outs = [header_name],
        cmd = "python3 $(location //bazel:gen_version_info.py) --format=cpp" +
              " --stable=bazel-out/stable-status.txt" +
              " --volatile=bazel-out/volatile-status.txt" +
              " --output=$(OUTS)",
        stamp = 1,
    )
    native.cc_library(
        name = name,
        hdrs = [":" + gen_name],
        **kwargs
    )
