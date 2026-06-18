"""Reusable Bazel macros for generating version info files.

Usage in BUILD files:

    # Python version module
    load("//bazel:version.bzl", "version_info_py")
    version_info_py(name = "gen_version_info")

    # C++ version library
    load("//bazel:version.bzl", "version_info_cc")
    version_info_cc(name = "build_version")
"""

def version_info_py(name, **kwargs):
    """Generate a Python _version_info.py module with build metadata.

    The generated file exposes VERSION, GIT_COMMIT, GIT_COMMIT_FULL,
    GIT_REPO, BUILD_DATE, BUILD_TIMESTAMP, BUILD_TIME, and FULL_VERSION variables.
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

def version_info_cc(name, header_name = "build_version.h", source_name = "build_version.cc", **kwargs):
    """Generate C++ build metadata constants and wrap them as cc_library.

    The generated header declares kKvcm* constants. The generated source defines
    the actual string values so timestamp changes only recompile this small
    source file before downstream relinks.
    """
    gen_name = name + "_gen"
    native.genrule(
        name = gen_name,
        srcs = ["//bazel:gen_version_info.py"],
        outs = [
            header_name,
            source_name,
        ],
        cmd = "python3 $(location //bazel:gen_version_info.py) --format=cpp" +
              " --stable=bazel-out/stable-status.txt" +
              " --volatile=bazel-out/volatile-status.txt" +
              " --output=$(@D)/" + header_name +
              " --source-output=$(@D)/" + source_name,
        stamp = 1,
    )
    native.cc_library(
        name = name,
        srcs = [":" + source_name],
        hdrs = [":" + header_name],
        **kwargs
    )
