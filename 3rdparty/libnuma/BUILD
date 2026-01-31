load("//bazel:defs.bzl", "rpm_library")

rpm_library(
    name = "libnuma",
    hdrs = [
        "numa.h",
    ],
    include_path = "usr/include",
    lib_path = "usr/lib64",
    rpms = [
        "@numactl-libs_rpm//file:file",
        "@numactl-devel_rpm//file:file",
    ],
    shared_libs = [
        "libnuma.so.1",
    ],
)
