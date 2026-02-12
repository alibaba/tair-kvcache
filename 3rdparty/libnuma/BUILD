load("//bazel:defs.bzl", "rpm_library")

rpm_library(
    name = "libnuma",
    hdrs = [
        "numa.h",
    ],
    include_path = "usr/include",
    lib_path = "usr/lib64",
    rpms = select({
        "@platforms//cpu:aarch64": [
            "@numactl-libs_rpm_aarch64//file:file",
            "@numactl-devel_rpm_aarch64//file:file",
        ],
        "@platforms//cpu:x86_64": [
            "@numactl-libs_rpm_x86_64//file:file",
            "@numactl-devel_rpm_x86_64//file:file",
        ],
        "//conditions:default": [
            "@numactl-libs_rpm_x86_64//file:file",
            "@numactl-devel_rpm_x86_64//file:file",
        ],
    }),
    shared_libs = [
        "libnuma.so.1",
    ],
)
