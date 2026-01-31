load("@grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@pip_cpu//:requirements.bzl", "requirement")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("//bazel:py_proto.bzl", "generate_grpc_proto")
load("//bazel:tf_proto.bzl", "cc_proto", "tf_proto_library_cc")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "client_only_header",
    values = {"define": "CLIENT_ONLY_HEADER=true"},
)

cc_library(
    name = "service_cc_proto",
    srcs = [
        "admin_service.pb.cc",
        "debug_service.pb.cc",
        "kv_meta_service.pb.cc",
        "meta_service.pb.cc",
    ],
    hdrs = [
        "admin_service.pb.h",
        "debug_service.pb.h",
        "kv_meta_service.pb.h",
        "meta_service.pb.h",
    ],
    include_prefix = "service/proto",
    deps = select({
        "client_only_header": [
            ":admin_service_cc_impl_grpc_header_only",
            ":debug_service_cc_impl_grpc_header_only",
            ":kv_meta_service_cc_impl_grpc_header_only",
            ":meta_service_cc_impl_grpc_header_only",
        ],
        "//conditions:default": [
            ":admin_service_cc_impl",
            ":debug_service_cc_impl",
            ":kv_meta_service_cc_impl",
            ":meta_service_cc_impl",
        ],
    }),
)

cc_library(
    name = "service_cc_grpc",
    srcs = [
        "admin_service.grpc.pb.cc",
        "debug_service.grpc.pb.cc",
        "meta_service.grpc.pb.cc",
    ],
    hdrs = [
        "admin_service.grpc.pb.h",
        "debug_service.grpc.pb.h",
        "meta_service.grpc.pb.h",
    ],
    include_prefix = "service/proto",
    visibility = ["//visibility:public"],
    deps = [":service_cc_proto"],
)

py_library(
    name = "meta_service_py_proto",
    srcs = [":meta_service_py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "admin_service_py_proto",
    srcs = [":admin_service_py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "debug_service_py_proto",
    srcs = [":debug_service_py"],
    visibility = ["//visibility:public"],
)

## ========================================================================

py_binary(
    name = "create_grpc_proto",
    srcs = ["create_grpc_proto.py"],
    data = [],
    visibility = ["//visibility:public"],
    deps = [
        "//kv_cache_manager:grpcio-tools",
        "//kv_cache_manager:setuptools",
    ],
)

generate_grpc_proto(
    name = "meta_service_py",
    create_grpc_proto = ":create_grpc_proto",
    proto = "meta_service.proto",
)

generate_grpc_proto(
    name = "kv_meta_service_py",
    create_grpc_proto = ":create_grpc_proto",
    proto = "kv_meta_service.proto",
)

generate_grpc_proto(
    name = "admin_service_py",
    create_grpc_proto = ":create_grpc_proto",
    proto = "admin_service.proto",
)

generate_grpc_proto(
    name = "debug_service_py",
    create_grpc_proto = ":create_grpc_proto",
    proto = "debug_service.proto",
)

tf_proto_library_cc(
    name = "meta_service",
    srcs = [
        "meta_service.proto",
    ],
    cc_grpc_version = True,
    default_header = True,
)

tf_proto_library_cc(
    name = "kv_meta_service",
    srcs = [
        "kv_meta_service.proto",
    ],
    cc_grpc_version = True,
    default_header = True,
)

tf_proto_library_cc(
    name = "admin_service",
    srcs = [
        "admin_service.proto",
    ],
    cc_grpc_version = True,
    default_header = True,
)

tf_proto_library_cc(
    name = "debug_service",
    srcs = [
        "debug_service.proto",
    ],
    cc_grpc_version = True,
    default_header = True,
)
