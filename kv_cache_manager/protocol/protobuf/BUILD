load("@grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@pip_cpu//:requirements.bzl", "requirement")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("//bazel:py_proto.bzl", "generate_grpc_proto")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "client_only_header",
    values = {"define": "CLIENT_ONLY_HEADER=true"},
)

# ---- Standard proto_library targets ----

proto_library(
    name = "meta_service_proto",
    srcs = ["meta_service.proto"],
)

proto_library(
    name = "kv_meta_service_proto",
    srcs = ["kv_meta_service.proto"],
)

proto_library(
    name = "admin_service_proto",
    srcs = ["admin_service.proto"],
    deps = ["@com_google_protobuf//:wrappers_proto"],
)

proto_library(
    name = "debug_service_proto",
    srcs = ["debug_service.proto"],
)

# ---- Standard cc_proto_library targets ----

cc_proto_library(
    name = "meta_service_cc_proto",
    deps = [":meta_service_proto"],
)

cc_proto_library(
    name = "kv_meta_service_cc_proto",
    deps = [":kv_meta_service_proto"],
)

cc_proto_library(
    name = "admin_service_cc_proto",
    deps = [":admin_service_proto"],
)

cc_proto_library(
    name = "debug_service_cc_proto",
    deps = [":debug_service_proto"],
)

# ---- Standard cc_grpc_library targets ----

cc_grpc_library(
    name = "meta_service_cc_grpc",
    srcs = [":meta_service_proto"],
    deps = [":meta_service_cc_proto"],
    grpc_only = True,
)

cc_grpc_library(
    name = "admin_service_cc_grpc",
    srcs = [":admin_service_proto"],
    deps = [":admin_service_cc_proto"],
    grpc_only = True,
)

cc_grpc_library(
    name = "debug_service_cc_grpc",
    srcs = [":debug_service_proto"],
    deps = [":debug_service_cc_proto"],
    grpc_only = True,
)

# ---- Compatibility aggregation targets ----

cc_library(
    name = "service_cc_proto",
    deps = [
        ":admin_service_cc_proto",
        ":debug_service_cc_proto",
        ":kv_meta_service_cc_proto",
        ":meta_service_cc_proto",
    ],
)

cc_library(
    name = "service_cc_grpc",
    visibility = ["//visibility:public"],
    deps = [
        ":admin_service_cc_grpc",
        ":debug_service_cc_grpc",
        ":meta_service_cc_grpc",
        ":service_cc_proto",
    ],
)

# ---- Python proto targets ----

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
