package(default_visibility = ["//visibility:private"])

load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")

cc_test(
    name = "ProtoMessageJsonUtilTest",
    srcs = [
        "proto_message_json_util_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        ":service_util_test_cc_proto",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/service/util:proto_message_json_util",
    ],
)

proto_library(
    name = "service_util_test_proto",
    srcs = ["service_util_test.proto"],
)

cc_proto_library(
    name = "service_util_test_cc_proto",
    deps = [":service_util_test_proto"],
)
