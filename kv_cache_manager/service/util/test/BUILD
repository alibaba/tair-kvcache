package(default_visibility = ["//visibility:private"])

load("//bazel:tf_proto.bzl", "tf_proto_library_cc")

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

tf_proto_library_cc(
    name = "service_util_test",
    srcs = [
        "service_util_test.proto",
    ],
    default_header = True,
)

cc_library(
    name = "service_util_test_cc_proto",
    srcs = [
        "service_util_test.pb.cc",
    ],
    hdrs = [
        "service_util_test.pb.h",
    ],
    include_prefix = "service/util/test",
    deps = [":service_util_test_cc_impl"],
)
