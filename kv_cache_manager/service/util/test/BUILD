load("//bazel:tf_proto.bzl", "tf_proto_library_cc")

package(default_visibility = ["//visibility:private"])

cc_binary(
    name = "proto_message_json_benchmark",
    srcs = ["proto_message_json_benchmark.cc"],
    tags = ["manual"],
    deps = [
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
        "//kv_cache_manager/service/util:proto_message_json_util",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "AccessLogWriterTest",
    srcs = ["access_log_writer_test.cc"],
    deps = [
        "//kv_cache_manager/common:request_context",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/service/util:access_log_writer",
        "@rapidjson",
    ],
)

cc_test(
    name = "ProtoMessageJsonUtilTest",
    srcs = [
        "proto_message_json_util_test.cc",
    ],
    copts = ["-fno-access-control"],
    deps = [
        ":service_util_test_cc_proto",
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
        "//kv_cache_manager/service/util:manager_message_proto_util",
        "//kv_cache_manager/service/util:proto_message_json_util",
        "//kv_cache_manager/service/util:report_event_json_parser",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "FastProtoJsonCodecConformanceTest",
    srcs = ["fast_proto_json_codec_conformance_test.cc"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/protocol/protobuf:service_cc_proto",
        "//kv_cache_manager/service/util:proto_message_json_util",
        "@com_google_protobuf//:protobuf",
        "@rapidjson",
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
