#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "google/protobuf/arena.h"
#include "google/protobuf/util/json_util.h"
#include "google/protobuf/util/message_differencer.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/protocol/protobuf/admin_service.pb.h"
#include "kv_cache_manager/protocol/protobuf/debug_service.pb.h"
#include "kv_cache_manager/protocol/protobuf/kv_meta_service.pb.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.pb.h"
#include "kv_cache_manager/service/util/manager_message_proto_util.h"
#include "kv_cache_manager/service/util/report_event_json_parser.h"
#include "service/util/fast_proto_json_codec.h"
#include "service/util/proto_message_json_util.h"
#include "service/util/test/service_util_test.pb.h"

namespace kv_cache_manager {

namespace {

bool ProtobufToJson(const google::protobuf::Message &message, std::string *json) {
    google::protobuf::util::JsonPrintOptions options;
    options.always_print_primitive_fields = true;
    options.preserve_proto_field_names = true;
    return google::protobuf::util::MessageToJsonString(message, json, options).ok();
}

bool ProtobufFromJson(std::string_view json, google::protobuf::Message *message) {
    google::protobuf::util::JsonParseOptions options;
    options.ignore_unknown_fields = true;
    const google::protobuf::StringPiece input(json.data(), static_cast<ptrdiff_t>(json.size()));
    return google::protobuf::util::JsonStringToMessage(input, message, options).ok();
}

} // namespace

class ProtoMessageJsonUtilTest : public TESTBASE {
public:
};

TEST_F(ProtoMessageJsonUtilTest, TestFastCodecSupportsAllProtocolMessages) {
    const google::protobuf::FileDescriptor *protocol_files[] = {
        proto::admin::Status::descriptor()->file(),
        proto::debug::Status::descriptor()->file(),
        proto::kv_meta::Status::descriptor()->file(),
        proto::meta::Status::descriptor()->file(),
        proto::optimizer::Status::descriptor()->file(),
    };
    for (const auto *file : protocol_files) {
        SCOPED_TRACE(file->name());
        for (int i = 0; i < file->message_type_count(); ++i) {
            SCOPED_TRACE(file->message_type(i)->full_name());
            const auto *descriptor = file->message_type(i);
            ASSERT_TRUE(FastProtoJsonCodec::Supports(descriptor));

            const auto *prototype = google::protobuf::MessageFactory::generated_factory()->GetPrototype(descriptor);
            ASSERT_NE(nullptr, prototype);
            std::unique_ptr<google::protobuf::Message> message(prototype->New());
            std::string protobuf_json;
            std::string fast_json;
            ASSERT_TRUE(ProtobufToJson(*message, &protobuf_json));
            ASSERT_TRUE(FastProtoJsonCodec::TryToJson(*message, fast_json));
            EXPECT_EQ(protobuf_json, fast_json);
        }
    }
}

TEST_F(ProtoMessageJsonUtilTest, TestFastCodecHandlesCurrentMapAndWrapperTypes) {
    proto::meta::ReportEventRequest event_request;
    auto *event = event_request.add_events();
    event->set_event_type(proto::meta::EVENT_HEARTBEAT);
    (*event->mutable_heartbeat()->mutable_system_status())["state"] = "ready";
    (*event->mutable_heartbeat()->mutable_system_status())["load"] = "7";

    std::string event_json;
    ASSERT_TRUE(FastProtoJsonCodec::TryToJson(event_request, event_json));
    std::string protobuf_event_json;
    ASSERT_TRUE(ProtobufToJson(event_request, &protobuf_event_json));
    EXPECT_EQ(protobuf_event_json, event_json);
    proto::meta::ReportEventRequest parsed_event;
    ASSERT_TRUE(FastProtoJsonCodec::TryFromJson(event_json, &parsed_event));
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(event_request, parsed_event));

    proto::admin::MigrationMarkMethodConfig wrapper_message;
    wrapper_message.set_enabled(true);
    wrapper_message.mutable_timeout_ms()->set_value(1234567890123LL);
    std::string wrapper_json;
    ASSERT_TRUE(FastProtoJsonCodec::TryToJson(wrapper_message, wrapper_json));
    EXPECT_EQ(R"({"enabled":true,"timeout_ms":"1234567890123"})", wrapper_json);
    proto::admin::MigrationMarkMethodConfig parsed_wrapper;
    ASSERT_TRUE(FastProtoJsonCodec::TryFromJson(wrapper_json, &parsed_wrapper));
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(wrapper_message, parsed_wrapper));
}

TEST_F(ProtoMessageJsonUtilTest, TestUnsupportedTypesFallBackToProtobufJsonUtil) {
    UnsupportedBytesMessage message;
    message.set_value(std::string("\0\1\2", 3));
    EXPECT_FALSE(FastProtoJsonCodec::Supports(message.GetDescriptor()));

    std::string json;
    ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&message, json));
    EXPECT_EQ(R"({"value":"AAEC"})", json);

    UnsupportedBytesMessage parsed;
    ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &parsed));
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(message, parsed));

    UnsupportedMapBytesMessage map_bytes;
    (*map_bytes.mutable_value())["key"] = std::string("\0\1\2", 3);
    EXPECT_FALSE(FastProtoJsonCodec::Supports(map_bytes.GetDescriptor()));
    std::string protobuf_map_json;
    std::string compatible_map_json;
    ASSERT_TRUE(ProtobufToJson(map_bytes, &protobuf_map_json));
    ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&map_bytes, compatible_map_json));
    EXPECT_EQ(protobuf_map_json, compatible_map_json);
    UnsupportedMapBytesMessage parsed_map_bytes;
    ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(compatible_map_json, &parsed_map_bytes));
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(map_bytes, parsed_map_bytes));

    UnsupportedNullValueMessage null_value;
    null_value.set_value(google::protobuf::NULL_VALUE);
    EXPECT_FALSE(FastProtoJsonCodec::Supports(null_value.GetDescriptor()));
    std::string protobuf_null_json;
    std::string compatible_null_json;
    ASSERT_TRUE(ProtobufToJson(null_value, &protobuf_null_json));
    ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&null_value, compatible_null_json));
    EXPECT_EQ(protobuf_null_json, compatible_null_json);
    UnsupportedNullValueMessage parsed_null_value;
    ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(compatible_null_json, &parsed_null_value));
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(null_value, parsed_null_value));
}

TEST_F(ProtoMessageJsonUtilTest, TestFastCodecParsesArenaMessageTransactionally) {
    google::protobuf::Arena arena;
    auto *request = google::protobuf::Arena::CreateMessage<proto::meta::GetCacheLocationsByBackendRequest>(&arena);
    request->set_trace_id("before");
    ASSERT_NE(nullptr, request->GetArena());

    ASSERT_TRUE(FastProtoJsonCodec::TryFromJson(
        R"({"trace_id":"after","instance_id":"instance","query_type":"QT_BATCH_GET","block_keys":["1","2"]})",
        request));
    EXPECT_EQ("after", request->trace_id());
    EXPECT_EQ(2, request->block_keys_size());

    EXPECT_FALSE(FastProtoJsonCodec::TryFromJson(R"({"block_keys":["invalid"]})", request));
    EXPECT_EQ("after", request->trace_id());
    EXPECT_EQ(2, request->block_keys_size());
}

TEST_F(ProtoMessageJsonUtilTest, TestFastCodecMatchesProtobufForScalarEdgeCases) {
    SimpleMessage source;
    source.set_int32value(std::numeric_limits<int32_t>::min());
    source.set_uint32value(std::numeric_limits<uint32_t>::max());
    source.set_int64value(std::numeric_limits<int64_t>::min());
    source.set_uint64value(std::numeric_limits<uint64_t>::max());
    source.set_doublevalue(std::numeric_limits<double>::infinity());
    source.set_floatvalue(-std::numeric_limits<float>::infinity());
    source.set_boolvalue(true);
    constexpr char kSpecialString[] = "quote:\" slash:\\ newline:\n nul:\0 unicode:雪";
    source.set_stringvalue(kSpecialString, sizeof(kSpecialString) - 1);

    std::string protobuf_json;
    std::string fast_json;
    ASSERT_TRUE(ProtobufToJson(source, &protobuf_json));
    ASSERT_TRUE(FastProtoJsonCodec::TryToJson(source, fast_json));
    EXPECT_EQ(protobuf_json, fast_json);

    const std::string compatible_json =
        R"({"int32Value":"-2147483648","uint32Value":"4294967295","int64Value":-9223372036854775808,"uint64Value":"18446744073709551615","doubleValue":"Infinity","floatValue":"-Infinity","boolValue":true,"stringValue":"unicode:\u96ea","future":{"ignored":true}})";
    SimpleMessage protobuf_parsed;
    SimpleMessage fast_parsed;
    ASSERT_TRUE(ProtobufFromJson(compatible_json, &protobuf_parsed));
    ASSERT_TRUE(FastProtoJsonCodec::TryFromJson(compatible_json, &fast_parsed));
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(protobuf_parsed, fast_parsed));

    SimpleMessage invalid_utf8;
    invalid_utf8.set_stringvalue(std::string(1, static_cast<char>(0xff)));
    std::string protobuf_invalid_json;
    std::string fast_invalid_json;
    std::string compatible_invalid_json;
    ASSERT_TRUE(ProtobufToJson(invalid_utf8, &protobuf_invalid_json));
    EXPECT_FALSE(FastProtoJsonCodec::TryToJson(invalid_utf8, fast_invalid_json));
    EXPECT_TRUE(fast_invalid_json.empty());
    ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&invalid_utf8, compatible_invalid_json));
    EXPECT_EQ(protobuf_invalid_json, compatible_invalid_json);

    proto::meta::ReportEventRequest invalid_map_utf8;
    auto *heartbeat = invalid_map_utf8.add_events()->mutable_heartbeat();
    (*heartbeat->mutable_system_status())["state"] = std::string(1, static_cast<char>(0xff));
    EXPECT_FALSE(FastProtoJsonCodec::TryToJson(invalid_map_utf8, fast_invalid_json));
}

TEST_F(ProtoMessageJsonUtilTest, TestToJsonSimple) {
    { // Empty msg
        SimpleMessage msg;
        std::string json;
        ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&msg, json));
        ASSERT_EQ("{\"int32Value\":0,\"uint32Value\":0,\"int64Value\":\"0\",\"uint64Value\":\"0\",\"doubleValue\":0,"
                  "\"floatValue\":0,\"boolValue\":false,\"stringValue\":\"\"}",
                  json);
    }
    { // full message
        SimpleMessage msg;
        msg.set_int32value(111);
        msg.set_uint32value(222);
        msg.set_int64value(333);
        msg.set_uint64value(444);
        msg.set_doublevalue(555.555);
        msg.set_floatvalue(666.666);
        msg.set_boolvalue(true);
        msg.set_stringvalue("hello");
        std::string json;
        std::string expected(
            "{\"int32Value\":111,\"uint32Value\":222,\"int64Value\":\"333\",\"uint64Value\":\"444\",\"doubleValue\":"
            "555.555,\"floatValue\":666.666,\"boolValue\":true,\"stringValue\":\"hello\"}");
        ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&msg, json));
        ASSERT_EQ(expected, json);
    }
    { // part message
        SimpleMessage msg;
        msg.set_int32value(111);
        msg.set_uint32value(222);
        msg.set_stringvalue("hello");
        std::string json;
        std::string expected("{\"int32Value\":111,\"uint32Value\":222,\"int64Value\":\"0\",\"uint64Value\":\"0\","
                             "\"doubleValue\":0,\"floatValue\":0,\"boolValue\":false,\"stringValue\":\"hello\"}");
        ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&msg, json));
        ASSERT_EQ(expected, json);
    }
}

TEST_F(ProtoMessageJsonUtilTest, TestToJsonNullPtr) {
    std::string json;
    ASSERT_FALSE(ProtoMessageJsonUtil::ToJson(nullptr, json));
    ASSERT_EQ("", json);
}

TEST_F(ProtoMessageJsonUtilTest, TestToJsonPreservesProtobufAppendBehavior) {
    SimpleMessage message;
    std::string json = "prefix:";
    ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&message, json));
    EXPECT_EQ("prefix:{\"int32Value\":0,\"uint32Value\":0,\"int64Value\":\"0\",\"uint64Value\":\"0\","
              "\"doubleValue\":0,\"floatValue\":0,\"boolValue\":false,\"stringValue\":\"\"}",
              json);
}

TEST_F(ProtoMessageJsonUtilTest, TestToJsonEnum) {
    { // normal
        EnumMessage msg;
        msg.set_enumvalue(TEST_ENUM_1);
        std::string json;
        ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&msg, json));
        ASSERT_EQ("{\"enumValue\":\"TEST_ENUM_1\"}", json);
    }
    { // unspecified
        EnumMessage msg;
        msg.set_enumvalue(TEST_NUM_UNSPECIFIED);
        std::string json;
        ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&msg, json));
        ASSERT_EQ("{\"enumValue\":\"TEST_NUM_UNSPECIFIED\"}", json);
    }
}

TEST_F(ProtoMessageJsonUtilTest, TestToJsonOneOf) {
    {
        OneOfMeaaage msg;
        msg.set_v1(11);
        std::string json;
        ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&msg, json));
        ASSERT_EQ("{\"v1\":\"11\"}", json);
    }
    {
        OneOfMeaaage msg;
        msg.set_v2(22);
        std::string json;
        ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&msg, json));
        ASSERT_EQ("{\"v2\":\"22\"}", json);
    }
    {
        OneOfMeaaage one_of_msg;
        SimpleMessage *simple_msg = new SimpleMessage;
        simple_msg->set_int32value(111);
        simple_msg->set_uint32value(222);
        simple_msg->set_stringvalue("hello");
        one_of_msg.set_allocated_v3(simple_msg);
        std::string json;
        ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&one_of_msg, json));
        ASSERT_EQ("{\"v3\":{\"int32Value\":111,\"uint32Value\":222,\"int64Value\":\"0\",\"uint64Value\":\"0\","
                  "\"doubleValue\":0,\"floatValue\":0,\"boolValue\":false,\"stringValue\":\"hello\"}}",
                  json);
    }
}

TEST_F(ProtoMessageJsonUtilTest, TestToJsonRepeated) {
    {
        RepeatMessage repeated_msg;
        std::string json;
        repeated_msg.add_int32vec(1);
        repeated_msg.add_int32vec(2);
        ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&repeated_msg, json));
        ASSERT_EQ("{\"int32Vec\":[1,2],\"simpleMsgVec\":[],\"enumMsgVec\":[],\"oneOfVec\":[]}", json);
    }
    {
        RepeatMessage repeated_msg;
        std::string json;
        repeated_msg.add_int32vec(1);
        repeated_msg.add_simplemsgvec();
        ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&repeated_msg, json));
        ASSERT_EQ("{\"int32Vec\":[1],\"simpleMsgVec\":[{\"int32Value\":0,\"uint32Value\":0,\"int64Value\":\"0\","
                  "\"uint64Value\":\"0\",\"doubleValue\":0,\"floatValue\":0,\"boolValue\":false,\"stringValue\":\"\"}],"
                  "\"enumMsgVec\":[],\"oneOfVec\":[]}",
                  json);
    }
    {
        RepeatMessage repeated_msg;
        std::string json;
        repeated_msg.add_int32vec(1);
        repeated_msg.add_simplemsgvec();
        repeated_msg.add_simplemsgvec();
        auto simple_value_1 = repeated_msg.mutable_simplemsgvec(0);
        simple_value_1->set_int32value(100);
        auto simple_value_2 = repeated_msg.mutable_simplemsgvec(1);
        simple_value_2->set_doublevalue(111.111);
        ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&repeated_msg, json));
        ASSERT_EQ("{\"int32Vec\":[1],\"simpleMsgVec\":[{\"int32Value\":100,\"uint32Value\":0,\"int64Value\":\"0\","
                  "\"uint64Value\":\"0\",\"doubleValue\":0,\"floatValue\":0,\"boolValue\":false,\"stringValue\":\"\"},"
                  "{\"int32Value\":0,\"uint32Value\":0,\"int64Value\":\"0\",\"uint64Value\":\"0\","
                  "\"doubleValue\":111.111,\"floatValue\":0,\"boolValue\":false,\"stringValue\":\"\"}],"
                  "\"enumMsgVec\":[],\"oneOfVec\":[]}",
                  json);
    }
    {
        RepeatMessage repeated_msg;
        std::string json;
        repeated_msg.add_int32vec(1);
        repeated_msg.add_simplemsgvec();
        repeated_msg.add_simplemsgvec();
        auto simple_value_1 = repeated_msg.mutable_simplemsgvec(0);
        simple_value_1->set_int32value(100);
        repeated_msg.add_enummsgvec();
        repeated_msg.add_enummsgvec();
        auto enum_value_1 = repeated_msg.mutable_enummsgvec(0);
        enum_value_1->set_enumvalue(TEST_ENUM_2);
        auto enum_value_2 = repeated_msg.mutable_enummsgvec(1);
        enum_value_2->set_enumvalue(TEST_ENUM_3);
        ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&repeated_msg, json));
        ASSERT_EQ("{\"int32Vec\":[1],\"simpleMsgVec\":[{\"int32Value\":100,\"uint32Value\":0,\"int64Value\":\"0\","
                  "\"uint64Value\":\"0\",\"doubleValue\":0,\"floatValue\":0,\"boolValue\":false,\"stringValue\":\"\"},"
                  "{\"int32Value\":0,\"uint32Value\":0,\"int64Value\":\"0\",\"uint64Value\":\"0\",\"doubleValue\":0,"
                  "\"floatValue\":0,\"boolValue\":false,\"stringValue\":\"\"}],"
                  "\"enumMsgVec\":[{\"enumValue\":\"TEST_ENUM_2\"},{\"enumValue\":\"TEST_ENUM_3\"}],\"oneOfVec\":[]}",
                  json);
    }
    {
        RepeatMessage repeated_msg;
        std::string json;
        repeated_msg.add_int32vec(1);
        repeated_msg.add_simplemsgvec();
        repeated_msg.add_simplemsgvec();
        auto simple_value_1 = repeated_msg.mutable_simplemsgvec(0);
        simple_value_1->set_int32value(100);
        repeated_msg.add_enummsgvec();
        repeated_msg.add_enummsgvec();
        auto enum_value_1 = repeated_msg.mutable_enummsgvec(0);
        enum_value_1->set_enumvalue(TEST_ENUM_2);
        auto enum_value_2 = repeated_msg.mutable_enummsgvec(1);
        enum_value_2->set_enumvalue(TEST_NUM_UNSPECIFIED);
        repeated_msg.add_oneofvec();
        repeated_msg.add_oneofvec();
        repeated_msg.add_oneofvec();
        auto oneof_value_1 = repeated_msg.mutable_oneofvec(0);
        oneof_value_1->set_v1(1);
        SimpleMessage *simple_msg = new SimpleMessage;
        simple_msg->set_int32value(111);
        simple_msg->set_uint32value(222);
        simple_msg->set_stringvalue("hello");
        auto oneof_value_2 = repeated_msg.mutable_oneofvec(2);
        oneof_value_2->set_allocated_v3(simple_msg);
        ASSERT_TRUE(ProtoMessageJsonUtil::ToJson(&repeated_msg, json));
        ASSERT_EQ("{\"int32Vec\":[1],\"simpleMsgVec\":[{\"int32Value\":100,\"uint32Value\":0,\"int64Value\":\"0\","
                  "\"uint64Value\":\"0\",\"doubleValue\":0,\"floatValue\":0,\"boolValue\":false,\"stringValue\":\"\"},"
                  "{\"int32Value\":0,\"uint32Value\":0,\"int64Value\":\"0\",\"uint64Value\":\"0\",\"doubleValue\":0,"
                  "\"floatValue\":0,\"boolValue\":false,\"stringValue\":\"\"}],"
                  "\"enumMsgVec\":[{\"enumValue\":\"TEST_ENUM_2\"},{\"enumValue\":\"TEST_NUM_UNSPECIFIED\"}],"
                  "\"oneOfVec\":[{\"v1\":\"1\"},{},{\"v3\":{\"int32Value\":111,\"uint32Value\":222,"
                  "\"int64Value\":\"0\",\"uint64Value\":\"0\",\"doubleValue\":0,\"floatValue\":0,\"boolValue\":false,"
                  "\"stringValue\":\"hello\"}}]}",
                  json);
    }
}

TEST_F(ProtoMessageJsonUtilTest, TestFromJsonSimple) {
    { // Empty msg
        SimpleMessage msg;
        std::string json = "{}";
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_EQ(0, msg.int32value());
        ASSERT_EQ(0, msg.uint32value());
        ASSERT_EQ(0, msg.int64value());
        ASSERT_EQ(0, msg.uint64value());
        ASSERT_EQ(0, msg.doublevalue());
        ASSERT_EQ(0, msg.floatvalue());
        ASSERT_EQ(false, msg.boolvalue());
        ASSERT_EQ(std::string(), msg.stringvalue());
    }
    { // full message
        SimpleMessage msg;
        std::string json(
            "{\"int32Value\":111,\"uint32Value\":222,\"int64Value\":\"333\",\"uint64Value\":\"444\",\"doubleValue\":"
            "555.555,\"floatValue\":666.666,\"boolValue\":true,\"stringValue\":\"hello\"}");
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_EQ(111, msg.int32value());
        ASSERT_EQ(222, msg.uint32value());
        ASSERT_EQ(333, msg.int64value());
        ASSERT_EQ(444, msg.uint64value());
        ASSERT_EQ(555.555, msg.doublevalue());
        ASSERT_EQ((float)666.666, msg.floatvalue());
        ASSERT_EQ(true, msg.boolvalue());
        ASSERT_EQ(std::string("hello"), msg.stringvalue());
    }
    { // part message
        SimpleMessage msg;
        std::string json("{\"int32Value\":111,\"uint32Value\":222,\"stringValue\":\"hello\"}");
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_EQ(111, msg.int32value());
        ASSERT_EQ(222, msg.uint32value());
        ASSERT_EQ(0, msg.int64value());
        ASSERT_EQ(0, msg.uint64value());
        ASSERT_EQ(0, msg.doublevalue());
        ASSERT_EQ(0, msg.floatvalue());
        ASSERT_EQ(false, msg.boolvalue());
        ASSERT_EQ(std::string("hello"), msg.stringvalue());
    }
    { // unknowns field
        SimpleMessage msg;
        std::string json("{\"int32Value\":111,\"uint32Value\":222,\"stringValue\":\"hello\","
                         " \"unknown\":\"unknown\"}");
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_EQ(111, msg.int32value());
        ASSERT_EQ(222, msg.uint32value());
        ASSERT_EQ(false, msg.boolvalue());
        ASSERT_EQ(std::string("hello"), msg.stringvalue());
    }
    { // null
        SimpleMessage msg;
        std::string json = "{\"int32Value\":null}";
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_EQ(0, msg.int32value());
    }
}

TEST_F(ProtoMessageJsonUtilTest, TestFromJsonError) {
    {
        SimpleMessage msg;
        msg.set_int32value(123);
        std::string json = "{\"int32Value\":111,";
        ASSERT_FALSE(ProtoMessageJsonUtil::FromJson(json, &msg));
        EXPECT_EQ(123, msg.int32value());
    }
    {
        SimpleMessage msg;
        msg.set_int32value(123);
        std::string json = "{\"int32Value\":\"not_int\"}";
        ASSERT_FALSE(ProtoMessageJsonUtil::FromJson(json, &msg));
        EXPECT_EQ(123, msg.int32value());
    }
}

TEST_F(ProtoMessageJsonUtilTest, TestFromJsonHonorsNonNullTerminatedViewBounds) {
    const std::string json = R"({"int32Value":123,"stringValue":"bounded"})";
    const std::string prefix = "ignored-prefix";
    const std::string backing = prefix + json + "!invalid-trailing-bytes";
    const std::string_view bounded(backing.data() + prefix.size(), json.size());

    SimpleMessage msg;
    ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(bounded, &msg));
    EXPECT_EQ(123, msg.int32value());
    EXPECT_EQ("bounded", msg.stringvalue());

    EXPECT_FALSE(ProtoMessageJsonUtil::FromJson(std::string_view(), &msg));
    EXPECT_FALSE(ProtoMessageJsonUtil::FromJson(bounded, nullptr));
}

TEST_F(ProtoMessageJsonUtilTest, TestReportEventFastJsonParserMatchesGenericParser) {
    const std::string json = R"json({
        "trace_id":"trace-fast",
        "instanceId":"instance-fast",
        "host_ip_port":"10.0.0.8:8080",
        "events":[
            {"eventType":"EVENT_NODE_REGISTER","nodeRegister":{"mediums":["mem","disk"],"ignored":1}},
            {"event_type":"EVENT_BLOCK_ADD","block_add":{"blockKey":"-1","uri":"legacy://uri","medium":"mem","specs":[{"name":"tp0","uri":"event_report://host/mem"},{"name":"tp1","uri":"event_report://host/mem?part=1"}]}},
            {"event_type":3,"blockDelete":{"block_key":"2","medium":"disk","specNames":["tp0","tp1"]}},
            {"event_type":"EVENT_HOST_DOWN","hostDown":{"ignored":{"nested":true}}},
            {"event_type":"EVENT_HEARTBEAT","heartbeat":{"systemStatus":{"state":"ready","load":"7"}}},
            {"event_type":"EVENT_BLOCK_SNAPSHOT","blockSnapshot":{"medium":"legacy","blocks":[{"blockKey":"3","medium":"mem","specs":[{"name":"tp0","uri":"event_report://host/mem?block=3"}]}]}}
        ],
        "storageType":8,
        "ignored_top_level":{"deep":[1,2,3]}
    })json";

    proto::meta::ReportEventRequest generic;
    ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &generic));

    proto::meta::ReportEventRequest fast;
    fast.set_trace_id("must-be-cleared");
    ASSERT_TRUE(ReportEventJsonParser::TryFromJson(json, &fast));
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(generic, fast));

    const std::string prefix = "ignored-prefix";
    const std::string backing = prefix + json + "!invalid-trailing-bytes";
    const std::string_view bounded(backing.data() + prefix.size(), json.size());
    proto::meta::ReportEventRequest bounded_fast;
    ASSERT_TRUE(ReportEventJsonParser::TryFromJson(bounded, &bounded_fast));
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(generic, bounded_fast));

    std::string mutable_small = json;
    proto::meta::ReportEventRequest mutable_small_fast;
    ASSERT_TRUE(ReportEventJsonParser::FromMutableNullTerminatedJson(
        mutable_small.data(), mutable_small.size(), &mutable_small_fast));
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(generic, mutable_small_fast));

    // Force the HTTP-only in-situ path with the same complete event matrix.
    // The existing small body above exercises the immutable compatibility
    // path, while production ReportEvent batches are normally larger than the
    // 32-KiB threshold and mutate cinatra's request-owned buffer directly.
    std::string large_json = json;
    ASSERT_EQ('}', large_json.back());
    large_json.pop_back();
    large_json += R"json(,"ignored_large_padding":")json";
    large_json.append(40 * 1024, 'p');
    large_json += R"json("})json";
    proto::meta::ReportEventRequest generic_large;
    ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(large_json, &generic_large));
    std::string mutable_large = large_json;
    proto::meta::ReportEventRequest mutable_large_fast;
    ASSERT_TRUE(ReportEventJsonParser::FromMutableNullTerminatedJson(
        mutable_large.data(), mutable_large.size(), &mutable_large_fast));
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(generic_large, mutable_large_fast));
    EXPECT_NE(large_json, mutable_large);
}

TEST_F(ProtoMessageJsonUtilTest, TestReportEventFastJsonParserFallsBackForCompatibleRareShapes) {
    const std::string json = R"json({
        "trace_id":null,
        "instance_id":"fallback-instance",
        "host_ip_port":"host:8080",
        "events":[{"event_type":"FUTURE_EVENT_NAME","heartbeat":{"system_status":{}}}],
        "storage_type":"ST_EVENT_REPORT_L2"
    })json";

    proto::meta::ReportEventRequest fast_attempt;
    EXPECT_FALSE(ReportEventJsonParser::TryFromJson(json, &fast_attempt));

    proto::meta::ReportEventRequest generic;
    ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &generic));
    proto::meta::ReportEventRequest with_fallback;
    ASSERT_TRUE(ReportEventJsonParser::FromJson(json, &with_fallback));
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(generic, with_fallback));
    EXPECT_EQ(proto::meta::EVENT_UNSPECIFIED, with_fallback.events(0).event_type());

    std::string mutable_json = json;
    mutable_json.pop_back();
    mutable_json += R"json(,"ignored_padding":")json";
    mutable_json.append(40 * 1024, 'p');
    mutable_json += R"json("})json";
    proto::meta::ReportEventRequest generic_large;
    ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(mutable_json, &generic_large));
    proto::meta::ReportEventRequest mutable_with_fallback;
    ASSERT_TRUE(ReportEventJsonParser::FromMutableNullTerminatedJson(
        mutable_json.data(), mutable_json.size(), &mutable_with_fallback));
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(generic_large, mutable_with_fallback));

    EXPECT_FALSE(ReportEventJsonParser::FromJson("{\"events\":[", &with_fallback));
    EXPECT_FALSE(ReportEventJsonParser::FromJson("{}", nullptr));
}

TEST_F(ProtoMessageJsonUtilTest, TestReportEventFastJsonParserValidatesNonAsciiInput) {
    const std::string unicode_json = R"json({
        "trace_id":"追踪",
        "instance_id":"实例",
        "host_ip_port":"host:8080",
        "events":[{"event_type":"EVENT_BLOCK_ADD","block_add":{
            "block_key":"1","medium":"内存","specs":[{"name":"分片","uri":"event_report://host/内存?标签=值"}]
        }}],
        "storage_type":"ST_EVENT_REPORT_L2"
    })json";
    proto::meta::ReportEventRequest generic;
    proto::meta::ReportEventRequest fast;
    ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(unicode_json, &generic));
    ASSERT_TRUE(ReportEventJsonParser::TryFromJson(unicode_json, &fast));
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(generic, fast));

    std::string invalid_json = "{\"trace_id\":\"";
    invalid_json.push_back(static_cast<char>(0xff));
    invalid_json += "\",\"instance_id\":\"i\",\"host_ip_port\":\"h\",\"events\":[],\"storage_type\":8}";
    EXPECT_FALSE(ReportEventJsonParser::TryFromJson(invalid_json, &fast));
    EXPECT_FALSE(ReportEventJsonParser::FromJson(invalid_json, &fast));
}

TEST_F(ProtoMessageJsonUtilTest, TestReportEventLargeInSituJsonParserPreservesEscapesAndRejectsRawNul) {
    std::string json = R"json({
        "trace_id":"quote:\" slash:\\ newline:\n nul:\u0000 snowman:\u2603",
        "instanceId":"large-insitu-instance",
        "host_ip_port":"host:8080",
        "events":[{"eventType":"EVENT_BLOCK_ADD","blockAdd":{
            "blockKey":"7","medium":"mem","specs":[{
                "name":"spec\u005f0","uri":"event_report://host/mem?escaped=a%5C%22b"
            }]
        }}],
        "storageType":"ST_EVENT_REPORT_L2",
        "ignored_padding":")json";
    json.append(40 * 1024, 'p');
    json += R"json("})json";
    ASSERT_GT(json.size(), 32U * 1024U);

    proto::meta::ReportEventRequest generic;
    proto::meta::ReportEventRequest fast;
    ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &generic));
    for (int iteration = 0; iteration < 4; ++iteration) {
        ASSERT_TRUE(ReportEventJsonParser::TryFromJson(json, &fast));
        EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(generic, fast));
    }

    std::string mutable_json = json;
    proto::meta::ReportEventRequest mutable_fast;
    ASSERT_TRUE(
        ReportEventJsonParser::FromMutableNullTerminatedJson(mutable_json.data(), mutable_json.size(), &mutable_fast));
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(generic, mutable_fast));

    std::string non_terminated_view = json + "!";
    proto::meta::ReportEventRequest non_terminated_fast;
    ASSERT_TRUE(ReportEventJsonParser::FromMutableNullTerminatedJson(
        non_terminated_view.data(), json.size(), &non_terminated_fast));
    EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(generic, non_terminated_fast));
    EXPECT_EQ('!', non_terminated_view[json.size()]);
    EXPECT_NE(std::string::npos, fast.trace_id().find('\0'));
    EXPECT_EQ("spec_0", fast.events(0).block_add().specs(0).name());

    std::string raw_nul = json;
    raw_nul.insert(raw_nul.size() - 1, 1, '\0');
    EXPECT_FALSE(ReportEventJsonParser::TryFromJson(raw_nul, &fast));
    EXPECT_FALSE(ReportEventJsonParser::FromJson(raw_nul, &fast));
    EXPECT_FALSE(ReportEventJsonParser::FromMutableNullTerminatedJson(raw_nul.data(), raw_nul.size(), &fast));
}

TEST_F(ProtoMessageJsonUtilTest, TestReportEventJsonParserCompatibilityCorpusMatchesProtobufParser) {
    struct CompatibilityCase {
        const char *name;
        std::string json;
    };
    const std::vector<CompatibilityCase> cases{
        {"empty", R"json({})json"},
        {"canonical",
         R"json({"trace_id":"t","instance_id":"i","host_ip_port":"h:1","events":[{"event_type":"EVENT_BLOCK_ADD","block_add":{"block_key":"1","medium":"mem","specs":[{"name":"tp0","uri":"event_report://h:1/mem?size=1"}]}}],"storage_type":"ST_EVENT_REPORT_L2"})json"},
        {"camel_case_and_numeric_enums",
         R"json({"traceId":"t","instanceId":"i","hostIpPort":"h:1","events":[{"eventType":5,"heartbeat":{"systemStatus":{"state":"ready"}}}],"storageType":8})json"},
        {"quoted_numeric_enums",
         R"json({"trace_id":"t","instance_id":"i","host_ip_port":"h:1","events":[{"event_type":"5","heartbeat":{"system_status":{"state":"ready"}}}],"storage_type":"8"})json"},
        {"known_null_fields",
         R"json({"trace_id":null,"instance_id":"i","host_ip_port":"h:1","events":null,"storage_type":null})json"},
        {"unknown_enum_names",
         R"json({"instance_id":"i","host_ip_port":"h:1","events":[{"event_type":"EVENT_ADDED_LATER","heartbeat":{"system_status":{}}}],"storage_type":"ST_ADDED_LATER"})json"},
        {"numeric_string_fields",
         R"json({"instance_id":"i","host_ip_port":"h:1","events":[{"event_type":"EVENT_BLOCK_ADD","block_add":{"block_key":18446744073709551615,"medium":"mem","specs":[]}}],"storage_type":8})json"},
        {"duplicate_aliases",
         R"json({"trace_id":"first","traceId":"second","instance_id":"i","host_ip_port":"h:1","events":[],"storage_type":8})json"},
        {"duplicate_map_keys",
         R"json({"instance_id":"i","host_ip_port":"h:1","events":[{"event_type":5,"heartbeat":{"system_status":{"state":"first","state":"second"}}}],"storage_type":8})json"},
        {"escaped_duplicate_map_keys",
         R"json({"instance_id":"i","host_ip_port":"h:1","events":[{"event_type":5,"heartbeat":{"system_status":{"state":"first","\u0073tate":"second"}}}],"storage_type":8})json"},
        {"unknown_numeric_enums",
         R"json({"instance_id":"i","host_ip_port":"h:1","events":[{"event_type":99,"heartbeat":{}}],"storage_type":99})json"},
        {"multiple_oneof_members",
         R"json({"instance_id":"i","host_ip_port":"h:1","events":[{"event_type":2,"heartbeat":{},"block_add":{"block_key":"1","medium":"mem","specs":[]}}],"storage_type":8})json"},
        {"payload_before_event_type",
         R"json({"instance_id":"i","host_ip_port":"h:1","events":[{"block_add":{"specs":[{"uri":"event_report://h:1/mem","name":"tp0"}],"medium":"mem","block_key":"1"},"event_type":2}],"storage_type":8})json"},
        {"duplicate_nested_field_aliases",
         R"json({"instance_id":"i","host_ip_port":"h:1","events":[{"event_type":2,"block_add":{"block_key":"1","blockKey":"2","medium":"mem","specs":[]}}],"storage_type":8})json"},
        {"duplicate_unknown_members",
         R"json({"instance_id":"i","host_ip_port":"h:1","events":[],"storage_type":8,"future":1,"future":2})json"},
        {"unknown_nested_values",
         R"json({"instance_id":"i","host_ip_port":"h:1","events":[{"event_type":4,"host_down":{"future":{"array":[1,true,null,{"x":"y"}]}}}],"storage_type":8,"future_top":{"deep":{"value":1}}})json"},
        {"escaped_unicode",
         R"json({"trace_id":"pair:\ud83d\ude80 nul:\u0000","instance_id":"\u5b9e\u4f8b","host_ip_port":"h:1","events":[{"event_type":1,"node_register":{"mediums":["m\u00e9m"]}}],"storage_type":8})json"},
        {"unpaired_surrogate",
         R"json({"trace_id":"bad:\ud800","instance_id":"i","host_ip_port":"h:1","events":[],"storage_type":8})json"},
        {"wrong_known_field_types",
         R"json({"trace_id":7,"instance_id":"i","host_ip_port":"h:1","events":{},"storage_type":true})json"},
        {"null_oneof_and_repeated_entries",
         R"json({"instance_id":"i","host_ip_port":"h:1","events":[null,{"event_type":2,"block_add":null},{"event_type":2,"block_add":{"block_key":"1","medium":"mem","specs":[null]}}],"storage_type":8})json"},
    };

    for (const auto &test_case : cases) {
        SCOPED_TRACE(test_case.name);
        proto::meta::ReportEventRequest protobuf_parsed;
        const bool protobuf_ok = ProtobufFromJson(test_case.json, &protobuf_parsed);

        // The fast codec may conservatively reject a compatible shape, but it
        // must never accept input with semantics different from protobuf.
        proto::meta::ReportEventRequest fast_codec_parsed;
        const bool fast_codec_ok = FastProtoJsonCodec::TryFromJson(test_case.json, &fast_codec_parsed);
        if (fast_codec_ok) {
            ASSERT_TRUE(protobuf_ok);
            EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(protobuf_parsed, fast_codec_parsed));
        }

        proto::meta::ReportEventRequest generic;
        const bool generic_ok = ProtoMessageJsonUtil::FromJson(test_case.json, &generic);
        EXPECT_EQ(protobuf_ok, generic_ok);
        if (protobuf_ok && generic_ok) {
            EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(protobuf_parsed, generic));
        }

        proto::meta::ReportEventRequest compatible;
        compatible.set_trace_id("must-be-cleared");
        const bool compatible_ok = ReportEventJsonParser::FromJson(test_case.json, &compatible);
        EXPECT_EQ(protobuf_ok, compatible_ok);
        if (protobuf_ok && compatible_ok) {
            EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(protobuf_parsed, compatible));
        }

        // Exercise the HTTP-only in-situ path as well. Appending an ignored
        // field keeps the protobuf meaning unchanged while forcing the body
        // above the mutable parser's 32-KiB threshold. This is especially
        // important for rare shapes: after RapidJSON mutates the source, the
        // compatibility fallback must reconstruct exactly the same protobuf
        // semantics from the complete DOM.
        ASSERT_FALSE(test_case.json.empty());
        ASSERT_EQ('}', test_case.json.back());
        std::string large_json = test_case.json;
        large_json.pop_back();
        if (large_json.back() != '{') {
            large_json.push_back(',');
        }
        large_json += R"json("ignored_padding":")json";
        large_json.append(40 * 1024, 'p');
        large_json += R"json("})json";

        proto::meta::ReportEventRequest protobuf_large;
        const bool protobuf_large_ok = ProtobufFromJson(large_json, &protobuf_large);
        EXPECT_EQ(protobuf_ok, protobuf_large_ok);

        proto::meta::ReportEventRequest fast_codec_large;
        const bool fast_codec_large_ok = FastProtoJsonCodec::TryFromJson(large_json, &fast_codec_large);
        if (fast_codec_large_ok) {
            ASSERT_TRUE(protobuf_large_ok);
            EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(protobuf_large, fast_codec_large));
        }

        proto::meta::ReportEventRequest generic_large;
        const bool generic_large_ok = ProtoMessageJsonUtil::FromJson(large_json, &generic_large);
        EXPECT_EQ(protobuf_large_ok, generic_large_ok);
        if (protobuf_large_ok && generic_large_ok) {
            EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(protobuf_large, generic_large));
        }

        proto::meta::ReportEventRequest mutable_compatible;
        const bool mutable_ok = ReportEventJsonParser::FromMutableNullTerminatedJson(
            large_json.data(), large_json.size(), &mutable_compatible);
        EXPECT_EQ(protobuf_large_ok, mutable_ok);
        if (protobuf_large_ok && mutable_ok) {
            EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(protobuf_large, mutable_compatible));
        }
    }
}

TEST_F(ProtoMessageJsonUtilTest, TestFromJsonEnum) {
    { // normal
        EnumMessage msg;
        std::string json("{\"enumValue\":\"TEST_ENUM_1\"}");
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_EQ(TEST_ENUM_1, msg.enumvalue());
    }
    { // unspecified
        EnumMessage msg;
        std::string json("{\"enumValue\":\"TEST_ENUM_NOT_EXIST\"}");
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_EQ(TEST_NUM_UNSPECIFIED, msg.enumvalue());
    }
    { // unspecified
        EnumMessage msg;
        std::string json("{\"not_exist\":\"TEST_ENUM_NOT_EXIST\"}");
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_EQ(TEST_NUM_UNSPECIFIED, msg.enumvalue());
    }
}

TEST_F(ProtoMessageJsonUtilTest, TestFromJsonOneOf) {
    {
        OneOfMeaaage msg;
        std::string json("{\"v1\":\"11\"}");
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_EQ(11, msg.v1());
        ASSERT_TRUE(msg.has_v1());
        ASSERT_FALSE(msg.has_v2());
        ASSERT_FALSE(msg.has_v3());
    }
    {
        OneOfMeaaage msg;
        std::string json("{\"v2\":\"22\"}");
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_EQ(22, msg.v2());
        ASSERT_FALSE(msg.has_v1());
        ASSERT_TRUE(msg.has_v2());
        ASSERT_FALSE(msg.has_v3());
    }
    {
        OneOfMeaaage msg;
        std::string json("{\"v3\":{\"int32Value\":111,\"uint32Value\":222,\"stringValue\":\"hello\"}}");
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_EQ(111, msg.v3().int32value());
        ASSERT_EQ(222, msg.v3().uint32value());
        ASSERT_EQ(std::string("hello"), msg.v3().stringvalue());
        ASSERT_FALSE(msg.has_v1());
        ASSERT_FALSE(msg.has_v2());
        ASSERT_TRUE(msg.has_v3());
    }
    {
        OneOfMeaaage msg;
        std::string json("{}");
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_FALSE(msg.has_v1());
        ASSERT_FALSE(msg.has_v2());
        ASSERT_FALSE(msg.has_v3());
    }
}

TEST_F(ProtoMessageJsonUtilTest, TestFromJsonOneOfError) {
    {
        OneOfMeaaage msg;
        std::string json("{\"v1\":\"11\", \"v2\":\"22\"}");
        ASSERT_FALSE(ProtoMessageJsonUtil::FromJson(json, &msg));
    }
    {
        OneOfMeaaage msg;
        std::string json("{\"v1\":\"11\", \"v3\":{\"int32Value\":111,\"uint32Value\":222,"
                         "\"stringValue\":\"hello\"}}");
        ASSERT_FALSE(ProtoMessageJsonUtil::FromJson(json, &msg));
    }
}

TEST_F(ProtoMessageJsonUtilTest, TestFromJsonRepeated) {
    {
        RepeatMessage msg;
        std::string json("{\"int32Vec\":[1,2]}");
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_EQ(2, msg.int32vec_size());
        ASSERT_EQ(1, msg.int32vec(0));
        ASSERT_EQ(2, msg.int32vec(1));
        ASSERT_EQ(0, msg.simplemsgvec_size());
        ASSERT_EQ(0, msg.enummsgvec_size());
        ASSERT_EQ(0, msg.oneofvec_size());
    }
    {
        RepeatMessage msg;
        std::string json("{\"int32Vec\":[1],\"simpleMsgVec\":[{}]}");
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_EQ(1, msg.int32vec_size());
        ASSERT_EQ(1, msg.int32vec(0));
        ASSERT_EQ(1, msg.simplemsgvec_size());
    }
    {
        RepeatMessage msg;
        std::string json("{\"int32Vec\":[1],\"simpleMsgVec\":[{\"int32Value\":100},{\"doubleValue\":111.111}]}");
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_EQ(1, msg.int32vec_size());
        ASSERT_EQ(1, msg.int32vec(0));
        ASSERT_EQ(2, msg.simplemsgvec_size());
        ASSERT_EQ(100, msg.simplemsgvec(0).int32value());
        ASSERT_EQ(0, msg.simplemsgvec(0).doublevalue());
        ASSERT_EQ(0, msg.simplemsgvec(1).int32value());
        ASSERT_EQ(111.111, msg.simplemsgvec(1).doublevalue());
    }
    {
        RepeatMessage msg;
        std::string json(
            "{\"int32Vec\":[1],\"simpleMsgVec\":[{\"int32Value\":100},{}],\"enumMsgVec\":[{\"enumValue\":\"TEST_"
            "ENUM_2\"},{\"enumValue\":\"TEST_ENUM_3\"}]}");
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_EQ(1, msg.int32vec_size());
        ASSERT_EQ(1, msg.int32vec(0));
        ASSERT_EQ(2, msg.simplemsgvec_size());
        ASSERT_EQ(100, msg.simplemsgvec(0).int32value());
        ASSERT_EQ(0, msg.simplemsgvec(0).doublevalue());
        ASSERT_EQ(0, msg.simplemsgvec(1).int32value());
        ASSERT_EQ(0, msg.simplemsgvec(1).doublevalue());
        ASSERT_EQ(2, msg.enummsgvec_size());
        ASSERT_EQ(TEST_ENUM_2, msg.enummsgvec(0).enumvalue());
        ASSERT_EQ(TEST_ENUM_3, msg.enummsgvec(1).enumvalue());
    }
    {
        RepeatMessage msg;
        std::string json(
            "{\"int32Vec\":[1],\"simpleMsgVec\":[{\"int32Value\":100},{}],\"enumMsgVec\":[{\"enumValue\":\"TEST_"
            "ENUM_2\"},{}],\"oneOfVec\":[{\"v1\":\"1\"},{},{\"v3\":{\"int32Value\":111,\"uint32Value\":222,"
            "\"stringValue\":\"hello\"}}]}");
        ASSERT_TRUE(ProtoMessageJsonUtil::FromJson(json, &msg));
        ASSERT_EQ(1, msg.int32vec_size());
        ASSERT_EQ(1, msg.int32vec(0));
        ASSERT_EQ(2, msg.simplemsgvec_size());
        ASSERT_EQ(100, msg.simplemsgvec(0).int32value());
        ASSERT_EQ(0, msg.simplemsgvec(0).doublevalue());
        ASSERT_EQ(0, msg.simplemsgvec(1).int32value());
        ASSERT_EQ(0, msg.simplemsgvec(1).doublevalue());
        ASSERT_EQ(2, msg.enummsgvec_size());
        ASSERT_EQ(TEST_ENUM_2, msg.enummsgvec(0).enumvalue());
        ASSERT_EQ(TEST_NUM_UNSPECIFIED, msg.enummsgvec(1).enumvalue());
        ASSERT_EQ(3, msg.oneofvec_size());
        {
            ASSERT_TRUE(msg.oneofvec(0).has_v1());
            ASSERT_FALSE(msg.oneofvec(0).has_v2());
            ASSERT_FALSE(msg.oneofvec(0).has_v3());
            ASSERT_EQ(1, msg.oneofvec(0).v1());
        }
        {
            ASSERT_FALSE(msg.oneofvec(1).has_v1());
            ASSERT_FALSE(msg.oneofvec(1).has_v2());
            ASSERT_FALSE(msg.oneofvec(1).has_v3());
        }
        {
            ASSERT_FALSE(msg.oneofvec(2).has_v1());
            ASSERT_FALSE(msg.oneofvec(2).has_v2());
            ASSERT_TRUE(msg.oneofvec(2).has_v3());
            ASSERT_EQ(111, msg.oneofvec(2).v3().int32value());
            ASSERT_EQ(222, msg.oneofvec(2).v3().uint32value());
            ASSERT_EQ("hello", msg.oneofvec(2).v3().stringvalue());
        }
    }
}

} // namespace kv_cache_manager

namespace kv_cache_manager {

TEST_F(ProtoMessageJsonUtilTest, TestModelDeploymentUseEaglePopConversion) {
    ModelDeployment default_model_deployment;
    EXPECT_FALSE(default_model_deployment.use_eagle_pop());

    proto::meta::ModelDeployment proto_model_deployment;
    proto_model_deployment.set_model_name("m");
    proto_model_deployment.set_dtype("fp8");
    proto_model_deployment.set_use_eagle_pop(false);

    ModelDeployment model_deployment;
    ProtoConvert::ModelDeploymentFromProto(&proto_model_deployment, model_deployment);
    EXPECT_FALSE(model_deployment.use_eagle_pop());

    ProtoConvert::ModelDeploymentToProto(model_deployment, &proto_model_deployment);
    EXPECT_FALSE(proto_model_deployment.use_eagle_pop());

    proto_model_deployment.set_use_eagle_pop(true);
    ProtoConvert::ModelDeploymentFromProto(&proto_model_deployment, model_deployment);
    EXPECT_TRUE(model_deployment.use_eagle_pop());
}

} // namespace kv_cache_manager
