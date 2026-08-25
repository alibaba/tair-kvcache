#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

#include "google/protobuf/util/message_differencer.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "kv_cache_manager/service/util/manager_message_proto_util.h"
#include "kv_cache_manager/service/util/report_event_json_parser.h"
#include "service/util/proto_message_json_util.h"
#include "service/util/test/service_util_test.pb.h"

namespace kv_cache_manager {

class ProtoMessageJsonUtilTest : public TESTBASE {
public:
};

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
        std::string json = "{\"int32Value\":111,";
        ASSERT_FALSE(ProtoMessageJsonUtil::FromJson(json, &msg));
    }
    {
        SimpleMessage msg;
        std::string json = "{\"int32Value\":\"not_int\"}";
        ASSERT_FALSE(ProtoMessageJsonUtil::FromJson(json, &msg));
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

TEST_F(ProtoMessageJsonUtilTest, TestReportEventJsonParserCompatibilityCorpusMatchesGenericParser) {
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
        proto::meta::ReportEventRequest generic;
        const bool generic_ok = ProtoMessageJsonUtil::FromJson(test_case.json, &generic);

        proto::meta::ReportEventRequest compatible;
        compatible.set_trace_id("must-be-cleared");
        const bool compatible_ok = ReportEventJsonParser::FromJson(test_case.json, &compatible);
        EXPECT_EQ(generic_ok, compatible_ok);
        if (generic_ok && compatible_ok) {
            EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(generic, compatible));
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

        proto::meta::ReportEventRequest generic_large;
        const bool generic_large_ok = ProtoMessageJsonUtil::FromJson(large_json, &generic_large);
        EXPECT_EQ(generic_ok, generic_large_ok);

        proto::meta::ReportEventRequest mutable_compatible;
        const bool mutable_ok = ReportEventJsonParser::FromMutableNullTerminatedJson(
            large_json.data(), large_json.size(), &mutable_compatible);
        EXPECT_EQ(generic_large_ok, mutable_ok);
        if (generic_large_ok && mutable_ok) {
            EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(generic_large, mutable_compatible));
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
