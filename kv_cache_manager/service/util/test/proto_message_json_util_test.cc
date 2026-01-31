#include <algorithm>

#include "kv_cache_manager/common/unittest.h"
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