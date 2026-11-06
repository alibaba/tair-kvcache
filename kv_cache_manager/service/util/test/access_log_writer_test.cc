#include <gtest/gtest.h>
#include <string>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/service/util/access_log_writer.h"
#include "rapidjson/document.h"

namespace kv_cache_manager {

class AccessLogWriterTest : public TESTBASE {};

TEST_F(AccessLogWriterTest, PreservesBasicFieldsWithoutPayloads) {
    RequestContext request_context("trace\"id");
    request_context.set_client_ip("127.0.0.1");
    request_context.set_api_name("GetCacheLocation");
    request_context.set_status_code(RequestContext::kOkStatusCode);
    request_context.set_request_debug_json({R"({"escaped":"\u003c","nested":{"key":1}})", true});
    request_context.set_response_debug_json_generator(
        []() { return RequestContext::JsonFragment{R"({"ok":true,"values":[1,2,3]})", true}; });

    const std::string access_log = AccessLogWriter::Build(request_context);

    rapidjson::Document document;
    document.Parse(access_log.data(), access_log.size());
    ASSERT_FALSE(document.HasParseError());
    ASSERT_TRUE(document.IsObject());
    EXPECT_EQ(7u, document.MemberCount());
    EXPECT_EQ(TimestampUtil::FormatTimestampUs(request_context.request_begin_time_us()),
              document["request_begin_time"].GetString());
    EXPECT_STREQ("127.0.0.1", document["client_ip"].GetString());
    EXPECT_STREQ("trace\"id", document["trace_id"].GetString());
    EXPECT_EQ(request_context.request_id(), document["request_id"].GetString());
    EXPECT_STREQ("GetCacheLocation", document["api_name"].GetString());
    EXPECT_EQ(RequestContext::kOkStatusCode, document["status_code"].GetInt());
    EXPECT_GE(document["request_cost_time_us"].GetInt64(), 0);
    EXPECT_LE(document["request_cost_time_us"].GetInt64(),
              TimestampUtil::GetCurrentTimeUs() - request_context.request_begin_time_us());
    EXPECT_FALSE(document.HasMember("request"));
    EXPECT_FALSE(document.HasMember("response"));
}

TEST_F(AccessLogWriterTest, EmptyPayloadsAreOmitted) {
    RequestContext request_context("trace_id");

    const std::string access_log = AccessLogWriter::Build(request_context);

    rapidjson::Document document;
    document.Parse(access_log.data(), access_log.size());
    ASSERT_FALSE(document.HasParseError());
    ASSERT_TRUE(document.IsObject());
    EXPECT_FALSE(document.HasMember("request"));
    EXPECT_FALSE(document.HasMember("response"));
}

TEST_F(AccessLogWriterTest, DoesNotMaterializeResponseOrPreventTransportReuse) {
    RequestContext request_context("trace_id");
    int response_serializations = 0;
    request_context.set_request_debug_json({R"({"request":1})", true});
    request_context.set_response_debug_json_generator(
        [&response_serializations]() {
            ++response_serializations;
            return RequestContext::JsonFragment{R"({"response":1})", true};
        },
        RequestContext::ResponseJsonKind::kFullMessage);

    AccessLogWriter::Build(request_context);
    AccessLogWriter::Build(request_context);

    EXPECT_EQ(0, response_serializations);
    auto response_json = request_context.TakeReusableResponseJson();
    ASSERT_TRUE(response_json.has_value());
    EXPECT_EQ(R"({"response":1})", *response_json);
    EXPECT_EQ(1, response_serializations);
}

TEST_F(AccessLogWriterTest, InvalidPayloadsDoNotAffectLog) {
    RequestContext request_context("trace_id");
    request_context.set_request_debug_json({R"({"partial":)", false});
    request_context.set_response_debug_json_generator(
        []() { return RequestContext::JsonFragment{R"({"partial":)", false}; });

    const std::string access_log = AccessLogWriter::Build(request_context);

    rapidjson::Document document;
    document.Parse(access_log.data(), access_log.size());
    ASSERT_FALSE(document.HasParseError());
    ASSERT_TRUE(document.IsObject());
    EXPECT_FALSE(document.HasMember("request"));
    EXPECT_FALSE(document.HasMember("response"));
}

} // namespace kv_cache_manager
