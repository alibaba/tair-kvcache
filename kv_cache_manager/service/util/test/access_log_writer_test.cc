#include <gtest/gtest.h>
#include <string>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/service/util/access_log_writer.h"
#include "rapidjson/document.h"

namespace kv_cache_manager {

class AccessLogWriterTest : public TESTBASE {};

TEST_F(AccessLogWriterTest, EmbedsTrustedFragmentsWithoutRewriting) {
    RequestContext request_context("trace\"id");
    request_context.set_client_ip("127.0.0.1");
    request_context.set_api_name("GetCacheLocation");
    request_context.set_status_code(RequestContext::kOkStatusCode);
    request_context.set_request_debug_json({R"({"escaped":"\u003c","nested":{"key":1}})", true});
    request_context.set_response_debug_json_generator(
        []() { return RequestContext::JsonFragment{R"({"ok":true,"values":[1,2,3]})", true}; });

    const std::string access_log = AccessLogWriter::Build(request_context);

    // RawValue preserves the trusted fragment bytes instead of parsing and
    // normalizing \u003c into a literal '<'.
    EXPECT_NE(std::string::npos, access_log.find(R"("request":{"escaped":"\u003c","nested":{"key":1}})"));

    rapidjson::Document document;
    document.Parse(access_log.data(), access_log.size());
    ASSERT_FALSE(document.HasParseError());
    ASSERT_TRUE(document.IsObject());
    EXPECT_STREQ("trace\"id", document["trace_id"].GetString());
    ASSERT_TRUE(document["request"].IsObject());
    EXPECT_STREQ("<", document["request"]["escaped"].GetString());
    EXPECT_EQ(1, document["request"]["nested"]["key"].GetInt());
    ASSERT_TRUE(document["response"].IsObject());
    EXPECT_TRUE(document["response"]["ok"].GetBool());
}

TEST_F(AccessLogWriterTest, EmptyFragmentsFallBackToObjects) {
    RequestContext request_context("trace_id");

    const std::string access_log = AccessLogWriter::Build(request_context);

    rapidjson::Document document;
    document.Parse(access_log.data(), access_log.size());
    ASSERT_FALSE(document.HasParseError());
    ASSERT_TRUE(document["request"].IsObject());
    EXPECT_TRUE(document["request"].ObjectEmpty());
    ASSERT_TRUE(document["response"].IsObject());
    EXPECT_TRUE(document["response"].ObjectEmpty());
}

TEST_F(AccessLogWriterTest, MaterializesLazyFragmentsOnce) {
    RequestContext request_context("trace_id");
    int response_serializations = 0;
    request_context.set_request_debug_json({R"({"request":1})", true});
    request_context.set_response_debug_json_generator([&response_serializations]() {
        ++response_serializations;
        return RequestContext::JsonFragment{R"({"response":1})", true};
    });

    AccessLogWriter::Build(request_context);
    AccessLogWriter::Build(request_context);

    EXPECT_EQ(1, response_serializations);
}

TEST_F(AccessLogWriterTest, InvalidFragmentsFallBackToObjects) {
    RequestContext request_context("trace_id");
    request_context.set_request_debug_json({R"({"partial":)", false});
    request_context.set_response_debug_json_generator(
        []() { return RequestContext::JsonFragment{R"({"partial":)", false}; });

    const std::string access_log = AccessLogWriter::Build(request_context);

    rapidjson::Document document;
    document.Parse(access_log.data(), access_log.size());
    ASSERT_FALSE(document.HasParseError());
    ASSERT_TRUE(document["request"].IsObject());
    EXPECT_TRUE(document["request"].ObjectEmpty());
    ASSERT_TRUE(document["response"].IsObject());
    EXPECT_TRUE(document["response"].ObjectEmpty());
}

} // namespace kv_cache_manager
