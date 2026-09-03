// request_context_test.cc

#include <chrono>
#include <gtest/gtest.h>
#include <thread>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

class RequestContextTest : public TESTBASE {
protected:
    void SetUp() override {
        // Setup code if needed
    }
    void TearDown() override {
        // Cleanup code if needed
    }
};

TEST_F(RequestContextTest, TestSimple) {
    RequestContext request_context("fake_trace_id");
    ASSERT_FALSE(request_context.request_id().empty());
    ASSERT_FALSE(request_context.trace_id().empty());
    ASSERT_EQ(nullptr, request_context.metrics_collector());
}

TEST_F(RequestContextTest, ResponseJsonIsMaterializedOnce) {
    RequestContext request_context("fake_trace_id");
    int response_serializations = 0;

    request_context.set_request_debug_json({R"({"request":true})", true});
    request_context.set_response_debug_json_generator(
        [&response_serializations]() {
            ++response_serializations;
            return RequestContext::JsonFragment{R"({"response":true})", true};
        },
        RequestContext::ResponseJsonKind::kFullMessage);

    EXPECT_EQ(0, response_serializations);
    request_context.MaterializeResponseJson();
    EXPECT_EQ(1, response_serializations);

    EXPECT_EQ(R"({"request":true})", request_context.request_debug_json().json);
    EXPECT_EQ(R"({"response":true})", request_context.response_debug_json().json);
    EXPECT_EQ(1, response_serializations);

    auto cached_response = request_context.TakeReusableResponseJson();
    ASSERT_TRUE(cached_response.has_value());
    EXPECT_EQ(R"({"response":true})", *cached_response);
    EXPECT_FALSE(request_context.TakeReusableResponseJson().has_value());
}

TEST_F(RequestContextTest, AccessLogSummaryIsNotReusable) {
    RequestContext request_context("fake_trace_id");
    request_context.set_response_debug_json_generator(
        []() { return RequestContext::JsonFragment{R"({"summary":true})", true}; },
        RequestContext::ResponseJsonKind::kAccessLogSummary);

    EXPECT_EQ(R"({"summary":true})", request_context.response_debug_json().json);
    EXPECT_FALSE(request_context.TakeReusableResponseJson().has_value());
    EXPECT_EQ(R"({"summary":true})", request_context.response_debug_json().json);
}

TEST_F(RequestContextTest, FailedResponseJsonIsNotReusable) {
    RequestContext request_context("fake_trace_id");
    request_context.set_response_debug_json_generator(
        []() { return RequestContext::JsonFragment{R"({"partial":)", false}; },
        RequestContext::ResponseJsonKind::kFullMessage);

    request_context.MaterializeResponseJson();
    EXPECT_FALSE(request_context.response_debug_json().valid);
    EXPECT_FALSE(request_context.TakeReusableResponseJson().has_value());
}

class SpanTracerTest {
public:
    void Function1(RequestContext *request_context) {
        SPAN_TRACER(request_context);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        Function2(request_context);
    }

    void Function2(RequestContext *request_context) {
        SPAN_TRACER(request_context);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        Function3(request_context);
        Function4(request_context);
        Function4(request_context);
    }
    void Function3(RequestContext *request_context) {
        SPAN_TRACER(request_context);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    void Function4(RequestContext *request_context) {
        SPAN_TRACER(request_context);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
};

TEST_F(RequestContextTest, TestSpanTracer) {
    RequestContext request_context("test__kvcm_need_span_tracer");
    SpanTracerTest test;
    test.Function1(&request_context);
    size_t n = 0;
    std::string span_tracer_str = request_context.EndAndGetSpanTracerDebugStr();
    n = span_tracer_str.find("Function1", n + 1);
    ASSERT_NE(std::string::npos, n);
    n = span_tracer_str.find("Function2", n + 1);
    ASSERT_NE(std::string::npos, n);
    n = span_tracer_str.find("Function3", n + 1);
    ASSERT_NE(std::string::npos, n);
    n = span_tracer_str.find("Function4", n + 1);
    ASSERT_NE(std::string::npos, n);
    n = span_tracer_str.find("Function4", n + 1);
    ASSERT_NE(std::string::npos, n);
    n = span_tracer_str.find("Function4", n + 1);
    ASSERT_EQ(std::string::npos, n);
}

} // namespace kv_cache_manager
