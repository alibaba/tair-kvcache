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