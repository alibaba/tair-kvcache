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

// is_replication defaults to false; setter flips it
TEST_F(RequestContextTest, IsReplicationDefaultAndSetter) {
    RequestContext rc("trace_v1_c4");
    EXPECT_FALSE(rc.is_replication());
    rc.set_is_replication(true);
    EXPECT_TRUE(rc.is_replication());
    rc.set_is_replication(false);
    EXPECT_FALSE(rc.is_replication());
}

// caller_supernode_id defaults to empty; setter round-trips
TEST_F(RequestContextTest, CallerSupernodeIdDefaultAndSetter) {
    RequestContext rc("trace_supernode");
    EXPECT_TRUE(rc.caller_supernode_id().empty());
    rc.set_caller_supernode_id("sn-42");
    EXPECT_EQ("sn-42", rc.caller_supernode_id());
}

// caller_node holds both node_id and supernode_id; the struct setter and
// the convenience per-field accessors stay consistent.
TEST_F(RequestContextTest, CallerNodeStructRoundTrips) {
    RequestContext rc("trace_caller_loc");
    EXPECT_TRUE(rc.caller_node().node_id.empty());
    EXPECT_TRUE(rc.caller_node().supernode_id.empty());

    rc.set_caller_node(CallerNode{"worker.9", "sn-7"});
    EXPECT_EQ("worker.9", rc.caller_node().node_id);
    EXPECT_EQ("sn-7", rc.caller_node().supernode_id);
    // Convenience accessors delegate to the same struct.
    EXPECT_EQ("worker.9", rc.caller_node_id());
    EXPECT_EQ("sn-7", rc.caller_supernode_id());

    // Per-field setters mutate the struct in place.
    rc.set_caller_node_id("worker.10");
    EXPECT_EQ("worker.10", rc.caller_node().node_id);
    EXPECT_EQ("sn-7", rc.caller_node().supernode_id);
}

// caller_node_id and is_replication are independent fields
TEST_F(RequestContextTest, CallerNodeIpIndependentOfReplicationFlag) {
    RequestContext rc("trace_v1_c4_2");
    rc.set_caller_node_id("worker.7");
    EXPECT_EQ("worker.7", rc.caller_node_id());
    EXPECT_FALSE(rc.is_replication());
    rc.set_is_replication(true);
    EXPECT_EQ("worker.7", rc.caller_node_id()); // 不被 set_is_replication 改写
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