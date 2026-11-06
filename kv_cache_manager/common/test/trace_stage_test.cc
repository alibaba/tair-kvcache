#include <gtest/gtest.h>

#include "kv_cache_manager/common/request_context.h"

namespace kv_cache_manager {
namespace {

TEST(TraceStageTest, DisabledTracingDoesNotCreateNodesOrChangeParent) {
    RequestContext context("ordinary-request");
    {
        ScopedTraceStage stage(&context, "test", "validate");
        EXPECT_EQ(nullptr, context.parent_span_tracer());
        stage.Next("write");
        stage.AddElapsed("read_total", 17);
        stage.AddCounter("keys", 3);
    }
    EXPECT_TRUE(context.EndAndGetSpanTracerDebugStr().empty());
    ScopedTraceStage null_stage(nullptr, "test", "disabled");
    null_stage.Next("still_disabled");
}

TEST(TraceStageTest, StagesRestoreParentsAndPreserveAggregatesOnEarlyExit) {
    RequestContext context("test__kvcm_need_span_tracer");
    auto *root = context.parent_span_tracer();
    const auto run = [&] {
        ScopedTraceStage outer(&context, "test", "snapshot");
        auto *snapshot = context.parent_span_tracer();
        {
            ScopedTraceStage stage(&context, "test", "validate");
            stage.Next("write");
            stage.AddElapsed("read_total", 17);
            stage.AddElapsed("upsert_total", 29);
            stage.AddCounter("keys", 3);
            stage.Finish();
            stage.Finish();
            EXPECT_EQ(snapshot, context.parent_span_tracer());
        }
        return; // The outer span must still be published.
    };
    run();
    EXPECT_EQ(root, context.parent_span_tracer());
    rapidjson::Document json;
    json.Parse(context.EndAndGetSpanTracerDebugStr().c_str());
    ASSERT_FALSE(json.HasParseError());
    const auto &snapshot = json["test__kvcm_need_span_tracer:test__kvcm_need_span_tracer"][0];
    const auto &stages = snapshot["test:snapshot"];
    ASSERT_EQ(2u, stages.Size());
    EXPECT_TRUE(stages[0].HasMember("test:validate"));
    const auto &write = stages[1];
    EXPECT_EQ(3, write["counters"]["keys"].GetInt64());
    const auto &totals = write["test:write"];
    ASSERT_EQ(2u, totals.Size());
    EXPECT_EQ(17, totals[0]["span_time_us"].GetInt64());
    EXPECT_TRUE(totals[0].HasMember("test:read_total"));
    EXPECT_EQ(29, totals[1]["span_time_us"].GetInt64());
    EXPECT_TRUE(totals[1].HasMember("test:upsert_total"));
}

} // namespace
} // namespace kv_cache_manager
