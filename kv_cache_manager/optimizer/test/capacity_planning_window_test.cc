#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"
#include "kv_cache_manager/optimizer/metrics/capacity_planning_window.h"

namespace kv_cache_manager {
class CapacityPlanningWindowTest : public TESTBASE {};

TEST_F(CapacityPlanningWindowTest, FinalizesOncePerFiveMinutesWithSampleWeightedInputs) {
    CapacityPlanningWindow window;
    constexpr int64_t start = 300000;
    EXPECT_FALSE(window.Complete(start));
    window.reuse()->Record(100000000000ULL);
    window.RecordFact(RequestFact{{{100, 1}}});
    for (int report = 1; report < 10; ++report) {
        EXPECT_FALSE(window.Complete(start + report * 30000));
    }
    for (int i = 0; i < 1000; ++i) {
        window.reuse()->Record(1000000000ULL);
        window.RecordFact(RequestFact{{{2, 1}}});
    }
    const auto result = window.Complete(start + 300000);
    ASSERT_TRUE(result);
    EXPECT_EQ(start + 300000 - 1, result->timestamp_ms);
    EXPECT_EQ(1001, result->reuse.count);
    EXPECT_GE(result->reuse.p99_ns, 1000000000ULL);
    EXPECT_LT(result->reuse.p99_ns, 1008000000ULL);
    EXPECT_EQ(2, result->required_blocks);
    EXPECT_FALSE(window.Complete(start + 300000)); // Exactly once.
    EXPECT_FALSE(window.Complete(start + 330000));
    window.reuse()->Record(0);
    window.RecordFact(RequestFact{{{1, 1}}});
    const auto next = window.Complete(start + 600000);
    ASSERT_TRUE(next);
    EXPECT_EQ(1, next->reuse.count); // No overlap with previous window.
    EXPECT_EQ(0, next->reuse.p99_ns);
    EXPECT_EQ(1, next->required_blocks);
}

TEST_F(CapacityPlanningWindowTest, LateReportsAnchorToOriginalWindowAndSkipIdleGaps) {
    CapacityPlanningWindow window;
    window.Complete(301000);
    window.reuse()->Record(200);
    window.RecordFact(RequestFact{{{9, 1}}});
    const auto result = window.Complete(1800000);
    ASSERT_TRUE(result);
    EXPECT_EQ(599999, result->timestamp_ms); // Not the late report time.
    EXPECT_EQ(1, result->reuse.count);
    EXPECT_FALSE(window.Complete(1800000));
    const auto empty = window.Complete(2100000);
    ASSERT_TRUE(empty);
    EXPECT_EQ(0, empty->reuse.count);
    EXPECT_EQ(0, empty->required_blocks);
    EXPECT_FALSE(window.Complete(300000)); // Clock regression cannot revive.
}

TEST_F(CapacityPlanningWindowTest, PartialStartupAndReset) {
    CapacityPlanningWindow window;
    EXPECT_FALSE(window.Complete(599999));
    window.reuse()->Record(10);
    const auto partial = window.Complete(600000);
    ASSERT_TRUE(partial);
    EXPECT_EQ(1, partial->reuse.count);
    window.reuse()->Record(20);
    window.Reset();
    EXPECT_FALSE(window.Complete(900000));
    const auto result = window.Complete(1200000);
    ASSERT_TRUE(result);
    EXPECT_EQ(0, result->reuse.count);
}
} // namespace kv_cache_manager
