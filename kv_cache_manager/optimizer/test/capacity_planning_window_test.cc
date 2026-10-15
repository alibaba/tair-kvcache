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
TEST_F(CapacityPlanningWindowTest, FourWindowsMergeRawSamplesAndDiluteShortBurst) {
    CapacityPlanningWindows windows;
    constexpr int64_t start = 3600000;
    windows.Complete(start);
    for (int slice = 0; slice < 12; ++slice) {
        const int count = slice == 0 ? 1 : 1000;
        for (int sample = 0; sample < count; ++sample) {
            windows.reuse()->Record(slice == 0 ? 36000000000000ULL : 1000000000ULL);
            windows.RecordFact(RequestFact{{{slice == 0 ? 100ULL : 2ULL, 1}}});
        }
        for (int report = 1; report < 10; ++report) {
            for (const auto &result : windows.Complete(start + slice * 300000 + report * 30000)) {
                EXPECT_FALSE(result);
            }
        }
        const auto results = windows.Complete(start + (slice + 1) * 300000);
        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto width = CapacityPlanningWindows::kWindowsMs[i] / 300000;
            ASSERT_EQ((slice + 1) % width == 0, results[i].has_value());
            if (!results[i]) {
                continue;
            }
            const auto &result = *results[i];
            EXPECT_EQ(start + (slice + 1) * 300000 - 1, result.timestamp_ms);
            EXPECT_EQ((slice + 1 == width ? (width - 1) * 1000 + 1 : width * 1000), result.reuse.count);
            if (slice == 0) {
                EXPECT_EQ(36000000000000ULL, result.reuse.p99_ns);
                EXPECT_EQ(100, result.required_blocks);
            } else {
                EXPECT_GE(result.reuse.p99_ns, 1000000000ULL);
                EXPECT_LT(result.reuse.p99_ns, 1008000000ULL);
                EXPECT_EQ(2, result.required_blocks);
            }
        }
        for (const auto &result : windows.Complete(start + (slice + 1) * 300000)) {
            EXPECT_FALSE(result); // No repeated merge on report/request boundary.
        }
    }
}

TEST_F(CapacityPlanningWindowTest, FourWindowsLateReportPartialStartupResetAndClockRegression) {
    CapacityPlanningWindows windows;
    windows.Complete(3599999);
    windows.reuse()->Record(100);
    windows.RecordFact(RequestFact{{{9, 1}}});
    const auto results = windows.Complete(10800000);
    for (const auto &result : results) {
        ASSERT_TRUE(result);
        EXPECT_EQ(3599999, result->timestamp_ms);
        EXPECT_EQ(1, result->reuse.count);
        EXPECT_EQ(9, result->required_blocks);
    }
    windows.reuse()->Record(200);
    for (const auto &result : windows.Complete(3600000)) {
        EXPECT_FALSE(result);
    }
    windows.Reset();
    windows.Complete(10800000);
    for (const auto &result : windows.Complete(14400000)) {
        ASSERT_TRUE(result);
        EXPECT_EQ(0, result->reuse.count);
        EXPECT_EQ(0, result->required_blocks);
    }
}
} // namespace kv_cache_manager
