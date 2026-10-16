#include <algorithm>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/metrics/reuse_interval.h"

namespace kv_cache_manager {

class ReuseIntervalTest : public TESTBASE {};

TEST_F(ReuseIntervalTest, ExactTimesAndSampleWeightedAverage) {
    ReuseInterval intervals;
    intervals.Record(0);
    intervals.Record(1);
    intervals.Record(8);
    const auto stats = intervals.Snapshot();
    EXPECT_EQ(3, stats.count);
    EXPECT_EQ(0, stats.min_ns);
    EXPECT_EQ(8, stats.max_ns);
    EXPECT_DOUBLE_EQ(3.0, stats.avg_ns);
    EXPECT_EQ(8, stats.p95_ns);
    EXPECT_EQ(8, stats.p99_ns);
}

TEST_F(ReuseIntervalTest, TakeClearsWindow) {
    ReuseInterval intervals;
    EXPECT_EQ(0, intervals.Take().count);
    intervals.Record(2);
    intervals.Record(6);
    auto stats = intervals.Take();
    EXPECT_EQ(2, stats.count);
    EXPECT_EQ(2, stats.min_ns);
    EXPECT_EQ(6, stats.max_ns);
    EXPECT_DOUBLE_EQ(4.0, stats.avg_ns);
    EXPECT_EQ(0, intervals.Take().count);

    intervals.Record(10);
    stats = intervals.Take();
    EXPECT_EQ(1, stats.count);
    EXPECT_EQ(10, stats.min_ns);
    EXPECT_EQ(10, stats.max_ns);
    EXPECT_DOUBLE_EQ(10.0, stats.avg_ns);
    EXPECT_EQ(10, stats.p95_ns);
    EXPECT_EQ(10, stats.p99_ns);
}

TEST_F(ReuseIntervalTest, LargeIntervalsDoNotOverflowSum) {
    ReuseInterval intervals;
    const auto value = std::numeric_limits<uint64_t>::max();
    intervals.Record(value);
    intervals.Record(value);
    const auto stats = intervals.Take();
    EXPECT_EQ(2, stats.count);
    EXPECT_EQ(value, stats.min_ns);
    EXPECT_EQ(value, stats.max_ns);
    EXPECT_DOUBLE_EQ(static_cast<double>(value), stats.avg_ns);
    EXPECT_EQ(value, stats.p95_ns);
    EXPECT_EQ(value, stats.p99_ns);
}

TEST_F(ReuseIntervalTest, ResetClearsWindow) {
    ReuseInterval intervals;
    intervals.Record(7);
    intervals.Reset();
    EXPECT_EQ(0, intervals.Snapshot().count);
    intervals.Record(3);
    EXPECT_EQ(3, intervals.Snapshot().min_ns);
    EXPECT_EQ(3, intervals.Snapshot().max_ns);
    EXPECT_DOUBLE_EQ(3.0, intervals.Snapshot().avg_ns);
    EXPECT_EQ(3, intervals.Snapshot().p95_ns);
    EXPECT_EQ(3, intervals.Snapshot().p99_ns);
}

TEST_F(ReuseIntervalTest, PercentilesUseNearestRankAndResetEachWindow) {
    ReuseInterval intervals;
    // Reverse arrival order must not affect the quantiles.
    for (uint64_t value = 100; value > 0; --value) {
        intervals.Record(value);
    }
    auto stats = intervals.Take();
    EXPECT_EQ(95, stats.p95_ns);
    EXPECT_EQ(99, stats.p99_ns);
    EXPECT_EQ(0, intervals.Snapshot().count);
    EXPECT_EQ(0, intervals.Snapshot().p95_ns);
    EXPECT_EQ(0, intervals.Snapshot().p99_ns);

    // Include real zero intervals and repeated samples in the rank.
    for (int i = 0; i < 95; ++i) {
        intervals.Record(0);
    }
    for (int i = 0; i < 4; ++i) {
        intervals.Record(7);
    }
    intervals.Record(100);
    stats = intervals.Take();
    EXPECT_EQ(0, stats.p95_ns);
    EXPECT_EQ(7, stats.p99_ns);
}

TEST_F(ReuseIntervalTest, BucketBoundariesRespectRelativeErrorAcrossUint64Range) {
    // A larger outlier prevents max-clamping from hiding bucket errors.
    for (unsigned exponent = 0; exponent < 64; ++exponent) {
        const uint64_t power = uint64_t{1} << exponent;
        for (uint64_t value : {power - 1, power, power + 1}) {
            ReuseInterval intervals;
            for (int i = 0; i < 100; ++i) {
                intervals.Record(value);
            }
            intervals.Record(std::numeric_limits<uint64_t>::max());
            const auto stats = intervals.Take();
            for (uint64_t quantile : {stats.p95_ns, stats.p99_ns}) {
                EXPECT_GE(quantile, value);
                EXPECT_LE(static_cast<long double>(quantile), static_cast<long double>(value) * (1.0L + 1.0L / 128));
            }
        }
    }
}

TEST_F(ReuseIntervalTest, QuantilesMatchSortedOracleWithinErrorBound) {
    ReuseInterval intervals;
    std::mt19937_64 rng(42);
    std::vector<uint64_t> values;
    for (int i = 0; i < 20000; ++i) {
        // Exercise time scales from nanoseconds through seconds and days.
        const auto value = rng() >> (rng() % 64);
        values.push_back(value);
        intervals.Record(value);
    }
    std::sort(values.begin(), values.end());
    const auto stats = intervals.Snapshot();
    for (const auto &[percent, estimate] : {std::pair<uint64_t, uint64_t>{95, stats.p95_ns}, {99, stats.p99_ns}}) {
        const auto expected = values[(values.size() * percent + 99) / 100 - 1];
        EXPECT_GE(estimate, expected);
        EXPECT_LE(static_cast<long double>(estimate), static_cast<long double>(expected) * (1.0L + 1.0L / 128));
    }
    EXPECT_LE(stats.p95_ns, stats.p99_ns);
    EXPECT_LE(stats.p99_ns, stats.max_ns);
}

} // namespace kv_cache_manager
