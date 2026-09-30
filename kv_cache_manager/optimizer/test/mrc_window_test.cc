#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"
#include "kv_cache_manager/optimizer/metrics/mrc_window.h"

namespace kv_cache_manager {

class MrcWindowTest : public TESTBASE {};

TEST_F(MrcWindowTest, TakesConfiguredCapacityCurve) {
    MrcWindow window;
    FullRequestFact fact;
    fact.hit_curve.push_back({1, 200});

    window.Record(fact);
    const auto curve = window.Take();

    ASSERT_EQ(6, curve.size());
    EXPECT_EQ(6000, curve[0].target_basis_points);
    EXPECT_EQ(120, curve[0].required_blocks);
    EXPECT_EQ(8000, curve[1].target_basis_points);
    EXPECT_EQ(160, curve[1].required_blocks);
    EXPECT_EQ(9000, curve[2].target_basis_points);
    EXPECT_EQ(180, curve[2].required_blocks);
    EXPECT_EQ(9500, curve[3].target_basis_points);
    EXPECT_EQ(190, curve[3].required_blocks);
    EXPECT_EQ(9900, curve[4].target_basis_points);
    EXPECT_EQ(198, curve[4].required_blocks);
    EXPECT_EQ(9950, curve[5].target_basis_points);
    EXPECT_EQ(199, curve[5].required_blocks);
}

TEST_F(MrcWindowTest, AppliesTargetsAfterAggregatingAllRequestsInWindow) {
    MrcWindow window;
    FullRequestFact large_request;
    large_request.hit_curve.push_back({1, 100});
    FullRequestFact high_capacity_request;
    high_capacity_request.hit_curve.push_back({1000, 1});
    FullRequestFact cold_request;

    window.Record(large_request);
    window.Record(high_capacity_request);
    window.Record(cold_request);
    const auto curve = window.Take();

    const std::vector<uint64_t> expected_required_blocks = {61, 81, 91, 96, 100, 1000};
    ASSERT_EQ(expected_required_blocks.size(), curve.size());
    const uint64_t total_theoretical_hits = HitCurveProjector::ProjectFullInfinite(large_request) +
                                            HitCurveProjector::ProjectFullInfinite(high_capacity_request) +
                                            HitCurveProjector::ProjectFullInfinite(cold_request);
    ASSERT_EQ(101, total_theoretical_hits);

    for (size_t i = 0; i < curve.size(); ++i) {
        EXPECT_EQ(expected_required_blocks[i], curve[i].required_blocks);
        const uint64_t target_hits = (total_theoretical_hits * curve[i].target_basis_points + 9999) / 10000;
        const uint64_t achieved_hits =
            HitCurveProjector::ProjectFullBlocks(large_request, curve[i].required_blocks) +
            HitCurveProjector::ProjectFullBlocks(high_capacity_request, curve[i].required_blocks);
        EXPECT_GE(achieved_hits, target_hits);

        const uint64_t smaller_capacity = curve[i].required_blocks - 1;
        const uint64_t smaller_capacity_hits =
            HitCurveProjector::ProjectFullBlocks(large_request, smaller_capacity) +
            HitCurveProjector::ProjectFullBlocks(high_capacity_request, smaller_capacity);
        EXPECT_LT(smaller_capacity_hits, target_hits);
    }
}

TEST_F(MrcWindowTest, UsesSparseRequiredCapacityPoints) {
    MrcWindow window;
    FullRequestFact fact;
    fact.hit_curve.push_back({1000000000, 1});

    window.Record(fact);
    const auto curve = window.Take();

    ASSERT_EQ(6, curve.size());
    for (const auto &point : curve) {
        EXPECT_EQ(1000000000, point.required_blocks);
    }
}

TEST_F(MrcWindowTest, TakeClearsReportingWindow) {
    MrcWindow window;
    FullRequestFact fact;
    fact.hit_curve.push_back({1, 1});
    window.Record(fact);
    window.Take();

    const auto empty_curve = window.Take();
    ASSERT_EQ(6, empty_curve.size());
    for (const auto &point : empty_curve) {
        EXPECT_EQ(0, point.required_blocks);
    }
}

TEST_F(MrcWindowTest, TakesByteAxisStepCurve) {
    ByteMrcWindow window;
    RequestFact fact;
    // The first byte threshold recovers three blocks at once; the second
    // raises the recoverable prefix from three to five blocks.
    fact.points = {{100, 3}, {250, 5}};

    window.Record(fact);
    const auto curve = window.Take();

    ASSERT_EQ(6, curve.size());
    EXPECT_EQ(6000, curve[0].target_basis_points);
    EXPECT_EQ(100, curve[0].required_bytes); // ceil(5 * 60%) = 3 hits
    for (size_t i = 1; i < curve.size(); ++i) {
        EXPECT_EQ(250, curve[i].required_bytes); // every remaining target needs at least 4 hits
    }
}

TEST_F(MrcWindowTest, AggregatesByteAxisRequestsBeforeApplyingTargets) {
    ByteMrcWindow window;
    RequestFact first;
    first.points = {{10, 2}, {100, 4}};
    RequestFact second;
    second.points = {{50, 3}};

    window.Record(first);
    window.Record(second);
    const auto curve = window.Take();

    const std::vector<uint64_t> expected_required_bytes = {50, 100, 100, 100, 100, 100};
    ASSERT_EQ(expected_required_bytes.size(), curve.size());
    constexpr uint64_t total_theoretical_hits = 7;
    for (size_t i = 0; i < curve.size(); ++i) {
        EXPECT_EQ(expected_required_bytes[i], curve[i].required_bytes);
        const uint64_t target_hits = (total_theoretical_hits * curve[i].target_basis_points + 9999) / 10000;
        const uint64_t achieved_hits = HitCurveProjector::ProjectBytes(first, curve[i].required_bytes) +
                                       HitCurveProjector::ProjectBytes(second, curve[i].required_bytes);
        EXPECT_GE(achieved_hits, target_hits);

        const uint64_t smaller_capacity = curve[i].required_bytes - 1;
        const uint64_t smaller_capacity_hits = HitCurveProjector::ProjectBytes(first, smaller_capacity) +
                                               HitCurveProjector::ProjectBytes(second, smaller_capacity);
        EXPECT_LT(smaller_capacity_hits, target_hits);
    }
}

TEST_F(MrcWindowTest, ByteAxisWindowKeepsLargeThresholdSparseAndClearsOnTake) {
    ByteMrcWindow window;
    RequestFact fact;
    fact.points = {{1000000000000ULL, 1}};

    window.Record(fact);
    auto curve = window.Take();
    ASSERT_EQ(6, curve.size());
    for (const auto &point : curve) {
        EXPECT_EQ(1000000000000ULL, point.required_bytes);
    }

    curve = window.Take();
    ASSERT_EQ(6, curve.size());
    for (const auto &point : curve) {
        EXPECT_EQ(0, point.required_bytes);
    }
}

} // namespace kv_cache_manager
