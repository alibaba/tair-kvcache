#include "kv_cache_manager/mrc/lite_hit_core.h"

#include <gtest/gtest.h>
#include <vector>

#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

namespace {

// Hits at capacity C = number of thresholds <= C.
int64_t HitsAt(const std::vector<uint64_t> &thresholds, uint64_t capacity) {
    int64_t hits = 0;
    for (const uint64_t t : thresholds) {
        hits += (t <= capacity) ? 1 : 0;
    }
    return hits;
}

std::vector<int64_t> Chain(int64_t base, int n) {
    // Prefix-chained ids: distinct per (base, position).
    std::vector<int64_t> keys;
    keys.reserve(n);
    for (int i = 0; i < n; ++i) {
        keys.push_back(base * 100000 + i);
    }
    return keys;
}

} // namespace

TEST(LiteHitCoreTest, ColdRequestEmitsNothing) {
    LiteHitCore core;
    std::vector<uint64_t> thresholds;
    core.ProcessRequest(Chain(1, 10), thresholds);
    EXPECT_TRUE(thresholds.empty());
    EXPECT_EQ(10, core.unique_blocks());
}

TEST(LiteHitCoreTest, RepeatedScanThresholdsAreOneToN) {
    // Identical request replayed: block j requires exactly j resident blocks
    // (LiteHit projection: hits(C) = min(C, N) per repeat).
    LiteHitCore core;
    std::vector<uint64_t> thresholds;
    const auto request = Chain(1, 100);
    core.ProcessRequest(request, thresholds);
    ASSERT_TRUE(thresholds.empty());

    core.ProcessRequest(request, thresholds);
    ASSERT_EQ(100u, thresholds.size());
    for (size_t i = 0; i < thresholds.size(); ++i) {
        EXPECT_EQ(i + 1, thresholds[i]);
    }
    EXPECT_EQ(16, HitsAt(thresholds, 16));
    EXPECT_EQ(100, HitsAt(thresholds, 4096));
}

TEST(LiteHitCoreTest, ColdKeyStopsPrefixForEveryCapacity) {
    LiteHitCore core;
    std::vector<uint64_t> thresholds;
    const auto request = Chain(1, 4);
    core.ProcessRequest(request, thresholds);

    // Same 2-block prefix, then a diverging (cold) suffix, then a resident
    // block again: the prefix stops at the cold key and never revives.
    std::vector<int64_t> mixed = {request[0], request[1], 999999, request[2]};
    thresholds.clear();
    core.ProcessRequest(mixed, thresholds);
    EXPECT_EQ(2u, thresholds.size());
}

TEST(LiteHitCoreTest, LeafFirstCommitKeepsChainHeadNewest) {
    LiteHitCore core;
    std::vector<uint64_t> thresholds;
    // Two chains A(4) and B(4); replay A: its head must need fewer resident
    // blocks than its tail (head committed last = newest).
    const auto chain_a = Chain(1, 4);
    const auto chain_b = Chain(2, 4);
    core.ProcessRequest(chain_a, thresholds);
    core.ProcessRequest(chain_b, thresholds);
    ASSERT_TRUE(thresholds.empty());

    core.ProcessRequest(chain_a, thresholds);
    ASSERT_EQ(4u, thresholds.size());
    // A committed before B: A's head has 4 B-blocks plus deeper A-blocks in
    // front of it. Thresholds must be non-decreasing along the chain and the
    // deepest block requires the full working set.
    for (size_t i = 1; i < thresholds.size(); ++i) {
        EXPECT_GE(thresholds[i], thresholds[i - 1]);
    }
    EXPECT_EQ(8u, thresholds.back());
}

TEST(LiteHitCoreTest, SurvivesPositionCompaction) {
    LiteHitCore core(/*initial_capacity=*/32);
    std::vector<uint64_t> thresholds;
    // Push far more distinct blocks than the initial position space.
    for (int r = 0; r < 100; ++r) {
        core.ProcessRequest(Chain(r + 10, 8), thresholds);
    }
    // A recent chain must still be resident with small thresholds.
    thresholds.clear();
    core.ProcessRequest(Chain(109, 8), thresholds);
    ASSERT_EQ(8u, thresholds.size());
    EXPECT_EQ(8u, thresholds.back());
}

TEST(LiteHitCoreTest, BoundedStackStaysExactInsideTrackingCapacity) {
    LiteHitCore bounded(/*initial_capacity=*/16, /*max_tracked_blocks=*/4);
    std::vector<uint64_t> thresholds;
    bounded.ProcessRequest(Chain(1, 8), thresholds);
    EXPECT_EQ(4, bounded.unique_blocks());

    thresholds.clear();
    bounded.ProcessRequest(Chain(1, 8), thresholds);
    ASSERT_EQ(4u, thresholds.size());
    for (size_t i = 0; i < thresholds.size(); ++i) {
        EXPECT_EQ(i + 1, thresholds[i]);
    }
}

} // namespace kv_cache_manager
