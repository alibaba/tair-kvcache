#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/online_optimizer/indexer/fenwick_cache_indexer.h"

#include <climits>
#include <unordered_set>
#include <vector>

namespace kv_cache_manager {

class FenwickTreeTest : public TESTBASE {};

TEST_F(FenwickTreeTest, BasicPrefixSum) {
    FenwickTree tree(10);
    tree.Update(0, 1);
    tree.Update(3, 1);
    tree.Update(7, 1);

    EXPECT_EQ(1, tree.PrefixSum(0));
    EXPECT_EQ(1, tree.PrefixSum(2));
    EXPECT_EQ(2, tree.PrefixSum(3));
    EXPECT_EQ(2, tree.PrefixSum(6));
    EXPECT_EQ(3, tree.PrefixSum(7));
    EXPECT_EQ(3, tree.PrefixSum(9));
}

TEST_F(FenwickTreeTest, RangeSum) {
    FenwickTree tree(10);
    tree.Update(1, 1);
    tree.Update(3, 1);
    tree.Update(5, 1);
    tree.Update(7, 1);

    EXPECT_EQ(2, tree.RangeSum(1, 3));
    EXPECT_EQ(3, tree.RangeSum(1, 5));
    EXPECT_EQ(0, tree.RangeSum(8, 9));
    EXPECT_EQ(4, tree.RangeSum(0, 9));
}

TEST_F(FenwickTreeTest, FindFirst) {
    FenwickTree tree(10);
    tree.Update(2, 1);
    tree.Update(5, 1);
    tree.Update(8, 1);

    EXPECT_EQ(2, tree.FindFirst(1));
    EXPECT_EQ(5, tree.FindFirst(2));
    EXPECT_EQ(8, tree.FindFirst(3));
}

TEST_F(FenwickTreeTest, UpdateAndDelete) {
    FenwickTree tree(10);
    tree.Update(3, 1);
    EXPECT_EQ(1, tree.PrefixSum(9));

    tree.Update(3, -1);
    EXPECT_EQ(0, tree.PrefixSum(9));
}

TEST_F(FenwickTreeTest, Reset) {
    FenwickTree tree(10);
    tree.Update(5, 1);
    EXPECT_EQ(1, tree.PrefixSum(9));

    tree.Reset(20);
    EXPECT_EQ(20, tree.capacity());
    EXPECT_EQ(0, tree.PrefixSum(19));
}

class FenwickCacheIndexerTest : public TESTBASE {};

TEST_F(FenwickCacheIndexerTest, FirstAccessReturnMaxSD) {
    FenwickCacheIndexer calc(100);
    EXPECT_EQ(INT64_MAX, calc.ComputeStackDistance(42));
    EXPECT_EQ(1, calc.unique_count());
}

TEST_F(FenwickCacheIndexerTest, ImmediateReaccessHasZeroSD) {
    FenwickCacheIndexer calc(100);
    calc.ComputeStackDistance(1);
    EXPECT_EQ(0, calc.ComputeStackDistance(1));
}

TEST_F(FenwickCacheIndexerTest, StackDistanceIsCorrect) {
    FenwickCacheIndexer calc(100);
    calc.ComputeStackDistance(1);
    calc.ComputeStackDistance(2);
    calc.ComputeStackDistance(3);
    int64_t sd = calc.ComputeStackDistance(1);
    EXPECT_EQ(2, sd);
}

TEST_F(FenwickCacheIndexerTest, StackDistanceSequence) {
    FenwickCacheIndexer calc(100);
    EXPECT_EQ(INT64_MAX, calc.ComputeStackDistance(1));
    EXPECT_EQ(INT64_MAX, calc.ComputeStackDistance(2));
    EXPECT_EQ(INT64_MAX, calc.ComputeStackDistance(3));
    EXPECT_EQ(INT64_MAX, calc.ComputeStackDistance(4));
    EXPECT_EQ(2, calc.ComputeStackDistance(2));
    EXPECT_EQ(3, calc.ComputeStackDistance(1));
    EXPECT_EQ(4, calc.unique_count());
}

TEST_F(FenwickCacheIndexerTest, EvictionRemovesOldestKeys) {
    FenwickCacheIndexer calc(3);
    calc.ComputeStackDistance(1);
    calc.ComputeStackDistance(2);
    calc.ComputeStackDistance(3);
    calc.ComputeStackDistance(4);
    EXPECT_EQ(4, calc.unique_count());

    calc.PostQueryMaintenance();
    EXPECT_EQ(3, calc.unique_count());

    EXPECT_EQ(INT64_MAX, calc.ComputeStackDistance(1));
}

TEST_F(FenwickCacheIndexerTest, CompactReducesArraySize) {
    FenwickCacheIndexer calc(1000);
    for (int i = 0; i < 100; i++) {
        calc.ComputeStackDistance(i);
    }
    EXPECT_EQ(100, calc.unique_count());

    for (int i = 0; i < 500; i++) {
        calc.ComputeStackDistance(0);
    }

    for (int i = 0; i < 1000; i++) {
        calc.ComputeStackDistance(0);
    }
    EXPECT_EQ(100, calc.unique_count());
}

TEST_F(FenwickCacheIndexerTest, CorrectAfterEviction) {
    FenwickCacheIndexer calc(5);
    for (int i = 0; i < 5; i++) {
        calc.ComputeStackDistance(i);
    }
    EXPECT_EQ(5, calc.unique_count());

    calc.ComputeStackDistance(100);
    calc.PostQueryMaintenance();
    EXPECT_EQ(5, calc.unique_count());

    EXPECT_NE(INT64_MAX, calc.ComputeStackDistance(1));
    EXPECT_NE(INT64_MAX, calc.ComputeStackDistance(2));
    EXPECT_NE(INT64_MAX, calc.ComputeStackDistance(3));
    EXPECT_NE(INT64_MAX, calc.ComputeStackDistance(4));

    EXPECT_EQ(INT64_MAX, calc.ComputeStackDistance(0));
}

TEST_F(FenwickCacheIndexerTest, LargeScaleCorrectness) {
    const int64_t capacity = 1000;
    FenwickCacheIndexer calc(capacity);

    for (int64_t i = 0; i < 500; i++) {
        EXPECT_EQ(INT64_MAX, calc.ComputeStackDistance(i));
    }
    EXPECT_EQ(500, calc.unique_count());

    for (int64_t i = 499; i >= 0; i--) {
        int64_t sd = calc.ComputeStackDistance(i);
        EXPECT_NE(INT64_MAX, sd);
        EXPECT_LT(sd, 500);
    }
}

TEST_F(FenwickCacheIndexerTest, NoEvictionWhenUnlimited) {
    FenwickCacheIndexer calc(0);
    for (int i = 0; i < 500; i++) {
        calc.ComputeStackDistance(i);
    }
    calc.PostQueryMaintenance();
    EXPECT_EQ(500, calc.unique_count());
}

TEST_F(FenwickCacheIndexerTest, ExpansionWhenUnlimited) {
    FenwickCacheIndexer calc(0);
    for (int i = 0; i < 2000; i++) {
        calc.ComputeStackDistance(i);
    }
    EXPECT_EQ(2000, calc.unique_count());

    for (int i = 0; i < 100; i++) {
        int64_t sd = calc.ComputeStackDistance(i);
        EXPECT_NE(INT64_MAX, sd);
    }
}

TEST_F(FenwickCacheIndexerTest, ProcessKeysHitCount) {
    FenwickCacheIndexer calc(100);
    // size_full_only = size_full_linear = 1GB, linear_step = 1
    // -> avg_bytes_per_block = 1GB, capacity_blocks = {2, 5, 10}
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    calc.Init({2.0, 5.0, 10.0}, kOneGB, kOneGB, 1);

    // First access: all miss -> hit_count = {0, 0, 0}
    std::vector<int64_t> hit_count;
    int64_t max_hit;
    calc.ProcessKeys({1}, hit_count, max_hit);
    EXPECT_EQ(3u, hit_count.size());
    EXPECT_EQ(0, hit_count[0]);
    EXPECT_EQ(0, hit_count[1]);
    EXPECT_EQ(0, hit_count[2]);

    // Access 2, 3 (populate cache)
    calc.ProcessKeys({2}, hit_count, max_hit);
    calc.ProcessKeys({3}, hit_count, max_hit);

    // Re-access 1: sd=2, so hit for cap>=3 (cap[1]=5, cap[2]=10) but miss for cap[0]=2
    calc.ProcessKeys({1}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]);  // sd=2 >= cap[0]=2 -> miss at first key
    EXPECT_EQ(1, hit_count[1]);  // sd=2 < cap[1]=5 -> hit
    EXPECT_EQ(1, hit_count[2]);  // sd=2 < cap[2]=10 -> hit
}

} // namespace kv_cache_manager
