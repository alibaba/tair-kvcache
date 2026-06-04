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
    EXPECT_EQ(INT64_MAX, calc.ProcessKey(42));
    EXPECT_EQ(1, calc.unique_count());
}

TEST_F(FenwickCacheIndexerTest, ImmediateReaccessHasZeroSD) {
    FenwickCacheIndexer calc(100);
    calc.ProcessKey(1);
    EXPECT_EQ(0, calc.ProcessKey(1));
}

TEST_F(FenwickCacheIndexerTest, StackDistanceIsCorrect) {
    FenwickCacheIndexer calc(100);
    calc.ProcessKey(1);
    calc.ProcessKey(2);
    calc.ProcessKey(3);
    int64_t sd = calc.ProcessKey(1);
    EXPECT_EQ(2, sd);
}

TEST_F(FenwickCacheIndexerTest, StackDistanceSequence) {
    FenwickCacheIndexer calc(100);
    EXPECT_EQ(INT64_MAX, calc.ProcessKey(1));
    EXPECT_EQ(INT64_MAX, calc.ProcessKey(2));
    EXPECT_EQ(INT64_MAX, calc.ProcessKey(3));
    EXPECT_EQ(INT64_MAX, calc.ProcessKey(4));
    EXPECT_EQ(2, calc.ProcessKey(2));
    EXPECT_EQ(3, calc.ProcessKey(1));
    EXPECT_EQ(4, calc.unique_count());
}

TEST_F(FenwickCacheIndexerTest, EvictionRemovesOldestKeys) {
    FenwickCacheIndexer calc(3);
    calc.ProcessKey(1);
    calc.ProcessKey(2);
    calc.ProcessKey(3);
    calc.ProcessKey(4);
    EXPECT_EQ(4, calc.unique_count());

    calc.PostQueryMaintenance();
    EXPECT_EQ(3, calc.unique_count());

    EXPECT_EQ(INT64_MAX, calc.ProcessKey(1));
}

TEST_F(FenwickCacheIndexerTest, CompactReducesArraySize) {
    FenwickCacheIndexer calc(1000);
    for (int i = 0; i < 100; i++) {
        calc.ProcessKey(i);
    }
    EXPECT_EQ(100, calc.unique_count());

    for (int i = 0; i < 500; i++) {
        calc.ProcessKey(0);
    }

    for (int i = 0; i < 1000; i++) {
        calc.ProcessKey(0);
    }
    EXPECT_EQ(100, calc.unique_count());
}

TEST_F(FenwickCacheIndexerTest, CorrectAfterEviction) {
    FenwickCacheIndexer calc(5);
    for (int i = 0; i < 5; i++) {
        calc.ProcessKey(i);
    }
    EXPECT_EQ(5, calc.unique_count());

    calc.ProcessKey(100);
    calc.PostQueryMaintenance();
    EXPECT_EQ(5, calc.unique_count());

    EXPECT_NE(INT64_MAX, calc.ProcessKey(1));
    EXPECT_NE(INT64_MAX, calc.ProcessKey(2));
    EXPECT_NE(INT64_MAX, calc.ProcessKey(3));
    EXPECT_NE(INT64_MAX, calc.ProcessKey(4));

    EXPECT_EQ(INT64_MAX, calc.ProcessKey(0));
}

TEST_F(FenwickCacheIndexerTest, LargeScaleCorrectness) {
    const int64_t capacity = 1000;
    FenwickCacheIndexer calc(capacity);

    for (int64_t i = 0; i < 500; i++) {
        EXPECT_EQ(INT64_MAX, calc.ProcessKey(i));
    }
    EXPECT_EQ(500, calc.unique_count());

    for (int64_t i = 499; i >= 0; i--) {
        int64_t sd = calc.ProcessKey(i);
        EXPECT_NE(INT64_MAX, sd);
        EXPECT_LT(sd, 500);
    }
}

TEST_F(FenwickCacheIndexerTest, NoEvictionWhenUnlimited) {
    FenwickCacheIndexer calc(0);
    for (int i = 0; i < 500; i++) {
        calc.ProcessKey(i);
    }
    calc.PostQueryMaintenance();
    EXPECT_EQ(500, calc.unique_count());
    EXPECT_EQ(500, calc.peak_unique_count());
}

TEST_F(FenwickCacheIndexerTest, ExpansionWhenUnlimited) {
    FenwickCacheIndexer calc(0);
    for (int i = 0; i < 2000; i++) {
        calc.ProcessKey(i);
    }
    EXPECT_EQ(2000, calc.unique_count());

    for (int i = 0; i < 100; i++) {
        int64_t sd = calc.ProcessKey(i);
        EXPECT_NE(INT64_MAX, sd);
    }
}

TEST_F(FenwickCacheIndexerTest, PeakUniqueCountTracked) {
    FenwickCacheIndexer calc(5);
    for (int i = 0; i < 10; i++) {
        calc.ProcessKey(i);
    }
    calc.PostQueryMaintenance();
    EXPECT_EQ(5, calc.unique_count());
    EXPECT_EQ(10, calc.peak_unique_count());
}

} // namespace kv_cache_manager
