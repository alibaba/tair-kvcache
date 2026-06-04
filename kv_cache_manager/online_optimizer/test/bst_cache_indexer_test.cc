#include <climits>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/online_optimizer/indexer/bst_cache_indexer.h"

namespace kv_cache_manager {

class AugmentedTreapTest : public TESTBASE {};

TEST_F(AugmentedTreapTest, InsertAndSize) {
    AugmentedTreap treap;
    EXPECT_EQ(0, treap.Size());
    treap.Insert(10);
    EXPECT_EQ(1, treap.Size());
    treap.Insert(20);
    treap.Insert(30);
    EXPECT_EQ(3, treap.Size());
}

TEST_F(AugmentedTreapTest, EraseAndSize) {
    AugmentedTreap treap;
    treap.Insert(10);
    treap.Insert(20);
    treap.Insert(30);
    treap.Erase(20);
    EXPECT_EQ(2, treap.Size());
    treap.Erase(10);
    EXPECT_EQ(1, treap.Size());
}

TEST_F(AugmentedTreapTest, CountGreater) {
    AugmentedTreap treap;
    treap.Insert(10);
    treap.Insert(20);
    treap.Insert(30);
    treap.Insert(40);
    treap.Insert(50);

    EXPECT_EQ(4, treap.CountGreater(10));
    EXPECT_EQ(2, treap.CountGreater(30));
    EXPECT_EQ(0, treap.CountGreater(50));
    EXPECT_EQ(5, treap.CountGreater(5));  // all are greater
    EXPECT_EQ(0, treap.CountGreater(55)); // none greater
}

TEST_F(AugmentedTreapTest, CountGreaterAfterErase) {
    AugmentedTreap treap;
    treap.Insert(10);
    treap.Insert(20);
    treap.Insert(30);

    EXPECT_EQ(2, treap.CountGreater(10));
    treap.Erase(30);
    EXPECT_EQ(1, treap.CountGreater(10));
}

TEST_F(AugmentedTreapTest, LargeScale) {
    AugmentedTreap treap;
    for (int64_t i = 0; i < 1000; i++) {
        treap.Insert(i);
    }
    EXPECT_EQ(1000, treap.Size());
    EXPECT_EQ(999, treap.CountGreater(0));
    EXPECT_EQ(0, treap.CountGreater(999));
    EXPECT_EQ(500, treap.CountGreater(499));
}

class BSTCacheIndexerTest : public TESTBASE {};

TEST_F(BSTCacheIndexerTest, FirstAccessReturnMaxSD) {
    BSTCacheIndexer calc(0);
    EXPECT_EQ(INT64_MAX, calc.ComputeStackDistance(42));
    EXPECT_EQ(1, calc.unique_count());
}

TEST_F(BSTCacheIndexerTest, ImmediateReaccessHasZeroSD) {
    BSTCacheIndexer calc(0);
    calc.ComputeStackDistance(1);
    EXPECT_EQ(0, calc.ComputeStackDistance(1));
    EXPECT_EQ(1, calc.unique_count());
}

TEST_F(BSTCacheIndexerTest, StackDistanceIsCorrect) {
    BSTCacheIndexer calc(0);
    calc.ComputeStackDistance(1);              // A first
    calc.ComputeStackDistance(2);              // B first
    calc.ComputeStackDistance(3);              // C first
    int64_t sd = calc.ComputeStackDistance(1); // A: since A, B and C accessed -> sd=2
    EXPECT_EQ(2, sd);
}

TEST_F(BSTCacheIndexerTest, StackDistanceSequence) {
    BSTCacheIndexer calc(0);
    EXPECT_EQ(INT64_MAX, calc.ComputeStackDistance(1)); // A
    EXPECT_EQ(INT64_MAX, calc.ComputeStackDistance(2)); // B
    EXPECT_EQ(INT64_MAX, calc.ComputeStackDistance(3)); // C
    EXPECT_EQ(INT64_MAX, calc.ComputeStackDistance(4)); // D
    EXPECT_EQ(2, calc.ComputeStackDistance(2));         // B: C,D accessed since B -> sd=2
    EXPECT_EQ(3, calc.ComputeStackDistance(1));         // A: {C,D,B} accessed since A -> sd=3
}

TEST_F(BSTCacheIndexerTest, ConsistencyWithFenwick) {
    BSTCacheIndexer bst(0);

    // Sequence: 1, 2, 3, 4, 2, 1
    EXPECT_EQ(INT64_MAX, bst.ComputeStackDistance(1));
    EXPECT_EQ(INT64_MAX, bst.ComputeStackDistance(2));
    EXPECT_EQ(INT64_MAX, bst.ComputeStackDistance(3));
    EXPECT_EQ(INT64_MAX, bst.ComputeStackDistance(4));
    EXPECT_EQ(2, bst.ComputeStackDistance(2)); // C,D accessed since B
    EXPECT_EQ(3, bst.ComputeStackDistance(1)); // B,C,D since A at position 0
}

TEST_F(BSTCacheIndexerTest, NoEviction) {
    BSTCacheIndexer calc(0);
    for (int64_t i = 0; i < 1000; i++) {
        calc.ComputeStackDistance(i);
    }
    EXPECT_EQ(1000, calc.unique_count());

    // Re-access doesn't change unique_count
    for (int64_t i = 0; i < 100; i++) {
        calc.ComputeStackDistance(i);
    }
    EXPECT_EQ(1000, calc.unique_count());
}

TEST_F(BSTCacheIndexerTest, LargeScaleCorrectness) {
    BSTCacheIndexer calc(0);

    for (int64_t i = 0; i < 500; i++) {
        EXPECT_EQ(INT64_MAX, calc.ComputeStackDistance(i));
    }
    EXPECT_EQ(500, calc.unique_count());

    // Re-access all in reverse
    for (int64_t i = 499; i >= 0; i--) {
        int64_t sd = calc.ComputeStackDistance(i);
        EXPECT_NE(INT64_MAX, sd);
        EXPECT_LT(sd, 500);
    }
}

TEST_F(BSTCacheIndexerTest, EvictionWithMaxKeyCount) {
    BSTCacheIndexer calc(3);
    calc.ComputeStackDistance(1);
    calc.ComputeStackDistance(2);
    calc.ComputeStackDistance(3);
    calc.ComputeStackDistance(4);
    EXPECT_EQ(4, calc.unique_count());

    calc.PostQueryMaintenance();
    EXPECT_EQ(3, calc.unique_count());

    EXPECT_EQ(INT64_MAX, calc.ComputeStackDistance(1));
}

TEST_F(BSTCacheIndexerTest, EvictionPreservesRecentKeys) {
    BSTCacheIndexer calc(5);
    for (int64_t i = 0; i < 5; i++) {
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

TEST_F(BSTCacheIndexerTest, NoEvictionWhenUnlimited) {
    BSTCacheIndexer calc(0);
    for (int64_t i = 0; i < 1000; i++) {
        calc.ComputeStackDistance(i);
    }
    calc.PostQueryMaintenance();
    EXPECT_EQ(1000, calc.unique_count());
}

TEST_F(BSTCacheIndexerTest, ProcessKeysHitCount) {
    BSTCacheIndexer calc(0);
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
    // Query {1}: cap[0]=2 -> miss at idx 0, cap[1]=5 -> hit, cap[2]=10 -> hit
    calc.ProcessKeys({1}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]); // sd=2 >= cap[0]=2 -> miss at first key
    EXPECT_EQ(1, hit_count[1]); // sd=2 < cap[1]=5 -> hit
    EXPECT_EQ(1, hit_count[2]); // sd=2 < cap[2]=10 -> hit
}

} // namespace kv_cache_manager
