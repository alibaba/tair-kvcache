#include "kv_cache_manager/mrc/reuse_distance_tracker.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

namespace {

// Naive reference: recency list scan, O(n) per access.
class NaiveStackDistance {
public:
    int64_t Access(int64_t key) {
        auto it = std::find(recency_.begin(), recency_.end(), key);
        int64_t distance = -1;
        if (it != recency_.end()) {
            distance = std::distance(recency_.begin(), it);
            recency_.erase(it);
        }
        recency_.insert(recency_.begin(), key);
        return distance;
    }

    bool Erase(int64_t key) {
        auto it = std::find(recency_.begin(), recency_.end(), key);
        if (it == recency_.end()) {
            return false;
        }
        recency_.erase(it);
        return true;
    }

private:
    std::vector<int64_t> recency_; // MRU first
};

} // namespace

TEST(ReuseDistanceTrackerTest, BasicDistances) {
    ReuseDistanceTracker tracker;
    // First accesses are cold.
    EXPECT_EQ(-1, tracker.Access(1));
    EXPECT_EQ(-1, tracker.Access(2));
    EXPECT_EQ(-1, tracker.Access(3));
    // Immediate re-access: no distinct key in between.
    EXPECT_EQ(0, tracker.Access(3));
    // 1 was followed by {2, 3}.
    EXPECT_EQ(2, tracker.Access(1));
    // 2 was followed by {3, 1}.
    EXPECT_EQ(2, tracker.Access(2));
    EXPECT_EQ(3, tracker.size());
}

TEST(ReuseDistanceTrackerTest, EraseRemovesFromStack) {
    ReuseDistanceTracker tracker;
    tracker.Access(1);
    tracker.Access(2);
    tracker.Access(3);
    EXPECT_TRUE(tracker.Erase(2));
    EXPECT_FALSE(tracker.Erase(2));
    // Only 3 stands between the two accesses of 1 now.
    EXPECT_EQ(1, tracker.Access(1));
    EXPECT_EQ(2, tracker.size());
}

TEST(ReuseDistanceTrackerTest, MatchesNaiveOnRandomTraceWithCompaction) {
    // Tiny initial capacity to force many Compact() rounds.
    ReuseDistanceTracker tracker(/*initial_capacity=*/16);
    NaiveStackDistance naive;

    std::mt19937_64 rng(20260806);
    std::uniform_int_distribution<int64_t> key_dist(0, 199);
    std::uniform_int_distribution<int> op_dist(0, 9);

    for (int i = 0; i < 20000; ++i) {
        const int64_t key = key_dist(rng);
        if (op_dist(rng) == 0) {
            EXPECT_EQ(naive.Erase(key), tracker.Erase(key)) << "step " << i;
        } else {
            EXPECT_EQ(naive.Access(key), tracker.Access(key)) << "step " << i;
        }
    }
}

TEST(ReuseDistanceTrackerTest, MatchesNaiveOnZipfLikeTrace) {
    ReuseDistanceTracker tracker(/*initial_capacity=*/64);
    NaiveStackDistance naive;

    // Zipf-ish skew via squaring a uniform variate.
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int i = 0; i < 20000; ++i) {
        const double u = uniform(rng);
        const int64_t key = static_cast<int64_t>(u * u * 500);
        EXPECT_EQ(naive.Access(key), tracker.Access(key)) << "step " << i;
    }
}

} // namespace kv_cache_manager
