#include "kv_cache_manager/mrc/mrc_profiler.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <list>
#include <random>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {
namespace {

class NaiveRequestLru {
public:
    explicit NaiveRequestLru(int64_t capacity) : capacity_(capacity) {}

    int64_t Process(const std::vector<int64_t> &keys) {
        int64_t prefix_hits = 0;
        for (const int64_t key : keys) {
            if (index_.find(key) == index_.end()) {
                break;
            }
            ++prefix_hits;
        }
        for (auto it = keys.rbegin(); it != keys.rend(); ++it) {
            auto current = index_.find(*it);
            if (current != index_.end()) {
                lru_.erase(current->second);
                index_.erase(current);
            }
            lru_.push_front(*it);
            index_[*it] = lru_.begin();
            while (static_cast<int64_t>(lru_.size()) > capacity_) {
                index_.erase(lru_.back());
                lru_.pop_back();
            }
        }
        return prefix_hits;
    }

private:
    int64_t capacity_;
    std::list<int64_t> lru_;
    std::unordered_map<int64_t, std::list<int64_t>::iterator> index_;
};

std::vector<int64_t> Chain(int64_t prefix, int length) {
    std::vector<int64_t> keys;
    for (int i = 0; i < length; ++i) {
        keys.push_back(prefix * 1000 + i);
    }
    return keys;
}

} // namespace

TEST(MrcProfilerTest, MatchesNaiveRequestLevelLruAtEveryTrackedCapacity) {
    MrcProfiler::Options options;
    options.max_tracked_blocks = 64;
    options.window_seconds = 0;
    MrcProfiler profiler(options);

    std::vector<double> capacities;
    std::vector<NaiveRequestLru> references;
    std::vector<int64_t> hits;
    for (int capacity = 1; capacity <= 64; ++capacity) {
        capacities.push_back(capacity);
        references.emplace_back(capacity);
        hits.push_back(0);
    }

    std::mt19937_64 rng(20260810);
    std::uniform_int_distribution<int64_t> prefix_dist(0, 99);
    std::uniform_int_distribution<int> length_dist(1, 16);
    int64_t total_blocks = 0;
    for (int request = 0; request < 20000; ++request) {
        const auto keys = Chain(prefix_dist(rng), length_dist(rng));
        total_blocks += keys.size();
        for (size_t i = 0; i < references.size(); ++i) {
            hits[i] += references[i].Process(keys);
        }
        profiler.Observe(keys, request);
    }

    MrcProfiler::Snapshot snapshot;
    profiler.QueryCumulative(capacities, snapshot);
    ASSERT_EQ(capacities.size(), snapshot.hit_rates.size());
    EXPECT_DOUBLE_EQ(static_cast<double>(total_blocks), snapshot.total_accesses);
    for (size_t i = 0; i < capacities.size(); ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(hits[i]) / total_blocks, snapshot.hit_rates[i])
            << "capacity " << capacities[i];
    }
    EXPECT_DOUBLE_EQ(snapshot.hit_rates.back(), snapshot.max_tracked_hit_rate);
}

TEST(MrcProfilerTest, MarksCapacitiesBeyondExactBoundaryUnavailable) {
    MrcProfiler::Options options;
    options.max_tracked_blocks = 8;
    options.window_seconds = 0;
    MrcProfiler profiler(options);
    profiler.Observe(Chain(1, 12), 0);
    profiler.Observe(Chain(1, 12), 1);

    MrcProfiler::Snapshot snapshot;
    profiler.QueryCumulative({1, 8, 9, 1024}, snapshot);
    EXPECT_GE(snapshot.hit_rates[0], 0.0);
    EXPECT_GE(snapshot.hit_rates[1], 0.0);
    EXPECT_EQ(-1.0, snapshot.hit_rates[2]);
    EXPECT_EQ(-1.0, snapshot.hit_rates[3]);
    EXPECT_LE(profiler.tracked_blocks(), 8);
}

TEST(MrcProfilerTest, WindowRotationKeepsLruWarm) {
    MrcProfiler::Options options;
    options.max_tracked_blocks = 16;
    options.window_seconds = 60;
    MrcProfiler profiler(options);
    const auto request = Chain(1, 4);

    MrcProfiler::Snapshot snapshot;
    profiler.Observe(request, 0);
    EXPECT_FALSE(profiler.QueryWindow({4}, snapshot));
    profiler.Observe(request, 61 * 1000000LL);
    ASSERT_TRUE(profiler.QueryWindow({4}, snapshot));
    EXPECT_DOUBLE_EQ(4.0, snapshot.total_accesses);
    EXPECT_DOUBLE_EQ(0.0, snapshot.hit_rates[0]);

    profiler.QueryCumulative({4}, snapshot);
    EXPECT_DOUBLE_EQ(8.0, snapshot.total_accesses);
    EXPECT_DOUBLE_EQ(0.5, snapshot.hit_rates[0]);
}

TEST(MrcProfilerTest, DumpCurveIsMonotonic) {
    MrcProfiler::Options options;
    options.max_tracked_blocks = 64;
    options.window_seconds = 0;
    MrcProfiler profiler(options);
    for (int i = 0; i < 1000; ++i) {
        // Keep the complete hot working set below the 64-block tracking cap;
        // otherwise a valid all-miss workload is allowed to produce no curve
        // points at all.
        profiler.Observe(Chain(i % 4, 1 + i % 12), i);
    }
    const auto points = profiler.DumpCurve(true);
    ASSERT_FALSE(points.empty());
    for (size_t i = 1; i < points.size(); ++i) {
        EXPECT_GT(points[i].capacity_blocks, points[i - 1].capacity_blocks);
        EXPECT_GT(points[i].hit_rate, points[i - 1].hit_rate);
    }
}

} // namespace kv_cache_manager
