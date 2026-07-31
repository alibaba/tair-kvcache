#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/index/online/cache_indexer_factory.h"
#include "kv_cache_manager/optimizer/index/online/ttl_cache_indexer_wrapper.h"

namespace kv_cache_manager {

class TtlCacheIndexerWrapperTest : public TESTBASE {};

static std::unique_ptr<CacheIndexer> MakeInnerIndexer(const std::string &policy_type = "lru",
                                                      bool enable_theoretical_max_cache = false,
                                                      double capacity_gb = 10.0) {
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    auto indexer = CacheIndexerFactory::CreateCacheIndexer(
        policy_type, enable_theoretical_max_cache, {capacity_gb}, kOneGB, kOneGB, 1);
    return indexer;
}

TEST_F(TtlCacheIndexerWrapperTest, BasicExpiration) {
    int64_t now = 1000;
    auto clock = [&now]() { return now; };
    TtlCacheIndexerWrapper wrapper(MakeInnerIndexer(), 10, clock);

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    wrapper.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(3, wrapper.unique_count());
    EXPECT_EQ(0, wrapper.ttl_eviction_count());

    // Re-access within TTL: should hit
    now = 1005;
    wrapper.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(3, hit_count[0]); // all hit (prefix match)

    // Advance past TTL (last access was at 1005, TTL=10, expires at 1015)
    now = 1016;
    wrapper.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]); // all expired -> all miss
    EXPECT_EQ(3, wrapper.ttl_eviction_count());
}

TEST_F(TtlCacheIndexerWrapperTest, SlidingTtlRefreshesExpiry) {
    int64_t now = 1000;
    auto clock = [&now]() { return now; };
    TtlCacheIndexerWrapper wrapper(MakeInnerIndexer(), 10, clock);

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    wrapper.ProcessKeys({1}, hit_count, max_hit);

    // Access at t=1008, refreshes TTL to expire at 1018
    now = 1008;
    wrapper.ProcessKeys({1}, hit_count, max_hit);
    EXPECT_EQ(1, hit_count[0]); // hit

    // At t=1012, would have expired without refresh (1000+10=1010), but was refreshed
    now = 1012;
    wrapper.ProcessKeys({1}, hit_count, max_hit);
    EXPECT_EQ(1, hit_count[0]); // still hit
    EXPECT_EQ(0, wrapper.ttl_eviction_count());

    // At t=1023, past refreshed expiry (1012+10=1022)
    now = 1023;
    wrapper.ProcessKeys({1}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]); // expired
    EXPECT_EQ(1, wrapper.ttl_eviction_count());
}

TEST_F(TtlCacheIndexerWrapperTest, TtlAndCapacityEvictionInteraction) {
    int64_t now = 1000;
    auto clock = [&now]() { return now; };
    TtlCacheIndexerWrapper wrapper(MakeInnerIndexer("lru", false, 3.0), 100, clock);

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    wrapper.ProcessKeys({1, 2, 3, 4, 5}, hit_count, max_hit);
    wrapper.PostQueryMaintenance();
    // Capacity eviction should have removed 2 keys
    EXPECT_EQ(3, wrapper.unique_count());
    EXPECT_EQ(2, wrapper.eviction_count()); // total = capacity(2) + ttl(0)
    EXPECT_EQ(0, wrapper.ttl_eviction_count());

    // Now expire remaining keys via TTL
    now = 1101;
    wrapper.ProcessKeys({10}, hit_count, max_hit);
    // 3 keys expired by TTL + 1 new key
    EXPECT_EQ(3, wrapper.ttl_eviction_count());
    EXPECT_EQ(1, wrapper.unique_count());
}

TEST_F(TtlCacheIndexerWrapperTest, WorksWithFactoryLruIndexer) {
    int64_t now = 1000;
    auto clock = [&now]() { return now; };
    TtlCacheIndexerWrapper wrapper(MakeInnerIndexer(), 10, clock);

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    wrapper.ProcessKeys({1, 2}, hit_count, max_hit);
    EXPECT_EQ(2, wrapper.unique_count());

    now = 1005;
    wrapper.ProcessKeys({1}, hit_count, max_hit);
    EXPECT_EQ(1, hit_count[0]);

    now = 1011;
    wrapper.ProcessKeys({1, 2}, hit_count, max_hit);
    // key 2 expired (last access 1000+10=1010), key 1 not (last access 1005+10=1015)
    EXPECT_EQ(1, hit_count[0]); // key 1 hit, then key 2 miss -> prefix hit = 1
    EXPECT_EQ(1, wrapper.ttl_eviction_count());
}

TEST_F(TtlCacheIndexerWrapperTest, WorksWithLruIndexer) {
    int64_t now = 1000;
    auto clock = [&now]() { return now; };
    TtlCacheIndexerWrapper wrapper(MakeInnerIndexer(), 10, clock);

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    wrapper.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(3, wrapper.unique_count());

    now = 1005;
    wrapper.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(3, hit_count[0]); // all hit

    now = 1016;
    wrapper.ProcessKeys({1}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]); // all expired
    EXPECT_EQ(3, wrapper.ttl_eviction_count());
}

TEST_F(TtlCacheIndexerWrapperTest, MetricsAccuracy) {
    int64_t now = 1000;
    auto clock = [&now]() { return now; };
    TtlCacheIndexerWrapper wrapper(MakeInnerIndexer(), 10, clock);

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    wrapper.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(3, wrapper.unique_count());
    EXPECT_EQ(0, wrapper.eviction_count());
    EXPECT_EQ(0, wrapper.ttl_eviction_count());
    EXPECT_GT(wrapper.memory_usage_bytes(), 0);
    EXPECT_GT(wrapper.kv_cache_usage_bytes(), 0);

    now = 1011;
    wrapper.ProcessKeys({4}, hit_count, max_hit);
    EXPECT_EQ(1, wrapper.unique_count());
    EXPECT_EQ(3, wrapper.ttl_eviction_count());
    EXPECT_EQ(3, wrapper.eviction_count()); // ttl evictions counted in total
}

TEST_F(TtlCacheIndexerWrapperTest, RemoveKeyCleansTtlState) {
    int64_t now = 1000;
    auto clock = [&now]() { return now; };
    TtlCacheIndexerWrapper wrapper(MakeInnerIndexer(), 10, clock);

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    wrapper.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(3, wrapper.unique_count());

    wrapper.RemoveKey(2);
    EXPECT_EQ(2, wrapper.unique_count());

    // Key 2 should be a miss now
    now = 1002;
    wrapper.ProcessKeys({2}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]); // miss (was removed, re-inserted as new)
    EXPECT_EQ(3, wrapper.unique_count());

    // Advance past TTL: key 2 (re-inserted at 1002) should survive, keys 1,3 (at 1000) should expire
    now = 1011;
    wrapper.ProcessKeys({2}, hit_count, max_hit);
    EXPECT_EQ(1, hit_count[0]);                 // key 2 hit (inserted at 1002, expires at 1012)
    EXPECT_EQ(2, wrapper.ttl_eviction_count()); // keys 1 and 3 expired
}

TEST_F(TtlCacheIndexerWrapperTest, PartialExpiration) {
    int64_t now = 1000;
    auto clock = [&now]() { return now; };
    TtlCacheIndexerWrapper wrapper(MakeInnerIndexer(), 10, clock);

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    wrapper.ProcessKeys({1}, hit_count, max_hit);

    now = 1003;
    wrapper.ProcessKeys({2}, hit_count, max_hit);

    now = 1006;
    wrapper.ProcessKeys({3}, hit_count, max_hit);

    // At t=1011: key 1 (access 1000) expires, keys 2,3 survive
    now = 1011;
    wrapper.ProcessKeys({2, 3}, hit_count, max_hit);
    EXPECT_EQ(2, hit_count[0]);
    EXPECT_EQ(1, wrapper.ttl_eviction_count()); // only key 1

    // At t=1014: key 2 (access refreshed to 1011) still alive
    now = 1014;
    wrapper.ProcessKeys({2}, hit_count, max_hit);
    EXPECT_EQ(1, hit_count[0]);

    // At t=1017: key 3 (access refreshed to 1011) expires
    now = 1022;
    wrapper.ProcessKeys({2, 3}, hit_count, max_hit);
    // key 2 refreshed at 1014, expires at 1024 -> still alive
    // key 3 refreshed at 1011, expires at 1021 -> expired
    EXPECT_EQ(1, hit_count[0]); // key 2 hits, key 3 misses -> prefix = 1
}

TEST_F(TtlCacheIndexerWrapperTest, ExactTtlBoundary) {
    int64_t now = 1000;
    auto clock = [&now]() { return now; };
    TtlCacheIndexerWrapper wrapper(MakeInnerIndexer(), 10, clock);

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    wrapper.ProcessKeys({1}, hit_count, max_hit);

    // One second before expiry: key should still be alive
    now = 1009;
    wrapper.ProcessKeys({1}, hit_count, max_hit);
    EXPECT_EQ(1, hit_count[0]);
    EXPECT_EQ(0, wrapper.ttl_eviction_count());

    // Sliding TTL: access at 1009 refreshes expiry to 1019
    // At exactly last_access + ttl_seconds (1009+10=1019): key should be expired
    now = 1019;
    wrapper.ProcessKeys({1}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]);
    EXPECT_EQ(1, wrapper.ttl_eviction_count());
}

TEST_F(TtlCacheIndexerWrapperTest, HitAgeBucketBasic) {
    int64_t now = 1000;
    auto clock = [&now]() { return now; };
    TtlCacheIndexerWrapper wrapper(MakeInnerIndexer(), 3600, clock);
    // Use small thresholds for easy testing
    wrapper.SetHitAgeBucketThresholds({10, 30, 60});
    // Buckets: [0,10) [10,30) [30,60) [60,+inf)

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // Insert keys 1, 2, 3
    wrapper.ProcessKeys({1, 2, 3}, hit_count, max_hit);

    // Re-access key 1 at age=5 (< 10 → bucket 0)
    now = 1005;
    wrapper.ProcessKeys({1}, hit_count, max_hit);

    // Re-access key 2 at age=15 (>= 10, < 30 → bucket 1)
    now = 1015;
    wrapper.ProcessKeys({2}, hit_count, max_hit);

    // Re-access key 3 at age=45 (>= 30, < 60 → bucket 2)
    now = 1045;
    wrapper.ProcessKeys({3}, hit_count, max_hit);

    auto buckets = wrapper.GetHitAgeBuckets();
    ASSERT_EQ(4u, buckets.size());

    // bucket [0,10): threshold=10, count=1 (key 1 at age 5)
    EXPECT_EQ(10, buckets[0].threshold_seconds);
    EXPECT_EQ(1, buckets[0].hit_count);

    // bucket [10,30): threshold=30, count=1 (key 2 at age 15)
    EXPECT_EQ(30, buckets[1].threshold_seconds);
    EXPECT_EQ(1, buckets[1].hit_count);

    // bucket [30,60): threshold=60, count=1 (key 3 at age 45)
    EXPECT_EQ(60, buckets[2].threshold_seconds);
    EXPECT_EQ(1, buckets[2].hit_count);

    // bucket [60,+inf): threshold=0, count=0
    EXPECT_EQ(0, buckets[3].threshold_seconds);
    EXPECT_EQ(0, buckets[3].hit_count);
}

TEST_F(TtlCacheIndexerWrapperTest, CapacityEvictedKeyDoesNotIncrementHitAgeBucket) {
    int64_t now = 1000;
    auto clock = [&now]() { return now; };
    TtlCacheIndexerWrapper wrapper(MakeInnerIndexer("lru", true, 2.0), 100, clock);
    wrapper.SetHitAgeBucketThresholds({10});

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    wrapper.ProcessKeys({1, 2}, hit_count, max_hit);
    wrapper.PostQueryMaintenance();

    now = 1005;
    wrapper.ProcessKeys({3}, hit_count, max_hit);
    wrapper.PostQueryMaintenance();

    now = 1010;
    wrapper.ProcessKeys({1}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]);
    EXPECT_EQ(1, max_hit);

    auto buckets = wrapper.GetHitAgeBuckets();
    ASSERT_EQ(2u, buckets.size());
    EXPECT_EQ(0, buckets[0].hit_count);
    EXPECT_EQ(0, buckets[1].hit_count);
}

TEST_F(TtlCacheIndexerWrapperTest, HitAgeBucketInfinityBucket) {
    int64_t now = 1000;
    auto clock = [&now]() { return now; };
    TtlCacheIndexerWrapper wrapper(MakeInnerIndexer(), 10000, clock);
    wrapper.SetHitAgeBucketThresholds({10, 100});
    // Buckets: [0,10) [10,100) [100,+inf)

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    wrapper.ProcessKeys({1}, hit_count, max_hit);

    // Re-access at age=200 (>= 100 → bucket 2, the +inf bucket)
    now = 1200;
    wrapper.ProcessKeys({1}, hit_count, max_hit);

    auto buckets = wrapper.GetHitAgeBuckets();
    ASSERT_EQ(3u, buckets.size());
    EXPECT_EQ(0, buckets[0].hit_count); // [0,10)
    EXPECT_EQ(0, buckets[1].hit_count); // [10,100)
    EXPECT_EQ(1, buckets[2].hit_count); // [100,+inf)
    EXPECT_EQ(0, buckets[2].threshold_seconds);
}

TEST_F(TtlCacheIndexerWrapperTest, HitAgeBucketDefaultThresholds) {
    int64_t now = 1000;
    auto clock = [&now]() { return now; };
    TtlCacheIndexerWrapper wrapper(MakeInnerIndexer(), 100000, clock);
    // Don't call SetHitAgeBucketThresholds — use defaults {5,30,60,120,300,600,1800,3600,7200}

    auto buckets = wrapper.GetHitAgeBuckets();
    // 9 thresholds + 1 infinity bucket = 10
    ASSERT_EQ(10u, buckets.size());
    EXPECT_EQ(5, buckets[0].threshold_seconds);
    EXPECT_EQ(30, buckets[1].threshold_seconds);
    EXPECT_EQ(60, buckets[2].threshold_seconds);
    EXPECT_EQ(120, buckets[3].threshold_seconds);
    EXPECT_EQ(300, buckets[4].threshold_seconds);
    EXPECT_EQ(600, buckets[5].threshold_seconds);
    EXPECT_EQ(1800, buckets[6].threshold_seconds);
    EXPECT_EQ(3600, buckets[7].threshold_seconds);
    EXPECT_EQ(7200, buckets[8].threshold_seconds);
    EXPECT_EQ(0, buckets[9].threshold_seconds);

    // All counts should be 0 initially
    for (const auto &b : buckets) {
        EXPECT_EQ(0, b.hit_count);
    }
}

TEST_F(TtlCacheIndexerWrapperTest, HitAgeBucketMultipleHitsSameBucket) {
    int64_t now = 1000;
    auto clock = [&now]() { return now; };
    TtlCacheIndexerWrapper wrapper(MakeInnerIndexer(), 3600, clock);
    wrapper.SetHitAgeBucketThresholds({10, 30});

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // Insert 3 keys
    wrapper.ProcessKeys({1, 2, 3}, hit_count, max_hit);

    // Re-access all 3 at age=5 → all fall into bucket [0,10)
    now = 1005;
    wrapper.ProcessKeys({1, 2, 3}, hit_count, max_hit);

    auto buckets = wrapper.GetHitAgeBuckets();
    ASSERT_EQ(3u, buckets.size());
    EXPECT_EQ(3, buckets[0].hit_count); // [0,10): 3 hits
    EXPECT_EQ(0, buckets[1].hit_count); // [10,30)
    EXPECT_EQ(0, buckets[2].hit_count); // [30,+inf)
}

TEST_F(TtlCacheIndexerWrapperTest, HitAgeBucketNoTrackingWithoutTtlWrapper) {
    // Base CacheIndexer should return empty buckets
    auto indexer = MakeInnerIndexer();
    auto buckets = indexer->GetHitAgeBuckets();
    EXPECT_TRUE(buckets.empty());
}

} // namespace kv_cache_manager
