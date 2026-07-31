#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/index/online/lru_cache_indexer.h"

namespace kv_cache_manager {

class LruCacheIndexerTest : public TESTBASE {};

// ==================== Basic Init Tests ====================

TEST_F(LruCacheIndexerTest, InitWithSingleCapacity) {
    LruCacheIndexer indexer(0);
    indexer.Init({1.0}, 1024, 2048, 1);
    EXPECT_EQ(0, indexer.unique_count());
    EXPECT_EQ(0, indexer.eviction_count());
}

TEST_F(LruCacheIndexerTest, InitWithMultipleCapacities) {
    LruCacheIndexer indexer(0);
    indexer.Init({0.5, 1.0, 2.0}, 1024, 2048, 1);
    EXPECT_EQ(0, indexer.unique_count());
    EXPECT_EQ(0, indexer.eviction_count());
    EXPECT_EQ(0, indexer.memory_usage_bytes());
    EXPECT_EQ(0, indexer.kv_cache_usage_bytes());
}

TEST_F(LruCacheIndexerTest, InitWithLinearStep) {
    LruCacheIndexer indexer(0);
    indexer.Init({1.0}, 512, 1024, 4);
    EXPECT_EQ(0, indexer.unique_count());
}

// ==================== First Access (All Miss) Tests ====================

TEST_F(LruCacheIndexerTest, FirstAccessIsMiss) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    indexer.Init({1.0, 2.0}, kOneGB, kOneGB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    indexer.ProcessKeys({1}, hit_count, max_hit);

    EXPECT_EQ(2u, hit_count.size());
    EXPECT_EQ(0, hit_count[0]);
    EXPECT_EQ(0, hit_count[1]);
}

TEST_F(LruCacheIndexerTest, FirstAccessMultipleKeysAllMiss) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    indexer.Init({1.0}, kOneGB, kOneGB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);

    EXPECT_EQ(1u, hit_count.size());
    EXPECT_EQ(0, hit_count[0]);
}

// ==================== Repeated Access (Hit) Tests ====================

TEST_F(LruCacheIndexerTest, RepeatedAccessIsHit) {
    LruCacheIndexer indexer(0);
    // Use small charge per key so cache can comfortably hold entries
    indexer.Init({1.0}, 1024, 1024, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // First access - miss
    indexer.ProcessKeys({1}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]);

    // Second access - hit
    indexer.ProcessKeys({1}, hit_count, max_hit);
    EXPECT_EQ(1, hit_count[0]);
}

TEST_F(LruCacheIndexerTest, RepeatedAccessMultipleKeys) {
    LruCacheIndexer indexer(0);
    // Use small charge per key so cache can comfortably hold entries
    indexer.Init({1.0}, 1024, 1024, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // Populate cache
    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]);

    // Re-access same keys - all hit
    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(3, hit_count[0]);
}

// ==================== Multiple Capacity Tiers ====================

TEST_F(LruCacheIndexerTest, MultipleCapacityTiersEviction) {
    LruCacheIndexer indexer(0);
    // tier0 capacity ~2.5MB (holds 2 keys of 1MB + metadata), cannot hold 3
    // tier1 capacity ~10MB (easily holds 5 keys of 1MB + metadata)
    // Note: kFullChargeCacheMetadata adds ~72 bytes overhead per entry
    constexpr int64_t kOneMB = 1024LL * 1024;
    double tier0_gb = 2.5 * kOneMB / (1024.0 * 1024.0 * 1024.0);
    double tier1_gb = 10.0 * kOneMB / (1024.0 * 1024.0 * 1024.0);
    indexer.Init({tier0_gb, tier1_gb}, kOneMB, kOneMB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // Insert keys 1..5: tier0 can hold ~2, tier1 can hold all 5
    for (int64_t i = 1; i <= 5; i++) {
        indexer.ProcessKeys({i}, hit_count, max_hit);
    }

    // Re-access key 1: should be evicted from tier0 but still in tier1
    indexer.ProcessKeys({1}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]); // evicted from small cache
    EXPECT_EQ(1, hit_count[1]); // still in larger cache
}

// ==================== Linear Step and Boundary Key Tests ====================

TEST_F(LruCacheIndexerTest, LinearStepZeroAllFullAttention) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    // linear_step=0: all full attention, simple prefix matching with size_full_only charge.
    indexer.Init({10.0}, kOneGB / 2, kOneGB, 0);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // First access: all miss
    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]);

    // Re-access: all hit, simple prefix = 3
    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(3, hit_count[0]);

    // Partial hit: key 4 not cached, prefix truncates at 3
    indexer.ProcessKeys({1, 2, 3, 4, 5}, hit_count, max_hit);
    EXPECT_EQ(3, hit_count[0]);
}

TEST_F(LruCacheIndexerTest, LinearStepGrouping) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    // linear_step=2: step_hit at pos1,3,5,...
    // Step_hit keys have charge = size_full_linear
    indexer.Init({10.0}, kOneGB / 2, kOneGB, 2);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // First access group {1, 2}: both miss, group invalid -> hit_count = 0
    indexer.ProcessKeys({1, 2}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]);

    // Re-access group {1, 2}: both should hit with correct boundary charge
    indexer.ProcessKeys({1, 2}, hit_count, max_hit);
    EXPECT_EQ(2, hit_count[0]);
}

TEST_F(LruCacheIndexerTest, LinearStepColdAllMissReturnsZeroForCapacitiesAndMaxCache) {
    LruCacheIndexer indexer(true);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    indexer.Init({10.0, 20.0}, kOneGB / 2, kOneGB, 2);

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    indexer.ProcessKeys({1, 2, 3, 4}, hit_count, max_hit);

    ASSERT_EQ(2u, hit_count.size());
    EXPECT_EQ(0, hit_count[0]);
    EXPECT_EQ(0, hit_count[1]);
    EXPECT_EQ(0, max_hit);
}

TEST_F(LruCacheIndexerTest, LinearStepBoundaryChargeMismatch) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    // linear_step=3: groups of 3, boundary key is 3rd key
    indexer.Init({10.0}, kOneGB / 2, kOneGB, 3);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // First access as linear_step=3: group {1,2,3}
    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]); // all miss

    // Re-access same group: all hit
    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(3, hit_count[0]);
}

TEST_F(LruCacheIndexerTest, LinearStepPartialGroup) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    // linear_step=3: step_hit positions are pos2,5,8,...
    // Last key of each request is also stored with size_full_linear_.
    indexer.Init({10.0}, kOneGB / 2, kOneGB, 3);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // First access {1,2}: both miss
    indexer.ProcessKeys({1, 2}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]);

    // Re-access {1,2}: pos1 is last key → stored with size_full_linear_ → checkpoint at pos1 → reuse = 2
    indexer.ProcessKeys({1, 2}, hit_count, max_hit);
    EXPECT_EQ(2, hit_count[0]);

    // Access {1,2,3,4,5}: keys 1,2 hit (checkpoint at pos1), key 3 miss at pos2 → prefix=2
    // last_checkpoint=1 within prefix → reuse = 2
    indexer.ProcessKeys({1, 2, 3, 4, 5}, hit_count, max_hit);
    EXPECT_EQ(2, hit_count[0]);

    // Re-access {1,2,3,4,5}: all hit.
    // Checkpoints at pos1(key2), pos2(key3:step_hit), pos4(key5:last)
    // Rightmost checkpoint at pos4 → reuse = 5
    indexer.ProcessKeys({1, 2, 3, 4, 5}, hit_count, max_hit);
    EXPECT_EQ(5, hit_count[0]);
}

TEST_F(LruCacheIndexerTest, LinearStepMultipleGroups) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    // linear_step=2, capacity enough for all keys
    indexer.Init({10.0}, kOneGB / 2, kOneGB, 2);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // First access: 2 groups {1,2} {3,4}
    indexer.ProcessKeys({1, 2, 3, 4}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]);

    // Re-access: both groups hit
    indexer.ProcessKeys({1, 2, 3, 4}, hit_count, max_hit);
    EXPECT_EQ(4, hit_count[0]);
}

// ==================== Prefix Hit Count Semantics ====================

TEST_F(LruCacheIndexerTest, PrefixHitTruncatesOnMiss) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    indexer.Init({10.0}, kOneGB, kOneGB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // Populate keys 1 and 3, but NOT 2
    indexer.ProcessKeys({1}, hit_count, max_hit);
    indexer.ProcessKeys({3}, hit_count, max_hit);

    // Access sequence {1, 2, 3}: key 1 hits, key 2 misses -> prefix = 1
    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(1, hit_count[0]);
}

TEST_F(LruCacheIndexerTest, PrefixHitAllHit) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    indexer.Init({10.0}, kOneGB, kOneGB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // Populate
    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);

    // All hit -> prefix = 3
    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(3, hit_count[0]);
}

// ==================== unique_count Tracking ====================

TEST_F(LruCacheIndexerTest, UniqueCountTracksNewKeys) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    indexer.Init({10.0}, kOneGB, kOneGB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(3, indexer.unique_count());

    // Re-access doesn't increase unique_count
    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(3, indexer.unique_count());

    // New key increases unique_count
    indexer.ProcessKeys({4}, hit_count, max_hit);
    EXPECT_EQ(4, indexer.unique_count());
}

// ==================== PostQueryMaintenance and Eviction ====================

TEST_F(LruCacheIndexerTest, PostQueryMaintenanceUpdatesEviction) {
    constexpr int64_t kOneMB = 1024LL * 1024;
    LruCacheIndexer indexer(false);
    double cap_gb = 3.5 * kOneMB / (1024.0 * 1024.0 * 1024.0);
    indexer.Init({cap_gb}, kOneMB, kOneMB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // Insert 10 keys -> largest configured capacity should evict older entries.
    for (int64_t i = 1; i <= 10; i++) {
        indexer.ProcessKeys({i}, hit_count, max_hit);
    }

    indexer.PostQueryMaintenance();
    // After maintenance, unique_count should equal actual cache occupancy
    // and eviction_count should reflect evicted keys
    EXPECT_GT(indexer.eviction_count(), 0);
    EXPECT_LT(indexer.unique_count(), 10);
}

TEST_F(LruCacheIndexerTest, PostQueryMaintenanceNoEvictionWithLargeCapacity) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    indexer.Init({10.0}, kOneGB, kOneGB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    indexer.ProcessKeys({1, 2, 3, 4, 5}, hit_count, max_hit);
    indexer.PostQueryMaintenance();

    EXPECT_EQ(5, indexer.unique_count());
    EXPECT_EQ(0, indexer.eviction_count());
}

// ==================== RemoveKey Tests ====================

TEST_F(LruCacheIndexerTest, RemoveKeyExisting) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    indexer.Init({10.0}, kOneGB, kOneGB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(3, indexer.unique_count());

    EXPECT_TRUE(indexer.RemoveKey(2));
    EXPECT_EQ(2, indexer.unique_count());
    EXPECT_EQ(1, indexer.eviction_count());

    // Removed key should miss on re-access
    indexer.ProcessKeys({2}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]);
}

TEST_F(LruCacheIndexerTest, RemoveKeyNonExisting) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    indexer.Init({10.0}, kOneGB, kOneGB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    indexer.ProcessKeys({1}, hit_count, max_hit);
    EXPECT_FALSE(indexer.RemoveKey(999));
    EXPECT_EQ(1, indexer.unique_count());
    EXPECT_EQ(0, indexer.eviction_count());
}

TEST_F(LruCacheIndexerTest, RemoveKeyFromMultipleTiers) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    indexer.Init({5.0, 10.0}, kOneGB, kOneGB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    indexer.ProcessKeys({1, 2}, hit_count, max_hit);
    EXPECT_TRUE(indexer.RemoveKey(1));

    // Key 1 should miss on both tiers
    indexer.ProcessKeys({1}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]);
    EXPECT_EQ(0, hit_count[1]);
}

// ==================== Theoretical Max Cache Tests ====================

TEST_F(LruCacheIndexerTest, TheoreticalMaxCacheDoesNotLimitTracking) {
    LruCacheIndexer indexer(true);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    indexer.Init({10.0}, kOneGB, kOneGB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    for (int64_t i = 1; i <= 5; i++) {
        indexer.ProcessKeys({i}, hit_count, max_hit);
    }
    EXPECT_EQ(5, indexer.unique_count());

    indexer.PostQueryMaintenance();
    // Theoretical max cache is an effectively unlimited baseline, not another bounded capacity.
    EXPECT_EQ(5, indexer.unique_count());
    EXPECT_EQ(0, indexer.eviction_count());
}

TEST_F(LruCacheIndexerTest, TheoreticalMaxCacheDisabledUsesLargestCapacity) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneKB = 1024LL;
    indexer.Init({10.0}, kOneKB, kOneKB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    for (int64_t i = 1; i <= 100; i++) {
        indexer.ProcessKeys({i}, hit_count, max_hit);
    }
    indexer.PostQueryMaintenance();
    EXPECT_EQ(100, indexer.unique_count());
    EXPECT_EQ(0, indexer.eviction_count());
}

// ==================== Memory and Usage Metrics ====================

TEST_F(LruCacheIndexerTest, MemoryUsageBytesGrows) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    indexer.Init({10.0}, kOneGB, kOneGB, 1);

    EXPECT_EQ(0, indexer.memory_usage_bytes());

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);

    EXPECT_GT(indexer.memory_usage_bytes(), 0);
}

TEST_F(LruCacheIndexerTest, KvCacheUsageBytesGrows) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    indexer.Init({10.0}, kOneGB, kOneGB, 1);

    EXPECT_EQ(0, indexer.kv_cache_usage_bytes());

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);

    EXPECT_GT(indexer.kv_cache_usage_bytes(), 0);
}

TEST_F(LruCacheIndexerTest, KvCacheUsageBytesReflectsCharge) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    // Each key uses kOneGB as charge
    indexer.Init({100.0}, kOneGB, kOneGB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    indexer.ProcessKeys({1}, hit_count, max_hit);

    // Usage should be approximately kOneGB (for 1 key)
    int64_t usage = indexer.kv_cache_usage_bytes();
    EXPECT_GE(usage, kOneGB);
}

// ==================== max_hit_count Tests ====================

TEST_F(LruCacheIndexerTest, MaxHitCountWithTheoreticalCapacity) {
    LruCacheIndexer indexer(true);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    indexer.Init({1.0}, kOneGB, kOneGB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // First access: miss
    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(0, max_hit);

    // Re-access: all hit in max_cache
    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(3, max_hit);
}

// ==================== Edge Cases ====================

TEST_F(LruCacheIndexerTest, EmptyKeysVector) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    indexer.Init({1.0}, kOneGB, kOneGB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;
    indexer.ProcessKeys({}, hit_count, max_hit);

    EXPECT_EQ(1u, hit_count.size());
    EXPECT_EQ(0, hit_count[0]);
}

TEST_F(LruCacheIndexerTest, SingleKeyRepeatedAccess) {
    LruCacheIndexer indexer(0);
    // Use small charge per key
    indexer.Init({1.0}, 1024, 1024, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // First miss
    indexer.ProcessKeys({42}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]);

    // Repeated hits
    for (int i = 0; i < 10; i++) {
        indexer.ProcessKeys({42}, hit_count, max_hit);
        EXPECT_EQ(1, hit_count[0]);
    }

    EXPECT_EQ(1, indexer.unique_count());
}

TEST_F(LruCacheIndexerTest, LinearStepOneIsDefault) {
    // linear_step = 1 means every key is its own boundary
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    indexer.Init({10.0}, kOneGB, kOneGB, 1);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]);

    // All keys should hit individually
    indexer.ProcessKeys({1, 2, 3}, hit_count, max_hit);
    EXPECT_EQ(3, hit_count[0]);
}

TEST_F(LruCacheIndexerTest, LargeLinearStepGroupBehavior) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    // linear_step=4: step_hit at pos3,7,11,...
    // Step_hit keys use charge=size_full_linear, others use size_full_only
    indexer.Init({10.0}, kOneGB / 2, kOneGB, 4);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // Full group of 4 keys
    indexer.ProcessKeys({10, 20, 30, 40}, hit_count, max_hit);
    EXPECT_EQ(0, hit_count[0]); // first access, all miss

    // Re-access same group
    indexer.ProcessKeys({10, 20, 30, 40}, hit_count, max_hit);
    EXPECT_EQ(4, hit_count[0]); // all hit with correct boundary charge
}

TEST_F(LruCacheIndexerTest, GroupInvalidationPropagation) {
    LruCacheIndexer indexer(0);
    constexpr int64_t kOneGB = 1024LL * 1024 * 1024;
    // linear_step=2
    indexer.Init({10.0}, kOneGB, kOneGB, 2);

    std::vector<int64_t> hit_count;
    int64_t max_hit;

    // Populate group {1, 2}
    indexer.ProcessKeys({1, 2}, hit_count, max_hit);

    // Access {1, 2, 3, 4}: group {1,2} hits, group {3,4} misses
    // Prefix hit should be 2 (first group all hit)
    indexer.ProcessKeys({1, 2, 3, 4}, hit_count, max_hit);
    EXPECT_EQ(2, hit_count[0]);
}

} // namespace kv_cache_manager
