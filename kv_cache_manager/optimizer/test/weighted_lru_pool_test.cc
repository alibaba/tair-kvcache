#include <algorithm>
#include <cstdint>
#include <list>
#include <random>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/liteHit/weighted_lru_pool.h"

namespace kv_cache_manager {

namespace {

// Naive weighted LRU oracle: a recency list of typed objects with fixed
// per-type charges. RequiredBytes = own charge + bytes of all newer objects.
class NaiveWeightedLru {
public:
    NaiveWeightedLru(uint64_t full_charge, uint64_t mamba_charge) : charges_{full_charge, mamba_charge} {}

    void Touch(const CacheObjectKey &key) {
        Remove(key);
        entries_.push_front(key);
    }

    bool RequiredBytes(const CacheObjectKey &key, uint64_t &required) const {
        uint64_t newer = 0;
        for (const auto &entry : entries_) {
            if (Same(entry, key)) {
                required = newer + charges_[static_cast<int>(key.type)];
                return true;
            }
            newer += charges_[static_cast<int>(entry.type)];
        }
        return false;
    }

    uint64_t TotalBytes() const {
        uint64_t total = 0;
        for (const auto &entry : entries_) {
            total += charges_[static_cast<int>(entry.type)];
        }
        return total;
    }

    uint64_t FullObjectsWithinBytes(uint64_t budget) const {
        uint64_t cumulative = 0;
        uint64_t count = 0;
        for (const auto &entry : entries_) {
            cumulative += charges_[static_cast<int>(entry.type)];
            if (cumulative > budget) {
                break;
            }
            if (entry.type == CacheObjectType::kFull) {
                ++count;
            }
        }
        return count;
    }

private:
    static bool Same(const CacheObjectKey &a, const CacheObjectKey &b) {
        return a.type == b.type && a.prefix_block_key == b.prefix_block_key;
    }
    void Remove(const CacheObjectKey &key) {
        entries_.remove_if([&](const CacheObjectKey &entry) { return Same(entry, key); });
    }

    uint64_t charges_[2];
    std::list<CacheObjectKey> entries_;
};

} // namespace

class WeightedLruPoolTest : public TESTBASE {};

TEST_F(WeightedLruPoolTest, TypedKeysDoNotCollide) {
    WeightedLruPool pool(/*full=*/100, /*mamba=*/30);
    const CacheObjectKey full_key{CacheObjectType::kFull, 42};
    const CacheObjectKey mamba_key{CacheObjectType::kMamba, 42};

    pool.Touch(full_key);
    EXPECT_TRUE(pool.IsResident(full_key));
    EXPECT_FALSE(pool.IsResident(mamba_key));

    pool.Touch(mamba_key);
    EXPECT_EQ(1u, pool.resident_full_count());
    EXPECT_EQ(1u, pool.resident_mamba_count());
    EXPECT_EQ(130u, pool.resident_bytes());

    // Mamba was touched later, so full requires both charges, mamba only its own.
    uint64_t required = 0;
    ASSERT_TRUE(pool.RequiredBytes(mamba_key, required));
    EXPECT_EQ(30u, required);
    ASSERT_TRUE(pool.RequiredBytes(full_key, required));
    EXPECT_EQ(130u, required);
}

TEST_F(WeightedLruPoolTest, TouchMovesToMostRecent) {
    WeightedLruPool pool(/*full=*/10, /*mamba=*/4);
    const CacheObjectKey a{CacheObjectType::kFull, 1};
    const CacheObjectKey b{CacheObjectType::kFull, 2};
    const CacheObjectKey m{CacheObjectType::kMamba, 1};

    pool.Touch(a);
    pool.Touch(b);
    pool.Touch(m);
    // Order (old->new): a, b, m
    uint64_t required = 0;
    ASSERT_TRUE(pool.RequiredBytes(a, required));
    EXPECT_EQ(24u, required); // 10 + (10 + 4)

    pool.Touch(a);
    // Order: b, m, a
    ASSERT_TRUE(pool.RequiredBytes(a, required));
    EXPECT_EQ(10u, required);
    ASSERT_TRUE(pool.RequiredBytes(b, required));
    EXPECT_EQ(24u, required);
    EXPECT_EQ(24u, pool.resident_bytes());
}

TEST_F(WeightedLruPoolTest, NonResidentReturnsFalse) {
    WeightedLruPool pool(8, 2);
    uint64_t required = 0;
    EXPECT_FALSE(pool.RequiredBytes({CacheObjectType::kFull, 7}, required));
    pool.Touch({CacheObjectType::kFull, 7});
    EXPECT_TRUE(pool.RequiredBytes({CacheObjectType::kFull, 7}, required));
    pool.Reset();
    EXPECT_FALSE(pool.RequiredBytes({CacheObjectType::kFull, 7}, required));
    EXPECT_EQ(0u, pool.resident_bytes());
}

TEST_F(WeightedLruPoolTest, RandomizedOracleComparison) {
    std::mt19937 rng(20260729);
    WeightedLruPool pool(/*full=*/48, /*mamba=*/16);
    NaiveWeightedLru oracle(48, 16);

    std::uniform_int_distribution<int64_t> key_dist(0, 63);
    std::uniform_int_distribution<int> type_dist(0, 4);
    for (int step = 0; step < 20000; ++step) {
        const CacheObjectKey key{type_dist(rng) == 0 ? CacheObjectType::kMamba : CacheObjectType::kFull, key_dist(rng)};
        pool.Touch(key);
        oracle.Touch(key);

        if (step % 7 == 0) {
            const CacheObjectKey probe{type_dist(rng) == 0 ? CacheObjectType::kMamba : CacheObjectType::kFull,
                                       key_dist(rng)};
            uint64_t expected = 0;
            uint64_t actual = 0;
            const bool expected_resident = oracle.RequiredBytes(probe, expected);
            const bool actual_resident = pool.RequiredBytes(probe, actual);
            ASSERT_EQ(expected_resident, actual_resident) << "step " << step;
            if (expected_resident) {
                ASSERT_EQ(expected, actual) << "step " << step;
            }
        }
        if (step % 11 == 0) {
            for (uint64_t budget : {0ull, 16ull, 48ull, 100ull, 500ull, 5000ull}) {
                ASSERT_EQ(oracle.FullObjectsWithinBytes(budget), pool.FullObjectsWithinBytes(budget))
                    << "step " << step << " budget " << budget;
            }
        }
        ASSERT_EQ(oracle.TotalBytes(), pool.resident_bytes()) << "step " << step;
    }
}

TEST_F(WeightedLruPoolTest, CompactionPreservesOrderAndBytes) {
    WeightedLruPool pool(/*full=*/3, /*mamba=*/5);
    NaiveWeightedLru oracle(3, 5);

    // Force many dead positions: repeatedly touch a small key set far beyond
    // the compaction slack.
    for (int round = 0; round < 3000; ++round) {
        for (int64_t key = 0; key < 8; ++key) {
            const CacheObjectKey object{key % 3 == 0 ? CacheObjectType::kMamba : CacheObjectType::kFull, key};
            pool.Touch(object);
            oracle.Touch(object);
        }
    }
    for (int64_t key = 0; key < 8; ++key) {
        const CacheObjectKey object{key % 3 == 0 ? CacheObjectType::kMamba : CacheObjectType::kFull, key};
        uint64_t expected = 0;
        uint64_t actual = 0;
        ASSERT_TRUE(oracle.RequiredBytes(object, expected));
        ASSERT_TRUE(pool.RequiredBytes(object, actual));
        EXPECT_EQ(expected, actual) << "key " << key;
    }
    EXPECT_EQ(oracle.TotalBytes(), pool.resident_bytes());
}

} // namespace kv_cache_manager
