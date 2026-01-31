//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/cache/lru_cache.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

class LRUCacheTest : public TESTBASE {
public:
    LRUCacheTest() = default;
    ~LRUCacheTest() override { DeleteCache(); }

    void DeleteCache() {
        if (cache_ != nullptr) {
            cache_->~LRUCacheShard();
            port::cacheline_aligned_free(cache_);
            cache_ = nullptr;
        }
    }

    void NewCache(size_t capacity,
                  double high_pri_pool_ratio = 0.0,
                  double low_pri_pool_ratio = 1.0,
                  bool use_adaptive_mutex = kDefaultToAdaptiveMutex) {
        DeleteCache();
        cache_ = static_cast<LRUCacheShard *>(port::cacheline_aligned_alloc(sizeof(LRUCacheShard)));
        new (cache_) LRUCacheShard(capacity,
                                   /*strict_capacity_limit=*/false,
                                   high_pri_pool_ratio,
                                   low_pri_pool_ratio,
                                   use_adaptive_mutex,
                                   kDontChargeCacheMetadata,
                                   /*max_upper_hash_bits=*/24,
                                   /*allocator*/ nullptr,
                                   &eviction_callback_);
    }

    void Insert(const std::string &key, Cache::Priority priority = Cache::Priority::LOW, size_t charge = 1) {
        ASSERT_EQ(EC_OK,
                  cache_->Insert(
                      key, 0 /*hash*/, nullptr /*value*/, &kNoopCacheItemHelper, charge, nullptr /*handle*/, priority));
    }

    void Insert(char key, Cache::Priority priority = Cache::Priority::LOW) { Insert(std::string(1, key), priority); }

    bool Lookup(const std::string &key) {
        auto handle = cache_->Lookup(key, 0 /*hash*/, nullptr, nullptr, Cache::Priority::LOW, nullptr);
        if (handle) {
            cache_->Release(handle, true /*useful*/, false /*erase*/);
            return true;
        }
        return false;
    }

    bool Lookup(char key) { return Lookup(std::string(1, key)); }

    void Erase(const std::string &key) { cache_->Erase(key, 0 /*hash*/); }

    void ValidateLRUList(std::vector<std::string> keys,
                         size_t num_high_pri_pool_keys = 0,
                         size_t num_low_pri_pool_keys = 0,
                         size_t num_bottom_pri_pool_keys = 0) {
        LRUHandle *lru;
        LRUHandle *lru_low_pri;
        LRUHandle *lru_bottom_pri;
        cache_->TEST_GetLRUList(&lru, &lru_low_pri, &lru_bottom_pri);

        LRUHandle *iter = lru;

        bool in_low_pri_pool = false;
        bool in_high_pri_pool = false;

        size_t high_pri_pool_keys = 0;
        size_t low_pri_pool_keys = 0;
        size_t bottom_pri_pool_keys = 0;

        if (iter == lru_bottom_pri) {
            in_low_pri_pool = true;
            in_high_pri_pool = false;
        }
        if (iter == lru_low_pri) {
            in_low_pri_pool = false;
            in_high_pri_pool = true;
        }

        for (const auto &key : keys) {
            iter = iter->next;
            ASSERT_NE(lru, iter);
            ASSERT_EQ(key, std::string(iter->key()));
            ASSERT_EQ(in_high_pri_pool, iter->InHighPriPool());
            ASSERT_EQ(in_low_pri_pool, iter->InLowPriPool());
            if (in_high_pri_pool) {
                ASSERT_FALSE(iter->InLowPriPool());
                high_pri_pool_keys++;
            } else if (in_low_pri_pool) {
                ASSERT_FALSE(iter->InHighPriPool());
                low_pri_pool_keys++;
            } else {
                bottom_pri_pool_keys++;
            }
            if (iter == lru_bottom_pri) {
                ASSERT_FALSE(in_low_pri_pool);
                ASSERT_FALSE(in_high_pri_pool);
                in_low_pri_pool = true;
                in_high_pri_pool = false;
            }
            if (iter == lru_low_pri) {
                ASSERT_TRUE(in_low_pri_pool);
                ASSERT_FALSE(in_high_pri_pool);
                in_low_pri_pool = false;
                in_high_pri_pool = true;
            }
        }
        ASSERT_EQ(lru, iter->next);
        ASSERT_FALSE(in_low_pri_pool);
        ASSERT_TRUE(in_high_pri_pool);
        ASSERT_EQ(num_high_pri_pool_keys, high_pri_pool_keys);
        ASSERT_EQ(num_low_pri_pool_keys, low_pri_pool_keys);
        ASSERT_EQ(num_bottom_pri_pool_keys, bottom_pri_pool_keys);
    }

protected:
    LRUCacheShard *cache_ = nullptr;

private:
    Cache::EvictionCallback eviction_callback_;
};

TEST_F(LRUCacheTest, BasicLRU) {
    NewCache(5);
    for (char ch = 'a'; ch <= 'e'; ch++) {
        Insert(ch);
    }
    ValidateLRUList({"a", "b", "c", "d", "e"}, 0, 5);
    for (char ch = 'x'; ch <= 'z'; ch++) {
        Insert(ch);
    }
    ValidateLRUList({"d", "e", "x", "y", "z"}, 0, 5);
    ASSERT_FALSE(Lookup("b"));
    ValidateLRUList({"d", "e", "x", "y", "z"}, 0, 5);
    ASSERT_TRUE(Lookup("e"));
    ValidateLRUList({"d", "x", "y", "z", "e"}, 0, 5);
    ASSERT_TRUE(Lookup("z"));
    ValidateLRUList({"d", "x", "y", "e", "z"}, 0, 5);
    Erase("x");
    ValidateLRUList({"d", "y", "e", "z"}, 0, 4);
    ASSERT_TRUE(Lookup("d"));
    ValidateLRUList({"y", "e", "z", "d"}, 0, 4);
    Insert("u");
    ValidateLRUList({"y", "e", "z", "d", "u"}, 0, 5);
    Insert("v");
    ValidateLRUList({"e", "z", "d", "u", "v"}, 0, 5);
}

TEST_F(LRUCacheTest, LowPriorityMidpointInsertion) {
    // Allocate 2 cache entries to high-pri pool and 3 to low-pri pool.
    NewCache(5, /* high_pri_pool_ratio */ 0.40, /* low_pri_pool_ratio */ 0.60);

    Insert("a", Cache::Priority::LOW);
    Insert("b", Cache::Priority::LOW);
    Insert("c", Cache::Priority::LOW);
    Insert("x", Cache::Priority::HIGH);
    Insert("y", Cache::Priority::HIGH);
    ValidateLRUList({"a", "b", "c", "x", "y"}, 2, 3);

    // Low-pri entries inserted to the tail of low-pri list (the midpoint).
    // After lookup, it will move to the tail of the full list.
    Insert("d", Cache::Priority::LOW);
    ValidateLRUList({"b", "c", "d", "x", "y"}, 2, 3);
    ASSERT_TRUE(Lookup("d"));
    ValidateLRUList({"b", "c", "x", "y", "d"}, 2, 3);

    // High-pri entries will be inserted to the tail of full list.
    Insert("z", Cache::Priority::HIGH);
    ValidateLRUList({"c", "x", "y", "d", "z"}, 2, 3);
}

TEST_F(LRUCacheTest, BottomPriorityMidpointInsertion) {
    // Allocate 2 cache entries to high-pri pool and 2 to low-pri pool.
    NewCache(6, /* high_pri_pool_ratio */ 0.35, /* low_pri_pool_ratio */ 0.35);

    Insert("a", Cache::Priority::BOTTOM);
    Insert("b", Cache::Priority::BOTTOM);
    Insert("i", Cache::Priority::LOW);
    Insert("j", Cache::Priority::LOW);
    Insert("x", Cache::Priority::HIGH);
    Insert("y", Cache::Priority::HIGH);
    ValidateLRUList({"a", "b", "i", "j", "x", "y"}, 2, 2, 2);

    // Low-pri entries will be inserted to the tail of low-pri list (the
    // midpoint). After lookup, 'k' will move to the tail of the full list, and
    // 'x' will spill over to the low-pri pool.
    Insert("k", Cache::Priority::LOW);
    ValidateLRUList({"b", "i", "j", "k", "x", "y"}, 2, 2, 2);
    ASSERT_TRUE(Lookup("k"));
    ValidateLRUList({"b", "i", "j", "x", "y", "k"}, 2, 2, 2);

    // High-pri entries will be inserted to the tail of full list. Although y was
    // inserted with high priority, it got spilled over to the low-pri pool. As
    // a result, j also got spilled over to the bottom-pri pool.
    Insert("z", Cache::Priority::HIGH);
    ValidateLRUList({"i", "j", "x", "y", "k", "z"}, 2, 2, 2);
    Erase("x");
    ValidateLRUList({"i", "j", "y", "k", "z"}, 2, 1, 2);
    Erase("y");
    ValidateLRUList({"i", "j", "k", "z"}, 2, 0, 2);

    // Bottom-pri entries will be inserted to the tail of bottom-pri list.
    Insert("c", Cache::Priority::BOTTOM);
    ValidateLRUList({"i", "j", "c", "k", "z"}, 2, 0, 3);
    Insert("d", Cache::Priority::BOTTOM);
    ValidateLRUList({"i", "j", "c", "d", "k", "z"}, 2, 0, 4);
    Insert("e", Cache::Priority::BOTTOM);
    ValidateLRUList({"j", "c", "d", "e", "k", "z"}, 2, 0, 4);

    // Low-pri entries will be inserted to the tail of low-pri list (the
    // midpoint).
    Insert("l", Cache::Priority::LOW);
    ValidateLRUList({"c", "d", "e", "l", "k", "z"}, 2, 1, 3);
    Insert("m", Cache::Priority::LOW);
    ValidateLRUList({"d", "e", "l", "m", "k", "z"}, 2, 2, 2);

    Erase("k");
    ValidateLRUList({"d", "e", "l", "m", "z"}, 1, 2, 2);
    Erase("z");
    ValidateLRUList({"d", "e", "l", "m"}, 0, 2, 2);

    // Bottom-pri entries will be inserted to the tail of bottom-pri list.
    Insert("f", Cache::Priority::BOTTOM);
    ValidateLRUList({"d", "e", "f", "l", "m"}, 0, 2, 3);
    Insert("g", Cache::Priority::BOTTOM);
    ValidateLRUList({"d", "e", "f", "g", "l", "m"}, 0, 2, 4);

    // High-pri entries will be inserted to the tail of full list.
    Insert("o", Cache::Priority::HIGH);
    ValidateLRUList({"e", "f", "g", "l", "m", "o"}, 1, 2, 3);
    Insert("p", Cache::Priority::HIGH);
    ValidateLRUList({"f", "g", "l", "m", "o", "p"}, 2, 2, 2);
}

TEST_F(LRUCacheTest, EntriesWithPriority) {
    // Allocate 2 cache entries to high-pri pool and 2 to low-pri pool.
    NewCache(6, /* high_pri_pool_ratio */ 0.35, /* low_pri_pool_ratio */ 0.35);

    Insert("a", Cache::Priority::LOW);
    Insert("b", Cache::Priority::LOW);
    ValidateLRUList({"a", "b"}, 0, 2, 0);
    // Low-pri entries can overflow to bottom-pri pool.
    Insert("c", Cache::Priority::LOW);
    ValidateLRUList({"a", "b", "c"}, 0, 2, 1);

    // Bottom-pri entries can take high-pri pool capacity if available
    Insert("t", Cache::Priority::LOW);
    Insert("u", Cache::Priority::LOW);
    ValidateLRUList({"a", "b", "c", "t", "u"}, 0, 2, 3);
    Insert("v", Cache::Priority::LOW);
    ValidateLRUList({"a", "b", "c", "t", "u", "v"}, 0, 2, 4);
    Insert("w", Cache::Priority::LOW);
    ValidateLRUList({"b", "c", "t", "u", "v", "w"}, 0, 2, 4);

    Insert("X", Cache::Priority::HIGH);
    Insert("Y", Cache::Priority::HIGH);
    ValidateLRUList({"t", "u", "v", "w", "X", "Y"}, 2, 2, 2);

    // After lookup, the high-pri entry 'X' got spilled over to the low-pri pool.
    // The low-pri entry 'v' got spilled over to the bottom-pri pool.
    Insert("Z", Cache::Priority::HIGH);
    ValidateLRUList({"u", "v", "w", "X", "Y", "Z"}, 2, 2, 2);

    // Low-pri entries will be inserted to head of low-pri pool.
    Insert("a", Cache::Priority::LOW);
    ValidateLRUList({"v", "w", "X", "a", "Y", "Z"}, 2, 2, 2);

    // After lookup, the high-pri entry 'Y' got spilled over to the low-pri pool.
    // The low-pri entry 'X' got spilled over to the bottom-pri pool.
    ASSERT_TRUE(Lookup("v"));
    ValidateLRUList({"w", "X", "a", "Y", "Z", "v"}, 2, 2, 2);

    // After lookup, the high-pri entry 'Z' got spilled over to the low-pri pool.
    // The low-pri entry 'a' got spilled over to the bottom-pri pool.
    ASSERT_TRUE(Lookup("X"));
    ValidateLRUList({"w", "a", "Y", "Z", "v", "X"}, 2, 2, 2);

    // After lookup, the low pri entry 'Z' got promoted back to high-pri pool. The
    // high-pri entry 'v' got spilled over to the low-pri pool.
    ASSERT_TRUE(Lookup("Z"));
    ValidateLRUList({"w", "a", "Y", "v", "X", "Z"}, 2, 2, 2);

    Erase("Y");
    ValidateLRUList({"w", "a", "v", "X", "Z"}, 2, 1, 2);
    Erase("X");
    ValidateLRUList({"w", "a", "v", "Z"}, 1, 1, 2);

    Insert("d", Cache::Priority::LOW);
    Insert("e", Cache::Priority::LOW);
    ValidateLRUList({"w", "a", "v", "d", "e", "Z"}, 1, 2, 3);

    Insert("f", Cache::Priority::LOW);
    Insert("g", Cache::Priority::LOW);
    ValidateLRUList({"v", "d", "e", "f", "g", "Z"}, 1, 2, 3);
    ASSERT_TRUE(Lookup("d"));
    ValidateLRUList({"v", "e", "f", "g", "Z", "d"}, 2, 2, 2);

    // Erase some entries.
    Erase("e");
    Erase("f");
    Erase("Z");
    ValidateLRUList({"v", "g", "d"}, 1, 1, 1);

    // Bottom-pri entries can take low- and high-pri pool capacity if available
    Insert("o", Cache::Priority::BOTTOM);
    ValidateLRUList({"v", "o", "g", "d"}, 1, 1, 2);
    Insert("p", Cache::Priority::BOTTOM);
    ValidateLRUList({"v", "o", "p", "g", "d"}, 1, 1, 3);
    Insert("q", Cache::Priority::BOTTOM);
    ValidateLRUList({"v", "o", "p", "q", "g", "d"}, 1, 1, 4);

    // High-pri entries can overflow to low-pri pool, and bottom-pri entries will
    // be evicted.
    Insert("x", Cache::Priority::HIGH);
    ValidateLRUList({"o", "p", "q", "g", "d", "x"}, 2, 1, 3);
    Insert("y", Cache::Priority::HIGH);
    ValidateLRUList({"p", "q", "g", "d", "x", "y"}, 2, 2, 2);
    Insert("z", Cache::Priority::HIGH);
    ValidateLRUList({"q", "g", "d", "x", "y", "z"}, 2, 2, 2);

    // 'g' is bottom-pri before this lookup, it will be inserted to head of
    // high-pri pool after lookup.
    ASSERT_TRUE(Lookup("g"));
    ValidateLRUList({"q", "d", "x", "y", "z", "g"}, 2, 2, 2);

    // High-pri entries will be inserted to head of high-pri pool after lookup.
    ASSERT_TRUE(Lookup("z"));
    ValidateLRUList({"q", "d", "x", "y", "g", "z"}, 2, 2, 2);

    // Bottom-pri entries will be inserted to head of high-pri pool after lookup.
    ASSERT_TRUE(Lookup("d"));
    ValidateLRUList({"q", "x", "y", "g", "z", "d"}, 2, 2, 2);

    // Bottom-pri entries will be inserted to the tail of bottom-pri list.
    Insert("m", Cache::Priority::BOTTOM);
    ValidateLRUList({"x", "m", "y", "g", "z", "d"}, 2, 2, 2);

    // Bottom-pri entries will be inserted to head of high-pri pool after lookup.
    ASSERT_TRUE(Lookup("m"));
    ValidateLRUList({"x", "y", "g", "z", "d", "m"}, 2, 2, 2);
}

} // namespace kv_cache_manager
