#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/online_optimizer/indexer/cache_indexer.h"

namespace kv_cache_manager {

class FenwickTree {
public:
    explicit FenwickTree(int64_t capacity);

    void Update(int64_t idx, int64_t delta);
    int64_t PrefixSum(int64_t idx) const;
    int64_t RangeSum(int64_t lo, int64_t hi) const;

    int64_t FindFirst(int64_t target) const;

    void Reset(int64_t new_capacity);
    int64_t capacity() const { return capacity_; }

private:
    int64_t capacity_;
    std::vector<int64_t> tree_;
};

class FenwickCacheIndexer : public CacheIndexer {
public:
    explicit FenwickCacheIndexer(int64_t max_key_count);

    int64_t ProcessKey(int64_t key) override;
    int64_t unique_count() const override { return unique_count_; }
    int64_t peak_unique_count() const override { return peak_unique_count_; }
    int64_t eviction_count() const override { return eviction_count_; }
    int64_t memory_usage_bytes() const override;

    void PostQueryMaintenance() override;

private:
    void EvictIfExceedsCapacity();
    void CompactIfNeeded();
    void DoEvictOne();
    void DoCompact();

    int64_t max_key_count_;
    int64_t logical_time_ = 0;
    int64_t unique_count_ = 0;
    int64_t peak_unique_count_ = 0;
    int64_t eviction_count_ = 0;
    int64_t total_slots_;
    FenwickTree fenwick_;
    std::unordered_map<int64_t, int64_t> last_access_;
    std::unordered_map<int64_t, int64_t> reverse_map_;
};

} // namespace kv_cache_manager
