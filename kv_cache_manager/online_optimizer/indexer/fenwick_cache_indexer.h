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

    void Init(const std::vector<double> &capacity_gb,
              int64_t size_full_only,
              int64_t size_full_linear,
              int32_t linear_step) override;

    void
    ProcessKeys(const std::vector<int64_t> &keys, std::vector<int64_t> &hit_count, int64_t &max_hit_count) override;
    int64_t unique_count() const override { return unique_count_; }
    int64_t eviction_count() const override { return eviction_count_; }
    int64_t memory_usage_bytes() const override;
    int64_t kv_cache_usage_bytes() const override;

    void PostQueryMaintenance() override;
    bool RemoveKey(int64_t key) override;

    // Exposed for testing: compute stack distance for a single key.
    int64_t ComputeStackDistance(int64_t key);

private:
    void EvictIfExceedsCapacity();
    void CompactIfNeeded();
    void DoEvictOne();
    void DoCompact();

    int64_t max_key_count_;
    int64_t avg_bytes_per_block_ = 0;
    std::vector<int64_t> capacity_blocks_;
    int64_t logical_time_ = 0;
    int64_t unique_count_ = 0;
    int64_t eviction_count_ = 0;
    int64_t total_slots_;
    FenwickTree fenwick_;
    std::unordered_map<int64_t, int64_t> last_access_;
    std::unordered_map<int64_t, int64_t> reverse_map_;
};

} // namespace kv_cache_manager
