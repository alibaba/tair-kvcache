#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "kv_cache_manager/common/cache/advanced_cache.h"
#include "kv_cache_manager/online_optimizer/indexer/cache_indexer.h"

namespace kv_cache_manager {

class LruCacheIndexer : public CacheIndexer {
public:
    explicit LruCacheIndexer(int64_t max_key_count);

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

private:
    void RebuildCaches();
    void
    ProcessKeysFullAttention(const std::vector<int64_t> &keys, std::vector<int64_t> &hit_count, int64_t &max_hit_count);
    void ProcessKeysLinearAttention(const std::vector<int64_t> &keys,
                                    std::vector<int64_t> &hit_count,
                                    int64_t &max_hit_count);

    // Full-attention helper: lookup key, insert with size_full_only_ on miss.
    bool LookupAndInsert(Cache *cache, std::string_view key_sv, bool &is_new_key);

    // Linear-attention helper: lookup key, insert with desired_charge on miss.
    // On hit with is_linear && charge mismatch, updates charge.
    // Returns true on hit. Sets is_checkpoint if stored_charge == size_full_linear_.
    bool LookupAndInsert(Cache *cache,
                         std::string_view key_sv,
                         bool is_linear,
                         int64_t desired_charge,
                         bool &is_new_key,
                         bool &is_checkpoint);

    int64_t max_key_count_;
    int64_t unique_count_ = 0;
    int64_t eviction_count_ = 0;

    int64_t size_full_only_ = 0;
    int64_t size_full_linear_ = 0;
    int32_t linear_step_ = 1;

    std::vector<int64_t> capacity_bytes_;

    static const Cache::CacheItemHelper kHelper;
    std::vector<std::shared_ptr<Cache>> caches_;
    std::shared_ptr<Cache> max_cache_;
};

} // namespace kv_cache_manager
