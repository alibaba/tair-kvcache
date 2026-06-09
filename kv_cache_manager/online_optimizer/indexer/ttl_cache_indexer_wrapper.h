#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kv_cache_manager/online_optimizer/indexer/cache_indexer.h"

namespace kv_cache_manager {

class TtlCacheIndexerWrapper : public CacheIndexer {
public:
    using ClockFunc = std::function<int64_t()>;

    TtlCacheIndexerWrapper(std::unique_ptr<CacheIndexer> inner,
                           int64_t ttl_seconds);

    TtlCacheIndexerWrapper(std::unique_ptr<CacheIndexer> inner,
                           int64_t ttl_seconds,
                           ClockFunc clock);

    void Init(const std::vector<double> &capacity_gb,
              int64_t size_full_only,
              int64_t size_full_linear,
              int32_t linear_step) override;

    void ProcessKeys(const std::vector<int64_t> &keys,
                     std::vector<int64_t> &hit_count,
                     int64_t &max_hit_count) override;

    int64_t unique_count() const override;
    int64_t eviction_count() const override;
    int64_t ttl_eviction_count() const override;
    int64_t memory_usage_bytes() const override;
    int64_t kv_cache_usage_bytes() const override;

    void PostQueryMaintenance() override;
    bool RemoveKey(int64_t key) override;

private:
    void HarvestExpired(int64_t now);

    std::unique_ptr<CacheIndexer> inner_;
    int64_t ttl_seconds_;
    ClockFunc clock_;

    std::unordered_map<int64_t, int64_t> key_access_time_;
    std::set<std::pair<int64_t, int64_t>> expire_set_;
    int64_t ttl_eviction_count_ = 0;
};

} // namespace kv_cache_manager
