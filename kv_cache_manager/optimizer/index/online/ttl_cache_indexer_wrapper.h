#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kv_cache_manager/optimizer/index/online/cache_indexer.h"

namespace kv_cache_manager {

class TtlCacheIndexerWrapper : public CacheIndexer {
public:
    using ClockFunc = std::function<int64_t()>;

    TtlCacheIndexerWrapper(std::unique_ptr<CacheIndexer> inner, int64_t ttl_seconds);

    TtlCacheIndexerWrapper(std::unique_ptr<CacheIndexer> inner, int64_t ttl_seconds, ClockFunc clock);

    void Init(const std::vector<double> &capacity_gb,
              int64_t size_full_only,
              int64_t size_full_linear,
              int32_t linear_step) override;

    void ProcessKeys(const std::vector<int64_t> &keys,
                     std::vector<int64_t> &hit_count,
                     int64_t &max_hit_count,
                     std::vector<bool> *key_hits = nullptr) override;

    int64_t unique_count() const override;
    int64_t eviction_count() const override;
    int64_t ttl_eviction_count() const override;
    int64_t memory_usage_bytes() const override;
    int64_t kv_cache_usage_bytes() const override;
    std::vector<int64_t> capacity_unique_counts() const override;

    void PostQueryMaintenance() override;
    bool RemoveKey(int64_t key) override;
    std::vector<HitAgeBucketInfo> GetHitAgeBuckets() const override;

    // Configure the age bucket thresholds (in seconds, ascending order).
    // The last bucket implicitly covers [last_threshold, +inf).
    // Default: {5, 30, 60, 120, 300, 600, 1800, 3600, 7200}.
    void SetHitAgeBucketThresholds(const std::vector<int64_t> &thresholds);

private:
    void HarvestExpired(int64_t now);
    size_t FindAgeBucket(int64_t age_seconds) const;

    std::unique_ptr<CacheIndexer> inner_;
    int64_t ttl_seconds_;
    ClockFunc clock_;

    std::unordered_map<int64_t, int64_t> key_access_time_;
    std::set<std::pair<int64_t, int64_t>> expire_set_;
    int64_t ttl_eviction_count_ = 0;

    // Age bucket thresholds (sorted ascending). Each value is the upper bound
    // of a bucket. A final "+inf" bucket is always appended.
    std::vector<int64_t> hit_age_thresholds_ = {5, 30, 60, 120, 300, 600, 1800, 3600, 7200};
    // hit_age_bucket_counts_ has size = hit_age_thresholds_.size() + 1
    // buckets: [0,5) [5,30) [30,60) ... [7200, +inf)
    std::vector<int64_t> hit_age_bucket_counts_;
};

} // namespace kv_cache_manager
