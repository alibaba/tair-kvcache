#include "kv_cache_manager/optimizer/index/online/ttl_cache_indexer_wrapper.h"

#include <algorithm>

#include "kv_cache_manager/common/timestamp_util.h"

namespace kv_cache_manager {

TtlCacheIndexerWrapper::TtlCacheIndexerWrapper(std::unique_ptr<CacheIndexer> inner, int64_t ttl_seconds)
    : TtlCacheIndexerWrapper(std::move(inner), ttl_seconds, []() { return TimestampUtil::GetCurrentTimeSec(); }) {}

TtlCacheIndexerWrapper::TtlCacheIndexerWrapper(std::unique_ptr<CacheIndexer> inner,
                                               int64_t ttl_seconds,
                                               ClockFunc clock)
    : inner_(std::move(inner))
    , ttl_seconds_(ttl_seconds)
    , clock_(std::move(clock))
    , hit_age_bucket_counts_(hit_age_thresholds_.size() + 1, 0) {}

void TtlCacheIndexerWrapper::Init(const std::vector<double> &capacity_gb,
                                  int64_t size_full_only,
                                  int64_t size_full_linear,
                                  int32_t linear_step) {
    inner_->Init(capacity_gb, size_full_only, size_full_linear, linear_step);
}

void TtlCacheIndexerWrapper::ProcessKeys(const std::vector<int64_t> &keys,
                                         std::vector<int64_t> &hit_count,
                                         int64_t &max_hit_count,
                                         std::vector<bool> *key_hits) {
    int64_t now = clock_();

    HarvestExpired(now);

    std::vector<bool> inner_key_hits;
    inner_->ProcessKeys(keys, hit_count, max_hit_count, &inner_key_hits);
    if (key_hits) {
        *key_hits = inner_key_hits;
    }

    for (size_t i = 0; i < keys.size(); ++i) {
        int64_t key = keys[i];
        auto it = key_access_time_.find(key);
        if (it != key_access_time_.end()) {
            if (i < inner_key_hits.size() && inner_key_hits[i]) {
                int64_t age_seconds = now - it->second;
                size_t bucket_index = FindAgeBucket(age_seconds);
                hit_age_bucket_counts_[bucket_index]++;
            }

            expire_set_.erase({it->second + ttl_seconds_, key});
            it->second = now;
            expire_set_.insert({now + ttl_seconds_, key});
        } else {
            key_access_time_[key] = now;
            expire_set_.insert({now + ttl_seconds_, key});
        }
    }
}

void TtlCacheIndexerWrapper::HarvestExpired(int64_t now) {
    while (!expire_set_.empty()) {
        auto it = expire_set_.begin();
        if (it->first > now) {
            break;
        }
        int64_t key = it->second;
        expire_set_.erase(it);
        key_access_time_.erase(key);
        if (inner_->RemoveKey(key)) {
            ttl_eviction_count_++;
        }
    }
}

int64_t TtlCacheIndexerWrapper::unique_count() const { return inner_->unique_count(); }

int64_t TtlCacheIndexerWrapper::eviction_count() const { return inner_->eviction_count(); }

int64_t TtlCacheIndexerWrapper::ttl_eviction_count() const { return ttl_eviction_count_; }

int64_t TtlCacheIndexerWrapper::memory_usage_bytes() const {
    constexpr int64_t kHashMapEntryBytes = 56;
    constexpr int64_t kSetNodeBytes = 48;
    int64_t ttl_bytes = static_cast<int64_t>(key_access_time_.size()) * kHashMapEntryBytes +
                        static_cast<int64_t>(expire_set_.size()) * kSetNodeBytes;
    return inner_->memory_usage_bytes() + ttl_bytes;
}

int64_t TtlCacheIndexerWrapper::kv_cache_usage_bytes() const { return inner_->kv_cache_usage_bytes(); }

std::vector<int64_t> TtlCacheIndexerWrapper::capacity_unique_counts() const { return inner_->capacity_unique_counts(); }

void TtlCacheIndexerWrapper::PostQueryMaintenance() { inner_->PostQueryMaintenance(); }

bool TtlCacheIndexerWrapper::RemoveKey(int64_t key) {
    auto it = key_access_time_.find(key);
    if (it != key_access_time_.end()) {
        expire_set_.erase({it->second + ttl_seconds_, key});
        key_access_time_.erase(it);
    }
    return inner_->RemoveKey(key);
}

std::vector<HitAgeBucketInfo> TtlCacheIndexerWrapper::GetHitAgeBuckets() const {
    std::vector<HitAgeBucketInfo> result;
    result.reserve(hit_age_bucket_counts_.size());
    for (size_t i = 0; i < hit_age_thresholds_.size(); i++) {
        result.push_back({hit_age_thresholds_[i], hit_age_bucket_counts_[i]});
    }
    // The last bucket covers [last_threshold, +inf), threshold=0 means infinity
    result.push_back({0, hit_age_bucket_counts_.back()});
    return result;
}

void TtlCacheIndexerWrapper::SetHitAgeBucketThresholds(const std::vector<int64_t> &thresholds) {
    hit_age_thresholds_ = thresholds;
    std::sort(hit_age_thresholds_.begin(), hit_age_thresholds_.end());
    hit_age_bucket_counts_.assign(hit_age_thresholds_.size() + 1, 0);
}

size_t TtlCacheIndexerWrapper::FindAgeBucket(int64_t age_seconds) const {
    // Find the first threshold >= age_seconds.
    // Buckets: [0, t0] (t0+1, t1] ... (t_{n-1}, +inf)
    // age=3 with thresholds {5,30} → lower_bound → 5 → bucket 0 ("5s")
    // age=10 with thresholds {5,30} → lower_bound → 30 → bucket 1 ("30s")
    // age=100 with thresholds {5,30} → lower_bound → end → bucket 2 ("+inf")
    auto it = std::lower_bound(hit_age_thresholds_.begin(), hit_age_thresholds_.end(), age_seconds);
    return static_cast<size_t>(it - hit_age_thresholds_.begin());
}

} // namespace kv_cache_manager
