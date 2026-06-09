#include "kv_cache_manager/online_optimizer/indexer/ttl_cache_indexer_wrapper.h"

#include "kv_cache_manager/common/timestamp_util.h"

namespace kv_cache_manager {

TtlCacheIndexerWrapper::TtlCacheIndexerWrapper(std::unique_ptr<CacheIndexer> inner,
                                               int64_t ttl_seconds)
    : TtlCacheIndexerWrapper(std::move(inner), ttl_seconds,
                             []() { return TimestampUtil::GetCurrentTimeSec(); }) {}

TtlCacheIndexerWrapper::TtlCacheIndexerWrapper(std::unique_ptr<CacheIndexer> inner,
                                               int64_t ttl_seconds,
                                               ClockFunc clock)
    : inner_(std::move(inner))
    , ttl_seconds_(ttl_seconds)
    , clock_(std::move(clock)) {}

void TtlCacheIndexerWrapper::Init(const std::vector<double> &capacity_gb,
                                  int64_t size_full_only,
                                  int64_t size_full_linear,
                                  int32_t linear_step) {
    inner_->Init(capacity_gb, size_full_only, size_full_linear, linear_step);
}

void TtlCacheIndexerWrapper::ProcessKeys(const std::vector<int64_t> &keys,
                                         std::vector<int64_t> &hit_count,
                                         int64_t &max_hit_count) {
    int64_t now = clock_();

    HarvestExpired(now);

    inner_->ProcessKeys(keys, hit_count, max_hit_count);

    for (int64_t key : keys) {
        auto it = key_access_time_.find(key);
        if (it != key_access_time_.end()) {
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

int64_t TtlCacheIndexerWrapper::unique_count() const {
    return inner_->unique_count();
}

int64_t TtlCacheIndexerWrapper::eviction_count() const {
    return inner_->eviction_count();
}

int64_t TtlCacheIndexerWrapper::ttl_eviction_count() const {
    return ttl_eviction_count_;
}

int64_t TtlCacheIndexerWrapper::memory_usage_bytes() const {
    constexpr int64_t kHashMapEntryBytes = 56;
    constexpr int64_t kSetNodeBytes = 48;
    int64_t ttl_bytes = static_cast<int64_t>(key_access_time_.size()) * kHashMapEntryBytes
                      + static_cast<int64_t>(expire_set_.size()) * kSetNodeBytes;
    return inner_->memory_usage_bytes() + ttl_bytes;
}

int64_t TtlCacheIndexerWrapper::kv_cache_usage_bytes() const {
    return inner_->kv_cache_usage_bytes();
}

void TtlCacheIndexerWrapper::PostQueryMaintenance() {
    inner_->PostQueryMaintenance();
}

bool TtlCacheIndexerWrapper::RemoveKey(int64_t key) {
    auto it = key_access_time_.find(key);
    if (it != key_access_time_.end()) {
        expire_set_.erase({it->second + ttl_seconds_, key});
        key_access_time_.erase(it);
    }
    return inner_->RemoveKey(key);
}

} // namespace kv_cache_manager
