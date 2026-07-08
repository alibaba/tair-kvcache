#include "kv_cache_manager/optimizer/index/online/lru_cache_indexer.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "kv_cache_manager/common/cache/cache.h"

namespace kv_cache_manager {

const Cache::CacheItemHelper LruCacheIndexer::kHelper(CacheEntryRole::kMisc, nullptr);

LruCacheIndexer::LruCacheIndexer(bool enable_theoretical_max_cache)
    : enable_theoretical_max_cache_(enable_theoretical_max_cache) {}

void LruCacheIndexer::Init(const std::vector<double> &capacity_gb,
                           int64_t size_full_only,
                           int64_t size_full_linear,
                           int32_t linear_step) {
    size_full_only_ = size_full_only;
    size_full_linear_ = size_full_linear;
    linear_step_ = std::max(linear_step, int32_t(0));

    capacity_bytes_.resize(capacity_gb.size());
    for (size_t i = 0; i < capacity_gb.size(); i++) {
        capacity_bytes_[i] = static_cast<int64_t>(capacity_gb[i] * 1024.0 * 1024.0 * 1024.0);
    }

    RebuildCaches();

    if (enable_theoretical_max_cache_) {
        constexpr size_t kTheoreticalMaxCacheBytes = std::numeric_limits<size_t>::max() / 4;
        max_cache_ = NewLRUCache(kTheoreticalMaxCacheBytes,
                                 0,
                                 false,
                                 false,
                                 0.0,
                                 nullptr,
                                 kDefaultToAdaptiveMutex,
                                 kDontChargeCacheMetadata);
    }
}

void LruCacheIndexer::RebuildCaches() {
    caches_.clear();
    caches_.reserve(capacity_bytes_.size());
    for (int64_t cap_bytes : capacity_bytes_) {
        auto cache = NewLRUCache(static_cast<size_t>(cap_bytes),
                                 0,
                                 false,
                                 false,
                                 0.0,
                                 nullptr,
                                 kDefaultToAdaptiveMutex,
                                 kDontChargeCacheMetadata);
        caches_.push_back(std::move(cache));
    }
    max_cache_.reset();
    unique_count_ = 0;
    eviction_count_ = 0;
}

bool LruCacheIndexer::LookupAndInsert(Cache *cache, std::string_view key_sv, bool &is_new_key) {
    auto *handle = cache->Lookup(key_sv);
    if (handle) {
        is_new_key = false;
        cache->Release(handle);
        return true;
    }
    cache->Insert(key_sv, nullptr, &kHelper, static_cast<size_t>(size_full_only_));
    return false;
}

bool LruCacheIndexer::LookupAndInsert(Cache *cache,
                                      std::string_view key_sv,
                                      bool is_linear,
                                      int64_t desired_charge,
                                      bool &is_new_key,
                                      bool &is_checkpoint) {
    is_checkpoint = false;
    auto *handle = cache->Lookup(key_sv);
    if (handle) {
        is_new_key = false;
        size_t stored_charge = cache->GetCharge(handle);
        cache->Release(handle);
        is_checkpoint = (static_cast<int64_t>(stored_charge) == size_full_linear_);
        if (is_linear && !is_checkpoint) {
            cache->Erase(key_sv);
            cache->Insert(key_sv, nullptr, &kHelper, static_cast<size_t>(desired_charge));
        }
        return true;
    }
    cache->Insert(key_sv, nullptr, &kHelper, static_cast<size_t>(desired_charge));
    return false;
}

void LruCacheIndexer::ProcessKeys(const std::vector<int64_t> &keys,
                                  std::vector<int64_t> &hit_count,
                                  int64_t &max_hit_count,
                                  std::vector<bool> *key_hits) {
    if (linear_step_ == 0) {
        ProcessKeysFullAttention(keys, hit_count, max_hit_count, key_hits);
    } else {
        ProcessKeysLinearAttention(keys, hit_count, max_hit_count, key_hits);
    }
}

void LruCacheIndexer::ProcessKeysFullAttention(const std::vector<int64_t> &keys,
                                               std::vector<int64_t> &hit_count,
                                               int64_t &max_hit_count,
                                               std::vector<bool> *key_hits) {
    const size_t num_caps = caches_.size();
    const int64_t total_keys = static_cast<int64_t>(keys.size());
    hit_count.assign(num_caps, total_keys);
    max_hit_count = max_cache_ ? total_keys : -1;
    if (key_hits) {
        key_hits->assign(keys.size(), false);
    }

    for (int64_t i = 0; i < total_keys; i++) {
        int64_t key = keys[i];
        std::string_view key_sv(reinterpret_cast<const char *>(&key), sizeof(key));

        bool is_new_key = true;
        bool largest_cache_hit = false;
        for (size_t j = 0; j < num_caps; j++) {
            const bool hit = LookupAndInsert(caches_[j].get(), key_sv, is_new_key);
            if (j + 1 == num_caps) {
                largest_cache_hit = hit;
            }
            if (!hit) {
                if (i < hit_count[j]) {
                    hit_count[j] = i;
                }
            }
        }
        bool max_cache_hit = false;
        if (max_cache_) {
            max_cache_hit = LookupAndInsert(max_cache_.get(), key_sv, is_new_key);
            if (!max_cache_hit) {
                if (i < max_hit_count) {
                    max_hit_count = i;
                }
            }
        }
        if (key_hits) {
            (*key_hits)[i] = largest_cache_hit;
        }

        if (is_new_key) {
            unique_count_++;
        }
    }
}

void LruCacheIndexer::ProcessKeysLinearAttention(const std::vector<int64_t> &keys,
                                                 std::vector<int64_t> &hit_count,
                                                 int64_t &max_hit_count,
                                                 std::vector<bool> *key_hits) {
    const size_t num_caps = caches_.size();
    const int64_t total_keys = static_cast<int64_t>(keys.size());
    std::vector<int64_t> first_miss(num_caps, total_keys);
    max_hit_count = max_cache_ ? 0 : -1;
    int64_t max_first_miss = total_keys;
    if (key_hits) {
        key_hits->assign(keys.size(), false);
    }

    std::vector<int64_t> last_checkpoint(num_caps, -1);
    int64_t max_last_checkpoint = -1;

    for (int64_t i = 0; i < total_keys; i++) {
        int64_t key = keys[i];
        std::string_view key_sv(reinterpret_cast<const char *>(&key), sizeof(key));
        const bool is_linear = (((i + 1) % linear_step_) == 0) || (i == total_keys - 1);
        const int64_t desired_charge = is_linear ? size_full_linear_ : size_full_only_;

        bool is_new_key = true;
        bool largest_cache_hit = false;
        for (size_t j = 0; j < num_caps; j++) {
            bool is_checkpoint = false;
            const bool hit =
                LookupAndInsert(caches_[j].get(), key_sv, is_linear, desired_charge, is_new_key, is_checkpoint);
            if (j + 1 == num_caps) {
                largest_cache_hit = hit;
            }
            if (hit) {
                if (is_checkpoint && i < first_miss[j]) {
                    last_checkpoint[j] = i;
                }
            } else {
                if (i < first_miss[j]) {
                    first_miss[j] = i;
                }
            }
        }
        bool max_cache_hit = false;
        if (max_cache_) {
            bool is_checkpoint = false;
            max_cache_hit =
                LookupAndInsert(max_cache_.get(), key_sv, is_linear, desired_charge, is_new_key, is_checkpoint);
            if (max_cache_hit) {
                if (is_checkpoint && i < max_first_miss) {
                    max_last_checkpoint = i;
                }
            } else {
                if (i < max_first_miss) {
                    max_first_miss = i;
                }
            }
        }
        if (key_hits) {
            (*key_hits)[i] = largest_cache_hit;
        }

        if (is_new_key) {
            unique_count_++;
        }
    }

    hit_count.assign(num_caps, 0);
    for (size_t j = 0; j < num_caps; j++) {
        if (last_checkpoint[j] >= 0 && last_checkpoint[j] < first_miss[j]) {
            hit_count[j] = last_checkpoint[j] + 1;
        }
    }
    if (max_cache_) {
        if (max_last_checkpoint >= 0 && max_last_checkpoint < max_first_miss) {
            max_hit_count = max_last_checkpoint + 1;
        }
    }
}

void LruCacheIndexer::PostQueryMaintenance() {
    Cache *largest = max_cache_ ? max_cache_.get() : (caches_.empty() ? nullptr : caches_.back().get());
    if (largest) {
        int64_t occupancy = static_cast<int64_t>(largest->GetOccupancyCount());
        if (unique_count_ > occupancy) {
            eviction_count_ += (unique_count_ - occupancy);
            unique_count_ = occupancy;
        }
    }
}

bool LruCacheIndexer::RemoveKey(int64_t key) {
    std::string_view key_sv(reinterpret_cast<const char *>(&key), sizeof(key));
    bool found = false;
    for (auto &cache : caches_) {
        if (cache->Erase(key_sv))
            found = true;
    }
    if (max_cache_ && max_cache_->Erase(key_sv))
        found = true;
    if (found) {
        unique_count_--;
        eviction_count_++;
    }
    return found;
}

int64_t LruCacheIndexer::kv_cache_usage_bytes() const {
    if (max_cache_) {
        return static_cast<int64_t>(max_cache_->GetUsage());
    }
    if (caches_.empty())
        return 0;
    return static_cast<int64_t>(caches_.back()->GetUsage());
}

int64_t LruCacheIndexer::memory_usage_bytes() const {
    int64_t total = 0;
    for (const auto &cache : caches_) {
        total += static_cast<int64_t>(cache->GetOccupancyCount()) * kEstimatedCacheEntryOverheadBytes;
    }
    if (max_cache_) {
        total += static_cast<int64_t>(max_cache_->GetOccupancyCount()) * kEstimatedCacheEntryOverheadBytes;
    }
    return total;
}

std::vector<int64_t> LruCacheIndexer::capacity_unique_counts() const {
    std::vector<int64_t> result;
    result.reserve(caches_.size());
    for (const auto &cache : caches_) {
        result.push_back(static_cast<int64_t>(cache->GetOccupancyCount()));
    }
    return result;
}

} // namespace kv_cache_manager
