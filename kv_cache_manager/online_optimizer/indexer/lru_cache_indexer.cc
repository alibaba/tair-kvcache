#include "kv_cache_manager/online_optimizer/indexer/lru_cache_indexer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "kv_cache_manager/common/cache/cache.h"

namespace kv_cache_manager {

const Cache::CacheItemHelper LruCacheIndexer::kHelper(CacheEntryRole::kMisc, nullptr);

LruCacheIndexer::LruCacheIndexer(int64_t max_key_count)
    : max_key_count_(max_key_count) {}

void LruCacheIndexer::Init(const std::vector<double> &capacity_gb,
                            int64_t size_full_only,
                            int64_t size_full_linear,
                            int32_t linear_step) {
    size_full_only_ = size_full_only;
    size_full_linear_ = size_full_linear;
    linear_step_ = std::max(linear_step, int32_t(1));

    capacity_bytes_.resize(capacity_gb.size());
    for (size_t i = 0; i < capacity_gb.size(); i++) {
        capacity_bytes_[i] = static_cast<int64_t>(capacity_gb[i] * 1024.0 * 1024.0 * 1024.0);
    }

    int64_t avg_bytes_per_block;
    if (linear_step_ <= 1) {
        avg_bytes_per_block = size_full_linear_;
    } else {
        avg_bytes_per_block =
            ((linear_step_ - 1) * size_full_only_ + size_full_linear_) / linear_step_;
    }

    RebuildCaches();

    int64_t max_cache_bytes = 0;
    if (max_key_count_ <= 0) {
        max_cache_bytes = static_cast<int64_t>(std::numeric_limits<size_t>::max() / 2);
    } else {
        int64_t max_cap_bytes = capacity_bytes_.empty() ? 0 : capacity_bytes_.back();
        int64_t mkc_bytes = max_key_count_ * avg_bytes_per_block;
        if (mkc_bytes > max_cap_bytes) {
            max_cache_bytes = mkc_bytes;
        }
    }
    if (max_cache_bytes > 0) {
        max_cache_ = NewLRUCache(
            static_cast<size_t>(max_cache_bytes), 0, false, false, 0.0);
    }
}

void LruCacheIndexer::RebuildCaches() {
    caches_.clear();
    caches_.reserve(capacity_bytes_.size());
    for (int64_t cap_bytes : capacity_bytes_) {
        auto cache = NewLRUCache(
            static_cast<size_t>(cap_bytes), 0, false, false, 0.0);
        caches_.push_back(std::move(cache));
    }
    max_cache_.reset();
    unique_count_ = 0;
    eviction_count_ = 0;
}

void LruCacheIndexer::ProcessKeys(const std::vector<int64_t> &keys,
                                  std::vector<int64_t> &hit_count,
                                  int64_t &max_hit_count) {
    const size_t num_caps = caches_.size();
    const int64_t total_keys = static_cast<int64_t>(keys.size());
    hit_count.assign(num_caps, total_keys);
    max_hit_count = (max_key_count_ <= 0 && max_cache_) ? total_keys : -1;

    for (int64_t i = 0; i < total_keys; i++) {
        int64_t key = keys[i];
        std::string_view key_sv(reinterpret_cast<const char *>(&key), sizeof(key));

        int64_t charge;
        if (linear_step_ <= 1) {
            charge = size_full_linear_;
        } else if (i == total_keys - 1) {
            charge = size_full_linear_;
        } else if (i % linear_step_ == 0) {
            charge = size_full_linear_;
        } else {
            charge = size_full_only_;
        }

        bool is_new_key = true;
        for (size_t j = 0; j < num_caps; j++) {
            auto *handle = caches_[j]->Lookup(key_sv);
            bool hit = (handle != nullptr);
            if (handle) {
                caches_[j]->Release(handle);
                is_new_key = false;
            }
            if (i < hit_count[j] && !hit) {
                hit_count[j] = i;
            }
            if (!hit) {
                caches_[j]->Insert(key_sv, nullptr, &kHelper, static_cast<size_t>(charge));
            }
        }

        if (max_cache_) {
            auto *handle = max_cache_->Lookup(key_sv);
            bool hit = (handle != nullptr);
            if (handle) {
                max_cache_->Release(handle);
                is_new_key = false;
            }
            if (max_hit_count > 0 && i < max_hit_count && !hit) {
                max_hit_count = i;
            }
            if (!hit) {
                max_cache_->Insert(key_sv, nullptr, &kHelper, static_cast<size_t>(charge));
            }
        }

        if (is_new_key) {
            unique_count_++;
        }
    }
}

void LruCacheIndexer::PostQueryMaintenance() {
    Cache *largest = max_cache_ ? max_cache_.get()
                                : (caches_.empty() ? nullptr : caches_.back().get());
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
        if (cache->Erase(key_sv)) found = true;
    }
    if (max_cache_ && max_cache_->Erase(key_sv)) found = true;
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
    if (caches_.empty()) return 0;
    return static_cast<int64_t>(caches_.back()->GetUsage());
}

int64_t LruCacheIndexer::memory_usage_bytes() const {
    int64_t total = 0;
    for (const auto &cache : caches_) {
        total += static_cast<int64_t>(cache->GetOccupancyCount()) * 200;
    }
    if (max_cache_) {
        total += static_cast<int64_t>(max_cache_->GetOccupancyCount()) * 200;
    }
    return total;
}

} // namespace kv_cache_manager
