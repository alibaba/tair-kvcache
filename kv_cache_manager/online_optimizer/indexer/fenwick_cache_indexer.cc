#include "kv_cache_manager/online_optimizer/indexer/fenwick_cache_indexer.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <utility>

namespace kv_cache_manager {

// ==================== FenwickTree ====================

FenwickTree::FenwickTree(int64_t capacity) : capacity_(capacity), tree_(capacity + 1, 0) {}

void FenwickTree::Update(int64_t idx, int64_t delta) {
    for (int64_t i = idx + 1; i <= capacity_; i += i & (-i)) {
        tree_[i] += delta;
    }
}

int64_t FenwickTree::PrefixSum(int64_t idx) const {
    int64_t sum = 0;
    for (int64_t i = idx + 1; i > 0; i -= i & (-i)) {
        sum += tree_[i];
    }
    return sum;
}

int64_t FenwickTree::RangeSum(int64_t lo, int64_t hi) const {
    if (lo > hi)
        return 0;
    int64_t hi_sum = PrefixSum(hi);
    int64_t lo_sum = lo > 0 ? PrefixSum(lo - 1) : 0;
    return hi_sum - lo_sum;
}

int64_t FenwickTree::FindFirst(int64_t target) const {
    int64_t pos = 0;
    int64_t bit_mask = 1;
    while (bit_mask <= capacity_)
        bit_mask <<= 1;

    for (bit_mask >>= 1; bit_mask > 0; bit_mask >>= 1) {
        int64_t next = pos + bit_mask;
        if (next <= capacity_ && tree_[next] < target) {
            target -= tree_[next];
            pos = next;
        }
    }
    return pos;
}

void FenwickTree::Reset(int64_t new_capacity) {
    capacity_ = new_capacity;
    tree_.assign(capacity_ + 1, 0);
}

// ==================== FenwickCacheIndexer ====================

FenwickCacheIndexer::FenwickCacheIndexer(int64_t max_key_count)
    : max_key_count_(max_key_count), total_slots_(1024), fenwick_(total_slots_) {}

void FenwickCacheIndexer::Init(const std::vector<double> &capacity_gb,
                               int64_t size_full_only,
                               int64_t size_full_linear,
                               int32_t linear_step) {
    int64_t avg_bytes_per_block;
    if (linear_step == 0) {
        avg_bytes_per_block = size_full_only;
    } else {
        avg_bytes_per_block = ((linear_step - 1) * size_full_only + size_full_linear) / linear_step;
    }

    avg_bytes_per_block_ = avg_bytes_per_block;

    capacity_blocks_.resize(capacity_gb.size());
    for (size_t i = 0; i < capacity_gb.size(); i++) {
        int64_t bytes = static_cast<int64_t>(capacity_gb[i] * 1024.0 * 1024.0 * 1024.0);
        capacity_blocks_[i] = (avg_bytes_per_block_ > 0) ? bytes / avg_bytes_per_block_ : 0;
    }
}

int64_t FenwickCacheIndexer::kv_cache_usage_bytes() const { return unique_count() * avg_bytes_per_block_; }

int64_t FenwickCacheIndexer::ComputeStackDistance(int64_t key) {
    int64_t sd = INT64_MAX;
    auto it = last_access_.find(key);
    if (it != last_access_.end()) {
        int64_t t_prev = it->second;
        int64_t total_active = fenwick_.PrefixSum(logical_time_ - 1);
        int64_t prefix_at_prev = fenwick_.PrefixSum(t_prev);
        sd = total_active - prefix_at_prev;

        fenwick_.Update(t_prev, -1);
        reverse_map_.erase(t_prev);
        last_access_.erase(it);
    } else {
        unique_count_++;
    }

    if (logical_time_ >= total_slots_) {
        DoCompact();
    }

    fenwick_.Update(logical_time_, +1);
    last_access_[key] = logical_time_;
    reverse_map_[logical_time_] = key;
    logical_time_++;

    return sd;
}

void FenwickCacheIndexer::ProcessKeys(const std::vector<int64_t> &keys,
                                      std::vector<int64_t> &hit_count,
                                      int64_t &max_hit_count) {
    const size_t num_caps = capacity_blocks_.size();
    const int64_t total_keys = static_cast<int64_t>(keys.size());
    hit_count.assign(num_caps, total_keys);
    max_hit_count = (max_key_count_ <= 0) ? total_keys : -1;

    for (int64_t i = 0; i < total_keys; i++) {
        int64_t sd = ComputeStackDistance(keys[i]);
        for (size_t j = 0; j < num_caps; j++) {
            bool is_hit = (sd != INT64_MAX && sd < capacity_blocks_[j]);
            if (i < hit_count[j] && !is_hit) {
                hit_count[j] = i;
            }
        }
        if (max_hit_count > 0 && i < max_hit_count && sd == INT64_MAX) {
            max_hit_count = i;
        }
    }
}

void FenwickCacheIndexer::PostQueryMaintenance() {
    EvictIfExceedsCapacity();
    CompactIfNeeded();
}

bool FenwickCacheIndexer::RemoveKey(int64_t key) {
    auto it = last_access_.find(key);
    if (it == last_access_.end())
        return false;
    int64_t t = it->second;
    fenwick_.Update(t, -1);
    reverse_map_.erase(t);
    last_access_.erase(it);
    unique_count_--;
    eviction_count_++;
    return true;
}

void FenwickCacheIndexer::EvictIfExceedsCapacity() {
    if (max_key_count_ <= 0)
        return;
    while (unique_count_ > max_key_count_) {
        DoEvictOne();
    }
}

void FenwickCacheIndexer::CompactIfNeeded() {
    if (logical_time_ > unique_count_ * 2 + 1024) {
        DoCompact();
    }
}

void FenwickCacheIndexer::DoEvictOne() {
    int64_t t_min = fenwick_.FindFirst(1);
    auto rev_it = reverse_map_.find(t_min);
    if (rev_it == reverse_map_.end())
        return;

    int64_t evicted_key = rev_it->second;
    fenwick_.Update(t_min, -1);
    reverse_map_.erase(rev_it);
    last_access_.erase(evicted_key);
    unique_count_--;
    eviction_count_++;
}

int64_t FenwickCacheIndexer::memory_usage_bytes() const {
    // FenwickTree: (capacity + 1) * sizeof(int64_t)
    int64_t fenwick_bytes = (fenwick_.capacity() + 1) * static_cast<int64_t>(sizeof(int64_t));
    // unordered_map overhead: ~56 bytes per entry (key + value + node overhead)
    constexpr int64_t kMapEntryBytes = 56;
    int64_t map_bytes = static_cast<int64_t>(last_access_.size() + reverse_map_.size()) * kMapEntryBytes;
    return fenwick_bytes + map_bytes;
}

void FenwickCacheIndexer::DoCompact() {
    std::vector<std::pair<int64_t, int64_t>> entries;
    entries.reserve(last_access_.size());
    for (auto &[key, ts] : last_access_) {
        entries.emplace_back(ts, key);
    }
    std::sort(entries.begin(), entries.end());

    int64_t min_slots = std::max(static_cast<int64_t>(entries.size()) * 4, int64_t(1024));
    if (min_slots >= total_slots_) {
        total_slots_ = std::max(min_slots, total_slots_ * 2);
    } else if (total_slots_ > min_slots * 2) {
        total_slots_ = min_slots;
    }

    fenwick_.Reset(total_slots_);
    last_access_.clear();
    reverse_map_.clear();

    for (int64_t new_ts = 0; new_ts < static_cast<int64_t>(entries.size()); new_ts++) {
        int64_t key = entries[new_ts].second;
        fenwick_.Update(new_ts, +1);
        last_access_[key] = new_ts;
        reverse_map_[new_ts] = key;
    }
    logical_time_ = static_cast<int64_t>(entries.size());
}

} // namespace kv_cache_manager
