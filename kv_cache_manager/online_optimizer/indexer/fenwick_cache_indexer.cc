#include "kv_cache_manager/online_optimizer/indexer/fenwick_cache_indexer.h"

#include <algorithm>
#include <climits>
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
    if (lo > hi) return 0;
    int64_t hi_sum = PrefixSum(hi);
    int64_t lo_sum = lo > 0 ? PrefixSum(lo - 1) : 0;
    return hi_sum - lo_sum;
}

int64_t FenwickTree::FindFirst(int64_t target) const {
    int64_t pos = 0;
    int64_t bit_mask = 1;
    while (bit_mask <= capacity_) bit_mask <<= 1;

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
    : max_key_count_(max_key_count)
    , total_slots_(1024)
    , fenwick_(total_slots_) {
}

int64_t FenwickCacheIndexer::ProcessKey(int64_t key) {
    int64_t sd = INT64_MAX;
    auto it = last_access_.find(key);
    if (it != last_access_.end()) {
        int64_t t_prev = it->second;
        int64_t total_active = fenwick_.PrefixSum(logical_time_ - 1);
        int64_t prefix_at_prev = fenwick_.PrefixSum(t_prev);
        sd = total_active - prefix_at_prev;

        fenwick_.Update(t_prev, -1);
        reverse_map_.erase(t_prev);
    } else {
        unique_count_++;
        if (unique_count_ > peak_unique_count_) {
            peak_unique_count_ = unique_count_;
        }
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

void FenwickCacheIndexer::PostQueryMaintenance() {
    EvictIfExceedsCapacity();
    CompactIfNeeded();
}

void FenwickCacheIndexer::EvictIfExceedsCapacity() {
    if (max_key_count_ <= 0) return;
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
    if (rev_it == reverse_map_.end()) return;

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
