#include "kv_cache_manager/mrc/reuse_distance_tracker.h"

#include <algorithm>

namespace kv_cache_manager {

ReuseDistanceTracker::ReuseDistanceTracker(int64_t initial_capacity) {
    capacity_ = std::max<int64_t>(initial_capacity, 16);
    tree_.assign(capacity_ + 1, 0);
}

int64_t ReuseDistanceTracker::Access(int64_t key) {
    int64_t distance = -1;
    auto it = last_pos_.find(key);
    if (it != last_pos_.end()) {
        const int64_t pos = it->second;
        // Distinct keys accessed after this key's last access: every live key
        // occupies exactly one position, so count live positions above pos.
        distance = static_cast<int64_t>(last_pos_.size()) - FenwickPrefixSum(pos);
        FenwickAdd(pos, -1);
        last_pos_.erase(it);
    }
    if (next_pos_ > capacity_) {
        Compact();
    }
    last_pos_[key] = next_pos_;
    FenwickAdd(next_pos_, 1);
    ++next_pos_;
    return distance;
}

bool ReuseDistanceTracker::Erase(int64_t key) {
    auto it = last_pos_.find(key);
    if (it == last_pos_.end()) {
        return false;
    }
    FenwickAdd(it->second, -1);
    last_pos_.erase(it);
    return true;
}

int64_t ReuseDistanceTracker::memory_usage_bytes() const {
    // Coarse estimate: Fenwick array plus per-entry hash map overhead
    // (bucket pointer + node with key, value and next pointer).
    constexpr int64_t kEstimatedMapEntryOverheadBytes = 48;
    return static_cast<int64_t>(tree_.capacity() * sizeof(int64_t)) +
           static_cast<int64_t>(last_pos_.size()) * kEstimatedMapEntryOverheadBytes;
}

void ReuseDistanceTracker::FenwickAdd(int64_t pos, int64_t delta) {
    for (; pos <= capacity_; pos += pos & (-pos)) {
        tree_[pos] += delta;
    }
}

int64_t ReuseDistanceTracker::FenwickPrefixSum(int64_t pos) const {
    int64_t sum = 0;
    for (; pos > 0; pos -= pos & (-pos)) {
        sum += tree_[pos];
    }
    return sum;
}

void ReuseDistanceTracker::Compact() {
    std::vector<std::pair<int64_t, int64_t>> entries; // (position, key), ordered by recency
    entries.reserve(last_pos_.size());
    for (const auto &[key, pos] : last_pos_) {
        entries.emplace_back(pos, key);
    }
    std::sort(entries.begin(), entries.end());

    if (static_cast<int64_t>(entries.size()) * 2 > capacity_) {
        capacity_ *= 2;
    }
    tree_.assign(capacity_ + 1, 0);
    next_pos_ = 1;
    for (const auto &[pos, key] : entries) {
        last_pos_[key] = next_pos_;
        FenwickAdd(next_pos_, 1);
        ++next_pos_;
    }
}

} // namespace kv_cache_manager
