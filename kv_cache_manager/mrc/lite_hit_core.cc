#include "kv_cache_manager/mrc/lite_hit_core.h"

#include <algorithm>

namespace kv_cache_manager {

namespace {
// Snapshot entry: 0 = cold; otherwise required_blocks (reuse distance + 1).
constexpr uint64_t kCold = 0;
} // namespace

LiteHitCore::LiteHitCore(int64_t initial_capacity, int64_t max_tracked_blocks)
    : max_tracked_blocks_(max_tracked_blocks) {
    capacity_ = std::max<int64_t>(initial_capacity, 16);
    if (max_tracked_blocks_ > 0) {
        capacity_ = std::min(capacity_, std::max<int64_t>(max_tracked_blocks_ * 2, 16));
    }
    tree_.assign(capacity_ + 1, 0);
}

void LiteHitCore::ProcessRequest(const std::vector<int64_t> &block_keys, std::vector<uint64_t> &out_thresholds) {
    if (block_keys.empty()) {
        return;
    }

    // Phase 1: prefix hit curve against the request-start snapshot.
    snapshot_required_.clear();
    for (const int64_t key : block_keys) {
        auto [it, inserted] = snapshot_required_.emplace(key, kCold);
        if (!inserted) {
            continue;
        }
        const auto previous = last_pos_.find(key);
        if (previous != last_pos_.end()) {
            const uint64_t distance =
                static_cast<uint64_t>(static_cast<int64_t>(last_pos_.size()) - FenwickPrefixSum(previous->second));
            it->second = distance + 1; // required_blocks
        }
    }

    uint64_t prefix_required = 0;
    uint64_t last_threshold = 0;
    for (const int64_t key : block_keys) {
        const uint64_t required = snapshot_required_.at(key);
        if (required == kCold) {
            break; // a cold key stops every capacity's prefix
        }
        prefix_required = std::max(prefix_required, required);
        // Keep thresholds strictly increasing (defensive for duplicate keys,
        // pessimistic never optimistic — mirrors the offline core).
        const uint64_t threshold = std::max(prefix_required, last_threshold + 1);
        out_thresholds.push_back(threshold);
        last_threshold = threshold;
    }

    // Phase 2: commit tail-to-head so the chain head ends most recent and
    // the LRU victim is always a chain leaf.
    first_occurrence_.clear();
    for (size_t i = block_keys.size(); i > 0; --i) {
        first_occurrence_[block_keys[i - 1]] = i - 1;
    }
    for (const auto &[key, _] : first_occurrence_) {
        const auto previous = last_pos_.find(key);
        if (previous != last_pos_.end()) {
            FenwickAdd(previous->second, -1);
            pos_to_key_.erase(previous->second);
            last_pos_.erase(previous);
        }
    }
    for (size_t i = block_keys.size(); i > 0; --i) {
        const int64_t key = block_keys[i - 1];
        if (first_occurrence_.at(key) != i - 1) {
            continue;
        }
        EnsureSlot();
        last_pos_[key] = next_pos_;
        pos_to_key_[next_pos_] = key;
        FenwickAdd(next_pos_, 1);
        ++next_pos_;
        TrimToLimit();
    }
}

void LiteHitCore::Reset() {
    next_pos_ = 1;
    std::fill(tree_.begin(), tree_.end(), 0);
    last_pos_.clear();
    pos_to_key_.clear();
    snapshot_required_.clear();
    first_occurrence_.clear();
}

void LiteHitCore::TrimToLimit() {
    while (max_tracked_blocks_ > 0 && static_cast<int64_t>(last_pos_.size()) > max_tracked_blocks_) {
        const auto oldest = pos_to_key_.begin();
        FenwickAdd(oldest->first, -1);
        last_pos_.erase(oldest->second);
        pos_to_key_.erase(oldest);
    }
}

void LiteHitCore::EnsureSlot() {
    if (next_pos_ > capacity_) {
        Compact();
    }
}

void LiteHitCore::Compact() {
    std::vector<std::pair<int64_t, int64_t>> entries; // (position, key)
    entries.reserve(last_pos_.size());
    for (const auto &[key, pos] : last_pos_) {
        entries.emplace_back(pos, key);
    }
    std::sort(entries.begin(), entries.end());

    if (static_cast<int64_t>(entries.size()) * 2 > capacity_) {
        capacity_ *= 2;
    }
    tree_.assign(capacity_ + 1, 0);
    pos_to_key_.clear();
    next_pos_ = 1;
    for (const auto &[pos, key] : entries) {
        last_pos_[key] = next_pos_;
        pos_to_key_[next_pos_] = key;
        FenwickAdd(next_pos_, 1);
        ++next_pos_;
    }
}

void LiteHitCore::FenwickAdd(int64_t pos, int64_t delta) {
    for (; pos <= capacity_; pos += pos & (-pos)) {
        tree_[pos] += delta;
    }
}

int64_t LiteHitCore::FenwickPrefixSum(int64_t pos) const {
    int64_t sum = 0;
    for (; pos > 0; pos -= pos & (-pos)) {
        sum += tree_[pos];
    }
    return sum;
}

int64_t LiteHitCore::memory_usage_bytes() const {
    constexpr int64_t kEstimatedHashEntryOverheadBytes = 48;
    constexpr int64_t kEstimatedTreeEntryOverheadBytes = 48;
    return static_cast<int64_t>(tree_.capacity() * sizeof(int64_t)) +
           static_cast<int64_t>(last_pos_.size()) *
               (kEstimatedHashEntryOverheadBytes + kEstimatedTreeEntryOverheadBytes);
}

} // namespace kv_cache_manager
