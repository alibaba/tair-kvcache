#include "kv_cache_manager/optimizer/liteHit/lite_hit.h"

#include <algorithm>
#include <utility>

namespace kv_cache_manager {

namespace {

constexpr std::size_t kCompactionSlackPositions = 4096;

} // namespace

RequestFact LiteHit::ProcessRequest(const std::vector<int64_t> &block_keys) {
    RequestFact fact = BuildHitCurve(block_keys);
    CommitRequest(block_keys);
    MaybeCompactPositions();
    return fact;
}

RequestFact LiteHit::BuildHitCurve(const std::vector<int64_t> &block_keys) const {
    RequestFact fact;
    if (block_keys.empty()) {
        return fact;
    }

    // All ranks are read from one immutable request-start snapshot. Repeated
    // keys reuse the same snapshot entry. A cold key stops every capacity's
    // prefix, so its later occurrence cannot revive the prefix.
    std::unordered_map<int64_t, SnapshotEntry> snapshot_entries;
    snapshot_entries.reserve(block_keys.size());
    for (int64_t block_key : block_keys) {
        auto [entry_it, inserted] = snapshot_entries.emplace(block_key, SnapshotEntry{});
        if (!inserted) {
            continue;
        }
        const auto previous = last_positions_.find(block_key);
        if (previous != last_positions_.end()) {
            entry_it->second.is_resident = true;
            entry_it->second.required_blocks = ReuseDistance(previous->second) + 1;
        }
    }

    uint64_t prefix_required_blocks = 0;
    uint64_t last_encoded_threshold = 0;
    for (int64_t block_key : block_keys) {
        const SnapshotEntry &entry = snapshot_entries.at(block_key);
        if (!entry.is_resident) {
            break;
        }
        prefix_required_blocks = std::max(prefix_required_blocks, entry.required_blocks);
        // Under the prefix-hash contract thresholds strictly increase and the
        // max() below is a no-op. For non-contract input (duplicate keys) it
        // keeps the encoded thresholds strictly increasing so the arithmetic
        // runs stay representable; the projection is then pessimistic by at
        // most the number of duplicates and never optimistic.
        const uint64_t threshold = std::max(prefix_required_blocks, last_encoded_threshold + 1);
        if (!fact.hit_curve.empty() &&
            fact.hit_curve.back().start_required_blocks + fact.hit_curve.back().run_length == threshold) {
            fact.hit_curve.back().run_length++;
        } else {
            fact.hit_curve.push_back(HitCurveSegment{threshold, 1});
        }
        last_encoded_threshold = threshold;
    }
    return fact;
}

void LiteHit::CommitRequest(const std::vector<int64_t> &block_keys) {
    if (block_keys.empty()) {
        return;
    }

    // State commits tail-to-head: sequentially touching the request in
    // reverse order produces a final LRU order determined only by each
    // distinct key's last touch, which is its first occurrence visited
    // back-to-front. The chain head therefore ends up most recent and the
    // eviction victim is always a chain leaf. Remove old markers once, then
    // append those touches in reverse request order.
    std::unordered_map<int64_t, std::size_t> first_occurrence;
    first_occurrence.reserve(block_keys.size());
    for (std::size_t i = block_keys.size(); i > 0; --i) {
        first_occurrence[block_keys[i - 1]] = i - 1;
    }

    for (const auto &[block_key, _] : first_occurrence) {
        const auto previous = last_positions_.find(block_key);
        if (previous != last_positions_.end()) {
            fenwick_.Add(previous->second, -1);
            last_positions_.erase(previous);
        }
    }

    for (std::size_t i = block_keys.size(); i > 0; --i) {
        const int64_t block_key = block_keys[i - 1];
        if (first_occurrence.at(block_key) != i - 1) {
            continue;
        }
        fenwick_.AppendZero();
        const std::size_t current_position = fenwick_.size();
        fenwick_.Add(current_position, 1);
        last_positions_[block_key] = current_position;
    }
}

void LiteHit::MaybeCompactPositions() {
    const std::size_t active_positions = last_positions_.size();
    if (fenwick_.size() <= kCompactionSlackPositions) {
        return;
    }
    const std::size_t positions_over_slack = fenwick_.size() - kCompactionSlackPositions;
    if (active_positions >= (positions_over_slack + 1) / 2) {
        return;
    }

    std::vector<std::pair<std::size_t, int64_t>> ordered_positions;
    ordered_positions.reserve(active_positions);
    for (const auto &[block_key, position] : last_positions_) {
        ordered_positions.emplace_back(position, block_key);
    }
    std::sort(ordered_positions.begin(), ordered_positions.end());

    DynamicFenwickTree compacted_fenwick;
    for (const auto &[_, block_key] : ordered_positions) {
        compacted_fenwick.AppendZero();
        const std::size_t compacted_position = compacted_fenwick.size();
        compacted_fenwick.Add(compacted_position, 1);
        last_positions_[block_key] = compacted_position;
    }
    fenwick_ = std::move(compacted_fenwick);
}

uint64_t LiteHit::ReuseDistance(std::size_t previous_position) const {
    return fenwick_.PrefixSum(fenwick_.size()) - fenwick_.PrefixSum(previous_position);
}

void LiteHit::Reset() {
    fenwick_.Clear();
    last_positions_.clear();
}

uint64_t LiteHit::memory_usage_bytes() const {
    uint64_t bytes = fenwick_.memory_usage_bytes();
    bytes += static_cast<uint64_t>(last_positions_.bucket_count()) * sizeof(void *);
    constexpr uint64_t kEstimatedHashNodeOverhead = sizeof(void *) * 2;
    bytes += static_cast<uint64_t>(last_positions_.size()) *
             (sizeof(std::pair<const int64_t, std::size_t>) + kEstimatedHashNodeOverhead);
    return bytes;
}

} // namespace kv_cache_manager
