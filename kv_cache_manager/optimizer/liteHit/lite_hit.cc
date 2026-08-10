#include "kv_cache_manager/optimizer/liteHit/lite_hit.h"

#include <algorithm>
#include <utility>

namespace kv_cache_manager {

namespace {

constexpr std::size_t kCompactionSlackPositions = 4096;

} // namespace

RequestFact LiteHit::ProcessRequest(const std::vector<int64_t> &block_keys, int64_t now_ns) {
    if (ttl_ns_ > 0) {
        AdvanceTtlWatermark(now_ns);
    }
    RequestFact fact = BuildHitCurve(block_keys);
    CommitRequest(block_keys, now_ns);
    MaybeCompactPositions();
    return fact;
}

void LiteHit::AdvanceTtlWatermark(int64_t now_ns) {
    // Strict boundary: an epoch whose deadline has been reached is dead,
    // matching the online TtlCacheIndexerWrapper harvest. Every position of
    // a dead epoch is below the next epoch's start (or below everything when
    // it is the last one).
    const std::size_t old_watermark = dead_below_position_;
    while (!position_epochs_.empty() &&
           position_epochs_.front().timestamp_ns <= now_ns - static_cast<int64_t>(ttl_ns_)) {
        position_epochs_.pop_front();
        dead_below_position_ = position_epochs_.empty() ? fenwick_.size() + 1 : position_epochs_.front().start_position;
    }
    // Markers the watermark sweeps over are blocks that reached the deadline
    // without a refreshing access (re-touched blocks moved their marker
    // above); this equals the online wrapper's harvested-eviction count.
    if (dead_below_position_ > old_watermark) {
        const uint64_t already_dead = old_watermark > 1 ? fenwick_.PrefixSum(old_watermark - 1) : 0;
        ttl_expired_blocks_ += fenwick_.PrefixSum(dead_below_position_ - 1) - already_dead;
    }
}

RequestFact LiteHit::BuildHitCurve(const std::vector<int64_t> &block_keys) const {
    RequestFact fact;
    if (block_keys.empty()) {
        return fact;
    }

    // All ranks are read from one immutable request-start snapshot. Repeated
    // keys reuse the same snapshot entry. A cold or expired key stops every
    // capacity's prefix, so its later occurrence cannot revive the prefix.
    std::unordered_map<int64_t, SnapshotEntry> snapshot_entries;
    snapshot_entries.reserve(block_keys.size());
    for (int64_t block_key : block_keys) {
        auto [entry_it, inserted] = snapshot_entries.emplace(block_key, SnapshotEntry{});
        if (!inserted) {
            continue;
        }
        const auto previous = last_positions_.find(block_key);
        if (previous != last_positions_.end() && previous->second >= dead_below_position_) {
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

void LiteHit::CommitRequest(const std::vector<int64_t> &block_keys, int64_t now_ns) {
    if (block_keys.empty()) {
        return;
    }

    if (ttl_ns_ > 0) {
        // Positions appended below belong to this commit's epoch. Merge with
        // the previous epoch when the timestamp did not advance; clamp a
        // defensively out-of-order timestamp so the deque stays monotone
        // (equivalent to the age-clamp of a per-key timestamp table).
        const int64_t epoch_ns =
            position_epochs_.empty() ? now_ns : std::max(now_ns, position_epochs_.back().timestamp_ns);
        if (position_epochs_.empty() || position_epochs_.back().timestamp_ns != epoch_ns) {
            position_epochs_.push_back(PositionEpoch{fenwick_.size() + 1, epoch_ns});
        }
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
    const std::size_t active_positions = static_cast<std::size_t>(alive_marker_count());
    if (fenwick_.size() <= kCompactionSlackPositions) {
        return;
    }
    const std::size_t positions_over_slack = fenwick_.size() - kCompactionSlackPositions;
    if (active_positions >= (positions_over_slack + 1) / 2) {
        return;
    }

    // Markers below the TTL watermark are a miss for every capacity forever,
    // so compaction drops them together with their table entries; the state
    // afterwards is bounded by the alive working set again.
    std::vector<std::pair<std::size_t, int64_t>> ordered_positions;
    ordered_positions.reserve(active_positions);
    for (auto it = last_positions_.begin(); it != last_positions_.end();) {
        if (it->second < dead_below_position_) {
            it = last_positions_.erase(it);
        } else {
            ordered_positions.emplace_back(it->second, it->first);
            ++it;
        }
    }
    std::sort(ordered_positions.begin(), ordered_positions.end());

    // The bucket array never shrinks on erase and would otherwise stay at
    // the historical peak. Shrink only when it is far above the surviving
    // set (with headroom) so oscillating working sets do not rehash back and
    // forth.
    if (last_positions_.bucket_count() > 4 * (last_positions_.size() + 1)) {
        last_positions_.rehash(2 * last_positions_.size());
    }

    // Old position boundary S maps to (surviving markers below S) + 1; the
    // mapping is monotone, so epoch starts and the watermark keep their
    // order and semantics.
    const auto remap_boundary = [&ordered_positions](std::size_t old_position) -> std::size_t {
        const auto it = std::lower_bound(
            ordered_positions.begin(),
            ordered_positions.end(),
            old_position,
            [](const std::pair<std::size_t, int64_t> &entry, std::size_t value) { return entry.first < value; });
        return static_cast<std::size_t>(it - ordered_positions.begin()) + 1;
    };
    for (PositionEpoch &epoch : position_epochs_) {
        epoch.start_position = remap_boundary(epoch.start_position);
    }
    // Equal remapped starts prove there is no alive marker between the two
    // epochs, so the earlier one governs an empty range and can be dropped
    // without changing any future liveness answer. Keeping the last epoch per
    // start bounds the deque by the alive working set (hot keys otherwise
    // accumulate one empty epoch per request until the TTL elapses).
    std::size_t deduped_size = 0;
    for (const PositionEpoch &epoch : position_epochs_) {
        if (deduped_size > 0 && position_epochs_[deduped_size - 1].start_position == epoch.start_position) {
            position_epochs_[deduped_size - 1].timestamp_ns = epoch.timestamp_ns;
        } else {
            position_epochs_[deduped_size++] = epoch;
        }
    }
    position_epochs_.resize(deduped_size);
    if (dead_below_position_ > 0) {
        dead_below_position_ = remap_boundary(dead_below_position_);
    }

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

uint64_t LiteHit::alive_marker_count() const {
    const uint64_t total = fenwick_.PrefixSum(fenwick_.size());
    if (dead_below_position_ <= 1) {
        return total;
    }
    return total - fenwick_.PrefixSum(dead_below_position_ - 1);
}

void LiteHit::Reset() {
    fenwick_.Clear();
    last_positions_.clear();
    position_epochs_.clear();
    dead_below_position_ = 0;
    ttl_expired_blocks_ = 0;
}

uint64_t LiteHit::memory_usage_bytes() const {
    uint64_t bytes = fenwick_.memory_usage_bytes();
    bytes += static_cast<uint64_t>(last_positions_.bucket_count()) * sizeof(void *);
    constexpr uint64_t kEstimatedHashNodeOverhead = sizeof(void *) * 2;
    bytes += static_cast<uint64_t>(last_positions_.size()) *
             (sizeof(std::pair<const int64_t, std::size_t>) + kEstimatedHashNodeOverhead);
    bytes += static_cast<uint64_t>(position_epochs_.size()) * sizeof(PositionEpoch);
    return bytes;
}

} // namespace kv_cache_manager
