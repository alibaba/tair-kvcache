#include "kv_cache_manager/optimizer/liteHit/weighted_lru_pool.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace kv_cache_manager {

namespace {

constexpr std::size_t kCompactionSlackPositions = 4096;

} // namespace

bool WeightedLruPool::RequiredBytes(const CacheObjectKey &key, uint64_t &required_bytes) const {
    const PositionMap &map = positions(key.type);
    const auto it = map.find(key.prefix_block_key);
    if (it == map.end()) {
        return false;
    }
    const uint64_t newer_bytes = fenwick_.PrefixSum(fenwick_.size()) - fenwick_.PrefixSum(it->second);
    required_bytes = newer_bytes + charge_of(key.type);
    return true;
}

void WeightedLruPool::Touch(const CacheObjectKey &key) {
    PositionMap &map = positions(key.type);
    const uint64_t charge = charge_of(key.type);
    const bool is_full = key.type == CacheObjectType::kFull;
    const auto previous = map.find(key.prefix_block_key);
    if (previous != map.end()) {
        fenwick_.Add(previous->second, -static_cast<int64_t>(charge));
        if (is_full) {
            full_count_fenwick_.Add(previous->second, -1);
        }
        map.erase(previous);
    }
    fenwick_.AppendZero();
    full_count_fenwick_.AppendZero();
    const std::size_t current_position = fenwick_.size();
    fenwick_.Add(current_position, static_cast<int64_t>(charge));
    if (is_full) {
        full_count_fenwick_.Add(current_position, 1);
    }
    map[key.prefix_block_key] = current_position;
    MaybeCompactPositions();
}

uint64_t WeightedLruPool::FullObjectsWithinBytes(uint64_t byte_budget) const {
    // Object at position p is resident under the budget iff the bytes of all
    // markers at positions >= p fit, i.e. total - PrefixSum(p - 1) <= budget.
    // That suffix-bytes function is nonincreasing in p, so binary search the
    // smallest resident position and count Full markers behind it.
    const std::size_t n = fenwick_.size();
    const uint64_t total_bytes = fenwick_.PrefixSum(n);
    if (n == 0 || byte_budget == 0) {
        return 0;
    }
    std::size_t lo = 1;
    std::size_t hi = n + 1; // n + 1 means nothing fits
    while (lo < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (total_bytes - fenwick_.PrefixSum(mid - 1) <= byte_budget) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    if (lo > n) {
        return 0;
    }
    return full_count_fenwick_.PrefixSum(n) - full_count_fenwick_.PrefixSum(lo - 1);
}

void WeightedLruPool::MaybeCompactPositions() {
    const std::size_t active_positions = full_positions_.size() + mamba_positions_.size();
    if (fenwick_.size() <= kCompactionSlackPositions) {
        return;
    }
    const std::size_t positions_over_slack = fenwick_.size() - kCompactionSlackPositions;
    if (active_positions >= (positions_over_slack + 1) / 2) {
        return;
    }

    struct Marker {
        std::size_t position;
        CacheObjectType type;
        int64_t prefix_block_key;
    };
    std::vector<Marker> ordered_markers;
    ordered_markers.reserve(active_positions);
    for (const auto &[block_key, position] : full_positions_) {
        ordered_markers.push_back({position, CacheObjectType::kFull, block_key});
    }
    for (const auto &[block_key, position] : mamba_positions_) {
        ordered_markers.push_back({position, CacheObjectType::kMamba, block_key});
    }
    std::sort(ordered_markers.begin(), ordered_markers.end(), [](const Marker &a, const Marker &b) {
        return a.position < b.position;
    });

    DynamicFenwickTree compacted_fenwick;
    DynamicFenwickTree compacted_full_count;
    for (const Marker &marker : ordered_markers) {
        compacted_fenwick.AppendZero();
        compacted_full_count.AppendZero();
        const std::size_t compacted_position = compacted_fenwick.size();
        compacted_fenwick.Add(compacted_position, static_cast<int64_t>(charge_of(marker.type)));
        if (marker.type == CacheObjectType::kFull) {
            compacted_full_count.Add(compacted_position, 1);
        }
        positions(marker.type)[marker.prefix_block_key] = compacted_position;
    }
    fenwick_ = std::move(compacted_fenwick);
    full_count_fenwick_ = std::move(compacted_full_count);
}

void WeightedLruPool::Reset() {
    fenwick_.Clear();
    full_count_fenwick_.Clear();
    full_positions_.clear();
    mamba_positions_.clear();
}

uint64_t WeightedLruPool::memory_usage_bytes() const {
    uint64_t bytes = fenwick_.memory_usage_bytes() + full_count_fenwick_.memory_usage_bytes();
    constexpr uint64_t kEstimatedHashNodeOverhead = sizeof(void *) * 2;
    for (const PositionMap *map : {&full_positions_, &mamba_positions_}) {
        bytes += static_cast<uint64_t>(map->bucket_count()) * sizeof(void *);
        bytes += static_cast<uint64_t>(map->size()) *
                 (sizeof(std::pair<const int64_t, std::size_t>) + kEstimatedHashNodeOverhead);
    }
    return bytes;
}

} // namespace kv_cache_manager
