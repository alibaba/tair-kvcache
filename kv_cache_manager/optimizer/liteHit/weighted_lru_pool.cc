#include "kv_cache_manager/optimizer/liteHit/weighted_lru_pool.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace kv_cache_manager {

namespace {

constexpr std::size_t kCompactionSlackPositions = 4096;

} // namespace

std::size_t WeightedLruPool::PositionRemap::RemapBoundary(std::size_t old_position) const {
    if (!compacted) {
        return old_position;
    }
    const auto it = std::lower_bound(surviving_old_positions.begin(), surviving_old_positions.end(), old_position);
    return static_cast<std::size_t>(it - surviving_old_positions.begin()) + 1;
}

bool WeightedLruPool::IsResident(const CacheObjectKey &key, std::size_t alive_from_position) const {
    const PositionMap &map = positions(key.type);
    const auto it = map.find(key.prefix_block_key);
    return it != map.end() && it->second >= NormalizeAliveFrom(alive_from_position);
}

bool WeightedLruPool::RequiredBytes(const CacheObjectKey &key,
                                    uint64_t &required_bytes,
                                    std::size_t alive_from_position) const {
    const PositionMap &map = positions(key.type);
    const auto it = map.find(key.prefix_block_key);
    if (it == map.end() || it->second < NormalizeAliveFrom(alive_from_position)) {
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
}

uint64_t WeightedLruPool::resident_full_count(std::size_t alive_from_position) const {
    const std::size_t alive_from = NormalizeAliveFrom(alive_from_position);
    if (alive_from > fenwick_.size()) {
        return 0;
    }
    return full_count_fenwick_.PrefixSum(fenwick_.size()) - full_count_fenwick_.PrefixSum(alive_from - 1);
}

uint64_t WeightedLruPool::resident_linear_count(std::size_t alive_from_position) const {
    const std::size_t alive_from = NormalizeAliveFrom(alive_from_position);
    if (alive_from <= 1) {
        return static_cast<uint64_t>(linear_positions_.size());
    }
    uint64_t count = 0;
    for (const auto &[_, position] : linear_positions_) {
        if (position >= alive_from) {
            ++count;
        }
    }
    return count;
}

uint64_t WeightedLruPool::resident_bytes(std::size_t alive_from_position) const {
    const std::size_t alive_from = NormalizeAliveFrom(alive_from_position);
    if (alive_from > fenwick_.size()) {
        return 0;
    }
    return fenwick_.PrefixSum(fenwick_.size()) - fenwick_.PrefixSum(alive_from - 1);
}

uint64_t WeightedLruPool::FullObjectsWithinBytes(uint64_t byte_budget, std::size_t alive_from_position) const {
    // Object at position p is resident under the budget iff the bytes of all
    // markers at positions >= p fit, i.e. total - PrefixSum(p - 1) <= budget.
    // That suffix-bytes function is nonincreasing in p, so binary search the
    // smallest resident position and count Full markers behind it.
    const std::size_t n = fenwick_.size();
    const std::size_t alive_from = NormalizeAliveFrom(alive_from_position);
    const uint64_t total_bytes = fenwick_.PrefixSum(n);
    if (n == 0 || byte_budget == 0 || alive_from > n) {
        return 0;
    }
    std::size_t lo = alive_from;
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

uint64_t WeightedLruPool::FullObjectsBefore(std::size_t boundary_position) const {
    if (boundary_position <= 1 || fenwick_.size() == 0) {
        return 0;
    }
    return full_count_fenwick_.PrefixSum(std::min(boundary_position - 1, fenwick_.size()));
}

WeightedLruPool::PositionRemap WeightedLruPool::MaybeCompactPositions(std::size_t alive_from_position) {
    PositionRemap remap;
    if (fenwick_.size() <= kCompactionSlackPositions) {
        return remap;
    }

    const std::size_t alive_from = NormalizeAliveFrom(alive_from_position);
    std::size_t active_positions = 0;
    for (const PositionMap *map : {&full_positions_, &linear_positions_}) {
        for (const auto &[_, position] : *map) {
            if (position >= alive_from) {
                ++active_positions;
            }
        }
    }
    const std::size_t positions_over_slack = fenwick_.size() - kCompactionSlackPositions;
    if (active_positions >= (positions_over_slack + 1) / 2) {
        return remap;
    }

    struct Marker {
        std::size_t position;
        CacheObjectType type;
        int64_t prefix_block_key;
    };
    std::vector<Marker> ordered_markers;
    ordered_markers.reserve(active_positions);
    for (const auto &[block_key, position] : full_positions_) {
        if (position >= alive_from) {
            ordered_markers.push_back({position, CacheObjectType::kFull, block_key});
        }
    }
    for (const auto &[block_key, position] : linear_positions_) {
        if (position >= alive_from) {
            ordered_markers.push_back({position, CacheObjectType::kLinear, block_key});
        }
    }
    std::sort(ordered_markers.begin(), ordered_markers.end(), [](const Marker &a, const Marker &b) {
        return a.position < b.position;
    });

    remap.compacted = true;
    remap.surviving_old_positions.reserve(ordered_markers.size());
    full_positions_.clear();
    linear_positions_.clear();

    DynamicFenwickTree compacted_fenwick;
    DynamicFenwickTree compacted_full_count;
    for (const Marker &marker : ordered_markers) {
        remap.surviving_old_positions.push_back(marker.position);
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

    constexpr std::size_t kBucketShrinkRatio = 4;
    for (PositionMap *map : {&full_positions_, &linear_positions_}) {
        if (map->bucket_count() > kBucketShrinkRatio * (map->size() + 1)) {
            map->rehash(2 * map->size());
        }
    }
    return remap;
}

void WeightedLruPool::Reset() {
    fenwick_.Clear();
    full_count_fenwick_.Clear();
    full_positions_.clear();
    linear_positions_.clear();
}

uint64_t WeightedLruPool::memory_usage_bytes() const {
    uint64_t bytes = fenwick_.memory_usage_bytes() + full_count_fenwick_.memory_usage_bytes();
    constexpr uint64_t kEstimatedHashNodeOverhead = sizeof(void *) * 2;
    for (const PositionMap *map : {&full_positions_, &linear_positions_}) {
        bytes += static_cast<uint64_t>(map->bucket_count()) * sizeof(void *);
        bytes += static_cast<uint64_t>(map->size()) *
                 (sizeof(std::pair<const int64_t, std::size_t>) + kEstimatedHashNodeOverhead);
    }
    return bytes;
}

} // namespace kv_cache_manager
