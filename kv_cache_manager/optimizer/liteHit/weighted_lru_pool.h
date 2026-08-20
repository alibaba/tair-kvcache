#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/optimizer/liteHit/dynamic_fenwick_tree.h"

namespace kv_cache_manager {

// Object identity inside a LiteHit cache pool. Full blocks and Linear state
// objects may share the same prefix-chained key; the type keeps them
// from colliding.
enum class CacheObjectType {
    kFull = 0,
    kLinear = 1
};

struct CacheObjectKey {
    CacheObjectType type = CacheObjectType::kFull;
    int64_t prefix_block_key = 0;
};

// Weighted generalization of the LiteHit LRU state: one global recency order
// over typed objects with a fixed positive byte charge per type. Range sums
// return resident BYTES instead of block counts, so residency for a byte
// capacity is decided by RequiredBytes(key) <= capacity_bytes (Mattson stack
// inclusion, byte-weighted). It is a LiteHit-internal component.
class WeightedLruPool {
public:
    // Linear state charge may be zero when the pool is configured for
    // Full-only replay; Linear objects are never touched in that mode.
    WeightedLruPool(uint64_t full_charge_bytes, uint64_t linear_charge_bytes)
        : charges_{full_charge_bytes, linear_charge_bytes} {}

    // A compaction maps every surviving old marker position to a dense new
    // position. TTL watermarks/epoch boundaries use the same lower-bound
    // mapping to keep their meaning after positions move.
    struct PositionRemap {
        bool compacted = false;
        std::vector<std::size_t> surviving_old_positions;

        std::size_t RemapBoundary(std::size_t old_position) const;
    };

    uint64_t charge_of(CacheObjectType type) const { return charges_[static_cast<int>(type)]; }

    bool IsResident(const CacheObjectKey &key, std::size_t alive_from_position = 0) const;

    // Minimum total pool bytes that keep `key` resident right now: its own
    // charge plus the bytes of every strictly newer resident object.
    // Returns false when the object is not resident.
    bool RequiredBytes(const CacheObjectKey &key, uint64_t &required_bytes, std::size_t alive_from_position = 0) const;

    // Moves the object to the most recent position (inserting it if absent)
    // with its fixed type charge.
    void Touch(const CacheObjectKey &key);

    void Reset();

    uint64_t resident_full_count(std::size_t alive_from_position = 0) const;
    uint64_t resident_linear_count(std::size_t alive_from_position = 0) const;
    // Total bytes of resident objects at/above an optional liveness boundary.
    uint64_t resident_bytes(std::size_t alive_from_position = 0) const;
    // Number of resident Full objects whose RequiredBytes fits into
    // byte_budget, i.e. the Full population of a cache bounded to that many
    // bytes (Mattson inclusion boundary).
    uint64_t FullObjectsWithinBytes(uint64_t byte_budget, std::size_t alive_from_position = 0) const;

    // Number of live Full markers strictly below boundary_position. Used by
    // the TTL policy to count markers crossed by its watermark.
    uint64_t FullObjectsBefore(std::size_t boundary_position) const;

    std::size_t position_count() const { return fenwick_.size(); }

    // Compacts dead marker positions when slack is large. Objects below
    // alive_from_position are dropped together with their key-map entries.
    // The caller must remap any external position boundaries with the result.
    PositionRemap MaybeCompactPositions(std::size_t alive_from_position = 0);

    uint64_t memory_usage_bytes() const;

private:
    using PositionMap = std::unordered_map<int64_t, std::size_t>;

    PositionMap &positions(CacheObjectType type) {
        return type == CacheObjectType::kFull ? full_positions_ : linear_positions_;
    }
    const PositionMap &positions(CacheObjectType type) const {
        return type == CacheObjectType::kFull ? full_positions_ : linear_positions_;
    }

    static std::size_t NormalizeAliveFrom(std::size_t alive_from_position) {
        return alive_from_position <= 1 ? 1 : alive_from_position;
    }

    uint64_t charges_[2];
    DynamicFenwickTree fenwick_;
    // Parallel order-statistics view counting 1 per Full marker at the same
    // positions as the byte tree; backs FullObjectsWithinBytes.
    DynamicFenwickTree full_count_fenwick_;
    PositionMap full_positions_;
    PositionMap linear_positions_;
};

} // namespace kv_cache_manager
