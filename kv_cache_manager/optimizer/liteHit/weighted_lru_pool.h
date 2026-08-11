#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "kv_cache_manager/optimizer/liteHit/dynamic_fenwick_tree.h"

namespace kv_cache_manager {

// Object identity inside a LiteHit cache pool. Full blocks and Mamba
// checkpoints may share the same prefix-chained key; the type keeps them
// from colliding.
enum class CacheObjectType {
    kFull = 0,
    kMamba = 1
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
    // Both charges are positive: one pool hosts Full blocks and Mamba
    // checkpoints on a single recency order.
    WeightedLruPool(uint64_t full_charge_bytes, uint64_t mamba_charge_bytes)
        : charges_{full_charge_bytes, mamba_charge_bytes} {}

    uint64_t charge_of(CacheObjectType type) const { return charges_[static_cast<int>(type)]; }

    bool IsResident(const CacheObjectKey &key) const { return positions(key.type).count(key.prefix_block_key) > 0; }

    // Minimum total pool bytes that keep `key` resident right now: its own
    // charge plus the bytes of every strictly newer resident object.
    // Returns false when the object is not resident.
    bool RequiredBytes(const CacheObjectKey &key, uint64_t &required_bytes) const;

    // Moves the object to the most recent position (inserting it if absent)
    // with its fixed type charge.
    void Touch(const CacheObjectKey &key);

    void Reset();

    uint64_t resident_full_count() const { return static_cast<uint64_t>(full_positions_.size()); }
    uint64_t resident_mamba_count() const { return static_cast<uint64_t>(mamba_positions_.size()); }
    // Total bytes of all resident objects (infinite-capacity working set).
    uint64_t resident_bytes() const { return fenwick_.PrefixSum(fenwick_.size()); }
    // Number of resident Full objects whose RequiredBytes fits into
    // byte_budget, i.e. the Full population of a cache bounded to that many
    // bytes (Mattson inclusion boundary).
    uint64_t FullObjectsWithinBytes(uint64_t byte_budget) const;
    uint64_t memory_usage_bytes() const;

private:
    using PositionMap = std::unordered_map<int64_t, std::size_t>;

    PositionMap &positions(CacheObjectType type) {
        return type == CacheObjectType::kFull ? full_positions_ : mamba_positions_;
    }
    const PositionMap &positions(CacheObjectType type) const {
        return type == CacheObjectType::kFull ? full_positions_ : mamba_positions_;
    }
    void MaybeCompactPositions();

    uint64_t charges_[2];
    DynamicFenwickTree fenwick_;
    // Parallel order-statistics view counting 1 per Full marker at the same
    // positions as the byte tree; backs FullObjectsWithinBytes.
    DynamicFenwickTree full_count_fenwick_;
    PositionMap full_positions_;
    PositionMap mamba_positions_;
};

} // namespace kv_cache_manager
