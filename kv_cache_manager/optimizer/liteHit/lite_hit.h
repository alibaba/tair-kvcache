#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/optimizer/liteHit/dynamic_fenwick_tree.h"
#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"

namespace kv_cache_manager {

// Exact, capacity-independent LRU replay core shared by online and offline.
class LiteHit {
public:
    LiteHit() = default;

    RequestFact ProcessRequest(const std::vector<int64_t> &block_keys);
    void Reset();

    uint64_t current_unique_blocks() const { return static_cast<uint64_t>(last_positions_.size()); }
    uint64_t memory_usage_bytes() const;

private:
    struct SnapshotEntry {
        bool is_resident = false;
        uint64_t required_blocks = 0;
    };

    RequestFact BuildHitCurve(const std::vector<int64_t> &block_keys) const;
    void CommitRequest(const std::vector<int64_t> &block_keys);
    void MaybeCompactPositions();
    uint64_t ReuseDistance(std::size_t previous_position) const;

    DynamicFenwickTree fenwick_;
    std::unordered_map<int64_t, std::size_t> last_positions_;
};

} // namespace kv_cache_manager
