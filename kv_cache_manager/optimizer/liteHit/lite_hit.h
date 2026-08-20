#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"
#include "kv_cache_manager/optimizer/liteHit/lite_hit_linear.h"
#include "kv_cache_manager/optimizer/liteHit/weighted_lru_pool.h"

namespace kv_cache_manager {

// One capacity-independent LRU replay core for both Full-only and Mamba
// instances. All cache objects share one weighted recency pool. Full-only is
// the uniform-charge specialization where every covered prefix position is a
// restore point; Mamba adds Linear state objects and only their resident
// positions are restore points.
//
// RequestFact is always evaluated on the byte axis. Full-only callers may
// losslessly encode it afterwards as FullRequestFact block-axis RLE. Linear
// state scheduling owns no duplicate LRU state. TTL is deliberately outside
// this core and is provided by the TtlLiteHit decorator.
class LiteHit {
public:
    // Describes the byte charge and placement of objects in the shared LRU
    // pool. It is not a general LiteHit runtime configuration.
    struct CacheObjectConfig {
        uint64_t full_charge_bytes = 1;
        uint64_t linear_charge_bytes = 0; // 0 = Full-only
        uint64_t linear_step_blocks = 0;
    };

    // Full-only constructor; unit charge preserves the public block-axis
    // FullRequestFact contract.
    LiteHit();
    explicit LiteHit(const CacheObjectConfig &object_config);

    // Default path for every instance kind: explicit byte-axis step curve.
    RequestFact ProcessRequest(const std::vector<int64_t> &block_keys);

    // Full-only specialization: losslessly encodes the default byte-step fact
    // as block-axis RLE. Mixed Full/Linear charge cannot use this encoding.
    FullRequestFact ProcessFullRequest(const std::vector<int64_t> &block_keys);

    void Reset();

    bool uses_linear() const { return linear_policy_.enabled(); }

    // Full objects currently resident. Linear state objects are excluded.
    uint64_t current_unique_blocks() const;

    // Resident Full objects under one total byte capacity. Linear state bytes
    // consume the same budget even though Linear objects are not counted.
    uint64_t FullObjectsWithinTotalBytes(uint64_t total_capacity_bytes) const;

    // Infinite-capacity resident working set, including Linear state bytes.
    uint64_t resident_bytes() const;
    uint64_t memory_usage_bytes() const;

private:
    friend class TtlLiteHit;

    RequestFact ProcessRequest(const std::vector<int64_t> &block_keys, std::size_t alive_from_position);
    FullRequestFact ProcessFullRequest(const std::vector<int64_t> &block_keys, std::size_t alive_from_position);
    RequestFact EvaluateRecoveryCurve(const std::vector<int64_t> &block_keys, std::size_t alive_from_position) const;
    FullRequestFact EncodeFullFact(const RequestFact &byte_fact) const;
    WeightedLruPool::PositionRemap CommitRequest(const std::vector<int64_t> &block_keys,
                                                 std::size_t alive_from_position);

    CacheObjectConfig object_config_;
    WeightedLruPool pool_;
    LiteHitLinearPolicy linear_policy_;
};

} // namespace kv_cache_manager
