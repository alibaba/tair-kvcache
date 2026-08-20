#pragma once

#include <cstddef>
#include <cstdint>

#include "kv_cache_manager/optimizer/liteHit/weighted_lru_pool.h"

namespace kv_cache_manager {

// Linear-attention restore/write policy layered on the shared LiteHit core.
// It owns no recency state: Full blocks and Linear state objects always live
// in the core's one WeightedLruPool.
class LiteHitLinearPolicy {
public:
    LiteHitLinearPolicy(uint64_t linear_charge_bytes = 0, uint64_t step_blocks = 0);

    bool enabled() const { return linear_charge_bytes_ > 0; }

    // Historical Linear states are valid read candidates regardless of
    // whether their position belongs to the current request's write schedule.
    bool RequiredLinearBytes(const WeightedLruPool &pool,
                             int64_t prefix_block_key,
                             std::size_t alive_from_position,
                             uint64_t &required_bytes) const;

    // Writes periodic Linear states and always the current request's tail.
    void CommitLinearIfNeeded(WeightedLruPool &pool,
                              int64_t prefix_block_key,
                              std::size_t position,
                              std::size_t total_blocks) const;

private:
    bool ShouldWriteLinear(std::size_t position, std::size_t total_blocks) const;

    uint64_t linear_charge_bytes_ = 0;
    uint64_t step_blocks_ = 1;
};

} // namespace kv_cache_manager
