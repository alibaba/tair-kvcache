#include "kv_cache_manager/optimizer/liteHit/lite_hit_linear.h"

namespace kv_cache_manager {

LiteHitLinearPolicy::LiteHitLinearPolicy(uint64_t linear_charge_bytes, uint64_t step_blocks)
    : linear_charge_bytes_(linear_charge_bytes), step_blocks_(step_blocks == 0 ? 1 : step_blocks) {}

bool LiteHitLinearPolicy::RequiredLinearBytes(const WeightedLruPool &pool,
                                              int64_t prefix_block_key,
                                              std::size_t alive_from_position,
                                              uint64_t &required_bytes) const {
    return enabled() &&
           pool.RequiredBytes({CacheObjectType::kLinear, prefix_block_key}, required_bytes, alive_from_position);
}

bool LiteHitLinearPolicy::ShouldWriteLinear(std::size_t position, std::size_t total_blocks) const {
    return (position + 1) % step_blocks_ == 0 || position == total_blocks - 1;
}

void LiteHitLinearPolicy::CommitLinearIfNeeded(WeightedLruPool &pool,
                                               int64_t prefix_block_key,
                                               std::size_t position,
                                               std::size_t total_blocks) const {
    if (enabled() && ShouldWriteLinear(position, total_blocks)) {
        pool.Touch({CacheObjectType::kLinear, prefix_block_key});
    }
}

} // namespace kv_cache_manager
