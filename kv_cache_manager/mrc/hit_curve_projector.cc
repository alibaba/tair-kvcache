#include "kv_cache_manager/mrc/hit_curve_projector.h"

#include <algorithm>

namespace kv_cache_manager {

HitCurveProjection
HitCurveProjector::Project(const std::deque<MrcRequestFact> &facts, uint64_t capacity_blocks, int32_t block_size) {
    HitCurveProjection result;
    if (block_size <= 0) {
        return result;
    }
    for (const auto &fact : facts) {
        if (fact.input_token_len <= 0) {
            continue;
        }
        const uint64_t input_tokens = static_cast<uint64_t>(fact.input_token_len);
        const uint64_t hit_blocks = static_cast<uint64_t>(
            std::upper_bound(fact.thresholds.begin(), fact.thresholds.end(), capacity_blocks) -
            fact.thresholds.begin());
        result.total_tokens += input_tokens;
        result.hit_tokens +=
            std::min<uint64_t>(hit_blocks * static_cast<uint64_t>(block_size), input_tokens);
    }
    return result;
}

} // namespace kv_cache_manager
