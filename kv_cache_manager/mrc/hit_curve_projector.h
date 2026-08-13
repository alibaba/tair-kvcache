#pragma once

#include <cstdint>
#include <deque>
#include <vector>

namespace kv_cache_manager {

// Capacity-independent result of one LiteHit request. Facts are snapshots of
// completed calculations and are currently retained for the process lifetime.
struct MrcRequestFact {
    int64_t input_token_len = 0;
    std::vector<uint64_t> thresholds;
};

struct HitCurveProjection {
    uint64_t total_tokens = 0;
    uint64_t hit_tokens = 0;

    double hit_rate() const {
        return total_tokens == 0 ? 0.0 : static_cast<double>(hit_tokens) / total_tokens;
    }
};

// Projects capacity-independent request facts onto a configured capacity.
// Keeping this separate from LiteHit makes capacity points reporting
// parameters instead of part of the simulation state.
class HitCurveProjector {
public:
    static HitCurveProjection
    Project(const std::deque<MrcRequestFact> &facts, uint64_t capacity_blocks, int32_t block_size);
};

} // namespace kv_cache_manager
