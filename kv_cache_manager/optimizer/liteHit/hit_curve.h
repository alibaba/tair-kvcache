#pragma once

#include <cstdint>
#include <vector>

namespace kv_cache_manager {

struct HitCurveSegment {
    uint64_t start_required_blocks = 0;
    uint64_t run_length = 0;

    bool operator==(const HitCurveSegment &other) const {
        return start_required_blocks == other.start_required_blocks && run_length == other.run_length;
    }
};

// Capacity-independent result for one request.
struct RequestFact {
    std::vector<HitCurveSegment> hit_curve;
};

class HitCurveProjector {
public:
    static uint64_t ProjectBlocks(const RequestFact &fact, uint64_t capacity_blocks);
    static uint64_t ProjectBytes(const RequestFact &fact, uint64_t capacity_bytes, uint64_t block_bytes);
    static uint64_t ProjectInfinite(const RequestFact &fact);
};

} // namespace kv_cache_manager
