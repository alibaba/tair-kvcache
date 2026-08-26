#pragma once

#include <cstdint>
#include <vector>

namespace kv_cache_manager {

// One arithmetic run of a per-request hit curve. The j-th block inside a
// segment (0-based) becomes a prefix hit once the LRU holds
// start_required_blocks + j blocks: every extra block of capacity adds
// exactly one hit. Segment starts are strictly increasing and adjacent
// segments are never mergeable (a gap of at least one threshold separates
// them).
struct HitCurveSegment {
    uint64_t start_required_blocks = 0;
    uint64_t run_length = 0;

    bool operator==(const HitCurveSegment &other) const {
        return start_required_blocks == other.start_required_blocks && run_length == other.run_length;
    }
};

// Capacity-independent facts of one request replay. An empty curve means no
// capacity can hit this request (cold prefix head).
struct RequestFact {
    std::vector<HitCurveSegment> hit_curve;
};

// Stateless projection from capacity-independent facts to hit blocks for a
// concrete capacity. Both the online path and the facts post-query must use
// this projector; no other component may reimplement the boundary logic.
class HitCurveProjector {
public:
    // Hit blocks when the cache holds exactly capacity_blocks blocks.
    static uint64_t ProjectBlocks(const RequestFact &fact, uint64_t capacity_blocks);

    // Hit blocks for a byte capacity; the capacity is floor-converted with the
    // per-block byte charge before projection. block_bytes must be positive.
    static uint64_t ProjectBytes(const RequestFact &fact, uint64_t capacity_bytes, uint64_t block_bytes);

    // Hit blocks with unbounded capacity: cold misses remain, capacity misses
    // disappear, so this is the total run length of the curve.
    static uint64_t ProjectInfinite(const RequestFact &fact);
};

} // namespace kv_cache_manager
