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

// Full-only compact representation. Equal Full charges make consecutive byte
// thresholds an arithmetic run on the block axis.
struct FullRequestFact {
    std::vector<HitCurveSegment> hit_curve;
};

// One breakpoint of the core's default byte-axis request curve. Linear
// attention jumps between Linear states; Full-only produces one point per
// covered block before optionally being encoded as FullRequestFact.
struct ByteStepPoint {
    uint64_t min_total_capacity_bytes = 0;
    uint64_t hit_blocks = 0;

    bool operator==(const ByteStepPoint &other) const {
        return min_total_capacity_bytes == other.min_total_capacity_bytes && hit_blocks == other.hit_blocks;
    }
};

// Points are strictly increasing in both fields (monotone envelope). Empty
// means no byte capacity can recover this request.
struct RequestFact {
    std::vector<ByteStepPoint> points;
};

// Stateless projection from capacity-independent facts to hit blocks for a
// concrete capacity. Both the online path and the facts post-query must use
// this projector; no other component may reimplement the boundary logic.
class HitCurveProjector {
public:
    // Default byte-step curve: largest hit_blocks whose threshold fits.
    static uint64_t ProjectBytes(const RequestFact &fact, uint64_t total_capacity_bytes);

    static uint64_t ProjectInfinite(const RequestFact &fact);

    // Full-only RLE projection on the block axis.
    static uint64_t ProjectFullBlocks(const FullRequestFact &fact, uint64_t capacity_blocks);

    // Full-only byte projection. block_bytes must be positive.
    static uint64_t ProjectFullBytes(const FullRequestFact &fact, uint64_t capacity_bytes, uint64_t block_bytes);

    static uint64_t ProjectFullInfinite(const FullRequestFact &fact);
};

} // namespace kv_cache_manager
