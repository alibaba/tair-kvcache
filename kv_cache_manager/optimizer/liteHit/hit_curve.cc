#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"

#include <algorithm>

namespace kv_cache_manager {

uint64_t HitCurveProjector::ProjectBlocks(const RequestFact &fact, uint64_t capacity_blocks) {
    uint64_t hits = 0;
    for (const HitCurveSegment &segment : fact.hit_curve) {
        if (segment.start_required_blocks > capacity_blocks) {
            break;
        }
        const uint64_t reachable = capacity_blocks - segment.start_required_blocks + 1;
        hits += std::min(segment.run_length, reachable);
    }
    return hits;
}

uint64_t HitCurveProjector::ProjectBytes(const RequestFact &fact, uint64_t capacity_bytes, uint64_t block_bytes) {
    return ProjectBlocks(fact, capacity_bytes / block_bytes);
}

uint64_t HitCurveProjector::ProjectInfinite(const RequestFact &fact) {
    uint64_t hits = 0;
    for (const HitCurveSegment &segment : fact.hit_curve) {
        hits += segment.run_length;
    }
    return hits;
}

uint64_t HitCurveProjector::ProjectMambaBytes(const MambaRequestFact &fact, uint64_t total_capacity_bytes) {
    uint64_t hits = 0;
    for (const MambaCurvePoint &point : fact.points) {
        if (point.min_total_capacity_bytes > total_capacity_bytes) {
            break;
        }
        hits = point.hit_blocks;
    }
    return hits;
}

uint64_t HitCurveProjector::ProjectMambaInfinite(const MambaRequestFact &fact) {
    return fact.points.empty() ? 0 : fact.points.back().hit_blocks;
}

} // namespace kv_cache_manager
