#pragma once

#include <cstdint>
#include <vector>

#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"
#include "kv_cache_manager/optimizer/liteHit/weighted_lru_pool.h"

namespace kv_cache_manager {

// LiteHit core for linear-attention (Mamba) instances: Full blocks plus
// checkpoint Mamba states replayed against one weighted LRU pool that both
// object types share. Per request it emits a capacity-independent
// MambaRequestFact; any total capacity is answered afterwards through
// HitCurveProjector::ProjectMambaBytes.
class LiteHitMamba {
public:
    struct Config {
        uint64_t full_charge_bytes = 0;  // size of one FullObject (one KV block)
        uint64_t mamba_charge_bytes = 0; // size of one checkpoint MambaObject
        uint64_t step_blocks = 1;        // checkpoint every step_blocks complete blocks (>= 1)
    };

    explicit LiteHitMamba(const Config &config);

    // One call is one request boundary; block_keys are the normalized
    // prefix-chained keys of all complete blocks (analysis granularity).
    //
    // Phase 1 evaluates, on the request-start snapshot, every checkpoint
    // candidate: it is recoverable at total capacity C iff its MambaObject fits
    // the pool AND the Full prefix covers up to it. The monotone envelope of
    // (min_total_capacity_bytes, position + 1) is the emitted fact. Phase 2
    // commits the fixed set: every complete block one FullObject touch, every
    // checkpoint one MambaObject touch, tail-to-head, with the MambaObject
    // touched before the FullObject at the same position.
    MambaRequestFact ProcessRequest(const std::vector<int64_t> &block_keys);

    void Reset();

    // Resident FullObject count (unique_keys reporting; Mamba objects are
    // intentionally not counted).
    uint64_t current_unique_blocks() const;
    // Resident FullObjects under one total byte capacity.
    uint64_t FullObjectsWithinTotalBytes(uint64_t total_capacity_bytes) const;
    // Infinite-capacity working set: bytes of all resident Full and Mamba
    // objects.
    uint64_t resident_bytes() const;
    uint64_t memory_usage_bytes() const;

private:
    // Checkpoint at position i iff (i + 1) % step_blocks == 0 or i == n - 1.
    bool IsCheckpoint(std::size_t position, std::size_t total_blocks) const;

    Config config_;
    // Both object types share one recency order and one byte budget, so a
    // request's own checkpoints compete with its Full blocks.
    WeightedLruPool pool_;
};

} // namespace kv_cache_manager
