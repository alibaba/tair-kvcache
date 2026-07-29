#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/optimizer/liteHit/dynamic_fenwick_tree.h"
#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"

namespace kv_cache_manager {

// LiteHit is an exact LRU replay core for equal-charge full-attention
// blocks. It keeps one global LRU order and, per request, emits
// capacity-independent facts (RequestFact); it never receives a capacity
// list and never accumulates per-capacity results. Any capacity is answered
// afterwards by projecting the facts with HitCurveProjector.
class LiteHit {
public:
    LiteHit() = default;

    // One call is one request boundary. block_keys must be the normalized
    // prefix-chained keys of all complete blocks of the request, in request
    // order; input length parsing, validation, and prefix hashing happen in
    // the shared preprocessing outside the core.
    //
    // Phase 1 evaluates the prefix hit curve against the request-start LRU
    // snapshot. Phase 2 commits every complete block tail-to-head (reverse
    // request order), including blocks after the first miss. With
    // prefix-chained keys a chain head is therefore always newer than its
    // resident descendants, so the global LRU victim is always a leaf,
    // matching leaf-first eviction used by production prefix caches
    // (vLLM free-queue order, SGLang radix cache).
    //
    // Under the prefix-hash contract per-block thresholds strictly increase,
    // so the arithmetic-run RLE in RequestFact is lossless. Non-contract
    // input (duplicate keys inside a request) is first defensively merged to
    // its longest prefix per threshold and then encoded monotonically; the
    // projection is never optimistic.
    RequestFact ProcessRequest(const std::vector<int64_t> &block_keys);

    void Reset();

    uint64_t current_unique_blocks() const { return static_cast<uint64_t>(last_positions_.size()); }

    // Coarse memory estimate for observability. It is derived from the state
    // already required by the algorithm and does not retain extra trace data.
    uint64_t memory_usage_bytes() const;

private:
    struct SnapshotEntry {
        bool is_resident = false;
        uint64_t required_blocks = 0;
    };

    RequestFact BuildHitCurve(const std::vector<int64_t> &block_keys) const;
    void CommitRequest(const std::vector<int64_t> &block_keys);
    void MaybeCompactPositions();
    uint64_t ReuseDistance(std::size_t previous_position) const;

    DynamicFenwickTree fenwick_;
    std::unordered_map<int64_t, std::size_t> last_positions_;
};

} // namespace kv_cache_manager
