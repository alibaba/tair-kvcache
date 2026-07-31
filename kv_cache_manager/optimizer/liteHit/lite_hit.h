#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
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
//
// An optional fixed TTL adds the online TtlCacheIndexerWrapper semantics on
// top of the LRU stack: a block whose age since last access reached the TTL
// is a miss for every capacity, every access (hit or miss) refreshes
// last_access, and time is the caller-provided trace timestamp so the replay
// stays deterministic. Ages are monotone along the LRU stack (younger blocks
// are always fresher), so expired blocks never inflate the reuse distance of
// alive ones and one replay stays exact for every capacity under the fixed
// TTL.
//
// The TTL needs no per-key timestamps: marker positions are assigned
// monotonically, so a block's last access time is the commit time of the
// request that assigned its current position. A small epoch deque maps
// position ranges to commit timestamps; expired epochs advance a single
// dead-below watermark and liveness is one integer comparison. The extra
// state is O(distinct commit timestamps inside one TTL window), independent
// of the number of unique blocks.
class LiteHit {
public:
    // ttl_ns == 0 disables the TTL (pure LRU).
    explicit LiteHit(uint64_t ttl_ns = 0) : ttl_ns_(ttl_ns) {}

    // One call is one request boundary. block_keys must be the normalized
    // prefix-chained keys of all complete blocks of the request, in request
    // order; input length parsing, validation, and prefix hashing happen in
    // the shared preprocessing outside the core. now_ns is the request trace
    // timestamp (time-sorted); it is only consulted when a TTL is configured.
    //
    // Phase 1 evaluates the prefix hit curve against the request-start LRU
    // snapshot; with a TTL an expired block stops the prefix like a cold
    // one. Phase 2 commits every complete block tail-to-head (reverse
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
    RequestFact ProcessRequest(const std::vector<int64_t> &block_keys, int64_t now_ns = 0);

    void Reset();

    uint64_t ttl_ns() const { return ttl_ns_; }

    // Advances the TTL watermark to now_ns without processing a request, so
    // observability reads reflect the current time instead of the last
    // request's. No-op without a TTL.
    void AdvanceTime(int64_t now_ns) {
        if (ttl_ns_ > 0) {
            AdvanceTtlWatermark(now_ns);
        }
    }

    // Unique blocks currently alive: expired markers below the TTL watermark
    // are excluded even though their table entries linger until compaction.
    uint64_t current_unique_blocks() const { return alive_marker_count(); }

    // Coarse memory estimate for observability. It is derived from the state
    // already required by the algorithm and does not retain extra trace data.
    uint64_t memory_usage_bytes() const;

private:
    struct SnapshotEntry {
        bool is_resident = false;
        uint64_t required_blocks = 0;
    };

    // Positions in [start_position, next epoch's start_position) were
    // assigned by a commit at timestamp_ns; timestamps are non-decreasing
    // along the deque.
    struct PositionEpoch {
        std::size_t start_position = 0;
        int64_t timestamp_ns = 0;
    };

    RequestFact BuildHitCurve(const std::vector<int64_t> &block_keys) const;
    void CommitRequest(const std::vector<int64_t> &block_keys, int64_t now_ns);
    void MaybeCompactPositions();
    uint64_t ReuseDistance(std::size_t previous_position) const;
    // Resident markers at or above the TTL watermark (all markers when no
    // watermark is active).
    uint64_t alive_marker_count() const;
    // Pops expired epochs and advances dead_below_position_.
    void AdvanceTtlWatermark(int64_t now_ns);

    uint64_t ttl_ns_ = 0;
    DynamicFenwickTree fenwick_;
    std::unordered_map<int64_t, std::size_t> last_positions_;
    // TTL state (empty when ttl_ns_ == 0): markers below the watermark are
    // expired.
    std::deque<PositionEpoch> position_epochs_;
    std::size_t dead_below_position_ = 0;
};

} // namespace kv_cache_manager
