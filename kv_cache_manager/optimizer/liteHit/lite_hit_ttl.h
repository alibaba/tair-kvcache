#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

#include "kv_cache_manager/optimizer/liteHit/lite_hit.h"

namespace kv_cache_manager {

// Optional time policy for the shared LiteHit recency core. Commit positions
// are monotone with access time, so all expired objects form one old-position
// prefix. Epochs map position ranges to request timestamps without retaining
// a timestamp per key.
class LiteHitTtlState {
public:
    explicit LiteHitTtlState(uint64_t ttl_ns = 0) : ttl_ns_(ttl_ns) {}

    bool enabled() const { return ttl_ns_ > 0; }
    uint64_t ttl_ns() const { return ttl_ns_; }
    std::size_t alive_from_position() const { return dead_below_position_; }
    uint64_t expired_full_blocks() const { return expired_full_blocks_; }

    // Advances the strict TTL deadline before a request or observability read.
    void AdvanceTime(int64_t now_ns, const WeightedLruPool &pool);

    // Records the timestamp of positions about to be appended by a non-empty
    // request. Out-of-order timestamps are clamped to the latest epoch.
    void BeginCommit(int64_t now_ns, std::size_t next_position);

    // Applies the pool's dense-position mapping after compaction and removes
    // epochs whose position ranges became empty.
    void ApplyPositionRemap(const WeightedLruPool::PositionRemap &remap);

    void Reset();
    uint64_t memory_usage_bytes() const;

private:
    struct PositionEpoch {
        std::size_t start_position = 0;
        int64_t timestamp_ns = 0;
    };

    static uint64_t AgeNs(int64_t now_ns, int64_t timestamp_ns);

    uint64_t ttl_ns_ = 0;
    std::deque<PositionEpoch> position_epochs_;
    std::size_t dead_below_position_ = 0;
    uint64_t expired_full_blocks_ = 0;
};

// TTL decorator for the capacity-independent LiteHit core. The wrapped core
// owns all Full/Linear recency state and remains unaware of timestamps. This
// layer supplies a liveness boundary while evaluating and committing, and
// translates pool position compaction back into its timestamp epochs.
class TtlLiteHit {
public:
    explicit TtlLiteHit(uint64_t ttl_ns = 0);
    TtlLiteHit(const LiteHit::CacheObjectConfig &object_config, uint64_t ttl_ns);

    RequestFact ProcessRequest(const std::vector<int64_t> &block_keys, int64_t now_ns = 0);
    FullRequestFact ProcessFullRequest(const std::vector<int64_t> &block_keys, int64_t now_ns = 0);

    void AdvanceTime(int64_t now_ns);
    void Reset();

    bool uses_linear() const { return core_.uses_linear(); }
    uint64_t ttl_ns() const { return ttl_state_.ttl_ns(); }
    uint64_t ttl_expired_blocks() const { return ttl_state_.expired_full_blocks(); }

    uint64_t current_unique_blocks() const;
    uint64_t FullObjectsWithinTotalBytes(uint64_t total_capacity_bytes) const;
    uint64_t resident_bytes() const;
    uint64_t memory_usage_bytes() const;

private:
    std::size_t alive_from_position() const { return ttl_state_.alive_from_position(); }

    LiteHit core_;
    LiteHitTtlState ttl_state_;
};

} // namespace kv_cache_manager
