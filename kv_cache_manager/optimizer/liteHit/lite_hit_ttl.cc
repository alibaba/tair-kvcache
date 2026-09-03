#include "kv_cache_manager/optimizer/liteHit/lite_hit_ttl.h"

#include <algorithm>
#include <stdexcept>

namespace kv_cache_manager {

uint64_t LiteHitTtlState::AgeNs(int64_t now_ns, int64_t timestamp_ns) {
    if (timestamp_ns >= now_ns) {
        return 0;
    }
    return static_cast<uint64_t>(now_ns) - static_cast<uint64_t>(timestamp_ns);
}

void LiteHitTtlState::AdvanceTime(int64_t now_ns, const WeightedLruPool &pool) {
    if (!enabled()) {
        return;
    }

    const std::size_t old_watermark = dead_below_position_;
    while (!position_epochs_.empty() && AgeNs(now_ns, position_epochs_.front().timestamp_ns) >= ttl_ns_) {
        position_epochs_.pop_front();
        dead_below_position_ =
            position_epochs_.empty() ? pool.position_count() + 1 : position_epochs_.front().start_position;
    }

    if (dead_below_position_ > old_watermark) {
        expired_full_blocks_ += pool.FullObjectsBefore(dead_below_position_) - pool.FullObjectsBefore(old_watermark);
    }
}

void LiteHitTtlState::BeginCommit(int64_t now_ns, std::size_t next_position) {
    if (!enabled()) {
        return;
    }

    const int64_t epoch_ns = position_epochs_.empty() ? now_ns : std::max(now_ns, position_epochs_.back().timestamp_ns);
    if (position_epochs_.empty() || position_epochs_.back().timestamp_ns != epoch_ns) {
        position_epochs_.push_back(PositionEpoch{next_position, epoch_ns});
    }
}

void LiteHitTtlState::ApplyPositionRemap(const WeightedLruPool::PositionRemap &remap) {
    if (!enabled() || !remap.compacted) {
        return;
    }

    for (PositionEpoch &epoch : position_epochs_) {
        epoch.start_position = remap.RemapBoundary(epoch.start_position);
    }

    std::size_t deduped_size = 0;
    for (const PositionEpoch &epoch : position_epochs_) {
        if (deduped_size > 0 && position_epochs_[deduped_size - 1].start_position == epoch.start_position) {
            position_epochs_[deduped_size - 1].timestamp_ns = epoch.timestamp_ns;
        } else {
            position_epochs_[deduped_size++] = epoch;
        }
    }
    position_epochs_.resize(deduped_size);
    if (dead_below_position_ > 0) {
        dead_below_position_ = remap.RemapBoundary(dead_below_position_);
    }
}

void LiteHitTtlState::Reset() {
    position_epochs_.clear();
    dead_below_position_ = 0;
    expired_full_blocks_ = 0;
}

uint64_t LiteHitTtlState::memory_usage_bytes() const {
    return static_cast<uint64_t>(position_epochs_.size()) * sizeof(PositionEpoch);
}

TtlLiteHit::TtlLiteHit(uint64_t ttl_ns) : TtlLiteHit(LiteHit::CacheObjectConfig{}, ttl_ns) {}

TtlLiteHit::TtlLiteHit(const LiteHit::CacheObjectConfig &object_config, uint64_t ttl_ns)
    : core_(object_config), ttl_state_(ttl_ns) {}

RequestFact TtlLiteHit::ProcessRequest(const std::vector<int64_t> &block_keys, int64_t now_ns) {
    ttl_state_.AdvanceTime(now_ns, core_.pool_);
    const RequestFact fact = core_.EvaluateRecoveryCurve(block_keys, alive_from_position());
    if (!block_keys.empty()) {
        ttl_state_.BeginCommit(now_ns, core_.pool_.position_count() + 1);
    }
    const WeightedLruPool::PositionRemap remap = core_.CommitRequest(block_keys, alive_from_position());
    ttl_state_.ApplyPositionRemap(remap);
    return fact;
}

FullRequestFact TtlLiteHit::ProcessFullRequest(const std::vector<int64_t> &block_keys, int64_t now_ns) {
    if (uses_linear()) {
        throw std::logic_error("ProcessFullRequest requires a Full-only LiteHit");
    }
    return core_.EncodeFullFact(ProcessRequest(block_keys, now_ns));
}

void TtlLiteHit::AdvanceTime(int64_t now_ns) { ttl_state_.AdvanceTime(now_ns, core_.pool_); }

void TtlLiteHit::Reset() {
    core_.Reset();
    ttl_state_.Reset();
}

uint64_t TtlLiteHit::current_unique_blocks() const { return core_.pool_.resident_full_count(alive_from_position()); }

uint64_t TtlLiteHit::FullObjectsWithinTotalBytes(uint64_t total_capacity_bytes) const {
    return core_.pool_.FullObjectsWithinBytes(total_capacity_bytes, alive_from_position());
}

uint64_t TtlLiteHit::resident_bytes() const { return core_.pool_.resident_bytes(alive_from_position()); }

uint64_t TtlLiteHit::memory_usage_bytes() const { return core_.memory_usage_bytes() + ttl_state_.memory_usage_bytes(); }

} // namespace kv_cache_manager
