#pragma once

#include <cstdint>
#include <optional>

#include "kv_cache_manager/optimizer/metrics/mrc_window.h"
#include "kv_cache_manager/optimizer/metrics/reuse_interval.h"

namespace kv_cache_manager {

// One non-overlapping five-minute input accumulator for capacity history.
// Membership uses local receipt time; reuse duration uses producer timestamps.
// No rolling slices, sample queue, or per-report snapshots are retained.
// Calls are serialized by the owning InstanceState mutex.
class CapacityPlanningWindow {
public:
    static constexpr int64_t kWindowMs = 5 * 60 * 1000;
    struct Result {
        int64_t timestamp_ms;
        ReuseIntervalStats reuse;
        uint64_t required_blocks;
    };

    // Called before recording a request and before reporting. Return each
    // completed window once, even if the report is late; idle gaps are skipped.
    std::optional<Result> Complete(int64_t now_ms) {
        if (now_ms < 0) {
            return std::nullopt;
        }
        const int64_t bucket = now_ms / kWindowMs;
        if (bucket_ < 0) {
            bucket_ = bucket;
            return std::nullopt;
        }
        if (bucket <= bucket_) {
            return std::nullopt;
        }
        // Anchor to the completed interval, not the delayed report time.
        Result result{(bucket_ + 1) * kWindowMs - 1, reuse_.Take(), mrc_.RequiredBlocks(9900)};
        mrc_.Reset();
        bucket_ = bucket;
        return result;
    }

    ReuseInterval *reuse() { return &reuse_; }
    void RecordFact(const RequestFact &fact) { mrc_.Record(fact); }

    void Reset() {
        bucket_ = -1;
        reuse_.Reset();
        mrc_.Reset();
    }

private:
    int64_t bucket_ = -1;
    ReuseInterval reuse_;
    MrcWindow mrc_;
};

} // namespace kv_cache_manager
