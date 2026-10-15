#pragma once

#include <cstdint>
#include <optional>

#include "kv_cache_manager/optimizer/metrics/mrc_window.h"
#include "kv_cache_manager/optimizer/metrics/reuse_interval.h"

namespace kv_cache_manager {

// One non-overlapping input accumulator for capacity history.
// Membership uses local receipt time; reuse duration uses producer timestamps.
// No rolling slices, sample queue, or per-report snapshots are retained.
// Calls are serialized by the owning InstanceState mutex.
class CapacityPlanningWindow {
public:
    static constexpr int64_t kWindowMs = 5 * 60 * 1000;
    explicit CapacityPlanningWindow(int64_t window_ms = kWindowMs) : window_ms_(window_ms) {}
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
        const int64_t bucket = now_ms / window_ms_;
        if (bucket_ < 0) {
            bucket_ = bucket;
            return std::nullopt;
        }
        if (bucket <= bucket_) {
            return std::nullopt;
        }
        // Anchor to the completed interval, not the delayed report time.
        Result result{(bucket_ + 1) * window_ms_ - 1, reuse_.Take(), mrc_.RequiredBlocks(9900)};
        mrc_.Reset();
        bucket_ = bucket;
        return result;
    }

    ReuseInterval *reuse() { return &reuse_; }
    void RecordFact(const RequestFact &fact) { mrc_.Record(fact); }
    bool IsComplete(int64_t now_ms) const { return bucket_ >= 0 && now_ms >= 0 && now_ms / window_ms_ > bucket_; }
    void MergeInputs(const CapacityPlanningWindow &other) {
        reuse_.Merge(other.reuse_);
        mrc_.Merge(other.mrc_);
    }

    void Reset() {
        bucket_ = -1;
        reuse_.Reset();
        mrc_.Reset();
    }

private:
    const int64_t window_ms_;
    int64_t bucket_ = -1;
    ReuseInterval reuse_;
    MrcWindow mrc_;
};

// All windows consume the same raw distributions. Only the five-minute base
// receives per-request updates; longer windows merge its counts before drain.
// Never average percentiles or merge finalized capacities.
class CapacityPlanningWindows {
public:
    static constexpr std::array<int64_t, 4> kWindowsMs = {300000, 900000, 1800000, 3600000};
    std::array<std::optional<CapacityPlanningWindow::Result>, 4> Complete(int64_t now_ms) {
        if (windows_[0].IsComplete(now_ms)) {
            for (std::size_t i = 1; i < windows_.size(); ++i) {
                windows_[i].MergeInputs(windows_[0]);
            }
        }
        std::array<std::optional<CapacityPlanningWindow::Result>, 4> results;
        for (std::size_t i = 0; i < windows_.size(); ++i) {
            results[i] = windows_[i].Complete(now_ms);
        }
        return results;
    }
    ReuseInterval *reuse() { return windows_[0].reuse(); }
    void RecordFact(const RequestFact &fact) { windows_[0].RecordFact(fact); }
    void Reset() {
        for (auto &window : windows_) {
            window.Reset();
        }
    }

private:
    std::array<CapacityPlanningWindow, 4> windows_ = {CapacityPlanningWindow(300000),
                                                      CapacityPlanningWindow(900000),
                                                      CapacityPlanningWindow(1800000),
                                                      CapacityPlanningWindow(3600000)};
};

} // namespace kv_cache_manager
