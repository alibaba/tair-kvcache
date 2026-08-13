#pragma once

#include <cstdint>
#include <vector>

#include "kv_cache_manager/mrc/lite_hit_core.h"

namespace kv_cache_manager {

// Exact, request-level online MRC profiler for one instance lane.
//
// Each Observe call is one complete prefix request. LiteHitCore evaluates the
// prefix against the request-start LRU snapshot and commits the request
// tail-to-head, matching the formal offline LiteHit engine. The LRU stack is
// capped at max_tracked_blocks; this does not approximate points at or below
// that capacity because older entries are misses for all such points.
//
// Not thread-safe; callers serialize access externally.
class MrcProfiler {
public:
    struct Options {
        int64_t max_tracked_blocks = 65536;
        int64_t window_seconds = 3600;
    };

    struct Snapshot {
        double total_accesses = 0;
        // Hit rate at max_tracked_blocks. This is a finite-capacity bound, not
        // an infinite-capacity compulsory-miss estimate.
        double max_tracked_hit_rate = 0;
        std::vector<double> hit_rates;
    };

    struct CurvePoint {
        double capacity_blocks = 0;
        double hit_rate = 0;
    };

    explicit MrcProfiler(const Options &options);

    MrcProfiler(const MrcProfiler &) = delete;
    MrcProfiler &operator=(const MrcProfiler &) = delete;

    // One complete request, in prefix order.
    void Observe(const std::vector<int64_t> &keys, int64_t now_us);

    bool QueryWindow(const std::vector<double> &capacity_blocks, Snapshot &out) const;
    void QueryCumulative(const std::vector<double> &capacity_blocks, Snapshot &out) const;
    std::vector<CurvePoint> DumpCurve(bool cumulative) const;

    int64_t tracked_blocks() const { return core_.unique_blocks(); }
    int64_t tracked_capacity_blocks() const { return options_.max_tracked_blocks; }
    int64_t memory_usage_bytes() const;

private:
    struct Histogram {
        explicit Histogram(int64_t max_capacity_blocks = 1);

        void AddThreshold(uint64_t threshold);
        void AddAccesses(uint64_t count) { total_accesses += count; }
        void Reset();
        uint64_t HitsAt(uint64_t capacity_blocks) const;

        int64_t max_capacity_blocks = 1;
        std::vector<uint64_t> fenwick;
        uint64_t total_accesses = 0;
    };

    void RotateWindowIfNeeded(int64_t now_us);
    void FillSnapshot(const Histogram &hist,
                      const std::vector<double> &capacity_blocks,
                      Snapshot &out) const;
    std::vector<CurvePoint> DumpCurveFrom(const Histogram &hist) const;

    Options options_;
    LiteHitCore core_;
    std::vector<uint64_t> thresholds_;

    Histogram window_;
    Histogram completed_window_;
    bool has_completed_window_ = false;
    int64_t window_start_us_ = -1;
    Histogram cumulative_;
};

} // namespace kv_cache_manager
