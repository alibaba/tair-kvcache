#include "kv_cache_manager/mrc/mrc_profiler.h"

#include <algorithm>
#include <cmath>

namespace kv_cache_manager {

namespace {

int64_t NormalizeMaxTrackedBlocks(int64_t value) { return value > 0 ? value : 65536; }

} // namespace

MrcProfiler::Histogram::Histogram(int64_t max_capacity)
    : max_capacity_blocks(std::max<int64_t>(max_capacity, 1))
    , fenwick(static_cast<size_t>(max_capacity_blocks) + 1, 0) {}

void MrcProfiler::Histogram::AddThreshold(uint64_t threshold) {
    if (threshold == 0 || threshold > static_cast<uint64_t>(max_capacity_blocks)) {
        return;
    }
    for (size_t i = static_cast<size_t>(threshold); i < fenwick.size(); i += i & (~i + 1)) {
        ++fenwick[i];
    }
}

void MrcProfiler::Histogram::Reset() {
    std::fill(fenwick.begin(), fenwick.end(), 0);
    total_accesses = 0;
}

uint64_t MrcProfiler::Histogram::HitsAt(uint64_t capacity_blocks) const {
    size_t i = static_cast<size_t>(std::min<uint64_t>(capacity_blocks, max_capacity_blocks));
    uint64_t hits = 0;
    for (; i > 0; i -= i & (~i + 1)) {
        hits += fenwick[i];
    }
    return hits;
}

MrcProfiler::MrcProfiler(const Options &options)
    : options_(options)
    , core_(std::min<int64_t>(NormalizeMaxTrackedBlocks(options.max_tracked_blocks), 4096),
            NormalizeMaxTrackedBlocks(options.max_tracked_blocks))
    , window_(NormalizeMaxTrackedBlocks(options.max_tracked_blocks))
    , completed_window_(NormalizeMaxTrackedBlocks(options.max_tracked_blocks))
    , cumulative_(NormalizeMaxTrackedBlocks(options.max_tracked_blocks)) {
    options_.max_tracked_blocks = NormalizeMaxTrackedBlocks(options.max_tracked_blocks);
}

void MrcProfiler::Observe(const std::vector<int64_t> &keys, int64_t now_us) {
    if (keys.empty()) {
        return;
    }
    RotateWindowIfNeeded(now_us);
    thresholds_.clear();
    core_.ProcessRequest(keys, thresholds_);
    window_.AddAccesses(keys.size());
    cumulative_.AddAccesses(keys.size());
    for (const uint64_t threshold : thresholds_) {
        window_.AddThreshold(threshold);
        cumulative_.AddThreshold(threshold);
    }
}

void MrcProfiler::RotateWindowIfNeeded(int64_t now_us) {
    if (window_start_us_ < 0) {
        window_start_us_ = now_us;
        return;
    }
    if (options_.window_seconds <= 0) {
        return;
    }
    const int64_t window_us = options_.window_seconds * 1000000;
    if (now_us - window_start_us_ >= window_us) {
        completed_window_ = window_;
        has_completed_window_ = true;
        window_.Reset();
        window_start_us_ = now_us;
    }
}

void MrcProfiler::FillSnapshot(const Histogram &hist,
                               const std::vector<double> &capacity_blocks,
                               Snapshot &out) const {
    out.total_accesses = static_cast<double>(hist.total_accesses);
    out.hit_rates.assign(capacity_blocks.size(), 0.0);
    if (hist.total_accesses == 0) {
        out.max_tracked_hit_rate = 0;
        return;
    }
    const double denominator = static_cast<double>(hist.total_accesses);
    out.max_tracked_hit_rate = static_cast<double>(hist.HitsAt(options_.max_tracked_blocks)) / denominator;
    for (size_t i = 0; i < capacity_blocks.size(); ++i) {
        if (capacity_blocks[i] < 1.0 || capacity_blocks[i] > static_cast<double>(options_.max_tracked_blocks)) {
            out.hit_rates[i] = -1.0; // explicitly outside the exact coverage boundary
            continue;
        }
        const auto capacity = static_cast<uint64_t>(std::floor(capacity_blocks[i]));
        out.hit_rates[i] = static_cast<double>(hist.HitsAt(capacity)) / denominator;
    }
}

bool MrcProfiler::QueryWindow(const std::vector<double> &capacity_blocks, Snapshot &out) const {
    if (!has_completed_window_) {
        return false;
    }
    FillSnapshot(completed_window_, capacity_blocks, out);
    return true;
}

void MrcProfiler::QueryCumulative(const std::vector<double> &capacity_blocks, Snapshot &out) const {
    FillSnapshot(cumulative_, capacity_blocks, out);
}

std::vector<MrcProfiler::CurvePoint> MrcProfiler::DumpCurveFrom(const Histogram &hist) const {
    std::vector<CurvePoint> points;
    if (hist.total_accesses == 0) {
        return points;
    }
    uint64_t previous_hits = 0;
    for (int64_t capacity = 1; capacity <= options_.max_tracked_blocks; ++capacity) {
        const uint64_t hits = hist.HitsAt(static_cast<uint64_t>(capacity));
        if (hits == previous_hits) {
            continue;
        }
        points.push_back(
            {static_cast<double>(capacity), static_cast<double>(hits) / static_cast<double>(hist.total_accesses)});
        previous_hits = hits;
    }
    return points;
}

std::vector<MrcProfiler::CurvePoint> MrcProfiler::DumpCurve(bool cumulative) const {
    return DumpCurveFrom(cumulative ? cumulative_ : completed_window_);
}

int64_t MrcProfiler::memory_usage_bytes() const {
    return core_.memory_usage_bytes() + static_cast<int64_t>(thresholds_.capacity() * sizeof(uint64_t)) +
           static_cast<int64_t>((window_.fenwick.capacity() + completed_window_.fenwick.capacity() +
                                 cumulative_.fenwick.capacity()) *
                                sizeof(uint64_t));
}

} // namespace kv_cache_manager
