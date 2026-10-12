#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace kv_cache_manager {

// A bounded, revisable window-level estimate, NOT an exact cache reconstruction.
// Synchronization belongs to InstanceState. No per-request work is performed here.
class CapacityHistory {
public:
    static constexpr int64_t kStepMs = 5 * 60 * 1000;
    static constexpr size_t kSlots = 288;
    struct Point {
        int64_t timestamp_ms = 0; // 0 means this slot has no observation.
        int64_t capacity_bytes = 0;
    };

    void Update(int64_t now_ms, uint64_t reuse_ns, int64_t capacity_bytes, bool has_sample) {
        if (now_ms <= 0 || now_ms < last_update_ms_) {
            return;
        }
        last_update_ms_ = now_ms;
        const int64_t end = now_ms / kStepMs;
        const int64_t oldest = std::max<int64_t>(1, end - static_cast<int64_t>(kSlots) + 1);
        for (auto &point : points_) {
            if (point.timestamp_ms / kStepMs < oldest) {
                point = {};
            }
        }
        if (!has_sample || capacity_bytes < 0) {
            return;
        }
        // Bucket-overlap semantics: round both endpoints down. Even a reuse
        // interval shorter than 5 minutes updates the current bucket.
        const auto reuse_ms = std::min<uint64_t>(reuse_ns / 1000000, static_cast<uint64_t>(now_ms));
        const int64_t begin = std::max(oldest, (now_ms - static_cast<int64_t>(reuse_ms)) / kStepMs);
        for (int64_t bucket = begin; bucket <= end; ++bucket) {
            auto &point = points_[static_cast<size_t>(bucket) % kSlots];
            const int64_t timestamp_ms = bucket * kStepMs;
            if (point.timestamp_ms != timestamp_ms) {
                point = {timestamp_ms, capacity_bytes};
            } else {
                point.capacity_bytes = std::max(point.capacity_bytes, capacity_bytes);
            }
        }
    }

    const std::array<Point, kSlots> &points() const { return points_; }
    void Reset() {
        points_ = {};
        last_update_ms_ = 0;
    }

private:
    std::array<Point, kSlots> points_{};
    int64_t last_update_ms_ = 0;
};

} // namespace kv_cache_manager
