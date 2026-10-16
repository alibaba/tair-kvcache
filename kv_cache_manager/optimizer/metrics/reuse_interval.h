#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace kv_cache_manager {

// Statistics since the last Take(). Times are meaningful only when count > 0.
struct ReuseIntervalStats {
    uint64_t count = 0;
    uint64_t min_ns = 0;
    uint64_t max_ns = 0;
    double avg_ns = 0.0;
    uint64_t p95_ns = 0;
    uint64_t p99_ns = 0;
};

// Accumulates capacity-independent intervals between two
// resident accesses to the same block. The owner supplies already-normalized
// intervals; synchronization is provided by the owning InstanceState mutex.
class ReuseInterval {
public:
    void Record(uint64_t interval_ns);
    ReuseIntervalStats Snapshot() const;
    ReuseIntervalStats Take();
    void Reset();

private:
    // Exact integer bins below 256 ns; above that, 128 bins per power
    // of two bound the upward quantile error to less than 1/128.
    // 7424 uint64 bins = 58 KiB, independent of traffic volume.
    static constexpr std::size_t kSubBuckets = 128;
    static constexpr std::size_t kBucketCount = kSubBuckets * (1 + 64 - 7);
    static std::size_t BucketIndex(uint64_t interval_ns);
    static uint64_t BucketUpperBound(std::size_t index);
    uint64_t Percentile(uint64_t percent) const;

    std::array<uint64_t, kBucketCount> buckets_{};
    ReuseIntervalStats window_;
    // A floating-point sum avoids uint64 nanosecond overflow in busy windows.
    long double total_ns_ = 0.0L;
};

} // namespace kv_cache_manager
