#include "kv_cache_manager/optimizer/metrics/reuse_interval.h"

#include <algorithm>
#include <limits>

namespace kv_cache_manager {

void ReuseInterval::Record(uint64_t interval_ns) {
    if (window_.count == std::numeric_limits<uint64_t>::max()) {
        return;
    }
    window_.min_ns = window_.count == 0 ? interval_ns : std::min(window_.min_ns, interval_ns);
    window_.max_ns = std::max(window_.max_ns, interval_ns);
    ++window_.count;
    total_ns_ += static_cast<long double>(interval_ns);
    ++buckets_[BucketIndex(interval_ns)];
}

std::size_t ReuseInterval::BucketIndex(uint64_t interval_ns) {
    if (interval_ns < kSubBuckets) {
        return static_cast<std::size_t>(interval_ns);
    }
    const unsigned exponent = 63U - __builtin_clzll(static_cast<unsigned long long>(interval_ns));
    const unsigned shift = exponent - 7U;
    return shift * kSubBuckets + static_cast<std::size_t>(interval_ns >> shift);
}

uint64_t ReuseInterval::BucketUpperBound(std::size_t index) {
    if (index < kSubBuckets) {
        return index;
    }
    const auto offset = index - kSubBuckets;
    const unsigned shift = offset / kSubBuckets;
    const uint64_t mantissa = kSubBuckets + offset % kSubBuckets;
    // The last bucket ends at UINT64_MAX, not the overflowing 2^64.
    if (shift == 56 && mantissa == 255) {
        return std::numeric_limits<uint64_t>::max();
    }
    return ((mantissa + 1) << shift) - 1;
}

uint64_t ReuseInterval::Percentile(uint64_t percent) const {
    if (window_.count == 0) {
        return 0;
    }
    // Nearest rank: ceil(count * percent / 100), without integer overflow.
    const uint64_t rank = (window_.count / 100) * percent + ((window_.count % 100) * percent + 99) / 100;
    uint64_t cumulative = 0;
    for (std::size_t i = 0; i < buckets_.size(); ++i) {
        cumulative += buckets_[i];
        if (cumulative >= rank) {
            return std::min(BucketUpperBound(i), window_.max_ns);
        }
    }
    return window_.max_ns;
}

ReuseIntervalStats ReuseInterval::Snapshot() const {
    auto result = window_;
    result.avg_ns = result.count > 0 ? static_cast<double>(total_ns_ / result.count) : 0.0;
    result.p95_ns = Percentile(95);
    result.p99_ns = Percentile(99);
    return result;
}

ReuseIntervalStats ReuseInterval::Take() {
    const auto result = Snapshot();
    Reset();
    return result;
}

void ReuseInterval::Reset() {
    buckets_.fill(0);
    window_ = {};
    total_ns_ = 0.0L;
}

} // namespace kv_cache_manager
