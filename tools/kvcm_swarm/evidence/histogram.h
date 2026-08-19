// Bounded latency histogram with logarithmic buckets.
//
// Quantiles are approximate but the bucket layout is fixed and reported, so
// the evaluator can reason about the resolution of any threshold it applies.
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace kvcm_swarm {

class Histogram {
public:
    static constexpr size_t kBucketCount = 64;

    void Add(double value_ms);
    void Merge(const Histogram &other);

    uint64_t count() const { return count_; }
    double sum_ms() const { return sum_ms_; }
    double min_ms() const { return count_ == 0 ? 0.0 : min_ms_; }
    double max_ms() const { return max_ms_; }
    double mean_ms() const { return count_ == 0 ? 0.0 : sum_ms_ / static_cast<double>(count_); }
    double Quantile(double q) const;

    // Bucket upper bound in milliseconds, exclusive.
    static double BucketUpperBoundMs(size_t index);

private:
    static size_t BucketFor(double value_ms);

    std::array<uint64_t, kBucketCount> buckets_{};
    uint64_t count_ = 0;
    double sum_ms_ = 0.0;
    double min_ms_ = 0.0;
    double max_ms_ = 0.0;
};

} // namespace kvcm_swarm
