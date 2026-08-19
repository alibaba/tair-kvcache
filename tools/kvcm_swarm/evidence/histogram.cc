#include "tools/kvcm_swarm/evidence/histogram.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace kvcm_swarm {
namespace {
// Bucket 0 covers [0, 0.01ms); every later bucket multiplies the bound by
// 2^(1/4), so the relative resolution is ~19% up to ~10 minutes.
constexpr double kFirstBound = 0.01;
const double kGrowth = std::pow(2.0, 0.25);
} // namespace

double Histogram::BucketUpperBoundMs(size_t index) {
    if (index + 1 >= kBucketCount) {
        return std::numeric_limits<double>::infinity();
    }
    return kFirstBound * std::pow(kGrowth, static_cast<double>(index));
}

size_t Histogram::BucketFor(double value_ms) {
    if (!(value_ms > 0.0)) {
        return 0;
    }
    if (value_ms < kFirstBound) {
        return 0;
    }
    const double ratio = std::log(value_ms / kFirstBound) / std::log(kGrowth);
    const auto index = static_cast<size_t>(std::floor(ratio)) + 1;
    return std::min(index, kBucketCount - 1);
}

void Histogram::Add(double value_ms) {
    if (std::isnan(value_ms)) {
        return;
    }
    const double clamped = value_ms < 0.0 ? 0.0 : value_ms;
    if (count_ == 0 || clamped < min_ms_) {
        min_ms_ = clamped;
    }
    if (clamped > max_ms_) {
        max_ms_ = clamped;
    }
    ++count_;
    sum_ms_ += clamped;
    ++buckets_[BucketFor(clamped)];
}

void Histogram::Merge(const Histogram &other) {
    if (other.count_ == 0) {
        return;
    }
    if (count_ == 0 || other.min_ms_ < min_ms_) {
        min_ms_ = other.min_ms_;
    }
    max_ms_ = std::max(max_ms_, other.max_ms_);
    count_ += other.count_;
    sum_ms_ += other.sum_ms_;
    for (size_t i = 0; i < kBucketCount; ++i) {
        buckets_[i] += other.buckets_[i];
    }
}

double Histogram::Quantile(double q) const {
    if (count_ == 0) {
        return 0.0;
    }
    const double clamped_q = std::min(1.0, std::max(0.0, q));
    const auto target = static_cast<uint64_t>(std::ceil(clamped_q * static_cast<double>(count_)));
    uint64_t cumulative = 0;
    for (size_t i = 0; i < kBucketCount; ++i) {
        cumulative += buckets_[i];
        if (cumulative >= std::max<uint64_t>(1, target)) {
            const double upper = BucketUpperBoundMs(i);
            if (std::isinf(upper)) {
                return max_ms_;
            }
            return std::min(upper, max_ms_);
        }
    }
    return max_ms_;
}

} // namespace kvcm_swarm
