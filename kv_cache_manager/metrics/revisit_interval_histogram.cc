#include "kv_cache_manager/metrics/revisit_interval_histogram.h"

#include <cmath>

namespace kv_cache_manager {

bool RevisitIntervalHistogram::Init(std::shared_ptr<MetricsRegistry> registry,
                                    const std::vector<double> &boundaries,
                                    const std::string &instance_id) {
    // Validate boundaries: non-empty, sorted, positive
    if (boundaries.empty()) {
        return false;
    }
    for (size_t i = 0; i < boundaries.size(); ++i) {
        if (boundaries[i] <= 0.0) {
            return false;
        }
        if (i > 0 && boundaries[i] <= boundaries[i - 1]) {
            return false;
        }
    }

    boundaries_ = boundaries;
    MetricsTags tags = {{"instance_id", instance_id}};

    // Register histogram family metadata (only once per registry, idempotent)
    static constexpr const char *kFamilyName = "revisit_interval_seconds";
    registry->RegisterHistogramFamily(kFamilyName);

    // Register bucket counters (N buckets + 1 for +Inf)
    bucket_counters_.resize(boundaries_.size() + 1);
    for (size_t i = 0; i < boundaries_.size(); ++i) {
        MetricsTags bucket_tags = tags;
        // Format "le" (less-or-equal) label — standard Prometheus histogram bucket tag.
        // Integers without decimals, non-integers as-is.
        double b = boundaries_[i];
        bucket_tags["le"] = (b == std::floor(b)) ? std::to_string(static_cast<int64_t>(b)) : std::to_string(b);
        bucket_counters_[i] = registry->GetCounter("revisit_interval_seconds_bucket", bucket_tags);
    }
    // +Inf bucket
    MetricsTags inf_tags = tags;
    inf_tags["le"] = "+Inf";
    bucket_counters_[boundaries_.size()] = registry->GetCounter("revisit_interval_seconds_bucket", inf_tags);

    // Register sum and count counters
    sum_counter_ = registry->GetCounter("revisit_interval_seconds_sum", tags);
    count_counter_ = registry->GetCounter("revisit_interval_seconds_count", tags);

    // Map metric names to family (idempotent, safe for concurrent Init calls)
    registry->MapMetricToFamily("revisit_interval_seconds_bucket", kFamilyName);
    registry->MapMetricToFamily("revisit_interval_seconds_sum", kFamilyName);
    registry->MapMetricToFamily("revisit_interval_seconds_count", kFamilyName);

    return true;
}

void RevisitIntervalHistogram::Observe(int64_t interval_us) {
    if (interval_us <= 0) {
        return;
    }

    // Convert microseconds to seconds
    double interval_s = static_cast<double>(interval_us) / 1e6;

    // Increment cumulative bucket counters
    // All buckets with boundary >= interval_s should be incremented
    for (size_t i = 0; i < boundaries_.size(); ++i) {
        if (boundaries_[i] >= interval_s) {
            ++bucket_counters_[i];
        }
    }
    // Always increment +Inf bucket
    ++bucket_counters_[boundaries_.size()];

    // Increment sum (stored in microseconds to preserve precision as uint64)
    // PrometheusExporter will convert to seconds on output
    sum_counter_ += static_cast<uint64_t>(interval_us);
    ++count_counter_;
}

void RevisitIntervalHistogram::ObserveBatch(const std::vector<int64_t> &intervals_us) {
    if (intervals_us.empty()) {
        return;
    }

    std::vector<uint64_t> bucket_deltas(boundaries_.size() + 1, 0);
    uint64_t sum_delta = 0;
    uint64_t count_delta = 0;
    for (const int64_t interval_us : intervals_us) {
        if (interval_us <= 0) {
            continue;
        }
        const double interval_s = static_cast<double>(interval_us) / 1e6;
        for (size_t i = 0; i < boundaries_.size(); ++i) {
            if (boundaries_[i] >= interval_s) {
                ++bucket_deltas[i];
            }
        }
        ++bucket_deltas.back();
        sum_delta += static_cast<uint64_t>(interval_us);
        ++count_delta;
    }

    if (count_delta == 0) {
        return;
    }
    for (size_t i = 0; i < bucket_deltas.size(); ++i) {
        if (bucket_deltas[i] != 0) {
            bucket_counters_[i] += bucket_deltas[i];
        }
    }
    sum_counter_ += sum_delta;
    count_counter_ += count_delta;
}

std::vector<uint64_t> RevisitIntervalHistogram::GetBucketCounts() const {
    std::vector<uint64_t> counts;
    counts.reserve(bucket_counters_.size());
    for (const auto &counter : bucket_counters_) {
        counts.push_back(counter.Get());
    }
    return counts;
}

uint64_t RevisitIntervalHistogram::GetSum() const { return sum_counter_.Get(); }

uint64_t RevisitIntervalHistogram::GetCount() const { return count_counter_.Get(); }

} // namespace kv_cache_manager
