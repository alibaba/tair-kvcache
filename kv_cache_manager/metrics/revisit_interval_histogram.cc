#include "kv_cache_manager/metrics/revisit_interval_histogram.h"

#include <algorithm>

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

    // Register bucket counters (N buckets + 1 for +Inf)
    bucket_counters_.resize(boundaries_.size() + 1);
    for (size_t i = 0; i < boundaries_.size(); ++i) {
        MetricsTags bucket_tags = tags;
        bucket_tags["le"] = std::to_string(boundaries_[i]);
        bucket_counters_[i] = registry->GetCounter("revisit_interval_seconds_bucket", bucket_tags);
    }
    // +Inf bucket
    MetricsTags inf_tags = tags;
    inf_tags["le"] = "+Inf";
    bucket_counters_[boundaries_.size()] = registry->GetCounter("revisit_interval_seconds_bucket", inf_tags);

    // Register sum and count counters
    sum_counter_ = registry->GetCounter("revisit_interval_seconds_sum", tags);
    count_counter_ = registry->GetCounter("revisit_interval_seconds_count", tags);

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
            bucket_counters_[i]++;
        }
    }
    // Always increment +Inf bucket
    bucket_counters_[boundaries_.size()]++;

    // Increment sum (in milliseconds) and count
    uint64_t interval_ms = static_cast<uint64_t>(interval_us / 1000);
    sum_counter_ += interval_ms;
    count_counter_++;
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
