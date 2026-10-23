#include "kv_cache_manager/metrics/block_key_lifetime_histogram.h"

#include <cmath>

namespace kv_cache_manager {

bool BlockKeyLifetimeHistogram::Init(std::shared_ptr<MetricsRegistry> registry,
                                     const std::vector<double> &boundaries,
                                     const std::string &instance_group,
                                     const std::string &instance_id) {
    if (!registry || boundaries.empty()) {
        return false;
    }
    for (size_t i = 0; i < boundaries.size(); ++i) {
        if (!std::isfinite(boundaries[i]) || boundaries[i] <= 0.0 || (i > 0 && boundaries[i] <= boundaries[i - 1])) {
            return false;
        }
    }
    boundaries_ = boundaries;
    MetricsTags tags = {{"instance_group", instance_group}, {"instance_id", instance_id}};
    static constexpr const char *kFamily = "block_key_lifetime_seconds";
    registry->RegisterHistogramFamily(kFamily);
    bucket_counters_.resize(boundaries_.size() + 1);
    for (size_t i = 0; i < boundaries_.size(); ++i) {
        MetricsTags bucket_tags = tags;
        const double boundary = boundaries_[i];
        bucket_tags["le"] = boundary == std::floor(boundary)
                                ? std::to_string(static_cast<int64_t>(boundary))
                                : std::to_string(boundary);
        bucket_counters_[i] = registry->GetCounter("block_key_lifetime_seconds_bucket", bucket_tags);
    }
    tags["le"] = "+Inf";
    bucket_counters_.back() = registry->GetCounter("block_key_lifetime_seconds_bucket", tags);
    tags.erase("le");
    sum_counter_ = registry->GetCounter("block_key_lifetime_seconds_sum", tags);
    count_counter_ = registry->GetCounter("block_key_lifetime_seconds_count", tags);
    registry->MapMetricToFamily("block_key_lifetime_seconds_bucket", kFamily);
    registry->MapMetricToFamily("block_key_lifetime_seconds_sum", kFamily);
    registry->MapMetricToFamily("block_key_lifetime_seconds_count", kFamily);
    return true;
}

void BlockKeyLifetimeHistogram::ObserveMs(uint64_t lifetime_ms) noexcept {
    const double lifetime_seconds = static_cast<double>(lifetime_ms) / 1000.0;
    for (size_t i = 0; i < boundaries_.size(); ++i) {
        if (lifetime_seconds <= boundaries_[i]) {
            ++bucket_counters_[i];
        }
    }
    ++bucket_counters_.back();
    sum_counter_ += lifetime_ms * 1000;
    ++count_counter_;
}

std::vector<uint64_t> BlockKeyLifetimeHistogram::GetBucketCounts() const {
    std::vector<uint64_t> counts;
    counts.reserve(bucket_counters_.size());
    for (const auto &bucket : bucket_counters_) {
        counts.push_back(bucket.Get());
    }
    return counts;
}

} // namespace kv_cache_manager
