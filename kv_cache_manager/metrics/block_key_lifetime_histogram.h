#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

// Completed block-key lifetimes. The sum counter stores microseconds so the
// existing Prometheus exporter can emit a standard seconds histogram.
class BlockKeyLifetimeHistogram {
public:
    bool Init(std::shared_ptr<MetricsRegistry> registry,
              const std::vector<double> &boundaries,
              const std::string &instance_group,
              const std::string &instance_id);
    void ObserveMs(uint64_t lifetime_ms) noexcept;

    uint64_t GetCount() const { return count_counter_.Get(); }
    uint64_t GetSumUs() const { return sum_counter_.Get(); }
    std::vector<uint64_t> GetBucketCounts() const;

private:
    std::vector<double> boundaries_;
    std::vector<Counter> bucket_counters_;
    Counter sum_counter_;
    Counter count_counter_;
};

} // namespace kv_cache_manager
