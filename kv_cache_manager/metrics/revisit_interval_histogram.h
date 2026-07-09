#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

// Prometheus-compatible cumulative histogram implemented using Counter.
//
// Each histogram instance maintains N+2 counters:
//   - N bucket counters (cumulative: bucket[i] counts observations <= boundary[i])
//   - 1 sum counter (total of all observed values, in microseconds)
//   - 1 count counter (total number of observations)
//
// The sum is stored in microseconds for uint64 precision. PrometheusExporter
// converts to seconds on output by dividing by 1e6.
//
// All counters are registered with an instance_id tag for per-instance isolation.
// Observe() uses only atomic increments (~10ns per call), no locks or allocations.
class RevisitIntervalHistogram {
public:
    RevisitIntervalHistogram() = default;
    ~RevisitIntervalHistogram() = default;

    // Non-copyable, non-movable
    RevisitIntervalHistogram(const RevisitIntervalHistogram &) = delete;
    RevisitIntervalHistogram &operator=(const RevisitIntervalHistogram &) = delete;
    RevisitIntervalHistogram(RevisitIntervalHistogram &&) = delete;
    RevisitIntervalHistogram &operator=(RevisitIntervalHistogram &&) = delete;

    // Initialize histogram with bucket boundaries and instance_id.
    // boundaries: sorted vector of positive bucket boundaries in seconds (ascending order).
    // instance_id: tag value for per-instance isolation.
    // Returns true on success, false on invalid parameters.
    bool Init(std::shared_ptr<MetricsRegistry> registry,
              const std::vector<double> &boundaries,
              const std::string &instance_id);

    // Record an observation (interval in microseconds).
    // Increments all bucket counters where boundary >= interval_seconds.
    // Increments sum counter by interval_us (stored in microseconds).
    // Increments count counter by 1.
    // Thread-safe: uses only atomic operations.
    void Observe(int64_t interval_us);

    // Get bucket boundaries (for testing/debugging).
    const std::vector<double> &GetBoundaries() const { return boundaries_; }

    // Get bucket counter values (for testing/debugging).
    std::vector<uint64_t> GetBucketCounts() const;

    // Get sum and count (for testing/debugging).
    uint64_t GetSum() const;
    uint64_t GetCount() const;

private:
    std::vector<double> boundaries_;       // bucket boundaries in seconds
    std::vector<Counter> bucket_counters_; // cumulative counters
    Counter sum_counter_;                  // sum of all observations (microseconds)
    Counter count_counter_;                // total number of observations
};

} // namespace kv_cache_manager
