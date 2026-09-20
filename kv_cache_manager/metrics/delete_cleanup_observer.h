#pragma once

#include <cstdint>
#include <memory>

namespace kv_cache_manager {

class MetricsRegistry;

// A deliberately closed set: these values are used as metric labels, so
// callers cannot introduce URI/key/free-text cardinality.
enum class DeleteCleanupFailureStage {
    kPhysicalDelete,
    kMetadataCad,
    kAuthoritativeFence,
    kDispatchOrWorker,
};

inline constexpr const char *kDeleteCleanupPermanentFailureMetricName =
    "cache_cleanup.permanent_failure_location_count";

const char *DeleteCleanupFailureStageName(DeleteCleanupFailureStage stage) noexcept;

// Creates and touches every fixed-label series at zero. This must run before
// serving traffic so Prometheus increase()/rate() has a pre-failure baseline
// even when the first cleanup failure is the only one in the process lifetime.
void InitializeDeleteCleanupMetrics(const std::shared_ptr<MetricsRegistry> &metrics_registry) noexcept;

// Records affected Location targets, not requests. This is a cumulative
// counter for terminal failures; it is intentionally not an automatic retry
// trigger. A matching ERROR log is emitted even when metrics are unavailable.
void RecordDeleteCleanupPermanentFailure(const std::shared_ptr<MetricsRegistry> &metrics_registry,
                                         DeleteCleanupFailureStage stage,
                                         std::uint64_t affected_location_count) noexcept;

} // namespace kv_cache_manager
