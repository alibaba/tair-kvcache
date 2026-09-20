#include "kv_cache_manager/metrics/delete_cleanup_observer.h"

#include <array>
#include <cinttypes>
#include <exception>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

const char *DeleteCleanupFailureStageName(DeleteCleanupFailureStage stage) noexcept {
    switch (stage) {
    case DeleteCleanupFailureStage::kPhysicalDelete:
        return "physical_delete";
    case DeleteCleanupFailureStage::kMetadataCad:
        return "metadata_cad";
    case DeleteCleanupFailureStage::kAuthoritativeFence:
        return "authoritative_fence";
    case DeleteCleanupFailureStage::kDispatchOrWorker:
        return "dispatch_or_worker";
    }
    return "unknown";
}

void InitializeDeleteCleanupMetrics(const std::shared_ptr<MetricsRegistry> &metrics_registry) noexcept {
    if (!metrics_registry) {
        return;
    }
    constexpr std::array stages{DeleteCleanupFailureStage::kPhysicalDelete,
                                DeleteCleanupFailureStage::kMetadataCad,
                                DeleteCleanupFailureStage::kAuthoritativeFence,
                                DeleteCleanupFailureStage::kDispatchOrWorker};
    try {
        for (const auto stage : stages) {
            // Counter += 0 intentionally marks the series as touched so both
            // Prometheus and KMonitor publish a zero baseline.
            metrics_registry->GetCounter(kDeleteCleanupPermanentFailureMetricName,
                                         {{"stage", DeleteCleanupFailureStageName(stage)}}) += 0;
        }
    } catch (const std::exception &) {
        KVCM_LOG_ERROR("failed to initialize terminal delete cleanup metrics");
    } catch (...) { KVCM_LOG_ERROR("failed to initialize terminal delete cleanup metrics"); }
}

void RecordDeleteCleanupPermanentFailure(const std::shared_ptr<MetricsRegistry> &metrics_registry,
                                         DeleteCleanupFailureStage stage,
                                         std::uint64_t affected_location_count) noexcept {
    if (affected_location_count == 0) {
        return;
    }

    const char *stage_name = DeleteCleanupFailureStageName(stage);
    KVCM_LOG_ERROR("terminal delete cleanup failure: stage[%s], affected_location_count[%" PRIu64
                   "]; manual reconciliation is required and automatic physical retry remains disabled",
                   stage_name,
                   affected_location_count);

    if (!metrics_registry) {
        return;
    }
    try {
        metrics_registry->GetCounter(kDeleteCleanupPermanentFailureMetricName, {{"stage", stage_name}}) +=
            affected_location_count;
    } catch (const std::exception &) {
        KVCM_LOG_ERROR("failed to record terminal delete cleanup metric for stage[%s]", stage_name);
    } catch (...) { KVCM_LOG_ERROR("failed to record terminal delete cleanup metric for stage[%s]", stage_name); }
}

} // namespace kv_cache_manager
