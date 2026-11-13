#pragma once

#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/metrics_reporter.h"

namespace kv_cache_manager {

class LocalMetricsReporter : public MetricsReporter {
public:
    bool Init(std::shared_ptr<CacheManager> cache_manager,
              std::shared_ptr<MetricsRegistry> metrics_registry,
              const std::string &config,
              const MetricsTags &common_tags = {}) override;
    void ReportPerQuery(MetricsCollector *collector, std::optional<bool> request_failed = std::nullopt) override;
    void ReportInterval() override;
    std::shared_ptr<MetricsRegistry> GetMetricsRegistry() override;

protected:
    // Publishes kvcm_is_leader. Does nothing until
    // a role snapshot is provided; callers must hold the metrics lifecycle
    // lock and guarantee cache_manager_ is non-null.
    void PublishNodeStatus(std::optional<bool> is_leader);

    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;

    // Reused by KMonitor so both outputs share one capacity eligibility sample per interval,
    // including when demotion has already removed the storage manager.
    bool report_storage_usage_ = true;

    // for intervallic reporting
    KVCM_DECLARE_METRICS_COLLECTOR_(MetaIndexerAccumulative);
    MetricsCollectors data_storage_interval_metrics_collectors_;
    KVCM_DECLARE_METRICS_COLLECTOR_(CacheManager);
    MetricsCollectors cache_manager_group_interval_metrics_collectors_;
    MetricsCollectors cache_manager_instance_interval_metrics_collectors_;
};

} // namespace kv_cache_manager
