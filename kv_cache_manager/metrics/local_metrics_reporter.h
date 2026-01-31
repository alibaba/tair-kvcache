#pragma once

#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/metrics_reporter.h"

namespace kv_cache_manager {

class LocalMetricsReporter : public MetricsReporter {
public:
    bool Init(std::shared_ptr<CacheManager> cache_manager,
              std::shared_ptr<MetricsRegistry> metrics_registry,
              const std::string &config) override;
    void ReportPerQuery(MetricsCollector *collector) override;
    void ReportInterval() override;
    std::shared_ptr<MetricsRegistry> GetMetricsRegistry() override;

protected:
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;

    // for intervallic reporting
    KVCM_DECLARE_METRICS_COLLECTOR_(MetaIndexerAccumulative);
    MetricsCollectors data_storage_interval_metrics_collectors_;
    KVCM_DECLARE_METRICS_COLLECTOR_(CacheManager);
    MetricsCollectors cache_manager_group_interval_metrics_collectors_;
    MetricsCollectors cache_manager_instance_interval_metrics_collectors_;
};

} // namespace kv_cache_manager
