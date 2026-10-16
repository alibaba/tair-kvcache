#pragma once

#include <memory>
#include <string>

#include "kv_cache_manager/metrics/local_metrics_reporter.h"

namespace kv_cache_manager {

class LoggingMetricsReporter final : public LocalMetricsReporter {
public:
    bool Init(std::shared_ptr<CacheManager> cache_manager,
              std::shared_ptr<MetricsRegistry> metrics_registry,
              const std::string &config,
              const MetricsTags &common_tags = {}) override;
    void ReportPerQuery(MetricsCollector *collector, std::optional<bool> request_failed = std::nullopt) override;
    void ReportInterval() override;
};

} // namespace kv_cache_manager
