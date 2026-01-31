#pragma once

#include <memory>
#include <string>

#include "kv_cache_manager/metrics/local_metrics_reporter.h"

namespace kv_cache_manager {

class LoggingMetricsReporter final : public LocalMetricsReporter {
public:
    bool Init(std::shared_ptr<CacheManager> cache_manager,
              std::shared_ptr<MetricsRegistry> metrics_registry,
              const std::string &config) override;
    void ReportPerQuery(MetricsCollector *collector) override;
    void ReportInterval() override;
};

} // namespace kv_cache_manager
