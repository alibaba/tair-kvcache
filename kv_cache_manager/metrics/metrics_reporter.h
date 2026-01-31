#pragma once

#include <memory>
#include <string>

namespace kv_cache_manager {

class CacheManager;
class MetricsCollector;
class MetricsRegistry;

class MetricsReporter {
public:
    virtual ~MetricsReporter() = default;
    virtual bool Init(std::shared_ptr<CacheManager> cache_manager,
                      std::shared_ptr<MetricsRegistry> metrics_registry,
                      const std::string &config) = 0;
    virtual void ReportPerQuery(MetricsCollector *collector) = 0;
    virtual void ReportInterval() = 0;
    virtual std::shared_ptr<MetricsRegistry> GetMetricsRegistry() = 0;
};

} // namespace kv_cache_manager
