#pragma once

#include <memory>
#include <optional>
#include <string>

#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

class CacheManager;
class MetricsCollector;
class MetricsRegistry;

class MetricsReporter {
public:
    virtual ~MetricsReporter() = default;
    virtual bool Init(std::shared_ptr<CacheManager> cache_manager,
                      std::shared_ptr<MetricsRegistry> metrics_registry,
                      const std::string &config,
                      const MetricsTags &common_tags = {}) = 0;
    virtual void ReportPerQuery(MetricsCollector *collector, std::optional<bool> request_failed = std::nullopt) = 0;
    virtual void ReportInterval() = 0;
    virtual std::shared_ptr<MetricsRegistry> GetMetricsRegistry() = 0;
    const MetricsTags &GetCommonTags() const { return common_tags_; }

protected:
    // Process-level tags resolved at startup. Each backend decides how to
    // export them; they never participate in MetricsRegistry series identity.
    MetricsTags common_tags_;
};

} // namespace kv_cache_manager
