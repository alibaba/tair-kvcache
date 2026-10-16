#pragma once

#include "kv_cache_manager/metrics/metrics_reporter.h"

namespace kv_cache_manager {

class DummyMetricsReporter final : public MetricsReporter {
public:
    bool Init(std::shared_ptr<CacheManager> cache_manager,
              std::shared_ptr<MetricsRegistry> metrics_registry,
              const std::string &config,
              const MetricsTags &common_tags = {}) override {
        common_tags_ = common_tags;
        return true;
    }
    void ReportPerQuery(MetricsCollector *collector, std::optional<bool> request_failed = std::nullopt) override {}
    void ReportInterval() override {}
    std::shared_ptr<MetricsRegistry> GetMetricsRegistry() override { return nullptr; }
};

} // namespace kv_cache_manager
