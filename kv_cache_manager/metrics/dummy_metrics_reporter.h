#pragma once

#include "kv_cache_manager/metrics/metrics_reporter.h"

namespace kv_cache_manager {

class DummyMetricsReporter final : public MetricsReporter {
public:
    bool Init(std::shared_ptr<CacheManager> cache_manager,
              std::shared_ptr<MetricsRegistry> metrics_registry,
              const std::string &config) override {
        return true;
    }
    void ReportPerQuery(MetricsCollector *collector) override {}
    void ReportInterval() override {}
    std::shared_ptr<MetricsRegistry> GetMetricsRegistry() override { return nullptr; }
};

} // namespace kv_cache_manager
