#pragma once

#include <functional>
#include <memory>

#include "kv_cache_manager/metrics/local_metrics_reporter.h"

namespace kv_cache_manager {

class KmonitorMetricsReporter final : public LocalMetricsReporter {
public:
    KmonitorMetricsReporter();
    ~KmonitorMetricsReporter();
    bool Init(std::shared_ptr<CacheManager> cache_manager,
              std::shared_ptr<MetricsRegistry> metrics_registry,
              const std::string &config) override;
    void ReportPerQuery(MetricsCollector *collector) override;
    void ReportInterval() override;

private:
    using MetricSampleConsumer = std::function<void(const MetricsTags &, double)>;

    bool InitMetrics();
    void VisitMetricSamples(const char *name, const MetricSampleConsumer &consumer) const;

    struct Context;
    std::unique_ptr<Context> ctx_;
};

} // namespace kv_cache_manager
