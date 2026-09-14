#pragma once

#include <memory>
#include <string>

#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

class MetricsCollector;
class OnlineOptimizerManager;
class OptimizerKmonitorMetricsReporter;
struct InstanceSummary;

class OptimizerMetricsReporter {
public:
    OptimizerMetricsReporter(std::shared_ptr<OnlineOptimizerManager> manager,
                             std::shared_ptr<MetricsRegistry> metrics_registry,
                             std::shared_ptr<OptimizerKmonitorMetricsReporter> kmonitor_reporter = nullptr);
    ~OptimizerMetricsReporter();

    OptimizerMetricsReporter(const OptimizerMetricsReporter &) = delete;
    OptimizerMetricsReporter &operator=(const OptimizerMetricsReporter &) = delete;

    void ReportInterval();
    void ReportPerQuery(MetricsCollector *collector);
    void RemoveInstanceMetrics(const std::string &instance_id);

private:
    std::shared_ptr<OnlineOptimizerManager> manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<OptimizerKmonitorMetricsReporter> kmonitor_reporter_;
};

} // namespace kv_cache_manager
