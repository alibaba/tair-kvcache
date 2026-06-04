#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

class OnlineOptimizerManager;
class MetricsCollector;

class OptimizerMetricsReporter {
public:
    OptimizerMetricsReporter(std::shared_ptr<OnlineOptimizerManager> manager,
                             std::shared_ptr<MetricsRegistry> metrics_registry,
                             const std::string &prefix = "kvcm_optimizer");
    ~OptimizerMetricsReporter();

    OptimizerMetricsReporter(const OptimizerMetricsReporter &) = delete;
    OptimizerMetricsReporter &operator=(const OptimizerMetricsReporter &) = delete;

    bool InitKmonitor();
    void ShutdownKmonitor();

    void ReportInterval();

    void ReportPerQuery(MetricsCollector *collector);

    void RemoveInstanceMetrics(const std::string &instance_id);

private:
    bool InitMetrics();

    std::shared_ptr<OnlineOptimizerManager> manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::string prefix_;

    struct KmonContext;
    std::unique_ptr<KmonContext> kmon_ctx_;
};

} // namespace kv_cache_manager
