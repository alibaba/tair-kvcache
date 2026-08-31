#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

struct InstanceSummary;
struct MrcMetricInfo;
class OptimizerServiceMetricsCollector;

class OptimizerKmonitorMetricsReporter {
public:
    explicit OptimizerKmonitorMetricsReporter(std::string prefix);
    ~OptimizerKmonitorMetricsReporter();

    OptimizerKmonitorMetricsReporter(const OptimizerKmonitorMetricsReporter &) = delete;
    OptimizerKmonitorMetricsReporter &operator=(const OptimizerKmonitorMetricsReporter &) = delete;

    bool Init();
    void Shutdown();

    void ReportPerQuery(OptimizerServiceMetricsCollector *collector,
                        const MetricsTags &service_tags,
                        const MetricsTags &query_tags);
    void ReportInterval(const std::vector<InstanceSummary> &summaries, const std::vector<MrcMetricInfo> &mrc_metrics);

private:
    bool InitMetrics();

    std::string prefix_;
    struct KmonContext;
    std::unique_ptr<KmonContext> kmon_ctx_;
};

} // namespace kv_cache_manager
