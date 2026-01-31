#include "kv_cache_manager/metrics/logging_metrics_reporter.h"

#include <sstream>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

bool LoggingMetricsReporter::Init(std::shared_ptr<CacheManager> cache_manager,
                                  std::shared_ptr<MetricsRegistry> metrics_registry,
                                  const std::string &config) {
    return LocalMetricsReporter::Init(cache_manager, metrics_registry, config);
}

void LoggingMetricsReporter::ReportPerQuery(MetricsCollector *collector) {
    LocalMetricsReporter::ReportPerQuery(collector);

    if (metrics_registry_ == nullptr) {
        return;
    }

    std::vector<MetricsRegistry::metrics_tuple_t> all_metrics;
    metrics_registry_->GetAllMetrics(all_metrics);

    std::ostringstream ss;
    ss << "[Report_Query_Metrics_By_Print]|";
    for (const auto &[name, tags, val] : all_metrics) {
        // name
        ss << "[" << name << "]:";

        // tags
        ss << "[";
        for (const auto &[tag_name, tag_value] : tags) {
            ss << "{" << tag_name << ":" << tag_value << "}";
        }
        ss << "]:";

        // value
        if (std::holds_alternative<CounterValue>(*val)) {
            ss << std::get<CounterValue>(*val).load();
        } else if (std::holds_alternative<GaugeValue>(*val)) {
            ss << std::get<GaugeValue>(*val).load();
        } else {
            ss << "<unknown_value_type>";
        }

        ss << "|";
    }

    // TODO 要使用专门的logger防止截断
     KVCM_METRICS_LOG(ss.str());
}

void LoggingMetricsReporter::ReportInterval() {
    LocalMetricsReporter::ReportInterval();

    if (metrics_registry_ == nullptr) {
        return;
    }

    std::vector<MetricsRegistry::metrics_tuple_t> all_metrics;
    metrics_registry_->GetAllMetrics(all_metrics);

    std::ostringstream ss;
    ss << "[Report_Interval_By_Print]|";
    for (const auto &[name, tags, val] : all_metrics) {
        // name
        ss << "[" << name << "]:";

        // tags
        ss << "[";
        for (const auto &[tag_name, tag_value] : tags) {
            ss << "{" << tag_name << ":" << tag_value << "}";
        }
        ss << "]:";

        // value
        if (std::holds_alternative<CounterValue>(*val)) {
                ss << std::get<CounterValue>(*val).load();
        } else if (std::holds_alternative<GaugeValue>(*val)) {
                ss << std::get<GaugeValue>(*val).load();
        } else {
            ss << "<unknown_value_type>";
        }

        ss << "|";
    }

    // TODO 要使用专门的logger防止截断
    KVCM_METRICS_LOG(ss.str());
}

} // namespace kv_cache_manager
