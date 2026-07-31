#include "kv_cache_manager/optimizer/service/metrics/optimizer_metrics_collector.h"

#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

#define DEFINE_METRICS_NAME_FOR_OPT_SERVICE(name) DEFINE_METRICS_NAME_(OptimizerServiceMetricsCollector, service, name)
#define REGISTER_COUNTER_METRICS_FOR_OPT_SERVICE(name)                                                                 \
    REGISTER_METRICS_W_TAGS_COUNTER_(metrics_registry_, service, name, metrics_tags_)
#define REGISTER_GAUGE_METRICS_FOR_OPT_SERVICE(name)                                                                   \
    REGISTER_METRICS_W_TAGS_GAUGE_(metrics_registry_, service, name, metrics_tags_)

DEFINE_METRICS_NAME_FOR_OPT_SERVICE(query_counter);
DEFINE_METRICS_NAME_FOR_OPT_SERVICE(query_rt_us);
DEFINE_METRICS_NAME_FOR_OPT_SERVICE(error_code);
DEFINE_METRICS_NAME_FOR_OPT_SERVICE(error_counter);

#undef DEFINE_METRICS_NAME_FOR_OPT_SERVICE

OptimizerServiceMetricsCollector::OptimizerServiceMetricsCollector(
    std::shared_ptr<MetricsRegistry> metrics_registry) noexcept
    : MetricsCollector(std::move(metrics_registry)) {}

OptimizerServiceMetricsCollector::OptimizerServiceMetricsCollector(std::shared_ptr<MetricsRegistry> metrics_registry,
                                                                   MetricsTags metrics_tags) noexcept
    : MetricsCollector(std::move(metrics_registry), std::move(metrics_tags)) {}

bool OptimizerServiceMetricsCollector::Init() {
    if (metrics_registry_ == nullptr) {
        return false;
    }

    REGISTER_COUNTER_METRICS_FOR_OPT_SERVICE(query_counter);
    REGISTER_GAUGE_METRICS_FOR_OPT_SERVICE(query_rt_us);
    REGISTER_GAUGE_METRICS_FOR_OPT_SERVICE(error_code);
    REGISTER_COUNTER_METRICS_FOR_OPT_SERVICE(error_counter);

    return true;
}

#undef REGISTER_COUNTER_METRICS_FOR_OPT_SERVICE
#undef REGISTER_GAUGE_METRICS_FOR_OPT_SERVICE

} // namespace kv_cache_manager
