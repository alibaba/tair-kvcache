#include "kv_cache_manager/metrics/metrics_reporter_factory.h"

#include <memory>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/dummy_metrics_reporter.h"
#include "kv_cache_manager/metrics/kmonitor_metrics_reporter.h"
#include "kv_cache_manager/metrics/local_metrics_reporter.h"
#include "kv_cache_manager/metrics/logging_metrics_reporter.h"

namespace kv_cache_manager {

bool MetricsReporterFactory::Init(std::shared_ptr<CacheManager> cache_manager,
                                  std::shared_ptr<MetricsRegistry> metrics_registry) {
    cache_manager_ = std::move(cache_manager);
    metrics_registry_ = std::move(metrics_registry);
    return true;
}

std::shared_ptr<MetricsReporter> MetricsReporterFactory::Create(const std::string &type,
                                                                const std::string &config) const {
    KVCM_LOG_INFO("creating metrics reporter with type: %s", type.c_str());
    if (type == "kmonitor") {
        auto reporter = std::make_shared<KmonitorMetricsReporter>();
        reporter->Init(cache_manager_, metrics_registry_, config);
        return reporter;
    } else if (type == "local") {
        auto reporter = std::make_shared<LocalMetricsReporter>();
        reporter->Init(cache_manager_, metrics_registry_, config);
        return reporter;
    } else if (type == "logging") {
        auto reporter = std::make_shared<LoggingMetricsReporter>();
        reporter->Init(cache_manager_, metrics_registry_, config);
        return reporter;
    } else if (type == "dummy") {
        auto reporter = std::make_shared<DummyMetricsReporter>();
        reporter->Init(cache_manager_, metrics_registry_, config);
        return reporter;
    }
    KVCM_LOG_INFO("metrics reporter type [%s] invalid, use logging reporter.", type.c_str());
    auto reporter = std::make_shared<LoggingMetricsReporter>();
    reporter->Init(cache_manager_, metrics_registry_, config);
    return reporter;
}

} // namespace kv_cache_manager
