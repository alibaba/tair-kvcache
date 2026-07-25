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

namespace {

enum class MetricsReporterType {
    kDummy,
    kLocal,
    kLogging,
    kKmonitor,
    kUnsupported,
};

MetricsReporterType ParseMetricsReporterType(const std::string &type) {
    if (type.empty() || type == "local") {
        return MetricsReporterType::kLocal;
    }
    if (type == "logging") {
        return MetricsReporterType::kLogging;
    }
    if (type == "dummy") {
        return MetricsReporterType::kDummy;
    }
    if (type == "kmonitor") {
        return MetricsReporterType::kKmonitor;
    }
    return MetricsReporterType::kUnsupported;
}

} // namespace

bool MetricsReporterFactory::IsSupportedType(const std::string &type) {
    return ParseMetricsReporterType(type) != MetricsReporterType::kUnsupported;
}

const char *MetricsReporterFactory::SupportedTypes() { return "dummy, local, logging, kmonitor"; }

bool MetricsReporterFactory::Init(std::shared_ptr<CacheManager> cache_manager,
                                  std::shared_ptr<MetricsRegistry> metrics_registry) {
    cache_manager_ = std::move(cache_manager);
    metrics_registry_ = std::move(metrics_registry);
    return true;
}

std::shared_ptr<MetricsReporter> MetricsReporterFactory::Create(const std::string &type,
                                                                const std::string &config) const {
    const auto reporter_type = ParseMetricsReporterType(type);
    if (reporter_type == MetricsReporterType::kUnsupported) {
        KVCM_LOG_ERROR("unsupported metrics reporter type [%s], supported types: %s", type.c_str(), SupportedTypes());
        return nullptr;
    }

    KVCM_LOG_INFO("creating metrics reporter with type: %s", type.empty() ? "local (default)" : type.c_str());
    if (reporter_type == MetricsReporterType::kKmonitor) {
        auto reporter = std::make_shared<KmonitorMetricsReporter>();
        reporter->Init(cache_manager_, metrics_registry_, config);
        return reporter;
    } else if (reporter_type == MetricsReporterType::kLocal) {
        auto reporter = std::make_shared<LocalMetricsReporter>();
        reporter->Init(cache_manager_, metrics_registry_, config);
        return reporter;
    } else if (reporter_type == MetricsReporterType::kLogging) {
        auto reporter = std::make_shared<LoggingMetricsReporter>();
        reporter->Init(cache_manager_, metrics_registry_, config);
        return reporter;
    } else if (reporter_type == MetricsReporterType::kDummy) {
        auto reporter = std::make_shared<DummyMetricsReporter>();
        reporter->Init(cache_manager_, metrics_registry_, config);
        return reporter;
    }
    return nullptr;
}

} // namespace kv_cache_manager
