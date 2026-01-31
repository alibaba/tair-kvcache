#pragma once

#include <memory>

namespace kv_cache_manager {

class CacheManager;
class MetricsRegistry;
class MetricsReporter;

class MetricsReporterFactory {
public:
    bool Init(std::shared_ptr<CacheManager> cache_manager, std::shared_ptr<MetricsRegistry> metrics_registry);
    [[nodiscard]] std::shared_ptr<MetricsReporter> Create(const std::string &type, const std::string &config) const;

private:
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

} // namespace kv_cache_manager
