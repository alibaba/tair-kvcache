#pragma once

#include <memory>
#include <string>

namespace kv_cache_manager {

class CacheManager;
class MetricsRegistry;
class MetricsReporter;

class MetricsReporterFactory {
public:
    // Empty type is valid and resolves to the default local reporter.
    static bool IsSupportedType(const std::string &type);
    static const char *SupportedTypes();

    bool Init(std::shared_ptr<CacheManager> cache_manager, std::shared_ptr<MetricsRegistry> metrics_registry);
    [[nodiscard]] std::shared_ptr<MetricsReporter> Create(const std::string &type, const std::string &config) const;

private:
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

} // namespace kv_cache_manager
