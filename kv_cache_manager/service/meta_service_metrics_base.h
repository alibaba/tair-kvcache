#pragma once

#include <shared_mutex>
#include <unordered_map>

#include "kv_cache_manager/metrics/metrics_collector.h"

namespace kv_cache_manager {

#ifndef KVCM_METRICS_COLLECTOR_MAP_
#define KVCM_METRICS_COLLECTOR_MAP_(name) metrics_collector_map_for_##name##_
#endif

#ifndef KVCM_DECLARE_METRICS_COLLECTOR_MAP_
#define KVCM_DECLARE_METRICS_COLLECTOR_MAP_(name)                                                                      \
    std::unordered_map<std::string, std::shared_ptr<MetricsCollector>> KVCM_METRICS_COLLECTOR_MAP_(name)
#endif

#ifndef KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_
#define KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(name)                                                               \
protected:                                                                                                             \
    std::shared_ptr<MetricsCollector> get_metrics_collector_from_map_for_##name(const std::string &instance_id);       \
                                                                                                                       \
private:                                                                                                               \
    std::shared_mutex mutex_##name##_
#endif

class RegistryManager;

class MetaServiceMetricsBase {
public:
    explicit MetaServiceMetricsBase(std::shared_ptr<MetricsRegistry> metrics_registry,
                                    std::shared_ptr<RegistryManager> registry_manager);
    void InitMetrics();

    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(GetCacheMeta);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(GetCacheLocation);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(StartWriteCache);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(FinishWriteCache);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(RemoveCache);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(TrimCache);

protected:
    KVCM_DECLARE_METRICS_COLLECTOR_(RegisterInstance);
    KVCM_DECLARE_METRICS_COLLECTOR_(GetInstanceInfo);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(GetCacheMeta);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(GetCacheLocation);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(StartWriteCache);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(FinishWriteCache);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(RemoveCache);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(TrimCache);

private:
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
};

} // namespace kv_cache_manager