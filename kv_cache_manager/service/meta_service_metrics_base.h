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

#ifndef KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_
#define KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_(name, instance_id)                                                      \
    do {                                                                                                               \
        std::scoped_lock guard(mutex_##name##_);                                                                       \
        KVCM_METRICS_COLLECTOR_MAP_(name).erase(instance_id);                                                          \
    } while (0)
#endif

class RegistryManager;
struct MetricsLifecycle;

class MetaServiceMetricsBase {
public:
    explicit MetaServiceMetricsBase(std::shared_ptr<MetricsRegistry> metrics_registry,
                                    std::shared_ptr<RegistryManager> registry_manager,
                                    std::shared_ptr<MetricsLifecycle> metrics_lifecycle = nullptr);
    void InitMetrics();

    // evict cached per-instance collectors so that purged registry
    // entries cannot be resurrected by stale handles
    void InvalidateCollectorCache(const std::string &instance_id);

    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(GetCacheMeta);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(GetCacheLocation);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(GetCacheLocationsByBackend);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(GetCacheLocationLen);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(StartWriteCache);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(FinishWriteCache);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(RemoveCache);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(TrimCache);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(GetClusterInfo);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(ReportEvent);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(EventBlockAdd);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(EventBlockDelete);

protected:
    KVCM_DECLARE_METRICS_COLLECTOR_(RegisterInstance);
    KVCM_DECLARE_METRICS_COLLECTOR_(GetInstanceInfo);
    KVCM_DECLARE_METRICS_COLLECTOR_(GetClusterInfo);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(GetCacheMeta);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(GetCacheLocation);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(GetCacheLocationsByBackend);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(GetCacheLocationLen);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(StartWriteCache);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(FinishWriteCache);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(RemoveCache);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(TrimCache);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(GetClusterInfo);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(ReportEvent);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(EventBlockAdd);
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(EventBlockDelete);

private:
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    // shared coarse-grained lock that excludes RemoveInstance /
    // RemoveInstanceGroup; held in shared mode while the slow-path
    // macro creates a new ServiceMetricsCollector so that the new
    // tagged entry cannot be registered concurrently with a purge
    std::shared_ptr<MetricsLifecycle> metrics_lifecycle_;
};

} // namespace kv_cache_manager