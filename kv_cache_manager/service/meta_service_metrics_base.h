#pragma once

#include <shared_mutex>
#include <string>
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

#ifndef KVCM_INVALIDATE_TYPED_METRICS_COLLECTOR_MAP_
#define KVCM_INVALIDATE_TYPED_METRICS_COLLECTOR_MAP_(name, instance_id)                                                \
    do {                                                                                                               \
        std::scoped_lock guard(mutex_##name##_);                                                                       \
        KVCM_METRICS_COLLECTOR_MAP_(name).erase(MakeTypedCollectorKey(instance_id, kEventReportL1P5MetricsType));      \
        KVCM_METRICS_COLLECTOR_MAP_(name).erase(MakeTypedCollectorKey(instance_id, kEventReportL2MetricsType));        \
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
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_METHOD_(GetHostCacheState);

protected:
    std::shared_ptr<MetricsCollector> GetTypedMetricsCollectorForReportEvent(const std::string &instance_id,
                                                                             const std::string &type);
    std::shared_ptr<MetricsCollector> GetTypedMetricsCollectorForEventBlockAdd(const std::string &instance_id,
                                                                               const std::string &type);
    std::shared_ptr<MetricsCollector> GetTypedMetricsCollectorForEventBlockDelete(const std::string &instance_id,
                                                                                  const std::string &type);

    static constexpr const char *kEventReportL1P5MetricsType = "event_report_l1p5";
    static constexpr const char *kEventReportL2MetricsType = "event_report_l2";

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
    KVCM_DECLARE_METRICS_COLLECTOR_MAP_(GetHostCacheState);

private:
    std::shared_ptr<MetricsCollector> GetMetricsCollectorFromMap(
        const std::string &api_name,
        std::unordered_map<std::string, std::shared_ptr<MetricsCollector>> &metrics_collector_map,
        std::shared_mutex &mutex,
        const std::string &instance_id,
        const std::string &collector_key,
        const MetricsTags &extra_tags);
    static std::string MakeTypedCollectorKey(const std::string &instance_id, const std::string &type);

    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    // shared coarse-grained lock that excludes RemoveInstance /
    // RemoveInstanceGroup; held in shared mode while the slow-path
    // macro creates a new ServiceMetricsCollector so that the new
    // tagged entry cannot be registered concurrently with a purge
    std::shared_ptr<MetricsLifecycle> metrics_lifecycle_;
};

} // namespace kv_cache_manager
