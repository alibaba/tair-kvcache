#include "kv_cache_manager/service/meta_service_metrics_base.h"

#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/metrics/metrics_lifecycle.h"
#include "kv_cache_manager/service/util/common.h"

#ifndef KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_
#define KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(name)                                                                \
    std::shared_ptr<MetricsCollector> MetaServiceMetricsBase::get_metrics_collector_from_map_for_##name(               \
        const std::string &instance_id) {                                                                              \
        {                                                                                                              \
            std::shared_lock read_guard(mutex_##name##_);                                                              \
            auto iter = KVCM_METRICS_COLLECTOR_MAP_(name).find(instance_id);                                           \
            if (iter != KVCM_METRICS_COLLECTOR_MAP_(name).end()) {                                                     \
                return iter->second;                                                                                   \
            }                                                                                                          \
        }                                                                                                              \
        {                                                                                                              \
            /* hold a shared lifecycle lock around the slow path so   */                                               \
            /* the tagged ServiceMetricsCollector below cannot be     */                                               \
            /* registered concurrently with a unique-locked           */                                               \
            /* RemoveInstance / RemoveInstanceGroup tag-filter purge. */                                               \
            std::shared_lock<std::shared_mutex> lifecycle_guard(metrics_lifecycle_->mut_);                             \
            std::scoped_lock write_guard(mutex_##name##_);                                                             \
                                                                                                                       \
            auto iter = KVCM_METRICS_COLLECTOR_MAP_(name).find(instance_id);                                           \
            if (iter != KVCM_METRICS_COLLECTOR_MAP_(name).end()) {                                                     \
                return iter->second;                                                                                   \
            }                                                                                                          \
                                                                                                                       \
            /* the GetInstanceGroupName check is independently useful: */                                              \
            /* it prevents creating collectors for instances that were */                                              \
            /* never registered, so it stays even though the lifecycle */                                              \
            /* lock alone would already exclude concurrent removals.   */                                              \
            auto instance_group = registry_manager_->GetInstanceGroupName(instance_id);                                \
            if (instance_group.empty()) {                                                                              \
                return nullptr;                                                                                        \
            }                                                                                                          \
                                                                                                                       \
            auto metrics_collector = std::make_shared<ServiceMetricsCollector>(                                        \
                metrics_registry_,                                                                                     \
                MetricsTags({{"api_name", #name}, {"instance_group", instance_group}, {"instance_id", instance_id}})); \
            if (!metrics_collector->Init()) {                                                                          \
                return nullptr;                                                                                        \
            }                                                                                                          \
            KVCM_METRICS_COLLECTOR_MAP_(name)[instance_id] = metrics_collector;                                        \
            return metrics_collector;                                                                                  \
        }                                                                                                              \
    }

#endif

namespace kv_cache_manager {

MetaServiceMetricsBase::MetaServiceMetricsBase(std::shared_ptr<MetricsRegistry> metrics_registry,
                                               std::shared_ptr<RegistryManager> registry_manager,
                                               std::shared_ptr<MetricsLifecycle> metrics_lifecycle)
    : metrics_registry_(std::move(metrics_registry))
    , registry_manager_(std::move(registry_manager))
    , metrics_lifecycle_(metrics_lifecycle ? std::move(metrics_lifecycle) : std::make_shared<MetricsLifecycle>()) {}

void MetaServiceMetricsBase::InitMetrics() {
    MAKE_SERVICE_METRICS_COLLECTOR(RegisterInstance);
    MAKE_SERVICE_METRICS_COLLECTOR(GetInstanceInfo);
    MAKE_SERVICE_METRICS_COLLECTOR(GetClusterInfo);
    // GetClusterInfo 的全局 collector 也预置到 MAP 中，以空 instance_id 为 key
    KVCM_METRICS_COLLECTOR_MAP_(GetClusterInfo)[""] = KVCM_METRICS_COLLECTOR_(GetClusterInfo);
}

void MetaServiceMetricsBase::InvalidateCollectorCache(const std::string &instance_id) {
    // guard against empty instance_id which would wipe the global
    // GetClusterInfo collector seeded at the empty-string key
    if (instance_id.empty()) {
        return;
    }

    KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_(GetCacheMeta, instance_id);
    KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_(GetCacheLocation, instance_id);
    KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_(GetCacheLocationsByBackend, instance_id);
    KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_(GetCacheLocationLen, instance_id);
    KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_(StartWriteCache, instance_id);
    KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_(FinishWriteCache, instance_id);
    KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_(RemoveCache, instance_id);
    KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_(TrimCache, instance_id);
    KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_(GetClusterInfo, instance_id);
    KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_(ReportEvent, instance_id);
    KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_(EventBlockAdd, instance_id);
    KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_(EventBlockDelete, instance_id);
}

KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(GetCacheMeta);
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(GetCacheLocation);
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(GetCacheLocationsByBackend);
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(GetCacheLocationLen);
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(StartWriteCache);
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(FinishWriteCache);
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(RemoveCache);
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(TrimCache);
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(GetClusterInfo);
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(ReportEvent);
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(EventBlockAdd);
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(EventBlockDelete);

} // namespace kv_cache_manager