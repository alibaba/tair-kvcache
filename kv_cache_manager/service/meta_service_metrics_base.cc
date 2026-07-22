#include "kv_cache_manager/service/meta_service_metrics_base.h"

#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/metrics/metrics_lifecycle.h"
#include "kv_cache_manager/service/util/common.h"

#ifndef KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_
#define KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(name)                                                                \
    std::shared_ptr<MetricsCollector> MetaServiceMetricsBase::get_metrics_collector_from_map_for_##name(               \
        const std::string &instance_id) {                                                                              \
        return GetMetricsCollectorFromMap(                                                                             \
            #name, KVCM_METRICS_COLLECTOR_MAP_(name), mutex_##name##_, instance_id, instance_id, MetricsTags{});       \
    }

#endif

#ifndef KVCM_DEFINE_TYPED_METRICS_COLLECTOR_MAP_METHOD_
#define KVCM_DEFINE_TYPED_METRICS_COLLECTOR_MAP_METHOD_(name)                                                          \
    std::shared_ptr<MetricsCollector> MetaServiceMetricsBase::get_metrics_collector_from_map_for_##name(               \
        const std::string &instance_id, const std::string &type) {                                                     \
        return GetMetricsCollectorFromMap(#name,                                                                       \
                                          KVCM_METRICS_COLLECTOR_MAP_(name),                                           \
                                          mutex_##name##_,                                                             \
                                          instance_id,                                                                 \
                                          MakeTypedCollectorKey(instance_id, type),                                    \
                                          MetricsTags{{"type", type}});                                                \
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
    KVCM_INVALIDATE_TYPED_METRICS_COLLECTOR_MAP_(ReportEvent, instance_id);
    KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_(EventBlockAdd, instance_id);
    KVCM_INVALIDATE_TYPED_METRICS_COLLECTOR_MAP_(EventBlockAdd, instance_id);
    KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_(EventBlockDelete, instance_id);
    KVCM_INVALIDATE_TYPED_METRICS_COLLECTOR_MAP_(EventBlockDelete, instance_id);
    KVCM_INVALIDATE_METRICS_COLLECTOR_MAP_(GetHostCacheState, instance_id);
}

std::shared_ptr<MetricsCollector> MetaServiceMetricsBase::GetMetricsCollectorFromMap(
    const std::string &api_name,
    std::unordered_map<std::string, std::shared_ptr<MetricsCollector>> &metrics_collector_map,
    std::shared_mutex &mutex,
    const std::string &instance_id,
    const std::string &collector_key,
    const MetricsTags &extra_tags) {
    {
        std::shared_lock read_guard(mutex);
        auto iter = metrics_collector_map.find(collector_key);
        if (iter != metrics_collector_map.end()) {
            return iter->second;
        }
    }
    {
        // Hold a shared lifecycle lock around the slow path so the
        // tagged ServiceMetricsCollector below cannot be registered
        // concurrently with a unique-locked RemoveInstance /
        // RemoveInstanceGroup tag-filter purge.
        std::shared_lock<std::shared_mutex> lifecycle_guard(metrics_lifecycle_->mut_);
        std::scoped_lock write_guard(mutex);

        auto iter = metrics_collector_map.find(collector_key);
        if (iter != metrics_collector_map.end()) {
            return iter->second;
        }

        // GetInstanceGroupName prevents creating collectors for instances that
        // were never registered; the lifecycle lock only covers concurrent removals.
        auto instance_group = registry_manager_->GetInstanceGroupName(instance_id);
        if (instance_group.empty()) {
            return nullptr;
        }

        MetricsTags metrics_tags = {
            {"api_name", api_name}, {"instance_group", instance_group}, {"instance_id", instance_id}};
        metrics_tags.insert(extra_tags.begin(), extra_tags.end());
        auto metrics_collector = std::make_shared<ServiceMetricsCollector>(metrics_registry_, std::move(metrics_tags));
        if (!metrics_collector->Init()) {
            return nullptr;
        }
        metrics_collector_map[collector_key] = metrics_collector;
        return metrics_collector;
    }
}

std::string MetaServiceMetricsBase::MakeTypedCollectorKey(const std::string &instance_id, const std::string &type) {
    return instance_id + "#" + type;
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
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(GetHostCacheState);
KVCM_DEFINE_TYPED_METRICS_COLLECTOR_MAP_METHOD_(ReportEvent);
KVCM_DEFINE_TYPED_METRICS_COLLECTOR_MAP_METHOD_(EventBlockAdd);
KVCM_DEFINE_TYPED_METRICS_COLLECTOR_MAP_METHOD_(EventBlockDelete);

} // namespace kv_cache_manager
