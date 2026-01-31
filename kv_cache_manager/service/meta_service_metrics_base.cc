#include "kv_cache_manager/service/meta_service_metrics_base.h"

#include "kv_cache_manager/config/registry_manager.h"
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
            std::scoped_lock write_guard(mutex_##name##_);                                                             \
                                                                                                                       \
            auto iter = KVCM_METRICS_COLLECTOR_MAP_(name).find(instance_id);                                           \
            if (iter != KVCM_METRICS_COLLECTOR_MAP_(name).end()) {                                                     \
                return iter->second;                                                                                   \
            }                                                                                                          \
                                                                                                                       \
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
                                               std::shared_ptr<RegistryManager> registry_manager)
    : metrics_registry_(std::move(metrics_registry)), registry_manager_(registry_manager) {}

void MetaServiceMetricsBase::InitMetrics() {
    MAKE_SERVICE_METRICS_COLLECTOR(RegisterInstance);
    MAKE_SERVICE_METRICS_COLLECTOR(GetInstanceInfo);
}

KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(GetCacheMeta);
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(GetCacheLocation);
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(StartWriteCache);
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(FinishWriteCache);
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(RemoveCache);
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(TrimCache);

} // namespace kv_cache_manager