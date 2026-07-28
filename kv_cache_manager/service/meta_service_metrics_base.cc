#include "kv_cache_manager/service/meta_service_metrics_base.h"

#include <array>
#include <cstdint>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/metrics/metrics_lifecycle.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
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
    std::shared_ptr<MetricsCollector> MetaServiceMetricsBase::GetTypedMetricsCollectorFor##name(                       \
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
    {
        std::scoped_lock guard(mutex_ReportEventType_);
        const std::string prefix = instance_id + "#";
        auto &collectors = KVCM_METRICS_COLLECTOR_MAP_(ReportEventType);
        for (auto iter = collectors.begin(); iter != collectors.end();) {
            if (iter->first.compare(0, prefix.size(), prefix) == 0) {
                iter = collectors.erase(iter);
            } else {
                ++iter;
            }
        }
    }
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

std::string MetaServiceMetricsBase::MakeEventTypeCollectorKey(const std::string &instance_id,
                                                              const std::string &type,
                                                              const std::string &event_type) {
    return instance_id + "#" + type + "#" + event_type;
}

std::shared_ptr<MetricsCollector> MetaServiceMetricsBase::GetEventTypeMetricsCollectorFromMap(
    const std::string &instance_id, const std::string &type, const std::string &event_type) {
    const std::string collector_key = MakeEventTypeCollectorKey(instance_id, type, event_type);
    {
        std::shared_lock read_guard(mutex_ReportEventType_);
        auto iter = KVCM_METRICS_COLLECTOR_MAP_(ReportEventType).find(collector_key);
        if (iter != KVCM_METRICS_COLLECTOR_MAP_(ReportEventType).end()) {
            return iter->second;
        }
    }

    std::shared_lock<std::shared_mutex> lifecycle_guard(metrics_lifecycle_->mut_);
    std::scoped_lock write_guard(mutex_ReportEventType_);
    auto iter = KVCM_METRICS_COLLECTOR_MAP_(ReportEventType).find(collector_key);
    if (iter != KVCM_METRICS_COLLECTOR_MAP_(ReportEventType).end()) {
        return iter->second;
    }
    auto instance_group = registry_manager_->GetInstanceGroupName(instance_id);
    if (instance_group.empty()) {
        return nullptr;
    }
    MetricsTags tags = {
        {"instance_group", instance_group}, {"instance_id", instance_id}, {"type", type}, {"event_type", event_type}};
    auto collector = std::make_shared<EventReportMetricsCollector>(metrics_registry_, std::move(tags));
    if (!collector->Init()) {
        return nullptr;
    }
    KVCM_METRICS_COLLECTOR_MAP_(ReportEventType)[collector_key] = collector;
    return collector;
}

std::shared_ptr<MetricsCollector> MetaServiceMetricsBase::GetTypedMetricsCollectorForReportEventType(
    const std::string &instance_id, const std::string &type, const std::string &event_type) {
    return GetEventTypeMetricsCollectorFromMap(instance_id, type, event_type);
}

void MetaServiceMetricsBase::AttachReportEventTypeMetricsCollectors(const proto::meta::ReportEventRequest &request,
                                                                    const std::string &type,
                                                                    RequestContext *request_context) {
    if (request.instance_id().empty() || type.empty() || request_context == nullptr) {
        return;
    }

    uint32_t event_type_mask = 0;
    for (const auto &event : request.events()) {
        const int event_type = static_cast<int>(event.event_type());
        event_type_mask |=
            1U << ((event_type >= proto::meta::EVENT_NODE_REGISTER && event_type <= proto::meta::EVENT_BLOCK_SNAPSHOT)
                       ? event_type
                       : 0);
    }

    static constexpr std::array<const char *, 7> kEventTypeTags = {
        "unknown", "node_register", "block_add", "block_delete", "host_down", "heartbeat", "block_snapshot"};
    for (size_t event_type = 0; event_type < kEventTypeTags.size(); ++event_type) {
        if ((event_type_mask & (1U << event_type)) == 0) {
            continue;
        }
        auto shared_collector =
            GetTypedMetricsCollectorForReportEventType(request.instance_id(), type, kEventTypeTags[event_type]);
        auto event_collector = std::dynamic_pointer_cast<EventReportMetricsCollector>(shared_collector);
        if (event_collector) {
            // The cached object owns the registry handles. Each request gets a
            // lightweight view with the same handles but private sample state,
            // avoiding both registry re-registration and cross-request races.
            request_context->GetMetricsCollectorsVehicle().AddMetricsCollector(
                std::make_shared<EventReportMetricsCollector>(*event_collector));
        }
    }
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
KVCM_DEFINE_METRICS_COLLECTOR_MAP_METHOD_(GetHostCacheState);
KVCM_DEFINE_TYPED_METRICS_COLLECTOR_MAP_METHOD_(ReportEvent);

} // namespace kv_cache_manager
