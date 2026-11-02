#include "kv_cache_manager/metrics/node_status_metrics.h"

#include <memory>
#include <utility>

#include "kv_cache_manager/data_storage/data_storage_backend.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

namespace {

// Capability flags OR-reduced across all backends. Available() gates the read
// path and Writable() the write path; Writable() defaults to Available(), and
// backends with a read-only degraded mode (TairMempool) override it to false.
struct StorageCapability {
    bool readable = false;
    bool writable = false;
};

StorageCapability CollectStorageCapability(const std::shared_ptr<DataStorageManager> &data_storage_manager) {
    StorageCapability capability;
    if (!data_storage_manager) {
        return capability;
    }

    for (const auto &unique_name : data_storage_manager->GetAllStorageNames()) {
        const auto backend = data_storage_manager->GetDataStorageBackend(unique_name);
        if (!backend) {
            continue;
        }
        capability.readable = capability.readable || backend->Available();
        capability.writable = capability.writable || backend->Writable();
    }
    return capability;
}

} // namespace

NodeAccessMode NodeStatusMetrics::ComputeAccessMode(const std::shared_ptr<DataStorageManager> &data_storage_manager) {
    const auto capability = CollectStorageCapability(data_storage_manager);
    if (capability.writable) {
        return NodeAccessMode::READ_WRITE;
    }
    if (capability.readable) {
        return NodeAccessMode::READ_ONLY;
    }
    return NodeAccessMode::NONE;
}

void NodeStatusMetrics::Publish(const std::shared_ptr<MetricsRegistry> &metrics_registry,
                                bool is_leader,
                                const std::shared_ptr<DataStorageManager> &data_storage_manager) {
    if (!metrics_registry) {
        return;
    }

    const auto access_mode = is_leader ? ComputeAccessMode(data_storage_manager) : NodeAccessMode::NONE;
    metrics_registry->GetGauge(kIsLeaderName) = is_leader ? 1.0 : 0.0;
    metrics_registry->GetGauge(kAccessModeName) = static_cast<double>(access_mode);
}

} // namespace kv_cache_manager
