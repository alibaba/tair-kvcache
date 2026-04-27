#include "kv_cache_manager/data_storage/vineyard_backend.h"

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

namespace {
constexpr std::string_view kVineyardLocationPrefix = "kvs_vineyard_";
constexpr int kDetectRetryCount = 10;
constexpr int kDetectRetryIntervalSeconds = 1;
constexpr int kDetectLoopIntervalSeconds = 10;
} // anonymous namespace

VineyardBackend::VineyardBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
    : DataStorageBackend(std::move(metrics_registry)) {}

VineyardBackend::~VineyardBackend() {
    if (IsOpen()) {
        Close();
    }
}

DataStorageType VineyardBackend::GetType() { return DataStorageType::DATA_STORAGE_TYPE_VINEYARD; }

bool VineyardBackend::Available() { return IsOpen(); }

double VineyardBackend::GetStorageUsageRatio(const std::string & /*trace_id*/) const { return 1.0; }

ErrorCode VineyardBackend::DoOpen(const StorageConfig &config, const std::string &trace_id) {
    auto spec = std::dynamic_pointer_cast<VineyardStorageSpec>(config.storage_spec());
    if (!spec) {
        KVCM_LOG_WARN("trace_id [%s] | VineyardBackend::DoOpen: unexpected config type, storage config: [%s]",
                      trace_id.c_str(),
                      config.ToString().c_str());
        return EC_ERROR;
    }
    spec_ = *spec;
    SetOpen(true);
    KVCM_LOG_INFO(
        "trace_id [%s] | VineyardBackend opened, cluster: [%s]", trace_id.c_str(), spec_.cluster_name().c_str());
    return EC_OK;
}

ErrorCode VineyardBackend::Close() {
    SetOpen(false);

    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    for (auto &kv : nodes_) {
        if (kv.second && kv.second->probe_thread.joinable()) {
            kv.second->probe_thread.join();
        }
    }
    nodes_.clear();

    KVCM_LOG_INFO("VineyardBackend closed, cluster: [%s]", spec_.cluster_name().c_str());
    return EC_OK;
}

ErrorCode VineyardBackend::RegisterNode(const std::string &host_ip_port) {
    {
        std::shared_lock<std::shared_mutex> rlock(nodes_mutex_);
        if (nodes_.count(host_ip_port) > 0) {
            KVCM_LOG_INFO("VineyardBackend: node [%s] already registered", host_ip_port.c_str());
            return EC_OK;
        }
    }

    auto info = std::make_unique<NodeInfo>();
    info->probe_thread = std::thread(&VineyardBackend::ProbeNodeLoop, this, host_ip_port);

    {
        std::unique_lock<std::shared_mutex> wlock(nodes_mutex_);
        // Double-check after acquiring write lock
        if (nodes_.count(host_ip_port) == 0) {
            nodes_[host_ip_port] = std::move(info);
        }
    }

    KVCM_LOG_INFO(
        "VineyardBackend: node [%s] registered in cluster [%s]", host_ip_port.c_str(), spec_.cluster_name().c_str());
    return EC_OK;
}

ErrorCode VineyardBackend::UnregisterNode(const std::string &host_ip_port) {
    std::unique_ptr<NodeInfo> node_to_destroy;

    {
        std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
        auto it = nodes_.find(host_ip_port);
        if (it == nodes_.end()) {
            KVCM_LOG_WARN("VineyardBackend: node [%s] not found for unregister", host_ip_port.c_str());
            return EC_NOENT;
        }
        node_to_destroy = std::move(it->second);
        nodes_.erase(it);
    }

    // Join outside the lock to avoid deadlock with ProbeNodeLoop
    if (node_to_destroy && node_to_destroy->probe_thread.joinable()) {
        node_to_destroy->probe_thread.join();
    }

    KVCM_LOG_INFO("VineyardBackend: node [%s] unregistered from cluster [%s]",
                  host_ip_port.c_str(),
                  spec_.cluster_name().c_str());
    return EC_OK;
}

void VineyardBackend::SetNodeAvailable(const std::string &host_ip_port, bool available) {
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    auto it = nodes_.find(host_ip_port);
    if (it != nodes_.end() && it->second) {
        it->second->available.store(available, std::memory_order_relaxed);
    }
}

bool VineyardBackend::IsNodeAvailable(const std::string &host_ip_port) const {
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    auto it = nodes_.find(host_ip_port);
    if (it == nodes_.end() || !it->second) {
        return false;
    }
    return it->second->available.load(std::memory_order_relaxed);
}

bool VineyardBackend::IsLocationAvailable(const std::string &location_id) const {
    if (location_id.size() <= kVineyardLocationPrefix.size() ||
        location_id.compare(0, kVineyardLocationPrefix.size(), kVineyardLocationPrefix) != 0) {
        return false;
    }
    const std::string host_ip_port = location_id.substr(kVineyardLocationPrefix.size());
    return IsNodeAvailable(host_ip_port);
}

bool VineyardBackend::IsNodeRegistered(const std::string &host_ip_port) const {
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    return nodes_.count(host_ip_port) > 0;
}

bool VineyardBackend::DetectNodeHealthy(const std::string &host_ip_port) const {
    // TODO: implement actual health check (e.g. TCP connect or gRPC ping).
    // Returning true here makes the backend always assume healthy until a real
    // probe mechanism is implemented by the Vineyard team.
    (void)host_ip_port;
    return true;
}

void VineyardBackend::ProbeNodeLoop(const std::string &host_ip_port) {
    while (IsOpen() && IsNodeRegistered(host_ip_port)) {
        bool healthy = false;

        for (int i = 0; i < kDetectRetryCount; i++) {
            if (!IsOpen() || !IsNodeRegistered(host_ip_port)) {
                return;
            }
            if (DetectNodeHealthy(host_ip_port)) {
                healthy = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(kDetectRetryIntervalSeconds));
        }

        bool prev_available = IsNodeAvailable(host_ip_port);
        SetNodeAvailable(host_ip_port, healthy);

        if (prev_available && !healthy) {
            KVCM_LOG_WARN("VineyardBackend: node [%s] became unavailable", host_ip_port.c_str());
            // TriggerNodeDownCleanup is called from CacheManager which owns the
            // VineyardBackend.  We only log here; CacheManager registers a
            // callback via the EventManager if needed, but for now the
            // check_loc_data_exist callback already filters unavailable nodes.
        } else if (!prev_available && healthy) {
            KVCM_LOG_INFO("VineyardBackend: node [%s] recovered", host_ip_port.c_str());
        }

        std::this_thread::sleep_for(std::chrono::seconds(kDetectLoopIntervalSeconds));
    }
}

std::vector<std::pair<ErrorCode, DataStorageUri>> VineyardBackend::Create(const std::vector<std::string> &keys,
                                                                          size_t /*size_per_key*/,
                                                                          const std::string &trace_id,
                                                                          std::function<void()> /*cb*/) {
    KVCM_LOG_WARN("trace_id [%s] | VineyardBackend::Create should not be called", trace_id.c_str());
    return std::vector<std::pair<ErrorCode, DataStorageUri>>(
        keys.size(), std::make_pair(ErrorCode::EC_UNIMPLEMENTED, DataStorageUri()));
}

std::vector<ErrorCode> VineyardBackend::Delete(const std::vector<DataStorageUri> &storage_uris,
                                               const std::string &trace_id,
                                               std::function<void()> /*cb*/) {
    KVCM_LOG_WARN("trace_id [%s] | VineyardBackend::Delete should not be called", trace_id.c_str());
    return std::vector<ErrorCode>(storage_uris.size(), ErrorCode::EC_UNIMPLEMENTED);
}

std::vector<bool> VineyardBackend::Exist(const std::vector<DataStorageUri> &storage_uris) {
    return std::vector<bool>(storage_uris.size(), false);
}

std::vector<ErrorCode> VineyardBackend::Lock(const std::vector<DataStorageUri> &storage_uris) {
    return std::vector<ErrorCode>(storage_uris.size(), ErrorCode::EC_UNIMPLEMENTED);
}

std::vector<ErrorCode> VineyardBackend::UnLock(const std::vector<DataStorageUri> &storage_uris) {
    return std::vector<ErrorCode>(storage_uris.size(), ErrorCode::EC_UNIMPLEMENTED);
}

} // namespace kv_cache_manager
