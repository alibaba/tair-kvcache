#include "kv_cache_manager/data_storage/vineyard_backend.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
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
    heartbeat_timeout_ms_ = spec_.heartbeat_timeout_ms();
    cleanup_grace_ms_ = spec_.cleanup_grace_ms();
    liveness_check_interval_ms_ = spec_.liveness_check_interval_ms();

    SetOpen(true);

    liveness_checker_running_.store(true, std::memory_order_relaxed);
    liveness_checker_thread_ = std::thread(&VineyardBackend::LivenessCheckerLoop, this);

    KVCM_LOG_INFO("trace_id [%s] | VineyardBackend opened, cluster: [%s], hb_timeout=%ldms, "
                  "cleanup_grace=%ldms, check_interval=%ldms",
                  trace_id.c_str(),
                  spec_.cluster_name().c_str(),
                  heartbeat_timeout_ms_,
                  cleanup_grace_ms_,
                  liveness_check_interval_ms_);
    return EC_OK;
}

ErrorCode VineyardBackend::Close() {
    SetOpen(false);
    liveness_checker_running_.store(false, std::memory_order_relaxed);
    if (liveness_checker_thread_.joinable()) {
        liveness_checker_thread_.join();
    }
    {
        std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
        nodes_.clear();
        node_generation_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(cleanup_cb_mutex_);
        cleanup_callback_ = nullptr;
        cleanup_cb_set_.store(false, std::memory_order_release);
    }
    KVCM_LOG_INFO("VineyardBackend closed, cluster: [%s]", spec_.cluster_name().c_str());
    return EC_OK;
}

void VineyardBackend::SetCleanupCallback(CleanupCallback cb) {
    std::lock_guard<std::mutex> lock(cleanup_cb_mutex_);
    cleanup_callback_ = std::move(cb);
    cleanup_cb_set_.store(cleanup_callback_ != nullptr, std::memory_order_release);
}

ErrorCode VineyardBackend::RegisterNode(const std::string &host_ip_port, const std::vector<std::string> &mediums) {
    if (host_ip_port.empty()) {
        return EC_BADARGS;
    }
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    ++node_generation_[host_ip_port];
    auto it = nodes_.find(host_ip_port);
    int64_t now_ms = NowMillis();
    if (it != nodes_.end()) {
        auto &info = *it->second;
        for (const auto &m : mediums) {
            if (std::find(info.mediums.begin(), info.mediums.end(), m) == info.mediums.end()) {
                info.mediums.push_back(m);
            }
        }
        info.last_heartbeat_ms.store(now_ms, std::memory_order_relaxed);
        info.available.store(true, std::memory_order_relaxed);
        info.unavailable_since_ms.store(0, std::memory_order_relaxed);
        KVCM_LOG_INFO("VineyardBackend: node [%s] already registered, mediums=%zu (refreshed heartbeat, gen=%lu)",
                      host_ip_port.c_str(),
                      info.mediums.size(),
                      node_generation_[host_ip_port]);
        return EC_OK;
    }

    auto info = std::make_unique<NodeInfo>();
    info->last_heartbeat_ms.store(now_ms, std::memory_order_relaxed);
    info->available.store(true, std::memory_order_relaxed);
    info->unavailable_since_ms.store(0, std::memory_order_relaxed);
    info->mediums = mediums;
    nodes_[host_ip_port] = std::move(info);

    KVCM_LOG_INFO("VineyardBackend: node [%s] registered in cluster [%s], mediums=%zu, gen=%lu",
                  host_ip_port.c_str(),
                  spec_.cluster_name().c_str(),
                  mediums.size(),
                  node_generation_[host_ip_port]);
    return EC_OK;
}

ErrorCode VineyardBackend::UnregisterNode(const std::string &host_ip_port) {
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    auto it = nodes_.find(host_ip_port);
    if (it == nodes_.end()) {
        KVCM_LOG_WARN("VineyardBackend: node [%s] not found for unregister", host_ip_port.c_str());
        return EC_NOENT;
    }
    nodes_.erase(it);
    KVCM_LOG_INFO("VineyardBackend: node [%s] unregistered from cluster [%s]",
                  host_ip_port.c_str(),
                  spec_.cluster_name().c_str());
    return EC_OK;
}

void VineyardBackend::OnHeartbeat(const std::string &host_ip_port,
                                  const std::map<std::string, std::string> &system_status) {
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    auto it = nodes_.find(host_ip_port);
    if (it == nodes_.end()) {
        KVCM_LOG_WARN("VineyardBackend: heartbeat from unregistered node [%s], skipped", host_ip_port.c_str());
        return;
    }
    auto &info = *it->second;
    int64_t now_ms = NowMillis();
    info.last_heartbeat_ms.store(now_ms, std::memory_order_release);
    bool prev = info.available.exchange(true, std::memory_order_relaxed);
    if (!prev) {
        info.unavailable_since_ms.store(0, std::memory_order_relaxed);
        KVCM_LOG_INFO("VineyardBackend: node [%s] recovered from unavailable", host_ip_port.c_str());
    }
    info.last_system_status = system_status;
}

void VineyardBackend::SetNodeUnavailable(const std::string &host_ip_port) {
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    auto it = nodes_.find(host_ip_port);
    if (it == nodes_.end()) {
        return;
    }
    auto &info = *it->second;
    bool prev = info.available.exchange(false, std::memory_order_relaxed);
    if (prev) {
        info.unavailable_since_ms.store(NowMillis(), std::memory_order_relaxed);
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

bool VineyardBackend::IsNodeRegistered(const std::string &host_ip_port) const {
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    return nodes_.count(host_ip_port) > 0;
}

bool VineyardBackend::IsLocationAvailable(const std::string &location_id) const {
    auto pos = location_id.rfind('#');
    if (pos == std::string::npos || pos + 1 >= location_id.size()) {
        return false;
    }
    return IsNodeAvailable(location_id.substr(pos + 1));
}

uint64_t VineyardBackend::GetNodeGeneration(const std::string &host_ip_port) const {
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    auto it = node_generation_.find(host_ip_port);
    return it != node_generation_.end() ? it->second : 0;
}

void VineyardBackend::LivenessCheckerLoop() {
    while (liveness_checker_running_.load(std::memory_order_relaxed) && IsOpen()) {
        int64_t now_ms = NowMillis();
        std::vector<std::pair<std::string, uint64_t>> to_cleanup;

        {
            std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
            for (auto &kv : nodes_) {
                if (!kv.second) {
                    continue;
                }
                auto &info = *kv.second;
                int64_t last_hb = info.last_heartbeat_ms.load(std::memory_order_relaxed);
                if (last_hb == 0 || now_ms - last_hb <= heartbeat_timeout_ms_) {
                    continue;
                }

                last_hb = info.last_heartbeat_ms.load(std::memory_order_acquire);
                int64_t fresh_now = NowMillis();
                if (fresh_now - last_hb <= heartbeat_timeout_ms_) {
                    continue;
                }

                bool prev = info.available.exchange(false, std::memory_order_relaxed);
                if (prev) {
                    info.unavailable_since_ms.store(now_ms, std::memory_order_relaxed);
                    KVCM_LOG_WARN("VineyardBackend: node [%s] timed out (no hb for %ldms), marked unavailable",
                                  kv.first.c_str(),
                                  now_ms - last_hb);
                }
                int64_t unavailable_since = info.unavailable_since_ms.load(std::memory_order_relaxed);
                if (unavailable_since > 0 && now_ms - unavailable_since >= cleanup_grace_ms_) {
                    auto gen_it = node_generation_.find(kv.first);
                    uint64_t gen = (gen_it != node_generation_.end()) ? gen_it->second : 0;
                    to_cleanup.emplace_back(kv.first, gen);
                }
            }
        }

        if (!to_cleanup.empty()) {
            CleanupCallback cb_copy;
            {
                std::lock_guard<std::mutex> lock(cleanup_cb_mutex_);
                cb_copy = cleanup_callback_;
            }
            for (const auto &[host, gen] : to_cleanup) {
                KVCM_LOG_WARN("VineyardBackend: node [%s] passed cleanup_grace_ms, triggering cleanup (gen=%lu)",
                              host.c_str(),
                              gen);
                if (cb_copy) {
                    cb_copy(host, gen);
                }
                UnregisterNode(host);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(liveness_check_interval_ms_));
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

std::vector<bool> VineyardBackend::MightExist(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<bool> result;
    result.reserve(storage_uris.size());
    for (const auto &uri : storage_uris) {
        if (uri.Valid() && !uri.GetHostName().empty()) {
            std::string host_ip_port = uri.GetHostName();
            if (uri.GetPort() > 0) {
                host_ip_port += ":" + std::to_string(uri.GetPort());
            }
            result.push_back(IsNodeRegistered(host_ip_port));
        } else {
            result.push_back(true);
        }
    }
    return result;
}

std::vector<ErrorCode> VineyardBackend::Lock(const std::vector<DataStorageUri> &storage_uris) {
    return std::vector<ErrorCode>(storage_uris.size(), ErrorCode::EC_UNIMPLEMENTED);
}

std::vector<ErrorCode> VineyardBackend::UnLock(const std::vector<DataStorageUri> &storage_uris) {
    return std::vector<ErrorCode>(storage_uris.size(), ErrorCode::EC_UNIMPLEMENTED);
}

} // namespace kv_cache_manager
