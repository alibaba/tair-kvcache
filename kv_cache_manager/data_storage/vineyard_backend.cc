#include "kv_cache_manager/data_storage/vineyard_backend.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
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

    KVCM_LOG_INFO("trace_id [%s] | VineyardBackend opened, storage: [%s], hb_timeout=%ldms, "
                  "cleanup_grace=%ldms, check_interval=%ldms",
                  trace_id.c_str(),
                  config_.global_unique_name().c_str(),
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
        instance_nodes_.clear();
        node_generation_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(cleanup_cb_mutex_);
        cleanup_callback_ = nullptr;
        cleanup_cb_set_.store(false, std::memory_order_release);
    }
    KVCM_LOG_INFO("VineyardBackend closed, storage: [%s]", config_.global_unique_name().c_str());
    return EC_OK;
}

void VineyardBackend::SetCleanupCallback(CleanupCallback cb) {
    std::lock_guard<std::mutex> lock(cleanup_cb_mutex_);
    cleanup_callback_ = std::move(cb);
    cleanup_cb_set_.store(cleanup_callback_ != nullptr, std::memory_order_release);
}

ErrorCode VineyardBackend::RegisterNode(const std::string &instance_id,
                                        const std::string &host_ip_port,
                                        const std::vector<std::string> &mediums) {
    if (host_ip_port.empty()) {
        return EC_BADARGS;
    }
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    ++node_generation_[instance_id][host_ip_port];
    auto &host_map = instance_nodes_[instance_id];
    auto it = host_map.find(host_ip_port);
    int64_t now_ms = NowMillis();
    if (it != host_map.end()) {
        auto &info = *it->second;
        for (const auto &m : mediums) {
            if (std::find(info.mediums.begin(), info.mediums.end(), m) == info.mediums.end()) {
                info.mediums.push_back(m);
            }
        }
        info.last_heartbeat_ms.store(now_ms, std::memory_order_relaxed);
        info.available.store(true, std::memory_order_relaxed);
        info.unavailable_since_ms.store(0, std::memory_order_relaxed);
        info.instance_id = instance_id;
        info.metrics_tags = {{"instance_id", instance_id}, {"host", host_ip_port}};
        KVCM_LOG_INFO("VineyardBackend: node [%s] already registered for instance [%s], "
                      "mediums=%zu (refreshed heartbeat, gen=%lu)",
                      host_ip_port.c_str(),
                      instance_id.c_str(),
                      info.mediums.size(),
                      node_generation_[instance_id][host_ip_port]);
        return EC_OK;
    }

    auto info = std::make_unique<NodeInfo>();
    info->last_heartbeat_ms.store(now_ms, std::memory_order_relaxed);
    info->available.store(true, std::memory_order_relaxed);
    info->unavailable_since_ms.store(0, std::memory_order_relaxed);
    info->mediums = mediums;
    info->instance_id = instance_id;
    info->metrics_tags = {{"instance_id", instance_id}, {"host", host_ip_port}};
    host_map[host_ip_port] = std::move(info);

    KVCM_LOG_INFO("VineyardBackend: node [%s] registered in storage [%s] for instance [%s], mediums=%zu, gen=%lu",
                  host_ip_port.c_str(),
                  config_.global_unique_name().c_str(),
                  instance_id.c_str(),
                  mediums.size(),
                  node_generation_[instance_id][host_ip_port]);
    return EC_OK;
}

ErrorCode VineyardBackend::UnregisterNode(const std::string &instance_id, const std::string &host_ip_port) {
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    auto inst_it = instance_nodes_.find(instance_id);
    if (inst_it == instance_nodes_.end()) {
        KVCM_LOG_WARN("VineyardBackend: instance [%s] not found for unregister node [%s]",
                      instance_id.c_str(),
                      host_ip_port.c_str());
        return EC_NOENT;
    }
    auto it = inst_it->second.find(host_ip_port);
    if (it == inst_it->second.end()) {
        KVCM_LOG_WARN("VineyardBackend: node [%s] not found for instance [%s] for unregister",
                      host_ip_port.c_str(),
                      instance_id.c_str());
        return EC_NOENT;
    }
    if (metrics_registry_) {
        auto &info = *it->second;
        for (const auto &kv : info.last_system_status) {
            auto data = metrics_registry_->GetMetricsData("v6d." + kv.first);
            if (data) {
                data->RemoveByTags(info.metrics_tags);
            }
        }
    }
    inst_it->second.erase(it);
    KVCM_LOG_INFO("VineyardBackend: node [%s] unregistered from storage [%s] for instance [%s]",
                  host_ip_port.c_str(),
                  config_.global_unique_name().c_str(),
                  instance_id.c_str());
    return EC_OK;
}

// kvcm重启nodes_信息会丢失，v6d侧发送心跳会收到EC_NODE_NOT_REGISTERED，触发v6d re-register
ErrorCode VineyardBackend::OnHeartbeat(const std::string &instance_id,
                                       const std::string &host_ip_port,
                                       const std::map<std::string, std::string> &system_status) {
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    auto inst_it = instance_nodes_.find(instance_id);
    if (inst_it == instance_nodes_.end()) {
        KVCM_LOG_WARN("VineyardBackend: heartbeat from unregistered instance [%s] node [%s], "
                      "returning NODE_NOT_REGISTERED",
                      instance_id.c_str(),
                      host_ip_port.c_str());
        return EC_NODE_NOT_REGISTERED;
    }
    auto it = inst_it->second.find(host_ip_port);
    if (it == inst_it->second.end()) {
        KVCM_LOG_WARN("VineyardBackend: heartbeat from unregistered node [%s] for instance [%s], "
                      "returning NODE_NOT_REGISTERED",
                      host_ip_port.c_str(),
                      instance_id.c_str());
        return EC_NODE_NOT_REGISTERED;
    }
    auto &info = *it->second;
    int64_t now_ms = NowMillis();
    info.last_heartbeat_ms.store(now_ms, std::memory_order_release);
    bool prev = info.available.exchange(true, std::memory_order_relaxed);
    if (!prev) {
        info.unavailable_since_ms.store(0, std::memory_order_relaxed);
        KVCM_LOG_INFO("VineyardBackend: node [%s] recovered from unavailable", host_ip_port.c_str());
    }
    {
        std::lock_guard<std::mutex> status_lock(info.status_mutex);
        info.last_system_status = system_status;
    }

    if (metrics_registry_) {
        const auto &tags = info.metrics_tags;
        for (const auto &kv : system_status) {
            const auto &s = kv.second;
            if (s.empty())
                continue;
            char *end = nullptr;
            double val = std::strtod(s.c_str(), &end);
            if (end == s.c_str() + s.size()) {
                REPORT_DYNAMIC_GAUGE_(metrics_registry_, "v6d." + kv.first, tags, val);
            }
        }
    }
    return EC_OK;
}

void VineyardBackend::SetNodeUnavailable(const std::string &instance_id, const std::string &host_ip_port) {
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    auto inst_it = instance_nodes_.find(instance_id);
    if (inst_it == instance_nodes_.end()) {
        return;
    }
    auto it = inst_it->second.find(host_ip_port);
    if (it == inst_it->second.end()) {
        return;
    }
    auto &info = *it->second;
    bool prev = info.available.exchange(false, std::memory_order_relaxed);
    if (prev) {
        info.unavailable_since_ms.store(NowMillis(), std::memory_order_relaxed);
        ClearNodeGauges(info);
    }
}

void VineyardBackend::ClearNodeGauges(const NodeInfo &info) {
    if (!metrics_registry_) {
        return;
    }
    std::lock_guard<std::mutex> status_lock(info.status_mutex);
    for (const auto &kv : info.last_system_status) {
        auto data = metrics_registry_->GetMetricsData("v6d." + kv.first);
        if (data) {
            auto gauge = data->GetGauge(info.metrics_tags);
            if (gauge) {
                *gauge = 0.0;
            }
        }
    }
}

bool VineyardBackend::IsNodeAvailable(const std::string &instance_id, const std::string &host_ip_port) const {
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    auto inst_it = instance_nodes_.find(instance_id);
    if (inst_it == instance_nodes_.end()) {
        return false;
    }
    auto it = inst_it->second.find(host_ip_port);
    if (it == inst_it->second.end() || !it->second) {
        return false;
    }
    return it->second->available.load(std::memory_order_relaxed);
}

uint64_t VineyardBackend::GetNodeGeneration(const std::string &instance_id, const std::string &host_ip_port) const {
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    auto inst_it = node_generation_.find(instance_id);
    if (inst_it == node_generation_.end()) {
        return 0;
    }
    auto it = inst_it->second.find(host_ip_port);
    return it != inst_it->second.end() ? it->second : 0;
}

void VineyardBackend::LivenessCheckerLoop() {
    while (liveness_checker_running_.load(std::memory_order_relaxed) && IsOpen()) {
        int64_t now_ms = NowMillis();
        struct CleanupEntry {
            std::string instance_id;
            std::string host;
            uint64_t gen;
        };
        std::vector<CleanupEntry> to_cleanup;

        {
            std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
            for (auto &[inst_id, hosts] : instance_nodes_) {
                for (auto &[host, info_ptr] : hosts) {
                    if (!info_ptr) {
                        continue;
                    }
                    auto &info = *info_ptr;
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
                        KVCM_LOG_WARN(
                            "VineyardBackend: node [%s] instance [%s] timed out (no hb for %ldms), marked unavailable",
                            host.c_str(),
                            inst_id.c_str(),
                            now_ms - last_hb);
                        ClearNodeGauges(info);
                    }
                    int64_t unavailable_since = info.unavailable_since_ms.load(std::memory_order_relaxed);
                    if (unavailable_since > 0 && now_ms - unavailable_since >= cleanup_grace_ms_) {
                        auto gen_inst_it = node_generation_.find(inst_id);
                        uint64_t gen = 0;
                        if (gen_inst_it != node_generation_.end()) {
                            auto gen_it = gen_inst_it->second.find(host);
                            if (gen_it != gen_inst_it->second.end()) {
                                gen = gen_it->second;
                            }
                        }
                        to_cleanup.push_back({inst_id, host, gen});
                    }
                }
            }
        }

        if (!to_cleanup.empty()) {
            CleanupCallback cb_copy;
            {
                std::lock_guard<std::mutex> lock(cleanup_cb_mutex_);
                cb_copy = cleanup_callback_;
            }
            for (const auto &entry : to_cleanup) {
                KVCM_LOG_WARN("VineyardBackend: node [%s] instance [%s] passed cleanup_grace_ms, "
                              "triggering cleanup (gen=%lu)",
                              entry.host.c_str(),
                              entry.instance_id.c_str(),
                              entry.gen);
                if (cb_copy) {
                    cb_copy(entry.instance_id, entry.host, entry.gen);
                }
                uint64_t current_gen = GetNodeGeneration(entry.instance_id, entry.host);
                if (current_gen == entry.gen) {
                    UnregisterNode(entry.instance_id, entry.host);
                } else {
                    KVCM_LOG_INFO("VineyardBackend: node [%s] re-registered (gen=%lu -> %lu), skipping unregister",
                                  entry.host.c_str(),
                                  entry.gen,
                                  current_gen);
                }
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
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    for (const auto &uri : storage_uris) {
        if (uri.Valid() && !uri.GetHostName().empty()) {
            std::string host_ip_port = uri.GetHostName();
            if (uri.GetPort() > 0) {
                host_ip_port += ":" + std::to_string(uri.GetPort());
            }
            bool available = false;
            for (const auto &[inst_id, hosts] : instance_nodes_) {
                auto it = hosts.find(host_ip_port);
                if (it != hosts.end() && it->second && it->second->available.load(std::memory_order_relaxed)) {
                    available = true;
                    break;
                }
            }
            result.push_back(available);
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

std::string VineyardBackend::BuildLocationId(const std::string &medium, const std::string &host_ip_port) const {
    std::string id;
    id.reserve(8 + medium.size() + 1 + host_ip_port.size());
    id.append("kvs#v6d#");
    id.append(medium);
    id.push_back('#');
    id.append(host_ip_port);
    return id;
}

std::string VineyardBackend::HostSuffix(const std::string &host_ip_port) const { return "#" + host_ip_port; }

DataStorageType VineyardBackend::GetStorageType() const { return DataStorageType::DATA_STORAGE_TYPE_VINEYARD; }

std::string VineyardBackend::GetProtocol() const { return "vineyard"; }

} // namespace kv_cache_manager
