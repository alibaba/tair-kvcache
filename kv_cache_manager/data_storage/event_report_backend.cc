#include "kv_cache_manager/data_storage/event_report_backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
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

std::string GenerateSnapshotVersionToken() {
    std::array<unsigned char, 16> bytes{};
    std::random_device random;
    for (auto &byte : bytes) {
        byte = static_cast<unsigned char>(random());
    }
    bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0fU) | 0x40U);
    bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3fU) | 0x80U);
    constexpr char kHex[] = "0123456789abcdef";
    std::string token(bytes.size() * 2, '0');
    for (size_t i = 0; i < bytes.size(); ++i) {
        token[i * 2] = kHex[bytes[i] >> 4];
        token[i * 2 + 1] = kHex[bytes[i] & 0x0fU];
    }
    return token;
}

} // namespace

EventReportBackend::EventReportBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
    : DataStorageBackend(std::move(metrics_registry)) {}

EventReportBackend::~EventReportBackend() {
    if (IsOpen()) {
        Close();
    }
}

std::shared_ptr<EventReportBackend::LifecycleFence>
EventReportBackend::GetOrCreateLifecycleFence(const ReporterSnapshotKey &reporter_key) {
    std::lock_guard<std::mutex> lock(lifecycle_fences_mutex_);
    auto &fence = lifecycle_fences_[reporter_key];
    if (!fence) {
        fence = std::make_shared<LifecycleFence>();
    }
    return fence;
}

std::shared_ptr<EventReportBackend::LifecycleFence>
EventReportBackend::FindLifecycleFence(const ReporterSnapshotKey &reporter_key) const {
    std::lock_guard<std::mutex> lock(lifecycle_fences_mutex_);
    const auto it = lifecycle_fences_.find(reporter_key);
    return it == lifecycle_fences_.end() ? nullptr : it->second;
}

// --- DataStorageBackend interface ---

DataStorageType EventReportBackend::GetType() { return config_.type(); }

bool EventReportBackend::Available() { return IsOpen() && IsAvailable(); }

double EventReportBackend::GetStorageUsageRatio(const std::string & /*trace_id*/) const { return 1.0; }

ErrorCode EventReportBackend::DoOpen(const StorageConfig &config, const std::string &trace_id) {
    auto spec = std::dynamic_pointer_cast<EventReportStorageSpec>(config.storage_spec());
    if (!spec) {
        KVCM_LOG_WARN("trace_id [%s] | EventReportBackend::DoOpen: unexpected config type, storage config: [%s]",
                      trace_id.c_str(),
                      config.ToString().c_str());
        return EC_ERROR;
    }
    if (!IsEventReportStorageType(config.type())) {
        KVCM_LOG_WARN("trace_id [%s] | EventReportBackend::DoOpen: unexpected storage type, storage config: [%s]",
                      trace_id.c_str(),
                      config.ToString().c_str());
        return EC_ERROR;
    }
    spec_ = *spec;
    heartbeat_timeout_ms_ = spec_.heartbeat_timeout_ms();
    cleanup_grace_ms_ = spec_.cleanup_grace_ms();
    liveness_check_interval_ms_ = spec_.liveness_check_interval_ms();
    snapshot_min_interval_ms_ = spec_.snapshot_min_interval_ms();
    snapshot_delta_drain_timeout_ms_ = spec_.snapshot_delta_drain_timeout_ms();

    SetOpen(true);
    SetAvailable(true);

    liveness_checker_running_.store(true, std::memory_order_relaxed);
    liveness_checker_thread_ = std::thread(&EventReportBackend::LivenessCheckerLoop, this);

    KVCM_LOG_INFO("trace_id [%s] | EventReportBackend opened, storage: [%s], type: [%s], hb_timeout=%ldms, "
                  "cleanup_grace=%ldms, check_interval=%ldms, snapshot_min_interval=%ldms, "
                  "snapshot_delta_drain_timeout=%ldms",
                  trace_id.c_str(),
                  config_.global_unique_name().c_str(),
                  ToString(config_.type()).c_str(),
                  heartbeat_timeout_ms_,
                  cleanup_grace_ms_,
                  liveness_check_interval_ms_,
                  snapshot_min_interval_ms_,
                  snapshot_delta_drain_timeout_ms_);
    return EC_OK;
}

ErrorCode EventReportBackend::Close() {
    SetOpen(false);
    SetAvailable(false);
    liveness_checker_running_.store(false, std::memory_order_relaxed);
    if (liveness_checker_thread_.joinable()) {
        liveness_checker_thread_.join();
    }
    std::lock_guard<std::mutex> fences_guard(lifecycle_fences_mutex_);
    std::vector<std::unique_lock<std::shared_mutex>> fence_locks;
    fence_locks.reserve(lifecycle_fences_.size());
    for (const auto &entry : lifecycle_fences_) {
        const auto &fence = entry.second;
        if (fence) {
            fence_locks.emplace_back(fence->mutex);
        }
    }
    {
        std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
        instance_nodes_.clear();
        node_generation_.clear();
        snapshot_versions_.clear();
        snapshot_token_owners_.clear();
    }
    lifecycle_fences_.clear();
    snapshot_state_cv_.notify_all();
    {
        std::lock_guard<std::mutex> lock(cleanup_cb_mutex_);
        cleanup_callback_ = nullptr;
        cleanup_cb_set_.store(false, std::memory_order_release);
    }
    KVCM_LOG_INFO("EventReportBackend closed, storage: [%s]", config_.global_unique_name().c_str());
    return EC_OK;
}

void EventReportBackend::SetCleanupCallback(CleanupCallback cb) {
    std::lock_guard<std::mutex> lock(cleanup_cb_mutex_);
    cleanup_callback_ = std::move(cb);
    cleanup_cb_set_.store(cleanup_callback_ != nullptr, std::memory_order_release);
}

ErrorCode EventReportBackend::RegisterNode(const std::string &instance_id,
                                           const std::string &host_ip_port,
                                           const std::vector<std::string> &mediums) {
    if (instance_id.empty() || !SnapshotUriUtils::IsValidLocationIdComponent(host_ip_port) ||
        std::any_of(mediums.begin(), mediums.end(), [](const std::string &medium) {
            return !SnapshotUriUtils::IsValidLocationIdComponent(medium);
        })) {
        return EC_BADARGS;
    }
    const ReporterSnapshotKey reporter_key{instance_id, host_ip_port};
    const auto lifecycle_fence = GetOrCreateLifecycleFence(reporter_key);
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_fence->mutex);
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    ++node_generation_[instance_id][host_ip_port];
    lifecycle_fence->generation = node_generation_[instance_id][host_ip_port];
    lifecycle_fence->registered = true;
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
        info.metrics_tags = {{"instance_id", instance_id}, {"host", host_ip_port}, {"type", ToString(config_.type())}};
        KVCM_LOG_INFO("EventReportBackend: node [%s] already registered for instance [%s], "
                      "mediums=%zu (refreshed heartbeat, gen=%" PRIu64 ")",
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
    info->metrics_tags = {{"instance_id", instance_id}, {"host", host_ip_port}, {"type", ToString(config_.type())}};
    host_map[host_ip_port] = std::move(info);

    KVCM_LOG_INFO("EventReportBackend: node [%s] registered in storage [%s] for instance [%s], "
                  "mediums=%zu, gen=%" PRIu64,
                  host_ip_port.c_str(),
                  config_.global_unique_name().c_str(),
                  instance_id.c_str(),
                  mediums.size(),
                  node_generation_[instance_id][host_ip_port]);
    return EC_OK;
}

ErrorCode EventReportBackend::EnsureNodeRegistered(const std::string &instance_id,
                                                   const std::string &host_ip_port,
                                                   const std::vector<std::string> &mediums) {
    if (instance_id.empty() || !SnapshotUriUtils::IsValidLocationIdComponent(host_ip_port) ||
        std::any_of(mediums.begin(), mediums.end(), [](const std::string &medium) {
            return !SnapshotUriUtils::IsValidLocationIdComponent(medium);
        })) {
        return EC_BADARGS;
    }

    auto merge_mediums = [&mediums](NodeInfo &info) {
        for (const auto &medium : mediums) {
            if (std::find(info.mediums.begin(), info.mediums.end(), medium) == info.mediums.end()) {
                info.mediums.push_back(medium);
            }
        }
    };

    // The common path only enriches an existing node. It must not wait for
    // the lifecycle write lock: an in-flight metadata mutation intentionally
    // holds a shared lifecycle lease, and new deltas still need to reach the
    // bounded snapshot gate instead of blocking here indefinitely.
    {
        std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
        auto instance_it = instance_nodes_.find(instance_id);
        if (instance_it != instance_nodes_.end()) {
            auto node_it = instance_it->second.find(host_ip_port);
            if (node_it != instance_it->second.end() && node_it->second) {
                merge_mediums(*node_it->second);
                return EC_OK;
            }
        }
    }

    const ReporterSnapshotKey reporter_key{instance_id, host_ip_port};
    const auto lifecycle_fence = GetOrCreateLifecycleFence(reporter_key);
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_fence->mutex);
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    auto &host_map = instance_nodes_[instance_id];
    if (auto it = host_map.find(host_ip_port); it != host_map.end()) {
        // Another request may have created the node between the fast-path
        // check and acquiring the lifecycle lock.
        merge_mediums(*it->second);
        lifecycle_fence->generation = node_generation_[instance_id][host_ip_port];
        lifecycle_fence->registered = true;
        return EC_OK;
    }
    const auto generation_it = node_generation_.find(instance_id);
    if (generation_it != node_generation_.end() && generation_it->second.count(host_ip_port) > 0) {
        return EC_NODE_NOT_REGISTERED;
    }

    const uint64_t generation = ++node_generation_[instance_id][host_ip_port];
    lifecycle_fence->generation = generation;
    lifecycle_fence->registered = true;
    auto info = std::make_unique<NodeInfo>();
    info->last_heartbeat_ms.store(NowMillis(), std::memory_order_relaxed);
    info->available.store(true, std::memory_order_relaxed);
    info->unavailable_since_ms.store(0, std::memory_order_relaxed);
    info->mediums = mediums;
    info->instance_id = instance_id;
    info->metrics_tags = {{"instance_id", instance_id}, {"host", host_ip_port}, {"type", ToString(config_.type())}};
    host_map[host_ip_port] = std::move(info);

    KVCM_LOG_INFO("EventReportBackend: lazily restored node [%s] in storage [%s] for instance [%s], "
                  "mediums=%zu, gen=%" PRIu64,
                  host_ip_port.c_str(),
                  config_.global_unique_name().c_str(),
                  instance_id.c_str(),
                  mediums.size(),
                  generation);
    return EC_OK;
}

ErrorCode EventReportBackend::UnregisterNode(const std::string &instance_id, const std::string &host_ip_port) {
    const ReporterSnapshotKey reporter_key{instance_id, host_ip_port};
    const auto lifecycle_fence = GetOrCreateLifecycleFence(reporter_key);
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_fence->mutex);
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    const ErrorCode ec = UnregisterNodeLocked(instance_id, host_ip_port);
    lifecycle_fence->generation = node_generation_[instance_id][host_ip_port];
    lifecycle_fence->registered = false;
    return ec;
}

ErrorCode EventReportBackend::UnregisterNodeForHostDown(const std::string &instance_id,
                                                        const std::string &host_ip_port,
                                                        uint64_t &out_generation) {
    const ReporterSnapshotKey reporter_key{instance_id, host_ip_port};
    const auto lifecycle_fence = GetOrCreateLifecycleFence(reporter_key);
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_fence->mutex);
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    out_generation = node_generation_[instance_id][host_ip_port];
    lifecycle_fence->generation = out_generation;
    lifecycle_fence->registered = false;
    const auto instance_it = instance_nodes_.find(instance_id);
    if (instance_it == instance_nodes_.end() || instance_it->second.find(host_ip_port) == instance_it->second.end()) {
        // HOST_DOWN is explicitly idempotent. Keep the tombstone generation
        // above, but do not emit the generic missing-node warning.
        return EC_OK;
    }
    return UnregisterNodeLocked(instance_id, host_ip_port);
}

ErrorCode EventReportBackend::UnregisterNodeIfGeneration(const std::string &instance_id,
                                                         const std::string &host_ip_port,
                                                         uint64_t expected_generation) {
    const ReporterSnapshotKey reporter_key{instance_id, host_ip_port};
    const auto lifecycle_fence = GetOrCreateLifecycleFence(reporter_key);
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_fence->mutex);
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    const auto generation_it = node_generation_.find(instance_id);
    uint64_t current_generation = 0;
    if (generation_it != node_generation_.end()) {
        const auto host_generation_it = generation_it->second.find(host_ip_port);
        if (host_generation_it != generation_it->second.end()) {
            current_generation = host_generation_it->second;
        }
    }
    if (current_generation != expected_generation) {
        return EC_MISMATCH;
    }
    const ErrorCode ec = UnregisterNodeLocked(instance_id, host_ip_port);
    lifecycle_fence->generation = current_generation;
    lifecycle_fence->registered = false;
    return ec;
}

ErrorCode EventReportBackend::UnregisterNodeLocked(const std::string &instance_id, const std::string &host_ip_port) {
    node_generation_[instance_id].try_emplace(host_ip_port, 0);
    auto inst_it = instance_nodes_.find(instance_id);
    if (inst_it == instance_nodes_.end()) {
        KVCM_LOG_WARN("EventReportBackend: instance [%s] not found for unregister node [%s]",
                      instance_id.c_str(),
                      host_ip_port.c_str());
        return EC_NOENT;
    }
    auto it = inst_it->second.find(host_ip_port);
    if (it == inst_it->second.end()) {
        KVCM_LOG_WARN("EventReportBackend: node [%s] not found for instance [%s] for unregister",
                      host_ip_port.c_str(),
                      instance_id.c_str());
        return EC_NOENT;
    }
    if (metrics_registry_) {
        auto &info = *it->second;
        auto prefix = "event_report.";
        for (const auto &kv : info.last_system_status) {
            auto data = metrics_registry_->GetMetricsData(prefix + kv.first);
            if (data) {
                data->RemoveByTags(info.metrics_tags);
            }
        }
    }
    inst_it->second.erase(it);
    const ReporterSnapshotKey reporter_key{instance_id, host_ip_port};
    const auto snapshot_it = snapshot_versions_.find(reporter_key);
    if (snapshot_it != snapshot_versions_.end() && !snapshot_it->second.committed.empty()) {
        snapshot_token_owners_.erase(snapshot_it->second.committed);
    }
    snapshot_versions_.erase(reporter_key);
    snapshot_state_cv_.notify_all();
    KVCM_LOG_INFO("EventReportBackend: node [%s] unregistered from storage [%s] for instance [%s]",
                  host_ip_port.c_str(),
                  config_.global_unique_name().c_str(),
                  instance_id.c_str());
    return EC_OK;
}

ErrorCode EventReportBackend::OnHeartbeat(const std::string &instance_id,
                                          const std::string &host_ip_port,
                                          const std::map<std::string, std::string> &system_status) {
    if (instance_id.empty() || !SnapshotUriUtils::IsValidLocationIdComponent(host_ip_port)) {
        return EC_BADARGS;
    }
    const ReporterSnapshotKey reporter_key{instance_id, host_ip_port};
    const auto lifecycle_fence = GetOrCreateLifecycleFence(reporter_key);
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_fence->mutex);
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    auto &host_map = instance_nodes_[instance_id];
    auto it = host_map.find(host_ip_port);
    if (it == host_map.end()) {
        const auto generation_it = node_generation_.find(instance_id);
        if (generation_it != node_generation_.end() && generation_it->second.count(host_ip_port) > 0) {
            KVCM_LOG_WARN("EventReportBackend: heartbeat from tombstoned node [%s] for instance [%s], "
                          "returning NODE_NOT_REGISTERED",
                          host_ip_port.c_str(),
                          instance_id.c_str());
            return EC_NODE_NOT_REGISTERED;
        }
        const uint64_t generation = ++node_generation_[instance_id][host_ip_port];
        lifecycle_fence->generation = generation;
        lifecycle_fence->registered = true;
        auto new_info = std::make_unique<NodeInfo>();
        new_info->last_heartbeat_ms.store(NowMillis(), std::memory_order_relaxed);
        new_info->available.store(true, std::memory_order_relaxed);
        new_info->unavailable_since_ms.store(0, std::memory_order_relaxed);
        new_info->instance_id = instance_id;
        new_info->metrics_tags = {
            {"instance_id", instance_id}, {"host", host_ip_port}, {"type", ToString(config_.type())}};
        it = host_map.emplace(host_ip_port, std::move(new_info)).first;
        KVCM_LOG_INFO("EventReportBackend: heartbeat lazily restored node [%s] for instance [%s] "
                      "(generation=%" PRIu64 ")",
                      host_ip_port.c_str(),
                      instance_id.c_str(),
                      generation);
    }
    auto &info = *it->second;
    int64_t now_ms = NowMillis();
    info.last_heartbeat_ms.store(now_ms, std::memory_order_release);
    bool prev = info.available.exchange(true, std::memory_order_relaxed);
    if (!prev) {
        // A cleanup task may already have selected the previous unavailable
        // generation and may be waiting before it scans metadata. Advance the
        // generation before acknowledging recovery so that both the cleanup
        // callback and the final unregister step reject that stale task.
        const uint64_t generation = ++node_generation_[instance_id][host_ip_port];
        lifecycle_fence->generation = generation;
        lifecycle_fence->registered = true;
        info.unavailable_since_ms.store(0, std::memory_order_relaxed);
        KVCM_LOG_INFO("EventReportBackend: node [%s] recovered from unavailable (generation=%" PRIu64 ")",
                      host_ip_port.c_str(),
                      generation);
    }
    lifecycle_fence->generation = node_generation_[instance_id][host_ip_port];
    lifecycle_fence->registered = true;
    {
        std::lock_guard<std::mutex> status_lock(info.status_mutex);
        info.last_system_status = system_status;
    }
    const auto metrics_tags = info.metrics_tags;

    // Keep the node lifecycle lock until the gauges are published.
    // UnregisterNodeLocked removes the same tagged gauges while holding this
    // lock; releasing it earlier would allow HOST_DOWN to remove the node and
    // an older heartbeat to recreate ghost metrics afterwards.
    if (metrics_registry_) {
        auto prefix = "event_report.";
        for (const auto &kv : system_status) {
            const auto &s = kv.second;
            if (s.empty())
                continue;
            char *end = nullptr;
            double val = std::strtod(s.c_str(), &end);
            if (end == s.c_str() + s.size()) {
                REPORT_DYNAMIC_GAUGE_(metrics_registry_, prefix + kv.first, metrics_tags, val);
            }
        }
    }
    return EC_OK;
}

void EventReportBackend::SetNodeUnavailable(const std::string &instance_id, const std::string &host_ip_port) {
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

void EventReportBackend::ClearNodeGauges(const NodeInfo &info) {
    if (!metrics_registry_) {
        return;
    }
    std::lock_guard<std::mutex> status_lock(info.status_mutex);
    auto prefix = "event_report.";
    for (const auto &kv : info.last_system_status) {
        auto data = metrics_registry_->GetMetricsData(prefix + kv.first);
        if (data) {
            auto gauge = data->GetGauge(info.metrics_tags);
            if (gauge) {
                *gauge = 0.0;
            }
        }
    }
}

bool EventReportBackend::IsNodeAvailable(const std::string &instance_id, const std::string &host_ip_port) const {
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

bool EventReportBackend::IsNodeRegistered(const std::string &instance_id, const std::string &host_ip_port) const {
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    const auto instance_it = instance_nodes_.find(instance_id);
    return instance_it != instance_nodes_.end() && instance_it->second.count(host_ip_port) > 0;
}

uint64_t EventReportBackend::GetNodeGeneration(const std::string &instance_id, const std::string &host_ip_port) const {
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    auto inst_it = node_generation_.find(instance_id);
    if (inst_it == node_generation_.end()) {
        return 0;
    }
    auto it = inst_it->second.find(host_ip_port);
    return it != inst_it->second.end() ? it->second : 0;
}

void EventReportBackend::LivenessCheckerLoop() {
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
                        KVCM_LOG_WARN("EventReportBackend: node [%s] instance [%s] timed out (no hb for %ldms), "
                                      "marked unavailable",
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
                KVCM_LOG_WARN("EventReportBackend: node [%s] instance [%s] passed cleanup_grace_ms, "
                              "triggering cleanup (gen=%" PRIu64 ")",
                              entry.host.c_str(),
                              entry.instance_id.c_str(),
                              entry.gen);
                if (cb_copy) {
                    cb_copy(entry.instance_id, entry.host, entry.gen);
                }
                const ErrorCode unregister_ec = UnregisterNodeIfGeneration(entry.instance_id, entry.host, entry.gen);
                if (unregister_ec == EC_MISMATCH) {
                    const uint64_t current_gen = GetNodeGeneration(entry.instance_id, entry.host);
                    KVCM_LOG_INFO("EventReportBackend: node [%s] re-registered "
                                  "(gen=%" PRIu64 " -> %" PRIu64 "), skipping unregister",
                                  entry.host.c_str(),
                                  entry.gen,
                                  current_gen);
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(liveness_check_interval_ms_));
    }
}

std::vector<std::pair<ErrorCode, DataStorageUri>> EventReportBackend::Create(const std::vector<std::string> &keys,
                                                                             size_t /*size_per_key*/,
                                                                             const std::string &trace_id,
                                                                             std::function<void()> /*cb*/) {
    KVCM_LOG_WARN("trace_id [%s] | EventReportBackend::Create should not be called", trace_id.c_str());
    return std::vector<std::pair<ErrorCode, DataStorageUri>>(
        keys.size(), std::make_pair(ErrorCode::EC_UNIMPLEMENTED, DataStorageUri()));
}

std::vector<ErrorCode> EventReportBackend::Delete(const std::vector<DataStorageUri> &storage_uris,
                                                  const std::string &trace_id,
                                                  std::function<void()> /*cb*/) {
    KVCM_LOG_WARN("trace_id [%s] | EventReportBackend::Delete should not be called", trace_id.c_str());
    return std::vector<ErrorCode>(storage_uris.size(), ErrorCode::EC_UNIMPLEMENTED);
}

std::vector<bool> EventReportBackend::Exist(const std::vector<DataStorageUri> &storage_uris) {
    return std::vector<bool>(storage_uris.size(), false);
}

std::vector<bool> EventReportBackend::MightExist(const std::vector<DataStorageUri> &storage_uris) {
    std::vector<bool> result;
    result.reserve(storage_uris.size());
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    for (const auto &uri : storage_uris) {
        SnapshotUriInfo info;
        if (!SnapshotUriUtils::ParseSnapshotUriInfo(uri, info)) {
            result.push_back(false);
            continue;
        }
        const auto owner_it = snapshot_token_owners_.find(info.version);
        if (owner_it == snapshot_token_owners_.end()) {
            result.push_back(false);
            continue;
        }
        const ReporterSnapshotKey &reporter_key = owner_it->second;
        const auto state_it = snapshot_versions_.find(reporter_key);
        const auto instance_it = instance_nodes_.find(reporter_key.instance_id);
        if (state_it == snapshot_versions_.end() || state_it->second.committed != info.version ||
            instance_it == instance_nodes_.end()) {
            result.push_back(false);
            continue;
        }
        const auto node_it = instance_it->second.find(reporter_key.host_ip_port);
        result.push_back(node_it != instance_it->second.end() && node_it->second &&
                         node_it->second->available.load(std::memory_order_relaxed));
    }
    return result;
}

std::vector<ErrorCode> EventReportBackend::Lock(const std::vector<DataStorageUri> &storage_uris) {
    return std::vector<ErrorCode>(storage_uris.size(), ErrorCode::EC_UNIMPLEMENTED);
}

std::vector<ErrorCode> EventReportBackend::UnLock(const std::vector<DataStorageUri> &storage_uris) {
    return std::vector<ErrorCode>(storage_uris.size(), ErrorCode::EC_UNIMPLEMENTED);
}

std::string EventReportBackend::BuildLocationId(const std::string &medium, const std::string &host_ip_port) const {
    if (!SnapshotUriUtils::IsValidLocationIdComponent(medium) ||
        !SnapshotUriUtils::IsValidLocationIdComponent(host_ip_port)) {
        return {};
    }
    const std::string type_token = ToString(config_.type());
    std::string result;
    result.reserve(4 + type_token.size() + 1 + medium.size() + 1 + host_ip_port.size());
    result.append("kvs#");
    result.append(type_token);
    result.push_back('#');
    result.append(medium);
    result.push_back('#');
    result.append(host_ip_port);
    return result;
}

bool EventReportBackend::ParseLocationId(const std::string &location_id,
                                         std::string &out_medium,
                                         std::string &out_host_ip_port) const {
    std::string storage_type;
    return SnapshotUriUtils::ParseEventReportLocationId(location_id, storage_type, out_medium, out_host_ip_port) &&
           storage_type == ToString(config_.type());
}

std::string EventReportBackend::HostSuffix(const std::string &host_ip_port) const { return "#" + host_ip_port; }

ErrorCode EventReportBackend::BeginDeltaMutation(const ReporterSnapshotKey &reporter_key,
                                                 std::string &out_committed_version,
                                                 uint64_t *out_lifecycle_generation) {
    out_committed_version.clear();
    if (out_lifecycle_generation) {
        *out_lifecycle_generation = 0;
    }
    if (reporter_key.instance_id.empty() || reporter_key.host_ip_port.empty()) {
        return EC_BADARGS;
    }
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    const int64_t snapshot_wait_timeout_ms = snapshot_delta_drain_timeout_ms_;
    const bool snapshot_finished =
        snapshot_state_cv_.wait_for(lock, std::chrono::milliseconds(snapshot_wait_timeout_ms), [&] {
            auto it = snapshot_versions_.find(reporter_key);
            return it == snapshot_versions_.end() || it->second.in_flight.empty();
        });
    if (!snapshot_finished) {
        return EC_SNAPSHOT_IN_PROGRESS;
    }
    auto state_it = snapshot_versions_.find(reporter_key);
    if (state_it == snapshot_versions_.end() || state_it->second.committed.empty()) {
        const auto instance_it = instance_nodes_.find(reporter_key.instance_id);
        if (instance_it == instance_nodes_.end() ||
            instance_it->second.find(reporter_key.host_ip_port) == instance_it->second.end()) {
            return EC_SNAPSHOT_REQUIRED;
        }

        auto token_in_use = [this](const std::string &token) {
            if (snapshot_token_owners_.count(token) > 0) {
                return true;
            }
            return std::any_of(snapshot_versions_.begin(), snapshot_versions_.end(), [&token](const auto &entry) {
                return entry.second.in_flight == token;
            });
        };
        std::string candidate;
        do {
            candidate = GenerateSnapshotVersionToken();
        } while (token_in_use(candidate));
        auto &new_state = snapshot_versions_[reporter_key];
        new_state.committed = candidate;
        snapshot_token_owners_[candidate] = reporter_key;
        state_it = snapshot_versions_.find(reporter_key);
    }
    auto &state = state_it->second;
    if (state.active_delta_mutations == std::numeric_limits<uint64_t>::max()) {
        return EC_ERROR;
    }
    ++state.active_delta_mutations;
    out_committed_version = state.committed;
    if (out_lifecycle_generation) {
        *out_lifecycle_generation = node_generation_[reporter_key.instance_id][reporter_key.host_ip_port];
    }
    return EC_OK;
}

void EventReportBackend::EndDeltaMutation(const ReporterSnapshotKey &reporter_key, uint64_t lifecycle_generation) {
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    auto it = snapshot_versions_.find(reporter_key);
    if (it == snapshot_versions_.end() || it->second.active_delta_mutations == 0) {
        const auto generation_it = node_generation_.find(reporter_key.instance_id);
        const auto node_it = instance_nodes_.find(reporter_key.instance_id);
        const bool lifecycle_ended =
            generation_it == node_generation_.end() ||
            generation_it->second.find(reporter_key.host_ip_port) == generation_it->second.end() ||
            (lifecycle_generation != 0 &&
             generation_it->second.at(reporter_key.host_ip_port) != lifecycle_generation) ||
            node_it == instance_nodes_.end() ||
            node_it->second.find(reporter_key.host_ip_port) == node_it->second.end();
        if (lifecycle_ended) {
            KVCM_LOG_DEBUG("EventReportBackend: delta mutation lease ended after reporter lifecycle changed, "
                           "instance [%s] host [%s]",
                           reporter_key.instance_id.c_str(),
                           reporter_key.host_ip_port.c_str());
            return;
        }
        KVCM_LOG_ERROR("EventReportBackend: unmatched delta mutation lease for instance [%s] host [%s]",
                       reporter_key.instance_id.c_str(),
                       reporter_key.host_ip_port.c_str());
        return;
    }
    --it->second.active_delta_mutations;
    const bool drained = it->second.active_delta_mutations == 0;
    lock.unlock();
    if (drained) {
        snapshot_state_cv_.notify_all();
    }
}

ErrorCode EventReportBackend::BeginSnapshot(const ReporterSnapshotKey &reporter_key,
                                            std::string &out_candidate_version,
                                            uint64_t &out_retry_after_ms,
                                            uint64_t *out_lifecycle_generation) {
    out_candidate_version.clear();
    out_retry_after_ms = 0;
    if (reporter_key.instance_id.empty() || reporter_key.host_ip_port.empty()) {
        return EC_BADARGS;
    }
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    const auto instance_it = instance_nodes_.find(reporter_key.instance_id);
    if (instance_it == instance_nodes_.end() ||
        instance_it->second.find(reporter_key.host_ip_port) == instance_it->second.end()) {
        return EC_SNAPSHOT_REQUIRED;
    }
    auto &state = snapshot_versions_[reporter_key];
    if (!state.in_flight.empty()) {
        return EC_SNAPSHOT_IN_PROGRESS;
    }
    const int64_t now_ms = NowMillis();
    if (state.last_commit_ms > 0 && snapshot_min_interval_ms_ > 0) {
        const int64_t elapsed_ms = now_ms - state.last_commit_ms;
        if (elapsed_ms < snapshot_min_interval_ms_) {
            out_retry_after_ms = static_cast<uint64_t>(snapshot_min_interval_ms_ - elapsed_ms);
            return EC_SNAPSHOT_RATE_LIMITED;
        }
    }
    auto token_in_use = [this](const std::string &token) {
        if (snapshot_token_owners_.count(token) > 0) {
            return true;
        }
        return std::any_of(snapshot_versions_.begin(), snapshot_versions_.end(), [&token](const auto &entry) {
            return entry.second.in_flight == token;
        });
    };
    do {
        out_candidate_version = GenerateSnapshotVersionToken();
    } while (token_in_use(out_candidate_version));
    // Close the reporter's write gate before waiting for already admitted
    // deltas. Deltas arriving from this point wait until commit or abort.
    ++state.attempt_epoch;
    state.in_flight = out_candidate_version;
    const int64_t delta_drain_timeout_ms = snapshot_delta_drain_timeout_ms_;
    const bool deltas_drained =
        snapshot_state_cv_.wait_for(lock, std::chrono::milliseconds(delta_drain_timeout_ms), [&] {
            auto it = snapshot_versions_.find(reporter_key);
            return it == snapshot_versions_.end() || it->second.in_flight != out_candidate_version ||
                   it->second.active_delta_mutations == 0;
        });
    if (!deltas_drained) {
        auto it = snapshot_versions_.find(reporter_key);
        const uint64_t active_delta_mutations = it == snapshot_versions_.end() ? 0 : it->second.active_delta_mutations;
        if (it != snapshot_versions_.end() && it->second.in_flight == out_candidate_version) {
            it->second.in_flight.clear();
        }
        out_candidate_version.clear();
        lock.unlock();
        snapshot_state_cv_.notify_all();
        KVCM_LOG_WARN("EventReportBackend: snapshot admission timed out after %" PRId64 "ms waiting for %" PRIu64
                      " active delta mutation(s), instance [%s] host [%s]; "
                      "candidate aborted and write gate reopened",
                      delta_drain_timeout_ms,
                      active_delta_mutations,
                      reporter_key.instance_id.c_str(),
                      reporter_key.host_ip_port.c_str());
        return EC_SNAPSHOT_IN_PROGRESS;
    }
    auto it = snapshot_versions_.find(reporter_key);
    if (it == snapshot_versions_.end() || it->second.in_flight != out_candidate_version) {
        out_candidate_version.clear();
        return EC_SNAPSHOT_REQUIRED;
    }
    if (out_lifecycle_generation) {
        *out_lifecycle_generation = node_generation_[reporter_key.instance_id][reporter_key.host_ip_port];
    }
    return EC_OK;
}

ErrorCode EventReportBackend::AcquireLifecycleMutationLease(const ReporterSnapshotKey &reporter_key,
                                                            uint64_t expected_generation,
                                                            LifecycleMutationLease &out_lease) const {
    out_lease.reset();
    const auto lifecycle_fence = FindLifecycleFence(reporter_key);
    if (!lifecycle_fence) {
        return EC_NODE_NOT_REGISTERED;
    }
    auto lease = std::make_shared<std::shared_lock<std::shared_mutex>>(lifecycle_fence->mutex, std::try_to_lock);
    if (!lease->owns_lock()) {
        // Do not wait behind a lifecycle writer while the caller may already
        // hold metadata locks. Cleanup takes the opposite order
        // (lifecycle -> metadata), so blocking here would deadlock.
        return EC_NODE_NOT_REGISTERED;
    }
    if (!lifecycle_fence->registered || lifecycle_fence->generation != expected_generation) {
        return EC_NODE_NOT_REGISTERED;
    }
    out_lease = std::move(lease);
    return EC_OK;
}

ErrorCode EventReportBackend::AcquireLifecycleCleanupLease(const ReporterSnapshotKey &reporter_key,
                                                           uint64_t expected_generation,
                                                           LifecycleMutationLease &out_lease) const {
    out_lease.reset();
    const auto lifecycle_fence = FindLifecycleFence(reporter_key);
    if (!lifecycle_fence) {
        return EC_MISMATCH;
    }
    auto lease = std::make_shared<std::shared_lock<std::shared_mutex>>(lifecycle_fence->mutex);
    if (lifecycle_fence->generation != expected_generation) {
        return EC_MISMATCH;
    }
    out_lease = std::move(lease);
    return EC_OK;
}

bool EventReportBackend::CommitSnapshotVersion(const ReporterSnapshotKey &reporter_key, const std::string &version) {
    if (!SnapshotUriUtils::IsValidSnapshotVersionToken(version)) {
        return false;
    }
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    auto it = snapshot_versions_.find(reporter_key);
    if (it == snapshot_versions_.end() || it->second.in_flight != version) {
        return false;
    }
    if (!it->second.committed.empty()) {
        snapshot_token_owners_.erase(it->second.committed);
    }
    it->second.committed = version;
    it->second.in_flight.clear();
    it->second.last_commit_ms = NowMillis();
    it->second.strict_query_visibility = true;
    snapshot_token_owners_[version] = reporter_key;
    lock.unlock();
    snapshot_state_cv_.notify_all();
    return true;
}

void EventReportBackend::AbortSnapshotVersion(const ReporterSnapshotKey &reporter_key, const std::string &version) {
    if (!SnapshotUriUtils::IsValidSnapshotVersionToken(version) || reporter_key.instance_id.empty() ||
        reporter_key.host_ip_port.empty()) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    auto it = snapshot_versions_.find(reporter_key);
    if (it != snapshot_versions_.end() && it->second.in_flight == version) {
        it->second.in_flight.clear();
        // Candidate metadata is written in place. Once an admitted attempt
        // aborts, accepting only committed could hide locations already
        // replaced with the failed candidate. Stay soft until a later
        // successful complete snapshot restores an authoritative fence.
        it->second.strict_query_visibility = false;
        lock.unlock();
        snapshot_state_cv_.notify_all();
    }
}

std::string EventReportBackend::GetSnapshotVersion(const ReporterSnapshotKey &reporter_key) const {
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    auto it = snapshot_versions_.find(reporter_key);
    return it == snapshot_versions_.end() ? std::string{} : it->second.committed;
}

void EventReportBackend::GetSnapshotVersionTokens(const ReporterSnapshotKey &reporter_key,
                                                  std::string &out_committed,
                                                  std::string &out_in_flight) const {
    out_committed.clear();
    out_in_flight.clear();
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    const auto it = snapshot_versions_.find(reporter_key);
    if (it != snapshot_versions_.end()) {
        out_committed = it->second.committed;
        out_in_flight = it->second.in_flight;
    }
}

bool EventReportBackend::GetQueryVisibilityState(const ReporterSnapshotKey &reporter_key,
                                                 bool &out_strict,
                                                 std::string &out_committed) const {
    out_strict = false;
    out_committed.clear();
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    const auto instance_it = instance_nodes_.find(reporter_key.instance_id);
    if (instance_it == instance_nodes_.end()) {
        return false;
    }
    const auto node_it = instance_it->second.find(reporter_key.host_ip_port);
    if (node_it == instance_it->second.end() || !node_it->second ||
        !node_it->second->available.load(std::memory_order_relaxed)) {
        return false;
    }
    const auto state_it = snapshot_versions_.find(reporter_key);
    if (state_it != snapshot_versions_.end()) {
        out_strict = state_it->second.strict_query_visibility;
        out_committed = state_it->second.committed;
    }
    return true;
}

uint64_t EventReportBackend::GetSnapshotAttemptEpoch(const ReporterSnapshotKey &reporter_key) const {
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    const auto it = snapshot_versions_.find(reporter_key);
    return it == snapshot_versions_.end() ? 0 : it->second.attempt_epoch;
}

void EventReportBackend::SetSnapshotMinIntervalMsForTest(int64_t interval_ms) {
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    snapshot_min_interval_ms_ = std::max<int64_t>(0, interval_ms);
}

void EventReportBackend::SetSnapshotDeltaDrainTimeoutMsForTest(int64_t timeout_ms) {
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    snapshot_delta_drain_timeout_ms_ = std::max<int64_t>(1, timeout_ms);
}

DataStorageType EventReportBackend::GetStorageType() const { return config_.type(); }

} // namespace kv_cache_manager
