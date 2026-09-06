#include "kv_cache_manager/data_storage/event_report_backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <exception>
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
#include "kv_cache_manager/common/timestamp_util.h"
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

template <typename LockType>
class EventReportBackend::TimedLock {
public:
    template <typename Mutex>
    TimedLock(Mutex &mutex, const LockMetrics &metrics)
        : metrics_(metrics), wait_begin_us_(TimestampUtil::GetSteadyTimeUs()), lock_(mutex) {
        RecordAcquired();
    }

    ~TimedLock() {
        if (lock_.owns_lock()) {
            Release();
        }
    }

    TimedLock(const TimedLock &) = delete;
    TimedLock &operator=(const TimedLock &) = delete;

    void lock() {
        wait_begin_us_ = TimestampUtil::GetSteadyTimeUs();
        lock_.lock();
        RecordAcquired();
    }

    void unlock() { Release(); }

private:
    void RecordAcquired() {
        hold_begin_us_ = TimestampUtil::GetSteadyTimeUs();
        wait_time_us_ = static_cast<std::uint64_t>(hold_begin_us_ - wait_begin_us_);
    }

    void Release() {
        const auto hold_time_us = static_cast<std::uint64_t>(TimestampUtil::GetSteadyTimeUs() - hold_begin_us_);
        lock_.unlock();
        metrics_.wait_time_us_sum += wait_time_us_;
        metrics_.hold_time_us_sum += hold_time_us;
        ++metrics_.acquire_counter;
        hold_begin_us_ = 0;
    }

    LockMetrics metrics_;
    std::int64_t wait_begin_us_;
    LockType lock_;
    std::int64_t hold_begin_us_ = 0;
    std::uint64_t wait_time_us_ = 0;
};

EventReportBackend::EventReportBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
    : DataStorageBackend(std::move(metrics_registry)) {}

EventReportBackend::~EventReportBackend() {
    if (IsOpen()) {
        Close();
    }
}

void EventReportBackend::InitLockMetrics(const StorageConfig &config) {
    if (!metrics_registry_) {
        return;
    }
    auto init_metrics = [&](LockMetrics &metrics, const std::string &lock_name, const std::string &operation) {
        const MetricsTags tags{{"lock_name", lock_name},
                               {"operation", operation},
                               {"type", ToString(config.type())},
                               {"unique_name", config.global_unique_name()}};
        metrics.wait_time_us_sum = metrics_registry_->GetCounter("event.lock_wait_time_us_sum", tags);
        metrics.hold_time_us_sum = metrics_registry_->GetCounter("event.lock_hold_time_us_sum", tags);
        metrics.acquire_counter = metrics_registry_->GetCounter("event.lock_acquire_counter", tags);
    };
    init_metrics(reporter_state_begin_metrics_, "reporter_state_mutex", "begin");
    init_metrics(reporter_state_end_metrics_, "reporter_state_mutex", "end");
    init_metrics(reporter_state_snapshot_get_metrics_, "reporter_state_mutex", "snapshot_get");
    init_metrics(nodes_mutex_ensure_node_metrics_, "nodes_mutex", "ensure_node");
    init_metrics(lifecycle_fences_mutex_metrics_, "lifecycle_fences_mutex", "access");
}

std::shared_ptr<EventReportBackend::LifecycleFence>
EventReportBackend::GetOrCreateLifecycleFence(const ReporterSnapshotKey &reporter_key) {
    if (const auto fence = FindLifecycleFence(reporter_key)) {
        return fence;
    }
    TimedLock<std::unique_lock<std::shared_mutex>> lock(lifecycle_fences_mutex_, lifecycle_fences_mutex_metrics_);
    auto &fence = lifecycle_fences_[reporter_key];
    if (!fence) {
        fence = std::make_shared<LifecycleFence>();
    }
    return fence;
}

std::shared_ptr<EventReportBackend::LifecycleFence>
EventReportBackend::FindLifecycleFence(const ReporterSnapshotKey &reporter_key) const {
    TimedLock<std::shared_lock<std::shared_mutex>> lock(lifecycle_fences_mutex_, lifecycle_fences_mutex_metrics_);
    const auto it = lifecycle_fences_.find(reporter_key);
    return it == lifecycle_fences_.end() ? nullptr : it->second;
}

std::vector<std::shared_ptr<EventReportBackend::LifecycleFence>> EventReportBackend::GetLifecycleFences() const {
    TimedLock<std::shared_lock<std::shared_mutex>> lock(lifecycle_fences_mutex_, lifecycle_fences_mutex_metrics_);
    std::vector<std::shared_ptr<LifecycleFence>> fences;
    fences.reserve(lifecycle_fences_.size());
    for (const auto &entry : lifecycle_fences_) {
        if (entry.second) {
            fences.push_back(entry.second);
        }
    }
    return fences;
}

std::string EventReportBackend::ReserveSnapshotVersionToken(const ReporterSnapshotKey &reporter_key) {
    std::unique_lock<std::shared_mutex> lock(snapshot_token_owners_mutex_);
    std::string token;
    do {
        token = GenerateSnapshotVersionToken();
    } while (snapshot_token_owners_.count(token) > 0);
    snapshot_token_owners_.emplace(token, reporter_key);
    return token;
}

void EventReportBackend::ReleaseSnapshotVersionToken(const ReporterSnapshotKey &reporter_key,
                                                     const std::string &token) {
    if (token.empty()) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(snapshot_token_owners_mutex_);
    const auto it = snapshot_token_owners_.find(token);
    if (it != snapshot_token_owners_.end() && it->second == reporter_key) {
        snapshot_token_owners_.erase(it);
    }
}

// --- DataStorageBackend interface ---

DataStorageType EventReportBackend::GetType() { return config_.type(); }

bool EventReportBackend::Available() { return IsOpen() && IsAvailable(); }

void EventReportBackend::SetAvailable(bool available) {
    DataStorageBackend::SetAvailable(available);
    if (!available) {
        for (const auto &fence : GetLifecycleFences()) {
            fence->snapshot_state_cv.notify_all();
        }
    }
}

double EventReportBackend::GetStorageUsageRatio(const std::string & /*trace_id*/) const { return 1.0; }

ErrorCode EventReportBackend::Open(const StorageConfig &config, const std::string &trace_id) {
    if (IsOpen() || Retired()) {
        KVCM_LOG_WARN("trace_id [%s] | EventReportBackend::Open: backend objects cannot be opened twice or reused "
                      "after Close",
                      trace_id.c_str());
        return EC_ERROR;
    }
    return DataStorageBackend::Open(config, trace_id);
}

ErrorCode EventReportBackend::DoOpen(const StorageConfig &config, const std::string &trace_id) {
    if (IsOpen() || Retired()) {
        return EC_ERROR;
    }
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
    InitLockMetrics(config);
    heartbeat_timeout_ms_ = spec_.heartbeat_timeout_ms();
    cleanup_grace_ms_ = spec_.cleanup_grace_ms();
    liveness_check_interval_ms_ = spec_.liveness_check_interval_ms();
    snapshot_min_interval_ms_ = spec_.snapshot_min_interval_ms();
    snapshot_delta_drain_timeout_ms_ = spec_.snapshot_delta_drain_timeout_ms();

    SetOpen(true);
    SetAvailable(true);

    liveness_checker_running_.store(true, std::memory_order_relaxed);
    try {
        liveness_checker_thread_ = std::thread(&EventReportBackend::LivenessCheckerLoop, this);
    } catch (const std::exception &e) {
        liveness_checker_running_.store(false, std::memory_order_release);
        SetAvailable(false);
        SetOpen(false);
        KVCM_LOG_ERROR("trace_id [%s] | EventReportBackend::DoOpen: start liveness checker failed: [%s]",
                       trace_id.c_str(),
                       e.what());
        return EC_ERROR;
    } catch (...) {
        liveness_checker_running_.store(false, std::memory_order_release);
        SetAvailable(false);
        SetOpen(false);
        KVCM_LOG_ERROR("trace_id [%s] | EventReportBackend::DoOpen: start liveness checker failed with unknown "
                       "exception",
                       trace_id.c_str());
        return EC_ERROR;
    }

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
    retired_.store(true, std::memory_order_release);
    SetOpen(false);
    SetAvailable(false);
    // Serialize the predicate update with wait_for().  An atomic predicate is
    // not sufficient to prevent a lost notification when the waiter is
    // between checking the predicate and actually sleeping.
    {
        std::lock_guard<std::mutex> wait_guard(liveness_wait_mutex_);
        liveness_checker_running_.store(false, std::memory_order_release);
    }
    liveness_wait_cv_.notify_all();
    if (liveness_checker_thread_.joinable()) {
        liveness_checker_thread_.join();
    }
    auto fence_refs = GetLifecycleFences();

    // Never wait for a lifecycle fence while holding lifecycle_fences_mutex_.
    // Cleanup deliberately takes lifecycle -> metadata, while a metadata RMW
    // may already hold metadata when it briefly looks up and try-locks its
    // lifecycle fence. Close holding the table mutex while waiting for the
    // cleanup lease would complete a three-lock cycle. The strong references
    // also keep each shared_mutex alive until its unique_lock is released.
    std::vector<std::unique_lock<std::shared_mutex>> fence_locks;
    fence_locks.reserve(fence_refs.size());
    for (const auto &fence : fence_refs) {
        fence_locks.emplace_back(fence->mutex);
    }
    {
        std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
        instance_nodes_.clear();
        node_generation_.clear();
    }
    for (const auto &fence : fence_refs) {
        {
            std::lock_guard<std::mutex> state_lock(fence->state_mutex);
            fence->registered = false;
            fence->snapshot_state = {};
        }
        fence->snapshot_state_cv.notify_all();
    }
    {
        std::unique_lock<std::shared_mutex> token_lock(snapshot_token_owners_mutex_);
        snapshot_token_owners_.clear();
    }
    {
        TimedLock<std::unique_lock<std::shared_mutex>> fences_guard(lifecycle_fences_mutex_,
                                                                    lifecycle_fences_mutex_metrics_);
        lifecycle_fences_.clear();
    }
    fence_locks.clear();
    fence_refs.clear();
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
    if (Retired()) {
        cleanup_callback_ = nullptr;
        cleanup_cb_set_.store(false, std::memory_order_release);
        return;
    }
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
    if (!AcceptingReports()) {
        return EC_INSTANCE_NOT_EXIST;
    }
    const ReporterSnapshotKey reporter_key{instance_id, host_ip_port};
    const auto lifecycle_fence = GetOrCreateLifecycleFence(reporter_key);
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_fence->mutex);
    if (!AcceptingReports()) {
        return EC_INSTANCE_NOT_EXIST;
    }
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    std::lock_guard<std::mutex> state_lock(lifecycle_fence->state_mutex);
    auto &host_map = instance_nodes_[instance_id];
    auto it = host_map.find(host_ip_port);
    ++node_generation_[instance_id][host_ip_port];
    lifecycle_fence->generation = node_generation_[instance_id][host_ip_port];
    lifecycle_fence->registered = true;
    int64_t now_ms = NowMillis();
    if (it != host_map.end()) {
        auto &info = *it->second;
        info.lifecycle_fence = lifecycle_fence;
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
    info->lifecycle_fence = lifecycle_fence;
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
    if (!AcceptingReports()) {
        return EC_INSTANCE_NOT_EXIST;
    }

    auto merge_mediums = [&mediums](NodeInfo &info) {
        for (const auto &medium : mediums) {
            if (std::find(info.mediums.begin(), info.mediums.end(), medium) == info.mediums.end()) {
                info.mediums.push_back(medium);
            }
        }
    };

    // Repeated reports normally carry a medium that is already known. Keep
    // that path read-only so it does not serialize all reporters on the node
    // table's exclusive lock. A missing medium is rechecked under the unique
    // lock before it is merged (lock-based double check; the map itself must
    // never be read without nodes_mutex_).
    {
        TimedLock<std::shared_lock<std::shared_mutex>> lock(nodes_mutex_, nodes_mutex_ensure_node_metrics_);
        auto instance_it = instance_nodes_.find(instance_id);
        if (instance_it != instance_nodes_.end()) {
            auto node_it = instance_it->second.find(host_ip_port);
            if (node_it != instance_it->second.end() && node_it->second) {
                const auto &known_mediums = node_it->second->mediums;
                const bool all_known =
                    std::all_of(mediums.begin(), mediums.end(), [&known_mediums](const auto &medium) {
                        return std::find(known_mediums.begin(), known_mediums.end(), medium) != known_mediums.end();
                    });
                if (all_known) {
                    return EC_OK;
                }
            }
        }
    }

    // Enriching an existing node must not wait for the lifecycle write lock:
    // an in-flight metadata mutation intentionally holds a shared lifecycle
    // lease, and new deltas still need to reach the bounded snapshot gate
    // instead of blocking here indefinitely.
    {
        TimedLock<std::unique_lock<std::shared_mutex>> lock(nodes_mutex_, nodes_mutex_ensure_node_metrics_);
        if (!AcceptingReports()) {
            return EC_INSTANCE_NOT_EXIST;
        }
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
    if (!AcceptingReports()) {
        return EC_INSTANCE_NOT_EXIST;
    }
    TimedLock<std::unique_lock<std::shared_mutex>> lock(nodes_mutex_, nodes_mutex_ensure_node_metrics_);
    std::lock_guard<std::mutex> state_lock(lifecycle_fence->state_mutex);
    auto &host_map = instance_nodes_[instance_id];
    if (auto it = host_map.find(host_ip_port); it != host_map.end()) {
        // Another request may have created the node between the fast-path
        // check and acquiring the lifecycle lock.
        merge_mediums(*it->second);
        it->second->lifecycle_fence = lifecycle_fence;
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
    info->lifecycle_fence = lifecycle_fence;
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
    if (Retired()) {
        return EC_INSTANCE_NOT_EXIST;
    }
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    return UnregisterNodeLocked(instance_id, host_ip_port, *lifecycle_fence);
}

ErrorCode EventReportBackend::UnregisterNodeForHostDown(const std::string &instance_id,
                                                        const std::string &host_ip_port,
                                                        uint64_t &out_generation) {
    const ReporterSnapshotKey reporter_key{instance_id, host_ip_port};
    const auto lifecycle_fence = GetOrCreateLifecycleFence(reporter_key);
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_fence->mutex);
    if (!AcceptingReports()) {
        return EC_INSTANCE_NOT_EXIST;
    }
    std::unique_lock<std::shared_mutex> lock(nodes_mutex_);
    out_generation = node_generation_[instance_id][host_ip_port];
    const auto instance_it = instance_nodes_.find(instance_id);
    if (instance_it == instance_nodes_.end() || instance_it->second.find(host_ip_port) == instance_it->second.end()) {
        // HOST_DOWN is explicitly idempotent. Keep the tombstone generation
        // above, but do not emit the generic missing-node warning.
        std::unique_lock<std::mutex> state_lock(lifecycle_fence->state_mutex);
        lifecycle_fence->generation = out_generation;
        lifecycle_fence->registered = false;
        ReleaseSnapshotVersionToken(reporter_key, lifecycle_fence->snapshot_state.committed);
        ReleaseSnapshotVersionToken(reporter_key, lifecycle_fence->snapshot_state.in_flight);
        lifecycle_fence->snapshot_state = {};
        state_lock.unlock();
        lifecycle_fence->snapshot_state_cv.notify_all();
        return EC_OK;
    }
    return UnregisterNodeLocked(instance_id, host_ip_port, *lifecycle_fence);
}

ErrorCode EventReportBackend::UnregisterNodeIfGeneration(const std::string &instance_id,
                                                         const std::string &host_ip_port,
                                                         uint64_t expected_generation) {
    const ReporterSnapshotKey reporter_key{instance_id, host_ip_port};
    const auto lifecycle_fence = GetOrCreateLifecycleFence(reporter_key);
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_fence->mutex);
    if (Retired()) {
        return EC_INSTANCE_NOT_EXIST;
    }
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
    return UnregisterNodeLocked(instance_id, host_ip_port, *lifecycle_fence);
}

ErrorCode EventReportBackend::UnregisterNodeLocked(const std::string &instance_id,
                                                   const std::string &host_ip_port,
                                                   LifecycleFence &lifecycle_fence) {
    node_generation_[instance_id].try_emplace(host_ip_port, 0);
    std::unique_lock<std::mutex> state_lock(lifecycle_fence.state_mutex);
    lifecycle_fence.generation = node_generation_[instance_id][host_ip_port];
    lifecycle_fence.registered = false;
    const ReporterSnapshotKey reporter_key{instance_id, host_ip_port};
    ReleaseSnapshotVersionToken(reporter_key, lifecycle_fence.snapshot_state.committed);
    ReleaseSnapshotVersionToken(reporter_key, lifecycle_fence.snapshot_state.in_flight);
    lifecycle_fence.snapshot_state = {};
    state_lock.unlock();
    lifecycle_fence.snapshot_state_cv.notify_all();
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

    // Fast path: a steady HEARTBEAT does not change lifecycle state. Use a
    // shared lease so same-generation ADD/DELETE can proceed; it still pins
    // NodeInfo and excludes REGISTER/HOST_DOWN writers.
    {
        std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_fence->mutex);
        if (!AcceptingReports()) {
            return EC_INSTANCE_NOT_EXIST;
        }
        std::unique_lock<std::shared_mutex> nodes_lock(nodes_mutex_);
        if (TryPublishSteadyHeartbeatLocked(reporter_key, *lifecycle_fence, system_status, nodes_lock)) {
            return EC_OK;
        }
    }

    // Slow path: HEARTBEAT may create or recover a node and change lifecycle
    // state. Drop the shared lease, acquire it exclusively, and revalidate
    // because shared_mutex has no atomic lock upgrade.
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_fence->mutex);
    if (!AcceptingReports()) {
        return EC_INSTANCE_NOT_EXIST;
    }
    std::unique_lock<std::shared_mutex> nodes_lock(nodes_mutex_);
    std::unique_lock<std::mutex> state_lock(lifecycle_fence->state_mutex);
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
        new_info->lifecycle_fence = lifecycle_fence;
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
    info.lifecycle_fence = lifecycle_fence;
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
    state_lock.unlock();
    PublishHeartbeatStatus(info, system_status, nodes_lock);
    return EC_OK;
}

bool EventReportBackend::TryPublishSteadyHeartbeatLocked(const ReporterSnapshotKey &reporter_key,
                                                         const LifecycleFence &lifecycle_fence,
                                                         const std::map<std::string, std::string> &system_status,
                                                         std::unique_lock<std::shared_mutex> &nodes_lock) {
    std::unique_lock<std::mutex> state_lock(lifecycle_fence.state_mutex);
    if (!lifecycle_fence.registered) {
        return false;
    }
    const auto instance_it = instance_nodes_.find(reporter_key.instance_id);
    const auto generation_it = node_generation_.find(reporter_key.instance_id);
    if (instance_it == instance_nodes_.end() || generation_it == node_generation_.end()) {
        return false;
    }
    const auto node_it = instance_it->second.find(reporter_key.host_ip_port);
    const auto host_generation_it = generation_it->second.find(reporter_key.host_ip_port);
    if (node_it == instance_it->second.end() || !node_it->second || host_generation_it == generation_it->second.end() ||
        lifecycle_fence.generation != host_generation_it->second ||
        !node_it->second->available.load(std::memory_order_relaxed)) {
        return false;
    }
    auto &info = *node_it->second;
    info.last_heartbeat_ms.store(NowMillis(), std::memory_order_release);
    state_lock.unlock();
    PublishHeartbeatStatus(info, system_status, nodes_lock);
    return true;
}

void EventReportBackend::PublishHeartbeatStatus(NodeInfo &info,
                                                const std::map<std::string, std::string> &system_status,
                                                std::unique_lock<std::shared_mutex> &nodes_lock) {
    std::unique_lock<std::mutex> status_lock(info.status_mutex);
    const std::map<std::string, std::string> previous_system_status = info.last_system_status;
    info.last_system_status = system_status;
    const auto metrics_tags = info.metrics_tags;

    // Lock handoff: status_mutex now serializes this publication with
    // SetNodeUnavailable's gauge reset, while the lifecycle lease pins
    // NodeInfo. Release the global node-table lock before slower metric work.
    nodes_lock.unlock();
    if (metrics_registry_) {
        const auto parse_gauge = [](const std::string &value, double &out) {
            if (value.empty()) {
                return false;
            }
            char *end = nullptr;
            out = std::strtod(value.c_str(), &end);
            return end == value.c_str() + value.size() && std::isfinite(out);
        };
        const std::string prefix = "event_report.";
        // system_status is a full heartbeat snapshot, not a patch. Remove a
        // prior numeric gauge when the next heartbeat omits it or changes it
        // to a non-numeric value; otherwise stale values would survive and a
        // later unregister could no longer discover the omitted key.
        for (const auto &[name, previous_value] : previous_system_status) {
            double ignored_previous = 0.0;
            if (!parse_gauge(previous_value, ignored_previous)) {
                continue;
            }
            const auto current_it = system_status.find(name);
            double ignored_current = 0.0;
            if (current_it == system_status.end() || !parse_gauge(current_it->second, ignored_current)) {
                if (auto data = metrics_registry_->GetMetricsData(prefix + name)) {
                    data->RemoveByTags(metrics_tags);
                }
            }
        }
        for (const auto &kv : system_status) {
            double val = 0.0;
            if (parse_gauge(kv.second, val)) {
                REPORT_DYNAMIC_GAUGE_(metrics_registry_, prefix + kv.first, metrics_tags, val);
            }
        }
    }
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
                const ErrorCode unregister_ec = UnregisterNodeIfGeneration(entry.instance_id, entry.host, entry.gen);
                if (unregister_ec == EC_MISMATCH) {
                    const uint64_t current_gen = GetNodeGeneration(entry.instance_id, entry.host);
                    KVCM_LOG_INFO("EventReportBackend: node [%s] re-registered "
                                  "(gen=%" PRIu64 " -> %" PRIu64 "), skipping unregister",
                                  entry.host.c_str(),
                                  entry.gen,
                                  current_gen);
                    continue;
                }
                if (unregister_ec != EC_OK) {
                    KVCM_LOG_WARN("EventReportBackend: failed to unregister expired node [%s] instance [%s], "
                                  "ec=%d (gen=%" PRIu64 ")",
                                  entry.host.c_str(),
                                  entry.instance_id.c_str(),
                                  unregister_ec,
                                  entry.gen);
                    continue;
                }

                // Unregister is the liveness-expiry linearization point. It
                // must happen before cleanup is dispatched: otherwise cleanup
                // can delete metadata while a heartbeat waits behind its
                // lifecycle lease, then let that heartbeat revive the old
                // committed snapshot without requiring reconciliation.
                KVCM_LOG_WARN("EventReportBackend: node [%s] instance [%s] passed cleanup_grace_ms, "
                              "unregistered and triggering cleanup (gen=%" PRIu64 ")",
                              entry.host.c_str(),
                              entry.instance_id.c_str(),
                              entry.gen);
                if (cb_copy) {
                    cb_copy(entry.instance_id, entry.host, entry.gen);
                }
            }
        }

        std::unique_lock<std::mutex> wait_lock(liveness_wait_mutex_);
        liveness_wait_cv_.wait_for(wait_lock, std::chrono::milliseconds(liveness_check_interval_ms_), [this] {
            return !liveness_checker_running_.load(std::memory_order_acquire) || !IsOpen();
        });
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
    if (!AcceptingReports()) {
        return std::vector<bool>(storage_uris.size(), false);
    }
    std::vector<bool> result;
    result.reserve(storage_uris.size());
    for (const auto &uri : storage_uris) {
        SnapshotUriInfo info;
        if (!SnapshotUriUtils::ParseSnapshotUriInfo(uri, info)) {
            result.push_back(false);
            continue;
        }
        ReporterSnapshotKey reporter_key;
        {
            std::shared_lock<std::shared_mutex> token_lock(snapshot_token_owners_mutex_);
            const auto owner_it = snapshot_token_owners_.find(info.version);
            if (owner_it == snapshot_token_owners_.end()) {
                result.push_back(false);
                continue;
            }
            reporter_key = owner_it->second;
        }
        const auto lifecycle_fence = FindLifecycleFence(reporter_key);
        if (!lifecycle_fence) {
            result.push_back(false);
            continue;
        }

        std::shared_lock<std::shared_mutex> nodes_lock(nodes_mutex_);
        const auto instance_it = instance_nodes_.find(reporter_key.instance_id);
        if (instance_it == instance_nodes_.end()) {
            result.push_back(false);
            continue;
        }
        const auto node_it = instance_it->second.find(reporter_key.host_ip_port);
        if (node_it == instance_it->second.end() || !node_it->second ||
            !node_it->second->available.load(std::memory_order_relaxed)) {
            result.push_back(false);
            continue;
        }
        std::lock_guard<std::mutex> state_lock(lifecycle_fence->state_mutex);
        result.push_back(lifecycle_fence->registered && lifecycle_fence->snapshot_state.committed == info.version);
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
    std::string_view medium;
    std::string_view host_ip_port;
    if (!ParseLocationIdView(location_id, medium, host_ip_port)) {
        out_medium.clear();
        out_host_ip_port.clear();
        return false;
    }
    out_medium.assign(medium.data(), medium.size());
    out_host_ip_port.assign(host_ip_port.data(), host_ip_port.size());
    return true;
}

bool EventReportBackend::ParseLocationIdView(std::string_view location_id,
                                             std::string_view &out_medium,
                                             std::string_view &out_host_ip_port) const noexcept {
    std::string_view storage_type;
    if (!SnapshotUriUtils::ParseEventReportLocationIdView(location_id, storage_type, out_medium, out_host_ip_port)) {
        return false;
    }
    switch (config_.type()) {
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5:
        return storage_type == "event_report_l1p5";
    case DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2:
        return storage_type == "event_report_l2";
    default:
        return false;
    }
}

std::string EventReportBackend::HostSuffix(const std::string &host_ip_port) const { return "#" + host_ip_port; }

ErrorCode EventReportBackend::BeginDeltaMutation(const ReporterSnapshotKey &reporter_key,
                                                 std::string &out_committed_version,
                                                 uint64_t *out_lifecycle_generation,
                                                 bool *out_created_generation) {
    out_committed_version.clear();
    if (out_lifecycle_generation) {
        *out_lifecycle_generation = 0;
    }
    if (out_created_generation) {
        *out_created_generation = false;
    }
    if (reporter_key.instance_id.empty() || reporter_key.host_ip_port.empty()) {
        return EC_BADARGS;
    }
    if (!AcceptingReports()) {
        return EC_INSTANCE_NOT_EXIST;
    }
    const auto lifecycle_fence = FindLifecycleFence(reporter_key);
    if (!lifecycle_fence) {
        return EC_SNAPSHOT_REQUIRED;
    }
    TimedLock<std::unique_lock<std::mutex>> lock(lifecycle_fence->state_mutex, reporter_state_begin_metrics_);
    const int64_t snapshot_wait_timeout_ms = snapshot_delta_drain_timeout_ms_;
    const bool snapshot_finished =
        lifecycle_fence->snapshot_state_cv.wait_for(lock, std::chrono::milliseconds(snapshot_wait_timeout_ms), [&] {
            return !AcceptingReports() || !lifecycle_fence->registered ||
                   lifecycle_fence->snapshot_state.in_flight.empty();
        });
    if (!snapshot_finished) {
        return EC_SNAPSHOT_IN_PROGRESS;
    }
    // Close()/dynamic disable or unregister can wake this waiter. Recheck the
    // reporter-local predicate before creating or incrementing mutation state.
    if (!AcceptingReports()) {
        return EC_INSTANCE_NOT_EXIST;
    }
    if (!lifecycle_fence->registered) {
        return EC_SNAPSHOT_REQUIRED;
    }
    auto &state = lifecycle_fence->snapshot_state;
    if (state.committed.empty()) {
        state.committed = ReserveSnapshotVersionToken(reporter_key);
        if (out_created_generation) {
            *out_created_generation = true;
        }
    }
    if (state.active_delta_mutations == std::numeric_limits<uint64_t>::max()) {
        return EC_ERROR;
    }
    ++state.active_delta_mutations;
    out_committed_version = state.committed;
    if (out_lifecycle_generation) {
        *out_lifecycle_generation = lifecycle_fence->generation;
    }
    return EC_OK;
}

void EventReportBackend::EndDeltaMutation(const ReporterSnapshotKey &reporter_key,
                                          uint64_t lifecycle_generation,
                                          const std::string &expected_snapshot_version) {
    const auto lifecycle_fence = FindLifecycleFence(reporter_key);
    if (!lifecycle_fence) {
        KVCM_LOG_DEBUG("EventReportBackend: delta mutation lease ended after reporter lifecycle changed, "
                       "instance [%s] host [%s]",
                       reporter_key.instance_id.c_str(),
                       reporter_key.host_ip_port.c_str());
        return;
    }
    TimedLock<std::unique_lock<std::mutex>> lock(lifecycle_fence->state_mutex, reporter_state_end_metrics_);
    auto &state = lifecycle_fence->snapshot_state;
    if (!expected_snapshot_version.empty() && state.committed != expected_snapshot_version) {
        // HOST_DOWN removes snapshot state before an already-admitted delta
        // necessarily reaches its final metadata lease. If the reporter is
        // then registered again, a new delta can recreate state at the same
        // key. The old guard must not drain that newer lifecycle's admission.
        KVCM_LOG_DEBUG("EventReportBackend: ignoring stale delta mutation lease for instance [%s] host [%s]",
                       reporter_key.instance_id.c_str(),
                       reporter_key.host_ip_port.c_str());
        return;
    }
    if (state.active_delta_mutations == 0) {
        const bool lifecycle_ended = !lifecycle_fence->registered ||
                                     (lifecycle_generation != 0 && lifecycle_fence->generation != lifecycle_generation);
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
    --state.active_delta_mutations;
    const bool drained = state.active_delta_mutations == 0;
    lock.unlock();
    if (drained) {
        lifecycle_fence->snapshot_state_cv.notify_all();
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
    // The in-flight token/attempt epoch is the snapshot-cleanup fence. Change
    // it while holding the reporter lifecycle writer so an older cleanup can
    // either acquire its read lease first and finish, or observe the newer
    // epoch after this transition; it can never delete across the boundary.
    const auto lifecycle_fence = GetOrCreateLifecycleFence(reporter_key);
    std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_fence->mutex);
    std::unique_lock<std::mutex> lock(lifecycle_fence->state_mutex);
    if (!AcceptingReports()) {
        return EC_INSTANCE_NOT_EXIST;
    }
    if (!lifecycle_fence->registered) {
        return EC_SNAPSHOT_REQUIRED;
    }
    const uint64_t admitted_lifecycle_generation = lifecycle_fence->generation;
    auto &state = lifecycle_fence->snapshot_state;
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
    out_candidate_version = ReserveSnapshotVersionToken(reporter_key);
    // Close the reporter's write gate before waiting for already admitted
    // deltas. Deltas arriving from this point wait until commit or abort.
    ++state.attempt_epoch;
    state.in_flight = out_candidate_version;
    // Do not retain the lifecycle writer while draining already-admitted
    // deltas: their final metadata phase needs a lifecycle read lease before
    // it can end its delta admission and wake this waiter.
    lifecycle_lock.unlock();
    const int64_t delta_drain_timeout_ms = snapshot_delta_drain_timeout_ms_;
    const bool deltas_drained =
        lifecycle_fence->snapshot_state_cv.wait_for(lock, std::chrono::milliseconds(delta_drain_timeout_ms), [&] {
            return !AcceptingReports() || !lifecycle_fence->registered || state.in_flight != out_candidate_version ||
                   state.active_delta_mutations == 0;
        });
    // The wait releases the reporter state mutex. If the backend was retired or disabled
    // meanwhile, do not return a usable candidate.  Close() has already
    // cleared the state; for a dynamic disable, reopen the reporter write gate
    // explicitly so a later re-enable is not stuck behind this abandoned
    // candidate.
    if (!AcceptingReports()) {
        if (state.in_flight == out_candidate_version) {
            state.in_flight.clear();
            ReleaseSnapshotVersionToken(reporter_key, out_candidate_version);
        }
        out_candidate_version.clear();
        lock.unlock();
        lifecycle_fence->snapshot_state_cv.notify_all();
        return EC_INSTANCE_NOT_EXIST;
    }
    if (!deltas_drained) {
        const uint64_t active_delta_mutations = state.active_delta_mutations;
        if (state.in_flight == out_candidate_version) {
            state.in_flight.clear();
            ReleaseSnapshotVersionToken(reporter_key, out_candidate_version);
        }
        out_candidate_version.clear();
        lock.unlock();
        lifecycle_fence->snapshot_state_cv.notify_all();
        KVCM_LOG_WARN("EventReportBackend: snapshot admission timed out after %" PRId64 "ms waiting for %" PRIu64
                      " active delta mutation(s), instance [%s] host [%s]; "
                      "candidate aborted and write gate reopened",
                      delta_drain_timeout_ms,
                      active_delta_mutations,
                      reporter_key.instance_id.c_str(),
                      reporter_key.host_ip_port.c_str());
        return EC_SNAPSHOT_IN_PROGRESS;
    }
    if (!lifecycle_fence->registered || state.in_flight != out_candidate_version) {
        out_candidate_version.clear();
        return EC_SNAPSHOT_REQUIRED;
    }
    if (out_lifecycle_generation) {
        *out_lifecycle_generation = admitted_lifecycle_generation;
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
    // ADD/DELETE mutate metadata, not reporter lifecycle. This shared lease
    // pins and validates the current registration/generation while the
    // metadata RMW is in progress.
    auto lease = std::make_shared<std::shared_lock<std::shared_mutex>>(lifecycle_fence->mutex, std::try_to_lock);
    if (!lease->owns_lock()) {
        // Do not wait behind a lifecycle writer while the caller may already
        // hold metadata locks. Cleanup takes the opposite order
        // (lifecycle -> metadata), so blocking here would deadlock.
        return EC_NODE_NOT_REGISTERED;
    }
    if (!AcceptingReports()) {
        return EC_INSTANCE_NOT_EXIST;
    }
    {
        std::lock_guard<std::mutex> state_lock(lifecycle_fence->state_mutex);
        if (!lifecycle_fence->registered || lifecycle_fence->generation != expected_generation) {
            return EC_NODE_NOT_REGISTERED;
        }
    }
    out_lease = std::move(lease);
    return EC_OK;
}

ErrorCode EventReportBackend::CommitSnapshotVersionIfGeneration(const ReporterSnapshotKey &reporter_key,
                                                                const std::string &version,
                                                                uint64_t expected_generation) {
    const auto lifecycle_fence = FindLifecycleFence(reporter_key);
    if (!lifecycle_fence) {
        return EC_NODE_NOT_REGISTERED;
    }
    // Commit runs after the metadata RMW has released its shard locks, so it
    // can safely wait for a transient lifecycle writer. Using the mutation
    // path's try-lock here would turn harmless lock contention into a failed
    // snapshot. REGISTER/HOST_DOWN and a lifecycle-changing HEARTBEAT still
    // serialize first and are rejected by the generation/registered check
    // below.
    std::shared_lock<std::shared_mutex> lifecycle_lease(lifecycle_fence->mutex);
    if (!AcceptingReports()) {
        return EC_INSTANCE_NOT_EXIST;
    }
    {
        std::lock_guard<std::mutex> state_lock(lifecycle_fence->state_mutex);
        if (!lifecycle_fence->registered || lifecycle_fence->generation != expected_generation) {
            return EC_NODE_NOT_REGISTERED;
        }
    }
    return CommitSnapshotVersion(reporter_key, version) ? EC_OK : EC_ERROR;
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
    // Host cleanup is only valid after HOST_DOWN/liveness has atomically
    // unregistered this exact reporter generation.  Checking `registered`
    // under the retained lifecycle lease prevents an accidental cleanup caller
    // from deleting metadata that still belongs to an active reporter, while
    // the generation check fences a concurrent re-registration.
    {
        std::lock_guard<std::mutex> state_lock(lifecycle_fence->state_mutex);
        if (!AcceptingReports() || lifecycle_fence->registered || lifecycle_fence->generation != expected_generation) {
            return EC_MISMATCH;
        }
    }
    out_lease = std::move(lease);
    return EC_OK;
}

ErrorCode EventReportBackend::AcquireSnapshotCleanupLease(const ReporterSnapshotKey &reporter_key,
                                                          uint64_t expected_generation,
                                                          const std::string &expected_snapshot_version,
                                                          uint64_t expected_attempt_epoch,
                                                          LifecycleMutationLease &out_lease) const {
    out_lease.reset();
    const auto lifecycle_fence = FindLifecycleFence(reporter_key);
    if (!lifecycle_fence) {
        return EC_MISMATCH;
    }
    // Cleanup acquires this lease before taking metadata locks. It can block
    // behind a short-lived REGISTER or lifecycle-changing HEARTBEAT writer
    // without creating the lifecycle->metadata / metadata->lifecycle
    // inversion that forces delta mutations to use try_lock.
    auto lease = std::make_shared<std::shared_lock<std::shared_mutex>>(lifecycle_fence->mutex);
    // BeginSnapshot publishes a new attempt epoch under the lifecycle writer
    // before releasing it. Holding the read lease here therefore makes this
    // validation atomic with respect to every later snapshot admission.
    std::lock_guard<std::mutex> state_lock(lifecycle_fence->state_mutex);
    const auto &state = lifecycle_fence->snapshot_state;
    if (!AcceptingReports() || !lifecycle_fence->registered || lifecycle_fence->generation != expected_generation ||
        state.committed != expected_snapshot_version ||
        (expected_attempt_epoch != 0 && state.attempt_epoch != expected_attempt_epoch)) {
        return EC_MISMATCH;
    }
    out_lease = std::move(lease);
    return EC_OK;
}

bool EventReportBackend::CommitSnapshotVersion(const ReporterSnapshotKey &reporter_key, const std::string &version) {
    if (!SnapshotUriUtils::IsValidSnapshotVersionToken(version)) {
        return false;
    }
    const auto lifecycle_fence = FindLifecycleFence(reporter_key);
    if (!lifecycle_fence) {
        return false;
    }
    std::unique_lock<std::mutex> lock(lifecycle_fence->state_mutex);
    auto &state = lifecycle_fence->snapshot_state;
    if (state.in_flight != version) {
        return false;
    }
    ReleaseSnapshotVersionToken(reporter_key, state.committed);
    state.committed = version;
    state.in_flight.clear();
    state.last_commit_ms = NowMillis();
    state.strict_query_visibility = true;
    lock.unlock();
    lifecycle_fence->snapshot_state_cv.notify_all();
    return true;
}

void EventReportBackend::AbortSnapshotVersion(const ReporterSnapshotKey &reporter_key, const std::string &version) {
    if (!SnapshotUriUtils::IsValidSnapshotVersionToken(version) || reporter_key.instance_id.empty() ||
        reporter_key.host_ip_port.empty()) {
        return;
    }
    const auto lifecycle_fence = FindLifecycleFence(reporter_key);
    if (!lifecycle_fence) {
        return;
    }
    std::unique_lock<std::mutex> lock(lifecycle_fence->state_mutex);
    auto &state = lifecycle_fence->snapshot_state;
    if (state.in_flight == version) {
        state.in_flight.clear();
        // Candidate metadata is written in place. Once an admitted attempt
        // aborts, accepting only committed could hide locations already
        // replaced with the failed candidate. Stay soft until a later
        // successful complete snapshot restores an authoritative fence.
        state.strict_query_visibility = false;
        ReleaseSnapshotVersionToken(reporter_key, version);
        lock.unlock();
        lifecycle_fence->snapshot_state_cv.notify_all();
    }
}

std::string EventReportBackend::GetSnapshotVersion(const ReporterSnapshotKey &reporter_key) const {
    const auto lifecycle_fence = FindLifecycleFence(reporter_key);
    if (!lifecycle_fence) {
        return {};
    }
    TimedLock<std::unique_lock<std::mutex>> lock(lifecycle_fence->state_mutex, reporter_state_snapshot_get_metrics_);
    return lifecycle_fence->snapshot_state.committed;
}

void EventReportBackend::GetSnapshotVersionTokens(const ReporterSnapshotKey &reporter_key,
                                                  std::string &out_committed,
                                                  std::string &out_in_flight) const {
    out_committed.clear();
    out_in_flight.clear();
    const auto lifecycle_fence = FindLifecycleFence(reporter_key);
    if (!lifecycle_fence) {
        return;
    }
    std::lock_guard<std::mutex> lock(lifecycle_fence->state_mutex);
    out_committed = lifecycle_fence->snapshot_state.committed;
    out_in_flight = lifecycle_fence->snapshot_state.in_flight;
}

bool EventReportBackend::GetQueryVisibilityState(const ReporterSnapshotKey &reporter_key,
                                                 bool &out_strict,
                                                 std::string &out_committed) const {
    out_strict = false;
    out_committed.clear();
    if (!AcceptingReports()) {
        return false;
    }
    const auto lifecycle_fence = FindLifecycleFence(reporter_key);
    if (!lifecycle_fence) {
        return false;
    }
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
    std::lock_guard<std::mutex> state_lock(lifecycle_fence->state_mutex);
    if (!lifecycle_fence->registered) {
        return false;
    }
    out_strict = lifecycle_fence->snapshot_state.strict_query_visibility;
    out_committed = lifecycle_fence->snapshot_state.committed;
    return true;
}

void EventReportBackend::GetQueryVisibilitySnapshot(const std::string &instance_id,
                                                    QueryVisibilitySnapshot &out_snapshot) const {
    out_snapshot.clear();
    if (!AcceptingReports()) {
        return;
    }
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    const auto instance_it = instance_nodes_.find(instance_id);
    if (instance_it == instance_nodes_.end()) {
        return;
    }
    for (const auto &[host_ip_port, node] : instance_it->second) {
        if (!node || !node->available.load(std::memory_order_relaxed)) {
            continue;
        }
        const auto lifecycle_fence = node->lifecycle_fence;
        if (!lifecycle_fence) {
            continue;
        }
        std::lock_guard<std::mutex> state_lock(lifecycle_fence->state_mutex);
        if (!lifecycle_fence->registered) {
            continue;
        }
        QueryVisibilityState state;
        state.strict = lifecycle_fence->snapshot_state.strict_query_visibility;
        state.committed_version = lifecycle_fence->snapshot_state.committed;
        out_snapshot.emplace(host_ip_port, std::move(state));
    }
}

uint64_t EventReportBackend::GetSnapshotAttemptEpoch(const ReporterSnapshotKey &reporter_key) const {
    const auto lifecycle_fence = FindLifecycleFence(reporter_key);
    if (!lifecycle_fence) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(lifecycle_fence->state_mutex);
    return lifecycle_fence->snapshot_state.attempt_epoch;
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
