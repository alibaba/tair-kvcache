#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/loop_thread.h"
#include "kv_cache_manager/data_storage/event_report_backend.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/types.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

inline constexpr int64_t kMinCacheGcOrphanWritingGracePeriodMs = 60LL * 60 * 1000;

#ifndef KVCM_COUNTER_METRICS_FOR_CACHE_GC
#define KVCM_COUNTER_METRICS_FOR_CACHE_GC(name)                                                                        \
public:                                                                                                                \
    DECLARE_METRICS_NAME_(cache_gc, name);                                                                             \
    DEFINE_GET_METRICS_COUNTER_(cache_gc, name)                                                                        \
                                                                                                                       \
private:                                                                                                               \
    DECLARE_METRICS_COUNTER_(cache_gc, name);
#endif

#ifndef KVCM_GAUGE_METRICS_FOR_CACHE_GC
#define KVCM_GAUGE_METRICS_FOR_CACHE_GC(name)                                                                          \
public:                                                                                                                \
    DECLARE_METRICS_NAME_(cache_gc, name);                                                                             \
    DEFINE_GET_METRICS_GAUGE_(cache_gc, name)                                                                          \
                                                                                                                       \
private:                                                                                                               \
    DECLARE_METRICS_GAUGE_(cache_gc, name);
#endif

class CacheLocation;
class DataStorageManager;
class MetaIndexerManager;
class MigrationManager;
class RegistryManager;
class RequestContext;
struct MaintenanceScanBatch;

class CacheGarbageCollector {
public:
    struct Config {
        bool enabled{false};
        int64_t scan_interval_ms{1000};
        int64_t round_pause_ms{2LL * 60 * 60 * 1000};
        size_t scan_batch_size{256};
        int64_t orphan_writing_grace_period_ms{24LL * 60 * 60 * 1000};
        size_t max_inflight_delete_requests{2};
        bool event_report_cleanup_enabled{true};
        size_t event_report_action_batch_size{32};
    };

    CacheGarbageCollector() = delete;
    CacheGarbageCollector(Config config,
                          std::shared_ptr<RegistryManager> registry_manager,
                          std::shared_ptr<MetaIndexerManager> meta_indexer_manager,
                          std::shared_ptr<DataStorageManager> data_storage_manager,
                          std::shared_ptr<SchedulePlanExecutor> schedule_plan_executor,
                          std::shared_ptr<MetricsRegistry> metrics_registry,
                          std::shared_ptr<MigrationManager> migration_manager);
    ~CacheGarbageCollector();

    CacheGarbageCollector(const CacheGarbageCollector &) = delete;
    CacheGarbageCollector(CacheGarbageCollector &&) = delete;
    CacheGarbageCollector &operator=(const CacheGarbageCollector &) = delete;
    CacheGarbageCollector &operator=(CacheGarbageCollector &&) = delete;

    [[nodiscard]] ErrorCode Validate() const noexcept;
    [[nodiscard]] ErrorCode Start() noexcept;
    void RequestStop() noexcept;
    void Join() noexcept;
    void Stop() noexcept;
    [[nodiscard]] bool IsRunning() const noexcept;
    [[nodiscard]] bool IsEnabled() const noexcept { return config_.enabled; }
    [[nodiscard]] bool IsEventReportCleanupEnabled() const noexcept {
        return config_.enabled && config_.event_report_cleanup_enabled;
    }

private:
    using Clock = std::chrono::steady_clock;
    static constexpr size_t kMaxScanFailuresPerInstancePerRound = 3;

    struct InstanceScanEntry {
        std::string instance_group;
        std::string instance_id;
        std::string cursor{SCAN_BASE_CURSOR};
        bool completed{false};
        size_t scan_failure_count{0};

        bool operator<(const InstanceScanEntry &other) const {
            return std::tie(instance_group, instance_id) < std::tie(other.instance_group, other.instance_id);
        }
        bool operator==(const InstanceScanEntry &other) const {
            return instance_group == other.instance_group && instance_id == other.instance_id;
        }
    };

    struct PendingLocationKey {
        std::string instance_id;
        int64_t block_key{0};
        std::string location_id;

        bool operator<(const PendingLocationKey &other) const noexcept {
            return std::tie(instance_id, block_key, location_id) <
                   std::tie(other.instance_id, other.block_key, other.location_id);
        }
    };

    struct InflightDelete {
        uint64_t round_id{0};
        std::string instance_id;
        std::string action_name;
        size_t target_count{0};
        Clock::time_point submitted_at;
        std::vector<PendingLocationKey> pending_locations;
        std::future<PlanExecuteResult> future;
    };

    enum class EventReportBackendRouteStatus {
        kResolved,
        kMissing,
        kAmbiguous,
        kUnavailable,
    };

    struct EventReportBackendRoute {
        EventReportBackendRouteStatus status{EventReportBackendRouteStatus::kMissing};
        std::string backend_unique_name;
        std::shared_ptr<EventReportBackend> backend;
    };

    struct ScanDeleteActions {
        CacheLocationDelRequest executor_request;
        EventReportMetadataDelRequest event_report_request;
    };

    void RunOneTick() noexcept;
    EventReportBackendRoute LookupEventReportBackend(const std::string &instance_id,
                                                      DataStorageType storage_type) const;
    void PollInflightDeletes() noexcept;
    void UpdateInflightDeleteMetrics() noexcept;
    void ReleasePendingLocations(const InflightDelete &inflight) noexcept;
    bool BeginRound();
    void CompleteRound() noexcept;
    void AdvanceInstance(bool completed_current) noexcept;
    ScanDeleteActions
    BuildDeleteActions(const std::string &instance_id, const MaintenanceScanBatch &batch, int64_t now_us);
    bool
    IsOrphanWriting(const std::string &map_location_id, const CacheLocation &location, int64_t now_us) const noexcept;
    void ResetWorkerState() noexcept;
    void RegisterMetrics();
    void RecordCandidateCount(const std::string &reason, size_t count) noexcept;
    void RecordCandidateDropped(const std::string &reason, const std::string &cause, size_t count) noexcept;
    void RecordEventReportProbe(const std::string &result, const std::string &unknown_cause = {}) noexcept;
    void RecordOperationError(const std::string &stage) noexcept;
    void RecordDeleteResult(const std::string &status) noexcept;

    const Config config_;
    const std::shared_ptr<RegistryManager> registry_manager_;
    const std::shared_ptr<MetaIndexerManager> meta_indexer_manager_;
    const std::shared_ptr<DataStorageManager> data_storage_manager_;
    const std::shared_ptr<SchedulePlanExecutor> schedule_plan_executor_;
    const std::shared_ptr<MetricsRegistry> metrics_registry_;
    const std::shared_ptr<MigrationManager> migration_manager_;

    mutable std::mutex lifecycle_mutex_;
    std::shared_ptr<LoopThread> worker_;
    std::atomic<bool> stop_requested_{true};
    bool running_{false};

    // Worker-thread-only state.
    std::vector<InstanceScanEntry> instances_;
    size_t instance_index_{0};
    uint64_t round_id_{0};
    bool round_active_{false};
    Clock::time_point round_started_at_;
    std::optional<Clock::time_point> next_round_at_;
    std::shared_ptr<RequestContext> round_context_;
    std::vector<InflightDelete> inflight_deletes_;
    std::set<PendingLocationKey> pending_locations_;

    KVCM_COUNTER_METRICS_FOR_CACHE_GC(scan_round_count)
    KVCM_COUNTER_METRICS_FOR_CACHE_GC(scan_key_count)
    KVCM_COUNTER_METRICS_FOR_CACHE_GC(candidate_count)
    KVCM_COUNTER_METRICS_FOR_CACHE_GC(delete_target_count)
    KVCM_GAUGE_METRICS_FOR_CACHE_GC(inflight_delete_count)
    KVCM_GAUGE_METRICS_FOR_CACHE_GC(inflight_delete_age_ms)
    KVCM_GAUGE_METRICS_FOR_CACHE_GC(round_duration_ms)
};

#undef KVCM_COUNTER_METRICS_FOR_CACHE_GC
#undef KVCM_GAUGE_METRICS_FOR_CACHE_GC

} // namespace kv_cache_manager
