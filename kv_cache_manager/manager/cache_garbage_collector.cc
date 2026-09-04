#include "kv_cache_manager/manager/cache_garbage_collector.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/event_report_backend.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/migration_manager.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/meta/types.h"

namespace kv_cache_manager {

#define DEFINE_METRICS_NAME_FOR_CACHE_GC(name) DEFINE_METRICS_NAME_(CacheGarbageCollector, cache_gc, name)
#define REGISTER_COUNTER_METRICS_FOR_CACHE_GC(name) REGISTER_METRICS_COUNTER_(metrics_registry_, cache_gc, name)
#define REGISTER_GAUGE_METRICS_FOR_CACHE_GC(name) REGISTER_METRICS_GAUGE_(metrics_registry_, cache_gc, name)

DEFINE_METRICS_NAME_FOR_CACHE_GC(scan_round_count);
DEFINE_METRICS_NAME_FOR_CACHE_GC(scan_key_count);
DEFINE_METRICS_NAME_FOR_CACHE_GC(candidate_count);
DEFINE_METRICS_NAME_FOR_CACHE_GC(delete_target_count);
DEFINE_METRICS_NAME_FOR_CACHE_GC(inflight_delete_count);
DEFINE_METRICS_NAME_FOR_CACHE_GC(inflight_delete_age_ms);
DEFINE_METRICS_NAME_FOR_CACHE_GC(round_duration_ms);

namespace {

constexpr const char *kDeleteResultMetricName = "cache_gc.delete_result_count";
constexpr const char *kOperationErrorMetricName = "cache_gc.operation_error_count";
constexpr const char *kOrphanWritingReason = "orphan_writing";
constexpr const char *kStorageMissingReason = "storage_missing";
constexpr const char *kEventReportStaleSnapshotReason = "event_report_stale_snapshot";
constexpr const char *kEventReportDownHostReason = "event_report_down_host";
constexpr const char *kEventReportRecoveryAbsentHostReason = "event_report_recovery_absent_host";
constexpr const char *kCandidateDroppedMetricName = "cache_gc.candidate_dropped_count";
constexpr const char *kEventReportProbeMetricName = "cache_gc.event_report_probe_count";
constexpr const char *kEventReportProbeUnknownMetricName = "cache_gc.event_report_probe_unknown_count";
// DataStorageManager keeps a shared registry lock while invoking MightExist().
constexpr size_t kMightExistProbeBatchSize = 512;
constexpr size_t kEventReportProbeBatchSize = 512;

using CandidateKey = std::pair<KeyType, std::string>;
using StorageProbeKey = std::tuple<std::string, DataStorageType>;

enum class CandidateReason {
    kOrphanWriting,
    kStorageMissing,
    kEventReportStaleSnapshot,
    kEventReportDownHost,
    kEventReportRecoveryAbsentHost,
};

const char *CandidateReasonName(CandidateReason reason) {
    switch (reason) {
    case CandidateReason::kOrphanWriting:
        return kOrphanWritingReason;
    case CandidateReason::kStorageMissing:
        return kStorageMissingReason;
    case CandidateReason::kEventReportDownHost:
        return kEventReportDownHostReason;
    case CandidateReason::kEventReportRecoveryAbsentHost:
        return kEventReportRecoveryAbsentHostReason;
    case CandidateReason::kEventReportStaleSnapshot:
        return kEventReportStaleSnapshotReason;
    }
    return "unknown";
}

int CandidatePriority(CandidateReason reason) {
    switch (reason) {
    case CandidateReason::kOrphanWriting:
        return 0;
    case CandidateReason::kStorageMissing:
        return 1;
    case CandidateReason::kEventReportDownHost:
        return 2;
    case CandidateReason::kEventReportRecoveryAbsentHost:
        return 3;
    case CandidateReason::kEventReportStaleSnapshot:
        return 4;
    }
    return 5;
}

bool IsEventReportReason(CandidateReason reason) {
    return reason == CandidateReason::kEventReportStaleSnapshot || reason == CandidateReason::kEventReportDownHost ||
           reason == CandidateReason::kEventReportRecoveryAbsentHost;
}

CandidateReason EventReportTokenReason(const EventReportBackend::MaintenanceCleanupToken &token) {
    switch (token.reason) {
    case EventReportBackend::MaintenanceCleanupReason::kStaleSnapshot:
        return CandidateReason::kEventReportStaleSnapshot;
    case EventReportBackend::MaintenanceCleanupReason::kDownHost:
        return CandidateReason::kEventReportDownHost;
    case EventReportBackend::MaintenanceCleanupReason::kRecoveryAbsentHost:
        return CandidateReason::kEventReportRecoveryAbsentHost;
    }
    return CandidateReason::kEventReportStaleSnapshot;
}

const char *MaintenanceUnknownReasonName(EventReportBackend::MaintenanceProbeUnknownReason reason) {
    switch (reason) {
    case EventReportBackend::MaintenanceProbeUnknownReason::kNone:
        return "unspecified";
    case EventReportBackend::MaintenanceProbeUnknownReason::kBackendUnavailable:
        return "backend_unavailable";
    case EventReportBackend::MaintenanceProbeUnknownReason::kReporterIdentityMalformed:
        return "reporter_identity_malformed";
    case EventReportBackend::MaintenanceProbeUnknownReason::kSnapshotState:
        return "snapshot_state";
    case EventReportBackend::MaintenanceProbeUnknownReason::kLocationMalformed:
        return "malformed";
    case EventReportBackend::MaintenanceProbeUnknownReason::kRecoveryGrace:
        return "recovery_grace";
    }
    return "unspecified";
}

struct DeleteCandidate {
    std::string expected_location_value;
    CandidateReason reason{CandidateReason::kOrphanWriting};
    std::set<std::string> confirmed_missing_uris;
    std::shared_ptr<EventReportBackend> event_report_backend;
    std::string event_report_backend_unique_name;
    std::optional<EventReportBackend::MaintenanceCleanupToken> event_report_cleanup_token;
};

struct ServingProbe {
    CandidateKey key;
    CacheLocationConstPtr location;
    std::set<std::string> confirmed_missing_uris;
};

struct StorageProbeBatch {
    std::vector<DataStorageUri> uris;
    // Parallel to uris; multiple specs can refer to the same Location.
    std::vector<size_t> serving_probe_indexes;
};

struct EventReportProbeEntry {
    CandidateKey key;
    CacheLocationConstPtr location;
};

struct EventReportProbeBatch {
    std::shared_ptr<EventReportBackend> backend;
    std::string backend_unique_name;
    const char *route_unknown_cause{nullptr};
    bool route_checked{false};
    std::vector<EventReportBackend::MaintenanceLocationProbe> probes;
    std::vector<size_t> probe_indexes;
};

} // namespace

CacheGarbageCollector::CacheGarbageCollector(Config config,
                                             std::shared_ptr<RegistryManager> registry_manager,
                                             std::shared_ptr<MetaIndexerManager> meta_indexer_manager,
                                             std::shared_ptr<DataStorageManager> data_storage_manager,
                                             std::shared_ptr<SchedulePlanExecutor> schedule_plan_executor,
                                             std::shared_ptr<MetricsRegistry> metrics_registry,
                                             std::shared_ptr<MigrationManager> migration_manager)
    : config_(std::move(config))
    , registry_manager_(std::move(registry_manager))
    , meta_indexer_manager_(std::move(meta_indexer_manager))
    , data_storage_manager_(std::move(data_storage_manager))
    , schedule_plan_executor_(std::move(schedule_plan_executor))
    , metrics_registry_(std::move(metrics_registry))
    , migration_manager_(std::move(migration_manager)) {}

CacheGarbageCollector::~CacheGarbageCollector() { Stop(); }

ErrorCode CacheGarbageCollector::Validate() const noexcept {
    if (!config_.enabled) {
        return EC_OK;
    }
    if (!registry_manager_ || !meta_indexer_manager_ || !data_storage_manager_ || !schedule_plan_executor_ ||
        !metrics_registry_ || !migration_manager_) {
        KVCM_LOG_ERROR("cache gc enabled with missing dependency");
        return EC_CONFIG_ERROR;
    }
    if (config_.scan_interval_ms <= 0 || config_.round_pause_ms < 0 || config_.scan_batch_size == 0 ||
        config_.max_inflight_delete_requests == 0 ||
        config_.orphan_writing_grace_period_ms < kMinCacheGcOrphanWritingGracePeriodMs ||
        (config_.event_report_cleanup_enabled && config_.event_report_action_batch_size == 0)) {
        KVCM_LOG_ERROR("invalid cache gc config, scan_interval_ms[%ld], round_pause_ms[%ld], scan_batch_size[%zu], "
                       "max_inflight_delete_requests[%zu], orphan_writing_grace_period_ms[%ld], "
                       "event_report_action_batch_size[%zu], grace must be at least[%ld]",
                       config_.scan_interval_ms,
                       config_.round_pause_ms,
                       config_.scan_batch_size,
                       config_.max_inflight_delete_requests,
                       config_.orphan_writing_grace_period_ms,
                       config_.event_report_action_batch_size,
                       kMinCacheGcOrphanWritingGracePeriodMs);
        return EC_CONFIG_ERROR;
    }
    if (config_.scan_batch_size > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        KVCM_LOG_ERROR("invalid cache gc scan_batch_size[%zu]", config_.scan_batch_size);
        return EC_CONFIG_ERROR;
    }
    const auto max_steady_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::duration::max()).count() / 2;
    if (config_.scan_interval_ms > max_steady_ms || config_.round_pause_ms > max_steady_ms) {
        KVCM_LOG_ERROR("cache gc duration config overflows steady clock");
        return EC_CONFIG_ERROR;
    }
    return EC_OK;
}

void CacheGarbageCollector::RegisterMetrics() {
    REGISTER_COUNTER_METRICS_FOR_CACHE_GC(scan_round_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_GC(scan_key_count);
    METRICS_(cache_gc, candidate_count) = metrics_registry_->GetCounter(METRICS_NAME_(cache_gc, candidate_count),
                                                                        MetricsTags{{"reason", kOrphanWritingReason}});
    REGISTER_COUNTER_METRICS_FOR_CACHE_GC(delete_target_count);
    REGISTER_GAUGE_METRICS_FOR_CACHE_GC(inflight_delete_count);
    REGISTER_GAUGE_METRICS_FOR_CACHE_GC(inflight_delete_age_ms);
    REGISTER_GAUGE_METRICS_FOR_CACHE_GC(round_duration_ms);
}

ErrorCode CacheGarbageCollector::Start() noexcept {
    if (!config_.enabled) {
        return EC_OK;
    }
    if (Validate() != EC_OK) {
        return EC_CONFIG_ERROR;
    }

    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (running_) {
        return EC_OK;
    }

    try {
        RegisterMetrics();
        ResetWorkerState();
        stop_requested_.store(false, std::memory_order_release);
        if (config_.event_report_cleanup_enabled) {
            // Backend objects can be opened well before leader recovery
            // finishes. Start recovery-absent grace when this leader is about
            // to scan and admit ReportEvent requests, not at process/backend
            // construction time.
            for (const auto &storage : data_storage_manager_->GetAvailableStorages()) {
                if (const auto backend = std::dynamic_pointer_cast<EventReportBackend>(storage)) {
                    backend->ResetMaintenanceRecoveryGrace();
                }
            }
        }
        worker_ = LoopThread::CreateLoopThread(
            [this]() { RunOneTick(); }, config_.scan_interval_ms * 1000, "CacheGarbageCollector", true);
        if (!worker_) {
            stop_requested_.store(true, std::memory_order_release);
            KVCM_LOG_ERROR("start cache gc worker failed: LoopThread creation returned null");
            return EC_ERROR;
        }
        running_ = true;
    } catch (const std::exception &e) {
        worker_.reset();
        running_ = false;
        stop_requested_.store(true, std::memory_order_release);
        KVCM_LOG_ERROR("start cache gc worker failed: %s", e.what());
        return EC_ERROR;
    } catch (...) {
        worker_.reset();
        running_ = false;
        stop_requested_.store(true, std::memory_order_release);
        KVCM_LOG_ERROR("start cache gc worker failed with unknown exception");
        return EC_ERROR;
    }
    KVCM_LOG_INFO("cache gc worker started");
    return EC_OK;
}

void CacheGarbageCollector::RequestStop() noexcept {
    stop_requested_.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (worker_) {
        // Wake a long tick/cooldown wait without joining here. RunOneTick observes
        // stop_requested_ and returns without touching GC dependencies.
        worker_->RunOnce();
    }
}

void CacheGarbageCollector::Join() noexcept {
    stop_requested_.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (worker_) {
        worker_->Stop();
        worker_.reset();
    }
    // Match CacheReclaimer's demotion policy: accepted actions continue best effort in the Executor, while GC never
    // waits for their end-to-end Futures. Clearing these observer handles does not cancel physical deletes or
    // EventReport metadata actions. Both paths revalidate their mutation preconditions in the Executor worker.
    inflight_deletes_.clear();
    pending_locations_.clear();
    METRICS_(cache_gc, inflight_delete_count) = 0;
    METRICS_(cache_gc, inflight_delete_age_ms) = 0;
    running_ = false;
}

void CacheGarbageCollector::Stop() noexcept {
    RequestStop();
    Join();
}

bool CacheGarbageCollector::IsRunning() const noexcept {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    return running_;
}

void CacheGarbageCollector::ResetWorkerState() noexcept {
    instances_.clear();
    instance_index_ = 0;
    round_active_ = false;
    next_round_at_.reset();
    round_context_.reset();
    inflight_deletes_.clear();
    pending_locations_.clear();
    METRICS_(cache_gc, inflight_delete_count) = 0;
    METRICS_(cache_gc, inflight_delete_age_ms) = 0;
    METRICS_(cache_gc, round_duration_ms) = 0;
}

void CacheGarbageCollector::RunOneTick() noexcept {
    try {
        if (stop_requested_.load(std::memory_order_acquire)) {
            return;
        }
        PollInflightDeletes();
        if (inflight_deletes_.size() >= config_.max_inflight_delete_requests) {
            if (round_active_ && instance_index_ < instances_.size()) {
                ++instances_[instance_index_].inflight_throttled_tick_count;
            }
            return;
        }
        if (stop_requested_.load(std::memory_order_acquire)) {
            return;
        }

        const auto now = Clock::now();
        if (next_round_at_.has_value() && now < next_round_at_.value()) {
            return;
        }
        if (!round_active_ && !BeginRound()) {
            return;
        }
        if (!round_active_) {
            return;
        }
        if (instance_index_ >= instances_.size()) {
            CompleteRound();
            return;
        }

        InstanceScanEntry &entry = instances_[instance_index_];
        if (stop_requested_.load(std::memory_order_acquire)) {
            return;
        }
        auto indexer = meta_indexer_manager_->GetMetaIndexer(entry.instance_id);
        if (!indexer) {
            RecordOperationError("indexer_missing");
            KVCM_LOG_WARN("cache gc round[%lu] skip missing indexer, group[%s] instance[%s]",
                          round_id_,
                          entry.instance_group.c_str(),
                          entry.instance_id.c_str());
            AdvanceInstance(true);
            return;
        }

        MaintenanceScanBatch batch;
        const std::string scan_cursor = entry.cursor;
        ErrorCode ec =
            indexer->ScanLocationsForMaintenance(round_context_.get(), scan_cursor, config_.scan_batch_size, batch);
        if (ec != EC_OK) {
            RecordOperationError("scan");
            ++entry.scan_failure_count;
            const bool retry_exhausted = entry.scan_failure_count >= kMaxScanFailuresPerInstancePerRound;
            KVCM_LOG_WARN("cache gc round[%lu] scan failed, group[%s] instance[%s] cursor[%s] ec[%d] "
                          "failure_count[%zu] retry_exhausted[%d]",
                          round_id_,
                          entry.instance_group.c_str(),
                          entry.instance_id.c_str(),
                          scan_cursor.c_str(),
                          static_cast<int>(ec),
                          entry.scan_failure_count,
                          retry_exhausted);
            if (retry_exhausted) {
                RecordOperationError("scan_retry_exhausted");
            }
            // Preserve the cursor while retries remain. Once this Instance
            // exhausts its per-round budget, defer the remaining keyspace to
            // the next round so one broken backend cannot wedge every other
            // Instance behind an active round forever.
            AdvanceInstance(retry_exhausted);
            return;
        }

        entry.scan_failure_count = 0;
        entry.scanned_key_count += batch.keys.size();
        ++entry.scan_batch_count;
        METRICS_(cache_gc, scan_key_count) += batch.keys.size();
        entry.cursor = batch.next_cursor;
        if (stop_requested_.load(std::memory_order_acquire)) {
            return;
        }
        ScanDeleteActions actions = BuildDeleteActions(entry.instance_id, batch, TimestampUtil::GetCurrentTimeUs());
        const std::string instance_id = entry.instance_id;
        const bool scan_completed = entry.cursor == SCAN_BASE_CURSOR;

        if (stop_requested_.load(std::memory_order_acquire)) {
            return;
        }

        auto submit_and_track = [&](auto &request,
                                    std::vector<PendingLocationKey> request_targets,
                                    const char *action_name,
                                    const std::map<std::string, size_t> &reason_counts) {
            if (request_targets.empty() || inflight_deletes_.size() >= config_.max_inflight_delete_requests) {
                return false;
            }
            InflightDelete tracked_action;
            try {
                // Prepare the tracking record before SubmitAsync. Pending-set insertion still allocates after
                // acceptance, so its exception path rolls back every successfully published target below.
                inflight_deletes_.reserve(inflight_deletes_.size() + 1);
                tracked_action.instance_id = instance_id;
                tracked_action.action_name = action_name;
                tracked_action.pending_locations.reserve(request_targets.size());
            } catch (...) {
                RecordOperationError("prepare_action_tracking");
                return false;
            }
            AsyncDeleteSubmitResult submit_result;
            try {
                submit_result = schedule_plan_executor_->SubmitAsync(request);
            } catch (const std::exception &e) {
                RecordOperationError("submit_exception");
                KVCM_LOG_ERROR("cache gc %s SubmitAsync threw, instance[%s] error[%s]",
                               action_name,
                               instance_id.c_str(),
                               e.what());
                return false;
            } catch (...) {
                RecordOperationError("submit_exception");
                KVCM_LOG_ERROR("cache gc %s SubmitAsync threw unknown exception, instance[%s]",
                               action_name,
                               instance_id.c_str());
                return false;
            }
            if (!submit_result.accepted) {
                if (submit_result.future.valid()) {
                    RecordOperationError("submit_contract");
                    KVCM_LOG_ERROR("cache gc %s SubmitAsync rejected with a valid Future, instance[%s]",
                                   action_name,
                                   instance_id.c_str());
                }
                return false;
            }
            if (!submit_result.future.valid()) {
                RecordOperationError("submit_contract");
                KVCM_LOG_ERROR("cache gc %s SubmitAsync accepted with an invalid Future, instance[%s]",
                               action_name,
                               instance_id.c_str());
                return false;
            }

            try {
                for (const auto &target : request_targets) {
                    if (pending_locations_.insert(target).second) {
                        tracked_action.pending_locations.push_back(target);
                    } else {
                        // BuildDeleteActions filters pending targets and this
                        // callback is single-threaded, so normal execution
                        // cannot reach this branch. The Executor's final
                        // conditional check remains the safety barrier for an
                        // already accepted invariant-violating request.
                        RecordOperationError("pending_contract");
                    }
                }
                tracked_action.round_id = round_id_;
                tracked_action.target_count = request_targets.size();
                tracked_action.submitted_at = Clock::now();
                tracked_action.future = std::move(submit_result.future);
                inflight_deletes_.push_back(std::move(tracked_action));
            } catch (...) {
                for (const auto &target : tracked_action.pending_locations) {
                    pending_locations_.erase(target);
                }
                RecordOperationError("track_accepted_action");
                KVCM_LOG_ERROR("cache gc failed to track accepted %s action, instance[%s]",
                               action_name,
                               instance_id.c_str());
                return false;
            }
            METRICS_(cache_gc, delete_target_count) += request_targets.size();
            for (const auto &[reason, count] : reason_counts) {
                entry.submitted_location_counts[reason] += count;
            }
            return true;
        };

        std::vector<PendingLocationKey> physical_targets;
        for (size_t i = 0; i < actions.executor_request.block_keys.size(); ++i) {
            for (const auto &location_id : actions.executor_request.location_ids[i]) {
                physical_targets.push_back({instance_id, actions.executor_request.block_keys[i], location_id});
            }
        }
        submit_and_track(
            actions.executor_request, std::move(physical_targets), "physical_delete", actions.executor_reason_counts);

        std::vector<PendingLocationKey> event_report_targets;
        for (size_t i = 0; i < actions.event_report_request.block_keys.size(); ++i) {
            for (const auto &target : actions.event_report_request.targets[i]) {
                event_report_targets.push_back(
                    {instance_id, actions.event_report_request.block_keys[i], target.location_id});
            }
        }
        if (!event_report_targets.empty() && inflight_deletes_.size() >= config_.max_inflight_delete_requests) {
            for (const auto &targets : actions.event_report_request.targets) {
                for (const auto &target : targets) {
                    RecordCandidateDropped(CandidateReasonName(EventReportTokenReason(target.cleanup_token)),
                                           "inflight_limit",
                                           1);
                }
            }
        } else {
            submit_and_track(actions.event_report_request,
                             std::move(event_report_targets),
                             "event_report_metadata",
                             actions.event_report_reason_counts);
        }
        UpdateInflightDeleteMetrics();
        if (scan_completed) {
            LogInstanceScanSummary(entry);
        }
        AdvanceInstance(scan_completed);
    } catch (const std::exception &e) {
        RecordOperationError("tick_exception");
        KVCM_LOG_ERROR("cache gc tick threw exception: %s", e.what());
    } catch (...) {
        RecordOperationError("tick_exception");
        KVCM_LOG_ERROR("cache gc tick threw unknown exception");
    }
}

CacheGarbageCollector::EventReportBackendRoute
CacheGarbageCollector::LookupEventReportBackend(const std::string &instance_id, DataStorageType storage_type) const {
    EventReportBackendRoute route;
    if (!registry_manager_ || !data_storage_manager_ || !IsEventReportStorageType(storage_type)) {
        return route;
    }
    const std::string group_name = registry_manager_->GetInstanceGroupName(instance_id);
    const auto group = registry_manager_->GetInstanceGroupConfig(group_name);
    if (!group) {
        return route;
    }
    std::set<std::string> matched_names;
    for (const auto &storage_name : group->event_report_storage_candidates()) {
        auto backend =
            std::dynamic_pointer_cast<EventReportBackend>(data_storage_manager_->GetDataStorageBackend(storage_name));
        if (!backend) {
            // Match the online ReportEvent routing contract: a candidate that
            // is not currently registered cannot own requests in this
            // process and must not hide a later, matching storage tier.
            continue;
        }
        if (backend->GetStorageType() != storage_type || !matched_names.insert(storage_name).second) {
            continue;
        }
        if (route.backend) {
            route = {};
            route.status = EventReportBackendRouteStatus::kAmbiguous;
            return route;
        }
        route.backend_unique_name = storage_name;
        route.backend = std::move(backend);
    }
    if (!route.backend) {
        return route;
    }
    route.status = route.backend->Available() ? EventReportBackendRouteStatus::kResolved
                                               : EventReportBackendRouteStatus::kUnavailable;
    return route;
}

void CacheGarbageCollector::PollInflightDeletes() noexcept {
    auto it = inflight_deletes_.begin();
    while (it != inflight_deletes_.end()) {
        if (!it->future.valid()) {
            RecordOperationError("future_invalid");
            RecordDeleteResult("invalid");
            KVCM_LOG_ERROR("cache gc stored invalid Future, round[%lu] instance[%s] action[%s] targets[%zu]",
                           it->round_id,
                           it->instance_id.c_str(),
                           it->action_name.c_str(),
                           it->target_count);
            ReleasePendingLocations(*it);
            it = inflight_deletes_.erase(it);
            continue;
        }

        const uint64_t round_id = it->round_id;
        const std::string instance_id = it->instance_id;
        const std::string action_name = it->action_name;
        const size_t target_count = it->target_count;
        try {
            if (it->future.wait_for(std::chrono::microseconds::zero()) != std::future_status::ready) {
                ++it;
                continue;
            }
            const PlanExecuteResult result = it->future.get();
            RecordDeleteResult(std::to_string(static_cast<int>(result.status)));
            if (result.status != EC_OK) {
                KVCM_LOG_WARN(
                    "cache gc action completed with status[%d], round[%lu] instance[%s] action[%s] targets[%zu] "
                    "error[%s]",
                    static_cast<int>(result.status),
                    round_id,
                    instance_id.c_str(),
                    action_name.c_str(),
                    target_count,
                    result.error_message.c_str());
            }
        } catch (const std::exception &e) {
            RecordOperationError("future_get");
            RecordDeleteResult("exception");
            KVCM_LOG_ERROR("cache gc Future get failed, round[%lu] instance[%s] action[%s] targets[%zu] error[%s]",
                           round_id,
                           instance_id.c_str(),
                           action_name.c_str(),
                           target_count,
                           e.what());
        } catch (...) {
            RecordOperationError("future_get");
            RecordDeleteResult("exception");
            KVCM_LOG_ERROR("cache gc Future get failed with unknown exception, round[%lu] instance[%s] action[%s] "
                           "targets[%zu]",
                           round_id,
                           instance_id.c_str(),
                           action_name.c_str(),
                           target_count);
        }
        ReleasePendingLocations(*it);
        it = inflight_deletes_.erase(it);
    }
    UpdateInflightDeleteMetrics();
}

void CacheGarbageCollector::UpdateInflightDeleteMetrics() noexcept {
    METRICS_(cache_gc, inflight_delete_count) = static_cast<double>(inflight_deletes_.size());
    int64_t oldest_age_ms = 0;
    const auto now = Clock::now();
    for (const auto &inflight : inflight_deletes_) {
        const auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - inflight.submitted_at).count();
        oldest_age_ms = std::max<int64_t>(oldest_age_ms, age_ms);
    }
    METRICS_(cache_gc, inflight_delete_age_ms) = static_cast<double>(oldest_age_ms);
}

void CacheGarbageCollector::ReleasePendingLocations(const InflightDelete &inflight) noexcept {
    for (const auto &pending_location : inflight.pending_locations) {
        pending_locations_.erase(pending_location);
    }
}

bool CacheGarbageCollector::BeginRound() {
    const uint64_t next_round_id = round_id_ + 1;
    auto context = std::make_shared<RequestContext>("cache_gc_round_" + std::to_string(next_round_id));
    auto [group_ec, groups] = registry_manager_->ListInstanceGroup(context.get());
    if (group_ec != EC_OK) {
        RecordOperationError("list_groups");
        KVCM_LOG_WARN("cache gc list instance groups failed, ec[%d]", static_cast<int>(group_ec));
        return false;
    }

    std::vector<InstanceScanEntry> snapshot;
    for (const auto &group : groups) {
        if (stop_requested_.load(std::memory_order_acquire)) {
            return false;
        }
        if (!group || group->name().empty()) {
            RecordOperationError("registry_snapshot");
            KVCM_LOG_WARN("cache gc encountered invalid instance group in snapshot");
            return false;
        }
        auto [instance_ec, instances] = registry_manager_->ListInstanceInfo(context.get(), group->name());
        if (instance_ec != EC_OK) {
            RecordOperationError("list_instances");
            KVCM_LOG_WARN("cache gc list instances failed, group[%s] ec[%d]",
                          group->name().c_str(),
                          static_cast<int>(instance_ec));
            return false;
        }
        for (const auto &instance : instances) {
            if (!instance || instance->instance_id().empty()) {
                RecordOperationError("registry_snapshot");
                KVCM_LOG_WARN("cache gc encountered invalid instance in group[%s]", group->name().c_str());
                return false;
            }
            snapshot.push_back({group->name(), instance->instance_id(), SCAN_BASE_CURSOR, false});
        }
    }

    std::sort(snapshot.begin(), snapshot.end());
    snapshot.erase(std::unique(snapshot.begin(), snapshot.end()), snapshot.end());
    instances_ = std::move(snapshot);
    instance_index_ = 0;
    round_id_ = next_round_id;
    round_active_ = true;
    round_started_at_ = Clock::now();
    next_round_at_.reset();
    round_context_ = std::move(context);
    KVCM_LOG_INFO("cache gc round[%lu] started with[%zu] instances", round_id_, instances_.size());
    return true;
}

void CacheGarbageCollector::CompleteRound() noexcept {
    const auto now = Clock::now();
    const auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - round_started_at_).count();
    METRICS_(cache_gc, round_duration_ms) = static_cast<double>(std::max<int64_t>(0, duration_ms));
    ++METRICS_(cache_gc, scan_round_count);
    KVCM_LOG_INFO(
        "cache gc round[%lu] completed, instances[%zu] duration_ms[%ld]", round_id_, instances_.size(), duration_ms);

    instances_.clear();
    instance_index_ = 0;
    round_active_ = false;
    round_context_.reset();
    next_round_at_ = now + std::chrono::milliseconds(config_.round_pause_ms);
}

void CacheGarbageCollector::LogInstanceScanSummary(const InstanceScanEntry &entry) const noexcept {
    const auto reason_count = [&entry](const char *reason) {
        const auto it = entry.submitted_location_counts.find(reason);
        return it == entry.submitted_location_counts.end() ? size_t{0} : it->second;
    };
    size_t submitted_location_count = 0;
    for (const auto &[reason, count] : entry.submitted_location_counts) {
        (void)reason;
        submitted_location_count += count;
    }
    KVCM_LOG_INFO(
        "cache gc instance scan completed, round[%lu] group[%s] instance[%s] scanned_keys[%zu] scan_batches[%zu] "
        "submitted_locations[%zu] reasons[orphan_writing=%zu,storage_missing=%zu,event_report_stale_snapshot=%zu,"
        "event_report_down_host=%zu,event_report_recovery_absent_host=%zu] inflight_throttled_ticks[%zu]",
        round_id_,
        entry.instance_group.c_str(),
        entry.instance_id.c_str(),
        entry.scanned_key_count,
        entry.scan_batch_count,
        submitted_location_count,
        reason_count(kOrphanWritingReason),
        reason_count(kStorageMissingReason),
        reason_count(kEventReportStaleSnapshotReason),
        reason_count(kEventReportDownHostReason),
        reason_count(kEventReportRecoveryAbsentHostReason),
        entry.inflight_throttled_tick_count);
}

void CacheGarbageCollector::AdvanceInstance(bool completed_current) noexcept {
    if (instances_.empty() || instance_index_ >= instances_.size()) {
        CompleteRound();
        return;
    }
    instances_[instance_index_].completed = instances_[instance_index_].completed || completed_current;
    if (std::all_of(instances_.begin(), instances_.end(), [](const InstanceScanEntry &entry) {
            return entry.completed;
        })) {
        CompleteRound();
        return;
    }
    for (size_t offset = 1; offset <= instances_.size(); ++offset) {
        const size_t candidate = (instance_index_ + offset) % instances_.size();
        if (!instances_[candidate].completed) {
            instance_index_ = candidate;
            return;
        }
    }
}

CacheGarbageCollector::ScanDeleteActions CacheGarbageCollector::BuildDeleteActions(const std::string &instance_id,
                                                                                   const MaintenanceScanBatch &batch,
                                                                                   const int64_t now_us) {
    ScanDeleteActions actions;
    actions.executor_request.instance_id = instance_id;
    actions.executor_request.authoritative_read = true;
    actions.event_report_request.instance_id = instance_id;

    if (instance_id.empty() || batch.keys.size() != batch.locations.size() ||
        batch.keys.size() != batch.location_results.size()) {
        RecordOperationError("scan_shape");
        return actions;
    }

    std::map<CandidateKey, DeleteCandidate> candidates;
    std::vector<ServingProbe> serving_probes;
    std::map<StorageProbeKey, StorageProbeBatch> storage_probe_batches;
    std::vector<EventReportProbeEntry> event_report_probe_entries;
    std::map<DataStorageType, EventReportProbeBatch> event_report_probe_batches;
    for (size_t i = 0; i < batch.keys.size(); ++i) {
        if (stop_requested_.load(std::memory_order_acquire)) {
            return actions;
        }
        if (batch.location_results[i] != EC_OK) {
            if (batch.location_results[i] != EC_NOENT) {
                RecordOperationError("scan_key");
            }
            continue;
        }
        for (const auto &[location_id, location] : batch.locations[i]) {
            if (!location || location_id.empty() || location->id().empty() || location_id != location->id()) {
                continue;
            }

            const CandidateKey candidate_key{batch.keys[i], location_id};
            if (IsOrphanWriting(location_id, *location, now_us)) {
                if (migration_manager_ &&
                    migration_manager_->HasActiveCopyTargetLocation(instance_id, batch.keys[i], location_id)) {
                    continue;
                }
                candidates.emplace(candidate_key,
                                   DeleteCandidate{
                                       .expected_location_value = location->ToJsonString(),
                                       .reason = CandidateReason::kOrphanWriting,
                                   });
                continue;
            }

            const auto storage_type = location->type();
            if (location->status() != CLS_SERVING || storage_type == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN ||
                ToIndex(storage_type) >= ToIndex(DataStorageType::COUNT)) {
                continue;
            }

            if (IsEventReportStorageType(storage_type)) {
                if (!config_.event_report_cleanup_enabled) {
                    continue;
                }
                auto &probe_batch = event_report_probe_batches[storage_type];
                if (!probe_batch.route_checked) {
                    probe_batch.route_checked = true;
                    const auto route = LookupEventReportBackend(instance_id, storage_type);
                    if (route.status == EventReportBackendRouteStatus::kResolved) {
                        probe_batch.backend = route.backend;
                        probe_batch.backend_unique_name = route.backend_unique_name;
                    } else {
                        probe_batch.route_unknown_cause =
                            route.status == EventReportBackendRouteStatus::kAmbiguous
                                ? "backend_ambiguous"
                                : (route.status == EventReportBackendRouteStatus::kUnavailable ? "backend_unavailable"
                                                                                              : "backend_missing");
                        if (route.status == EventReportBackendRouteStatus::kAmbiguous) {
                            KVCM_LOG_WARN("cache gc skips ambiguous EventReport owner, round[%lu] instance[%s] "
                                          "storage_type[%d]",
                                          round_id_,
                                          instance_id.c_str(),
                                          static_cast<int>(storage_type));
                        }
                    }
                }
                if (!probe_batch.backend) {
                    RecordEventReportProbe(
                        "unknown", probe_batch.route_unknown_cause ? probe_batch.route_unknown_cause : "backend_missing");
                    continue;
                }
                EventReportBackend::MaintenanceLocationProbe probe{
                    .instance_id = instance_id,
                    .location_id = location_id,
                };
                probe.storage_uris.reserve(location->location_specs().size());
                for (const auto &spec : location->location_specs()) {
                    probe.storage_uris.push_back(spec.uri());
                }
                const size_t probe_index = event_report_probe_entries.size();
                event_report_probe_entries.push_back({candidate_key, location});
                probe_batch.probes.emplace_back(std::move(probe));
                probe_batch.probe_indexes.push_back(probe_index);
                continue;
            }

            if (location->location_specs().empty()) {
                continue;
            }

            std::vector<std::pair<std::string, DataStorageUri>> parsed_uris;
            parsed_uris.reserve(location->location_specs().size());
            for (const auto &spec : location->location_specs()) {
                DataStorageUri uri(spec.uri());
                if (uri.Valid() && !uri.GetHostName().empty()) {
                    parsed_uris.emplace_back(uri.GetHostName(), std::move(uri));
                }
            }
            if (parsed_uris.empty()) {
                continue;
            }

            const size_t serving_probe_index = serving_probes.size();
            serving_probes.push_back({candidate_key, location});
            for (auto &[storage_name, uri] : parsed_uris) {
                auto &probe_batch = storage_probe_batches[{storage_name, storage_type}];
                probe_batch.uris.emplace_back(std::move(uri));
                probe_batch.serving_probe_indexes.push_back(serving_probe_index);
            }
        }
    }

    for (const auto &[storage_key, probe_batch] : storage_probe_batches) {
        if (stop_requested_.load(std::memory_order_acquire)) {
            return actions;
        }
        const auto &[storage_name, expected_storage_type] = storage_key;
        try {
            auto storage_backend = data_storage_manager_->GetDataStorageBackend(storage_name);
            if (!storage_backend) {
                RecordOperationError("might_exist_storage_not_found");
                continue;
            }
            const DataStorageType actual_storage_type = storage_backend->GetType();
            if (actual_storage_type != expected_storage_type) {
                RecordOperationError("might_exist_storage_type_mismatch");
                KVCM_LOG_WARN("cache gc storage type mismatch, round[%lu] instance[%s] storage[%s] expected[%d] "
                              "actual[%d]",
                              round_id_,
                              instance_id.c_str(),
                              storage_name.c_str(),
                              static_cast<int>(expected_storage_type),
                              static_cast<int>(actual_storage_type));
                continue;
            }

            for (size_t offset = 0; offset < probe_batch.uris.size(); offset += kMightExistProbeBatchSize) {
                if (stop_requested_.load(std::memory_order_acquire)) {
                    return actions;
                }
                const size_t end = std::min(offset + kMightExistProbeBatchSize, probe_batch.uris.size());
                const std::vector<DataStorageUri> uris(probe_batch.uris.begin() + offset,
                                                       probe_batch.uris.begin() + end);
                const std::vector<bool> might_exist =
                    data_storage_manager_->Exist(storage_name, uris, /*fastpath=*/true);
                if (stop_requested_.load(std::memory_order_acquire)) {
                    return actions;
                }
                if (might_exist.size() != uris.size()) {
                    RecordOperationError("might_exist_shape");
                    KVCM_LOG_WARN("cache gc MightExist result shape mismatch, round[%lu] instance[%s] storage[%s] "
                                  "offset[%zu] uris[%zu] results[%zu]",
                                  round_id_,
                                  instance_id.c_str(),
                                  storage_name.c_str(),
                                  offset,
                                  uris.size(),
                                  might_exist.size());
                    continue;
                }
                for (size_t i = 0; i < might_exist.size(); ++i) {
                    if (!might_exist[i]) {
                        serving_probes[probe_batch.serving_probe_indexes[offset + i]].confirmed_missing_uris.insert(
                            uris[i].ToUriString());
                    }
                }
            }
        } catch (const std::exception &e) {
            RecordOperationError("might_exist_exception");
            KVCM_LOG_WARN("cache gc MightExist threw, round[%lu] instance[%s] storage[%s] error[%s]",
                          round_id_,
                          instance_id.c_str(),
                          storage_name.c_str(),
                          e.what());
        } catch (...) {
            RecordOperationError("might_exist_exception");
            KVCM_LOG_WARN("cache gc MightExist threw unknown exception, round[%lu] instance[%s] storage[%s]",
                          round_id_,
                          instance_id.c_str(),
                          storage_name.c_str());
        }
    }

    for (const auto &[storage_type, probe_batch] : event_report_probe_batches) {
        (void)storage_type;
        if (stop_requested_.load(std::memory_order_acquire)) {
            return actions;
        }
        try {
            for (size_t offset = 0; offset < probe_batch.probes.size(); offset += kEventReportProbeBatchSize) {
                if (stop_requested_.load(std::memory_order_acquire)) {
                    return actions;
                }
                const size_t end = std::min(offset + kEventReportProbeBatchSize, probe_batch.probes.size());
                const std::vector<EventReportBackend::MaintenanceLocationProbe> probes(
                    probe_batch.probes.begin() + offset, probe_batch.probes.begin() + end);
                const auto probe_results = probe_batch.backend->ProbeLocationsForMaintenance(probes);
                if (stop_requested_.load(std::memory_order_acquire)) {
                    return actions;
                }
                if (probe_results.size() != probes.size()) {
                    RecordOperationError("event_report_probe_shape");
                    RecordEventReportProbe("error", "shape");
                    continue;
                }
                for (size_t i = 0; i < probe_results.size(); ++i) {
                    const auto &probe_result = probe_results[i];
                    if (probe_result.decision == EventReportBackend::MaintenanceCleanupDecision::kKeep) {
                        RecordEventReportProbe("keep");
                        continue;
                    }
                    if (probe_result.decision == EventReportBackend::MaintenanceCleanupDecision::kUnknown) {
                        RecordEventReportProbe("unknown", MaintenanceUnknownReasonName(probe_result.unknown_reason));
                        continue;
                    }
                    RecordEventReportProbe("delete");
                    const auto &entry = event_report_probe_entries[probe_batch.probe_indexes[offset + i]];
                    CandidateReason reason = CandidateReason::kEventReportStaleSnapshot;
                    if (probe_result.token.reason == EventReportBackend::MaintenanceCleanupReason::kDownHost) {
                        reason = CandidateReason::kEventReportDownHost;
                    } else if (probe_result.token.reason ==
                               EventReportBackend::MaintenanceCleanupReason::kRecoveryAbsentHost) {
                        reason = CandidateReason::kEventReportRecoveryAbsentHost;
                    }
                    candidates.emplace(entry.key,
                                       DeleteCandidate{
                                           .expected_location_value = entry.location->ToJsonString(),
                                           .reason = reason,
                                           .event_report_backend = probe_batch.backend,
                                           .event_report_backend_unique_name = probe_batch.backend_unique_name,
                                           .event_report_cleanup_token = probe_result.token,
                                       });
                }
            }
        } catch (const std::exception &e) {
            RecordOperationError("event_report_probe_exception");
            RecordEventReportProbe("error", "exception");
            KVCM_LOG_WARN("cache gc EventReport maintenance probe threw, round[%lu] instance[%s] error[%s]",
                          round_id_,
                          instance_id.c_str(),
                          e.what());
        } catch (...) {
            RecordOperationError("event_report_probe_exception");
            RecordEventReportProbe("error", "exception");
            KVCM_LOG_WARN("cache gc EventReport maintenance probe threw unknown exception, round[%lu] instance[%s]",
                          round_id_,
                          instance_id.c_str());
        }
    }

    for (auto &probe : serving_probes) {
        if (!probe.confirmed_missing_uris.empty()) {
            candidates.emplace(std::move(probe.key),
                               DeleteCandidate{
                                   .expected_location_value = probe.location->ToJsonString(),
                                   .reason = CandidateReason::kStorageMissing,
                                   .confirmed_missing_uris = std::move(probe.confirmed_missing_uris),
                               });
        }
    }

    std::map<CandidateReason, size_t> candidate_counts;
    for (const auto &[candidate_key, candidate] : candidates) {
        (void)candidate_key;
        ++candidate_counts[candidate.reason];
    }
    for (const auto &[reason, count] : candidate_counts) {
        RecordCandidateCount(CandidateReasonName(reason), count);
    }

    struct RequestTargets {
        std::vector<std::string> location_ids;
        std::vector<std::string> expected_location_values;
    };
    std::map<KeyType, RequestTargets> executor_targets;
    std::map<KeyType, std::vector<EventReportMetadataDeleteTarget>> event_report_targets;
    std::vector<std::pair<CandidateKey, const DeleteCandidate *>> ordered_candidates;
    ordered_candidates.reserve(candidates.size());
    for (const auto &[candidate_key, candidate] : candidates) {
        ordered_candidates.emplace_back(candidate_key, &candidate);
    }
    std::sort(ordered_candidates.begin(), ordered_candidates.end(), [](const auto &lhs, const auto &rhs) {
        const int lhs_priority = CandidatePriority(lhs.second->reason);
        const int rhs_priority = CandidatePriority(rhs.second->reason);
        return lhs_priority != rhs_priority ? lhs_priority < rhs_priority : lhs.first < rhs.first;
    });

    size_t selected = 0;
    std::set<KeyType> event_report_selected_keys;
    for (const auto &[candidate_key, candidate_ptr] : ordered_candidates) {
        const auto &candidate = *candidate_ptr;
        const auto &[block_key, location_id] = candidate_key;
        if (pending_locations_.find(PendingLocationKey{instance_id, block_key, location_id}) !=
            pending_locations_.end()) {
            continue;
        }
        if (selected >= config_.scan_batch_size) {
            RecordCandidateDropped(CandidateReasonName(candidate.reason), "total_budget", 1);
            continue;
        }
        if (IsEventReportReason(candidate.reason)) {
            const bool new_key = event_report_selected_keys.find(block_key) == event_report_selected_keys.end();
            if (new_key && event_report_selected_keys.size() >= config_.event_report_action_batch_size) {
                RecordCandidateDropped(CandidateReasonName(candidate.reason), "event_report_budget", 1);
                continue;
            }
            if (!candidate.event_report_backend || !candidate.event_report_cleanup_token.has_value()) {
                RecordOperationError("event_report_candidate_contract");
                continue;
            }
            event_report_targets[block_key].push_back(EventReportMetadataDeleteTarget{
                .location_id = location_id,
                .expected_location_value = candidate.expected_location_value,
                .backend_unique_name = candidate.event_report_backend_unique_name,
                .storage_type = candidate.event_report_backend->GetStorageType(),
                .expected_backend = candidate.event_report_backend,
                .cleanup_token = candidate.event_report_cleanup_token.value(),
            });
            event_report_selected_keys.insert(block_key);
            ++actions.event_report_reason_counts[CandidateReasonName(candidate.reason)];
        } else {
            auto &block_targets = executor_targets[block_key];
            block_targets.location_ids.push_back(location_id);
            block_targets.expected_location_values.push_back(candidate.expected_location_value);
            actions.executor_request.confirmed_missing_uris.insert(candidate.confirmed_missing_uris.begin(),
                                                                   candidate.confirmed_missing_uris.end());
            ++actions.executor_reason_counts[CandidateReasonName(candidate.reason)];
        }
        ++selected;
    }
    actions.executor_request.block_keys.reserve(executor_targets.size());
    actions.executor_request.location_ids.reserve(executor_targets.size());
    actions.executor_request.expected_location_values.reserve(executor_targets.size());
    for (auto &[block_key, block_targets] : executor_targets) {
        actions.executor_request.block_keys.push_back(block_key);
        actions.executor_request.location_ids.emplace_back(std::move(block_targets.location_ids));
        actions.executor_request.expected_location_values.emplace_back(
            std::move(block_targets.expected_location_values));
    }
    actions.event_report_request.block_keys.reserve(event_report_targets.size());
    actions.event_report_request.targets.reserve(event_report_targets.size());
    for (auto &[block_key, targets] : event_report_targets) {
        actions.event_report_request.block_keys.push_back(block_key);
        actions.event_report_request.targets.push_back(std::move(targets));
    }
    return actions;
}

bool CacheGarbageCollector::IsOrphanWriting(const std::string &map_location_id,
                                            const CacheLocation &location,
                                            const int64_t now_us) const noexcept {
    if (map_location_id.empty() || location.id().empty() || map_location_id != location.id() ||
        location.status() != CLS_WRITING || location.create_time() <= 0 || now_us < location.create_time()) {
        return false;
    }
    const int64_t age_us = now_us - location.create_time();
    return age_us / 1000 >= config_.orphan_writing_grace_period_ms;
}

void CacheGarbageCollector::RecordCandidateCount(const std::string &reason, const size_t count) noexcept {
    if (count == 0) {
        return;
    }
    try {
        if (metrics_registry_) {
            auto counter = metrics_registry_->GetCounter(METRICS_NAME_(cache_gc, candidate_count),
                                                         MetricsTags{{"reason", reason}});
            counter += count;
        }
    } catch (...) { KVCM_LOG_ERROR("cache gc failed to record candidate metric"); }
}

void CacheGarbageCollector::RecordCandidateDropped(const std::string &reason,
                                                   const std::string &cause,
                                                   const size_t count) noexcept {
    if (count == 0) {
        return;
    }
    try {
        if (metrics_registry_) {
            auto counter = metrics_registry_->GetCounter(kCandidateDroppedMetricName,
                                                         MetricsTags{{"reason", reason}, {"cause", cause}});
            counter += count;
        }
    } catch (...) { KVCM_LOG_ERROR("cache gc failed to record candidate drop metric"); }
}

void CacheGarbageCollector::RecordEventReportProbe(const std::string &result,
                                                   const std::string &unknown_cause) noexcept {
    try {
        if (!metrics_registry_) {
            return;
        }
        ++metrics_registry_->GetCounter(kEventReportProbeMetricName, MetricsTags{{"result", result}});
        if (!unknown_cause.empty()) {
            ++metrics_registry_->GetCounter(kEventReportProbeUnknownMetricName, MetricsTags{{"cause", unknown_cause}});
        }
    } catch (...) { KVCM_LOG_ERROR("cache gc failed to record EventReport probe metric"); }
}

void CacheGarbageCollector::RecordOperationError(const std::string &stage) noexcept {
    try {
        if (metrics_registry_) {
            auto counter = metrics_registry_->GetCounter(kOperationErrorMetricName, MetricsTags{{"stage", stage}});
            ++counter;
        }
    } catch (...) { KVCM_LOG_ERROR("cache gc failed to record operation error metric"); }
}

void CacheGarbageCollector::RecordDeleteResult(const std::string &status) noexcept {
    try {
        if (metrics_registry_) {
            auto counter = metrics_registry_->GetCounter(kDeleteResultMetricName, MetricsTags{{"status", status}});
            ++counter;
        }
    } catch (...) { KVCM_LOG_ERROR("cache gc failed to record delete result metric"); }
}

} // namespace kv_cache_manager
