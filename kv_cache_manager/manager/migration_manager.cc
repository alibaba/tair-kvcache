#include "kv_cache_manager/manager/migration_manager.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <limits>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/event/spec_events/migration_event.h"
#include "kv_cache_manager/manager/data_storage_selector.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"

namespace kv_cache_manager {

static_assert(MigrationConfig::kMinCopyHttpTimeoutWindowsPerDeadline ==
                  AsyncCopyOptions::kMinHttpTimeoutWindowsPerDeadline,
              "async Copy HTTP timeout ratio defaults must remain aligned");

namespace {
constexpr auto kMonitorIdleSleep = std::chrono::milliseconds(50);
constexpr auto kFutureWaitTime = std::chrono::microseconds(200);
constexpr auto kGuardSyncRetryInterval = std::chrono::milliseconds(100);
constexpr auto kGuardSyncRetryIdleSleep = std::chrono::milliseconds(5);
constexpr auto kLocationCleanupRetryInterval = std::chrono::seconds(1);
constexpr int64_t kAsyncGuardRecoveryScanBatchSize = 256;
constexpr auto kAsyncGuardRecoveryScanPause = std::chrono::milliseconds(2);
constexpr auto kAsyncGuardRecoveryMaxDuration = std::chrono::minutes(5);

struct AddLocationRollbackItem {
    int64_t block_key = 0;
    std::string storage_name;
    std::vector<DataStorageUri> uris;
    ErrorCode ec = EC_UNKNOWN;
    std::string location_id;
};

// BatchAddLocation 失败后的补偿规则与返回契约一致，统一收敛在
// MetaSearcher::ReconcileAddLocationRollback：
// - 已知成功项走标准 Location 删除流水线，由流水线回收 usage；
// - 失败但已生成 ID 的项先幂等删除元数据并 Sync，确认无引用后才删 URI；
// - 未生成 ID 的项可直接删 URI。
// 任一元数据状态无法确认时保留 URI，避免制造悬空引用。
void RollbackAddedLocations(RequestContext *request_context,
                            const std::string &trace_id,
                            const std::string &instance_id,
                            const std::shared_ptr<MetaIndexer> &indexer,
                            const std::shared_ptr<DataStorageManager> &data_storage_manager,
                            const std::shared_ptr<SchedulePlanExecutor> &schedule_plan_executor,
                            const std::vector<AddLocationRollbackItem> &items) {
    MetaSearcher::AddLocationRollbackPlan plan;
    KeyVector keys;
    std::vector<MetaSearcher::AddLocationResult> add_results;
    keys.reserve(items.size());
    add_results.reserve(items.size());
    for (const auto &item : items) {
        keys.push_back(item.block_key);
        add_results.push_back({item.ec, item.location_id});
    }
    if (!indexer) {
        // 无元数据访问时无法 reconcile uncertain 项，只能保留其 URI；但
        // confirmed-success 与无 ID 项的清理不依赖元数据，照常执行。
        const size_t uncertain_count = MetaSearcher::ClassifyAddLocationRollback(keys, add_results, plan);
        KVCM_LOG_WARN("[%s] rollback migration locations without meta indexer: uncertain URIs retained, instance %s, "
                      "uncertain_count %zu, key_count %zu",
                      trace_id.c_str(),
                      instance_id.c_str(),
                      uncertain_count,
                      items.size());
    } else {
        MetaSearcher meta_searcher(indexer);
        if (meta_searcher.ReconcileAddLocationRollback(request_context, keys, add_results, plan) != EC_OK) {
            KVCM_LOG_WARN("[%s] rollback migration locations reconcile failed, instance %s, key_count %zu",
                          trace_id.c_str(),
                          instance_id.c_str(),
                          items.size());
            return;
        }
    }

    if (!plan.pipeline_keys.empty()) {
        CacheLocationDelRequest known_success_request;
        known_success_request.instance_id = instance_id;
        known_success_request.block_keys = plan.pipeline_keys;
        known_success_request.location_ids.reserve(plan.pipeline_location_ids.size());
        for (const auto &location_id : plan.pipeline_location_ids) {
            known_success_request.location_ids.push_back({location_id});
        }
        if (!schedule_plan_executor || !schedule_plan_executor->SubmitNonBlocking(
                                           known_success_request, ScheduleTaskClass::kMigrationContinuation)) {
            KVCM_LOG_WARN("[%s] rollback known successful migration locations failed to submit delete, instance %s, "
                          "key_count %zu",
                          trace_id.c_str(),
                          instance_id.c_str(),
                          known_success_request.block_keys.size());
        }
    }

    std::unordered_map<std::string, std::vector<DataStorageUri>> direct_delete_uris;
    for (const size_t index : plan.direct_delete_indices) {
        const auto &item = items[index];
        if (item.uris.empty()) {
            continue;
        }
        auto &uris = direct_delete_uris[item.storage_name];
        uris.insert(uris.end(), item.uris.begin(), item.uris.end());
    }
    if (direct_delete_uris.empty()) {
        return;
    }
    if (!data_storage_manager) {
        KVCM_LOG_WARN("[%s] rollback migration URIs failed: data storage manager missing, instance %s",
                      trace_id.c_str(),
                      instance_id.c_str());
        return;
    }
    for (const auto &[storage_name, uris] : direct_delete_uris) {
        const auto delete_results = data_storage_manager->Delete(request_context, storage_name, uris, nullptr);
        if (delete_results.size() != uris.size()) {
            KVCM_LOG_WARN("[%s] rollback migration URI result count mismatch, instance %s, storage %s, expected %zu, "
                          "got %zu",
                          trace_id.c_str(),
                          instance_id.c_str(),
                          storage_name.c_str(),
                          uris.size(),
                          delete_results.size());
        }
        const std::size_t result_count = std::min(delete_results.size(), uris.size());
        for (std::size_t i = 0; i < result_count; ++i) {
            if (delete_results[i] != EC_OK && delete_results[i] != EC_NOENT) {
                KVCM_LOG_WARN("[%s] rollback migration URI failed, instance %s, storage %s, uri %s, ec %d",
                              trace_id.c_str(),
                              instance_id.c_str(),
                              storage_name.c_str(),
                              uris[i].ToUriString().c_str(),
                              delete_results[i]);
            }
        }
    }
}

std::optional<int64_t> ParsePositiveInt64(const std::string &value) {
    if (value.empty()) {
        return std::nullopt;
    }
    int64_t parsed = 0;
    const char *begin = value.data();
    const char *end = begin + value.size();
    const auto [ptr, ec] = std::from_chars(begin, end, parsed);
    if (ec != std::errc{} || ptr != end || parsed <= 0) {
        return std::nullopt;
    }
    return parsed;
}

std::optional<uint64_t> ParsePositiveUriSize(const DataStorageUri &uri) {
    if (!uri.Valid() || !uri.HasParam("size")) {
        return std::nullopt;
    }
    const std::string value = uri.GetParam("size");
    uint64_t parsed = 0;
    const char *begin = value.data();
    const char *end = begin + value.size();
    const auto [ptr, ec] = std::from_chars(begin, end, parsed);
    if (ec != std::errc{} || ptr != end || parsed == 0) {
        return std::nullopt;
    }
    return parsed;
}

std::optional<uint64_t> SumSourceUriSizes(const std::vector<LocationSpec> &specs) {
    if (specs.empty()) {
        return std::nullopt;
    }
    uint64_t total = 0;
    for (const auto &spec : specs) {
        const auto size = ParsePositiveUriSize(DataStorageUri(spec.uri()));
        if (!size.has_value() || total > std::numeric_limits<uint64_t>::max() - *size) {
            return std::nullopt;
        }
        total += *size;
    }
    return total;
}

// mark property 解析结果。target 为空表示无标记或已清；target 非空时 deadline
// 必须是合法正整数，malformed 表示缺失、非数字、越界或非正值；expired 表示已过期待清理。
struct MarkInfo {
    std::string target;
    int64_t deadline_ms = 0;
    bool malformed = false;
    bool expired = false;
};

static MarkInfo ParseMarkFromProperties(const PropertyMap &props, int64_t now_ms) {
    MarkInfo info;
    auto tit = props.find(MigrationManager::PROPERTY_TIERED_WRITE_TARGET);
    if (tit == props.end() || tit->second.empty()) {
        return info;
    }
    info.target = tit->second;
    auto dit = props.find(MigrationManager::PROPERTY_TIERED_WRITE_DEADLINE_MS);
    const auto deadline = dit == props.end() ? std::nullopt : ParsePositiveInt64(dit->second);
    if (!deadline.has_value()) {
        info.malformed = true;
        return info;
    }
    info.deadline_ms = *deadline;
    if (info.deadline_ms <= now_ms) {
        info.expired = true;
    }
    return info;
}

// 收集目标 storage 上指定 status 的 location 联合覆盖的 spec name 集合。
// 一次 O(L·S) 扫描替代旧的 per-location 全覆盖判断（O(L²·S²)）。
std::unordered_set<std::string> CollectCoveredSpecNames(const CacheLocationMap &loc_map,
    const std::string &storage_name,
    std::initializer_list<CacheLocationStatus> statuses) {
    std::unordered_set<std::string> covered;
    for (const auto &[_, loc_ptr] : loc_map) {
        if (!loc_ptr || std::find(statuses.begin(), statuses.end(), loc_ptr->status()) == statuses.end()) {
            continue;
        }
        for (const auto &spec : loc_ptr->location_specs()) {
            if (DataStorageUri uri(spec.uri()); uri.Valid() && uri.GetHostName() == storage_name) {
                covered.insert(spec.name());
            }
        }
    }
    return covered;
}

std::vector<const CacheLocation *> FindLocationsOnStorage(const CacheLocationMap &loc_map,
                                                          const std::string &storage_name,
                                                          std::initializer_list<CacheLocationStatus> statuses) {
    std::vector<const CacheLocation *> locations;
    for (const auto &[_, loc_ptr] : loc_map) {
        if (!loc_ptr || std::find(statuses.begin(), statuses.end(), loc_ptr->status()) == statuses.end()) {
            continue;
        }
        for (const auto &spec : loc_ptr->location_specs()) {
            if (DataStorageUri uri(spec.uri()); uri.Valid() && uri.GetHostName() == storage_name) {
                locations.push_back(loc_ptr.get());
                break;
            }
        }
    }
    return locations;
}

template <typename T>
void HashCombine(std::size_t &seed, const T &value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
}
} // namespace

// Mark 持久化属性名（block 级 property）。带 inner 前缀避免与业务属性冲突。
const std::string MigrationManager::PROPERTY_TIERED_WRITE_TARGET = "__mig_tier_target__";
const std::string MigrationManager::PROPERTY_TIERED_WRITE_DEADLINE_MS = "__mig_tier_deadline_ms__";

std::size_t MigrationManager::CopySourceFailureKeyHash::operator()(const CopySourceFailureKey &key) const {
    std::size_t seed = 0;
    HashCombine(seed, key.instance_id);
    HashCombine(seed, key.block_key);
    HashCombine(seed, key.location_id);
    HashCombine(seed, key.location_create_time);
    HashCombine(seed, key.target_storage_name);
    return seed;
}

MigrationManager::MigrationManager(std::shared_ptr<SchedulePlanExecutor> schedule_plan_executor,
                                   std::shared_ptr<MetaIndexerManager> meta_indexer_manager,
                                   std::shared_ptr<DataStorageManager> data_storage_manager,
                                   std::shared_ptr<MetricsRegistry> metrics_registry,
                                   std::shared_ptr<EventManager> event_manager,
                                   std::shared_ptr<RegistryManager> registry_manager,
                                   std::shared_ptr<DataStorageSelector> data_storage_selector)
    : schedule_plan_executor_(std::move(schedule_plan_executor))
    , meta_indexer_manager_(std::move(meta_indexer_manager))
    , data_storage_manager_(std::move(data_storage_manager))
    , metrics_registry_(std::move(metrics_registry))
    , event_manager_(std::move(event_manager))
    , registry_manager_(std::move(registry_manager))
    , data_storage_selector_(std::move(data_storage_selector)) {
    if (metrics_registry_ != nullptr) {
        metrics_enabled_ = true;
        m_tasks_submitted_total_ = metrics_registry_->GetCounter("migration.tasks_submitted_total");
        m_tasks_completed_success_ =
            metrics_registry_->GetCounter("migration.tasks_completed_total", {{"status", "success"}});
        m_tasks_completed_failed_ =
            metrics_registry_->GetCounter("migration.tasks_completed_total", {{"status", "failed"}});
        m_tasks_completed_cancelled_ =
            metrics_registry_->GetCounter("migration.tasks_completed_total", {{"status", "cancelled"}});
        m_tasks_active_ = metrics_registry_->GetGauge("migration.tasks_active");
        m_copy_bytes_total_ = metrics_registry_->GetCounter("migration.copy_bytes_total");
        m_copy_duration_ms_ = metrics_registry_->GetGauge("migration.copy_duration_ms");
        m_source_failures_recorded_total_ = metrics_registry_->GetCounter("migration.source_failures_recorded_total");
        m_source_retries_suppressed_total_ = metrics_registry_->GetCounter("migration.source_retries_suppressed_total");
        m_source_switches_total_ = metrics_registry_->GetCounter("migration.source_switches_total");
        m_no_usable_source_total_ = metrics_registry_->GetCounter("migration.no_usable_source_total");
        m_source_failure_entries_ = metrics_registry_->GetGauge("migration.source_failure_entries");
        m_marks_active_ = metrics_registry_->GetGauge("migration.marks_active");
        m_marks_consumed_total_ = metrics_registry_->GetCounter("migration.marks_consumed_total");
        m_marks_expired_total_ = metrics_registry_->GetCounter("migration.marks_expired_total");
        m_mark_query_errors_total_ = metrics_registry_->GetCounter("migration.mark_query_errors_total");
    }
}

void MigrationManager::UpdateActiveTasksGauge() {
    if (metrics_enabled_) {
        m_tasks_active_ = static_cast<double>(ActiveTaskCountUnsafe());
    }
}

void MigrationManager::UpdateMarksActiveGauge() {
    if (metrics_enabled_) {
        // best-effort：持久化方案下无内存表，近似为 added - cleared（不计随 block 回收）。
        const int64_t added = static_cast<int64_t>(stat_marks_added_.load(std::memory_order_relaxed));
        const int64_t cleared = static_cast<int64_t>(stat_marks_cleared_.load(std::memory_order_relaxed));
        m_marks_active_ = static_cast<double>(added > cleared ? added - cleared : 0);
    }
}

void MigrationManager::UpdateCopySourceFailureGaugeLocked() {
    if (metrics_enabled_) {
        m_source_failure_entries_ = static_cast<double>(copy_source_failures_.size());
    }
}

MigrationManager::~MigrationManager() { Stop(); }

ErrorCode MigrationManager::CheckTargetStorageAdmission(const std::string &trace_id,
                                                        const std::string &instance_group_name,
                                                        const std::string &instance_id,
                                                        const std::string &target_storage_name) const {
    if (data_storage_manager_ == nullptr || target_storage_name.empty()) {
        return EC_BADARGS;
    }
    std::string group_name = instance_group_name;
    if (group_name.empty() && registry_manager_ != nullptr) {
        auto request_context =
            std::make_shared<RequestContext>(trace_id.empty() ? "migration_target_admission" : trace_id);
        const auto instance_info = registry_manager_->GetInstanceInfo(request_context.get(), instance_id);
        if (instance_info == nullptr) {
            return EC_INSTANCE_NOT_EXIST;
        }
        group_name = instance_info->instance_group_name();
    }
    if (data_storage_selector_ != nullptr && !group_name.empty()) {
        auto request_context =
            std::make_shared<RequestContext>(trace_id.empty() ? "migration_target_admission" : trace_id);
        const auto admissions =
            data_storage_selector_->CheckExplicitWriteTargets(request_context.get(), group_name, {target_storage_name});
        if (admissions.size() != 1 || !admissions[0].Allowed()) {
            const auto ec = admissions.size() == 1 ? admissions[0].ec : EC_ERROR;
            KVCM_LOG_WARN("[%s] reject migration target storage %s for instance %s, admission status %d, ec %d",
                          trace_id.c_str(),
                          target_storage_name.c_str(),
                          instance_id.c_str(),
                          admissions.size() == 1 ? static_cast<int>(admissions[0].status)
                                                 : static_cast<int>(StorageTargetAdmissionStatus::kReadError),
                          ec);
            return ec;
        }
        return EC_OK;
    }

    // Test-only/legacy construction may not provide RegistryManager. Availability is still a
    // mandatory target invariant; quota evaluation requires the instance-group context.
    const auto backend = data_storage_manager_->GetDataStorageBackend(target_storage_name);
    if (backend == nullptr || !backend->Available()) {
        return EC_NOENT;
    }
    return EC_OK;
}

bool MigrationManager::ReserveAsyncCopyCredit(const MigrationRequest &request, uint64_t total_bytes) {
    if (request.copy_execution_mode != MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        return true;
    }
    if (request.instance_group_name.empty() || request.async_operation_id.empty() || total_bytes == 0 ||
        request.copy_max_inflight_bytes == 0 || request.copy_max_quarantine_operations <= 0 ||
        request.copy_max_quarantine_bytes == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(async_copy_usage_mutex_);
    if (async_copy_inflight_operation_ids_.count(request.async_operation_id) > 0 ||
        async_copy_quarantine_by_operation_.count(request.async_operation_id) > 0) {
        return false;
    }
    auto &usage = async_copy_usage_by_group_[request.instance_group_name];
    const uint64_t max_quarantine_operations = static_cast<uint64_t>(request.copy_max_quarantine_operations);
    // Every inflight operation may have to fail closed into quarantine. Reserve
    // that worst-case capacity at admission time; checking only the already
    // quarantined usage lets a wave of concurrent failures exceed the advertised
    // hard limits by the entire inflight set.
    const bool quarantine_operation_exhausted =
        usage.quarantine_operations >= max_quarantine_operations ||
        usage.inflight_operations >= max_quarantine_operations - usage.quarantine_operations;
    const bool quarantine_bytes_exhausted =
        total_bytes > request.copy_max_quarantine_bytes ||
        usage.quarantine_bytes > request.copy_max_quarantine_bytes - total_bytes ||
        usage.inflight_bytes > request.copy_max_quarantine_bytes - total_bytes - usage.quarantine_bytes;
    if (quarantine_operation_exhausted || quarantine_bytes_exhausted || total_bytes > request.copy_max_inflight_bytes ||
        usage.inflight_bytes > request.copy_max_inflight_bytes - total_bytes) {
        return false;
    }
    async_copy_inflight_operation_ids_.insert(request.async_operation_id);
    ++usage.inflight_operations;
    usage.inflight_bytes += total_bytes;
    return true;
}

void MigrationManager::ReleaseAsyncCopyCredit(const CopyTaskContext &ctx) {
    if (!ctx.async_credit_active || ctx.instance_group_name.empty() || ctx.async_operation_id.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(async_copy_usage_mutex_);
    if (async_copy_inflight_operation_ids_.erase(ctx.async_operation_id) == 0) {
        return;
    }
    const auto iter = async_copy_usage_by_group_.find(ctx.instance_group_name);
    if (iter == async_copy_usage_by_group_.end()) {
        return;
    }
    auto &usage = iter->second;
    usage.inflight_operations = usage.inflight_operations > 0 ? usage.inflight_operations - 1 : 0;
    usage.inflight_bytes = usage.inflight_bytes >= ctx.total_bytes ? usage.inflight_bytes - ctx.total_bytes : 0;
    if (usage.inflight_operations == 0 && usage.inflight_bytes == 0 && usage.quarantine_operations == 0 &&
        usage.quarantine_bytes == 0) {
        async_copy_usage_by_group_.erase(iter);
    }
}

void MigrationManager::MoveAsyncCopyCreditToQuarantine(const CopyTaskContext &ctx, const std::string &reason) {
    if (!ctx.async_credit_active || ctx.instance_group_name.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(async_copy_usage_mutex_);
    if (ctx.async_operation_id.empty() || async_copy_quarantine_by_operation_.count(ctx.async_operation_id) > 0 ||
        async_copy_inflight_operation_ids_.erase(ctx.async_operation_id) == 0) {
        return;
    }
    MigrationCopyGuard guard;
    const int64_t now_us = TimestampUtil::GetCurrentTimeUs();
    guard.set_schema_version(MigrationCopyGuard::kCurrentSchemaVersion);
    guard.set_state(MigrationCopyGuardState::MCGS_UNKNOWN);
    guard.set_operation_id(ctx.async_operation_id);
    guard.set_source_location_id(ctx.src_location_id);
    guard.set_source_location_create_time(ctx.src_create_time);
    guard.set_source_storage_name(ctx.src_storage_name);
    guard.set_target_storage_name(ctx.dst_storage_name);
    guard.set_migration_retention(static_cast<int32_t>(ctx.retention));
    guard.set_mark_target(ctx.mark_target);
    guard.set_mark_deadline_ms(ctx.mark_deadline_ms);
    guard.set_total_bytes(ctx.total_bytes);
    guard.set_backend_task_ids(ctx.async_backend_task_ids);
    guard.set_create_time_us(ctx.async_guard_create_time_us > 0 ? ctx.async_guard_create_time_us : now_us);
    guard.set_update_time_us(now_us);
    guard.set_last_error(reason);
    async_copy_quarantine_by_operation_.emplace(
        ctx.async_operation_id,
        AsyncCopyQuarantineRecord{
            ctx.instance_group_name, ctx.instance_id, ctx.block_key, ctx.dst_location_id, std::move(guard)});
    auto &usage = async_copy_usage_by_group_[ctx.instance_group_name];
    usage.inflight_operations = usage.inflight_operations > 0 ? usage.inflight_operations - 1 : 0;
    usage.inflight_bytes = usage.inflight_bytes >= ctx.total_bytes ? usage.inflight_bytes - ctx.total_bytes : 0;
    ++usage.quarantine_operations;
    if (usage.quarantine_bytes <= std::numeric_limits<uint64_t>::max() - ctx.total_bytes) {
        usage.quarantine_bytes += ctx.total_bytes;
    } else {
        usage.quarantine_bytes = std::numeric_limits<uint64_t>::max();
    }
}

bool MigrationManager::RestoreAsyncCopyInflightCredit(const CopyTaskContext &ctx) {
    if (!ctx.async_credit_active || ctx.instance_group_name.empty() || ctx.async_operation_id.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(async_copy_usage_mutex_);
    if (!async_copy_inflight_operation_ids_.insert(ctx.async_operation_id).second) {
        return false;
    }
    auto &usage = async_copy_usage_by_group_[ctx.instance_group_name];
    ++usage.inflight_operations;
    usage.inflight_bytes = usage.inflight_bytes <= std::numeric_limits<uint64_t>::max() - ctx.total_bytes
                               ? usage.inflight_bytes + ctx.total_bytes
                               : std::numeric_limits<uint64_t>::max();
    return true;
}

std::vector<MigrationManager::AsyncCopyQuarantineRecord>
MigrationManager::ListAsyncCopyQuarantine(const std::string &instance_group_name) const {
    std::vector<AsyncCopyQuarantineRecord> records;
    std::lock_guard<std::mutex> lock(async_copy_usage_mutex_);
    records.reserve(async_copy_quarantine_by_operation_.size());
    for (const auto &[_, record] : async_copy_quarantine_by_operation_) {
        if (instance_group_name.empty() || record.instance_group_name == instance_group_name) {
            records.push_back(record);
        }
    }
    std::sort(records.begin(), records.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.instance_group_name != rhs.instance_group_name) {
            return lhs.instance_group_name < rhs.instance_group_name;
        }
        if (lhs.instance_id != rhs.instance_id) {
            return lhs.instance_id < rhs.instance_id;
        }
        if (lhs.block_key != rhs.block_key) {
            return lhs.block_key < rhs.block_key;
        }
        return lhs.guard.operation_id() < rhs.guard.operation_id();
    });
    return records;
}

ErrorCode MigrationManager::BreakGlassReleaseAsyncCopy(const std::string &operation_id,
                                                        const std::string &operator_name,
                                                        const std::string &external_fencing_evidence) {
    if (operation_id.empty() || operator_name.empty() || external_fencing_evidence.empty()) {
        return EC_BADARGS;
    }
    AsyncCopyQuarantineRecord record;
    {
        std::lock_guard<std::mutex> lock(async_copy_usage_mutex_);
        const auto iter = async_copy_quarantine_by_operation_.find(operation_id);
        if (iter == async_copy_quarantine_by_operation_.end()) {
            return EC_NOENT;
        }
        record = iter->second;
    }
    if (record.guard.state() != MigrationCopyGuardState::MCGS_UNKNOWN) {
        return EC_MISMATCH;
    }
    auto indexer = meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(record.instance_id) : nullptr;
    if (!indexer) {
        return EC_INSTANCE_NOT_EXIST;
    }
    MetaSearcher meta_searcher(indexer);
    auto request_context = std::make_shared<RequestContext>("migration_break_glass_release");
    MetaSearcher::LocationCASTask task;
    task.location_id = record.target_location_id;
    task.old_status = CLS_WRITING;
    task.new_status = CLS_DELETING;
    task.expected_operation_id = operation_id;
    task.expected_migration_copy_guard_state = MigrationCopyGuardState::MCGS_UNKNOWN;
    task.clear_migration_copy_guard = true;
    std::vector<std::vector<ErrorCode>> results;
    const auto ec = meta_searcher.BatchCASLocationStatus(
        request_context.get(), {record.block_key}, {{std::move(task)}}, results, true);
    const auto transition_is_persistent = [&]() {
        CacheLocationMapVector location_maps;
        const auto get_result =
            indexer->GetLocationsFromPersistent(request_context.get(), {record.block_key}, location_maps);
        if (get_result.error_codes.size() != 1 || get_result.error_codes[0] != EC_OK || location_maps.size() != 1) {
            return false;
        }
        const auto location_iter = location_maps[0].find(record.target_location_id);
        return location_iter != location_maps[0].end() && location_iter->second &&
               location_iter->second->status() == CLS_DELETING && !location_iter->second->has_migration_copy_guard();
    };
    const bool cas_applied = ec == EC_OK && results.size() == 1 && results[0].size() == 1 && results[0][0] == EC_OK;
    if (cas_applied) {
        // Sync timeout is an ambiguous durability result, not proof that the
        // transition failed. Re-read the persistent backend before returning;
        // otherwise a retry would CAS guarded WRITING after the first attempt
        // had already durably produced guard-free DELETING and cleanup would
        // be stranded forever.
        if (!indexer->Sync({record.block_key}) && !transition_is_persistent()) {
            return EC_ERROR;
        }
    } else if (!transition_is_persistent()) {
        if (ec != EC_OK) {
            return ec;
        }
        if (results.size() != 1 || results[0].size() != 1) {
            return EC_MISMATCH;
        }
        return results[0][0];
    }

    CopyTaskContext ctx;
    ctx.instance_group_name = record.instance_group_name;
    ctx.instance_id = record.instance_id;
    ctx.block_key = record.block_key;
    ctx.dst_location_id = record.target_location_id;
    ctx.dst_storage_name = record.guard.target_storage_name();
    SubmitPreparedTargetLocationDelete(ctx);
    {
        std::lock_guard<std::mutex> lock(async_copy_usage_mutex_);
        const auto iter = async_copy_quarantine_by_operation_.find(operation_id);
        if (iter != async_copy_quarantine_by_operation_.end()) {
            auto usage_iter = async_copy_usage_by_group_.find(iter->second.instance_group_name);
            if (usage_iter != async_copy_usage_by_group_.end()) {
                auto &usage = usage_iter->second;
                usage.quarantine_operations = usage.quarantine_operations > 0 ? usage.quarantine_operations - 1 : 0;
                usage.quarantine_bytes = usage.quarantine_bytes >= iter->second.guard.total_bytes()
                                             ? usage.quarantine_bytes - iter->second.guard.total_bytes()
                                             : 0;
                if (usage.inflight_operations == 0 && usage.inflight_bytes == 0 && usage.quarantine_operations == 0 &&
                    usage.quarantine_bytes == 0) {
                    async_copy_usage_by_group_.erase(usage_iter);
                }
            }
            async_copy_quarantine_by_operation_.erase(iter);
        }
    }
    KVCM_LOG_ERROR("BREAK_GLASS asynchronous Copy quarantine released after external fencing: "
                   "operator %s operation_id %s instance %s block_key %lld target_location %s evidence %s",
                   operator_name.c_str(),
                   operation_id.c_str(),
                   record.instance_id.c_str(),
                   static_cast<long long>(record.block_key),
                   record.target_location_id.c_str(),
                   external_fencing_evidence.c_str());
    return EC_OK;
}

bool MigrationManager::HasAsyncCopyStorageReference(const std::string &storage_name) const {
    if (storage_name.empty()) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        for (const auto &[_, tasks] : active_tasks_by_instance_) {
            for (const auto &[__, ctx] : tasks) {
                if (ctx.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED &&
                    (ctx.src_storage_name == storage_name || ctx.dst_storage_name == storage_name)) {
                    return true;
                }
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(async_copy_usage_mutex_);
        if (std::any_of(async_copy_quarantine_by_operation_.begin(),
                        async_copy_quarantine_by_operation_.end(),
                        [&storage_name](const auto &entry) {
                            const auto &guard = entry.second.guard;
                            return guard.source_storage_name() == storage_name ||
                                   guard.target_storage_name() == storage_name;
                        })) {
            return true;
        }
    }
    std::lock_guard<std::mutex> cleanup_lock(pending_cleanup_mutex_);
    return std::any_of(pending_location_cleanups_.begin(),
                       pending_location_cleanups_.end(),
                       [&storage_name](const auto &cleanup) { return cleanup.storage_name == storage_name; });
}

bool MigrationManager::HasAsyncCopyInstanceReference(const std::string &instance_id) const {
    if (instance_id.empty()) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        const auto iter = active_tasks_by_instance_.find(instance_id);
        if (iter != active_tasks_by_instance_.end() && !iter->second.empty()) {
            return true;
        }
    }
    {
        std::lock_guard<std::mutex> lock(async_copy_usage_mutex_);
        if (std::any_of(async_copy_quarantine_by_operation_.begin(),
                        async_copy_quarantine_by_operation_.end(),
                        [&instance_id](const auto &entry) { return entry.second.instance_id == instance_id; })) {
            return true;
        }
    }
    std::lock_guard<std::mutex> cleanup_lock(pending_cleanup_mutex_);
    return std::any_of(pending_location_cleanups_.begin(),
                       pending_location_cleanups_.end(),
                       [&instance_id](const auto &cleanup) { return cleanup.instance_id == instance_id; });
}

bool MigrationManager::RecoverAsyncCopyGuards() {
    if (!registry_manager_ || !meta_indexer_manager_ || !schedule_plan_executor_) {
        return true;
    }
    struct RecoveryCandidate {
        CopyTaskContext ctx;
        MigrationCopyGuard guard;
        size_t expected_items = 0;
        AsyncCopyOptions options;
    };
    std::vector<RecoveryCandidate> candidates;
    const auto recovery_started_at = std::chrono::steady_clock::now();
    uint64_t scanned_batches = 0;
    uint64_t scanned_keys = 0;
    auto request_context = std::make_shared<RequestContext>("migration_async_guard_recovery");
    auto [groups_ec, groups] = registry_manager_->ListInstanceGroup(request_context.get());
    if (groups_ec != EC_OK) {
        KVCM_LOG_ERROR("asynchronous Copy guard recovery failed to list instance groups, ec %d",
                       static_cast<int>(groups_ec));
        return false;
    }
    for (const auto &group : groups) {
        if (!group) {
            continue;
        }
        auto [instances_ec, instances] = registry_manager_->ListInstanceInfo(request_context.get(), group->name());
        if (instances_ec != EC_OK) {
            KVCM_LOG_ERROR("asynchronous Copy guard recovery failed to list instances for group %s, ec %d",
                           group->name().c_str(),
                           static_cast<int>(instances_ec));
            return false;
        }
        AsyncCopyOptions options;
        if (group->cache_config()) {
            options.operation_deadline_ms = group->cache_config()->migration_copy_operation_deadline_ms();
            options.initial_poll_interval_ms = group->cache_config()->migration_copy_poll_initial_interval_ms();
            options.max_poll_interval_ms = group->cache_config()->migration_copy_poll_max_interval_ms();
            options.connect_timeout_ms = group->cache_config()->migration_copy_connect_timeout_ms();
            options.submit_timeout_ms = group->cache_config()->migration_copy_submit_timeout_ms();
            options.query_timeout_ms = group->cache_config()->migration_copy_query_timeout_ms();
        }
        for (const auto &instance : instances) {
            if (!instance) {
                continue;
            }
            auto indexer = meta_indexer_manager_->GetMetaIndexer(instance->instance_id());
            if (!indexer) {
                KVCM_LOG_ERROR("asynchronous Copy guard recovery missing indexer for instance %s",
                               instance->instance_id().c_str());
                return false;
            }
            std::string cursor = SCAN_BASE_CURSOR;
            do {
                if (std::chrono::steady_clock::now() - recovery_started_at >= kAsyncGuardRecoveryMaxDuration) {
                    KVCM_LOG_ERROR("asynchronous Copy guard recovery exceeded its %lld ms startup budget: "
                                   "batches %llu keys %llu",
                                   static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                              kAsyncGuardRecoveryMaxDuration)
                                                              .count()),
                                   static_cast<unsigned long long>(scanned_batches),
                                   static_cast<unsigned long long>(scanned_keys));
                    return false;
                }
                MaintenanceScanBatch batch;
                const auto scan_ec = indexer->ScanLocationsForMaintenance(
                    request_context.get(), cursor, kAsyncGuardRecoveryScanBatchSize, batch);
                if (scan_ec != EC_OK || batch.keys.size() != batch.locations.size() ||
                    batch.keys.size() != batch.location_results.size()) {
                    KVCM_LOG_ERROR("asynchronous Copy guard recovery scan failed for instance %s cursor %s ec %d",
                                   instance->instance_id().c_str(),
                                   cursor.c_str(),
                                   static_cast<int>(scan_ec));
                    return false;
                }
                ++scanned_batches;
                scanned_keys += batch.keys.size();
                for (size_t i = 0; i < batch.keys.size(); ++i) {
                    if (batch.location_results[i] == EC_NOENT) {
                        continue;
                    }
                    if (batch.location_results[i] != EC_OK) {
                        // A malformed/future migration guard is rejected by
                        // CacheLocation parsing.  Silently skipping that block
                        // would enable admission while its durable ownership
                        // fence is not understood by this binary.  Keep leader
                        // startup fail-closed and require operator/version
                        // intervention.
                        KVCM_LOG_ERROR("asynchronous Copy guard recovery cannot decode block metadata: "
                                       "instance %s block_key %lld ec %d",
                                       instance->instance_id().c_str(),
                                       static_cast<long long>(batch.keys[i]),
                                       static_cast<int>(batch.location_results[i]));
                        return false;
                    }
                    for (const auto &[location_id, location] : batch.locations[i]) {
                        if (!location || !location->has_migration_copy_guard()) {
                            continue;
                        }
                        const auto &guard = location->migration_copy_guard();
                        CopyTaskContext ctx;
                        ctx.instance_group_name = group->name();
                        ctx.instance_id = instance->instance_id();
                        ctx.block_key = batch.keys[i];
                        ctx.src_location_id = guard.source_location_id();
                        ctx.src_create_time = guard.source_location_create_time();
                        ctx.src_storage_name = guard.source_storage_name();
                        ctx.dst_storage_name = guard.target_storage_name();
                        ctx.dst_location_id = location_id;
                        ctx.copy_execution_mode = MigrationCopyExecutionMode::ASYNC_REQUIRED;
                        ctx.async_operation_id = guard.operation_id();
                        ctx.async_backend_task_ids = guard.backend_task_ids();
                        ctx.async_guard_create_time_us = guard.create_time_us();
                        ctx.async_credit_active = true;
                        ctx.async_transition_mutex = std::make_shared<std::mutex>();
                        ctx.total_bytes = guard.total_bytes();
                        ctx.mark_target = guard.mark_target();
                        ctx.mark_deadline_ms = guard.mark_deadline_ms();
                        ctx.submit_time = std::chrono::steady_clock::now();
                        const auto retention = static_cast<MigrationRetention>(guard.migration_retention());
                        ctx.retention = retention == MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE ||
                                                retention == MigrationRetention::MIGRATION_RETENTION_KEEP_BOTH
                                            ? retention
                                            : MigrationRetention::MIGRATION_RETENTION_KEEP_BOTH;
                        candidates.push_back(
                            RecoveryCandidate{std::move(ctx), guard, location->location_specs().size(), options});
                    }
                }
                cursor = batch.next_cursor;
                if (cursor != SCAN_BASE_CURSOR) {
                    std::this_thread::sleep_for(kAsyncGuardRecoveryScanPause);
                }
            } while (cursor != SCAN_BASE_CURSOR);
        }
    }

    // Validate the complete recovery set before replacing the in-memory
    // accounting tables.  A partial rebuild would either double-charge an
    // operation or lose its capacity fence.
    {
        std::unordered_map<std::string, std::string> operation_identities;
        operation_identities.reserve(candidates.size());
        std::vector<RecoveryCandidate> deduplicated;
        deduplicated.reserve(candidates.size());
        for (auto &candidate : candidates) {
            const std::string identity = candidate.ctx.instance_id + "\x1f" + std::to_string(candidate.ctx.block_key) +
                                         "\x1f" + candidate.ctx.dst_location_id + "\x1f" +
                                         candidate.guard.ToJsonString();
            const auto [identity_iter, inserted] =
                operation_identities.emplace(candidate.ctx.async_operation_id, identity);
            if (candidate.ctx.instance_group_name.empty() || candidate.ctx.instance_id.empty() ||
                candidate.ctx.dst_location_id.empty() || candidate.ctx.total_bytes == 0 ||
                candidate.ctx.async_operation_id.empty() || (!inserted && identity_iter->second != identity)) {
                KVCM_LOG_ERROR("asynchronous Copy guard recovery found invalid candidate: group %s instance %s "
                               "block %lld target %s operation %s bytes %llu",
                               candidate.ctx.instance_group_name.c_str(),
                               candidate.ctx.instance_id.c_str(),
                               static_cast<long long>(candidate.ctx.block_key),
                               candidate.ctx.dst_location_id.c_str(),
                               candidate.ctx.async_operation_id.c_str(),
                               static_cast<unsigned long long>(candidate.ctx.total_bytes));
                return false;
            }
            if (!inserted) {
                KVCM_LOG_DEBUG("deduplicate asynchronous Copy guard returned by maintenance scan: "
                               "instance %s block %lld target %s operation %s",
                               candidate.ctx.instance_id.c_str(),
                               static_cast<long long>(candidate.ctx.block_key),
                               candidate.ctx.dst_location_id.c_str(),
                               candidate.ctx.async_operation_id.c_str());
                continue;
            }
            deduplicated.push_back(std::move(candidate));
        }
        candidates = std::move(deduplicated);
    }
    {
        std::lock_guard<std::mutex> lock(async_copy_usage_mutex_);
        async_copy_usage_by_group_.clear();
        async_copy_quarantine_by_operation_.clear();
        async_copy_inflight_operation_ids_.clear();
    }
    size_t resumed = 0;
    size_t quarantined = 0;
    for (auto &candidate : candidates) {
        auto &ctx = candidate.ctx;
        const auto state = candidate.guard.state();
        if (!RestoreAsyncCopyInflightCredit(ctx)) {
            KVCM_LOG_ERROR("asynchronous Copy guard recovery found duplicate/invalid operation id %s",
                           ctx.async_operation_id.c_str());
            return false;
        }
        const bool recoverable =
            (state == MigrationCopyGuardState::MCGS_ACTIVE || state == MigrationCopyGuardState::MCGS_CANCELLING) &&
            !ctx.async_operation_id.empty() && !ctx.src_storage_name.empty() && ctx.total_bytes > 0 &&
            candidate.expected_items > 0 && ctx.async_backend_task_ids.size() == candidate.expected_items;
        if (!recoverable) {
            if (state != MigrationCopyGuardState::MCGS_UNKNOWN) {
                const auto update_result = UpdateAsyncCopyGuard(ctx,
                                                                state,
                                                                MigrationCopyGuardState::MCGS_UNKNOWN,
                                                                ctx.async_backend_task_ids,
                                                                "leader_recovery_cannot_resume_guard");
                if (update_result != GuardMutationResult::kAppliedDurably) {
                    KVCM_LOG_ERROR("asynchronous Copy recovery could not durably quarantine operation %s",
                                   ctx.async_operation_id.c_str());
                    return false;
                }
            }
            MoveAsyncCopyCreditToQuarantine(ctx, "leader_recovery_cannot_resume_guard");
            ++quarantined;
            continue;
        }
        ctx.state =
            state == MigrationCopyGuardState::MCGS_CANCELLING ? CopyTaskState::kCancelling : CopyTaskState::kRunning;
        bool inserted = false;
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            inserted = InsertActiveTaskLocked(ctx);
        }
        if (!inserted) {
            const auto update_result = UpdateAsyncCopyGuard(ctx,
                                                            state,
                                                            MigrationCopyGuardState::MCGS_UNKNOWN,
                                                            ctx.async_backend_task_ids,
                                                            "leader_recovery_duplicate_block_operation");
            if (update_result != GuardMutationResult::kAppliedDurably) {
                KVCM_LOG_ERROR("asynchronous Copy recovery could not durably quarantine duplicate block "
                               "operation %s",
                               ctx.async_operation_id.c_str());
                return false;
            }
            MoveAsyncCopyCreditToQuarantine(ctx, "leader_recovery_duplicate_block_operation");
            ++quarantined;
            continue;
        }
        const int64_t age_ms =
            candidate.guard.create_time_us() > 0
                ? std::max<int64_t>(0, (TimestampUtil::GetCurrentTimeUs() - candidate.guard.create_time_us()) / 1000)
                                   : 0;
        const int64_t minimum_deadline = candidate.options.max_poll_interval_ms + 1;
        candidate.options.operation_deadline_ms =
            std::max(minimum_deadline, candidate.options.operation_deadline_ms - age_ms);
        auto submit = schedule_plan_executor_->ResumeAsyncCopy(ctx.src_storage_name,
                                                               ctx.async_backend_task_ids,
                                                               candidate.expected_items,
                                                               ctx.async_operation_id,
                                                               candidate.options);
        if (!submit.submit_result.accepted || !submit.future.valid()) {
            if (!CompleteCopyTaskAsUnknown(ctx,
                                      submit.submit_result.detail.empty()
                                          ? "leader_recovery_backend_rejected_resume"
                                               : submit.submit_result.detail)) {
                KVCM_LOG_ERROR("asynchronous Copy recovery could not durably publish UNKNOWN operation %s",
                               ctx.async_operation_id.c_str());
                return false;
            }
            ++quarantined;
            continue;
        }
        if (state == MigrationCopyGuardState::MCGS_CANCELLING) {
            schedule_plan_executor_->RequestCancelAsyncCopy(ctx.src_storage_name, ctx.async_operation_id);
        }
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_copies_.push_back(
                PendingCopy{ctx.instance_id, ctx.block_key, candidate.expected_items, std::move(submit.future), true});
        }
        ++resumed;
    }
    if (resumed > 0) {
        pending_cv_.notify_all();
    }
    KVCM_LOG_INFO("asynchronous Copy guard recovery completed: batches %llu keys %llu discovered %zu resumed %zu "
                  "quarantined %zu",
                  static_cast<unsigned long long>(scanned_batches),
                  static_cast<unsigned long long>(scanned_keys),
                  candidates.size(),
                  resumed,
                  quarantined);
    return true;
}

ErrorCode MigrationManager::Start() {
    std::unique_lock<std::shared_mutex> lifecycle_lock(copy_submission_lifecycle_mutex_);
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return EC_OK; // already running
    }
    async_prepare_generation_.fetch_add(1, std::memory_order_acq_rel);
    accepting_copy_submissions_.store(false, std::memory_order_release);
    if (!RecoverAsyncCopyGuards()) {
        // Guard-aware Reclaimer/GC fencing remains safe, but opening migration
        // admission before the source-of-truth scan succeeds could create more
        // operations without a complete quarantine budget.
        KVCM_LOG_ERROR("MigrationManager guard recovery failed; new Copy submissions remain disabled");
        running_.store(false, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_copies_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            active_tasks_by_instance_.clear();
            UpdateActiveTasksGauge();
        }
        {
            std::lock_guard<std::mutex> lock(async_copy_usage_mutex_);
            async_copy_usage_by_group_.clear();
            async_copy_quarantine_by_operation_.clear();
            async_copy_inflight_operation_ids_.clear();
        }
        return EC_ERROR;
    }
    // Recovery first installs every in-memory reservation and pending future.
    // Starting the monitor earlier would let a fast completion remove the first
    // reservation before a duplicate durable guard for the same block is seen.
    monitor_thread_ = std::thread([this]() { MonitorLoop(); });
    accepting_copy_submissions_.store(true, std::memory_order_release);
    KVCM_LOG_INFO("MigrationManager started");
    return EC_OK;
}

void MigrationManager::Stop() {
    // 先关全局 gate，后来的 Submit 可在获取 lifecycle shared lock 前快速失败；随后用 unique lock
    // 等待已经持有 shared lock 的提交函数完整返回。该 barrier 只等待 prepare/enqueue，不等待
    // executor 中已异步运行的数据 Copy。
    accepting_copy_submissions_.store(false, std::memory_order_release);
    async_prepare_generation_.fetch_add(1, std::memory_order_acq_rel);
    {
        std::lock_guard<std::mutex> lock(async_prepare_mutex_);
        pending_async_prepare_jobs_.clear();
    }
    std::unique_lock<std::shared_mutex> lifecycle_lock(copy_submission_lifecycle_mutex_);

    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
        return; // not running
    }
    pending_cv_.notify_all();
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
    // Monitor 已停止，丢弃旧 Leader 的 future。这些 future 只是进程内 completion
    // channel；持久 guard + backend task ids 才是切主后的恢复权威。若不清空，同一
    // Manager 重新 Start 后会同时处理 stale future 和恢复出来的新 future。
    std::deque<PendingCopy> dropped_pending_copies;
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        dropped_pending_copies.swap(pending_copies_);
    }
    const size_t dropped_pending = dropped_pending_copies.size();
    size_t harvested_remote_submits = 0;
    size_t finalized_remote_rejections = 0;
    for (auto &cell : dropped_pending_copies) {
        if (!cell.native_async || cell.remote_submit_processed || !cell.remote_submit_future.valid() ||
            cell.remote_submit_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            continue;
        }
        try {
            auto remote_result = cell.remote_submit_future.get();
            if (!remote_result.accepted) {
                if (!remote_result.acceptance_unknown) {
                    const bool completed = OnTaskFailedInternal(
                        cell.instance_id,
                        cell.block_key,
                        remote_result.status,
                        remote_result.detail.empty() ? "copy_submit_rejected_during_stop" : remote_result.detail,
                        false,
                        false);
                    if (completed) {
                        ++finalized_remote_rejections;
                    } else {
                        KVCM_LOG_WARN("MigrationManager stop retained a definitively rejected submit while its guard "
                                      "finalization awaits durable confirmation: instance %s block_key %lld",
                                      cell.instance_id.c_str(),
                                      static_cast<long long>(cell.block_key));
                    }
                }
                continue;
            }
            CopyTaskContext accepted_ctx;
            bool cancel_requested = false;
            if (!PersistRemoteAsyncCopyAcceptance(cell.instance_id,
                                                  cell.block_key,
                                                  cell.expected_items,
                                                  remote_result,
                                                  accepted_ctx,
                                                  cancel_requested)) {
                if (!accepted_ctx.src_storage_name.empty() && schedule_plan_executor_) {
                    schedule_plan_executor_->RequestCancelAsyncCopy(accepted_ctx.src_storage_name,
                                                                    remote_result.operation_id);
                }
                KVCM_LOG_ERROR("MigrationManager stop could not persist ready remote-submit handles: "
                               "instance %s block_key %lld operation %s handles %zu expected %zu",
                               cell.instance_id.c_str(),
                               static_cast<long long>(cell.block_key),
                               remote_result.operation_id.c_str(),
                               remote_result.backend_task_ids.size(),
                               cell.expected_items);
                continue;
            }
            ++harvested_remote_submits;
            if (cancel_requested && schedule_plan_executor_) {
                schedule_plan_executor_->RequestCancelAsyncCopy(accepted_ctx.src_storage_name,
                                                                accepted_ctx.async_operation_id);
            }
        } catch (const std::exception &e) {
            KVCM_LOG_ERROR("MigrationManager stop failed to harvest remote-submit future: %s", e.what());
        } catch (...) {
            KVCM_LOG_ERROR("MigrationManager stop failed to harvest remote-submit future with unknown exception");
        }
    }

    // 退出时丢弃进程内活跃任务表。同步任务沿用孤儿 WRITING 清理；原生异步任务若已
    // 持久完整 task ids，则保留 ACTIVE/CANCELLING guard 交给新 Leader Resume，不将一次
    // graceful demotion 伪造成远端 outcome unknown。只有缺少恢复句柄的 SUBMITTING 任务
    // 才 fail-closed 转 UNKNOWN quarantine。
    size_t dropped = 0;
    size_t detached = 0;
    size_t quarantined = 0;
    std::vector<CopyTaskContext> dropped_contexts;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        dropped = ActiveTaskCountUnsafe();
        dropped_contexts.reserve(dropped);
        for (const auto &[_, tasks] : active_tasks_by_instance_) {
            for (const auto &[__, ctx] : tasks) {
                dropped_contexts.push_back(ctx);
            }
        }
        active_tasks_by_instance_.clear();
        UpdateActiveTasksGauge();
    }
    for (const auto &ctx : dropped_contexts) {
        if (ctx.copy_execution_mode != MigrationCopyExecutionMode::ASYNC_REQUIRED) {
            continue;
        }
        if (ctx.async_guard_pending_action != CopyCompletionAction::kNone) {
            // A successful durability retry must also execute the already
            // claimed terminal action (source retention, target cleanup and
            // credit release/quarantine publication).  Merely dropping the
            // in-memory action here would strand those side effects forever.
            if (!ResumePendingGuardFinalization(ctx)) {
                // The final CAS may already have promoted/deleted the target and
                // cleared its guard.  Rewriting UNKNOWN here could regress that
                // terminal state.  Leave the durable outcome untouched: either
                // the terminal CAS is present, or the prior fenced guard will be
                // recovered by the next leader.
                KVCM_LOG_ERROR("MigrationManager stopped with guard finalization durability still unknown: "
                               "instance %s block_key %lld operation_id %s",
                               ctx.instance_id.c_str(),
                               static_cast<long long>(ctx.block_key),
                               ctx.async_operation_id.c_str());
                stat_async_copy_unknown_.fetch_add(1, std::memory_order_relaxed);
            } else {
                KVCM_LOG_WARN("MigrationManager stopped after completing a previously applied guard finalization: "
                              "instance %s block_key %lld operation_id %s",
                              ctx.instance_id.c_str(),
                              static_cast<long long>(ctx.block_key),
                              ctx.async_operation_id.c_str());
            }
            continue;
        }

        const auto expected_guard_state = ExpectedGuardState(ctx);
        const bool recoverable_after_detach =
            !ctx.async_backend_task_ids.empty() && (expected_guard_state == MigrationCopyGuardState::MCGS_ACTIVE ||
             expected_guard_state == MigrationCopyGuardState::MCGS_CANCELLING);
        if (recoverable_after_detach) {
            // Do not cancel the PACE task and do not mutate the durable guard.
            // DataStorage backend Close() drops its local coordinator job; the
            // next Leader reconstructs accounting and attaches with these ids.
            ++detached;
            KVCM_LOG_INFO("MigrationManager detached recoverable asynchronous Copy on stop: "
                          "instance %s block_key %lld operation_id %s tasks %zu state %d",
                          ctx.instance_id.c_str(),
                          static_cast<long long>(ctx.block_key),
                          ctx.async_operation_id.c_str(),
                          ctx.async_backend_task_ids.size(),
                          static_cast<int>(expected_guard_state));
            continue;
        }
        const auto guard_result = UpdateAsyncCopyGuard(ctx,
                             expected_guard_state,
                             MigrationCopyGuardState::MCGS_UNKNOWN,
                             ctx.async_backend_task_ids,
                             "manager_stopped_before_authoritative_completion");
        if (guard_result == GuardMutationResult::kAppliedDurably || HasDurableUnknownGuard(ctx)) {
        MoveAsyncCopyCreditToQuarantine(ctx, "manager_stopped_before_authoritative_completion");
        stat_async_copy_unknown_.fetch_add(1, std::memory_order_relaxed);
        ++quarantined;
        } else {
            // Runtime quarantine is only a projection of durable UNKNOWN.  A
            // new leader will rescan the still-fenced guard and retry the
            // transition; publishing a process-local record here would make
            // break-glass unable to match persistent meta.
            KVCM_LOG_ERROR("MigrationManager stop could not durably persist UNKNOWN guard: instance %s "
                           "block_key %lld operation_id %s mutation_result %d",
                           ctx.instance_id.c_str(),
                           static_cast<long long>(ctx.block_key),
                           ctx.async_operation_id.c_str(),
                           static_cast<int>(guard_result));
        }
    }
    if (dropped > 0) {
        KVCM_LOG_WARN("MigrationManager stopped with %zu active copy task(s) and %zu pending future(s) dropped; "
                      "remote submits harvested %zu definite rejections finalized %zu asynchronous detached %zu "
                      "quarantined %zu",
                      dropped,
                      dropped_pending,
                      harvested_remote_submits,
                      finalized_remote_rejections,
                      detached,
                      quarantined);
    } else {
        KVCM_LOG_INFO("MigrationManager stopped");
    }
}

ErrorCode MigrationManager::PrepareCopyTask(const std::string &trace_id,
                                            MigrationRequest &request,
                                            CopyTaskContext &out_ctx,
                                            std::vector<DataStorageUri> &out_src_uris,
                                            std::vector<DataStorageUri> &out_dst_uris) {
    if (request.instance_id.empty() || request.src_location_id.empty() || request.src_storage_name.empty() ||
        request.dst_storage_name.empty()) {
        KVCM_LOG_WARN("[%s] migration bad args: instance_id/src_location_id/src_storage/dst_storage required",
                      trace_id.c_str());
        return EC_BADARGS;
    }
    if (!data_storage_manager_) {
        return EC_ERROR;
    }
    auto indexer = meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(request.instance_id) : nullptr;
    if (!indexer) {
        KVCM_LOG_WARN("[%s] MetaIndexer not found for instance %s", trace_id.c_str(), request.instance_id.c_str());
        return EC_INSTANCE_NOT_EXIST;
    }

    // 1. 读取源 location specs + create_time。
    // 如果调用方已在 admission 阶段取得 src_specs，直接使用并跳过冗余 BatchGetLocation；
    // public 单条 Submit 仍允许不带快照，此时在这里重读源 location。private BatchSubmit 不走本函数。
    const std::vector<LocationSpec> *src_specs_ptr = nullptr;
    int64_t src_create_time = 0;
    if (!request.src_specs.empty()) {
        src_specs_ptr = &request.src_specs;
        src_create_time = request.src_create_time;
    } else {
        MetaSearcher meta_searcher_for_src(indexer);
        auto ctx_for_src = std::make_shared<RequestContext>(trace_id.empty() ? "migration_prepare" : trace_id);
        std::vector<CacheLocationMap> location_maps;
        BlockMask empty_mask;
        ErrorCode ec =
            meta_searcher_for_src.BatchGetLocation(ctx_for_src.get(), {request.block_key}, empty_mask, location_maps);
        if (ec != EC_OK || location_maps.empty()) {
            KVCM_LOG_WARN(
                "[%s] BatchGetLocation failed for block_key %ld, ec %d", trace_id.c_str(), request.block_key, ec);
            return EC_NOENT;
        }
        auto src_iter = location_maps[0].find(request.src_location_id);
        if (src_iter == location_maps[0].end()) {
            KVCM_LOG_WARN("[%s] source location %s not found for block_key %ld",
                          trace_id.c_str(),
                          request.src_location_id.c_str(),
                          request.block_key);
            return EC_NOENT;
        }
        if (src_iter->second == nullptr) {
            return EC_ERROR;
        }
        const CacheLocation &src_location = *src_iter->second;
        if (src_location.status() != CLS_SERVING) {
            KVCM_LOG_WARN("[%s] source location %s not SERVING (status %d), skip migration",
                          trace_id.c_str(),
                          request.src_location_id.c_str(),
                          src_location.status());
            return EC_MISMATCH;
        }
        if (src_location.location_specs().empty()) {
            KVCM_LOG_WARN("[%s] source location %s has no specs", trace_id.c_str(), request.src_location_id.c_str());
            return EC_ERROR;
        }
        request.src_specs = src_location.location_specs();
        request.src_create_time = src_location.create_time();
        src_specs_ptr = &request.src_specs;
        src_create_time = request.src_create_time;
    }

    const auto &src_specs = *src_specs_ptr;
    const auto total_bytes_result = SumSourceUriSizes(src_specs);
    if (!total_bytes_result.has_value()) {
        KVCM_LOG_WARN("[%s] source location %s has a missing, zero, invalid, or overflowing URI size",
                      trace_id.c_str(),
                      request.src_location_id.c_str());
        return EC_BADARGS;
    }

    // 2. 目标 storage 类型。
    auto dst_backend = data_storage_manager_->GetDataStorageBackend(request.dst_storage_name);
    if (!dst_backend) {
        KVCM_LOG_WARN("[%s] target storage %s not found", trace_id.c_str(), request.dst_storage_name.c_str());
        return EC_NOENT;
    }
    const DataStorageType dst_type = dst_backend->GetType();

    // 3. 逐 spec 在目标 storage 预分配空间，构建目标 location specs + src/dst uri 对。
    MetaSearcher meta_searcher(indexer);
    auto ctx = std::make_shared<RequestContext>(trace_id.empty() ? "migration_prepare" : trace_id);
    std::vector<LocationSpec> dst_specs;
    dst_specs.reserve(src_specs.size());
    out_src_uris.clear();
    out_dst_uris.clear();
    std::vector<DataStorageUri> allocated_for_rollback;
    const std::uint64_t total_bytes = *total_bytes_result;

    auto rollback = [&]() {
        if (!allocated_for_rollback.empty()) {
            data_storage_manager_->Delete(ctx.get(), request.dst_storage_name, allocated_for_rollback, nullptr);
        }
    };

    for (const auto &src_spec : src_specs) {
        DataStorageUri src_uri(src_spec.uri());
        if (!src_uri.Valid()) {
            KVCM_LOG_WARN("[%s] invalid source uri for spec %s", trace_id.c_str(), src_spec.name().c_str());
            rollback();
            return EC_ERROR;
        }
        const auto spec_size_result = ParsePositiveUriSize(src_uri);
        if (!spec_size_result.has_value()) {
            rollback();
            return EC_BADARGS;
        }
        const std::uint64_t spec_size = *spec_size_result;

        std::string dst_key =
            request.instance_id + "/" + src_spec.name() + "/" + StringUtil::Uint64ToHex(request.block_key);
        auto create_results = data_storage_manager_->Create(
            ctx.get(), request.dst_storage_name, {dst_key}, static_cast<size_t>(spec_size), nullptr);
        if (create_results.size() != 1 || create_results[0].first != EC_OK) {
            KVCM_LOG_WARN("[%s] allocate dst space failed on %s for spec %s",
                          trace_id.c_str(),
                          request.dst_storage_name.c_str(),
                          src_spec.name().c_str());
            rollback();
            return EC_ERROR;
        }
        DataStorageUri dst_uri = create_results[0].second;
        allocated_for_rollback.push_back(dst_uri);
        out_src_uris.push_back(src_uri);
        out_dst_uris.push_back(dst_uri);
        dst_specs.emplace_back(src_spec.name(), dst_uri.ToUriString());
    }

    // 4. 建目标 location（BatchAddLocation 总是写 CLS_WRITING 并生成随机 location_id）。
    auto dst_location = std::make_shared<CacheLocation>();
    dst_location->set_type(dst_type);
    dst_location->set_spec_size(dst_specs.size());
    for (auto &spec : dst_specs) {
        dst_location->push_location_spec(std::move(spec));
    }
    std::vector<MetaSearcher::AddLocationResult> add_results;
    ErrorCode ec = meta_searcher.BatchAddLocation(ctx.get(), {request.block_key}, {dst_location}, add_results);
    if (add_results.size() != 1) {
        KVCM_LOG_WARN("[%s] BatchAddLocation returned unexpected result count for block_key %ld: expected 1, got %zu; "
                      "retaining allocated URIs",
                      trace_id.c_str(),
                      request.block_key,
                      add_results.size());
        return EC_ERROR;
    }
    const auto &add_result = add_results.front();
    if (ec != EC_OK || add_result.ec != EC_OK || add_result.location_id.empty()) {
        KVCM_LOG_WARN("[%s] BatchAddLocation failed for block_key %ld, aggregate ec %d, item ec %d",
                      trace_id.c_str(),
                      request.block_key,
                      ec,
                      add_result.ec);
        RollbackAddedLocations(ctx.get(),
                               trace_id,
                               request.instance_id,
                               indexer,
                               data_storage_manager_,
                               schedule_plan_executor_,
                               {{request.block_key,
                                 request.dst_storage_name,
                                 allocated_for_rollback,
                                 add_result.ec,
                                 add_result.location_id}});
        return EC_ERROR;
    }

    out_ctx.instance_group_name = request.instance_group_name;
    out_ctx.instance_id = request.instance_id;
    out_ctx.block_key = request.block_key;
    out_ctx.src_location_id = request.src_location_id;
    out_ctx.src_create_time = src_create_time; // 记录源 location 的创建时间
    out_ctx.src_storage_name = request.src_storage_name;
    out_ctx.dst_storage_name = request.dst_storage_name;
    out_ctx.dst_location_id = add_result.location_id;
    out_ctx.retention = request.retention;
    out_ctx.copy_execution_mode = request.copy_execution_mode;
    out_ctx.async_operation_id = request.async_operation_id;
    out_ctx.total_bytes = total_bytes;
    return EC_OK;
}

ErrorCode MigrationManager::Submit(const std::string &trace_id, MigrationRequest request) {
    auto reject_not_accepting = [&]() {
        KVCM_LOG_WARN("[%s] reject migration copy submit for instance %s block_key %ld: "
                      "manager is not accepting submissions",
                      trace_id.c_str(),
                      request.instance_id.c_str(),
                      request.block_key);
        return EC_ERROR;
    };
    // Stop 先把 accepting 置 false，再等待 lifecycle unique lock。快检避免 shutdown 期间的新请求
    // 排在 unique lock 后面；shared lock 则保证一旦进入，Stop 必须等本函数完整收口。
    if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
        return reject_not_accepting();
    }
    std::shared_lock<std::shared_mutex> lifecycle_lock(copy_submission_lifecycle_mutex_);
    if (request.instance_id.empty() || request.src_location_id.empty() || request.src_storage_name.empty() ||
        request.dst_storage_name.empty()) {
        return EC_BADARGS;
    }
    if (GetIndexer(request.instance_id) == nullptr) {
        return EC_INSTANCE_NOT_EXIST;
    }
    if (const auto target_ec = CheckTargetStorageAdmission(
            trace_id, request.instance_group_name, request.instance_id, request.dst_storage_name);
        target_ec != EC_OK) {
        return target_ec;
    }
    if (request.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        const auto results = BatchSubmit(trace_id, {std::move(request)}, CopyConcurrencyLimit(), true);
        return results.empty() ? EC_ERROR : results.front();
    }
    {
        std::lock_guard<std::mutex> submission_lock(copy_submission_mutex_);
        // shared lock 可能排在一次 Stop 后才拿到，必须在真正准入点复查。
        if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
            return reject_not_accepting();
        }
        // instance 正在 drain（RemoveInstance 中）时拒绝新提交，避免 trim-vs-write 竞态。
        if (draining_instances_.count(request.instance_id) > 0) {
            KVCM_LOG_WARN("[%s] reject migration copy submit for draining instance %s block_key %ld",
                          trace_id.c_str(),
                          request.instance_id.c_str(),
                          request.block_key);
            return EC_ERROR;
        }

        // 在释放短准入锁、执行任何目标 I/O 前原子登记 preparing 占位。它同时用于
        // 防重复、copy 并发预算、Reclaimer 保护和 drain 快照，不能拆成 check-then-insert。
        std::lock_guard<std::mutex> lock(task_mutex_);
        if (!ReservePreparingTaskLocked(request)) {
            KVCM_LOG_INFO("[%s] instance %s block_key %ld already has an active migration task, skip",
                          trace_id.c_str(),
                          request.instance_id.c_str(),
                          request.block_key);
            return EC_EXIST;
        }
    }
    auto release_preparing = [&]() {
        std::lock_guard<std::mutex> lock(task_mutex_);
        RemovePreparingTaskLocked(request.instance_id, request.block_key);
    };

    CopyTaskContext ctx;
    std::vector<DataStorageUri> src_uris;
    std::vector<DataStorageUri> dst_uris;
    ErrorCode prepare_ec = PrepareCopyTask(trace_id, request, ctx, src_uris, dst_uris);
    if (prepare_ec != EC_OK) {
        // PrepareCopyTask 已按逐 key AddLocation 结果回滚目标元数据和 URI；补偿状态无法确认时
        // 会保守保留 URI，由残留 WRITING 元数据的孤儿清理路径继续收敛。
        release_preparing();
        return prepare_ec;
    }

    // BatchAddLocation 可能在返回前已经对 Reclaimer 可见；此前空 id 的 preparing 按 block 临时保护。
    // 返回后立即绑定真实 id，缩短对同 block 其他 WRITING location 的宽泛保护时间。
    bool target_bound = false;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        target_bound = UpdatePreparingTaskLocked(ctx);
    }
    if (!target_bound) {
        // Cancel 可能已把 reservation 转成 kPrepareCancelling。先在 reservation 保护下提交精确的
        // 异步目标删除，再释放未提交任务；这里保证提交顺序，不等待实际删除完成。
        SubmitTargetLocationDelete(ctx);
        release_preparing();
        return EC_ERROR;
    }
    ctx.submit_time = std::chrono::steady_clock::now();
    // 只有指向本次 Copy 目标的 mark 才能绑定到 ctx，供 OnTaskSuccess 做 match-clear。
    // 其他目标的 mark 表示独立的迁移意图，不能由本次 Copy 消费。
    {
        std::vector<MarkQueryResult> mark_snap;
        const auto mark_ec = BatchGetTieredWriteTargets(request.instance_id, {request.block_key}, mark_snap);
        if (mark_ec == EC_OK && !mark_snap.empty() && mark_snap[0].HasValidMark() &&
            mark_snap[0].target == ctx.dst_storage_name) {
            ctx.mark_target = mark_snap[0].target;
            ctx.mark_deadline_ms = mark_snap[0].deadline_ms;
        }
    }

    // future 对 monitor 可见前，先把完整快照写回并进入可认领的 kRunning。
    // Cancel 若在 executor Submit 前命中 kRunning，采用 deferred-cancel 语义：copy 完成后丢弃目标。
    bool task_running = false;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        task_running = UpdatePreparingTaskLocked(ctx) && MarkTaskRunningLocked(ctx.instance_id, ctx.block_key);
    }
    if (!task_running) {
        SubmitTargetLocationDelete(ctx);
        release_preparing();
        return EC_ERROR;
    }

    // 提交 copy 任务。
    CacheLocationCopyRequest copy_req;
    copy_req.instance_id = ctx.instance_id;
    copy_req.block_key = ctx.block_key;
    copy_req.exec_storage_name = ctx.src_storage_name; // 一期 = 源 storage
    copy_req.src_uris = std::move(src_uris);
    copy_req.dst_uris = std::move(dst_uris);

    std::future<PlanExecuteResult> future = schedule_plan_executor_->Submit(copy_req);
    if (!future.valid()) {
        KVCM_LOG_WARN("[%s] submit copy task failed for block_key %ld", trace_id.c_str(), ctx.block_key);
        SubmitTargetLocationDelete(ctx);
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            RemoveActiveTaskLocked(ctx.instance_id, ctx.block_key);
        }
        return EC_ERROR;
    }

    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_copies_.push_back(PendingCopy{ctx.instance_id, ctx.block_key, 0, std::move(future)});
    }
    pending_cv_.notify_one();

    stat_copy_submitted_.fetch_add(1, std::memory_order_relaxed);
    if (metrics_enabled_) {
        ++m_tasks_submitted_total_;
        // 统一口径：copy 任务成功提交 executor（派发不可逆完成）即计入，
        // 与普通写路径（StartWriteCache 在 BatchAddLocation 成功后、location 交付
        // client 前）的统计点对成；之后的取消/copy 失败/源丢失均不影响该计数。
        data_storage_manager_->RecordWriteBytes(ctx.dst_storage_name, ctx.total_bytes);
    }
    if (event_manager_ != nullptr) {
        auto ev = std::make_shared<MigrationSubmittedEvent>(ctx.instance_id);
        ev->SetEventTriggerTime();
        ev->SetAdditionalArgs(ctx.block_key, ctx.src_storage_name, ctx.dst_storage_name, trace_id);
        event_manager_->Publish(ev);
    }
    KVCM_LOG_INFO("[%s] migration copy submitted: instance %s block_key %ld src_loc %s -> dst_storage %s (dst_loc %s)",
                  trace_id.c_str(),
                  ctx.instance_id.c_str(),
                  ctx.block_key,
                  ctx.src_location_id.c_str(),
                  ctx.dst_storage_name.c_str(),
                  ctx.dst_location_id.c_str());
    return EC_OK;
}

std::vector<ErrorCode> MigrationManager::BatchSubmit(const std::string &trace_id,
                                                     std::vector<MigrationRequest> requests,
                                                     CopyConcurrencyLimit copy_limit,
                                                     bool lifecycle_lock_held) {
    if (requests.empty()) {
        return {};
    }

    // Keep all long-lived per-request state together. The temporary vectors passed to batch APIs
    // remain below, but their compact result indexes are mapped back to these stable items immediately.
    struct BatchCopyItem {
        explicit BatchCopyItem(MigrationRequest input_request) : request(std::move(input_request)) {}

        void MarkFailed(ErrorCode ec) {
            result = ec;
            eligible = false;
        }

        MigrationRequest request;
        ErrorCode result = EC_ERROR;
        bool eligible = true;
        bool reservation_active = false;
        bool async_credit_active = false;
        std::vector<DataStorageUri> src_uris;
        std::vector<DataStorageUri> dst_uris;
        std::vector<LocationSpec> dst_specs;
        std::uint64_t total_bytes = 0;
        CopyTaskContext context;
        bool target_bound = false;
    };

    std::vector<BatchCopyItem> items;
    items.reserve(requests.size());
    for (auto &request : requests) {
        items.emplace_back(std::move(request));
    }
    auto collect_results = [&items]() {
        std::vector<ErrorCode> results;
        results.reserve(items.size());
        for (const auto &item : items) {
            results.push_back(item.result);
        }
        return results;
    };

    // BatchSubmit 是 DispatchMigrationBatch 的内部 prepared-request API，不再
    // 兼容空 src_specs 后回读 meta 的第二套提交流程。整批共享的 indexer/target 均取 first_req，
    // 因此任何 item 违反同 instance/target 或缺少源快照时，都必须在 reservation/I/O 前拒绝整批。
    const auto &first_req = items.front().request;
    const bool prepared_batch = !first_req.instance_id.empty() && !first_req.dst_storage_name.empty() &&
                                std::all_of(items.begin(), items.end(), [&first_req](const BatchCopyItem &item) {
                                    const auto &request = item.request;
                                    return request.instance_id == first_req.instance_id &&
                                           request.dst_storage_name == first_req.dst_storage_name &&
                                           !request.src_location_id.empty() && !request.src_storage_name.empty() &&
                                           !request.src_specs.empty();
                                });
    if (!prepared_batch) {
        KVCM_LOG_WARN(
            "[%s] reject invalid prepared migration batch: requests must share a non-empty instance/target and "
            "carry non-empty source location/specs",
            trace_id.c_str());
        for (auto &item : items) {
            item.MarkFailed(EC_BADARGS);
        }
        return collect_results();
    }

    const auto batch_mode = first_req.copy_execution_mode;
    if (batch_mode != MigrationCopyExecutionMode::SYNC && batch_mode != MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        for (auto &item : items) {
            item.MarkFailed(EC_BADARGS);
        }
        return collect_results();
    }
    for (auto &item : items) {
        if (item.request.copy_execution_mode != batch_mode) {
            item.MarkFailed(EC_BADARGS);
            continue;
        }
        const auto total_bytes = SumSourceUriSizes(item.request.src_specs);
        if (!total_bytes.has_value()) {
            KVCM_LOG_WARN("[%s] reject migration block %lld: source URI size is missing, zero, invalid, or overflowed",
                          trace_id.c_str(),
                          static_cast<long long>(item.request.block_key));
            item.MarkFailed(EC_BADARGS);
            continue;
        }
        item.total_bytes = *total_bytes;
        if (batch_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED && item.request.async_operation_id.empty()) {
            item.request.async_operation_id = StringUtil::GenerateRandomString(32);
        }
    }

    // Stop 的 lifecycle barrier 与 Submit 相同：快检拒绝 shutdown 后的新调用；shared lock 覆盖本函数
    // 余下生命周期，但不串行其他 Submit/BatchSubmit。
    if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
        return collect_results();
    }
    std::shared_lock<std::shared_mutex> lifecycle_lock;
    if (!lifecycle_lock_held) {
        lifecycle_lock = std::shared_lock<std::shared_mutex>(copy_submission_lifecycle_mutex_);
        if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
            return collect_results();
        }
    }
    if (const auto target_ec = CheckTargetStorageAdmission(
            trace_id, first_req.instance_group_name, first_req.instance_id, first_req.dst_storage_name);
        target_ec != EC_OK) {
        for (auto &item : items) {
            item.MarkFailed(target_ec);
        }
        return collect_results();
    }
    if (batch_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        std::unordered_set<std::string> checked_source_storages;
        for (const auto &item : items) {
            if (!item.eligible || !checked_source_storages.insert(item.request.src_storage_name).second) {
                continue;
            }
            if (!data_storage_manager_ || !data_storage_manager_->SupportsAsyncCopy(item.request.src_storage_name)) {
                KVCM_LOG_WARN(
                    "[%s] reject ASYNC_REQUIRED migration: source storage %s has no native async Copy capability",
                    trace_id.c_str(),
                    item.request.src_storage_name.c_str());
                for (auto &candidate : items) {
                    if (candidate.eligible) {
                        candidate.MarkFailed(EC_UNIMPLEMENTED);
                    }
                }
                return collect_results();
            }
        }
    }

    // ---- phase 0: group 级 Copy 硬限流 + per-block dedup + draining gate + preparing reservation ----
    // 所有 eligible item 必须在释放短准入锁、执行任何 batch Create/AddLocation 前进入
    // active 表。copy_submission_mutex_ 只串行本 phase 的 gate；后续 backend/meta I/O 可跨 instance 并行。
    // MarkFailed 只收拢 result/eligible。reservation 必须等 URI/Location rollback 完成后再释放，
    // 避免失败 item 尚在清理目标资源时，同 block 的新 Submit 已经进入并与旧清理相互覆盖。
    auto release_async_credit = [&](BatchCopyItem &item) {
        if (!item.async_credit_active) {
            return;
        }
        CopyTaskContext credit;
        credit.instance_group_name = item.request.instance_group_name;
        credit.async_operation_id = item.request.async_operation_id;
        credit.total_bytes = item.total_bytes;
        credit.async_credit_active = true;
        ReleaseAsyncCopyCredit(credit);
        item.async_credit_active = false;
    };
    auto release_preparing = [&](BatchCopyItem &item) {
        if (item.reservation_active) {
            std::lock_guard<std::mutex> lock(task_mutex_);
            RemovePreparingTaskLocked(item.request.instance_id, item.request.block_key);
            item.reservation_active = false;
        }
        release_async_credit(item);
    };
    auto release_all_preparing = [&]() {
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            for (auto &item : items) {
                if (item.reservation_active) {
                    RemovePreparingTaskLocked(item.request.instance_id, item.request.block_key);
                    item.reservation_active = false;
                }
            }
        }
        for (auto &item : items) {
            release_async_credit(item);
        }
    };
    {
        std::lock_guard<std::mutex> submission_lock(copy_submission_mutex_);
        if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
            return collect_results();
        }
        std::lock_guard<std::mutex> lock(task_mutex_);
        std::size_t available_group_slots = SIZE_MAX;
        if (copy_limit.enabled()) {
            const auto active = ActiveTaskCountForGroupUnsafe(copy_limit.instance_group_name);
            available_group_slots = copy_limit.max_concurrency > active ? copy_limit.max_concurrency - active : 0;
        }
        for (auto &item : items) {
            auto &request = item.request;
            // instance 正在 drain 时拒绝（覆盖 reclaimer + admin 两路，与 Submit 一致）。
            if (draining_instances_.count(request.instance_id) > 0) {
                item.MarkFailed(EC_ERROR);
                continue;
            }
            if (copy_limit.enabled() && request.instance_group_name != copy_limit.instance_group_name) {
                item.MarkFailed(EC_BADARGS);
                continue;
            }
            // 原子 reserve 同时挡住并发重复和同一输入 batch 内的重复 block；逐项失败不影响其他 item。
            if (!ReservePreparingTaskLocked(request)) {
                item.MarkFailed(EC_EXIST);
                continue;
            }
            // 在同一个 task_mutex_ 临界区内先 reserve 再检查剩余 slot：这样即使 slot 已满，
            // 同 block 重复项仍稳定返回 EC_EXIST；非重复项则立即回滚临时 reservation。
            if (copy_limit.enabled() && available_group_slots == 0) {
                RemovePreparingTaskLocked(request.instance_id, request.block_key);
                item.MarkFailed(EC_OUT_OF_LIMIT);
                continue;
            }
            item.reservation_active = true;
            if (copy_limit.enabled()) {
                --available_group_slots;
            }
            if (request.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
                if (!ReserveAsyncCopyCredit(request, item.total_bytes)) {
                    RemovePreparingTaskLocked(request.instance_id, request.block_key);
                    item.reservation_active = false;
                    item.MarkFailed(EC_OUT_OF_LIMIT);
                    continue;
                }
                item.async_credit_active = true;
            }
        }
    }

    // ---- phase 1: batch Create (按 size 分组) ----
    // 收集 eligible requests 的 src_specs → 构建 dst_key 列表 → 按 (dst_storage, size) 分组 batch Create。
    // 每个 item 的 src_uris/dst_uris 按原 request 的 spec 顺序保存。
    auto indexer = meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(first_req.instance_id) : nullptr;
    if (!indexer) {
        release_all_preparing();
        return collect_results();
    }
    auto dst_backend =
        data_storage_manager_ ? data_storage_manager_->GetDataStorageBackend(first_req.dst_storage_name) : nullptr;
    if (!dst_backend) {
        release_all_preparing();
        return collect_results();
    }
    const DataStorageType dst_type = dst_backend->GetType();

    // 按 (size) 分组：group_key=size → {(item_idx, spec_idx, dst_key, src_uri)}
    struct CreateEntry {
        std::size_t item_idx;
        std::size_t spec_idx;
        std::string dst_key;
        DataStorageUri src_uri;
    };
    std::unordered_map<std::uint64_t, std::vector<CreateEntry>> create_groups;

    for (std::size_t i = 0; i < items.size(); ++i) {
        auto &item = items[i];
        if (!item.eligible) {
            continue;
        }
        auto &req = item.request;
        for (std::size_t s = 0; s < req.src_specs.size(); ++s) {
            DataStorageUri src_uri(req.src_specs[s].uri());
            if (!src_uri.Valid()) {
                item.MarkFailed(EC_ERROR);
                break;
            }
            const auto spec_size_result = ParsePositiveUriSize(src_uri);
            if (!spec_size_result.has_value()) {
                item.MarkFailed(EC_BADARGS);
                break;
            }
            const std::uint64_t spec_size = *spec_size_result;
            std::string dst_key =
                req.instance_id + "/" + req.src_specs[s].name() + "/" + StringUtil::Uint64ToHex(req.block_key);
            create_groups[spec_size].push_back({i, s, std::move(dst_key), std::move(src_uri)});
        }
    }

    // 执行 batch Create + 回填 per-block URI
    auto batch_ctx = std::make_shared<RequestContext>(trace_id.empty() ? "migration_batch_prepare" : trace_id);
    // 异构 size 的 spec 分散在不同 create_group；某 group 使一个 block ineligible 后，
    // 之后处理的 group 里同 block 的 spec 可能已被 batch Create 成功分配 URI，却在下面
    // `!item.eligible` 分支被跳过、不进 item.dst_uris，导致后续 rollback 删不到 → 永久 orphan
    // （无 CacheLocation meta，reclaimer 也 GC 不到）。Create 返回 shape 异常时，整组返回的
    // 成功 URI 同样不能被采用。统一收集这些无 owner 的 URI，在 group 循环后删除。
    std::vector<DataStorageUri> unowned_created_uris;
    for (auto &[size, entries] : create_groups) {
        std::vector<std::string> keys;
        keys.reserve(entries.size());
        for (const auto &e : entries) {
            keys.push_back(e.dst_key);
        }
        auto create_results =
            data_storage_manager_->Create(batch_ctx.get(), first_req.dst_storage_name, keys, size, nullptr);
        if (create_results.size() != entries.size()) {
            KVCM_LOG_WARN(
                "[%s] batch Create returned unexpected result count on storage %s for size %llu: expected %zu, got "
                "%zu; rolling back the entire group",
                trace_id.c_str(),
                first_req.dst_storage_name.c_str(),
                static_cast<unsigned long long>(size),
                entries.size(),
                create_results.size());
            // 位置映射以 vector 下标为契约；shape 一旦异常，连前 min(N, M) 项也不能再可信地
            // 绑定到 request。删除本组实际返回的全部成功 URI，并将本组 request 整体置失败。
            for (const auto &[ec, uri] : create_results) {
                if (ec == EC_OK) {
                    unowned_created_uris.push_back(uri);
                }
            }
            for (const auto &entry : entries) {
                items[entry.item_idx].MarkFailed(EC_ERROR);
            }
            continue;
        }
        for (std::size_t j = 0; j < entries.size(); ++j) {
            const auto &e = entries[j];
            auto &item = items[e.item_idx];
            if (!item.eligible) {
                // 该 block 已被其他 spec 的失败标记 ineligible；本 spec 若已成功分配，
                // 需补删（否则永不进 dst_uris，rollback 漏删）。
                if (create_results[j].first == EC_OK) {
                    unowned_created_uris.push_back(create_results[j].second);
                }
                continue;
            }
            if (create_results[j].first == EC_OK) {
                item.src_uris.push_back(e.src_uri);
                item.dst_uris.push_back(create_results[j].second);
                item.dst_specs.emplace_back(item.request.src_specs[e.spec_idx].name(),
                                            create_results[j].second.ToUriString());
            } else {
                item.MarkFailed(EC_ERROR);
            }
        }
    }
    if (!unowned_created_uris.empty()) {
        data_storage_manager_->Delete(batch_ctx.get(), first_req.dst_storage_name, unowned_created_uris, nullptr);
    }

    // 回滚 Create 部分失败的 block 的已分配 URI
    for (auto &item : items) {
        if (item.eligible) {
            continue;
        }
        if (!item.dst_uris.empty()) {
            data_storage_manager_->Delete(batch_ctx.get(), item.request.dst_storage_name, item.dst_uris, nullptr);
            item.dst_uris.clear();
            item.src_uris.clear();
            item.dst_specs.clear();
        }
        release_preparing(item);
    }

    // 收集成功 Create 的 block → batch AddLocation
    // add_items[k] 显式对应 add_block_keys[k]/add_locations[k]，并用于接回 add_results[k]。
    std::vector<BatchCopyItem *> add_items;
    std::vector<int64_t> add_block_keys;
    CacheLocationVector add_locations;
    for (auto &item : items) {
        if (!item.eligible) {
            continue;
        }
        if (item.dst_specs.size() != item.request.src_specs.size()) {
            // spec 数不对（Create 返回的结果数和请求数不匹配）
            if (!item.dst_uris.empty()) {
                data_storage_manager_->Delete(batch_ctx.get(), item.request.dst_storage_name, item.dst_uris, nullptr);
            }
            item.MarkFailed(EC_ERROR);
            release_preparing(item);
            continue;
        }
        auto dst_loc = std::make_shared<CacheLocation>();
        dst_loc->set_type(dst_type);
        dst_loc->set_spec_size(item.dst_specs.size());
        for (auto &spec : item.dst_specs) {
            dst_loc->push_location_spec(std::move(spec));
        }
        add_items.push_back(&item);
        add_block_keys.push_back(item.request.block_key);
        add_locations.push_back(std::move(dst_loc));
    }

    // ---- phase 2+3: batch AddLocation + bind preparing reservations + executor Submit ----
    if (!add_block_keys.empty()) {
        MetaSearcher meta_searcher(indexer);
        std::vector<MetaSearcher::AddLocationResult> add_results;
        ErrorCode add_ec = meta_searcher.BatchAddLocation(batch_ctx.get(), add_block_keys, add_locations, add_results);
        if (add_results.size() != add_items.size()) {
            KVCM_LOG_WARN("[%s] BatchAddLocation returned unexpected result count for migration batch: expected %zu, "
                          "got %zu, aggregate ec %d; retaining allocated URIs",
                          trace_id.c_str(),
                          add_items.size(),
                          add_results.size(),
                          add_ec);
            for (auto *item : add_items) {
                item->MarkFailed(EC_MISMATCH);
            }
            release_all_preparing();
            return collect_results();
        }
        if (add_ec != EC_OK) {
            KVCM_LOG_WARN("[%s] BatchAddLocation partially failed for migration batch, aggregate ec %d",
                          trace_id.c_str(),
                          add_ec);
        }

        // ---- phase 3a: 先为整个批次绑定 location_id，再做任何逐项 mark 查询/submit ----
        // BatchAddLocation 一次公开全部 WRITING；若仍在逐项循环里绑定，后半批会继续暴露宽泛窗口。
        std::vector<BatchCopyItem *> bind_items;
        bind_items.reserve(add_items.size());
        std::vector<AddLocationRollbackItem> rollback_items;
        std::vector<BatchCopyItem *> rollback_add_items;
        for (std::size_t k = 0; k < add_items.size(); ++k) {
            auto &item = *add_items[k];
            const auto &add_result = add_results[k];
            if (!item.eligible || add_result.ec != EC_OK || add_result.location_id.empty()) {
                const ErrorCode item_ec = add_result.ec == EC_OK ? EC_MISMATCH : add_result.ec;
                rollback_items.push_back({item.request.block_key,
                                          item.request.dst_storage_name,
                                          item.dst_uris,
                                          item_ec,
                                          add_result.location_id});
                item.dst_uris.clear();
                item.MarkFailed(item_ec);
                rollback_add_items.push_back(&item);
                continue;
            }
            auto &req = item.request;
            CopyTaskContext &ctx = item.context;
            ctx.instance_group_name = req.instance_group_name;
            ctx.instance_id = req.instance_id;
            ctx.block_key = req.block_key;
            ctx.src_location_id = req.src_location_id;
            ctx.src_create_time = req.src_create_time;
            ctx.src_storage_name = req.src_storage_name;
            ctx.dst_storage_name = req.dst_storage_name;
            ctx.dst_location_id = add_result.location_id;
            ctx.retention = req.retention;
            ctx.copy_execution_mode = req.copy_execution_mode;
            ctx.async_operation_id = req.async_operation_id;
            ctx.async_credit_active = item.async_credit_active;
            if (ctx.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
                ctx.async_transition_mutex = std::make_shared<std::mutex>();
            }
            ctx.total_bytes = item.total_bytes;
            bind_items.push_back(&item);
        }
        if (!rollback_items.empty()) {
            RollbackAddedLocations(batch_ctx.get(),
                                   trace_id,
                                   first_req.instance_id,
                                   indexer,
                                   data_storage_manager_,
                                   schedule_plan_executor_,
                                   rollback_items);
            for (auto *item : rollback_add_items) {
                release_preparing(*item);
            }
        }
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            for (auto *item : bind_items) {
                // UpdatePreparingTaskLocked copies from const&, so item.context stays available for executor
                // submission, target cleanup and event fields after the active reservation is updated.
                item->target_bound = UpdatePreparingTaskLocked(item->context);
            }
        }
        // 与单条 Submit 保持相同边界：phase-3 失败只保证先提交基于精确 location id 的异步删除，
        // 再释放 reservation；逐项 ownership 收拢不把该既有异步清理改成同步等待。
        for (auto *item : bind_items) {
            if (item->target_bound) {
                continue;
            }
            SubmitTargetLocationDelete(item->context);
            item->MarkFailed(EC_ERROR);
            release_preparing(*item);
        }

        // ---- phase 3b: 逐项补齐提交快照、转 kRunning、提交 executor ----
        for (auto *item : bind_items) {
            if (!item->eligible || !item->target_bound) {
                continue;
            }
            auto &req = item->request;
            CopyTaskContext &ctx = item->context;
            ctx.submit_time = std::chrono::steady_clock::now();
            // 仅绑定与本次 Copy 目标一致的 mark；其他目标的 mark 必须保留。
            {
                std::vector<MarkQueryResult> mark_snap;
                const auto mark_ec = BatchGetTieredWriteTargets(req.instance_id, {req.block_key}, mark_snap);
                if (mark_ec == EC_OK && !mark_snap.empty() && mark_snap[0].HasValidMark() &&
                    mark_snap[0].target == ctx.dst_storage_name) {
                    ctx.mark_target = mark_snap[0].target;
                    ctx.mark_deadline_ms = mark_snap[0].deadline_ms;
                }
            }

            // The persistent guard is installed only after AddLocation and
            // before any remote POST.  This keeps ordinary AddLocation
            // rollback guard-free while preserving the no-unfenced-submit
            // invariant.
            if (ctx.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
                ctx.async_guard_create_time_us = TimestampUtil::GetCurrentTimeUs();
                if (UpdateAsyncCopyGuard(
                        ctx, MigrationCopyGuardState::MCGS_NONE, MigrationCopyGuardState::MCGS_SUBMITTING, {}, "") !=
                    GuardMutationResult::kAppliedDurably) {
                    item->reservation_active = false;
                    item->async_credit_active = false;
                    // No executor/backend handoff and therefore no remote POST
                    // has happened yet.  A failed persistence fence is a
                    // definite pre-submit rollback, not an unknown remote
                    // operation.  Treating it as UNKNOWN would consume
                    // quarantine forever for a task that never existed.
                    CompleteCopyTaskAsFailed(ctx, "guard_install_failed_before_submit", false);
                    item->MarkFailed(EC_ERROR);
                    continue;
                }
                // The source snapshot predates destination allocation and
                // guard persistence. Re-read both the hot view and persistent
                // meta after the pin exists; once this succeeds, all later
                // deletion admission observes the durable source pin.
                if (!IsSourceLocationServing(ctx)) {
                    item->reservation_active = false;
                    item->async_credit_active = false;
                    CompleteCopyTaskAsFailed(ctx, "source_changed_after_guard_install", false);
                    item->MarkFailed(EC_MISMATCH);
                    continue;
                }
            }

            bool task_running = false;
            {
                std::lock_guard<std::mutex> lock(task_mutex_);
                task_running = UpdatePreparingTaskLocked(ctx) && MarkTaskRunningLocked(ctx.instance_id, ctx.block_key);
                if (task_running) {
                    item->reservation_active = false; // 已成为正式 active task，不能再由 preparing guard 清理
                    item->async_credit_active = false; // credit ownership moved into the active task context
                }
            }
            if (!task_running) {
                if (ctx.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
                    item->reservation_active = false;
                    item->async_credit_active = false;
                    CompleteCopyTaskAsFailed(ctx, "task_activation_failed_before_submit", false);
                } else {
                    SubmitTargetLocationDelete(ctx);
                    item->MarkFailed(EC_ERROR);
                    release_preparing(*item);
                }
                item->MarkFailed(EC_ERROR);
                continue;
            }

            CacheLocationCopyRequest copy_req;
            copy_req.instance_id = ctx.instance_id;
            copy_req.block_key = ctx.block_key;
            copy_req.exec_storage_name = ctx.src_storage_name;
            copy_req.src_uris = std::move(item->src_uris);
            copy_req.dst_uris = std::move(item->dst_uris);

            std::future<PlanExecuteResult> future;
            std::future<AsyncCopyRemoteSubmitResult> remote_submit_future;
            bool native_async = false;
            if (ctx.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
                auto async_submit =
                    schedule_plan_executor_->SubmitAsyncCopy(copy_req, ctx.async_operation_id, req.async_copy_options);
                future = std::move(async_submit.future);
                if (!async_submit.submit_result.accepted) {
                    if (async_submit.submit_result.acceptance_unknown) {
                        OnTaskUnknown(ctx.instance_id,
                                      ctx.block_key,
                                      async_submit.submit_result.detail.empty() ? "copy_submit_acceptance_unknown"
                                                                                : async_submit.submit_result.detail);
                    } else {
                        OnTaskFailedInternal(ctx.instance_id,
                                             ctx.block_key,
                                             async_submit.submit_result.status,
                                             async_submit.submit_result.detail.empty()
                                                 ? "copy_submit_rejected"
                                                 : async_submit.submit_result.detail,
                                             false,
                                             false);
                    }
                    item->MarkFailed(async_submit.submit_result.status);
                    continue;
                }
                if (!async_submit.remote_submit_future.valid()) {
                    OnTaskUnknown(ctx.instance_id, ctx.block_key, "copy_submit_missing_remote_acceptance_channel");
                    item->MarkFailed(EC_ERROR);
                    continue;
                }
                remote_submit_future = std::move(async_submit.remote_submit_future);
                native_async = true;
            } else {
                future = schedule_plan_executor_->Submit(copy_req);
            }
            if (!future.valid()) {
                if (native_async) {
                    // accepted=true means the remote operation may already be
                    // running.  Losing the completion channel is not a safe
                    // failure and must never authorize target reuse.
                    OnTaskUnknown(ctx.instance_id, ctx.block_key, "copy_submit_invalid_future_after_acceptance");
                } else {
                    CompleteCopyTaskAsFailed(ctx, "copy_submit_invalid_future");
                }
                item->MarkFailed(EC_ERROR);
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                PendingCopy pending{
                    ctx.instance_id, ctx.block_key, copy_req.src_uris.size(), std::move(future), native_async};
                if (native_async) {
                    pending.remote_submit_future = std::move(remote_submit_future);
                    pending.remote_submit_processed = false;
                }
                pending_copies_.push_back(std::move(pending));
            }
            pending_cv_.notify_one();

            stat_copy_submitted_.fetch_add(1, std::memory_order_relaxed);
            if (metrics_enabled_) {
                ++m_tasks_submitted_total_;
                // 与单条 Submit 同口径：任务成功提交 executor 即计入，后续结果不影响。
                data_storage_manager_->RecordWriteBytes(ctx.dst_storage_name, ctx.total_bytes);
            }
            if (event_manager_ != nullptr) {
                auto ev = std::make_shared<MigrationSubmittedEvent>(ctx.instance_id);
                ev->SetEventTriggerTime();
                ev->SetAdditionalArgs(ctx.block_key, ctx.src_storage_name, ctx.dst_storage_name, trace_id);
                event_manager_->Publish(ev);
            }
            item->result = EC_OK;
        }
    } // if (!add_block_keys.empty())

    // 防御性收口：正常路径此时只剩已转 kRunning 的 task（reservation_active=false）。任何遗漏的
    // preparing item 都不应泄漏到函数返回之后。
    release_all_preparing();
    return collect_results();
}

bool MigrationManager::IsSourceLocationServing(const CopyTaskContext &ctx) const {
    auto indexer = meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(ctx.instance_id) : nullptr;
    if (!indexer) {
        return false;
    }

    const auto matches_snapshot = [&ctx](const std::vector<CacheLocationMap> &location_maps) {
        if (location_maps.size() != 1) {
            return false;
        }
        const auto iter = location_maps[0].find(ctx.src_location_id);
        // id + status + create_time 三者同时匹配，防止 id 复用导致误判新 location 为原始源。
        return iter != location_maps[0].end() && iter->second != nullptr && iter->second->status() == CLS_SERVING &&
               iter->second->create_time() == ctx.src_create_time;
    };

    auto rc = std::make_shared<RequestContext>("migration_check_source");
    MetaSearcher meta_searcher(indexer);
    std::vector<CacheLocationMap> hot_locations;
    BlockMask empty_mask;
    if (meta_searcher.BatchGetLocation(rc.get(), {ctx.block_key}, empty_mask, hot_locations) != EC_OK ||
        !matches_snapshot(hot_locations)) {
        return false;
    }
    if (ctx.copy_execution_mode != MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        return true;
    }

    // A delete CAS may already be visible in the hot layer but not yet durable,
    // or persistent meta may have advanced while a stale cache entry remains.
    // Requiring both views closes both halves of the pre-guard deletion race.
    std::vector<CacheLocationMap> persistent_locations;
    const auto persistent_result = indexer->GetLocationsFromPersistent(rc.get(), {ctx.block_key}, persistent_locations);
    return persistent_result.error_codes.size() == 1 && persistent_result.error_codes[0] == EC_OK &&
           matches_snapshot(persistent_locations);
}

MigrationManager::CopySourceFailureKey MigrationManager::MakeCopySourceFailureKey(const CopyTaskContext &ctx) {
    return CopySourceFailureKey{
        ctx.instance_id, ctx.block_key, ctx.src_location_id, ctx.src_create_time, ctx.dst_storage_name};
}

MigrationManager::CopySourceFailureKey
MigrationManager::MakeCopySourceFailureKey(const std::string &instance_id,
                                           int64_t block_key,
                                           const CacheLocation &source,
                                           const std::string &target_storage_name) {
    return CopySourceFailureKey{instance_id, block_key, source.id(), source.create_time(), target_storage_name};
}

std::chrono::milliseconds MigrationManager::ComputeCopySourceBackoff(const CopySourceFailureKey &key,
                                                                     uint32_t consecutive_failures) {
    int64_t delay_ms = kCopySourceInitialBackoff.count();
    uint32_t remaining_doublings = consecutive_failures > 0 ? consecutive_failures - 1 : 0;
    while (remaining_doublings-- > 0 && delay_ms < kCopySourceMaxBackoff.count()) {
        delay_ms = std::min(kCopySourceMaxBackoff.count(), delay_ms * 2);
    }

    // 0~25% 的稳定 jitter，避免大量 stale location 在同一个 retry deadline 上形成惊群。
    // 使用 identity + failure count 的稳定 hash，便于故障复现，同时不引入共享随机数锁。
    const int64_t jitter_window_ms = delay_ms / 4;
    if (jitter_window_ms > 0 && delay_ms < kCopySourceMaxBackoff.count()) {
        std::size_t jitter_seed = CopySourceFailureKeyHash{}(key);
        HashCombine(jitter_seed, consecutive_failures);
        const int64_t jitter_ms = static_cast<int64_t>(jitter_seed % static_cast<std::size_t>(jitter_window_ms + 1));
        delay_ms = std::min(kCopySourceMaxBackoff.count(), delay_ms + jitter_ms);
    }
    return std::chrono::milliseconds(delay_ms);
}

void MigrationManager::PruneCopySourceFailuresLocked(std::chrono::steady_clock::time_point now) {
    while (!copy_source_failure_history_.empty()) {
        const auto &oldest = copy_source_failure_history_.front();
        const bool expired = now - oldest.failure_time >= kCopySourceFailureRetention;
        const bool over_capacity = copy_source_failures_.size() > kMaxCopySourceFailureEntries;
        if (!expired && !over_capacity) {
            break;
        }

        auto iter = copy_source_failures_.find(oldest.key);
        if (iter != copy_source_failures_.end() && iter->second.last_failure_time == oldest.failure_time) {
            copy_source_failures_.erase(iter);
        }
        copy_source_failure_history_.pop_front();
    }

    // 同一 source 的每次失败都会留下一个过期队列节点。正常 backoff 下增长很慢，但测试直调或
    // 异常回调风暴仍可能制造大量 stale 节点；超过硬上限时按 map 当前快照重建一次队列。
    if (copy_source_failure_history_.size() > kMaxCopySourceFailureHistoryEntries) {
        std::vector<CopySourceFailureHistoryEntry> compacted;
        compacted.reserve(copy_source_failures_.size());
        for (const auto &[key, state] : copy_source_failures_) {
            compacted.push_back(CopySourceFailureHistoryEntry{key, state.last_failure_time});
        }
        std::sort(compacted.begin(), compacted.end(), [](const auto &lhs, const auto &rhs) {
            return lhs.failure_time < rhs.failure_time;
        });
        copy_source_failure_history_ = std::deque<CopySourceFailureHistoryEntry>(compacted.begin(), compacted.end());
    }
}

void MigrationManager::RecordCopySourceFailure(const CopyTaskContext &ctx, ErrorCode reason) {
    const auto key = MakeCopySourceFailureKey(ctx);
    const auto now = std::chrono::steady_clock::now();
    uint32_t failure_count = 0;
    std::chrono::milliseconds backoff{0};
    {
        std::lock_guard<std::mutex> lock(copy_source_failure_mutex_);
        PruneCopySourceFailuresLocked(now);

        auto &state = copy_source_failures_[key];
        if (state.consecutive_failures < std::numeric_limits<uint32_t>::max()) {
            ++state.consecutive_failures;
        }
        failure_count = state.consecutive_failures;
        backoff = ComputeCopySourceBackoff(key, failure_count);
        state.last_error = reason;
        state.last_failure_time = now;
        state.retry_after = now + backoff;
        copy_source_failure_history_.push_back(CopySourceFailureHistoryEntry{key, now});

        PruneCopySourceFailuresLocked(now);
        UpdateCopySourceFailureGaugeLocked();
    }

    stat_source_failures_recorded_.fetch_add(1, std::memory_order_relaxed);
    if (metrics_enabled_) {
        ++m_source_failures_recorded_total_;
    }
    KVCM_LOG_WARN("migration Copy source enters retry backoff: instance %s block_key %lld source_location %s "
                  "source_create_time %lld target_storage %s reason %d consecutive_failures %u backoff_ms %lld",
                  ctx.instance_id.c_str(),
                  static_cast<long long>(ctx.block_key),
                  ctx.src_location_id.c_str(),
                  static_cast<long long>(ctx.src_create_time),
                  ctx.dst_storage_name.c_str(),
                  static_cast<int>(reason),
                  failure_count,
                  static_cast<long long>(backoff.count()));
}

void MigrationManager::ClearCopySourceFailure(const CopyTaskContext &ctx) {
    const auto key = MakeCopySourceFailureKey(ctx);
    std::lock_guard<std::mutex> lock(copy_source_failure_mutex_);
    if (copy_source_failures_.erase(key) > 0) {
        UpdateCopySourceFailureGaugeLocked();
    }
}

MigrationManager::CopySourceFailureSnapshot
MigrationManager::GetCopySourceFailure(const std::string &instance_id,
                                       int64_t block_key,
                                       const CacheLocation &source,
                                       const std::string &target_storage_name,
                                       std::chrono::steady_clock::time_point now) const {
    const auto key = MakeCopySourceFailureKey(instance_id, block_key, source, target_storage_name);
    std::lock_guard<std::mutex> lock(copy_source_failure_mutex_);
    auto iter = copy_source_failures_.find(key);
    if (iter == copy_source_failures_.end() || now - iter->second.last_failure_time >= kCopySourceFailureRetention) {
        return {};
    }
    return CopySourceFailureSnapshot{
        true, now < iter->second.retry_after, iter->second.consecutive_failures, iter->second.last_error};
}

MigrationCopyGuardState MigrationManager::ExpectedGuardState(const CopyTaskContext &ctx) {
    if (ctx.state == CopyTaskState::kCancelling || ctx.completion_was_cancelling) {
        return MigrationCopyGuardState::MCGS_CANCELLING;
    }
    return ctx.async_backend_task_ids.empty() ? MigrationCopyGuardState::MCGS_SUBMITTING
                                              : MigrationCopyGuardState::MCGS_ACTIVE;
}

std::shared_ptr<std::mutex> MigrationManager::GetAsyncTransitionMutex(const std::string &instance_id,
                                                                      int64_t block_key) const {
    std::lock_guard<std::mutex> lock(task_mutex_);
    const auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return nullptr;
    }
    const auto task_iter = instance_iter->second.find(block_key);
    return task_iter == instance_iter->second.end() ? nullptr : task_iter->second.async_transition_mutex;
}

void MigrationManager::MarkGuardFinalizationPendingSync(const CopyTaskContext &ctx,
                                                        CopyCompletionAction action,
                                                        const std::string &reason) {
    std::lock_guard<std::mutex> lock(task_mutex_);
    const auto instance_iter = active_tasks_by_instance_.find(ctx.instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return;
    }
    const auto task_iter = instance_iter->second.find(ctx.block_key);
    if (task_iter == instance_iter->second.end() || task_iter->second.async_operation_id != ctx.async_operation_id ||
        task_iter->second.state != CopyTaskState::kCompleting) {
        return;
    }
    task_iter->second.async_guard_finalization_pending_sync = true;
    task_iter->second.completion_was_cancelling = ctx.completion_was_cancelling;
    task_iter->second.async_guard_pending_action = action;
    task_iter->second.async_guard_pending_reason = reason;
}

void MigrationManager::MarkUnknownGuardPersistencePending(const CopyTaskContext &ctx,
                                                          const std::string &reason,
                                                          bool sync_only) {
    std::lock_guard<std::mutex> lock(task_mutex_);
    const auto instance_iter = active_tasks_by_instance_.find(ctx.instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return;
    }
    const auto task_iter = instance_iter->second.find(ctx.block_key);
    if (task_iter == instance_iter->second.end() || task_iter->second.async_operation_id != ctx.async_operation_id) {
        return;
    }
    // Retain the active owner and its inflight credit until persistent meta
    // itself authorizes publishing the runtime quarantine record.
    task_iter->second.state = CopyTaskState::kCompleting;
    task_iter->second.completion_was_cancelling =
        ctx.completion_was_cancelling || ctx.state == CopyTaskState::kCancelling;
    task_iter->second.async_guard_finalization_pending_sync = sync_only;
    task_iter->second.async_guard_pending_action = CopyCompletionAction::kUnknown;
    task_iter->second.async_guard_pending_reason = reason;
}

MigrationManager::GuardMutationResult
MigrationManager::UpdateAsyncCopyGuard(const CopyTaskContext &ctx,
                                       MigrationCopyGuardState expected_state,
                                       MigrationCopyGuardState state,
                                       const std::vector<std::string> &backend_task_ids,
                                       const std::string &last_error) {
    if (ctx.copy_execution_mode != MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        return GuardMutationResult::kAppliedDurably;
    }
    auto indexer = meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(ctx.instance_id) : nullptr;
    if (!indexer || ctx.async_operation_id.empty() || ctx.dst_location_id.empty()) {
        return GuardMutationResult::kNotApplied;
    }

    MigrationCopyGuard guard;
    const int64_t now_us = TimestampUtil::GetCurrentTimeUs();
    guard.set_schema_version(MigrationCopyGuard::kCurrentSchemaVersion);
    guard.set_state(state);
    guard.set_operation_id(ctx.async_operation_id);
    guard.set_source_location_id(ctx.src_location_id);
    guard.set_source_location_create_time(ctx.src_create_time);
    guard.set_source_storage_name(ctx.src_storage_name);
    guard.set_target_storage_name(ctx.dst_storage_name);
    guard.set_migration_retention(static_cast<int32_t>(ctx.retention));
    guard.set_mark_target(ctx.mark_target);
    guard.set_mark_deadline_ms(ctx.mark_deadline_ms);
    guard.set_total_bytes(ctx.total_bytes);
    guard.set_backend_task_ids(backend_task_ids);
    guard.set_create_time_us(ctx.async_guard_create_time_us > 0 ? ctx.async_guard_create_time_us : now_us);
    guard.set_update_time_us(now_us);
    guard.set_last_error(last_error);

    MetaSearcher meta_searcher(indexer);
    auto request_context = std::make_shared<RequestContext>("migration_update_async_guard");
    MetaSearcher::LocationCASTask task;
    task.location_id = ctx.dst_location_id;
    task.old_status = CLS_WRITING;
    task.new_status = CLS_WRITING;
    if (expected_state == MigrationCopyGuardState::MCGS_NONE) {
        task.expected_migration_copy_guard_absent = true;
    } else {
        task.expected_operation_id = ctx.async_operation_id;
        task.expected_migration_copy_guard_state = expected_state;
    }
    task.new_migration_copy_guard = std::make_shared<MigrationCopyGuard>(std::move(guard));
    std::vector<std::vector<ErrorCode>> results;
    const auto ec =
        meta_searcher.BatchCASLocationStatus(request_context.get(), {ctx.block_key}, {{std::move(task)}}, results);
    const bool updated = ec == EC_OK && results.size() == 1 && results[0].size() == 1 && results[0][0] == EC_OK;
    if (!updated) {
        return GuardMutationResult::kNotApplied;
    }
    // ReadModifyWriteLocation may first update the local/async layer.  A guard
    // is not a crash fence until the exact block has reached persistent meta.
    return indexer->Sync({ctx.block_key}) ? GuardMutationResult::kAppliedDurably
                                          : GuardMutationResult::kAppliedDurabilityUnknown;
}

MigrationManager::GuardMutationResult MigrationManager::FinalizeAsyncCopyGuard(const CopyTaskContext &ctx,
                                                                               MigrationCopyGuardState expected_state,
                                                                               CacheLocationStatus new_status) {
    if (ctx.copy_execution_mode != MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        return GuardMutationResult::kAppliedDurably;
    }
    auto indexer = meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(ctx.instance_id) : nullptr;
    if (!indexer || ctx.async_operation_id.empty() || ctx.dst_location_id.empty()) {
        return GuardMutationResult::kNotApplied;
    }
    if (ctx.async_guard_finalization_pending_sync) {
        return indexer->Sync({ctx.block_key}) ? GuardMutationResult::kAppliedDurably
                                              : GuardMutationResult::kAppliedDurabilityUnknown;
    }
    MetaSearcher meta_searcher(indexer);
    auto request_context = std::make_shared<RequestContext>("migration_finalize_async_guard");
    MetaSearcher::LocationCASTask task;
    task.location_id = ctx.dst_location_id;
    task.old_status = CLS_WRITING;
    task.new_status = new_status;
    task.expected_operation_id = ctx.async_operation_id;
    task.expected_migration_copy_guard_state = expected_state;
    task.clear_migration_copy_guard = true;
    std::vector<std::vector<ErrorCode>> results;
    const auto ec =
        meta_searcher.BatchCASLocationStatus(request_context.get(), {ctx.block_key}, {{std::move(task)}}, results);
    const bool updated = ec == EC_OK && results.size() == 1 && results[0].size() == 1 && results[0][0] == EC_OK;
    if (!updated) {
        return GuardMutationResult::kNotApplied;
    }
    return indexer->Sync({ctx.block_key}) ? GuardMutationResult::kAppliedDurably
                                          : GuardMutationResult::kAppliedDurabilityUnknown;
}

bool MigrationManager::PersistRemoteAsyncCopyAcceptance(const std::string &instance_id,
    int64_t block_key,
    size_t expected_items,
    const AsyncCopyRemoteSubmitResult &remote_result,
    CopyTaskContext &out_ctx,
    bool &out_cancel_requested) {
    out_ctx = CopyTaskContext{};
    out_cancel_requested = false;
    if (!remote_result.accepted || remote_result.operation_id.empty()) {
        return false;
    }

    // Cancel() and completion take this operation's mutex too.  Unrelated Copy
    // operations remain parallel, while a late submit response can never
    // overwrite MCGS_CANCELLING with MCGS_ACTIVE.
    auto transition_mutex = GetAsyncTransitionMutex(instance_id, block_key);
    if (!transition_mutex) {
        return false;
    }
    std::lock_guard<std::mutex> transition_lock(*transition_mutex);
    {
        std::lock_guard<std::mutex> task_lock(task_mutex_);
        const auto instance_iter = active_tasks_by_instance_.find(instance_id);
        if (instance_iter == active_tasks_by_instance_.end()) {
            return false;
        }
        const auto task_iter = instance_iter->second.find(block_key);
        if (task_iter == instance_iter->second.end() ||
            task_iter->second.copy_execution_mode != MigrationCopyExecutionMode::ASYNC_REQUIRED ||
            task_iter->second.async_operation_id != remote_result.operation_id ||
            (task_iter->second.state != CopyTaskState::kRunning &&
             task_iter->second.state != CopyTaskState::kCancelling)) {
            return false;
        }
        out_ctx = task_iter->second;
        out_ctx.async_backend_task_ids = remote_result.backend_task_ids;
        out_cancel_requested = task_iter->second.state == CopyTaskState::kCancelling;
    }

    // Populate out_ctx before rejecting a malformed accepted response so the
    // caller can still request cancellation through the correct backend.  Do
    // not persist ACTIVE unless every copied item has exactly one recovery
    // handle; a partial set cannot be resumed safely after leader change.
    if (expected_items == 0 || remote_result.backend_task_ids.size() != expected_items) {
        return false;
    }

    const auto expected_state =
        out_cancel_requested ? MigrationCopyGuardState::MCGS_CANCELLING : MigrationCopyGuardState::MCGS_SUBMITTING;
    const auto guard_state =
        out_cancel_requested ? MigrationCopyGuardState::MCGS_CANCELLING : MigrationCopyGuardState::MCGS_ACTIVE;
    const auto guard_result = UpdateAsyncCopyGuard(out_ctx,
                                                   expected_state,
                                                   guard_state,
                                                   out_ctx.async_backend_task_ids,
                                                   out_cancel_requested ? "cancel_requested" : "");

    if (guard_result != GuardMutationResult::kAppliedDurably) {
        return false;
    }

    {
        std::lock_guard<std::mutex> task_lock(task_mutex_);
        const auto instance_iter = active_tasks_by_instance_.find(instance_id);
        if (instance_iter == active_tasks_by_instance_.end()) {
            return false;
        }
        const auto task_iter = instance_iter->second.find(block_key);
        if (task_iter == instance_iter->second.end() ||
            task_iter->second.async_operation_id != remote_result.operation_id ||
            (task_iter->second.state != CopyTaskState::kRunning &&
             task_iter->second.state != CopyTaskState::kCancelling)) {
            return false;
        }
        task_iter->second.async_backend_task_ids = remote_result.backend_task_ids;
        out_ctx = task_iter->second;
    }
    return true;
}

void MigrationManager::SubmitPreparedTargetLocationDelete(const CopyTaskContext &ctx) {
    if (!schedule_plan_executor_ || ctx.dst_location_id.empty()) {
        return;
    }
    CacheLocationDelRequest del_req;
    del_req.instance_id = ctx.instance_id;
    del_req.block_keys = {ctx.block_key};
    del_req.location_ids = {{ctx.dst_location_id}};
    del_req.prepared_deleting = true;
    TrackLocationCleanup(ctx.instance_id,
                         ctx.dst_storage_name,
                         del_req,
                         schedule_plan_executor_->SubmitLocationDelete(del_req,
                                                                       ScheduleTaskClass::kMigrationContinuation));
}

void MigrationManager::TrackLocationCleanup(const std::string &instance_id,
                                            const std::string &storage_name,
                                            CacheLocationDelRequest request,
                                            std::future<PlanExecuteResult> future) {
    if (storage_name.empty() || !future.valid()) {
        KVCM_LOG_ERROR("cannot track asynchronous Copy location cleanup: instance %s storage %s valid_future %d",
                       instance_id.c_str(),
                       storage_name.c_str(),
                       static_cast<int>(future.valid()));
        return;
    }
    {
        std::lock_guard<std::mutex> lock(pending_cleanup_mutex_);
        pending_location_cleanups_.push_back(
            PendingLocationCleanup{instance_id, storage_name, std::move(request), std::move(future), {}, 0});
    }
    pending_cv_.notify_one();
}

void MigrationManager::ProcessCompletedLocationCleanups() {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(pending_cleanup_mutex_);
    for (auto iter = pending_location_cleanups_.begin(); iter != pending_location_cleanups_.end();) {
        if (!iter->future.valid()) {
            if (now < iter->retry_after) {
                ++iter;
                continue;
            }
            if (!schedule_plan_executor_) {
                iter->retry_after = now + kLocationCleanupRetryInterval;
                ++iter;
                continue;
            }
            iter->request.authoritative_read = true;
            iter->request.resume_deleting = true;
            iter->future =
                schedule_plan_executor_->SubmitLocationDelete(iter->request, ScheduleTaskClass::kMigrationContinuation);
            ++iter->retry_count;
            KVCM_LOG_WARN("retry asynchronous Copy location cleanup: instance %s storage %s retry %u",
                          iter->instance_id.c_str(),
                          iter->storage_name.c_str(),
                          iter->retry_count);
            ++iter;
            continue;
        }
        if (iter->future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            ++iter;
            continue;
        }

        bool completed = false;
        std::string failure_detail;
        ErrorCode failure_ec = EC_ERROR;
        try {
            const auto result = iter->future.get();
            completed = result.status == EC_OK;
            failure_ec = result.status;
            failure_detail = result.error_message;
        } catch (const std::exception &e) {
            failure_detail = e.what();
        } catch (...) {
            failure_detail = "unknown cleanup future exception";
        }
        if (completed) {
            iter = pending_location_cleanups_.erase(iter);
            continue;
        }

        // get() invalidates the future. Keep this record (and therefore the
        // backend/instance reference) while a bounded-delay retry is pending.
        iter->retry_after = now + kLocationCleanupRetryInterval;
        KVCM_LOG_WARN("asynchronous Copy location cleanup retained for retry: instance %s storage %s ec %d "
                      "detail %s retry %u",
                      iter->instance_id.c_str(),
                      iter->storage_name.c_str(),
                      static_cast<int>(failure_ec),
                      failure_detail.c_str(),
                      iter->retry_count);
        ++iter;
    }
}

bool MigrationManager::HasDurableUnknownGuard(const CopyTaskContext &ctx) const {
    auto indexer = meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(ctx.instance_id) : nullptr;
    if (!indexer || ctx.dst_location_id.empty() || ctx.async_operation_id.empty()) {
        return false;
    }
    auto request_context = std::make_shared<RequestContext>("migration_reconcile_unknown_guard");
    std::vector<CacheLocationMap> locations;
    const auto result = indexer->GetLocationsFromPersistent(request_context.get(), {ctx.block_key}, locations);
    if (result.error_codes.size() != 1 || result.error_codes[0] != EC_OK || locations.size() != 1) {
        return false;
    }
    const auto location_iter = locations[0].find(ctx.dst_location_id);
    return location_iter != locations[0].end() && location_iter->second != nullptr &&
           location_iter->second->has_migration_copy_guard() &&
           location_iter->second->migration_copy_guard().operation_id() == ctx.async_operation_id &&
           location_iter->second->migration_copy_guard().state() == MigrationCopyGuardState::MCGS_UNKNOWN;
}

bool MigrationManager::CompleteCopyTaskAsUnknown(const CopyTaskContext &ctx, const std::string &fail_reason) {
    if (ctx.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        GuardMutationResult guard_result = GuardMutationResult::kNotApplied;
        if (ctx.async_guard_pending_action == CopyCompletionAction::kUnknown &&
            ctx.async_guard_finalization_pending_sync) {
            auto indexer = meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(ctx.instance_id) : nullptr;
            guard_result = indexer && indexer->Sync({ctx.block_key}) ? GuardMutationResult::kAppliedDurably
                                                                     : GuardMutationResult::kAppliedDurabilityUnknown;
        } else {
            guard_result = UpdateAsyncCopyGuard(ctx,
                                                ExpectedGuardState(ctx),
                                                MigrationCopyGuardState::MCGS_UNKNOWN,
                                                ctx.async_backend_task_ids,
                                                fail_reason);
        }
        if (guard_result != GuardMutationResult::kAppliedDurably && HasDurableUnknownGuard(ctx)) {
            guard_result = GuardMutationResult::kAppliedDurably;
        }
        if (guard_result != GuardMutationResult::kAppliedDurably) {
            MarkUnknownGuardPersistencePending(
                ctx, fail_reason, guard_result == GuardMutationResult::kAppliedDurabilityUnknown);
            KVCM_LOG_ERROR("migration asynchronous Copy retained until UNKNOWN guard is durable: instance %s "
                           "block_key %lld dst_loc %s operation_id %s mutation_result %d reason %s",
                           ctx.instance_id.c_str(),
                           static_cast<long long>(ctx.block_key),
                           ctx.dst_location_id.c_str(),
                           ctx.async_operation_id.c_str(),
                           static_cast<int>(guard_result),
                           fail_reason.c_str());
            return false;
        }
        MoveAsyncCopyCreditToQuarantine(ctx, fail_reason);
    }
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        RemoveActiveTaskLocked(ctx.instance_id, ctx.block_key);
    }
    stat_copy_failed_.fetch_add(1, std::memory_order_relaxed);
    if (ctx.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        stat_async_copy_unknown_.fetch_add(1, std::memory_order_relaxed);
    }
    const int64_t duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ctx.submit_time)
                                    .count();
    if (metrics_enabled_) {
        ++m_tasks_completed_failed_;
    }
    if (event_manager_ != nullptr) {
        auto ev = std::make_shared<MigrationCompletedEvent>(ctx.instance_id);
        ev->SetEventTriggerTime();
        ev->SetAdditionalArgs(ctx.block_key,
                              ctx.src_storage_name,
                              ctx.dst_storage_name,
                              duration_ms,
                              ctx.total_bytes,
                              false,
                              fail_reason);
        event_manager_->Publish(ev);
    }
    KVCM_LOG_ERROR("migration asynchronous Copy entered quarantine: instance %s block_key %lld dst_loc %s "
                   "operation_id %s reason %s",
                   ctx.instance_id.c_str(),
                   static_cast<long long>(ctx.block_key),
                   ctx.dst_location_id.c_str(),
                   ctx.async_operation_id.c_str(),
                   fail_reason.c_str());
    return true;
}

bool MigrationManager::CompleteCopyTaskAsFailed(const CopyTaskContext &ctx,
                                                const std::string &fail_reason,
                                                bool remote_side_effect_possible) {
    if (ctx.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        const auto finalize_result = FinalizeAsyncCopyGuard(ctx, ExpectedGuardState(ctx), CLS_DELETING);
        if (finalize_result == GuardMutationResult::kAppliedDurabilityUnknown && remote_side_effect_possible) {
            MarkGuardFinalizationPendingSync(ctx, CopyCompletionAction::kFailed, fail_reason);
            return false;
        }
        if (finalize_result != GuardMutationResult::kAppliedDurably) {
            if (remote_side_effect_possible) {
                return CompleteCopyTaskAsUnknown(ctx, "guard_finalize_failed:" + fail_reason);
            }
            // The caller proves no backend handoff/POST occurred.  It is safe
            // to attempt ordinary target cleanup even if the SUBMITTING guard
            // CAS/Sync outcome is ambiguous.  The delete executor itself
            // still refuses a guard it can observe, so this can leak capacity
            // on a persistent-meta outage but can never recycle a remotely
            // writable destination.  Recovery will quarantine any durable
            // residual guard after restart.
            KVCM_LOG_WARN("pre-submit async Copy guard cleanup was not durably finalized: instance %s "
                          "block_key %lld dst_loc %s operation_id %s reason %s",
                          ctx.instance_id.c_str(),
                          static_cast<long long>(ctx.block_key),
                          ctx.dst_location_id.c_str(),
                          ctx.async_operation_id.c_str(),
                          fail_reason.c_str());
            SubmitTargetLocationDelete(ctx);
        } else {
            SubmitPreparedTargetLocationDelete(ctx);
        }
        ReleaseAsyncCopyCredit(ctx);
    } else {
        SubmitTargetLocationDelete(ctx);
    }
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        RemoveActiveTaskLocked(ctx.instance_id, ctx.block_key);
    }
    stat_copy_failed_.fetch_add(1, std::memory_order_relaxed);
    // 失败路径也填真实 duration（提交到失败的耗时），而非硬编码 0，
    // 便于区分"快速失败"与"跑很久才失败"。
    const int64_t duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ctx.submit_time)
                                    .count();
    if (metrics_enabled_) {
        ++m_tasks_completed_failed_;
    }
    if (event_manager_ != nullptr) {
        auto ev = std::make_shared<MigrationCompletedEvent>(ctx.instance_id);
        ev->SetEventTriggerTime();
        ev->SetAdditionalArgs(ctx.block_key,
                              ctx.src_storage_name,
                              ctx.dst_storage_name,
                              duration_ms,
                              ctx.total_bytes,
                              false,
                              fail_reason);
        event_manager_->Publish(ev);
    }
    KVCM_LOG_WARN("migration failed: instance %s block_key %lld dst_loc %s reason %s",
                  ctx.instance_id.c_str(),
                  static_cast<long long>(ctx.block_key),
                  ctx.dst_location_id.c_str(),
                  fail_reason.c_str());
    return true;
}

bool MigrationManager::CompleteCopyTaskAsSucceeded(const CopyTaskContext &ctx) {
    bool promoted = false;
    if (ctx.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        const auto finalize_result = FinalizeAsyncCopyGuard(ctx, ExpectedGuardState(ctx), CLS_SERVING);
        if (finalize_result == GuardMutationResult::kAppliedDurabilityUnknown) {
            MarkGuardFinalizationPendingSync(ctx, CopyCompletionAction::kSuccess);
            return false;
        }
        if (finalize_result == GuardMutationResult::kNotApplied) {
            return CompleteCopyTaskAsUnknown(ctx, "guard_finalize_failed:promote");
        }
        promoted = true;
    } else if (auto indexer = meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(ctx.instance_id) : nullptr;
               indexer) {
        MetaSearcher meta_searcher(indexer);
        auto rc = std::make_shared<RequestContext>("migration_on_success");
        std::vector<std::vector<MetaSearcher::LocationCASTask>> cas_tasks{
            {MetaSearcher::LocationCASTask{ctx.dst_location_id, CLS_WRITING, CLS_SERVING}}};
        std::vector<std::vector<ErrorCode>> cas_results;
        ErrorCode ec = meta_searcher.BatchCASLocationStatus(rc.get(), {ctx.block_key}, cas_tasks, cas_results);
        promoted = (ec == EC_OK && !cas_results.empty() && !cas_results[0].empty() && cas_results[0][0] == EC_OK);
    }

    if (!promoted) {
        // 目标提升失败：清理目标半成品，源端保持不动（数据未受损），按失败收尾。
        KVCM_LOG_WARN("migration promote dst location failed, block_key %ld dst_loc %s, treat as failed",
                      ctx.block_key,
                      ctx.dst_location_id.c_str());
        return CompleteCopyTaskAsFailed(ctx, "promote_failed");
    }

    // 按提交时快照的 mark target/deadline 做条件清除，避免清掉后续同 block 新 mark。
    if (!ctx.mark_target.empty()) {
        ClearTieredWriteMarkIfMatchInternal(ctx.instance_id, ctx.block_key, ctx.mark_target, ctx.mark_deadline_ms);
    }

    // 按 retention 处理源端。
    if (ctx.retention == MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE) {
        SubmitSourceLocationDelete(ctx);
    }

    ReleaseAsyncCopyCredit(ctx);

    // 最后移除活跃任务。
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        RemoveActiveTaskLocked(ctx.instance_id, ctx.block_key);
    }
    stat_copy_completed_.fetch_add(1, std::memory_order_relaxed);
    const int64_t duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ctx.submit_time)
                                    .count();
    if (metrics_enabled_) {
        ++m_tasks_completed_success_;
        m_copy_bytes_total_ += ctx.total_bytes;
        m_copy_duration_ms_ = static_cast<double>(duration_ms);
    }
    if (event_manager_ != nullptr) {
        auto ev = std::make_shared<MigrationCompletedEvent>(ctx.instance_id);
        ev->SetEventTriggerTime();
        ev->SetAdditionalArgs(
            ctx.block_key, ctx.src_storage_name, ctx.dst_storage_name, duration_ms, ctx.total_bytes, true, "");
        event_manager_->Publish(ev);
    }
    KVCM_LOG_INFO("migration completed: instance %s block_key %ld dst_loc %s retention %d",
                  ctx.instance_id.c_str(),
                  ctx.block_key,
                  ctx.dst_location_id.c_str(),
                  static_cast<int>(ctx.retention));
    return true;
}

bool MigrationManager::ResumePendingGuardFinalization(const CopyTaskContext &ctx) {
    switch (ctx.async_guard_pending_action) {
    case CopyCompletionAction::kSuccess:
        return CompleteCopyTaskAsSucceeded(ctx);
    case CopyCompletionAction::kFailed:
        return CompleteCopyTaskAsFailed(ctx, ctx.async_guard_pending_reason);
    case CopyCompletionAction::kCancelled:
        return CompleteCancelledTask(ctx);
    case CopyCompletionAction::kUnknown:
        return CompleteCopyTaskAsUnknown(
            ctx, ctx.async_guard_pending_reason.empty() ? "copy_completion_unknown" : ctx.async_guard_pending_reason);
    case CopyCompletionAction::kNone:
        KVCM_LOG_ERROR("guard Sync retry is missing its completion action: instance %s block_key %lld operation %s",
                       ctx.instance_id.c_str(),
                       static_cast<long long>(ctx.block_key),
                       ctx.async_operation_id.c_str());
        return CompleteCopyTaskAsUnknown(ctx, "guard_sync_retry_missing_action");
    }
    return true;
}

bool MigrationManager::OnTaskSuccess(const std::string &instance_id, int64_t block_key) {
    std::unique_lock<std::mutex> transition_lock;
    if (auto transition_mutex = GetAsyncTransitionMutex(instance_id, block_key)) {
        transition_lock = std::unique_lock<std::mutex>(*transition_mutex);
    }
    CopyTaskContext ctx;
    ClaimResult claim;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        claim = ClaimForCompletionLocked(instance_id, block_key, ctx);
    }
    if (claim == ClaimResult::kNotFound || claim == ClaimResult::kBusyCompleting) {
        return true; // 已处理过 / 完成认领中
    }
    if (claim == ClaimResult::kBusyPreparing) {
        KVCM_LOG_WARN("ignore premature migration success callback for preparing task: instance %s block_key %ld",
                      instance_id.c_str(),
                      block_key);
        return true;
    }
    if (claim == ClaimResult::kRetryingGuardTransition) {
        return ResumePendingGuardFinalization(ctx);
    }
    if (claim == ClaimResult::kWasCancelling) {
        // 用户已取消。copy 虽成功也丢弃：不 promote、不删源，删掉仍为 WRITING 的目标半成品。
        return CompleteCancelledTask(ctx);
    }
    // kClaimedRunning：状态已在锁内置 kCompleting，并发 Cancel 会走 busy 分支。

    // backend 已明确完成 Copy，说明这条 source->target 路由至少在本次可用；即使后续因为
    // source metadata 并发变化或 target promote 失败而按任务失败收尾，也不应继承旧的 Copy 退避。
    ClearCopySourceFailure(ctx);

    if (!IsSourceLocationServing(ctx)) {
        KVCM_LOG_WARN("migration source lost, block_key %ld src_loc %s, discard dst_loc %s",
                      block_key,
                      ctx.src_location_id.c_str(),
                      ctx.dst_location_id.c_str());
        return CompleteCopyTaskAsFailed(ctx, "source_lost");
    }
    return CompleteCopyTaskAsSucceeded(ctx);
}

bool MigrationManager::OnTaskFailed(const std::string &instance_id, int64_t block_key, ErrorCode reason) {
    return OnTaskFailedInternal(
        instance_id, block_key, reason, "copy_failed:" + std::to_string(static_cast<int>(reason)), true, true);
}

bool MigrationManager::OnTaskFailedInternal(const std::string &instance_id,
                                            int64_t block_key,
                                            ErrorCode reason,
                                            const std::string &fail_reason,
                                            bool remote_side_effect_possible,
                                            bool record_source_failure) {
    std::unique_lock<std::mutex> transition_lock;
    if (auto transition_mutex = GetAsyncTransitionMutex(instance_id, block_key)) {
        transition_lock = std::unique_lock<std::mutex>(*transition_mutex);
    }
    CopyTaskContext ctx;
    ClaimResult claim;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        claim = ClaimForCompletionLocked(instance_id, block_key, ctx);
    }
    if (claim == ClaimResult::kNotFound || claim == ClaimResult::kBusyCompleting) {
        return true;
    }
    if (claim == ClaimResult::kBusyPreparing) {
        KVCM_LOG_WARN("ignore premature migration failure callback for preparing task: instance %s block_key %ld",
                      instance_id.c_str(),
                      block_key);
        return true;
    }
    if (claim == ClaimResult::kRetryingGuardTransition) {
        return ResumePendingGuardFinalization(ctx);
    }
    if (claim == ClaimResult::kWasCancelling) {
        // 已取消，无论 copy 结果如何一律按取消收尾（清 WRITING 目标，记 cancelled 终态）。
        return CompleteCancelledTask(ctx);
    }

    // 失败：CAS 目标 WRITING -> DELETING 并删除目标半成品，源端不动。
    // 必须先登记 source 退避，再从 active table 移除任务，避免 admission 在两步之间重新选中同一源。
    if (record_source_failure) {
        RecordCopySourceFailure(ctx, reason);
    }
    return CompleteCopyTaskAsFailed(ctx, fail_reason, remote_side_effect_possible);
}

bool MigrationManager::OnTaskUnknown(const std::string &instance_id, int64_t block_key, const std::string &reason) {
    std::unique_lock<std::mutex> transition_lock;
    if (auto transition_mutex = GetAsyncTransitionMutex(instance_id, block_key)) {
        transition_lock = std::unique_lock<std::mutex>(*transition_mutex);
    }
    CopyTaskContext ctx;
    ClaimResult claim;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        claim = ClaimForCompletionLocked(instance_id, block_key, ctx);
    }
    if (claim == ClaimResult::kNotFound || claim == ClaimResult::kBusyCompleting) {
        return true;
    }
    if (claim == ClaimResult::kBusyPreparing) {
        KVCM_LOG_WARN("ignore premature unknown Copy callback for preparing task: instance %s block_key %lld",
                      instance_id.c_str(),
                      static_cast<long long>(block_key));
        return true;
    }
    if (claim == ClaimResult::kRetryingGuardTransition) {
        return ResumePendingGuardFinalization(ctx);
    }
    // Unknown ownership supersedes a concurrent cancel request: without a
    // terminal+drained proof, cancellation must not authorize target reuse.
    return CompleteCopyTaskAsUnknown(ctx, reason.empty() ? "copy_completion_unknown" : reason);
}

void MigrationManager::SubmitTargetLocationDelete(const CopyTaskContext &ctx) {
    if (!schedule_plan_executor_ || ctx.dst_location_id.empty()) {
        return;
    }
    // SubmitNonBlocking 只把任务放入 executor 队列；随后才会按精确 location id CAS 到 DELETING，
    // 再删存储和元数据。调用方不能把本函数返回理解为目标已经删除完成。
    CacheLocationDelRequest del_req;
    del_req.instance_id = ctx.instance_id;
    del_req.block_keys = {ctx.block_key};
    del_req.location_ids = {{ctx.dst_location_id}};
    if (ctx.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        TrackLocationCleanup(
            ctx.instance_id,
                             ctx.dst_storage_name,
            del_req,
            schedule_plan_executor_->SubmitLocationDelete(del_req, ScheduleTaskClass::kMigrationContinuation));
    } else {
        schedule_plan_executor_->SubmitNonBlocking(del_req, ScheduleTaskClass::kMigrationContinuation);
    }
}

void MigrationManager::SubmitSourceLocationDelete(const CopyTaskContext &ctx) {
    if (!schedule_plan_executor_ || ctx.src_location_id.empty()) {
        return;
    }
    CacheLocationDelRequest del_req;
    del_req.instance_id = ctx.instance_id;
    del_req.block_keys = {ctx.block_key};
    del_req.location_ids = {{ctx.src_location_id}};
    if (ctx.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        TrackLocationCleanup(
            ctx.instance_id,
                             ctx.src_storage_name,
            del_req,
            schedule_plan_executor_->SubmitLocationDelete(del_req, ScheduleTaskClass::kMigrationContinuation));
    } else {
        schedule_plan_executor_->SubmitNonBlocking(del_req, ScheduleTaskClass::kMigrationContinuation);
    }
}

void MigrationManager::MonitorLoop() {
    while (running_.load(std::memory_order_relaxed)) {
        ProcessExpiredMarks();
        ProcessCompletedLocationCleanups();

        PendingCopy cell;
        bool has_cell = false;
        {
            std::unique_lock<std::mutex> lock(pending_mutex_);
            if (pending_copies_.empty()) {
                pending_cv_.wait_for(lock, kMonitorIdleSleep, [this]() {
                    return !running_.load(std::memory_order_relaxed) || !pending_copies_.empty();
                });
            }
            if (!pending_copies_.empty()) {
                cell = std::move(pending_copies_.front());
                pending_copies_.pop_front();
                has_cell = true;
            }
        }
        if (!has_cell) {
            continue;
        }
        if (cell.completed_result.has_value() && std::chrono::steady_clock::now() < cell.completion_retry_after) {
            {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                pending_copies_.push_back(std::move(cell));
            }
            // Avoid a tight loop when this is the only pending operation.  No
            // task/guard lock is held while waiting for persistent metadata.
            std::this_thread::sleep_for(kGuardSyncRetryIdleSleep);
            continue;
        }

        if (cell.native_async && !cell.remote_submit_processed) {
            if (!cell.remote_submit_future.valid()) {
                OnTaskUnknown(cell.instance_id, cell.block_key, "copy_submit_acceptance_channel_lost");
                continue;
            }
            const auto remote_status = cell.remote_submit_future.wait_for(kFutureWaitTime);
            if (remote_status != std::future_status::ready) {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                pending_copies_.push_back(std::move(cell));
                continue;
            }

            AsyncCopyRemoteSubmitResult remote_result = cell.remote_submit_future.get();
            if (!remote_result.accepted) {
                if (remote_result.acceptance_unknown) {
                    OnTaskUnknown(cell.instance_id,
                                  cell.block_key,
                                  remote_result.detail.empty() ? "copy_submit_acceptance_unknown"
                                                               : remote_result.detail);
                } else {
                    const bool completed = OnTaskFailedInternal(cell.instance_id,
                                                                cell.block_key,
                                                                remote_result.status,
                                                                remote_result.detail.empty() ? "copy_submit_rejected"
                                                                                             : remote_result.detail,
                                                                false,
                                                                false);
                    if (!completed) {
                        cell.remote_submit_processed = true;
                        cell.completed_result =
                            PlanExecuteResult{remote_result.status, remote_result.detail, true, true};
                        cell.completion_retry_after = std::chrono::steady_clock::now() + kGuardSyncRetryInterval;
                        std::lock_guard<std::mutex> lock(pending_mutex_);
                        pending_copies_.push_back(std::move(cell));
                    }
                }
                continue;
            }

            CopyTaskContext accepted_ctx;
            bool cancel_requested = false;
            if (!PersistRemoteAsyncCopyAcceptance(cell.instance_id,
                                                  cell.block_key,
                                                  cell.expected_items,
                                                  remote_result,
                                                  accepted_ctx,
                                                  cancel_requested)) {
                if (!accepted_ctx.src_storage_name.empty() && schedule_plan_executor_) {
                    schedule_plan_executor_->RequestCancelAsyncCopy(accepted_ctx.src_storage_name,
                                                                    remote_result.operation_id);
                }
                OnTaskUnknown(cell.instance_id, cell.block_key, "guard_activate_failed_after_remote_submit");
                continue;
            }
            if (cancel_requested && schedule_plan_executor_) {
                schedule_plan_executor_->RequestCancelAsyncCopy(accepted_ctx.src_storage_name,
                                                                accepted_ctx.async_operation_id);
            }
            cell.remote_submit_processed = true;
        }

        if (!cell.completed_result.has_value()) {
            auto status = cell.future.wait_for(kFutureWaitTime);
            if (status != std::future_status::ready) {
                // 尚未完成，放回队尾稍后再查。
                std::lock_guard<std::mutex> lock(pending_mutex_);
                pending_copies_.push_back(std::move(cell));
                continue;
            }
            cell.completed_result = cell.future.get();
        }

        const PlanExecuteResult &result = *cell.completed_result;
        bool completed = true;
        if (result.status == EC_OK) {
            completed = OnTaskSuccess(cell.instance_id, cell.block_key);
        } else if (cell.native_async && (!result.terminal || !result.safe_to_reuse_dst)) {
            completed = OnTaskUnknown(cell.instance_id, cell.block_key, result.error_message);
        } else {
            completed = OnTaskFailed(cell.instance_id, cell.block_key, result.status);
        }
        if (!completed) {
            cell.completion_retry_after = std::chrono::steady_clock::now() + kGuardSyncRetryInterval;
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_copies_.push_back(std::move(cell));
        }
    }

    // Stop() owns the final handoff of pending futures.  A ready remote-submit
    // future may contain the only recoverable PACE task handles, so the monitor
    // must not discard process-local channels on its way out.
}

std::shared_ptr<MetaIndexer> MigrationManager::GetIndexer(const std::string &instance_id) const {
    return meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(instance_id) : nullptr;
}

ErrorCode MigrationManager::MarkForTieredWrite(const std::string &instance_id,
                                               const std::vector<int64_t> &block_keys,
                                               const std::string &dst_storage_name,
                                               int64_t timeout_ms) {
    if (dst_storage_name.empty() || block_keys.empty()) {
        return EC_OK;
    }
    // 新 Mark 只写入当前可分配的 target；已存在 Mark 遇到 quota 满或 unavailable 时由消费
    // 路径暂时忽略并保留，待容量/可用性恢复后自愈。
    const auto target_ec = CheckTargetStorageAdmission("mark_for_tiered_write", "", instance_id, dst_storage_name);
    if (target_ec != EC_OK) {
        KVCM_LOG_WARN("MarkForTieredWrite: target storage [%s] is not writable, skip marking (instance %s, ec %d)",
                      dst_storage_name.c_str(),
                      instance_id.c_str(),
                      target_ec);
        return target_ec;
    }
    if (timeout_ms <= 0) {
        timeout_ms = MigrationMarkMethod::kDefaultTimeoutMs;
    }
    auto indexer = GetIndexer(instance_id);
    if (indexer == nullptr) {
        KVCM_LOG_WARN("MarkForTieredWrite: meta indexer not found for instance %s", instance_id.c_str());
        return EC_INSTANCE_NOT_EXIST;
    }
    const int64_t now_ms = TimestampUtil::GetCurrentTimeMs();
    if (timeout_ms > std::numeric_limits<int64_t>::max() - now_ms) {
        KVCM_LOG_WARN(
            "MarkForTieredWrite: timeout %ld overflows deadline for instance %s", timeout_ms, instance_id.c_str());
        return EC_BADARGS;
    }
    const int64_t deadline_ms = now_ms + timeout_ms;
    // 记录实际成功打标的 key index，供 stat/expiry/event 仅按 actual 口径更新。
    // modifier 在 RMW 批次内按 global_idx 回调，顺序对齐 block_keys。
    std::vector<bool> mark_succeeded(block_keys.size(), false);
    // RMW：只写 property（out_new_locations 留空，不动 location）。不存在的 block 跳过（不给空 block 打标）。
    auto modifier =
        [&dst_storage_name, deadline_ms, &mark_succeeded](const LocationIdVector & /*existing*/,
                                                                     ErrorCode get_ec,
                                                                     size_t idx,
                                                                     PropertyMap &upsert_property_map,
                                                                     CacheLocationMap & /*out_new*/) -> ModifierResult {
        if (get_ec == EC_NOENT) {
            return {MA_SKIP, EC_OK};
        }
        if (get_ec != EC_OK) {
            return {MA_FAIL, get_ec};
        }
        upsert_property_map[PROPERTY_TIERED_WRITE_TARGET] = dst_storage_name;
        upsert_property_map[PROPERTY_TIERED_WRITE_DEADLINE_MS] = std::to_string(deadline_ms);
        if (idx < mark_succeeded.size()) {
            mark_succeeded[idx] = true;
        }
        return {MA_OK, EC_OK};
    };
    RequestContext rc("migration_mark");
    KeyVector keys(block_keys.begin(), block_keys.end());
    auto result = indexer->ReadModifyWriteBlock(&rc, keys, modifier);
    // stat/expiry/event 按实际成功数更新（actual 口径），不再用 block_keys.size()（request 口径）。
    const size_t actual_marked = static_cast<size_t>(std::count(mark_succeeded.begin(), mark_succeeded.end(), true));
    if (actual_marked > 0) {
        stat_marks_added_.fetch_add(actual_marked, std::memory_order_relaxed);
    }
    UpdateMarksActiveGauge();
    for (size_t i = 0; i < block_keys.size(); ++i) {
        if (!mark_succeeded[i]) {
            continue;
        }
        EnqueueMarkExpiry(instance_id, block_keys[i], dst_storage_name, deadline_ms);
        if (event_manager_ != nullptr) {
            auto ev = std::make_shared<MigrationMarkAddEvent>(instance_id);
            ev->SetEventTriggerTime();
            ev->SetAdditionalArgs(block_keys[i], dst_storage_name);
            event_manager_->Publish(ev);
        }
    }
    return result.ec;
}

bool MigrationManager::ClearTieredWriteMarkIfMatch(const std::string &instance_id,
                                                    int64_t block_key,
                                                    const std::string &expected_target,
                                                    int64_t expected_deadline_ms) {
    if (expected_target.empty() || expected_deadline_ms <= 0) {
        return false;
    }
    return ClearTieredWriteMarkIfMatchInternal(instance_id, block_key, expected_target, expected_deadline_ms);
}

bool MigrationManager::IsMarkedForTieredWrite(const std::string &instance_id, int64_t block_key) {
    std::vector<MarkQueryResult> results;
    const auto ec = BatchGetTieredWriteTargets(instance_id, {block_key}, results);
    return ec == EC_OK && !results.empty() && results[0].HasValidMark();
}

std::string MigrationManager::GetTieredWriteTarget(const std::string &instance_id, int64_t block_key) {
    std::vector<MarkQueryResult> results;
    const auto ec = BatchGetTieredWriteTargets(instance_id, {block_key}, results);
    return (ec == EC_OK && !results.empty() && results[0].HasValidMark()) ? results[0].target : std::string();
}

// match 检查在 RMW modifier 闭包内执行（shard lock 保原子），不需要外部 mark_mutex_。
// modifier 内通过捕获的 indexer 读 GetProperties（GetProperties 不走 shard lock，不死锁），
// 读-比较-写三步在同一个 shard lock 内完成，与 BatchCASLocationStatus 同模式。
bool MigrationManager::ClearTieredWriteMarkIfMatchInternal(const std::string &instance_id,
                                                           int64_t block_key,
                                                           const std::string &expected_target,
                                                           int64_t expected_deadline_ms,
                                                           bool is_expiry) {
    if (expected_target.empty()) {
        return false;
    }
    // private 调用以非正 deadline 表示“仅匹配当前仍 malformed 的同 target Mark”。
    // public ClearTieredWriteMarkIfMatch 仍拒绝非正 deadline，避免改变其精确快照语义。
    const bool expect_malformed = expected_deadline_ms <= 0;
    auto indexer = GetIndexer(instance_id);
    if (indexer == nullptr) {
        return false;
    }

    bool cleared = false;
    auto modifier = [&indexer, block_key, &expected_target, expected_deadline_ms, expect_malformed, &cleared](
                        const LocationIdVector & /*existing*/,
                        ErrorCode get_ec,
                        size_t /*idx*/,
                        PropertyMap &upsert_property_map,
                        CacheLocationMap & /*out_new*/) -> ModifierResult {
        if (get_ec != EC_OK) {
            return {MA_SKIP, EC_OK};
        }
        RequestContext check_rc("migration_match_clear_check");
        PropertyMapVector check_props;
        const auto check_result = indexer->GetProperties(
            &check_rc, {block_key}, {PROPERTY_TIERED_WRITE_TARGET, PROPERTY_TIERED_WRITE_DEADLINE_MS}, check_props);
        if (check_result.ec != EC_OK || check_result.error_codes.size() != 1 || check_result.error_codes[0] != EC_OK ||
            check_props.size() != 1) {
            return {MA_SKIP, EC_OK};
        }
        auto mark = ParseMarkFromProperties(check_props[0], 0);
        if (mark.target != expected_target ||
            (expect_malformed ? !mark.malformed : mark.deadline_ms != expected_deadline_ms)) {
            return {MA_SKIP, EC_OK};
        }
        upsert_property_map[PROPERTY_TIERED_WRITE_TARGET] = "";
        upsert_property_map[PROPERTY_TIERED_WRITE_DEADLINE_MS] = "";
        cleared = true;
        return {MA_OK, EC_OK};
    };
    RequestContext rc("migration_conditional_clear_mark");
    indexer->ReadModifyWriteBlock(&rc, {block_key}, modifier);
    if (cleared && !expect_malformed) {
        stat_marks_cleared_.fetch_add(1, std::memory_order_relaxed);
        UpdateMarksActiveGauge();
        if (metrics_enabled_) {
            if (is_expiry) {
                ++m_marks_expired_total_;
            } else {
                ++m_marks_consumed_total_;
            }
        }
        if (event_manager_ != nullptr) {
            if (is_expiry) {
                auto ev = std::make_shared<MigrationMarkExpiredEvent>(instance_id);
                ev->SetEventTriggerTime();
                ev->SetAdditionalArgs(block_key, expected_target);
                event_manager_->Publish(ev);
            } else {
                auto ev = std::make_shared<MigrationMarkConsumedEvent>(instance_id);
                ev->SetEventTriggerTime();
                ev->SetAdditionalArgs(block_key, expected_target);
                event_manager_->Publish(ev);
            }
        }
    }
    return cleared;
}

ErrorCode MigrationManager::BatchGetTieredWriteTargets(const std::string &instance_id,
                                                       const std::vector<int64_t> &block_keys,
                                                       std::vector<MarkQueryResult> &out) {
    out.assign(block_keys.size(), MarkQueryResult{});
    if (block_keys.empty()) {
        return EC_OK;
    }
    auto fail_all = [&](ErrorCode ec, const char *reason) {
        for (auto &result : out) {
            result.state = MarkQueryState::kReadError;
            result.ec = ec;
        }
        if (metrics_enabled_) {
            m_mark_query_errors_total_ += block_keys.size();
        }
        KVCM_LOG_WARN("mark query failed for instance %s, keys %zu, ec %d: %s",
                      instance_id.c_str(),
                      block_keys.size(),
                      ec,
                      reason);
        return ec;
    };

    auto indexer = GetIndexer(instance_id);
    if (indexer == nullptr) {
        return fail_all(EC_INSTANCE_NOT_EXIST, "meta indexer not found");
    }
    RequestContext rc("migration_query_mark_batch");
    KeyVector keys(block_keys.begin(), block_keys.end());
    PropertyMapVector props;
    const auto query_result =
        indexer->GetProperties(&rc, keys, {PROPERTY_TIERED_WRITE_TARGET, PROPERTY_TIERED_WRITE_DEADLINE_MS}, props);
    if (query_result.error_codes.size() != block_keys.size() || props.size() != block_keys.size()) {
        KVCM_LOG_WARN("mark query result shape mismatch for instance %s: keys %zu, per_key_ec %zu, props %zu, "
                      "aggregate ec %d",
                      instance_id.c_str(),
                      block_keys.size(),
                      query_result.error_codes.size(),
                      props.size(),
                      query_result.ec);
        return fail_all(EC_ERROR, "result shape mismatch");
    }

    const auto backend_error_count = static_cast<std::size_t>(std::count_if(
        query_result.error_codes.begin(), query_result.error_codes.end(), [](ErrorCode ec) { return ec != EC_OK; }));
    const ErrorCode expected_aggregate_ec =
        backend_error_count == 0 ? EC_OK : (backend_error_count == block_keys.size() ? EC_ERROR : EC_PARTIAL_OK);
    if (query_result.ec != expected_aggregate_ec) {
        return fail_all(EC_ERROR, "aggregate and per-key errors are inconsistent");
    }

    std::vector<ExpiringMark> expired_marks;
    std::vector<std::pair<int64_t, std::string>> malformed_marks;
    const int64_t now_ms = TimestampUtil::GetCurrentTimeMs();
    std::size_t read_error_count = 0;
    for (size_t i = 0; i < block_keys.size(); ++i) {
        out[i].ec = query_result.error_codes[i];
        if (out[i].ec == EC_NOENT) {
            out[i].state = MarkQueryState::kBlockNotFound;
            continue;
        }
        if (out[i].ec != EC_OK) {
            out[i].state = MarkQueryState::kReadError;
            ++read_error_count;
            continue;
        }
        auto mark = ParseMarkFromProperties(props[i], now_ms);
        if (mark.malformed) {
            KVCM_LOG_WARN("malformed tiered write mark for instance %s, block %ld, target %s; "
                          "treating as no-mark and clearing conditionally",
                          instance_id.c_str(),
                          block_keys[i],
                          mark.target.c_str());
            malformed_marks.emplace_back(block_keys[i], std::move(mark.target));
            out[i].state = MarkQueryState::kNoMark;
            continue;
        }
        if (mark.target.empty()) {
            out[i].state = MarkQueryState::kNoMark;
            continue;
        }
        out[i].target = std::move(mark.target);
        out[i].deadline_ms = mark.deadline_ms;
        if (mark.expired) {
            out[i].state = MarkQueryState::kExpired;
            expired_marks.push_back(ExpiringMark{out[i].deadline_ms, instance_id, block_keys[i], out[i].target});
            continue;
        }
        out[i].state = MarkQueryState::kValid;
    }
    for (const auto &em : expired_marks) {
        ClearTieredWriteMarkIfMatchInternal(em.instance_id, em.block_key, em.target_storage, em.deadline_ms, true);
    }
    for (const auto &[block_key, target] : malformed_marks) {
        ClearTieredWriteMarkIfMatchInternal(instance_id, block_key, target, 0);
    }
    if (read_error_count == 0) {
        return EC_OK;
    }
    if (metrics_enabled_) {
        m_mark_query_errors_total_ += read_error_count;
    }
    KVCM_LOG_WARN("mark query partially failed for instance %s: keys %zu, read errors %zu",
                  instance_id.c_str(),
                  block_keys.size(),
                  read_error_count);
    return read_error_count == block_keys.size() ? EC_ERROR : EC_PARTIAL_OK;
}

void MigrationManager::EnqueueMarkExpiry(const std::string &instance_id,
                                         int64_t block_key,
                                         const std::string &target_storage,
                                         int64_t deadline_ms) {
    if (deadline_ms <= 0 || target_storage.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mark_expiry_mutex_);
        mark_expiry_queue_.push(ExpiringMark{deadline_ms, instance_id, block_key, target_storage});
    }
    pending_cv_.notify_one();
}

void MigrationManager::ProcessExpiredMarks() {
    const int64_t now_ms = TimestampUtil::GetCurrentTimeMs();
    std::vector<ExpiringMark> expired_marks;
    {
        std::lock_guard<std::mutex> lock(mark_expiry_mutex_);
        while (!mark_expiry_queue_.empty() && mark_expiry_queue_.top().deadline_ms <= now_ms) {
            expired_marks.push_back(mark_expiry_queue_.top());
            mark_expiry_queue_.pop();
        }
    }
    for (const auto &mark : expired_marks) {
        ClearTieredWriteMarkIfMatchInternal(
            mark.instance_id, mark.block_key, mark.target_storage, mark.deadline_ms, true);
    }
}

bool MigrationManager::CompleteCancelledTask(const CopyTaskContext &ctx) {
    // 取消任务的延迟收尾（monitor 线程，认领到 kWasCancelling 时调用）。
    // 目标此时仍为 WRITING（cancelling 任务从未被 promote）；CAS WRITING->DELETING 删半成品，源端不动。
    if (ctx.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        const auto finalize_result = FinalizeAsyncCopyGuard(ctx, ExpectedGuardState(ctx), CLS_DELETING);
        if (finalize_result == GuardMutationResult::kAppliedDurabilityUnknown) {
            MarkGuardFinalizationPendingSync(ctx, CopyCompletionAction::kCancelled, "cancelled");
            return false;
        }
        if (finalize_result == GuardMutationResult::kNotApplied) {
            return CompleteCopyTaskAsUnknown(ctx, "cancel_guard_finalize_failed");
        }
        SubmitPreparedTargetLocationDelete(ctx);
        ReleaseAsyncCopyCredit(ctx);
    } else {
        SubmitTargetLocationDelete(ctx);
    }
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        RemoveActiveTaskLocked(ctx.instance_id, ctx.block_key);
    }
    stat_copy_cancelled_.fetch_add(1, std::memory_order_relaxed);
    // cancelled 是与 success/failed 对称的终态：在实际清理时计数，保持 submitted==success+failed+cancelled。
    // 被取消任务无论底层 copy 成/败一律记 cancelled（用户意图优先）。
    const int64_t duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ctx.submit_time)
                                    .count();
    if (metrics_enabled_) {
        ++m_tasks_completed_cancelled_;
    }
    if (event_manager_ != nullptr) {
        auto ev = std::make_shared<MigrationCompletedEvent>(ctx.instance_id);
        ev->SetEventTriggerTime();
        ev->SetAdditionalArgs(ctx.block_key,
                              ctx.src_storage_name,
                              ctx.dst_storage_name,
                              duration_ms,
                              ctx.total_bytes,
                              /*success=*/false,
                              /*fail_reason=*/"cancelled");
        event_manager_->Publish(ev);
    }
    KVCM_LOG_INFO("migration cancelled (deferred cleanup): instance %s block_key %ld dst_loc %s",
                  ctx.instance_id.c_str(),
                  ctx.block_key,
                  ctx.dst_location_id.c_str());
    return true;
}

ErrorCode MigrationManager::Cancel(const std::string &instance_id, int64_t block_key) {
    // preparing 取消只标记 kPrepareCancelling，由仍在同步 I/O 中的提交线程在下一安全
    // 边界停止进入 copy、清目标并释放 reservation；running 取消仍等待 future 完成后由 monitor 收尾。
    // 两种 cancelling 期间任务都保留在活跃表，继续挡重复 Submit 并保护 WRITING 目标。
    CancelResult result;
    CopyTaskContext cancelling_ctx;
    bool persist_async_cancelling = false;
    // Serialize this operation's in-memory cancellation transition and durable
    // guard update with submit acceptance and completion finalization.
    std::unique_lock<std::mutex> guard_transition_lock;
    if (auto transition_mutex = GetAsyncTransitionMutex(instance_id, block_key)) {
        guard_transition_lock = std::unique_lock<std::mutex>(*transition_mutex);
    }
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        result = MarkCancellingLocked(instance_id, block_key);
        if (result == CancelResult::kMarked) {
            const auto instance_iter = active_tasks_by_instance_.find(instance_id);
            if (instance_iter != active_tasks_by_instance_.end()) {
                const auto task_iter = instance_iter->second.find(block_key);
                if (task_iter != instance_iter->second.end() &&
                    task_iter->second.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
                    cancelling_ctx = task_iter->second;
                    persist_async_cancelling = true;
                }
            }
        }
    }
    if (persist_async_cancelling &&
        UpdateAsyncCopyGuard(cancelling_ctx,
                             cancelling_ctx.async_backend_task_ids.empty() ? MigrationCopyGuardState::MCGS_SUBMITTING
                                                                           : MigrationCopyGuardState::MCGS_ACTIVE,
                             MigrationCopyGuardState::MCGS_CANCELLING,
                             cancelling_ctx.async_backend_task_ids,
                             "cancel_requested") != GuardMutationResult::kAppliedDurably) {
        // The in-memory cancellation must not outrun its persistent fence.
        // Drop ownership into fail-closed quarantine; a late completion can no
        // longer publish or release the target through this process.
        CompleteCopyTaskAsUnknown(cancelling_ctx, "cancel_guard_persist_failed");
        return EC_ERROR;
    }
    if (guard_transition_lock.owns_lock()) {
        guard_transition_lock.unlock();
    }
    if (persist_async_cancelling && schedule_plan_executor_) {
        const auto cancel_ec = schedule_plan_executor_->RequestCancelAsyncCopy(cancelling_ctx.src_storage_name,
                                                                               cancelling_ctx.async_operation_id);
        if (cancel_ec != EC_OK) {
            // The persistent state is already CANCELLING.  A failed immediate
            // signal is not a drain proof; the coordinator keeps polling and
            // will retry cancellation at its deadline.
            KVCM_LOG_WARN("[cancel] async coordinator did not accept cancellation immediately: "
                          "instance %s block_key %lld operation_id %s ec %d",
                          instance_id.c_str(),
                          static_cast<long long>(block_key),
                          cancelling_ctx.async_operation_id.c_str(),
                          static_cast<int>(cancel_ec));
        }
    }
    switch (result) {
    case CancelResult::kNotFound:
        return EC_NOENT;
    case CancelResult::kMarkedPreparing:
        KVCM_LOG_INFO("[cancel] instance %s block_key %ld cancelled while preparing; submitter will clean up",
                      instance_id.c_str(),
                      block_key);
        return EC_OK;
    case CancelResult::kBusyCompleting:
        KVCM_LOG_INFO(
            "[cancel] instance %s block_key %ld already completing, cancel too late", instance_id.c_str(), block_key);
        return EC_EXIST; // 完成中，取消太晚；迁移将照常完成
    case CancelResult::kMarked:
        KVCM_LOG_INFO("[cancel] instance %s block_key %ld marked cancelling; cleanup deferred to copy completion",
                      instance_id.c_str(),
                      block_key);
        return EC_OK;
    case CancelResult::kAlreadyCancelling:
        return EC_OK; // 幂等
    }
    return EC_OK; // 不可达
}

std::vector<ErrorCode> MigrationManager::BatchCancel(const std::string &instance_id,
                                                     const std::vector<int64_t> &block_keys) {
    std::vector<ErrorCode> results;
    results.reserve(block_keys.size());
    for (int64_t block_key : block_keys) {
        results.push_back(Cancel(instance_id, block_key));
    }
    return results;
}

bool MigrationManager::InsertActiveTaskLocked(CopyTaskContext ctx) {
    const std::string instance_id = ctx.instance_id;
    const int64_t block_key = ctx.block_key;
    auto &instance_tasks = active_tasks_by_instance_[instance_id];
    if (instance_tasks.count(block_key) > 0) {
        return false; // 已存在：调用方决定如何处理（回滚等）
    }
    if (ctx.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED && !ctx.async_transition_mutex) {
        ctx.async_transition_mutex = std::make_shared<std::mutex>();
    }
    instance_tasks[block_key] = std::move(ctx);
    UpdateActiveTasksGauge();
    return true;
}

bool MigrationManager::ReservePreparingTaskLocked(const MigrationRequest &request) {
    CopyTaskContext ctx;
    ctx.instance_group_name = request.instance_group_name;
    ctx.instance_id = request.instance_id;
    ctx.block_key = request.block_key;
    ctx.src_location_id = request.src_location_id;
    ctx.src_create_time = request.src_create_time;
    ctx.src_storage_name = request.src_storage_name;
    ctx.dst_storage_name = request.dst_storage_name;
    ctx.retention = request.retention;
    ctx.copy_execution_mode = request.copy_execution_mode;
    ctx.async_operation_id = request.async_operation_id;
    if (ctx.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        ctx.async_transition_mutex = std::make_shared<std::mutex>();
    }
    ctx.state = CopyTaskState::kPreparing;
    // InsertActiveTaskLocked 在同一个 task_mutex_ 临界区内完成存在性检查与插入，避免 check-then-insert 窗口。
    return InsertActiveTaskLocked(std::move(ctx));
}

bool MigrationManager::UpdatePreparingTaskLocked(CopyTaskContext &ctx) {
    auto instance_iter = active_tasks_by_instance_.find(ctx.instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return false;
    }
    auto task_iter = instance_iter->second.find(ctx.block_key);
    if (task_iter == instance_iter->second.end() || task_iter->second.state != CopyTaskState::kPreparing) {
        return false;
    }
    if (ctx.copy_execution_mode == MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        // Preserve the mutex published with the preparing reservation.  A
        // Cancel caller may already hold a shared_ptr to it while waiting for
        // task_mutex_; replacing it would split one operation across locks.
        if (!task_iter->second.async_transition_mutex) {
            task_iter->second.async_transition_mutex = std::make_shared<std::mutex>();
        }
        ctx.async_transition_mutex = task_iter->second.async_transition_mutex;
    }
    task_iter->second = ctx;
    task_iter->second.state = CopyTaskState::kPreparing;
    return true;
}

bool MigrationManager::MarkTaskRunningLocked(const std::string &instance_id, int64_t block_key) {
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return false;
    }
    auto task_iter = instance_iter->second.find(block_key);
    if (task_iter == instance_iter->second.end() || task_iter->second.state != CopyTaskState::kPreparing) {
        return false;
    }
    task_iter->second.state = CopyTaskState::kRunning;
    return true;
}

bool MigrationManager::RemovePreparingTaskLocked(const std::string &instance_id, int64_t block_key) {
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return false;
    }
    auto task_iter = instance_iter->second.find(block_key);
    if (task_iter == instance_iter->second.end() || (task_iter->second.state != CopyTaskState::kPreparing &&
                                                     task_iter->second.state != CopyTaskState::kPrepareCancelling)) {
        return false;
    }
    return RemoveActiveTaskLocked(instance_id, block_key);
}

bool MigrationManager::RemoveActiveTaskLocked(const std::string &instance_id, int64_t block_key) {
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return false;
    }
    const size_t erased = instance_iter->second.erase(block_key);
    if (erased == 0) {
        return false;
    }
    if (instance_iter->second.empty()) {
        active_tasks_by_instance_.erase(instance_iter); // 内层空则收回外层，避免 stale 空 map
    }
    UpdateActiveTasksGauge();
    return true;
}

bool MigrationManager::HasActiveTaskLocked(const std::string &instance_id, int64_t block_key) const {
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    return instance_iter != active_tasks_by_instance_.end() && instance_iter->second.count(block_key) > 0;
}

MigrationManager::ClaimResult MigrationManager::ClaimForCompletionLocked(const std::string &instance_id,
                                                                        int64_t block_key,
                                                                        CopyTaskContext &out_ctx) {
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return ClaimResult::kNotFound;
    }
    auto task_iter = instance_iter->second.find(block_key);
    if (task_iter == instance_iter->second.end()) {
        return ClaimResult::kNotFound;
    }
    CopyTaskContext &task = task_iter->second;
    if (task.state == CopyTaskState::kPreparing || task.state == CopyTaskState::kPrepareCancelling) {
        return ClaimResult::kBusyPreparing;
    }
    if (task.state == CopyTaskState::kCancelling) {
        task.state = CopyTaskState::kCompleting;
        task.completion_was_cancelling = true;
        out_ctx = task; // 交由 CompleteCancelledTask 收尾（删 WRITING 目标、不 promote）
        return ClaimResult::kWasCancelling;
    }
    if (task.state == CopyTaskState::kCompleting) {
        if (task.async_guard_pending_action != CopyCompletionAction::kNone) {
            out_ctx = task;
            return ClaimResult::kRetryingGuardTransition;
        }
        return ClaimResult::kBusyCompleting; // 防御：完成路径仅单一 monitor 线程，正常不会重入
    }
    task.state = CopyTaskState::kCompleting; // 认领：此后并发 Cancel 走 busy 分支
    out_ctx = task;
    return ClaimResult::kClaimedRunning;
}

MigrationManager::CancelResult MigrationManager::MarkCancellingLocked(const std::string &instance_id,
                                                                     int64_t block_key) {
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return CancelResult::kNotFound;
    }
    auto task_iter = instance_iter->second.find(block_key);
    if (task_iter == instance_iter->second.end()) {
        return CancelResult::kNotFound;
    }
    CopyTaskContext &task = task_iter->second;
    if (task.state == CopyTaskState::kPreparing) {
        task.state = CopyTaskState::kPrepareCancelling;
        return CancelResult::kMarkedPreparing;
    }
    if (task.state == CopyTaskState::kCompleting) {
        return CancelResult::kBusyCompleting; // 完成中，取消太晚
    }
    if (task.state == CopyTaskState::kPrepareCancelling || task.state == CopyTaskState::kCancelling) {
        return CancelResult::kAlreadyCancelling; // 幂等
    }
    task.state = CopyTaskState::kCancelling; // 标记；收尾推迟到 future 完成时由 monitor 执行
    return CancelResult::kMarked;
}

bool MigrationManager::HasMigrationTask(const std::string &instance_id, int64_t block_key) const {
    std::lock_guard<std::mutex> lock(task_mutex_);
    return HasActiveTaskLocked(instance_id, block_key);
}

bool MigrationManager::HasActiveCopyTargetLocation(const std::string &instance_id,
                                                   int64_t block_key,
                                                   const std::string &location_id) const {
    if (location_id.empty()) {
        return false;
    }
    // 作用域限定为 (instance_id, block_key)：一次 copy 任务的目标 location 属于同一 block；
    // 直接 O(1) 定位，避免全表扫描，也避免跨 instance/block 的 id 碰撞误判。
    std::lock_guard<std::mutex> lock(task_mutex_);
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return false;
    }
    auto task_iter = instance_iter->second.find(block_key);
    if (task_iter == instance_iter->second.end()) {
        return false;
    }
    const CopyTaskContext &task = task_iter->second;
    // BatchAddLocation 可能已经提交、但 location_id 尚未返回给提交线程。
    // 此时按 (instance, block) 临时保护所有 WRITING location；id 一旦绑定即恢复精确匹配。
    if ((task.state == CopyTaskState::kPreparing || task.state == CopyTaskState::kPrepareCancelling) &&
        task.dst_location_id.empty()) {
        return true;
    }
    return task.dst_location_id == location_id;
}

size_t MigrationManager::ActiveTaskCountUnsafe() const {
    size_t total = 0;
    for (const auto &instance_entry : active_tasks_by_instance_) {
        total += instance_entry.second.size();
    }
    return total;
}

size_t MigrationManager::ActiveTaskCount() const {
    std::lock_guard<std::mutex> lock(task_mutex_);
    return ActiveTaskCountUnsafe();
}

size_t MigrationManager::ActiveTaskCountForGroupUnsafe(const std::string &instance_group_name) const {
    size_t total = 0;
    for (const auto &instance_entry : active_tasks_by_instance_) {
        for (const auto &task_entry : instance_entry.second) {
            if (task_entry.second.instance_group_name == instance_group_name) {
                ++total;
            }
        }
    }
    return total;
}

size_t MigrationManager::ActiveTaskCountForGroup(const std::string &instance_group_name) const {
    std::lock_guard<std::mutex> lock(task_mutex_);
    return ActiveTaskCountForGroupUnsafe(instance_group_name);
}

size_t MigrationManager::ActiveTaskCountForInstances(const std::vector<std::string> &instance_ids) const {
    std::lock_guard<std::mutex> lock(task_mutex_);
    size_t total = 0;
    for (const auto &id : instance_ids) {
        auto it = active_tasks_by_instance_.find(id);
        if (it != active_tasks_by_instance_.end()) {
            total += it->second.size();
        }
    }
    return total;
}

std::vector<int64_t> MigrationManager::GetActiveBlockKeysForInstance(const std::string &instance_id) const {
    std::lock_guard<std::mutex> lock(task_mutex_);
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return {};
    }
    std::vector<int64_t> keys;
    keys.reserve(instance_iter->second.size());
    for (const auto &entry : instance_iter->second) {
        keys.push_back(entry.first);
    }
    return keys;
}

void MigrationManager::BeginDrainingInstance(const std::string &instance_id) {
    std::lock_guard<std::mutex> lock(copy_submission_mutex_);
    draining_instances_.insert(instance_id);
}

void MigrationManager::EndDrainingInstance(const std::string &instance_id) {
    std::lock_guard<std::mutex> lock(copy_submission_mutex_);
    draining_instances_.erase(instance_id);
}

std::string MigrationManager::GetActiveTaskDstLocation(const std::string &instance_id, int64_t block_key) const {
    std::lock_guard<std::mutex> lock(task_mutex_);
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return std::string();
    }
    auto iter = instance_iter->second.find(block_key);
    if (iter == instance_iter->second.end()) {
        return std::string();
    }
    return iter->second.dst_location_id;
}

void MigrationManager::DebugInsertActiveCopyTask(const std::string &instance_id,
                                                 int64_t block_key,
                                                 const std::string &dst_location_id) {
    std::lock_guard<std::mutex> lock(task_mutex_);
    CopyTaskContext ctx;
    ctx.instance_id = instance_id;
    ctx.block_key = block_key;
    ctx.dst_location_id = dst_location_id;
    InsertActiveTaskLocked(std::move(ctx));
}

void MigrationManager::DebugEnableCopySubmissionsForTest() {
    accepting_copy_submissions_.store(true, std::memory_order_release);
}

MigrationManager::CopyAdmission MigrationManager::CheckCopyAdmission(const std::string &instance_id,
                                                                     int64_t block_key,
                                                                     const CacheLocationMap &loc_map,
                                                                     const std::string &src_storage_name,
                                                                     const std::string &dst_storage_name) const {
    if (HasMigrationTask(instance_id, block_key)) {
        return {CopyAdmissionStatus::kAlreadyMigrating, nullptr};
    }
    const auto src_locations = FindLocationsOnStorage(loc_map, src_storage_name, {CacheLocationStatus::CLS_SERVING});
    if (src_locations.empty()) {
        return {CopyAdmissionStatus::kSourceServingNotFound, nullptr};
    }

    // 按目标 storage 上多个 location 的联合覆盖判断，替代旧的 per-location 全覆盖。
    // 建索引一次 O(L·S),之后 per-source O(S) set lookup。
    const auto serving_covered = CollectCoveredSpecNames(loc_map, dst_storage_name, {CacheLocationStatus::CLS_SERVING});
    const auto all_covered = CollectCoveredSpecNames(
        loc_map, dst_storage_name, {CacheLocationStatus::CLS_SERVING, CacheLocationStatus::CLS_WRITING});

    auto specs_covered_by = [](const CacheLocation &src, const std::unordered_set<std::string> &covered) {
        return std::all_of(src.location_specs().begin(), src.location_specs().end(), [&covered](const auto &spec) {
            return covered.count(spec.name()) > 0;
        });
    };

    struct EligibleSource {
        const CacheLocation *location = nullptr;
        CopySourceFailureSnapshot failure;
    };

    const auto now = std::chrono::steady_clock::now();
    bool all_sources_covered_by_serving = true;
    std::size_t uncovered_source_count = 0;
    std::size_t suppressed_source_count = 0;
    std::size_t pinned_source_count = 0;
    std::vector<EligibleSource> eligible_sources;
    for (const auto *src_loc : src_locations) {
        if (specs_covered_by(*src_loc, serving_covered)) {
            continue;
        }
        all_sources_covered_by_serving = false;
        if (specs_covered_by(*src_loc, all_covered)) {
            continue;
        }
        // 该 source 既未被 SERVING 也未被 SERVING∪WRITING 联合覆盖 → 需要 copy。
        ++uncovered_source_count;
        // A durable guard pins the exact source identity until that operation
        // reaches a proven terminal state or an operator releases quarantine.
        // Starting another route from the same source could later succeed with
        // DELETE_SOURCE while deletion is still fenced by the first guard,
        // leaving retention work with no durable retry owner.  Defer the second
        // migration instead; after the guard is cleared, ordinary sampling can
        // select this source again and DELETE_SOURCE can complete normally.
        if (FindPersistentMigrationSourcePin(*src_loc, loc_map) != nullptr) {
            ++pinned_source_count;
            continue;
        }
        const auto failure = GetCopySourceFailure(instance_id, block_key, *src_loc, dst_storage_name, now);
        if (failure.suppressed) {
            ++suppressed_source_count;
            continue;
        }
        eligible_sources.push_back(EligibleSource{src_loc, failure});
    }

    if (!eligible_sources.empty()) {
        // 优先选择从未失败的 source；都失败过时选择连续失败次数更少的 source。这样即使
        // Reclaimer 周期长于首次 backoff，已知坏源也不会在健康副本之前被反复选中。
        const auto selected = std::min_element(
            eligible_sources.begin(), eligible_sources.end(), [](const EligibleSource &lhs, const EligibleSource &rhs) {
                if (lhs.failure.consecutive_failures != rhs.failure.consecutive_failures) {
                    return lhs.failure.consecutive_failures < rhs.failure.consecutive_failures;
                }
                return lhs.location->id() < rhs.location->id();
            });

        const bool switched_source =
            uncovered_source_count > 1 &&
            (suppressed_source_count > 0 ||
             std::any_of(eligible_sources.begin(), eligible_sources.end(), [&](const auto &item) {
                 return item.location != selected->location &&
                        item.failure.consecutive_failures > selected->failure.consecutive_failures;
             }));
        if (switched_source) {
            stat_source_switches_.fetch_add(1, std::memory_order_relaxed);
            if (metrics_enabled_) {
                ++m_source_switches_total_;
            }
            KVCM_LOG_INFO("migration admission switches Copy source: instance %s block_key %lld target_storage %s "
                          "selected_source %s selected_prior_failures %u suppressed_sources %zu",
                          instance_id.c_str(),
                          static_cast<long long>(block_key),
                          dst_storage_name.c_str(),
                          selected->location->id().c_str(),
                          selected->failure.consecutive_failures,
                          suppressed_source_count);
        }
        if (suppressed_source_count > 0) {
            stat_source_retries_suppressed_.fetch_add(suppressed_source_count, std::memory_order_relaxed);
            if (metrics_enabled_) {
                m_source_retries_suppressed_total_ += suppressed_source_count;
            }
        }
        return {CopyAdmissionStatus::kAccept, selected->location};
    }

    if (suppressed_source_count > 0) {
        stat_source_retries_suppressed_.fetch_add(suppressed_source_count, std::memory_order_relaxed);
        stat_no_usable_source_.fetch_add(1, std::memory_order_relaxed);
        if (metrics_enabled_) {
            m_source_retries_suppressed_total_ += suppressed_source_count;
            ++m_no_usable_source_total_;
        }
        return {CopyAdmissionStatus::kSourceRetrySuppressed, nullptr};
    }

    if (pinned_source_count > 0) {
        return {CopyAdmissionStatus::kSourcePinnedByGuard, nullptr};
    }

    // 循环走完未 return kAccept：每个 source 都被 serving∪writing 覆盖。
    if (all_sources_covered_by_serving) {
        return {CopyAdmissionStatus::kTargetServingExists, nullptr};
    }
    // 上面已排除"全部 SERVING 覆盖"，故此处必为有 WRITING 参与的联合覆盖。
    return {CopyAdmissionStatus::kTargetWritingExists, nullptr};
}

std::pair<ErrorCode, std::vector<std::int64_t>>
MigrationManager::SelectMigrationCandidateKeys(RequestContext *request_context,
                                               const std::string &trace_id,
                                               const std::vector<int64_t> &explicit_block_keys,
                                               int64_t sample_count,
                                               const std::shared_ptr<MetaIndexer> &meta_indexer) const {
    std::vector<std::int64_t> candidate_keys;
    if (!explicit_block_keys.empty()) {
        candidate_keys.reserve(explicit_block_keys.size());
        std::unordered_set<int64_t> seen;
        seen.reserve(explicit_block_keys.size());
        for (const auto block_key : explicit_block_keys) {
            if (seen.insert(block_key).second) {
                candidate_keys.push_back(block_key);
            }
        }
        return {EC_OK, std::move(candidate_keys)};
    }

    constexpr int64_t kDefaultMigrateSampleCount = 100;
    const int64_t count = sample_count > 0 ? sample_count : kDefaultMigrateSampleCount;
    if (const auto ec = meta_indexer->SampleReclaimKeys(request_context, count, candidate_keys); ec != EC_OK) {
        KVCM_LOG_WARN("[%s] MigrateCache sample keys failed, ec %d", trace_id.c_str(), ec);
        return {ec, {}};
    }
    return {EC_OK, std::move(candidate_keys)};
}

std::size_t MigrationManager::AsyncMigrationPrepareKeyHash::operator()(const AsyncMigrationPrepareKey &key) const {
    std::size_t seed = std::hash<std::uint64_t>{}(key.generation);
    const auto combine = [&seed](std::size_t value) { seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2); };
    combine(std::hash<std::string>{}(key.instance_group_name));
    combine(std::hash<std::string>{}(key.instance_id));
    combine(std::hash<std::string>{}(key.source_storage_name));
    combine(std::hash<std::string>{}(key.target_storage_name));
    return seed;
}

std::size_t MigrationManager::PendingAsyncMigrationPrepareCountForTest() const {
    std::lock_guard<std::mutex> lock(async_prepare_mutex_);
    return pending_async_prepare_jobs_.size();
}

std::size_t MigrationManager::PendingAsyncMigrationPrepareCountForGroup(const std::string &instance_group_name) const {
    std::lock_guard<std::mutex> lock(async_prepare_mutex_);
    return static_cast<std::size_t>(std::count_if(pending_async_prepare_jobs_.begin(),
                      pending_async_prepare_jobs_.end(),
                      [&instance_group_name](const AsyncMigrationPrepareKey &key) {
                          return key.instance_group_name == instance_group_name;
                      }));
}

void MigrationManager::FinishAsyncMigrationPrepare(const AsyncMigrationPrepareKey &key) {
    std::lock_guard<std::mutex> lock(async_prepare_mutex_);
    pending_async_prepare_jobs_.erase(key);
}

bool MigrationManager::SubmitAsyncMigrationPrepare(AsyncMigrationPrepareJob job) {
    if (!schedule_plan_executor_ || !accepting_copy_submissions_.load(std::memory_order_acquire) ||
        job.instance_group_name.empty() || job.instance_id.empty() || job.source_storage_name.empty() ||
        job.target_storage_name.empty() || job.block_keys.empty() ||
        job.pending_location_ids_by_block.size() != job.block_keys.size()) {
        return false;
    }

    const auto generation = async_prepare_generation_.load(std::memory_order_acquire);
    AsyncMigrationPrepareKey key{
        generation, job.instance_group_name, job.instance_id, job.source_storage_name, job.target_storage_name};
    {
        std::lock_guard<std::mutex> lock(async_prepare_mutex_);
        if (!accepting_copy_submissions_.load(std::memory_order_acquire) ||
            async_prepare_generation_.load(std::memory_order_acquire) != generation ||
            pending_async_prepare_jobs_.size() >= kMaxPendingAsyncMigrationPrepareJobs ||
            !pending_async_prepare_jobs_.insert(key).second) {
            return false;
        }
    }

    auto weak_self = weak_from_this();
    if (weak_self.expired()) {
        FinishAsyncMigrationPrepare(key);
        return false;
    }
    auto execute_task = [weak_self, key, job = std::move(job)]() mutable {
        auto self = weak_self.lock();
        if (!self) {
            return;
        }
        try {
            self->RunAsyncMigrationPrepare(std::move(job), key.generation);
        } catch (const std::exception &e) {
            KVCM_LOG_ERROR("async migration prepare threw exception: %s", e.what());
        } catch (...) { KVCM_LOG_ERROR("async migration prepare threw unknown exception"); }
        self->FinishAsyncMigrationPrepare(key);
    };
    auto cancel_task = [weak_self, key]() {
        if (auto self = weak_self.lock()) {
            self->FinishAsyncMigrationPrepare(key);
        }
    };
    if (!schedule_plan_executor_->SubmitTask(ScheduleTaskClass::kMigrationPrepare,
                                             std::move(execute_task),
                                             std::chrono::microseconds::zero(),
                                             std::move(cancel_task))) {
        FinishAsyncMigrationPrepare(key);
        return false;
    }
    return true;
}

void MigrationManager::RunAsyncMigrationPrepare(AsyncMigrationPrepareJob job, std::uint64_t generation) {
    const auto prepare_and_dispatch = [&]() -> DispatchBatchResult {
        if (!accepting_copy_submissions_.load(std::memory_order_acquire) ||
            async_prepare_generation_.load(std::memory_order_acquire) != generation || !registry_manager_ ||
            !meta_indexer_manager_ || !data_storage_manager_) {
            return {};
        }

        // Prepare 会读取 Registry/Meta/DataStorage，必须与 leader cleanup 使用同一个 lifecycle
        // barrier。Stop 先关闭 gate，再取 unique lock：已运行 Job 收口后才能清理依赖；排队 Job
        // 随后取得 shared lock 时会因 generation/gate 变化直接退出，不再访问已清理资源。
        std::shared_lock<std::shared_mutex> lifecycle_lock(copy_submission_lifecycle_mutex_);
        if (!accepting_copy_submissions_.load(std::memory_order_acquire) ||
            async_prepare_generation_.load(std::memory_order_acquire) != generation) {
            return {};
        }

        auto request_context = std::make_shared<RequestContext>(job.trace_id);
        const auto instance_info = registry_manager_->GetInstanceInfo(request_context.get(), job.instance_id);
        if (!instance_info || instance_info->instance_group_name() != job.instance_group_name) {
            return {};
        }
        const auto cache_config = registry_manager_->GetCacheConfig(job.instance_group_name);
        if (!cache_config) {
            return {};
        }

        std::shared_ptr<MigrationStrategy> current_strategy;
        for (const auto &strategy : cache_config->migration_strategies()) {
            if (strategy && strategy->source_storage_name() == job.source_storage_name &&
                strategy->target_storage_name() == job.target_storage_name) {
                if (current_strategy) {
                    // Config validation rejects duplicate routes. Fail closed here as well because recovered
                    // legacy data or direct in-process mutation may bypass the normal validation entrypoint.
                    KVCM_LOG_ERROR("duplicate migration route in instance group [%s]: [%s] -> [%s]",
                                   job.instance_group_name.c_str(),
                                   job.source_storage_name.c_str(),
                                   job.target_storage_name.c_str());
                    return {};
                }
                current_strategy = strategy;
            }
        }
        if (!current_strategy) {
            return {};
        }
        const bool copy_enabled = current_strategy->methods().copy().enabled();
        const bool mark_enabled = current_strategy->methods().mark().enabled();
        if ((!copy_enabled && !mark_enabled) ||
            data_storage_manager_->GetDataStorageBackend(job.source_storage_name) == nullptr ||
            data_storage_manager_->GetDataStorageBackend(job.target_storage_name) == nullptr) {
            return {};
        }

        const auto indexer = meta_indexer_manager_->GetMetaIndexer(job.instance_id);
        if (!indexer) {
            return {};
        }
        MetaSearcher meta_searcher(indexer);
        std::vector<CacheLocationMap> loc_maps;
        BlockMask empty_mask;
        if (meta_searcher.BatchGetLocation(request_context.get(), job.block_keys, empty_mask, loc_maps) != EC_OK ||
            loc_maps.size() != job.block_keys.size()) {
            return {};
        }

        for (std::size_t block_idx = 0; block_idx < loc_maps.size(); ++block_idx) {
            if (job.pending_location_ids_by_block[block_idx].empty()) {
                continue;
            }
            const std::unordered_set<std::string> pending_ids(job.pending_location_ids_by_block[block_idx].begin(),
                                                              job.pending_location_ids_by_block[block_idx].end());
            auto &loc_map = loc_maps[block_idx];
            for (auto it = loc_map.begin(); it != loc_map.end();) {
                if (pending_ids.count(it->first) > 0) {
                    it = loc_map.erase(it);
                } else {
                    ++it;
                }
            }
        }

        const auto configured_copy_concurrency = cache_config->migration_copy_max_concurrency();
        const std::size_t max_concurrent_copy =
            configured_copy_concurrency > 0 ? static_cast<std::size_t>(configured_copy_concurrency) : 0;
        DispatchBatchParams params;
        params.do_copy = copy_enabled;
        params.do_mark = mark_enabled;
        params.copy_limit = CopyConcurrencyLimit{job.instance_group_name, max_concurrent_copy};
        const auto active_copy_count = ActiveTaskCountForGroup(job.instance_group_name);
        params.max_copy_slots = max_concurrent_copy > active_copy_count ? max_concurrent_copy - active_copy_count : 0;
        params.retention = current_strategy->retention();
        params.copy_execution_mode = cache_config->migration_copy_execution_mode();
        params.copy_max_inflight_bytes = cache_config->migration_copy_max_inflight_bytes();
        params.copy_max_quarantine_operations = cache_config->migration_copy_max_quarantine_operations();
        params.copy_max_quarantine_bytes = cache_config->migration_copy_max_quarantine_bytes();
        params.async_copy_options.operation_deadline_ms = cache_config->migration_copy_operation_deadline_ms();
        params.async_copy_options.initial_poll_interval_ms = cache_config->migration_copy_poll_initial_interval_ms();
        params.async_copy_options.max_poll_interval_ms = cache_config->migration_copy_poll_max_interval_ms();
        params.async_copy_options.connect_timeout_ms = cache_config->migration_copy_connect_timeout_ms();
        params.async_copy_options.submit_timeout_ms = cache_config->migration_copy_submit_timeout_ms();
        params.async_copy_options.query_timeout_ms = cache_config->migration_copy_query_timeout_ms();
        params.mark_timeout_ms = current_strategy->methods().mark().timeout_ms();
        params.dedup_marks = true;
        return DispatchMigrationBatchWithLifecycleLockHeld(job.trace_id,
                                                            job.instance_id,
                                                            job.source_storage_name,
                                                            job.target_storage_name,
                                                            job.block_keys,
                                                            loc_maps,
                                                            params);
    };

    const auto dispatch = prepare_and_dispatch();
    // Callback is deliberately outside the lifecycle shared lock; it must not extend Stop latency
    // or create a shared-to-exclusive lock upgrade path.
    if (!job.on_dispatched) {
        return;
    }
    try {
        job.on_dispatched(dispatch);
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("async migration dispatch callback threw exception: %s", e.what());
    } catch (...) { KVCM_LOG_ERROR("async migration dispatch callback threw unknown exception"); }
}

MigrationManager::MigrateResult MigrationManager::MigrateCache(RequestContext *request_context,
                                                              const std::string &trace_id,
                                                              const std::string &instance_group_name,
                                                              const std::string &instance_id,
                                                              const std::shared_ptr<MetaIndexer> &meta_indexer,
                                                              const std::string &src_name,
                                                              const std::string &dst_name,
                                                              bool do_copy,
                                                              bool do_mark,
                                                              const std::vector<int64_t> &explicit_block_keys,
                                                              int64_t sample_count,
                                                              std::size_t copy_max_concurrency,
                                                              int64_t mark_timeout_ms) {
    MigrateResult result;

    // meta_indexer 由 facade 保证非空（instance 存在性已前置校验）；防御性再查一次。
    if (meta_indexer == nullptr) {
        result.ec = EC_INSTANCE_NOT_EXIST;
        result.message = "instance not found: " + instance_id;
        return result;
    }

    // 1. 选候选 block_keys：显式 block_keys 优先；否则按 rule 采样 + 过滤。
    auto [select_ec, candidate_keys] =
        SelectMigrationCandidateKeys(request_context, trace_id, explicit_block_keys, sample_count, meta_indexer);
    if (select_ec != EC_OK) {
        result.ec = select_ec;
        result.message = "select migration candidate keys failed";
        return result;
    }

    const int64_t total = static_cast<int64_t>(candidate_keys.size());
    if (candidate_keys.empty()) {
        result.ec = EC_OK;
        result.accepted = 0;
        result.rejected = 0;
        result.message = "no candidate to migrate";
        return result;
    }

    // 2. 取各 block 的 location，准入过滤（在源 storage 上、SERVING、未在迁移中、目标无副本）。
    MetaSearcher meta_searcher(meta_indexer);
    std::vector<CacheLocationMap> loc_maps;
    BlockMask empty_mask;
    if (const auto ec = meta_searcher.BatchGetLocation(request_context, candidate_keys, empty_mask, loc_maps);
        ec != EC_OK || loc_maps.size() != candidate_keys.size()) {
        result.ec = EC_ERROR;
        result.message = "get cache location failed";
        return result;
    }

    // 准入、分发和 fallback 委派共享 DispatchMigrationBatch。
    DispatchBatchParams params;
    params.do_copy = do_copy;
    params.do_mark = do_mark;
    params.copy_limit = CopyConcurrencyLimit{instance_group_name, copy_max_concurrency};
    params.mark_timeout_ms = mark_timeout_ms;
    if (registry_manager_) {
        if (const auto cache_config = registry_manager_->GetCacheConfig(instance_group_name); cache_config) {
            params.copy_execution_mode = cache_config->migration_copy_execution_mode();
            params.copy_max_inflight_bytes = cache_config->migration_copy_max_inflight_bytes();
            params.copy_max_quarantine_operations = cache_config->migration_copy_max_quarantine_operations();
            params.copy_max_quarantine_bytes = cache_config->migration_copy_max_quarantine_bytes();
            params.async_copy_options.operation_deadline_ms = cache_config->migration_copy_operation_deadline_ms();
            params.async_copy_options.initial_poll_interval_ms =
                cache_config->migration_copy_poll_initial_interval_ms();
            params.async_copy_options.max_poll_interval_ms = cache_config->migration_copy_poll_max_interval_ms();
            params.async_copy_options.connect_timeout_ms = cache_config->migration_copy_connect_timeout_ms();
            params.async_copy_options.submit_timeout_ms = cache_config->migration_copy_submit_timeout_ms();
            params.async_copy_options.query_timeout_ms = cache_config->migration_copy_query_timeout_ms();
        }
    }
    const auto dispatch =
        DispatchMigrationBatch(trace_id, instance_id, src_name, dst_name, candidate_keys, loc_maps, params);

    result.ec = EC_OK;
    result.accepted = dispatch.copy_submitted + dispatch.mark_submitted;
    result.rejected = total - result.accepted;
    result.message = "migrate cache dispatched";
    return result;
}

MigrationManager::DispatchBatchResult
MigrationManager::DispatchMigrationBatch(const std::string &trace_id,
    const std::string &instance_id,
    const std::string &src_name,
    const std::string &dst_name,
    const std::vector<int64_t> &batch,
    const std::vector<CacheLocationMap> &loc_maps,
    const DispatchBatchParams &params) {
    DispatchBatchResult result;
    if (batch.empty() || loc_maps.size() != batch.size()) {
        return result;
    }
    if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
        return result;
    }
    // Copy 与 Mark 必须位于同一个 leader-lifecycle barrier 内。否则 Stop 可能在异步 Job
    // 完成 fresh read 后关闭 Copy gate，但 Job 仍继续写入 Mark。
    std::shared_lock<std::shared_mutex> lifecycle_lock(copy_submission_lifecycle_mutex_);
    if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
        return result;
    }
    return DispatchMigrationBatchWithLifecycleLockHeld(
        trace_id, instance_id, src_name, dst_name, batch, loc_maps, params);
}

MigrationManager::DispatchBatchResult
MigrationManager::DispatchMigrationBatchWithLifecycleLockHeld(const std::string &trace_id,
    const std::string &instance_id,
    const std::string &src_name,
    const std::string &dst_name,
    const std::vector<int64_t> &batch,
    const std::vector<CacheLocationMap> &loc_maps,
    const DispatchBatchParams &params) {
    DispatchBatchResult result;
    if (batch.empty() || loc_maps.size() != batch.size()) {
        return result;
    }

    // 10a: mark 去重用 batch 查询替代逐 block IsMarkedForTieredWrite（N 次 meta 往返 → 1 次）。
    std::unordered_set<int64_t> already_marked;
    std::unordered_set<int64_t> mark_query_failed;
    if (params.do_mark && params.dedup_marks) {
        std::vector<MarkQueryResult> mark_results;
        const auto mark_query_ec = BatchGetTieredWriteTargets(instance_id, batch, mark_results);
        if (mark_query_ec != EC_OK) {
            KVCM_LOG_WARN("[%s] mark dedup query partially failed for instance %s, ec %d; failed keys will retry",
                          trace_id.c_str(),
                          instance_id.c_str(),
                          mark_query_ec);
        }
        for (std::size_t i = 0; i < batch.size() && i < mark_results.size(); ++i) {
            if (mark_results[i].HasValidMark()) {
                already_marked.insert(batch[i]);
            } else if (mark_results[i].IsReadError()) {
                // 无法确认已有 mark 时不覆盖未知状态，留待下一轮 reclaimer 重试。
                mark_query_failed.insert(batch[i]);
            }
        }
    }

    // 准入过滤 + 收集 copy / mark 候选
    std::vector<MigrationRequest> copy_reqs;
    std::vector<int64_t> copy_block_keys;
    std::vector<int64_t> mark_keys;
    for (std::size_t i = 0; i < batch.size(); ++i) {
        const int64_t block_key = batch[i];
        const auto admission = CheckCopyAdmission(instance_id, block_key, loc_maps[i], src_name, dst_name);
        if (admission.status != CopyAdmissionStatus::kAccept || admission.src_location == nullptr) {
            continue;
        }
        if (params.do_copy && copy_reqs.size() < params.max_copy_slots) {
            MigrationRequest req;
            req.instance_group_name = params.copy_limit.instance_group_name;
            req.instance_id = instance_id;
            req.block_key = block_key;
            req.src_location_id = admission.src_location->id();
            req.src_storage_name = src_name;
            req.dst_storage_name = dst_name;
            req.retention = params.retention;
            req.copy_execution_mode = params.copy_execution_mode;
            req.copy_max_inflight_bytes = params.copy_max_inflight_bytes;
            req.copy_max_quarantine_operations = params.copy_max_quarantine_operations;
            req.copy_max_quarantine_bytes = params.copy_max_quarantine_bytes;
            req.async_copy_options = params.async_copy_options;
            req.src_specs = admission.src_location->location_specs();
            req.src_create_time = admission.src_location->create_time();
            copy_block_keys.push_back(block_key);
            copy_reqs.push_back(std::move(req));
            continue; // Copy 优先：已进入 copy 的 block 不再重复 mark。
        }
        if (params.do_mark && already_marked.count(block_key) == 0 && mark_query_failed.count(block_key) == 0) {
            mark_keys.push_back(block_key);
        }
    }

    // 分发
    if (!copy_reqs.empty()) {
        const auto results = BatchSubmit(trace_id, std::move(copy_reqs), params.copy_limit, true);
        for (std::size_t i = 0; i < results.size(); ++i) {
            if (results[i] == EC_OK) {
                ++result.copy_submitted;
            } else {
                ++result.copy_failed;
                if (params.do_mark && i < copy_block_keys.size()) {
                    mark_keys.push_back(copy_block_keys[i]); // copy 失败时 fallback 到 mark
                }
            }
        }
    }
    if (params.do_mark && !mark_keys.empty()) {
        const auto mark_ec = MarkForTieredWrite(instance_id, mark_keys, dst_name, params.mark_timeout_ms);
        if (mark_ec == EC_OK) {
            result.mark_submitted = static_cast<int64_t>(mark_keys.size());
        }
    }
    return result;
}

MigrationManager::MigrationStats MigrationManager::GetStats() const {
    MigrationStats stats;
    stats.copy_submitted = stat_copy_submitted_.load(std::memory_order_relaxed);
    stats.copy_completed = stat_copy_completed_.load(std::memory_order_relaxed);
    stats.copy_failed = stat_copy_failed_.load(std::memory_order_relaxed);
    stats.copy_cancelled = stat_copy_cancelled_.load(std::memory_order_relaxed);
    stats.source_failures_recorded = stat_source_failures_recorded_.load(std::memory_order_relaxed);
    stats.source_retries_suppressed = stat_source_retries_suppressed_.load(std::memory_order_relaxed);
    stats.source_switches = stat_source_switches_.load(std::memory_order_relaxed);
    stats.no_usable_source = stat_no_usable_source_.load(std::memory_order_relaxed);
    stats.async_copy_unknown = stat_async_copy_unknown_.load(std::memory_order_relaxed);
    stats.marks_added = stat_marks_added_.load(std::memory_order_relaxed);
    stats.marks_cleared = stat_marks_cleared_.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        stats.active_copy_tasks = ActiveTaskCountUnsafe();
    }
    {
        std::lock_guard<std::mutex> lock(copy_source_failure_mutex_);
        stats.source_failure_entries = copy_source_failures_.size();
    }
    {
        std::lock_guard<std::mutex> lock(async_copy_usage_mutex_);
        for (const auto &[_, usage] : async_copy_usage_by_group_) {
            stats.async_copy_inflight_operations += usage.inflight_operations;
            stats.async_copy_inflight_bytes += usage.inflight_bytes;
            stats.async_copy_quarantine_operations += usage.quarantine_operations;
            stats.async_copy_quarantine_bytes += usage.quarantine_bytes;
        }
    }
    // 持久化方案下无内存表，active_marks 为 best-effort 近似（added - cleared）。
    const uint64_t added = stats.marks_added;
    const uint64_t cleared = stats.marks_cleared;
    stats.active_marks = added > cleared ? static_cast<size_t>(added - cleared) : 0;
    return stats;
}

} // namespace kv_cache_manager
