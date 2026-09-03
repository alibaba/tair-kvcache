#include "kv_cache_manager/manager/migration_manager.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <iterator>
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
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/event/spec_events/migration_event.h"
#include "kv_cache_manager/manager/data_storage_selector.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/manager/migration_admission_internal.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"

namespace kv_cache_manager {

namespace {
constexpr auto kMonitorIdleSleep = std::chrono::milliseconds(50);
constexpr auto kFutureWaitTime = std::chrono::microseconds(200);

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

// mark property 解析结果。target 为空表示无标记或已清；target 非空时 deadline
// 必须是合法正整数，malformed 表示缺失、非数字、越界或非正值；expired 表示已过期待清理。
struct MarkInfo {
    std::string target;
    int64_t deadline_ms = 0;
    bool malformed = false;
    bool expired = false;
};

struct SpecByteSummary {
    std::uint64_t bytes = 0;
    std::uint64_t unknown_specs = 0;
};

SpecByteSummary SummarizeSpecBytes(const std::vector<LocationSpec> &specs) noexcept {
    SpecByteSummary summary;
    for (const auto &spec : specs) {
        const DataStorageUri uri(spec.uri());
        const std::string size_value = uri.Valid() ? uri.GetParam("size") : std::string{};
        std::uint64_t size = 0;
        const auto [ptr, ec] =
            std::from_chars(size_value.data(), size_value.data() + size_value.size(), size);
        if (size_value.empty() || ec != std::errc{} || ptr != size_value.data() + size_value.size() ||
            size > std::numeric_limits<std::uint64_t>::max() - summary.bytes) {
            ++summary.unknown_specs;
            continue;
        }
        summary.bytes += size;
    }
    return summary;
}

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
std::unordered_set<std::string> CollectCoveredSpecNames(
    const CacheLocationMap &loc_map,
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

std::vector<LocationSpec> FindFirstUncoveredSourceSpecs(const CacheLocationMap &loc_map,
                                                        const std::string &src_storage_name,
                                                        const std::string &dst_storage_name) {
    const auto covered = CollectCoveredSpecNames(
        loc_map, dst_storage_name, {CacheLocationStatus::CLS_SERVING, CacheLocationStatus::CLS_WRITING});
    for (const auto *source :
         FindLocationsOnStorage(loc_map, src_storage_name, {CacheLocationStatus::CLS_SERVING})) {
        std::vector<LocationSpec> missing_specs;
        missing_specs.reserve(source->location_specs().size());
        std::copy_if(source->location_specs().begin(),
                     source->location_specs().end(),
                     std::back_inserter(missing_specs),
                     [&covered](const LocationSpec &spec) { return covered.count(spec.name()) == 0; });
        if (!missing_specs.empty()) {
            return missing_specs;
        }
    }
    return {};
}

void AddOutcome(MigrationOutcomeCounts &outcomes,
                MigrationOutcomeStage stage,
                MigrationOutcomeClass outcome_class,
                MigrationOutcomeReason reason,
                int64_t count,
                bool terminal) {
    if (count <= 0) {
        return;
    }
    const auto existing = std::find_if(outcomes.begin(), outcomes.end(), [&](const MigrationOutcomeCount &item) {
        return item.stage == stage && item.outcome_class == outcome_class && item.reason == reason &&
               item.terminal == terminal;
    });
    if (existing != outcomes.end()) {
        existing->count += count;
        return;
    }
    outcomes.push_back(MigrationOutcomeCount{stage, outcome_class, reason, count, terminal});
}

MigrationOutcomeReason ToOutcomeReason(MigrationAdmissionReason reason) noexcept {
    switch (reason) {
    case MigrationAdmissionReason::kSatisfied:
        return MigrationOutcomeReason::kValueAccepted;
    case MigrationAdmissionReason::kNotRecent:
        return MigrationOutcomeReason::kNotRecent;
    case MigrationAdmissionReason::kFeatureMissing:
        return MigrationOutcomeReason::kFeatureMissing;
    case MigrationAdmissionReason::kFeatureInvalid:
        return MigrationOutcomeReason::kFeatureInvalid;
    case MigrationAdmissionReason::kFeatureUnsupported:
        return MigrationOutcomeReason::kFeatureUnsupported;
    case MigrationAdmissionReason::kFeatureReadError:
        return MigrationOutcomeReason::kFeatureReadError;
    case MigrationAdmissionReason::kInsufficientBusinessAccessCount:
        return MigrationOutcomeReason::kFeatureInvalid;
    }
    return MigrationOutcomeReason::kUnspecified;
}

MigrationOutcomeReason ToOutcomeReason(MigrationManager::CopyAdmissionStatus status) noexcept {
    switch (status) {
    case MigrationManager::CopyAdmissionStatus::kAlreadyMigrating:
        return MigrationOutcomeReason::kAlreadyMigrating;
    case MigrationManager::CopyAdmissionStatus::kTargetServingExists:
    case MigrationManager::CopyAdmissionStatus::kTargetWritingExists:
        return MigrationOutcomeReason::kTargetAlreadyCovered;
    case MigrationManager::CopyAdmissionStatus::kSourceServingNotFound:
        return MigrationOutcomeReason::kSourceNotFound;
    case MigrationManager::CopyAdmissionStatus::kAccept:
        break;
    }
    return MigrationOutcomeReason::kUnspecified;
}

MigrationOutcomeReason CopySubmitOutcomeReason(ErrorCode ec) noexcept {
    switch (ec) {
    case EC_NOENT:
    case EC_MISMATCH:
        return MigrationOutcomeReason::kSourceRecheckFailed;
    case EC_EXIST:
        return MigrationOutcomeReason::kAlreadyMigrating;
    case EC_OUT_OF_LIMIT:
        return MigrationOutcomeReason::kCopySlotExhausted;
    case EC_NOSPC:
        return MigrationOutcomeReason::kTargetRejected;
    default:
        return MigrationOutcomeReason::kCopySubmitFailed;
    }
}

const char *TriggerName(MigrationManager::DispatchBatchParams::Trigger trigger) noexcept {
    return trigger == MigrationManager::DispatchBatchParams::Trigger::kReclaimer ? "reclaimer" : "admin";
}

const char *AdmissionModeName(MigrationAdmissionMode mode) noexcept {
    switch (mode) {
    case MigrationAdmissionMode::DISABLED:
        return "disabled";
    case MigrationAdmissionMode::SHADOW:
        return "shadow";
    case MigrationAdmissionMode::ENFORCE:
        return "enforce";
    }
    return "unknown";
}

const char *OutcomeStageName(MigrationOutcomeStage stage) noexcept {
    switch (stage) {
    case MigrationOutcomeStage::kSnapshot:
        return "snapshot";
    case MigrationOutcomeStage::kValue:
        return "value";
    case MigrationOutcomeStage::kExecution:
        return "execution";
    case MigrationOutcomeStage::kCopy:
        return "copy";
    case MigrationOutcomeStage::kMark:
        return "mark";
    }
    return "unknown";
}

const char *OutcomeClassName(MigrationOutcomeClass outcome_class) noexcept {
    switch (outcome_class) {
    case MigrationOutcomeClass::kAccepted:
        return "accepted";
    case MigrationOutcomeClass::kRejected:
        return "rejected";
    case MigrationOutcomeClass::kNoopAlreadySatisfied:
        return "noop_already_satisfied";
    case MigrationOutcomeClass::kFailed:
        return "failed";
    }
    return "unknown";
}

const char *OutcomeReasonName(MigrationOutcomeReason reason) noexcept {
    switch (reason) {
    case MigrationOutcomeReason::kUnspecified:
        return "unspecified";
    case MigrationOutcomeReason::kNotRecent:
        return "not_recent";
    case MigrationOutcomeReason::kFeatureMissing:
        return "feature_missing";
    case MigrationOutcomeReason::kFeatureInvalid:
        return "feature_invalid";
    case MigrationOutcomeReason::kFeatureUnsupported:
        return "feature_unsupported";
    case MigrationOutcomeReason::kFeatureReadError:
        return "feature_read_error";
    case MigrationOutcomeReason::kRouteNotReady:
        return "route_not_ready";
    case MigrationOutcomeReason::kLocationReadError:
        return "location_read_error";
    case MigrationOutcomeReason::kSnapshotShapeError:
        return "snapshot_shape_error";
    case MigrationOutcomeReason::kSourceNotFound:
        return "source_not_found";
    case MigrationOutcomeReason::kTargetAlreadyCovered:
        return "target_already_covered";
    case MigrationOutcomeReason::kAlreadyMigrating:
        return "already_migrating";
    case MigrationOutcomeReason::kTargetRejected:
        return "target_rejected";
    case MigrationOutcomeReason::kSourceRecheckFailed:
        return "source_recheck_failed";
    case MigrationOutcomeReason::kCopySubmitted:
        return "copy_submitted";
    case MigrationOutcomeReason::kCopySubmitFailed:
        return "copy_submit_failed";
    case MigrationOutcomeReason::kMarkInserted:
        return "mark_inserted";
    case MigrationOutcomeReason::kMarkAlreadySameTarget:
        return "mark_already_same_target";
    case MigrationOutcomeReason::kMarkConflictDifferentTarget:
        return "mark_conflict_different_target";
    case MigrationOutcomeReason::kMarkMalformed:
        return "mark_malformed";
    case MigrationOutcomeReason::kBlockNotFound:
        return "block_not_found";
    case MigrationOutcomeReason::kMarkReadError:
        return "mark_read_error";
    case MigrationOutcomeReason::kMarkWriteError:
        return "mark_write_error";
    case MigrationOutcomeReason::kPolicyContractError:
        return "policy_contract_error";
    case MigrationOutcomeReason::kBudgetExhausted:
        return "budget_exhausted";
    case MigrationOutcomeReason::kValueAccepted:
        return "value_accepted";
    case MigrationOutcomeReason::kNoExecutionMethod:
        return "no_execution_method";
    case MigrationOutcomeReason::kCopySlotExhausted:
        return "copy_slot_exhausted";
    case MigrationOutcomeReason::kDispatchNotAvailable:
        return "dispatch_not_available";
    }
    return "unknown";
}

const char *FeatureStatusName(MigrationOutcomeReason reason) noexcept {
    switch (reason) {
    case MigrationOutcomeReason::kFeatureMissing:
        return "missing";
    case MigrationOutcomeReason::kFeatureInvalid:
        return "invalid";
    case MigrationOutcomeReason::kFeatureUnsupported:
        return "unsupported";
    case MigrationOutcomeReason::kFeatureReadError:
        return "read_error";
    case MigrationOutcomeReason::kNotRecent:
    case MigrationOutcomeReason::kValueAccepted:
        return "available";
    default:
        return nullptr;
    }
}
} // namespace

// Mark 持久化属性名（block 级 property）。带 inner 前缀避免与业务属性冲突。
const std::string MigrationManager::PROPERTY_TIERED_WRITE_TARGET = "__mig_tier_target__";
const std::string MigrationManager::PROPERTY_TIERED_WRITE_DEADLINE_MS = "__mig_tier_deadline_ms__";

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
        m_marks_active_ = metrics_registry_->GetGauge("migration.marks_active");
        m_marks_consumed_total_ = metrics_registry_->GetCounter("migration.marks_consumed_total");
        m_marks_expired_total_ = metrics_registry_->GetCounter("migration.marks_expired_total");
        m_mark_query_errors_total_ = metrics_registry_->GetCounter("migration.mark_query_errors_total");
    }
}

void MigrationManager::AddCounterMetric(const std::string &name,
                                        MetricsTags tags,
                                        std::uint64_t value) const noexcept {
    if (!metrics_enabled_ || metrics_registry_ == nullptr || value == 0) {
        return;
    }
    try {
        metrics_registry_->GetCounter(name, tags) += value;
    } catch (const std::exception &e) {
        KVCM_LOG_WARN("record migration metric %s failed: %s", name.c_str(), e.what());
    } catch (...) { KVCM_LOG_WARN("record migration metric %s failed with unknown error", name.c_str()); }
}

void MigrationManager::SetGaugeMetric(const std::string &name, MetricsTags tags, double value) const noexcept {
    if (!metrics_enabled_ || metrics_registry_ == nullptr) {
        return;
    }
    try {
        metrics_registry_->GetGauge(name, tags) = value;
    } catch (const std::exception &e) {
        KVCM_LOG_WARN("record migration gauge %s failed: %s", name.c_str(), e.what());
    } catch (...) { KVCM_LOG_WARN("record migration gauge %s failed with unknown error", name.c_str()); }
}

void MigrationManager::ObserveAdmissionAccessAge(const std::string &src_name,
                                                 const std::string &dst_name,
                                                 const DispatchBatchParams &params,
                                                 std::int64_t age_us) const noexcept {
    if (!metrics_enabled_ || metrics_registry_ == nullptr || age_us < 0) {
        return;
    }
    try {
        static constexpr std::array<std::int64_t, 8> kBucketSeconds = {
            1, 10, 60, 5 * 60, 60 * 60, 6 * 60 * 60, 24 * 60 * 60, 7 * 24 * 60 * 60};
        static constexpr const char *kFamily = "migration_admission_access_age_seconds";
        static constexpr const char *kBucket = "migration_admission_access_age_seconds_bucket";
        static constexpr const char *kSum = "migration_admission_access_age_seconds_sum";
        static constexpr const char *kCount = "migration_admission_access_age_seconds_count";
        metrics_registry_->RegisterHistogramFamily(kFamily);
        metrics_registry_->MapMetricToFamily(kBucket, kFamily);
        metrics_registry_->MapMetricToFamily(kSum, kFamily);
        metrics_registry_->MapMetricToFamily(kCount, kFamily);

        MetricsTags tags{{"trigger", TriggerName(params.trigger)}, {"src", src_name}, {"dst", dst_name}};
        for (const auto boundary_seconds : kBucketSeconds) {
            if (age_us <= boundary_seconds * 1000 * 1000) {
                auto bucket_tags = tags;
                bucket_tags["le"] = std::to_string(boundary_seconds);
                ++metrics_registry_->GetCounter(kBucket, bucket_tags);
            }
        }
        auto infinite_tags = tags;
        infinite_tags["le"] = "+Inf";
        ++metrics_registry_->GetCounter(kBucket, infinite_tags);
        metrics_registry_->GetCounter(kSum, tags) += static_cast<std::uint64_t>(age_us);
        ++metrics_registry_->GetCounter(kCount, tags);
    } catch (const std::exception &e) {
        KVCM_LOG_WARN("record migration admission access age failed: %s", e.what());
    } catch (...) { KVCM_LOG_WARN("record migration admission access age failed with unknown error"); }
}

void MigrationManager::ObserveSourceLeaseDuration(const std::string &src_name,
                                                  const std::string &dst_name,
                                                  std::int64_t duration_us) const noexcept {
    if (!metrics_enabled_ || metrics_registry_ == nullptr || duration_us < 0) {
        return;
    }
    try {
        static constexpr std::array<std::int64_t, 7> kBucketSeconds = {1, 10, 60, 5 * 60, 30 * 60, 60 * 60, 6 * 60 * 60};
        static constexpr const char *kFamily = "migration_source_lease_duration_seconds";
        static constexpr const char *kBucket = "migration_source_lease_duration_seconds_bucket";
        static constexpr const char *kSum = "migration_source_lease_duration_seconds_sum";
        static constexpr const char *kCount = "migration_source_lease_duration_seconds_count";
        metrics_registry_->RegisterHistogramFamily(kFamily);
        metrics_registry_->MapMetricToFamily(kBucket, kFamily);
        metrics_registry_->MapMetricToFamily(kSum, kFamily);
        metrics_registry_->MapMetricToFamily(kCount, kFamily);

        MetricsTags tags{{"src", src_name}, {"dst", dst_name}};
        for (const auto boundary_seconds : kBucketSeconds) {
            if (duration_us <= boundary_seconds * 1000 * 1000) {
                auto bucket_tags = tags;
                bucket_tags["le"] = std::to_string(boundary_seconds);
                ++metrics_registry_->GetCounter(kBucket, bucket_tags);
            }
        }
        auto infinite_tags = tags;
        infinite_tags["le"] = "+Inf";
        ++metrics_registry_->GetCounter(kBucket, infinite_tags);
        metrics_registry_->GetCounter(kSum, tags) += static_cast<std::uint64_t>(duration_us);
        ++metrics_registry_->GetCounter(kCount, tags);
    } catch (const std::exception &e) {
        KVCM_LOG_WARN("record migration source lease duration failed: %s", e.what());
    } catch (...) { KVCM_LOG_WARN("record migration source lease duration failed with unknown error"); }
}

MigrationManager::DispatchBatchResult MigrationManager::FinalizeDispatchMetrics(
    const std::string &instance_id,
    const std::string &src_name,
    const std::string &dst_name,
    std::size_t candidate_count,
    const DispatchBatchParams &params,
    DispatchBatchResult result) const {
    MetricsTags route_tags{{"trigger", TriggerName(params.trigger)},
                           {"src", src_name},
                           {"dst", dst_name},
                           {"mode", AdmissionModeName(params.admission.mode())}};
    AddCounterMetric("migration_admission_candidates_total", route_tags, candidate_count);

    auto pool_tags = route_tags;
    pool_tags.erase("trigger");
    pool_tags.erase("mode");
    pool_tags["stage"] = "sampled";
    AddCounterMetric("migration_candidate_pool_total", pool_tags, candidate_count);

    std::int64_t terminal_count = 0;
    std::int64_t dispatched_count = 0;
    for (const auto &outcome : result.outcome_counts) {
        auto outcome_tags = route_tags;
        outcome_tags.erase("mode");
        outcome_tags["stage"] = OutcomeStageName(outcome.stage);
        outcome_tags["class"] = OutcomeClassName(outcome.outcome_class);
        outcome_tags["reason"] = OutcomeReasonName(outcome.reason);
        outcome_tags["terminal"] = outcome.terminal ? "true" : "false";
        AddCounterMetric("migration_dispatch_outcomes_total", std::move(outcome_tags), outcome.count);

        if (outcome.terminal) {
            terminal_count += outcome.count;
            if (outcome.outcome_class == MigrationOutcomeClass::kAccepted) {
                dispatched_count += outcome.count;
            }
        }
        if (outcome.stage == MigrationOutcomeStage::kValue &&
            outcome.reason != MigrationOutcomeReason::kPolicyContractError) {
            auto policy_tags = route_tags;
            policy_tags.erase("mode");
            policy_tags["policy"] = "recent_access";
            policy_tags["reason"] = OutcomeReasonName(outcome.reason);
            if (outcome.outcome_class == MigrationOutcomeClass::kAccepted) {
                policy_tags["verdict"] = "accept";
                AddCounterMetric("migration_admission_accepted_total", route_tags, outcome.count);
            } else {
                policy_tags["verdict"] = outcome.reason == MigrationOutcomeReason::kNotRecent ? "reject" : "unknown";
                auto rejected_tags = route_tags;
                rejected_tags["reason"] = OutcomeReasonName(outcome.reason);
                AddCounterMetric("migration_admission_rejected_total", std::move(rejected_tags), outcome.count);
            }
            AddCounterMetric("migration_admission_policy_evaluations_total", std::move(policy_tags), outcome.count);
        }
        // Feature transport failures are snapshot failures in ENFORCE and
        // projected policy UNKNOWNs in SHADOW. Record feature availability
        // independently of the stage so both shapes share one metric.
        if (const char *feature_status = FeatureStatusName(outcome.reason); feature_status != nullptr) {
            MetricsTags feature_tags{{"src", src_name},
                                     {"dst", dst_name},
                                     {"feature", "last_access_time"},
                                     {"status", feature_status}};
            AddCounterMetric("migration_admission_feature_status_total", std::move(feature_tags), outcome.count);
        }

        const char *read_component = nullptr;
        if (outcome.reason == MigrationOutcomeReason::kLocationReadError) {
            read_component = "location";
        } else if (outcome.reason == MigrationOutcomeReason::kSnapshotShapeError) {
            read_component = "snapshot";
        } else if (outcome.reason == MigrationOutcomeReason::kFeatureReadError) {
            read_component = "property";
        }
        if (read_component != nullptr) {
            MetricsTags read_tags{{"src", src_name},
                                  {"dst", dst_name},
                                  {"component", read_component},
                                  {"reason", OutcomeReasonName(outcome.reason)}};
            AddCounterMetric("migration_admission_read_error_total", std::move(read_tags), outcome.count);
        }
    }

    pool_tags["stage"] = "dispatched";
    AddCounterMetric("migration_candidate_pool_total", std::move(pool_tags), dispatched_count);
    if (terminal_count != static_cast<std::int64_t>(candidate_count)) {
        auto invariant_tags = route_tags;
        invariant_tags["reason"] = "terminal_count_mismatch";
        AddCounterMetric("migration_dispatch_invariant_errors_total", std::move(invariant_tags));
        KVCM_LOG_ERROR("migration dispatch terminal outcome mismatch for instance %s, route %s -> %s: candidates %zu, "
                       "terminal %lld",
                       instance_id.c_str(),
                       src_name.c_str(),
                       dst_name.c_str(),
                       candidate_count,
                       static_cast<long long>(terminal_count));
    }
    return result;
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
        auto request_context = std::make_shared<RequestContext>(
            trace_id.empty() ? "migration_target_admission" : trace_id);
        const auto instance_info = registry_manager_->GetInstanceInfo(request_context.get(), instance_id);
        if (instance_info == nullptr) {
            return EC_INSTANCE_NOT_EXIST;
        }
        group_name = instance_info->instance_group_name();
    }
    if (data_storage_selector_ != nullptr && !group_name.empty()) {
        auto request_context = std::make_shared<RequestContext>(
            trace_id.empty() ? "migration_target_admission" : trace_id);
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

void MigrationManager::Start() {
    std::unique_lock<std::shared_mutex> lifecycle_lock(copy_submission_lifecycle_mutex_);
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return; // already running
    }
    async_prepare_generation_.fetch_add(1, std::memory_order_acq_rel);
    if (meta_indexer_manager_ != nullptr) {
        for (const auto &[unused_instance_id, indexer] : meta_indexer_manager_->GetIndexers()) {
            (void)unused_instance_id;
            if (indexer != nullptr) {
                indexer->ResetMaintenancePropertyReadinessEpoch();
            }
        }
    }
    if (schedule_plan_executor_ != nullptr) {
        schedule_plan_executor_->SetSourceLocationLeaseChecker(
            [this](const std::string &instance_id,
                   int64_t block_key,
                   const std::string &location_id,
                   int64_t create_time) {
                return HasActiveCopySourceLocation(instance_id, block_key, location_id, create_time);
            });
    }
    accepting_copy_submissions_.store(true, std::memory_order_release);
    monitor_thread_ = std::thread([this]() { MonitorLoop(); });
    KVCM_LOG_INFO("MigrationManager started");
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
    // 退出时丢弃活跃任务表：futures 已随 MonitorLoop 退出释放，不再有人驱动 OnTaskSuccess/OnTaskFailed，
    // 残留条目会让 HasMigrationTask/ActiveTaskCount/HasActiveCopyTargetLocation 永远 stale。
    // 半成品的 WRITING 目标 location 由 Reclaimer 走孤儿清理路径回收。
    size_t dropped = 0;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        dropped = ActiveTaskCountUnsafe();
        active_tasks_by_instance_.clear();
        UpdateActiveTasksGauge();
    }
    // Keep the checker installed until every lifecycle-protected submit and
    // monitor completion has stopped and the active lease table is empty.
    // Clearing it earlier opens a shutdown-only window in which an already
    // admitted submit still prepares Copy while delete admission no longer
    // recognizes its source lease.
    if (schedule_plan_executor_ != nullptr) {
        schedule_plan_executor_->SetSourceLocationLeaseChecker({});
    }
    if (dropped > 0) {
        KVCM_LOG_WARN("MigrationManager stopped with %zu active copy task(s) dropped; "
                      "WRITING dst locations will be reclaimed as orphans",
                      dropped);
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
    // 如果调用方已在 admission 阶段取得 src_specs，直接使用并跳过冗余读取；
    // public 单条 Submit 仍允许不带快照，此时也必须通过 NoTouch 接口重读源 location。
    const std::vector<LocationSpec> *src_specs_ptr = nullptr;
    int64_t src_create_time = 0;
    if (!request.src_specs.empty()) {
        src_specs_ptr = &request.src_specs;
        src_create_time = request.src_create_time;
    } else {
        RequestContext source_context(trace_id.empty() ? "migration_prepare" : trace_id);
        CacheLocationMapVector location_maps;
        PropertyMapVector unused_properties;
        const auto source_read = indexer->GetForMaintenance(
            &source_context, {request.block_key}, {}, location_maps, unused_properties);
        const ErrorCode source_ec = source_read.locations.error_codes.size() == 1
                                        ? source_read.locations.error_codes.front()
                                        : EC_MISMATCH;
        if (source_ec != EC_OK || location_maps.size() != 1) {
            KVCM_LOG_WARN("[%s] NoTouch source location read failed for block_key %ld, ec %d",
                          trace_id.c_str(),
                          request.block_key,
                          source_ec);
            return source_ec == EC_NOENT ? EC_NOENT : EC_ERROR;
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
            KVCM_LOG_WARN("[%s] source location %s has no specs",
                          trace_id.c_str(),
                          request.src_location_id.c_str());
            return EC_ERROR;
        }
        request.src_specs = src_location.location_specs();
        request.src_create_time = src_location.create_time();
        src_specs_ptr = &request.src_specs;
        src_create_time = request.src_create_time;
    }

    const auto &src_specs = *src_specs_ptr;

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
    std::uint64_t total_bytes = 0;

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
        std::uint64_t spec_size = 0;
        src_uri.GetParamAs<std::uint64_t>("size", spec_size);
        total_bytes += spec_size;

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
    ErrorCode ec =
        meta_searcher.BatchAddLocation(ctx.get(), {request.block_key}, {dst_location}, add_results, true);
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
    if (schedule_plan_executor_ == nullptr) {
        return EC_ERROR;
    }
    if (const auto target_ec = CheckTargetStorageAdmission(
            trace_id, request.instance_group_name, request.instance_id, request.dst_storage_name);
        target_ec != EC_OK) {
        return target_ec;
    }
    auto source_lease_guard = schedule_plan_executor_->AcquireSourceLocationLeaseReservationGuard();
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

    // Reservation and this exact NoTouch recheck are atomic with respect to
    // every automatic delete admission. Once the guard is released, the
    // active task's (id, create_time) source lease makes later deletes skip it.
    auto indexer = GetIndexer(request.instance_id);
    CacheLocationMapVector source_location_maps;
    PropertyMapVector unused_properties;
    RequestContext source_recheck_context(trace_id.empty() ? "migration_source_recheck" : trace_id);
    const auto source_read = indexer->GetForMaintenance(&source_recheck_context,
                                                        {request.block_key},
                                                        {},
                                                        source_location_maps,
                                                        unused_properties);
    ErrorCode source_check_ec = EC_OK;
    const CacheLocation *source_location = nullptr;
    if (source_read.locations.error_codes.size() != 1 || source_location_maps.size() != 1) {
        source_check_ec = EC_ERROR;
    } else if (source_read.locations.error_codes[0] != EC_OK) {
        source_check_ec = source_read.locations.error_codes[0] == EC_NOENT
                              ? EC_NOENT
                              : source_read.locations.error_codes[0];
    } else {
        const auto source_iter = source_location_maps[0].find(request.src_location_id);
        if (source_iter != source_location_maps[0].end() && source_iter->second != nullptr) {
            source_location = source_iter->second.get();
        } else {
            source_check_ec = EC_NOENT;
        }
    }
    // A prepared request always carries an exact generation, including the
    // legacy value zero. An unprepared public Submit binds to the generation
    // returned by this recheck; zero must not act as a wildcard for prepared
    // requests or a same-id replacement could be accepted accidentally.
    const bool has_prepared_source_snapshot = !request.src_specs.empty();
    const bool source_matches =
        source_location != nullptr && source_location->status() == CLS_SERVING &&
        (!has_prepared_source_snapshot ||
         (request.src_create_time >= 0 && request.src_create_time == source_location->create_time()));
    if (source_check_ec != EC_OK || !source_matches || source_location->location_specs().empty()) {
        std::lock_guard<std::mutex> lock(task_mutex_);
        RemovePreparingTaskLocked(request.instance_id, request.block_key);
        return source_check_ec != EC_OK ? source_check_ec : EC_MISMATCH;
    }
    request.src_create_time = source_location->create_time();
    request.src_specs = source_location->location_specs();
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        const auto instance_iter = active_tasks_by_instance_.find(request.instance_id);
        if (instance_iter == active_tasks_by_instance_.end()) {
            return EC_ERROR;
        }
        const auto task_iter = instance_iter->second.find(request.block_key);
        if (task_iter == instance_iter->second.end() || task_iter->second.state != CopyTaskState::kPreparing) {
            RemovePreparingTaskLocked(request.instance_id, request.block_key);
            return EC_ERROR;
        }
        task_iter->second.src_create_time = request.src_create_time;
    }
    source_lease_guard.unlock();
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
        pending_copies_.push_back(PendingCopy{ctx.instance_id, ctx.block_key, std::move(future)});
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
                                           !request.src_specs.empty() && request.src_create_time >= 0;
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
    if (schedule_plan_executor_ == nullptr) {
        for (auto &item : items) {
            item.MarkFailed(EC_ERROR);
        }
        return collect_results();
    }
    auto source_lease_guard = schedule_plan_executor_->AcquireSourceLocationLeaseReservationGuard();

    // ---- phase 0: group 级 Copy 硬限流 + per-block dedup + draining gate + preparing reservation ----
    // 所有 eligible item 必须在释放短准入锁、执行任何 batch Create/AddLocation 前进入
    // active 表。copy_submission_mutex_ 只串行本 phase 的 gate；后续 backend/meta I/O 可跨 instance 并行。
    // MarkFailed 只收拢 result/eligible。reservation 必须等 URI/Location rollback 完成后再释放，
    // 避免失败 item 尚在清理目标资源时，同 block 的新 Submit 已经进入并与旧清理相互覆盖。
    auto release_preparing = [&](BatchCopyItem &item) {
        if (!item.reservation_active) {
            return;
        }
        std::lock_guard<std::mutex> lock(task_mutex_);
        RemovePreparingTaskLocked(item.request.instance_id, item.request.block_key);
        item.reservation_active = false;
    };
    auto release_all_preparing = [&]() {
        std::lock_guard<std::mutex> lock(task_mutex_);
        for (auto &item : items) {
            if (item.reservation_active) {
                RemovePreparingTaskLocked(item.request.instance_id, item.request.block_key);
                item.reservation_active = false;
            }
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

    // ---- phase 0.5: exact source generation recheck, before target allocation ----
    // The reservation guard is still held, so delete admission cannot pass
    // its lease check and CAS between this read and lease publication.
    KeyVector source_recheck_keys;
    std::vector<BatchCopyItem *> source_recheck_items;
    for (auto &item : items) {
        if (item.eligible) {
            source_recheck_keys.push_back(item.request.block_key);
            source_recheck_items.push_back(&item);
        }
    }
    if (!source_recheck_keys.empty()) {
        CacheLocationMapVector source_location_maps;
        PropertyMapVector unused_properties;
        RequestContext source_recheck_context(trace_id.empty() ? "migration_batch_source_recheck" : trace_id);
        const auto source_read = indexer->GetForMaintenance(&source_recheck_context,
                                                            source_recheck_keys,
                                                            {},
                                                            source_location_maps,
                                                            unused_properties);
        const bool shape_ok = source_read.locations.error_codes.size() == source_recheck_items.size() &&
                              source_location_maps.size() == source_recheck_items.size();
        for (size_t i = 0; i < source_recheck_items.size(); ++i) {
            auto &item = *source_recheck_items[i];
            ErrorCode source_check_ec = EC_OK;
            const CacheLocation *source_location = nullptr;
            if (!shape_ok) {
                source_check_ec = EC_ERROR;
            } else if (source_read.locations.error_codes[i] != EC_OK) {
                source_check_ec = source_read.locations.error_codes[i] == EC_NOENT
                                      ? EC_NOENT
                                      : source_read.locations.error_codes[i];
            } else {
                const auto source_iter = source_location_maps[i].find(item.request.src_location_id);
                if (source_iter != source_location_maps[i].end() && source_iter->second != nullptr) {
                    source_location = source_iter->second.get();
                } else {
                    source_check_ec = EC_NOENT;
                }
            }
            if (source_check_ec != EC_OK || source_location == nullptr ||
                source_location->status() != CLS_SERVING ||
                source_location->create_time() != item.request.src_create_time) {
                item.MarkFailed(source_check_ec != EC_OK ? source_check_ec : EC_MISMATCH);
                release_preparing(item);
            }
        }
    }
    source_lease_guard.unlock();

    auto dst_backend = data_storage_manager_ ? data_storage_manager_->GetDataStorageBackend(first_req.dst_storage_name)
                                             : nullptr;
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
            std::uint64_t spec_size = 0;
            src_uri.GetParamAs<std::uint64_t>("size", spec_size);
            item.total_bytes += spec_size;
            std::string dst_key = req.instance_id + "/" + req.src_specs[s].name() + "/" +
                                  StringUtil::Uint64ToHex(req.block_key);
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
                item.dst_specs.emplace_back(
                    item.request.src_specs[e.spec_idx].name(), create_results[j].second.ToUriString());
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
        ErrorCode add_ec =
            meta_searcher.BatchAddLocation(batch_ctx.get(), add_block_keys, add_locations, add_results, true);
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

            bool task_running = false;
            {
                std::lock_guard<std::mutex> lock(task_mutex_);
                task_running = UpdatePreparingTaskLocked(ctx) && MarkTaskRunningLocked(ctx.instance_id, ctx.block_key);
                if (task_running) {
                    item->reservation_active = false; // 已成为正式 active task，不能再由 preparing guard 清理
                }
            }
            if (!task_running) {
                SubmitTargetLocationDelete(ctx);
                item->MarkFailed(EC_ERROR);
                release_preparing(*item);
                continue;
            }

            CacheLocationCopyRequest copy_req;
            copy_req.instance_id = ctx.instance_id;
            copy_req.block_key = ctx.block_key;
            copy_req.exec_storage_name = ctx.src_storage_name;
            copy_req.src_uris = std::move(item->src_uris);
            copy_req.dst_uris = std::move(item->dst_uris);

            std::future<PlanExecuteResult> future = schedule_plan_executor_->Submit(copy_req);
            if (!future.valid()) {
                SubmitTargetLocationDelete(ctx);
                std::lock_guard<std::mutex> lock(task_mutex_);
                RemoveActiveTaskLocked(ctx.instance_id, ctx.block_key);
                item->MarkFailed(EC_ERROR);
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                pending_copies_.push_back(PendingCopy{ctx.instance_id, ctx.block_key, std::move(future)});
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

    auto rc = std::make_shared<RequestContext>("migration_check_source");
    std::vector<CacheLocationMap> location_maps;
    PropertyMapVector unused_properties;
    const auto read_result =
        indexer->GetForMaintenance(rc.get(), {ctx.block_key}, {}, location_maps, unused_properties);
    if (read_result.locations.error_codes.size() != 1 || read_result.locations.error_codes[0] != EC_OK ||
        location_maps.size() != 1) {
        return false;
    }
    auto iter = location_maps[0].find(ctx.src_location_id);
    if (iter == location_maps[0].end() || iter->second == nullptr) {
        return false;
    }
    // id + status + create_time 三者同时匹配，防止 id 复用导致误判新 location 为原始源。
    return iter->second->status() == CLS_SERVING &&
           iter->second->create_time() == ctx.src_create_time;
}

void MigrationManager::CompleteCopyTaskAsFailed(const CopyTaskContext &ctx, const std::string &fail_reason) {
    SubmitTargetLocationDelete(ctx);
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        RemoveActiveTaskLocked(ctx.instance_id, ctx.block_key);
    }
    stat_copy_failed_.fetch_add(1, std::memory_order_relaxed);
    // 失败路径也填真实 duration（提交到失败的耗时），而非硬编码 0，
    // 便于区分"快速失败"与"跑很久才失败"。
    const int64_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - ctx.submit_time)
                                    .count();
    if (metrics_enabled_) {
        ++m_tasks_completed_failed_;
    }
    if (fail_reason == "source_lost") {
        AddCounterMetric("migration_copy_source_lost_written_bytes_total",
                         {{"src", ctx.src_storage_name}, {"dst", ctx.dst_storage_name}},
                         ctx.total_bytes);
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
}

void MigrationManager::OnTaskSuccess(const std::string &instance_id, int64_t block_key) {
    CopyTaskContext ctx;
    ClaimResult claim;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        claim = ClaimForCompletionLocked(instance_id, block_key, ctx);
    }
    if (claim == ClaimResult::kNotFound || claim == ClaimResult::kBusyCompleting) {
        return; // 已处理过 / 完成认领中
    }
    if (claim == ClaimResult::kBusyPreparing) {
        KVCM_LOG_WARN("ignore premature migration success callback for preparing task: instance %s block_key %ld",
                      instance_id.c_str(),
                      block_key);
        return;
    }

    if (claim == ClaimResult::kWasCancelling) {
        // 用户已取消。copy 虽成功也丢弃：不 promote、不删源，删掉仍为 WRITING 的目标半成品。
        CompleteCancelledTask(ctx);
        return;
    }
    // kClaimedRunning：状态已在锁内置 kCompleting，并发 Cancel 会走 busy 分支。

    if (!IsSourceLocationServing(ctx)) {
        KVCM_LOG_WARN("migration source lost, block_key %ld src_loc %s, discard dst_loc %s",
                      block_key,
                      ctx.src_location_id.c_str(),
                      ctx.dst_location_id.c_str());
        CompleteCopyTaskAsFailed(ctx, "source_lost");
        return;
    }

    // CAS 目标 location WRITING -> SERVING。
    auto indexer = meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(ctx.instance_id) : nullptr;
    bool promoted = false;
    if (indexer) {
        MetaSearcher meta_searcher(indexer);
        auto rc = std::make_shared<RequestContext>("migration_on_success");
        std::vector<std::vector<MetaSearcher::LocationCASTask>> cas_tasks{
            {MetaSearcher::LocationCASTask{ctx.dst_location_id, CLS_WRITING, CLS_SERVING}}};
        std::vector<std::vector<ErrorCode>> cas_results;
        ErrorCode ec =
            meta_searcher.BatchCASLocationStatus(rc.get(), {block_key}, cas_tasks, cas_results, false, true);
        promoted = (ec == EC_OK && !cas_results.empty() && !cas_results[0].empty() && cas_results[0][0] == EC_OK);
    }

    if (!promoted) {
        // 目标提升失败：清理目标半成品，源端保持不动（数据未受损），按失败收尾。
        KVCM_LOG_WARN("migration promote dst location failed, block_key %ld dst_loc %s, treat as failed",
                      block_key,
                      ctx.dst_location_id.c_str());
        CompleteCopyTaskAsFailed(ctx, "promote_failed");
        return;
    }

    // 按提交时快照的 mark target/deadline 做条件清除，避免清掉后续同 block 新 mark。
    if (!ctx.mark_target.empty()) {
        ClearTieredWriteMarkIfMatchInternal(ctx.instance_id, block_key, ctx.mark_target, ctx.mark_deadline_ms);
    }

    // 先释放 active task/source lease，再提交带 create_time generation guard
    // 的源删除。否则删除 admission 会正确地把本任务自己的源删除也挡掉。
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        RemoveActiveTaskLocked(ctx.instance_id, block_key);
    }
    if (ctx.retention == MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE) {
        SubmitSourceLocationDelete(ctx);
    }
    stat_copy_completed_.fetch_add(1, std::memory_order_relaxed);
    const int64_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - ctx.submit_time)
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
            block_key, ctx.src_storage_name, ctx.dst_storage_name, duration_ms, ctx.total_bytes, true, "");
        event_manager_->Publish(ev);
    }
    KVCM_LOG_INFO("migration completed: instance %s block_key %ld dst_loc %s retention %d",
                  ctx.instance_id.c_str(),
                  block_key,
                  ctx.dst_location_id.c_str(),
                  static_cast<int>(ctx.retention));
}

void MigrationManager::OnTaskFailed(const std::string &instance_id, int64_t block_key, ErrorCode reason) {
    CopyTaskContext ctx;
    ClaimResult claim;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        claim = ClaimForCompletionLocked(instance_id, block_key, ctx);
    }
    if (claim == ClaimResult::kNotFound || claim == ClaimResult::kBusyCompleting) {
        return;
    }
    if (claim == ClaimResult::kBusyPreparing) {
        KVCM_LOG_WARN("ignore premature migration failure callback for preparing task: instance %s block_key %ld",
                      instance_id.c_str(),
                      block_key);
        return;
    }
    if (claim == ClaimResult::kWasCancelling) {
        // 已取消，无论 copy 结果如何一律按取消收尾（清 WRITING 目标，记 cancelled 终态）。
        CompleteCancelledTask(ctx);
        return;
    }

    // 失败：CAS 目标 WRITING -> DELETING 并删除目标半成品，源端不动。
    const std::string fail_reason = "copy_failed:" + std::to_string(static_cast<int>(reason));
    CompleteCopyTaskAsFailed(ctx, fail_reason);
    KVCM_LOG_WARN("migration failed: instance %s block_key %ld dst_loc %s reason %d",
                  ctx.instance_id.c_str(),
                  block_key,
                  ctx.dst_location_id.c_str(),
                  reason);
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
    schedule_plan_executor_->SubmitNonBlocking(del_req, ScheduleTaskClass::kMigrationContinuation);
}

void MigrationManager::SubmitSourceLocationDelete(const CopyTaskContext &ctx) {
    if (!schedule_plan_executor_ || ctx.src_location_id.empty()) {
        return;
    }
    CacheLocationDelRequest del_req;
    del_req.instance_id = ctx.instance_id;
    del_req.block_keys = {ctx.block_key};
    del_req.location_ids = {{ctx.src_location_id}};
    del_req.expected_location_create_times = {{ctx.src_create_time}};
    schedule_plan_executor_->SubmitNonBlocking(del_req, ScheduleTaskClass::kMigrationContinuation);
}

void MigrationManager::MonitorLoop() {
    while (running_.load(std::memory_order_relaxed)) {
        ProcessExpiredMarks();

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

        auto status = cell.future.wait_for(kFutureWaitTime);
        if (status != std::future_status::ready) {
            // 尚未完成，放回队尾稍后再查。
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_copies_.push_back(std::move(cell));
            continue;
        }

        PlanExecuteResult result = cell.future.get();
        if (result.status == EC_OK) {
            OnTaskSuccess(cell.instance_id, cell.block_key);
        } else {
            OnTaskFailed(cell.instance_id, cell.block_key, result.status);
        }
    }

    // 退出时排空剩余 future（不再驱动状态流转，仅释放）。
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_copies_.clear();
}

std::shared_ptr<MetaIndexer> MigrationManager::GetIndexer(const std::string &instance_id) const {
    return meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(instance_id) : nullptr;
}

ErrorCode MigrationManager::MarkForTieredWrite(const std::string &instance_id,
                                               const std::vector<int64_t> &block_keys,
                                               const std::string &dst_storage_name,
                                               int64_t timeout_ms) {
    return MarkForTieredWriteDetailed(instance_id, block_keys, dst_storage_name, timeout_ms).ec;
}

MigrationManager::MarkWriteResult MigrationManager::MarkForTieredWriteDetailed(
    const std::string &instance_id,
    const std::vector<int64_t> &block_keys,
    const std::string &dst_storage_name,
    int64_t timeout_ms) {
    MarkWriteResult write_result;
    write_result.outcomes.resize(block_keys.size());
    if (dst_storage_name.empty() || block_keys.empty()) {
        if (dst_storage_name.empty() && !block_keys.empty()) {
            write_result.ec = EC_BADARGS;
            for (auto &outcome : write_result.outcomes) {
                outcome.status = MarkWriteStatus::kWriteError;
                outcome.ec = EC_BADARGS;
            }
        }
        return write_result;
    }
    // 新 Mark 只写入当前可分配的 target；已存在 Mark 遇到 quota 满或 unavailable 时由消费
    // 路径暂时忽略并保留，待容量/可用性恢复后自愈。
    const auto target_ec =
        CheckTargetStorageAdmission("mark_for_tiered_write", "", instance_id, dst_storage_name);
    if (target_ec != EC_OK) {
        KVCM_LOG_WARN("MarkForTieredWrite: target storage [%s] is not writable, skip marking (instance %s, ec %d)",
                      dst_storage_name.c_str(),
                      instance_id.c_str(),
                      target_ec);
        write_result.ec = target_ec;
        for (auto &outcome : write_result.outcomes) {
            outcome.status = MarkWriteStatus::kWriteError;
            outcome.ec = target_ec;
        }
        return write_result;
    }
    if (timeout_ms <= 0) {
        timeout_ms = MigrationMarkMethod::kDefaultTimeoutMs;
    }
    auto indexer = GetIndexer(instance_id);
    if (indexer == nullptr) {
        KVCM_LOG_WARN("MarkForTieredWrite: meta indexer not found for instance %s", instance_id.c_str());
        write_result.ec = EC_INSTANCE_NOT_EXIST;
        for (auto &outcome : write_result.outcomes) {
            outcome.status = MarkWriteStatus::kReadError;
            outcome.ec = EC_INSTANCE_NOT_EXIST;
        }
        return write_result;
    }
    const int64_t now_ms = TimestampUtil::GetCurrentTimeMs();
    if (timeout_ms > std::numeric_limits<int64_t>::max() - now_ms) {
        KVCM_LOG_WARN(
            "MarkForTieredWrite: timeout %ld overflows deadline for instance %s", timeout_ms, instance_id.c_str());
        write_result.ec = EC_BADARGS;
        for (auto &outcome : write_result.outcomes) {
            outcome.status = MarkWriteStatus::kWriteError;
            outcome.ec = EC_BADARGS;
        }
        return write_result;
    }
    const int64_t deadline_ms = now_ms + timeout_ms;
    // 条件判断和 property 写入共享同一个 metadata shard 临界区。预查询只用于减少 I/O，
    // 不承担并发正确性；modifier 只记录计划结果，最终以逐 key backend commit 为准。
    auto modifier = [&dst_storage_name, now_ms, deadline_ms, &write_result](
                        const LocationIdVector & /*existing*/,
                        const PropertyMap &existing_properties,
                        ErrorCode get_ec,
                        size_t idx,
                        PropertyMap &upsert_property_map,
                        CacheLocationMap & /*out_new*/) -> ModifierResult {
        auto &outcome = write_result.outcomes[idx];
        if (get_ec == EC_NOENT) {
            outcome.status = MarkWriteStatus::kBlockNotFound;
            outcome.ec = EC_NOENT;
            return {MA_SKIP, EC_OK};
        }
        if (get_ec != EC_OK) {
            outcome.status = MarkWriteStatus::kReadError;
            outcome.ec = get_ec;
            return {MA_FAIL, get_ec};
        }
        const MarkInfo existing_mark = ParseMarkFromProperties(existing_properties, now_ms);
        if (existing_mark.malformed) {
            outcome.status = MarkWriteStatus::kMalformedExistingMark;
            return {MA_SKIP, EC_OK};
        }
        if (!existing_mark.target.empty() && !existing_mark.expired) {
            outcome.status = existing_mark.target == dst_storage_name
                                 ? MarkWriteStatus::kAlreadySameTarget
                                 : MarkWriteStatus::kConflictDifferentTarget;
            return {MA_SKIP, EC_OK};
        }
        upsert_property_map[PROPERTY_TIERED_WRITE_TARGET] = dst_storage_name;
        upsert_property_map[PROPERTY_TIERED_WRITE_DEADLINE_MS] = std::to_string(deadline_ms);
        outcome.status = MarkWriteStatus::kInserted;
        return {MA_OK, EC_OK};
    };
    RequestContext rc("migration_mark");
    KeyVector keys(block_keys.begin(), block_keys.end());
    auto result = indexer->ReadModifyWriteBlockPropertiesForMaintenance(
        &rc, keys, {PROPERTY_TIERED_WRITE_TARGET, PROPERTY_TIERED_WRITE_DEADLINE_MS}, modifier);

    size_t actual_marked = 0;
    size_t io_error_count = 0;
    for (size_t i = 0; i < write_result.outcomes.size(); ++i) {
        auto &outcome = write_result.outcomes[i];
        const ErrorCode commit_ec = i < result.error_codes.size() ? result.error_codes[i] : EC_MISMATCH;
        if (outcome.status == MarkWriteStatus::kInserted && commit_ec != EC_OK) {
            outcome.status = MarkWriteStatus::kWriteError;
            outcome.ec = commit_ec;
        }
        if (outcome.status == MarkWriteStatus::kReadError && outcome.ec == EC_OK) {
            outcome.ec = commit_ec != EC_OK ? commit_ec : (result.ec == EC_OK ? EC_ERROR : result.ec);
        }
        if (outcome.status == MarkWriteStatus::kInserted) {
            ++actual_marked;
        } else if (outcome.status == MarkWriteStatus::kReadError ||
                   outcome.status == MarkWriteStatus::kWriteError) {
            ++io_error_count;
        }
    }
    write_result.ec = io_error_count == 0
                          ? EC_OK
                          : (io_error_count == write_result.outcomes.size() ? EC_ERROR : EC_PARTIAL_OK);
    if (actual_marked > 0) {
        stat_marks_added_.fetch_add(actual_marked, std::memory_order_relaxed);
    }
    UpdateMarksActiveGauge();
    for (size_t i = 0; i < block_keys.size(); ++i) {
        if (write_result.outcomes[i].status != MarkWriteStatus::kInserted) {
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
    return write_result;
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

// match 检查与写空值在同一个 maintenance block/property RMW 内完成；读取和写入均不 touch。
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

    bool planned_clear = false;
    auto modifier = [&expected_target, expected_deadline_ms, expect_malformed, &planned_clear](
                        const LocationIdVector & /*existing*/,
                        const PropertyMap &existing_properties,
                        ErrorCode get_ec,
                        size_t /*idx*/,
                        PropertyMap &upsert_property_map,
                        CacheLocationMap & /*out_new*/) -> ModifierResult {
        if (get_ec != EC_OK) {
            return {MA_SKIP, EC_OK};
        }
        const auto mark = ParseMarkFromProperties(existing_properties, 0);
        if (mark.target != expected_target ||
            (expect_malformed ? !mark.malformed : mark.deadline_ms != expected_deadline_ms)) {
            return {MA_SKIP, EC_OK};
        }
        upsert_property_map[PROPERTY_TIERED_WRITE_TARGET] = "";
        upsert_property_map[PROPERTY_TIERED_WRITE_DEADLINE_MS] = "";
        planned_clear = true;
        return {MA_OK, EC_OK};
    };
    RequestContext rc("migration_conditional_clear_mark");
    const auto rmw_result = indexer->ReadModifyWriteBlockPropertiesForMaintenance(
        &rc,
        {block_key},
        {PROPERTY_TIERED_WRITE_TARGET, PROPERTY_TIERED_WRITE_DEADLINE_MS},
        modifier);
    const bool cleared = planned_clear && rmw_result.error_codes.size() == 1 && rmw_result.error_codes[0] == EC_OK;
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
    const auto query_result = indexer->GetPropertiesForMaintenance(
        &rc, keys, {PROPERTY_TIERED_WRITE_TARGET, PROPERTY_TIERED_WRITE_DEADLINE_MS}, props);
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

void MigrationManager::CompleteCancelledTask(const CopyTaskContext &ctx) {
    // 取消任务的延迟收尾（monitor 线程，认领到 kWasCancelling 时调用）。
    // 目标此时仍为 WRITING（cancelling 任务从未被 promote）；CAS WRITING->DELETING 删半成品，源端不动。
    SubmitTargetLocationDelete(ctx);
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        RemoveActiveTaskLocked(ctx.instance_id, ctx.block_key);
    }
    stat_copy_cancelled_.fetch_add(1, std::memory_order_relaxed);
    // cancelled 是与 success/failed 对称的终态：在实际清理时计数，保持 submitted==success+failed+cancelled。
    // 被取消任务无论底层 copy 成/败一律记 cancelled（用户意图优先）。
    const int64_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - ctx.submit_time)
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
}

ErrorCode MigrationManager::Cancel(const std::string &instance_id, int64_t block_key) {
    // preparing 取消只标记 kPrepareCancelling，由仍在同步 I/O 中的提交线程在下一安全
    // 边界停止进入 copy、清目标并释放 reservation；running 取消仍等待 future 完成后由 monitor 收尾。
    // 两种 cancelling 期间任务都保留在活跃表，继续挡重复 Submit 并保护 WRITING 目标。
    CancelResult result;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        result = MarkCancellingLocked(instance_id, block_key);
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
        KVCM_LOG_INFO("[cancel] instance %s block_key %ld already completing, cancel too late",
                      instance_id.c_str(),
                      block_key);
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
    ctx.state = CopyTaskState::kPreparing;
    ctx.source_lease_start_time = std::chrono::steady_clock::now();
    // InsertActiveTaskLocked 在同一个 task_mutex_ 临界区内完成存在性检查与插入，避免 check-then-insert 窗口。
    return InsertActiveTaskLocked(std::move(ctx));
}

bool MigrationManager::UpdatePreparingTaskLocked(const CopyTaskContext &ctx) {
    auto instance_iter = active_tasks_by_instance_.find(ctx.instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return false;
    }
    auto task_iter = instance_iter->second.find(ctx.block_key);
    if (task_iter == instance_iter->second.end() || task_iter->second.state != CopyTaskState::kPreparing) {
        return false;
    }
    const auto source_lease_start_time = task_iter->second.source_lease_start_time;
    task_iter->second = ctx;
    task_iter->second.source_lease_start_time = source_lease_start_time;
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
    const auto task_iter = instance_iter->second.find(block_key);
    if (task_iter == instance_iter->second.end()) {
        return false;
    }
    const CopyTaskContext removed_task = task_iter->second;
    instance_iter->second.erase(task_iter);
    if (instance_iter->second.empty()) {
        active_tasks_by_instance_.erase(instance_iter); // 内层空则收回外层，避免 stale 空 map
    }
    UpdateActiveTasksGauge();
    if (removed_task.source_lease_start_time.time_since_epoch().count() != 0) {
        const auto lease_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                           std::chrono::steady_clock::now() -
                                           removed_task.source_lease_start_time)
                                           .count();
        ObserveSourceLeaseDuration(
            removed_task.src_storage_name, removed_task.dst_storage_name, lease_duration_us);
    }
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
        out_ctx = task; // 交由 CompleteCancelledTask 收尾（删 WRITING 目标、不 promote）
        return ClaimResult::kWasCancelling;
    }
    if (task.state == CopyTaskState::kCompleting) {
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

bool MigrationManager::HasActiveCopySourceLocation(const std::string &instance_id,
                                                   int64_t block_key,
                                                   const std::string &location_id,
                                                   int64_t create_time) const {
    if (location_id.empty() || create_time <= 0) {
        return false;
    }
    std::string src_storage_name;
    std::string dst_storage_name;
    bool leased = false;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        const auto instance_iter = active_tasks_by_instance_.find(instance_id);
        if (instance_iter == active_tasks_by_instance_.end()) {
            return false;
        }
        const auto task_iter = instance_iter->second.find(block_key);
        if (task_iter == instance_iter->second.end()) {
            return false;
        }
        const CopyTaskContext &task = task_iter->second;
        leased = task.src_location_id == location_id && task.src_create_time == create_time;
        if (leased) {
            src_storage_name = task.src_storage_name;
            dst_storage_name = task.dst_storage_name;
        }
    }
    if (leased) {
        AddCounterMetric("migration_source_lease_conflicts_total",
                         {{"src", src_storage_name}, {"dst", dst_storage_name}, {"deleter", "scheduled_delete"}});
    }
    return leased;
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
    const auto all_covered =
        CollectCoveredSpecNames(loc_map, dst_storage_name, {CacheLocationStatus::CLS_SERVING, CacheLocationStatus::CLS_WRITING});

    auto specs_covered_by = [](const CacheLocation &src, const std::unordered_set<std::string> &covered) {
        return std::all_of(src.location_specs().begin(), src.location_specs().end(),
                           [&covered](const auto &spec) { return covered.count(spec.name()) > 0; });
    };

    bool all_sources_covered_by_serving = true;
    for (const auto *src_loc : src_locations) {
        if (specs_covered_by(*src_loc, serving_covered)) {
            continue;
        }
        all_sources_covered_by_serving = false;
        if (specs_covered_by(*src_loc, all_covered)) {
            continue;
        }
        // 该 source 既未被 SERVING 也未被 SERVING∪WRITING 联合覆盖 → 需要 copy。
        std::vector<LocationSpec> missing_specs;
        missing_specs.reserve(src_loc->location_specs().size());
        std::copy_if(src_loc->location_specs().begin(),
                     src_loc->location_specs().end(),
                     std::back_inserter(missing_specs),
                     [&all_covered](const LocationSpec &spec) { return all_covered.count(spec.name()) == 0; });
        return {CopyAdmissionStatus::kAccept, src_loc, std::move(missing_specs)};
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
    // Sampling backends may return the same block more than once. Preserve
    // first-seen order so a block consumes one action/Copy slot and receives
    // exactly one terminal outcome, just like explicit Admin keys and the
    // Reclaimer candidate builder.
    std::unordered_set<int64_t> seen;
    seen.reserve(candidate_keys.size());
    candidate_keys.erase(
        std::remove_if(candidate_keys.begin(),
                       candidate_keys.end(),
                       [&seen](int64_t block_key) { return !seen.insert(block_key).second; }),
        candidate_keys.end());
    return {EC_OK, std::move(candidate_keys)};
}

std::size_t MigrationManager::AsyncMigrationPrepareKeyHash::operator()(const AsyncMigrationPrepareKey &key) const {
    std::size_t seed = std::hash<std::uint64_t>{}(key.generation);
    const auto combine = [&seed](std::size_t value) {
        seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    };
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

std::size_t
MigrationManager::PendingAsyncMigrationPrepareCountForGroup(const std::string &instance_group_name) const {
    std::lock_guard<std::mutex> lock(async_prepare_mutex_);
    return static_cast<std::size_t>(
        std::count_if(pending_async_prepare_jobs_.begin(),
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
    AsyncMigrationPrepareKey key{generation,
                                 job.instance_group_name,
                                 job.instance_id,
                                 job.source_storage_name,
                                 job.target_storage_name};
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
        params.mark_timeout_ms = current_strategy->methods().mark().timeout_ms();
        params.admission = current_strategy->admission();
        params.trigger = DispatchBatchParams::Trigger::kReclaimer;
        params.action_budget = job.action_budget;
        params.pending_location_ids_by_block = std::move(job.pending_location_ids_by_block);
        return DispatchMigrationBatchWithLifecycleLockHeld(job.trace_id,
                                                            job.instance_id,
                                                            job.source_storage_name,
                                                            job.target_storage_name,
                                                            job.block_keys,
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
                                                              int64_t mark_timeout_ms,
                                                              MigrationRetention retention,
                                                              const MigrationAdmissionConfig &admission_config) {
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

    if (candidate_keys.empty()) {
        result.ec = EC_OK;
        result.accepted = 0;
        result.rejected = 0;
        result.message = "no candidate to migrate";
        return result;
    }

    // 准入、分发和 fallback 委派共享 DispatchMigrationBatch。
    DispatchBatchParams params;
    params.do_copy = do_copy;
    params.do_mark = do_mark;
    params.copy_limit = CopyConcurrencyLimit{instance_group_name, copy_max_concurrency};
    params.retention = retention;
    params.mark_timeout_ms = mark_timeout_ms;
    params.admission = admission_config;
    params.trigger = DispatchBatchParams::Trigger::kAdmin;
    const auto dispatch = DispatchMigrationBatch(trace_id, instance_id, src_name, dst_name, candidate_keys, params);

    result.ec = dispatch.ec;
    result.accepted = dispatch.copy_submitted + dispatch.mark_submitted;
    // Legacy projection: rejected means "did not create a new Copy/Mark" and
    // therefore includes precise rejected/noop/failed terminal classes. New
    // callers should use outcome_counts for the authoritative breakdown.
    result.rejected = dispatch.rejected + dispatch.noop + dispatch.failed;
    result.outcome_counts = dispatch.outcome_counts;
    result.message = dispatch.ec == EC_OK ? "migrate cache dispatched" : "migration dispatch failed";
    return result;
}

MigrationManager::DispatchBatchResult MigrationManager::DispatchMigrationBatch(
    const std::string &trace_id,
    const std::string &instance_id,
    const std::string &src_name,
    const std::string &dst_name,
    const std::vector<int64_t> &batch,
    const DispatchBatchParams &params) {
    DispatchBatchResult result;
    if (batch.empty()) {
        return FinalizeDispatchMetrics(instance_id, src_name, dst_name, batch.size(), params, std::move(result));
    }
    if (!params.pending_location_ids_by_block.empty() &&
        params.pending_location_ids_by_block.size() != batch.size()) {
        result.ec = EC_BADARGS;
        result.failed = static_cast<int64_t>(batch.size());
        AddOutcome(result.outcome_counts,
                   MigrationOutcomeStage::kSnapshot,
                   MigrationOutcomeClass::kFailed,
                   MigrationOutcomeReason::kSnapshotShapeError,
                   result.failed,
                   true);
        return FinalizeDispatchMetrics(instance_id, src_name, dst_name, batch.size(), params, std::move(result));
    }
    if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
        result.ec = EC_ERROR;
        result.failed = static_cast<int64_t>(batch.size());
        AddOutcome(result.outcome_counts,
                   MigrationOutcomeStage::kSnapshot,
                   MigrationOutcomeClass::kFailed,
                   MigrationOutcomeReason::kDispatchNotAvailable,
                   result.failed,
                   true);
        return FinalizeDispatchMetrics(instance_id, src_name, dst_name, batch.size(), params, std::move(result));
    }
    // Copy 与 Mark 必须位于同一个 leader-lifecycle barrier 内。否则 Stop 可能在异步 Job
    // 完成 fresh read 后关闭 Copy gate，但 Job 仍继续写入 Mark。
    std::shared_lock<std::shared_mutex> lifecycle_lock(copy_submission_lifecycle_mutex_);
    if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
        result.ec = EC_ERROR;
        result.failed = static_cast<int64_t>(batch.size());
        AddOutcome(result.outcome_counts,
                   MigrationOutcomeStage::kSnapshot,
                   MigrationOutcomeClass::kFailed,
                   MigrationOutcomeReason::kDispatchNotAvailable,
                   result.failed,
                   true);
        return FinalizeDispatchMetrics(instance_id, src_name, dst_name, batch.size(), params, std::move(result));
    }
    return DispatchMigrationBatchWithLifecycleLockHeld(
        trace_id, instance_id, src_name, dst_name, batch, params);
}

MigrationManager::DispatchBatchResult MigrationManager::DispatchMigrationBatchWithLifecycleLockHeld(
    const std::string &trace_id,
    const std::string &instance_id,
    const std::string &src_name,
    const std::string &dst_name,
    const std::vector<int64_t> &batch,
    const DispatchBatchParams &params) {
    DispatchBatchResult result;
    if (batch.empty()) {
        return result;
    }
    if (!params.pending_location_ids_by_block.empty() &&
        params.pending_location_ids_by_block.size() != batch.size()) {
        result.ec = EC_BADARGS;
        result.failed = static_cast<int64_t>(batch.size());
        AddOutcome(result.outcome_counts,
                   MigrationOutcomeStage::kSnapshot,
                   MigrationOutcomeClass::kFailed,
                   MigrationOutcomeReason::kSnapshotShapeError,
                   result.failed,
                   true);
        return FinalizeDispatchMetrics(instance_id, src_name, dst_name, batch.size(), params, std::move(result));
    }
    const auto indexer = GetIndexer(instance_id);
    if (indexer == nullptr) {
        result.ec = EC_INSTANCE_NOT_EXIST;
        result.failed = static_cast<int64_t>(batch.size());
        AddOutcome(result.outcome_counts,
                   MigrationOutcomeStage::kSnapshot,
                   MigrationOutcomeClass::kFailed,
                   MigrationOutcomeReason::kDispatchNotAvailable,
                   result.failed,
                   true);
        return FinalizeDispatchMetrics(instance_id, src_name, dst_name, batch.size(), params, std::move(result));
    }

    std::string policy_error;
    auto policy = MigrationAdmissionPolicyFactory::Build(params.admission, policy_error);
    if (params.admission.mode() != MigrationAdmissionMode::DISABLED && policy == nullptr) {
        KVCM_LOG_WARN("[%s] invalid migration admission policy for %s -> %s: %s",
                      trace_id.c_str(),
                      src_name.c_str(),
                      dst_name.c_str(),
                      policy_error.c_str());
        if (params.admission.mode() != MigrationAdmissionMode::SHADOW) {
            result.ec = EC_CONFIG_ERROR;
            result.failed = static_cast<int64_t>(batch.size());
            AddOutcome(result.outcome_counts,
                       MigrationOutcomeStage::kValue,
                       MigrationOutcomeClass::kFailed,
                       MigrationOutcomeReason::kPolicyContractError,
                       result.failed,
                       true);
            return FinalizeDispatchMetrics(instance_id, src_name, dst_name, batch.size(), params, std::move(result));
        }
        // SHADOW is behavior-neutral even when a malformed/recovered config
        // reaches this defensive runtime check. Record the projected fault and
        // continue through the legacy location/execution path.
        AddOutcome(result.outcome_counts,
                   MigrationOutcomeStage::kValue,
                   MigrationOutcomeClass::kFailed,
                   MigrationOutcomeReason::kPolicyContractError,
                   static_cast<int64_t>(batch.size()),
                   false);
    }

    const MigrationAdmissionFeatureSet required_features =
        policy ? policy->RequiredFeatures() : MigrationAdmissionFeatureSet{};
    std::vector<std::string> property_names;
    bool last_access_provider_ready = true;
    if (required_features.test(static_cast<size_t>(MigrationAdmissionFeature::kLastAccessTime))) {
        property_names.push_back(PROPERTY_LRU_TIME);
        const auto readiness = indexer->GetMaintenancePropertyReadiness(PROPERTY_LRU_TIME);
        last_access_provider_ready = readiness.capability == MaintenancePropertyCapability::kDurableAcrossRecovery;
        if (readiness.capability == MaintenancePropertyCapability::kProcessLocalVolatile) {
            const auto &policies = params.admission.policies();
            const int64_t window_seconds = policies.empty() || policies.front() == nullptr ||
                                                   policies.front()->recent_access() == nullptr
                                               ? 0
                                               : policies.front()->recent_access()->window_seconds();
            const int64_t steady_now = TimestampUtil::GetSteadyTimeUs();
            const bool window_valid = window_seconds > 0 &&
                                      window_seconds <= std::numeric_limits<int64_t>::max() / (1000 * 1000);
            last_access_provider_ready = window_valid && readiness.valid_since_steady_us > 0 &&
                                         steady_now >= readiness.valid_since_steady_us &&
                                         steady_now - readiness.valid_since_steady_us >=
                                             window_seconds * 1000 * 1000;
        }
        MetricsTags readiness_tags{{"instance", instance_id},
                                   {"src", src_name},
                                   {"dst", dst_name},
                                   {"feature", "last_access_time"}};
        SetGaugeMetric("migration_admission_readiness", readiness_tags, last_access_provider_ready ? 1.0 : 0.0);
        if (!last_access_provider_ready) {
            readiness_tags["reason"] =
                readiness.capability == MaintenancePropertyCapability::kUnsupported ? "unsupported" : "warmup";
            AddCounterMetric("migration_admission_readiness_not_ready_total", std::move(readiness_tags));
        }
        if (params.admission.mode() == MigrationAdmissionMode::ENFORCE && !last_access_provider_ready) {
            result.ec = readiness.capability == MaintenancePropertyCapability::kUnsupported ? EC_CONFIG_ERROR
                                                                                             : EC_ERROR;
            result.failed = static_cast<int64_t>(batch.size());
            AddOutcome(result.outcome_counts,
                       MigrationOutcomeStage::kSnapshot,
                       MigrationOutcomeClass::kFailed,
                       MigrationOutcomeReason::kRouteNotReady,
                       result.failed,
                       true);
            KVCM_LOG_WARN("[%s] migration admission route not ready for instance %s, src %s, dst %s, "
                          "capability %d, generation %llu",
                          trace_id.c_str(),
                          instance_id.c_str(),
                          src_name.c_str(),
                          dst_name.c_str(),
                          static_cast<int>(readiness.capability),
                          static_cast<unsigned long long>(readiness.generation));
            return FinalizeDispatchMetrics(instance_id, src_name, dst_name, batch.size(), params, std::move(result));
        }
    }

    CacheLocationMapVector location_maps;
    PropertyMapVector property_maps;
    RequestContext snapshot_context(trace_id.empty() ? "migration_admission_snapshot" : trace_id);
    const auto snapshot_result =
        indexer->GetForMaintenance(&snapshot_context, batch, property_names, location_maps, property_maps);
    if (snapshot_result.locations.error_codes.size() != batch.size() || location_maps.size() != batch.size() ||
        snapshot_result.properties.error_codes.size() != batch.size() || property_maps.size() != batch.size()) {
        result.ec = EC_MISMATCH;
        result.failed = static_cast<int64_t>(batch.size());
        AddOutcome(result.outcome_counts,
                   MigrationOutcomeStage::kSnapshot,
                   MigrationOutcomeClass::kFailed,
                   MigrationOutcomeReason::kSnapshotShapeError,
                   result.failed,
                   true);
        return FinalizeDispatchMetrics(instance_id, src_name, dst_name, batch.size(), params, std::move(result));
    }

    struct PreparedCandidate {
        size_t original_index = 0;
        int64_t block_key = 0;
        CacheLocationMap locations;
        MigrationCandidateFeatures features;
        int64_t last_access_time_us = 0;
    };
    std::vector<PreparedCandidate> location_valid_candidates;
    std::vector<MigrationCandidateFeatures> policy_inputs;
    location_valid_candidates.reserve(batch.size());
    policy_inputs.reserve(batch.size());
    int64_t hard_location_failures = 0;
    int64_t hard_feature_failures = 0;
    const int64_t admission_now_us = TimestampUtil::GetCurrentTimeUs();
    for (size_t i = 0; i < batch.size(); ++i) {
        const ErrorCode location_ec = snapshot_result.locations.error_codes[i];
        if (location_ec != EC_OK) {
            if (location_ec == EC_NOENT) {
                ++result.rejected;
                AddOutcome(result.outcome_counts,
                           MigrationOutcomeStage::kSnapshot,
                           MigrationOutcomeClass::kRejected,
                           MigrationOutcomeReason::kSourceNotFound,
                           1,
                           true);
            } else {
                ++result.failed;
                ++hard_location_failures;
                AddOutcome(result.outcome_counts,
                           MigrationOutcomeStage::kSnapshot,
                           MigrationOutcomeClass::kFailed,
                           MigrationOutcomeReason::kLocationReadError,
                           1,
                           true);
            }
            continue;
        }
        PreparedCandidate candidate;
        candidate.original_index = i;
        candidate.block_key = batch[i];
        candidate.locations = std::move(location_maps[i]);
        bool terminal_feature_failure = false;

        if (required_features.test(static_cast<size_t>(MigrationAdmissionFeature::kLastAccessTime))) {
            ObservedFeature feature;
            if (!last_access_provider_ready) {
                feature.status = ObservedFeatureStatus::kUnsupported;
            } else if (snapshot_result.properties.error_codes[i] == EC_NOENT) {
                // In a dual backend, location may be recovered from the
                // persistent side while process-local recency remains
                // intentionally unavailable. That is a missing feature, not
                // an infrastructure read failure.
                feature.status = ObservedFeatureStatus::kMissing;
            } else if (snapshot_result.properties.error_codes[i] != EC_OK) {
                feature.status = ObservedFeatureStatus::kReadError;
                if (params.admission.mode() == MigrationAdmissionMode::ENFORCE) {
                    ++hard_feature_failures;
                    ++result.failed;
                    terminal_feature_failure = true;
                    AddOutcome(result.outcome_counts,
                               MigrationOutcomeStage::kSnapshot,
                               MigrationOutcomeClass::kFailed,
                               MigrationOutcomeReason::kFeatureReadError,
                               1,
                               true);
                }
            } else {
                const auto property = property_maps[i].find(PROPERTY_LRU_TIME);
                if (property == property_maps[i].end()) {
                    feature.status = ObservedFeatureStatus::kMissing;
                } else if (const auto value = ParsePositiveInt64(property->second); value.has_value()) {
                    feature.status = ObservedFeatureStatus::kAvailable;
                    feature.value = *value;
                    candidate.last_access_time_us = *value;
                    if (*value <= admission_now_us) {
                        ObserveAdmissionAccessAge(src_name, dst_name, params, admission_now_us - *value);
                    }
                } else {
                    feature.status = ObservedFeatureStatus::kInvalid;
                }
            }
            candidate.features.Set(MigrationAdmissionFeature::kLastAccessTime, std::move(feature));
        }
        if (terminal_feature_failure) {
            continue;
        }
        policy_inputs.push_back(candidate.features);
        location_valid_candidates.push_back(std::move(candidate));
    }
    const int64_t hard_snapshot_failures = hard_location_failures + hard_feature_failures;
    if (hard_snapshot_failures > 0) {
        result.ec = hard_snapshot_failures == static_cast<int64_t>(batch.size()) ? EC_ERROR : EC_PARTIAL_OK;
    }

    std::vector<MigrationAdmissionDecision> decisions;
    if (policy != nullptr) {
        decisions = policy->EvaluateBatch(policy_inputs, MigrationAdmissionContext{admission_now_us});
        if (decisions.size() != location_valid_candidates.size()) {
            if (params.admission.mode() == MigrationAdmissionMode::ENFORCE) {
                result.ec = EC_ERROR;
                result.failed += static_cast<int64_t>(location_valid_candidates.size());
                AddOutcome(result.outcome_counts,
                           MigrationOutcomeStage::kValue,
                           MigrationOutcomeClass::kFailed,
                           MigrationOutcomeReason::kPolicyContractError,
                           static_cast<int64_t>(location_valid_candidates.size()),
                           true);
                return FinalizeDispatchMetrics(
                    instance_id, src_name, dst_name, batch.size(), params, std::move(result));
            }
            AddOutcome(result.outcome_counts,
                       MigrationOutcomeStage::kValue,
                       MigrationOutcomeClass::kFailed,
                       MigrationOutcomeReason::kPolicyContractError,
                       static_cast<int64_t>(location_valid_candidates.size()),
                       false);
            decisions.clear();
        }
    }

    std::vector<PreparedCandidate> execution_candidates;
    execution_candidates.reserve(location_valid_candidates.size());
    for (size_t i = 0; i < location_valid_candidates.size(); ++i) {
        const bool has_decision = i < decisions.size();
        if (has_decision) {
            const bool accepted = decisions[i].verdict == MigrationAdmissionVerdict::kAccept;
            const bool terminal = params.admission.mode() == MigrationAdmissionMode::ENFORCE && !accepted;
            AddOutcome(result.outcome_counts,
                       MigrationOutcomeStage::kValue,
                       accepted ? MigrationOutcomeClass::kAccepted : MigrationOutcomeClass::kRejected,
                       ToOutcomeReason(decisions[i].reason),
                       1,
                       terminal);
            if (params.do_copy) {
                const auto projected_specs = FindFirstUncoveredSourceSpecs(
                    location_valid_candidates[i].locations, src_name, dst_name);
                const auto projected = SummarizeSpecBytes(projected_specs);
                const MetricsTags byte_tags{{"src", src_name},
                                             {"dst", dst_name},
                                             {"decision", accepted ? "value_accept" : "value_reject"}};
                AddCounterMetric("migration_copy_planned_bytes_total", byte_tags, projected.bytes);
                AddCounterMetric(
                    "migration_copy_planned_bytes_unknown_specs_total", byte_tags, projected.unknown_specs);
            }
        }
        if (params.admission.mode() == MigrationAdmissionMode::ENFORCE &&
            (!has_decision || decisions[i].verdict != MigrationAdmissionVerdict::kAccept)) {
            ++result.rejected;
            continue;
        }
        execution_candidates.push_back(std::move(location_valid_candidates[i]));
    }
    AddCounterMetric("migration_candidate_pool_total",
                     {{"src", src_name}, {"dst", dst_name}, {"stage", "value_qualified"}},
                     execution_candidates.size());
    if (params.trigger == DispatchBatchParams::Trigger::kReclaimer &&
        params.admission.mode() == MigrationAdmissionMode::ENFORCE) {
        std::stable_sort(execution_candidates.begin(),
                         execution_candidates.end(),
                         [](const PreparedCandidate &lhs, const PreparedCandidate &rhs) {
                             return lhs.last_access_time_us < rhs.last_access_time_us;
                         });
    }
    if (params.trigger == DispatchBatchParams::Trigger::kReclaimer &&
        execution_candidates.size() > params.action_budget) {
        const int64_t budget_rejected =
            static_cast<int64_t>(execution_candidates.size() - params.action_budget);
        result.rejected += budget_rejected;
        AddOutcome(result.outcome_counts,
                   MigrationOutcomeStage::kExecution,
                   MigrationOutcomeClass::kRejected,
                   MigrationOutcomeReason::kBudgetExhausted,
                   budget_rejected,
                   true);
        execution_candidates.resize(params.action_budget);
    }

    // Execution admission consumes the exact locations from the same final
    // NoTouch snapshot. Pending reclaimer deletions are filtered only now,
    // after the aligned snapshot has been built.
    std::vector<MigrationRequest> copy_reqs;
    std::vector<int64_t> copy_block_keys;
    std::vector<int64_t> mark_keys;
    std::unordered_map<int64_t, SpecByteSummary> eligible_spec_bytes_by_block;
    for (auto &candidate : execution_candidates) {
        if (!params.pending_location_ids_by_block.empty()) {
            for (const auto &pending_id : params.pending_location_ids_by_block[candidate.original_index]) {
                candidate.locations.erase(pending_id);
            }
        }
        const int64_t block_key = candidate.block_key;
        const auto admission = CheckCopyAdmission(instance_id, block_key, candidate.locations, src_name, dst_name);
        if (admission.status != CopyAdmissionStatus::kAccept || admission.src_location == nullptr) {
            ++result.rejected;
            AddOutcome(result.outcome_counts,
                       MigrationOutcomeStage::kExecution,
                       MigrationOutcomeClass::kRejected,
                       ToOutcomeReason(admission.status),
                       1,
                       true);
            continue;
        }
        if (admission.missing_specs.empty()) {
            ++result.rejected;
            AddOutcome(result.outcome_counts,
                       MigrationOutcomeStage::kExecution,
                       MigrationOutcomeClass::kRejected,
                       MigrationOutcomeReason::kTargetAlreadyCovered,
                       1,
                       true);
            continue;
        }
        eligible_spec_bytes_by_block[block_key] = SummarizeSpecBytes(admission.missing_specs);
        if (params.do_copy && copy_reqs.size() < params.max_copy_slots) {
            MigrationRequest req;
            req.instance_group_name = params.copy_limit.instance_group_name;
            req.instance_id = instance_id;
            req.block_key = block_key;
            req.src_location_id = admission.src_location->id();
            req.src_storage_name = src_name;
            req.dst_storage_name = dst_name;
            req.retention = params.retention;
            req.src_specs = admission.missing_specs;
            req.src_create_time = admission.src_location->create_time();
            copy_block_keys.push_back(block_key);
            copy_reqs.push_back(std::move(req));
            const auto &planned = eligible_spec_bytes_by_block[block_key];
            const MetricsTags byte_tags{{"src", src_name}, {"dst", dst_name}, {"decision", "dispatched"}};
            AddCounterMetric("migration_copy_planned_bytes_total", byte_tags, planned.bytes);
            AddCounterMetric(
                "migration_copy_planned_bytes_unknown_specs_total", byte_tags, planned.unknown_specs);
            continue; // Copy 优先：已进入 copy 的 block 不再重复 mark。
        }
        if (params.do_mark) {
            mark_keys.push_back(block_key);
        } else {
            ++result.rejected;
            AddOutcome(result.outcome_counts,
                       MigrationOutcomeStage::kExecution,
                       MigrationOutcomeClass::kRejected,
                       params.do_copy ? MigrationOutcomeReason::kCopySlotExhausted
                                      : MigrationOutcomeReason::kNoExecutionMethod,
                       1,
                       true);
        }
    }

    // 分发
    if (!copy_reqs.empty()) {
        const auto results = BatchSubmit(trace_id, std::move(copy_reqs), params.copy_limit, true);
        const std::size_t aligned_result_count = std::min(results.size(), copy_block_keys.size());
        for (std::size_t i = 0; i < aligned_result_count; ++i) {
            if (results[i] == EC_OK) {
                ++result.copy_submitted;
                AddOutcome(result.outcome_counts,
                           MigrationOutcomeStage::kCopy,
                           MigrationOutcomeClass::kAccepted,
                           MigrationOutcomeReason::kCopySubmitted,
                           1,
                           true);
            } else {
                ++result.copy_failed;
                if (params.do_mark && i < copy_block_keys.size()) {
                    AddOutcome(result.outcome_counts,
                               MigrationOutcomeStage::kCopy,
                               MigrationOutcomeClass::kFailed,
                               CopySubmitOutcomeReason(results[i]),
                               1,
                               false);
                    mark_keys.push_back(copy_block_keys[i]); // copy 失败时 fallback 到 mark
                } else if (results[i] == EC_OUT_OF_LIMIT || results[i] == EC_EXIST ||
                           results[i] == EC_NOENT || results[i] == EC_NOSPC) {
                    // 并发额度、重复任务、源状态或目标配额已不满足都是
                    // execution/target admission reject，不是基础设施故障。这也保证
                    // legacy accepted/rejected 响应不会遗漏这些候选。
                    ++result.rejected;
                    AddOutcome(result.outcome_counts,
                               MigrationOutcomeStage::kCopy,
                               MigrationOutcomeClass::kRejected,
                               CopySubmitOutcomeReason(results[i]),
                               1,
                               true);
                } else {
                    ++result.failed;
                    AddOutcome(result.outcome_counts,
                               MigrationOutcomeStage::kCopy,
                               MigrationOutcomeClass::kFailed,
                               CopySubmitOutcomeReason(results[i]),
                               1,
                               true);
                }
            }
        }
        if (aligned_result_count < copy_block_keys.size()) {
            const int64_t missing_results = static_cast<int64_t>(copy_block_keys.size() - aligned_result_count);
            result.failed += missing_results;
            result.ec = result.ec == EC_OK ? EC_MISMATCH : EC_ERROR;
            AddOutcome(result.outcome_counts,
                       MigrationOutcomeStage::kCopy,
                       MigrationOutcomeClass::kFailed,
                       MigrationOutcomeReason::kCopySubmitFailed,
                       missing_results,
                       true);
        }
    }
    if (params.do_mark && !mark_keys.empty()) {
        std::uint64_t mark_eligible_bytes = 0;
        std::uint64_t mark_unknown_specs = 0;
        for (const auto block_key : mark_keys) {
            if (const auto summary = eligible_spec_bytes_by_block.find(block_key);
                summary != eligible_spec_bytes_by_block.end()) {
                mark_unknown_specs += summary->second.unknown_specs;
                if (summary->second.bytes > std::numeric_limits<std::uint64_t>::max() - mark_eligible_bytes) {
                    ++mark_unknown_specs;
                } else {
                    mark_eligible_bytes += summary->second.bytes;
                }
            }
        }
        const MetricsTags byte_tags{{"src", src_name}, {"dst", dst_name}, {"decision", "eligible"}};
        AddCounterMetric("migration_mark_eligible_source_bytes_total", byte_tags, mark_eligible_bytes);
        AddCounterMetric(
            "migration_mark_eligible_source_bytes_unknown_specs_total", byte_tags, mark_unknown_specs);
        const auto mark_result =
            MarkForTieredWriteDetailed(instance_id, mark_keys, dst_name, params.mark_timeout_ms);
        const std::size_t aligned_outcome_count = std::min(mark_result.outcomes.size(), mark_keys.size());
        for (std::size_t i = 0; i < aligned_outcome_count; ++i) {
            const auto &outcome = mark_result.outcomes[i];
            switch (outcome.status) {
            case MarkWriteStatus::kInserted:
                ++result.mark_submitted;
                AddOutcome(result.outcome_counts,
                           MigrationOutcomeStage::kMark,
                           MigrationOutcomeClass::kAccepted,
                           MigrationOutcomeReason::kMarkInserted,
                           1,
                           true);
                break;
            case MarkWriteStatus::kAlreadySameTarget:
                ++result.noop;
                AddOutcome(result.outcome_counts,
                           MigrationOutcomeStage::kMark,
                           MigrationOutcomeClass::kNoopAlreadySatisfied,
                           MigrationOutcomeReason::kMarkAlreadySameTarget,
                           1,
                           true);
                break;
            case MarkWriteStatus::kConflictDifferentTarget:
                ++result.rejected;
                AddOutcome(result.outcome_counts,
                           MigrationOutcomeStage::kMark,
                           MigrationOutcomeClass::kRejected,
                           MigrationOutcomeReason::kMarkConflictDifferentTarget,
                           1,
                           true);
                break;
            case MarkWriteStatus::kMalformedExistingMark:
                ++result.rejected;
                AddOutcome(result.outcome_counts,
                           MigrationOutcomeStage::kMark,
                           MigrationOutcomeClass::kRejected,
                           MigrationOutcomeReason::kMarkMalformed,
                           1,
                           true);
                break;
            case MarkWriteStatus::kBlockNotFound:
                ++result.rejected;
                AddOutcome(result.outcome_counts,
                           MigrationOutcomeStage::kMark,
                           MigrationOutcomeClass::kRejected,
                           MigrationOutcomeReason::kBlockNotFound,
                           1,
                           true);
                break;
            case MarkWriteStatus::kWriteError:
                ++result.failed;
                AddOutcome(result.outcome_counts,
                           MigrationOutcomeStage::kMark,
                           MigrationOutcomeClass::kFailed,
                           MigrationOutcomeReason::kMarkWriteError,
                           1,
                           true);
                break;
            case MarkWriteStatus::kReadError:
                ++result.failed;
                AddOutcome(result.outcome_counts,
                           MigrationOutcomeStage::kMark,
                           MigrationOutcomeClass::kFailed,
                           MigrationOutcomeReason::kMarkReadError,
                           1,
                           true);
                break;
            }
        }
        if (aligned_outcome_count < mark_keys.size()) {
            const int64_t missing_outcomes = static_cast<int64_t>(mark_keys.size() - aligned_outcome_count);
            result.failed += missing_outcomes;
            result.ec = result.ec == EC_OK ? EC_MISMATCH : EC_ERROR;
            AddOutcome(result.outcome_counts,
                       MigrationOutcomeStage::kMark,
                       MigrationOutcomeClass::kFailed,
                       MigrationOutcomeReason::kMarkWriteError,
                       missing_outcomes,
                       true);
        }
        if (mark_result.ec != EC_OK) {
            result.ec = result.ec == EC_OK ? mark_result.ec : EC_ERROR;
        }
    }
    if (result.failed > 0) {
        result.ec = result.failed == static_cast<int64_t>(batch.size()) ? EC_ERROR : EC_PARTIAL_OK;
    }
    return FinalizeDispatchMetrics(instance_id, src_name, dst_name, batch.size(), params, std::move(result));
}

MigrationManager::MigrationStats MigrationManager::GetStats() const {
    MigrationStats stats;
    stats.copy_submitted = stat_copy_submitted_.load(std::memory_order_relaxed);
    stats.copy_completed = stat_copy_completed_.load(std::memory_order_relaxed);
    stats.copy_failed = stat_copy_failed_.load(std::memory_order_relaxed);
    stats.copy_cancelled = stat_copy_cancelled_.load(std::memory_order_relaxed);
    stats.marks_added = stat_marks_added_.load(std::memory_order_relaxed);
    stats.marks_cleared = stat_marks_cleared_.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        stats.active_copy_tasks = ActiveTaskCountUnsafe();
    }
    // 持久化方案下无内存表，active_marks 为 best-effort 近似（added - cleared）。
    const uint64_t added = stats.marks_added;
    const uint64_t cleared = stats.marks_cleared;
    stats.active_marks = added > cleared ? static_cast<size_t>(added - cleared) : 0;
    return stats;
}

} // namespace kv_cache_manager
