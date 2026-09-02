#include "kv_cache_manager/manager/cache_manager.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/meta_cache_policy_config.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/event_report_backend.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/event/spec_events/optimizer_event.h"
#include "kv_cache_manager/manager/cache_manager_metrics_recorder.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
#include "kv_cache_manager/manager/data_storage_selector.h"
#include "kv_cache_manager/manager/hash_util.h"
#include "kv_cache_manager/manager/meta_searcher_manager.h"
#include "kv_cache_manager/manager/migration_manager.h"
#include "kv_cache_manager/manager/reclaimer_task_supervisor.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/manager/select_location_policy.h"
#include "kv_cache_manager/manager/startup_config_loader.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/meta/types.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_lifecycle.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"

namespace kv_cache_manager {

#define PREFIX_LOG(LEVEL, format, args...)                                                                             \
    do {                                                                                                               \
        KVCM_LOG_##LEVEL("trace_id [%s] instance [%s] | " format, trace_id.c_str(), instance_id.c_str(), ##args);      \
    } while (0)

#define RETURN_IF_EC_NOT_OK(ec)                                                                                        \
    do {                                                                                                               \
        if ((ec) != EC_OK) {                                                                                           \
            return ec;                                                                                                 \
        }                                                                                                              \
    } while (0)

#define RETURN_IF_EC_NOT_OK_WITH_TYPE(ec, Type)                                                                        \
    do {                                                                                                               \
        if ((ec) != EC_OK) {                                                                                           \
            return {ec, Type()};                                                                                       \
        }                                                                                                              \
    } while (0)

#define RETURN_IF_EC_NOT_OK_WITH_LOG(LEVEL, ec, format, args...)                                                       \
    do {                                                                                                               \
        if ((ec) != EC_OK) {                                                                                           \
            PREFIX_LOG(LEVEL, format, ##args);                                                                         \
            return ec;                                                                                                 \
        }                                                                                                              \
    } while (0)

#define RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(LEVEL, ec, Type, format, args...)                                            \
    do {                                                                                                               \
        if ((ec) != EC_OK) {                                                                                           \
            PREFIX_LOG(LEVEL, format, ##args);                                                                         \
            return {ec, Type()};                                                                                       \
        }                                                                                                              \
    } while (0)

namespace {
CacheManager::KeyVector GenKeyVector(const CacheManager::TokenIdsVector &tokens, int64_t block_size) {
    std::vector<int64_t> block_keys;
    size_t total_blocks = tokens.size() / block_size;

    int64_t hash = 0;
    for (int index = 0; index < total_blocks; index++) {
        auto pos = index * block_size;
        hash = hashInt64Array(hash, &tokens[pos], &tokens[pos + block_size]);
        block_keys.push_back(hash);
    }
    return block_keys;
}

inline std::pair<kv_cache_manager::ErrorCode, bool>
IsSpecNameInSpecGroup(const std::string &trace_id,
                      const std::string &instance_id,
                      std::string_view spec_name,
                      std::string_view group_name,
                      const std::vector<kv_cache_manager::LocationSpecGroup> &location_spec_groups) {
    // we have sorted location_spec_groups before
    auto it_group = std::lower_bound(location_spec_groups.begin(),
                                     location_spec_groups.end(),
                                     group_name,
                                     [](const auto &location_spec_group, std::string_view group_name) {
                                         return location_spec_group.name() < group_name;
                                     });
    if (it_group == location_spec_groups.end() || it_group->name() != group_name) {
        PREFIX_LOG(WARN, "not find group [%s]", group_name.data());
        return {EC_ERROR, false};
    }
    const auto &group = *it_group;
    auto it_spec_name = std::lower_bound(
        group.spec_names().begin(),
        group.spec_names().end(),
        spec_name,
        [](const std::string &src_spec_name, std::string_view dst_spec_name) { return src_spec_name < dst_spec_name; });
    if (it_spec_name == group.spec_names().end() || *it_spec_name != spec_name) {
        PREFIX_LOG(DEBUG, "not find spec_name [%s] in group [%s]", spec_name.data(), group_name.data());
        return {EC_OK, false};
    }
    return {EC_OK, true};
}
class DeltaMutationGuard {
public:
    struct LeaseInfo {
        std::string snapshot_version;
        uint64_t lifecycle_generation = 0;
    };
    DeltaMutationGuard(std::shared_ptr<EventReportBackend> backend, ReporterSnapshotKey reporter_key)
        : backend_(std::move(backend)), reporter_key_(std::move(reporter_key)) {}
    DeltaMutationGuard(const DeltaMutationGuard &) = delete;
    DeltaMutationGuard &operator=(const DeltaMutationGuard &) = delete;

    ~DeltaMutationGuard() {
        if (lease_) {
            backend_->EndDeltaMutation(reporter_key_, lease_->lifecycle_generation, lease_->snapshot_version);
        }
    }

    ErrorCode Acquire(const LeaseInfo *&out_lease, bool &out_created_generation) {
        out_lease = nullptr;
        if (snapshot_wait_failure_) {
            out_created_generation = false;
            return *snapshot_wait_failure_;
        }
        if (lease_) {
            out_lease = &*lease_;
            out_created_generation = false;
            return EC_OK;
        }
        LeaseInfo lease;
        const ErrorCode ec = backend_->BeginDeltaMutation(
            reporter_key_, lease.snapshot_version, &lease.lifecycle_generation, &out_created_generation);
        if (ec != EC_OK) {
            out_created_generation = false;
            if (ec == EC_SNAPSHOT_IN_PROGRESS) {
                snapshot_wait_failure_ = ec;
            }
            return ec;
        }
        lease_.emplace(std::move(lease));
        out_lease = &*lease_;
        return EC_OK;
    }

    void AdoptLifecycleGeneration(uint64_t lifecycle_generation) {
        if (lease_) {
            lease_->lifecycle_generation = lifecycle_generation;
        }
    }

private:
    std::shared_ptr<EventReportBackend> backend_;
    ReporterSnapshotKey reporter_key_;
    std::optional<LeaseInfo> lease_;
    std::optional<ErrorCode> snapshot_wait_failure_;
};

// 共享 helper：收集目标 storage 上指定 status 的 location 联合覆盖的 spec name 集合。
// 一次 O(L·S) 扫描。exclude_loc_ids 用于排除 stale location。
std::unordered_set<std::string> CollectCoveredSpecNames(const CacheLocationMap &loc_map,
                                                        const std::string &storage_name,
                                                        std::initializer_list<CacheLocationStatus> statuses,
                                                        const std::vector<std::string> &exclude_loc_ids = {}) {
    std::unordered_set<std::string> covered;
    for (const auto &[loc_id, loc_ptr] : loc_map) {
        if (!loc_ptr || std::find(statuses.begin(), statuses.end(), loc_ptr->status()) == statuses.end()) {
            continue;
        }
        if (!exclude_loc_ids.empty() &&
            std::find(exclude_loc_ids.begin(), exclude_loc_ids.end(), loc_id) != exclude_loc_ids.end()) {
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

// 判断目标 storage 上是否有 SERVING/WRITING location（联合）覆盖 requested specs。
// requested_spec_names 空时只判"有任何 spec 在该 storage 上"。
bool HasServingOrWritingLocOnStorage(const CacheLocationMap &loc_map,
                                     const std::string &storage_name,
                                     const std::vector<std::string> &requested_spec_names = {},
                                     const std::vector<std::string> &exclude_loc_ids = {}) {
    const auto covered = CollectCoveredSpecNames(
        loc_map, storage_name, {CacheLocationStatus::CLS_SERVING, CacheLocationStatus::CLS_WRITING}, exclude_loc_ids);
    if (requested_spec_names.empty()) {
        return !covered.empty();
    }
    return std::all_of(requested_spec_names.begin(), requested_spec_names.end(), [&covered](const auto &name) {
        return covered.count(name) > 0;
    });
}

const CacheLocation *FindLocationById(const CacheLocationMap &loc_map, const std::string &location_id) {
    auto it = loc_map.find(location_id);
    if (it == loc_map.end()) {
        return nullptr;
    }
    return it->second.get();
}

bool LocationHasSpecOnStorage(const CacheLocation &loc, const std::string &storage_name) {
    return std::any_of(loc.location_specs().begin(), loc.location_specs().end(), [&storage_name](const auto &spec) {
        const DataStorageUri uri(spec.uri());
        return uri.Valid() && uri.GetHostName() == storage_name;
    });
}

std::vector<std::string> BuildAllLocationSpecNames(const std::shared_ptr<const InstanceInfo> &instance_info) {
    std::vector<std::string> spec_names;
    if (instance_info == nullptr) {
        return spec_names;
    }
    spec_names.reserve(instance_info->location_spec_infos().size());
    for (const auto &spec_info : instance_info->location_spec_infos()) {
        spec_names.push_back(spec_info.name());
    }
    return spec_names;
}

bool LocationsCoverFullBlockOnStorage(const CacheLocationMap &loc_map,
                                      const std::string &storage_name,
                                      const std::shared_ptr<const InstanceInfo> &instance_info) {
    if (instance_info == nullptr || instance_info->location_spec_infos().empty()) {
        return false;
    }
    return std::all_of(instance_info->location_spec_infos().begin(),
                       instance_info->location_spec_infos().end(),
                       [&loc_map, &storage_name](const LocationSpecInfo &spec_info) {
                           return std::any_of(
                               loc_map.begin(), loc_map.end(), [&spec_info, &storage_name](const auto &entry) {
                                   const auto &loc_ptr = entry.second;
                                   if (!loc_ptr || loc_ptr->status() != CacheLocationStatus::CLS_SERVING) {
                                       return false;
                                   }
                                   return std::any_of(loc_ptr->location_specs().begin(),
                                                      loc_ptr->location_specs().end(),
                                                      [&spec_info, &storage_name](const auto &spec) {
                                                          const DataStorageUri uri(spec.uri());
                                                          return spec.name() == spec_info.name() && uri.Valid() &&
                                                                 uri.GetHostName() == storage_name;
                                                      });
                               });
                       });
}

MigrationMarkClearPolicy GetMigrationMarkClearPolicy(RequestContext *request_context,
                                                     const std::shared_ptr<RegistryManager> &registry_manager,
                                                     const std::shared_ptr<const InstanceInfo> &instance_info) {
    if (registry_manager == nullptr || instance_info == nullptr) {
        return MigrationMarkClearPolicy::CLEAR_ON_NEXT_WRITE_SUCCESS;
    }
    auto [ec, instance_group] =
        registry_manager->GetInstanceGroup(request_context, instance_info->instance_group_name());
    if (ec != EC_OK || instance_group == nullptr || instance_group->cache_config() == nullptr) {
        return MigrationMarkClearPolicy::CLEAR_ON_NEXT_WRITE_SUCCESS;
    }
    return instance_group->cache_config()->migration_mark_clear_policy();
}

bool IsTieredMigrationEnabled(RequestContext *request_context,
                              const std::shared_ptr<RegistryManager> &registry_manager,
                              const std::shared_ptr<const InstanceInfo> &instance_info) {
    if (registry_manager == nullptr || instance_info == nullptr) {
        return false;
    }
    auto [ec, instance_group] =
        registry_manager->GetInstanceGroup(request_context, instance_info->instance_group_name());
    return ec == EC_OK && instance_group != nullptr && instance_group->cache_config() != nullptr &&
           !instance_group->cache_config()->migration_strategies().empty();
}

const char *StorageTargetAdmissionStatusName(StorageTargetAdmissionStatus status) {
    switch (status) {
    case StorageTargetAdmissionStatus::kAllowed:
        return "allowed";
    case StorageTargetAdmissionStatus::kNotFound:
        return "not_found";
    case StorageTargetAdmissionStatus::kUnavailable:
        return "unavailable";
    case StorageTargetAdmissionStatus::kGroupQuotaExceeded:
        return "group_quota_exceeded";
    case StorageTargetAdmissionStatus::kStorageTypeQuotaExceeded:
        return "storage_type_quota_exceeded";
    case StorageTargetAdmissionStatus::kReadError:
        return "read_error";
    }
    return "unknown";
}

// 只把 target 可用且 quota 未超限的 valid Mark 交给写入过滤逻辑。target 被注销时按查询
// 快照精确清标；unavailable/quota 满属于瞬时状态，保留 Mark 并让本次调用使用普通 write 判定。
void ResolveUsableTieredWriteTargets(RequestContext *request_context,
                                     const std::string &instance_id,
                                     const std::string &instance_group_name,
                                     const CacheManager::KeyVector &keys,
                                     const std::vector<MigrationManager::MarkQueryResult> &mark_results,
                                     const std::shared_ptr<DataStorageManager> &data_storage_manager,
                                     const std::shared_ptr<DataStorageSelector> &data_storage_selector,
                                     const std::shared_ptr<MigrationManager> &migration_manager,
                                     std::vector<std::string> &out_targets) {
    const size_t result_count = std::min({keys.size(), mark_results.size(), out_targets.size()});
    std::vector<std::string> unique_targets;
    std::unordered_map<std::string, std::size_t> target_indexes;
    for (size_t i = 0; i < result_count; ++i) {
        const auto &mark = mark_results[i];
        if (mark.HasValidMark() && target_indexes.emplace(mark.target, unique_targets.size()).second) {
            unique_targets.push_back(mark.target);
        }
    }

    std::vector<StorageTargetAdmissionResult> admissions;
    if (data_storage_selector != nullptr) {
        admissions =
            data_storage_selector->CheckExplicitWriteTargets(request_context, instance_group_name, unique_targets);
    }
    for (size_t i = 0; i < result_count; ++i) {
        const auto &mark = mark_results[i];
        if (!mark.HasValidMark()) {
            continue;
        }
        StorageTargetAdmissionResult admission;
        if (const auto it = target_indexes.find(mark.target);
            it != target_indexes.end() && it->second < admissions.size()) {
            admission = admissions[it->second];
        } else if (data_storage_manager != nullptr) {
            const auto backend = data_storage_manager->GetDataStorageBackend(mark.target);
            if (backend == nullptr) {
                admission.status = StorageTargetAdmissionStatus::kNotFound;
                admission.ec = EC_NOENT;
            } else if (!backend->Available()) {
                admission.status = StorageTargetAdmissionStatus::kUnavailable;
                admission.ec = EC_NOENT;
            } else {
                admission.status = StorageTargetAdmissionStatus::kAllowed;
                admission.ec = EC_OK;
                admission.type = backend->GetType();
            }
        }
        if (admission.Allowed()) {
            out_targets[i] = mark.target;
            continue;
        }

        bool cleared = false;
        if (admission.status == StorageTargetAdmissionStatus::kNotFound && migration_manager != nullptr) {
            cleared =
                migration_manager->ClearTieredWriteMarkIfMatch(instance_id, keys[i], mark.target, mark.deadline_ms);
        }
        KVCM_LOG_WARN("trace_id [%s] instance [%s] block [%ld] tiered target storage [%s] rejected: %s; "
                      "conditional mark clear [%s], use ordinary write policy",
                      request_context->trace_id().c_str(),
                      instance_id.c_str(),
                      keys[i],
                      mark.target.c_str(),
                      StorageTargetAdmissionStatusName(admission.status),
                      cleared ? "succeeded" : "skipped");
    }
}

} // namespace

CacheManager::CacheManager(std::shared_ptr<MetricsRegistry> metrics_registry,
                           std::shared_ptr<RegistryManager> registry_manager,
                           std::shared_ptr<MetricsLifecycle> metrics_lifecycle)
    : meta_indexer_manager_(std::make_shared<MetaIndexerManager>())
    , write_location_manager_(std::make_shared<WriteLocationManager>())
    , meta_searcher_manager_(std::make_shared<MetaSearcherManager>(registry_manager, meta_indexer_manager_))
    , data_storage_selector_(std::make_shared<DataStorageSelector>(meta_indexer_manager_, registry_manager))
    , metrics_registry_(std::move(metrics_registry))
    , registry_manager_(std::move(registry_manager))
    , metrics_lifecycle_(metrics_lifecycle ? std::move(metrics_lifecycle) : std::make_shared<MetricsLifecycle>())
    , metrics_recorder_(std::make_shared<CacheManagerMetricsRecorder>(
          meta_indexer_manager_, write_location_manager_, registry_manager_, metrics_lifecycle_)) {}

CacheManager::~CacheManager() {
    if (cache_garbage_collector_) {
        cache_garbage_collector_->Stop();
        cache_garbage_collector_.reset();
    }
    ClearEventCleanupCallbacks();
    StopRecoverRetryLoop();
    DeactivateEventCleanupCallbacks();
    if (write_location_manager_) {
        write_location_manager_->Stop();
        write_location_manager_.reset();
    }
    if (cache_reclaimer_) {
        cache_reclaimer_->Stop();
        cache_reclaimer_.reset();
    }
    reclaimer_task_supervisor_.reset();
    // Background plans capture this CacheManager.  Stop and join their worker
    // threads before member destruction begins.
    schedule_plan_executor_.reset();
}

bool CacheManager::Init(int32_t schedule_plan_executor_thread_count,
                        uint64_t cache_reclaimer_key_sampling_size_total,
                        uint64_t cache_reclaimer_key_sampling_size_per_task,
                        uint64_t cache_reclaimer_del_batch_size,
                        uint32_t cache_reclaimer_idle_interval_ms,
                        uint32_t cache_reclaimer_worker_size,
                        CacheReclaimerAsyncDeleteConfig cache_reclaimer_async_delete_config,
                        uint32_t schedule_plan_migration_worker_budget,
                        uint32_t meta_query_worker_count,
                        std::size_t meta_query_parallel_threshold,
                        std::size_t meta_query_chunk_size,
                        CacheGarbageCollector::Config cache_gc_config) {
    if (schedule_plan_executor_thread_count <= 1 || schedule_plan_migration_worker_budget == 0 ||
        schedule_plan_migration_worker_budget >= static_cast<uint32_t>(schedule_plan_executor_thread_count)) {
        KVCM_LOG_ERROR("invalid schedule executor budget: worker_count=%d migration_worker_budget=%u",
                       schedule_plan_executor_thread_count,
                       schedule_plan_migration_worker_budget);
        return false;
    }
    if (meta_query_worker_count == 0 || meta_query_worker_count > 64 || meta_query_parallel_threshold == 0 ||
        meta_query_chunk_size == 0 || meta_query_chunk_size > meta_query_parallel_threshold ||
        !meta_indexer_manager_->ConfigureQueryExecutor(
            meta_query_worker_count, meta_query_parallel_threshold, meta_query_chunk_size)) {
        KVCM_LOG_ERROR("invalid meta query executor config: workers=%u threshold=%zu chunk_size=%zu",
                       meta_query_worker_count,
                       meta_query_parallel_threshold,
                       meta_query_chunk_size);
        return false;
    }
    schedule_plan_executor_ = std::make_shared<SchedulePlanExecutor>(schedule_plan_executor_thread_count,
                                                                     meta_indexer_manager_,
                                                                     registry_manager_->data_storage_manager(),
                                                                     metrics_registry_,
                                                                     schedule_plan_migration_worker_budget);
    event_manager_ = std::make_shared<EventManager>();
    if (!event_manager_) {
        KVCM_LOG_WARN("create EventManager failed");
    }
    if (!event_manager_->Init()) {
        KVCM_LOG_ERROR("event_manager init failed");
    }

    migration_manager_ = std::make_shared<MigrationManager>(schedule_plan_executor_,
                                                            meta_indexer_manager_,
                                                            registry_manager_->data_storage_manager(),
                                                            metrics_registry_,
                                                            event_manager_,
                                                            registry_manager_,
                                                            data_storage_selector_);
    // Invariant: migration_manager_ is always constructed here. Feature enablement is a per
    // instance-group property (IsTieredMigrationEnabled), never expressed via pointer nullness.
    assert(migration_manager_ != nullptr);
    registry_manager_->data_storage_manager()->SetAsyncCopyReferenceChecker(
        [weak_manager = std::weak_ptr<MigrationManager>(migration_manager_)](const std::string &storage_name) {
            const auto manager = weak_manager.lock();
            return manager && manager->HasAsyncCopyStorageReference(storage_name);
        });

    cache_garbage_collector_ = std::make_shared<CacheGarbageCollector>(std::move(cache_gc_config),
                                                                       registry_manager_,
                                                                       meta_indexer_manager_,
                                                                       registry_manager_->data_storage_manager(),
                                                                       schedule_plan_executor_,
                                                                       metrics_registry_,
                                                                       migration_manager_);
    if (cache_garbage_collector_->Validate() != EC_OK) {
        KVCM_LOG_ERROR("CacheManager init failed: invalid CacheGarbageCollector config");
        return false;
    }

    cache_reclaimer_ = std::make_shared<CacheReclaimer>(cache_reclaimer_key_sampling_size_total,
                                                        cache_reclaimer_key_sampling_size_per_task,
                                                        cache_reclaimer_del_batch_size,
                                                        cache_reclaimer_idle_interval_ms,
                                                        cache_reclaimer_worker_size,
                                                        registry_manager_,
                                                        meta_indexer_manager_,
                                                        meta_searcher_manager_,
                                                        schedule_plan_executor_,
                                                        metrics_registry_,
                                                        event_manager_,
                                                        write_location_manager_,
                                                        std::move(cache_reclaimer_async_delete_config),
                                                        migration_manager_);
    if (cache_reclaimer_->Start() != EC_OK) {
        KVCM_LOG_ERROR("CacheManager init failed");
        return false;
    }
    reclaimer_task_supervisor_ = std::make_unique<ReclaimerTaskSupervisor>(schedule_plan_executor_);
    reclaimer_task_supervisor_->Start();
    write_location_manager_->Start();
    metrics_recorder_->Start();
    KVCM_LOG_INFO("CacheManager init OK");
    return true;
}

std::string CacheManager::GetExtraInfo(RequestContext *request_context, const std::string &instance_id) {
    auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (!instance_info) {
        return "";
    }
    auto group_name = instance_info->instance_group_name();
    auto [ec, group] = registry_manager_->GetInstanceGroup(request_context, group_name);
    if (ec == EC_OK && group) {
        return group->extra_info();
    }
    return "";
}

void CacheManager::SetRevisitHistogramConfig(const std::vector<double> &boundaries) {
    if (meta_indexer_manager_ && metrics_registry_ && !boundaries.empty()) {
        meta_indexer_manager_->SetRevisitHistogramConfig(metrics_registry_, boundaries);
        KVCM_LOG_INFO("Set revisit histogram config with %zu boundaries", boundaries.size());
    }
}

std::pair<ErrorCode, std::string>
CacheManager::RegisterInstance(RequestContext *request_context,
                               const std::string &instance_group,
                               const std::string &instance_id,
                               int32_t block_size,
                               const std::vector<LocationSpecInfo> &location_spec_infos,
                               const ModelDeployment &model_deployment,
                               const std::vector<LocationSpecGroup> &location_spec_groups,
                               QueryType default_query_type) {
    SPAN_TRACER(request_context);
    // TODO : not thread safe now
    const auto &trace_id = request_context->trace_id();
    auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (instance_info) {
        auto mismatched = instance_info->MismatchFields(block_size,
                                                        location_spec_infos,
                                                        model_deployment,
                                                        location_spec_groups,
                                                        static_cast<int32_t>(default_query_type));
        if (instance_info->instance_group_name() != instance_group) {
            mismatched.insert(mismatched.begin(), "instance_group_name");
        }
        if (!mismatched.empty()) {
            auto mismatched_str = StringUtil::Join(mismatched, ", ");
            request_context->error_tracer()->AddErrorMsg(
                "register instance failed: instance_id '" + instance_id +
                "' already exists with different configuration, mismatched fields: [" + mismatched_str + "]");
            PREFIX_LOG(
                WARN, "register instance failed: duplicate instance, mismatched fields: [%s]", mismatched_str.c_str());
            return {EC_DUPLICATE_ENTITY, {}};
        }
        auto ec = TryCreateMetaSearcher(request_context, instance_id);
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, std::string, "register instance failed with errorcode: %d", ec);
        PREFIX_LOG(INFO, "register instance OK");
        return {ec, GetStorageConfigStr(request_context, instance_id)};
    }
    auto ec = registry_manager_->RegisterInstance(request_context,
                                                  instance_group,
                                                  instance_id,
                                                  block_size,
                                                  location_spec_infos,
                                                  model_deployment,
                                                  location_spec_groups,
                                                  static_cast<int32_t>(default_query_type));
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, std::string, "register instance failed with errorcode: %d", ec);
    ec = TryCreateMetaSearcher(request_context, instance_id);
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, std::string, "register instance failed with errorcode: %d", ec);
    PREFIX_LOG(INFO, "register instance OK");
    return {ec, GetStorageConfigStr(request_context, instance_id)};
}

ErrorCode CacheManager::RemoveInstance(RequestContext *request_context,
                                       const std::string &instance_group,
                                       const std::string &instance_id) {
    SPAN_TRACER(request_context);
    const auto &trace_id = request_context->trace_id();

    // drain 活跃迁移 copy 后再 trim，避免 trim 与 backend copy 竞态。
    // （trim 把 active copy 的 WRITING 目标 CAS→DELETING 删掉 / copy 成功后 promote 的 SERVING 目标被 trim 删）。
    // 步骤：draining gate（阻止该 instance 所有新 Copy，覆盖 reclaimer + admin 两路）
    //       → cancel 活跃 copy → 有界等待完成 → 删除并 trim。
    // RAII guard 保证 EndDraining 在任何出口（含宏 return）都执行，避免 draining set 泄漏。
    // 不暂停全局 Reclaimer：删除一个 instance 不应阻塞其他 instance，也不能覆盖 Server 生命周期的暂停状态。
    struct DrainGuard {
        CacheManager *mgr;
        std::string instance_id;
        bool active = false;
        ~DrainGuard() {
            if (active) {
                if (auto mm = mgr->migration_manager()) {
                    mm->EndDrainingInstance(instance_id);
                }
            }
        }
    } drain_guard{this, instance_id, false};

    if (migration_manager_ != nullptr) {
        migration_manager_->BeginDrainingInstance(instance_id); // 契约保证：happens-before 所有后续提交
        drain_guard.active = true;
        const auto active_keys = migration_manager_->GetActiveBlockKeysForInstance(instance_id);
        if (!active_keys.empty()) {
            migration_manager_->BatchCancel(instance_id, active_keys);
            // 有界等待：running/cancelling 任务由 monitor 在 copy future 完成后清理；快照中的
            // preparing 任务则已被置为 kPrepareCancelling，由提交线程在下一安全边界回滚。
            // draining gate 与锁内 reservation 保证快照不漏掉已准入任务，故无需重复 Cancel。
            constexpr int kDrainTimeoutMs = 5000;
            constexpr int kPollIntervalMs = 50;
            int waited_ms = 0;
            while (waited_ms < kDrainTimeoutMs) {
                if (migration_manager_->GetActiveBlockKeysForInstance(instance_id).empty()) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
                waited_ms += kPollIntervalMs;
            }
            if (!migration_manager_->GetActiveBlockKeysForInstance(instance_id).empty()) {
                KVCM_LOG_WARN("[%s] RemoveInstance drain timeout (%dms) for instance %s, "
                              "refusing removal while asynchronous Copy state is still owned",
                              trace_id.c_str(),
                              kDrainTimeoutMs,
                              instance_id.c_str());
            }
        }
        // A quarantined guard is intentionally invisible to the active task
        // table, and an accepted physical cleanup may outlive task completion.
        // Removing the registry entry here would make those durable guards
        // undiscoverable on the next leader and could close their backend while
        // it is still being used. Keep the instance registered until every
        // active/quarantined/cleanup reference has drained.
        if (migration_manager_->HasAsyncCopyInstanceReference(instance_id)) {
            PREFIX_LOG(WARN, "remove instance rejected: asynchronous Copy reference remains");
            return EC_EXIST;
        }
    }

    auto ec = registry_manager_->RemoveInstance(request_context, instance_group, instance_id);
    RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "remove instance failed"); // drain_guard 析构自动收尾

    InvalidateInstanceMetrics(instance_id);

    ec = TrimCache(request_context, instance_id, proto::meta::TrimStrategy::TS_REMOVE_ALL_CACHE);
    RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "remove instance failed"); // drain_guard 析构自动收尾
    PREFIX_LOG(INFO, "remove instance OK");
    return ec;
}

void CacheManager::InvalidateInstanceMetrics(const std::string &instance_id) const {
    if (instance_id.empty()) {
        return;
    }

    // callers must hold a unique lock on metrics_lifecycle_->mut_ while
    // invoking this, so that no producer (recorder publish span,
    // reporter ReportInterval, MetaServiceMetricsBase slow path) can
    // register new instance_id-tagged metrics during the steps below

    // 1) prune the recorder snapshot so the reporter cannot recreate
    //    entries on its next cycle
    if (metrics_recorder_) {
        metrics_recorder_->RemoveInstance(instance_id);
    }
    // 2) remove existing metrics entries from the registry
    if (metrics_registry_) {
        metrics_registry_->RemoveByTagFilter({{"instance_id", instance_id}});
    }
    // 3) evict cached per-instance collectors so subsequent requests
    //    do not resurrect entries through stale handles
    if (on_instance_removed_) {
        on_instance_removed_(instance_id);
    }
}

std::pair<ErrorCode, InstanceInfoConstPtr> CacheManager::GetInstanceInfo(RequestContext *request_context,
                                                                         const std::string &instance_id) {
    SPAN_TRACER(request_context);
    const auto &trace_id = request_context->trace_id();
    InstanceInfoConstPtr info_ptr = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (info_ptr == nullptr) {
        PREFIX_LOG(DEBUG, "get instance info failed");
        return {EC_INSTANCE_NOT_EXIST, nullptr};
    }
    return {EC_OK, info_ptr};
}

std::pair<ErrorCode, std::vector<InstanceInfoConstPtr>> ListInstanceInfo(RequestContext *request_context,
                                                                         const std::string &instance_group) {
    SPAN_TRACER(request_context);
    return {EC_OK, std::vector<InstanceInfoConstPtr>()};
}

std::pair<ErrorCode, CacheMetaVecWrapper> CacheManager::GetCacheMeta(RequestContext *request_context,
                                                                     const std::string &instance_id,
                                                                     const KeyVector &keys,
                                                                     const TokenIdsVector &tokens,
                                                                     const BlockMask &block_mask,
                                                                     int32_t detail_level /*TODO*/) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    auto [ec, meta_searcher] = CheckInputAndGetMetaSearcher(request_context, instance_id, keys, tokens);
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(DEBUG, ec, CacheMetaVecWrapper, "get cache meta failed");
    std::vector<CacheLocationMap> location_maps;
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, ManagerBatchGetLocation);
    if (!keys.empty()) {
        KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, request_key_count, keys.size());
        ec = meta_searcher->BatchGetLocation(request_context, keys, block_mask, location_maps);
    } else {
        auto [ec_temp, block_size] = GetBlockSize(request_context, instance_id);
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(DEBUG, ec_temp, CacheMetaVecWrapper, "get cache meta failed");
        auto gen_keys = GenKeyVector(tokens, block_size);
        KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, request_key_count, gen_keys.size());
        ec = meta_searcher->BatchGetLocation(request_context, gen_keys, block_mask, location_maps);
    }
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, ManagerBatchGetLocation);
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(DEBUG, ec, CacheMetaVecWrapper, "get cache meta failed: BatchGetLocation fail");
    // TODO, 现在BatchGetLocation接口还未返回 location properties 信息, 先置空
    // 另外现在BatchGetLocation接口返回的是一个block key对应的location map, 和proto定义不同,
    // 先临时只返回map里的第一个 location(不管是不是在serving状态), 将serving状态保存在meta里 这里现在非常 ugly
    CacheLocationVector cache_locations;
    std::vector<std::string> metas;
    std::map<std::string, std::string> meta;
    for (CacheLocationMap &location_map : location_maps) {
        auto iter = location_map.begin();
        if (iter != location_map.end() && iter->second) {
            cache_locations.push_back(iter->second);
            meta["id"] = cache_locations.back()->id();
        } else {
            auto not_found_loc = std::make_shared<CacheLocation>();
            not_found_loc->set_status(CacheLocationStatus::CLS_NOT_FOUND);
            cache_locations.push_back(std::move(not_found_loc));
        }
        meta["status"] = CacheLocation::CacheLocationStatusToString(cache_locations.back()->status());
        metas.push_back(Jsonizable::ToJsonString(meta));
    }

    return {ec, CacheMetaVecWrapper(std::move(metas), std::move(cache_locations))};
}

ErrorCode CacheManager::PerformCacheLocationQuery(RequestContext *request_context,
                                                  ServiceMetricsCollector *service_metrics_collector,
                                                  MetaSearcher *meta_searcher,
                                                  const std::string &instance_id,
                                                  QueryType query_type,
                                                  const KeyVector &keys,
                                                  const TokenIdsVector &tokens,
                                                  const BlockMask &block_mask,
                                                  int32_t sw_size,
                                                  KeyVector &query_keys,
                                                  CacheLocationVector &cache_locations) const {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    ErrorCode ec = EC_ERROR;
    if (!keys.empty()) {
        KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, request_key_count, keys.size());
        ec = GetCacheLocationByQueryType(
            meta_searcher, request_context, instance_id, query_type, keys, block_mask, sw_size, cache_locations);
    } else {
        auto [ec_temp, block_size] = GetBlockSize(request_context, instance_id);
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec_temp, "get block_size failed");
        auto gen_keys = GenKeyVector(tokens, block_size);
        KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, request_key_count, gen_keys.size());
        query_keys = gen_keys;
        ec = GetCacheLocationByQueryType(
            meta_searcher, request_context, instance_id, query_type, gen_keys, block_mask, sw_size, cache_locations);
    }
    return ec;
}

std::pair<ErrorCode, CacheLocationViewVecWrapper>
CacheManager::GetCacheLocation(RequestContext *request_context,
                               const std::string &instance_id,
                               QueryType query_type,
                               const KeyVector &keys,
                               const TokenIdsVector &tokens,
                               const BlockMask &block_mask,
                               int32_t sw_size,
                               const std::vector<std::string> &location_spec_names) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    auto [ec, meta_searcher] = CheckInputAndGetMetaSearcher(request_context, instance_id, keys, tokens);
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, CacheLocationViewVecWrapper, "check input or get meta searcher failed");
    if (query_type == QueryType::QT_UNSPECIFIED) {
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, EC_ERROR, CacheLocationViewVecWrapper, "unknown query type");
    }
    auto query_scope = (query_type == QueryType::QT_BATCH_GET)
                           ? KVCM_METRICS_COLLECTOR_CHRONO_SCOPE(service_metrics_collector, ManagerBatchGet)
                           : KVCM_METRICS_COLLECTOR_CHRONO_SCOPE(service_metrics_collector, ManagerPrefixMatch);
    CacheLocationVector cache_locations;
    KeyVector query_keys = keys;
    ec = PerformCacheLocationQuery(request_context,
                                   service_metrics_collector,
                                   meta_searcher,
                                   instance_id,
                                   query_type,
                                   keys,
                                   tokens,
                                   block_mask,
                                   sw_size,
                                   query_keys,
                                   cache_locations);
    query_scope = ChronoScopeGuard{};
    // prefix_match_len: count actual hits (non-empty id), not total returned entries.
    // BatchGet/ReverseRollSW pad misses with empty CacheLocation objects.
    {
        size_t match_len = 0;
        for (const auto &loc : cache_locations) {
            if (loc && !loc->id().empty()) {
                ++match_len;
            }
        }
        KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, prefix_match_len, match_len);
    }
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, CacheLocationViewVecWrapper, "get cache location failed");
    // accumulate hit/query block counters for hit-rate monitoring (only on success)
    if (service_metrics_collector) {
        size_t query_count = query_keys.size();
        size_t hit_count = 0;
        if (query_type == QueryType::QT_PREFIX_MATCH) {
            // PrefixMatch only returns matched blocks; size() == hit count
            hit_count = cache_locations.size();
        } else {
            // BatchGet / ReverseRollSW pad misses with empty CacheLocation
            for (const auto &loc : cache_locations) {
                if (loc && !loc->id().empty()) {
                    ++hit_count;
                }
            }
        }
        Counter query_counter, hit_counter;
        COPY_METRICS_(service_metrics_collector, manager, get_cache_location_query_block_counter, query_counter);
        COPY_METRICS_(service_metrics_collector, manager, get_cache_location_hit_block_counter, hit_counter);
        query_counter += query_count;
        hit_counter += hit_count;
    }
    FilterLocationSpecByName(cache_locations, location_spec_names);

    auto cache_get_event = std::make_shared<CacheGetEvent>(instance_id);
    cache_get_event->SetEventTriggerTime();
    cache_get_event->SetAddtionalArgs(
        QueryTypeToString(query_type), query_keys, tokens, block_mask, sw_size, location_spec_names);
    if (event_manager_) {
        event_manager_->Publish(cache_get_event);
    }
    return {ec, CacheLocationViewVecWrapper(std::move(cache_locations))};
}

void CacheManager::FillEmptyLocationSpecs(const std::vector<LocationSpecInfo> &location_spec_infos,
                                          CacheLocationVector &locations) {
    for (auto &location : locations) {
        if (!location || location->spec_size() == 0) {
            auto mutable_loc =
                location ? std::make_shared<CacheLocation>(*location) : std::make_shared<CacheLocation>();
            mutable_loc->set_spec_size(location_spec_infos.size());
            for (auto &spec_info : location_spec_infos) {
                mutable_loc->push_location_spec(LocationSpec(spec_info.name(), ""));
            }
            location = std::move(mutable_loc);
        }
    }
}

std::pair<ErrorCode, BatchLocationsView>
CacheManager::GetCacheLocationsByBackend(RequestContext *request_context,
                                         const std::string &instance_id,
                                         QueryType query_type,
                                         const KeyVector &keys,
                                         const TokenIdsVector &tokens,
                                         const BlockMask &block_mask,
                                         int32_t sw_size,
                                         const std::vector<std::string> &location_spec_names,
                                         const std::vector<BackendSelector> &backend_selectors) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    auto [ec, meta_searcher] = CheckInputAndGetMetaSearcher(request_context, instance_id, keys, tokens);
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, BatchLocationsView, "check input or get meta searcher failed");
    if (query_type != QueryType::QT_BATCH_GET) {
        request_context->error_tracer()->AddErrorMsg("GetCacheLocationsByBackend only supports QT_BATCH_GET");
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(
            WARN, EC_BADARGS, BatchLocationsView, "GetCacheLocationsByBackend only supports QT_BATCH_GET");
    }

    auto policy = genSelectLocationPolicy(request_context, instance_id);
    if (policy == nullptr) {
        request_context->error_tracer()->AddErrorMsg("gen select location policy failed");
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, EC_ERROR, BatchLocationsView, "gen select location policy failed");
    }

    KeyVector query_keys = keys;
    if (keys.empty()) {
        auto [ec_temp, block_size] = GetBlockSize(request_context, instance_id);
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec_temp, BatchLocationsView, "get block_size failed");
        query_keys = GenKeyVector(tokens, block_size);
    }

    const bool has_implicit_empty_mask =
        std::holds_alternative<BlockMaskVector>(block_mask) && std::get<BlockMaskVector>(block_mask).empty();
    if (!has_implicit_empty_mask && !IsBlockMaskValid(block_mask, query_keys.size())) {
        request_context->error_tracer()->AddErrorMsg("block_mask must match the number of query keys");
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(
            WARN, EC_BADARGS, BatchLocationsView, "block_mask must match the number of query keys");
    }

    if (!location_spec_names.empty()) {
        if (location_spec_names.size() != query_keys.size() ||
            std::any_of(location_spec_names.begin(), location_spec_names.end(), [](const std::string &name) {
                return name.empty();
            })) {
            request_context->error_tracer()->AddErrorMsg(
                "location_spec_names must be empty or contain one non-empty name per query key");
            RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(
                WARN, EC_BADARGS, BatchLocationsView, "invalid per-key location_spec_names");
        }
    }

    auto query_scope = KVCM_METRICS_COLLECTOR_CHRONO_SCOPE(service_metrics_collector, ManagerBatchGet);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, request_key_count, query_keys.size());

    if (backend_selectors.empty()) {
        request_context->error_tracer()->AddErrorMsg("backend_selectors must not be empty");
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, EC_BADARGS, BatchLocationsView, "backend_selectors must not be empty");
    }
    std::unordered_set<DataStorageType> selected_backend_types;
    for (const auto &selector : backend_selectors) {
        const auto backend_index = ToIndex(selector.backend_type);
        if (selector.backend_type == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN ||
            backend_index >= ToIndex(DataStorageType::COUNT)) {
            request_context->error_tracer()->AddErrorMsg("backend selector has invalid backend_type");
            RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(
                WARN, EC_BADARGS, BatchLocationsView, "backend selector has invalid backend_type");
        }
        if (!selected_backend_types.insert(selector.backend_type).second) {
            request_context->error_tracer()->AddErrorMsg("backend selector contains duplicate backend_type");
            RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(
                WARN, EC_BADARGS, BatchLocationsView, "backend selector contains duplicate backend_type");
        }
        switch (selector.strategy) {
        case LocationSelectStrategy::LSS_WEIGHTED_RANDOM:
            break;
        case LocationSelectStrategy::LSS_V6D_PREFIX:
        case LocationSelectStrategy::LSS_V6D_COVERAGE:
            if (selector.backend_type == DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2) {
                break;
            }
            [[fallthrough]];
        default:
            request_context->error_tracer()->AddErrorMsg("backend selector has invalid strategy for backend_type");
            RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(
                WARN, EC_BADARGS, BatchLocationsView, "backend selector has invalid strategy for backend_type");
        }
    }

    LocationsPerKey locations_per_key;
    ec = meta_searcher->BatchGetBestLocationByBackend(request_context,
                                                      query_keys,
                                                      locations_per_key,
                                                      policy.get(),
                                                      backend_selectors,
                                                      location_spec_names,
                                                      block_mask);
    query_scope = ChronoScopeGuard{};
    // prefix_match_len: count keys with at least one hit (non-empty id).
    // Miss keys have empty CacheLocation objects with no id.
    {
        size_t match_len = 0;
        for (const auto &key_locs : locations_per_key) {
            for (const auto &loc : key_locs) {
                if (loc && !loc->id().empty()) {
                    ++match_len;
                    break;
                }
            }
        }
        KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, prefix_match_len, match_len);
    }
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, BatchLocationsView, "batch get multi locations failed");

    auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (instance_info == nullptr) {
        request_context->error_tracer()->AddErrorMsg("instance not found");
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, EC_INSTANCE_NOT_EXIST, BatchLocationsView, "instance not found");
    }
    for (auto &key_locs : locations_per_key) {
        FillEmptyLocationSpecs(instance_info->location_spec_infos(), key_locs);
    }
    if (!location_spec_names.empty()) {
        for (size_t i = 0; i < locations_per_key.size(); ++i) {
            FilterLocationSpecByName(locations_per_key[i], {location_spec_names[i]});
        }
    }

    auto cache_get_event = std::make_shared<CacheGetEvent>(instance_id);
    cache_get_event->SetEventTriggerTime();
    cache_get_event->SetAddtionalArgs(
        QueryTypeToString(query_type), query_keys, tokens, block_mask, sw_size, location_spec_names);
    if (event_manager_) {
        event_manager_->Publish(cache_get_event);
    }

    BatchLocationsView result;
    result.reserve(locations_per_key.size());
    for (auto &key_locs : locations_per_key) {
        result.emplace_back(std::move(key_locs));
    }
    return {EC_OK, std::move(result)};
}

std::pair<ErrorCode, int64_t> CacheManager::GetCacheLocationLen(RequestContext *request_context,
                                                                const std::string &instance_id,
                                                                QueryType query_type,
                                                                const KeyVector &keys,
                                                                const TokenIdsVector &tokens,
                                                                int32_t sw_size) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    auto [ec, meta_searcher] = CheckInputAndGetMetaSearcher(request_context, instance_id, keys, tokens);
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, int64_t, "check input or get meta searcher failed");
    if (query_type == QueryType::QT_UNSPECIFIED) {
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, EC_ERROR, int64_t, "unknown query type");
    }
    CacheLocationVector cache_locations;
    KeyVector query_keys = keys;
    ec = PerformCacheLocationQuery(request_context,
                                   service_metrics_collector,
                                   meta_searcher,
                                   instance_id,
                                   query_type,
                                   keys,
                                   tokens,
                                   BlockMask(),
                                   sw_size,
                                   query_keys,
                                   cache_locations);
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, int64_t, "get cache location length failed");
    int64_t cache_location_len = 0;
    switch (query_type) {
    case QueryType::QT_BATCH_GET:
    case QueryType::QT_REVERSE_ROLL_SW_MATCH: {
        for (const auto &loc_ptr : cache_locations) {
            if (!loc_ptr) {
                continue;
            }
            bool has_valid_uri = false;
            for (const auto &spec : loc_ptr->location_specs()) {
                if (!spec.uri().empty()) {
                    has_valid_uri = true;
                    break;
                }
            }
            if (has_valid_uri) {
                cache_location_len++;
            }
        }
        break;
    }
    case QueryType::QT_PREFIX_MATCH: {
        cache_location_len = static_cast<int64_t>(cache_locations.size());
        break;
    }
    default:
        break;
    }
    auto cache_get_event = std::make_shared<CacheGetEvent>(instance_id);
    cache_get_event->SetEventTriggerTime();
    cache_get_event->SetAddtionalArgs(QueryTypeToString(query_type), query_keys, tokens, BlockMask(), sw_size, {});
    if (event_manager_) {
        event_manager_->Publish(cache_get_event);
    }
    return {ec, cache_location_len};
}

std::pair<ErrorCode, StartWriteCacheInfo>
CacheManager::StartWriteCache(RequestContext *request_context,
                              const std::string &instance_id,
                              const KeyVector &keys,
                              const TokenIdsVector &tokens,
                              const std::vector<std::string> &location_spec_group_names,
                              int64_t write_timeout_seconds,
                              int32_t min_replica_count) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    if (!location_spec_group_names.empty()) {
        // The group names are per *block*. A token-only request carries no
        // block keys (they are generated from the tokens below), so the block
        // count must come from the tokens in that case -- otherwise a valid
        // token-only request with group names is always rejected against a
        // key count of 0.
        size_t block_count = keys.size();
        if (block_count == 0 && !tokens.empty()) {
            auto [ec_bs, block_size] = GetBlockSize(request_context, instance_id);
            RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec_bs, StartWriteCacheInfo, "start write cache failed");
            block_count = tokens.size() / block_size;
        }
        auto check_ec =
            CheckLocationSpecGroupNames(request_context, instance_id, block_count, location_spec_group_names);
        RETURN_IF_EC_NOT_OK_WITH_TYPE(check_ec, StartWriteCacheInfo);
    }
    auto [ec, meta_searcher] = CheckInputAndGetMetaSearcher(request_context, instance_id, keys, tokens);
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, StartWriteCacheInfo, "start write cache failed");
    CacheLocationVector new_locations;
    BlockMask block_mask;
    KeyVector new_keys;
    std::vector<std::string_view> new_location_spec_group_names;
    std::vector<std::string> new_keys_tiered_targets;
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, ManagerFilterWriteCache);
    KeyVector query_keys = keys;

    ErrorCode filter_ec;
    if (!keys.empty()) {
        KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, request_key_count, keys.size());
        filter_ec = FilterWriteCache(request_context,
                                     instance_id,
                                     meta_searcher,
                                     keys,
                                     new_keys,
                                     location_spec_group_names,
                                     new_location_spec_group_names,
                                     block_mask,
                                     min_replica_count,
                                     new_keys_tiered_targets);
    } else {
        auto [ec_temp, block_size] = GetBlockSize(request_context, instance_id);
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec_temp, StartWriteCacheInfo, "start write cache failed");
        auto gen_keys = GenKeyVector(tokens, block_size);
        query_keys = gen_keys;
        KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, request_key_count, gen_keys.size());
        filter_ec = FilterWriteCache(request_context,
                                     instance_id,
                                     meta_searcher,
                                     gen_keys,
                                     new_keys,
                                     location_spec_group_names,
                                     new_location_spec_group_names,
                                     block_mask,
                                     min_replica_count,
                                     new_keys_tiered_targets);
    }
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, ManagerFilterWriteCache);
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, filter_ec, StartWriteCacheInfo, "filter write cache failed");

    std::vector<std::string> location_ids;
    std::string write_session_id = StringUtil::GenerateRandomString(32);
    if (new_keys.empty()) {
        // if no new keys, delete this write_session_id as soon as possible
        write_timeout_seconds = 10; // seconds
    } else {
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, StartWriteCacheInfo, "start write cache failed");
        KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, GenWriteLocation);
        ec = GenWriteLocation(request_context,
                              instance_id,
                              new_keys,
                              new_location_spec_group_names,
                              new_keys_tiered_targets,
                              new_locations);
        KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, GenWriteLocation);
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, StartWriteCacheInfo, "start write cache failed");
        KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, ManagerBatchAddLocation);
        std::vector<MetaSearcher::AddLocationResult> add_results;
        ec = meta_searcher->BatchAddLocation(request_context, new_keys, new_locations, add_results);
        KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, ManagerBatchAddLocation);
        if (ec != EC_OK) {
            RollbackAddLocations(request_context, instance_id, new_keys, new_locations, add_results);
        } else {
            location_ids.reserve(add_results.size());
            for (const auto &add_result : add_results) {
                location_ids.push_back(add_result.location_id);
            }
            RecordWriteBytesForLocations(new_locations);  // 记录写入量
        }
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, StartWriteCacheInfo, "start write cache failed");
    }
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, PutWriteLocationManager);
    constexpr int64_t kMaxWriteTimeoutSeconds = 1800;
    write_location_manager_->Put(
        write_session_id,
        std::move(new_keys),
        std::move(location_ids),
        std::min(kMaxWriteTimeoutSeconds, write_timeout_seconds),
        [this, trace_id, instance_id, write_session_id](
            std::unique_ptr<WriteLocationManager::WriteLocationInfo> write_location_info) {
            RequestContext temp_request_context(trace_id + "_timeout_callback");
            BlockMaskOffset succeed_block = 0;
            auto ec = this->FinishWriteCache(
                &temp_request_context, instance_id, write_session_id, succeed_block, std::move(write_location_info));
            static_cast<void>(ec);
        });
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, PutWriteLocationManager);
    auto start_write_event = std::make_shared<StartWriteCacheEvent>(instance_id);
    start_write_event->SetEventTriggerTime();
    start_write_event->SetAddtionalArgs(
        write_session_id, query_keys, tokens, block_mask, location_spec_group_names, write_timeout_seconds);
    if (event_manager_) {
        event_manager_->Publish(start_write_event);
    }
    return {EC_OK,
            StartWriteCacheInfo(std::move(write_session_id),
                                std::move(block_mask),
                                CacheLocationViewVecWrapper(std::move(new_locations)))};
}

void CacheManager::RollbackAddLocations(RequestContext *request_context,
                                        const std::string &instance_id,
                                        const KeyVector &keys,
                                        const CacheLocationVector &locations,
                                        const std::vector<MetaSearcher::AddLocationResult> &add_results) {
    const std::string &trace_id = request_context->trace_id();
    if (keys.size() != locations.size() || keys.size() != add_results.size()) {
        PREFIX_LOG(ERROR,
                   "rollback add locations size mismatch, keys[%lu], locations[%lu], results[%lu]",
                   keys.size(),
                   locations.size(),
                   add_results.size());
        return;
    }

    MetaSearcher *meta_searcher = meta_searcher_manager_->GetMetaSearcher(instance_id);
    MetaSearcher::AddLocationRollbackPlan plan;
    if (!meta_searcher) {
        // 无元数据访问时无法 reconcile uncertain 项，只能保留其 URI；但
        // confirmed-success 与无 ID 项的清理不依赖元数据，照常执行。
        const size_t uncertain_count = MetaSearcher::ClassifyAddLocationRollback(keys, add_results, plan);
        PREFIX_LOG(ERROR,
                   "rollback add locations without meta searcher: uncertain URIs retained, uncertain_count[%lu], "
                   "key_count[%lu]",
                   uncertain_count,
                   keys.size());
    } else if (meta_searcher->ReconcileAddLocationRollback(request_context, keys, add_results, plan) != EC_OK) {
        PREFIX_LOG(ERROR, "rollback add locations reconcile failed, key_count[%lu]", keys.size());
        return;
    }

    // 已成功写入元数据的 location 复用标准删除流水线：标记 DELETING、Sync、删 URI、删元数据。
    if (!plan.pipeline_keys.empty()) {
        CacheLocationDelRequest metadata_rollback{.instance_id = instance_id, .delay = std::chrono::seconds(0)};
        metadata_rollback.block_keys = plan.pipeline_keys;
        metadata_rollback.location_ids.reserve(plan.pipeline_location_ids.size());
        for (const auto &location_id : plan.pipeline_location_ids) {
            metadata_rollback.location_ids.push_back({location_id});
        }
        reclaimer_task_supervisor_->Submit(trace_id, std::move(metadata_rollback));
    }

    std::map<std::string, std::vector<DataStorageUri>> direct_delete_uris;
    auto queue_uri_delete = [&](size_t index) {
        if (!locations[index]) {
            PREFIX_LOG(WARN, "rollback add location has null location, key[%lu](%lu)", index, keys[index]);
            return;
        }
        for (const auto &spec : locations[index]->location_specs()) {
            DataStorageUri uri(spec.uri());
            if (!uri.Valid()) {
                PREFIX_LOG(WARN,
                           "rollback add location has invalid uri, key[%lu](%lu), uri[%s]",
                           index,
                           keys[index],
                           spec.uri().c_str());
                continue;
            }
            direct_delete_uris[uri.GetHostName()].push_back(std::move(uri));
        }
    };
    for (const size_t index : plan.direct_delete_indices) {
        queue_uri_delete(index);
    }
    if (direct_delete_uris.empty()) {
        return;
    }

    auto data_storage_manager = registry_manager_->data_storage_manager();
    if (!data_storage_manager) {
        PREFIX_LOG(ERROR, "rollback add locations failed: data storage manager not found");
        return;
    }
    // 已确认元数据无引用的项直接释放其已分配 URI。
    for (const auto &[storage_name, uris] : direct_delete_uris) {
        const auto delete_results = data_storage_manager->Delete(request_context, storage_name, uris, nullptr);
        if (delete_results.size() != uris.size()) {
            PREFIX_LOG(WARN,
                       "rollback data uri result size mismatch, storage[%s], expect[%lu], actual[%lu]",
                       storage_name.c_str(),
                       uris.size(),
                       delete_results.size());
        }
        const size_t result_count = std::min(delete_results.size(), uris.size());
        for (size_t i = 0; i < result_count; ++i) {
            if (delete_results[i] != EC_OK && delete_results[i] != EC_NOENT) {
                PREFIX_LOG(WARN,
                           "rollback data uri failed, storage[%s], uri[%s], ec[%d]",
                           storage_name.c_str(),
                           uris[i].ToUriString().c_str(),
                           delete_results[i]);
            }
        }
    }
}

ErrorCode
CacheManager::FinishWriteCache(RequestContext *request_context,
                               const std::string &instance_id,
                               const std::string &write_session_id,
                               const BlockMask &success_block_mask,
                               std::unique_ptr<WriteLocationManager::WriteLocationInfo> write_location_info_internal) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    WriteLocationManager::WriteLocationInfo location_info;
    if (write_location_info_internal != nullptr) {
        location_info = std::move(*write_location_info_internal);
    } else if (!write_location_manager_->GetAndDelete(write_session_id, location_info)) {
        request_context->error_tracer()->AddErrorMsg("write_session_id has been deleted");
        RETURN_IF_EC_NOT_OK_WITH_LOG(
            WARN, EC_ERROR, "finish write cache failed: write_session_id not found: %s", write_session_id.c_str());
    }
    if (!IsBlockMaskValid(success_block_mask, location_info.keys.size())) {
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN,
                                     EC_BADARGS,
                                     "invalid block mask, mask type: %zu, size: %zu",
                                     success_block_mask.index(),
                                     location_info.keys.size());
    }

    MetaSearcher *meta_searcher = meta_searcher_manager_->GetMetaSearcher(instance_id);
    if (!meta_searcher) {
        request_context->error_tracer()->AddErrorMsg("instance not exist");
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, EC_INSTANCE_NOT_EXIST, "finish write cache failed: meta searcher not found");
    }
    std::vector<KeyType> success_batch_keys;
    std::vector<std::string> success_batch_location_ids;
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> success_batch_update_tasks;
    CacheLocationDelRequest failed_del_request{.instance_id = instance_id, .delay = std::chrono::seconds(0)};

    for (size_t block_key_idx = 0; block_key_idx < location_info.keys.size(); block_key_idx++) {
        if (IsIndexInMaskRange(success_block_mask, block_key_idx)) {
            // success
            success_batch_keys.push_back(location_info.keys[block_key_idx]);
            success_batch_location_ids.push_back(location_info.location_ids[block_key_idx]);
            success_batch_update_tasks.push_back(
                {{location_info.location_ids[block_key_idx], CacheLocationStatus::CLS_SERVING}});
        } else {
            // failed
            failed_del_request.block_keys.push_back(location_info.keys[block_key_idx]);
            failed_del_request.location_ids.push_back({location_info.location_ids[block_key_idx]});
        }
    }

    ErrorCode ec = ErrorCode::EC_OK;
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, manager, request_key_count, success_batch_keys.size());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, ManagerBatchUpdateLocation);
    std::vector<std::vector<ErrorCode>> out_batch_results;
    if (!success_batch_keys.empty()) {
        ec = meta_searcher->BatchUpdateLocationStatus(
            request_context, success_batch_keys, success_batch_update_tasks, out_batch_results);
        if (ec != EC_OK) {
            std::string detail_ec_str = MetaSearcher::BatchErrorCodeToStr(out_batch_results);
            PREFIX_LOG(WARN, "update location status failed, ec: %d, ec_batches: %s", ec, detail_ec_str.c_str());
        }
    }
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, ManagerBatchUpdateLocation);

    // 多层存储 Mark 消费完成：仅当本次 target location 成功 SERVING 后，按配置策略清标。
    // 仅处理启用了 tiered migration（配置了 migration_strategies）的 instance group，与
    // FilterWriteCache 的 mark 消费入口保持对称；非分层 group 无 mark 可清（admin 旁路打的标由 timeout 兜底）。
    const auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (!success_batch_keys.empty() && IsTieredMigrationEnabled(request_context, registry_manager_, instance_info)) {
        KeyVector mark_candidate_keys;
        std::vector<std::string> mark_candidate_location_ids;
        for (size_t i = 0; i < success_batch_keys.size(); ++i) {
            if (i >= out_batch_results.size() || out_batch_results[i].empty() ||
                out_batch_results[i][0] != ErrorCode::EC_OK) {
                continue;
            }
            mark_candidate_keys.push_back(success_batch_keys[i]);
            mark_candidate_location_ids.push_back(success_batch_location_ids[i]);
        }
        if (!mark_candidate_keys.empty()) {
            std::vector<MigrationManager::MarkQueryResult> tiered_targets;
            const auto mark_query_ec =
                migration_manager_->BatchGetTieredWriteTargets(instance_id, mark_candidate_keys, tiered_targets);
            if (mark_query_ec != EC_OK) {
                PREFIX_LOG(WARN,
                           "tiered mark query partially failed after write, ec %d; valid keys will still be cleared",
                           mark_query_ec);
            }
            std::vector<CacheLocationMap> loc_maps;
            static const BlockMask empty_block_mask = static_cast<size_t>(0);
            const auto get_loc_ec =
                meta_searcher->BatchGetLocation(request_context, mark_candidate_keys, empty_block_mask, loc_maps);
            if (get_loc_ec == EC_OK && loc_maps.size() == mark_candidate_keys.size()) {
                for (size_t i = 0; i < mark_candidate_keys.size(); ++i) {
                    if (i >= tiered_targets.size() || !tiered_targets[i].HasValidMark()) {
                        continue;
                    }
                    const auto *loc = FindLocationById(loc_maps[i], mark_candidate_location_ids[i]);
                    if (loc == nullptr || loc->status() != CacheLocationStatus::CLS_SERVING ||
                        !LocationHasSpecOnStorage(*loc, tiered_targets[i].target)) {
                        continue;
                    }
                    const auto clear_policy =
                        GetMigrationMarkClearPolicy(request_context, registry_manager_, instance_info);
                    if (clear_policy == MigrationMarkClearPolicy::CLEAR_ON_NEXT_WRITE_SUCCESS ||
                        (clear_policy == MigrationMarkClearPolicy::CLEAR_ON_FULL_BLOCK_COVERED &&
                         LocationsCoverFullBlockOnStorage(loc_maps[i], tiered_targets[i].target, instance_info))) {
                        // 按 target+deadline 条件清除，避免清掉后续同 block 新 mark。
                        migration_manager_->ClearTieredWriteMarkIfMatch(instance_id,
                                                                        mark_candidate_keys[i],
                                                                        tiered_targets[i].target,
                                                                        tiered_targets[i].deadline_ms);
                    }
                }
            } else {
                PREFIX_LOG(
                    WARN,
                    "skip tiered mark clear because reloading finished locations failed, ec %d, got %zu, want %zu",
                    get_loc_ec,
                    loc_maps.size(),
                    mark_candidate_keys.size());
            }
        }
    }

    if (!failed_del_request.block_keys.empty()) {
        reclaimer_task_supervisor_->Submit(request_context->trace_id(), std::move(failed_del_request));
        // no need to wait delete finish here
    }
    auto finish_write_event = std::make_shared<FinishWriteCacheEvent>(instance_id);
    finish_write_event->SetEventTriggerTime();
    finish_write_event->SetAddtionalArgs(write_session_id, success_block_mask);
    if (event_manager_) {
        event_manager_->Publish(finish_write_event);
    }
    return ec;
}

ErrorCode CacheManager::RemoveCache(RequestContext *request_context,
                                    const std::string &instance_id,
                                    const KeyVector &keys,
                                    const TokenIdsVector &tokens,
                                    const BlockMask &block_mask /*TODO*/) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    assert(schedule_plan_executor_);
    if (keys.empty() && tokens.empty()) {
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, EC_BADARGS, "remove cache failed: empty input");
    }
    CacheMetaDelRequest request;
    request.instance_id = instance_id;
    if (!keys.empty()) {
        request.block_keys = keys;
    } else {
        auto [ec, block_size] = GetBlockSize(request_context, instance_id);
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "remove cache failed");
        auto gen_keys = GenKeyVector(tokens, block_size);
        request.block_keys = std::move(gen_keys);
    }
    reclaimer_task_supervisor_->Submit(trace_id, std::move(request));
    return EC_OK;
}

ErrorCode CacheManager::TrimCache(RequestContext *request_context,
                                  const std::string &instance_id,
                                  const proto::meta::TrimStrategy &trim_strategy,
                                  std::int32_t begin_ts,
                                  std::int32_t end_ts) const noexcept {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();

    if (trim_strategy != proto::meta::TS_REMOVE_ALL_CACHE) {
        PREFIX_LOG(WARN, "trim strategy not implemented");
        return ErrorCode::EC_UNIMPLEMENTED;
    }

    const auto meta_indexer = meta_indexer_manager_->GetMetaIndexer(instance_id);
    if (meta_indexer == nullptr) {
        PREFIX_LOG(WARN, "meta indexer is nullptr");
        return ErrorCode::EC_INSTANCE_NOT_EXIST;
    }

    std::string cursor = SCAN_BASE_CURSOR;
    do {
        constexpr std::size_t limit = 256;
        std::string next_cursor;

        CacheMetaDelRequest request;
        request.instance_id = instance_id;

        if (const ErrorCode ec = meta_indexer->Scan(request_context, cursor, limit, next_cursor, request.block_keys);
            ec != ErrorCode::EC_OK) {
            // TODO (rui): cache reclaimer should reclaim the dangling blocks
            RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "trim cache failed");
        }

        reclaimer_task_supervisor_->Submit(trace_id, std::move(request));
        cursor = next_cursor;
    } while (cursor != SCAN_BASE_CURSOR);

    return ErrorCode::EC_OK;
}
void CacheManager::PauseReclaimer() { cache_reclaimer_->Pause(); }
void CacheManager::ResumeReclaimer() { cache_reclaimer_->Resume(); }

ErrorCode CacheManager::StartCacheGarbageCollector() {
    return cache_garbage_collector_ ? cache_garbage_collector_->Start() : EC_ERROR;
}

void CacheManager::RequestStopCacheGarbageCollector() {
    if (cache_garbage_collector_) {
        cache_garbage_collector_->RequestStop();
    }
}

void CacheManager::JoinCacheGarbageCollector() {
    if (cache_garbage_collector_) {
        cache_garbage_collector_->Join();
    }
}

ErrorCode CacheManager::StartMigrationManager() { return migration_manager_->Start(); }
void CacheManager::StopMigrationManager() { migration_manager_->Stop(); }

CacheManager::MigrateCacheResult CacheManager::MigrateCache(RequestContext *request_context,
                                                            const std::string &trace_id,
                                                            const std::string &instance_id,
                                                            const std::string &src_name,
                                                            const std::string &dst_name,
                                                            bool do_copy,
                                                            bool do_mark,
                                                            const std::vector<int64_t> &explicit_block_keys,
                                                            int64_t sample_count) {
    // 薄 facade：只做前置校验（依赖、instance、target storage 和 migration strategy），
    // 迁移编排（候选/meta/admission/dispatch/计数）下沉 MigrationManager::MigrateCache。
    MigrateCacheResult result;

    if (migration_manager_ == nullptr || meta_indexer_manager_ == nullptr) {
        result.ec = EC_ERROR;
        result.message = "migration manager not available";
        return result;
    }
    // instance 存在性前置（保持错误优先级：instance → target storage → migration strategy）。
    auto meta_indexer = meta_indexer_manager_->GetMetaIndexer(instance_id);
    if (meta_indexer == nullptr) {
        result.ec = EC_INSTANCE_NOT_EXIST;
        result.message = "instance not found: " + instance_id;
        return result;
    }

    // target storage 必须已注册：COPY 需在其上分配空间，MARK 需要它可被写路径满足。
    // 否则会静默降级（写回落热层）并留下永不被满足的 mark。上层直接拒绝，避免误导性的 accepted。
    {
        auto data_storage_manager = registry_manager_->data_storage_manager();
        if (data_storage_manager == nullptr || data_storage_manager->GetDataStorageBackend(dst_name) == nullptr) {
            result.ec = EC_BADARGS;
            result.message = "target storage not registered: " + dst_name;
            return result;
        }
    }

    // Admin Copy 与 Reclaimer 共享 instance group 级并发上限。这里取得 group 配置并传给
    // MigrationManager；最终的原子检查在 BatchSubmit 内完成，避免并发入口的 TOCTOU。
    const auto group_name = registry_manager_->GetInstanceGroupName(instance_id);
    auto [group_ec, instance_group] = registry_manager_->GetInstanceGroup(request_context, group_name);
    if (group_ec != EC_OK || instance_group == nullptr || instance_group->cache_config() == nullptr) {
        result.ec = group_ec == EC_OK ? EC_BADARGS : group_ec;
        result.message = "instance group or cache config not found: " + group_name;
        return result;
    }
    const auto cache_config = instance_group->cache_config();
    const auto configured_copy_concurrency = cache_config->migration_copy_max_concurrency();
    const std::size_t copy_max_concurrency =
        configured_copy_concurrency > 0 ? static_cast<std::size_t>(configured_copy_concurrency) : 0;
    int64_t mark_timeout_ms = MigrationMarkMethod::kDefaultTimeoutMs;

    // IsTieredMigrationEnabled 以“group 至少有一条 migration strategy”作为写路径消费 Mark 的
    // group 级开关，因此无 strategy 时打标会成为“成功的 no-op”，这里直接拒绝。Admin 显式 target
    // 不受 strategy route 约束；精确匹配到 enabled Mark route 只影响下方 timeout 的选择。
    if (do_mark) {
        const bool has_migration_strategy = !cache_config->migration_strategies().empty();
        if (!has_migration_strategy) {
            result.ec = EC_BADARGS;
            result.message = "MARK/BOTH requires a migration strategy configured on the instance group: " + instance_id;
            return result;
        }

        // Admin 保留显式指定任意已注册 target 的能力。精确匹配到启用 Mark 的 strategy 时复用其
        // timeout；没有匹配 route 时仍允许迁移，并使用默认 timeout。
        for (const auto &strategy : cache_config->migration_strategies()) {
            if (strategy != nullptr && strategy->source_storage_name() == src_name &&
                strategy->target_storage_name() == dst_name && strategy->methods().mark().enabled()) {
                mark_timeout_ms = strategy->methods().mark().timeout_ms();
                break;
            }
        }
    }

    // 前置通过 → 委派编排。meta_indexer 已取得并校验，直接传入避免二次查找。
    const auto domain = migration_manager_->MigrateCache(request_context,
                                                         trace_id,
                                                         group_name,
                                                         instance_id,
                                                         meta_indexer,
                                                         src_name,
                                                         dst_name,
                                                         do_copy,
                                                         do_mark,
                                                         explicit_block_keys,
                                                         sample_count,
                                                         copy_max_concurrency,
                                                         mark_timeout_ms);
    result.ec = domain.ec;
    result.accepted = domain.accepted;
    result.rejected = domain.rejected;
    result.message = domain.message;
    return result;
}

void CacheManager::FilterLocationSpecByName(CacheLocationVector &locations,
                                            const std::vector<std::string> &location_spec_names) {
    if (location_spec_names.empty()) {
        return;
    }

    const std::unordered_set<std::string> names_set(location_spec_names.begin(), location_spec_names.end());
    for (auto &loc_ptr : locations) {
        if (!loc_ptr) {
            continue;
        }
        std::vector<LocationSpec> new_specs;
        for (const auto &spec : loc_ptr->location_specs()) {
            if (names_set.count(spec.name()) == 0) {
                continue;
            }
            new_specs.push_back(spec);
        }
        // COW: copy, modify, replace
        auto new_loc = std::make_shared<CacheLocation>(*loc_ptr);
        new_loc->set_location_specs(std::move(new_specs));
        new_loc->set_spec_size(new_loc->location_specs().size());
        loc_ptr = std::move(new_loc);
    }
}

ErrorCode CacheManager::CheckLocationSpecGroupNames(RequestContext *request_context,
                                                    const std::string &instance_id,
                                                    size_t key_count,
                                                    const std::vector<std::string> &location_spec_group_names) {
    const std::string &trace_id = request_context->trace_id();
    if (key_count != location_spec_group_names.size()) {
        PREFIX_LOG(WARN,
                   "location_spec_group_names size not match, expect[%zu], real[%zu]",
                   key_count,
                   location_spec_group_names.size());
        return EC_ERROR;
    }
    auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (!instance_info) {
        return EC_OK;
    }
    const auto &groups = instance_info->location_spec_groups();
    std::set<std::string_view> checked;
    for (const auto &group_name : location_spec_group_names) {
        if (group_name.empty() || !checked.insert(group_name).second) {
            continue;
        }
        auto it = std::lower_bound(
            groups.begin(), groups.end(), group_name, [](const LocationSpecGroup &g, const std::string_view &name) {
                return g.name() < name;
            });
        if (it == groups.end() || it->name() != group_name) {
            PREFIX_LOG(WARN, "location_spec_group_name [%s] not found in registered groups", group_name.c_str());
            return EC_BADARGS;
        }
    }
    return EC_OK;
}

ErrorCode CacheManager::FilterWriteCache(RequestContext *request_context,
                                         const std::string &instance_id,
                                         MetaSearcher *meta_searcher,
                                         const KeyVector &keys,
                                         KeyVector &new_keys,
                                         const std::vector<std::string> &location_spec_group_names,
                                         std::vector<std::string_view> &new_location_spec_group_names,
                                         BlockMask &block_mask,
                                         int32_t min_replica_count,
                                         std::vector<std::string> &new_keys_tiered_targets) {
    if (min_replica_count > 1) {
        return FilterWriteCacheWithMinReplica(request_context,
                                              instance_id,
                                              meta_searcher,
                                              keys,
                                              new_keys,
                                              location_spec_group_names,
                                              new_location_spec_group_names,
                                              block_mask,
                                              min_replica_count,
                                              new_keys_tiered_targets);
    }
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    static const BlockMask empty_block_mask = static_cast<size_t>(0);
    std::vector<CacheLocationMap> location_maps;
    auto ec = meta_searcher->BatchGetLocation(request_context, keys, empty_block_mask, location_maps);
    RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "BatchGetLocation failed");
    assert(keys.size() == location_maps.size());
    auto policy = genSelectLocationPolicy(request_context, instance_id);
    if (!policy) {
        return EC_ERROR;
    }

    const auto check_loc_data_exist = GetCheckLocDataExistFunc(instance_id);
    const auto submit_del_req = GetSubmitDelReqFunc(instance_id);
    KeyVector prune_keys;
    std::vector<std::vector<std::string>> prune_loc_ids_vec;

    // Resolve instance_info for spec-group-aware filtering and full-block tiered target checks.
    // When location_spec_group_names is provided, we check per-spec-group
    // coverage instead of block-level existence, allowing complementary
    // locations (e.g. KV-only and Mamba-only) to coexist in the same block.
    // Fetched unconditionally: tiered_migration_enabled depends on it, so gating the fetch on
    // "tiered enabled" would be circular. (Equivalent to prior behavior, which always fetched.)
    std::shared_ptr<const InstanceInfo> instance_info =
        registry_manager_->GetInstanceInfo(request_context, instance_id);
    const bool tiered_migration_enabled = IsTieredMigrationEnabled(request_context, registry_manager_, instance_info);
    const std::vector<std::string> all_spec_names = BuildAllLocationSpecNames(instance_info);

    auto requestedSpecNames = [&](size_t i) -> const std::vector<std::string> * {
        if (!instance_info || i >= location_spec_group_names.size() || location_spec_group_names[i].empty()) {
            return nullptr;
        }
        const auto &groups = instance_info->location_spec_groups();
        auto it =
            std::lower_bound(groups.begin(),
                             groups.end(),
                             location_spec_group_names[i],
                             [](const LocationSpecGroup &g, const std::string_view &name) { return g.name() < name; });
        if (it == groups.end() || it->name() != location_spec_group_names[i]) {
            return nullptr;
        }
        return &it->spec_names();
    };
    auto existsForWrite =
        [&](size_t i, const CacheLocationMap &m, std::vector<std::string> &out_prune_loc_ids) -> bool {
        const auto *spec_names = requestedSpecNames(i);
        if (spec_names == nullptr) {
            return policy->ExistsForWrite(m, check_loc_data_exist, out_prune_loc_ids);
        }
        return policy->ExistsForWrite(m, *spec_names, check_loc_data_exist, out_prune_loc_ids);
    };

    // Single pass: decide whether each block can skip this write.
    std::vector<bool> skip_write_flags(location_maps.size());
    // 多层存储 Mark 消费：批量预取各 block 的冷层目标（一次元数据读，避免逐块往返）。
    // 命中（target 非空）时仅在目标冷 storage 尚无 SERVING/WRITING 副本时纳入待写集合，
    // 并记录其目标冷 storage，由 GenWriteLocation 按 block 路由到冷层。
    std::vector<std::string> tiered_target_per_key(location_maps.size());
    if (tiered_migration_enabled) {
        std::vector<MigrationManager::MarkQueryResult> mark_results;
        const auto mark_query_ec = migration_manager_->BatchGetTieredWriteTargets(instance_id, keys, mark_results);
        if (mark_query_ec != EC_OK) {
            PREFIX_LOG(WARN,
                       "tiered mark query partially failed, ec %d; failed keys use ordinary write policy",
                       mark_query_ec);
        }
        ResolveUsableTieredWriteTargets(request_context,
                                        instance_id,
                                        instance_info->instance_group_name(),
                                        keys,
                                        mark_results,
                                        registry_manager_->data_storage_manager(),
                                        data_storage_selector_,
                                        migration_manager_,
                                        tiered_target_per_key);
    }
    for (size_t i = 0; i < location_maps.size(); ++i) {
        std::vector<std::string> prune_loc_ids;
        skip_write_flags[i] = existsForWrite(i, location_maps[i], prune_loc_ids);
        if (!prune_loc_ids.empty()) {
            prune_keys.emplace_back(keys[i]);
            prune_loc_ids_vec.emplace_back(prune_loc_ids);
        }
        if (!tiered_target_per_key[i].empty()) {
            const auto *spec_names = requestedSpecNames(i);
            if (spec_names == nullptr && !all_spec_names.empty()) {
                spec_names = &all_spec_names;
            }
            skip_write_flags[i] =
                spec_names == nullptr
                    ? HasServingOrWritingLocOnStorage(location_maps[i], tiered_target_per_key[i], {}, prune_loc_ids)
                    : HasServingOrWritingLocOnStorage(
                          location_maps[i], tiered_target_per_key[i], *spec_names, prune_loc_ids);
        }
    }
    if (!prune_keys.empty() && submit_del_req) {
        submit_del_req(prune_keys, prune_loc_ids_vec, {}, false);
    }

    // Find the first write and check whether all blocks that need writing form a suffix.
    size_t first_write_idx = location_maps.size();
    bool writes_form_suffix = true;
    for (size_t i = 0; i < location_maps.size(); ++i) {
        if (!skip_write_flags[i]) {
            if (first_write_idx == location_maps.size()) {
                first_write_idx = i;
            }
        } else if (first_write_idx != location_maps.size()) {
            // Found a skipped block after a block that needs writing, so writes do not form a suffix.
            writes_form_suffix = false;
            break;
        }
    }
    if (writes_form_suffix) {
        block_mask = static_cast<BlockMaskOffset>(first_write_idx);
        new_keys.insert(new_keys.end(), keys.begin() + first_write_idx, keys.end());
        new_keys_tiered_targets.insert(new_keys_tiered_targets.end(),
                                       tiered_target_per_key.begin() + first_write_idx,
                                       tiered_target_per_key.end());
        if (!location_spec_group_names.empty()) {
            new_location_spec_group_names.insert(new_location_spec_group_names.end(),
                                                 location_spec_group_names.begin() + first_write_idx,
                                                 location_spec_group_names.end());
        }
        return EC_OK;
    }
    block_mask = BlockMaskVector(location_maps.size(), false);
    for (size_t i = 0; i < location_maps.size(); ++i) {
        if (skip_write_flags[i]) {
            std::get<BlockMaskVector>(block_mask)[i] = true;
        } else {
            new_keys.push_back(keys[i]);
            new_keys_tiered_targets.push_back(tiered_target_per_key[i]);
            if (!location_spec_group_names.empty()) {
                new_location_spec_group_names.push_back(location_spec_group_names[i]);
            }
        }
    }
    return EC_OK;
}

ErrorCode CacheManager::FilterWriteCacheWithMinReplica(RequestContext *request_context,
                                                       const std::string &instance_id,
                                                       MetaSearcher *meta_searcher,
                                                       const KeyVector &keys,
                                                       KeyVector &new_keys,
                                                       const std::vector<std::string> &location_spec_group_names,
                                                       std::vector<std::string_view> &new_location_spec_group_names,
                                                       BlockMask &block_mask,
                                                       int32_t min_replica_count,
                                                       std::vector<std::string> &new_keys_tiered_targets) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    static const BlockMask empty_block_mask = static_cast<size_t>(0);
    std::vector<CacheLocationMap> location_maps;
    auto ec = meta_searcher->BatchGetLocation(request_context, keys, empty_block_mask, location_maps);
    RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "BatchGetLocation failed");
    assert(keys.size() == location_maps.size());
    auto policy = genSelectLocationPolicy(request_context, instance_id);
    if (!policy) {
        return EC_ERROR;
    }

    auto *weight_policy = dynamic_cast<WeightSLPolicy *>(policy.get());
    const auto check_loc_data_exist = GetCheckLocDataExistFunc(instance_id);
    const auto submit_del_req = GetSubmitDelReqFunc(instance_id);
    KeyVector prune_keys;
    std::vector<std::vector<std::string>> prune_loc_ids_vec;

    // Fetched unconditionally: tiered_migration_enabled depends on instance_info (see FilterWriteCache above).
    std::shared_ptr<const InstanceInfo> instance_info =
        registry_manager_->GetInstanceInfo(request_context, instance_id);
    const bool tiered_migration_enabled = IsTieredMigrationEnabled(request_context, registry_manager_, instance_info);
    const std::vector<std::string> all_spec_names = BuildAllLocationSpecNames(instance_info);

    static const std::vector<std::string> empty_spec_names;
    auto existsForWrite =
        [&](size_t i, const CacheLocationMap &m, std::vector<std::string> &out_prune_loc_ids) -> bool {
        const std::vector<std::string> *spec_names = &empty_spec_names;
        if (instance_info && i < location_spec_group_names.size() && !location_spec_group_names[i].empty()) {
            const auto &groups = instance_info->location_spec_groups();
            auto it = std::lower_bound(
                groups.begin(),
                groups.end(),
                location_spec_group_names[i],
                [](const LocationSpecGroup &g, const std::string_view &name) { return g.name() < name; });
            if (it != groups.end() && it->name() == location_spec_group_names[i]) {
                spec_names = &it->spec_names();
            }
        }
        if (weight_policy) {
            return weight_policy->ExistsForWriteWithMinCount(
                m, min_replica_count, *spec_names, check_loc_data_exist, out_prune_loc_ids);
        }
        return policy->ExistsForWrite(m, *spec_names, check_loc_data_exist, out_prune_loc_ids);
    };

    std::vector<bool> skip_write_flags(location_maps.size());
    std::vector<std::string> tiered_target_per_key(location_maps.size());
    std::vector<std::string> tiered_target_to_write(location_maps.size());
    if (tiered_migration_enabled) {
        std::vector<MigrationManager::MarkQueryResult> mark_results;
        const auto mark_query_ec = migration_manager_->BatchGetTieredWriteTargets(instance_id, keys, mark_results);
        if (mark_query_ec != EC_OK) {
            PREFIX_LOG(WARN,
                       "tiered mark query partially failed, ec %d; failed keys use ordinary min-replica policy",
                       mark_query_ec);
        }
        ResolveUsableTieredWriteTargets(request_context,
                                        instance_id,
                                        instance_info->instance_group_name(),
                                        keys,
                                        mark_results,
                                        registry_manager_->data_storage_manager(),
                                        data_storage_selector_,
                                        migration_manager_,
                                        tiered_target_per_key);
    }
    for (size_t i = 0; i < location_maps.size(); ++i) {
        std::vector<std::string> prune_loc_ids;
        skip_write_flags[i] = existsForWrite(i, location_maps[i], prune_loc_ids);
        if (!prune_loc_ids.empty()) {
            prune_keys.emplace_back(keys[i]);
            // 复制而非 move：下方 tiered 检查还要用 prune_loc_ids 排除 stale 目标副本。
            prune_loc_ids_vec.emplace_back(prune_loc_ids);
        }
        if (!tiered_target_per_key[i].empty()) {
            const std::vector<std::string> *spec_names = &empty_spec_names;
            if (instance_info && i < location_spec_group_names.size() && !location_spec_group_names[i].empty()) {
                const auto &groups = instance_info->location_spec_groups();
                auto it = std::lower_bound(
                    groups.begin(),
                    groups.end(),
                    location_spec_group_names[i],
                    [](const LocationSpecGroup &g, const std::string_view &name) { return g.name() < name; });
                if (it != groups.end() && it->name() == location_spec_group_names[i]) {
                    spec_names = &it->spec_names();
                }
            }
            if (spec_names->empty() && !all_spec_names.empty()) {
                spec_names = &all_spec_names;
            }
            const bool target_satisfied =
                spec_names->empty()
                    ? HasServingOrWritingLocOnStorage(location_maps[i], tiered_target_per_key[i], {}, prune_loc_ids)
                    : HasServingOrWritingLocOnStorage(
                          location_maps[i], tiered_target_per_key[i], *spec_names, prune_loc_ids);
            if (!target_satisfied) {
                skip_write_flags[i] = false;
                tiered_target_to_write[i] = tiered_target_per_key[i];
            }
        }
    }
    if (!prune_keys.empty() && submit_del_req) {
        submit_del_req(prune_keys, prune_loc_ids_vec, {}, false);
    }

    size_t first_write_idx = location_maps.size();
    bool writes_form_suffix = true;
    for (size_t i = 0; i < location_maps.size(); ++i) {
        if (!skip_write_flags[i]) {
            if (first_write_idx == location_maps.size()) {
                first_write_idx = i;
            }
        } else if (first_write_idx != location_maps.size()) {
            writes_form_suffix = false;
            break;
        }
    }
    if (writes_form_suffix) {
        block_mask = static_cast<BlockMaskOffset>(first_write_idx);
        new_keys.insert(new_keys.end(), keys.begin() + first_write_idx, keys.end());
        new_keys_tiered_targets.insert(new_keys_tiered_targets.end(),
                                       tiered_target_to_write.begin() + first_write_idx,
                                       tiered_target_to_write.end());
        if (!location_spec_group_names.empty()) {
            new_location_spec_group_names.insert(new_location_spec_group_names.end(),
                                                 location_spec_group_names.begin() + first_write_idx,
                                                 location_spec_group_names.end());
        }
        return EC_OK;
    }
    block_mask = BlockMaskVector(location_maps.size(), false);
    for (size_t i = 0; i < location_maps.size(); ++i) {
        if (skip_write_flags[i]) {
            std::get<BlockMaskVector>(block_mask)[i] = true;
        } else {
            new_keys.push_back(keys[i]);
            new_keys_tiered_targets.push_back(tiered_target_to_write[i]);
            if (!location_spec_group_names.empty()) {
                new_location_spec_group_names.push_back(location_spec_group_names[i]);
            }
        }
    }
    return EC_OK;
}

ErrorCode
CacheManager::CreateInSingleBatch(RequestContext *request_context,
                                  const std::string &instance_id,
                                  const CacheManager::KeyVector &keys,
                                  const std::vector<std::string_view> &location_spec_group_names,
                                  const std::shared_ptr<const InstanceInfo> &instance_info,
                                  const std::shared_ptr<DataStorageManager> &data_storage_manager,
                                  const std::string &unique_name,
                                  std::vector<DataStorageUri> &allocated_uris,
                                  std::vector<std::vector<std::pair<size_t, const LocationSpecInfo *>>> &key_to_uris,
                                  bool &is_create_success,
                                  int64_t common_size) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    std::vector<std::string> merged_block_keys;
    std::vector<size_t> merged_keys_idx;
    std::vector<const LocationSpecInfo *> spec_info_mapping;
    merged_block_keys.reserve(instance_info->location_spec_infos().size() * keys.size());
    merged_keys_idx.reserve(instance_info->location_spec_infos().size() * keys.size());
    spec_info_mapping.reserve(instance_info->location_spec_infos().size() * keys.size());

    for (const auto &spec_info : instance_info->location_spec_infos()) {
        if (location_spec_group_names.empty()) {
            for (size_t i = 0; i < keys.size(); i++) {
                std::string block_key = instance_id + "/" + spec_info.name() + "/" + StringUtil::Uint64ToHex(keys[i]);
                merged_block_keys.push_back(block_key);
                merged_keys_idx.push_back(i);
                spec_info_mapping.push_back(&spec_info);
            }
        } else {
            for (size_t i = 0; i < keys.size(); i++) {
                auto [ec, found] = IsSpecNameInSpecGroup(trace_id,
                                                         instance_id,
                                                         spec_info.name(),
                                                         location_spec_group_names[i],
                                                         instance_info->location_spec_groups());
                RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "IsSpecNameInSpecGroup failed");
                if (found) {
                    std::string block_key =
                        instance_id + "/" + spec_info.name() + "/" + StringUtil::Uint64ToHex(keys[i]);
                    merged_block_keys.push_back(block_key);
                    merged_keys_idx.push_back(i);
                    spec_info_mapping.push_back(&spec_info);
                }
            }
        }
    }

    std::vector<std::pair<ErrorCode, DataStorageUri>> results = data_storage_manager->Create(
        request_context, unique_name, merged_block_keys, common_size, []() { /* do nothing */ });

    for (size_t i = 0; i < results.size(); i++) {
        if (results[i].first == ErrorCode::EC_OK) {
            allocated_uris.push_back(results[i].second);
            key_to_uris[merged_keys_idx[i]].push_back({allocated_uris.size() - 1, spec_info_mapping[i]});
        }
    }
    // TODO: move check to another function
    if (results.size() != merged_block_keys.size()) {
        is_create_success = false;
        PREFIX_LOG(WARN,
                   "create data storage fail, results size:%ld, request size: %ld",
                   results.size(),
                   merged_block_keys.size());
    }
    for (auto &result : results) {
        if (result.first != ErrorCode::EC_OK) {
            is_create_success = false;
            PREFIX_LOG(WARN, "create data storage fail, ec_code: %d", result.first);
            break;
        }
    }
    return EC_OK;
}

ErrorCode CacheManager::CreateBySpec(RequestContext *request_context,
                                     const std::string &instance_id,
                                     const CacheManager::KeyVector &keys,
                                     const std::vector<std::string_view> &location_spec_group_names,
                                     const std::shared_ptr<const InstanceInfo> &instance_info,
                                     const std::shared_ptr<DataStorageManager> &data_storage_manager,
                                     const std::string &unique_name,
                                     std::vector<DataStorageUri> &allocated_uris,
                                     std::vector<std::vector<std::pair<size_t, const LocationSpecInfo *>>> &key_to_uris,
                                     bool &is_create_success) {
    // avoid use file across tp ranks
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    for (const auto &spec_info : instance_info->location_spec_infos()) {
        std::vector<std::string> block_keys;
        std::vector<size_t> keys_idx;
        block_keys.reserve(keys.size());
        keys_idx.reserve(keys.size());
        if (location_spec_group_names.empty()) {
            for (size_t i = 0; i < keys.size(); i++) {
                std::string block_key = instance_id + "/" + spec_info.name() + "/" + StringUtil::Uint64ToHex(keys[i]);
                block_keys.push_back(block_key);
                keys_idx.push_back(i);
            }
        } else {
            for (size_t i = 0; i < keys.size(); i++) {
                auto [ec, found] = IsSpecNameInSpecGroup(trace_id,
                                                         instance_id,
                                                         spec_info.name(),
                                                         location_spec_group_names[i],
                                                         instance_info->location_spec_groups());
                RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "IsSpecNameInSpecGroup failed");
                if (found) {
                    std::string block_key =
                        instance_id + "/" + spec_info.name() + "/" + StringUtil::Uint64ToHex(keys[i]);
                    block_keys.push_back(block_key);
                    keys_idx.push_back(i);
                }
            }
        }

        if (block_keys.empty()) {
            continue;
        }

        std::vector<std::pair<ErrorCode, DataStorageUri>> results = data_storage_manager->Create(
            request_context, unique_name, block_keys, spec_info.size(), []() { /* do nothing */ });

        for (size_t i = 0; i < results.size(); i++) {
            if (results[i].first == ErrorCode::EC_OK) {
                allocated_uris.push_back(results[i].second);
                key_to_uris[keys_idx[i]].push_back({allocated_uris.size() - 1, &spec_info});
            }
        }

        // TODO: move check to another function
        if (results.size() != block_keys.size()) {
            is_create_success = false;
            PREFIX_LOG(WARN,
                       "create data storage fail, results size:%ld, request size: %ld",
                       results.size(),
                       block_keys.size());
        }
        for (auto &result : results) {
            if (result.first != ErrorCode::EC_OK) {
                is_create_success = false;
                PREFIX_LOG(WARN, "create data storage fail, ec_code: %d", result.first);
                break;
            }
        }

        if (!is_create_success) {
            break;
        }
    }
    return EC_OK;
}

ErrorCode CacheManager::GenWriteLocationOnStorage(RequestContext *request_context,
                                                  const std::string &instance_id,
                                                  const CacheManager::KeyVector &keys,
                                                  const std::vector<std::string_view> &location_spec_group_names,
                                                  const std::shared_ptr<const InstanceInfo> &instance_info,
                                                  const std::shared_ptr<DataStorageManager> &data_storage_manager,
                                                  const std::string &storage_name,
                                                  DataStorageType storage_type,
                                                  CacheLocationVector &out_locations) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();

    if (keys.empty()) {
        return EC_OK;
    }

    std::vector<DataStorageUri> allocated_uris;
    allocated_uris.reserve(instance_info->location_spec_infos().size() * keys.size());
    std::vector<std::vector<std::pair<size_t, const LocationSpecInfo *>>> key_to_uris(keys.size());
    bool is_create_success = true;

    bool merge = instance_info->location_spec_infos().empty() ||
                 std::all_of(instance_info->location_spec_infos().begin() + 1,
                             instance_info->location_spec_infos().end(),
                             [&instance_info](const auto &spec) {
                                 return spec.size() == instance_info->location_spec_infos().front().size();
                             });
    int64_t common_size = merge && !instance_info->location_spec_infos().empty()
                              ? instance_info->location_spec_infos().front().size()
                              : 0;

    if (merge) {
        auto ec = CreateInSingleBatch(request_context,
                                      instance_id,
                                      keys,
                                      location_spec_group_names,
                                      instance_info,
                                      data_storage_manager,
                                      storage_name,
                                      allocated_uris,
                                      key_to_uris,
                                      is_create_success,
                                      common_size);
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "CreateInSingleBatch failed");
    } else {
        auto ec = CreateBySpec(request_context,
                               instance_id,
                               keys,
                               location_spec_group_names,
                               instance_info,
                               data_storage_manager,
                               storage_name,
                               allocated_uris,
                               key_to_uris,
                               is_create_success);
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "CreateBySpec failed");
    }

    if (!is_create_success) {
        request_context->error_tracer()->AddErrorMsg("some internal error when GenWriteLocation");
        auto error_codes =
            data_storage_manager->Delete(request_context, storage_name, allocated_uris, []() { /* do nothing */ });
        for (size_t i = 0; i < error_codes.size(); i++) {
            if (i >= allocated_uris.size()) {
                PREFIX_LOG(WARN,
                           "wrong error code num from Delete, results size:%ld, request size: %ld",
                           error_codes.size(),
                           allocated_uris.size());
                break;
            }
            if (error_codes[i] != ErrorCode::EC_OK) {
                PREFIX_LOG(WARN,
                           "delete data uri failed, storage unique name: %s, uri: %s",
                           storage_name.c_str(),
                           allocated_uris[i].ToUriString().c_str());
            }
        }
        return EC_ERROR;
    }

    for (const auto &uris : key_to_uris) {
        auto cache_location = std::make_shared<CacheLocation>();
        cache_location->set_type(storage_type);
        for (const auto &[data_storage_uri_idx, location_spec_info] : uris) {
            LocationSpec location_spec;
            location_spec.set_name(location_spec_info->name());
            location_spec.set_uri(allocated_uris[data_storage_uri_idx].ToUriString());
            cache_location->push_location_spec(std::move(location_spec));
        }
        cache_location->set_spec_size(uris.size());
        out_locations.push_back(std::move(cache_location));
    }
    return EC_OK;
}

ErrorCode CacheManager::GenWriteLocation(RequestContext *request_context,
                                         const std::string &instance_id,
                                         const CacheManager::KeyVector &keys,
                                         const std::vector<std::string_view> &location_spec_group_names,
                                         const std::vector<std::string> &tiered_targets,
                                         CacheLocationVector &new_locations) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();

    if (keys.empty()) {
        PREFIX_LOG(INFO, "new keys empty, no need to generate write location");
        return EC_OK;
    }

    auto data_storage_manager = registry_manager_->data_storage_manager();
    if (data_storage_manager == nullptr) {
        request_context->error_tracer()->AddErrorMsg("data storage manager not found");
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, EC_ERROR, "data storage manager not found");
    }

    auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (instance_info == nullptr) {
        request_context->error_tracer()->AddErrorMsg("instance not found");
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, EC_INSTANCE_NOT_EXIST, "instance not found");
    }

    // 按目标 storage 对 keys 分组：Mark 命中且目标 storage 存在 -> 路由到冷层；未命中 -> 默认 storage。
    // FilterWriteCache 已验证 target；若 backend 在两阶段之间消失，拒绝本次写并由重试重新过滤，
    // 不能 fallback 默认层，否则可能重复写一个原本已满足普通写条件的 block。
    const bool has_tiered = tiered_targets.size() == keys.size();
    std::map<std::string, std::vector<size_t>> indices_by_storage;
    std::map<std::string, DataStorageType> type_by_storage;
    std::vector<size_t> default_indices;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (has_tiered && !tiered_targets[i].empty()) {
            const auto backend = data_storage_manager->GetDataStorageBackend(tiered_targets[i]);
            if (backend != nullptr) {
                indices_by_storage[tiered_targets[i]].push_back(i);
                type_by_storage[tiered_targets[i]] = backend->GetType();
                continue;
            }
            PREFIX_LOG(WARN,
                       "tiered target storage [%s] disappeared after filtering, reject write instead of falling "
                       "back to default storage",
                       tiered_targets[i].c_str());
            return EC_NOENT;
        }
        default_indices.push_back(i);
    }
    if (!default_indices.empty()) {
        const auto select_result = data_storage_selector_->SelectCacheWriteDataStorageBackend(
            request_context, instance_info->instance_group_name());
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, select_result.ec, "select storage backend failed");
        for (const auto idx : default_indices) {
            indices_by_storage[select_result.name].push_back(idx);
        }
        type_by_storage[select_result.name] = select_result.type;
    }

    // 输出与 keys 同序对齐（CacheLocationVector 为 shared_ptr 向量，先填空）
    new_locations.resize(keys.size());
    auto collect_location_uris = [&](const CacheLocationVector &locations,
                                     std::map<std::string, std::vector<DataStorageUri>> &uris_by_storage) {
        for (const auto &location : locations) {
            if (location == nullptr) {
                continue;
            }
            for (const auto &location_spec : location->location_specs()) {
                DataStorageUri uri(location_spec.uri());
                if (!uri.Valid() || uri.GetHostName().empty()) {
                    PREFIX_LOG(WARN, "skip rollback invalid generated uri: %s", location_spec.uri().c_str());
                    continue;
                }
                uris_by_storage[uri.GetHostName()].push_back(std::move(uri));
            }
        }
    };
    auto rollback_generated_locations = [&](const CacheLocationVector *extra_locations = nullptr) {
        std::map<std::string, std::vector<DataStorageUri>> uris_by_storage;
        collect_location_uris(new_locations, uris_by_storage);
        if (extra_locations != nullptr) {
            collect_location_uris(*extra_locations, uris_by_storage);
        }
        for (const auto &[storage_name, uris] : uris_by_storage) {
            const auto results =
                data_storage_manager->Delete(request_context, storage_name, uris, []() { /* do nothing */ });
            if (results.size() != uris.size()) {
                PREFIX_LOG(WARN,
                           "rollback generated locations on storage %s returned %zu results, request size %zu",
                           storage_name.c_str(),
                           results.size(),
                           uris.size());
            }
            for (const auto ec : results) {
                if (ec != EC_OK) {
                    PREFIX_LOG(
                        WARN, "rollback generated location on storage %s failed, ec %d", storage_name.c_str(), ec);
                }
            }
        }
        new_locations.clear();
    };
    const bool has_sgn = !location_spec_group_names.empty();
    for (const auto &[storage_name, indices] : indices_by_storage) {
        KeyVector sub_keys;
        std::vector<std::string_view> sub_sgn;
        sub_keys.reserve(indices.size());
        if (has_sgn) {
            sub_sgn.reserve(indices.size());
        }
        for (size_t idx : indices) {
            sub_keys.push_back(keys[idx]);
            if (has_sgn) {
                sub_sgn.push_back(location_spec_group_names[idx]);
            }
        }
        CacheLocationVector sub_locations;
        auto ec = GenWriteLocationOnStorage(request_context,
                                            instance_id,
                                            sub_keys,
                                            sub_sgn,
                                            instance_info,
                                            data_storage_manager,
                                            storage_name,
                                            type_by_storage[storage_name],
                                            sub_locations);
        if (ec != EC_OK) {
            rollback_generated_locations(&sub_locations);
            RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "GenWriteLocationOnStorage failed");
        }
        if (sub_locations.size() != indices.size()) {
            PREFIX_LOG(WARN,
                       "sub_locations size %zu != indices size %zu for storage %s",
                       sub_locations.size(),
                       indices.size(),
                       storage_name.c_str());
            rollback_generated_locations(&sub_locations);
            return EC_ERROR;
        }
        for (size_t j = 0; j < indices.size(); ++j) {
            new_locations[indices[j]] = std::move(sub_locations[j]);
        }
    }
    return EC_OK;
}

void CacheManager::RecordWriteBytesForLocations(const CacheLocationVector &locations) {
    if (metrics_registry_ == nullptr) {
        return;
    }
    if (locations.empty()) {
        return;
    }
    auto data_storage_manager = registry_manager_->data_storage_manager();
    if (data_storage_manager == nullptr) {
        return;
    }
    std::map<std::string, std::uint64_t> bytes_by_storage;
    for (const auto &location : locations) {
        if (location == nullptr) {
            continue;
        }
        for (const auto &spec : location->location_specs()) {
            const DataStorageUri uri = DataStorageUri::FromUri(spec.uri());
            if (!uri.Valid() || uri.GetHostName().empty()) {
                continue;
            }
            std::uint64_t size = 0;
            uri.GetParamAs<std::uint64_t>("size", size);
            if (size == 0) {
                continue; // URI 无 size 参数或解析失败时 GetParamAs 保持 0，跳过
            }
            bytes_by_storage[uri.GetHostName()] += size;
        }
    }
    for (const auto &[unique_name, bytes] : bytes_by_storage) {
        data_storage_manager->RecordWriteBytes(unique_name, bytes);
    }
}

namespace {

std::shared_ptr<DataStorageBackend> LookupEventReportBackend(const std::shared_ptr<RegistryManager> &registry_manager,
                                                             const std::string &instance_id,
                                                             DataStorageType requested_type,
                                                             bool require_available = false) {
    if (!registry_manager || !registry_manager->data_storage_manager()) {
        return nullptr;
    }
    if (!IsEventReportStorageType(requested_type)) {
        return nullptr;
    }
    std::string group_name = registry_manager->GetInstanceGroupName(instance_id);
    if (group_name.empty()) {
        return nullptr;
    }
    auto ig = registry_manager->GetInstanceGroupConfig(group_name);
    if (!ig || ig->event_report_storage_candidates().empty()) {
        return nullptr;
    }
    auto dsm = registry_manager->data_storage_manager();
    for (const auto &candidate_name : ig->event_report_storage_candidates()) {
        auto backend = dsm->GetDataStorageBackend(candidate_name);
        auto *event_backend = dynamic_cast<EventReportBackend *>(backend.get());
        if (event_backend && event_backend->GetStorageType() == requested_type &&
            (!require_available || event_backend->Available())) {
            return backend;
        }
    }
    return nullptr;
}

bool IsCurrentEventReportBackend(const std::shared_ptr<RegistryManager> &registry_manager,
                                 const std::string &instance_id,
                                 DataStorageType requested_type,
                                 const std::shared_ptr<EventReportBackend> &expected_backend) {
    if (!expected_backend || expected_backend->GetStorageType() != requested_type) {
        return false;
    }
    // Match the same first-available-candidate rule used by ReportEvent and
    // GetHostCacheState. Merely remaining in the candidate list is not enough:
    // an older backend incarnation must never clean metadata owned by the
    // backend that currently wins routing for this storage tier.
    const auto current = LookupEventReportBackend(registry_manager, instance_id, requested_type, true);
    return current.get() == expected_backend.get();
}

bool ParseInt64(const std::string &s, int64_t &out) {
    if (s.empty() || s.front() == '+') {
        return false;
    }

    const bool negative = s.front() == '-';
    const char *begin = s.data() + (negative ? 1 : 0);
    const char *end = s.data() + s.size();
    if (begin == end) {
        return false;
    }

    uint64_t magnitude = 0;
    constexpr uint64_t kMaxMagnitude = std::numeric_limits<uint64_t>::max();
    for (const char *cursor = begin; cursor != end; ++cursor) {
        const unsigned char ch = static_cast<unsigned char>(*cursor);
        if (ch < '0' || ch > '9') {
            return false;
        }
        const uint64_t digit = ch - '0';
        if (magnitude > (kMaxMagnitude - digit) / 10) {
            return false;
        }
        magnitude = magnitude * 10 + digit;
    }

    constexpr uint64_t kSignBit = uint64_t{1} << 63;
    if (negative) {
        if (magnitude > kSignBit) {
            return false;
        }
        out = magnitude == kSignBit ? std::numeric_limits<int64_t>::min() : -static_cast<int64_t>(magnitude);
        return true;
    }

    // vLLM serializes its external block hash as an unsigned uint64 decimal.
    // Preserve that 64-bit pattern when KVCM stores and queries signed int64
    // keys instead of rejecting values above INT64_MAX.
    out = magnitude < kSignBit ? static_cast<int64_t>(magnitude)
                               : std::numeric_limits<int64_t>::min() + static_cast<int64_t>(magnitude - kSignBit);
    return true;
}

struct ValidatedEventLocationSpec {
    std::string_view name;
    std::string_view raw_uri;
    // Canonical ReportEvent URIs never need the heavyweight StandardUri
    // object. Allocate it only for the compatibility fallback so the common
    // validation result stays small and cheap to move through inline storage.
    std::unique_ptr<DataStorageUri> parsed_uri;
    CanonicalSnapshotUriAppendInfo canonical_uri;
    std::uint64_t size = 0;
    std::string versioned_uri;
    bool is_canonical_uri = false;

    bool AddPrevalidatedSnapshotVersion(const std::string &version) {
        if (is_canonical_uri) {
            return SnapshotUriUtils::AddPrevalidatedSnapshotVersionToCanonicalUri(
                raw_uri, canonical_uri, version, versioned_uri);
        }
        if (!parsed_uri) {
            return false;
        }
        versioned_uri = parsed_uri->ToUriStringWithExtraParam(SnapshotUriUtils::kSnapshotVersionParam, version);
        return !versioned_uri.empty();
    }
};

bool ValidateEventLocationSpec(const proto::meta::LocationSpec &spec, ValidatedEventLocationSpec &out) {
    out = {};
    if (spec.name().empty()) {
        return false;
    }
    out.name = spec.name();
    out.raw_uri = spec.uri();
    if (SnapshotUriUtils::ParseCanonicalUriForSnapshotAppend(out.raw_uri, out.canonical_uri)) {
        out.is_canonical_uri = true;
        out.size = out.canonical_uri.size;
        return true;
    }
    out.parsed_uri = std::make_unique<DataStorageUri>(spec.uri());
    if (!out.parsed_uri->Valid() || SnapshotUriUtils::HasEventReportInternalUriMetadata(*out.parsed_uri)) {
        return false;
    }
    out.parsed_uri->GetParamAs<std::uint64_t>("size", out.size);
    return true;
}

class ValidatedEventLocationSpecs {
public:
    void Push(ValidatedEventLocationSpec &&spec, size_t max_spec_count) {
        if (many_.empty() && !one_) {
            one_.emplace(std::move(spec));
            return;
        }
        if (one_) {
            many_.reserve(max_spec_count < 2 ? 2 : max_spec_count);
            many_.push_back(std::move(*one_));
            one_.reset();
        }
        many_.push_back(std::move(spec));
    }

    [[nodiscard]] size_t Size() const noexcept { return many_.size() + (one_ ? 1 : 0); }
    [[nodiscard]] ValidatedEventLocationSpec &At(size_t index) noexcept {
        return index < many_.size() ? many_[index] : *one_;
    }

private:
    std::optional<ValidatedEventLocationSpec> one_;
    std::vector<ValidatedEventLocationSpec> many_;
};

bool IsSnapshotLocationStale(const EventReportBackend *event_backend,
                             const std::string &instance_id,
                             const CacheLocation &location,
                             bool preserve_in_flight = false) {
    if (!event_backend) {
        return false;
    }

    std::string medium;
    std::string reporter_host;
    if (!event_backend->ParseLocationId(location.id(), medium, reporter_host)) {
        return false;
    }

    const ReporterSnapshotKey reporter_key{instance_id, reporter_host};
    std::string committed_version;
    std::string in_flight_version;
    event_backend->GetSnapshotVersionTokens(reporter_key, committed_version, in_flight_version);
    if (location.location_specs().empty()) {
        return true;
    }
    bool contains_committed = false;
    bool contains_in_flight = false;
    for (const auto &spec : location.location_specs()) {
        const size_t version_param_count =
            SnapshotUriUtils::CountUriParam(spec.uri(), SnapshotUriUtils::kSnapshotVersionParam);
        if (version_param_count == 0) {
            // Legacy metadata is a stale reconciliation component, but a
            // current delta spec in the same stable location still protects
            // the location from coarse-grained cleanup.
            continue;
        }
        SnapshotUriInfo info;
        if (version_param_count != 1 || !SnapshotUriUtils::ParseSnapshotUriInfo(spec.uri(), info)) {
            return true;
        }
        contains_committed = contains_committed || (!committed_version.empty() && info.version == committed_version);
        contains_in_flight = contains_in_flight ||
                             (preserve_in_flight && !in_flight_version.empty() && info.version == in_flight_version);
    }
    // Delta merge is spec-granular and can temporarily leave multiple
    // generations in one stable location. Cleanup is location-granular, so it
    // must preserve the whole location when any current/in-flight spec is
    // present; deleting stale sibling specs is deferred to a later complete
    // snapshot rather than risking a false negative for a successful delta.
    return !contains_committed && !contains_in_flight;
}

bool IsEventReportLocationReadable(const CacheLocation &location,
                                   bool strict_query_visibility,
                                   const std::string &committed_version) {
    if (location.location_specs().empty()) {
        return false;
    }
    bool contains_readable_version = !strict_query_visibility;
    const bool uri_structure_prevalidated = location.HasValidatedLocationSpecs();
    for (const auto &spec : location.location_specs()) {
        std::string_view snapshot_version;
        if (!SnapshotUriUtils::InspectSnapshotUriForVisibility(
                spec.uri(), snapshot_version, uri_structure_prevalidated)) {
            return false;
        }
        if (!snapshot_version.empty() && strict_query_visibility &&
            snapshot_version == std::string_view(committed_version)) {
            contains_readable_version = true;
        }
    }
    // Delta merge is spec-granular. A post-snapshot ADD may refresh one spec
    // in a stable location while an untouched sibling still carries an older
    // generation. The committed spec keeps that location readable; a
    // location composed entirely of candidate/older/legacy specs is fenced
    // until the candidate commits or the attempt fails back to soft mode.
    return contains_readable_version;
}

} // namespace

ErrorCode CacheManager::ReportEvent(RequestContext *request_context,
                                    const proto::meta::ReportEventRequest *request,
                                    proto::meta::ReportEventResponse *response) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    const std::string &instance_id = request->instance_id();
    const std::string &host_ip_port = request->host_ip_port();
    auto *response_status = response->mutable_header()->mutable_status();

    if (instance_id.empty() || !SnapshotUriUtils::IsValidLocationIdComponent(host_ip_port)) {
        KVCM_LOG_WARN("trace_id [%s] | ReportEvent: invalid instance_id or host_ip_port", trace_id.c_str());
        response_status->set_code(proto::meta::INVALID_ARGUMENT);
        response_status->set_message("invalid instance_id or host_ip_port");
        return EC_BADARGS;
    }
    if (request->events_size() == 0) {
        response_status->set_code(proto::meta::OK);
        return EC_OK;
    }

    bool has_snapshot_event = false;
    bool has_delta_event = false;
    bool has_host_down_event = false;
    int snapshot_event_count = 0;
    for (const auto &event : request->events()) {
        if (event.event_type() == proto::meta::EVENT_BLOCK_SNAPSHOT) {
            has_snapshot_event = true;
            ++snapshot_event_count;
        }
        has_delta_event = has_delta_event || event.event_type() == proto::meta::EVENT_BLOCK_ADD ||
                          event.event_type() == proto::meta::EVENT_BLOCK_DELETE;
        has_host_down_event = has_host_down_event || event.event_type() == proto::meta::EVENT_HOST_DOWN;
    }
    if (has_snapshot_event && has_delta_event) {
        response_status->set_code(proto::meta::INVALID_ARGUMENT);
        response_status->set_message("snapshot and delta mutations must use separate ReportEvent requests");
        return EC_BADARGS;
    }
    if (snapshot_event_count > 1) {
        response_status->set_code(proto::meta::INVALID_ARGUMENT);
        response_status->set_message("a ReportEvent request may contain only one complete snapshot");
        return EC_BADARGS;
    }
    if (has_host_down_event && request->events_size() != 1) {
        response_status->set_code(proto::meta::INVALID_ARGUMENT);
        response_status->set_message("host-down must be the only event in a ReportEvent request");
        return EC_BADARGS;
    }

    DataStorageType requested_type = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
    switch (request->storage_type()) {
    case proto::meta::ST_EVENT_REPORT_L1P5:
        requested_type = DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5;
        break;
    case proto::meta::ST_EVENT_REPORT_L2:
        requested_type = DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2;
        break;
    case proto::meta::ST_UNSPECIFIED:
        KVCM_LOG_WARN("trace_id [%s] | ReportEvent: storage_type is required but not specified", trace_id.c_str());
        response_status->set_code(proto::meta::INVALID_ARGUMENT);
        response_status->set_message("storage_type is required");
        return EC_BADARGS;
    default:
        // Do not cast the open protobuf enum directly to DataStorageType (whose
        // underlying type is uint8_t): an unknown wire value such as 263 would
        // truncate to 7 and be misrouted to the L1P5 backend.
        KVCM_LOG_WARN("trace_id [%s] | ReportEvent: unsupported event-report storage_type [%d]",
                      trace_id.c_str(),
                      static_cast<int>(request->storage_type()));
        response_status->set_code(proto::meta::INVALID_ARGUMENT);
        response_status->set_message("unsupported event-report storage_type: " +
                                     std::to_string(static_cast<int>(request->storage_type())));
        return EC_BADARGS;
    }

    auto event_backend_holder = LookupEventReportBackend(registry_manager_, instance_id, requested_type, true);
    auto event_backend = std::dynamic_pointer_cast<EventReportBackend>(event_backend_holder);
    if (!event_backend) {
        KVCM_LOG_WARN("trace_id [%s] | ReportEvent: EventReportBackend not found for instance [%s] type [%d]",
                      trace_id.c_str(),
                      instance_id.c_str(),
                      static_cast<int>(requested_type));
        response_status->set_code(proto::meta::INSTANCE_NOT_EXIST);
        response_status->set_message("EventReportBackend not found for instance: " + instance_id +
                                     ", type: " + ToString(requested_type));
        return EC_INSTANCE_NOT_EXIST;
    }
    if (event_backend->GetStorageType() != requested_type) {
        KVCM_LOG_WARN("trace_id [%s] | ReportEvent: storage_type mismatch for instance [%s], "
                      "requested [%d] but backend is [%d]",
                      trace_id.c_str(),
                      instance_id.c_str(),
                      static_cast<int>(requested_type),
                      static_cast<int>(event_backend->GetStorageType()));
        response_status->set_code(proto::meta::INVALID_ARGUMENT);
        response_status->set_message("storage_type mismatch");
        return EC_BADARGS;
    }

    const ReporterSnapshotKey reporter_key{instance_id, host_ip_port};
    bool request_created_generation = false;
    auto refresh_snapshot_response = [&](bool force_snapshot_required) {
        const std::string committed = event_backend->GetSnapshotVersion(reporter_key);
        response->set_committed_snapshot_version(committed);
        response->set_snapshot_required(committed.empty() || force_snapshot_required);
    };
    refresh_snapshot_response(false);

    if (!event_backend->IsCleanupCallbackSet()) {
        const std::weak_ptr<EventReportBackend> expected_backend = event_backend;
        const auto callback_state = event_cleanup_callback_state_;
        event_backend->SetCleanupCallback([this, requested_type, expected_backend, callback_state](
                                              const std::string &cleanup_instance,
                                              const std::string &down_host,
                                              uint64_t generation) {
            std::shared_lock<std::shared_mutex> callback_lease(callback_state->mutex);
            if (!callback_state->accepting) {
                return;
            }
            const uint64_t callback_epoch = callback_state->epoch;
            auto cleanup_backend = expected_backend.lock();
            if (!cleanup_backend) {
                return;
            }
            const auto cleanup = [this,
                                  cleanup_instance,
                                  down_host,
                                  generation,
                                  requested_type,
                                  cleanup_backend,
                                  callback_state,
                                  callback_epoch] {
                // A queued task may start after the backend callback that
                // submitted it has returned. Hold the same lifetime lease for
                // the whole cleanup so DoCleanup/destruction drains running
                // tasks and makes tasks that are still queued harmless.
                std::shared_lock<std::shared_mutex> task_lease(callback_state->mutex);
                if (!callback_state->accepting || callback_state->epoch != callback_epoch) {
                    return;
                }
                this->CleanupHostLocations(cleanup_instance, down_host, generation, requested_type, cleanup_backend);
            };
            if (!this->schedule_plan_executor_ || !this->schedule_plan_executor_->SubmitTask(cleanup)) {
                // This callback runs on EventReportBackend's liveness thread.
                // Running inline could enter DataStorageManager while an
                // unregister operation holds its write lock and waits for this
                // backend thread to join. The node is still removed from the
                // visibility table below, so dropping best-effort physical
                // metadata cleanup during executor shutdown is fail-closed.
                KVCM_LOG_WARN("ReportEvent liveness cleanup queue unavailable; skipping metadata cleanup for host [%s]",
                              down_host.c_str());
            }
        });
    }

    MetaSearcher *meta_searcher = meta_searcher_manager_->GetMetaSearcher(instance_id);
    if (!meta_searcher) {
        KVCM_LOG_WARN("trace_id [%s] | ReportEvent: meta searcher not found for instance [%s]",
                      trace_id.c_str(),
                      instance_id.c_str());
        response_status->set_code(proto::meta::INSTANCE_NOT_EXIST);
        response_status->set_message("meta searcher not found for instance: " + instance_id);
        return EC_INSTANCE_NOT_EXIST;
    }

    auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (!instance_info) {
        KVCM_LOG_WARN("trace_id [%s] | ReportEvent: instance info not found for instance [%s]",
                      trace_id.c_str(),
                      instance_id.c_str());
        response_status->set_code(proto::meta::INSTANCE_NOT_EXIST);
        response_status->set_message("instance info not found for instance: " + instance_id);
        return EC_INSTANCE_NOT_EXIST;
    }
    std::unordered_set<std::string_view> registered_spec_names;
    registered_spec_names.reserve(instance_info->location_spec_infos().size());
    for (const auto &spec_info : instance_info->location_spec_infos()) {
        registered_spec_names.insert(spec_info.name());
    }

    const int events_size = request->events_size();
    std::vector<ErrorCode> per_item_ec(events_size, EC_OK);
    DeltaMutationGuard delta_mutations(event_backend, reporter_key);
    uint64_t mutation_lifecycle_generation = 0;
    bool delta_snapshot_version_validated = false;

    bool has_heartbeat = false;
    bool has_host_down = false;
    std::vector<int> valid_heartbeat_indices;
    std::vector<std::string> register_mediums;
    std::map<std::string, std::string> heartbeat_status;
    bool registration_items_prepared = false;
    bool registration_applied = false;
    ErrorCode register_ec = EC_OK;
    bool node_registration_ensured = false;
    struct RequestMediumState {
        InternedLocationId location_id;
        bool registration_ensured = false;
    };
    // A report overwhelmingly uses one medium. Keep registration state and
    // the interned location id in one table, and retain the last node so the
    // common run of thousands of `mem` events performs no repeated string
    // hash or table probe. References to unordered_map elements survive a
    // rehash and every resulting CacheLocation retains the shared id.
    std::unordered_map<std::string, RequestMediumState> medium_states;
    medium_states.reserve(register_mediums.size() + 1);
    std::string_view last_medium;
    RequestMediumState *last_medium_state = nullptr;
    auto get_medium_state = [&](const std::string &medium) -> RequestMediumState & {
        if (last_medium_state != nullptr && last_medium == medium) {
            return *last_medium_state;
        }
        auto [it, inserted] = medium_states.try_emplace(medium);
        (void)inserted;
        last_medium = it->first;
        last_medium_state = &it->second;
        return *last_medium_state;
    };
    auto get_location_id_for_state = [&](const std::string &medium,
                                         RequestMediumState &medium_state) -> const InternedLocationId & {
        if (!medium_state.location_id) {
            medium_state.location_id =
                std::make_shared<const std::string>(event_backend->BuildLocationId(medium, host_ip_port));
        }
        return medium_state.location_id;
    };
    auto get_location_id = [&](const std::string &medium) -> const InternedLocationId & {
        return get_location_id_for_state(medium, get_medium_state(medium));
    };
    auto prepare_registration_items = [&] {
        if (registration_items_prepared) {
            return;
        }
        registration_items_prepared = true;
        for (int event_index = 0; event_index < events_size; ++event_index) {
            const auto &event = request->events(event_index);
            if (event.event_type() != proto::meta::EVENT_NODE_REGISTER || !event.has_node_register()) {
                continue;
            }
            const bool valid_mediums = std::all_of(
                event.node_register().mediums().begin(),
                event.node_register().mediums().end(),
                [](const std::string &medium) { return SnapshotUriUtils::IsValidLocationIdComponent(medium); });
            if (!valid_mediums) {
                per_item_ec[event_index] = EC_BADARGS;
                continue;
            }
            for (const auto &medium : event.node_register().mediums()) {
                if (std::find(register_mediums.begin(), register_mediums.end(), medium) == register_mediums.end()) {
                    register_mediums.push_back(medium);
                }
            }
        }
        medium_states.reserve(register_mediums.size() + 1);
    };
    auto ensure_node_medium = [&](const std::string &medium, RequestMediumState &medium_state) {
        if (medium_state.registration_ensured) {
            return EC_OK;
        }
        const ErrorCode ec = event_backend->EnsureNodeRegistered(instance_id, host_ip_port, {medium});
        if (ec == EC_OK) {
            node_registration_ensured = true;
            medium_state.registration_ensured = true;
        }
        return ec;
    };
    auto ensure_node_mediums = [&](const std::vector<std::string> &mediums) {
        std::vector<std::string> missing_mediums;
        missing_mediums.reserve(mediums.size());
        for (const auto &medium : mediums) {
            if (!get_medium_state(medium).registration_ensured) {
                missing_mediums.push_back(medium);
            }
        }
        if (node_registration_ensured && missing_mediums.empty()) {
            return EC_OK;
        }
        const ErrorCode ec = event_backend->EnsureNodeRegistered(instance_id, host_ip_port, missing_mediums);
        if (ec == EC_OK) {
            node_registration_ensured = true;
            for (const auto &medium : missing_mediums) {
                get_medium_state(medium).registration_ensured = true;
            }
        }
        return ec;
    };

    struct SnapshotReplaceEntry {
        const InternedLocationId *location_id = nullptr;
        std::vector<LocationSpec> specs;
        int event_index;
    };
    struct SnapshotCommitTask {
        ReporterSnapshotKey reporter_key;
        std::string version;
        uint64_t attempt_epoch;
        int event_index;
    };
    struct DeltaSpecMutation {
        bool is_add = false;
        LocationSpec spec;
        std::uint64_t size = 0;
        size_t next = std::numeric_limits<size_t>::max();
    };
    struct DeltaEventMutation {
        std::uint32_t event_index = 0;
        std::uint32_t next = std::numeric_limits<std::uint32_t>::max();
        bool materialized = false;
    };
    struct DeltaLocationMutation {
        int64_t block_key = 0;
        const InternedLocationId *location_id = nullptr;
        size_t first_spec = std::numeric_limits<size_t>::max();
        size_t last_spec = std::numeric_limits<size_t>::max();
        size_t spec_count = 0;
        // Keep the overwhelmingly common sole event inline. Only repeated
        // mutations of one block/location need nodes in the extra-event list.
        // ADD and DELETE are persisted in separate phases; if either phase
        // fails, every related event must still be retried together.
        std::uint32_t first_event_index = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t first_extra_event = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t last_extra_event = std::numeric_limits<std::uint32_t>::max();
        bool first_event_materialized = false;
    };
    constexpr size_t kInvalidDeltaIndex = std::numeric_limits<size_t>::max();
    constexpr std::uint32_t kInvalidDeltaEventIndex = std::numeric_limits<std::uint32_t>::max();
    std::vector<DeltaLocationMutation> delta_locations;
    std::vector<DeltaSpecMutation> delta_spec_mutations;
    std::vector<DeltaEventMutation> delta_event_mutations;
    delta_locations.reserve(events_size);
    delta_spec_mutations.reserve(events_size);
    bool has_materialized_delta_add = false;
    bool has_materialized_delta_delete = false;

    // Reporters normally emit unique block keys in ascending order. Fold that
    // stream directly into the contiguous location vector: allocating and
    // clearing an O(events) hash table only to prove every key is new costs
    // hundreds of KiB per large request. On the first non-adjacent duplicate
    // or out-of-order pair, lazily index all locations accumulated so far and
    // retain the same last-op-wins semantics for arbitrary event order.
    const size_t event_count = static_cast<size_t>(events_size);
    size_t delta_slot_mask = 0;
    std::vector<size_t> delta_location_slots;
    bool delta_locations_sorted = true;
    auto delta_location_hash = [](int64_t block_key, const InternedLocationId &location_id) {
        size_t hash = std::hash<int64_t>{}(block_key);
        hash ^= std::hash<const void *>{}(location_id.get()) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
        return hash;
    };
    auto find_delta_location = [&](int64_t block_key, const InternedLocationId &location_id) {
        size_t slot = delta_location_hash(block_key, location_id) & delta_slot_mask;
        while (delta_location_slots[slot] != kInvalidDeltaIndex) {
            const size_t location_index = delta_location_slots[slot];
            const auto &location = delta_locations[location_index];
            if (location.block_key == block_key && location.location_id->get() == location_id.get()) {
                return std::pair<size_t, size_t>{slot, location_index};
            }
            slot = (slot + 1) & delta_slot_mask;
        }
        return std::pair<size_t, size_t>{slot, kInvalidDeltaIndex};
    };
    auto build_delta_location_index = [&] {
        size_t delta_slot_count = 8;
        const size_t max_delta_slot_count = std::numeric_limits<size_t>::max() / 2;
        while (event_count > delta_slot_count / 2 && delta_slot_count <= max_delta_slot_count) {
            delta_slot_count <<= 1;
        }
        delta_location_slots.assign(delta_slot_count, kInvalidDeltaIndex);
        delta_slot_mask = delta_slot_count - 1;
        for (size_t location_index = 0; location_index < delta_locations.size(); ++location_index) {
            const auto &location = delta_locations[location_index];
            size_t slot = delta_location_hash(location.block_key, *location.location_id) & delta_slot_mask;
            while (delta_location_slots[slot] != kInvalidDeltaIndex) {
                slot = (slot + 1) & delta_slot_mask;
            }
            delta_location_slots[slot] = location_index;
        }
    };
    auto append_delta_location = [&](int64_t block_key, const InternedLocationId &location_id) {
        if (!delta_locations.empty()) {
            const auto &previous = delta_locations.back();
            if (block_key < previous.block_key ||
                (block_key == previous.block_key && *location_id < **previous.location_id)) {
                delta_locations_sorted = false;
            }
        }
        const size_t location_index = delta_locations.size();
        delta_locations.push_back(DeltaLocationMutation{block_key, &location_id});
        return location_index;
    };
    auto record_delta_event =
        [&](int64_t block_key, const InternedLocationId &location_id, int event_index) -> std::pair<size_t, size_t> {
        size_t location_index = kInvalidDeltaIndex;
        if (delta_locations.empty()) {
            location_index = append_delta_location(block_key, location_id);
        } else {
            const auto &last = delta_locations.back();
            if (last.block_key == block_key && last.location_id->get() == location_id.get()) {
                location_index = delta_locations.size() - 1;
            } else if (delta_location_slots.empty() &&
                       (last.block_key < block_key ||
                        (last.block_key == block_key && **last.location_id < *location_id))) {
                location_index = append_delta_location(block_key, location_id);
            } else {
                if (delta_location_slots.empty()) {
                    build_delta_location_index();
                }
                auto [slot, indexed_location] = find_delta_location(block_key, location_id);
                location_index = indexed_location;
                if (location_index == kInvalidDeltaIndex) {
                    location_index = append_delta_location(block_key, location_id);
                    delta_location_slots[slot] = location_index;
                }
            }
        }
        auto &location = delta_locations[location_index];
        if (location.first_event_index == kInvalidDeltaEventIndex) {
            location.first_event_index = static_cast<std::uint32_t>(event_index);
            return {location_index, kInvalidDeltaIndex};
        }
        if (delta_event_mutations.empty()) {
            // A request contains at most INT_MAX protobuf events, so uint32_t
            // indexes are sufficient. Delay this allocation until a repeated
            // block/location actually needs dependency closure.
            delta_event_mutations.reserve(event_count);
        }
        const auto event_mutation_index = static_cast<std::uint32_t>(delta_event_mutations.size());
        delta_event_mutations.push_back(
            DeltaEventMutation{static_cast<std::uint32_t>(event_index), kInvalidDeltaEventIndex, false});
        if (location.first_extra_event == kInvalidDeltaEventIndex) {
            location.first_extra_event = event_mutation_index;
        } else {
            delta_event_mutations[location.last_extra_event].next = event_mutation_index;
        }
        location.last_extra_event = event_mutation_index;
        return {location_index, event_mutation_index};
    };
    auto apply_delta_spec =
        [&](DeltaLocationMutation &location, bool is_add, LocationSpec spec, std::uint64_t size = 0) {
            has_materialized_delta_add = has_materialized_delta_add || is_add;
            has_materialized_delta_delete = has_materialized_delta_delete || !is_add;
            for (size_t index = location.first_spec; index != kInvalidDeltaIndex;
                 index = delta_spec_mutations[index].next) {
                auto &mutation = delta_spec_mutations[index];
                if (mutation.spec.name() != spec.name()) {
                    continue;
                }
                mutation.is_add = is_add;
                mutation.spec = std::move(spec);
                mutation.size = size;
                return;
            }
            const size_t mutation_index = delta_spec_mutations.size();
            delta_spec_mutations.push_back(DeltaSpecMutation{is_add, std::move(spec), size, kInvalidDeltaIndex});
            if (location.first_spec == kInvalidDeltaIndex) {
                location.first_spec = mutation_index;
            } else {
                delta_spec_mutations[location.last_spec].next = mutation_index;
            }
            location.last_spec = mutation_index;
            ++location.spec_count;
        };
    std::map<int64_t, std::vector<SnapshotReplaceEntry>> snapshot_to_replace;
    std::vector<SnapshotCommitTask> snapshot_commit_tasks;
    std::string request_snapshot_version;

    for (int i = 0; i < events_size; ++i) {
        const auto &item = request->events(i);
        switch (item.event_type()) {
        case proto::meta::EVENT_NODE_REGISTER: {
            prepare_registration_items();
            if (!item.has_node_register()) {
                per_item_ec[i] = EC_BADARGS;
                break;
            }
            if (per_item_ec[i] != EC_OK) {
                break;
            }
            if (!registration_applied) {
                register_ec = event_backend->RegisterNode(instance_id, host_ip_port, register_mediums);
                registration_applied = true;
                if (register_ec == EC_OK) {
                    node_registration_ensured = true;
                    for (const auto &medium : register_mediums) {
                        get_medium_state(medium).registration_ensured = true;
                    }
                    mutation_lifecycle_generation = event_backend->GetNodeGeneration(instance_id, host_ip_port);
                    delta_mutations.AdoptLifecycleGeneration(mutation_lifecycle_generation);
                }
            }
            per_item_ec[i] = register_ec;
            break;
        }
        case proto::meta::EVENT_HEARTBEAT: {
            if (!item.has_heartbeat()) {
                per_item_ec[i] = EC_BADARGS;
                break;
            }
            has_heartbeat = true;
            valid_heartbeat_indices.push_back(i);
            heartbeat_status.clear();
            for (const auto &kv : item.heartbeat().system_status()) {
                heartbeat_status[kv.first] = kv.second;
            }
            break;
        }
        case proto::meta::EVENT_HOST_DOWN:
            if (!item.has_host_down()) {
                per_item_ec[i] = EC_BADARGS;
                break;
            }
            has_host_down = true;
            break;
        case proto::meta::EVENT_BLOCK_ADD: {
            if (!item.has_block_add()) {
                per_item_ec[i] = EC_BADARGS;
                break;
            }
            const auto &params = item.block_add();
            int64_t block_key = 0;
            if (!ParseInt64(params.block_key(), block_key) ||
                !SnapshotUriUtils::IsValidLocationIdComponent(params.medium()) || params.specs_size() == 0) {
                per_item_ec[i] = EC_BADARGS;
                break;
            }

            ValidatedEventLocationSpecs specs;
            std::unordered_set<std::string_view> seen_spec_names;
            if (params.specs_size() > 1) {
                seen_spec_names.reserve(params.specs_size());
            }
            std::uint64_t event_total_size = 0;
            for (const auto &spec : params.specs()) {
                if (registered_spec_names.find(spec.name()) == registered_spec_names.end() ||
                    (params.specs_size() > 1 && !seen_spec_names.insert(std::string_view(spec.name())).second)) {
                    per_item_ec[i] = EC_BADARGS;
                    break;
                }
                ValidatedEventLocationSpec validated_spec;
                if (!ValidateEventLocationSpec(spec, validated_spec) ||
                    validated_spec.size > std::numeric_limits<std::uint64_t>::max() - event_total_size) {
                    per_item_ec[i] = EC_BADARGS;
                    break;
                }
                event_total_size += validated_spec.size;
                specs.Push(std::move(validated_spec), params.specs_size());
            }
            if (per_item_ec[i] != EC_OK) {
                break;
            }
            auto &medium_state = get_medium_state(params.medium());
            const auto &location_id = get_location_id_for_state(params.medium(), medium_state);
            // Record the retry dependency as soon as the event is
            // structurally valid. Admission may still fail before a mutation
            // is materialized (for example, a tombstoned reporter followed by
            // REGISTER and a later related mutation). In that case a later
            // success must also be retried, otherwise retrying only the early
            // failed item could reverse the request's final operation order.
            const auto [location_mutation_index, event_mutation_index] = record_delta_event(block_key, location_id, i);
            auto &location_mutation = delta_locations[location_mutation_index];
            const ErrorCode ensure_node_ec = ensure_node_medium(params.medium(), medium_state);
            if (ensure_node_ec != EC_OK) {
                per_item_ec[i] = ensure_node_ec;
                break;
            }

            const DeltaMutationGuard::LeaseInfo *lease = nullptr;
            bool created_generation = false;
            const ErrorCode fence_ec = delta_mutations.Acquire(lease, created_generation);
            if (fence_ec != EC_OK) {
                per_item_ec[i] = fence_ec;
                break;
            }
            mutation_lifecycle_generation = lease->lifecycle_generation;
            if (created_generation) {
                request_created_generation = true;
            }
            if (!delta_snapshot_version_validated) {
                if (!SnapshotUriUtils::IsValidSnapshotVersionToken(lease->snapshot_version)) {
                    per_item_ec[i] = EC_BADARGS;
                    break;
                }
                delta_snapshot_version_validated = true;
            }
            for (size_t spec_index = 0; spec_index < specs.Size(); ++spec_index) {
                if (!specs.At(spec_index).AddPrevalidatedSnapshotVersion(lease->snapshot_version)) {
                    per_item_ec[i] = EC_BADARGS;
                    break;
                }
            }
            if (per_item_ec[i] != EC_OK) {
                break;
            }
            for (size_t spec_index = 0; spec_index < specs.Size(); ++spec_index) {
                auto &spec = specs.At(spec_index);
                LocationSpec versioned_spec;
                versioned_spec.set_name_view(spec.name);
                versioned_spec.set_uri(std::move(spec.versioned_uri));
                apply_delta_spec(location_mutation, true, std::move(versioned_spec), spec.size);
            }
            if (event_mutation_index == kInvalidDeltaIndex) {
                location_mutation.first_event_materialized = true;
            } else {
                delta_event_mutations[event_mutation_index].materialized = true;
            }
            break;
        }
        case proto::meta::EVENT_BLOCK_DELETE: {
            if (!item.has_block_delete()) {
                per_item_ec[i] = EC_BADARGS;
                break;
            }
            const auto &params = item.block_delete();
            int64_t block_key = 0;
            if (!ParseInt64(params.block_key(), block_key) ||
                !SnapshotUriUtils::IsValidLocationIdComponent(params.medium()) || params.spec_names_size() == 0) {
                per_item_ec[i] = EC_BADARGS;
                break;
            }
            std::unordered_set<std::string_view> seen_spec_names;
            if (params.spec_names_size() > 1) {
                seen_spec_names.reserve(params.spec_names_size());
            }
            for (const auto &spec_name : params.spec_names()) {
                if (spec_name.empty() || registered_spec_names.find(spec_name) == registered_spec_names.end() ||
                    (params.spec_names_size() > 1 && !seen_spec_names.insert(std::string_view(spec_name)).second)) {
                    per_item_ec[i] = EC_BADARGS;
                    break;
                }
            }
            if (per_item_ec[i] != EC_OK) {
                break;
            }
            auto &medium_state = get_medium_state(params.medium());
            const auto &location_id = get_location_id_for_state(params.medium(), medium_state);
            const auto [location_mutation_index, event_mutation_index] = record_delta_event(block_key, location_id, i);
            auto &location_mutation = delta_locations[location_mutation_index];
            const ErrorCode ensure_node_ec = ensure_node_medium(params.medium(), medium_state);
            if (ensure_node_ec != EC_OK) {
                per_item_ec[i] = ensure_node_ec;
                break;
            }

            const DeltaMutationGuard::LeaseInfo *lease = nullptr;
            bool created_generation = false;
            const ErrorCode fence_ec = delta_mutations.Acquire(lease, created_generation);
            if (fence_ec != EC_OK) {
                per_item_ec[i] = fence_ec;
                break;
            }
            mutation_lifecycle_generation = lease->lifecycle_generation;
            if (created_generation) {
                request_created_generation = true;
            }
            for (const auto &spec_name : params.spec_names()) {
                apply_delta_spec(location_mutation, false, LocationSpec(spec_name, ""));
            }
            if (event_mutation_index == kInvalidDeltaIndex) {
                location_mutation.first_event_materialized = true;
            } else {
                delta_event_mutations[event_mutation_index].materialized = true;
            }
            break;
        }
        case proto::meta::EVENT_BLOCK_SNAPSHOT: {
            if (!item.has_block_snapshot()) {
                per_item_ec[i] = EC_BADARGS;
                break;
            }
            const auto &params = item.block_snapshot();
            struct ValidatedBlock {
                int64_t block_key;
                std::string medium;
                ValidatedEventLocationSpecs specs;
            };
            std::vector<ValidatedBlock> validated_blocks;
            validated_blocks.reserve(params.blocks_size());
            std::unordered_set<std::string> seen_blocks;
            std::uint64_t snapshot_total_size = 0;
            for (const auto &block : params.blocks()) {
                int64_t block_key = 0;
                const std::string &block_medium = block.medium().empty() ? params.medium() : block.medium();
                if (!ParseInt64(block.block_key(), block_key) ||
                    !SnapshotUriUtils::IsValidLocationIdComponent(block_medium) || block.specs_size() == 0) {
                    per_item_ec[i] = EC_BADARGS;
                    break;
                }
                const std::string duplicate_key = block_medium + "\n" + std::to_string(block_key);
                if (!seen_blocks.insert(duplicate_key).second) {
                    per_item_ec[i] = EC_BADARGS;
                    break;
                }
                ValidatedEventLocationSpecs specs;
                std::unordered_set<std::string_view> seen_spec_names;
                if (block.specs_size() > 1) {
                    seen_spec_names.reserve(block.specs_size());
                }
                for (const auto &spec : block.specs()) {
                    if (registered_spec_names.find(spec.name()) == registered_spec_names.end() ||
                        (block.specs_size() > 1 && !seen_spec_names.insert(std::string_view(spec.name())).second)) {
                        per_item_ec[i] = EC_BADARGS;
                        break;
                    }
                    ValidatedEventLocationSpec validated_spec;
                    if (!ValidateEventLocationSpec(spec, validated_spec) ||
                        validated_spec.size > std::numeric_limits<std::uint64_t>::max() - snapshot_total_size) {
                        per_item_ec[i] = EC_BADARGS;
                        break;
                    }
                    snapshot_total_size += validated_spec.size;
                    specs.Push(std::move(validated_spec), block.specs_size());
                }
                if (per_item_ec[i] != EC_OK) {
                    break;
                }
                validated_blocks.push_back(ValidatedBlock{block_key, block_medium, std::move(specs)});
            }
            if (per_item_ec[i] != EC_OK) {
                break;
            }
            std::vector<std::string> snapshot_mediums;
            snapshot_mediums.reserve(validated_blocks.size());
            for (const auto &block : validated_blocks) {
                if (std::find(snapshot_mediums.begin(), snapshot_mediums.end(), block.medium) ==
                    snapshot_mediums.end()) {
                    snapshot_mediums.push_back(block.medium);
                }
            }
            const ErrorCode ensure_node_ec = ensure_node_mediums(snapshot_mediums);
            if (ensure_node_ec != EC_OK) {
                per_item_ec[i] = ensure_node_ec;
                break;
            }

            // Keep the reporter write gate short: all request parsing and
            // validation above is complete before BeginSnapshot closes it.
            uint64_t retry_after_ms = 0;
            const ErrorCode begin_ec = event_backend->BeginSnapshot(
                reporter_key, request_snapshot_version, retry_after_ms, &mutation_lifecycle_generation);
            if (begin_ec != EC_OK) {
                per_item_ec[i] = begin_ec;
                if (begin_ec == EC_SNAPSHOT_RATE_LIMITED) {
                    response->set_retry_after_ms(retry_after_ms);
                }
                break;
            }
            if (!SnapshotUriUtils::IsValidSnapshotVersionToken(request_snapshot_version)) {
                per_item_ec[i] = EC_BADARGS;
                event_backend->AbortSnapshotVersion(reporter_key, request_snapshot_version);
                request_snapshot_version.clear();
                break;
            }

            for (auto &block : validated_blocks) {
                std::vector<LocationSpec> versioned_specs;
                versioned_specs.reserve(block.specs.Size());
                for (size_t spec_index = 0; spec_index < block.specs.Size(); ++spec_index) {
                    auto &spec = block.specs.At(spec_index);
                    if (!spec.AddPrevalidatedSnapshotVersion(request_snapshot_version)) {
                        per_item_ec[i] = EC_BADARGS;
                        break;
                    }
                    LocationSpec versioned_spec;
                    versioned_spec.set_name_view(spec.name);
                    versioned_spec.set_uri(std::move(spec.versioned_uri));
                    versioned_specs.push_back(std::move(versioned_spec));
                }
                if (per_item_ec[i] != EC_OK) {
                    break;
                }
                snapshot_to_replace[block.block_key].push_back(
                    SnapshotReplaceEntry{&get_location_id(block.medium), std::move(versioned_specs), i});
            }
            if (per_item_ec[i] != EC_OK) {
                event_backend->AbortSnapshotVersion(reporter_key, request_snapshot_version);
                snapshot_to_replace.clear();
                request_snapshot_version.clear();
                break;
            }
            snapshot_commit_tasks.push_back(SnapshotCommitTask{
                reporter_key, request_snapshot_version, event_backend->GetSnapshotAttemptEpoch(reporter_key), i});
            break;
        }
        default:
            per_item_ec[i] = EC_BADARGS;
            break;
        }
    }

    // A request is an ordered event stream. Specs and event dependencies are
    // linked through request-wide contiguous arrays, so the common one-spec,
    // one-event block does not allocate three tiny vectors of its own.
    // Preserve deterministic block/location task ordering for arbitrary input
    // without allocating an identity permutation for the already-sorted fast
    // path.
    std::vector<size_t> ordered_delta_location_indices;
    if (!delta_locations_sorted) {
        ordered_delta_location_indices.resize(delta_locations.size());
        for (size_t i = 0; i < delta_locations.size(); ++i) {
            ordered_delta_location_indices[i] = i;
        }
        std::sort(ordered_delta_location_indices.begin(),
                  ordered_delta_location_indices.end(),
                  [&delta_locations](size_t lhs_index, size_t rhs_index) {
                      const auto &lhs = delta_locations[lhs_index];
                      const auto &rhs = delta_locations[rhs_index];
                      return lhs.block_key != rhs.block_key ? lhs.block_key < rhs.block_key
                                                            : **lhs.location_id < **rhs.location_id;
                  });
    }
    auto delta_location_index_at = [&ordered_delta_location_indices](size_t position) {
        return ordered_delta_location_indices.empty() ? position : ordered_delta_location_indices[position];
    };

    KeyVector add_keys_aggr;
    std::vector<MetaSearcher::MergeLocationSpecsTask> merge_tasks;
    std::vector<size_t> merge_task_offsets{0};
    KeyVector del_keys_aggr;
    std::vector<std::vector<MetaSearcher::DeleteLocationSpecsTask>> delete_tasks;
    if (has_materialized_delta_add) {
        add_keys_aggr.reserve(delta_locations.size());
        merge_tasks.reserve(delta_locations.size());
        merge_task_offsets.reserve(delta_locations.size() + 1);
    }
    if (has_materialized_delta_delete) {
        del_keys_aggr.reserve(delta_locations.size());
        delete_tasks.reserve(delta_locations.size());
    }

    for (size_t block_begin = 0; block_begin < delta_locations.size();) {
        const int64_t block_key = delta_locations[delta_location_index_at(block_begin)].block_key;
        size_t block_end = block_begin + 1;
        while (block_end < delta_locations.size() &&
               delta_locations[delta_location_index_at(block_end)].block_key == block_key) {
            ++block_end;
        }
        std::vector<MetaSearcher::DeleteLocationSpecsTask> block_delete_tasks;
        const size_t add_task_begin = merge_tasks.size();
        for (size_t location_position = block_begin; location_position < block_end; ++location_position) {
            auto &location = delta_locations[delta_location_index_at(location_position)];
            MetaSearcher::MergeLocationSpecsTask add_task{
                {},
                requested_type,
                CacheLocationStatus::CLS_SERVING,
                {},
            };
            add_task.borrowed_interned_location_id = location.location_id;
            std::uint64_t add_task_total_size = 0;
            bool add_task_size_overflow = false;
            MetaSearcher::DeleteLocationSpecsTask delete_task{{}, {}};
            delete_task.borrowed_interned_location_id = location.location_id;
            for (size_t mutation_index = location.first_spec; mutation_index != kInvalidDeltaIndex;
                 mutation_index = delta_spec_mutations[mutation_index].next) {
                auto &mutation = delta_spec_mutations[mutation_index];
                if (mutation.is_add) {
                    if (mutation.size > std::numeric_limits<std::uint64_t>::max() - add_task_total_size) {
                        add_task_size_overflow = true;
                    } else {
                        add_task_total_size += mutation.size;
                    }
                    add_task.PushReportEventSpec(std::move(mutation.spec), location.spec_count);
                } else {
                    if (delete_task.spec_names.empty()) {
                        delete_task.spec_names.reserve(location.spec_count);
                    }
                    delete_task.spec_names.push_back(mutation.spec.name());
                }
            }
            if (add_task.specs.size() > 1) {
                std::sort(add_task.specs.begin(), add_task.specs.end(), [](const auto &lhs, const auto &rhs) {
                    return lhs.name() < rhs.name();
                });
            }
            std::sort(delete_task.spec_names.begin(), delete_task.spec_names.end());
            if (!add_task.SpecsEmpty()) {
                // A pathological overflow falls back to the strict validator,
                // which rejects the task without trusting a wrapped total.
                if (!add_task_size_overflow) {
                    add_task.prevalidated_total_size = MetaSearcher::PrevalidatedTotalSize(add_task_total_size);
                }
                merge_tasks.push_back(std::move(add_task));
            }
            if (!delete_task.spec_names.empty()) {
                if (block_delete_tasks.empty()) {
                    block_delete_tasks.reserve(block_end - block_begin);
                }
                block_delete_tasks.push_back(std::move(delete_task));
            }
        }
        if (merge_tasks.size() != add_task_begin) {
            add_keys_aggr.push_back(block_key);
            merge_task_offsets.push_back(merge_tasks.size());
        }
        if (!block_delete_tasks.empty()) {
            del_keys_aggr.push_back(block_key);
            delete_tasks.push_back(std::move(block_delete_tasks));
        }
        block_begin = block_end;
    }

    // Failure propagation is exceptional. Avoid writing a 24-byte range for
    // every successful block; locate the affected range in the already-sorted
    // location view only if a backend error actually occurs.
    auto find_delta_block_range = [&](int64_t block_key) {
        size_t lower = 0;
        size_t upper = delta_locations.size();
        while (lower < upper) {
            const size_t middle = lower + (upper - lower) / 2;
            if (delta_locations[delta_location_index_at(middle)].block_key < block_key) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        const size_t block_begin = lower;
        upper = delta_locations.size();
        while (lower < upper) {
            const size_t middle = lower + (upper - lower) / 2;
            if (delta_locations[delta_location_index_at(middle)].block_key <= block_key) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        assert(block_begin < lower && delta_locations[delta_location_index_at(block_begin)].block_key == block_key);
        return std::pair<size_t, size_t>{block_begin, lower};
    };

    auto visit_delta_events = [&delta_event_mutations, kInvalidDeltaEventIndex](const DeltaLocationMutation &location,
                                                                                const auto &visitor) {
        if (location.first_event_index != kInvalidDeltaEventIndex &&
            !visitor(static_cast<int>(location.first_event_index), location.first_event_materialized)) {
            return false;
        }
        for (std::uint32_t event_index = location.first_extra_event; event_index != kInvalidDeltaEventIndex;
             event_index = delta_event_mutations[event_index].next) {
            const auto &event = delta_event_mutations[event_index];
            if (!visitor(static_cast<int>(event.event_index), event.materialized)) {
                return false;
            }
        }
        return true;
    };
    auto mark_delta_phase_failure =
        [&per_item_ec, request, &delta_location_index_at, &delta_locations, &visit_delta_events](
            size_t location_begin, size_t location_end, ErrorCode ec, const auto &spec_participates) {
            for (size_t location_position = location_begin; location_position < location_end; ++location_position) {
                const auto &location = delta_locations[delta_location_index_at(location_position)];
                visit_delta_events(location, [&](int event_index, bool materialized) {
                    if (!materialized || per_item_ec[event_index] != EC_OK) {
                        return true;
                    }
                    const auto &event = request->events(event_index);
                    bool participates = false;
                    if (event.event_type() == proto::meta::EVENT_BLOCK_ADD && event.has_block_add()) {
                        participates = std::any_of(event.block_add().specs().begin(),
                                                   event.block_add().specs().end(),
                                                   [&spec_participates, &location](const auto &spec) {
                                                       return spec_participates(**location.location_id, spec.name());
                                                   });
                    } else if (event.event_type() == proto::meta::EVENT_BLOCK_DELETE && event.has_block_delete()) {
                        participates = std::any_of(event.block_delete().spec_names().begin(),
                                                   event.block_delete().spec_names().end(),
                                                   [&spec_participates, &location](const auto &spec_name) {
                                                       return spec_participates(**location.location_id, spec_name);
                                                   });
                    }
                    if (participates) {
                        per_item_ec[event_index] = ec;
                    }
                    return true;
                });
            }
        };

    if (has_heartbeat) {
        const ErrorCode ec = event_backend->OnHeartbeat(instance_id, host_ip_port, heartbeat_status);
        if (ec != EC_OK) {
            for (const int event_index : valid_heartbeat_indices) {
                per_item_ec[event_index] = ec;
            }
        } else {
            // Recovering an unavailable reporter advances its lifecycle to
            // fence cleanup selected for the old generation. Mutations
            // admitted by this same RPC belong to the recovered lifecycle;
            // retain that generation through metadata writes and lease drain.
            mutation_lifecycle_generation = event_backend->GetNodeGeneration(instance_id, host_ip_port);
            delta_mutations.AdoptLifecycleGeneration(mutation_lifecycle_generation);
        }
    }

    // MetaSearcher calls this once after the fused target-location read and
    // before its first mutation. This removes the old per-key lock/allocation
    // amplification while preserving the window in which HOST_DOWN can fence
    // a request that is stalled in metadata I/O.
    auto acquire_write_lease = [event_backend, reporter_key, mutation_lifecycle_generation] {
        EventReportBackend::LifecycleMutationLease lease;
        const ErrorCode ec =
            event_backend->AcquireLifecycleMutationLease(reporter_key, mutation_lifecycle_generation, lease);
        return std::make_pair(ec, std::static_pointer_cast<void>(std::move(lease)));
    };

    if (!add_keys_aggr.empty()) {
        std::vector<ErrorCode> per_key_ec;
        meta_searcher->BatchMergeLocationSpecsFlat(
            request_context, add_keys_aggr, merge_task_offsets, merge_tasks, per_key_ec, acquire_write_lease);

        for (size_t k = 0; k < add_keys_aggr.size(); ++k) {
            ErrorCode key_ec = (k < per_key_ec.size()) ? per_key_ec[k] : EC_ERROR;
            if (key_ec == EC_OK) {
                continue;
            }
            const auto tasks_begin = merge_tasks.begin() + merge_task_offsets[k];
            const auto tasks_end = merge_tasks.begin() + merge_task_offsets[k + 1];
            const auto [location_begin, location_end] = find_delta_block_range(add_keys_aggr[k]);
            mark_delta_phase_failure(
                location_begin,
                location_end,
                key_ec,
                [tasks_begin, tasks_end](const std::string &location_id, const std::string &spec_name) {
                    const auto task = std::find_if(tasks_begin, tasks_end, [&location_id](const auto &candidate) {
                        return candidate.ResolvedLocationId() == location_id;
                    });
                    if (task == tasks_end) {
                        return false;
                    }
                    for (size_t spec_index = 0; spec_index < task->SpecCount(); ++spec_index) {
                        if (task->SpecAt(spec_index).name() == spec_name) {
                            return true;
                        }
                    }
                    return false;
                });
        }
    }

    if (!del_keys_aggr.empty()) {
        std::vector<std::vector<ErrorCode>> per_location_ec;
        std::vector<std::vector<bool>> missing_delete_targets;
        size_t delete_target_count = 0;
        for (const auto &tasks : delete_tasks) {
            delete_target_count += tasks.size();
        }
        meta_searcher->BatchDeleteLocationSpecs(request_context,
                                                del_keys_aggr,
                                                delete_tasks,
                                                per_location_ec,
                                                &missing_delete_targets,
                                                acquire_write_lease);

        for (size_t k = 0; k < del_keys_aggr.size(); ++k) {
            ErrorCode key_ec = EC_OK;
            if (k < per_location_ec.size() && per_location_ec[k].size() == delete_tasks[k].size()) {
                for (const auto &loc_ec : per_location_ec[k]) {
                    if (loc_ec != EC_OK && loc_ec != EC_NOENT) {
                        key_ec = loc_ec;
                        break;
                    }
                }
            } else {
                key_ec = EC_ERROR;
            }
            if (key_ec == EC_OK) {
                continue;
            }
            const auto &tasks = delete_tasks[k];
            const auto [location_begin, location_end] = find_delta_block_range(del_keys_aggr[k]);
            mark_delta_phase_failure(
                location_begin,
                location_end,
                key_ec,
                [&tasks](const std::string &location_id, const std::string &spec_name) {
                    const auto task = std::find_if(tasks.begin(), tasks.end(), [&location_id](const auto &candidate) {
                        return candidate.ResolvedLocationId() == location_id;
                    });
                    return task != tasks.end() &&
                           std::find(task->spec_names.begin(), task->spec_names.end(), spec_name) !=
                               task->spec_names.end();
                });
        }
        size_t missing_block_count = 0;
        for (const auto &missing_per_key : missing_delete_targets) {
            missing_block_count +=
                static_cast<size_t>(std::count(missing_per_key.begin(), missing_per_key.end(), true));
        }
        if (missing_block_count > 0) {
            KVCM_LOG_DEBUG("trace_id [%s] | EVENT_BLOCK_DELETE: attempted to delete %zu non-existent block(s) "
                           "(block_key + medium) out of %zu unique target(s); treated as idempotent success, "
                           "instance [%s], host [%s], storage_type [%s]",
                           trace_id.c_str(),
                           missing_block_count,
                           delete_target_count,
                           instance_id.c_str(),
                           host_ip_port.c_str(),
                           ToString(requested_type).c_str());
        }
    }

    if (!snapshot_to_replace.empty()) {
        KeyVector snapshot_keys;
        std::vector<std::vector<MetaSearcher::ReplaceLocationSpecsTask>> replace_tasks;
        std::vector<std::vector<int>> event_indices;
        snapshot_keys.reserve(snapshot_to_replace.size());
        replace_tasks.reserve(snapshot_to_replace.size());
        event_indices.reserve(snapshot_to_replace.size());
        for (auto &[block_key, entries] : snapshot_to_replace) {
            snapshot_keys.push_back(block_key);
            auto &tasks = replace_tasks.emplace_back();
            auto &indices = event_indices.emplace_back();
            tasks.reserve(entries.size());
            indices.reserve(entries.size());
            for (auto &entry : entries) {
                MetaSearcher::ReplaceLocationSpecsTask task{
                    {},
                    requested_type,
                    CacheLocationStatus::CLS_SERVING,
                    std::move(entry.specs),
                };
                task.borrowed_interned_location_id = entry.location_id;
                tasks.push_back(std::move(task));
                indices.push_back(entry.event_index);
            }
        }

        std::vector<ErrorCode> per_key_ec;
        meta_searcher->BatchReplaceLocationSpecs(
            request_context, snapshot_keys, replace_tasks, per_key_ec, acquire_write_lease);
        for (size_t key_index = 0; key_index < snapshot_keys.size(); ++key_index) {
            const ErrorCode key_ec = key_index < per_key_ec.size() ? per_key_ec[key_index] : EC_ERROR;
            if (key_ec == EC_OK) {
                continue;
            }
            for (int event_index : event_indices[key_index]) {
                if (per_item_ec[event_index] == EC_OK) {
                    per_item_ec[event_index] = key_ec;
                }
            }
        }
    }

    if (!snapshot_commit_tasks.empty()) {
        const auto &task = snapshot_commit_tasks.front();
        bool snapshot_failed = per_item_ec[task.event_index] != EC_OK;
        // Event-report metadata is a soft cache index. Like delta mutations,
        // a snapshot commits after every persistent write has been accepted
        // by the async backend and mirrored into the local cache; it does not
        // wait for Redis consumers to flush their queues.
        if (!snapshot_failed) {
            const ErrorCode commit_ec = event_backend->CommitSnapshotVersionIfGeneration(
                task.reporter_key, task.version, mutation_lifecycle_generation);
            if (commit_ec != EC_OK) {
                KVCM_LOG_ERROR("trace_id [%s] | EVENT_BLOCK_SNAPSHOT: failed to publish host [%s] token [%s], ec [%d]",
                               trace_id.c_str(),
                               task.reporter_key.host_ip_port.c_str(),
                               task.version.c_str(),
                               commit_ec);
                per_item_ec[task.event_index] = commit_ec;
                snapshot_failed = true;
            }
        }
        if (snapshot_failed) {
            if (per_item_ec[task.event_index] == EC_OK) {
                per_item_ec[task.event_index] = EC_ERROR;
            }
            event_backend->AbortSnapshotVersion(task.reporter_key, task.version);
        } else if (schedule_plan_executor_) {
            const auto cleanup_backend = event_backend;
            const auto cleanup_state = event_cleanup_callback_state_;
            uint64_t cleanup_epoch = 0;
            {
                std::shared_lock<std::shared_mutex> cleanup_lease(cleanup_state->mutex);
                if (cleanup_state->accepting) {
                    cleanup_epoch = cleanup_state->epoch;
                }
            }
            if (!schedule_plan_executor_->SubmitTask([this,
                                                      reporter_key = task.reporter_key,
                                                      version = task.version,
                                                      attempt_epoch = task.attempt_epoch,
                                                      lifecycle_generation = mutation_lifecycle_generation,
                                                      requested_type,
                                                      cleanup_backend,
                                                      cleanup_state,
                                                      cleanup_epoch] {
                    std::shared_lock<std::shared_mutex> cleanup_lease(cleanup_state->mutex);
                    if (!cleanup_state->accepting || cleanup_state->epoch != cleanup_epoch) {
                        return;
                    }
                    this->CleanupStaleSnapshotLocations(
                        reporter_key, version, requested_type, cleanup_backend, attempt_epoch, lifecycle_generation);
                })) {
                KVCM_LOG_WARN("trace_id [%s] | EVENT_BLOCK_SNAPSHOT: failed to submit stale-data scan for host [%s]",
                              trace_id.c_str(),
                              task.reporter_key.host_ip_port.c_str());
            }
        }
    }

    if (has_host_down) {
        uint64_t gen_at_trigger = 0;
        const ErrorCode host_down_ec =
            event_backend->UnregisterNodeForHostDown(instance_id, host_ip_port, gen_at_trigger);
        if (host_down_ec != EC_OK) {
            per_item_ec[0] = host_down_ec;
        }
        bool cleanup_dispatched = false;
        if (host_down_ec == EC_OK) {
            const auto cleanup_state = event_cleanup_callback_state_;
            uint64_t cleanup_epoch = 0;
            {
                std::shared_lock<std::shared_mutex> cleanup_lease(cleanup_state->mutex);
                if (cleanup_state->accepting) {
                    cleanup_epoch = cleanup_state->epoch;
                }
            }
            const auto cleanup = [this,
                                  instance_id,
                                  host_ip_port,
                                  gen_at_trigger,
                                  requested_type,
                                  event_backend,
                                  cleanup_state,
                                  cleanup_epoch] {
                std::shared_lock<std::shared_mutex> cleanup_lease(cleanup_state->mutex);
                if (!cleanup_state->accepting || cleanup_state->epoch != cleanup_epoch) {
                    return;
                }
                this->CleanupHostLocations(instance_id, host_ip_port, gen_at_trigger, requested_type, event_backend);
            };
            cleanup_dispatched = schedule_plan_executor_ && schedule_plan_executor_->SubmitTask(cleanup);
            if (!cleanup_dispatched) {
                KVCM_LOG_WARN("trace_id [%s] | HOST_DOWN: cleanup queue unavailable for host [%s], "
                              "instance [%s], gen=%" PRIu64 "; running inline",
                              trace_id.c_str(),
                              host_ip_port.c_str(),
                              instance_id.c_str(),
                              gen_at_trigger);
                cleanup();
                cleanup_dispatched = true;
            }
            KVCM_LOG_INFO("trace_id [%s] | HOST_DOWN: host [%s] removed from node table, cleanup_dispatched=%s "
                          "(gen=%" PRIu64 ")",
                          trace_id.c_str(),
                          host_ip_port.c_str(),
                          cleanup_dispatched ? "true" : "false",
                          gen_at_trigger);
        } else {
            KVCM_LOG_WARN("trace_id [%s] | HOST_DOWN: failed to end lifecycle for host [%s], "
                          "instance [%s], ec=%d",
                          trace_id.c_str(),
                          host_ip_port.c_str(),
                          instance_id.c_str(),
                          host_down_ec);
        }
    }

    for (const auto &location : delta_locations) {
        ErrorCode group_failure = EC_OK;
        visit_delta_events(location, [&](int event_index, bool /*materialized*/) {
            if (per_item_ec[event_index] != EC_OK) {
                group_failure = per_item_ec[event_index];
                return false;
            }
            return true;
        });
        if (group_failure == EC_OK) {
            continue;
        }
        visit_delta_events(location, [&](int event_index, bool /*materialized*/) {
            if (per_item_ec[event_index] == EC_OK) {
                per_item_ec[event_index] = group_failure;
            }
            return true;
        });
    }

    bool any_failure = false;
    ErrorCode first_failure = EC_OK;
    for (const ErrorCode ec : per_item_ec) {
        if (ec != EC_OK) {
            any_failure = true;
            if (first_failure == EC_OK) {
                first_failure = ec;
            }
        }
    }

    auto map_error = [](ErrorCode ec) {
        switch (ec) {
        case EC_OK:
            return proto::meta::OK;
        case EC_BADARGS:
            return proto::meta::INVALID_ARGUMENT;
        case EC_INSTANCE_NOT_EXIST:
            return proto::meta::INSTANCE_NOT_EXIST;
        case EC_NODE_NOT_REGISTERED:
            return proto::meta::NODE_NOT_REGISTERED;
        case EC_SNAPSHOT_IN_PROGRESS:
            return proto::meta::SNAPSHOT_IN_PROGRESS;
        case EC_SNAPSHOT_RATE_LIMITED:
            return proto::meta::SNAPSHOT_RATE_LIMITED;
        case EC_SNAPSHOT_REQUIRED:
            return proto::meta::SNAPSHOT_REQUIRED;
        default:
            return proto::meta::INTERNAL_ERROR;
        }
    };

    refresh_snapshot_response(request_created_generation);
    if (any_failure) {
        std::map<std::pair<proto::meta::ReportEventType, proto::meta::ErrorCode>, size_t> failure_counts;
        size_t failed_item_count = 0;
        for (int i = 0; i < request->events_size(); ++i) {
            if (per_item_ec[i] == EC_OK) {
                continue;
            }
            ++failed_item_count;
            ++failure_counts[{request->events(i).event_type(), map_error(per_item_ec[i])}];
        }
        std::string failure_summary;
        for (const auto &[event_and_error, count] : failure_counts) {
            if (!failure_summary.empty()) {
                failure_summary.append(", ");
            }
            const int event_type = static_cast<int>(event_and_error.first);
            const std::string event_type_name = proto::meta::ReportEventType_IsValid(event_type)
                                                    ? proto::meta::ReportEventType_Name(event_and_error.first)
                                                    : "EVENT_UNKNOWN(" + std::to_string(event_type) + ")";
            failure_summary.append(event_type_name)
                .append(":")
                .append(proto::meta::ErrorCode_Name(event_and_error.second))
                .append("=")
                .append(std::to_string(count));
        }
        KVCM_LOG_WARN("trace_id [%s] | ReportEvent partial failure: failed_items [%zu/%d], summary [%s], "
                      "instance [%s], host [%s], storage_type [%s]",
                      trace_id.c_str(),
                      failed_item_count,
                      request->events_size(),
                      failure_summary.c_str(),
                      instance_id.c_str(),
                      host_ip_port.c_str(),
                      ToString(requested_type).c_str());
        for (const ErrorCode ec : per_item_ec) {
            response->add_item_results(map_error(ec));
        }
        response_status->set_code(map_error(first_failure));
        response_status->set_message("ReportEvent partially failed; see item_results");
        return EC_PARTIAL_OK;
    }
    response_status->set_code(proto::meta::OK);
    return EC_OK;
}

void CacheManager::CleanupHostLocations(const std::string &instance_id,
                                        const std::string &host_ip_port,
                                        uint64_t cleanup_generation,
                                        DataStorageType storage_type,
                                        const std::shared_ptr<EventReportBackend> &expected_backend) {
    if (!IsCurrentEventReportBackend(registry_manager_, instance_id, storage_type, expected_backend)) {
        KVCM_LOG_INFO("CleanupHostLocations: skipping stale backend incarnation for host [%s] instance [%s]",
                      host_ip_port.c_str(),
                      instance_id.c_str());
        return;
    }
    const auto &event_backend = expected_backend;

    uint64_t current_gen = event_backend->GetNodeGeneration(instance_id, host_ip_port);
    if (current_gen != cleanup_generation) {
        KVCM_LOG_INFO("CleanupHostLocations: skipping stale cleanup for host [%s] instance [%s] "
                      "(trigger_gen=%" PRIu64 ", current_gen=%" PRIu64 " — node re-registered)",
                      host_ip_port.c_str(),
                      instance_id.c_str(),
                      cleanup_generation,
                      current_gen);
        return;
    }

    MetaSearcher *meta_searcher = meta_searcher_manager_->GetMetaSearcher(instance_id);
    if (!meta_searcher) {
        KVCM_LOG_WARN("CleanupHostLocations: meta searcher not found for instance [%s]", instance_id.c_str());
        return;
    }

    RequestContext cleanup_ctx("cleanup_host_" + host_ip_port);
    const std::string host_suffix = event_backend->HostSuffix(host_ip_port);

    auto backend_is_current = [registry_manager = registry_manager_, instance_id, storage_type, event_backend] {
        return IsCurrentEventReportBackend(registry_manager, instance_id, storage_type, event_backend);
    };
    auto abort_if_reregistered =
        [event_backend, instance_id, host_ip_port, cleanup_generation, backend_is_current]() -> bool {
        return !backend_is_current() ||
               event_backend->GetNodeGeneration(instance_id, host_ip_port) != cleanup_generation;
    };
    auto acquire_cleanup_lease = [event_backend, instance_id, host_ip_port, cleanup_generation, backend_is_current] {
        EventReportBackend::LifecycleMutationLease lease;
        if (!backend_is_current()) {
            return std::make_pair(EC_MISMATCH, MetaSearcher::MetadataWriteLease{});
        }
        const ErrorCode ec =
            event_backend->AcquireLifecycleCleanupLease({instance_id, host_ip_port}, cleanup_generation, lease);
        // Keep the global lock order DataStorageManager -> backend lifecycle.
        // Re-entering backend_is_current() while holding this lease would
        // invert UnRegisterStorage's DataStorageManager -> Close order.
        return std::make_pair(ec, std::static_pointer_cast<void>(std::move(lease)));
    };

    auto ec = meta_searcher->CleanupLocationsByHost(&cleanup_ctx,
                                                    host_suffix,
                                                    storage_type,
                                                    /*scan_batch_size=*/1000,
                                                    abort_if_reregistered,
                                                    acquire_cleanup_lease);

    if (ec == EC_OK) {
        KVCM_LOG_INFO("CleanupHostLocations: finished cleaning host [%s] from instance [%s]",
                      host_ip_port.c_str(),
                      instance_id.c_str());
    } else {
        KVCM_LOG_WARN("CleanupHostLocations: finished with partial failures for host [%s] instance [%s]",
                      host_ip_port.c_str(),
                      instance_id.c_str());
    }
}

ErrorCode CacheManager::CleanupStaleSnapshotLocations(const ReporterSnapshotKey &reporter_key,
                                                      const std::string &snapshot_version,
                                                      DataStorageType storage_type,
                                                      const std::shared_ptr<EventReportBackend> &event_backend,
                                                      uint64_t snapshot_attempt_epoch,
                                                      uint64_t lifecycle_generation) {
    if (!IsCurrentEventReportBackend(registry_manager_, reporter_key.instance_id, storage_type, event_backend)) {
        return EC_OK;
    }
    if (!event_backend || snapshot_version.empty() || event_backend->GetStorageType() != storage_type ||
        event_backend->GetSnapshotVersion(reporter_key) != snapshot_version ||
        (snapshot_attempt_epoch != 0 &&
         event_backend->GetSnapshotAttemptEpoch(reporter_key) != snapshot_attempt_epoch)) {
        return EC_OK;
    }
    MetaSearcher *meta_searcher = meta_searcher_manager_->GetMetaSearcher(reporter_key.instance_id);
    if (!meta_searcher) {
        KVCM_LOG_WARN("CleanupStaleSnapshotLocations: meta searcher not found for instance [%s]",
                      reporter_key.instance_id.c_str());
        return EC_NOENT;
    }

    const auto scan_begin = std::chrono::steady_clock::now();
    RequestContext cleanup_ctx("reclaim_snapshot_" + reporter_key.host_ip_port);
    auto should_delete = [event_backend, reporter_key, snapshot_attempt_epoch](
                             int64_t, const std::string &location_id, const CacheLocation &location) {
        if (snapshot_attempt_epoch != 0 &&
            event_backend->GetSnapshotAttemptEpoch(reporter_key) != snapshot_attempt_epoch) {
            return false;
        }
        std::string medium;
        std::string reporter_host;
        return event_backend->ParseLocationId(location_id, medium, reporter_host) &&
               reporter_host == reporter_key.host_ip_port &&
               IsSnapshotLocationStale(
                   event_backend.get(), reporter_key.instance_id, location, /*preserve_in_flight=*/true);
    };
    auto backend_is_current = [registry_manager = registry_manager_, reporter_key, storage_type, event_backend] {
        return IsCurrentEventReportBackend(registry_manager, reporter_key.instance_id, storage_type, event_backend);
    };
    auto should_abort = [event_backend, reporter_key, snapshot_attempt_epoch, backend_is_current] {
        return !backend_is_current() || (snapshot_attempt_epoch != 0 && event_backend->GetSnapshotAttemptEpoch(
                                                                            reporter_key) != snapshot_attempt_epoch);
    };
    if (lifecycle_generation == 0) {
        lifecycle_generation = event_backend->GetNodeGeneration(reporter_key.instance_id, reporter_key.host_ip_port);
    }
    auto acquire_cleanup_lease = [event_backend,
                                  reporter_key,
                                  snapshot_version,
                                  snapshot_attempt_epoch,
                                  lifecycle_generation,
                                  backend_is_current] {
        EventReportBackend::LifecycleMutationLease lease;
        if (!backend_is_current()) {
            return std::make_pair(EC_MISMATCH, MetaSearcher::MetadataWriteLease{});
        }
        const ErrorCode ec = event_backend->AcquireSnapshotCleanupLease(
            reporter_key, lifecycle_generation, snapshot_version, snapshot_attempt_epoch, lease);
        // Do not query DataStorageManager again while holding the backend
        // lifecycle lease; storage unregister takes those locks in the
        // opposite (global-manager then backend) order.
        return std::make_pair(ec, std::static_pointer_cast<void>(std::move(lease)));
    };
    const ErrorCode ec = meta_searcher->CleanupLocationsByPredicate(&cleanup_ctx,
                                                                    storage_type,
                                                                    /*scan_batch_size=*/1000,
                                                                    std::move(should_delete),
                                                                    std::move(should_abort),
                                                                    std::move(acquire_cleanup_lease));
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - scan_begin).count();
    KVCM_LOG_INFO("SnapshotReclaimer: scanned instance [%s] host [%s] token [%s] in [%" PRId64 "] ms, ec [%d]",
                  reporter_key.instance_id.c_str(),
                  reporter_key.host_ip_port.c_str(),
                  snapshot_version.c_str(),
                  static_cast<int64_t>(elapsed_ms),
                  ec);
    if (metrics_registry_) {
        const MetricsTags metrics_tags = {
            {"instance_id", reporter_key.instance_id},
            {"host", reporter_key.host_ip_port},
            {"type", ToString(storage_type)},
        };
        REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                              "event_report.snapshot_cleanup_scan_latency_ms",
                              metrics_tags,
                              static_cast<double>(elapsed_ms));
    }
    return ec;
}

ErrorCode CacheManager::TryCreateMetaSearcher(RequestContext *request_context, const std::string &instance_id) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    const auto check_loc_data_exist = GetCheckLocDataExistFunc(instance_id);
    const auto submit_del_req = GetSubmitDelReqFunc(instance_id);
    MetaSearcher *meta_searcher = meta_searcher_manager_->TryCreateMetaSearcher(
        request_context, instance_id, check_loc_data_exist, submit_del_req);
    if (!meta_searcher) {
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, EC_ERROR, "create meta searcher failed");
    }
    PREFIX_LOG(INFO, "create meta searcher success");
    return EC_OK;
}

std::pair<ErrorCode, MetaSearcher *> CacheManager::CheckInputAndGetMetaSearcher(RequestContext *request_context,
                                                                                const std::string &instance_id,
                                                                                const KeyVector &keys,
                                                                                const TokenIdsVector &tokens) const {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    MetaSearcher *meta_searcher = meta_searcher_manager_->GetMetaSearcher(instance_id);
    if (!meta_searcher) {
        PREFIX_LOG(WARN, "meta searcher not found");
        request_context->error_tracer()->AddErrorMsg("instance not exist");
        return {EC_INSTANCE_NOT_EXIST, nullptr};
    }
    if (keys.empty() && tokens.empty()) {
        PREFIX_LOG(WARN, "empty input");
        request_context->error_tracer()->AddErrorMsg("empty input");
        return {EC_BADARGS, nullptr};
    }
    return {EC_OK, meta_searcher};
}

std::pair<ErrorCode, int64_t> CacheManager::GetBlockSize(RequestContext *request_context,
                                                         const std::string &instance_id) const {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (!instance_info) {
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, EC_INSTANCE_NOT_EXIST, int64_t, "instance not found");
    };
    int64_t block_size = instance_info->block_size();
    if (block_size <= 0) {
        request_context->error_tracer()->AddErrorMsg("tokens size per block error");
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, EC_BADARGS, int64_t, "tokens size per block error [%ld]", block_size);
    };
    return {EC_OK, block_size};
}

std::string CacheManager::GetStorageConfigStr(RequestContext *request_context, const std::string &instance_id) const {
    SPAN_TRACER(request_context);
    const auto &trace_id = request_context->trace_id();
    auto all_configs = registry_manager_->data_storage_manager()->ListStorageConfig();
    auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (instance_info == nullptr) {
        PREFIX_LOG(WARN, "instance not found");
        return {};
    }
    const auto &instance_group_name = instance_info->instance_group_name();
    auto [ec, instance_group] = registry_manager_->GetInstanceGroup(request_context, instance_group_name);
    if (instance_group == nullptr) {
        PREFIX_LOG(WARN, "instance group not found: %s", instance_group_name.c_str());
        return {};
    }
    // storage_candidates only describes normal write placement. Tiered migration may move data to a
    // source/target storage that is intentionally not a normal write candidate, but workers still need
    // the corresponding SDK config to read copied data or write data directed by a migration mark.
    std::set<std::string_view> accessible_storage_names(instance_group->storage_candidates().begin(),
                                                        instance_group->storage_candidates().end());
    const auto cache_config = instance_group->cache_config();
    if (cache_config != nullptr) {
        for (const auto &strategy : cache_config->migration_strategies()) {
            if (strategy == nullptr) {
                continue;
            }
            accessible_storage_names.insert(strategy->source_storage_name());
            accessible_storage_names.insert(strategy->target_storage_name());
        }
    }
    // TODO : try optimize these copy operation
    std::vector<const StorageConfig *> result;
    for (const auto &config : all_configs) {
        // dummy backend is server-internal (testing only);
        // clients have no SDK for it and would fail on init
        if (config.type() == DataStorageType::DATA_STORAGE_TYPE_DUMMY) {
            continue;
        }
        if (accessible_storage_names.find(config.global_unique_name()) != accessible_storage_names.end()) {
            result.push_back(&config);
        }
    }
    return Jsonizable::ToJsonString(result);
}

ErrorCode CacheManager::GetCacheLocationByQueryType(MetaSearcher *meta_searcher,
                                                    RequestContext *request_context,
                                                    const std::string &instance_id,
                                                    QueryType query_type,
                                                    const KeyVector &keys,
                                                    const BlockMask &block_mask,
                                                    int32_t sw_size,
                                                    CacheLocationVector &cache_locations) const {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    auto policy = genSelectLocationPolicy(request_context, instance_id);
    if (policy == nullptr) {
        return EC_ERROR;
    }
    ErrorCode ec = EC_ERROR;
    switch (query_type) {
    case QueryType::QT_BATCH_GET: {
        ec = meta_searcher->BatchGetBestLocation(request_context, keys, cache_locations, policy.get());
        break;
    }
    case QueryType::QT_PREFIX_MATCH: {
        ec = meta_searcher->PrefixMatch(request_context, keys, block_mask, cache_locations, policy.get());
        break;
    }
    case QueryType::QT_REVERSE_ROLL_SW_MATCH: {
        if (keys.size() < sw_size || sw_size < 0) {
            request_context->error_tracer()->AddErrorMsg("QT_REVERSE_ROLL_SW_MATCH bad args");
            RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, EC_BADARGS, "bad keys size: %zu, %d", keys.size(), sw_size);
        }
        ec = meta_searcher->ReverseRollSlideWindowMatch(request_context, keys, sw_size, cache_locations, policy.get());
        break;
    }
    default:
        assert(false);
    }
    if (ec == EC_OK && query_type != QueryType::QT_PREFIX_MATCH) {
        auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
        if (instance_info == nullptr) {
            request_context->error_tracer()->AddErrorMsg("instance not found");
            RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, EC_INSTANCE_NOT_EXIST, "instance not found");
        }
        FillEmptyLocationSpecs(instance_info->location_spec_infos(), cache_locations);
    }
    return ec;
}

ErrorCode CacheManager::DoRecoverOnce() {
    ActivateEventCleanupCallbacks();
    if (!registry_manager_) {
        KVCM_LOG_ERROR("CacheManager do recover failed, registry_manager is nullptr");
        return EC_ERROR;
    }
    size_t error_count = 0;
    auto request_context = std::make_shared<RequestContext>("cache_manager_recover_trace");
    auto [ec1, instance_groups] = registry_manager_->ListInstanceGroup(request_context.get());
    if (ec1 != EC_OK) {
        KVCM_LOG_WARN("CacheManager ListInstanceGroup failed when recover, ec[%d], will retry later", ec1);
        return EC_ERROR;
    }
    for (const auto &instance_group : instance_groups) {
        std::string group_name = instance_group->name();
        auto [ec2, instance_infos] = registry_manager_->ListInstanceInfo(request_context.get(), group_name);
        if (ec2 != EC_OK) {
            KVCM_LOG_WARN("CacheManager ListInstanceInfo failed when recover, skip. ec[%d] instance_group name[%s]",
                          ec2,
                          group_name.c_str());
            ++error_count;
            continue;
        }
        for (const auto &instance_info : instance_infos) {
            auto [ec3, config_str] = RegisterInstance(request_context.get(),
                                                      group_name,
                                                      instance_info->instance_id(),
                                                      instance_info->block_size(),
                                                      instance_info->location_spec_infos(),
                                                      instance_info->model_deployment(),
                                                      instance_info->location_spec_groups(),
                                                      static_cast<QueryType>(instance_info->default_query_type()));
            if (ec3 != EC_OK) {
                KVCM_LOG_WARN("CacheManager RegisterInstance failed when recover, skip. ec[%d] instance_group "
                              "name[%s] instance_id[%s]",
                              ec3,
                              group_name.c_str(),
                              instance_info->instance_id().c_str());
                ++error_count;
                continue;
            }
            KVCM_LOG_INFO("CacheManager RegisterInstance success when recover, instance_id[%s], storage_config[%s]",
                          instance_info->instance_id().c_str(),
                          config_str.c_str());
        }
    }

    // CacheManager recover is only complete when RegistryManager recover is also complete
    if (!registry_manager_->IsRecoverComplete()) {
        KVCM_LOG_WARN("CacheManager recover waiting for RegistryManager recover to complete");
        ++error_count;
    }

    KVCM_LOG_INFO("CacheManager do recover once done, error_count[%lu]", error_count);
    return error_count > 0 ? EC_ERROR : EC_OK;
}

ErrorCode CacheManager::DoRecover() {
    auto ec = DoRecoverOnce();
    if (ec == EC_OK) {
        return EC_OK;
    }
    KVCM_LOG_WARN("CacheManager DoRecover partially failed, starting retry loop in background");
    StartRecoverRetryLoop();
    return EC_OK;
}

void CacheManager::StartRecoverRetryLoop() {
    StopRecoverRetryLoop();
    recover_retry_stop_.store(false);
    recover_retry_thread_ = std::thread([this]() {
        while (!recover_retry_stop_.load()) {
            for (int i = 0; i < 100 && !recover_retry_stop_.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (recover_retry_stop_.load()) {
                break;
            }
            KVCM_LOG_INFO("CacheManager recover retry loop executing...");
            auto ec = DoRecoverOnce();
            if (ec == EC_OK) {
                KVCM_LOG_INFO("CacheManager recover retry loop completed successfully, stopping retry");
                break;
            }
        }
    });
}

void CacheManager::StopRecoverRetryLoop() {
    recover_retry_stop_.store(true);
    if (recover_retry_thread_.joinable()) {
        recover_retry_thread_.join();
    }
}
void CacheManager::ClearEventCleanupCallbacks() {
    if (!registry_manager_ || !registry_manager_->data_storage_manager()) {
        return;
    }
    auto dsm = registry_manager_->data_storage_manager();
    for (const auto &name : dsm->GetAllStorageNames()) {
        auto backend = dsm->GetDataStorageBackend(name);
        auto event_backend = std::dynamic_pointer_cast<EventReportBackend>(std::move(backend));
        if (event_backend) {
            event_backend->SetCleanupCallback(nullptr);
        }
    }
}

void CacheManager::DeactivateEventCleanupCallbacks() {
    if (!event_cleanup_callback_state_) {
        return;
    }
    std::unique_lock<std::shared_mutex> callback_fence(event_cleanup_callback_state_->mutex);
    event_cleanup_callback_state_->accepting = false;
    ++event_cleanup_callback_state_->epoch;
}

void CacheManager::ActivateEventCleanupCallbacks() {
    if (!event_cleanup_callback_state_) {
        return;
    }
    std::unique_lock<std::shared_mutex> callback_fence(event_cleanup_callback_state_->mutex);
    event_cleanup_callback_state_->accepting = true;
}

ErrorCode CacheManager::DoCleanup() {
    if (cache_garbage_collector_) {
        cache_garbage_collector_->Stop();
    }
    ClearEventCleanupCallbacks();
    StopRecoverRetryLoop();
    DeactivateEventCleanupCallbacks();
    // aborting write session need meta indexer
    if (write_location_manager_) {
        write_location_manager_->DoCleanup();
    }
    if (meta_searcher_manager_) {
        meta_searcher_manager_->DoCleanup();
    }
    if (meta_indexer_manager_) {
        meta_indexer_manager_->DoCleanup();
    }
    if (metrics_recorder_) {
        metrics_recorder_->DoCleanup();
    }
    if (data_storage_selector_) {
        data_storage_selector_->DoCleanup();
    }

    return EC_OK;
}

std::unique_ptr<SelectLocationPolicy> CacheManager::genSelectLocationPolicy(RequestContext *request_context,
                                                                            const std::string &instance_id) const {
    const auto &trace_id = request_context->trace_id();
    auto all_storages = registry_manager_->data_storage_manager()->GetAllStorageNames();
    auto all_available_storages = registry_manager_->data_storage_manager()->GetAvailableStorages();
    if (all_available_storages.size() >= all_storages.size()) {
        return std::make_unique<StaticWeightSLPolicy>();
    }
    auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (instance_info == nullptr) {
        request_context->error_tracer()->AddErrorMsg("instance not found");
        PREFIX_LOG(WARN, "instance not found");
        return nullptr;
    }
    const auto &instance_group_name = instance_info->instance_group_name();
    auto [ec, instance_group] = registry_manager_->GetInstanceGroup(request_context, instance_group_name);
    if (instance_group == nullptr) {
        request_context->error_tracer()->AddErrorMsg("instance group not found");
        PREFIX_LOG(WARN, "instance group not found: %s", instance_group_name.c_str());
        return nullptr;
    }
    const auto &storage_candadates = instance_group->storage_candidates();
    std::vector<std::shared_ptr<DataStorageBackend>> group_storages;
    std::vector<std::shared_ptr<DataStorageBackend>> group_available_storages;
    group_storages.reserve(all_available_storages.size());
    group_available_storages.reserve(group_storages.size());
    for (const auto &candadate : storage_candadates) {
        if (auto iter = std::find_if(
                all_storages.begin(), all_storages.end(), [&candadate](const auto &name) { return candadate == name; });
            iter != all_storages.end()) {
            if (auto storage_backend = registry_manager_->data_storage_manager()->GetDataStorageBackend(*iter);
                storage_backend != nullptr) {
                group_storages.push_back(storage_backend);
            }
        }
        if (auto iter = std::find_if(all_available_storages.begin(),
                                     all_available_storages.end(),
                                     [&candadate](const auto &backend) {
                                         return candadate == backend->GetStorageConfig().global_unique_name();
                                     });
            iter != all_available_storages.end()) {
            group_available_storages.push_back(*iter);
        }
    }
    if (group_available_storages.size() >= group_storages.size()) {
        return std::make_unique<StaticWeightSLPolicy>();
    }
    if (group_available_storages.empty()) {
        request_context->error_tracer()->AddErrorMsg("all storages are unavailable");
        KVCM_INTERVAL_LOG_WARN(10, "all storages are unavailable!");
        return nullptr;
    }
    std::array<uint32_t, static_cast<std::size_t>(DataStorageType::COUNT)> data_storage_counts{};
    bool is_all_type_only_one = true;
    for (const auto &storage : group_storages) {
        size_t idx = static_cast<size_t>(storage->GetType());
        if (++data_storage_counts[idx] > 1) {
            is_all_type_only_one = false;
            break;
        }
    }
    if (is_all_type_only_one) {
        StaticWeightSLPolicy::WeightArray weight_array{};
        for (const auto &storage : group_available_storages) {
            size_t idx = static_cast<size_t>(storage->GetType());
            weight_array[idx] = 1;
        }
        return std::make_unique<DynamicWeightSLPoliy>(weight_array);
    }

    NamedStorageWeightedSLPolicy::WeightMap weight_map;
    for (const auto &storage : group_available_storages) {
        weight_map[storage->GetStorageConfig().global_unique_name()] = 1;
    }
    return std::make_unique<NamedStorageWeightedSLPolicy>(std::move(weight_map));
}

CheckLocDataExistFunc CacheManager::GetCheckLocDataExistFunc(const std::string &instance_id) const {
    return [this, instance_id](const CacheLocation &loc) -> bool {
        if (!registry_manager_ || !registry_manager_->data_storage_manager()) {
            return true;
        }

        if (IsEventReportStorageType(loc.type())) {
            auto event_backend_holder = LookupEventReportBackend(registry_manager_, instance_id, loc.type(), true);
            auto *event_backend = dynamic_cast<EventReportBackend *>(event_backend_holder.get());
            if (!event_backend || loc.type() != event_backend->GetStorageType()) {
                return false;
            }
            std::string reporter_medium;
            std::string reporter_host;
            if (!event_backend->ParseLocationId(loc.id(), reporter_medium, reporter_host)) {
                return false;
            }
            bool strict_query_visibility = false;
            std::string committed_version;
            if (!event_backend->GetQueryVisibilityState(
                    {instance_id, reporter_host}, strict_query_visibility, committed_version)) {
                return false;
            }
            return IsEventReportLocationReadable(loc, strict_query_visibility, committed_version);
        }

        std::vector<DataStorageUri> storage_uris;
        for (const auto &spec : loc.location_specs()) {
            if (const DataStorageUri uri{spec.uri()}; uri.Valid()) {
                storage_uris.emplace_back(uri);
            }
        }
        if (storage_uris.empty()) {
            return true;
        }

        const std::string storage_unique_name = storage_uris.front().GetHostName();
        const auto result = registry_manager_->data_storage_manager()->Exist(storage_unique_name, storage_uris, true);
        return result.size() == storage_uris.size() &&
               std::all_of(result.cbegin(), result.cend(), [](bool value) { return value; });
    };
}

MetaSearcher::CheckHostCacheLocationFunc
CacheManager::GetHostCacheStateCheckLocDataExistFunc(const std::string &instance_id) const {
    auto fallback = GetCheckLocDataExistFunc(instance_id);
    struct EventVisibilitySnapshot {
        std::shared_ptr<EventReportBackend> backend;
        EventReportBackend::QueryVisibilitySnapshot reporters;
    };
    struct EventVisibilitySnapshots {
        std::once_flag initialize_once;
        std::map<DataStorageType, EventVisibilitySnapshot> by_storage_type;
    };
    auto event_snapshots = std::make_shared<EventVisibilitySnapshots>();
    auto initialize_event_snapshots = [registry_manager = registry_manager_, instance_id, event_snapshots] {
        if (!registry_manager || !registry_manager->data_storage_manager()) {
            return;
        }
        const std::string group_name = registry_manager->GetInstanceGroupName(instance_id);
        const auto instance_group = registry_manager->GetInstanceGroupConfig(group_name);
        const auto storage_manager = registry_manager->data_storage_manager();
        if (!instance_group || !storage_manager) {
            return;
        }
        for (const auto &candidate_name : instance_group->event_report_storage_candidates()) {
            auto event_backend =
                std::dynamic_pointer_cast<EventReportBackend>(storage_manager->GetDataStorageBackend(candidate_name));
            if (!event_backend) {
                continue;
            }
            const DataStorageType storage_type = event_backend->GetStorageType();
            if (event_snapshots->by_storage_type.find(storage_type) != event_snapshots->by_storage_type.end()) {
                continue;
            }
            EventVisibilitySnapshot snapshot;
            snapshot.backend = std::move(event_backend);
            snapshot.backend->GetQueryVisibilitySnapshot(instance_id, snapshot.reporters);
            event_snapshots->by_storage_type.emplace(storage_type, std::move(snapshot));
        }
    };

    return [fallback = std::move(fallback),
            event_snapshots = std::move(event_snapshots),
            initialize_event_snapshots = std::move(initialize_event_snapshots)](
               const CacheLocation &location, MetaSearcher::HostCacheLocationInfo &out_info) -> bool {
        out_info = {};
        if (!IsEventReportStorageType(location.type())) {
            return fallback ? fallback(location) : true;
        }
        // Initialization is intentionally lazy: MetaSearcher reads metadata
        // before invoking this callback, so the request-level liveness/version
        // snapshot is taken after the potentially expensive metadata I/O.
        std::call_once(event_snapshots->initialize_once, initialize_event_snapshots);
        const auto snapshot_it = event_snapshots->by_storage_type.find(location.type());
        if (snapshot_it == event_snapshots->by_storage_type.end() || !snapshot_it->second.backend) {
            return false;
        }
        std::string_view reporter_medium;
        std::string_view reporter_host;
        if (!snapshot_it->second.backend->ParseLocationIdView(location.id(), reporter_medium, reporter_host)) {
            return false;
        }
        const auto reporter_it = snapshot_it->second.reporters.find(reporter_host);
        if (reporter_it == snapshot_it->second.reporters.end()) {
            return false;
        }
        if (!IsEventReportLocationReadable(
                location, reporter_it->second.strict, reporter_it->second.committed_version)) {
            return false;
        }
        out_info.has_reporter_identity = true;
        out_info.reporter_medium = reporter_medium;
        out_info.reporter_host = reporter_host;
        return true;
    };
}

SubmitDelReqFunc CacheManager::GetSubmitDelReqFunc(const std::string &instance_id) const {
    return [this, instance_id](const std::vector<std::int64_t> &blk_keys,
                               const std::vector<std::vector<std::string>> &loc_ids,
                               const std::vector<std::vector<std::string>> &expected_location_values,
                               bool metadata_only) -> void {
        CacheLocationDelRequest request;
        request.instance_id = instance_id;
        request.delay = std::chrono::seconds(0);
        request.block_keys = blk_keys;
        request.location_ids = loc_ids;
        request.expected_location_values = expected_location_values;
        request.metadata_only = metadata_only;
        if (reclaimer_task_supervisor_) {
            reclaimer_task_supervisor_->Submit(instance_id, std::move(request));
            KVCM_LOG_DEBUG("meta data del request submitted to reclaimer supervisor");
        } else {
            KVCM_LOG_WARN("meta data del request dropped: reclaimer supervisor is unavailable");
        }
    };
}

std::pair<ErrorCode, std::vector<CacheManager::HostCacheMatch>>
CacheManager::GetHostCacheState(RequestContext *request_context,
                                const std::string &instance_id,
                                QueryType query_type,
                                const KeyVector &block_cache_keys,
                                const std::vector<std::string> &medium_filter,
                                size_t p2p_host_count) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());

    MetaSearcher *meta_searcher = meta_searcher_manager_->GetMetaSearcher(instance_id);
    if (!meta_searcher) {
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN,
                                          EC_INSTANCE_NOT_EXIST,
                                          std::vector<HostCacheMatch>,
                                          "meta searcher not found for instance: %s",
                                          instance_id.c_str());
    }

    auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (!instance_info) {
        request_context->error_tracer()->AddErrorMsg("instance not found");
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(
            WARN, EC_INSTANCE_NOT_EXIST, std::vector<HostCacheMatch>, "instance not found");
    }
    if (query_type == QueryType::QT_UNSPECIFIED) {
        query_type = static_cast<QueryType>(instance_info->default_query_type());
        if (query_type == QueryType::QT_UNSPECIFIED) {
            RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, EC_ERROR, std::vector<HostCacheMatch>, "unknown query type");
        }
    }
    const bool use_eagle_pop = instance_info->model_deployment().use_eagle_pop();
    if (query_type != QueryType::QT_PREFIX_MATCH && query_type != QueryType::QT_PREFIX_MATCH_WITH_MAMBA) {
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN,
                                          EC_BADARGS,
                                          std::vector<HostCacheMatch>,
                                          "unsupported query type for GetHostCacheState: %s",
                                          QueryTypeToString(query_type).c_str());
    }

    PREFIX_LOG(DEBUG, "GetHostCacheState query_type [%s]", QueryTypeToString(query_type).c_str());

    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, request_key_count, block_cache_keys.size());
    auto query_scope = KVCM_METRICS_COLLECTOR_CHRONO_SCOPE(service_metrics_collector, ManagerPrefixMatch);
    const auto request_check_location = GetHostCacheStateCheckLocDataExistFunc(instance_id);
    std::vector<MetaSearcher::HostCacheMatch> host_matches;
    ErrorCode ec = EC_ERROR;
    switch (query_type) {
    case QueryType::QT_PREFIX_MATCH: {
        ec = meta_searcher->PrefixMatchByHost(request_context,
                                              block_cache_keys,
                                              use_eagle_pop,
                                              medium_filter,
                                              host_matches,
                                              &request_check_location,
                                              p2p_host_count);
        break;
    }
    case QueryType::QT_PREFIX_MATCH_WITH_MAMBA: {
        ec = meta_searcher->PrefixMatchWithMambaByHost(request_context,
                                                       block_cache_keys,
                                                       use_eagle_pop,
                                                       medium_filter,
                                                       instance_info->location_spec_groups(),
                                                       host_matches,
                                                       &request_check_location,
                                                       p2p_host_count);
        break;
    }
    default:
        assert(false);
    }
    query_scope = ChronoScopeGuard{};
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(
        WARN, ec, std::vector<HostCacheMatch>, "get host cache state failed for instance: %s", instance_id.c_str());

    std::vector<HostCacheMatch> result;
    result.reserve(host_matches.size());
    for (const auto &match : host_matches) {
        result.push_back(HostCacheMatch{match.host_ip_port, match.local, match.p2p_1_fetch, match.p2p_1_total_match});
    }

    return {EC_OK, std::move(result)};
}

} // namespace kv_cache_manager
