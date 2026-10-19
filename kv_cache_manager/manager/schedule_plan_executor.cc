#include "kv_cache_manager/manager/schedule_plan_executor.h"

#include <algorithm>
#include <cassert>
#include <exception>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

#define DEFINE_METRICS_NAME_FOR_SCHEDULE_PLAN_EXECUTOR(name)                                                           \
    DEFINE_METRICS_NAME_(SchedulePlanExecutor, schedule_plan_executor, name)

#define REGISTER_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(name)                                                              \
    REGISTER_METRICS_GAUGE_(metrics_registry_, schedule_plan_executor, name)

namespace {
template <typename Request>
Request BindStorageBackends(const Request &request, DataStorageManager &manager) {
    auto bound = request;
    if (!bound.storage_backends) {
        bound.storage_backends = manager.CaptureStorageBackends();
    }
    return bound;
}

PlanExecuteResult MakeErrorResult(ErrorCode error_code, std::string error_message) {
    return PlanExecuteResult{
        .status = error_code,
        .error_message = std::move(error_message),
    };
}

template <typename ResultType, typename... Args>
void HandleErrorPromise(const std::shared_ptr<std::promise<ResultType>> &promise,
                        ErrorCode error_code,
                        const std::string &err_msg_format,
                        Args &&...args) {
    std::string error_message = StringUtil::FormatString(err_msg_format, std::forward<Args>(args)...);
    promise->set_value(ResultType{
        .status = error_code,
        .error_message = std::move(error_message),
    });
}

const char *EventReportCleanupReasonName(EventReportBackend::MaintenanceCleanupReason reason) {
    switch (reason) {
    case EventReportBackend::MaintenanceCleanupReason::kStaleSnapshot:
        return "stale_snapshot";
    case EventReportBackend::MaintenanceCleanupReason::kDownHost:
        return "down_host";
    case EventReportBackend::MaintenanceCleanupReason::kRecoveryAbsentHost:
        return "recovery_absent_host";
    }
    return "unknown";
}
} // namespace

struct SchedulePlanExecutor::PromiseCompletion {
    explicit PromiseCompletion(std::shared_ptr<std::promise<PlanExecuteResult>> promise)
        : promise_(std::move(promise)) {}

    void Complete(PlanExecuteResult result) noexcept {
        if (completed_.exchange(true, std::memory_order_acq_rel)) {
            KVCM_LOG_ERROR("schedule plan executor promise completed more than once");
            return;
        }
        try {
            promise_->set_value(std::move(result));
        } catch (const std::exception &e) {
            KVCM_LOG_ERROR("complete schedule plan executor promise failed: %s", e.what());
        } catch (...) { KVCM_LOG_ERROR("complete schedule plan executor promise failed with unknown exception"); }
    }

    void Complete(ErrorCode error_code, const std::string &error_message) noexcept {
        Complete(MakeErrorResult(error_code, error_message));
    }

private:
    std::shared_ptr<std::promise<PlanExecuteResult>> promise_;
    std::atomic<bool> completed_{false};
};

struct SchedulePlanExecutor::PendingDeleteOperation {
    CacheLocationDelRequest request;
    std::shared_ptr<PromiseCompletion> completion;
    ScheduleTaskClass task_class;
    uint32_t retry_count{0};
    bool delay_after_admission{false};
    // Positive Delete acknowledgements belong to this exact operation, not
    // GC's weaker Exist/MightExist hints. Never delete an acknowledged URI
    // again: an allocator may already have reused its address for a new key.
    std::set<std::string> confirmed_deleted_uris;
    std::set<std::string> abandoned_uris;
    std::atomic<bool> finished{false};
};

DEFINE_METRICS_NAME_FOR_SCHEDULE_PLAN_EXECUTOR(waiting_task_count);
DEFINE_METRICS_NAME_FOR_SCHEDULE_PLAN_EXECUTOR(executing_task_count);
DEFINE_METRICS_NAME_FOR_SCHEDULE_PLAN_EXECUTOR(waiting_migration_task_count);
DEFINE_METRICS_NAME_FOR_SCHEDULE_PLAN_EXECUTOR(executing_migration_task_count);

SchedulePlanExecutor::SchedulePlanExecutor(unsigned int thread_count,
                                           const std::shared_ptr<MetaIndexerManager> &meta_manager,
                                           const std::shared_ptr<DataStorageManager> &storage_manager,
                                           const std::shared_ptr<MetricsRegistry> &metrics_registry,
                                           unsigned int migration_worker_budget)
    : meta_manager_(meta_manager)
    , data_storage_manager_(storage_manager)
    , metrics_registry_(metrics_registry)
    , stop_(false) {
    REGISTER_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(waiting_task_count);
    REGISTER_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(executing_task_count);
    REGISTER_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(waiting_migration_task_count);
    REGISTER_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(executing_migration_task_count);

    if (thread_count == 0)
        thread_count = 1;
    const std::size_t requested_migration_budget =
        migration_worker_budget == std::numeric_limits<unsigned int>::max() ? thread_count : migration_worker_budget;
    // CacheManager/ServerConfig enforce the production invariant budget < worker count. Keep the
    // standalone executor total here so an invalid direct caller cannot permanently strand migration tasks.
    migration_worker_budget_ =
        std::max<std::size_t>(1, std::min<std::size_t>(requested_migration_budget, thread_count));

    for (unsigned int i = 0; i < thread_count; ++i) {
        workers_.emplace_back([this]() { WorkerRoutine(); });
    }
    KVCM_LOG_INFO("SchedulePlanExecutor initialized with %u worker threads, migration worker budget %zu",
                  thread_count,
                  migration_worker_budget_);
}
void SchedulePlanExecutor::Stop() {
    KVCM_LOG_DEBUG("Stopping SchedulePlanExecutor...");
    std::vector<std::function<void()>> cancel_tasks;
    {
        // The stop predicate and both condition-variable waits are synchronized by
        // queue_mutex_. This prevents a worker from observing stop_ == false and
        // going to sleep after the shutdown notification has already been sent.
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
        const auto waiting_task_count = WaitingTaskCountLocked();
        std::size_t waiting_migration_task_count = 0;
        cancel_tasks.reserve(waiting_task_count);
        for (std::size_t class_idx = 0; class_idx < task_queues_.size(); ++class_idx) {
            auto &queue = task_queues_[class_idx];
            if (IsMigrationTaskClass(static_cast<ScheduleTaskClass>(class_idx))) {
                waiting_migration_task_count += queue.size();
            }
            for (const auto &scheduled_task : queue) {
                if (scheduled_task.cancel_task) {
                    cancel_tasks.emplace_back(scheduled_task.cancel_task);
                }
            }
            queue.clear();
        }
        if (waiting_task_count > 0) {
            METRICS_(schedule_plan_executor, waiting_task_count) -= waiting_task_count;
        }
        if (waiting_migration_task_count > 0) {
            METRICS_(schedule_plan_executor, waiting_migration_task_count) -= waiting_migration_task_count;
        }
    }
    condition_.notify_all();

    for (auto &cancel_task : cancel_tasks) {
        try {
            cancel_task();
        } catch (const std::exception &e) {
            KVCM_LOG_ERROR("cancel schedule plan executor task failed: %s", e.what());
        } catch (...) { KVCM_LOG_ERROR("cancel schedule plan executor task failed with unknown exception"); }
    }

    for (auto &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    KVCM_LOG_DEBUG("SchedulePlanExecutor stopped");
}

SchedulePlanExecutor::~SchedulePlanExecutor() { Stop(); }

bool SchedulePlanExecutor::IsMigrationTaskClass(ScheduleTaskClass task_class) {
    return task_class == ScheduleTaskClass::kMigrationPrepare ||
           task_class == ScheduleTaskClass::kMigrationContinuation;
}

std::size_t SchedulePlanExecutor::TaskClassIndex(ScheduleTaskClass task_class) {
    return static_cast<std::size_t>(task_class);
}

std::size_t SchedulePlanExecutor::WaitingTaskCountLocked() const {
    std::size_t total = 0;
    for (const auto &queue : task_queues_) {
        total += queue.size();
    }
    return total;
}

void SchedulePlanExecutor::WorkerRoutine() {
    while (true) {
        std::function<void()> task;
        bool is_migration_task = false;

        {
            auto wait_until_time = std::chrono::steady_clock::time_point::max();
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                return;
            }

            const auto now = std::chrono::steady_clock::now();
            // task_queues_ 的下标同时是优先级。每类只检查队首，避免 migration budget 满时
            // 在线性队列中扫描 ready reclaim task，把 queue_mutex_ 变成热点。
            for (std::size_t class_idx = 0; class_idx < task_queues_.size(); ++class_idx) {
                auto &queue = task_queues_[class_idx];
                if (queue.empty()) {
                    continue;
                }
                const auto task_class = static_cast<ScheduleTaskClass>(class_idx);
                const bool migration_class = IsMigrationTaskClass(task_class);
                if (migration_class && running_migration_tasks_ >= migration_worker_budget_) {
                    continue;
                }
                auto earliest_task = queue.begin();
                if (earliest_task->execute_time > now) {
                    wait_until_time = std::min(wait_until_time, earliest_task->execute_time);
                    continue;
                }
                task = earliest_task->task;
                is_migration_task = migration_class;
                queue.erase(earliest_task);
                if (is_migration_task) {
                    ++running_migration_tasks_;
                }
                break;
            }

            if (!task) {
                if (wait_until_time != std::chrono::steady_clock::time_point::max()) {
                    condition_.wait_until(lock, wait_until_time);
                } else {
                    // 队列为空，或只剩 budget 已满的 ready migration。migration 完成、
                    // 新 reclaim 入队和 Stop 都会通知这里重新选择。
                    condition_.wait(lock);
                }
                continue;
            }
        }

        if (task) {
            METRICS_(schedule_plan_executor, waiting_task_count) -= 1;
            METRICS_(schedule_plan_executor, executing_task_count) += 1;
            if (is_migration_task) {
                METRICS_(schedule_plan_executor, waiting_migration_task_count) -= 1;
                METRICS_(schedule_plan_executor, executing_migration_task_count) += 1;
            }
            try {
                task();
            } catch (const std::exception &e) {
                KVCM_LOG_ERROR("schedule plan executor task threw exception: %s", e.what());
            } catch (...) { KVCM_LOG_ERROR("schedule plan executor task threw unknown exception"); }
            METRICS_(schedule_plan_executor, executing_task_count) -= 1;
            if (is_migration_task) {
                METRICS_(schedule_plan_executor, executing_migration_task_count) -= 1;
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    assert(running_migration_tasks_ > 0);
                    --running_migration_tasks_;
                }
                condition_.notify_one();
            }
        }
    }
}

PlanExecuteResult SchedulePlanExecutor::DoLocationDelTask(const CacheLocationDelRequest &task,
                                                          std::set<std::string> *confirmed_deleted_uris,
                                                          std::set<std::string> *abandoned_uris,
                                                          const std::shared_ptr<MetaIndexer> &provided_indexer) {
    auto indexer = provided_indexer ? provided_indexer : meta_manager_->GetMetaIndexer(task.instance_id);
    if (!indexer) {
        return MakeErrorResult(EC_NOENT, "meta indexer not found for location delete");
    }
    PlanExecuteResult result;
    result.status = ErrorCode::EC_OK;

    if (task.block_keys.size() != task.location_ids.size() ||
        (!task.expected_location_values.empty() && task.expected_location_values.size() != task.block_keys.size())) {
        return MakeErrorResult(ErrorCode::EC_BADARGS,
                               StringUtil::FormatString("block_keys size %zu != location_ids size %zu",
                                                        task.block_keys.size(),
                                                        task.location_ids.size()));
    }

    MetaSearcher meta_searcher(indexer);
    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    auto ctx = std::make_shared<RequestContext>("schedule_plan_executor_call");
    ErrorCode get_locations_ec = EC_OK;
    if (task.authoritative_read) {
        const auto read_result = indexer->GetLocationsFromPrimary(ctx.get(), task.block_keys, location_maps);
        if (read_result.error_codes.size() != task.block_keys.size() ||
            std::any_of(read_result.error_codes.begin(), read_result.error_codes.end(), [](ErrorCode ec) {
                return ec != EC_OK && ec != EC_NOENT;
            })) {
            get_locations_ec = EC_ERROR;
        }
    } else {
        get_locations_ec = meta_searcher.BatchGetLocation(ctx.get(), task.block_keys, empty_mask, location_maps);
    }
    if (get_locations_ec != ErrorCode::EC_OK) {
        return MakeErrorResult(ErrorCode::EC_ERROR,
                               StringUtil::FormatString("Failed to get block locations, ec: %d", get_locations_ec));
    }
    if (location_maps.size() != task.block_keys.size()) {
        return MakeErrorResult(ErrorCode::EC_ERROR,
                               StringUtil::FormatString("block location result size %zu != block_keys size %zu",
                                                        location_maps.size(),
                                                        task.block_keys.size()));
    }

    std::vector<int64_t> block_keys_to_delete;
    std::vector<std::vector<MetaSearcher::LocationCADTask>> batch_cad_tasks;
    size_t total_locations_to_delete = 0;

    std::map<std::string, std::vector<DataStorageUri>> delete_uris_by_unique_name;
    for (size_t i = 0; i < task.block_keys.size(); i++) {
        auto &block_key = task.block_keys[i];
        auto &location_map = location_maps[i];
        auto &need_delete_location_ids = task.location_ids[i];

        std::vector<MetaSearcher::LocationCADTask> cad_tasks;

        if (!task.expected_location_values.empty() &&
            task.expected_location_values[i].size() != need_delete_location_ids.size()) {
            return MakeErrorResult(EC_BADARGS, "delete location snapshot shape mismatch");
        }
        for (size_t location_idx = 0; location_idx < need_delete_location_ids.size(); ++location_idx) {
            const auto &location_id = need_delete_location_ids[location_idx];
            auto iter = location_map.find(location_id);
            if (iter == location_map.end() || !iter->second) {
                continue;
            }
            if (iter->second->status() != CacheLocationStatus::CLS_DELETING) {
                continue;
            }
            const auto current_value = iter->second->ToJsonString();
            if (!task.expected_location_values.empty() &&
                current_value != task.expected_location_values[i][location_idx]) {
                continue; // A replacement with the same ID is not this delete's allocation.
            }
            if (!task.metadata_only && !IsEventReportStorageType(iter->second->type())) {
                for (const auto &loc_spec : iter->second->location_specs()) {
                    DataStorageUri uri(loc_spec.uri());
                    if (uri.Valid()) {
                        std::string storage_unique_name = uri.GetHostName();
                        delete_uris_by_unique_name[storage_unique_name].emplace_back(uri);
                    }
                }
            }
            cad_tasks.push_back({iter->first, CacheLocationStatus::CLS_DELETING, current_value});
        }
        if (!cad_tasks.empty()) {
            block_keys_to_delete.push_back(block_key);
            total_locations_to_delete += cad_tasks.size();
            batch_cad_tasks.emplace_back(std::move(cad_tasks));
        }
    }

    KVCM_LOG_DEBUG("Found %zu meta location(s) to delete", total_locations_to_delete);

    // delete storage uris
    auto request_context = std::make_shared<RequestContext>("location_del_task_trace");
    bool all_physical_deletes_succeeded = true;
    std::set<std::string> local_abandoned_uris;
    if (!abandoned_uris) {
        abandoned_uris = &local_abandoned_uris;
    }
    for (auto &storage_uris_pair : delete_uris_by_unique_name) {
        const std::string &storage_unique_name = storage_uris_pair.first;
        auto &storage_uris = storage_uris_pair.second;
        if (confirmed_deleted_uris != nullptr) {
            storage_uris.erase(std::remove_if(storage_uris.begin(),
                                              storage_uris.end(),
                                              [&](const DataStorageUri &uri) {
                                                  return confirmed_deleted_uris->count(uri.ToUriString()) != 0;
                                              }),
                               storage_uris.end());
        }
        storage_uris.erase(
            std::remove_if(storage_uris.begin(),
                           storage_uris.end(),
                           [&](const DataStorageUri &uri) { return abandoned_uris->count(uri.ToUriString()) != 0; }),
            storage_uris.end());
        if (!task.confirmed_missing_uris.empty()) {
            std::shared_ptr<DataStorageBackend> backend;
            if (task.storage_backends) {
                const auto expected = task.storage_backends->find(storage_unique_name);
                backend = expected == task.storage_backends->end() ? nullptr : expected->second;
            } else {
                backend = data_storage_manager_->GetDataStorageBackend(storage_unique_name);
            }
            if (!backend || backend->ShouldSkipConfirmedMissingBackendDelete()) {
                storage_uris.erase(std::remove_if(storage_uris.begin(),
                                                  storage_uris.end(),
                                                  [&](const DataStorageUri &uri) {
                                                      return task.confirmed_missing_uris.count(uri.ToUriString()) != 0;
                                                  }),
                                   storage_uris.end());
            }
        }
        if (storage_uris.empty()) {
            continue;
        }
        KVCM_LOG_DEBUG("Deleting %zu entries from storage: %s", storage_uris.size(), storage_unique_name.c_str());
        bool storage_removed = false;
        std::vector<ErrorCode> delete_results = data_storage_manager_->Delete(
            request_context.get(), storage_unique_name, storage_uris, nullptr, task.storage_backends, &storage_removed);
        if (storage_removed) {
            for (const auto &uri : storage_uris) {
                abandoned_uris->insert(uri.ToUriString());
            }
            continue; // Other storages in this batch still own their cleanup.
        }
        if (delete_results.size() != storage_uris.size()) {
            result.status = ErrorCode::EC_PARTIAL_OK;
            all_physical_deletes_succeeded = false;
            result.error_message =
                StringUtil::FormatString("storage delete result size %zu != request size %zu, instance[%s] storage[%s]",
                                         delete_results.size(),
                                         storage_uris.size(),
                                         task.instance_id.c_str(),
                                         storage_unique_name.c_str());
            KVCM_LOG_WARN("%s", result.error_message.c_str());
        }
        const auto result_count = std::min(delete_results.size(), storage_uris.size());
        if (confirmed_deleted_uris != nullptr && delete_results.size() == storage_uris.size()) {
            for (size_t i = 0; i < delete_results.size(); ++i) {
                if (delete_results[i] == EC_OK || delete_results[i] == EC_NOENT) {
                    confirmed_deleted_uris->insert(storage_uris[i].ToUriString());
                }
            }
        }
        size_t failed_count = 0;
        size_t first_failed_index = 0;
        for (size_t i = 0; i < result_count; ++i) {
            if (delete_results[i] != ErrorCode::EC_OK && delete_results[i] != ErrorCode::EC_NOENT) {
                if (failed_count == 0) {
                    first_failed_index = i;
                }
                ++failed_count;
                KVCM_LOG_WARN("storage delete failed, instance[%s] storage[%s] uri[%s] ec[%d]",
                              task.instance_id.c_str(),
                              storage_unique_name.c_str(),
                              storage_uris[i].ToUriString().c_str(),
                              static_cast<int>(delete_results[i]));
            }
        }
        if (failed_count > 0) {
            result.status = ErrorCode::EC_PARTIAL_OK;
            all_physical_deletes_succeeded = false;
            const std::string failure_message = StringUtil::FormatString(
                "storage delete failed, instance[%s] storage[%s] failed[%zu] total[%zu] first_uri[%s] first_ec[%d]",
                task.instance_id.c_str(),
                storage_unique_name.c_str(),
                failed_count,
                storage_uris.size(),
                storage_uris[first_failed_index].ToUriString().c_str(),
                static_cast<int>(delete_results[first_failed_index]));
            if (result.error_message.empty()) {
                result.error_message = failure_message;
            }
        }
    }

    if (!all_physical_deletes_succeeded) {
        // CLS_DELETING metadata is the retry anchor.  Never CAD it while any
        // URI may still exist. The executor keeps its future and references
        // pending, and retries the exact snapshot through resume_deleting.
        if (result.error_message.empty()) {
            result.error_message = "one or more physical storage deletes failed";
        }
        KVCM_LOG_WARN("retain %zu CLS_DELETING location(s) after physical delete failure for instance %s",
                      total_locations_to_delete,
                      task.instance_id.c_str());
        result.error_logged = true;
        return result;
    }

    // An administrative retirement is not physical-delete success. Leave the
    // exact metadata of affected Locations, while completing unrelated ones.
    for (size_t i = 0; i < block_keys_to_delete.size();) {
        const auto original = std::find(task.block_keys.begin(), task.block_keys.end(), block_keys_to_delete[i]);
        const auto &locations = location_maps[std::distance(task.block_keys.begin(), original)];
        auto &tasks = batch_cad_tasks[i];
        tasks.erase(std::remove_if(tasks.begin(),
                                   tasks.end(),
                                   [&](const auto &cad) {
                                       const auto &location = locations.at(cad.location_id);
                                       return std::any_of(
                                           location->location_specs().begin(),
                                           location->location_specs().end(),
                                           [&](const auto &spec) { return abandoned_uris->count(spec.uri()) != 0; });
                                   }),
                    tasks.end());
        if (tasks.empty()) {
            block_keys_to_delete.erase(block_keys_to_delete.begin() + i);
            batch_cad_tasks.erase(batch_cad_tasks.begin() + i);
        } else {
            ++i;
        }
    }
    result.cleanup_abandoned = !abandoned_uris->empty();

    // delete locations
    if (!batch_cad_tasks.empty()) {
        std::vector<std::vector<ErrorCode>> delete_meta_results;
        ErrorCode delete_meta_ec =
            meta_searcher.BatchCADLocationStatus(ctx.get(), block_keys_to_delete, batch_cad_tasks, delete_meta_results);
        if (delete_meta_ec != ErrorCode::EC_OK || delete_meta_results.size() != batch_cad_tasks.size()) {
            result.status = ErrorCode::EC_PARTIAL_OK;
            result.error_message =
                StringUtil::FormatString("location CAD failed, instance[%s] ec: %d, result size %zu != task size %zu",
                                         task.instance_id.c_str(),
                                         static_cast<int>(delete_meta_ec),
                                         delete_meta_results.size(),
                                         batch_cad_tasks.size());
            KVCM_LOG_WARN("%s", result.error_message.c_str());
        }
        const auto block_result_count = std::min(delete_meta_results.size(), batch_cad_tasks.size());
        for (size_t block_key_idx = 0; block_key_idx < block_result_count; ++block_key_idx) {
            auto &results = delete_meta_results[block_key_idx];
            if (results.size() != batch_cad_tasks[block_key_idx].size()) {
                result.status = ErrorCode::EC_PARTIAL_OK;
                KVCM_LOG_WARN("location CAD result size %zu != task size %zu, instance[%s] block key %ld",
                              results.size(),
                              batch_cad_tasks[block_key_idx].size(),
                              task.instance_id.c_str(),
                              block_keys_to_delete[block_key_idx]);
            }
            const auto location_result_count = std::min(results.size(), batch_cad_tasks[block_key_idx].size());
            for (size_t location_idx = 0; location_idx < location_result_count; location_idx++) {
                if (results[location_idx] != ErrorCode::EC_OK) {
                    result.status = ErrorCode::EC_PARTIAL_OK;
                    KVCM_LOG_WARN("Failed to CAD meta, instance[%s] key %ld, location: %s, error_code: %d",
                                  task.instance_id.c_str(),
                                  block_keys_to_delete[block_key_idx],
                                  batch_cad_tasks[block_key_idx][location_idx].location_id.c_str(),
                                  static_cast<int>(results[location_idx]));
                }
            }
        }
    }
    // All storage/CAD failures above have detailed diagnostics. Early returns
    // and worker exceptions keep the default false so callers still log them.
    result.error_logged = result.status != ErrorCode::EC_OK;
    return result;
}

bool SchedulePlanExecutor::SubmitRaw(std::function<void()> task,
                                     std::chrono::microseconds delay,
                                     std::function<void()> cancel_task,
                                     ScheduleTaskClass task_class) {
    const auto task_class_index = TaskClassIndex(task_class);
    if (task_class_index >= task_queues_.size()) {
        return false;
    }
    try {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stop_) {
                return false;
            }

            auto execute_time = std::chrono::steady_clock::now() + delay;
            uint64_t sequence_id = sequence_counter_.fetch_add(1, std::memory_order_relaxed);
            task_queues_[task_class_index].emplace(
                ScheduledTask{std::move(task), std::move(cancel_task), execute_time, sequence_id});
            METRICS_(schedule_plan_executor, waiting_task_count) += 1;
            if (IsMigrationTaskClass(task_class)) {
                METRICS_(schedule_plan_executor, waiting_migration_task_count) += 1;
            }
        }
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("enqueue schedule plan executor task failed: %s", e.what());
        return false;
    } catch (...) {
        KVCM_LOG_ERROR("enqueue schedule plan executor task failed with unknown exception");
        return false;
    }
    condition_.notify_one();
    return true;
}

std::future<PlanExecuteResult> SchedulePlanExecutor::Submit(const CacheMetaDelRequest &task) {
    return SubmitMetaDelete(task, ScheduleTaskClass::kSystem, meta_manager_->GetMetaIndexer(task.instance_id));
}

std::future<PlanExecuteResult> SchedulePlanExecutor::Submit(const CacheMetaDelRequest &task,
                                                            std::shared_ptr<MetaIndexer> indexer) {
    return SubmitMetaDelete(task, ScheduleTaskClass::kSystem, std::move(indexer));
}

std::future<PlanExecuteResult> SchedulePlanExecutor::SubmitMetaDelete(const CacheMetaDelRequest &request,
                                                                      ScheduleTaskClass task_class,
                                                                      std::shared_ptr<MetaIndexer> indexer) {
    auto task = BindStorageBackends(request, *data_storage_manager_);
    KVCM_LOG_DEBUG("Submitting meta delete task for instance_id: %s, block_keys count: %zu",
                   task.instance_id.c_str(),
                   task.block_keys.size());

    auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    std::future<PlanExecuteResult> future = promise->get_future();

    if (stop_) {
        HandleErrorPromise(promise, ErrorCode::EC_ERROR, "SchedulePlanExecutor stopped.");
        return future;
    }

    // 1. sync set block status to deleting
    if (indexer == nullptr) {
        HandleErrorPromise(promise, ErrorCode::EC_NOENT, "MetaIndexer %s not found", task.instance_id.c_str());
        return future;
    }

    MetaSearcher meta_searcher(indexer);

    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    auto request_context = std::make_shared<RequestContext>("schedule_plan_executor_call");
    ErrorCode get_locations_ec =
        meta_searcher.BatchGetLocation(request_context.get(), task.block_keys, empty_mask, location_maps);
    if (get_locations_ec != ErrorCode::EC_OK) {
        HandleErrorPromise(promise, ErrorCode::EC_ERROR, "Failed to get block locations");
        return future;
    }

    std::vector<int64_t> batch_cas_block_keys;
    std::vector<std::vector<MetaSearcher::LocationCASTask>> batch_cas_tasks;
    for (size_t block_key_idx = 0; block_key_idx < task.block_keys.size(); ++block_key_idx) {
        std::vector<MetaSearcher::LocationCASTask> cas_tasks;
        const auto &location_map = location_maps[block_key_idx];
        for (const auto &loc_kv : location_map) {
            if (!loc_kv.second) {
                continue;
            }
            cas_tasks.push_back({loc_kv.first, loc_kv.second->status(), CLS_DELETING, loc_kv.second->ToJsonString()});
        }
        if (cas_tasks.empty()) {
            continue;
        }
        batch_cas_block_keys.emplace_back(task.block_keys[block_key_idx]);
        batch_cas_tasks.emplace_back(std::move(cas_tasks));
    }

    if (batch_cas_block_keys.empty()) {
        promise->set_value({ErrorCode::EC_OK, ""});
        return future;
    }

    std::vector<std::vector<ErrorCode>> cas_results;
    ErrorCode update_ec =
        meta_searcher.BatchCASLocationStatus(request_context.get(), batch_cas_block_keys, batch_cas_tasks, cas_results);
    if (update_ec != ErrorCode::EC_OK) {
        KVCM_LOG_DEBUG("Location status BatchCASLocationStatus not ok, ec: %d", update_ec);
    }

    std::string error_message;
    CacheLocationDelRequest actual_task{task.instance_id, {}, {}, task.delay};
    actual_task.storage_backends = task.storage_backends;
    if (!FillActualTask(batch_cas_block_keys, batch_cas_tasks, cas_results, actual_task, error_message)) {
        HandleErrorPromise(promise, ErrorCode::EC_ERROR, "FillActualTask error: %s", error_message.c_str());
        return future;
    }
    if (actual_task.block_keys.empty()) {
        if (update_ec == ErrorCode::EC_OK) {
            promise->set_value(PlanExecuteResult{ErrorCode::EC_OK, ""});
        } else {
            HandleErrorPromise(promise,
                               update_ec,
                               "Failed to admit location delete metadata update, instance[%s]",
                               task.instance_id.c_str());
        }
        return future;
    }

    // Sync: ensure CAS(→DELETING) is persisted before scheduling Phase 2.
    if (!indexer->Sync(actual_task.block_keys)) {
        HandleErrorPromise(promise,
                           ErrorCode::EC_ERROR,
                           "Sync failed or timed out for location delete, instance[%s]",
                           task.instance_id.c_str());
        return future;
    }

    KVCM_LOG_DEBUG("Location statuses updated, submitting task to worker pool with delay: %lld microseconds",
                   static_cast<long long>(task.delay.count()));

    auto execute_task = [this, promise, actual_task, indexer = std::move(indexer)]() {
        try {
            promise->set_value(DoLocationDelTask(actual_task, nullptr, nullptr, indexer));
        } catch (const std::exception &e) {
            HandleErrorPromise(promise, ErrorCode::EC_ERROR, "location delete task threw exception: %s", e.what());
        } catch (...) {
            HandleErrorPromise(promise, ErrorCode::EC_ERROR, "location delete task threw unknown exception");
        }
    };
    auto cancel_task = [promise]() {
        HandleErrorPromise(promise, ErrorCode::EC_ERROR, "SchedulePlanExecutor stopped before task execution.");
    };
    bool submit_result = this->SubmitRaw(execute_task, task.delay, cancel_task, task_class);
    if (!submit_result) {
        HandleErrorPromise(promise, ErrorCode::EC_ERROR, "submit task failed");
        return future;
    }
    return future;
}

bool SchedulePlanExecutor::FillActualTask(
    const std::vector<int64_t> &batch_cas_block_keys,
    const std::vector<std::vector<MetaSearcher::LocationCASTask>> &batch_cas_tasks,
    const std::vector<std::vector<ErrorCode>> &batch_results,
    CacheLocationDelRequest &actual_task,
    std::string &error_message) {

    if (batch_results.size() != batch_cas_block_keys.size() || batch_results.size() != batch_cas_tasks.size()) {
        error_message = StringUtil::FormatString(
            "Size mismatch between batch_results(%zu), batch_cas_block_keys(%zu) and batch_cas_tasks(%zu).",
            batch_results.size(),
            batch_cas_block_keys.size(),
            batch_cas_tasks.size());
        KVCM_LOG_ERROR("%s", error_message.c_str());
        return false;
    }

    for (size_t key_idx = 0; key_idx < batch_results.size(); key_idx++) {
        auto &results = batch_results[key_idx];
        if (results.size() != batch_cas_tasks[key_idx].size()) {
            error_message = StringUtil::FormatString(
                "Location CAS result size mismatch for block key %ld: results(%zu), tasks(%zu).",
                batch_cas_block_keys[key_idx],
                results.size(),
                batch_cas_tasks[key_idx].size());
            KVCM_LOG_ERROR("%s", error_message.c_str());
            return false;
        }
        std::vector<std::string> location_ids;
        std::vector<std::string> expected_values;
        for (size_t location_idx = 0; location_idx < results.size(); location_idx++) {
            if (results[location_idx] != EC_OK) {
                KVCM_LOG_INFO("Location status CAS failed, block key: %ld, location_id: %s",
                              batch_cas_block_keys[key_idx],
                              batch_cas_tasks[key_idx][location_idx].location_id.c_str());
                continue;
            }
            const auto &cas = batch_cas_tasks[key_idx][location_idx];
            CacheLocation snapshot;
            if (!snapshot.FromJsonString(cas.expected_location_value)) {
                error_message = "delete admission did not retain a valid location snapshot";
                return false;
            }
            snapshot.set_status(CLS_DELETING);
            location_ids.push_back(cas.location_id);
            expected_values.push_back(snapshot.ToJsonString());
        }
        if (location_ids.empty()) {
            continue;
        }
        actual_task.block_keys.push_back(batch_cas_block_keys[key_idx]);
        actual_task.location_ids.emplace_back(std::move(location_ids));
        actual_task.expected_location_values.emplace_back(std::move(expected_values));
    }
    return true;
}

bool SchedulePlanExecutor::IsLocationBlockedByReadOnlyStorage(
    const CacheLocation &location, std::unordered_map<std::string, bool> &read_only_by_storage) const {
    if (!data_storage_manager_ || IsEventReportStorageType(location.type())) {
        return false;
    }
    for (const auto &loc_spec : location.location_specs()) {
        const DataStorageUri uri(loc_spec.uri());
        if (!uri.Valid()) {
            continue;
        }
        const std::string &storage_unique_name = uri.GetHostName();
        const auto memo = read_only_by_storage.find(storage_unique_name);
        if (memo != read_only_by_storage.end()) {
            if (memo->second) {
                return true;
            }
            continue;
        }
        const auto backend = data_storage_manager_->GetDataStorageBackend(storage_unique_name);
        const bool read_only = backend != nullptr && backend->InReadOnlyMode();
        read_only_by_storage.emplace(storage_unique_name, read_only);
        if (read_only) {
            return true;
        }
    }
    return false;
}

SchedulePlanExecutor::LocationDelAdmissionResult
SchedulePlanExecutor::PrepareDeleteTask(const CacheMetaDelRequest &task) {
    auto result = PrepareDeleteTaskImpl(task.instance_id, task.block_keys, nullptr, nullptr, task.delay, false, false);
    result.actual_task.storage_backends = task.storage_backends;
    result.retry_task.storage_backends = task.storage_backends;
    return result;
}

SchedulePlanExecutor::LocationDelAdmissionResult
SchedulePlanExecutor::PrepareDeleteTask(const CacheLocationDelRequest &request) {
    auto task = request;
    if (task.block_keys.size() != task.location_ids.size() ||
        (!task.expected_location_values.empty() && task.block_keys.size() != task.expected_location_values.size())) {
        LocationDelAdmissionResult admission_result;
        admission_result.result = MakeErrorResult(
            ErrorCode::EC_BADARGS, "block_keys, location_ids and expected_location_values sizes do not match");
        return admission_result;
    }
    if (!task.expected_location_values.empty()) {
        for (size_t i = 0; i < task.location_ids.size(); ++i) {
            if (task.location_ids[i].size() != task.expected_location_values[i].size()) {
                LocationDelAdmissionResult admission_result;
                admission_result.result = MakeErrorResult(
                    ErrorCode::EC_BADARGS,
                    StringUtil::FormatString(
                        "location_ids and expected_location_values sizes do not match at index %zu", i));
                return admission_result;
            }
        }
    }
    for (size_t i = 0; i < task.block_keys.size(); ++i) {
        const auto required = task.required_serving_location_values.find(task.block_keys[i]);
        if (required == task.required_serving_location_values.end()) {
            continue;
        }
        CacheLocation sibling;
        if (!sibling.FromJsonString(required->second) || sibling.id().empty() || sibling.status() != CLS_SERVING ||
            sibling.create_time() < 0 ||
            std::find(task.location_ids[i].begin(), task.location_ids[i].end(), sibling.id()) !=
                task.location_ids[i].end()) {
            LocationDelAdmissionResult result;
            result.result = MakeErrorResult(EC_BADARGS, "invalid serving sibling condition for location delete");
            return result;
        }
    }
    std::set<std::string> abandoned_uris;
    if (!task.expected_location_values.empty() && task.storage_backends) {
        auto operation = std::make_shared<PendingDeleteOperation>();
        operation->request = std::move(task);
        DropRetiredDeleteLocations(operation);
        task = std::move(operation->request);
        abandoned_uris = std::move(operation->abandoned_uris);
        if (task.block_keys.empty()) {
            LocationDelAdmissionResult result;
            result.result = MakeErrorResult(EC_ERROR, "storage registration removed before delete admission");
            result.result.cleanup_abandoned = !abandoned_uris.empty();
            result.result.safe_to_reuse_dst = false;
            result.abandoned_uris = std::move(abandoned_uris);
            return result;
        }
    }
    const auto *expected_location_values =
        task.expected_location_values.empty() ? nullptr : &task.expected_location_values;
    auto result = PrepareDeleteTaskImpl(task.instance_id,
                                        task.block_keys,
                                        &task.location_ids,
                                        expected_location_values,
                                        task.delay,
                                        task.authoritative_read,
                                        task.resume_deleting,
                                        &task.required_serving_location_values);
    result.actual_task.storage_backends = task.storage_backends;
    result.retry_task.storage_backends = task.storage_backends;
    result.actual_task.metadata_only = task.metadata_only;
    result.actual_task.confirmed_missing_uris = task.confirmed_missing_uris;
    result.retry_task.metadata_only = task.metadata_only;
    result.retry_task.confirmed_missing_uris = task.confirmed_missing_uris;
    result.retry_task.required_serving_location_values = task.required_serving_location_values;
    if (!abandoned_uris.empty() && result.result.status != EC_OK && result.actual_task.block_keys.empty() &&
        result.retry_task.block_keys.empty()) {
        // Retirement of one storage must not discard the other exact identities
        // when their metadata admission happens to fail in the same batch.
        result.retry_task = task;
    }
    result.abandoned_uris = std::move(abandoned_uris);
    return result;
}

SchedulePlanExecutor::LocationDelAdmissionResult
SchedulePlanExecutor::PrepareDeleteTaskImpl(const std::string &instance_id,
                                            const std::vector<int64_t> &block_keys,
                                            const std::vector<std::vector<std::string>> *target_location_ids,
                                            const std::vector<std::vector<std::string>> *expected_location_values,
                                            std::chrono::microseconds delay,
                                            bool authoritative_read,
                                            bool resume_deleting,
                                            const std::map<int64_t, std::string> *required_serving_location_values) {
    LocationDelAdmissionResult admission_result;
    admission_result.actual_task = CacheLocationDelRequest{instance_id, {}, {}, delay};
    admission_result.actual_task.authoritative_read = authoritative_read;

    if (stop_) {
        admission_result.result = MakeErrorResult(ErrorCode::EC_ERROR, "SchedulePlanExecutor stopped.");
        return admission_result;
    }
    std::shared_ptr<MetaIndexer> indexer = meta_manager_->GetMetaIndexer(instance_id);
    if (!indexer) {
        admission_result.result = MakeErrorResult(
            ErrorCode::EC_NOENT, StringUtil::FormatString("MetaIndexer %s not found", instance_id.c_str()));
        return admission_result;
    }
    MetaSearcher meta_searcher(indexer);
    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    auto request_context = std::make_shared<RequestContext>("schedule_plan_executor_call");
    ErrorCode get_locations_ec = ErrorCode::EC_OK;
    if (authoritative_read) {
        const auto get_result = indexer->GetLocationsFromPrimary(request_context.get(), block_keys, location_maps);
        if (get_result.error_codes.size() != block_keys.size()) {
            get_locations_ec = ErrorCode::EC_ERROR;
        } else {
            for (const ErrorCode ec : get_result.error_codes) {
                if (ec != ErrorCode::EC_OK && ec != ErrorCode::EC_NOENT) {
                    get_locations_ec = ErrorCode::EC_ERROR;
                    break;
                }
            }
        }
    } else {
        get_locations_ec = meta_searcher.BatchGetLocation(request_context.get(), block_keys, empty_mask, location_maps);
    }
    if (get_locations_ec != ErrorCode::EC_OK) {
        admission_result.result = MakeErrorResult(
            ErrorCode::EC_ERROR, StringUtil::FormatString("Failed to get block locations, ec: %d", get_locations_ec));
        return admission_result;
    }
    if (location_maps.size() != block_keys.size()) {
        admission_result.result = MakeErrorResult(
            ErrorCode::EC_ERROR,
            StringUtil::FormatString(
                "block location result size %zu != block_keys size %zu", location_maps.size(), block_keys.size()));
        return admission_result;
    }

    std::vector<int64_t> batch_cas_block_keys;
    std::vector<std::vector<MetaSearcher::LocationCASTask>> batch_cas_tasks;
    size_t read_only_deferred_locations = 0;
    std::unordered_map<std::string, bool> read_only_by_storage;
    // A null target list represents CacheMetaDelRequest and selects every non-deleting location.
    for (size_t block_key_idx = 0; block_key_idx < block_keys.size(); ++block_key_idx) {
        std::unordered_set<std::string> target_ids;
        if (target_location_ids != nullptr) {
            target_ids.insert((*target_location_ids)[block_key_idx].begin(),
                              (*target_location_ids)[block_key_idx].end());
        }
        std::unordered_map<std::string, std::string> expected_values_by_location;
        if (expected_location_values != nullptr) {
            for (size_t i = 0; i < (*target_location_ids)[block_key_idx].size(); ++i) {
                expected_values_by_location.emplace((*target_location_ids)[block_key_idx][i],
                                                    (*expected_location_values)[block_key_idx][i]);
            }
        }
        auto block_key = block_keys[block_key_idx];
        auto &location_map = location_maps[block_key_idx];
        CacheLocation required_sibling;
        const std::string *required_sibling_value = nullptr;
        if (required_serving_location_values) {
            const auto required = required_serving_location_values->find(block_key);
            if (required != required_serving_location_values->end()) {
                required_sibling_value = &required->second;
                if (!required_sibling.FromJsonString(*required_sibling_value)) {
                    admission_result.result = MakeErrorResult(EC_BADARGS, "invalid serving sibling snapshot");
                    return admission_result;
                }
            }
        }
        std::vector<MetaSearcher::LocationCASTask> location_cas_tasks;
        for (const auto &loc_kv : location_map) {
            if (!loc_kv.second) {
                continue;
            }
            const auto &location = *loc_kv.second;
            if (location.status() == CacheLocationStatus::CLS_DELETING) {
                if (!resume_deleting ||
                    (target_location_ids != nullptr && target_ids.find(location.id()) == target_ids.end())) {
                    continue;
                }
                if (expected_location_values != nullptr) {
                    const auto expected = expected_values_by_location.find(location.id());
                    CacheLocation original;
                    if (expected == expected_values_by_location.end() ||
                        !original.FromJsonString(expected->second)) {
                        continue;
                    }
                    original.set_status(CLS_DELETING);
                    if (original.ToJsonString() != location.ToJsonString()) {
                        continue; // This ID belongs to a replacement, not our cleanup.
                    }
                }
                // Memory-primary metadata may already be DELETING locally
                // even if the previous backup enqueue failed. Re-admit this
                // exact value before Sync; an empty queue is not evidence
                // that the deletion marker reached the secondary backend.
                location_cas_tasks.push_back(
                    {location.id(), CLS_DELETING, CLS_DELETING, location.ToJsonString()});
                continue;
            }
            if (target_location_ids != nullptr && target_ids.find(location.id()) == target_ids.end()) {
                continue;
            }
            if (IsLocationBlockedByReadOnlyStorage(location, read_only_by_storage)) {
                // Defer instead of admitting: the location stays CLS_SERVING so
                // reads keep hitting it and reclaim/GC can retry once the
                // backend is writable again.
                ++read_only_deferred_locations;
                continue;
            }
            if (required_sibling_value) {
                const auto sibling = location_map.find(required_sibling.id());
                if (sibling == location_map.end() || !sibling->second ||
                    sibling->second->ToJsonString() != *required_sibling_value) {
                    // The replacement copy is no longer usable. Retain the
                    // source instead of retrying its deletion indefinitely.
                    continue;
                }
            }
            std::string expected_location_value;
            if (expected_location_values != nullptr) {
                const auto expected = expected_values_by_location.find(location.id());
                if (expected == expected_values_by_location.end() || location.ToJsonString() != expected->second) {
                    continue; // stable location was refreshed after the cleanup scan
                }
                expected_location_value = expected->second;
            } else {
                // Always bind the status transition to the exact location
                // snapshot.  A refresh between admission and CAS must not be
                // deleted as if it were the older object.
                expected_location_value = location.ToJsonString();
            }
            location_cas_tasks.push_back({location.id(),
                                          location.status(),
                                          CacheLocationStatus::CLS_DELETING,
                                          std::move(expected_location_value)});
            if (required_sibling_value) {
                auto &cas = location_cas_tasks.back();
                cas.expected_source_location_id = required_sibling.id();
                cas.expected_source_create_time = required_sibling.create_time();
                cas.expected_source_status = CLS_SERVING;
                cas.expected_source_location_value = *required_sibling_value;
            }
        }
        if (location_cas_tasks.empty()) {
            continue;
        }
        batch_cas_block_keys.push_back(block_key);
        batch_cas_tasks.emplace_back(std::move(location_cas_tasks));
    }
    if (read_only_deferred_locations > 0) {
        KVCM_INTERVAL_LOG_WARN(10,
                               "location delete deferred by read-only storage, instance[%s] deferred_locations[%zu] "
                               "admitted_blocks[%zu]",
                               instance_id.c_str(),
                               read_only_deferred_locations,
                               batch_cas_tasks.size());
    }
    if (batch_cas_block_keys.empty()) {
        return admission_result;
    }

    if (resume_deleting) {
        // A CAS loser is not proof that the old allocation disappeared:
        // another deletion may have changed only local-primary metadata to
        // DELETING while its backup admission failed. Keep every candidate,
        // including this batch's CAS winners, until an authoritative reread
        // either admits that exact identity or proves it has been replaced.
        admission_result.retry_task = CacheLocationDelRequest{instance_id, batch_cas_block_keys, {}, delay};
        for (const auto &tasks : batch_cas_tasks) {
            std::vector<std::string> ids;
            std::vector<std::string> values;
            for (const auto &task : tasks) {
                ids.push_back(task.location_id);
                values.push_back(task.expected_location_value);
            }
            admission_result.retry_task.location_ids.emplace_back(std::move(ids));
            admission_result.retry_task.expected_location_values.emplace_back(std::move(values));
        }
    }

    std::vector<std::vector<ErrorCode>> batch_results;
    ErrorCode update_ec = meta_searcher.BatchCASLocationStatus(
        request_context.get(), batch_cas_block_keys, batch_cas_tasks, batch_results, authoritative_read);
    if (update_ec != ErrorCode::EC_OK) {
        KVCM_LOG_DEBUG("Location status BatchCASLocationStatus not ok, ec: %d", update_ec);
    }

    std::string error_message;
    if (!FillActualTask(
            batch_cas_block_keys, batch_cas_tasks, batch_results, admission_result.actual_task, error_message)) {
        admission_result.result = MakeErrorResult(
            ErrorCode::EC_ERROR, StringUtil::FormatString("FillActualTask error: %s", error_message.c_str()));
        return admission_result;
    }
    if (resume_deleting || admission_result.actual_task.block_keys.empty()) {
        for (const auto &results : batch_results) {
            for (const auto ec : results) {
                if (ec != EC_OK && ec != EC_NOENT && ec != EC_MISMATCH) {
                    // Preserve the RMW aggregate contract: if every key's
                    // backup admission failed, the operation is EC_ERROR
                    // even when a per-location slot reports EC_PARTIAL_OK.
                    admission_result.result = MakeErrorResult(update_ec != EC_OK ? update_ec : ec,
                                                               "delete metadata admission must be retried");
                    return admission_result;
                }
            }
        }
    }
    if (resume_deleting) {
        for (const auto &results : batch_results) {
            if (std::find(results.begin(), results.end(), EC_MISMATCH) != results.end()) {
                admission_result.result = MakeErrorResult(EC_MISMATCH, "delete identity changed during admission; retry");
                return admission_result;
            }
        }
    }
    if (admission_result.actual_task.block_keys.empty()) {
        if (update_ec != ErrorCode::EC_OK) {
            admission_result.result = MakeErrorResult(
                update_ec,
                StringUtil::FormatString("Failed to admit location delete metadata update, instance[%s]",
                                         instance_id.c_str()));
        }
        return admission_result;
    }

    // Newly transitioned and resumed DELETING keys have both passed backup
    // admission above; complete their persistence barrier before physical IO.
    if (!indexer->Sync(admission_result.actual_task.block_keys)) {
        admission_result.result =
            MakeErrorResult(ErrorCode::EC_ERROR,
                            StringUtil::FormatString("Sync failed or timed out for location delete, instance[%s]",
                                                     instance_id.c_str()));
        return admission_result;
    }
    admission_result.indexer = std::move(indexer);
    admission_result.needs_physical_delete = !admission_result.actual_task.block_keys.empty();
    return admission_result;
}

void SchedulePlanExecutor::RunDeleteAdmission(const std::shared_ptr<PromiseCompletion> &completion,
                                              std::chrono::microseconds delay,
                                              const std::function<LocationDelAdmissionResult()> &prepare,
                                              ScheduleTaskClass task_class) {
    try {
        auto admission_result = prepare();
        if (admission_result.result.status != EC_OK && !admission_result.retry_task.block_keys.empty()) {
            StartLocationDelete(completion,
                                std::move(admission_result.retry_task),
                                std::chrono::seconds(1),
                                task_class,
                                true,
                                std::move(admission_result.abandoned_uris));
            return;
        }
        if (admission_result.actual_task.block_keys.empty()) {
            if (!admission_result.abandoned_uris.empty()) {
                admission_result.result.status = EC_ERROR;
                admission_result.result.cleanup_abandoned = true;
                admission_result.result.safe_to_reuse_dst = false;
                admission_result.result.error_message = "storage registration removed; unfinished cleanup abandoned";
            }
            completion->Complete(std::move(admission_result.result));
            return;
        }

        KVCM_LOG_DEBUG("Location statuses updated, submitting task to worker pool with delay: %lld microseconds",
                       static_cast<long long>(delay.count()));
        auto actual_task = std::move(admission_result.actual_task);
        auto indexer = std::move(admission_result.indexer);
        auto execute_task = [this, completion, actual_task, indexer = std::move(indexer)]() {
            try {
                completion->Complete(DoLocationDelTask(actual_task, nullptr, nullptr, indexer));
            } catch (const std::exception &e) {
                completion->Complete(ErrorCode::EC_ERROR,
                                     StringUtil::FormatString("location delete task threw exception: %s", e.what()));
            } catch (...) { completion->Complete(ErrorCode::EC_ERROR, "location delete task threw unknown exception"); }
        };
        auto cancel_task = [completion]() {
            completion->Complete(ErrorCode::EC_ERROR, "SchedulePlanExecutor stopped before physical delete.");
        };
        if (!SubmitRaw(execute_task, delay, cancel_task, task_class)) {
            completion->Complete(ErrorCode::EC_ERROR, "submit physical delete task failed");
        }
    } catch (const std::exception &e) {
        completion->Complete(ErrorCode::EC_ERROR,
                             StringUtil::FormatString("location delete admission threw exception: %s", e.what()));
    } catch (...) { completion->Complete(ErrorCode::EC_ERROR, "location delete admission threw unknown exception"); }
}

void SchedulePlanExecutor::StartLocationDelete(const std::shared_ptr<PromiseCompletion> &completion,
                                               CacheLocationDelRequest task,
                                               std::chrono::microseconds delay,
                                               ScheduleTaskClass task_class,
                                               bool readmit,
                                               std::set<std::string> abandoned_uris) {
    auto operation = std::make_shared<PendingDeleteOperation>();
    operation->request = std::move(task);
    operation->request.resume_deleting = true;
    operation->request.authoritative_read = true;
    operation->completion = completion;
    operation->abandoned_uris = std::move(abandoned_uris);
    operation->task_class = task_class;
    operation->delay_after_admission = readmit && operation->request.delay.count() > 0;
    for (const auto &values : operation->request.expected_location_values) {
        for (const auto &value : values) {
            CacheLocation location;
            if (!location.FromJsonString(value)) {
                completion->Complete(EC_BADARGS, "delete admission snapshot is invalid");
                return;
            }
        }
    }
    QueueLocationDelete(operation, delay, readmit);
}

void SchedulePlanExecutor::QueueLocationDelete(const std::shared_ptr<PendingDeleteOperation> &operation,
                                               std::chrono::microseconds delay,
                                               bool readmit) {
    auto cancel = [this, operation]() {
        FinishLocationDelete(operation, MakeErrorResult(EC_ERROR, "SchedulePlanExecutor stopped before delete completed."));
    };
    if (!SubmitRaw([this, operation, readmit]() { RunLocationDelete(operation, readmit); },
                   delay,
                   cancel,
                   operation->task_class)) {
        cancel();
    }
}

void SchedulePlanExecutor::RunLocationDelete(const std::shared_ptr<PendingDeleteOperation> &operation, bool readmit) {
    try {
        DropRetiredDeleteLocations(operation);
        if (operation->request.block_keys.empty()) {
            FinishLocationDelete(operation, PlanExecuteResult{EC_OK, ""});
            return;
        }
        CacheLocationDelRequest actual_task;
        const CacheLocationDelRequest *physical_task = &operation->request;
        if (readmit) {
            auto admission = PrepareDeleteTask(operation->request);
            operation->abandoned_uris.insert(admission.abandoned_uris.begin(), admission.abandoned_uris.end());
            if (admission.result.cleanup_abandoned) {
                FinishLocationDelete(operation, std::move(admission.result));
                return;
            }
            if (admission.result.status != EC_OK) {
                RetryLocationDelete(operation);
                return;
            }
            if (!admission.needs_physical_delete) {
                FinishLocationDelete(operation, std::move(admission.result));
                return;
            }
            if (operation->delay_after_admission) {
                operation->delay_after_admission = false;
                // Preserve the caller's grace interval after the first
                // successful Sync, even when that barrier initially failed.
                // The retained request may contain pre-CAS snapshots after
                // an admission race. Re-read after the grace interval rather
                // than treating those identities as admitted DELETING values.
                QueueLocationDelete(operation, operation->request.delay, true);
                return;
            }
            actual_task = std::move(admission.actual_task);
            physical_task = &actual_task;
        }
        auto result = DoLocationDelTask(*physical_task, &operation->confirmed_deleted_uris, &operation->abandoned_uris);
        if (result.status == EC_OK) {
            if (readmit && (actual_task.block_keys != operation->request.block_keys ||
                           actual_task.location_ids != operation->request.location_ids)) {
                // A subset may have disappeared or failed a conditional
                // admission. Keep the original responsibility until a fresh
                // read proves no original allocation remains to be deleted.
                QueueLocationDelete(operation, std::chrono::microseconds(0), true);
            } else {
                FinishLocationDelete(operation, std::move(result));
            }
        } else {
            RetryLocationDelete(operation);
        }
    } catch (const std::exception &e) {
        KVCM_LOG_WARN("delete attempt threw; retain exact cleanup for retry: %s", e.what());
        RetryLocationDelete(operation);
    } catch (...) {
        KVCM_LOG_WARN("delete attempt threw; retain exact cleanup for retry");
        RetryLocationDelete(operation);
    }
}

void SchedulePlanExecutor::DropRetiredDeleteLocations(const std::shared_ptr<PendingDeleteOperation> &operation) {
    auto &task = operation->request;
    if (!task.storage_backends || task.metadata_only) {
        return;
    }
    if (task.expected_location_values.size() != task.block_keys.size() ||
        task.location_ids.size() != task.block_keys.size()) {
        return; // Only identity-bound requests may be pruned without rereading metadata.
    }
    // Run before metadata read/Sync: a removed storage must not leave its task
    // retrying forever merely because the metadata backend is also unavailable.
    for (size_t i = 0; i < task.block_keys.size();) {
        if (task.expected_location_values[i].size() != task.location_ids[i].size()) {
            return;
        }
        for (size_t j = 0; j < task.location_ids[i].size();) {
            CacheLocation location;
            if (!location.FromJsonString(task.expected_location_values[i][j]) ||
                IsEventReportStorageType(location.type())) {
                ++j;
                continue;
            }
            bool retired = false;
            bool remaining = false;
            for (const auto &spec : location.location_specs()) {
                const DataStorageUri uri(spec.uri());
                if (!uri.Valid()) {
                    continue;
                }
                const auto expected = task.storage_backends->find(uri.GetHostName());
                const auto backend = expected == task.storage_backends->end() ? nullptr : expected->second;
                if (task.confirmed_missing_uris.count(spec.uri()) != 0 &&
                    (!backend || backend->ShouldSkipConfirmedMissingBackendDelete())) {
                    continue;
                }
                if (!data_storage_manager_->IsStorageBackendCurrent(task.storage_backends, uri.GetHostName())) {
                    operation->abandoned_uris.insert(spec.uri());
                }
                if (operation->abandoned_uris.count(spec.uri()) != 0) {
                    retired = true;
                } else if (operation->confirmed_deleted_uris.count(spec.uri()) == 0) {
                    remaining = true;
                }
            }
            if (retired && !remaining) {
                task.location_ids[i].erase(task.location_ids[i].begin() + j);
                task.expected_location_values[i].erase(task.expected_location_values[i].begin() + j);
            } else {
                ++j;
            }
        }
        if (task.location_ids[i].empty()) {
            task.block_keys.erase(task.block_keys.begin() + i);
            task.location_ids.erase(task.location_ids.begin() + i);
            task.expected_location_values.erase(task.expected_location_values.begin() + i);
        } else {
            ++i;
        }
    }
}

void SchedulePlanExecutor::RetryLocationDelete(const std::shared_ptr<PendingDeleteOperation> &operation) {
    // Keep the same future pending. Delayed tasks consume no worker while
    // waiting. Positively acknowledged URIs are never sent again, including
    // after partial physical success or a later metadata CAD failure. Unknown
    // outcomes still require the backend's safe retry/identity contract.
    const auto seconds = std::min<uint32_t>(30, 1u << std::min<uint32_t>(operation->retry_count, 5));
    operation->retry_count = std::min<uint32_t>(operation->retry_count + 1, 5);
    QueueLocationDelete(operation, std::chrono::seconds(seconds), true);
}

void SchedulePlanExecutor::FinishLocationDelete(const std::shared_ptr<PendingDeleteOperation> &operation,
                                                PlanExecuteResult result) {
    if (operation->finished.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    if (!operation->abandoned_uris.empty()) {
        result.cleanup_abandoned = true;
        result.safe_to_reuse_dst = false;
        result.status = EC_ERROR;
        result.error_message = "storage registration removed; unfinished cleanup abandoned";
        result.error_logged = false;
    }
    operation->completion->Complete(std::move(result));
}

std::future<PlanExecuteResult> SchedulePlanExecutor::Submit(const CacheLocationDelRequest &task) {
    return SubmitLocationDelete(task, ScheduleTaskClass::kSystem);
}

std::future<PlanExecuteResult> SchedulePlanExecutor::SubmitLocationDelete(const CacheLocationDelRequest &request,
                                                                          ScheduleTaskClass task_class) {
    auto task = BindStorageBackends(request, *data_storage_manager_);
    KVCM_LOG_DEBUG("Submitting location delete task for instance_id: %s, block_keys count: %zu",
                   task.instance_id.c_str(),
                   task.block_keys.size());

    auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    auto future = promise->get_future();
    auto completion = std::make_shared<PromiseCompletion>(promise);
    RunDeleteAdmission(
        completion, task.delay, [this, &task]() { return PrepareDeleteTask(task); }, task_class);
    return future;
}

AsyncDeleteSubmitResult SchedulePlanExecutor::SubmitAsync(const CacheMetaDelRequest &request) {
    auto task = BindStorageBackends(request, *data_storage_manager_);
    KVCM_LOG_DEBUG("Submitting async meta delete task for instance_id: %s, block_keys count: %zu",
                   task.instance_id.c_str(),
                   task.block_keys.size());
    return SubmitDeleteTaskAsync(task.delay, [this, task]() { return PrepareDeleteTask(task); });
}

AsyncDeleteSubmitResult SchedulePlanExecutor::SubmitAsync(const CacheLocationDelRequest &request) {
    return SubmitAsync(request, ScheduleTaskClass::kReclaim);
}

AsyncDeleteSubmitResult SchedulePlanExecutor::SubmitAsync(const CacheLocationDelRequest &request,
                                                          ScheduleTaskClass task_class) {
    auto task = BindStorageBackends(request, *data_storage_manager_);
    KVCM_LOG_DEBUG("Submitting async location delete task for instance_id: %s, block_keys count: %zu",
                   task.instance_id.c_str(),
                   task.block_keys.size());

    return SubmitDeleteTaskAsync(task.delay, [this, task]() { return PrepareDeleteTask(task); }, task_class);
}

AsyncDeleteSubmitResult SchedulePlanExecutor::SubmitAsync(const EventReportMetadataDelRequest &task) {
    if (task.instance_id.empty() || task.block_keys.empty() || task.block_keys.size() != task.targets.size()) {
        return {};
    }
    std::set<int64_t> unique_block_keys;
    if (!std::all_of(task.block_keys.begin(), task.block_keys.end(), [&](int64_t block_key) {
            return unique_block_keys.insert(block_key).second;
        })) {
        return {};
    }
    for (const auto &targets : task.targets) {
        if (targets.empty()) {
            return {};
        }
        std::set<std::string> unique_location_ids;
        for (const auto &target : targets) {
            if (target.location_id.empty() || target.expected_location_value.empty() ||
                target.backend_unique_name.empty() || target.expected_backend.expired() ||
                !IsEventReportStorageType(target.storage_type) ||
                target.cleanup_token.reporter_key.instance_id != task.instance_id ||
                target.cleanup_token.reporter_key.host_ip_port.empty() ||
                !unique_location_ids.insert(target.location_id).second) {
                return {};
            }
        }
    }

    auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    auto future = promise->get_future();
    auto completion = std::make_shared<PromiseCompletion>(promise);
    auto execute_task = [this, completion, task]() {
        try {
            completion->Complete(DoEventReportMetadataDelTask(task));
        } catch (const std::exception &e) {
            completion->Complete(
                EC_ERROR, StringUtil::FormatString("EventReport metadata delete threw exception: %s", e.what()));
        } catch (...) { completion->Complete(EC_ERROR, "EventReport metadata delete threw unknown exception"); }
    };
    auto cancel_task = [completion]() {
        completion->Complete(EC_ERROR, "SchedulePlanExecutor stopped before EventReport metadata delete.");
    };
    if (!SubmitRaw(execute_task, std::chrono::microseconds::zero(), cancel_task, ScheduleTaskClass::kReclaim)) {
        return {};
    }
    return AsyncDeleteSubmitResult{true, std::move(future)};
}

PlanExecuteResult
SchedulePlanExecutor::DoEventReportMetadataDelTask(const EventReportMetadataDelRequest &task) {
    auto indexer = meta_manager_ ? meta_manager_->GetMetaIndexer(task.instance_id) : nullptr;
    if (!indexer) {
        return MakeErrorResult(EC_NOENT,
                               StringUtil::FormatString("MetaIndexer %s not found", task.instance_id.c_str()));
    }
    MetaSearcher meta_searcher(indexer);
    size_t completed_targets = 0;
    size_t hard_error_targets = 0;
    const auto record_delete_result = [this](EventReportBackend::MaintenanceCleanupReason reason,
                                             const char *status) noexcept {
        if (!metrics_registry_) {
            return;
        }
        try {
            metrics_registry_->GetCounter(
                "cache_gc.event_report_delete_location_count",
                MetricsTags{{"reason", EventReportCleanupReasonName(reason)}, {"status", status}}) += 1;
        } catch (const std::exception &e) {
            KVCM_LOG_ERROR("record EventReport metadata delete metric failed: %s", e.what());
        } catch (...) { KVCM_LOG_ERROR("record EventReport metadata delete metric failed with unknown exception"); }
    };

    for (size_t key_index = 0; key_index < task.block_keys.size(); ++key_index) {
        const auto &targets = task.targets[key_index];
        using LeaseKey = std::tuple<std::string, std::string, uint64_t, int, std::string, uint64_t>;
        std::map<std::string, std::shared_ptr<EventReportBackend>> cleanup_backends;
        std::map<LeaseKey, std::pair<std::shared_ptr<EventReportBackend>, EventReportBackend::MaintenanceCleanupToken>>
            cleanup_tokens;
        bool key_stale = false;
        bool key_busy = false;
        for (const auto &target : targets) {
            if (!IsEventReportStorageType(target.storage_type) ||
                target.cleanup_token.reporter_key.instance_id != task.instance_id) {
                key_stale = true;
                break;
            }
            auto expected_backend = target.expected_backend.lock();
            auto current_backend = data_storage_manager_
                                       ? std::dynamic_pointer_cast<EventReportBackend>(
                                             data_storage_manager_->GetDataStorageBackend(target.backend_unique_name))
                                       : nullptr;
            if (!expected_backend || !current_backend || current_backend.get() != expected_backend.get() ||
                current_backend->GetStorageConfig().global_unique_name() != target.backend_unique_name ||
                current_backend->GetStorageType() != target.storage_type) {
                key_stale = true;
                break;
            }
            if (!current_backend->Available()) {
                key_busy = true;
                break;
            }
            const auto [backend_it, backend_inserted] =
                cleanup_backends.emplace(target.backend_unique_name, current_backend);
            if (!backend_inserted && backend_it->second.get() != current_backend.get()) {
                key_stale = true;
                break;
            }
            const auto &token = target.cleanup_token;
            CacheLocation expected_location;
            std::string reporter_medium;
            std::string reporter_host;
            if (!expected_location.FromJsonString(target.expected_location_value) ||
                expected_location.id() != target.location_id || expected_location.type() != target.storage_type ||
                expected_location.status() != CLS_SERVING ||
                !current_backend->ParseLocationId(target.location_id, reporter_medium, reporter_host) ||
                reporter_host != token.reporter_key.host_ip_port) {
                key_stale = true;
                break;
            }
            cleanup_tokens.emplace(LeaseKey{target.backend_unique_name,
                                            token.reporter_key.host_ip_port,
                                            token.lifecycle_generation,
                                            static_cast<int>(token.reason),
                                            token.committed_version,
                                            token.snapshot_attempt_epoch},
                                   std::make_pair(std::move(current_backend), token));
        }

        std::vector<EventReportBackend::MaintenanceBackendLease> backend_leases;
        if (!key_stale && !key_busy) {
            backend_leases.reserve(cleanup_backends.size());
            for (const auto &[_, backend] : cleanup_backends) {
                EventReportBackend::MaintenanceBackendLease lease;
                if (backend->AcquireMaintenanceBackendLease(lease) !=
                    EventReportBackend::CleanupLeaseAcquireResult::kAcquired) {
                    key_busy = true;
                    break;
                }
                backend_leases.push_back(std::move(lease));
            }
        }

        std::vector<EventReportBackend::LifecycleMutationLease> leases;
        if (!key_stale && !key_busy) {
            leases.reserve(cleanup_tokens.size());
            for (const auto &[_, backend_and_token] : cleanup_tokens) {
                EventReportBackend::LifecycleMutationLease lease;
                const auto lease_result =
                    backend_and_token.first->AcquireMaintenanceCleanupLease(backend_and_token.second, lease);
                if (lease_result == EventReportBackend::CleanupLeaseAcquireResult::kBusy) {
                    key_busy = true;
                    break;
                }
                if (lease_result == EventReportBackend::CleanupLeaseAcquireResult::kStale) {
                    key_stale = true;
                    break;
                }
                leases.push_back(std::move(lease));
            }
        }

        if (key_stale || key_busy) {
            const char *status = key_stale ? "mismatch" : "error";
            for (const auto &target : targets) {
                record_delete_result(target.cleanup_token.reason, status);
            }
            if (key_stale) {
                completed_targets += targets.size();
            } else {
                hard_error_targets += targets.size();
            }
            continue;
        }

        LocationIdsPerKey location_ids(1);
        std::vector<std::vector<std::string>> expected_values(1);
        location_ids.front().reserve(targets.size());
        expected_values.front().reserve(targets.size());
        for (const auto &target : targets) {
            location_ids.front().push_back(target.location_id);
            expected_values.front().push_back(target.expected_location_value);
        }
        std::vector<std::vector<ErrorCode>> per_location_ec;
        RequestContext context("event_report_metadata_delete");
        const ErrorCode ec = meta_searcher.BatchDeleteLocations(&context,
                                                                 {task.block_keys[key_index]},
                                                                 location_ids,
                                                                 per_location_ec,
                                                                 expected_values,
                                                                 true,
                                                                 true,
                                                                 true);
        if (per_location_ec.size() != 1 || per_location_ec.front().size() != targets.size()) {
            hard_error_targets += targets.size();
            for (const auto &target : targets) {
                record_delete_result(target.cleanup_token.reason, "error");
            }
            continue;
        }
        for (size_t target_index = 0; target_index < targets.size(); ++target_index) {
            const ErrorCode target_ec = per_location_ec.front()[target_index];
            const bool hard_error = target_ec != EC_OK && target_ec != EC_NOENT && target_ec != EC_MISMATCH;
            hard_error_targets += static_cast<size_t>(hard_error);
            completed_targets += static_cast<size_t>(!hard_error);
            const char *status = target_ec == EC_OK       ? "deleted"
                                 : target_ec == EC_NOENT  ? "noent"
                                 : target_ec == EC_MISMATCH ? "mismatch"
                                                           : "error";
            record_delete_result(targets[target_index].cleanup_token.reason, status);
        }
        if (ec != EC_OK && ec != EC_PARTIAL_OK &&
            std::none_of(per_location_ec.front().begin(), per_location_ec.front().end(), [](ErrorCode target_ec) {
                return target_ec != EC_OK && target_ec != EC_NOENT && target_ec != EC_MISMATCH;
            })) {
            // A malformed aggregate result must not be hidden by otherwise
            // successful per-target values.
            ++hard_error_targets;
        }
    }

    if (hard_error_targets == 0) {
        return PlanExecuteResult{EC_OK, ""};
    }
    return MakeErrorResult(completed_targets == 0 ? EC_ERROR : EC_PARTIAL_OK,
                           StringUtil::FormatString("EventReport metadata delete hard failures: %zu",
                                                    hard_error_targets));
}

AsyncDeleteSubmitResult SchedulePlanExecutor::SubmitDeleteTaskAsync(std::chrono::microseconds delay,
                                                                    std::function<LocationDelAdmissionResult()> prepare,
                                                                    ScheduleTaskClass task_class) {
    auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    auto future = promise->get_future();
    auto completion = std::make_shared<PromiseCompletion>(promise);
    auto admission_task = [this, completion, delay, prepare = std::move(prepare), task_class]() {
        RunDeleteAdmission(completion, delay, prepare, task_class);
    };
    auto cancel_task = [completion]() {
        completion->Complete(ErrorCode::EC_ERROR, "SchedulePlanExecutor stopped before delete admission.");
    };
    if (!SubmitRaw(admission_task, std::chrono::microseconds::zero(), cancel_task, task_class)) {
        return AsyncDeleteSubmitResult{};
    }
    return AsyncDeleteSubmitResult{true, std::move(future)};
}

bool SchedulePlanExecutor::SubmitNonBlocking(const CacheMetaDelRequest &request, ScheduleTaskClass task_class) {
    auto req = BindStorageBackends(request, *data_storage_manager_);
    return SubmitRaw(
        [this, req, task_class]() {
            SubmitMetaDelete(req, task_class, meta_manager_->GetMetaIndexer(req.instance_id));
        },
        std::chrono::microseconds{0},
        {},
        task_class);
}

bool SchedulePlanExecutor::SubmitNonBlocking(const CacheLocationDelRequest &request, ScheduleTaskClass task_class) {
    auto req = BindStorageBackends(request, *data_storage_manager_);
    return SubmitRaw([this, req, task_class]() { SubmitLocationDelete(req, task_class); },
                     std::chrono::microseconds{0},
                     {},
                     task_class);
}

bool SchedulePlanExecutor::SubmitTask(std::function<void()> task, std::chrono::microseconds delay) {
    return SubmitTask(ScheduleTaskClass::kSystem, std::move(task), delay);
}

bool SchedulePlanExecutor::SubmitTask(ScheduleTaskClass task_class,
                                      std::function<void()> task,
                                      std::chrono::microseconds delay,
                                      std::function<void()> cancel_task) {
    if (!task) {
        return false;
    }
    return SubmitRaw(std::move(task), delay, std::move(cancel_task), task_class);
}

void SchedulePlanExecutor::DoCopyTask(const std::shared_ptr<std::promise<PlanExecuteResult>> &promise,
                                      const CacheLocationCopyRequest &task) {
    PlanExecuteResult result;
    result.status = ErrorCode::EC_OK;

    if (task.src_uris.size() != task.dst_uris.size()) {
        HandleErrorPromise(promise,
                           ErrorCode::EC_BADARGS,
                           "src_uris size %zu != dst_uris size %zu",
                           task.src_uris.size(),
                           task.dst_uris.size());
        return;
    }
    if (task.src_uris.empty()) {
        promise->set_value(result);
        return;
    }

    auto request_context = std::make_shared<RequestContext>("location_copy_task_trace");
    std::vector<ErrorCode> copy_results = data_storage_manager_->Copy(
        request_context.get(), task.exec_storage_name, task.src_uris, task.dst_uris, task.storage_backends);

    // 后端必须为每个输入 URI 返回一个结果（接口 postcondition）。短返回意味着部分 spec 的
    // 复制状态不可知——整体判为失败,防止 MigrationManager promote 不完整的目标 location。
    if (copy_results.size() != task.src_uris.size()) {
        HandleErrorPromise(promise,
                           ErrorCode::EC_ERROR,
                           "Copy returned %zu results, expected %zu (storage: %s, block_key: %ld)",
                           copy_results.size(),
                           task.src_uris.size(),
                           task.exec_storage_name.c_str(),
                           task.block_key);
        return;
    }

    for (size_t i = 0; i < copy_results.size(); ++i) {
        if (copy_results[i] != ErrorCode::EC_OK) {
            result.status = ErrorCode::EC_PARTIAL_OK;
            KVCM_LOG_WARN("Failed to copy kvcache via storage %s, block_key: %ld, src: %s, dst: %s, ec: %d",
                          task.exec_storage_name.c_str(),
                          task.block_key,
                          i < task.src_uris.size() ? task.src_uris[i].ToUriString().c_str() : "",
                          i < task.dst_uris.size() ? task.dst_uris[i].ToUriString().c_str() : "",
                          copy_results[i]);
        }
    }
    KVCM_LOG_DEBUG("DoCopyTask completed for instance_id: %s, block_key: %ld, status: %d",
                   task.instance_id.c_str(),
                   task.block_key,
                   result.status);
    promise->set_value(result);
}

std::future<PlanExecuteResult> SchedulePlanExecutor::Submit(const CacheLocationCopyRequest &task) {
    KVCM_LOG_DEBUG("Submitting copy task for instance_id: %s, block_key: %ld, uris: %zu",
                   task.instance_id.c_str(),
                   task.block_key,
                   task.src_uris.size());

    auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    std::future<PlanExecuteResult> future = promise->get_future();

    if (stop_) {
        HandleErrorPromise(promise, ErrorCode::EC_ERROR, "SchedulePlanExecutor stopped.");
        return future;
    }

    // copy 任务是 URI 级（URI 已由 MigrationManager 解析/预分配），无需 meta 解析，直接异步执行。
    auto execute_task = [this, promise, task]() { DoCopyTask(promise, task); };
    auto cancel_task = [promise]() {
        HandleErrorPromise(promise, ErrorCode::EC_ERROR, "SchedulePlanExecutor stopped before copy task execution.");
    };
    bool submit_result =
        this->SubmitRaw(execute_task, task.delay, cancel_task, ScheduleTaskClass::kMigrationContinuation);
    if (!submit_result) {
        HandleErrorPromise(promise, ErrorCode::EC_ERROR, "submit copy task failed");
        return future;
    }
    return future;
}

AsyncCopyExecuteSubmitResult SchedulePlanExecutor::SubmitAsyncCopy(const CacheLocationCopyRequest &task,
                                                                   const std::string &operation_id,
                                                                   const AsyncCopyOptions &options) {
    AsyncCopyExecuteSubmitResult result;
    auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    result.future = promise->get_future();
    auto completion = std::make_shared<PromiseCompletion>(promise);
    if (stop_) {
        result.submit_result.status = EC_ERROR;
        result.submit_result.operation_id = operation_id;
        result.submit_result.detail = "SchedulePlanExecutor stopped";
        completion->Complete(EC_ERROR, result.submit_result.detail);
        return result;
    }
    if (task.src_uris.empty() || task.src_uris.size() != task.dst_uris.size() || operation_id.empty()) {
        result.submit_result.status = EC_BADARGS;
        result.submit_result.operation_id = operation_id;
        result.submit_result.detail = "invalid asynchronous copy request";
        completion->Complete(EC_BADARGS, result.submit_result.detail);
        return result;
    }

    auto request_context = std::make_shared<RequestContext>("location_async_copy_submit");
    result.submit_result = data_storage_manager_->CopyAsync(
        request_context.get(),
        task.exec_storage_name,
        task.src_uris,
        task.dst_uris,
        operation_id,
        options,
        [completion, expected_items = task.src_uris.size()](AsyncCopyBatchResult batch_result) mutable {
            PlanExecuteResult plan_result;
            plan_result.status = batch_result.status;
            plan_result.error_message = std::move(batch_result.detail);
            plan_result.terminal = batch_result.items.size() == expected_items &&
                                   std::all_of(batch_result.items.begin(), batch_result.items.end(), [](const auto &x) {
                                       return x.terminal;
                                   });
            plan_result.safe_to_reuse_dst = batch_result.items.size() == expected_items &&
                                            std::all_of(batch_result.items.begin(),
                                                        batch_result.items.end(),
                                                        [](const auto &x) { return x.safe_to_reuse_dst; });
            if (batch_result.items.size() != expected_items) {
                plan_result.status = EC_MISMATCH;
                plan_result.terminal = false;
                plan_result.safe_to_reuse_dst = false;
                plan_result.error_message = StringUtil::FormatString(
                    "async Copy returned %zu items, expected %zu", batch_result.items.size(), expected_items);
            } else if (batch_result.status != EC_OK) {
                // Item-level success cannot override an aggregate backend
                // failure, including an unknown overall completion result.
                plan_result.status = batch_result.status;
            } else if (batch_result.AllSucceeded()) {
                plan_result.status = EC_OK;
            } else if (plan_result.terminal && plan_result.safe_to_reuse_dst) {
                plan_result.status = EC_PARTIAL_OK;
            } else {
                plan_result.status = EC_ERROR;
            }
            completion->Complete(std::move(plan_result));
        },
        task.storage_backends);

    if (!result.submit_result.accepted) {
        PlanExecuteResult rejected;
        rejected.status = result.submit_result.status;
        rejected.error_message = result.submit_result.detail;
        rejected.terminal = !result.submit_result.acceptance_unknown;
        rejected.safe_to_reuse_dst = !result.submit_result.acceptance_unknown;
        completion->Complete(std::move(rejected));
    }
    return result;
}

ErrorCode SchedulePlanExecutor::RequestCancelAsyncCopy(const std::string &storage_name,
                                                       const std::string &operation_id,
                                                       const StorageBackendSnapshot &snapshot) {
    if (stop_ || !data_storage_manager_) {
        return EC_ERROR;
    }
    return data_storage_manager_->RequestCancelAsyncCopy(storage_name, operation_id, snapshot);
}

} // namespace kv_cache_manager
