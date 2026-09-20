#include "kv_cache_manager/manager/schedule_plan_executor.h"

#include <algorithm>
#include <cassert>
#include <exception>
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
#include "kv_cache_manager/metrics/delete_cleanup_observer.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

#define DEFINE_METRICS_NAME_FOR_SCHEDULE_PLAN_EXECUTOR(name)                                                           \
    DEFINE_METRICS_NAME_(SchedulePlanExecutor, schedule_plan_executor, name)

#define REGISTER_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(name)                                                              \
    REGISTER_METRICS_GAUGE_(metrics_registry_, schedule_plan_executor, name)

namespace {
PlanExecuteResult MakeErrorResult(ErrorCode error_code, std::string error_message) {
    return PlanExecuteResult{
        .status = error_code,
        .error_message = std::move(error_message),
    };
}

std::uint64_t CountLocationTargets(const CacheLocationDelRequest &request) {
    std::uint64_t count = 0;
    for (const auto &ids : request.location_ids) {
        count += static_cast<std::uint64_t>(ids.size());
    }
    return count;
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
    InitializeDeleteCleanupMetrics(metrics_registry_);

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

PlanExecuteResult SchedulePlanExecutor::DoLocationDelTask(const CacheLocationDelRequest &task) {
    PlanExecuteResult result;
    result.status = ErrorCode::EC_OK;

    if (task.block_keys.size() != task.location_ids.size() ||
        (!task.expected_location_values.empty() && task.block_keys.size() != task.expected_location_values.size())) {
        return MakeErrorResult(ErrorCode::EC_BADARGS,
                               StringUtil::FormatString("delete request shape mismatch: block_keys[%zu], "
                                                        "location_ids[%zu], expected_values[%zu]",
                                                        task.block_keys.size(),
                                                        task.location_ids.size(),
                                                        task.expected_location_values.size()));
    }
    for (size_t i = 0; i < task.location_ids.size(); ++i) {
        if (!task.expected_location_values.empty() &&
            task.location_ids[i].size() != task.expected_location_values[i].size()) {
            return MakeErrorResult(ErrorCode::EC_BADARGS,
                                   StringUtil::FormatString("location_ids[%zu] size %zu != expected_values size %zu",
                                                            i,
                                                            task.location_ids[i].size(),
                                                            task.expected_location_values[i].size()));
        }
    }

    std::shared_ptr<MetaIndexer> indexer = meta_manager_->GetMetaIndexer(task.instance_id);
    if (!indexer) {
        return MakeErrorResult(ErrorCode::EC_NOENT,
                               StringUtil::FormatString("MetaIndexer %s not found", task.instance_id.c_str()));
    }

    MetaSearcher meta_searcher(indexer);
    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    auto ctx = std::make_shared<RequestContext>("schedule_plan_executor_call");
    ErrorCode get_locations_ec = EC_OK;
    if (task.pre_fenced_deleting) {
        const auto get_result = indexer->GetLocationsFromPersistent(ctx.get(), task.block_keys, location_maps);
        if (get_result.error_codes.size() != task.block_keys.size() ||
            std::any_of(get_result.error_codes.begin(), get_result.error_codes.end(), [](ErrorCode ec) {
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

    struct LocationDeleteCandidate {
        size_t request_index{0};
        std::string location_id;
        std::string expected_location_value;
        bool storage_delete_ok{true};
    };
    struct StorageDeleteBatch {
        std::vector<DataStorageUri> uris;
        std::vector<size_t> candidate_indices;
    };
    std::vector<LocationDeleteCandidate> candidates;
    std::map<std::string, StorageDeleteBatch> delete_batches_by_unique_name;
    for (size_t i = 0; i < task.block_keys.size(); i++) {
        auto &location_map = location_maps[i];
        auto &need_delete_location_ids = task.location_ids[i];
        for (size_t location_index = 0; location_index < need_delete_location_ids.size(); ++location_index) {
            const auto &location_id = need_delete_location_ids[location_index];
            auto iter = location_map.find(location_id);
            if (iter == location_map.end() || !iter->second) {
                continue;
            }
            if (iter->second->status() != CacheLocationStatus::CLS_DELETING) {
                continue;
            }
            if (!task.expected_location_values.empty() &&
                iter->second->ToJsonString() != task.expected_location_values[i][location_index]) {
                continue;
            }
            const size_t candidate_index = candidates.size();
            candidates.push_back({.request_index = i,
                                  .location_id = iter->first,
                                  .expected_location_value = iter->second->ToJsonString()});
            if (!task.metadata_only) {
                for (const auto &loc_spec : iter->second->location_specs()) {
                    DataStorageUri uri(loc_spec.uri());
                    if (!uri.Valid()) {
                        candidates[candidate_index].storage_delete_ok = false;
                        result.status = ErrorCode::EC_PARTIAL_OK;
                        const std::string failure_message = StringUtil::FormatString(
                            "invalid storage URI prevents location cleanup, instance[%s] key[%ld] location[%s]",
                            task.instance_id.c_str(),
                            task.block_keys[i],
                            iter->first.c_str());
                        if (result.error_message.empty()) {
                            result.error_message = failure_message;
                        }
                        KVCM_LOG_WARN("%s", failure_message.c_str());
                        continue;
                    }
                    if (!task.confirmed_missing_uris.empty() &&
                        task.confirmed_missing_uris.find(uri.ToUriString()) != task.confirmed_missing_uris.end()) {
                        continue;
                    }
                    auto &batch = delete_batches_by_unique_name[uri.GetHostName()];
                    batch.uris.emplace_back(std::move(uri));
                    batch.candidate_indices.push_back(candidate_index);
                }
            }
        }
    }

    KVCM_LOG_DEBUG("Found %zu meta location(s) to delete", candidates.size());

    // delete storage uris
    auto request_context = std::make_shared<RequestContext>("location_del_task_trace");
    for (const auto &[storage_unique_name, batch] : delete_batches_by_unique_name) {
        const auto &storage_uris = batch.uris;
        KVCM_LOG_DEBUG("Deleting %zu entries from storage: %s", storage_uris.size(), storage_unique_name.c_str());
        std::vector<ErrorCode> delete_results =
            data_storage_manager_->Delete(request_context.get(), storage_unique_name, storage_uris, nullptr);
        if (delete_results.size() != storage_uris.size()) {
            result.status = ErrorCode::EC_PARTIAL_OK;
            result.error_message =
                StringUtil::FormatString("storage delete result size %zu != request size %zu, instance[%s] storage[%s]",
                                         delete_results.size(),
                                         storage_uris.size(),
                                         task.instance_id.c_str(),
                                         storage_unique_name.c_str());
            KVCM_LOG_WARN("%s", result.error_message.c_str());
            for (const size_t candidate_index : batch.candidate_indices) {
                candidates[candidate_index].storage_delete_ok = false;
            }
        }
        const auto result_count = std::min(delete_results.size(), storage_uris.size());
        size_t failed_count = storage_uris.size() - result_count;
        size_t first_failed_index = result_count < storage_uris.size() ? result_count : storage_uris.size();
        for (size_t i = 0; i < result_count; ++i) {
            if (delete_results[i] != ErrorCode::EC_OK && delete_results[i] != ErrorCode::EC_NOENT) {
                if (first_failed_index == storage_uris.size()) {
                    first_failed_index = i;
                }
                ++failed_count;
                const auto &candidate = candidates[batch.candidate_indices[i]];
                KVCM_LOG_WARN("storage delete failed, instance[%s] storage[%s] key[%ld] location[%s] item[%zu] ec[%d]",
                              task.instance_id.c_str(),
                              storage_unique_name.c_str(),
                              task.block_keys[candidate.request_index],
                              candidate.location_id.c_str(),
                              i,
                              static_cast<int>(delete_results[i]));
                candidates[batch.candidate_indices[i]].storage_delete_ok = false;
            }
        }
        for (size_t i = result_count; i < storage_uris.size(); ++i) {
            candidates[batch.candidate_indices[i]].storage_delete_ok = false;
        }
        if (failed_count > 0) {
            result.status = ErrorCode::EC_PARTIAL_OK;
            const bool has_returned_failure = first_failed_index < result_count;
            const auto &first_failed_candidate = candidates[batch.candidate_indices[first_failed_index]];
            const std::string failure_message = StringUtil::FormatString(
                "storage delete failed, instance[%s] storage[%s] failed[%zu] total[%zu] first_key[%ld] "
                "first_location[%s] first_item[%zu] first_ec[%d]",
                task.instance_id.c_str(),
                storage_unique_name.c_str(),
                failed_count,
                storage_uris.size(),
                task.block_keys[first_failed_candidate.request_index],
                first_failed_candidate.location_id.c_str(),
                first_failed_index,
                has_returned_failure ? static_cast<int>(delete_results[first_failed_index])
                                     : static_cast<int>(ErrorCode::EC_ERROR));
            if (result.error_message.empty()) {
                result.error_message = failure_message;
            }
        }
    }

    const auto physical_delete_failure_count = static_cast<std::uint64_t>(std::count_if(
        candidates.begin(), candidates.end(), [](const auto &candidate) { return !candidate.storage_delete_ok; }));
    RecordDeleteCleanupPermanentFailure(
        metrics_registry_, DeleteCleanupFailureStage::kPhysicalDelete, physical_delete_failure_count);

    // Retain a DELETING metadata reference whenever any URI for the Location
    // failed or returned an ambiguous result. Dropping metadata here would
    // hide the leak; automatic retries are unsafe until the backend supports
    // a lease or conditional delete that prevents deleting a reused URI.
    std::vector<std::vector<MetaSearcher::LocationCADTask>> cad_tasks_by_request(task.block_keys.size());
    for (const auto &candidate : candidates) {
        if (candidate.storage_delete_ok) {
            cad_tasks_by_request[candidate.request_index].push_back(
                {candidate.location_id, CacheLocationStatus::CLS_DELETING, candidate.expected_location_value});
        }
    }
    std::vector<int64_t> block_keys_to_delete;
    std::vector<std::vector<MetaSearcher::LocationCADTask>> batch_cad_tasks;
    for (size_t i = 0; i < cad_tasks_by_request.size(); ++i) {
        if (!cad_tasks_by_request[i].empty()) {
            block_keys_to_delete.push_back(task.block_keys[i]);
            batch_cad_tasks.push_back(std::move(cad_tasks_by_request[i]));
        }
    }

    // delete locations
    if (!batch_cad_tasks.empty()) {
        std::vector<std::vector<ErrorCode>> delete_meta_results;
        ErrorCode delete_meta_ec = meta_searcher.BatchCADLocationStatus(
            ctx.get(), block_keys_to_delete, batch_cad_tasks, delete_meta_results, task.pre_fenced_deleting);
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
        std::uint64_t cad_failure_count = 0;
        if (delete_meta_ec != ErrorCode::EC_OK) {
            for (const auto &tasks : batch_cad_tasks) {
                cad_failure_count += static_cast<std::uint64_t>(tasks.size());
            }
        } else {
            for (size_t block_key_idx = 0; block_key_idx < batch_cad_tasks.size(); ++block_key_idx) {
                if (block_key_idx >= delete_meta_results.size()) {
                    cad_failure_count += static_cast<std::uint64_t>(batch_cad_tasks[block_key_idx].size());
                    continue;
                }
                const auto &results = delete_meta_results[block_key_idx];
                for (size_t location_idx = 0; location_idx < batch_cad_tasks[block_key_idx].size(); ++location_idx) {
                    if (location_idx >= results.size() ||
                        (results[location_idx] != ErrorCode::EC_OK && results[location_idx] != ErrorCode::EC_NOENT)) {
                        ++cad_failure_count;
                    }
                }
            }
        }
        RecordDeleteCleanupPermanentFailure(
            metrics_registry_, DeleteCleanupFailureStage::kMetadataCad, cad_failure_count);
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
                if (results[location_idx] != ErrorCode::EC_OK && results[location_idx] != ErrorCode::EC_NOENT) {
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
    return SubmitMetaDelete(task, ScheduleTaskClass::kSystem);
}

std::future<PlanExecuteResult> SchedulePlanExecutor::SubmitMetaDelete(const CacheMetaDelRequest &task,
                                                                      ScheduleTaskClass task_class) {
    KVCM_LOG_DEBUG("Submitting meta delete task for instance_id: %s, block_keys count: %zu",
                   task.instance_id.c_str(),
                   task.block_keys.size());

    auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    auto future = promise->get_future();
    auto completion = std::make_shared<PromiseCompletion>(promise);
    RunDeleteAdmission(
        completion, task.delay, [this, task]() { return PrepareDeleteTask(task); }, task_class, 0);
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
        for (size_t location_idx = 0; location_idx < results.size(); location_idx++) {
            if (results[location_idx] != EC_OK) {
                KVCM_LOG_INFO("Location status CAS failed, block key: %ld, location_id: %s",
                              batch_cas_block_keys[key_idx],
                              batch_cas_tasks[key_idx][location_idx].location_id.c_str());
                continue;
            }
            location_ids.push_back(batch_cas_tasks[key_idx][location_idx].location_id);
        }
        if (location_ids.empty()) {
            continue;
        }
        actual_task.block_keys.push_back(batch_cas_block_keys[key_idx]);
        actual_task.location_ids.emplace_back(std::move(location_ids));
    }
    return true;
}
SchedulePlanExecutor::LocationDelAdmissionResult
SchedulePlanExecutor::PrepareDeleteTask(const CacheMetaDelRequest &task) {
    std::vector<int64_t> unique_block_keys;
    unique_block_keys.reserve(task.block_keys.size());
    std::unordered_set<int64_t> seen_block_keys;
    seen_block_keys.reserve(task.block_keys.size());
    for (const int64_t block_key : task.block_keys) {
        if (seen_block_keys.insert(block_key).second) {
            unique_block_keys.push_back(block_key);
        }
    }
    return PrepareDeleteTaskImpl(task.instance_id, unique_block_keys, nullptr, nullptr, task.delay, false, false);
}

SchedulePlanExecutor::LocationDelAdmissionResult
SchedulePlanExecutor::PrepareDeleteTask(const CacheLocationDelRequest &task) {
    if (task.pre_fenced_deleting &&
        (!task.authoritative_read || task.block_keys.empty() || task.expected_location_values.empty() ||
         std::any_of(
             task.location_ids.begin(), task.location_ids.end(), [](const auto &ids) { return ids.empty(); }))) {
        LocationDelAdmissionResult admission_result;
        admission_result.result =
            MakeErrorResult(ErrorCode::EC_BADARGS,
                            "pre_fenced_deleting requires authoritative_read, explicit location_ids and exact values");
        return admission_result;
    }
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
    // Canonicalize caller-controlled targets before either status mutation or
    // physical I/O. A repeated key/id must never be deleted or accounted
    // twice; conflicting exact snapshots are an invalid request rather than a
    // reason to guess which generation the caller intended.
    std::vector<int64_t> normalized_block_keys;
    std::vector<std::vector<std::string>> normalized_location_ids;
    std::vector<std::vector<std::string>> normalized_expected_values;
    std::unordered_map<int64_t, size_t> key_to_index;
    std::vector<std::unordered_map<std::string, size_t>> id_to_index;
    normalized_block_keys.reserve(task.block_keys.size());
    normalized_location_ids.reserve(task.block_keys.size());
    id_to_index.reserve(task.block_keys.size());
    const bool has_expected_values = !task.expected_location_values.empty();
    if (has_expected_values) {
        normalized_expected_values.reserve(task.block_keys.size());
    }
    for (size_t key_index = 0; key_index < task.block_keys.size(); ++key_index) {
        const int64_t block_key = task.block_keys[key_index];
        auto [key_iter, inserted] = key_to_index.emplace(block_key, normalized_block_keys.size());
        if (inserted) {
            normalized_block_keys.push_back(block_key);
            normalized_location_ids.emplace_back();
            id_to_index.emplace_back();
            if (has_expected_values) {
                normalized_expected_values.emplace_back();
            }
        }
        const size_t normalized_key_index = key_iter->second;
        for (size_t location_index = 0; location_index < task.location_ids[key_index].size(); ++location_index) {
            const auto &location_id = task.location_ids[key_index][location_index];
            if (location_id.empty()) {
                LocationDelAdmissionResult admission_result;
                admission_result.result = MakeErrorResult(ErrorCode::EC_BADARGS, "location id must not be empty");
                return admission_result;
            }
            auto [id_iter, id_inserted] = id_to_index[normalized_key_index].emplace(
                location_id, normalized_location_ids[normalized_key_index].size());
            if (id_inserted) {
                normalized_location_ids[normalized_key_index].push_back(location_id);
                if (has_expected_values) {
                    normalized_expected_values[normalized_key_index].push_back(
                        task.expected_location_values[key_index][location_index]);
                }
                continue;
            }
            if (has_expected_values && normalized_expected_values[normalized_key_index][id_iter->second] !=
                                           task.expected_location_values[key_index][location_index]) {
                LocationDelAdmissionResult admission_result;
                admission_result.result =
                    MakeErrorResult(ErrorCode::EC_BADARGS, "duplicate location target has conflicting expected values");
                return admission_result;
            }
        }
    }
    const auto *expected_location_values = has_expected_values ? &normalized_expected_values : nullptr;
    auto result = PrepareDeleteTaskImpl(task.instance_id,
                                        normalized_block_keys,
                                        &normalized_location_ids,
                                        expected_location_values,
                                        task.delay,
                                        task.authoritative_read,
                                        task.pre_fenced_deleting);
    result.actual_task.metadata_only = task.metadata_only;
    result.actual_task.authoritative_read = task.authoritative_read;
    result.actual_task.pre_fenced_deleting = task.pre_fenced_deleting;
    result.actual_task.confirmed_missing_uris = task.confirmed_missing_uris;
    return result;
}

SchedulePlanExecutor::LocationDelAdmissionResult
SchedulePlanExecutor::PrepareDeleteTaskImpl(const std::string &instance_id,
                                            const std::vector<int64_t> &block_keys,
                                            const std::vector<std::vector<std::string>> *target_location_ids,
                                            const std::vector<std::vector<std::string>> *expected_location_values,
                                            std::chrono::microseconds delay,
                                            bool authoritative_read,
                                            bool pre_fenced_deleting) {
    LocationDelAdmissionResult admission_result;
    admission_result.actual_task = CacheLocationDelRequest{instance_id, {}, {}, delay};

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
        const auto get_result = indexer->GetLocationsFromPersistent(request_context.get(), block_keys, location_maps);
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
    // A null target list represents CacheMetaDelRequest and selects every non-deleting location.
    for (size_t block_key_idx = 0; block_key_idx < block_keys.size(); ++block_key_idx) {
        std::unordered_map<std::string, std::string> expected_values_by_location;
        if (expected_location_values != nullptr) {
            for (size_t i = 0; i < (*target_location_ids)[block_key_idx].size(); ++i) {
                expected_values_by_location.emplace((*target_location_ids)[block_key_idx][i],
                                                    (*expected_location_values)[block_key_idx][i]);
            }
        }
        auto block_key = block_keys[block_key_idx];
        auto &location_map = location_maps[block_key_idx];
        std::vector<MetaSearcher::LocationCASTask> location_cas_tasks;
        std::vector<std::string> selected_location_ids;
        if (target_location_ids != nullptr) {
            selected_location_ids = (*target_location_ids)[block_key_idx];
        } else {
            selected_location_ids.reserve(location_map.size());
            for (const auto &[location_id, location] : location_map) {
                (void)location;
                selected_location_ids.push_back(location_id);
            }
        }
        for (const auto &selected_location_id : selected_location_ids) {
            const auto location_iter = location_map.find(selected_location_id);
            if (location_iter == location_map.end() || !location_iter->second) {
                continue;
            }
            const auto &location = *location_iter->second;
            std::string expected_location_value;
            if (expected_location_values != nullptr) {
                const auto expected = expected_values_by_location.find(location.id());
                if (expected == expected_values_by_location.end() || location.ToJsonString() != expected->second) {
                    continue; // stable location was refreshed after the cleanup scan
                }
                expected_location_value = expected->second;
            }
            if (pre_fenced_deleting) {
                // The authoritative FinishWrite cleanup has already persisted
                // both the final URI and CLS_DELETING. Only adopt that exact,
                // still-hidden generation for physical cleanup. Never turn a
                // different status into DELETING on this privileged path.
                if (location.status() != CacheLocationStatus::CLS_DELETING) {
                    continue;
                }
                if (admission_result.actual_task.block_keys.empty() ||
                    admission_result.actual_task.block_keys.back() != block_key) {
                    admission_result.actual_task.block_keys.push_back(block_key);
                    admission_result.actual_task.location_ids.emplace_back();
                    admission_result.actual_task.expected_location_values.emplace_back();
                }
                admission_result.actual_task.location_ids.back().push_back(location.id());
                admission_result.actual_task.expected_location_values.back().push_back(location.ToJsonString());
                continue;
            }
            if (location.status() == CacheLocationStatus::CLS_DELETING) {
                continue;
            }
            location_cas_tasks.push_back({location.id(),
                                          location.status(),
                                          CacheLocationStatus::CLS_DELETING,
                                          std::move(expected_location_value)});
        }
        if (location_cas_tasks.empty()) {
            continue;
        }
        batch_cas_block_keys.push_back(block_key);
        batch_cas_tasks.emplace_back(std::move(location_cas_tasks));
    }
    if (pre_fenced_deleting) {
        admission_result.needs_physical_delete = !admission_result.actual_task.block_keys.empty();
        return admission_result;
    }
    if (batch_cas_block_keys.empty()) {
        return admission_result;
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
    if (admission_result.actual_task.block_keys.empty()) {
        return admission_result;
    }

    if (!indexer->Sync(admission_result.actual_task.block_keys)) {
        admission_result.result =
            MakeErrorResult(ErrorCode::EC_ERROR,
                            StringUtil::FormatString("Sync failed or timed out for location delete, instance[%s]",
                                                     instance_id.c_str()));
        return admission_result;
    }

    admission_result.needs_physical_delete = true;
    return admission_result;
}

void SchedulePlanExecutor::RunDeleteAdmission(const std::shared_ptr<PromiseCompletion> &completion,
                                              std::chrono::microseconds delay,
                                              const std::function<LocationDelAdmissionResult()> &prepare,
                                              ScheduleTaskClass task_class,
                                              std::uint64_t already_fenced_target_count) {
    const auto record_dispatch_failure = [this](std::uint64_t count) {
        RecordDeleteCleanupPermanentFailure(metrics_registry_, DeleteCleanupFailureStage::kDispatchOrWorker, count);
    };
    try {
        auto admission_result = prepare();
        const std::uint64_t admitted_target_count = CountLocationTargets(admission_result.actual_task);
        if (admission_result.result.status != ErrorCode::EC_OK || !admission_result.needs_physical_delete) {
            const std::uint64_t stranded_target_count =
                admitted_target_count == 0 ? already_fenced_target_count : admitted_target_count;
            if (admission_result.result.status != ErrorCode::EC_OK && stranded_target_count != 0) {
                record_dispatch_failure(stranded_target_count);
                admission_result.result.error_logged = true;
            }
            completion->Complete(std::move(admission_result.result));
            return;
        }

        KVCM_LOG_DEBUG("Location statuses updated, submitting task to worker pool with delay: %lld microseconds",
                       static_cast<long long>(delay.count()));
        auto actual_task = std::move(admission_result.actual_task);
        auto execute_task =
            [this, completion, admitted_target_count, actual_task = std::move(actual_task), record_dispatch_failure]() {
                try {
                    auto result = DoLocationDelTask(actual_task);
                    if (result.status != ErrorCode::EC_OK && !result.error_logged) {
                        record_dispatch_failure(admitted_target_count);
                        result.error_logged = true;
                    }
                    completion->Complete(std::move(result));
                } catch (const std::exception &e) {
                    record_dispatch_failure(admitted_target_count);
                    auto result =
                        MakeErrorResult(ErrorCode::EC_ERROR,
                                        StringUtil::FormatString("location delete task threw exception: %s", e.what()));
                    result.error_logged = true;
                    completion->Complete(std::move(result));
                } catch (...) {
                    record_dispatch_failure(admitted_target_count);
                    auto result = MakeErrorResult(ErrorCode::EC_ERROR, "location delete task threw unknown exception");
                    result.error_logged = true;
                    completion->Complete(std::move(result));
                }
            };
        auto cancel_task = [completion, admitted_target_count, record_dispatch_failure]() {
            record_dispatch_failure(admitted_target_count);
            auto result = MakeErrorResult(ErrorCode::EC_ERROR, "SchedulePlanExecutor stopped before physical delete.");
            result.error_logged = true;
            completion->Complete(std::move(result));
        };
        if (!SubmitRaw(execute_task, delay, cancel_task, task_class)) {
            record_dispatch_failure(admitted_target_count);
            auto result = MakeErrorResult(ErrorCode::EC_ERROR, "submit physical delete task failed");
            result.error_logged = true;
            completion->Complete(std::move(result));
        }
    } catch (const std::exception &e) {
        record_dispatch_failure(already_fenced_target_count);
        auto result = MakeErrorResult(
            ErrorCode::EC_ERROR, StringUtil::FormatString("location delete admission threw exception: %s", e.what()));
        result.error_logged = already_fenced_target_count != 0;
        completion->Complete(std::move(result));
    } catch (...) {
        record_dispatch_failure(already_fenced_target_count);
        auto result = MakeErrorResult(ErrorCode::EC_ERROR, "location delete admission threw unknown exception");
        result.error_logged = already_fenced_target_count != 0;
        completion->Complete(std::move(result));
    }
}

std::future<PlanExecuteResult> SchedulePlanExecutor::Submit(const CacheLocationDelRequest &task) {
    return SubmitLocationDelete(task, ScheduleTaskClass::kSystem);
}

std::future<PlanExecuteResult> SchedulePlanExecutor::SubmitLocationDelete(const CacheLocationDelRequest &task,
                                                                          ScheduleTaskClass task_class) {
    KVCM_LOG_DEBUG("Submitting location delete task for instance_id: %s, block_keys count: %zu",
                   task.instance_id.c_str(),
                   task.block_keys.size());

    auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    auto future = promise->get_future();
    auto completion = std::make_shared<PromiseCompletion>(promise);
    RunDeleteAdmission(
        completion,
        task.delay,
        [this, &task]() { return PrepareDeleteTask(task); },
        task_class,
        task.pre_fenced_deleting ? CountLocationTargets(task) : 0);
    return future;
}

AsyncDeleteSubmitResult SchedulePlanExecutor::SubmitAsync(const CacheMetaDelRequest &task) {
    KVCM_LOG_DEBUG("Submitting async meta delete task for instance_id: %s, block_keys count: %zu",
                   task.instance_id.c_str(),
                   task.block_keys.size());
    return SubmitDeleteTaskAsync(
        task.delay, [this, task]() { return PrepareDeleteTask(task); }, 0);
}

AsyncDeleteSubmitResult SchedulePlanExecutor::SubmitAsync(const CacheLocationDelRequest &task) {
    KVCM_LOG_DEBUG("Submitting async location delete task for instance_id: %s, block_keys count: %zu",
                   task.instance_id.c_str(),
                   task.block_keys.size());

    return SubmitDeleteTaskAsync(
        task.delay,
        [this, task]() { return PrepareDeleteTask(task); },
        task.pre_fenced_deleting ? CountLocationTargets(task) : 0);
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
            completion->Complete(EC_ERROR,
                                 StringUtil::FormatString("EventReport metadata delete threw exception: %s", e.what()));
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

PlanExecuteResult SchedulePlanExecutor::DoEventReportMetadataDelTask(const EventReportMetadataDelRequest &task) {
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
        const ErrorCode ec = meta_searcher.BatchDeleteLocations(
            &context, {task.block_keys[key_index]}, location_ids, per_location_ec, expected_values, true, true, true);
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
            const char *status = target_ec == EC_OK         ? "deleted"
                                 : target_ec == EC_NOENT    ? "noent"
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
    return MakeErrorResult(
        completed_targets == 0 ? EC_ERROR : EC_PARTIAL_OK,
        StringUtil::FormatString("EventReport metadata delete hard failures: %zu", hard_error_targets));
}

AsyncDeleteSubmitResult SchedulePlanExecutor::SubmitDeleteTaskAsync(std::chrono::microseconds delay,
                                                                    std::function<LocationDelAdmissionResult()> prepare,
                                                                    std::uint64_t already_fenced_target_count) {
    auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    auto future = promise->get_future();
    auto completion = std::make_shared<PromiseCompletion>(promise);
    auto admission_task = [this, completion, delay, prepare = std::move(prepare), already_fenced_target_count]() {
        RunDeleteAdmission(completion, delay, prepare, ScheduleTaskClass::kReclaim, already_fenced_target_count);
    };
    auto cancel_task = [this, completion, already_fenced_target_count]() {
        RecordDeleteCleanupPermanentFailure(
            metrics_registry_, DeleteCleanupFailureStage::kDispatchOrWorker, already_fenced_target_count);
        auto result = MakeErrorResult(ErrorCode::EC_ERROR, "SchedulePlanExecutor stopped before delete admission.");
        result.error_logged = already_fenced_target_count != 0;
        completion->Complete(std::move(result));
    };
    if (!SubmitRaw(admission_task, std::chrono::microseconds::zero(), cancel_task, ScheduleTaskClass::kReclaim)) {
        RecordDeleteCleanupPermanentFailure(
            metrics_registry_, DeleteCleanupFailureStage::kDispatchOrWorker, already_fenced_target_count);
        return AsyncDeleteSubmitResult{};
    }
    return AsyncDeleteSubmitResult{true, std::move(future)};
}

bool SchedulePlanExecutor::SubmitNonBlocking(const CacheMetaDelRequest &req, ScheduleTaskClass task_class) {
    return SubmitRaw(
        [this, req, task_class]() { SubmitMetaDelete(req, task_class); }, std::chrono::microseconds{0}, {}, task_class);
}

bool SchedulePlanExecutor::SubmitNonBlocking(const CacheLocationDelRequest &req, ScheduleTaskClass task_class) {
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
    std::vector<ErrorCode> copy_results =
        data_storage_manager_->Copy(request_context.get(), task.exec_storage_name, task.src_uris, task.dst_uris);

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

} // namespace kv_cache_manager
