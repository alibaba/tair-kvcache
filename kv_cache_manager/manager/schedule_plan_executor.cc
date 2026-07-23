#include "kv_cache_manager/manager/schedule_plan_executor.h"

#include <algorithm>
#include <cassert>
#include <exception>
#include <memory>
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

    if (task.block_keys.size() != task.location_ids.size()) {
        return MakeErrorResult(ErrorCode::EC_BADARGS,
                               StringUtil::FormatString("block_keys size %zu != location_ids size %zu",
                                                        task.block_keys.size(),
                                                        task.location_ids.size()));
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
    ErrorCode get_locations_ec = meta_searcher.BatchGetLocation(ctx.get(), task.block_keys, empty_mask, location_maps);
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

        for (auto &location_id : need_delete_location_ids) {
            auto iter = location_map.find(location_id);
            if (iter == location_map.end() || !iter->second) {
                continue;
            }
            if (iter->second->status() != CacheLocationStatus::CLS_DELETING) {
                continue;
            }
            for (const auto &loc_spec : iter->second->location_specs()) {
                DataStorageUri uri(loc_spec.uri());
                if (uri.Valid()) {
                    std::string storage_unique_name = uri.GetHostName();
                    delete_uris_by_unique_name[storage_unique_name].emplace_back(uri);
                }
            }
            cad_tasks.push_back({iter->first, CacheLocationStatus::CLS_DELETING});
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
    for (const auto &storage_uris_pair : delete_uris_by_unique_name) {
        const std::string &storage_unique_name = storage_uris_pair.first;
        const std::vector<DataStorageUri> &storage_uris = storage_uris_pair.second;
        KVCM_LOG_DEBUG("Deleting %zu entries from storage: %s", storage_uris.size(), storage_unique_name.c_str());
        std::vector<ErrorCode> delete_results =
            data_storage_manager_->Delete(request_context.get(), storage_unique_name, storage_uris, nullptr);
        if (delete_results.size() != storage_uris.size()) {
            result.status = ErrorCode::EC_PARTIAL_OK;
            result.error_message = StringUtil::FormatString(
                "storage delete result size %zu != request size %zu", delete_results.size(), storage_uris.size());
            KVCM_LOG_WARN("%s", result.error_message.c_str());
        }
        const auto result_count = std::min(delete_results.size(), storage_uris.size());
        for (size_t i = 0; i < result_count; ++i) {
            if (delete_results[i] != ErrorCode::EC_OK) {
                // 这里存储删除报错暂且不管，报个warn表示哪个storageUri删失败了
                result.status = ErrorCode::EC_PARTIAL_OK;
                KVCM_LOG_WARN("storage delete failed, instance[%s] storage[%s] uri[%s] ec[%d]",
                              task.instance_id.c_str(),
                              storage_unique_name.c_str(),
                              storage_uris[i].ToUriString().c_str(),
                              static_cast<int>(delete_results[i]));
            }
        }
    }

    // delete locations
    if (!batch_cad_tasks.empty()) {
        std::vector<std::vector<ErrorCode>> delete_meta_results;
        ErrorCode delete_meta_ec =
            meta_searcher.BatchCADLocationStatus(ctx.get(), block_keys_to_delete, batch_cad_tasks, delete_meta_results);
        if (delete_meta_ec != ErrorCode::EC_OK || delete_meta_results.size() != batch_cad_tasks.size()) {
            result.status = ErrorCode::EC_PARTIAL_OK;
            result.error_message =
                StringUtil::FormatString("location CAD failed, ec: %d, result size %zu != task size %zu",
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
                KVCM_LOG_WARN("location CAD result size %zu != task size %zu for block key %ld",
                              results.size(),
                              batch_cad_tasks[block_key_idx].size(),
                              block_keys_to_delete[block_key_idx]);
            }
            const auto location_result_count = std::min(results.size(), batch_cad_tasks[block_key_idx].size());
            for (size_t location_idx = 0; location_idx < location_result_count; location_idx++) {
                if (results[location_idx] != ErrorCode::EC_OK) {
                    result.status = ErrorCode::EC_PARTIAL_OK;
                    KVCM_LOG_WARN("Failed to CAD meta key %ld, location: %s, error_code: %d",
                                  block_keys_to_delete[block_key_idx],
                                  batch_cad_tasks[block_key_idx][location_idx].location_id.c_str(),
                                  static_cast<int>(results[location_idx]));
                }
            }
        }
    }
    KVCM_LOG_DEBUG("DoDelLocationTask completed successfully for instance_id: %s", task.instance_id.c_str());

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
    std::future<PlanExecuteResult> future = promise->get_future();

    if (stop_) {
        HandleErrorPromise(promise, ErrorCode::EC_ERROR, "SchedulePlanExecutor stopped.");
        return future;
    }

    // 1. sync set block status to deleting
    const std::shared_ptr<MetaIndexer> indexer = meta_manager_->GetMetaIndexer(task.instance_id);
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
        for (const auto &loc_kv : location_maps[block_key_idx]) {
            if (!loc_kv.second) {
                continue;
            }
            cas_tasks.push_back({loc_kv.first, loc_kv.second->status(), CLS_DELETING});
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
    if (!FillActualTask(batch_cas_block_keys, batch_cas_tasks, cas_results, actual_task, error_message)) {
        HandleErrorPromise(promise, ErrorCode::EC_ERROR, "FillActualTask error: %s", error_message.c_str());
        return future;
    }
    if (actual_task.block_keys.empty()) {
        promise->set_value(PlanExecuteResult{ErrorCode::EC_OK, ""});
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

    auto execute_task = [this, promise, actual_task]() {
        try {
            promise->set_value(DoLocationDelTask(actual_task));
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
    return PrepareDeleteTaskImpl(task.instance_id, task.block_keys, nullptr, task.delay);
}

SchedulePlanExecutor::LocationDelAdmissionResult
SchedulePlanExecutor::PrepareDeleteTask(const CacheLocationDelRequest &task) {
    if (task.block_keys.size() != task.location_ids.size()) {
        LocationDelAdmissionResult admission_result;
        admission_result.result = MakeErrorResult(
            ErrorCode::EC_BADARGS,
            StringUtil::FormatString(
                "block_keys size %zu != location_ids size %zu", task.block_keys.size(), task.location_ids.size()));
        return admission_result;
    }
    return PrepareDeleteTaskImpl(task.instance_id, task.block_keys, &task.location_ids, task.delay);
}

SchedulePlanExecutor::LocationDelAdmissionResult
SchedulePlanExecutor::PrepareDeleteTaskImpl(const std::string &instance_id,
                                            const std::vector<int64_t> &block_keys,
                                            const std::vector<std::vector<std::string>> *target_location_ids,
                                            std::chrono::microseconds delay) {
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
    ErrorCode get_locations_ec =
        meta_searcher.BatchGetLocation(request_context.get(), block_keys, empty_mask, location_maps);
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
        std::unordered_set<std::string> target_ids;
        if (target_location_ids != nullptr) {
            target_ids.insert((*target_location_ids)[block_key_idx].begin(),
                              (*target_location_ids)[block_key_idx].end());
        }
        auto block_key = block_keys[block_key_idx];
        auto &location_map = location_maps[block_key_idx];
        std::vector<MetaSearcher::LocationCASTask> location_cas_tasks;
        for (const auto &loc_kv : location_map) {
            if (!loc_kv.second) {
                continue;
            }
            const auto &location = *loc_kv.second;
            if (location.status() == CacheLocationStatus::CLS_DELETING) {
                continue;
            }
            if (target_location_ids != nullptr && target_ids.find(location.id()) == target_ids.end()) {
                continue;
            }
            location_cas_tasks.push_back({location.id(), location.status(), CacheLocationStatus::CLS_DELETING});
        }
        if (location_cas_tasks.empty()) {
            continue;
        }
        batch_cas_block_keys.push_back(block_key);
        batch_cas_tasks.emplace_back(std::move(location_cas_tasks));
    }
    if (batch_cas_block_keys.empty()) {
        return admission_result;
    }

    std::vector<std::vector<ErrorCode>> batch_results;
    ErrorCode update_ec = meta_searcher.BatchCASLocationStatus(
        request_context.get(), batch_cas_block_keys, batch_cas_tasks, batch_results);
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
                                              ScheduleTaskClass task_class) {
    try {
        auto admission_result = prepare();
        if (admission_result.result.status != ErrorCode::EC_OK || !admission_result.needs_physical_delete) {
            completion->Complete(std::move(admission_result.result));
            return;
        }

        KVCM_LOG_DEBUG("Location statuses updated, submitting task to worker pool with delay: %lld microseconds",
                       static_cast<long long>(delay.count()));
        auto actual_task = std::move(admission_result.actual_task);
        auto execute_task = [this, completion, actual_task]() {
            try {
                completion->Complete(DoLocationDelTask(actual_task));
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

std::future<PlanExecuteResult> SchedulePlanExecutor::Submit(const CacheLocationDelRequest &task) {
    return SubmitLocationDelete(task, ScheduleTaskClass::kSystem);
}

std::future<PlanExecuteResult>
SchedulePlanExecutor::SubmitLocationDelete(const CacheLocationDelRequest &task, ScheduleTaskClass task_class) {
    KVCM_LOG_DEBUG("Submitting location delete task for instance_id: %s, block_keys count: %zu",
                   task.instance_id.c_str(),
                   task.block_keys.size());

    auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    auto future = promise->get_future();
    auto completion = std::make_shared<PromiseCompletion>(promise);
    RunDeleteAdmission(completion, task.delay, [this, &task]() { return PrepareDeleteTask(task); }, task_class);
    return future;
}

AsyncDeleteSubmitResult SchedulePlanExecutor::SubmitAsync(const CacheMetaDelRequest &task) {
    KVCM_LOG_DEBUG("Submitting async meta delete task for instance_id: %s, block_keys count: %zu",
                   task.instance_id.c_str(),
                   task.block_keys.size());
    return SubmitDeleteTaskAsync(task.delay, [this, task]() { return PrepareDeleteTask(task); });
}

AsyncDeleteSubmitResult SchedulePlanExecutor::SubmitAsync(const CacheLocationDelRequest &task) {
    KVCM_LOG_DEBUG("Submitting async location delete task for instance_id: %s, block_keys count: %zu",
                   task.instance_id.c_str(),
                   task.block_keys.size());

    return SubmitDeleteTaskAsync(task.delay, [this, task]() { return PrepareDeleteTask(task); });
}

AsyncDeleteSubmitResult
SchedulePlanExecutor::SubmitDeleteTaskAsync(std::chrono::microseconds delay,
                                            std::function<LocationDelAdmissionResult()> prepare) {
    auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    auto future = promise->get_future();
    auto completion = std::make_shared<PromiseCompletion>(promise);
    auto admission_task = [this, completion, delay, prepare = std::move(prepare)]() {
        RunDeleteAdmission(completion, delay, prepare, ScheduleTaskClass::kReclaim);
    };
    auto cancel_task = [completion]() {
        completion->Complete(ErrorCode::EC_ERROR, "SchedulePlanExecutor stopped before delete admission.");
    };
    if (!SubmitRaw(admission_task,
                   std::chrono::microseconds::zero(),
                   cancel_task,
                   ScheduleTaskClass::kReclaim)) {
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
    bool submit_result = this->SubmitRaw(
        execute_task, task.delay, cancel_task, ScheduleTaskClass::kMigrationContinuation);
    if (!submit_result) {
        HandleErrorPromise(promise, ErrorCode::EC_ERROR, "submit copy task failed");
        return future;
    }
    return future;
}

} // namespace kv_cache_manager
