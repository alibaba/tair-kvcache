#include "kv_cache_manager/manager/schedule_plan_executor.h"

#include <memory>
#include <unordered_set>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/manager/cache_location.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

#define DEFINE_METRICS_NAME_FOR_SCHEDULE_PLAN_EXECUTOR(name) \
    DEFINE_METRICS_NAME_(SchedulePlanExecutor, schedule_plan_executor, name)

#define REGISTER_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(name) \
    REGISTER_METRICS_COUNTER_(metrics_registry_, schedule_plan_executor, name)

namespace {
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

DEFINE_METRICS_NAME_FOR_SCHEDULE_PLAN_EXECUTOR(waiting_task_count);
DEFINE_METRICS_NAME_FOR_SCHEDULE_PLAN_EXECUTOR(executing_task_count);

SchedulePlanExecutor::SchedulePlanExecutor(unsigned int thread_count,
                                           const std::shared_ptr<MetaIndexerManager> &meta_manager,
                                           const std::shared_ptr<DataStorageManager> &storage_manager,
                                           const std::shared_ptr<MetricsRegistry> &metrics_registry)
    : meta_manager_(meta_manager)
    , data_storage_manager_(storage_manager)
    , metrics_registry_(metrics_registry)
    , stop_(false) {
    REGISTER_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(waiting_task_count);
    REGISTER_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(executing_task_count);

    if (thread_count == 0)
        thread_count = 1;

    for (unsigned int i = 0; i < thread_count; ++i) {
        workers_.emplace_back([this]() { WorkerRoutine(); });
    }
    KVCM_LOG_INFO("SchedulePlanExecutor initialized with %u worker threads", thread_count);
}
void SchedulePlanExecutor::Stop() {
    KVCM_LOG_DEBUG("Stopping SchedulePlanExecutor...");
    stop_ = true;
    condition_.notify_all();

    for (auto &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    KVCM_LOG_DEBUG("SchedulePlanExecutor stopped");
}

SchedulePlanExecutor::~SchedulePlanExecutor() { Stop(); }

void SchedulePlanExecutor::WorkerRoutine() {
    while (!stop_) {
        std::function<void()> task;

        {
            auto wait_until_time = std::chrono::steady_clock::time_point::max();
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // 检查是否有可执行的任务
            if (!tasks_.empty()) {
                auto earliest_task = tasks_.begin();
                auto now = std::chrono::steady_clock::now();

                if (earliest_task->execute_time <= now) {
                    // 租约时间已到，取出任务执行
                    task = earliest_task->task;
                    tasks_.erase(earliest_task);
                } else {
                    // 租约时间未到，记录等待时间点
                    wait_until_time = earliest_task->execute_time;
                }
            }

            if (!task) {
                if (wait_until_time != std::chrono::steady_clock::time_point::max()) {
                    // 等待到最早任务的时间或新任务到达
                    condition_.wait_until(lock, wait_until_time);
                } else if (tasks_.empty()) {
                    // 队列为空，等待新任务
                    condition_.wait(lock, [this] { return stop_ || (!tasks_.empty()); });
                }
                continue;
            }
        }

        if (task) {
            --METRICS_(schedule_plan_executor, waiting_task_count);
            ++METRICS_(schedule_plan_executor, executing_task_count);
            task();
            --METRICS_(schedule_plan_executor, executing_task_count);
        }
    }
}

void SchedulePlanExecutor::DoLocationDelTask(const std::shared_ptr<std::promise<PlanExecuteResult>> &promise,
                                             const CacheLocationDelRequest &task) {
    PlanExecuteResult result;
    result.status = ErrorCode::EC_OK;

    if (task.block_keys.size() != task.location_ids.size()) {
        HandleErrorPromise(promise,
                           ErrorCode::EC_BADARGS,
                           "block_keys size %d != location_ids size %d",
                           task.block_keys.size(),
                           task.location_ids.size());
        return;
    }

    std::shared_ptr<MetaIndexer> indexer = meta_manager_->GetMetaIndexer(task.instance_id);
    if (!indexer) {
        HandleErrorPromise(promise, ErrorCode::EC_NOENT, "MetaIndexer %s not found", task.instance_id.c_str());
        return;
    }

    MetaSearcher meta_searcher(indexer);
    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    auto ctx = std::make_shared<RequestContext>("schedule_plan_executor_call");
    ErrorCode get_locations_ec = meta_searcher.BatchGetLocation(ctx.get(), task.block_keys, empty_mask, location_maps);
    if (get_locations_ec != ErrorCode::EC_OK) {
        HandleErrorPromise(promise, ErrorCode::EC_ERROR, "Failed to get block locations, ec: %d", get_locations_ec);
        return;
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
            if (iter == location_map.end()) {
                continue;
            }
            if (iter->second.status() != CacheLocationStatus::CLS_DELETING) {
                continue;
            }
            for (const auto &loc_spec : iter->second.location_specs()) {
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
        for (size_t i = 0; i < delete_results.size(); ++i) {
            if (delete_results[i] != ErrorCode::EC_OK) {
                // 这里存储删除报错暂且不管，报个warn表示哪个storageUri删失败了
                result.status = ErrorCode::EC_PARTIAL_OK;
                KVCM_LOG_WARN("Failed to delete kvcache from storage %s, uri: %s",
                              storage_unique_name.c_str(),
                              storage_uris[i].ToUriString().c_str());
            }
        }
    }

    // delete locations
    if (!batch_cad_tasks.empty()) {
        std::vector<std::vector<ErrorCode>> delete_meta_results;
        ErrorCode delete_meta_ec =
            meta_searcher.BatchCADLocationStatus(ctx.get(), block_keys_to_delete, batch_cad_tasks, delete_meta_results);
        (void)delete_meta_ec; // 忽略返回值
        for (size_t block_key_idx = 0; block_key_idx < delete_meta_results.size(); ++block_key_idx) {
            auto &results = delete_meta_results[block_key_idx];
            for (size_t location_idx = 0; location_idx < results.size(); location_idx++) {
                if (results[location_idx] != ErrorCode::EC_OK) {
                    result.status = ErrorCode::EC_PARTIAL_OK;
                    KVCM_LOG_WARN("Failed to CAD meta key %ld, location: %s, error_code: %d",
                                  block_keys_to_delete[block_key_idx],
                                  batch_cad_tasks[block_key_idx][location_idx].location_id.c_str(),
                                  results[location_idx]);
                }
            }
        }
    }
    KVCM_LOG_DEBUG("DoDelLocationTask completed successfully for instance_id: %s", task.instance_id.c_str());

    promise->set_value(result);
}

bool SchedulePlanExecutor::SubmitRaw(const std::function<void()> &task, std::chrono::microseconds delay) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (stop_) {
            return false;
        }

        auto execute_time = std::chrono::steady_clock::now() + delay;
        uint64_t sequence_id = sequence_counter_.fetch_add(1, std::memory_order_relaxed);
        tasks_.emplace(ScheduledTask{task, execute_time, sequence_id});
    }
    ++METRICS_(schedule_plan_executor, waiting_task_count);

    condition_.notify_one();
    return true;
}

std::future<PlanExecuteResult> SchedulePlanExecutor::Submit(const CacheMetaDelRequest &task) {
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
            cas_tasks.push_back({loc_kv.first, loc_kv.second.status(), CLS_DELETING});
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
    KVCM_LOG_DEBUG("Location statuses updated, submitting task to worker pool with delay: %lld microseconds",
                   static_cast<long long>(task.delay.count()));

    bool submit_result =
        this->SubmitRaw([this, promise, actual_task]() { DoLocationDelTask(promise, actual_task); }, task.delay);
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
            "Size mismatch between batch_results(%d), batch_cas_block_keys(%d) and batch_cas_tasks(%d).",
            batch_results.size(),
            batch_cas_block_keys.size(),
            batch_cas_tasks.size());
        KVCM_LOG_ERROR("%s", error_message.c_str());
        return false;
    }

    for (size_t key_idx = 0; key_idx < batch_results.size(); key_idx++) {
        auto &results = batch_results[key_idx];
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
std::future<PlanExecuteResult> SchedulePlanExecutor::Submit(const CacheLocationDelRequest &task) {
    KVCM_LOG_DEBUG("Submitting location delete task for instance_id: %s, block_keys count: %zu",
                   task.instance_id.c_str(),
                   task.block_keys.size());

    auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    std::future<PlanExecuteResult> future = promise->get_future();

    if (stop_) {
        HandleErrorPromise(promise, ErrorCode::EC_ERROR, "SchedulePlanExecutor stopped.");
        return future;
    }

    std::shared_ptr<MetaIndexer> indexer = meta_manager_->GetMetaIndexer(task.instance_id);
    if (!indexer) {
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
        HandleErrorPromise(promise, ErrorCode::EC_ERROR, "Failed to get block locations, ec: %d", get_locations_ec);
        return future;
    }

    std::vector<int64_t> batch_cas_block_keys;
    std::vector<std::vector<MetaSearcher::LocationCASTask>> batch_cas_tasks;
    for (size_t block_key_idx = 0; block_key_idx < task.block_keys.size(); ++block_key_idx) {
        std::unordered_set<std::string> target_location_ids(task.location_ids[block_key_idx].begin(),
                                                            task.location_ids[block_key_idx].end());
        auto block_key = task.block_keys[block_key_idx];
        auto &location_map = location_maps[block_key_idx];
        std::vector<MetaSearcher::LocationCASTask> location_cas_tasks;
        for (const auto &loc_kv : location_map) {
            const auto &location = loc_kv.second;
            if (location.status() == CacheLocationStatus::CLS_DELETING) {
                continue; // ignore already deleting
            }
            if (target_location_ids.find(location.id()) == target_location_ids.end()) {
                continue; // ignore not target location
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
        promise->set_value(PlanExecuteResult{ErrorCode::EC_OK, ""});
        return future;
    }

    std::vector<std::vector<ErrorCode>> batch_results;
    ErrorCode update_ec = meta_searcher.BatchCASLocationStatus(
        request_context.get(), batch_cas_block_keys, batch_cas_tasks, batch_results);
    if (update_ec != ErrorCode::EC_OK) {
        KVCM_LOG_DEBUG("Location status BatchCASLocationStatus not ok, ec: %d", update_ec);
    }

    std::string error_message;
    CacheLocationDelRequest actual_task{task.instance_id, {}, {}, task.delay};
    if (!FillActualTask(batch_cas_block_keys, batch_cas_tasks, batch_results, actual_task, error_message)) {
        HandleErrorPromise(promise, ErrorCode::EC_ERROR, "FillActualTask error: %s", error_message.c_str());
        return future;
    }
    if (actual_task.block_keys.empty()) {
        promise->set_value(PlanExecuteResult{ErrorCode::EC_OK, ""});
        return future;
    }

    KVCM_LOG_DEBUG("Location statuses updated, submitting task to worker pool with delay: %lld microseconds",
                   static_cast<long long>(task.delay.count()));

    bool submit_result =
        this->SubmitRaw([this, promise, actual_task]() { DoLocationDelTask(promise, actual_task); }, task.delay);
    if (!submit_result) {
        HandleErrorPromise(promise, ErrorCode::EC_ERROR, "submit task failed");
        return future;
    }

    return future;
}

} // namespace kv_cache_manager
