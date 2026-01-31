#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

#ifndef KVCM_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR
#define KVCM_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(name)                                                                  \
public:                                                                                                                \
    DECLARE_METRICS_NAME_(schedule_plan_executor, name);                                                               \
    DEFINE_GET_METRICS_COUNTER_(schedule_plan_executor, name)                                                          \
                                                                                                                       \
private:                                                                                                               \
    DECLARE_METRICS_COUNTER_(schedule_plan_executor, name);
#endif

class MetaIndexerManager;
class DataStorageManager;

struct CacheMetaDelRequest {
    std::string instance_id;
    std::vector<int64_t> block_keys;
    std::chrono::microseconds delay{std::chrono::seconds(0)};
};

struct PlanExecuteResult {
    ErrorCode status;
    std::string error_message;
};

struct CacheLocationDelRequest {
    std::string instance_id;
    std::vector<int64_t> block_keys;
    std::vector<std::vector<std::string>> location_ids;
    std::chrono::microseconds delay{std::chrono::seconds(0)};
};

struct ScheduledTask {
    std::function<void()> task;
    std::chrono::steady_clock::time_point execute_time;
    uint64_t sequence_id;

    bool operator<(const ScheduledTask &other) const {
        if (execute_time != other.execute_time) {
            return execute_time < other.execute_time;
        }
        // ensure strict weak ordering when execute_time is same
        return sequence_id < other.sequence_id;
    }
};

class SchedulePlanExecutor {
public:
    explicit SchedulePlanExecutor(unsigned int thread_count,
                                  const std::shared_ptr<MetaIndexerManager> &meta_manager,
                                  const std::shared_ptr<DataStorageManager> &storage_manager,
                                  const std::shared_ptr<MetricsRegistry> &metrics_registry);
    ~SchedulePlanExecutor();

    std::future<PlanExecuteResult> Submit(const CacheMetaDelRequest &task);
    std::future<PlanExecuteResult> Submit(const CacheLocationDelRequest &task);

private:
    std::shared_ptr<MetaIndexerManager> meta_manager_;
    std::shared_ptr<DataStorageManager> data_storage_manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_;

    std::multiset<ScheduledTask> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<uint64_t> sequence_counter_{0};

    void WorkerRoutine();

    void Stop();
    bool SubmitRaw(const std::function<void()> &task, std::chrono::microseconds delay);
    static bool FillActualTask(const std::vector<int64_t> &batch_cas_block_keys,
                               const std::vector<std::vector<MetaSearcher::LocationCASTask>> &batch_cas_tasks,
                               const std::vector<std::vector<ErrorCode>> &batch_results,
                               CacheLocationDelRequest &actual_task,
                               std::string &error_message);
    void DoLocationDelTask(const std::shared_ptr<std::promise<PlanExecuteResult>> &promise,
                           const CacheLocationDelRequest &task);

    KVCM_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(waiting_task_count)
    KVCM_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(executing_task_count)
};

#undef KVCM_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR

} // namespace kv_cache_manager
