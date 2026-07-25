#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

#ifndef KVCM_GAUGE_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR
#define KVCM_GAUGE_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(name)                                                            \
public:                                                                                                                \
    DECLARE_METRICS_NAME_(schedule_plan_executor, name);                                                               \
    DEFINE_GET_METRICS_GAUGE_(schedule_plan_executor, name)                                                            \
                                                                                                                       \
private:                                                                                                               \
    DECLARE_METRICS_GAUGE_(schedule_plan_executor, name);
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

struct AsyncDeleteSubmitResult {
    bool accepted{false};
    std::future<PlanExecuteResult> future;
};

struct CacheLocationDelRequest {
    std::string instance_id;
    std::vector<int64_t> block_keys;
    std::vector<std::vector<std::string>> location_ids;
    std::chrono::microseconds delay{std::chrono::seconds(0)};
    // Optional serialized values observed by the submitter, parallel to
    // location_ids. A location is reclaimed only if it is still unchanged.
    std::vector<std::vector<std::string>> expected_location_values;
};

// 单个 block 的跨存储复制请求。URI 由上层（MigrationManager）解析与预分配后传入；
// SchedulePlanExecutor 只负责异步调用 backend.Copy 搬字节并回报结果，不创建 location/不做 CAS
// （目标 location 的创建与 CLS_WRITING->SERVING 的状态流转由 MigrationManager 负责）。
struct CacheLocationCopyRequest {
    std::string instance_id;
    int64_t block_key;
    std::string exec_storage_name;        // 执行 Copy 的 backend（一期 = 源 storage 的 unique name）
    std::vector<DataStorageUri> src_uris; // 源端各 spec 的 uri
    std::vector<DataStorageUri> dst_uris; // 目标端各 spec 预分配的 uri（与 src_uris 一一对应）
    std::chrono::microseconds delay{std::chrono::seconds(0)};
};

// 任务类别同时定义 ready task 的调度优先级。Migration continuation 已经持有活跃
// reservation/WRITING 目标，必须先于新的 Prepare 收敛；所有 migration 类别共同受
// 进程级 worker budget 约束。
enum class ScheduleTaskClass : std::uint8_t {
    kReclaim = 0,
    kSystem = 1,
    kMigrationContinuation = 2,
    kMigrationPrepare = 3,
    kCount = 4,
};

struct ScheduledTask {
    std::function<void()> task;
    std::function<void()> cancel_task;
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
                                  const std::shared_ptr<MetricsRegistry> &metrics_registry,
                                  unsigned int migration_worker_budget = std::numeric_limits<unsigned int>::max());
    ~SchedulePlanExecutor();

    std::future<PlanExecuteResult> Submit(const CacheMetaDelRequest &task);
    std::future<PlanExecuteResult> Submit(const CacheLocationDelRequest &task);
    std::future<PlanExecuteResult> Submit(const CacheLocationCopyRequest &task);
    AsyncDeleteSubmitResult SubmitAsync(const CacheMetaDelRequest &task);
    AsyncDeleteSubmitResult SubmitAsync(const CacheLocationDelRequest &task);

    bool SubmitNonBlocking(const CacheMetaDelRequest &req,
                           ScheduleTaskClass task_class = ScheduleTaskClass::kSystem);
    bool SubmitNonBlocking(const CacheLocationDelRequest &req,
                           ScheduleTaskClass task_class = ScheduleTaskClass::kSystem);

    bool SubmitTask(std::function<void()> task, std::chrono::microseconds delay = std::chrono::microseconds(0));
    bool SubmitTask(ScheduleTaskClass task_class,
                    std::function<void()> task,
                    std::chrono::microseconds delay = std::chrono::microseconds(0),
                    std::function<void()> cancel_task = {});

private:
    struct PromiseCompletion;
    struct LocationDelAdmissionResult {
        PlanExecuteResult result{ErrorCode::EC_OK, ""};
        CacheLocationDelRequest actual_task;
        bool needs_physical_delete{false};
    };

    std::shared_ptr<MetaIndexerManager> meta_manager_;
    std::shared_ptr<DataStorageManager> data_storage_manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_;

    static constexpr std::size_t kTaskClassCount = static_cast<std::size_t>(ScheduleTaskClass::kCount);
    std::array<std::multiset<ScheduledTask>, kTaskClassCount> task_queues_;
    std::size_t migration_worker_budget_{0};
    std::size_t running_migration_tasks_{0};
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<uint64_t> sequence_counter_{0};

    void WorkerRoutine();

    void Stop();
    bool SubmitRaw(std::function<void()> task,
                   std::chrono::microseconds delay,
                   std::function<void()> cancel_task = {},
                   ScheduleTaskClass task_class = ScheduleTaskClass::kSystem);
    static bool IsMigrationTaskClass(ScheduleTaskClass task_class);
    static std::size_t TaskClassIndex(ScheduleTaskClass task_class);
    std::size_t WaitingTaskCountLocked() const;
    static bool FillActualTask(const std::vector<int64_t> &batch_cas_block_keys,
                               const std::vector<std::vector<MetaSearcher::LocationCASTask>> &batch_cas_tasks,
                               const std::vector<std::vector<ErrorCode>> &batch_results,
                               CacheLocationDelRequest &actual_task,
                               std::string &error_message);
    LocationDelAdmissionResult PrepareDeleteTask(const CacheMetaDelRequest &task);
    LocationDelAdmissionResult PrepareDeleteTask(const CacheLocationDelRequest &task);
    LocationDelAdmissionResult PrepareDeleteTaskImpl(const std::string &instance_id,
                                                     const std::vector<int64_t> &block_keys,
                                                     const std::vector<std::vector<std::string>> *target_location_ids,
                                                     const std::vector<std::vector<std::string>>
                                                         *expected_location_values,
                                                     std::chrono::microseconds delay);
    void RunDeleteAdmission(const std::shared_ptr<PromiseCompletion> &completion,
                            std::chrono::microseconds delay,
                            const std::function<LocationDelAdmissionResult()> &prepare,
                            ScheduleTaskClass task_class);
    AsyncDeleteSubmitResult SubmitDeleteTaskAsync(std::chrono::microseconds delay,
                                                  std::function<LocationDelAdmissionResult()> prepare);
    std::future<PlanExecuteResult> SubmitMetaDelete(const CacheMetaDelRequest &task, ScheduleTaskClass task_class);
    std::future<PlanExecuteResult> SubmitLocationDelete(const CacheLocationDelRequest &task,
                                                        ScheduleTaskClass task_class);
    PlanExecuteResult DoLocationDelTask(const CacheLocationDelRequest &task);
    void DoCopyTask(const std::shared_ptr<std::promise<PlanExecuteResult>> &promise,
                    const CacheLocationCopyRequest &task);

    KVCM_GAUGE_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(waiting_task_count)
    KVCM_GAUGE_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(executing_task_count)
    KVCM_GAUGE_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(waiting_migration_task_count)
    KVCM_GAUGE_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR(executing_migration_task_count)
};

#undef KVCM_GAUGE_METRICS_FOR_SCHEDULE_PLAN_EXECUTOR

} // namespace kv_cache_manager
