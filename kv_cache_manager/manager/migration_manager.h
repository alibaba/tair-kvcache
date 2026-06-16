#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/config/migration_strategy.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

class MetaIndexerManager;
class DataStorageManager;
class EventManager;

/**
 * MigrationManager —— 多层存储迁移统一控制面。
 *
 * 不区分触发来源（水位 / API），按配置的执行方式分发：
 *   - Copy 路径：建目标 location(CLS_WRITING) -> 提交 CacheLocationCopyRequest 给
 *     SchedulePlanExecutor 异步搬字节 -> copy 完成后确认源端仍 SERVING ->
 *     CAS 目标 WRITING->SERVING -> 按 retention 处理源端 -> 清标 -> 移除活跃任务；
 *     失败则 CAS 目标 WRITING->DELETING。
 *   - Mark 路径：通过 MetaIndexer 在 block 级 property 上打标并持久化（Redis + local cache），
 *     供写路径批量查询并由 StartWriteCache 消费。打标天然随元数据持久化，
 *     crash 后自动可见（继续按持久化的 Mark 迁移）。
 *
 * 不引入新状态机（复用 CLS_WRITING）；crash 恢复交给 Reclaimer 孤儿检测。
 * MigrationManager 维护一个活跃任务集合（instance_id + block_key）用于防重复迁移与并发预算统计。
 *
 * 线程模型（一期）：copy 任务在 SchedulePlanExecutor 内同步执行；MigrationManager 起一个监控线程
 * 轮询 copy future，就绪后驱动状态流转。
 */
class MigrationManager {
public:
    // 一条 Copy 迁移请求：把 instance/block 下 src_location 的数据复制到 dst_storage。
    struct MigrationRequest {
        std::string instance_id;
        int64_t block_key = 0;
        std::string src_location_id;
        std::string src_storage_name; // 执行 Copy 的 backend（一期 = 源 storage 的 unique name）
        std::string dst_storage_name; // 目标 storage（冷层）
        MigrationRetention retention = MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE;
    };

    struct MigrationStats {
        uint64_t copy_submitted = 0;
        uint64_t copy_completed = 0;
        uint64_t copy_failed = 0;
        uint64_t copy_cancelled = 0;
        uint64_t marks_added = 0;
        uint64_t marks_cleared = 0;
        size_t active_copy_tasks = 0;
        size_t active_marks = 0;
    };

    MigrationManager(std::shared_ptr<SchedulePlanExecutor> schedule_plan_executor,
                     std::shared_ptr<MetaIndexerManager> meta_indexer_manager,
                     std::shared_ptr<DataStorageManager> data_storage_manager,
                     std::shared_ptr<MetricsRegistry> metrics_registry = nullptr,
                     std::shared_ptr<EventManager> event_manager = nullptr);
    ~MigrationManager();

    MigrationManager(const MigrationManager &) = delete;
    MigrationManager &operator=(const MigrationManager &) = delete;

    void Start();
    void Stop();

    // ---- Copy 路径 ----
    // 同步完成"建目标 location + 提交 copy 任务"，copy 字节复制异步执行；
    // 返回 EC_OK 表示已成功登记任务（不代表复制完成）。
    ErrorCode Submit(const std::string &trace_id, MigrationRequest request);
    // 批量提交，返回逐项结果（与 requests 一一对应）。
    std::vector<ErrorCode> BatchSubmit(const std::string &trace_id, std::vector<MigrationRequest> requests);

    // ---- copy 任务完成回调（监控线程驱动，亦可供测试直接调用） ----
    void OnTaskSuccess(const std::string &instance_id, int64_t block_key);
    void OnTaskFailed(const std::string &instance_id, int64_t block_key, ErrorCode reason);

    // ---- Mark 路径（MetaIndexer 持久化：block 级 property，随元数据落 Redis + local cache） ----
    // 打标/清标走 ReadModifyWriteBlock 只写 property（不动 location）；查标走 GetProperties。
    ErrorCode MarkForTieredWrite(const std::string &instance_id,
                                 const std::vector<int64_t> &block_keys,
                                 const std::string &dst_storage_name,
                                 int64_t timeout_ms = MigrationMarkMethod::kDefaultTimeoutMs);
    bool IsMarkedForTieredWrite(const std::string &instance_id, int64_t block_key) const;
    std::string GetTieredWriteTarget(const std::string &instance_id, int64_t block_key) const;
    void ClearTieredWriteMark(const std::string &instance_id, int64_t block_key);

    // 批量查询（写路径优化，避免逐 block 元数据往返）；out_targets 与 block_keys 同序，空串=未打标。
    ErrorCode BatchGetTieredWriteTargets(const std::string &instance_id,
                                         const std::vector<int64_t> &block_keys,
                                         std::vector<std::string> &out_targets) const;

    // Mark 持久化属性名（block 级 property）。
    static const std::string PROPERTY_TIERED_WRITE_TARGET; // 值=目标冷 storage；空串=未打标/已清
    static const std::string PROPERTY_TIERED_WRITE_DEADLINE_MS; // 值=过期时间戳(ms)；空/缺失=不过期旧标记

    // ---- 任务控制 ----
    ErrorCode Cancel(const std::string &instance_id, int64_t block_key);
    std::vector<ErrorCode> BatchCancel(const std::string &instance_id, const std::vector<int64_t> &block_keys);

    // ---- 查询（防重复迁移 + 并发预算） ----
    bool HasMigrationTask(const std::string &instance_id, int64_t block_key) const;
    // Reclaimer 孤儿检测使用：运行期不能回收活跃 Copy 任务正在写入的目标 location。
    bool HasActiveCopyTargetLocation(const std::string &location_id) const;
    size_t ActiveTaskCount() const;

    // 仅供测试/诊断：返回某 block_key 当前活跃 Copy 任务的目标 location_id（无任务返回空串）。
    std::string GetActiveTaskDstLocation(const std::string &instance_id, int64_t block_key) const;
    // 仅供测试/诊断：注入一个伪造的活跃 Copy 任务上下文（用于覆盖活跃目标保护等分支）。
    void DebugInsertActiveCopyTask(const std::string &instance_id, int64_t block_key, const std::string &dst_location_id);
    // 仅供测试：允许不开 monitor 直接提交 Copy，再手动驱动 OnTaskSuccess/OnTaskFailed。
    void DebugEnableCopySubmissionsForTest();

    // ---- Copy 准入策略（CacheReclaimer / AdminServiceImpl 共用） ----
    enum class CopyAdmissionStatus {
        kAccept,
        kAlreadyMigrating,       // 该 block_key 已有活跃 Copy 任务
        kTargetServingExists,    // 目标 storage 上已存在 SERVING 副本
        kTargetWritingExists,    // 目标 storage 上存在 WRITING 副本（可能为其他迁移半成品）
        kSourceServingNotFound,  // 源 storage 上没有 SERVING 副本，无可复制源
    };

    struct CopyAdmission {
        CopyAdmissionStatus status = CopyAdmissionStatus::kAccept;
        const CacheLocation *src_location = nullptr; // 仅 kAccept 时有效
    };

    // 在某个候选 block 上做 Copy 准入判断（无副作用）。
    // - instance_id/block_key：候选 block 的实例作用域与 block id；
    // - loc_map：该 block 当前所有 CacheLocation（来自 MetaSearcher::BatchGetLocation）；
    // - src_storage_name / dst_storage_name：迁移源/目标 storage 的 unique name。
    CopyAdmission CheckCopyAdmission(const std::string &instance_id,
                                     int64_t block_key,
                                     const CacheLocationMap &loc_map,
                                     const std::string &src_storage_name,
                                     const std::string &dst_storage_name) const;

    // ---- 统计 ----
    MigrationStats GetStats() const;

private:
    // 单个活跃 Copy 任务的上下文。
    struct CopyTaskContext {
        std::string instance_id;
        int64_t block_key = 0;
        std::string src_location_id;
        std::string src_storage_name;
        std::string dst_storage_name;
        std::string dst_location_id;
        MigrationRetention retention = MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE;
        std::chrono::steady_clock::time_point submit_time;
        uint64_t total_bytes = 0; // 源端各 spec 字节数之和（取自 uri size 参数）
    };

    // 监控线程待处理的 copy future。
    struct PendingCopy {
        std::string instance_id;
        int64_t block_key = 0;
        std::future<PlanExecuteResult> future;
    };

    struct ExpiringMark {
        int64_t deadline_ms = 0;
        std::string instance_id;
        int64_t block_key = 0;
        std::string target_storage;
    };

    struct ExpiringMarkGreater {
        bool operator()(const ExpiringMark &lhs, const ExpiringMark &rhs) const {
            return lhs.deadline_ms > rhs.deadline_ms;
        }
    };

    // 解析源 location -> 在目标 storage 预分配 dst_uris -> 建目标 location(CLS_WRITING)。
    // 成功时填充 out_ctx、out_src_uris、out_dst_uris。
    ErrorCode PrepareCopyTask(const std::string &trace_id,
                              const MigrationRequest &request,
                              CopyTaskContext &out_ctx,
                              std::vector<DataStorageUri> &out_src_uris,
                              std::vector<DataStorageUri> &out_dst_uris);

    // 提交目标 location 的删除任务（失败 / 取消时清理半成品）。
    void SubmitTargetLocationDelete(const CopyTaskContext &ctx);
    // 提交源 location 的删除任务（delete_source retention）。
    void SubmitSourceLocationDelete(const CopyTaskContext &ctx);
    // Copy 成功回调收尾前，确认源 location 仍可作为有效源副本。
    bool IsSourceLocationServing(const CopyTaskContext &ctx) const;
    void CompleteCopyTaskAsFailed(const CopyTaskContext &ctx, const std::string &fail_reason);

    void MonitorLoop();

    // 取指定 instance 的 MetaIndexer（可能为 nullptr）。
    std::shared_ptr<MetaIndexer> GetIndexer(const std::string &instance_id) const;
    bool ClearTieredWriteMarkInternal(const std::string &instance_id, int64_t block_key);
    bool ClearTieredWriteMarkIfMatchInternal(const std::string &instance_id,
                                             int64_t block_key,
                                             const std::string &expected_target,
                                             int64_t expected_deadline_ms);
    void EnqueueMarkExpiry(const std::string &instance_id,
                           int64_t block_key,
                           const std::string &target_storage,
                           int64_t deadline_ms);
    void ProcessExpiredMarks();
    // Mark 策略：查询 block 级 property，命中时输出目标 storage。
    bool ShouldWriteToTieredStorageByMark(const std::string &instance_id,
                                          int64_t block_key,
                                          std::string &target) const;
    size_t ActiveTaskCountUnsafe() const; // 调用方持有 task_mutex_

    std::shared_ptr<SchedulePlanExecutor> schedule_plan_executor_;
    std::shared_ptr<MetaIndexerManager> meta_indexer_manager_;
    std::shared_ptr<DataStorageManager> data_storage_manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<EventManager> event_manager_;
    bool metrics_enabled_ = false;

    // 活跃 Copy 任务表。
    mutable std::mutex task_mutex_;
    std::unordered_map<std::string, std::unordered_map<int64_t, CopyTaskContext>> active_tasks_by_instance_;

    // 监控线程待处理队列。
    std::mutex pending_mutex_;
    std::condition_variable pending_cv_;
    std::deque<PendingCopy> pending_copies_;

    mutable std::mutex mark_mutex_;
    std::mutex mark_expiry_mutex_;
    std::priority_queue<ExpiringMark, std::vector<ExpiringMark>, ExpiringMarkGreater> mark_expiry_queue_;

    // 线程与生命周期。
    std::thread monitor_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> accepting_copy_submissions_{false};
    std::mutex copy_submission_mutex_;

    // 统计计数（原子，GetStats 汇总）。
    std::atomic<uint64_t> stat_copy_submitted_{0};
    std::atomic<uint64_t> stat_copy_completed_{0};
    std::atomic<uint64_t> stat_copy_failed_{0};
    std::atomic<uint64_t> stat_copy_cancelled_{0};
    std::atomic<uint64_t> stat_marks_added_{0};
    std::atomic<uint64_t> stat_marks_cleared_{0};

    // 可观测指标（仅 metrics_enabled_ 时使用）
    Counter m_tasks_submitted_total_;
    Counter m_tasks_completed_success_;
    Counter m_tasks_completed_failed_;
    Gauge m_tasks_active_;
    Counter m_copy_bytes_total_;
    Gauge m_copy_duration_ms_;
    Gauge m_marks_active_;
    Counter m_marks_consumed_total_;

    void UpdateActiveTasksGauge(); // 调用方持有 task_mutex_
    void UpdateMarksActiveGauge(); // best-effort：基于 added-cleared 原子计数
};

} // namespace kv_cache_manager
