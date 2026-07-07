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
class RequestContext;

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
    // F-18: 按 (instance_id, block_key) 作用域查找——location_id 不保证跨 instance/block 全局唯一，
    // 且作用域查找为 O(1)（避免全表扫描）。
    bool HasActiveCopyTargetLocation(const std::string &instance_id,
                                     int64_t block_key,
                                     const std::string &location_id) const;
    size_t ActiveTaskCount() const;

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

    // ---- 编排入口（API 触发；F-05 domain API，供 CacheManager::MigrateCache 薄 facade 调用）----
    // 承载迁移编排本身：候选选择 + location 批查(MetaSearcher) + 逐 block 准入(CheckCopyAdmission)
    // + Copy(BatchSubmit)/Mark(MarkForTieredWrite) 分发与 copy 失败回落 + accepted/rejected 统计。
    // 前置条件（instance 存在=meta_indexer 非空、target 已注册=F-02、group 有 strategy=F-01）由 facade
    // 完成后再调本方法；meta_indexer 由调用方传入（facade 已按 instance 取得并做非空校验）。
    struct MigrateResult {
        ErrorCode ec = EC_OK;
        int64_t accepted = 0;
        int64_t rejected = 0;
        std::string message;
    };
    MigrateResult MigrateCache(RequestContext *request_context,
                               const std::string &trace_id,
                               const std::string &instance_id,
                               const std::shared_ptr<MetaIndexer> &meta_indexer,
                               const std::string &src_name,
                               const std::string &dst_name,
                               bool do_copy,
                               bool do_mark,
                               const std::vector<int64_t> &explicit_block_keys,
                               int64_t sample_count);

    // ---- 统计 ----
    MigrationStats GetStats() const;

private:
    // ---- 仅供测试/诊断（BUILD 通过 -fno-access-control 访问；F-07）----
    std::string GetActiveTaskDstLocation(const std::string &instance_id, int64_t block_key) const;
    void DebugInsertActiveCopyTask(const std::string &instance_id, int64_t block_key, const std::string &dst_location_id);
    void DebugEnableCopySubmissionsForTest();

    // F-11: 活跃 Copy 任务的收尾状态机。用于让外部 Cancel 线程与单一 monitor 完成线程在
    // task_mutex_ 内原子认领"谁负责收尾"，避免 cancel 与 promote 并发误删刚提升的目标。
    //   kRunning    —— copy 提交后、future 未完成，正常态；
    //   kCompleting —— monitor 已认领完成（promote/失败清理进行中），此后 Cancel 太晚；
    //   kCancelling —— 外部已请求取消，收尾推迟到 future 完成时由 monitor 执行（删 WRITING 目标、不 promote）。
    enum class CopyTaskState { kRunning, kCompleting, kCancelling };

    // 单个活跃 Copy 任务的上下文。
    struct CopyTaskContext {
        std::string instance_id;
        int64_t block_key = 0;
        std::string src_location_id;
        int64_t src_create_time = 0; // F-08: 提交时源 location 的 create_time，OnTaskSuccess 比对以防 id 复用
        std::string src_storage_name;
        std::string dst_storage_name;
        std::string dst_location_id;
        MigrationRetention retention = MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE;
        std::chrono::steady_clock::time_point submit_time;
        uint64_t total_bytes = 0;              // 源端各 spec 字节数之和（取自 uri size 参数）
        CopyTaskState state = CopyTaskState::kRunning; // F-11: 收尾认领状态
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
    // MigrateCache 候选选取：显式 block_keys 优先；否则按 sample_count 采样（<=0 用默认 100）。
    std::pair<ErrorCode, std::vector<std::int64_t>>
    SelectMigrationCandidateKeys(RequestContext *request_context,
                                 const std::string &trace_id,
                                 const std::vector<int64_t> &explicit_block_keys,
                                 int64_t sample_count,
                                 const std::shared_ptr<MetaIndexer> &meta_indexer) const;
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

    // ---- 活跃任务表操作收口（F-13 Part A；均要求调用方持有 task_mutex_）----
    // 统一处理"内层 map 空则 erase 外层 + 更新 active gauge"，避免各业务路径重复手写、遗漏。
    // 业务语义的 stats/event 仍由各调用方按转换语义各自发出。
    // 返回该 (instance, block) 之前是否已存在活跃任务（true 表示重复）。
    bool InsertActiveTaskLocked(CopyTaskContext ctx);
    // 移除 (instance, block) 的活跃任务；不存在返回 false。
    bool RemoveActiveTaskLocked(const std::string &instance_id, int64_t block_key);
    // 仅判断 (instance, block) 是否已有活跃任务（存在性查询，不拷贝 ctx）。
    bool HasActiveTaskLocked(const std::string &instance_id, int64_t block_key) const;

    // ---- F-11 收尾认领（均要求调用方持有 task_mutex_）----
    enum class ClaimResult { kClaimedRunning, kWasCancelling, kBusyCompleting, kNotFound };
    // monitor 完成路径认领：kRunning→置 kCompleting 并拷 ctx（kClaimedRunning）；
    // kCancelling→拷 ctx 返回 kWasCancelling（不改状态，交由 CompleteCancelledTask 收尾）。
    ClaimResult ClaimForCompletionLocked(const std::string &instance_id, int64_t block_key, CopyTaskContext &out_ctx);
    enum class CancelResult { kMarked, kAlreadyCancelling, kBusyCompleting, kNotFound };
    // 外部 Cancel 认领：kRunning→置 kCancelling（kMarked）；kCancelling→幂等；kCompleting→太晚。
    CancelResult MarkCancellingLocked(const std::string &instance_id, int64_t block_key);
    // 取消任务的延迟收尾（monitor 线程，入口不持锁）：删 WRITING 目标（源不动）+ 移除活跃任务
    // + cancelled 终态 metric/event/log。仅由 OnTaskSuccess/OnTaskFailed 在认领到 kWasCancelling 时调用。
    void CompleteCancelledTask(const CopyTaskContext &ctx);

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
    Counter m_tasks_completed_cancelled_; // F-16: cancelled 终态，与 success/failed 对称
    Gauge m_tasks_active_;
    Counter m_copy_bytes_total_;
    Gauge m_copy_duration_ms_;
    Gauge m_marks_active_;
    Counter m_marks_consumed_total_;
    Counter m_marks_expired_total_; // F-16: 超时过期清除的 mark（与 consumed 区分：浪费 vs 有效消费）

    void UpdateActiveTasksGauge(); // 调用方持有 task_mutex_
    void UpdateMarksActiveGauge(); // best-effort：基于 added-cleared 原子计数
};

} // namespace kv_cache_manager
