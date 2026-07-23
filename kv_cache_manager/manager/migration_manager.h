#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/config/migration_strategy.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

class MetaIndexerManager;
class DataStorageManager;
class DataStorageSelector;
class EventManager;
class RegistryManager;
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
 * 线程模型：Reclaimer 的 Prepare 与所有 Copy/迁移 cleanup 在共享 SchedulePlanExecutor 中执行，
 * Admin Prepare 保持调用线程同步语义；MigrationManager 监控线程轮询 copy future 并驱动状态流转。
 */
class MigrationManager : public std::enable_shared_from_this<MigrationManager> {
public:
    // 一条 Copy 迁移请求：把 instance/block 下 src_location 的数据复制到 dst_storage。
    struct MigrationRequest {
        std::string instance_group_name; // Copy 并发限流作用域；空表示未启用 group 限流（测试/内部直调）
        std::string instance_id;
        int64_t block_key = 0;
        std::string src_location_id;
        std::string src_storage_name; // 执行 Copy 的 backend（一期 = 源 storage 的 unique name）
        std::string dst_storage_name; // 目标 storage（冷层）
        MigrationRetention retention = MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE;
        // admission 已定位的源 location 信息，避免批量提交时冗余 BatchGetLocation。
        // DispatchMigrationBatch 生成的 prepared request 必须非空；单条 Submit 可留空，由
        // PrepareCopyTask 兼容性地重读 meta。
        std::vector<LocationSpec> src_specs;
        int64_t src_create_time = 0;
    };

    // Copy 提交的 group 级硬上限。BatchSubmit 在短准入锁 + task_mutex_ 内统计 active 并 reserve，
    // 保证 Admin / Reclaimer 并发提交时不会同时看到同一个空闲 slot。
    struct CopyConcurrencyLimit {
        std::string instance_group_name;
        std::size_t max_concurrency;

        CopyConcurrencyLimit(std::string group_name = "", std::size_t concurrency = SIZE_MAX)
            : instance_group_name(group_name), max_concurrency(concurrency) {}

        bool enabled() const { return !instance_group_name.empty() && max_concurrency != SIZE_MAX; }
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
                     std::shared_ptr<EventManager> event_manager = nullptr,
                     std::shared_ptr<RegistryManager> registry_manager = nullptr,
                     std::shared_ptr<DataStorageSelector> data_storage_selector = nullptr);
    ~MigrationManager();

    MigrationManager(const MigrationManager &) = delete;
    MigrationManager &operator=(const MigrationManager &) = delete;

    void Start();
    void Stop();

    // ---- Copy 路径 ----
    // 同步完成"建目标 location + 提交 copy 任务"，copy 字节复制异步执行；
    // 返回 EC_OK 表示已成功登记任务（不代表复制完成）。
    // 低层单条接口，当前仅单元测试直接调用；它不接受 group CopyConcurrencyLimit。生产 Admin/Reclaimer
    // 必须通过 DispatchMigrationBatch -> BatchSubmit 进入统一的 group 级原子限流，勿新增生产直调。
    ErrorCode Submit(const std::string &trace_id, MigrationRequest request);

    // ---- copy 任务完成回调（监控线程驱动，亦可供测试直接调用） ----
    void OnTaskSuccess(const std::string &instance_id, int64_t block_key);
    void OnTaskFailed(const std::string &instance_id, int64_t block_key, ErrorCode reason);

    // ---- Mark 路径（MetaIndexer 持久化：block 级 property，随元数据落 Redis + local cache） ----
    // 打标/清标走 ReadModifyWriteBlock 只写 property（不动 location）；查标走 GetProperties。
    ErrorCode MarkForTieredWrite(const std::string &instance_id,
                                 const std::vector<int64_t> &block_keys,
                                 const std::string &dst_storage_name,
                                 int64_t timeout_ms = MigrationMarkMethod::kDefaultTimeoutMs);
    bool IsMarkedForTieredWrite(const std::string &instance_id, int64_t block_key);
    std::string GetTieredWriteTarget(const std::string &instance_id, int64_t block_key);
    // 按 expected target+deadline 条件清除 mark（不匹配则不清，防止清掉同 block 新 mark）。
    // FinishWriteCache / OnTaskSuccess 使用此接口替代旧的无条件 ClearTieredWriteMark。
    bool ClearTieredWriteMarkIfMatch(const std::string &instance_id,
                                     int64_t block_key,
                                     const std::string &expected_target,
                                     int64_t expected_deadline_ms);

    // mark 查询结果严格与输入 key 对齐；读取失败不能伪装成 no-mark。
    enum class MarkQueryState {
        kNoMark,
        kValid,
        kExpired,
        kBlockNotFound,
        kReadError,
    };
    struct MarkQueryResult {
        MarkQueryState state = MarkQueryState::kNoMark;
        ErrorCode ec = EC_OK;
        // target/deadline 在 kValid/kExpired 时保存查询快照，供 match-clear 使用。
        std::string target;
        int64_t deadline_ms = 0;

        bool HasValidMark() const { return state == MarkQueryState::kValid; }
        bool IsReadError() const { return state == MarkQueryState::kReadError; }
    };

    // 批量查询（写路径优化，避免逐 block 元数据往返）；out 与 block_keys 严格同序、同长度。
    // EC_NOENT 映射为 kBlockNotFound，不算读取故障；其他逐 key 错误返回 EC_PARTIAL_OK/EC_ERROR。
    // 非 const：发现 kExpired mark 时会触发条件清理。
    ErrorCode BatchGetTieredWriteTargets(const std::string &instance_id,
                                         const std::vector<int64_t> &block_keys,
                                         std::vector<MarkQueryResult> &out);

    // Mark 持久化属性名（block 级 property）。
    static const std::string PROPERTY_TIERED_WRITE_TARGET; // 值=目标冷 storage；空串=未打标/已清
    static const std::string PROPERTY_TIERED_WRITE_DEADLINE_MS; // 值=正数过期时间戳(ms)；非法值按 malformed 清理

    // ---- 任务控制 ----
    ErrorCode Cancel(const std::string &instance_id, int64_t block_key);
    std::vector<ErrorCode> BatchCancel(const std::string &instance_id, const std::vector<int64_t> &block_keys);

    // ---- 查询（防重复迁移 + 并发预算） ----
    bool HasMigrationTask(const std::string &instance_id, int64_t block_key) const;
    // Reclaimer 孤儿检测使用：运行期不能回收活跃 Copy 任务正在写入的目标 location。
    // 按 (instance_id, block_key) 作用域查找：location_id 不保证跨 instance/block 全局唯一，
    // 且作用域查找为 O(1)（避免全表扫描）。
    bool HasActiveCopyTargetLocation(const std::string &instance_id,
                                     int64_t block_key,
                                     const std::string &location_id) const;
    size_t ActiveTaskCount() const;
    // 按任务提交时记录的 instance group 统计活跃 Copy，用于统一的 group 级硬限流。
    size_t ActiveTaskCountForGroup(const std::string &instance_group_name) const;
    // 仅供诊断/测试；生产 group 并发预算统一使用 ActiveTaskCountForGroup。
    size_t ActiveTaskCountForInstances(const std::vector<std::string> &instance_ids) const;
    // 取指定 instance 的活跃 copy task block_key 列表（用于 drain 前 BatchCancel）。
    std::vector<int64_t> GetActiveBlockKeysForInstance(const std::string &instance_id) const;

    // instance 级 draining：RemoveInstance 期间阻止该 instance 的所有新迁移提交。
    // Submit/BatchSubmit 在短准入锁下检查此集合，并在释放锁前登记 preparing reservation；
    // BeginDrainingInstance 拿同一把锁 insert 后，早于它准入的任务已对 drain 快照可见，晚于它的
    // 提交则直接拒绝。覆盖 reclaimer + admin 两条提交路径，且不等待其他 instance 的慢 I/O。
    void BeginDrainingInstance(const std::string &instance_id);
    void EndDrainingInstance(const std::string &instance_id);

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

    // ---- 编排入口（API 触发，供 CacheManager::MigrateCache 薄 facade 调用）----
    // 承载迁移编排本身：候选选择 + location 批查(MetaSearcher) + 逐 block 准入(CheckCopyAdmission)
    // + Copy(BatchSubmit)/Mark(MarkForTieredWrite) 分发与 copy 失败回落 + accepted/rejected 统计。
    // 前置条件（instance 存在、meta_indexer 非空、target 已注册、group 有 strategy）由 facade
    // 完成后再调本方法；meta_indexer 由调用方传入（facade 已按 instance 取得并做非空校验）。
    struct MigrateResult {
        ErrorCode ec = EC_OK;
        int64_t accepted = 0;
        int64_t rejected = 0;
        std::string message;
    };
    MigrateResult MigrateCache(RequestContext *request_context,
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
                               int64_t mark_timeout_ms);

    // ---- 共享迁移分发（Admin MigrateCache 与 CacheReclaimer 两路共用）----
    // 对已准备好的 (batch, loc_maps) 执行：逐 block 准入(CheckCopyAdmission) + copy/mark 分类
    // + BatchSubmit + copy 失败回落 mark + MarkForTieredWrite。
    // 两路差异通过 params 参数化（copy slot 限制 / retention / mark timeout / mark 去重）。
    struct DispatchBatchParams {
        bool do_copy = false;
        bool do_mark = false;
        // max_copy_slots 是调用方本批次的提前剪枝；copy_limit 是 BatchSubmit 内的原子硬限制。
        std::size_t max_copy_slots = SIZE_MAX;
        CopyConcurrencyLimit copy_limit;
        MigrationRetention retention = MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE;
        int64_t mark_timeout_ms = MigrationMarkMethod::kDefaultTimeoutMs;
        bool dedup_marks = false; // true=跳过已打标的 block（reclaimer 用；Admin 显式指定 block 不需要）
    };
    struct DispatchBatchResult {
        int64_t copy_submitted = 0;
        int64_t copy_failed = 0;
        int64_t mark_submitted = 0;
    };

    // Reclaimer 只在其单线程中选择候选并生成 pending-delete 快照；Job 跨异步边界后完全
    // own 输入。worker 会重新读取当前 instance/config/strategy/location，再进入统一分发。
    struct AsyncMigrationPrepareJob {
        std::string trace_id;
        std::string instance_group_name;
        std::string instance_id;
        std::string source_storage_name;
        std::string target_storage_name;
        std::vector<int64_t> block_keys;
        std::vector<std::vector<std::string>> pending_location_ids_by_block;
        std::function<void(const DispatchBatchResult &)> on_dispatched;
    };

    // 同一路由最多保留一个 queued/running Prepare。返回 true 仅表示 Job 已入队；实际
    // Copy/Mark 准入结果通过 on_dispatched 和 MigrationManager 指标观察。
    bool SubmitAsyncMigrationPrepare(AsyncMigrationPrepareJob job);
    // 用于 Reclaimer 的轻量 backpressure；包含 queued 与正在执行的 Prepare，不替代 Copy 硬准入。
    std::size_t PendingAsyncMigrationPrepareCountForGroup(const std::string &instance_group_name) const;

    DispatchBatchResult DispatchMigrationBatch(const std::string &trace_id,
                                              const std::string &instance_id,
                                              const std::string &src_name,
                                              const std::string &dst_name,
                                              const std::vector<int64_t> &batch,
                                              const std::vector<CacheLocationMap> &loc_maps,
                                              const DispatchBatchParams &params);

    // ---- 统计 ----
    MigrationStats GetStats() const;

private:
    // 仅供 DispatchMigrationBatch 使用的 prepared-request API，返回结果与 requests 逐项对齐。
    // 前置条件：同批 request 属于同一 instance、目标 storage 相同、block_key 已去重，且每项已经
    // 完成 admission 并填充非空 src_location_id/src_storage_name/src_specs。实现仍在入口做轻量
    // shape 校验，违反 contract 时整批返回 EC_BADARGS，不进入 reservation/backend/meta I/O。
    std::vector<ErrorCode> BatchSubmit(const std::string &trace_id,
                                       std::vector<MigrationRequest> requests,
                                       CopyConcurrencyLimit copy_limit = CopyConcurrencyLimit(),
                                       bool lifecycle_lock_held = false);

    // ---- 仅供测试/诊断（BUILD 通过 -fno-access-control 访问）----
    std::string GetActiveTaskDstLocation(const std::string &instance_id, int64_t block_key) const;
    void DebugInsertActiveCopyTask(const std::string &instance_id, int64_t block_key, const std::string &dst_location_id);
    void DebugEnableCopySubmissionsForTest();
    std::size_t PendingAsyncMigrationPrepareCountForTest() const;

    // 活跃 Copy 任务的收尾状态机。用于让外部 Cancel 线程与单一 monitor 完成线程在
    // task_mutex_ 内原子认领"谁负责收尾"，避免 cancel 与 promote 并发误删刚提升的目标。
    //   kPreparing  —— 已通过准入并占位，目标 location 可能尚未发布/尚未绑定 id；
    //   kPrepareCancelling —— prepare 阶段收到取消，提交线程负责停止提交并清理 reservation/目标；
    //   kRunning    —— copy 已进入提交阶段、future 未完成，正常态；
    //   kCompleting —— monitor 已认领完成（promote/失败清理进行中），此后 Cancel 太晚；
    //   kCancelling —— 外部已请求取消，收尾推迟到 future 完成时由 monitor 执行（删 WRITING 目标、不 promote）。
    enum class CopyTaskState { kPreparing, kPrepareCancelling, kRunning, kCompleting, kCancelling };

    // 单个活跃 Copy 任务的上下文。
    struct CopyTaskContext {
        std::string instance_group_name;
        std::string instance_id;
        int64_t block_key = 0;
        std::string src_location_id;
        int64_t src_create_time = 0; // 提交时源 location 的 create_time，OnTaskSuccess 比对以防 id 复用
        std::string src_storage_name;
        std::string dst_storage_name;
        std::string dst_location_id;
        MigrationRetention retention = MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE;
        std::chrono::steady_clock::time_point submit_time;
        uint64_t total_bytes = 0;              // 源端各 spec 字节数之和（取自 uri size 参数）
        std::string mark_target;               // 提交时 mark 的 target（空=无 mark，用于 match-clear）
        int64_t mark_deadline_ms = 0;          // 提交时 mark 的 deadline（用于 match-clear）
        CopyTaskState state = CopyTaskState::kRunning; // 收尾认领状态
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
                              MigrationRequest &request,
                              CopyTaskContext &out_ctx,
                              std::vector<DataStorageUri> &out_src_uris,
                              std::vector<DataStorageUri> &out_dst_uris);

    // 非阻塞提交目标 location 的精确删除任务（失败 / 取消时清理半成品）。返回不代表删除已完成。
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
    bool ClearTieredWriteMarkIfMatchInternal(const std::string &instance_id,
                                             int64_t block_key,
                                             const std::string &expected_target,
                                             int64_t expected_deadline_ms,
                                             bool is_expiry = false);
    void EnqueueMarkExpiry(const std::string &instance_id,
                           int64_t block_key,
                           const std::string &target_storage,
                           int64_t deadline_ms);
    void ProcessExpiredMarks();
    size_t ActiveTaskCountUnsafe() const; // 调用方持有 task_mutex_
    size_t ActiveTaskCountForGroupUnsafe(const std::string &instance_group_name) const; // 调用方持有 task_mutex_

    // ---- 活跃任务表操作收口（均要求调用方持有 task_mutex_）----
    // 统一处理"内层 map 空则 erase 外层 + 更新 active gauge"，避免各业务路径重复手写、遗漏。
    // 业务语义的 stats/event 仍由各调用方按转换语义各自发出。
    // 插入成功返回 true；若该 (instance, block) 已存在活跃任务则不插入并返回 false（调用方据此回滚）。
    bool InsertActiveTaskLocked(CopyTaskContext ctx);
    // 原子完成 (instance, block) 存在性检查与 preparing 占位；占位会立即参与去重、并发预算和
    // Reclaimer active-target 判断。调用方不得把 check/insert 拆成两个临界区。
    bool ReservePreparingTaskLocked(const MigrationRequest &request);
    // 将 PrepareCopyTask 产出的完整上下文（尤其 dst_location_id）绑定到既有 preparing 占位；
    // 不改变状态。仅当对应 entry 仍为 kPreparing 时成功。
    bool UpdatePreparingTaskLocked(const CopyTaskContext &ctx);
    // kPreparing -> kRunning；其他状态一律失败，避免覆盖并发状态转换。
    bool MarkTaskRunningLocked(const std::string &instance_id, int64_t block_key);
    // 仅移除仍由提交线程持有的 kPreparing/kPrepareCancelling entry，避免失败清理误删已进入
    // 运行/收尾阶段的任务。
    bool RemovePreparingTaskLocked(const std::string &instance_id, int64_t block_key);
    // 移除 (instance, block) 的活跃任务；不存在返回 false。
    bool RemoveActiveTaskLocked(const std::string &instance_id, int64_t block_key);
    // 仅判断 (instance, block) 是否已有活跃任务（存在性查询，不拷贝 ctx）。
    bool HasActiveTaskLocked(const std::string &instance_id, int64_t block_key) const;

    // ---- 收尾认领（均要求调用方持有 task_mutex_）----
    enum class ClaimResult { kClaimedRunning, kWasCancelling, kBusyPreparing, kBusyCompleting, kNotFound };
    // monitor 完成路径认领：kRunning→置 kCompleting 并拷 ctx（kClaimedRunning）；
    // kCancelling→拷 ctx 返回 kWasCancelling（不改状态，交由 CompleteCancelledTask 收尾）。
    ClaimResult ClaimForCompletionLocked(const std::string &instance_id, int64_t block_key, CopyTaskContext &out_ctx);
    enum class CancelResult { kMarked, kMarkedPreparing, kAlreadyCancelling, kBusyCompleting, kNotFound };
    // 外部 Cancel 认领：kPreparing→kPrepareCancelling（由提交线程清理）；
    // kRunning→kCancelling（future 完成后清理）；两个 cancelling 状态均幂等；kCompleting→太晚。
    CancelResult MarkCancellingLocked(const std::string &instance_id, int64_t block_key);
    // 取消任务的延迟收尾（monitor 线程，入口不持锁）：删 WRITING 目标（源不动）+ 移除活跃任务
    // + cancelled 终态 metric/event/log。仅由 OnTaskSuccess/OnTaskFailed 在认领到 kWasCancelling 时调用。
    void CompleteCancelledTask(const CopyTaskContext &ctx);

    struct AsyncMigrationPrepareKey {
        std::uint64_t generation = 0;
        std::string instance_group_name;
        std::string instance_id;
        std::string source_storage_name;
        std::string target_storage_name;

        bool operator==(const AsyncMigrationPrepareKey &other) const {
            return generation == other.generation && instance_group_name == other.instance_group_name &&
                   instance_id == other.instance_id && source_storage_name == other.source_storage_name &&
                   target_storage_name == other.target_storage_name;
        }
    };
    struct AsyncMigrationPrepareKeyHash {
        std::size_t operator()(const AsyncMigrationPrepareKey &key) const;
    };

    void RunAsyncMigrationPrepare(AsyncMigrationPrepareJob job, std::uint64_t generation);
    void FinishAsyncMigrationPrepare(const AsyncMigrationPrepareKey &key);
    DispatchBatchResult DispatchMigrationBatchWithLifecycleLockHeld(const std::string &trace_id,
                                                                    const std::string &instance_id,
                                                                    const std::string &src_name,
                                                                    const std::string &dst_name,
                                                                    const std::vector<int64_t> &batch,
                                                                    const std::vector<CacheLocationMap> &loc_maps,
                                                                    const DispatchBatchParams &params);
    ErrorCode CheckTargetStorageAdmission(const std::string &trace_id,
                                          const std::string &instance_group_name,
                                          const std::string &instance_id,
                                          const std::string &target_storage_name) const;

    std::shared_ptr<SchedulePlanExecutor> schedule_plan_executor_;
    std::shared_ptr<MetaIndexerManager> meta_indexer_manager_;
    std::shared_ptr<DataStorageManager> data_storage_manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<EventManager> event_manager_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<DataStorageSelector> data_storage_selector_;
    bool metrics_enabled_ = false;

    // 活跃 Copy 任务表。
    mutable std::mutex task_mutex_;
    std::unordered_map<std::string, std::unordered_map<int64_t, CopyTaskContext>> active_tasks_by_instance_;

    // 监控线程待处理队列。
    std::mutex pending_mutex_;
    std::condition_variable pending_cv_;
    std::deque<PendingCopy> pending_copies_;

    std::mutex mark_expiry_mutex_;
    std::priority_queue<ExpiringMark, std::vector<ExpiringMark>, ExpiringMarkGreater> mark_expiry_queue_;

    // 线程与生命周期。
    std::thread monitor_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> accepting_copy_submissions_{false};
    // Submit/BatchSubmit 全程持 shared lock，彼此可并行；Stop 持 unique lock，精确等待
    // 已准入的提交函数完成后再停止 monitor/清 active table。锁序固定为：
    // copy_submission_lifecycle_mutex_ -> copy_submission_mutex_ -> task_mutex_。
    std::shared_mutex copy_submission_lifecycle_mutex_;
    // 短准入锁：仅保护 accepting/draining 检查与 preparing reservation，不覆盖 backend/meta I/O。
    std::mutex copy_submission_mutex_;
    // draining 中的 instance（copy_submission_mutex_ 保护）。非空即拒绝该 instance 的新提交。
    std::unordered_set<std::string> draining_instances_;

    // 异步 Prepare 仅记录轻量 route key；不持有 Reclaimer 指针。Stop 递增 generation 使
    // 已排队的旧 leader Job 自动失效，避免切主后执行陈旧策略。
    static constexpr std::size_t kMaxPendingAsyncMigrationPrepareJobs = 1024;
    mutable std::mutex async_prepare_mutex_;
    std::unordered_set<AsyncMigrationPrepareKey, AsyncMigrationPrepareKeyHash> pending_async_prepare_jobs_;
    std::atomic<std::uint64_t> async_prepare_generation_{0};

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
    Counter m_tasks_completed_cancelled_; // cancelled 终态，与 success/failed 对称
    Gauge m_tasks_active_;
    Counter m_copy_bytes_total_;
    Gauge m_copy_duration_ms_;
    Gauge m_marks_active_;
    Counter m_marks_consumed_total_;
    Counter m_marks_expired_total_; // 超时过期清除的 mark（与 consumed 区分：浪费 vs 有效消费）
    Counter m_mark_query_errors_total_;

    void UpdateActiveTasksGauge(); // 调用方持有 task_mutex_
    void UpdateMarksActiveGauge(); // best-effort：基于 added-cleared 原子计数
};

} // namespace kv_cache_manager
