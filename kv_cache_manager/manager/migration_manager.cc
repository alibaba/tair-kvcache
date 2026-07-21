#include "kv_cache_manager/manager/migration_manager.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <limits>
#include <optional>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/event/spec_events/migration_event.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"

namespace kv_cache_manager {

namespace {
constexpr auto kMonitorIdleSleep = std::chrono::milliseconds(50);
constexpr auto kFutureWaitTime = std::chrono::microseconds(200);

std::optional<int64_t> ParsePositiveInt64(const std::string &value) {
    if (value.empty()) {
        return std::nullopt;
    }
    int64_t parsed = 0;
    const char *begin = value.data();
    const char *end = begin + value.size();
    const auto [ptr, ec] = std::from_chars(begin, end, parsed);
    if (ec != std::errc{} || ptr != end || parsed <= 0) {
        return std::nullopt;
    }
    return parsed;
}

// mark property 解析结果。target 为空表示无标记或已清；target 非空时 deadline
// 必须是合法正整数，malformed 表示缺失、非数字、越界或非正值；expired 表示已过期待清理。
struct MarkInfo {
    std::string target;
    int64_t deadline_ms = 0;
    bool malformed = false;
    bool expired = false;
};

static MarkInfo ParseMarkFromProperties(const PropertyMap &props, int64_t now_ms) {
    MarkInfo info;
    auto tit = props.find(MigrationManager::PROPERTY_TIERED_WRITE_TARGET);
    if (tit == props.end() || tit->second.empty()) {
        return info;
    }
    info.target = tit->second;
    auto dit = props.find(MigrationManager::PROPERTY_TIERED_WRITE_DEADLINE_MS);
    const auto deadline = dit == props.end() ? std::nullopt : ParsePositiveInt64(dit->second);
    if (!deadline.has_value()) {
        info.malformed = true;
        return info;
    }
    info.deadline_ms = *deadline;
    if (info.deadline_ms <= now_ms) {
        info.expired = true;
    }
    return info;
}

// 收集目标 storage 上指定 status 的 location 联合覆盖的 spec name 集合。
// 一次 O(L·S) 扫描替代旧的 per-location 全覆盖判断（O(L²·S²)）。
std::unordered_set<std::string> CollectCoveredSpecNames(
    const CacheLocationMap &loc_map,
    const std::string &storage_name,
    std::initializer_list<CacheLocationStatus> statuses) {
    std::unordered_set<std::string> covered;
    for (const auto &[_, loc_ptr] : loc_map) {
        if (!loc_ptr || std::find(statuses.begin(), statuses.end(), loc_ptr->status()) == statuses.end()) {
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

std::vector<const CacheLocation *> FindLocationsOnStorage(const CacheLocationMap &loc_map,
                                                          const std::string &storage_name,
                                                          std::initializer_list<CacheLocationStatus> statuses) {
    std::vector<const CacheLocation *> locations;
    for (const auto &[_, loc_ptr] : loc_map) {
        if (!loc_ptr || std::find(statuses.begin(), statuses.end(), loc_ptr->status()) == statuses.end()) {
            continue;
        }
        for (const auto &spec : loc_ptr->location_specs()) {
            if (DataStorageUri uri(spec.uri()); uri.Valid() && uri.GetHostName() == storage_name) {
                locations.push_back(loc_ptr.get());
                break;
            }
        }
    }
    return locations;
}
} // namespace

// Mark 持久化属性名（block 级 property）。带 inner 前缀避免与业务属性冲突。
const std::string MigrationManager::PROPERTY_TIERED_WRITE_TARGET = "__mig_tier_target__";
const std::string MigrationManager::PROPERTY_TIERED_WRITE_DEADLINE_MS = "__mig_tier_deadline_ms__";

MigrationManager::MigrationManager(std::shared_ptr<SchedulePlanExecutor> schedule_plan_executor,
                                   std::shared_ptr<MetaIndexerManager> meta_indexer_manager,
                                   std::shared_ptr<DataStorageManager> data_storage_manager,
                                   std::shared_ptr<MetricsRegistry> metrics_registry,
                                   std::shared_ptr<EventManager> event_manager,
                                   std::shared_ptr<RegistryManager> registry_manager)
    : schedule_plan_executor_(std::move(schedule_plan_executor))
    , meta_indexer_manager_(std::move(meta_indexer_manager))
    , data_storage_manager_(std::move(data_storage_manager))
    , metrics_registry_(std::move(metrics_registry))
    , event_manager_(std::move(event_manager))
    , registry_manager_(std::move(registry_manager)) {
    if (metrics_registry_ != nullptr) {
        metrics_enabled_ = true;
        m_tasks_submitted_total_ = metrics_registry_->GetCounter("migration.tasks_submitted_total");
        m_tasks_completed_success_ =
            metrics_registry_->GetCounter("migration.tasks_completed_total", {{"status", "success"}});
        m_tasks_completed_failed_ =
            metrics_registry_->GetCounter("migration.tasks_completed_total", {{"status", "failed"}});
        m_tasks_completed_cancelled_ =
            metrics_registry_->GetCounter("migration.tasks_completed_total", {{"status", "cancelled"}});
        m_tasks_active_ = metrics_registry_->GetGauge("migration.tasks_active");
        m_copy_bytes_total_ = metrics_registry_->GetCounter("migration.copy_bytes_total");
        m_copy_duration_ms_ = metrics_registry_->GetGauge("migration.copy_duration_ms");
        m_marks_active_ = metrics_registry_->GetGauge("migration.marks_active");
        m_marks_consumed_total_ = metrics_registry_->GetCounter("migration.marks_consumed_total");
        m_marks_expired_total_ = metrics_registry_->GetCounter("migration.marks_expired_total");
        m_mark_query_errors_total_ = metrics_registry_->GetCounter("migration.mark_query_errors_total");
    }
}

void MigrationManager::UpdateActiveTasksGauge() {
    if (metrics_enabled_) {
        m_tasks_active_ = static_cast<double>(ActiveTaskCountUnsafe());
    }
}

void MigrationManager::UpdateMarksActiveGauge() {
    if (metrics_enabled_) {
        // best-effort：持久化方案下无内存表，近似为 added - cleared（不计随 block 回收）。
        const int64_t added = static_cast<int64_t>(stat_marks_added_.load(std::memory_order_relaxed));
        const int64_t cleared = static_cast<int64_t>(stat_marks_cleared_.load(std::memory_order_relaxed));
        m_marks_active_ = static_cast<double>(added > cleared ? added - cleared : 0);
    }
}

MigrationManager::~MigrationManager() { Stop(); }

void MigrationManager::Start() {
    std::unique_lock<std::shared_mutex> lifecycle_lock(copy_submission_lifecycle_mutex_);
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return; // already running
    }
    async_prepare_generation_.fetch_add(1, std::memory_order_acq_rel);
    accepting_copy_submissions_.store(true, std::memory_order_release);
    monitor_thread_ = std::thread([this]() { MonitorLoop(); });
    KVCM_LOG_INFO("MigrationManager started");
}

void MigrationManager::Stop() {
    // 先关全局 gate，后来的 Submit 可在获取 lifecycle shared lock 前快速失败；随后用 unique lock
    // 等待已经持有 shared lock 的提交函数完整返回。该 barrier 只等待 prepare/enqueue，不等待
    // executor 中已异步运行的数据 Copy。
    accepting_copy_submissions_.store(false, std::memory_order_release);
    async_prepare_generation_.fetch_add(1, std::memory_order_acq_rel);
    {
        std::lock_guard<std::mutex> lock(async_prepare_mutex_);
        pending_async_prepare_jobs_.clear();
    }
    std::unique_lock<std::shared_mutex> lifecycle_lock(copy_submission_lifecycle_mutex_);

    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
        return; // not running
    }
    pending_cv_.notify_all();
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
    // 退出时丢弃活跃任务表：futures 已随 MonitorLoop 退出释放，不再有人驱动 OnTaskSuccess/OnTaskFailed，
    // 残留条目会让 HasMigrationTask/ActiveTaskCount/HasActiveCopyTargetLocation 永远 stale。
    // 半成品的 WRITING 目标 location 由 Reclaimer 走孤儿清理路径回收。
    size_t dropped = 0;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        dropped = ActiveTaskCountUnsafe();
        active_tasks_by_instance_.clear();
        UpdateActiveTasksGauge();
    }
    if (dropped > 0) {
        KVCM_LOG_WARN("MigrationManager stopped with %zu active copy task(s) dropped; "
                      "WRITING dst locations will be reclaimed as orphans",
                      dropped);
    } else {
        KVCM_LOG_INFO("MigrationManager stopped");
    }
}

ErrorCode MigrationManager::PrepareCopyTask(const std::string &trace_id,
                                            MigrationRequest &request,
                                            CopyTaskContext &out_ctx,
                                            std::vector<DataStorageUri> &out_src_uris,
                                            std::vector<DataStorageUri> &out_dst_uris) {
    if (request.instance_id.empty() || request.src_location_id.empty() || request.src_storage_name.empty() ||
        request.dst_storage_name.empty()) {
        KVCM_LOG_WARN("[%s] migration bad args: instance_id/src_location_id/src_storage/dst_storage required",
                      trace_id.c_str());
        return EC_BADARGS;
    }
    if (!data_storage_manager_) {
        return EC_ERROR;
    }
    auto indexer = meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(request.instance_id) : nullptr;
    if (!indexer) {
        KVCM_LOG_WARN("[%s] MetaIndexer not found for instance %s", trace_id.c_str(), request.instance_id.c_str());
        return EC_INSTANCE_NOT_EXIST;
    }

    // 1. 读取源 location specs + create_time。
    // 如果调用方已在 admission 阶段取得 src_specs，直接使用并跳过冗余 BatchGetLocation；
    // public 单条 Submit 仍允许不带快照，此时在这里重读源 location。private BatchSubmit 不走本函数。
    const std::vector<LocationSpec> *src_specs_ptr = nullptr;
    int64_t src_create_time = 0;
    if (!request.src_specs.empty()) {
        src_specs_ptr = &request.src_specs;
        src_create_time = request.src_create_time;
    } else {
        MetaSearcher meta_searcher_for_src(indexer);
        auto ctx_for_src =
            std::make_shared<RequestContext>(trace_id.empty() ? "migration_prepare" : trace_id);
        std::vector<CacheLocationMap> location_maps;
        BlockMask empty_mask;
        ErrorCode ec =
            meta_searcher_for_src.BatchGetLocation(ctx_for_src.get(), {request.block_key}, empty_mask, location_maps);
        if (ec != EC_OK || location_maps.empty()) {
            KVCM_LOG_WARN("[%s] BatchGetLocation failed for block_key %ld, ec %d",
                          trace_id.c_str(),
                          request.block_key,
                          ec);
            return EC_NOENT;
        }
        auto src_iter = location_maps[0].find(request.src_location_id);
        if (src_iter == location_maps[0].end()) {
            KVCM_LOG_WARN("[%s] source location %s not found for block_key %ld",
                          trace_id.c_str(),
                          request.src_location_id.c_str(),
                          request.block_key);
            return EC_NOENT;
        }
        if (src_iter->second == nullptr) {
            return EC_ERROR;
        }
        const CacheLocation &src_location = *src_iter->second;
        if (src_location.status() != CLS_SERVING) {
            KVCM_LOG_WARN("[%s] source location %s not SERVING (status %d), skip migration",
                          trace_id.c_str(),
                          request.src_location_id.c_str(),
                          src_location.status());
            return EC_MISMATCH;
        }
        if (src_location.location_specs().empty()) {
            KVCM_LOG_WARN("[%s] source location %s has no specs",
                          trace_id.c_str(),
                          request.src_location_id.c_str());
            return EC_ERROR;
        }
        request.src_specs = src_location.location_specs();
        request.src_create_time = src_location.create_time();
        src_specs_ptr = &request.src_specs;
        src_create_time = request.src_create_time;
    }

    const auto &src_specs = *src_specs_ptr;

    // 2. 目标 storage 类型。
    auto dst_backend = data_storage_manager_->GetDataStorageBackend(request.dst_storage_name);
    if (!dst_backend) {
        KVCM_LOG_WARN("[%s] target storage %s not found", trace_id.c_str(), request.dst_storage_name.c_str());
        return EC_NOENT;
    }
    const DataStorageType dst_type = dst_backend->GetType();

    // 3. 逐 spec 在目标 storage 预分配空间，构建目标 location specs + src/dst uri 对。
    MetaSearcher meta_searcher(indexer);
    auto ctx = std::make_shared<RequestContext>(trace_id.empty() ? "migration_prepare" : trace_id);
    std::vector<LocationSpec> dst_specs;
    dst_specs.reserve(src_specs.size());
    out_src_uris.clear();
    out_dst_uris.clear();
    std::vector<DataStorageUri> allocated_for_rollback;
    std::uint64_t total_bytes = 0;

    auto rollback = [&]() {
        if (!allocated_for_rollback.empty()) {
            data_storage_manager_->Delete(ctx.get(), request.dst_storage_name, allocated_for_rollback, nullptr);
        }
    };

    for (const auto &src_spec : src_specs) {
        DataStorageUri src_uri(src_spec.uri());
        if (!src_uri.Valid()) {
            KVCM_LOG_WARN("[%s] invalid source uri for spec %s", trace_id.c_str(), src_spec.name().c_str());
            rollback();
            return EC_ERROR;
        }
        std::uint64_t spec_size = 0;
        src_uri.GetParamAs<std::uint64_t>("size", spec_size);
        total_bytes += spec_size;

        std::string dst_key =
            request.instance_id + "/" + src_spec.name() + "/" + StringUtil::Uint64ToHex(request.block_key);
        auto create_results = data_storage_manager_->Create(
            ctx.get(), request.dst_storage_name, {dst_key}, static_cast<size_t>(spec_size), nullptr);
        if (create_results.size() != 1 || create_results[0].first != EC_OK) {
            KVCM_LOG_WARN("[%s] allocate dst space failed on %s for spec %s",
                          trace_id.c_str(),
                          request.dst_storage_name.c_str(),
                          src_spec.name().c_str());
            rollback();
            return EC_ERROR;
        }
        DataStorageUri dst_uri = create_results[0].second;
        allocated_for_rollback.push_back(dst_uri);
        out_src_uris.push_back(src_uri);
        out_dst_uris.push_back(dst_uri);
        dst_specs.emplace_back(src_spec.name(), dst_uri.ToUriString());
    }

    // 4. 建目标 location（BatchAddLocation 总是写 CLS_WRITING 并生成随机 location_id）。
    auto dst_location = std::make_shared<CacheLocation>();
    dst_location->set_type(dst_type);
    dst_location->set_spec_size(dst_specs.size());
    for (auto &spec : dst_specs) {
        dst_location->push_location_spec(std::move(spec));
    }
    std::vector<std::string> out_location_ids;
    ErrorCode ec =
        meta_searcher.BatchAddLocation(ctx.get(), {request.block_key}, {dst_location}, out_location_ids);
    if (ec != EC_OK || out_location_ids.empty() || out_location_ids[0].empty()) {
        KVCM_LOG_WARN("[%s] BatchAddLocation failed for block_key %ld, ec %d",
                      trace_id.c_str(),
                      request.block_key,
                      ec);
        rollback();
        return EC_ERROR;
    }

    out_ctx.instance_group_name = request.instance_group_name;
    out_ctx.instance_id = request.instance_id;
    out_ctx.block_key = request.block_key;
    out_ctx.src_location_id = request.src_location_id;
    out_ctx.src_create_time = src_create_time; // 记录源 location 的创建时间
    out_ctx.src_storage_name = request.src_storage_name;
    out_ctx.dst_storage_name = request.dst_storage_name;
    out_ctx.dst_location_id = out_location_ids[0];
    out_ctx.retention = request.retention;
    out_ctx.total_bytes = total_bytes;
    return EC_OK;
}

ErrorCode MigrationManager::Submit(const std::string &trace_id, MigrationRequest request) {
    auto reject_not_accepting = [&]() {
        KVCM_LOG_WARN("[%s] reject migration copy submit for instance %s block_key %ld: "
                      "manager is not accepting submissions",
                      trace_id.c_str(),
                      request.instance_id.c_str(),
                      request.block_key);
        return EC_ERROR;
    };
    // Stop 先把 accepting 置 false，再等待 lifecycle unique lock。快检避免 shutdown 期间的新请求
    // 排在 unique lock 后面；shared lock 则保证一旦进入，Stop 必须等本函数完整收口。
    if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
        return reject_not_accepting();
    }
    std::shared_lock<std::shared_mutex> lifecycle_lock(copy_submission_lifecycle_mutex_);
    {
        std::lock_guard<std::mutex> submission_lock(copy_submission_mutex_);
        // shared lock 可能排在一次 Stop 后才拿到，必须在真正准入点复查。
        if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
            return reject_not_accepting();
        }
        // instance 正在 drain（RemoveInstance 中）时拒绝新提交，避免 trim-vs-write 竞态。
        if (draining_instances_.count(request.instance_id) > 0) {
            KVCM_LOG_WARN("[%s] reject migration copy submit for draining instance %s block_key %ld",
                          trace_id.c_str(),
                          request.instance_id.c_str(),
                          request.block_key);
            return EC_ERROR;
        }

        // 在释放短准入锁、执行任何目标 I/O 前原子登记 preparing 占位。它同时用于
        // 防重复、copy 并发预算、Reclaimer 保护和 drain 快照，不能拆成 check-then-insert。
        std::lock_guard<std::mutex> lock(task_mutex_);
        if (!ReservePreparingTaskLocked(request)) {
            KVCM_LOG_INFO("[%s] instance %s block_key %ld already has an active migration task, skip",
                          trace_id.c_str(),
                          request.instance_id.c_str(),
                          request.block_key);
            return EC_EXIST;
        }
    }
    auto release_preparing = [&]() {
        std::lock_guard<std::mutex> lock(task_mutex_);
        RemovePreparingTaskLocked(request.instance_id, request.block_key);
    };

    CopyTaskContext ctx;
    std::vector<DataStorageUri> src_uris;
    std::vector<DataStorageUri> dst_uris;
    ErrorCode prepare_ec = PrepareCopyTask(trace_id, request, ctx, src_uris, dst_uris);
    if (prepare_ec != EC_OK) {
        // PrepareCopyTask 已回滚其成功分配的 URI；若 BatchAddLocation 部分成功留下 WRITING，
        // 释放 reservation 后它会成为无 copy 在跑的真实 orphan，由 Reclaimer 安全清理。
        release_preparing();
        return prepare_ec;
    }

    // BatchAddLocation 可能在返回前已经对 Reclaimer 可见；此前空 id 的 preparing 按 block 临时保护。
    // 返回后立即绑定真实 id，缩短对同 block 其他 WRITING location 的宽泛保护时间。
    bool target_bound = false;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        target_bound = UpdatePreparingTaskLocked(ctx);
    }
    if (!target_bound) {
        // Cancel 可能已把 reservation 转成 kPrepareCancelling。先在 reservation 保护下提交精确的
        // 异步目标删除，再释放未提交任务；这里保证提交顺序，不等待实际删除完成。
        SubmitTargetLocationDelete(ctx);
        release_preparing();
        return EC_ERROR;
    }
    ctx.submit_time = std::chrono::steady_clock::now();
    // 只有指向本次 Copy 目标的 mark 才能绑定到 ctx，供 OnTaskSuccess 做 match-clear。
    // 其他目标的 mark 表示独立的迁移意图，不能由本次 Copy 消费。
    {
        std::vector<MarkQueryResult> mark_snap;
        const auto mark_ec = BatchGetTieredWriteTargets(request.instance_id, {request.block_key}, mark_snap);
        if (mark_ec == EC_OK && !mark_snap.empty() && mark_snap[0].HasValidMark() &&
            mark_snap[0].target == ctx.dst_storage_name) {
            ctx.mark_target = mark_snap[0].target;
            ctx.mark_deadline_ms = mark_snap[0].deadline_ms;
        }
    }

    // future 对 monitor 可见前，先把完整快照写回并进入可认领的 kRunning。
    // Cancel 若在 executor Submit 前命中 kRunning，采用 deferred-cancel 语义：copy 完成后丢弃目标。
    bool task_running = false;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        task_running = UpdatePreparingTaskLocked(ctx) && MarkTaskRunningLocked(ctx.instance_id, ctx.block_key);
    }
    if (!task_running) {
        SubmitTargetLocationDelete(ctx);
        release_preparing();
        return EC_ERROR;
    }

    // 提交 copy 任务。
    CacheLocationCopyRequest copy_req;
    copy_req.instance_id = ctx.instance_id;
    copy_req.block_key = ctx.block_key;
    copy_req.exec_storage_name = ctx.src_storage_name; // 一期 = 源 storage
    copy_req.src_uris = std::move(src_uris);
    copy_req.dst_uris = std::move(dst_uris);

    std::future<PlanExecuteResult> future = schedule_plan_executor_->Submit(copy_req);
    if (!future.valid()) {
        KVCM_LOG_WARN("[%s] submit copy task failed for block_key %ld", trace_id.c_str(), ctx.block_key);
        SubmitTargetLocationDelete(ctx);
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            RemoveActiveTaskLocked(ctx.instance_id, ctx.block_key);
        }
        return EC_ERROR;
    }

    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_copies_.push_back(PendingCopy{ctx.instance_id, ctx.block_key, std::move(future)});
    }
    pending_cv_.notify_one();

    stat_copy_submitted_.fetch_add(1, std::memory_order_relaxed);
    if (metrics_enabled_) {
        ++m_tasks_submitted_total_;
    }
    if (event_manager_ != nullptr) {
        auto ev = std::make_shared<MigrationSubmittedEvent>(ctx.instance_id);
        ev->SetEventTriggerTime();
        ev->SetAdditionalArgs(ctx.block_key, ctx.src_storage_name, ctx.dst_storage_name, trace_id);
        event_manager_->Publish(ev);
    }
    KVCM_LOG_INFO("[%s] migration copy submitted: instance %s block_key %ld src_loc %s -> dst_storage %s (dst_loc %s)",
                  trace_id.c_str(),
                  ctx.instance_id.c_str(),
                  ctx.block_key,
                  ctx.src_location_id.c_str(),
                  ctx.dst_storage_name.c_str(),
                  ctx.dst_location_id.c_str());
    return EC_OK;
}

std::vector<ErrorCode> MigrationManager::BatchSubmit(const std::string &trace_id,
                                                     std::vector<MigrationRequest> requests,
                                                     CopyConcurrencyLimit copy_limit,
                                                     bool lifecycle_lock_held) {
    if (requests.empty()) {
        return {};
    }

    // Keep all long-lived per-request state together. The temporary vectors passed to batch APIs
    // remain below, but their compact result indexes are mapped back to these stable items immediately.
    struct BatchCopyItem {
        explicit BatchCopyItem(MigrationRequest input_request) : request(std::move(input_request)) {}

        void MarkFailed(ErrorCode ec) {
            result = ec;
            eligible = false;
        }

        MigrationRequest request;
        ErrorCode result = EC_ERROR;
        bool eligible = true;
        bool reservation_active = false;
        std::vector<DataStorageUri> src_uris;
        std::vector<DataStorageUri> dst_uris;
        std::vector<LocationSpec> dst_specs;
        std::uint64_t total_bytes = 0;
        CopyTaskContext context;
        bool target_bound = false;
    };

    std::vector<BatchCopyItem> items;
    items.reserve(requests.size());
    for (auto &request : requests) {
        items.emplace_back(std::move(request));
    }
    auto collect_results = [&items]() {
        std::vector<ErrorCode> results;
        results.reserve(items.size());
        for (const auto &item : items) {
            results.push_back(item.result);
        }
        return results;
    };

    // BatchSubmit 是 DispatchMigrationBatch 的内部 prepared-request API，不再
    // 兼容空 src_specs 后回读 meta 的第二套提交流程。整批共享的 indexer/target 均取 first_req，
    // 因此任何 item 违反同 instance/target 或缺少源快照时，都必须在 reservation/I/O 前拒绝整批。
    const auto &first_req = items.front().request;
    const bool prepared_batch = !first_req.instance_id.empty() && !first_req.dst_storage_name.empty() &&
                                std::all_of(items.begin(), items.end(), [&first_req](const BatchCopyItem &item) {
                                    const auto &request = item.request;
                                    return request.instance_id == first_req.instance_id &&
                                           request.dst_storage_name == first_req.dst_storage_name &&
                                           !request.src_location_id.empty() && !request.src_storage_name.empty() &&
                                           !request.src_specs.empty();
                                });
    if (!prepared_batch) {
        KVCM_LOG_WARN(
            "[%s] reject invalid prepared migration batch: requests must share a non-empty instance/target and "
            "carry non-empty source location/specs",
            trace_id.c_str());
        for (auto &item : items) {
            item.MarkFailed(EC_BADARGS);
        }
        return collect_results();
    }

    // Stop 的 lifecycle barrier 与 Submit 相同：快检拒绝 shutdown 后的新调用；shared lock 覆盖本函数
    // 余下生命周期，但不串行其他 Submit/BatchSubmit。
    if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
        return collect_results();
    }
    std::shared_lock<std::shared_mutex> lifecycle_lock;
    if (!lifecycle_lock_held) {
        lifecycle_lock = std::shared_lock<std::shared_mutex>(copy_submission_lifecycle_mutex_);
        if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
            return collect_results();
        }
    }

    // ---- phase 0: group 级 Copy 硬限流 + per-block dedup + draining gate + preparing reservation ----
    // 所有 eligible item 必须在释放短准入锁、执行任何 batch Create/AddLocation 前进入
    // active 表。copy_submission_mutex_ 只串行本 phase 的 gate；后续 backend/meta I/O 可跨 instance 并行。
    // MarkFailed 只收拢 result/eligible。reservation 必须等 URI/Location rollback 完成后再释放，
    // 避免失败 item 尚在清理目标资源时，同 block 的新 Submit 已经进入并与旧清理相互覆盖。
    auto release_preparing = [&](BatchCopyItem &item) {
        if (!item.reservation_active) {
            return;
        }
        std::lock_guard<std::mutex> lock(task_mutex_);
        RemovePreparingTaskLocked(item.request.instance_id, item.request.block_key);
        item.reservation_active = false;
    };
    auto release_all_preparing = [&]() {
        std::lock_guard<std::mutex> lock(task_mutex_);
        for (auto &item : items) {
            if (item.reservation_active) {
                RemovePreparingTaskLocked(item.request.instance_id, item.request.block_key);
                item.reservation_active = false;
            }
        }
    };
    {
        std::lock_guard<std::mutex> submission_lock(copy_submission_mutex_);
        if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
            return collect_results();
        }
        std::lock_guard<std::mutex> lock(task_mutex_);
        std::size_t available_group_slots = SIZE_MAX;
        if (copy_limit.enabled()) {
            const auto active = ActiveTaskCountForGroupUnsafe(copy_limit.instance_group_name);
            available_group_slots = copy_limit.max_concurrency > active ? copy_limit.max_concurrency - active : 0;
        }
        for (auto &item : items) {
            auto &request = item.request;
            // instance 正在 drain 时拒绝（覆盖 reclaimer + admin 两路，与 Submit 一致）。
            if (draining_instances_.count(request.instance_id) > 0) {
                item.MarkFailed(EC_ERROR);
                continue;
            }
            if (copy_limit.enabled() && request.instance_group_name != copy_limit.instance_group_name) {
                item.MarkFailed(EC_BADARGS);
                continue;
            }
            // 原子 reserve 同时挡住并发重复和同一输入 batch 内的重复 block；逐项失败不影响其他 item。
            if (!ReservePreparingTaskLocked(request)) {
                item.MarkFailed(EC_EXIST);
                continue;
            }
            // 在同一个 task_mutex_ 临界区内先 reserve 再检查剩余 slot：这样即使 slot 已满，
            // 同 block 重复项仍稳定返回 EC_EXIST；非重复项则立即回滚临时 reservation。
            if (copy_limit.enabled() && available_group_slots == 0) {
                RemovePreparingTaskLocked(request.instance_id, request.block_key);
                item.MarkFailed(EC_OUT_OF_LIMIT);
                continue;
            }
            item.reservation_active = true;
            if (copy_limit.enabled()) {
                --available_group_slots;
            }
        }
    }

    // ---- phase 1: batch Create (按 size 分组) ----
    // 收集 eligible requests 的 src_specs → 构建 dst_key 列表 → 按 (dst_storage, size) 分组 batch Create。
    // 每个 item 的 src_uris/dst_uris 按原 request 的 spec 顺序保存。
    auto indexer = meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(first_req.instance_id) : nullptr;
    if (!indexer) {
        release_all_preparing();
        return collect_results();
    }
    auto dst_backend = data_storage_manager_ ? data_storage_manager_->GetDataStorageBackend(first_req.dst_storage_name)
                                             : nullptr;
    if (!dst_backend) {
        release_all_preparing();
        return collect_results();
    }
    const DataStorageType dst_type = dst_backend->GetType();

    // 按 (size) 分组：group_key=size → {(item_idx, spec_idx, dst_key, src_uri)}
    struct CreateEntry {
        std::size_t item_idx;
        std::size_t spec_idx;
        std::string dst_key;
        DataStorageUri src_uri;
    };
    std::unordered_map<std::uint64_t, std::vector<CreateEntry>> create_groups;

    for (std::size_t i = 0; i < items.size(); ++i) {
        auto &item = items[i];
        if (!item.eligible) {
            continue;
        }
        auto &req = item.request;
        for (std::size_t s = 0; s < req.src_specs.size(); ++s) {
            DataStorageUri src_uri(req.src_specs[s].uri());
            if (!src_uri.Valid()) {
                item.MarkFailed(EC_ERROR);
                break;
            }
            std::uint64_t spec_size = 0;
            src_uri.GetParamAs<std::uint64_t>("size", spec_size);
            item.total_bytes += spec_size;
            std::string dst_key = req.instance_id + "/" + req.src_specs[s].name() + "/" +
                                  StringUtil::Uint64ToHex(req.block_key);
            create_groups[spec_size].push_back({i, s, std::move(dst_key), std::move(src_uri)});
        }
    }

    // 执行 batch Create + 回填 per-block URI
    auto batch_ctx = std::make_shared<RequestContext>(trace_id.empty() ? "migration_batch_prepare" : trace_id);
    // 异构 size 的 spec 分散在不同 create_group；某 group 使一个 block ineligible 后，
    // 之后处理的 group 里同 block 的 spec 可能已被 batch Create 成功分配 URI，却在下面
    // `!item.eligible` 分支被跳过、不进 item.dst_uris，导致后续 rollback 删不到 → 永久 orphan
    // （无 CacheLocation meta，reclaimer 也 GC 不到）。Create 返回 shape 异常时，整组返回的
    // 成功 URI 同样不能被采用。统一收集这些无 owner 的 URI，在 group 循环后删除。
    std::vector<DataStorageUri> unowned_created_uris;
    for (auto &[size, entries] : create_groups) {
        std::vector<std::string> keys;
        keys.reserve(entries.size());
        for (const auto &e : entries) {
            keys.push_back(e.dst_key);
        }
        auto create_results =
            data_storage_manager_->Create(batch_ctx.get(), first_req.dst_storage_name, keys, size, nullptr);
        if (create_results.size() != entries.size()) {
            KVCM_LOG_WARN(
                "[%s] batch Create returned unexpected result count on storage %s for size %llu: expected %zu, got "
                "%zu; rolling back the entire group",
                trace_id.c_str(),
                first_req.dst_storage_name.c_str(),
                static_cast<unsigned long long>(size),
                entries.size(),
                create_results.size());
            // 位置映射以 vector 下标为契约；shape 一旦异常，连前 min(N, M) 项也不能再可信地
            // 绑定到 request。删除本组实际返回的全部成功 URI，并将本组 request 整体置失败。
            for (const auto &[ec, uri] : create_results) {
                if (ec == EC_OK) {
                    unowned_created_uris.push_back(uri);
                }
            }
            for (const auto &entry : entries) {
                items[entry.item_idx].MarkFailed(EC_ERROR);
            }
            continue;
        }
        for (std::size_t j = 0; j < entries.size(); ++j) {
            const auto &e = entries[j];
            auto &item = items[e.item_idx];
            if (!item.eligible) {
                // 该 block 已被其他 spec 的失败标记 ineligible；本 spec 若已成功分配，
                // 需补删（否则永不进 dst_uris，rollback 漏删）。
                if (create_results[j].first == EC_OK) {
                    unowned_created_uris.push_back(create_results[j].second);
                }
                continue;
            }
            if (create_results[j].first == EC_OK) {
                item.src_uris.push_back(e.src_uri);
                item.dst_uris.push_back(create_results[j].second);
                item.dst_specs.emplace_back(
                    item.request.src_specs[e.spec_idx].name(), create_results[j].second.ToUriString());
            } else {
                item.MarkFailed(EC_ERROR);
            }
        }
    }
    if (!unowned_created_uris.empty()) {
        data_storage_manager_->Delete(batch_ctx.get(), first_req.dst_storage_name, unowned_created_uris, nullptr);
    }

    // 回滚 Create 部分失败的 block 的已分配 URI
    for (auto &item : items) {
        if (item.eligible) {
            continue;
        }
        if (!item.dst_uris.empty()) {
            data_storage_manager_->Delete(batch_ctx.get(), item.request.dst_storage_name, item.dst_uris, nullptr);
            item.dst_uris.clear();
            item.src_uris.clear();
            item.dst_specs.clear();
        }
        release_preparing(item);
    }

    // 收集成功 Create 的 block → batch AddLocation
    // add_items[k] 显式对应 add_block_keys[k]/add_locations[k]，并用于接回 location_ids[k]。
    std::vector<BatchCopyItem *> add_items;
    std::vector<int64_t> add_block_keys;
    CacheLocationVector add_locations;
    for (auto &item : items) {
        if (!item.eligible) {
            continue;
        }
        if (item.dst_specs.size() != item.request.src_specs.size()) {
            // spec 数不对（Create 返回的结果数和请求数不匹配）
            if (!item.dst_uris.empty()) {
                data_storage_manager_->Delete(batch_ctx.get(), item.request.dst_storage_name, item.dst_uris, nullptr);
            }
            item.MarkFailed(EC_ERROR);
            release_preparing(item);
            continue;
        }
        auto dst_loc = std::make_shared<CacheLocation>();
        dst_loc->set_type(dst_type);
        dst_loc->set_spec_size(item.dst_specs.size());
        for (auto &spec : item.dst_specs) {
            dst_loc->push_location_spec(std::move(spec));
        }
        add_items.push_back(&item);
        add_block_keys.push_back(item.request.block_key);
        add_locations.push_back(std::move(dst_loc));
    }

    // ---- phase 2+3: batch AddLocation + bind preparing reservations + executor Submit ----
    if (!add_block_keys.empty()) {
        MetaSearcher meta_searcher(indexer);
        std::vector<std::string> location_ids;
        ErrorCode add_ec =
            meta_searcher.BatchAddLocation(batch_ctx.get(), add_block_keys, add_locations, location_ids);
        if (add_ec != EC_OK) {
            // FIXME: 与正常写路径（cache_manager.cc:858）相同的已知局限——BatchAddLocation 部分成功时
            // 无法精确识别哪些 block 的 meta 已写入，可能留下孤儿 CLS_WRITING location，由 reclaimer
            // 孤儿检测被动清理。Delete 所有已分配 URI 作为 best-effort 回滚。
            for (auto *item : add_items) {
                if (!item->dst_uris.empty()) {
                    data_storage_manager_->Delete(
                        batch_ctx.get(), item->request.dst_storage_name, item->dst_uris, nullptr);
                }
                item->MarkFailed(EC_ERROR);
            }
            // 部分成功但未返回 id 的 WRITING 会在释放后成为真实 orphan；此时没有 copy 被提交。
            release_all_preparing();
            return collect_results();
        }

        // ---- phase 3a: 先为整个批次绑定 location_id，再做任何逐项 mark 查询/submit ----
        // BatchAddLocation 一次公开全部 WRITING；若仍在逐项循环里绑定，后半批会继续暴露宽泛窗口。
        std::vector<BatchCopyItem *> bind_items;
        bind_items.reserve(add_items.size());
        for (std::size_t k = 0; k < add_items.size(); ++k) {
            auto &item = *add_items[k];
            if (!item.eligible || k >= location_ids.size() || location_ids[k].empty()) {
                // 无法识别目标 location id，不能提交 copy。URI 先 best-effort 删除；若 meta 已部分写入，
                // reservation 释放后由 Reclaimer 清理残留 WRITING metadata。
                if (!item.dst_uris.empty()) {
                    data_storage_manager_->Delete(
                        batch_ctx.get(), item.request.dst_storage_name, item.dst_uris, nullptr);
                    item.dst_uris.clear();
                }
                item.MarkFailed(EC_ERROR);
                release_preparing(item);
                continue;
            }
            auto &req = item.request;
            CopyTaskContext &ctx = item.context;
            ctx.instance_group_name = req.instance_group_name;
            ctx.instance_id = req.instance_id;
            ctx.block_key = req.block_key;
            ctx.src_location_id = req.src_location_id;
            ctx.src_create_time = req.src_create_time;
            ctx.src_storage_name = req.src_storage_name;
            ctx.dst_storage_name = req.dst_storage_name;
            ctx.dst_location_id = location_ids[k];
            ctx.retention = req.retention;
            ctx.total_bytes = item.total_bytes;
            bind_items.push_back(&item);
        }
        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            for (auto *item : bind_items) {
                // UpdatePreparingTaskLocked copies from const&, so item.context stays available for executor
                // submission, target cleanup and event fields after the active reservation is updated.
                item->target_bound = UpdatePreparingTaskLocked(item->context);
            }
        }
        // 与单条 Submit 保持相同边界：phase-3 失败只保证先提交基于精确 location id 的异步删除，
        // 再释放 reservation；逐项 ownership 收拢不把该既有异步清理改成同步等待。
        for (auto *item : bind_items) {
            if (item->target_bound) {
                continue;
            }
            SubmitTargetLocationDelete(item->context);
            item->MarkFailed(EC_ERROR);
            release_preparing(*item);
        }

        // ---- phase 3b: 逐项补齐提交快照、转 kRunning、提交 executor ----
        for (auto *item : bind_items) {
            if (!item->eligible || !item->target_bound) {
                continue;
            }
            auto &req = item->request;
            CopyTaskContext &ctx = item->context;
            ctx.submit_time = std::chrono::steady_clock::now();
            // 仅绑定与本次 Copy 目标一致的 mark；其他目标的 mark 必须保留。
            {
                std::vector<MarkQueryResult> mark_snap;
                const auto mark_ec = BatchGetTieredWriteTargets(req.instance_id, {req.block_key}, mark_snap);
                if (mark_ec == EC_OK && !mark_snap.empty() && mark_snap[0].HasValidMark() &&
                    mark_snap[0].target == ctx.dst_storage_name) {
                    ctx.mark_target = mark_snap[0].target;
                    ctx.mark_deadline_ms = mark_snap[0].deadline_ms;
                }
            }

            bool task_running = false;
            {
                std::lock_guard<std::mutex> lock(task_mutex_);
                task_running = UpdatePreparingTaskLocked(ctx) && MarkTaskRunningLocked(ctx.instance_id, ctx.block_key);
                if (task_running) {
                    item->reservation_active = false; // 已成为正式 active task，不能再由 preparing guard 清理
                }
            }
            if (!task_running) {
                SubmitTargetLocationDelete(ctx);
                item->MarkFailed(EC_ERROR);
                release_preparing(*item);
                continue;
            }

            CacheLocationCopyRequest copy_req;
            copy_req.instance_id = ctx.instance_id;
            copy_req.block_key = ctx.block_key;
            copy_req.exec_storage_name = ctx.src_storage_name;
            copy_req.src_uris = std::move(item->src_uris);
            copy_req.dst_uris = std::move(item->dst_uris);

            std::future<PlanExecuteResult> future = schedule_plan_executor_->Submit(copy_req);
            if (!future.valid()) {
                SubmitTargetLocationDelete(ctx);
                std::lock_guard<std::mutex> lock(task_mutex_);
                RemoveActiveTaskLocked(ctx.instance_id, ctx.block_key);
                item->MarkFailed(EC_ERROR);
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(pending_mutex_);
                pending_copies_.push_back(PendingCopy{ctx.instance_id, ctx.block_key, std::move(future)});
            }
            pending_cv_.notify_one();

            stat_copy_submitted_.fetch_add(1, std::memory_order_relaxed);
            if (metrics_enabled_) {
                ++m_tasks_submitted_total_;
            }
            if (event_manager_ != nullptr) {
                auto ev = std::make_shared<MigrationSubmittedEvent>(ctx.instance_id);
                ev->SetEventTriggerTime();
                ev->SetAdditionalArgs(ctx.block_key, ctx.src_storage_name, ctx.dst_storage_name, trace_id);
                event_manager_->Publish(ev);
            }
            item->result = EC_OK;
        }
    } // if (!add_block_keys.empty())

    // 防御性收口：正常路径此时只剩已转 kRunning 的 task（reservation_active=false）。任何遗漏的
    // preparing item 都不应泄漏到函数返回之后。
    release_all_preparing();
    return collect_results();
}

bool MigrationManager::IsSourceLocationServing(const CopyTaskContext &ctx) const {
    auto indexer = meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(ctx.instance_id) : nullptr;
    if (!indexer) {
        return false;
    }

    MetaSearcher meta_searcher(indexer);
    auto rc = std::make_shared<RequestContext>("migration_check_source");
    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    ErrorCode ec = meta_searcher.BatchGetLocation(rc.get(), {ctx.block_key}, empty_mask, location_maps);
    if (ec != EC_OK || location_maps.empty()) {
        return false;
    }
    auto iter = location_maps[0].find(ctx.src_location_id);
    if (iter == location_maps[0].end() || iter->second == nullptr) {
        return false;
    }
    // id + status + create_time 三者同时匹配，防止 id 复用导致误判新 location 为原始源。
    return iter->second->status() == CLS_SERVING &&
           iter->second->create_time() == ctx.src_create_time;
}

void MigrationManager::CompleteCopyTaskAsFailed(const CopyTaskContext &ctx, const std::string &fail_reason) {
    SubmitTargetLocationDelete(ctx);
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        RemoveActiveTaskLocked(ctx.instance_id, ctx.block_key);
    }
    stat_copy_failed_.fetch_add(1, std::memory_order_relaxed);
    // 失败路径也填真实 duration（提交到失败的耗时），而非硬编码 0，
    // 便于区分"快速失败"与"跑很久才失败"。
    const int64_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - ctx.submit_time)
                                    .count();
    if (metrics_enabled_) {
        ++m_tasks_completed_failed_;
    }
    if (event_manager_ != nullptr) {
        auto ev = std::make_shared<MigrationCompletedEvent>(ctx.instance_id);
        ev->SetEventTriggerTime();
        ev->SetAdditionalArgs(ctx.block_key,
                              ctx.src_storage_name,
                              ctx.dst_storage_name,
                              duration_ms,
                              ctx.total_bytes,
                              false,
                              fail_reason);
        event_manager_->Publish(ev);
    }
}

void MigrationManager::OnTaskSuccess(const std::string &instance_id, int64_t block_key) {
    CopyTaskContext ctx;
    ClaimResult claim;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        claim = ClaimForCompletionLocked(instance_id, block_key, ctx);
    }
    if (claim == ClaimResult::kNotFound || claim == ClaimResult::kBusyCompleting) {
        return; // 已处理过 / 完成认领中
    }
    if (claim == ClaimResult::kBusyPreparing) {
        KVCM_LOG_WARN("ignore premature migration success callback for preparing task: instance %s block_key %ld",
                      instance_id.c_str(),
                      block_key);
        return;
    }
    if (claim == ClaimResult::kWasCancelling) {
        // 用户已取消。copy 虽成功也丢弃：不 promote、不删源，删掉仍为 WRITING 的目标半成品。
        CompleteCancelledTask(ctx);
        return;
    }
    // kClaimedRunning：状态已在锁内置 kCompleting，并发 Cancel 会走 busy 分支。

    if (!IsSourceLocationServing(ctx)) {
        KVCM_LOG_WARN("migration source lost, block_key %ld src_loc %s, discard dst_loc %s",
                      block_key,
                      ctx.src_location_id.c_str(),
                      ctx.dst_location_id.c_str());
        CompleteCopyTaskAsFailed(ctx, "source_lost");
        return;
    }

    // CAS 目标 location WRITING -> SERVING。
    auto indexer = meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(ctx.instance_id) : nullptr;
    bool promoted = false;
    if (indexer) {
        MetaSearcher meta_searcher(indexer);
        auto rc = std::make_shared<RequestContext>("migration_on_success");
        std::vector<std::vector<MetaSearcher::LocationCASTask>> cas_tasks{
            {MetaSearcher::LocationCASTask{ctx.dst_location_id, CLS_WRITING, CLS_SERVING}}};
        std::vector<std::vector<ErrorCode>> cas_results;
        ErrorCode ec = meta_searcher.BatchCASLocationStatus(rc.get(), {block_key}, cas_tasks, cas_results);
        promoted = (ec == EC_OK && !cas_results.empty() && !cas_results[0].empty() && cas_results[0][0] == EC_OK);
    }

    if (!promoted) {
        // 目标提升失败：清理目标半成品，源端保持不动（数据未受损），按失败收尾。
        KVCM_LOG_WARN("migration promote dst location failed, block_key %ld dst_loc %s, treat as failed",
                      block_key,
                      ctx.dst_location_id.c_str());
        CompleteCopyTaskAsFailed(ctx, "promote_failed");
        return;
    }

    // 按提交时快照的 mark target/deadline 做条件清除，避免清掉后续同 block 新 mark。
    if (!ctx.mark_target.empty()) {
        ClearTieredWriteMarkIfMatchInternal(ctx.instance_id, block_key, ctx.mark_target, ctx.mark_deadline_ms);
    }

    // 按 retention 处理源端。
    if (ctx.retention == MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE) {
        SubmitSourceLocationDelete(ctx);
    }

    // 最后移除活跃任务。
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        RemoveActiveTaskLocked(ctx.instance_id, block_key);
    }
    stat_copy_completed_.fetch_add(1, std::memory_order_relaxed);
    const int64_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - ctx.submit_time)
                                    .count();
    if (metrics_enabled_) {
        ++m_tasks_completed_success_;
        m_copy_bytes_total_ += ctx.total_bytes;
        m_copy_duration_ms_ = static_cast<double>(duration_ms);
    }
    if (event_manager_ != nullptr) {
        auto ev = std::make_shared<MigrationCompletedEvent>(ctx.instance_id);
        ev->SetEventTriggerTime();
        ev->SetAdditionalArgs(
            block_key, ctx.src_storage_name, ctx.dst_storage_name, duration_ms, ctx.total_bytes, true, "");
        event_manager_->Publish(ev);
    }
    KVCM_LOG_INFO("migration completed: instance %s block_key %ld dst_loc %s retention %d",
                  ctx.instance_id.c_str(),
                  block_key,
                  ctx.dst_location_id.c_str(),
                  static_cast<int>(ctx.retention));
}

void MigrationManager::OnTaskFailed(const std::string &instance_id, int64_t block_key, ErrorCode reason) {
    CopyTaskContext ctx;
    ClaimResult claim;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        claim = ClaimForCompletionLocked(instance_id, block_key, ctx);
    }
    if (claim == ClaimResult::kNotFound || claim == ClaimResult::kBusyCompleting) {
        return;
    }
    if (claim == ClaimResult::kBusyPreparing) {
        KVCM_LOG_WARN("ignore premature migration failure callback for preparing task: instance %s block_key %ld",
                      instance_id.c_str(),
                      block_key);
        return;
    }
    if (claim == ClaimResult::kWasCancelling) {
        // 已取消，无论 copy 结果如何一律按取消收尾（清 WRITING 目标，记 cancelled 终态）。
        CompleteCancelledTask(ctx);
        return;
    }

    // 失败：CAS 目标 WRITING -> DELETING 并删除目标半成品，源端不动。
    const std::string fail_reason = "copy_failed:" + std::to_string(static_cast<int>(reason));
    CompleteCopyTaskAsFailed(ctx, fail_reason);
    KVCM_LOG_WARN("migration failed: instance %s block_key %ld dst_loc %s reason %d",
                  ctx.instance_id.c_str(),
                  block_key,
                  ctx.dst_location_id.c_str(),
                  reason);
}

void MigrationManager::SubmitTargetLocationDelete(const CopyTaskContext &ctx) {
    if (!schedule_plan_executor_ || ctx.dst_location_id.empty()) {
        return;
    }
    // SubmitNonBlocking 只把任务放入 executor 队列；随后才会按精确 location id CAS 到 DELETING，
    // 再删存储和元数据。调用方不能把本函数返回理解为目标已经删除完成。
    CacheLocationDelRequest del_req;
    del_req.instance_id = ctx.instance_id;
    del_req.block_keys = {ctx.block_key};
    del_req.location_ids = {{ctx.dst_location_id}};
    schedule_plan_executor_->SubmitNonBlocking(del_req, ScheduleTaskClass::kMigrationContinuation);
}

void MigrationManager::SubmitSourceLocationDelete(const CopyTaskContext &ctx) {
    if (!schedule_plan_executor_ || ctx.src_location_id.empty()) {
        return;
    }
    CacheLocationDelRequest del_req;
    del_req.instance_id = ctx.instance_id;
    del_req.block_keys = {ctx.block_key};
    del_req.location_ids = {{ctx.src_location_id}};
    schedule_plan_executor_->SubmitNonBlocking(del_req, ScheduleTaskClass::kMigrationContinuation);
}

void MigrationManager::MonitorLoop() {
    while (running_.load(std::memory_order_relaxed)) {
        ProcessExpiredMarks();

        PendingCopy cell;
        bool has_cell = false;
        {
            std::unique_lock<std::mutex> lock(pending_mutex_);
            if (pending_copies_.empty()) {
                pending_cv_.wait_for(lock, kMonitorIdleSleep, [this]() {
                    return !running_.load(std::memory_order_relaxed) || !pending_copies_.empty();
                });
            }
            if (!pending_copies_.empty()) {
                cell = std::move(pending_copies_.front());
                pending_copies_.pop_front();
                has_cell = true;
            }
        }
        if (!has_cell) {
            continue;
        }

        auto status = cell.future.wait_for(kFutureWaitTime);
        if (status != std::future_status::ready) {
            // 尚未完成，放回队尾稍后再查。
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_copies_.push_back(std::move(cell));
            continue;
        }

        PlanExecuteResult result = cell.future.get();
        if (result.status == EC_OK) {
            OnTaskSuccess(cell.instance_id, cell.block_key);
        } else {
            OnTaskFailed(cell.instance_id, cell.block_key, result.status);
        }
    }

    // 退出时排空剩余 future（不再驱动状态流转，仅释放）。
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_copies_.clear();
}

std::shared_ptr<MetaIndexer> MigrationManager::GetIndexer(const std::string &instance_id) const {
    return meta_indexer_manager_ ? meta_indexer_manager_->GetMetaIndexer(instance_id) : nullptr;
}

ErrorCode MigrationManager::MarkForTieredWrite(const std::string &instance_id,
                                               const std::vector<int64_t> &block_keys,
                                               const std::string &dst_storage_name,
                                               int64_t timeout_ms) {
    if (dst_storage_name.empty() || block_keys.empty()) {
        return EC_OK;
    }
    // 目标 storage 必须是已注册的 storage。否则打标会产生"永不被满足"的 mark：
    // 下次写入因目标不存在静默回落热层，且清标条件（目标覆盖 spec）永远不成立，mark 只能等超时。
    // 在此统一拦截，覆盖 admin 与 reclaimer 两条打标路径。
    if (data_storage_manager_ == nullptr || data_storage_manager_->GetDataStorageBackend(dst_storage_name) == nullptr) {
        KVCM_LOG_WARN("MarkForTieredWrite: target storage [%s] not registered, skip marking (instance %s)",
                      dst_storage_name.c_str(),
                      instance_id.c_str());
        return EC_NOENT;
    }
    if (timeout_ms <= 0) {
        timeout_ms = MigrationMarkMethod::kDefaultTimeoutMs;
    }
    auto indexer = GetIndexer(instance_id);
    if (indexer == nullptr) {
        KVCM_LOG_WARN("MarkForTieredWrite: meta indexer not found for instance %s", instance_id.c_str());
        return EC_INSTANCE_NOT_EXIST;
    }
    const int64_t now_ms = TimestampUtil::GetCurrentTimeMs();
    if (timeout_ms > std::numeric_limits<int64_t>::max() - now_ms) {
        KVCM_LOG_WARN(
            "MarkForTieredWrite: timeout %ld overflows deadline for instance %s", timeout_ms, instance_id.c_str());
        return EC_BADARGS;
    }
    const int64_t deadline_ms = now_ms + timeout_ms;
    // 记录实际成功打标的 key index，供 stat/expiry/event 仅按 actual 口径更新。
    // modifier 在 RMW 批次内按 global_idx 回调，顺序对齐 block_keys。
    std::vector<bool> mark_succeeded(block_keys.size(), false);
    // RMW：只写 property（out_new_locations 留空，不动 location）。不存在的 block 跳过（不给空 block 打标）。
    auto modifier = [&dst_storage_name, deadline_ms, &mark_succeeded](const LocationIdVector & /*existing*/,
                                                                     ErrorCode get_ec,
                                                                     size_t idx,
                                                                     PropertyMap &upsert_property_map,
                                                                     CacheLocationMap & /*out_new*/) -> ModifierResult {
        if (get_ec == EC_NOENT) {
            return {MA_SKIP, EC_OK};
        }
        if (get_ec != EC_OK) {
            return {MA_FAIL, get_ec};
        }
        upsert_property_map[PROPERTY_TIERED_WRITE_TARGET] = dst_storage_name;
        upsert_property_map[PROPERTY_TIERED_WRITE_DEADLINE_MS] = std::to_string(deadline_ms);
        if (idx < mark_succeeded.size()) {
            mark_succeeded[idx] = true;
        }
        return {MA_OK, EC_OK};
    };
    RequestContext rc("migration_mark");
    KeyVector keys(block_keys.begin(), block_keys.end());
    auto result = indexer->ReadModifyWriteBlock(&rc, keys, modifier);
    // stat/expiry/event 按实际成功数更新（actual 口径），不再用 block_keys.size()（request 口径）。
    const size_t actual_marked = static_cast<size_t>(std::count(mark_succeeded.begin(), mark_succeeded.end(), true));
    if (actual_marked > 0) {
        stat_marks_added_.fetch_add(actual_marked, std::memory_order_relaxed);
    }
    UpdateMarksActiveGauge();
    for (size_t i = 0; i < block_keys.size(); ++i) {
        if (!mark_succeeded[i]) {
            continue;
        }
        EnqueueMarkExpiry(instance_id, block_keys[i], dst_storage_name, deadline_ms);
        if (event_manager_ != nullptr) {
            auto ev = std::make_shared<MigrationMarkAddEvent>(instance_id);
            ev->SetEventTriggerTime();
            ev->SetAdditionalArgs(block_keys[i], dst_storage_name);
            event_manager_->Publish(ev);
        }
    }
    return result.ec;
}

bool MigrationManager::ClearTieredWriteMarkIfMatch(const std::string &instance_id,
                                                    int64_t block_key,
                                                    const std::string &expected_target,
                                                    int64_t expected_deadline_ms) {
    if (expected_target.empty() || expected_deadline_ms <= 0) {
        return false;
    }
    return ClearTieredWriteMarkIfMatchInternal(instance_id, block_key, expected_target, expected_deadline_ms);
}

bool MigrationManager::IsMarkedForTieredWrite(const std::string &instance_id, int64_t block_key) {
    std::vector<MarkQueryResult> results;
    const auto ec = BatchGetTieredWriteTargets(instance_id, {block_key}, results);
    return ec == EC_OK && !results.empty() && results[0].HasValidMark();
}

std::string MigrationManager::GetTieredWriteTarget(const std::string &instance_id, int64_t block_key) {
    std::vector<MarkQueryResult> results;
    const auto ec = BatchGetTieredWriteTargets(instance_id, {block_key}, results);
    return (ec == EC_OK && !results.empty() && results[0].HasValidMark()) ? results[0].target : std::string();
}

// match 检查在 RMW modifier 闭包内执行（shard lock 保原子），不需要外部 mark_mutex_。
// modifier 内通过捕获的 indexer 读 GetProperties（GetProperties 不走 shard lock，不死锁），
// 读-比较-写三步在同一个 shard lock 内完成，与 BatchCASLocationStatus 同模式。
bool MigrationManager::ClearTieredWriteMarkIfMatchInternal(const std::string &instance_id,
                                                           int64_t block_key,
                                                           const std::string &expected_target,
                                                           int64_t expected_deadline_ms,
                                                           bool is_expiry) {
    if (expected_target.empty()) {
        return false;
    }
    // private 调用以非正 deadline 表示“仅匹配当前仍 malformed 的同 target Mark”。
    // public ClearTieredWriteMarkIfMatch 仍拒绝非正 deadline，避免改变其精确快照语义。
    const bool expect_malformed = expected_deadline_ms <= 0;
    auto indexer = GetIndexer(instance_id);
    if (indexer == nullptr) {
        return false;
    }

    bool cleared = false;
    auto modifier = [&indexer, block_key, &expected_target, expected_deadline_ms, expect_malformed, &cleared](
                        const LocationIdVector & /*existing*/,
                        ErrorCode get_ec,
                        size_t /*idx*/,
                        PropertyMap &upsert_property_map,
                        CacheLocationMap & /*out_new*/) -> ModifierResult {
        if (get_ec != EC_OK) {
            return {MA_SKIP, EC_OK};
        }
        RequestContext check_rc("migration_match_clear_check");
        PropertyMapVector check_props;
        const auto check_result = indexer->GetProperties(&check_rc, {block_key},
            {PROPERTY_TIERED_WRITE_TARGET, PROPERTY_TIERED_WRITE_DEADLINE_MS}, check_props);
        if (check_result.ec != EC_OK || check_result.error_codes.size() != 1 || check_result.error_codes[0] != EC_OK ||
            check_props.size() != 1) {
            return {MA_SKIP, EC_OK};
        }
        auto mark = ParseMarkFromProperties(check_props[0], 0);
        if (mark.target != expected_target ||
            (expect_malformed ? !mark.malformed : mark.deadline_ms != expected_deadline_ms)) {
            return {MA_SKIP, EC_OK};
        }
        upsert_property_map[PROPERTY_TIERED_WRITE_TARGET] = "";
        upsert_property_map[PROPERTY_TIERED_WRITE_DEADLINE_MS] = "";
        cleared = true;
        return {MA_OK, EC_OK};
    };
    RequestContext rc("migration_conditional_clear_mark");
    indexer->ReadModifyWriteBlock(&rc, {block_key}, modifier);
    if (cleared && !expect_malformed) {
        stat_marks_cleared_.fetch_add(1, std::memory_order_relaxed);
        UpdateMarksActiveGauge();
        if (metrics_enabled_) {
            if (is_expiry) {
                ++m_marks_expired_total_;
            } else {
                ++m_marks_consumed_total_;
            }
        }
        if (event_manager_ != nullptr) {
            if (is_expiry) {
                auto ev = std::make_shared<MigrationMarkExpiredEvent>(instance_id);
                ev->SetEventTriggerTime();
                ev->SetAdditionalArgs(block_key, expected_target);
                event_manager_->Publish(ev);
            } else {
                auto ev = std::make_shared<MigrationMarkConsumedEvent>(instance_id);
                ev->SetEventTriggerTime();
                ev->SetAdditionalArgs(block_key, expected_target);
                event_manager_->Publish(ev);
            }
        }
    }
    return cleared;
}

ErrorCode MigrationManager::BatchGetTieredWriteTargets(const std::string &instance_id,
                                                       const std::vector<int64_t> &block_keys,
                                                       std::vector<MarkQueryResult> &out) {
    out.assign(block_keys.size(), MarkQueryResult{});
    if (block_keys.empty()) {
        return EC_OK;
    }
    auto fail_all = [&](ErrorCode ec, const char *reason) {
        for (auto &result : out) {
            result.state = MarkQueryState::kReadError;
            result.ec = ec;
        }
        if (metrics_enabled_) {
            m_mark_query_errors_total_ += block_keys.size();
        }
        KVCM_LOG_WARN("mark query failed for instance %s, keys %zu, ec %d: %s",
                      instance_id.c_str(),
                      block_keys.size(),
                      ec,
                      reason);
        return ec;
    };

    auto indexer = GetIndexer(instance_id);
    if (indexer == nullptr) {
        return fail_all(EC_INSTANCE_NOT_EXIST, "meta indexer not found");
    }
    RequestContext rc("migration_query_mark_batch");
    KeyVector keys(block_keys.begin(), block_keys.end());
    PropertyMapVector props;
    const auto query_result =
        indexer->GetProperties(&rc, keys, {PROPERTY_TIERED_WRITE_TARGET, PROPERTY_TIERED_WRITE_DEADLINE_MS}, props);
    if (query_result.error_codes.size() != block_keys.size() || props.size() != block_keys.size()) {
        KVCM_LOG_WARN("mark query result shape mismatch for instance %s: keys %zu, per_key_ec %zu, props %zu, "
                      "aggregate ec %d",
                      instance_id.c_str(),
                      block_keys.size(),
                      query_result.error_codes.size(),
                      props.size(),
                      query_result.ec);
        return fail_all(EC_ERROR, "result shape mismatch");
    }

    const auto backend_error_count = static_cast<std::size_t>(std::count_if(
        query_result.error_codes.begin(), query_result.error_codes.end(), [](ErrorCode ec) { return ec != EC_OK; }));
    const ErrorCode expected_aggregate_ec =
        backend_error_count == 0 ? EC_OK : (backend_error_count == block_keys.size() ? EC_ERROR : EC_PARTIAL_OK);
    if (query_result.ec != expected_aggregate_ec) {
        return fail_all(EC_ERROR, "aggregate and per-key errors are inconsistent");
    }

    std::vector<ExpiringMark> expired_marks;
    std::vector<std::pair<int64_t, std::string>> malformed_marks;
    const int64_t now_ms = TimestampUtil::GetCurrentTimeMs();
    std::size_t read_error_count = 0;
    for (size_t i = 0; i < block_keys.size(); ++i) {
        out[i].ec = query_result.error_codes[i];
        if (out[i].ec == EC_NOENT) {
            out[i].state = MarkQueryState::kBlockNotFound;
            continue;
        }
        if (out[i].ec != EC_OK) {
            out[i].state = MarkQueryState::kReadError;
            ++read_error_count;
            continue;
        }
        auto mark = ParseMarkFromProperties(props[i], now_ms);
        if (mark.malformed) {
            KVCM_LOG_WARN("malformed tiered write mark for instance %s, block %ld, target %s; "
                          "treating as no-mark and clearing conditionally",
                          instance_id.c_str(),
                          block_keys[i],
                          mark.target.c_str());
            malformed_marks.emplace_back(block_keys[i], std::move(mark.target));
            out[i].state = MarkQueryState::kNoMark;
            continue;
        }
        if (mark.target.empty()) {
            out[i].state = MarkQueryState::kNoMark;
            continue;
        }
        out[i].target = std::move(mark.target);
        out[i].deadline_ms = mark.deadline_ms;
        if (mark.expired) {
            out[i].state = MarkQueryState::kExpired;
            expired_marks.push_back(ExpiringMark{out[i].deadline_ms, instance_id, block_keys[i], out[i].target});
            continue;
        }
        out[i].state = MarkQueryState::kValid;
    }
    for (const auto &em : expired_marks) {
        ClearTieredWriteMarkIfMatchInternal(em.instance_id, em.block_key, em.target_storage, em.deadline_ms, true);
    }
    for (const auto &[block_key, target] : malformed_marks) {
        ClearTieredWriteMarkIfMatchInternal(instance_id, block_key, target, 0);
    }
    if (read_error_count == 0) {
        return EC_OK;
    }
    if (metrics_enabled_) {
        m_mark_query_errors_total_ += read_error_count;
    }
    KVCM_LOG_WARN("mark query partially failed for instance %s: keys %zu, read errors %zu",
                  instance_id.c_str(),
                  block_keys.size(),
                  read_error_count);
    return read_error_count == block_keys.size() ? EC_ERROR : EC_PARTIAL_OK;
}

void MigrationManager::EnqueueMarkExpiry(const std::string &instance_id,
                                         int64_t block_key,
                                         const std::string &target_storage,
                                         int64_t deadline_ms) {
    if (deadline_ms <= 0 || target_storage.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mark_expiry_mutex_);
        mark_expiry_queue_.push(ExpiringMark{deadline_ms, instance_id, block_key, target_storage});
    }
    pending_cv_.notify_one();
}

void MigrationManager::ProcessExpiredMarks() {
    const int64_t now_ms = TimestampUtil::GetCurrentTimeMs();
    std::vector<ExpiringMark> expired_marks;
    {
        std::lock_guard<std::mutex> lock(mark_expiry_mutex_);
        while (!mark_expiry_queue_.empty() && mark_expiry_queue_.top().deadline_ms <= now_ms) {
            expired_marks.push_back(mark_expiry_queue_.top());
            mark_expiry_queue_.pop();
        }
    }
    for (const auto &mark : expired_marks) {
        ClearTieredWriteMarkIfMatchInternal(
            mark.instance_id, mark.block_key, mark.target_storage, mark.deadline_ms, true);
    }
}

void MigrationManager::CompleteCancelledTask(const CopyTaskContext &ctx) {
    // 取消任务的延迟收尾（monitor 线程，认领到 kWasCancelling 时调用）。
    // 目标此时仍为 WRITING（cancelling 任务从未被 promote）；CAS WRITING->DELETING 删半成品，源端不动。
    SubmitTargetLocationDelete(ctx);
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        RemoveActiveTaskLocked(ctx.instance_id, ctx.block_key);
    }
    stat_copy_cancelled_.fetch_add(1, std::memory_order_relaxed);
    // cancelled 是与 success/failed 对称的终态：在实际清理时计数，保持 submitted==success+failed+cancelled。
    // 被取消任务无论底层 copy 成/败一律记 cancelled（用户意图优先）。
    const int64_t duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - ctx.submit_time)
                                    .count();
    if (metrics_enabled_) {
        ++m_tasks_completed_cancelled_;
    }
    if (event_manager_ != nullptr) {
        auto ev = std::make_shared<MigrationCompletedEvent>(ctx.instance_id);
        ev->SetEventTriggerTime();
        ev->SetAdditionalArgs(ctx.block_key,
                              ctx.src_storage_name,
                              ctx.dst_storage_name,
                              duration_ms,
                              ctx.total_bytes,
                              /*success=*/false,
                              /*fail_reason=*/"cancelled");
        event_manager_->Publish(ev);
    }
    KVCM_LOG_INFO("migration cancelled (deferred cleanup): instance %s block_key %ld dst_loc %s",
                  ctx.instance_id.c_str(),
                  ctx.block_key,
                  ctx.dst_location_id.c_str());
}

ErrorCode MigrationManager::Cancel(const std::string &instance_id, int64_t block_key) {
    // preparing 取消只标记 kPrepareCancelling，由仍在同步 I/O 中的提交线程在下一安全
    // 边界停止进入 copy、清目标并释放 reservation；running 取消仍等待 future 完成后由 monitor 收尾。
    // 两种 cancelling 期间任务都保留在活跃表，继续挡重复 Submit 并保护 WRITING 目标。
    CancelResult result;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        result = MarkCancellingLocked(instance_id, block_key);
    }
    switch (result) {
    case CancelResult::kNotFound:
        return EC_NOENT;
    case CancelResult::kMarkedPreparing:
        KVCM_LOG_INFO("[cancel] instance %s block_key %ld cancelled while preparing; submitter will clean up",
                      instance_id.c_str(),
                      block_key);
        return EC_OK;
    case CancelResult::kBusyCompleting:
        KVCM_LOG_INFO("[cancel] instance %s block_key %ld already completing, cancel too late",
                      instance_id.c_str(),
                      block_key);
        return EC_EXIST; // 完成中，取消太晚；迁移将照常完成
    case CancelResult::kMarked:
        KVCM_LOG_INFO("[cancel] instance %s block_key %ld marked cancelling; cleanup deferred to copy completion",
                      instance_id.c_str(),
                      block_key);
        return EC_OK;
    case CancelResult::kAlreadyCancelling:
        return EC_OK; // 幂等
    }
    return EC_OK; // 不可达
}

std::vector<ErrorCode> MigrationManager::BatchCancel(const std::string &instance_id,
                                                     const std::vector<int64_t> &block_keys) {
    std::vector<ErrorCode> results;
    results.reserve(block_keys.size());
    for (int64_t block_key : block_keys) {
        results.push_back(Cancel(instance_id, block_key));
    }
    return results;
}

bool MigrationManager::InsertActiveTaskLocked(CopyTaskContext ctx) {
    const std::string instance_id = ctx.instance_id;
    const int64_t block_key = ctx.block_key;
    auto &instance_tasks = active_tasks_by_instance_[instance_id];
    if (instance_tasks.count(block_key) > 0) {
        return false; // 已存在：调用方决定如何处理（回滚等）
    }
    instance_tasks[block_key] = std::move(ctx);
    UpdateActiveTasksGauge();
    return true;
}

bool MigrationManager::ReservePreparingTaskLocked(const MigrationRequest &request) {
    CopyTaskContext ctx;
    ctx.instance_group_name = request.instance_group_name;
    ctx.instance_id = request.instance_id;
    ctx.block_key = request.block_key;
    ctx.src_location_id = request.src_location_id;
    ctx.src_create_time = request.src_create_time;
    ctx.src_storage_name = request.src_storage_name;
    ctx.dst_storage_name = request.dst_storage_name;
    ctx.retention = request.retention;
    ctx.state = CopyTaskState::kPreparing;
    // InsertActiveTaskLocked 在同一个 task_mutex_ 临界区内完成存在性检查与插入，避免 check-then-insert 窗口。
    return InsertActiveTaskLocked(std::move(ctx));
}

bool MigrationManager::UpdatePreparingTaskLocked(const CopyTaskContext &ctx) {
    auto instance_iter = active_tasks_by_instance_.find(ctx.instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return false;
    }
    auto task_iter = instance_iter->second.find(ctx.block_key);
    if (task_iter == instance_iter->second.end() || task_iter->second.state != CopyTaskState::kPreparing) {
        return false;
    }
    task_iter->second = ctx;
    task_iter->second.state = CopyTaskState::kPreparing;
    return true;
}

bool MigrationManager::MarkTaskRunningLocked(const std::string &instance_id, int64_t block_key) {
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return false;
    }
    auto task_iter = instance_iter->second.find(block_key);
    if (task_iter == instance_iter->second.end() || task_iter->second.state != CopyTaskState::kPreparing) {
        return false;
    }
    task_iter->second.state = CopyTaskState::kRunning;
    return true;
}

bool MigrationManager::RemovePreparingTaskLocked(const std::string &instance_id, int64_t block_key) {
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return false;
    }
    auto task_iter = instance_iter->second.find(block_key);
    if (task_iter == instance_iter->second.end() || (task_iter->second.state != CopyTaskState::kPreparing &&
                                                     task_iter->second.state != CopyTaskState::kPrepareCancelling)) {
        return false;
    }
    return RemoveActiveTaskLocked(instance_id, block_key);
}

bool MigrationManager::RemoveActiveTaskLocked(const std::string &instance_id, int64_t block_key) {
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return false;
    }
    const size_t erased = instance_iter->second.erase(block_key);
    if (erased == 0) {
        return false;
    }
    if (instance_iter->second.empty()) {
        active_tasks_by_instance_.erase(instance_iter); // 内层空则收回外层，避免 stale 空 map
    }
    UpdateActiveTasksGauge();
    return true;
}

bool MigrationManager::HasActiveTaskLocked(const std::string &instance_id, int64_t block_key) const {
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    return instance_iter != active_tasks_by_instance_.end() && instance_iter->second.count(block_key) > 0;
}

MigrationManager::ClaimResult MigrationManager::ClaimForCompletionLocked(const std::string &instance_id,
                                                                        int64_t block_key,
                                                                        CopyTaskContext &out_ctx) {
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return ClaimResult::kNotFound;
    }
    auto task_iter = instance_iter->second.find(block_key);
    if (task_iter == instance_iter->second.end()) {
        return ClaimResult::kNotFound;
    }
    CopyTaskContext &task = task_iter->second;
    if (task.state == CopyTaskState::kPreparing || task.state == CopyTaskState::kPrepareCancelling) {
        return ClaimResult::kBusyPreparing;
    }
    if (task.state == CopyTaskState::kCancelling) {
        out_ctx = task; // 交由 CompleteCancelledTask 收尾（删 WRITING 目标、不 promote）
        return ClaimResult::kWasCancelling;
    }
    if (task.state == CopyTaskState::kCompleting) {
        return ClaimResult::kBusyCompleting; // 防御：完成路径仅单一 monitor 线程，正常不会重入
    }
    task.state = CopyTaskState::kCompleting; // 认领：此后并发 Cancel 走 busy 分支
    out_ctx = task;
    return ClaimResult::kClaimedRunning;
}

MigrationManager::CancelResult MigrationManager::MarkCancellingLocked(const std::string &instance_id,
                                                                     int64_t block_key) {
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return CancelResult::kNotFound;
    }
    auto task_iter = instance_iter->second.find(block_key);
    if (task_iter == instance_iter->second.end()) {
        return CancelResult::kNotFound;
    }
    CopyTaskContext &task = task_iter->second;
    if (task.state == CopyTaskState::kPreparing) {
        task.state = CopyTaskState::kPrepareCancelling;
        return CancelResult::kMarkedPreparing;
    }
    if (task.state == CopyTaskState::kCompleting) {
        return CancelResult::kBusyCompleting; // 完成中，取消太晚
    }
    if (task.state == CopyTaskState::kPrepareCancelling || task.state == CopyTaskState::kCancelling) {
        return CancelResult::kAlreadyCancelling; // 幂等
    }
    task.state = CopyTaskState::kCancelling; // 标记；收尾推迟到 future 完成时由 monitor 执行
    return CancelResult::kMarked;
}

bool MigrationManager::HasMigrationTask(const std::string &instance_id, int64_t block_key) const {
    std::lock_guard<std::mutex> lock(task_mutex_);
    return HasActiveTaskLocked(instance_id, block_key);
}

bool MigrationManager::HasActiveCopyTargetLocation(const std::string &instance_id,
                                                   int64_t block_key,
                                                   const std::string &location_id) const {
    if (location_id.empty()) {
        return false;
    }
    // 作用域限定为 (instance_id, block_key)：一次 copy 任务的目标 location 属于同一 block；
    // 直接 O(1) 定位，避免全表扫描，也避免跨 instance/block 的 id 碰撞误判。
    std::lock_guard<std::mutex> lock(task_mutex_);
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return false;
    }
    auto task_iter = instance_iter->second.find(block_key);
    if (task_iter == instance_iter->second.end()) {
        return false;
    }
    const CopyTaskContext &task = task_iter->second;
    // BatchAddLocation 可能已经提交、但 location_id 尚未返回给提交线程。
    // 此时按 (instance, block) 临时保护所有 WRITING location；id 一旦绑定即恢复精确匹配。
    if ((task.state == CopyTaskState::kPreparing || task.state == CopyTaskState::kPrepareCancelling) &&
        task.dst_location_id.empty()) {
        return true;
    }
    return task.dst_location_id == location_id;
}

size_t MigrationManager::ActiveTaskCountUnsafe() const {
    size_t total = 0;
    for (const auto &instance_entry : active_tasks_by_instance_) {
        total += instance_entry.second.size();
    }
    return total;
}

size_t MigrationManager::ActiveTaskCount() const {
    std::lock_guard<std::mutex> lock(task_mutex_);
    return ActiveTaskCountUnsafe();
}

size_t MigrationManager::ActiveTaskCountForGroupUnsafe(const std::string &instance_group_name) const {
    size_t total = 0;
    for (const auto &instance_entry : active_tasks_by_instance_) {
        for (const auto &task_entry : instance_entry.second) {
            if (task_entry.second.instance_group_name == instance_group_name) {
                ++total;
            }
        }
    }
    return total;
}

size_t MigrationManager::ActiveTaskCountForGroup(const std::string &instance_group_name) const {
    std::lock_guard<std::mutex> lock(task_mutex_);
    return ActiveTaskCountForGroupUnsafe(instance_group_name);
}

size_t MigrationManager::ActiveTaskCountForInstances(const std::vector<std::string> &instance_ids) const {
    std::lock_guard<std::mutex> lock(task_mutex_);
    size_t total = 0;
    for (const auto &id : instance_ids) {
        auto it = active_tasks_by_instance_.find(id);
        if (it != active_tasks_by_instance_.end()) {
            total += it->second.size();
        }
    }
    return total;
}

std::vector<int64_t> MigrationManager::GetActiveBlockKeysForInstance(const std::string &instance_id) const {
    std::lock_guard<std::mutex> lock(task_mutex_);
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return {};
    }
    std::vector<int64_t> keys;
    keys.reserve(instance_iter->second.size());
    for (const auto &entry : instance_iter->second) {
        keys.push_back(entry.first);
    }
    return keys;
}

void MigrationManager::BeginDrainingInstance(const std::string &instance_id) {
    std::lock_guard<std::mutex> lock(copy_submission_mutex_);
    draining_instances_.insert(instance_id);
}

void MigrationManager::EndDrainingInstance(const std::string &instance_id) {
    std::lock_guard<std::mutex> lock(copy_submission_mutex_);
    draining_instances_.erase(instance_id);
}

std::string MigrationManager::GetActiveTaskDstLocation(const std::string &instance_id, int64_t block_key) const {
    std::lock_guard<std::mutex> lock(task_mutex_);
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    if (instance_iter == active_tasks_by_instance_.end()) {
        return std::string();
    }
    auto iter = instance_iter->second.find(block_key);
    if (iter == instance_iter->second.end()) {
        return std::string();
    }
    return iter->second.dst_location_id;
}

void MigrationManager::DebugInsertActiveCopyTask(const std::string &instance_id,
                                                 int64_t block_key,
                                                 const std::string &dst_location_id) {
    std::lock_guard<std::mutex> lock(task_mutex_);
    CopyTaskContext ctx;
    ctx.instance_id = instance_id;
    ctx.block_key = block_key;
    ctx.dst_location_id = dst_location_id;
    InsertActiveTaskLocked(std::move(ctx));
}

void MigrationManager::DebugEnableCopySubmissionsForTest() {
    accepting_copy_submissions_.store(true, std::memory_order_release);
}

MigrationManager::CopyAdmission MigrationManager::CheckCopyAdmission(const std::string &instance_id,
                                                                     int64_t block_key,
                                                                     const CacheLocationMap &loc_map,
                                                                     const std::string &src_storage_name,
                                                                     const std::string &dst_storage_name) const {
    if (HasMigrationTask(instance_id, block_key)) {
        return {CopyAdmissionStatus::kAlreadyMigrating, nullptr};
    }
    const auto src_locations = FindLocationsOnStorage(loc_map, src_storage_name, {CacheLocationStatus::CLS_SERVING});
    if (src_locations.empty()) {
        return {CopyAdmissionStatus::kSourceServingNotFound, nullptr};
    }

    // 按目标 storage 上多个 location 的联合覆盖判断，替代旧的 per-location 全覆盖。
    // 建索引一次 O(L·S),之后 per-source O(S) set lookup。
    const auto serving_covered = CollectCoveredSpecNames(loc_map, dst_storage_name, {CacheLocationStatus::CLS_SERVING});
    const auto all_covered =
        CollectCoveredSpecNames(loc_map, dst_storage_name, {CacheLocationStatus::CLS_SERVING, CacheLocationStatus::CLS_WRITING});

    auto specs_covered_by = [](const CacheLocation &src, const std::unordered_set<std::string> &covered) {
        return std::all_of(src.location_specs().begin(), src.location_specs().end(),
                           [&covered](const auto &spec) { return covered.count(spec.name()) > 0; });
    };

    bool all_sources_covered_by_serving = true;
    for (const auto *src_loc : src_locations) {
        if (specs_covered_by(*src_loc, serving_covered)) {
            continue;
        }
        all_sources_covered_by_serving = false;
        if (specs_covered_by(*src_loc, all_covered)) {
            continue;
        }
        // 该 source 既未被 SERVING 也未被 SERVING∪WRITING 联合覆盖 → 需要 copy。
        return {CopyAdmissionStatus::kAccept, src_loc};
    }

    // 循环走完未 return kAccept：每个 source 都被 serving∪writing 覆盖。
    if (all_sources_covered_by_serving) {
        return {CopyAdmissionStatus::kTargetServingExists, nullptr};
    }
    // 上面已排除"全部 SERVING 覆盖"，故此处必为有 WRITING 参与的联合覆盖。
    return {CopyAdmissionStatus::kTargetWritingExists, nullptr};
}

std::pair<ErrorCode, std::vector<std::int64_t>>
MigrationManager::SelectMigrationCandidateKeys(RequestContext *request_context,
                                               const std::string &trace_id,
                                               const std::vector<int64_t> &explicit_block_keys,
                                               int64_t sample_count,
                                               const std::shared_ptr<MetaIndexer> &meta_indexer) const {
    std::vector<std::int64_t> candidate_keys;
    if (!explicit_block_keys.empty()) {
        candidate_keys.reserve(explicit_block_keys.size());
        std::unordered_set<int64_t> seen;
        seen.reserve(explicit_block_keys.size());
        for (const auto block_key : explicit_block_keys) {
            if (seen.insert(block_key).second) {
                candidate_keys.push_back(block_key);
            }
        }
        return {EC_OK, std::move(candidate_keys)};
    }

    constexpr int64_t kDefaultMigrateSampleCount = 100;
    const int64_t count = sample_count > 0 ? sample_count : kDefaultMigrateSampleCount;
    if (const auto ec = meta_indexer->SampleReclaimKeys(request_context, count, candidate_keys); ec != EC_OK) {
        KVCM_LOG_WARN("[%s] MigrateCache sample keys failed, ec %d", trace_id.c_str(), ec);
        return {ec, {}};
    }
    return {EC_OK, std::move(candidate_keys)};
}

std::size_t MigrationManager::AsyncMigrationPrepareKeyHash::operator()(const AsyncMigrationPrepareKey &key) const {
    std::size_t seed = std::hash<std::uint64_t>{}(key.generation);
    const auto combine = [&seed](std::size_t value) {
        seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    };
    combine(std::hash<std::string>{}(key.instance_group_name));
    combine(std::hash<std::string>{}(key.instance_id));
    combine(std::hash<std::string>{}(key.source_storage_name));
    combine(std::hash<std::string>{}(key.target_storage_name));
    return seed;
}

std::size_t MigrationManager::PendingAsyncMigrationPrepareCountForTest() const {
    std::lock_guard<std::mutex> lock(async_prepare_mutex_);
    return pending_async_prepare_jobs_.size();
}

std::size_t
MigrationManager::PendingAsyncMigrationPrepareCountForGroup(const std::string &instance_group_name) const {
    std::lock_guard<std::mutex> lock(async_prepare_mutex_);
    return static_cast<std::size_t>(
        std::count_if(pending_async_prepare_jobs_.begin(),
                      pending_async_prepare_jobs_.end(),
                      [&instance_group_name](const AsyncMigrationPrepareKey &key) {
                          return key.instance_group_name == instance_group_name;
                      }));
}

void MigrationManager::FinishAsyncMigrationPrepare(const AsyncMigrationPrepareKey &key) {
    std::lock_guard<std::mutex> lock(async_prepare_mutex_);
    pending_async_prepare_jobs_.erase(key);
}

bool MigrationManager::SubmitAsyncMigrationPrepare(AsyncMigrationPrepareJob job) {
    if (!schedule_plan_executor_ || !accepting_copy_submissions_.load(std::memory_order_acquire) ||
        job.instance_group_name.empty() || job.instance_id.empty() || job.source_storage_name.empty() ||
        job.target_storage_name.empty() || job.block_keys.empty() ||
        job.pending_location_ids_by_block.size() != job.block_keys.size()) {
        return false;
    }

    const auto generation = async_prepare_generation_.load(std::memory_order_acquire);
    AsyncMigrationPrepareKey key{generation,
                                 job.instance_group_name,
                                 job.instance_id,
                                 job.source_storage_name,
                                 job.target_storage_name};
    {
        std::lock_guard<std::mutex> lock(async_prepare_mutex_);
        if (!accepting_copy_submissions_.load(std::memory_order_acquire) ||
            async_prepare_generation_.load(std::memory_order_acquire) != generation ||
            pending_async_prepare_jobs_.size() >= kMaxPendingAsyncMigrationPrepareJobs ||
            !pending_async_prepare_jobs_.insert(key).second) {
            return false;
        }
    }

    auto weak_self = weak_from_this();
    if (weak_self.expired()) {
        FinishAsyncMigrationPrepare(key);
        return false;
    }
    auto execute_task = [weak_self, key, job = std::move(job)]() mutable {
        auto self = weak_self.lock();
        if (!self) {
            return;
        }
        try {
            self->RunAsyncMigrationPrepare(std::move(job), key.generation);
        } catch (const std::exception &e) {
            KVCM_LOG_ERROR("async migration prepare threw exception: %s", e.what());
        } catch (...) { KVCM_LOG_ERROR("async migration prepare threw unknown exception"); }
        self->FinishAsyncMigrationPrepare(key);
    };
    auto cancel_task = [weak_self, key]() {
        if (auto self = weak_self.lock()) {
            self->FinishAsyncMigrationPrepare(key);
        }
    };
    if (!schedule_plan_executor_->SubmitTask(ScheduleTaskClass::kMigrationPrepare,
                                             std::move(execute_task),
                                             std::chrono::microseconds::zero(),
                                             std::move(cancel_task))) {
        FinishAsyncMigrationPrepare(key);
        return false;
    }
    return true;
}

void MigrationManager::RunAsyncMigrationPrepare(AsyncMigrationPrepareJob job, std::uint64_t generation) {
    const auto prepare_and_dispatch = [&]() -> DispatchBatchResult {
        if (!accepting_copy_submissions_.load(std::memory_order_acquire) ||
            async_prepare_generation_.load(std::memory_order_acquire) != generation || !registry_manager_ ||
            !meta_indexer_manager_ || !data_storage_manager_) {
            return {};
        }

        // Prepare 会读取 Registry/Meta/DataStorage，必须与 leader cleanup 使用同一个 lifecycle
        // barrier。Stop 先关闭 gate，再取 unique lock：已运行 Job 收口后才能清理依赖；排队 Job
        // 随后取得 shared lock 时会因 generation/gate 变化直接退出，不再访问已清理资源。
        std::shared_lock<std::shared_mutex> lifecycle_lock(copy_submission_lifecycle_mutex_);
        if (!accepting_copy_submissions_.load(std::memory_order_acquire) ||
            async_prepare_generation_.load(std::memory_order_acquire) != generation) {
            return {};
        }

        auto request_context = std::make_shared<RequestContext>(job.trace_id);
        const auto instance_info = registry_manager_->GetInstanceInfo(request_context.get(), job.instance_id);
        if (!instance_info || instance_info->instance_group_name() != job.instance_group_name) {
            return {};
        }
        const auto cache_config = registry_manager_->GetCacheConfig(job.instance_group_name);
        if (!cache_config) {
            return {};
        }

        std::shared_ptr<MigrationStrategy> current_strategy;
        for (const auto &strategy : cache_config->migration_strategies()) {
            if (strategy && strategy->source_storage_name() == job.source_storage_name &&
                strategy->target_storage_name() == job.target_storage_name) {
                current_strategy = strategy;
                break;
            }
        }
        if (!current_strategy) {
            return {};
        }
        const bool copy_enabled = current_strategy->methods().copy().enabled();
        const bool mark_enabled = current_strategy->methods().mark().enabled();
        if ((!copy_enabled && !mark_enabled) ||
            data_storage_manager_->GetDataStorageBackend(job.source_storage_name) == nullptr ||
            data_storage_manager_->GetDataStorageBackend(job.target_storage_name) == nullptr) {
            return {};
        }

        const auto indexer = meta_indexer_manager_->GetMetaIndexer(job.instance_id);
        if (!indexer) {
            return {};
        }
        MetaSearcher meta_searcher(indexer);
        std::vector<CacheLocationMap> loc_maps;
        BlockMask empty_mask;
        if (meta_searcher.BatchGetLocation(request_context.get(), job.block_keys, empty_mask, loc_maps) != EC_OK ||
            loc_maps.size() != job.block_keys.size()) {
            return {};
        }

        for (std::size_t block_idx = 0; block_idx < loc_maps.size(); ++block_idx) {
            if (job.pending_location_ids_by_block[block_idx].empty()) {
                continue;
            }
            const std::unordered_set<std::string> pending_ids(job.pending_location_ids_by_block[block_idx].begin(),
                                                              job.pending_location_ids_by_block[block_idx].end());
            auto &loc_map = loc_maps[block_idx];
            for (auto it = loc_map.begin(); it != loc_map.end();) {
                if (pending_ids.count(it->first) > 0) {
                    it = loc_map.erase(it);
                } else {
                    ++it;
                }
            }
        }

        const auto configured_copy_concurrency = cache_config->migration_copy_max_concurrency();
        const std::size_t max_concurrent_copy =
            configured_copy_concurrency > 0 ? static_cast<std::size_t>(configured_copy_concurrency) : 0;
        DispatchBatchParams params;
        params.do_copy = copy_enabled;
        params.do_mark = mark_enabled;
        params.copy_limit = CopyConcurrencyLimit{job.instance_group_name, max_concurrent_copy};
        const auto active_copy_count = ActiveTaskCountForGroup(job.instance_group_name);
        params.max_copy_slots = max_concurrent_copy > active_copy_count ? max_concurrent_copy - active_copy_count : 0;
        params.retention = current_strategy->retention();
        params.mark_timeout_ms = current_strategy->methods().mark().timeout_ms();
        params.dedup_marks = true;
        return DispatchMigrationBatchWithLifecycleLockHeld(job.trace_id,
                                                            job.instance_id,
                                                            job.source_storage_name,
                                                            job.target_storage_name,
                                                            job.block_keys,
                                                            loc_maps,
                                                            params);
    };

    const auto dispatch = prepare_and_dispatch();
    // Callback is deliberately outside the lifecycle shared lock; it must not extend Stop latency
    // or create a shared-to-exclusive lock upgrade path.
    if (!job.on_dispatched) {
        return;
    }
    try {
        job.on_dispatched(dispatch);
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("async migration dispatch callback threw exception: %s", e.what());
    } catch (...) { KVCM_LOG_ERROR("async migration dispatch callback threw unknown exception"); }
}

MigrationManager::MigrateResult MigrationManager::MigrateCache(RequestContext *request_context,
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
                                                              int64_t mark_timeout_ms) {
    MigrateResult result;

    // meta_indexer 由 facade 保证非空（instance 存在性已前置校验）；防御性再查一次。
    if (meta_indexer == nullptr) {
        result.ec = EC_INSTANCE_NOT_EXIST;
        result.message = "instance not found: " + instance_id;
        return result;
    }

    // 1. 选候选 block_keys：显式 block_keys 优先；否则按 rule 采样 + 过滤。
    auto [select_ec, candidate_keys] =
        SelectMigrationCandidateKeys(request_context, trace_id, explicit_block_keys, sample_count, meta_indexer);
    if (select_ec != EC_OK) {
        result.ec = select_ec;
        result.message = "select migration candidate keys failed";
        return result;
    }

    const int64_t total = static_cast<int64_t>(candidate_keys.size());
    if (candidate_keys.empty()) {
        result.ec = EC_OK;
        result.accepted = 0;
        result.rejected = 0;
        result.message = "no candidate to migrate";
        return result;
    }

    // 2. 取各 block 的 location，准入过滤（在源 storage 上、SERVING、未在迁移中、目标无副本）。
    MetaSearcher meta_searcher(meta_indexer);
    std::vector<CacheLocationMap> loc_maps;
    BlockMask empty_mask;
    if (const auto ec = meta_searcher.BatchGetLocation(request_context, candidate_keys, empty_mask, loc_maps);
        ec != EC_OK || loc_maps.size() != candidate_keys.size()) {
        result.ec = EC_ERROR;
        result.message = "get cache location failed";
        return result;
    }

    // 准入、分发和 fallback 委派共享 DispatchMigrationBatch。
    DispatchBatchParams params;
    params.do_copy = do_copy;
    params.do_mark = do_mark;
    params.copy_limit = CopyConcurrencyLimit{instance_group_name, copy_max_concurrency};
    params.mark_timeout_ms = mark_timeout_ms;
    const auto dispatch =
        DispatchMigrationBatch(trace_id, instance_id, src_name, dst_name, candidate_keys, loc_maps, params);

    result.ec = EC_OK;
    result.accepted = dispatch.copy_submitted + dispatch.mark_submitted;
    result.rejected = total - result.accepted;
    result.message = "migrate cache dispatched";
    return result;
}

MigrationManager::DispatchBatchResult MigrationManager::DispatchMigrationBatch(
    const std::string &trace_id,
    const std::string &instance_id,
    const std::string &src_name,
    const std::string &dst_name,
    const std::vector<int64_t> &batch,
    const std::vector<CacheLocationMap> &loc_maps,
    const DispatchBatchParams &params) {
    DispatchBatchResult result;
    if (batch.empty() || loc_maps.size() != batch.size()) {
        return result;
    }
    if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
        return result;
    }
    // Copy 与 Mark 必须位于同一个 leader-lifecycle barrier 内。否则 Stop 可能在异步 Job
    // 完成 fresh read 后关闭 Copy gate，但 Job 仍继续写入 Mark。
    std::shared_lock<std::shared_mutex> lifecycle_lock(copy_submission_lifecycle_mutex_);
    if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
        return result;
    }
    return DispatchMigrationBatchWithLifecycleLockHeld(
        trace_id, instance_id, src_name, dst_name, batch, loc_maps, params);
}

MigrationManager::DispatchBatchResult MigrationManager::DispatchMigrationBatchWithLifecycleLockHeld(
    const std::string &trace_id,
    const std::string &instance_id,
    const std::string &src_name,
    const std::string &dst_name,
    const std::vector<int64_t> &batch,
    const std::vector<CacheLocationMap> &loc_maps,
    const DispatchBatchParams &params) {
    DispatchBatchResult result;
    if (batch.empty() || loc_maps.size() != batch.size()) {
        return result;
    }

    // 10a: mark 去重用 batch 查询替代逐 block IsMarkedForTieredWrite（N 次 meta 往返 → 1 次）。
    std::unordered_set<int64_t> already_marked;
    std::unordered_set<int64_t> mark_query_failed;
    if (params.do_mark && params.dedup_marks) {
        std::vector<MarkQueryResult> mark_results;
        const auto mark_query_ec = BatchGetTieredWriteTargets(instance_id, batch, mark_results);
        if (mark_query_ec != EC_OK) {
            KVCM_LOG_WARN("[%s] mark dedup query partially failed for instance %s, ec %d; failed keys will retry",
                          trace_id.c_str(),
                          instance_id.c_str(),
                          mark_query_ec);
        }
        for (std::size_t i = 0; i < batch.size() && i < mark_results.size(); ++i) {
            if (mark_results[i].HasValidMark()) {
                already_marked.insert(batch[i]);
            } else if (mark_results[i].IsReadError()) {
                // 无法确认已有 mark 时不覆盖未知状态，留待下一轮 reclaimer 重试。
                mark_query_failed.insert(batch[i]);
            }
        }
    }

    // 准入过滤 + 收集 copy / mark 候选
    std::vector<MigrationRequest> copy_reqs;
    std::vector<int64_t> copy_block_keys;
    std::vector<int64_t> mark_keys;
    for (std::size_t i = 0; i < batch.size(); ++i) {
        const int64_t block_key = batch[i];
        const auto admission = CheckCopyAdmission(instance_id, block_key, loc_maps[i], src_name, dst_name);
        if (admission.status != CopyAdmissionStatus::kAccept || admission.src_location == nullptr) {
            continue;
        }
        if (params.do_copy && copy_reqs.size() < params.max_copy_slots) {
            MigrationRequest req;
            req.instance_group_name = params.copy_limit.instance_group_name;
            req.instance_id = instance_id;
            req.block_key = block_key;
            req.src_location_id = admission.src_location->id();
            req.src_storage_name = src_name;
            req.dst_storage_name = dst_name;
            req.retention = params.retention;
            req.src_specs = admission.src_location->location_specs();
            req.src_create_time = admission.src_location->create_time();
            copy_block_keys.push_back(block_key);
            copy_reqs.push_back(std::move(req));
            continue; // Copy 优先：已进入 copy 的 block 不再重复 mark。
        }
        if (params.do_mark && already_marked.count(block_key) == 0 && mark_query_failed.count(block_key) == 0) {
            mark_keys.push_back(block_key);
        }
    }

    // 分发
    if (!copy_reqs.empty()) {
        const auto results = BatchSubmit(trace_id, std::move(copy_reqs), params.copy_limit, true);
        for (std::size_t i = 0; i < results.size(); ++i) {
            if (results[i] == EC_OK) {
                ++result.copy_submitted;
            } else {
                ++result.copy_failed;
                if (params.do_mark && i < copy_block_keys.size()) {
                    mark_keys.push_back(copy_block_keys[i]); // copy 失败时 fallback 到 mark
                }
            }
        }
    }
    if (params.do_mark && !mark_keys.empty()) {
        MarkForTieredWrite(instance_id, mark_keys, dst_name, params.mark_timeout_ms);
        result.mark_submitted = static_cast<int64_t>(mark_keys.size());
    }
    return result;
}

MigrationManager::MigrationStats MigrationManager::GetStats() const {
    MigrationStats stats;
    stats.copy_submitted = stat_copy_submitted_.load(std::memory_order_relaxed);
    stats.copy_completed = stat_copy_completed_.load(std::memory_order_relaxed);
    stats.copy_failed = stat_copy_failed_.load(std::memory_order_relaxed);
    stats.copy_cancelled = stat_copy_cancelled_.load(std::memory_order_relaxed);
    stats.marks_added = stat_marks_added_.load(std::memory_order_relaxed);
    stats.marks_cleared = stat_marks_cleared_.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        stats.active_copy_tasks = ActiveTaskCountUnsafe();
    }
    // 持久化方案下无内存表，active_marks 为 best-effort 近似（added - cleared）。
    const uint64_t added = stats.marks_added;
    const uint64_t cleared = stats.marks_cleared;
    stats.active_marks = added > cleared ? static_cast<size_t>(added - cleared) : 0;
    return stats;
}

} // namespace kv_cache_manager
