#include "kv_cache_manager/manager/migration_manager.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <tuple>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
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

int64_t ParseInt64OrZero(const std::string &value) {
    if (value.empty()) {
        return 0;
    }
    char *end = nullptr;
    const auto parsed = std::strtoll(value.c_str(), &end, 10);
    return end != nullptr && *end == '\0' ? parsed : 0;
}

bool LocationCoversSourceSpecsOnStorage(const CacheLocation &target_location,
                                        const std::string &storage_name,
                                        const CacheLocation &source_location) {
    const auto &source_specs = source_location.location_specs();
    if (source_specs.empty()) {
        return false;
    }
    const auto &target_specs = target_location.location_specs();
    return std::all_of(source_specs.begin(), source_specs.end(), [&target_specs, &storage_name](const auto &src_spec) {
        return std::any_of(target_specs.begin(), target_specs.end(), [&src_spec, &storage_name](const auto &dst_spec) {
            const DataStorageUri uri(dst_spec.uri());
            return dst_spec.name() == src_spec.name() && uri.Valid() && uri.GetHostName() == storage_name;
        });
    });
}

const CacheLocation *FindLocationCoveringSourceSpecsOnStorage(
    const CacheLocationMap &loc_map,
    const std::string &storage_name,
    std::initializer_list<CacheLocationStatus> statuses,
    const CacheLocation &source_location) {
    for (const auto &[_, loc_ptr] : loc_map) {
        if (!loc_ptr || std::find(statuses.begin(), statuses.end(), loc_ptr->status()) == statuses.end()) {
            continue;
        }
        if (LocationCoversSourceSpecsOnStorage(*loc_ptr, storage_name, source_location)) {
            return loc_ptr.get();
        }
    }
    return nullptr;
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
                                   std::shared_ptr<EventManager> event_manager)
    : schedule_plan_executor_(std::move(schedule_plan_executor))
    , meta_indexer_manager_(std::move(meta_indexer_manager))
    , data_storage_manager_(std::move(data_storage_manager))
    , metrics_registry_(std::move(metrics_registry))
    , event_manager_(std::move(event_manager)) {
    if (metrics_registry_ != nullptr) {
        metrics_enabled_ = true;
        m_tasks_submitted_total_ = metrics_registry_->GetCounter("migration.tasks_submitted_total");
        m_tasks_completed_success_ =
            metrics_registry_->GetCounter("migration.tasks_completed_total", {{"status", "success"}});
        m_tasks_completed_failed_ =
            metrics_registry_->GetCounter("migration.tasks_completed_total", {{"status", "failed"}});
        m_tasks_active_ = metrics_registry_->GetGauge("migration.tasks_active");
        m_copy_bytes_total_ = metrics_registry_->GetCounter("migration.copy_bytes_total");
        m_copy_duration_ms_ = metrics_registry_->GetGauge("migration.copy_duration_ms");
        m_marks_active_ = metrics_registry_->GetGauge("migration.marks_active");
        m_marks_consumed_total_ = metrics_registry_->GetCounter("migration.marks_consumed_total");
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
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return; // already running
    }
    accepting_copy_submissions_.store(true, std::memory_order_release);
    monitor_thread_ = std::thread([this]() { MonitorLoop(); });
    KVCM_LOG_INFO("MigrationManager started");
}

void MigrationManager::Stop() {
    accepting_copy_submissions_.store(false, std::memory_order_release);
    std::lock_guard<std::mutex> submission_lock(copy_submission_mutex_);

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
                                            const MigrationRequest &request,
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

    // 1. 读取源 location。
    MetaSearcher meta_searcher(indexer);
    auto ctx = std::make_shared<RequestContext>(trace_id.empty() ? "migration_prepare" : trace_id);
    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    ErrorCode ec = meta_searcher.BatchGetLocation(ctx.get(), {request.block_key}, empty_mask, location_maps);
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
        KVCM_LOG_WARN("[%s] source location %s has no specs", trace_id.c_str(), request.src_location_id.c_str());
        return EC_ERROR;
    }

    // 2. 目标 storage 类型。
    auto dst_backend = data_storage_manager_->GetDataStorageBackend(request.dst_storage_name);
    if (!dst_backend) {
        KVCM_LOG_WARN("[%s] target storage %s not found", trace_id.c_str(), request.dst_storage_name.c_str());
        return EC_NOENT;
    }
    const DataStorageType dst_type = dst_backend->GetType();

    // 3. 逐 spec 在目标 storage 预分配空间，构建目标 location specs + src/dst uri 对。
    std::vector<LocationSpec> dst_specs;
    dst_specs.reserve(src_location.location_specs().size());
    out_src_uris.clear();
    out_dst_uris.clear();
    std::vector<DataStorageUri> allocated_for_rollback;
    std::uint64_t total_bytes = 0;

    auto rollback = [&]() {
        if (!allocated_for_rollback.empty()) {
            data_storage_manager_->Delete(ctx.get(), request.dst_storage_name, allocated_for_rollback, nullptr);
        }
    };

    for (const auto &src_spec : src_location.location_specs()) {
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
    ec = meta_searcher.BatchAddLocation(ctx.get(), {request.block_key}, {dst_location}, out_location_ids);
    if (ec != EC_OK || out_location_ids.empty() || out_location_ids[0].empty()) {
        KVCM_LOG_WARN("[%s] BatchAddLocation failed for block_key %ld, ec %d",
                      trace_id.c_str(),
                      request.block_key,
                      ec);
        rollback();
        return EC_ERROR;
    }

    out_ctx.instance_id = request.instance_id;
    out_ctx.block_key = request.block_key;
    out_ctx.src_location_id = request.src_location_id;
    out_ctx.src_create_time = src_location.create_time(); // F-08: 记录源 location 的创建时间
    out_ctx.src_storage_name = request.src_storage_name;
    out_ctx.dst_storage_name = request.dst_storage_name;
    out_ctx.dst_location_id = out_location_ids[0];
    out_ctx.retention = request.retention;
    out_ctx.total_bytes = total_bytes;
    return EC_OK;
}

ErrorCode MigrationManager::Submit(const std::string &trace_id, MigrationRequest request) {
    std::lock_guard<std::mutex> submission_lock(copy_submission_mutex_);
    if (!accepting_copy_submissions_.load(std::memory_order_acquire)) {
        KVCM_LOG_WARN("[%s] reject migration copy submit for instance %s block_key %ld: "
                      "manager is not accepting submissions",
                      trace_id.c_str(),
                      request.instance_id.c_str(),
                      request.block_key);
        return EC_ERROR;
    }

    // 同一 instance 内同一 block 已有活跃任务则拒绝（防重复迁移）。
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        auto instance_iter = active_tasks_by_instance_.find(request.instance_id);
        if (instance_iter != active_tasks_by_instance_.end() &&
            instance_iter->second.count(request.block_key) > 0) {
            KVCM_LOG_INFO("[%s] instance %s block_key %ld already has an active migration task, skip",
                          trace_id.c_str(),
                          request.instance_id.c_str(),
                          request.block_key);
            return EC_EXIST;
        }
    }

    CopyTaskContext ctx;
    std::vector<DataStorageUri> src_uris;
    std::vector<DataStorageUri> dst_uris;
    ErrorCode prepare_ec = PrepareCopyTask(trace_id, request, ctx, src_uris, dst_uris);
    if (prepare_ec != EC_OK) {
        return prepare_ec;
    }
    ctx.submit_time = std::chrono::steady_clock::now();

    // 登记活跃任务（用于防重复迁移和 copy 并发预算统计）。
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        auto &instance_tasks = active_tasks_by_instance_[request.instance_id];
        if (instance_tasks.count(request.block_key) > 0) {
            // 并发竞争：已被其它请求占用，回滚刚建好的目标 location。
            SubmitTargetLocationDelete(ctx);
            return EC_EXIST;
        }
        instance_tasks.emplace(request.block_key, ctx);
        UpdateActiveTasksGauge();
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
        std::lock_guard<std::mutex> lock(task_mutex_);
        auto instance_iter = active_tasks_by_instance_.find(ctx.instance_id);
        if (instance_iter != active_tasks_by_instance_.end()) {
            instance_iter->second.erase(ctx.block_key);
            if (instance_iter->second.empty()) {
                active_tasks_by_instance_.erase(instance_iter);
            }
        }
        UpdateActiveTasksGauge();
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
                                                     std::vector<MigrationRequest> requests) {
    std::vector<ErrorCode> results;
    results.reserve(requests.size());
    for (auto &req : requests) {
        results.push_back(Submit(trace_id, std::move(req)));
    }
    return results;
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
    // F-08: id + status + create_time 三者同时匹配，防止 id 复用导致误判新 location 为原始源。
    return iter->second->status() == CLS_SERVING &&
           iter->second->create_time() == ctx.src_create_time;
}

void MigrationManager::CompleteCopyTaskAsFailed(const CopyTaskContext &ctx, const std::string &fail_reason) {
    SubmitTargetLocationDelete(ctx);
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        auto instance_iter = active_tasks_by_instance_.find(ctx.instance_id);
        if (instance_iter != active_tasks_by_instance_.end()) {
            instance_iter->second.erase(ctx.block_key);
            if (instance_iter->second.empty()) {
                active_tasks_by_instance_.erase(instance_iter);
            }
        }
        UpdateActiveTasksGauge();
    }
    stat_copy_failed_.fetch_add(1, std::memory_order_relaxed);
    if (metrics_enabled_) {
        ++m_tasks_completed_failed_;
    }
    if (event_manager_ != nullptr) {
        auto ev = std::make_shared<MigrationCompletedEvent>(ctx.instance_id);
        ev->SetEventTriggerTime();
        ev->SetAdditionalArgs(
            ctx.block_key, ctx.src_storage_name, ctx.dst_storage_name, 0, ctx.total_bytes, false, fail_reason);
        event_manager_->Publish(ev);
    }
}

void MigrationManager::OnTaskSuccess(const std::string &instance_id, int64_t block_key) {
    CopyTaskContext ctx;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        auto instance_iter = active_tasks_by_instance_.find(instance_id);
        if (instance_iter == active_tasks_by_instance_.end()) {
            return; // 已被取消或处理过
        }
        auto iter = instance_iter->second.find(block_key);
        if (iter == instance_iter->second.end()) {
            return; // 已被取消或处理过
        }
        ctx = iter->second;
    }

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

    // 目标已可用，先清除可能存在的 Mark，避免后续 StartWriteCache 重复写冷层。
    ClearTieredWriteMark(ctx.instance_id, block_key);

    // 按 retention 处理源端。
    if (ctx.retention == MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE) {
        SubmitSourceLocationDelete(ctx);
    }

    // 最后移除活跃任务。
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        auto instance_iter = active_tasks_by_instance_.find(ctx.instance_id);
        if (instance_iter != active_tasks_by_instance_.end()) {
            instance_iter->second.erase(block_key);
            if (instance_iter->second.empty()) {
                active_tasks_by_instance_.erase(instance_iter);
            }
        }
        UpdateActiveTasksGauge();
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
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        auto instance_iter = active_tasks_by_instance_.find(instance_id);
        if (instance_iter == active_tasks_by_instance_.end()) {
            return;
        }
        auto iter = instance_iter->second.find(block_key);
        if (iter == instance_iter->second.end()) {
            return;
        }
        ctx = iter->second;
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
    // CacheLocationDelRequest 会把目标 location CAS 到 DELETING 后删存储/删元数据，清理半成品。
    CacheLocationDelRequest del_req;
    del_req.instance_id = ctx.instance_id;
    del_req.block_keys = {ctx.block_key};
    del_req.location_ids = {{ctx.dst_location_id}};
    schedule_plan_executor_->SubmitNonBlocking(del_req);
}

void MigrationManager::SubmitSourceLocationDelete(const CopyTaskContext &ctx) {
    if (!schedule_plan_executor_ || ctx.src_location_id.empty()) {
        return;
    }
    CacheLocationDelRequest del_req;
    del_req.instance_id = ctx.instance_id;
    del_req.block_keys = {ctx.block_key};
    del_req.location_ids = {{ctx.src_location_id}};
    schedule_plan_executor_->SubmitNonBlocking(del_req);
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
    // F-02: 目标 storage 必须是已注册的 storage。否则打标会产生"永不被满足"的 mark：
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
    const int64_t deadline_ms = TimestampUtil::GetCurrentTimeMs() + timeout_ms;
    // F-14: 记录实际成功打标的 key index，供 stat/expiry/event 仅按 actual 口径更新。
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
    MetaIndexer::Result result(EC_OK);
    {
        std::lock_guard<std::mutex> mark_lock(mark_mutex_);
        result = indexer->ReadModifyWriteBlock(&rc, keys, modifier);
    }
    // F-14: stat/expiry/event 按实际成功数更新（actual 口径），不再用 block_keys.size()（request 口径）。
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

bool MigrationManager::IsMarkedForTieredWrite(const std::string &instance_id, int64_t block_key) const {
    std::string target;
    return ShouldWriteToTieredStorageByMark(instance_id, block_key, target);
}

std::string MigrationManager::GetTieredWriteTarget(const std::string &instance_id, int64_t block_key) const {
    std::string target;
    return ShouldWriteToTieredStorageByMark(instance_id, block_key, target) ? target : std::string();
}

bool MigrationManager::ClearTieredWriteMarkInternal(const std::string &instance_id, int64_t block_key) {
    std::lock_guard<std::mutex> mark_lock(mark_mutex_);
    auto indexer = GetIndexer(instance_id);
    if (indexer == nullptr) {
        return false;
    }
    std::string event_dst_storage;
    bool had_mark = false;
    {
        RequestContext prop_rc("migration_clear_mark_props");
        PropertyMapVector props;
        indexer->GetProperties(&prop_rc, {block_key}, {PROPERTY_TIERED_WRITE_TARGET}, props);
        if (!props.empty()) {
            const auto target_it = props[0].find(PROPERTY_TIERED_WRITE_TARGET);
            if (target_it != props[0].end() && !target_it->second.empty()) {
                had_mark = true;
                event_dst_storage = target_it->second;
            }
        }
    }
    bool cleared = false;
    // RMW：把 mark 属性置空作墓碑（读侧把空视为未打标）。block 不存在则 no-op（幂等）。
    auto modifier = [&cleared](const LocationIdVector & /*existing*/,
                               ErrorCode get_ec,
                               size_t /*idx*/,
                               PropertyMap &upsert_property_map,
                               CacheLocationMap & /*out_new*/) -> ModifierResult {
        if (get_ec != EC_OK) {
            return {MA_SKIP, EC_OK};
        }
        upsert_property_map[PROPERTY_TIERED_WRITE_TARGET] = "";
        upsert_property_map[PROPERTY_TIERED_WRITE_DEADLINE_MS] = "";
        cleared = true;
        return {MA_OK, EC_OK};
    };
    RequestContext rc("migration_clear_mark");
    indexer->ReadModifyWriteBlock(&rc, {block_key}, modifier);
    if (cleared) {
        if (had_mark) {
            stat_marks_cleared_.fetch_add(1, std::memory_order_relaxed);
            UpdateMarksActiveGauge();
            if (metrics_enabled_) {
                ++m_marks_consumed_total_;
            }
        }
        if (event_manager_ != nullptr && had_mark) {
            auto ev = std::make_shared<MigrationMarkConsumedEvent>(instance_id);
            ev->SetEventTriggerTime();
            ev->SetAdditionalArgs(block_key, event_dst_storage);
            event_manager_->Publish(ev);
        }
    }
    return cleared;
}

void MigrationManager::ClearTieredWriteMark(const std::string &instance_id, int64_t block_key) {
    static_cast<void>(ClearTieredWriteMarkInternal(instance_id, block_key));
}

bool MigrationManager::ClearTieredWriteMarkIfMatchInternal(const std::string &instance_id,
                                                           int64_t block_key,
                                                           const std::string &expected_target,
                                                           int64_t expected_deadline_ms) {
    if (expected_target.empty() || expected_deadline_ms <= 0) {
        return false;
    }
    std::lock_guard<std::mutex> mark_lock(mark_mutex_);
    auto indexer = GetIndexer(instance_id);
    if (indexer == nullptr) {
        return false;
    }
    RequestContext prop_rc("migration_conditional_clear_mark_props");
    PropertyMapVector props;
    indexer->GetProperties(
        &prop_rc, {block_key}, {PROPERTY_TIERED_WRITE_TARGET, PROPERTY_TIERED_WRITE_DEADLINE_MS}, props);
    if (props.empty()) {
        return false;
    }
    const auto target_it = props[0].find(PROPERTY_TIERED_WRITE_TARGET);
    const auto deadline_it = props[0].find(PROPERTY_TIERED_WRITE_DEADLINE_MS);
    const int64_t current_deadline_ms =
        deadline_it == props[0].end() ? 0 : ParseInt64OrZero(deadline_it->second);
    if (target_it == props[0].end() || target_it->second != expected_target ||
        current_deadline_ms != expected_deadline_ms) {
        return false;
    }

    bool cleared = false;
    auto modifier = [&cleared](const LocationIdVector & /*existing*/,
                               ErrorCode get_ec,
                               size_t /*idx*/,
                               PropertyMap &upsert_property_map,
                               CacheLocationMap & /*out_new*/) -> ModifierResult {
        if (get_ec != EC_OK) {
            return {MA_SKIP, EC_OK};
        }
        upsert_property_map[PROPERTY_TIERED_WRITE_TARGET] = "";
        upsert_property_map[PROPERTY_TIERED_WRITE_DEADLINE_MS] = "";
        cleared = true;
        return {MA_OK, EC_OK};
    };
    RequestContext rc("migration_conditional_clear_mark");
    indexer->ReadModifyWriteBlock(&rc, {block_key}, modifier);
    if (cleared) {
        stat_marks_cleared_.fetch_add(1, std::memory_order_relaxed);
        UpdateMarksActiveGauge();
        if (metrics_enabled_) {
            ++m_marks_consumed_total_;
        }
    }
    return cleared;
}

bool MigrationManager::ShouldWriteToTieredStorageByMark(const std::string &instance_id,
                                                        int64_t block_key,
                                                        std::string &target) const {
    auto indexer = GetIndexer(instance_id);
    if (indexer == nullptr) {
        return false;
    }
    RequestContext rc("migration_query_mark");
    PropertyMapVector props;
    indexer->GetProperties(&rc, {block_key}, {PROPERTY_TIERED_WRITE_TARGET, PROPERTY_TIERED_WRITE_DEADLINE_MS}, props);
    if (props.empty()) {
        return false;
    }
    auto tit = props[0].find(PROPERTY_TIERED_WRITE_TARGET);
    if (tit == props[0].end() || tit->second.empty()) {
        return false; // 未打标 / 已清（墓碑）
    }
    auto deadline_it = props[0].find(PROPERTY_TIERED_WRITE_DEADLINE_MS);
    const int64_t deadline_ms = deadline_it == props[0].end() ? 0 : ParseInt64OrZero(deadline_it->second);
    if (deadline_ms > 0 && deadline_ms <= TimestampUtil::GetCurrentTimeMs()) {
        const_cast<MigrationManager *>(this)->ClearTieredWriteMarkIfMatchInternal(
            instance_id, block_key, tit->second, deadline_ms);
        return false;
    }
    target = tit->second;
    return true;
}

ErrorCode MigrationManager::BatchGetTieredWriteTargets(const std::string &instance_id,
                                                       const std::vector<int64_t> &block_keys,
                                                       std::vector<std::string> &out_targets) const {
    out_targets.assign(block_keys.size(), std::string());
    if (block_keys.empty()) {
        return EC_OK;
    }
    auto indexer = GetIndexer(instance_id);
    if (indexer == nullptr) {
        return EC_INSTANCE_NOT_EXIST;
    }
    RequestContext rc("migration_query_mark_batch");
    KeyVector keys(block_keys.begin(), block_keys.end());
    PropertyMapVector props;
    // 注：部分 block 不存在时 GetProperties 可能返回非 OK 聚合 ec，但 props 仍逐 key 填充
    //（缺失 key 为空 map）。因此不按聚合 ec 早退，直接按 props 逐项解析。
    indexer->GetProperties(&rc, keys, {PROPERTY_TIERED_WRITE_TARGET, PROPERTY_TIERED_WRITE_DEADLINE_MS}, props);
    std::vector<ExpiringMark> expired_marks;
    const int64_t now_ms = TimestampUtil::GetCurrentTimeMs();
    for (size_t i = 0; i < props.size() && i < out_targets.size(); ++i) {
        auto tit = props[i].find(PROPERTY_TIERED_WRITE_TARGET);
        if (tit == props[i].end() || tit->second.empty()) {
            continue;
        }
        auto deadline_it = props[i].find(PROPERTY_TIERED_WRITE_DEADLINE_MS);
        const int64_t deadline_ms = deadline_it == props[i].end() ? 0 : ParseInt64OrZero(deadline_it->second);
        if (deadline_ms > 0 && deadline_ms <= now_ms) {
            expired_marks.push_back(ExpiringMark{deadline_ms, instance_id, block_keys[i], tit->second});
            continue;
        }
        out_targets[i] = tit->second;
    }
    for (const auto &mark : expired_marks) {
        const_cast<MigrationManager *>(this)->ClearTieredWriteMarkIfMatchInternal(
            mark.instance_id, mark.block_key, mark.target_storage, mark.deadline_ms);
    }
    return EC_OK;
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
            mark.instance_id, mark.block_key, mark.target_storage, mark.deadline_ms);
    }
}

ErrorCode MigrationManager::Cancel(const std::string &instance_id, int64_t block_key) {
    CopyTaskContext ctx;
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        auto instance_iter = active_tasks_by_instance_.find(instance_id);
        if (instance_iter == active_tasks_by_instance_.end()) {
            return EC_NOENT;
        }
        auto iter = instance_iter->second.find(block_key);
        if (iter == instance_iter->second.end()) {
            return EC_NOENT;
        }
        ctx = iter->second;
        instance_iter->second.erase(iter);
        if (instance_iter->second.empty()) {
            active_tasks_by_instance_.erase(instance_iter);
        }
        UpdateActiveTasksGauge();
    }
    // 清理目标半成品（CAS WRITING->DELETING + 删除）。源端不动。
    SubmitTargetLocationDelete(ctx);
    stat_copy_cancelled_.fetch_add(1, std::memory_order_relaxed);
    KVCM_LOG_INFO("migration cancelled: instance %s block_key %ld dst_loc %s",
                  ctx.instance_id.c_str(),
                  block_key,
                  ctx.dst_location_id.c_str());
    return EC_OK;
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

bool MigrationManager::HasMigrationTask(const std::string &instance_id, int64_t block_key) const {
    std::lock_guard<std::mutex> lock(task_mutex_);
    auto instance_iter = active_tasks_by_instance_.find(instance_id);
    return instance_iter != active_tasks_by_instance_.end() && instance_iter->second.count(block_key) > 0;
}

bool MigrationManager::HasActiveCopyTargetLocation(const std::string &location_id) const {
    if (location_id.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(task_mutex_);
    for (const auto &instance_entry : active_tasks_by_instance_) {
        const auto &instance_tasks = instance_entry.second;
        for (const auto &task_entry : instance_tasks) {
            const auto &ctx = task_entry.second;
            if (ctx.dst_location_id == location_id) {
                return true;
            }
        }
    }
    return false;
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
    active_tasks_by_instance_[instance_id][block_key] = std::move(ctx);
    UpdateActiveTasksGauge();
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

    bool all_sources_covered_by_serving = true;
    bool all_sources_covered_by_writing = true;
    for (const auto *src_loc : src_locations) {
        if (FindLocationCoveringSourceSpecsOnStorage(
                loc_map, dst_storage_name, {CacheLocationStatus::CLS_SERVING}, *src_loc) != nullptr) {
            continue;
        }
        all_sources_covered_by_serving = false;
        if (FindLocationCoveringSourceSpecsOnStorage(
                loc_map, dst_storage_name, {CacheLocationStatus::CLS_WRITING}, *src_loc) != nullptr) {
            continue;
        }
        all_sources_covered_by_writing = false;
        return {CopyAdmissionStatus::kAccept, src_loc};
    }

    if (all_sources_covered_by_serving) {
        return {CopyAdmissionStatus::kTargetServingExists, nullptr};
    }
    if (all_sources_covered_by_writing) {
        return {CopyAdmissionStatus::kTargetWritingExists, nullptr};
    }
    return {CopyAdmissionStatus::kTargetServingExists, nullptr};
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
