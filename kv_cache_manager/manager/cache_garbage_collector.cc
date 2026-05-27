#include "kv_cache_manager/manager/cache_garbage_collector.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/manager/write_location_manager.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

#define DEFINE_METRICS_NAME_FOR_CACHE_GC(name) DEFINE_METRICS_NAME_(CacheGarbageCollector, cache_gc, name)

#define REGISTER_COUNTER_METRICS_FOR_CACHE_GC(name) REGISTER_METRICS_COUNTER_(metrics_registry_, cache_gc, name)

#define REGISTER_GAUGE_METRICS_FOR_CACHE_GC(name) REGISTER_METRICS_GAUGE_(metrics_registry_, cache_gc, name)

DEFINE_METRICS_NAME_FOR_CACHE_GC(scan_round_count);
DEFINE_METRICS_NAME_FOR_CACHE_GC(orphaned_writing_count);
DEFINE_METRICS_NAME_FOR_CACHE_GC(stale_serving_count);
DEFINE_METRICS_NAME_FOR_CACHE_GC(location_submit_count);
DEFINE_METRICS_NAME_FOR_CACHE_GC(scan_batch_duration_us);

const std::string CacheGarbageCollector::kTraceIDPrefix{"cache_gc_internal_trace_"};

inline std::string CacheGarbageCollector::GenTraceID() {
    static std::random_device rd;
    static std::mt19937_64 rng(rd());
    static std::uniform_int_distribution<std::uint64_t> dis;
    const std::uint64_t rand_val = dis(rng);
    std::stringstream ss;
    ss << kTraceIDPrefix << std::right << std::setfill('0') << std::setw(16) << std::hex << std::noshowbase << rand_val;
    return ss.str();
}

static std::string VineyardStorageNameFromInstance(const std::string &instance_id) { return "v6d_" + instance_id; }

CacheGarbageCollector::CacheGarbageCollector(Config config,
                                             std::shared_ptr<RegistryManager> registry_manager,
                                             std::shared_ptr<MetaIndexerManager> meta_indexer_manager,
                                             std::shared_ptr<SchedulePlanExecutor> sched_plan_executor,
                                             std::shared_ptr<MetricsRegistry> metrics_registry,
                                             std::shared_ptr<EventManager> event_manager,
                                             std::shared_ptr<WriteLocationManager> write_location_manager)
    : config_(config)
    , registry_manager_(std::move(registry_manager))
    , meta_indexer_manager_(std::move(meta_indexer_manager))
    , sched_plan_executor_(std::move(sched_plan_executor))
    , metrics_registry_(std::move(metrics_registry))
    , event_manager_(std::move(event_manager))
    , write_location_manager_(std::move(write_location_manager)) {}

CacheGarbageCollector::~CacheGarbageCollector() { Stop(); }

ErrorCode CacheGarbageCollector::Start() noexcept {
    if (!config_.enabled) {
        KVCM_LOG_INFO("cache garbage collector is disabled");
        return ErrorCode::EC_OK;
    }
    if (registry_manager_ == nullptr) {
        KVCM_LOG_ERROR("registry manager is nullptr");
        return ErrorCode::EC_ERROR;
    }
    if (meta_indexer_manager_ == nullptr) {
        KVCM_LOG_ERROR("meta indexer manager is nullptr");
        return ErrorCode::EC_ERROR;
    }
    if (sched_plan_executor_ == nullptr) {
        KVCM_LOG_ERROR("schedule plan executor is nullptr");
        return ErrorCode::EC_ERROR;
    }
    if (metrics_registry_ == nullptr) {
        KVCM_LOG_ERROR("metrics registry is nullptr");
        return ErrorCode::EC_ERROR;
    }

    REGISTER_COUNTER_METRICS_FOR_CACHE_GC(scan_round_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_GC(orphaned_writing_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_GC(stale_serving_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_GC(location_submit_count);
    REGISTER_GAUGE_METRICS_FOR_CACHE_GC(scan_batch_duration_us);

    {
        std::unique_lock<std::mutex> lock(state_mutex_);
        if (running_) {
            KVCM_LOG_ERROR("cache garbage collector is already running");
            return ErrorCode::EC_EXIST;
        }
        running_ = true;
    }

    gc_thread_ = std::thread([this]() { this->GCScanLoop(); });
    KVCM_LOG_INFO("cache garbage collector start OK");
    return ErrorCode::EC_OK;
}

void CacheGarbageCollector::Stop() noexcept {
    {
        std::unique_lock<std::mutex> lock(state_mutex_);
        if (!running_) {
            return;
        }
        running_ = false;
        cv_state_.notify_one();
    }
    if (gc_thread_.joinable()) {
        gc_thread_.join();
    }
    KVCM_LOG_INFO("cache garbage collector stopped");
}

bool CacheGarbageCollector::IsRunning() const noexcept {
    std::unique_lock<std::mutex> lock(const_cast<std::mutex &>(state_mutex_));
    return running_;
}

void CacheGarbageCollector::Pause() noexcept { paused_.store(true); }

void CacheGarbageCollector::Resume() noexcept { paused_.store(false); }

bool CacheGarbageCollector::IsPaused() const noexcept { return paused_.load(); }

void CacheGarbageCollector::GCScanLoop() noexcept {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(state_mutex_);
            cv_state_.wait_for(lock, std::chrono::milliseconds(config_.inter_round_sleep_ms), [this]() {
                return !running_;
            });
            if (!running_) {
                return;
            }
        }

        if (paused_.load()) {
            continue;
        }

        auto request_context = std::make_shared<RequestContext>(GenTraceID());

        const auto [ec, instance_groups] = registry_manager_->ListInstanceGroup(request_context.get());
        if (ec != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("cache gc: list instance group failed, ec: [%d]", static_cast<std::int32_t>(ec));
            continue;
        }

        for (const auto &instance_group : instance_groups) {
            const auto [ec2, instance_infos] =
                registry_manager_->ListInstanceInfo(request_context.get(), instance_group->name());
            if (ec2 != ErrorCode::EC_OK) {
                continue;
            }

            for (const auto &instance_info : instance_infos) {
                if (!running_ || paused_.load()) {
                    break;
                }
                const std::string &instance_id = instance_info->instance_id();
                auto meta_indexer = meta_indexer_manager_->GetMetaIndexer(instance_id);
                if (!meta_indexer) {
                    continue;
                }
                ScanInstance(request_context.get(), instance_id, meta_indexer);
            }

            if (!running_) {
                return;
            }
        }

        ++cache_gc_scan_round_count_metrics_;
    }
}

void CacheGarbageCollector::ScanInstance(RequestContext *request_context,
                                         const std::string &instance_id,
                                         const std::shared_ptr<MetaIndexer> &meta_indexer) noexcept {
    std::string cursor = SCAN_BASE_CURSOR;

    do {
        if (!running_ || paused_.load()) {
            return;
        }

        const auto batch_start_us = TimestampUtil::GetCurrentTimeUs();

        std::string next_cursor;
        KeyVector keys;
        auto ec = meta_indexer->Scan(request_context, cursor, config_.scan_batch_size, next_cursor, keys);
        if (ec != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("cache gc: scan failed for instance [%s], ec: [%d]",
                          instance_id.c_str(),
                          static_cast<std::int32_t>(ec));
            return;
        }

        if (keys.empty()) {
            cursor = next_cursor;
            continue;
        }

        CacheLocationMapVector location_maps;
        auto get_result = meta_indexer->GetLocations(request_context, keys, location_maps);

        std::vector<std::int64_t> del_block_keys;
        std::vector<std::vector<std::string>> del_location_ids;
        std::size_t deletion_count = 0;

        for (std::size_t i = 0; i < keys.size() && i < location_maps.size(); ++i) {
            if (deletion_count >= config_.max_deletions_per_batch) {
                break;
            }

            std::vector<std::string> dirty_loc_ids;
            for (const auto &[loc_id, loc_ptr] : location_maps[i]) {
                if (!loc_ptr) {
                    continue;
                }
                if (IsOrphanedWriting(*loc_ptr)) {
                    dirty_loc_ids.push_back(loc_id);
                    ++cache_gc_orphaned_writing_count_metrics_;
                } else if (IsStaleServing(instance_id, *loc_ptr)) {
                    dirty_loc_ids.push_back(loc_id);
                    ++cache_gc_stale_serving_count_metrics_;
                }
            }

            if (!dirty_loc_ids.empty()) {
                del_block_keys.push_back(keys[i]);
                deletion_count += dirty_loc_ids.size();
                del_location_ids.push_back(std::move(dirty_loc_ids));
            }
        }

        if (!del_block_keys.empty()) {
            CacheLocationDelRequest del_request;
            del_request.instance_id = instance_id;
            del_request.block_keys = std::move(del_block_keys);
            del_request.location_ids = std::move(del_location_ids);
            sched_plan_executor_->SubmitNonBlocking(del_request);
            cache_gc_location_submit_count_metrics_ += deletion_count;
        }

        const auto batch_end_us = TimestampUtil::GetCurrentTimeUs();
        cache_gc_scan_batch_duration_us_metrics_ = static_cast<double>(batch_end_us - batch_start_us);

        cursor = next_cursor;

        // rate limiting sleep between batches
        if (cursor != SCAN_BASE_CURSOR && config_.inter_batch_sleep_ms > 0) {
            std::unique_lock<std::mutex> lock(state_mutex_);
            cv_state_.wait_for(lock, std::chrono::milliseconds(config_.inter_batch_sleep_ms), [this]() {
                return !running_;
            });
            if (!running_) {
                return;
            }
        }
    } while (cursor != SCAN_BASE_CURSOR);
}

bool CacheGarbageCollector::IsOrphanedWriting(const CacheLocation &loc) const noexcept {
    if (loc.status() != CacheLocationStatus::CLS_WRITING) {
        return false;
    }
    if (write_location_manager_ && write_location_manager_->HasLocationId(loc.id())) {
        return false;
    }
    if (loc.create_time() > 0) {
        const std::int64_t now_us = TimestampUtil::GetCurrentTimeUs();
        const std::int64_t age_us = now_us - loc.create_time();
        if (age_us < config_.writing_orphan_grace_period_us) {
            return false;
        }
    }
    return true;
}

bool CacheGarbageCollector::IsStaleServing(const std::string &instance_id, const CacheLocation &loc) const noexcept {
    if (loc.status() != CacheLocationStatus::CLS_SERVING) {
        return false;
    }
    if (!config_.check_serving_data_exist) {
        return false;
    }
    if (!registry_manager_ || !registry_manager_->data_storage_manager()) {
        return false;
    }

    std::vector<DataStorageUri> storage_uris;
    for (const auto &spec : loc.location_specs()) {
        if (const DataStorageUri uri{spec.uri()}; uri.Valid()) {
            storage_uris.emplace_back(uri);
        }
    }
    if (storage_uris.empty()) {
        return false;
    }

    std::string storage_unique_name = storage_uris.front().GetHostName();
    if (storage_uris.front().GetProtocol() == "vineyard") {
        storage_unique_name = VineyardStorageNameFromInstance(instance_id);
    }
    const auto result = registry_manager_->data_storage_manager()->Exist(storage_unique_name, storage_uris, false);
    return std::any_of(result.cbegin(), result.cend(), [](const bool v) -> bool { return !v; });
}

} // namespace kv_cache_manager
