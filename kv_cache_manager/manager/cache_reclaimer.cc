#include "kv_cache_manager/manager/cache_reclaimer.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/cache_reclaim_strategy.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_group_quota.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/migration_strategy.h"
#include "kv_cache_manager/config/quota_config.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/config/trigger_strategy.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/event/spec_events/cache_reclaim_event.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/manager/meta_searcher_manager.h"
#include "kv_cache_manager/manager/migration_manager.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/manager/write_location_manager.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace {

struct KeySamplingResult {
    kv_cache_manager::ErrorCode ec;
    std::shared_ptr<std::vector<std::int64_t>> keys;
    std::shared_ptr<std::vector<std::map<std::string, std::string>>> maps;
};

} // namespace

namespace kv_cache_manager {

#define LOG_WITH_TRACE(LEVEL, format, args...)                                                                         \
    do {                                                                                                               \
        KVCM_LOG_##LEVEL("trace_id [%s] | " format, request_context->trace_id().c_str(), ##args);                      \
    } while (0)

#define LOG_WITH_GR(LEVEL, format, args...)                                                                            \
    do {                                                                                                               \
        KVCM_LOG_##LEVEL("trace_id [%s] | instance_group [%s] | " format,                                              \
                         request_context->trace_id().c_str(),                                                          \
                         ins_gr.c_str(),                                                                               \
                         ##args);                                                                                      \
    } while (0)

#define LOG_WITH_ID(LEVEL, format, args...)                                                                            \
    do {                                                                                                               \
        KVCM_LOG_##LEVEL("trace_id [%s] | instance_id [%s] | instance_group [%s] | " format,                           \
                         request_context->trace_id().c_str(),                                                          \
                         ins_id.c_str(),                                                                               \
                         ins_gr.c_str(),                                                                               \
                         ##args);                                                                                      \
    } while (0)

#define INTERVAL_LOG_WITH_ID(LEVEL, interval, format, args...)                                                         \
    do {                                                                                                               \
        KVCM_INTERVAL_LOG_##LEVEL(interval,                                                                            \
                                  "trace_id [%s] | instance_id [%s] | instance_group [%s] | " format,                  \
                                  request_context->trace_id().c_str(),                                                 \
                                  ins_id.c_str(),                                                                      \
                                  ins_gr.c_str(),                                                                      \
                                  ##args);                                                                             \
    } while (0)

#define DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(name) DEFINE_METRICS_NAME_(CacheReclaimer, cache_reclaimer, name)

#define REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(name)                                                             \
    REGISTER_METRICS_COUNTER_(metrics_registry_, cache_reclaimer, name)

#define REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(name)                                                               \
    REGISTER_METRICS_GAUGE_(metrics_registry_, cache_reclaimer, name)

DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_cron_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_job_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(block_submit_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(location_submit_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(block_del_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(location_del_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(credit_timeout_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(pending_limit_reject_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(duplicate_pending_location_filtered_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_no_progress_backoff_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(delete_submit_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(delete_complete_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(delete_fail_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(migration_copy_submitted_total);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(migration_mark_submitted_total);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(fair_plan_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(fair_planned_batch_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(fair_planned_sample_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(fair_zero_weight_skip_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(fair_plan_truncated_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(fair_plan_truncated_instance_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(fair_item_capped_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(fair_sampling_size_normalized_count);

DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_cron_duration_us);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_quota_duration_us);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_job_duration_us);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_res_duration_us);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_lru_sample_duration_us);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_lru_batch_duration_us);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_lru_filter_duration_us);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_lru_submit_duration_us);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(pending_delete_handler_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(pending_location_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(pending_delete_bytes);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(credited_delete_bytes);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(predicted_deleted_key_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(oldest_pending_request_age_ms);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(fair_effective_instance_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(fair_planned_instance_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(fair_sampled_instance_count);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(fair_submitted_instance_count);

DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_batch_lru_age_min_us);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_batch_lru_age_max_us);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_batch_lru_age_avg_us);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_batch_create_age_min_us);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_batch_create_age_max_us);
DEFINE_METRICS_NAME_FOR_CACHE_RECLAIMER(reclaim_batch_create_age_avg_us);

void CacheReclaimer::AgeStats::Clear() {
    min_us = 0;
    max_us = 0;
    avg_us = 0;
}

const std::string CacheReclaimer::kTraceIDPrefix{"cache_reclaimer_internal_trace_"};

inline std::string CacheReclaimer::GenTraceID() {
    // generate a random 64-bit unsigned integer
    static std::random_device rd;
    static std::mt19937_64 rng(rd());
    static std::uniform_int_distribution<std::uint64_t> dis;

    const std::uint64_t rand_val = dis(rng);

    // convert to hexadecimal string representation
    std::stringstream ss;
    ss << kTraceIDPrefix << std::right << std::setfill('0') << std::setw(16) << std::hex << std::noshowbase << rand_val;

    return ss.str();
}

// instance group key & storage usage data
struct CacheReclaimer::GroupUsageData {
    std::size_t grp_used_key_cnt_;
    std::size_t grp_max_key_cnt_;
    std::size_t grp_used_byte_sz_;

    GroupUsageData();
    ~GroupUsageData() = default;

    [[nodiscard]] std::size_t GetGroupUsageByType(const DataStorageType &type) const noexcept;
    void AddGroupUsageByType(const DataStorageType &type, std::size_t value) noexcept;

private:
    using array_t_ = std::array<std::size_t, static_cast<std::size_t>(DataStorageType::COUNT)>;
    using size_t_ = array_t_::size_type;

    // group storage usage data array aggregated by storage type
    // slot 0: DATA_STORAGE_TYPE_UNKNOWN **UNUSED**
    // slot 1: DATA_STORAGE_TYPE_HF3FS usage data
    // slot 2: DATA_STORAGE_TYPE_MOONCAKE usage data
    // slot 3: DATA_STORAGE_TYPE_TAIR_MEMPOOL usage data
    // slot 4: DATA_STORAGE_TYPE_NFS usage data
    // slot 5: DATA_STORAGE_TYPE_VCNS_HF3FS **UNUSED** (merged into HF3FS)
    // slot 6: DATA_STORAGE_TYPE_DUMMY usage data (testing only)
    // slot 7: DATA_STORAGE_TYPE_EVENT_REPORT_L1P5 usage data
    // slot 8: DATA_STORAGE_TYPE_EVENT_REPORT_L2 usage data
    // slot 9: DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD usage data
    array_t_ grp_storage_usage_by_type_;
};

CacheReclaimer::GroupUsageData::GroupUsageData()
    : grp_used_key_cnt_(0), grp_max_key_cnt_(0), grp_used_byte_sz_(0), grp_storage_usage_by_type_{} {
    grp_storage_usage_by_type_.fill(0);
}

std::size_t CacheReclaimer::GroupUsageData::GetGroupUsageByType(const DataStorageType &type) const noexcept {
    const size_t_ idx = ToIndex(ToBaseType(type));
    if (idx >= grp_storage_usage_by_type_.size()) {
        KVCM_LOG_WARN("data storage type to index out of range, array size: [%zu], type as index: [%zu]",
                      grp_storage_usage_by_type_.size(),
                      idx);
        return 0;
    }
    return grp_storage_usage_by_type_.at(idx);
}

void CacheReclaimer::GroupUsageData::AddGroupUsageByType(const DataStorageType &type,
                                                         const std::size_t value) noexcept {
    const size_t_ idx = ToIndex(ToBaseType(type));
    if (idx >= grp_storage_usage_by_type_.size()) {
        KVCM_LOG_WARN("data storage type to index out of range, array size: [%zu], type as index: [%zu]",
                      grp_storage_usage_by_type_.size(),
                      idx);
        return;
    }
    grp_storage_usage_by_type_.at(idx) += value;
}

std::uint64_t CacheReclaimer::SaturatingSub(const std::uint64_t lhs, const std::uint64_t rhs) noexcept {
    return lhs > rhs ? lhs - rhs : 0;
}

std::uint64_t CacheReclaimer::SaturatingAdd(const std::uint64_t lhs, const std::uint64_t rhs) noexcept {
    const auto max = std::numeric_limits<std::uint64_t>::max();
    return rhs > max - lhs ? max : lhs + rhs;
}

std::uint64_t CacheReclaimer::GetLocationBytes(const CacheLocation &location) noexcept {
    std::uint64_t total = 0;
    for (const auto &location_spec : location.location_specs()) {
        DataStorageUri uri(location_spec.uri());
        if (!uri.Valid()) {
            continue;
        }
        std::uint64_t size = 0;
        uri.GetParamAs<std::uint64_t>("size", size);
        total = SaturatingAdd(total, size);
    }
    return total;
}

CacheReclaimer::BytesByStorageType
CacheReclaimer::GetCreditedDeleteBytes(const std::string &instance_group) const noexcept {
    const auto it = credited_delete_bytes_by_group_.find(instance_group);
    return it == credited_delete_bytes_by_group_.end() ? BytesByStorageType{} : it->second;
}

std::uint64_t CacheReclaimer::GetPredictedDeletedKeys(const std::string &instance_group) const noexcept {
    const auto it = predicted_deleted_keys_by_group_.find(instance_group);
    return it == predicted_deleted_keys_by_group_.end() ? 0 : it->second;
}

void CacheReclaimer::AddDeleteHandlerState(const DeleteHandler &handler) noexcept {
    for (const auto &pending_location : handler.pending_locations_) {
        const auto [_, inserted] = pending_locations_.insert(pending_location);
        if (!inserted) {
            KVCM_LOG_ERROR("duplicate pending location admitted, instance[%s] block[%ld] location[%s]",
                           pending_location.instance_id.c_str(),
                           pending_location.block_key,
                           pending_location.location_id.c_str());
        }
    }

    for (std::size_t idx = 0; idx < handler.bytes_by_type_.size(); ++idx) {
        const auto type = static_cast<DataStorageType>(idx);
        if (type != DataStorageType::DATA_STORAGE_TYPE_UNKNOWN &&
            type != DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS && handler.bytes_by_type_[idx] != 0) {
            auto &credited_bytes = credited_delete_bytes_by_group_[handler.ins_gr_];
            credited_bytes[idx] = SaturatingAdd(credited_bytes[idx], handler.bytes_by_type_[idx]);
        }
        if (handler.location_counts_by_type_[idx] == 0 && handler.bytes_by_type_[idx] == 0) {
            continue;
        }
        auto &pending_quota = pending_quota_by_group_type_[{handler.ins_gr_, type}];
        pending_quota.location_count =
            SaturatingAdd(pending_quota.location_count, handler.location_counts_by_type_[idx]);
        pending_quota.bytes = SaturatingAdd(pending_quota.bytes, handler.bytes_by_type_[idx]);
    }
    if (handler.predicted_deleted_keys_ != 0) {
        auto &predicted_keys = predicted_deleted_keys_by_group_[handler.ins_gr_];
        predicted_keys = SaturatingAdd(predicted_keys, handler.predicted_deleted_keys_);
    }
    pending_delete_handler_count_ = SaturatingAdd(pending_delete_handler_count_, 1);
    for (const auto bytes : handler.bytes_by_type_) {
        pending_delete_bytes_ = SaturatingAdd(pending_delete_bytes_, bytes);
    }
}

void CacheReclaimer::DisableCredit(DeleteHandler &handler, const char *reason) noexcept {
    if (!handler.credit_enabled_) {
        return;
    }

    auto credited_it = credited_delete_bytes_by_group_.find(handler.ins_gr_);
    if (credited_it != credited_delete_bytes_by_group_.end()) {
        bool all_zero = true;
        for (std::size_t idx = 0; idx < credited_it->second.size(); ++idx) {
            credited_it->second[idx] = SaturatingSub(credited_it->second[idx], handler.bytes_by_type_[idx]);
            all_zero = all_zero && credited_it->second[idx] == 0;
        }
        if (all_zero) {
            credited_delete_bytes_by_group_.erase(credited_it);
        }
    }

    auto predicted_it = predicted_deleted_keys_by_group_.find(handler.ins_gr_);
    if (predicted_it != predicted_deleted_keys_by_group_.end()) {
        predicted_it->second = SaturatingSub(predicted_it->second, handler.predicted_deleted_keys_);
        if (predicted_it->second == 0) {
            predicted_deleted_keys_by_group_.erase(predicted_it);
        }
    }

    handler.credit_enabled_ = false;
    if (reason != nullptr) {
        const auto &request_context = handler.req_ctx_;
        const std::string &ins_id = handler.ins_id_;
        const std::string &ins_gr = handler.ins_gr_;
        LOG_WITH_ID(WARN,
                    "disable in-flight delete credit, reason: [%s], block count: [%" PRIu64
                    "], location count: [%" PRIu64 "]",
                    reason,
                    handler.blk_count_,
                    handler.loc_count_);
    }
}

void CacheReclaimer::ReleaseDeleteHandlerState(DeleteHandler &handler) noexcept {
    DisableCredit(handler, nullptr);

    for (const auto &pending_location : handler.pending_locations_) {
        pending_locations_.erase(pending_location);
    }
    for (std::size_t idx = 0; idx < handler.location_counts_by_type_.size(); ++idx) {
        if (handler.location_counts_by_type_[idx] == 0 && handler.bytes_by_type_[idx] == 0) {
            continue;
        }
        const auto key = GroupStorageTypeKey{handler.ins_gr_, static_cast<DataStorageType>(idx)};
        auto pending_it = pending_quota_by_group_type_.find(key);
        if (pending_it == pending_quota_by_group_type_.end()) {
            continue;
        }
        pending_it->second.location_count =
            SaturatingSub(pending_it->second.location_count, handler.location_counts_by_type_[idx]);
        pending_it->second.bytes = SaturatingSub(pending_it->second.bytes, handler.bytes_by_type_[idx]);
        if (pending_it->second.location_count == 0 && pending_it->second.bytes == 0) {
            pending_quota_by_group_type_.erase(pending_it);
        }
    }
    pending_delete_handler_count_ = SaturatingSub(pending_delete_handler_count_, 1);
    for (const auto bytes : handler.bytes_by_type_) {
        pending_delete_bytes_ = SaturatingSub(pending_delete_bytes_, bytes);
    }
}

void CacheReclaimer::RecordPendingLimitReject(const std::string &instance_group,
                                              const std::string &storage_type_scope) noexcept {
    METRICS_(cache_reclaimer, pending_limit_reject_count) += 1;
    if (metrics_registry_ == nullptr) {
        return;
    }
    try {
        metrics_registry_->GetCounter(
            METRICS_NAME_(cache_reclaimer, pending_limit_reject_count),
            MetricsTags{{"instance_group", instance_group}, {"storage_type", storage_type_scope}}) += 1;
    } catch (const std::exception &e) {
        KVCM_LOG_WARN("record tagged pending-limit metric failed: %s", e.what());
    } catch (...) { KVCM_LOG_WARN("record tagged pending-limit metric failed with unknown exception"); }
}

void CacheReclaimer::UpdateAsyncDeleteMetrics() noexcept {
    std::uint64_t credited_bytes = 0;
    for (const auto &[_, by_type] : credited_delete_bytes_by_group_) {
        for (const auto bytes : by_type) {
            credited_bytes = SaturatingAdd(credited_bytes, bytes);
        }
    }
    std::uint64_t predicted_keys = 0;
    for (const auto &[_, keys] : predicted_deleted_keys_by_group_) {
        predicted_keys = SaturatingAdd(predicted_keys, keys);
    }

    std::uint64_t oldest_age_ms = 0;
    const auto now = std::chrono::steady_clock::now();
    for (const auto &handler : delete_handlers_) {
        const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - handler.submitted_at_).count();
        if (age > 0) {
            oldest_age_ms = std::max(oldest_age_ms, static_cast<std::uint64_t>(age));
        }
    }

    METRICS_(cache_reclaimer, pending_delete_handler_count) = static_cast<double>(pending_delete_handler_count_);
    METRICS_(cache_reclaimer, pending_location_count) = static_cast<double>(pending_locations_.size());
    METRICS_(cache_reclaimer, pending_delete_bytes) = static_cast<double>(pending_delete_bytes_);
    METRICS_(cache_reclaimer, credited_delete_bytes) = static_cast<double>(credited_bytes);
    METRICS_(cache_reclaimer, predicted_deleted_key_count) = static_cast<double>(predicted_keys);
    METRICS_(cache_reclaimer, oldest_pending_request_age_ms) = static_cast<double>(oldest_age_ms);

    if (metrics_registry_ == nullptr) {
        return;
    }

    std::set<GroupStorageTypeKey> current_group_type_keys;
    for (const auto &[instance_group, by_type] : credited_delete_bytes_by_group_) {
        for (std::size_t idx = 0; idx < by_type.size(); ++idx) {
            if (by_type[idx] != 0) {
                current_group_type_keys.insert(GroupStorageTypeKey{instance_group, static_cast<DataStorageType>(idx)});
            }
        }
    }
    for (const auto &[key, quota] : pending_quota_by_group_type_) {
        if (quota.location_count != 0 || quota.bytes != 0) {
            current_group_type_keys.insert(key);
        }
    }

    auto group_type_keys_to_report = reported_group_type_metric_keys_;
    group_type_keys_to_report.insert(current_group_type_keys.begin(), current_group_type_keys.end());
    for (const auto &key : group_type_keys_to_report) {
        const auto credited_it = credited_delete_bytes_by_group_.find(key.instance_group);
        const auto type_idx = ToIndex(key.storage_type);
        const std::uint64_t group_type_credit =
            credited_it != credited_delete_bytes_by_group_.end() && type_idx < credited_it->second.size()
                ? credited_it->second[type_idx]
                : 0;
        const auto quota_it = pending_quota_by_group_type_.find(key);
        const PendingQuota quota = quota_it == pending_quota_by_group_type_.end() ? PendingQuota{} : quota_it->second;
        const MetricsTags tags{{"instance_group", key.instance_group}, {"storage_type", ToString(key.storage_type)}};
        metrics_registry_->GetGauge(METRICS_NAME_(cache_reclaimer, credited_delete_bytes), tags) =
            static_cast<double>(group_type_credit);
        metrics_registry_->GetGauge(METRICS_NAME_(cache_reclaimer, pending_location_count), tags) =
            static_cast<double>(quota.location_count);
        metrics_registry_->GetGauge(METRICS_NAME_(cache_reclaimer, pending_delete_bytes), tags) =
            static_cast<double>(quota.bytes);
    }
    reported_group_type_metric_keys_ = std::move(current_group_type_keys);

    std::set<std::string> current_group_keys;
    for (const auto &[instance_group, keys] : predicted_deleted_keys_by_group_) {
        if (keys != 0) {
            current_group_keys.insert(instance_group);
        }
    }
    auto group_keys_to_report = reported_group_metric_keys_;
    group_keys_to_report.insert(current_group_keys.begin(), current_group_keys.end());
    for (const auto &instance_group : group_keys_to_report) {
        const auto predicted_it = predicted_deleted_keys_by_group_.find(instance_group);
        const std::uint64_t group_predicted_keys =
            predicted_it == predicted_deleted_keys_by_group_.end() ? 0 : predicted_it->second;
        metrics_registry_->GetGauge(METRICS_NAME_(cache_reclaimer, predicted_deleted_key_count),
                                    MetricsTags{{"instance_group", instance_group}}) =
            static_cast<double>(group_predicted_keys);
    }
    reported_group_metric_keys_ = std::move(current_group_keys);
}

CacheReclaimer::WaterLevelExceed::WaterLevelExceed()
    : group_bytes_water_level_exceed_(false), group_keys_water_level_exceed_(false), water_level_exceed_by_type_{} {
    water_level_exceed_by_type_.fill(false);
}

bool CacheReclaimer::WaterLevelExceed::GetGeneralWaterLevelExceed() const noexcept {
    return group_bytes_water_level_exceed_ || group_keys_water_level_exceed_;
}

bool CacheReclaimer::WaterLevelExceed::GetGroupBytesWaterLevelExceed() const noexcept {
    return group_bytes_water_level_exceed_;
}

bool CacheReclaimer::WaterLevelExceed::GetGroupKeysWaterLevelExceed() const noexcept {
    return group_keys_water_level_exceed_;
}

bool CacheReclaimer::WaterLevelExceed::GetWaterLevelExceedByType(const DataStorageType &type) const noexcept {
    const size_t_ idx = ToIndex(ToBaseType(type));
    if (idx >= water_level_exceed_by_type_.size()) {
        KVCM_LOG_WARN("data storage type to index out of range, array size: [%zu], type as index: [%zu]",
                      water_level_exceed_by_type_.size(),
                      idx);
        return false;
    }
    return water_level_exceed_by_type_.at(idx);
}

void CacheReclaimer::WaterLevelExceed::SetGroupBytesWaterLevelExceed(const bool value) noexcept {
    group_bytes_water_level_exceed_ = value;
}

void CacheReclaimer::WaterLevelExceed::SetGroupKeysWaterLevelExceed(const bool value) noexcept {
    group_keys_water_level_exceed_ = value;
}

void CacheReclaimer::WaterLevelExceed::SetWaterLevelExceedByType(const DataStorageType &type,
                                                                 const bool value) noexcept {
    const size_t_ idx = ToIndex(ToBaseType(type));
    if (idx >= water_level_exceed_by_type_.size()) {
        KVCM_LOG_WARN("data storage type to index out of range, array size: [%zu], type as index: [%zu]",
                      water_level_exceed_by_type_.size(),
                      idx);
        return;
    }
    water_level_exceed_by_type_.at(idx) = value;
}

bool CacheReclaimer::WaterLevelExceed::CheckGroupWaterLevelExceed() const noexcept {
    return GetGeneralWaterLevelExceed() || CheckStorageTypeWaterLevelExceed();
}

bool CacheReclaimer::WaterLevelExceed::CheckStorageTypeWaterLevelExceed() const noexcept {
    for (size_t_ i = 0; i != water_level_exceed_by_type_.size(); ++i) {
        if (i == static_cast<size_t_>(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN) ||
            i == static_cast<size_t_>(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS)) {
            continue;
        }
        if (water_level_exceed_by_type_.at(i)) {
            return true;
        }
    }
    return false;
}

CacheReclaimer::CacheReclaimer(const std::size_t sampling_size_total,
                               const std::size_t sampling_size_per_task,
                               const std::size_t batching_size,
                               const std::uint32_t sleep_interval_ms,
                               std::uint32_t worker_size,
                               std::shared_ptr<RegistryManager> registry_manager,
                               std::shared_ptr<MetaIndexerManager> meta_indexer_manager,
                               std::shared_ptr<MetaSearcherManager> meta_searcher_manager,
                               std::shared_ptr<SchedulePlanExecutor> sched_plan_executor,
                               std::shared_ptr<MetricsRegistry> metrics_registry,
                               std::shared_ptr<EventManager> event_manager,
                               std::shared_ptr<WriteLocationManager> write_location_manager,
                               CacheReclaimerAsyncDeleteConfig async_delete_config,
                               std::shared_ptr<MigrationManager> migration_manager)
    : registry_manager_(std::move(registry_manager))
    , meta_indexer_manager_(std::move(meta_indexer_manager))
    , meta_searcher_manager_(std::move(meta_searcher_manager))
    , sched_plan_executor_(std::move(sched_plan_executor))
    , metrics_registry_(std::move(metrics_registry))
    , event_manager_(std::move(event_manager))
    , write_location_manager_(std::move(write_location_manager))
    , migration_manager_(std::move(migration_manager))
    , job_state_flag_(false)
    , pause_flag_(false)
    , sampling_size_(sampling_size_total)
    , sampling_size_per_task_(sampling_size_per_task)
    , batching_size_(batching_size)
    , sleep_interval_ms_(sleep_interval_ms)
    , future_timeout_ms_(kFutureTimeoutMs)
    , async_delete_config_(std::move(async_delete_config))
    , worker_stop_(false) {
    if (worker_size == 0) {
        worker_size = 1;
    }
    for (std::uint32_t i = 0; i != worker_size; ++i) {
        workers_.emplace_back([this] { WorkerRoutine(); });
    }
    KVCM_LOG_INFO("cache reclaimer initialized with [%u] worker(s)", worker_size);
}

CacheReclaimer::~CacheReclaimer() {
    this->Stop();

    // stop workers
    worker_stop_ = true;
    cv_task_queue_.notify_all();

    for (auto &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

ErrorCode CacheReclaimer::Start() noexcept {
    if (registry_manager_ == nullptr) {
        KVCM_LOG_ERROR("registry manager is nullptr");
        return ErrorCode::EC_ERROR;
    }

    if (meta_indexer_manager_ == nullptr) {
        KVCM_LOG_ERROR("meta indexer manager is nullptr");
        return ErrorCode::EC_ERROR;
    }

    if (meta_searcher_manager_ == nullptr) {
        KVCM_LOG_ERROR("meta searcher manager is nullptr");
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

    // allow event_manager_ to be nullptr

    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(reclaim_cron_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(reclaim_job_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(block_submit_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(location_submit_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(block_del_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(location_del_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(credit_timeout_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(pending_limit_reject_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(duplicate_pending_location_filtered_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(reclaim_no_progress_backoff_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(delete_submit_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(delete_complete_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(delete_fail_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(migration_copy_submitted_total);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(migration_mark_submitted_total);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(fair_plan_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(fair_planned_batch_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(fair_planned_sample_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(fair_zero_weight_skip_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(fair_plan_truncated_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(fair_plan_truncated_instance_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(fair_item_capped_count);
    REGISTER_COUNTER_METRICS_FOR_CACHE_RECLAIMER(fair_sampling_size_normalized_count);

    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(reclaim_cron_duration_us);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(reclaim_quota_duration_us);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(reclaim_job_duration_us);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(reclaim_res_duration_us);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(reclaim_lru_sample_duration_us);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(reclaim_lru_batch_duration_us);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(reclaim_lru_filter_duration_us);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(reclaim_lru_submit_duration_us);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(pending_delete_handler_count);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(pending_location_count);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(pending_delete_bytes);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(credited_delete_bytes);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(predicted_deleted_key_count);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(oldest_pending_request_age_ms);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(fair_effective_instance_count);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(fair_planned_instance_count);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(fair_sampled_instance_count);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(fair_submitted_instance_count);

    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(reclaim_batch_lru_age_min_us);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(reclaim_batch_lru_age_max_us);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(reclaim_batch_lru_age_avg_us);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(reclaim_batch_create_age_min_us);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(reclaim_batch_create_age_max_us);
    REGISTER_GAUGE_METRICS_FOR_CACHE_RECLAIMER(reclaim_batch_create_age_avg_us);

    {
        std::unique_lock<std::mutex> lock(job_state_mutex_);
        if (job_state_flag_
            // there is no need to check on reclaimer_.joinable(), because
            // under our usage model assumption, there will be no parallel
            // calls to this->Start() and this->Stop(); any previous
            // this->Stop() call would block the calling thread to wait for
            // its finishing, and synchronise with the completion of the
            // working thread
        ) {
            KVCM_LOG_ERROR("cannot start new reclaiming job; there is already a running one");
            return ErrorCode::EC_EXIST;
        }
        job_state_flag_ = true;
    }

    reclaimer_ = std::thread([this]() -> void { this->ReclaimCron(); });
    KVCM_LOG_INFO("cache reclaimer start OK");
    return ErrorCode::EC_OK;
}

void CacheReclaimer::Stop() noexcept {
    {
        std::unique_lock<std::mutex> lock(job_state_mutex_);
        if (job_state_flag_) {
            job_state_flag_ = false;
            cv_job_state_.notify_one();
        }
    }

    if (reclaimer_.joinable()) {
        reclaimer_.join();
    }

    KVCM_LOG_DEBUG("cache reclaimer stop OK");
}

bool CacheReclaimer::IsRunning() noexcept {
    std::unique_lock<std::mutex> lock(job_state_mutex_);
    return job_state_flag_;
}

void CacheReclaimer::Pause() noexcept { pause_flag_.store(true); }

void CacheReclaimer::Resume() noexcept { pause_flag_.store(false); }

bool CacheReclaimer::IsPaused() const noexcept { return pause_flag_.load(); }

std::size_t CacheReclaimer::GetSamplingSize(const RequestContext *request_context) const noexcept {
    const std::size_t sampling_size = sampling_size_.load();
    LOG_WITH_TRACE(DEBUG, "sampling size is [%zu]", sampling_size);
    return sampling_size;
}

ErrorCode CacheReclaimer::SetSamplingSize(const RequestContext *request_context,
                                          const std::size_t sampling_size) noexcept {
    if (sampling_size >= kSizeLimit) {
        LOG_WITH_TRACE(
            ERROR, "set sampling size failed: sampling_size [%zu] >= kSizeLimit [%zu]", sampling_size, kSizeLimit);
        return ErrorCode::EC_OUT_OF_RANGE;
    }
    sampling_size_.store(sampling_size);
    LOG_WITH_TRACE(DEBUG, "set sampling size [%zu]", sampling_size);
    return ErrorCode::EC_OK;
}

std::size_t CacheReclaimer::GetBatchingSize(const RequestContext *request_context) const noexcept {
    const std::size_t batching_size = batching_size_.load();
    LOG_WITH_TRACE(DEBUG, "batching size is [%zu]", batching_size);
    return batching_size;
}

ErrorCode CacheReclaimer::SetBatchingSize(const RequestContext *request_context,
                                          const std::size_t batching_size) noexcept {
    if (batching_size >= kSizeLimit) {
        LOG_WITH_TRACE(
            ERROR, "set batching size failed: batching_size [%zu] >= kSizeLimit [%zu]", batching_size, kSizeLimit);
        return ErrorCode::EC_OUT_OF_RANGE;
    }
    batching_size_.store(batching_size);
    LOG_WITH_TRACE(DEBUG, "set batching size [%zu]", batching_size);
    return ErrorCode::EC_OK;
}

std::uint32_t CacheReclaimer::GetSleepIntervalMs(const RequestContext *request_context) const noexcept {
    const std::uint32_t sleep_interval_ms = sleep_interval_ms_.load();
    LOG_WITH_TRACE(DEBUG, "sleep interval is [%u] ms", sleep_interval_ms);
    return sleep_interval_ms;
}

void CacheReclaimer::SetSleepIntervalMs(const RequestContext *request_context,
                                        const std::uint32_t sleep_interval_ms) noexcept {
    sleep_interval_ms_.store(sleep_interval_ms);
    LOG_WITH_TRACE(DEBUG, "set sleep interval to [%u] ms", sleep_interval_ms);
}

std::shared_ptr<CacheReclaimer::WaterLevelExceed>
CacheReclaimer::GetWaterLevelExceed(const RequestContext *request_context,
                                    const std::string &ins_gr,
                                    const InstanceGroupQuota &instance_group_quota,
                                    const std::shared_ptr<CacheReclaimStrategy> &reclaim_strategy,
                                    const std::vector<std::shared_ptr<const InstanceInfo>> &instance_infos) noexcept {
    if (!IsRunning() || IsPaused()) {
        // fast exiting in the middle of one job round
        return nullptr;
    }

    // TODO (rui): per storage backend reclaiming is not supported due
    //             to lacking of capacity and usage info

    // NOTE: reclaim_strategy->storage_unique_name() is ignored
    //       reclaim_strategy->trigger_strategy().used_size() is ignored

    // NOTE: the trigger detecting strategy is based on:
    //       1. the entire instance group usage and capacity quota
    //       2. storage type usage and capacity quota for this group

    const auto water_level_exceed = std::make_shared<WaterLevelExceed>();

    // 1. calculate the key count and used byte size of this group
    const auto data = GetGroupUsageData(request_context, instance_infos);
    if (data == nullptr) {
        LOG_WITH_GR(ERROR, "group usage data is nullptr");
        return nullptr;
    }

    const auto credited_bytes = GetCreditedDeleteBytes(ins_gr);
    std::uint64_t total_credited_bytes = 0;
    for (const auto bytes : credited_bytes) {
        total_credited_bytes = SaturatingAdd(total_credited_bytes, bytes);
    }
    const std::uint64_t effective_group_bytes =
        SaturatingSub(static_cast<std::uint64_t>(data->grp_used_byte_sz_), total_credited_bytes);
    const std::uint64_t effective_group_keys =
        SaturatingSub(static_cast<std::uint64_t>(data->grp_used_key_cnt_), GetPredictedDeletedKeys(ins_gr));

    // 2. generate the result water level exceeding array
    const double threshold_used_percentage = reclaim_strategy->trigger_strategy().used_percentage();

    // 2.1. results for each storage type
    for (const auto &storage_quota : instance_group_quota.quota_config()) {
        const auto &type = storage_quota.storage_spec();
        if (type == DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS) {
            // skip vcns_hf3fs, because it is treated as hf3fs
            continue;
        }
        if (IsEventReportStorageType(type)) {
            // EventReport metadata is not reclaimed by CacheReclaimer.
            continue;
        }

        const std::size_t type_idx = ToIndex(ToBaseType(type));
        const std::uint64_t type_credit = type_idx < credited_bytes.size() ? credited_bytes[type_idx] : 0;
        const std::uint64_t effective_type_bytes =
            SaturatingSub(static_cast<std::uint64_t>(data->GetGroupUsageByType(type)), type_credit);

        if (effective_type_bytes == 0) {
            continue;
        }

        if (storage_quota.capacity() <= 0) {
            LOG_WITH_GR(DEBUG,
                        "instance group storage type [%d] capacity quota used percentage [inf] "
                        "has reached or exceeded the threshold percentage [%f]",
                        static_cast<std::uint8_t>(type),
                        threshold_used_percentage);
            water_level_exceed->SetWaterLevelExceedByType(type, true);
            continue;
        }
        if (const double storage_type_wl =
                static_cast<double>(effective_type_bytes) / static_cast<double>(storage_quota.capacity());
            storage_type_wl + kEpsilon > threshold_used_percentage) {
            LOG_WITH_GR(DEBUG,
                        "instance group storage type [%d] capacity quota used percentage [%f] "
                        "has reached or exceeded the threshold percentage [%f]",
                        static_cast<std::uint8_t>(type),
                        storage_type_wl,
                        threshold_used_percentage);
            water_level_exceed->SetWaterLevelExceedByType(type, true);
        }
    }

    // 2.2. result for the entire instance group. Byte and key-count
    // credits are independent: credit reducing one dimension to zero
    // must not suppress the other dimension's water-level check.

    // 2.2.1. trigger_strategy:used_percent for instance group quota capacity
    if (effective_group_bytes > 0) {
        if (instance_group_quota.capacity() <= 0) {
            // proceed as group quota capacity is 0
            LOG_WITH_GR(DEBUG,
                        "instance group capacity quota used percentage [inf] "
                        "has reached or exceeded the threshold percentage [%f]",
                        threshold_used_percentage);
            water_level_exceed->SetGroupBytesWaterLevelExceed(true);
        } else if (const double group_used_percentage = static_cast<double>(effective_group_bytes) /
                                                        static_cast<double>(instance_group_quota.capacity());
                   group_used_percentage + kEpsilon > threshold_used_percentage) {
            LOG_WITH_GR(DEBUG,
                        "instance group capacity quota used percentage [%f] "
                        "has reached or exceeded the threshold percentage [%f]",
                        group_used_percentage,
                        threshold_used_percentage);
            water_level_exceed->SetGroupBytesWaterLevelExceed(true);
        }
    }

    // 2.2.2. trigger_strategy:used_percent for group key count
    if (effective_group_keys > 0) {
        if (data->grp_max_key_cnt_ == 0) {
            LOG_WITH_GR(DEBUG,
                        "instance group total key count used percentage [inf] "
                        "has reached or exceeded the threshold percentage [%f]",
                        threshold_used_percentage);
            water_level_exceed->SetGroupKeysWaterLevelExceed(true);
        } else if (const double group_used_percentage =
                       static_cast<double>(effective_group_keys) / static_cast<double>(data->grp_max_key_cnt_);
                   group_used_percentage + kEpsilon > threshold_used_percentage) {
            LOG_WITH_GR(DEBUG,
                        "instance group total key count used percentage [%f] "
                        "has reached or exceeded the threshold percentage [%f]",
                        group_used_percentage,
                        threshold_used_percentage);
            water_level_exceed->SetGroupKeysWaterLevelExceed(true);
        }
    }

    return water_level_exceed;
}

bool CacheReclaimer::IsTriggerReclaiming(const std::shared_ptr<WaterLevelExceed> &water_level_exceed) {
    if (water_level_exceed == nullptr || !water_level_exceed->CheckGroupWaterLevelExceed()) {
        return false;
    }
    return true;
}

bool CacheReclaimer::ReclaimByLRU(const std::shared_ptr<RequestContext> &request_context,
                                  const InstanceInfoConstPtr &instance_info,
                                  const WaterLevelExceed &water_level_exceed,
                                  const std::int32_t delay_before_delete_ms) noexcept {
    return ReclaimByLRUImpl(request_context, instance_info, water_level_exceed, delay_before_delete_ms, false, 0, 0);
}

bool CacheReclaimer::ReclaimByLRUWithBudget(const std::shared_ptr<RequestContext> &request_context,
                                            const InstanceInfoConstPtr &instance_info,
                                            const WaterLevelExceed &water_level_exceed,
                                            const std::int32_t delay_before_delete_ms,
                                            const std::size_t sampling_size,
                                            const std::size_t batching_size) noexcept {
    return ReclaimByLRUImpl(
        request_context, instance_info, water_level_exceed, delay_before_delete_ms, true, sampling_size, batching_size);
}

bool CacheReclaimer::ReclaimByLRUImpl(const std::shared_ptr<RequestContext> &request_context,
                                      const InstanceInfoConstPtr &instance_info,
                                      const WaterLevelExceed &water_level_exceed,
                                      const std::int32_t delay_before_delete_ms,
                                      const bool use_fair_budget,
                                      const std::size_t sampling_size,
                                      const std::size_t batching_size) noexcept {
    if (!IsRunning() || IsPaused()) {
        // fast exiting in the middle of one job round
        return false;
    }

    if (instance_info == nullptr) {
        LOG_WITH_TRACE(WARN, "instance is nullptr");
        return false;
    }

    const std::string &ins_id = instance_info->instance_id();
    const std::string &ins_gr = instance_info->instance_group_name();

    std::vector<std::int64_t> keys;
    std::vector<std::map<std::string, std::string>> maps;

    // 1. get the sampled block keys and the LRU timestamp info from
    // the meta indexer
    const std::int64_t begin_tp_sample = TimestampUtil::GetSteadyTimeUs();
    const bool sampled = use_fair_budget
                             ? DoKeySamplingWithSize(request_context, instance_info, sampling_size, true, keys, maps)
                             : DoKeySampling(request_context, instance_info, keys, maps);
    if (!sampled) {
        LOG_WITH_ID(DEBUG, "key sampling failed");
        return false;
    }
    METRICS_(cache_reclaimer, reclaim_lru_sample_duration_us) =
        static_cast<double>(TimestampUtil::GetSteadyTimeUs() - begin_tp_sample);
    LOG_WITH_ID(DEBUG, "[%zu] key(s) sampled", keys.size());

    // init the deleting request with content to be filled later
    // here the cache location form of deleting request is used to
    // permit the cache location status aware deleting control
    CacheLocationDelRequest request;
    request.instance_id = ins_id;
    request.delay = std::chrono::milliseconds(delay_before_delete_ms);

    // 2. constitute the batch based on the LRU timestamp info
    const std::int64_t begin_tp_batch = TimestampUtil::GetSteadyTimeUs();
    AgeStats lru_age_stats;
    const bool batch_made =
        use_fair_budget
            ? MakeBatchByLRUWithSize(
                  request_context.get(), instance_info, keys, maps, batching_size, request.block_keys, lru_age_stats)
            : MakeBatchByLRU(request_context.get(), instance_info, keys, maps, request.block_keys, lru_age_stats);
    if (!batch_made) {
        LOG_WITH_ID(DEBUG, "make batch failed");
        return false;
    }
    METRICS_(cache_reclaimer, reclaim_lru_batch_duration_us) =
        static_cast<double>(TimestampUtil::GetSteadyTimeUs() - begin_tp_batch);
    LOG_WITH_ID(DEBUG, "batch is made with size [%zu]", request.block_keys.size());
    if (request.block_keys.empty()) {
        return false;
    }
    METRICS_(cache_reclaimer, reclaim_batch_lru_age_min_us) = static_cast<double>(lru_age_stats.min_us);
    METRICS_(cache_reclaimer, reclaim_batch_lru_age_max_us) = static_cast<double>(lru_age_stats.max_us);
    METRICS_(cache_reclaimer, reclaim_batch_lru_age_avg_us) = static_cast<double>(lru_age_stats.avg_us);

    // 3. inspect the cache location status for every blocks so that:
    //    a) cache locations in CLS_SERVING status
    //    b) cache locations in CLS_WRITING status *and* is orphaned
    //    are submitted to be deleted
    const std::int64_t begin_tp_filter = TimestampUtil::GetSteadyTimeUs();
    AgeStats create_age_stats;
    BytesByStorageType bytes_by_type{};
    CountsByStorageType location_counts_by_type{};
    std::uint64_t predicted_deleted_keys = 0;
    if (!FilterLocID(request_context.get(),
                     instance_info,
                     request.block_keys,
                     water_level_exceed,
                     request.location_ids,
                     bytes_by_type,
                     location_counts_by_type,
                     predicted_deleted_keys,
                     create_age_stats)) {
        LOG_WITH_ID(DEBUG, "filter location ID failed");
        return false;
    }
    METRICS_(cache_reclaimer, reclaim_lru_filter_duration_us) =
        static_cast<double>(TimestampUtil::GetSteadyTimeUs() - begin_tp_filter);
    METRICS_(cache_reclaimer, reclaim_batch_create_age_min_us) = static_cast<double>(create_age_stats.min_us);
    METRICS_(cache_reclaimer, reclaim_batch_create_age_max_us) = static_cast<double>(create_age_stats.max_us);
    METRICS_(cache_reclaimer, reclaim_batch_create_age_avg_us) = static_cast<double>(create_age_stats.avg_us);

    // 4. submit the final deleting request to the executor
    const std::int64_t begin_tp_submit = TimestampUtil::GetSteadyTimeUs();
    const bool submitted = SubmitDelReq(
        request_context, instance_info, request, bytes_by_type, location_counts_by_type, predicted_deleted_keys);
    METRICS_(cache_reclaimer, reclaim_lru_submit_duration_us) =
        static_cast<double>(TimestampUtil::GetSteadyTimeUs() - begin_tp_submit);

    if (submitted) {
        METRICS_(cache_reclaimer, reclaim_job_count) += 1;
    }
    return submitted;
}

bool CacheReclaimer::ReclaimByLFU(const std::shared_ptr<RequestContext> &request_context,
                                  const InstanceInfoConstPtr &instance_info,
                                  const WaterLevelExceed &water_level_exceed,
                                  const std::int32_t delay_before_delete_ms) noexcept {
    if (!IsRunning() || IsPaused()) {
        // fast exiting in the middle of one job round
        return false;
    }

    // TODO: impl LFU policy
    LOG_WITH_TRACE(WARN, "LFU reclaim policy not supported yet; fall back to LRU policy");
    return ReclaimByLRU(request_context, instance_info, water_level_exceed, delay_before_delete_ms);
}

bool CacheReclaimer::ReclaimByTTL(const std::shared_ptr<RequestContext> &request_context,
                                  const InstanceInfoConstPtr &instance_info,
                                  const WaterLevelExceed &water_level_exceed,
                                  const std::int32_t delay_before_delete_ms) noexcept {
    if (!IsRunning() || IsPaused()) {
        // fast exiting in the middle of one job round
        return false;
    }

    // TODO: impl TTL policy
    LOG_WITH_TRACE(WARN, "TTL reclaim policy not supported yet; fall back to LRU policy");
    return ReclaimByLRU(request_context, instance_info, water_level_exceed, delay_before_delete_ms);
}

void CacheReclaimer::ReclaimCron() noexcept {
    std::uint32_t sleep_interval_ms = sleep_interval_ms_.load();
    while (true) {
        const std::int64_t begin_tp = TimestampUtil::GetSteadyTimeUs();

        {
            std::unique_lock<std::mutex> lock(job_state_mutex_);
            if (!job_state_flag_) {
                // prevent unnecessary sleeping
                break;
            }

            cv_job_state_.wait_for(lock, std::chrono::milliseconds(sleep_interval_ms));
            if (!job_state_flag_) {
                break;
            }
        }

        {
            const std::int64_t res_begin_tp = TimestampUtil::GetSteadyTimeUs();
            HandleDelRes();
            METRICS_(cache_reclaimer, reclaim_res_duration_us) =
                static_cast<double>(TimestampUtil::GetSteadyTimeUs() - res_begin_tp);
        }

        if (IsPaused()) {
            sleep_interval_ms = sleep_interval_ms_.load();
            continue;
        }

        const auto request_context = std::make_shared<RequestContext>(GenTraceID());

        const auto [ec, instance_groups] = registry_manager_->ListInstanceGroup(request_context.get());
        if (ec != ErrorCode::EC_OK) {
            LOG_WITH_TRACE(WARN, "list instance group failed, error code: [%d]", static_cast<std::int32_t>(ec));
            sleep_interval_ms = sleep_interval_ms_.load();
            continue;
        }

        bool made_progress = false;
        bool needs_no_progress_backoff = false;
        for (const auto &instance_group : instance_groups) {
            const auto result = TryReclaimOnGroup(request_context, instance_group);
            made_progress = made_progress || result.made_progress;
            needs_no_progress_backoff =
                needs_no_progress_backoff || (result.water_level_exceeded && !result.made_progress);
        }

        if (made_progress) {
            UpdateAsyncDeleteMetrics();
            sleep_interval_ms = 0;
        } else {
            sleep_interval_ms = sleep_interval_ms_.load();
            if (needs_no_progress_backoff) {
                sleep_interval_ms = std::max<std::uint32_t>(sleep_interval_ms, 1);
                METRICS_(cache_reclaimer, reclaim_no_progress_backoff_count) += 1;
                LOG_WITH_TRACE(DEBUG,
                               "reclaiming water level exceeded without accepted delete request; back off for [%u] ms",
                               sleep_interval_ms);
            }
        }

        METRICS_(cache_reclaimer, reclaim_cron_duration_us) =
            static_cast<double>(TimestampUtil::GetSteadyTimeUs() - begin_tp);
        METRICS_(cache_reclaimer, reclaim_cron_count) += 1;
    }
}

bool CacheReclaimer::DoKeySampling(const std::shared_ptr<RequestContext> &request_context,
                                   const std::shared_ptr<const InstanceInfo> &instance_info,
                                   std::vector<std::int64_t> &out_keys,
                                   std::vector<std::map<std::string, std::string>> &out_maps) noexcept {
    return DoKeySamplingWithSize(request_context, instance_info, 0, false, out_keys, out_maps);
}

bool CacheReclaimer::DoKeySamplingWithSize(const std::shared_ptr<RequestContext> &request_context,
                                           const std::shared_ptr<const InstanceInfo> &instance_info,
                                           const std::size_t total_sampling_size,
                                           const bool bounded_waves,
                                           std::vector<std::int64_t> &out_keys,
                                           std::vector<std::map<std::string, std::string>> &out_maps) noexcept {
    const std::string &ins_id = instance_info->instance_id();
    const std::string &ins_gr = instance_info->instance_group_name();

    const auto meta_indexer = meta_indexer_manager_->GetMetaIndexer(ins_id);
    if (meta_indexer == nullptr) {
        LOG_WITH_ID(WARN, "meta indexer is nullptr");
        return false;
    }

    const std::size_t total_sampling_sz = bounded_waves ? total_sampling_size : sampling_size_.load();
    std::size_t sampling_sz_per_task = sampling_size_per_task_.load();
    if (total_sampling_sz == 0) {
        KVCM_LOG_ERROR("sampling size == 0");
        return false;
    }
    if (sampling_sz_per_task == 0 || sampling_sz_per_task > total_sampling_sz) {
        sampling_sz_per_task = total_sampling_sz;
    }

    auto cancelled = std::make_shared<std::atomic<bool>>(false);
    auto sample = [this, request_context, ins_id, ins_gr, meta_indexer, cancelled, bounded_waves](
                      std::size_t sampling_sz,
                      std::vector<std::int64_t> &keys,
                      std::vector<std::map<std::string, std::string>> &maps) -> ErrorCode {
        if (cancelled->load(std::memory_order_relaxed) || (bounded_waves && (!IsRunning() || IsPaused()))) {
            return ErrorCode::EC_ERROR;
        }
        if (const auto ec = meta_indexer->SampleReclaimKeys(request_context.get(), sampling_sz, keys);
            ec != ErrorCode::EC_OK) {
            LOG_WITH_ID(WARN, "random sample failed, error code: [%d]", static_cast<std::int32_t>(ec));
            return ec;
        }
        if (keys.empty()) {
            LOG_WITH_ID(DEBUG, "random sample got empty keys");
            return ErrorCode::EC_NOENT;
        }
        if (keys.size() != sampling_sz) {
            LOG_WITH_ID(DEBUG, "random sample key size mismatch, expect: [%zu], got: [%zu]", sampling_sz, keys.size());
        }

        if (cancelled->load(std::memory_order_relaxed) || (bounded_waves && (!IsRunning() || IsPaused()))) {
            return ErrorCode::EC_ERROR;
        }
        if (const auto res = meta_indexer->GetProperties(request_context.get(), keys, {PROPERTY_LRU_TIME}, maps);
            res.ec != ErrorCode::EC_OK) {
            LOG_WITH_ID(WARN,
                        "get properties failed, error code: [%d], proceed with empty lru_time",
                        static_cast<std::int32_t>(res.ec));
            maps.clear();
            maps.resize(keys.size());
        } else if (keys.size() != maps.size()) {
            LOG_WITH_ID(
                WARN, "num of sampled keys [%zu] and property maps [%zu] do not match", keys.size(), maps.size());
            maps.clear();
            maps.resize(keys.size());
        }
        return ErrorCode::EC_OK;
    };

    out_keys.clear();
    out_keys.reserve(total_sampling_sz);
    out_maps.clear();
    out_maps.reserve(total_sampling_sz);
    const std::size_t worker_sz = (total_sampling_sz + sampling_sz_per_task - 1) / sampling_sz_per_task;
    if (worker_sz == 1 && !bounded_waves) {
        return sample(total_sampling_sz, out_keys, out_maps) == ErrorCode::EC_OK;
    }

    if (bounded_waves) {
        std::vector<std::int64_t> sampled_keys;
        sampled_keys.reserve(total_sampling_sz);
        std::vector<std::map<std::string, std::string>> sampled_maps;
        sampled_maps.reserve(total_sampling_sz);

        std::size_t sampling_sz_todo = total_sampling_sz;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(future_timeout_ms_.load());
        while (sampling_sz_todo > 0) {
            const std::size_t in_flight = in_flight_sampling_tasks_.load();
            if (in_flight >= workers_.size()) {
                LOG_WITH_ID(WARN,
                            "skipping fair key sampling wave: [%zu] tasks still in-flight, worker pool saturated",
                            in_flight);
                cancelled->store(true, std::memory_order_relaxed);
                return false;
            }

            const std::size_t available_workers = workers_.size() - in_flight;
            const std::size_t remaining_task_count = (sampling_sz_todo - 1) / sampling_sz_per_task + 1;
            const std::size_t wave_task_count = std::min(remaining_task_count, available_workers);
            std::vector<std::future<KeySamplingResult>> futures;
            futures.reserve(wave_task_count);
            for (std::size_t i = 0; i != wave_task_count; ++i) {
                const std::size_t sampling_sz = std::min(sampling_sz_per_task, sampling_sz_todo);
                auto promise = std::make_shared<std::promise<KeySamplingResult>>();
                futures.emplace_back(promise->get_future());
                in_flight_sampling_tasks_.fetch_add(1);
                SubmitTask([this, sample, sampling_sz, promise]() {
                    std::vector<std::int64_t> keys;
                    std::vector<std::map<std::string, std::string>> maps;
                    const auto ec = sample(sampling_sz, keys, maps);
                    in_flight_sampling_tasks_.fetch_sub(1);
                    if (ec != ErrorCode::EC_OK) {
                        promise->set_value({ec, nullptr, nullptr});
                    } else {
                        promise->set_value(
                            {ErrorCode::EC_OK,
                             std::make_shared<std::vector<std::int64_t>>(std::move(keys)),
                             std::make_shared<std::vector<std::map<std::string, std::string>>>(std::move(maps))});
                    }
                });
                sampling_sz_todo -= sampling_sz;
            }

            bool wave_succeeded = true;
            for (auto &future : futures) {
                if (!future.valid()) {
                    wave_succeeded = false;
                    cancelled->store(true, std::memory_order_relaxed);
                    break;
                }
                const auto remaining = deadline - std::chrono::steady_clock::now();
                if (remaining <= std::chrono::milliseconds::zero() ||
                    future.wait_for(remaining) != std::future_status::ready) {
                    LOG_WITH_ID(WARN, "fair key sampling wave timed out, shared deadline exceeded");
                    wave_succeeded = false;
                    cancelled->store(true, std::memory_order_relaxed);
                    break;
                }
                auto key_sampling_result = future.get();
                if (key_sampling_result.ec != ErrorCode::EC_OK) {
                    wave_succeeded = false;
                    cancelled->store(true, std::memory_order_relaxed);
                    break;
                }
                sampled_keys.insert(sampled_keys.end(),
                                    std::make_move_iterator(key_sampling_result.keys->begin()),
                                    std::make_move_iterator(key_sampling_result.keys->end()));
                sampled_maps.insert(sampled_maps.end(),
                                    std::make_move_iterator(key_sampling_result.maps->begin()),
                                    std::make_move_iterator(key_sampling_result.maps->end()));
            }
            if (!wave_succeeded) {
                cancelled->store(true, std::memory_order_relaxed);
                out_keys.clear();
                out_maps.clear();
                return false;
            }
        }

        out_keys = std::move(sampled_keys);
        out_maps = std::move(sampled_maps);
        return true;
    }

    // guard: skip submitting new tasks if too many prior tasks are still
    // in-flight (stuck on backend), to prevent worker pool exhaustion
    if (const std::size_t in_flight = in_flight_sampling_tasks_.load(); in_flight >= workers_.size()) {
        LOG_WITH_ID(WARN, "skipping key sampling: [%zu] tasks still in-flight, worker pool saturated", in_flight);
        return false;
    }

    std::size_t sampling_sz_todo = total_sampling_sz;
    std::vector<std::future<KeySamplingResult>> futures;
    for (std::size_t i = 0; i != worker_sz; ++i) {
        auto promise = std::make_shared<std::promise<KeySamplingResult>>();
        futures.emplace_back(promise->get_future());

        // final task do sample with left key size
        std::size_t sampling_sz = (i == worker_sz - 1) ? sampling_sz_todo : sampling_sz_per_task;
        in_flight_sampling_tasks_.fetch_add(1);
        SubmitTask([this, sample, sampling_sz, promise]() {
            std::vector<std::int64_t> keys;
            std::vector<std::map<std::string, std::string>> maps;
            const auto ec = sample(sampling_sz, keys, maps);
            in_flight_sampling_tasks_.fetch_sub(1);
            if (ec != ErrorCode::EC_OK) {
                promise->set_value({ec, nullptr, nullptr});
            } else {
                promise->set_value(
                    {ErrorCode::EC_OK,
                     std::make_shared<std::vector<std::int64_t>>(std::move(keys)),
                     std::make_shared<std::vector<std::map<std::string, std::string>>>(std::move(maps))});
            }
        });
        sampling_sz_todo -= sampling_sz;
    }

    bool result = true;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(future_timeout_ms_.load());
    for (auto &fut : futures) {
        if (fut.valid()) {
            if (const auto remaining = deadline - std::chrono::steady_clock::now();
                remaining <= std::chrono::milliseconds::zero() ||
                fut.wait_for(remaining) != std::future_status::ready) {
                LOG_WITH_ID(WARN, "key sampling task timed out, deadline exceeded");
                result = false;
                break; // timeout must have happened, break early
            }
            if (!result) {
                // some tasks already failed, no need to extract data any further
                continue;
            }

            if (auto key_sampling_res = fut.get(); key_sampling_res.ec != ErrorCode::EC_OK) {
                result = false;
            } else {
                out_keys.insert(out_keys.end(),
                                std::make_move_iterator(key_sampling_res.keys->begin()),
                                std::make_move_iterator(key_sampling_res.keys->end()));
                out_maps.insert(out_maps.end(),
                                std::make_move_iterator(key_sampling_res.maps->begin()),
                                std::make_move_iterator(key_sampling_res.maps->end()));
            }
        } else {
            result = false;
        }
    }
    if (!result) {
        // signal cancellation so still-running tasks abort early at the
        // next checkpoint
        cancelled->store(true, std::memory_order_relaxed);
    }
    return result;
}

bool CacheReclaimer::MakeBatchByLRU(const RequestContext *request_context,
                                    const std::shared_ptr<const InstanceInfo> &instance_info,
                                    const std::vector<std::int64_t> &sampled_keys,
                                    const std::vector<std::map<std::string, std::string>> &property_maps,
                                    std::vector<std::int64_t> &out_batch,
                                    AgeStats &out_lru_age_stats) const noexcept {
    const std::size_t batching_size = batching_size_.load();
    return MakeBatchByLRUWithSize(
        request_context, instance_info, sampled_keys, property_maps, batching_size, out_batch, out_lru_age_stats);
}

bool CacheReclaimer::MakeBatchByLRUWithSize(const RequestContext *request_context,
                                            const std::shared_ptr<const InstanceInfo> &instance_info,
                                            const std::vector<std::int64_t> &sampled_keys,
                                            const std::vector<std::map<std::string, std::string>> &property_maps,
                                            const std::size_t batching_size,
                                            std::vector<std::int64_t> &out_batch,
                                            AgeStats &out_lru_age_stats) const noexcept {
    if (batching_size == 0) {
        out_batch.clear();
        out_lru_age_stats.Clear();
        return true;
    }

    // invariant:
    // the 2 vectors' size must be guaranteed to be equal, and the
    // content must be guaranteed to be correlative when iterated by
    // index
    assert(sampled_keys.size() == property_maps.size());

    const std::string &ins_id = instance_info->instance_id();
    const std::string &ins_gr = instance_info->instance_group_name();

    std::vector<std::pair<std::int64_t, std::int64_t>> key_tp_vec; // vector of {key, last_access_time}
    key_tp_vec.reserve(sampled_keys.size());

    for (std::size_t i = 0; i != sampled_keys.size(); ++i) {
        const auto &k = sampled_keys[i];
        const auto &m = property_maps[i];
        int64_t lru_ts = 0;
        // if PROPERTY_LRU_TIME is not found, use 0 as the timestamp, the reclaim strategy will degrade
        if (const auto it = m.find(PROPERTY_LRU_TIME); it != m.end()) {
            // the PROPERTY_LRU_TIME value is represented as an int64_t type
            // timepoint string; parse them into integers
            const auto &lru_ts_str = it->second;
            if (!StringUtil::StrToInt64(lru_ts_str.c_str(), lru_ts)) {
                INTERVAL_LOG_WITH_ID(
                    WARN, 10000, "lru_time str [%s] to int64 failed, use 0 instead", lru_ts_str.c_str());
                lru_ts = 0;
            }
        } else {
            INTERVAL_LOG_WITH_ID(WARN, 10000, "PROPERTY_LRU_TIME not found, use 0 instead");
        }
        key_tp_vec.emplace_back(k, lru_ts);
    }

    if (sampled_keys.size() > batching_size) {
        std::sort(key_tp_vec.begin(),
                  key_tp_vec.end(),
                  [](const std::pair<std::int64_t, std::int64_t> &a,
                     const std::pair<std::int64_t, std::int64_t> &b) -> bool { return a.second < b.second; });
    }

    // constitute the batch to be submitted for deleting
    // the first N timestamp would be picked out
    const std::size_t effective_batch_size = std::min(batching_size, sampled_keys.size());
    std::unordered_set<std::int64_t> deduped_batch;
    const int64_t now_us = TimestampUtil::GetCurrentTimeUs();
    int64_t age_sum = 0;
    int64_t age_count = 0;
    for (const auto &[key, tp] : key_tp_vec) {
        if (auto [_, r] = deduped_batch.insert(key); r) {
            if (tp > 0) {
                const int64_t age = now_us - tp;
                out_lru_age_stats.min_us = std::min(out_lru_age_stats.min_us, age);
                out_lru_age_stats.max_us = std::max(out_lru_age_stats.max_us, age);
                age_sum += age;
                ++age_count;
            }
            if (deduped_batch.size() == effective_batch_size) {
                break;
            }
        }
    }

    if (deduped_batch.empty()) {
        out_batch.clear();
        out_lru_age_stats.Clear();
        return true;
    }

    if (age_count > 0) {
        out_lru_age_stats.avg_us = age_sum / age_count;
    } else {
        out_lru_age_stats.Clear();
    }

    if (deduped_batch.size() < batching_size) {
        if (deduped_batch.size() != sampled_keys.size()) {
            // sampled_keys contains duplicated keys, log the event
            LOG_WITH_ID(DEBUG,
                        "shortened batch size (likely duplicated keys sampled), final batch size: [%zu], "
                        "sampled keys size: [%zu], intended batching size: [%zu]",
                        deduped_batch.size(),
                        sampled_keys.size(),
                        batching_size);
        } else {
            // the batch size is equal to the size of sampled keys;
            // * possibility 1: not enough keys sampled
            // * possibility 2: sampling_size_ < batching_size_
            LOG_WITH_ID(DEBUG,
                        "shortened batch size, final batch size: [%zu], "
                        "sampled keys size: [%zu], intended batching size: [%zu]",
                        deduped_batch.size(),
                        sampled_keys.size(),
                        batching_size);
        }
    }

    out_batch.assign(deduped_batch.begin(), deduped_batch.end());
    return true;
}

bool CacheReclaimer::FilterLocID(RequestContext *request_context,
                                 const std::shared_ptr<const InstanceInfo> &instance_info,
                                 const std::vector<std::int64_t> &batch,
                                 const WaterLevelExceed &water_level_exceed,
                                 std::vector<std::vector<std::string>> &out_loc_ids,
                                 BytesByStorageType &out_bytes_by_type,
                                 CountsByStorageType &out_location_counts_by_type,
                                 std::uint64_t &out_predicted_deleted_keys,
                                 AgeStats &out_create_age_stats) noexcept {
    const std::string &ins_id = instance_info->instance_id();
    const std::string &ins_gr = instance_info->instance_group_name();

    out_loc_ids.clear();
    out_bytes_by_type.fill(0);
    out_location_counts_by_type.fill(0);
    out_predicted_deleted_keys = 0;
    out_create_age_stats = AgeStats{};

    if (pending_delete_handler_count_ >= async_delete_config_.pending_delete_handler_limit ||
        pending_delete_bytes_ >= async_delete_config_.pending_bytes_limit) {
        RecordPendingLimitReject(ins_gr, "process");
        LOG_WITH_ID(WARN,
                    "process pending delete limit reached, handlers: [%" PRIu64 "/%" PRIu64 "], bytes: [%" PRIu64
                    "/%" PRIu64 "]",
                    pending_delete_handler_count_,
                    async_delete_config_.pending_delete_handler_limit,
                    pending_delete_bytes_,
                    async_delete_config_.pending_bytes_limit);
        out_loc_ids.resize(batch.size());
        out_create_age_stats.Clear();
        return true;
    }

    const auto meta_searcher = meta_searcher_manager_->GetMetaSearcher(ins_id);
    if (meta_searcher == nullptr) {
        LOG_WITH_ID(WARN, "meta searcher is nullptr");
        return false;
    }

    // get the location map of each block in the batch
    std::vector<CacheLocationMap> loc_maps;
    const BlockMask blk_mask(std::in_place_type<BlockMaskVector>, batch.size(), false);
    assert(std::holds_alternative<BlockMaskVector>(blk_mask));
    if (const auto ec = meta_searcher->BatchGetLocation(request_context, batch, blk_mask, loc_maps);
        ec != ErrorCode::EC_OK) {
        LOG_WITH_ID(WARN, "get cache location maps failed, error code: [%d]", static_cast<std::int32_t>(ec));
        return false;
    }

    if (loc_maps.size() != batch.size()) {
        LOG_WITH_ID(WARN,
                    "get cache location maps failed: result vec size [%zu] not match batch size [%zu]",
                    loc_maps.size(),
                    batch.size());
        return false;
    }

    // inspect the cache location status of each block and get the
    // filtered location ID vecs
    const bool in_storage_type_eviction_zone = water_level_exceed.CheckStorageTypeWaterLevelExceed();
    out_loc_ids.reserve(loc_maps.size());

    // 多层存储 keep_both 配套：构建"冷层" storage 集合（迁移策略的 target_storage）。
    // 在"总水位超限"分支下，对同时拥有冷/热副本的 block 优先淘汰热(源)副本、保留冷副本，
    // 避免 LRU 不分层导致先淘汰冷副本而使 keep_both 失去读加速收益。类型超限(硬驱逐)分支不参与。
    std::unordered_set<std::string> cold_storages;
    if (!in_storage_type_eviction_zone && registry_manager_ != nullptr) {
        if (auto [ig_ec, ig] = registry_manager_->GetInstanceGroup(request_context, ins_gr);
            ig_ec == ErrorCode::EC_OK && ig != nullptr && ig->cache_config() != nullptr) {
            for (const auto &s : ig->cache_config()->migration_strategies()) {
                if (s != nullptr && !s->target_storage_name().empty()) {
                    cold_storages.insert(s->target_storage_name());
                }
            }
        }
    }
    const auto loc_on_cold_storage = [&cold_storages](const CacheLocation &loc) -> bool {
        if (cold_storages.empty() || loc.location_specs().empty()) {
            return false;
        }
        const DataStorageUri uri(loc.location_specs().front().uri());
        return cold_storages.count(uri.GetHostName()) > 0;
    };
    const auto is_pending_location = [this, &ins_id](const std::int64_t block_key,
                                                     const std::string &location_id) -> bool {
        return pending_locations_.find(PendingLocationKey{ins_id, block_key, location_id}) != pending_locations_.end();
    };
    const auto cold_locations_cover_specs =
        [&is_pending_location, &loc_on_cold_storage](
            const std::int64_t block_key, const CacheLocationMap &loc_map, const CacheLocation &source_loc) -> bool {
        if (source_loc.location_specs().empty()) {
            return false;
        }
        for (const auto &source_spec : source_loc.location_specs()) {
            const bool covered =
                std::any_of(loc_map.begin(),
                            loc_map.end(),
                            [&is_pending_location, &loc_on_cold_storage, block_key, &source_spec](const auto &entry) {
                                const auto &loc_ptr = entry.second;
                                if (!loc_ptr || loc_ptr->status() != CacheLocationStatus::CLS_SERVING ||
                                    IsEventReportStorageType(loc_ptr->type()) ||
                                    is_pending_location(block_key, loc_ptr->id()) || !loc_on_cold_storage(*loc_ptr)) {
                                    return false;
                                }
                                return std::any_of(loc_ptr->location_specs().begin(),
                                                   loc_ptr->location_specs().end(),
                                                   [&source_spec](const auto &cold_spec) {
                                                       return cold_spec.name() == source_spec.name();
                                                   });
                            });
            if (!covered) {
                return false;
            }
        }
        return true;
    };

    const int64_t now_us = TimestampUtil::GetCurrentTimeUs();
    int64_t create_age_sum = 0;
    int64_t create_age_count = 0;
    std::uint64_t selected_total_bytes = 0;
    for (std::size_t block_idx = 0; block_idx < loc_maps.size(); ++block_idx) {
        const auto &loc_map = loc_maps[block_idx];
        const std::int64_t block_key = batch[block_idx]; // 传入 block scope，供 active copy target 精确判断
        std::vector<std::string> loc_id_vec;
        std::size_t valid_location_count = 0;

        // 总水位超限分支：若 block 同时有冷层与非冷层(热/源)副本，则只淘汰已被冷层覆盖的热副本、保留冷副本。
        bool keep_cold_evict_hot = false;
        if (!in_storage_type_eviction_zone && !cold_storages.empty()) {
            bool has_cold = false;
            bool has_hot = false;
            for (const auto &[_, loc_ptr] : loc_map) {
                if (!loc_ptr) {
                    continue;
                }
                const auto &loc = *loc_ptr;
                if (IsEventReportStorageType(loc.type())) {
                    continue;
                }
                if (is_pending_location(block_key, loc.id())) {
                    continue;
                }
                const bool is_active_copy_target =
                    loc.status() == CacheLocationStatus::CLS_WRITING && migration_manager_ != nullptr &&
                    migration_manager_->HasActiveCopyTargetLocation(ins_id, block_key, loc.id());
                const bool is_orphaned_writing =
                    loc.status() == CacheLocationStatus::CLS_WRITING && write_location_manager_ != nullptr &&
                    !write_location_manager_->HasLocationId(loc.id()) && !is_active_copy_target;
                if (loc.status() != CacheLocationStatus::CLS_SERVING && !is_orphaned_writing) {
                    continue;
                }
                if (loc.status() == CacheLocationStatus::CLS_SERVING && loc_on_cold_storage(loc)) {
                    has_cold = true;
                } else if (loc.status() == CacheLocationStatus::CLS_SERVING) {
                    has_hot = true;
                }
            }
            keep_cold_evict_hot = has_cold && has_hot;
        }

        for (const auto &[_, loc_ptr] : loc_map) {
            if (!loc_ptr) {
                continue;
            }
            // Reporter-owned locations are not reclaim candidates, but still
            // keep the metadata key alive after all ordinary locations have
            // been removed. Count them before filtering so key-count credit is
            // only granted when the deletion can actually remove the key.
            ++valid_location_count;
            const auto &loc = *loc_ptr;
            if (IsEventReportStorageType(loc.type())) {
                // Generic reclamation must not remove metadata owned by a
                // ReportEvent reporter. Dedicated lifecycle/snapshot cleanup
                // uses metadata-only conditional deletion instead.
                continue;
            }
            if (is_pending_location(block_key, loc.id())) {
                METRICS_(cache_reclaimer, duplicate_pending_location_filtered_count) += 1;
                continue;
            }
            // a location is eligible for eviction if:
            // 1. it is in CLS_SERVING status, OR
            // 2. it is in CLS_WRITING status but its write session is
            //    no longer active (orphaned after a server restart) and
            //    it is not an active migration copy target.
            const bool is_active_copy_target =
                loc.status() == CacheLocationStatus::CLS_WRITING && migration_manager_ != nullptr &&
                migration_manager_->HasActiveCopyTargetLocation(ins_id, block_key, loc.id());
            const bool is_orphaned_writing =
                loc.status() == CacheLocationStatus::CLS_WRITING && write_location_manager_ != nullptr &&
                !write_location_manager_->HasLocationId(loc.id()) && !is_active_copy_target;
            if (loc.status() == CacheLocationStatus::CLS_SERVING || is_orphaned_writing) {
                bool selected_by_water_level = false;
                if (in_storage_type_eviction_zone) {
                    // some storage type water level exceeded; only collect the
                    // location with matched type but fairness is ignored
                    // TODO (rui): implement the fair eviction
                    if (water_level_exceed.GetWaterLevelExceedByType(loc.type())) {
                        selected_by_water_level = true;
                    }
                } else if (keep_cold_evict_hot && loc.status() == CacheLocationStatus::CLS_SERVING &&
                           loc_on_cold_storage(loc)) {
                    // 多副本 keep_both：保留冷层副本，不淘汰（优先腾出热层空间）。
                } else if (keep_cold_evict_hot && loc.status() == CacheLocationStatus::CLS_SERVING &&
                           !cold_locations_cover_specs(block_key, loc_map, loc)) {
                    // spec group 场景下，只有当该热 location 的 specs 已被冷层 SERVING location 覆盖时才删热。
                } else {
                    // there's no storage type water level exceeded
                    // and since the reclaiming is triggered, the total
                    // usage water level must be exceeded; ignore the
                    // type detection
                    selected_by_water_level = true;
                }
                if (!selected_by_water_level) {
                    continue;
                }

                const DataStorageType base_type = ToBaseType(loc.type());
                const std::size_t type_idx = ToIndex(base_type);
                if (type_idx >= out_bytes_by_type.size()) {
                    LOG_WITH_ID(WARN, "skip location with invalid storage type index: [%zu]", type_idx);
                    continue;
                }

                const auto quota_it = pending_quota_by_group_type_.find({ins_gr, base_type});
                const PendingQuota current_quota =
                    quota_it == pending_quota_by_group_type_.end() ? PendingQuota{} : quota_it->second;
                const std::uint64_t location_bytes = GetLocationBytes(loc);

                const bool location_limit_reached =
                    current_quota.location_count >= async_delete_config_.pending_location_limit_per_group_type ||
                    out_location_counts_by_type[type_idx] >=
                        async_delete_config_.pending_location_limit_per_group_type -
                            std::min(current_quota.location_count,
                                     async_delete_config_.pending_location_limit_per_group_type);
                const std::uint64_t current_and_selected_type_bytes =
                    SaturatingAdd(current_quota.bytes, out_bytes_by_type[type_idx]);
                const bool type_bytes_limit_reached =
                    current_and_selected_type_bytes >= async_delete_config_.pending_bytes_limit_per_group_type ||
                    location_bytes > async_delete_config_.pending_bytes_limit_per_group_type -
                                         std::min(current_and_selected_type_bytes,
                                                  async_delete_config_.pending_bytes_limit_per_group_type);
                const std::uint64_t current_and_selected_process_bytes =
                    SaturatingAdd(pending_delete_bytes_, selected_total_bytes);
                const bool process_bytes_limit_reached =
                    current_and_selected_process_bytes >= async_delete_config_.pending_bytes_limit ||
                    location_bytes >
                        async_delete_config_.pending_bytes_limit -
                            std::min(current_and_selected_process_bytes, async_delete_config_.pending_bytes_limit);
                if (location_limit_reached || type_bytes_limit_reached || process_bytes_limit_reached) {
                    RecordPendingLimitReject(ins_gr, ToString(base_type));
                    INTERVAL_LOG_WITH_ID(WARN,
                                         10000,
                                         "skip pending-limit location, block: [%ld], location: [%s], storage type: "
                                         "[%d], bytes: [%" PRIu64 "]",
                                         batch[block_idx],
                                         loc.id().c_str(),
                                         static_cast<int>(base_type),
                                         location_bytes);
                    continue;
                }

                loc_id_vec.emplace_back(loc.id());
                out_location_counts_by_type[type_idx] = SaturatingAdd(out_location_counts_by_type[type_idx], 1);
                out_bytes_by_type[type_idx] = SaturatingAdd(out_bytes_by_type[type_idx], location_bytes);
                selected_total_bytes = SaturatingAdd(selected_total_bytes, location_bytes);
                if (loc.create_time() > 0) {
                    const int64_t age = now_us - loc.create_time();
                    out_create_age_stats.min_us = std::min(out_create_age_stats.min_us, age);
                    out_create_age_stats.max_us = std::max(out_create_age_stats.max_us, age);
                    create_age_sum += age;
                    ++create_age_count;
                }
            }
        }
        if (!loc_id_vec.empty() && loc_id_vec.size() == valid_location_count) {
            out_predicted_deleted_keys = SaturatingAdd(out_predicted_deleted_keys, 1);
        }
        out_loc_ids.emplace_back(std::move(loc_id_vec));
    }
    if (create_age_count == 0) {
        out_create_age_stats.Clear();
        return true;
    }
    out_create_age_stats.avg_us = create_age_sum / create_age_count;
    return true;
}

void CacheReclaimer::TryMigrateOnGroup(
    const std::shared_ptr<RequestContext> &request_context,
    const std::shared_ptr<const InstanceGroup> &instance_group,
    const std::vector<std::shared_ptr<const InstanceInfo>> &instance_infos) noexcept {
    if (!IsRunning() || IsPaused()) {
        return;
    }
    if (instance_group == nullptr || migration_manager_ == nullptr || registry_manager_ == nullptr) {
        return;
    }
    const std::string &ins_gr = instance_group->name();

    const auto cache_config = instance_group->cache_config();
    if (cache_config == nullptr) {
        return;
    }
    const auto &strategies = cache_config->migration_strategies();
    if (strategies.empty()) {
        return;
    }
    const auto data_storage_manager = registry_manager_->data_storage_manager();
    if (data_storage_manager == nullptr) {
        LOG_WITH_GR(WARN, "data storage manager is nullptr; skip migration");
        return;
    }
    const auto usage = GetGroupUsageData(request_context.get(), instance_infos);
    if (usage == nullptr) {
        LOG_WITH_GR(WARN, "group usage data is nullptr; skip migration");
        return;
    }

    const auto &quota = instance_group->quota();
    struct CachedMigrationBatch {
        bool built = false;
        bool ok = false;
        std::vector<std::int64_t> batch;
    };
    std::unordered_map<std::string, CachedMigrationBatch> migration_batch_cache;
    const auto configured_copy_concurrency = cache_config->migration_copy_max_concurrency();
    const std::size_t max_concurrent_copy =
        configured_copy_concurrency > 0 ? static_cast<std::size_t>(configured_copy_concurrency) : 0;
    // active Copy 与已排队/运行的异步 Prepare 一起用于提前剪枝，避免慢 Create 期间每轮 cron
    // 都继续给同一 group 堆积新 Job；最终硬限制仍在 BatchSubmit 内原子执行。
    const std::size_t active_copy = migration_manager_->ActiveTaskCountForGroup(ins_gr);
    const std::size_t pending_prepare = migration_manager_->PendingAsyncMigrationPrepareCountForGroup(ins_gr);
    constexpr auto kMaxSize = std::numeric_limits<std::size_t>::max();
    const std::size_t estimated_inflight =
        active_copy > kMaxSize - pending_prepare ? kMaxSize : active_copy + pending_prepare;
    std::size_t available_copy_slots =
        max_concurrent_copy > estimated_inflight ? max_concurrent_copy - estimated_inflight : 0;

    for (const auto &strategy : strategies) {
        if (strategy == nullptr) {
            continue;
        }
        const std::string &src_name = strategy->source_storage_name();
        const auto backend = data_storage_manager->GetDataStorageBackend(src_name);
        if (backend == nullptr) {
            LOG_WITH_GR(WARN, "migration source storage [%s] not found; skip", src_name.c_str());
            continue;
        }
        const DataStorageType src_type = ToBaseType(backend->GetType());

        // NOTE: reclaimer 水位粒度仅到存储 type（per-storage 容量/用量暂不可得，见 GetWaterLevelExceed
        // 的 TODO），故迁移触发水位用 source storage 对应 type 的 type 级近似。
        std::int64_t capacity = 0;
        for (const auto &storage_quota : quota.quota_config()) {
            if (storage_quota.storage_spec() == src_type) {
                capacity = storage_quota.capacity();
                break;
            }
        }
        if (capacity <= 0) {
            LOG_WITH_GR(DEBUG, "no positive capacity for source type of [%s]; skip migration", src_name.c_str());
            continue;
        }

        const double water_level =
            static_cast<double>(usage->GetGroupUsageByType(src_type)) / static_cast<double>(capacity);
        const double migration_threshold = strategy->trigger_threshold();
        if (water_level + kEpsilon <= migration_threshold) {
            continue; // 低于迁移区间下界
        }

        const bool copy_enabled = strategy->methods().copy().enabled();
        const bool mark_enabled = strategy->methods().mark().enabled();
        if (!copy_enabled && !mark_enabled) {
            continue;
        }
        if (copy_enabled && !mark_enabled && available_copy_slots == 0) {
            continue;
        }

        LOG_WITH_GR(DEBUG,
                    "migration triggered: src_storage [%s] type [%d] wl [%f] migration_thr [%f] "
                    "group_copy_concurrency [%zu] active_copy [%zu] pending_prepare [%zu] "
                    "available_copy_slots [%zu]",
                    src_name.c_str(),
                    static_cast<std::int32_t>(src_type),
                    water_level,
                    migration_threshold,
                    max_concurrent_copy,
                    active_copy,
                    pending_prepare,
                    available_copy_slots);

        for (const auto &instance_info : instance_infos) {
            if (instance_info == nullptr) {
                continue;
            }
            // Copy-only route 没有 fallback Mark 可做。异步化后无法在本线程获知每个 Job
            // 实际提交了几个 Copy，因此按剩余 block slot 至多放行同样数量的 Prepare Job，
            // 防止 group limit 很小时仍为大量 instance 排队做无效的 fresh meta read。
            if (copy_enabled && !mark_enabled && available_copy_slots == 0) {
                break;
            }
            auto &cached = migration_batch_cache[instance_info->instance_id()];
            if (!cached.built) {
                cached.built = true;
                cached.ok = BuildMigrationCandidateBatch(request_context, instance_info, cached.batch);
            }
            if (!cached.ok || cached.batch.empty()) {
                continue;
            }
            const bool submitted = SubmitMigrationPrepareJob(request_context, instance_info, *strategy, cached.batch);
            if (submitted && copy_enabled && !mark_enabled) {
                --available_copy_slots;
            }
        }
    }
}

bool CacheReclaimer::BuildMigrationCandidateBatch(const std::shared_ptr<RequestContext> &request_context,
                                                  const std::shared_ptr<const InstanceInfo> &instance_info,
                                                  std::vector<std::int64_t> &out_batch) noexcept {
    out_batch.clear();
    if (instance_info == nullptr) {
        return false;
    }

    // 采样 + LRU 取最冷 batch（复用回收侧的采样与排序）。
    std::vector<std::int64_t> keys;
    std::vector<std::map<std::string, std::string>> maps;
    if (!DoKeySampling(request_context, instance_info, keys, maps)) {
        return false;
    }
    AgeStats lru_age_stats;
    if (!MakeBatchByLRU(request_context.get(), instance_info, keys, maps, out_batch, lru_age_stats)) {
        return false;
    }
    return true;
}

std::vector<std::vector<std::string>>
CacheReclaimer::SnapshotPendingLocations(const std::string &instance_id, const std::vector<std::int64_t> &batch) const {
    std::vector<std::vector<std::string>> snapshot(batch.size());
    for (std::size_t block_idx = 0; block_idx < batch.size(); ++block_idx) {
        const auto block_key = batch[block_idx];
        auto it = pending_locations_.lower_bound(PendingLocationKey{instance_id, block_key, ""});
        while (it != pending_locations_.end() && it->instance_id == instance_id && it->block_key == block_key) {
            snapshot[block_idx].push_back(it->location_id);
            ++it;
        }
    }
    return snapshot;
}

bool CacheReclaimer::SubmitMigrationPrepareJob(const std::shared_ptr<RequestContext> &request_context,
                                               const std::shared_ptr<const InstanceInfo> &instance_info,
                                               const MigrationStrategy &strategy,
                                               const std::vector<std::int64_t> &batch) noexcept {
    if (!IsRunning() || IsPaused()) {
        return false;
    }
    if (instance_info == nullptr || migration_manager_ == nullptr) {
        return false;
    }
    if (batch.empty()) {
        return false;
    }

    const bool copy_enabled = strategy.methods().copy().enabled();
    const bool mark_enabled = strategy.methods().mark().enabled();
    if (!copy_enabled && !mark_enabled) {
        return false;
    }

    const std::string &ins_id = instance_info->instance_id();
    const std::string &ins_gr = instance_info->instance_group_name();
    const std::string &src_name = strategy.source_storage_name();
    const std::string &dst_name = strategy.target_storage_name();

    auto copy_counter = METRICS_(cache_reclaimer, migration_copy_submitted_total);
    auto mark_counter = METRICS_(cache_reclaimer, migration_mark_submitted_total);
    MigrationManager::AsyncMigrationPrepareJob job;
    job.trace_id = request_context->trace_id();
    job.instance_group_name = ins_gr;
    job.instance_id = ins_id;
    job.source_storage_name = src_name;
    job.target_storage_name = dst_name;
    job.block_keys = batch;
    // pending_locations_ 由 Reclaimer cron 单线程维护。这里只复制与本 batch 相关的前缀范围；
    // executor worker 不得读取 CacheReclaimer 的任何可变状态。
    job.pending_location_ids_by_block = SnapshotPendingLocations(ins_id, batch);
    job.on_dispatched = [copy_counter, mark_counter, ins_id, src_name, dst_name](
                            const MigrationManager::DispatchBatchResult &dispatch) mutable {
        if (dispatch.copy_submitted > 0) {
            copy_counter += dispatch.copy_submitted;
        }
        if (dispatch.mark_submitted > 0) {
            mark_counter += dispatch.mark_submitted;
        }
        KVCM_LOG_DEBUG("async migration dispatched: instance [%s] src [%s] dst [%s] "
                       "copy_submitted [%lld] mark_submitted [%lld]",
                       ins_id.c_str(),
                       src_name.c_str(),
                       dst_name.c_str(),
                       static_cast<long long>(dispatch.copy_submitted),
                       static_cast<long long>(dispatch.mark_submitted));
    };
    return migration_manager_->SubmitAsyncMigrationPrepare(std::move(job));
}

bool CacheReclaimer::SubmitDelReq(const std::shared_ptr<RequestContext> &request_context,
                                  const std::shared_ptr<const InstanceInfo> &instance_info,
                                  const CacheLocationDelRequest &req,
                                  const BytesByStorageType &bytes_by_type,
                                  const CountsByStorageType &location_counts_by_type,
                                  const std::uint64_t predicted_deleted_keys) noexcept {
    if (!IsRunning() || IsPaused()) {
        // fast exiting in the middle of one job round
        return false;
    }

    const std::string &ins_id = instance_info->instance_id();
    const std::string &ins_gr = instance_info->instance_group_name();
    if (req.block_keys.size() != req.location_ids.size()) {
        LOG_WITH_ID(WARN,
                    "skip invalid reclaim request: block key count [%zu] != location vector count [%zu]",
                    req.block_keys.size(),
                    req.location_ids.size());
        return false;
    }

    std::uint64_t blk_count = 0;
    std::uint64_t loc_count = 0;
    CacheLocationDelRequest final_req{ins_id, {}, {}, req.delay};
    std::vector<PendingLocationKey> pending_locations;
    std::set<PendingLocationKey> request_pending_locations;
    for (std::size_t i = 0; i != req.block_keys.size(); ++i) {
        if (!req.location_ids[i].empty()) {
            blk_count = SaturatingAdd(blk_count, 1);
            loc_count = SaturatingAdd(loc_count, req.location_ids[i].size());
            final_req.block_keys.emplace_back(req.block_keys[i]);
            final_req.location_ids.emplace_back(req.location_ids[i]);
            for (const auto &location_id : req.location_ids[i]) {
                PendingLocationKey pending_location{ins_id, req.block_keys[i], location_id};
                if (!request_pending_locations.insert(pending_location).second) {
                    LOG_WITH_ID(WARN,
                                "skip reclaim request containing duplicate location, block: [%ld], location: [%s]",
                                req.block_keys[i],
                                location_id.c_str());
                    return false;
                }
                if (pending_locations_.find(pending_location) != pending_locations_.end()) {
                    METRICS_(cache_reclaimer, duplicate_pending_location_filtered_count) += 1;
                    LOG_WITH_ID(WARN,
                                "skip reclaim request containing pending location, block: [%ld], location: [%s]",
                                req.block_keys[i],
                                location_id.c_str());
                    return false;
                }
                pending_locations.emplace_back(std::move(pending_location));
            }
        }
    }

    if (blk_count == 0 || loc_count == 0) {
        LOG_WITH_ID(DEBUG, "skip empty reclaim request");
        return false;
    }
    if (pending_delete_handler_count_ >= async_delete_config_.pending_delete_handler_limit) {
        RecordPendingLimitReject(ins_gr, "process");
        LOG_WITH_ID(WARN, "skip reclaim request: process pending handler limit reached");
        return false;
    }

    std::uint64_t request_bytes = 0;
    for (std::size_t idx = 0; idx < bytes_by_type.size(); ++idx) {
        request_bytes = SaturatingAdd(request_bytes, bytes_by_type[idx]);
        if (location_counts_by_type[idx] == 0 && bytes_by_type[idx] == 0) {
            continue;
        }
        const auto type = static_cast<DataStorageType>(idx);
        const auto pending_it = pending_quota_by_group_type_.find({ins_gr, type});
        const PendingQuota pending_quota =
            pending_it == pending_quota_by_group_type_.end() ? PendingQuota{} : pending_it->second;
        if (pending_quota.location_count >= async_delete_config_.pending_location_limit_per_group_type ||
            location_counts_by_type[idx] > async_delete_config_.pending_location_limit_per_group_type -
                                               std::min(pending_quota.location_count,
                                                        async_delete_config_.pending_location_limit_per_group_type) ||
            pending_quota.bytes >= async_delete_config_.pending_bytes_limit_per_group_type ||
            bytes_by_type[idx] >
                async_delete_config_.pending_bytes_limit_per_group_type -
                    std::min(pending_quota.bytes, async_delete_config_.pending_bytes_limit_per_group_type)) {
            RecordPendingLimitReject(ins_gr, ToString(type));
            LOG_WITH_ID(WARN, "skip reclaim request: group/type pending limit reached for type [%zu]", idx);
            return false;
        }
    }
    if (pending_delete_bytes_ >= async_delete_config_.pending_bytes_limit ||
        request_bytes > async_delete_config_.pending_bytes_limit -
                            std::min(pending_delete_bytes_, async_delete_config_.pending_bytes_limit)) {
        RecordPendingLimitReject(ins_gr, "process");
        LOG_WITH_ID(WARN, "skip reclaim request: process pending bytes limit reached");
        return false;
    }

    auto submit_result = sched_plan_executor_->SubmitAsync(final_req);
    if (!submit_result.accepted) {
        LOG_WITH_ID(WARN, "schedule plan executor rejected async reclaim request");
        return false;
    }

    const auto submitted_at = std::chrono::steady_clock::now();
    const auto non_negative_delay = std::max(final_req.delay, std::chrono::microseconds::zero());
    const auto credit_deadline =
        submitted_at + non_negative_delay + std::chrono::milliseconds(async_delete_config_.inflight_delete_timeout_ms);
    delete_handlers_.emplace_front(request_context,
                                   ins_id,
                                   ins_gr,
                                   blk_count,
                                   loc_count,
                                   std::move(pending_locations),
                                   bytes_by_type,
                                   location_counts_by_type,
                                   predicted_deleted_keys,
                                   submitted_at,
                                   credit_deadline,
                                   std::move(submit_result.future));
    AddDeleteHandlerState(delete_handlers_.front());

    METRICS_(cache_reclaimer, block_submit_count) += blk_count;
    METRICS_(cache_reclaimer, location_submit_count) += loc_count;
    METRICS_(cache_reclaimer, delete_submit_count) += 1;

    if (event_manager_ != nullptr) {
        auto reclaim_submit_event = std::make_shared<CacheReclaimSubmitEvent>(ins_id);
        reclaim_submit_event->SetEventTriggerTime();
        reclaim_submit_event->SetAdditionalArgs(final_req.block_keys, final_req.location_ids, final_req.delay.count());
        event_manager_->Publish(reclaim_submit_event);
    }

    LOG_WITH_ID(DEBUG,
                "submit reclaim request to schedule plan executor OK, "
                "with effective cache block count: [%" PRIu64 "], "
                "cache location count: [%" PRIu64 "], bytes: [%" PRIu64 "]",
                blk_count,
                loc_count,
                request_bytes);
    return true;
}

std::shared_ptr<CacheReclaimer::GroupUsageData> CacheReclaimer::GetGroupUsageData(
    const RequestContext *request_context,
    const std::vector<std::shared_ptr<const InstanceInfo>> &instance_infos) const noexcept {
    const auto data = std::make_shared<GroupUsageData>();
    for (const auto &instance_info : instance_infos) {
        if (instance_info == nullptr) {
            LOG_WITH_TRACE(WARN, "instance is nullptr");
            continue;
        }

        const std::string &ins_id = instance_info->instance_id();
        const std::string &ins_gr = instance_info->instance_group_name();
        const auto meta_indexer = meta_indexer_manager_->GetMetaIndexer(ins_id);
        if (meta_indexer == nullptr) {
            LOG_WITH_ID(WARN, "meta indexer is nullptr");
            continue;
        }

        meta_indexer->PersistMetaData();
        const std::size_t ins_used_key_cnt = meta_indexer->GetKeyCount();
        const std::size_t ins_max_key_cnt = meta_indexer->GetMaxKeyCount();

        const std::size_t ins_used_byte_size = meta_indexer->GetStorageUsage();

        data->grp_used_key_cnt_ += ins_used_key_cnt;
        data->grp_max_key_cnt_ += ins_max_key_cnt;
        data->grp_used_byte_sz_ += ins_used_byte_size;

        for (std::size_t idx = 1; idx < static_cast<std::size_t>(DataStorageType::COUNT); ++idx) {
            const auto type = static_cast<DataStorageType>(idx);
            if (type == DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS || IsEventReportStorageType(type)) {
                continue;
            }
            data->AddGroupUsageByType(type, meta_indexer->GetStorageUsageByType(type));
        }
    }
    return data;
}

void CacheReclaimer::HandleDelRes() noexcept {
    auto it_pre = delete_handlers_.before_begin();
    auto it = delete_handlers_.begin();
    while (it != delete_handlers_.end()) {
        const auto &request_context = it->req_ctx_;
        const std::string &ins_id = it->ins_id_;
        const std::string &ins_gr = it->ins_gr_;

        if (!it->fut_.valid()) {
            if (!it->outcome_unknown_reported_) {
                LOG_WITH_ID(WARN,
                            "reclaim request got invalid future; preserve pending locations and hard-limit quota");
                it->outcome_unknown_reported_ = true;
            }
            DisableCredit(*it, "invalid future");
            ++it_pre;
            ++it;
            continue;
        }

        const auto future_status = it->fut_.wait_for(std::chrono::seconds::zero());
        if (future_status == std::future_status::ready) {
            bool terminal = false;
            ErrorCode result_code = ErrorCode::EC_ERROR;
            try {
                const auto [ec, err_msg] = it->fut_.get();
                terminal = true;
                result_code = ec;
                if (ec != ErrorCode::EC_OK) {
                    LOG_WITH_ID(WARN,
                                "reclaim request execute failed, error_code: [%d], error message: [%s]",
                                static_cast<std::int32_t>(ec),
                                err_msg.c_str());
                } else {
                    METRICS_(cache_reclaimer, block_del_count) += it->blk_count_;
                    METRICS_(cache_reclaimer, location_del_count) += it->loc_count_;
                    LOG_WITH_ID(DEBUG,
                                "reclaim request execute finished successfully, "
                                "with effective cache block count: [%" PRIu64 "], "
                                "cache location count: [%" PRIu64 "]",
                                it->blk_count_,
                                it->loc_count_);
                }
            } catch (const std::exception &e) {
                if (!it->outcome_unknown_reported_) {
                    LOG_WITH_ID(WARN,
                                "reclaim request future outcome unknown: [%s]; preserve pending locations and quota",
                                e.what());
                    it->outcome_unknown_reported_ = true;
                }
            } catch (...) {
                if (!it->outcome_unknown_reported_) {
                    LOG_WITH_ID(WARN, "reclaim request future outcome unknown; preserve pending locations and quota");
                    it->outcome_unknown_reported_ = true;
                }
            }

            if (!terminal) {
                DisableCredit(*it, "broken or exceptional future");
                ++it_pre;
                ++it;
                continue;
            }

            METRICS_(cache_reclaimer, delete_complete_count) += 1;
            if (result_code != ErrorCode::EC_OK) {
                METRICS_(cache_reclaimer, delete_fail_count) += 1;
            }
            ReleaseDeleteHandlerState(*it);
            it = delete_handlers_.erase_after(it_pre);
            continue;
        }

        if (future_status == std::future_status::deferred) {
            if (!it->outcome_unknown_reported_) {
                LOG_WITH_ID(WARN,
                            "reclaim request got deferred future; preserve pending locations and hard-limit quota");
                it->outcome_unknown_reported_ = true;
            }
            DisableCredit(*it, "deferred future");
        } else if (it->credit_enabled_ && std::chrono::steady_clock::now() >= it->credit_deadline_) {
            METRICS_(cache_reclaimer, credit_timeout_count) += 1;
            DisableCredit(*it, "credit deadline exceeded");
        }
        ++it_pre;
        ++it;
    }
    UpdateAsyncDeleteMetrics();
}

const char *CacheReclaimer::FairWeightDimensionName(const FairWeightDimension dimension) noexcept {
    switch (dimension) {
    case FairWeightDimension::STORAGE_TYPE_BYTES:
        return "storage_type_bytes";
    case FairWeightDimension::GROUP_BYTES:
        return "group_bytes";
    case FairWeightDimension::GROUP_KEYS:
        return "group_keys";
    case FairWeightDimension::NONE:
    default:
        return "none";
    }
}

bool CacheReclaimer::AllocateFairBudget(const std::size_t total_budget,
                                        const std::vector<std::uint64_t> &weights,
                                        const std::vector<std::string> &instance_ids,
                                        std::vector<std::size_t> &out_allocations) noexcept {
    out_allocations.clear();
    if (weights.size() != instance_ids.size()) {
        return false;
    }
    out_allocations.resize(weights.size(), 0);
    if (total_budget == 0) {
        return true;
    }

    using uint128_t = unsigned __int128;
    constexpr uint128_t kUint128Max = ~static_cast<uint128_t>(0);
    uint128_t total_weight = 0;
    for (const std::uint64_t weight : weights) {
        if (total_weight > kUint128Max - static_cast<uint128_t>(weight)) {
            return false;
        }
        total_weight += static_cast<uint128_t>(weight);
    }
    if (total_weight == 0) {
        return false;
    }

    struct RemainderEntry {
        std::size_t index;
        uint128_t remainder;
    };
    std::vector<RemainderEntry> remainders;
    remainders.reserve(weights.size());
    std::size_t allocated = 0;
    for (std::size_t i = 0; i != weights.size(); ++i) {
        if (weights[i] == 0) {
            continue;
        }
        const uint128_t numerator = static_cast<uint128_t>(total_budget) * static_cast<uint128_t>(weights[i]);
        const uint128_t quotient = numerator / total_weight;
        if (quotient > static_cast<uint128_t>(std::numeric_limits<std::size_t>::max())) {
            return false;
        }
        const std::size_t allocation = static_cast<std::size_t>(quotient);
        if (allocation > total_budget - allocated) {
            return false;
        }
        out_allocations[i] = allocation;
        allocated += allocation;
        remainders.push_back({i, numerator % total_weight});
    }

    const std::size_t unallocated = total_budget - allocated;
    if (unallocated > remainders.size()) {
        return false;
    }
    std::sort(remainders.begin(), remainders.end(), [&instance_ids](const auto &lhs, const auto &rhs) {
        if (lhs.remainder != rhs.remainder) {
            return lhs.remainder > rhs.remainder;
        }
        if (instance_ids[lhs.index] != instance_ids[rhs.index]) {
            return instance_ids[lhs.index] < instance_ids[rhs.index];
        }
        return lhs.index < rhs.index;
    });
    for (std::size_t i = 0; i != unallocated; ++i) {
        ++out_allocations[remainders[i].index];
    }
    return true;
}

bool CacheReclaimer::BuildFairReclaimPlan(const RequestContext *request_context,
                                          const WaterLevelExceed &water_level_exceed,
                                          const std::vector<std::shared_ptr<const InstanceInfo>> &instance_infos,
                                          const std::size_t configured_sampling_size,
                                          const std::size_t configured_batch_size,
                                          FairReclaimPlan &out_plan) const noexcept {
    out_plan = FairReclaimPlan{};
    out_plan.configured_sampling_size = configured_sampling_size;
    out_plan.configured_batch_size = configured_batch_size;
    out_plan.normalized_sampling_size = configured_sampling_size;

    if (configured_sampling_size == 0 || configured_batch_size == 0) {
        LOG_WITH_TRACE(WARN,
                       "skip fair reclaim plan because configured sampling size [%zu] or batching size [%zu] is zero",
                       configured_sampling_size,
                       configured_batch_size);
        return false;
    }
    if (configured_sampling_size < configured_batch_size) {
        out_plan.normalized_sampling_size = configured_batch_size;
        out_plan.sampling_size_normalized = true;
        KVCM_INTERVAL_LOG_WARN(10,
                               "trace_id [%s] | normalize fair sampling size from [%zu] to batching size [%zu]",
                               request_context->trace_id().c_str(),
                               configured_sampling_size,
                               configured_batch_size);
    }

    if (water_level_exceed.CheckStorageTypeWaterLevelExceed()) {
        out_plan.weight_dimension = FairWeightDimension::STORAGE_TYPE_BYTES;
    } else if (water_level_exceed.GetGroupBytesWaterLevelExceed()) {
        out_plan.weight_dimension = FairWeightDimension::GROUP_BYTES;
    } else if (water_level_exceed.GetGroupKeysWaterLevelExceed()) {
        out_plan.weight_dimension = FairWeightDimension::GROUP_KEYS;
    } else {
        return false;
    }

    out_plan.items.reserve(instance_infos.size());
    for (std::size_t i = 0; i != instance_infos.size(); ++i) {
        const auto &instance_info = instance_infos[i];
        if (instance_info == nullptr) {
            LOG_WITH_TRACE(WARN, "skip nullptr instance when building fair reclaim plan");
            continue;
        }

        const std::string &ins_id = instance_info->instance_id();
        const std::string &ins_gr = instance_info->instance_group_name();
        const auto meta_indexer = meta_indexer_manager_->GetMetaIndexer(ins_id);
        if (meta_indexer == nullptr) {
            LOG_WITH_ID(WARN, "skip instance without meta indexer when building fair reclaim plan");
            continue;
        }

        std::uint64_t weight = 0;
        switch (out_plan.weight_dimension) {
        case FairWeightDimension::STORAGE_TYPE_BYTES:
            for (std::size_t type_index = 1; type_index < static_cast<std::size_t>(DataStorageType::COUNT);
                 ++type_index) {
                const auto type = static_cast<DataStorageType>(type_index);
                if (type == DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS || IsEventReportStorageType(type) ||
                    !water_level_exceed.GetWaterLevelExceedByType(type)) {
                    continue;
                }
                weight = SaturatingAdd(weight, meta_indexer->GetStorageUsageByType(type));
            }
            break;
        case FairWeightDimension::GROUP_BYTES:
            // Group byte watermarks exclude reporter-owned EventReport usage. Keep the
            // fair weight on the same quota-chargeable storage types. VCNS_HF3FS aliases
            // the HF3FS slot and must be skipped to avoid double counting.
            for (std::size_t type_index = 1; type_index < static_cast<std::size_t>(DataStorageType::COUNT);
                 ++type_index) {
                const auto type = static_cast<DataStorageType>(type_index);
                if (type == DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS || IsEventReportStorageType(type)) {
                    continue;
                }
                weight = SaturatingAdd(weight, meta_indexer->GetStorageUsageByType(type));
            }
            break;
        case FairWeightDimension::GROUP_KEYS:
            weight = static_cast<std::uint64_t>(meta_indexer->GetKeyCount());
            break;
        case FairWeightDimension::NONE:
        default:
            break;
        }

        if (weight == 0) {
            ++out_plan.zero_weight_instance_count;
            continue;
        }
        out_plan.items.push_back({instance_info, ins_id, weight, 0, 0, 0, 0, i});
    }

    out_plan.effective_instance_count = out_plan.items.size();
    if (out_plan.effective_instance_count == 0) {
        LOG_WITH_TRACE(WARN,
                       "water level exceeded but all instances have zero weight for fair dimension [%s]",
                       FairWeightDimensionName(out_plan.weight_dimension));
        return false;
    }

    const auto checked_multiply = [](const std::size_t lhs, const std::size_t rhs, std::size_t &out) {
        if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
            return false;
        }
        out = lhs * rhs;
        return true;
    };
    if (!checked_multiply(configured_batch_size, out_plan.effective_instance_count, out_plan.group_batch_size) ||
        !checked_multiply(
            out_plan.normalized_sampling_size, out_plan.effective_instance_count, out_plan.group_sampling_size)) {
        LOG_WITH_TRACE(ERROR,
                       "fair reclaim Group budget overflow, batch [%zu], sampling [%zu], instance count [%zu]",
                       configured_batch_size,
                       out_plan.normalized_sampling_size,
                       out_plan.effective_instance_count);
        return false;
    }

    std::vector<std::uint64_t> weights;
    std::vector<std::string> instance_ids;
    weights.reserve(out_plan.items.size());
    instance_ids.reserve(out_plan.items.size());
    for (const auto &item : out_plan.items) {
        weights.push_back(item.weight);
        instance_ids.push_back(item.instance_id);
    }

    std::vector<std::size_t> batch_allocations;
    if (!AllocateFairBudget(out_plan.group_batch_size, weights, instance_ids, batch_allocations)) {
        LOG_WITH_TRACE(ERROR, "allocate fair batch budget failed");
        return false;
    }
    for (std::size_t i = 0; i != out_plan.items.size(); ++i) {
        out_plan.items[i].raw_batch_size = batch_allocations[i];
        out_plan.items[i].raw_sampling_size = batch_allocations[i];
    }

    const std::size_t extra_sampling_budget = out_plan.group_sampling_size - out_plan.group_batch_size;
    std::vector<std::size_t> sampling_item_indexes;
    std::vector<std::uint64_t> sampling_weights;
    std::vector<std::string> sampling_instance_ids;
    for (std::size_t i = 0; i != out_plan.items.size(); ++i) {
        if (out_plan.items[i].raw_batch_size == 0) {
            continue;
        }
        sampling_item_indexes.push_back(i);
        sampling_weights.push_back(out_plan.items[i].weight);
        sampling_instance_ids.push_back(out_plan.items[i].instance_id);
    }

    std::vector<std::size_t> extra_sampling_allocations;
    if (!AllocateFairBudget(
            extra_sampling_budget, sampling_weights, sampling_instance_ids, extra_sampling_allocations)) {
        LOG_WITH_TRACE(ERROR, "allocate fair extra sampling budget failed");
        return false;
    }
    for (std::size_t i = 0; i != sampling_item_indexes.size(); ++i) {
        auto &item = out_plan.items[sampling_item_indexes[i]];
        if (extra_sampling_allocations[i] > std::numeric_limits<std::size_t>::max() - item.raw_sampling_size) {
            LOG_WITH_TRACE(ERROR, "fair sampling item budget overflow");
            return false;
        }
        item.raw_sampling_size += extra_sampling_allocations[i];
    }

    constexpr std::size_t kPerInstanceHardLimit = kSizeLimit - 1;
    const std::size_t per_instance_batch_limit = std::min(configured_batch_size, kPerInstanceHardLimit);
    const std::size_t per_instance_sampling_limit = std::min(out_plan.normalized_sampling_size, kPerInstanceHardLimit);
    using uint128_t = unsigned __int128;
    for (auto &item : out_plan.items) {
        item.sampling_size = std::min(item.raw_sampling_size, per_instance_sampling_limit);
        item.batch_size = std::min(item.raw_batch_size, per_instance_batch_limit);
        const bool sampling_size_capped = item.sampling_size != item.raw_sampling_size;

        // Preserve the configured sampling amplification after clipping.
        // Otherwise sampling can hit its limit before batching and gradually
        // degrade into selecting every sampled key instead of the oldest subset.
        if (sampling_size_capped && out_plan.normalized_sampling_size > configured_batch_size) {
            uint128_t max_batch_for_sampling = static_cast<uint128_t>(item.sampling_size) *
                                               static_cast<uint128_t>(configured_batch_size) /
                                               static_cast<uint128_t>(out_plan.normalized_sampling_size);
            // Ratios above the hard sampling limit cannot be preserved exactly.
            // Do not turn an already allocated nonzero batch into permanent
            // no-progress; this does not grant a minimum to raw zero allocations.
            if (max_batch_for_sampling == 0 && item.batch_size > 0) {
                max_batch_for_sampling = 1;
            }
            item.batch_size = std::min(item.batch_size, static_cast<std::size_t>(max_batch_for_sampling));
        }
        const bool item_capped = item.batch_size != item.raw_batch_size || sampling_size_capped;
        if (item.batch_size == 0) {
            if (item_capped) {
                LOG_WITH_TRACE(DEBUG,
                               "drop fair plan item [%s] because capped batch size is zero, raw batch/sample "
                               "[%zu/%zu], capped sample [%zu]",
                               item.instance_id.c_str(),
                               item.raw_batch_size,
                               item.raw_sampling_size,
                               item.sampling_size);
            }
            item.sampling_size = 0;
            continue;
        }
        if (item_capped) {
            ++out_plan.capped_item_count;
        }
    }
    out_plan.items.erase(std::remove_if(out_plan.items.begin(),
                                        out_plan.items.end(),
                                        [](const FairReclaimPlanItem &item) { return item.batch_size == 0; }),
                         out_plan.items.end());
    std::sort(out_plan.items.begin(), out_plan.items.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.weight != rhs.weight) {
            return lhs.weight > rhs.weight;
        }
        if (lhs.batch_size != rhs.batch_size) {
            return lhs.batch_size > rhs.batch_size;
        }
        if (lhs.instance_id != rhs.instance_id) {
            return lhs.instance_id < rhs.instance_id;
        }
        return lhs.original_index < rhs.original_index;
    });
    return true;
}

CacheReclaimer::ReclaimResult
CacheReclaimer::TryReclaimOnGroup(const std::shared_ptr<RequestContext> &request_context,
                                  const std::shared_ptr<const InstanceGroup> &instance_group) noexcept {
    ReclaimResult result;
    if (!IsRunning() || IsPaused()) {
        // fast exiting in the middle of one job round
        return result;
    }

    if (instance_group == nullptr) {
        LOG_WITH_TRACE(WARN, "instance group is nullptr");
        return result;
    }

    const std::string &ins_gr = instance_group->name();
    const auto cache_config = instance_group->cache_config();
    if (cache_config == nullptr) {
        LOG_WITH_GR(WARN, "cache config is nullptr");
        return result;
    }

    const auto &reclaim_strategy = cache_config->reclaim_strategy();
    if (reclaim_strategy == nullptr) {
        LOG_WITH_GR(WARN, "reclaim strategy is nullptr");
        return result;
    }

    // Retrieve the list before branching, as in the legacy path. The budget
    // policy only changes planning and execution after this point.
    const auto [ec, instance_infos] = registry_manager_->ListInstanceInfo(request_context.get(), ins_gr);
    if (ec != ErrorCode::EC_OK) {
        LOG_WITH_GR(WARN, "list instances info failed, error code: [%d]", static_cast<std::int32_t>(ec));
        return result;
    }

    const auto budget_policy = reclaim_strategy->instance_reclaim_budget_policy();
    if (budget_policy == InstanceReclaimBudgetPolicy::FIXED_PER_INSTANCE) {
        result = TryReclaimOnGroupLegacy(request_context, instance_group, reclaim_strategy, instance_infos);
    } else {
        if (budget_policy != InstanceReclaimBudgetPolicy::USAGE_PROPORTIONAL) {
            LOG_WITH_GR(WARN,
                        "unknown instance reclaim budget policy: [%d], falling back to usage-proportional",
                        static_cast<std::int32_t>(budget_policy));
        }
        result = TryReclaimOnGroupFair(request_context, instance_group, reclaim_strategy, instance_infos);
    }

    // Reclaim admission precedes migration preparation in the same cron round. An accepted
    // delete is synchronously recorded in pending_locations_, so the migration job snapshot
    // excludes that exact location before asynchronous Create/Copy starts. This only orders
    // admission; it does not wait for physical deletion. Migration still runs independently
    // when its lower watermark is reached before the reclaim threshold.
    if (migration_manager_ != nullptr && !cache_config->migration_strategies().empty()) {
        TryMigrateOnGroup(request_context, instance_group, instance_infos);
    }

    return result;
}

CacheReclaimer::ReclaimResult CacheReclaimer::TryReclaimOnGroupLegacy(
    const std::shared_ptr<RequestContext> &request_context,
    const std::shared_ptr<const InstanceGroup> &instance_group,
    const std::shared_ptr<CacheReclaimStrategy> &reclaim_strategy,
    const std::vector<std::shared_ptr<const InstanceInfo>> &instance_infos) noexcept {
    ReclaimResult result;
    const std::string &ins_gr = instance_group->name();
    const std::int32_t delay_before_delete_ms = reclaim_strategy->delay_before_delete_ms();
    for (const auto &instance_info : instance_infos) {
        const std::int64_t quota_begin_tp = TimestampUtil::GetSteadyTimeUs();
        const auto water_level_exceed = GetWaterLevelExceed(
            request_context.get(), ins_gr, instance_group->quota(), reclaim_strategy, instance_infos);
        METRICS_(cache_reclaimer, reclaim_quota_duration_us) =
            static_cast<double>(TimestampUtil::GetSteadyTimeUs() - quota_begin_tp);
        if (!IsTriggerReclaiming(water_level_exceed)) {
            if (!result.water_level_exceeded) {
                LOG_WITH_GR(DEBUG, "instance group does not trigger reclaiming");
            } else {
                LOG_WITH_GR(DEBUG, "instance group water level satisfied by in-flight delete credit");
            }
            break;
        }

        result.water_level_exceeded = true;
        const std::int64_t begin_tp = TimestampUtil::GetSteadyTimeUs();
        bool submitted = false;
        switch (reclaim_strategy->reclaim_policy()) {
        case ReclaimPolicy::POLICY_LFU:
            submitted = ReclaimByLFU(request_context, instance_info, *water_level_exceed, delay_before_delete_ms);
            break;
        case ReclaimPolicy::POLICY_TTL:
            submitted = ReclaimByTTL(request_context, instance_info, *water_level_exceed, delay_before_delete_ms);
            break;
        case ReclaimPolicy::POLICY_UNSPECIFIED:
        case ReclaimPolicy::POLICY_LRU:
        default:
            submitted = ReclaimByLRU(request_context, instance_info, *water_level_exceed, delay_before_delete_ms);
            break;
        }
        METRICS_(cache_reclaimer, reclaim_job_duration_us) =
            static_cast<double>(TimestampUtil::GetSteadyTimeUs() - begin_tp);
        result.made_progress = result.made_progress || submitted;
    }
    return result;
}

CacheReclaimer::ReclaimResult
CacheReclaimer::TryReclaimOnGroupFair(const std::shared_ptr<RequestContext> &request_context,
                                      const std::shared_ptr<const InstanceGroup> &instance_group,
                                      const std::shared_ptr<CacheReclaimStrategy> &reclaim_strategy,
                                      const std::vector<std::shared_ptr<const InstanceInfo>> &instance_infos) noexcept {
    ReclaimResult result;
    const std::string &ins_gr = instance_group->name();
    const std::int32_t delay_before_delete_ms = reclaim_strategy->delay_before_delete_ms();
    METRICS_(cache_reclaimer, fair_effective_instance_count) = 0;
    METRICS_(cache_reclaimer, fair_planned_instance_count) = 0;
    METRICS_(cache_reclaimer, fair_sampled_instance_count) = 0;
    METRICS_(cache_reclaimer, fair_submitted_instance_count) = 0;

    const auto read_water_level = [&]() {
        const std::int64_t quota_begin_tp = TimestampUtil::GetSteadyTimeUs();
        auto water_level = GetWaterLevelExceed(
            request_context.get(), ins_gr, instance_group->quota(), reclaim_strategy, instance_infos);
        METRICS_(cache_reclaimer, reclaim_quota_duration_us) =
            static_cast<double>(TimestampUtil::GetSteadyTimeUs() - quota_begin_tp);
        return water_level;
    };

    const auto log_water_level_unavailable = [&]() {
        if (!IsRunning() || IsPaused()) {
            LOG_WITH_GR(DEBUG, "fair plan stopped because reclaimer is stopping or paused");
        } else {
            LOG_WITH_GR(WARN, "fair plan stopped because water level could not be read");
        }
    };

    const auto initial_water_level = read_water_level();
    if (initial_water_level == nullptr) {
        log_water_level_unavailable();
        return result;
    }
    if (!IsTriggerReclaiming(initial_water_level)) {
        LOG_WITH_GR(DEBUG, "instance group does not trigger reclaiming");
        return result;
    }
    result.water_level_exceeded = true;

    FairReclaimPlan plan;
    const std::size_t configured_sampling_size = sampling_size_.load();
    const std::size_t configured_batch_size = batching_size_.load();
    const bool plan_built = BuildFairReclaimPlan(request_context.get(),
                                                 *initial_water_level,
                                                 instance_infos,
                                                 configured_sampling_size,
                                                 configured_batch_size,
                                                 plan);
    METRICS_(cache_reclaimer, fair_zero_weight_skip_count) += plan.zero_weight_instance_count;
    if (plan.sampling_size_normalized) {
        METRICS_(cache_reclaimer, fair_sampling_size_normalized_count) += 1;
    }
    METRICS_(cache_reclaimer, fair_effective_instance_count) = static_cast<double>(plan.effective_instance_count);
    if (!plan_built) {
        return result;
    }

    std::uint64_t planned_batch_count = 0;
    std::uint64_t planned_sample_count = 0;
    const std::uint64_t capped_item_count = static_cast<std::uint64_t>(plan.capped_item_count);
    for (const auto &item : plan.items) {
        planned_batch_count = SaturatingAdd(planned_batch_count, item.batch_size);
        planned_sample_count = SaturatingAdd(planned_sample_count, item.sampling_size);
        LOG_WITH_GR(DEBUG,
                    "fair plan item instance [%s], weight [%" PRIu64 "], raw batch [%zu], raw sample [%zu], "
                    "batch [%zu], sample [%zu]",
                    item.instance_id.c_str(),
                    item.weight,
                    item.raw_batch_size,
                    item.raw_sampling_size,
                    item.batch_size,
                    item.sampling_size);
    }
    METRICS_(cache_reclaimer, fair_plan_count) += 1;
    METRICS_(cache_reclaimer, fair_planned_batch_count) += planned_batch_count;
    METRICS_(cache_reclaimer, fair_planned_sample_count) += planned_sample_count;
    METRICS_(cache_reclaimer, fair_item_capped_count) += capped_item_count;
    METRICS_(cache_reclaimer, fair_planned_instance_count) = static_cast<double>(plan.items.size());

    LOG_WITH_GR(DEBUG,
                "fair plan dimension [%s], group bytes exceeded [%d], group keys exceeded [%d], "
                "type exceeded [%d], configured batch/sample [%zu/%zu], normalized sample [%zu], "
                "group batch/sample [%zu/%zu], effective/planned instances [%zu/%zu]",
                FairWeightDimensionName(plan.weight_dimension),
                initial_water_level->GetGroupBytesWaterLevelExceed(),
                initial_water_level->GetGroupKeysWaterLevelExceed(),
                initial_water_level->CheckStorageTypeWaterLevelExceed(),
                plan.configured_batch_size,
                plan.configured_sampling_size,
                plan.normalized_sampling_size,
                plan.group_batch_size,
                plan.group_sampling_size,
                plan.effective_instance_count,
                plan.items.size());

    const auto plan_scope_still_active = [&](const WaterLevelExceed &current_water_level) {
        switch (plan.weight_dimension) {
        case FairWeightDimension::STORAGE_TYPE_BYTES:
            for (std::size_t type_index = 1; type_index < static_cast<std::size_t>(DataStorageType::COUNT);
                 ++type_index) {
                const auto type = static_cast<DataStorageType>(type_index);
                if (type == DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS) {
                    continue;
                }
                if (initial_water_level->GetWaterLevelExceedByType(type) !=
                    current_water_level.GetWaterLevelExceedByType(type)) {
                    return false;
                }
            }
            return current_water_level.CheckStorageTypeWaterLevelExceed();
        case FairWeightDimension::GROUP_BYTES:
            return !current_water_level.CheckStorageTypeWaterLevelExceed() &&
                   current_water_level.GetGroupBytesWaterLevelExceed();
        case FairWeightDimension::GROUP_KEYS:
            return !current_water_level.CheckStorageTypeWaterLevelExceed() &&
                   !current_water_level.GetGroupBytesWaterLevelExceed() &&
                   current_water_level.GetGroupKeysWaterLevelExceed();
        case FairWeightDimension::NONE:
        default:
            return false;
        }
    };

    std::size_t sampled_instance_count = 0;
    std::size_t submitted_instance_count = 0;
    std::shared_ptr<WaterLevelExceed> water_level_before_next_item;
    for (std::size_t i = 0; i != plan.items.size(); ++i) {
        auto water_level_exceed =
            water_level_before_next_item != nullptr ? std::move(water_level_before_next_item) : read_water_level();
        if (water_level_exceed == nullptr) {
            log_water_level_unavailable();
            break;
        }
        if (!IsTriggerReclaiming(water_level_exceed) || !plan_scope_still_active(*water_level_exceed)) {
            const std::size_t remaining = plan.items.size() - i;
            METRICS_(cache_reclaimer, fair_plan_truncated_count) += 1;
            METRICS_(cache_reclaimer, fair_plan_truncated_instance_count) += remaining;
            LOG_WITH_GR(DEBUG,
                        "fair plan stopped before instance [%s], water level satisfied or trigger scope "
                        "changed, remaining items [%zu]",
                        plan.items[i].instance_id.c_str(),
                        remaining);
            break;
        }

        const auto &item = plan.items[i];
        ++sampled_instance_count;
        const std::int64_t begin_tp = TimestampUtil::GetSteadyTimeUs();
        switch (reclaim_strategy->reclaim_policy()) {
        case ReclaimPolicy::POLICY_LFU:
            LOG_WITH_TRACE(WARN, "LFU reclaim policy not supported yet; fall back to fair LRU policy");
            break;
        case ReclaimPolicy::POLICY_TTL:
            LOG_WITH_TRACE(WARN, "TTL reclaim policy not supported yet; fall back to fair LRU policy");
            break;
        case ReclaimPolicy::POLICY_UNSPECIFIED:
        case ReclaimPolicy::POLICY_LRU:
        default:
            break;
        }
        const bool submitted = ReclaimByLRUWithBudget(request_context,
                                                      item.instance_info,
                                                      *initial_water_level,
                                                      delay_before_delete_ms,
                                                      item.sampling_size,
                                                      item.batch_size);
        METRICS_(cache_reclaimer, reclaim_job_duration_us) =
            static_cast<double>(TimestampUtil::GetSteadyTimeUs() - begin_tp);
        result.made_progress = result.made_progress || submitted;
        if (!submitted) {
            continue;
        }

        ++submitted_instance_count;
        if (i + 1 == plan.items.size()) {
            continue;
        }
        water_level_before_next_item = read_water_level();
        if (water_level_before_next_item == nullptr) {
            log_water_level_unavailable();
            break;
        }
        if (!IsTriggerReclaiming(water_level_before_next_item) ||
            !plan_scope_still_active(*water_level_before_next_item)) {
            const std::size_t remaining = plan.items.size() - i - 1;
            METRICS_(cache_reclaimer, fair_plan_truncated_count) += 1;
            METRICS_(cache_reclaimer, fair_plan_truncated_instance_count) += remaining;
            LOG_WITH_GR(DEBUG,
                        "fair plan stopped after accepted instance [%s], credit satisfied water level or "
                        "trigger scope changed, "
                        "remaining items [%zu]",
                        item.instance_id.c_str(),
                        remaining);
            break;
        }
    }

    METRICS_(cache_reclaimer, fair_sampled_instance_count) = static_cast<double>(sampled_instance_count);
    METRICS_(cache_reclaimer, fair_submitted_instance_count) = static_cast<double>(submitted_instance_count);
    return result;
}

CacheReclaimer::DeleteHandler::DeleteHandler(std::shared_ptr<RequestContext> req_ctx,
                                             std::string ins_id,
                                             std::string ins_gr,
                                             const std::uint64_t blk_count,
                                             const std::uint64_t loc_count,
                                             std::vector<PendingLocationKey> pending_locations,
                                             BytesByStorageType bytes_by_type,
                                             CountsByStorageType location_counts_by_type,
                                             const std::uint64_t predicted_deleted_keys,
                                             const std::chrono::steady_clock::time_point submitted_at,
                                             const std::chrono::steady_clock::time_point credit_deadline,
                                             std::future<PlanExecuteResult> fut)
    : req_ctx_(std::move(req_ctx))
    , ins_id_(std::move(ins_id))
    , ins_gr_(std::move(ins_gr))
    , blk_count_(blk_count)
    , loc_count_(loc_count)
    , pending_locations_(std::move(pending_locations))
    , bytes_by_type_(std::move(bytes_by_type))
    , location_counts_by_type_(std::move(location_counts_by_type))
    , predicted_deleted_keys_(predicted_deleted_keys)
    , submitted_at_(submitted_at)
    , credit_deadline_(credit_deadline)
    , credit_enabled_(true)
    , outcome_unknown_reported_(false)
    , fut_(std::move(fut)) {}

void CacheReclaimer::WorkerRoutine() {
    while (!worker_stop_) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(task_queue_mutex_);

            if (!task_queue_.empty()) {
                task = task_queue_.front();
                task_queue_.pop_front();
            }

            if (!task) {
                if (task_queue_.empty()) {
                    cv_task_queue_.wait(lock, [this] { return worker_stop_ || (!task_queue_.empty()); });
                }
                continue;
            }
        }

        if (task) {
            task();
        }
    }
}

void CacheReclaimer::SubmitTask(const std::function<void()> &task) {
    if (worker_stop_) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(task_queue_mutex_);
        task_queue_.emplace_back(task);
    }

    cv_task_queue_.notify_one();
}

} // namespace kv_cache_manager
