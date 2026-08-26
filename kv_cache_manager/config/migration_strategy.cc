#include "kv_cache_manager/config/migration_strategy.h"

#include <cmath>
#include <set>

namespace kv_cache_manager {

namespace {

bool IsValidMigrationRetention(MigrationRetention retention) {
    return retention == MigrationRetention::MIGRATION_RETENTION_UNSPECIFIED ||
           retention == MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE ||
           retention == MigrationRetention::MIGRATION_RETENTION_KEEP_BOTH;
}

} // namespace

MigrationCopyMethod::~MigrationCopyMethod() = default;

bool MigrationCopyMethod::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "enabled", enabled_);
    return true;
}

void MigrationCopyMethod::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "enabled", enabled_);
}

MigrationMarkMethod::~MigrationMarkMethod() = default;

bool MigrationMarkMethod::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "enabled", enabled_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "timeout_ms", timeout_ms_, MigrationMarkMethod::kDefaultTimeoutMs);
    return true;
}

void MigrationMarkMethod::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "enabled", enabled_);
    Put(writer, "timeout_ms", timeout_ms_);
}

MigrationMethods::~MigrationMethods() = default;

bool MigrationMethods::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "copy", copy_);
    KVCM_JSON_GET_MACRO(rapid_value, "mark", mark_);
    return true;
}

void MigrationMethods::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "copy", copy_);
    Put(writer, "mark", mark_);
}

MigrationStrategy::~MigrationStrategy() = default;

bool MigrationStrategy::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "source_storage_name", source_storage_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "target_storage_name", target_storage_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "trigger_threshold", trigger_threshold_);
    KVCM_JSON_GET_MACRO(rapid_value, "methods", methods_);
    KVCM_JSON_GET_MACRO(rapid_value, "retention", retention_);
    return true;
}

void MigrationStrategy::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "source_storage_name", source_storage_name_);
    Put(writer, "target_storage_name", target_storage_name_);
    Put(writer, "trigger_threshold", trigger_threshold_);
    Put(writer, "methods", methods_);
    Put(writer, "retention", retention_);
}

bool MigrationStrategy::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    if (source_storage_name_.empty()) {
        valid = false;
        local_invalid_fields += "{source_storage_name}";
    }
    if (target_storage_name_.empty()) {
        valid = false;
        local_invalid_fields += "{target_storage_name}";
    }
    if (!source_storage_name_.empty() && source_storage_name_ == target_storage_name_) {
        valid = false;
        local_invalid_fields += "{target_storage_equals_source}";
    }
    // trigger_threshold 是水位比例，应落在 (0, 1) 区间
    if (!std::isfinite(trigger_threshold_) || trigger_threshold_ <= 0.0 || trigger_threshold_ >= 1.0) {
        valid = false;
        local_invalid_fields += "{trigger_threshold}";
    }
    // 至少要启用一种执行方式
    if (!methods_.copy().enabled() && !methods_.mark().enabled()) {
        valid = false;
        local_invalid_fields += "{methods_none_enabled}";
    }
    if (methods_.mark().enabled() && methods_.mark().timeout_ms() <= 0) {
        valid = false;
        local_invalid_fields += "{mark_timeout_ms}";
    }
    if (!IsValidMigrationRetention(retention_)) {
        valid = false;
        local_invalid_fields += "{retention}";
    }
    // Copy 路径会按 retention 处理源端，必须显式指定
    if (methods_.copy().enabled() && retention_ == MigrationRetention::MIGRATION_RETENTION_UNSPECIFIED) {
        valid = false;
        local_invalid_fields += "{retention}";
    }
    if (!valid) {
        invalid_fields += "{MigrationStrategy: " + local_invalid_fields + "}";
    }
    return valid;
}

MigrationConfig::~MigrationConfig() = default;

bool MigrationConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value,
                                "copy_max_concurrency",
                                copy_max_concurrency_,
                                MigrationConfig::kDefaultCopyMaxConcurrency);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value,
                                "mark_clear_policy",
                                mark_clear_policy_,
                                MigrationMarkClearPolicy::CLEAR_ON_NEXT_WRITE_SUCCESS);
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "copy_execution_mode", copy_execution_mode_, MigrationCopyExecutionMode::SYNC);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "copy_max_inflight_bytes", copy_max_inflight_bytes_, uint64_t{0});
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "copy_max_quarantine_operations", copy_max_quarantine_operations_, int64_t{0});
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "copy_max_quarantine_bytes", copy_max_quarantine_bytes_, uint64_t{0});
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value,
                                "copy_operation_deadline_ms",
                                copy_operation_deadline_ms_,
                                MigrationConfig::kDefaultCopyOperationDeadlineMs);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value,
                                "copy_poll_initial_interval_ms",
                                copy_poll_initial_interval_ms_,
                                MigrationConfig::kDefaultCopyPollInitialIntervalMs);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value,
                                "copy_poll_max_interval_ms",
                                copy_poll_max_interval_ms_,
                                MigrationConfig::kDefaultCopyPollMaxIntervalMs);
    KVCM_JSON_GET_MACRO(rapid_value, "strategies", strategies_);
    return true;
}

void MigrationConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "copy_max_concurrency", copy_max_concurrency_);
    Put(writer, "mark_clear_policy", mark_clear_policy_);
    Put(writer, "copy_execution_mode", copy_execution_mode_);
    Put(writer, "copy_max_inflight_bytes", copy_max_inflight_bytes_);
    Put(writer, "copy_max_quarantine_operations", copy_max_quarantine_operations_);
    Put(writer, "copy_max_quarantine_bytes", copy_max_quarantine_bytes_);
    Put(writer, "copy_operation_deadline_ms", copy_operation_deadline_ms_);
    Put(writer, "copy_poll_initial_interval_ms", copy_poll_initial_interval_ms_);
    Put(writer, "copy_poll_max_interval_ms", copy_poll_max_interval_ms_);
    Put(writer, "strategies", strategies_);
}

bool MigrationConfig::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    std::set<std::pair<std::string, std::string>> migration_routes;
    if (copy_max_concurrency_ <= 0) {
        valid = false;
        local_invalid_fields += "{copy_max_concurrency}";
    }
    if (mark_clear_policy_ != MigrationMarkClearPolicy::CLEAR_ON_NEXT_WRITE_SUCCESS &&
        mark_clear_policy_ != MigrationMarkClearPolicy::CLEAR_ON_FULL_BLOCK_COVERED) {
        valid = false;
        local_invalid_fields += "{mark_clear_policy}";
    }
    if (copy_execution_mode_ != MigrationCopyExecutionMode::SYNC &&
        copy_execution_mode_ != MigrationCopyExecutionMode::ASYNC_REQUIRED) {
        valid = false;
        local_invalid_fields += "{copy_execution_mode}";
    }
    if (copy_operation_deadline_ms_ <= 0 || copy_poll_initial_interval_ms_ <= 0 ||
        copy_poll_max_interval_ms_ < copy_poll_initial_interval_ms_ ||
        copy_poll_max_interval_ms_ >= copy_operation_deadline_ms_) {
        valid = false;
        local_invalid_fields += "{copy_async_timing}";
    }
    if (copy_execution_mode_ == MigrationCopyExecutionMode::ASYNC_REQUIRED &&
        (copy_max_inflight_bytes_ == 0 || copy_max_quarantine_operations_ <= 0 ||
         copy_max_quarantine_bytes_ == 0)) {
        valid = false;
        local_invalid_fields += "{copy_async_limits}";
    }
    for (const auto &strategy : strategies_) {
        if (strategy == nullptr) {
            valid = false;
            local_invalid_fields += "{strategies:null_entry}";
            continue;
        }
        if (!strategy->ValidateRequiredFields(local_invalid_fields)) {
            valid = false;
        }
        if (!strategy->source_storage_name().empty() && !strategy->target_storage_name().empty() &&
            !migration_routes.emplace(strategy->source_storage_name(), strategy->target_storage_name()).second) {
            // Async Prepare uses the route as its stable identity. Allowing two rules for the same route
            // would make threshold/method/retention selection depend on vector order.
            valid = false;
            local_invalid_fields += "{strategies:duplicate_route}";
        }
    }
    if (!valid) {
        invalid_fields += "{MigrationConfig: " + local_invalid_fields + "}";
    }
    return valid;
}

} // namespace kv_cache_manager
