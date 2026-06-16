#include "kv_cache_manager/config/migration_strategy.h"

#include <cmath>

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
    KVCM_JSON_GET_MACRO(rapid_value, "storage_unique_name", storage_unique_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "target_storage", target_storage_);
    KVCM_JSON_GET_MACRO(rapid_value, "trigger_threshold", trigger_threshold_);
    KVCM_JSON_GET_MACRO(rapid_value, "methods", methods_);
    KVCM_JSON_GET_MACRO(rapid_value, "retention", retention_);
    return true;
}

void MigrationStrategy::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "storage_unique_name", storage_unique_name_);
    Put(writer, "target_storage", target_storage_);
    Put(writer, "trigger_threshold", trigger_threshold_);
    Put(writer, "methods", methods_);
    Put(writer, "retention", retention_);
}

bool MigrationStrategy::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    if (storage_unique_name_.empty()) {
        valid = false;
        local_invalid_fields += "{storage_unique_name}";
    }
    if (target_storage_.empty()) {
        valid = false;
        local_invalid_fields += "{target_storage}";
    }
    // 源与目标必须是不同的 storage，否则迁移无意义
    if (!storage_unique_name_.empty() && storage_unique_name_ == target_storage_) {
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
    KVCM_JSON_GET_MACRO(rapid_value, "strategies", strategies_);
    return true;
}

void MigrationConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "copy_max_concurrency", copy_max_concurrency_);
    Put(writer, "mark_clear_policy", mark_clear_policy_);
    Put(writer, "strategies", strategies_);
}

bool MigrationConfig::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    if (copy_max_concurrency_ <= 0) {
        valid = false;
        local_invalid_fields += "{copy_max_concurrency}";
    }
    if (mark_clear_policy_ != MigrationMarkClearPolicy::CLEAR_ON_NEXT_WRITE_SUCCESS &&
        mark_clear_policy_ != MigrationMarkClearPolicy::CLEAR_ON_FULL_BLOCK_COVERED) {
        valid = false;
        local_invalid_fields += "{mark_clear_policy}";
    }
    for (const auto &strategy : strategies_) {
        if (strategy == nullptr) {
            valid = false;
            local_invalid_fields += "{strategies:null_entry}";
        } else if (!strategy->ValidateRequiredFields(local_invalid_fields)) {
            valid = false;
        }
    }
    if (!valid) {
        invalid_fields += "{MigrationConfig: " + local_invalid_fields + "}";
    }
    return valid;
}

} // namespace kv_cache_manager
