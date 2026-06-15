#include "kv_cache_manager/config/migration_strategy.h"

namespace kv_cache_manager {

MigrationCopyMethod::~MigrationCopyMethod() = default;

bool MigrationCopyMethod::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "enabled", enabled_);
    KVCM_JSON_GET_MACRO(rapid_value, "max_concurrency", max_concurrency_);
    return true;
}

void MigrationCopyMethod::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "enabled", enabled_);
    Put(writer, "max_concurrency", max_concurrency_);
}

MigrationMarkMethod::~MigrationMarkMethod() = default;

bool MigrationMarkMethod::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "enabled", enabled_);
    KVCM_JSON_GET_MACRO(rapid_value, "mark_timeout_ms", mark_timeout_ms_);
    return true;
}

void MigrationMarkMethod::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "enabled", enabled_);
    Put(writer, "mark_timeout_ms", mark_timeout_ms_);
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
    if (trigger_threshold_ <= 0.0 || trigger_threshold_ >= 1.0) {
        valid = false;
        local_invalid_fields += "{trigger_threshold}";
    }
    // 至少要启用一种执行方式
    if (!methods_.copy().enabled() && !methods_.mark().enabled()) {
        valid = false;
        local_invalid_fields += "{methods_none_enabled}";
    }
    // Copy 路径会按 retention 处理源端，必须显式指定
    if (methods_.copy().enabled() && retention_ == MigrationRetention::MIGRATION_RETENTION_UNSPECIFIED) {
        valid = false;
        local_invalid_fields += "{retention}";
    }
    // Copy 并发上限固定由配置控制；启用 Copy 时必须为正数
    if (methods_.copy().enabled() && methods_.copy().max_concurrency() <= 0) {
        valid = false;
        local_invalid_fields += "{copy_max_concurrency}";
    }
    if (!valid) {
        invalid_fields += "{MigrationStrategy: " + local_invalid_fields + "}";
    }
    return valid;
}

} // namespace kv_cache_manager
