#include "kv_cache_manager/config/migration_strategy.h"

#include <cmath>
#include <limits>
#include <set>

namespace kv_cache_manager {

namespace {

bool IsValidMigrationRetention(MigrationRetention retention) {
    return retention == MigrationRetention::MIGRATION_RETENTION_UNSPECIFIED ||
           retention == MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE ||
           retention == MigrationRetention::MIGRATION_RETENTION_KEEP_BOTH;
}

bool IsValidMigrationAdmissionMode(MigrationAdmissionMode mode) {
    return mode == MigrationAdmissionMode::DISABLED || mode == MigrationAdmissionMode::SHADOW ||
           mode == MigrationAdmissionMode::ENFORCE;
}

} // namespace

RecentAccessAdmissionConfig::~RecentAccessAdmissionConfig() = default;

bool RecentAccessAdmissionConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "window_seconds", window_seconds_);
    return true;
}

void RecentAccessAdmissionConfig::ToRapidWriter(
    rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "window_seconds", window_seconds_);
}

bool RecentAccessAdmissionConfig::ValidateRequiredFields(std::string &invalid_fields) const {
    if (window_seconds_ > 0 && window_seconds_ <= std::numeric_limits<int64_t>::max() / (1000 * 1000)) {
        return true;
    }
    invalid_fields += "{RecentAccessAdmissionConfig:{window_seconds}}";
    return false;
}

MigrationAdmissionPolicyConfig::~MigrationAdmissionPolicyConfig() = default;

MigrationAdmissionPolicyConfig::MigrationAdmissionPolicyConfig(
    const MigrationAdmissionPolicyConfig &other) {
    if (other.recent_access_ != nullptr) {
        recent_access_ = std::make_shared<RecentAccessAdmissionConfig>(*other.recent_access_);
    }
}

MigrationAdmissionPolicyConfig &MigrationAdmissionPolicyConfig::operator=(
    const MigrationAdmissionPolicyConfig &other) {
    if (this == &other) {
        return *this;
    }
    recent_access_ = other.recent_access_ == nullptr
                         ? nullptr
                         : std::make_shared<RecentAccessAdmissionConfig>(*other.recent_access_);
    return *this;
}

bool MigrationAdmissionPolicyConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "recent_access", recent_access_, std::shared_ptr<RecentAccessAdmissionConfig>());
    return true;
}

void MigrationAdmissionPolicyConfig::ToRapidWriter(
    rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    if (recent_access_ != nullptr) {
        Put(writer, "recent_access", recent_access_);
    }
}

bool MigrationAdmissionPolicyConfig::ValidateRequiredFields(std::string &invalid_fields) const {
    if (recent_access_ == nullptr) {
        invalid_fields += "{MigrationAdmissionPolicyConfig:{empty_policy}}";
        return false;
    }
    return recent_access_->ValidateRequiredFields(invalid_fields);
}

MigrationAdmissionConfig::~MigrationAdmissionConfig() = default;

MigrationAdmissionConfig::MigrationAdmissionConfig(const MigrationAdmissionConfig &other)
    : mode_(other.mode_) {
    policies_.reserve(other.policies_.size());
    for (const auto &policy : other.policies_) {
        policies_.push_back(policy == nullptr ? nullptr : std::make_shared<MigrationAdmissionPolicyConfig>(*policy));
    }
}

MigrationAdmissionConfig &MigrationAdmissionConfig::operator=(const MigrationAdmissionConfig &other) {
    if (this == &other) {
        return *this;
    }
    mode_ = other.mode_;
    policies_.clear();
    policies_.reserve(other.policies_.size());
    for (const auto &policy : other.policies_) {
        policies_.push_back(policy == nullptr ? nullptr : std::make_shared<MigrationAdmissionPolicyConfig>(*policy));
    }
    return *this;
}

bool MigrationAdmissionConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "mode", mode_, MigrationAdmissionMode::DISABLED);
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "policies", policies_, std::vector<std::shared_ptr<MigrationAdmissionPolicyConfig>>{});
    return true;
}

void MigrationAdmissionConfig::ToRapidWriter(
    rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "mode", mode_);
    Put(writer, "policies", policies_);
}

bool MigrationAdmissionConfig::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    if (!IsValidMigrationAdmissionMode(mode_)) {
        valid = false;
        local_invalid_fields += "{mode}";
    }
    // DISABLED may carry one validated dormant policy so operators can stage
    // policy parameters before switching to SHADOW. The factory intentionally
    // avoids constructing it until the mode is enabled, while JSON/proto keep
    // round-tripping the configuration.
    if (policies_.size() > 1 || (mode_ != MigrationAdmissionMode::DISABLED && policies_.size() != 1)) {
        valid = false;
        local_invalid_fields += "{policies_count}";
    }
    for (const auto &policy : policies_) {
        if (policy == nullptr) {
            valid = false;
            local_invalid_fields += "{policies:null_entry}";
            continue;
        }
        if (!policy->ValidateRequiredFields(local_invalid_fields)) {
            valid = false;
        }
    }
    if (!valid) {
        invalid_fields += "{MigrationAdmissionConfig:" + local_invalid_fields + "}";
    }
    return valid;
}

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
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "admission", admission_, MigrationAdmissionConfig{});
    return true;
}

void MigrationStrategy::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "source_storage_name", source_storage_name_);
    Put(writer, "target_storage_name", target_storage_name_);
    Put(writer, "trigger_threshold", trigger_threshold_);
    Put(writer, "methods", methods_);
    Put(writer, "retention", retention_);
    Put(writer, "admission", admission_);
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
    if (!admission_.ValidateRequiredFields(local_invalid_fields)) {
        valid = false;
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
