#include "kv_cache_manager/config/cache_reclaim_strategy.h"

namespace kv_cache_manager {

CacheReclaimStrategy::~CacheReclaimStrategy() = default;

bool CacheReclaimStrategy::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "storage_unique_name", storage_unique_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "reclaim_policy", reclaim_policy_);
    KVCM_JSON_GET_MACRO(rapid_value, "trigger_strategy", trigger_strategy_);
    KVCM_JSON_GET_MACRO(rapid_value, "trigger_period_seconds", trigger_period_seconds_);
    KVCM_JSON_GET_MACRO(rapid_value, "reclaim_step_size", reclaim_step_size_);
    KVCM_JSON_GET_MACRO(rapid_value, "reclaim_step_percentage", reclaim_step_percentage_);
    KVCM_JSON_GET_MACRO(rapid_value, "delay_before_delete_ms", delay_before_delete_ms_);
    return true;
}

void CacheReclaimStrategy::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "storage_unique_name", storage_unique_name_);
    Put(writer, "reclaim_policy", reclaim_policy_);
    Put(writer, "trigger_strategy", trigger_strategy_);
    Put(writer, "trigger_period_seconds", trigger_period_seconds_);
    Put(writer, "reclaim_step_size", reclaim_step_size_);
    Put(writer, "reclaim_step_percentage", reclaim_step_percentage_);
    Put(writer, "delay_before_delete_ms", delay_before_delete_ms_);
}

bool CacheReclaimStrategy::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    if (storage_unique_name_.empty()) {
        valid = false;
        local_invalid_fields += "{storage_unique_name}";
    }
    if (!valid) {
        invalid_fields += "{CacheReclaimStrategy: " + local_invalid_fields + "}";
    }
    return valid;
}
} // namespace kv_cache_manager