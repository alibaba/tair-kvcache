#include "kv_cache_manager/config/instance_group.h"

namespace kv_cache_manager {

InstanceGroup::~InstanceGroup() = default;

bool InstanceGroup::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "name", name_);
    KVCM_JSON_GET_MACRO(rapid_value, "storage_candidates", storage_candidates_);
    KVCM_JSON_GET_MACRO(rapid_value, "global_quota_group_name", global_quota_group_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "max_instance_count", max_instance_count_);
    KVCM_JSON_GET_MACRO(rapid_value, "quota", quota_);
    KVCM_JSON_GET_MACRO(rapid_value, "cache_config", cache_config_);
    KVCM_JSON_GET_MACRO(rapid_value, "user_data", user_data_);
    KVCM_JSON_GET_MACRO(rapid_value, "version", version_);
    return true;
}

void InstanceGroup::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "name", name_);
    Put(writer, "storage_candidates", storage_candidates_);
    Put(writer, "global_quota_group_name", global_quota_group_name_);
    Put(writer, "max_instance_count", max_instance_count_);
    Put(writer, "quota", quota_);
    Put(writer, "cache_config", cache_config_);
    Put(writer, "user_data", user_data_);
    Put(writer, "version", version_);
}

bool InstanceGroup::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    if (name_.empty()) {
        valid = false;
        local_invalid_fields += "{name}";
    }
    if (storage_candidates_.empty()) {
        valid = false;
        local_invalid_fields += "{storage_candidates}";
    }
    if (global_quota_group_name_.empty()) {
        valid = false;
        local_invalid_fields += "{global_quota_group_name}";
    }
    if (!quota_.ValidateRequiredFields(local_invalid_fields)) {
        valid = false;
    }
    if (cache_config_ == nullptr) {
        valid = false;
        local_invalid_fields += "{cache_config}";
    } else if (!cache_config_->ValidateRequiredFields(local_invalid_fields)) {
        valid = false;
    }

    if (!valid) {
        invalid_fields += "{InstanceGroup: " + local_invalid_fields + "}";
    }
    return valid;
}
} // namespace kv_cache_manager