#include "kv_cache_manager/config/instance_group.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"

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
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "extra_info", extra_info_, std::string(""));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value,
                                "event_reporting_storage_candidates",
                                event_reporting_storage_candidates_,
                                std::vector<std::string>());
    std::string buckets_str;
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "revisit_interval_buckets", buckets_str, std::string(""));
    set_revisit_interval_buckets(buckets_str);
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
    Put(writer, "extra_info", extra_info_);
    Put(writer, "event_reporting_storage_candidates", event_reporting_storage_candidates_);
    Put(writer, "revisit_interval_buckets", revisit_interval_buckets_str_);
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
    // Reject non-empty but invalid revisit_interval_buckets.
    // raw string is non-empty but parsed vector is empty → user provided invalid config.
    // Empty raw string is fine (means "use server default").
    if (!revisit_interval_buckets_str_.empty() && parsed_revisit_interval_buckets_.empty()) {
        valid = false;
        local_invalid_fields += "{revisit_interval_buckets}";
    }

    if (!valid) {
        invalid_fields += "{InstanceGroup: " + local_invalid_fields + "}";
    }
    return valid;
}

void InstanceGroup::set_revisit_interval_buckets(const std::string &buckets_str) {
    revisit_interval_buckets_str_ = buckets_str;
    if (buckets_str.empty()) {
        parsed_revisit_interval_buckets_.clear();
        return;
    }
    parsed_revisit_interval_buckets_ = StringUtil::ParseBucketBoundaries(buckets_str);
    if (parsed_revisit_interval_buckets_.empty()) {
        KVCM_LOG_WARN("InstanceGroup [%s]: invalid revisit_interval_buckets '%s', will use server default",
                      name_.c_str(),
                      buckets_str.c_str());
    }
}

} // namespace kv_cache_manager
