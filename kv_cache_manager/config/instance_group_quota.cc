#include "kv_cache_manager/config/instance_group_quota.h"

namespace kv_cache_manager {

InstanceGroupQuota::~InstanceGroupQuota() = default;

bool InstanceGroupQuota::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "capacity", capacity_);
    KVCM_JSON_GET_MACRO(rapid_value, "quota_config", quota_config_);
    return true;
}

void InstanceGroupQuota::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "capacity", capacity_);
    Put(writer, "quota_config", quota_config_);
}

bool InstanceGroupQuota::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    // TODO quota_config_现在没有使用，后续做更改的时候再处理
    // for (size_t i = 0; i < quota_config_.size(); ++i) {
    //     if (!quota_config_[i].ValidateRequiredFields(invalid_fields)) {
    //         // valid = false; 这里不设置false，允许部分quota_config不合法
    //         valid = true;
    //     }
    // }
    // invalid_fields += "}";
    if (!valid) {
        invalid_fields += "{InstanceGroupQuota: " + local_invalid_fields + "}";
    }
    return valid;
}
} // namespace kv_cache_manager