#include "kv_cache_manager/config/instance_group_quota.h"

#include <set>

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
    if (capacity_ < -1) {
        valid = false;
        local_invalid_fields += "{capacity}";
    }
    std::set<DataStorageType> types;
    for (const auto &config : quota_config_) {
        valid = config.ValidateRequiredFields(local_invalid_fields) && valid;
        if (!types.insert(ToBaseType(config.storage_spec())).second) {
            valid = false;
            local_invalid_fields += "{duplicate storage_type}";
        }
    }
    if (!valid) {
        invalid_fields += "{InstanceGroupQuota: " + local_invalid_fields + "}";
    }
    return valid;
}
} // namespace kv_cache_manager
