#include "kv_cache_manager/optimizer/config/instance_group_config.h"

namespace kv_cache_manager {

bool OptInstanceGroupConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "group_name", group_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "quota_capacity", quota_capacity_);
    KVCM_JSON_GET_MACRO(rapid_value, "used_percentage", used_percentage_);
    KVCM_JSON_GET_MACRO(rapid_value, "hierarchical_eviction_enabled", hierarchical_eviction_enabled_);
    KVCM_JSON_GET_MACRO(rapid_value, "storages", storages_);
    KVCM_JSON_GET_MACRO(rapid_value, "instances", instances_);
    return true;
};

void OptInstanceGroupConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "group_name", group_name_);
    Put(writer, "quota_capacity", quota_capacity_);
    Put(writer, "used_percentage", used_percentage_);
    Put(writer, "hierarchical_eviction_enabled", hierarchical_eviction_enabled_);
    Put(writer, "storages", storages_);
    Put(writer, "instances", instances_);
};
} // namespace kv_cache_manager