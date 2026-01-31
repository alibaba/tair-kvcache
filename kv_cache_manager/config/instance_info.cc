#include "kv_cache_manager/config/instance_info.h"

namespace kv_cache_manager {

InstanceInfo::~InstanceInfo() = default;

bool InstanceInfo::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "quota_group_name", quota_group_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "instance_group_name", instance_group_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "instance_id", instance_id_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "block_size", block_size_, static_cast<int32_t>(-1));
    KVCM_JSON_GET_MACRO(rapid_value, "location_spec_infos", location_spec_infos_);
    KVCM_JSON_GET_MACRO(rapid_value, "model_deployment", model_deployment_);
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "location_spec_groups", location_spec_groups_, std::vector<LocationSpecGroup>());
    SortLocationSpecGroups();
    return true;
}

void InstanceInfo::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "quota_group_name", quota_group_name_);
    Put(writer, "instance_group_name", instance_group_name_);
    Put(writer, "instance_id", instance_id_);
    Put(writer, "block_size", block_size_);
    Put(writer, "location_spec_infos", location_spec_infos_);
    Put(writer, "model_deployment", model_deployment_);
    Put(writer, "location_spec_groups", location_spec_groups_);
}

std::string InstanceInfo::ToString() const {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    ToRapidWriter(writer);
    return buffer.GetString();
}

void InstanceInfo::SortLocationSpecGroups() {
    // to use binary search
    std::sort(location_spec_groups_.begin(), location_spec_groups_.end(), [](const auto &a, const auto &b) {
        return a.name() < b.name();
    });
}

} // namespace kv_cache_manager