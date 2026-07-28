#include "kv_cache_manager/optimizer/config/optimizer_instance_info.h"

namespace kv_cache_manager {

bool OptimizerStateInfo::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "full_location_spec_group_name", full_location_spec_group_name_, std::string(""));
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "linear_location_spec_group_name", linear_location_spec_group_name_, std::string(""));
    return true;
}

void OptimizerStateInfo::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "full_location_spec_group_name", full_location_spec_group_name_);
    Put(writer, "linear_location_spec_group_name", linear_location_spec_group_name_);
}

bool OptimizerInstanceInfo::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "instance_group_name", instance_group_name_, std::string(""));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "instance_id", instance_id_, std::string(""));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "block_size", block_size_, int32_t(0));
    KVCM_JSON_GET_MACRO(rapid_value, "location_spec_infos", location_spec_infos_);
    KVCM_JSON_GET_MACRO(rapid_value, "location_spec_groups", location_spec_groups_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "linear_step", linear_step_, int32_t(0));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "optimizer_state_info", optimizer_state_info_, OptimizerStateInfo());
    return true;
}

void OptimizerInstanceInfo::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "instance_group_name", instance_group_name_);
    Put(writer, "instance_id", instance_id_);
    Put(writer, "block_size", block_size_);
    Put(writer, "location_spec_infos", location_spec_infos_);
    Put(writer, "location_spec_groups", location_spec_groups_);
    Put(writer, "linear_step", linear_step_);
    Put(writer, "optimizer_state_info", optimizer_state_info_);
}

} // namespace kv_cache_manager
