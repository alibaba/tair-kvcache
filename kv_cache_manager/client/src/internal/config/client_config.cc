#include "kv_cache_manager/client/src/internal/config/client_config.h"

#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/common/logger.h"
namespace kv_cache_manager {

bool ClientConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "block_size", block_size_, static_cast<int32_t>(-1));
    KVCM_JSON_GET_MACRO(rapid_value, "instance_group", instance_group_);
    KVCM_JSON_GET_MACRO(rapid_value, "instance_id", instance_id_);
    KVCM_JSON_GET_MACRO(rapid_value, "location_spec_infos", location_spec_infos_);
    KVCM_JSON_GET_MACRO(rapid_value, "address", addresses_);
    KVCM_JSON_GET_MACRO(rapid_value, "meta_channel_config", meta_channel_config_);
    KVCM_JSON_GET_MACRO(rapid_value, "sdk_config", sdk_wrapper_config_);
    KVCM_JSON_GET_MACRO(rapid_value, "model_deployment", model_deployment_);
    KVCM_JSON_GET_MACRO(rapid_value, "location_spec_groups", location_spec_groups_);
    return Check();
}
void ClientConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "block_size", block_size_);
    Put(writer, "instance_group", instance_group_);
    Put(writer, "instance_id", instance_id_);
    Put(writer, "location_spec_infos", location_spec_infos_);
    Put(writer, "address", addresses_);
    Put(writer, "meta_channel_config", meta_channel_config_);
    Put(writer, "sdk_config", sdk_wrapper_config_);
    Put(writer, "model_deployment", model_deployment_);
    Put(writer, "location_spec_groups", location_spec_groups_);
}

bool ClientConfig::operator==(const ClientConfig &other) const {
    // Compare shared_ptr objects by their content if both are not null
    bool sdk_configs_equal = false;
    if (sdk_wrapper_config_ && other.sdk_wrapper_config_) {
        sdk_configs_equal = (*sdk_wrapper_config_ == *other.sdk_wrapper_config_);
    } else if (!sdk_wrapper_config_ && !other.sdk_wrapper_config_) {
        sdk_configs_equal = true;
    }

    return block_size_ == other.block_size_ && instance_group_ == other.instance_group_ &&
           instance_id_ == other.instance_id_ && location_spec_infos_ == other.location_spec_infos_ &&
           addresses_ == other.addresses_ && sdk_configs_equal && model_deployment_ == other.model_deployment_;
}

bool ClientConfig::Check() const {
    if (block_size_ < 1) {
        KVCM_LOG_ERROR("block_size [%d] is invalid", block_size_);
        return false;
    }
    if (instance_group_.empty() || instance_id_.empty()) {
        KVCM_LOG_ERROR("instance_group or instance_id empty");
        return false;
    }
    if (location_spec_infos_.empty()) {
        KVCM_LOG_ERROR("location_spec_infos is empty");
        return false;
    }
    // Check that all byte sizes in location_spec_infos are positive
    for (const auto &entry : location_spec_infos_) {
        if (entry.second <= 0) {
            KVCM_LOG_ERROR("location_spec_info [%s] has invalid byte size [%ld]", entry.first.c_str(), entry.second);
            return false;
        }
    }
    for (const auto &entry : location_spec_groups_) {
        for (const std::string &spec_name : entry.second) {
            if (location_spec_infos_.count(spec_name) == 0) {
                KVCM_LOG_ERROR(
                    "in group [%s], not find spec [%s] in location_spec_infos", entry.first.c_str(), spec_name.c_str());
                return false;
            }
        }
    }
    return true;
}

} // namespace kv_cache_manager