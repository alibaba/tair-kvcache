#include "kv_cache_manager/optimizer/config/instance_config.h"

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {
bool OptInstanceConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "instance_id", instance_id_);
    KVCM_JSON_GET_MACRO(rapid_value, "block_size", block_size_);
    KVCM_JSON_GET_MACRO(rapid_value, "instance_group_name", instance_group_name_);
    std::string eviction_policy_type_str;
    KVCM_JSON_GET_MACRO(rapid_value, "eviction_policy_type", eviction_policy_type_str);
    eviction_policy_type_ = ToEvictionPolicyType(eviction_policy_type_str);
    if (eviction_policy_type_ == EvictionPolicyType::POLICY_LRU ||
        eviction_policy_type_ == EvictionPolicyType::POLICY_LEAF_AWARE_LRU) {
        LruParams lru_params;
        KVCM_JSON_GET_MACRO(rapid_value, "eviction_policy_params", lru_params);
        eviction_policy_param_ = lru_params;
    } else if (eviction_policy_type_ == EvictionPolicyType::POLICY_RANDOM_LRU) {
        RandomLruParams random_lru_params;
        KVCM_JSON_GET_MACRO(rapid_value, "eviction_policy_params", random_lru_params);
        eviction_policy_param_ = random_lru_params;
    } else {
        KVCM_LOG_ERROR("Unknown eviction policy type: %s", eviction_policy_type_str.c_str());
    }
    return true;
};

void OptInstanceConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "instance_id", instance_id_);
    Put(writer, "block_size", block_size_);
    Put(writer, "instance_group_name", instance_group_name_);
    Put(writer, "eviction_policy_type", ToString(eviction_policy_type_));
    if (eviction_policy_type_ == EvictionPolicyType::POLICY_LRU ||
        eviction_policy_type_ == EvictionPolicyType::POLICY_LEAF_AWARE_LRU) {
        Put(writer, "eviction_policy_params", std::get<LruParams>(eviction_policy_param_));
    } else if (eviction_policy_type_ == EvictionPolicyType::POLICY_RANDOM_LRU) {
        Put(writer, "eviction_policy_params", std::get<RandomLruParams>(eviction_policy_param_));
    }
};
} // namespace kv_cache_manager