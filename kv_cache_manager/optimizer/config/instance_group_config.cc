#include "kv_cache_manager/optimizer/config/instance_group_config.h"

namespace kv_cache_manager {

bool OptTierStrategyConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    return ParseFields(rapid_value, "write_mode", "access_propagation_enabled", "promote_enabled");
}

bool OptTierStrategyConfig::FromLegacyGroupRapidValue(const rapidjson::Value &rapid_value) {
    return ParseFields(rapid_value, "tier_write_mode", "tier_access_propagation_enabled", "enable_promote");
}

bool OptTierStrategyConfig::ParseFields(const rapidjson::Value &rapid_value,
                                        const char *write_mode_key,
                                        const char *access_propagation_key,
                                        const char *promote_key) {
    if (!rapid_value.IsObject()) {
        return false;
    }
    KVCM_JSON_GET_MACRO(rapid_value, "hierarchical_eviction_enabled", hierarchical_eviction_enabled_);
    std::string write_mode_str;
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, write_mode_key, write_mode_str, std::string("write_through"));
    if (!IsValidTierWriteMode(write_mode_str)) {
        return false;
    }
    write_mode_ = ToTierWriteMode(write_mode_str);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, access_propagation_key, access_propagation_enabled_, true);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, promote_key, promote_enabled_, true);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "selective_write_threshold", selective_write_threshold_, int64_t(2));
    if (selective_write_threshold_ <= 0) {
        return false;
    }
    return true;
}

void OptTierStrategyConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "hierarchical_eviction_enabled", hierarchical_eviction_enabled_);
    Put(writer, "write_mode", ToString(write_mode_));
    Put(writer, "access_propagation_enabled", access_propagation_enabled_);
    Put(writer, "promote_enabled", promote_enabled_);
    Put(writer, "selective_write_threshold", selective_write_threshold_);
}

bool OptInstanceGroupConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "group_name", group_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "used_percentage", used_percentage_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "default_block_ttl_seconds", default_block_ttl_seconds_, int64_t(0));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "ttl_refresh_on_read", ttl_refresh_on_read_, true);
    KVCM_JSON_GET_MACRO(rapid_value, "instances", instances_);
    // quota_capacity in config is in GB; convert to bytes
    double quota_capacity_gb = 0.0;
    KVCM_JSON_GET_MACRO(rapid_value, "quota_capacity", quota_capacity_gb);
    quota_capacity_ = static_cast<int64_t>(quota_capacity_gb * static_cast<double>(1LL << 30));
    // Parse storages; tier capacity is in GB in config, OptTierConfig::FromRapidValue handles conversion
    KVCM_JSON_GET_MACRO(rapid_value, "storages", storages_);
    if (rapid_value.HasMember("tier_strategy")) {
        if (!tier_strategy_.FromRapidValue(rapid_value["tier_strategy"])) {
            return false;
        }
    } else if (!tier_strategy_.FromLegacyGroupRapidValue(rapid_value)) {
        return false;
    }
    return true;
};

void OptInstanceGroupConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "group_name", group_name_);
    // Write quota_capacity in GB
    const double quota_gb = static_cast<double>(quota_capacity_) / static_cast<double>(1LL << 30);
    Put(writer, "quota_capacity", quota_gb);
    Put(writer, "used_percentage", used_percentage_);
    Put(writer, "tier_strategy", tier_strategy_);
    Put(writer, "default_block_ttl_seconds", default_block_ttl_seconds_);
    Put(writer, "ttl_refresh_on_read", ttl_refresh_on_read_);
    Put(writer, "storages", storages_);
    Put(writer, "instances", instances_);
};
} // namespace kv_cache_manager
