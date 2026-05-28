#include "kv_cache_manager/optimizer/config/instance_group_config.h"

namespace kv_cache_manager {

bool OptInstanceGroupConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "group_name", group_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "used_percentage", used_percentage_);
    KVCM_JSON_GET_MACRO(rapid_value, "hierarchical_eviction_enabled", hierarchical_eviction_enabled_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "default_block_ttl_seconds", default_block_ttl_seconds_, int64_t(0));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "ttl_refresh_on_read", ttl_refresh_on_read_, true);
    KVCM_JSON_GET_MACRO(rapid_value, "instances", instances_);
    // quota_capacity in config is in GB; convert to bytes
    double quota_capacity_gb = 0.0;
    KVCM_JSON_GET_MACRO(rapid_value, "quota_capacity", quota_capacity_gb);
    quota_capacity_ = static_cast<int64_t>(quota_capacity_gb * static_cast<double>(1LL << 30));
    // Parse storages; tier capacity is in GB in config, OptTierConfig::FromRapidValue handles conversion
    KVCM_JSON_GET_MACRO(rapid_value, "storages", storages_);
    // tier_write_mode 为可选字段，缺省回退到 "write_through" 保证向后兼容
    std::string tier_write_mode_str;
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "tier_write_mode", tier_write_mode_str, std::string("write_through"));
    tier_write_mode_ = ToTierWriteMode(tier_write_mode_str);
    return true;
};

void OptInstanceGroupConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "group_name", group_name_);
    // Write quota_capacity in GB
    const double quota_gb = static_cast<double>(quota_capacity_) / static_cast<double>(1LL << 30);
    Put(writer, "quota_capacity", quota_gb);
    Put(writer, "used_percentage", used_percentage_);
    Put(writer, "hierarchical_eviction_enabled", hierarchical_eviction_enabled_);
    Put(writer, "tier_write_mode", ToString(tier_write_mode_));
    Put(writer, "default_block_ttl_seconds", default_block_ttl_seconds_);
    Put(writer, "ttl_refresh_on_read", ttl_refresh_on_read_);
    Put(writer, "storages", storages_);
    Put(writer, "instances", instances_);
};
} // namespace kv_cache_manager
