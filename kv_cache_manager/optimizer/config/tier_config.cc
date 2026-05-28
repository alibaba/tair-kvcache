#include "kv_cache_manager/optimizer/config/tier_config.h"
namespace kv_cache_manager {
bool OptTierConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "unique_name", unique_name_);
    std::string storage_type_str;
    KVCM_JSON_GET_MACRO(rapid_value, "storage_type", storage_type_str);
    storage_type_ = ToDataStorageType(storage_type_str);
    KVCM_JSON_GET_MACRO(rapid_value, "band_width_mbps", band_width_mbps_);
    KVCM_JSON_GET_MACRO(rapid_value, "priority", priority_);
    // capacity in config is in GB; convert to bytes
    double capacity_gb = 0.0;
    KVCM_JSON_GET_MACRO(rapid_value, "capacity", capacity_gb);
    capacity_ = capacity_gb < 0 ? -1 : static_cast<int64_t>(capacity_gb * static_cast<double>(1LL << 30));
    return true;
};

void OptTierConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "unique_name", unique_name_);
    Put(writer, "storage_type", ToString(storage_type_));
    Put(writer, "band_width_mbps", band_width_mbps_);
    Put(writer, "priority", priority_);
    // Write capacity in GB
    const double cap_gb = capacity_ < 0 ? -1.0 : static_cast<double>(capacity_) / static_cast<double>(1LL << 30);
    Put(writer, "capacity", cap_gb);
};
} // namespace kv_cache_manager
