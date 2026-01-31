#include "kv_cache_manager/optimizer/config/tier_config.h"
namespace kv_cache_manager {
bool OptTierConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "unique_name", unique_name_);
    std::string storage_type_str;
    KVCM_JSON_GET_MACRO(rapid_value, "storage_type", storage_type_str);
    storage_type_ = ToDataStorageType(storage_type_str);
    KVCM_JSON_GET_MACRO(rapid_value, "band_width_mbps", band_width_mbps_);
    KVCM_JSON_GET_MACRO(rapid_value, "priority", priority_);
    KVCM_JSON_GET_MACRO(rapid_value, "capacity", capacity_);
    return true;
};

void OptTierConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "unique_name", unique_name_);
    Put(writer, "storage_type", ToString(storage_type_));
    Put(writer, "band_width_mbps", band_width_mbps_);
    Put(writer, "priority", priority_);
    Put(writer, "capacity", capacity_);
};
} // namespace kv_cache_manager