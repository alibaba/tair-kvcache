#include "kv_cache_manager/config/quota_config.h"

namespace kv_cache_manager {

QuotaConfig::~QuotaConfig() = default;

bool QuotaConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "capacity", capacity_);
    std::string storage_type_str;
    KVCM_JSON_GET_MACRO(rapid_value, "storage_type", storage_type_str);
    storage_type_ = ToDataStorageType(storage_type_str);
    return true;
}

void QuotaConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "capacity", capacity_);
    Put(writer, "storage_type", ToString(storage_type_));
    Put(writer, "storage_type", storage_type_);
}
bool QuotaConfig::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;

    if (storage_type_ == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN) {
        valid = false;
        invalid_fields += "{storage_spec}";
    }
    if (!valid) {
        invalid_fields = "{QuotaConfig: " + invalid_fields + "}";
    }
    return valid;
}
} // namespace kv_cache_manager