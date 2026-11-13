#include "kv_cache_manager/config/quota_config.h"

#include <cmath>

namespace kv_cache_manager {

QuotaConfig::~QuotaConfig() = default;

bool QuotaConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "capacity", capacity_);
    std::string storage_type_str;
    KVCM_JSON_GET_MACRO(rapid_value, "storage_type", storage_type_str);
    storage_type_ = ToDataStorageType(storage_type_str);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "mode", mode_, std::string("fixed"));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "ratio", ratio_, 1.0);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "storage_names", storage_names_, std::vector<std::string>());
    return true;
}

void QuotaConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "capacity", capacity_);
    Put(writer, "storage_type", ToString(storage_type_));
    Put(writer, "mode", mode_);
    Put(writer, "ratio", ratio_);
    Put(writer, "storage_names", storage_names_);
}
bool QuotaConfig::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    if ((mode_ != "auto" && mode_ != "fixed") || !std::isfinite(ratio_) || ratio_ <= 0 || ratio_ > 1) {
        valid = false;
        invalid_fields += "{mode/ratio}";
    }

    if (storage_type_ == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN ||
        ToIndex(storage_type_) >= ToIndex(DataStorageType::COUNT)) {
        valid = false;
        invalid_fields += "{storage_spec}";
    }
    for (const auto &name : storage_names_) {
        if (name.empty()) {
            valid = false;
            invalid_fields += "{storage_names}";
        }
    }
    if (!valid) {
        invalid_fields = "{QuotaConfig: " + invalid_fields + "}";
    }
    return valid;
}
} // namespace kv_cache_manager