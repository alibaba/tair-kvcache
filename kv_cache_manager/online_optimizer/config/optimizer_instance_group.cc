#include "kv_cache_manager/online_optimizer/config/optimizer_instance_group.h"

namespace kv_cache_manager {

bool OptimizerInstanceGroup::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "name", name_, std::string(""));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "enabled", enabled_, false);
    KVCM_JSON_GET_MACRO(rapid_value, "capacity_gb", capacity_gb_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "primary_capacity_index", primary_capacity_index_, int32_t(0));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "indexer_type", indexer_type_, std::string("fenwick_lru"));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "max_key_count", max_key_count_, int64_t(0));
    return true;
}

void OptimizerInstanceGroup::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "name", name_);
    Put(writer, "enabled", enabled_);
    Put(writer, "capacity_gb", capacity_gb_);
    Put(writer, "primary_capacity_index", primary_capacity_index_);
    Put(writer, "indexer_type", indexer_type_);
    Put(writer, "max_key_count", max_key_count_);
}

bool OptimizerInstanceGroup::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    if (name_.empty()) {
        valid = false;
        local_invalid_fields += "{name}";
    }
    if (enabled_ && capacity_gb_.empty()) {
        valid = false;
        local_invalid_fields += "{capacity_gb}";
    }
    if (enabled_ && primary_capacity_index_ < 0) {
        valid = false;
        local_invalid_fields += "{primary_capacity_index}";
    }
    if (enabled_ && !capacity_gb_.empty() &&
        primary_capacity_index_ >= static_cast<int32_t>(capacity_gb_.size())) {
        valid = false;
        local_invalid_fields += "{primary_capacity_index out of range}";
    }
    if (indexer_type_ != "bst_lru" && indexer_type_ != "fenwick_lru") {
        valid = false;
        local_invalid_fields += "{indexer_type}";
    }
    if (max_key_count_ < 0) {
        valid = false;
        local_invalid_fields += "{max_key_count}";
    }
    if (!valid) {
        invalid_fields += "{OptimizerInstanceGroup: " + local_invalid_fields + "}";
    }
    return valid;
}

} // namespace kv_cache_manager
