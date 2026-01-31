#include "kv_cache_manager/config/cache_config.h"

namespace kv_cache_manager {

CacheConfig::~CacheConfig() = default;

bool CacheConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "reclaim_strategy", reclaim_strategy_);
    KVCM_JSON_GET_MACRO(rapid_value, "cache_prefer_strategy", cache_prefer_strategy_);
    KVCM_JSON_GET_MACRO(rapid_value, "meta_indexer_config", meta_indexer_config_);
    return true;
}

void CacheConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "reclaim_strategy", reclaim_strategy_);
    Put(writer, "cache_prefer_strategy", cache_prefer_strategy_);
    Put(writer, "meta_indexer_config", meta_indexer_config_);
}
bool CacheConfig::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    if (cache_prefer_strategy_ == CachePreferStrategy::CPS_UNSPECIFIED) {
        valid = false;
        local_invalid_fields += "{cache_prefer_strategy}";
    }
    if (reclaim_strategy_ == nullptr) {
        valid = false;
        local_invalid_fields += "{reclaim_strategy}";
    } else if (!reclaim_strategy_->ValidateRequiredFields(local_invalid_fields)) {
        valid = false;
    }
    if (meta_indexer_config_ == nullptr) {
        valid = false;
        local_invalid_fields += "{meta_indexer_config}";
    } else if (!meta_indexer_config_->ValidateRequiredFields(local_invalid_fields)) {
        valid = false;
    }
    if (!valid) {
        invalid_fields += "{CacheConfig: " + local_invalid_fields + "}";
    }
    return valid;
}
} // namespace kv_cache_manager