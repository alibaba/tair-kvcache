#include "kv_cache_manager/optimizer/config/eviction_config.h"

namespace kv_cache_manager {

bool EvictionConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "eviction_mode", eviction_mode_);
    KVCM_JSON_GET_MACRO(rapid_value, "eviction_batch_size_per_instance", eviction_batch_size_per_instance_);
    return true;
}
void EvictionConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "eviction_mode", eviction_mode_);
    Put(writer, "eviction_batch_size_per_instance", eviction_batch_size_per_instance_);
}
} // namespace kv_cache_manager