#include "kv_cache_manager/config/trigger_strategy.h"

namespace kv_cache_manager {

TriggerStrategy::~TriggerStrategy() = default;

bool TriggerStrategy::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "used_size", used_size_);
    KVCM_JSON_GET_MACRO(rapid_value, "used_percentage", used_percentage_);
    return true;
}

void TriggerStrategy::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "used_size", used_size_);
    Put(writer, "used_percentage", used_percentage_);
}

} // namespace kv_cache_manager