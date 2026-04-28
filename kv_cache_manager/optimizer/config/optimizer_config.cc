#include "kv_cache_manager/optimizer/config/optimizer_config.h"

namespace kv_cache_manager {
bool OptimizerConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "trace_file_path", trace_file_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "output_result_path", output_result_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "eviction_params", eviction_config_);
    KVCM_JSON_GET_MACRO(rapid_value, "instance_groups", instance_groups_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "enable_remote_matching", enable_remote_matching_, false);
    return true;
};

void OptimizerConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "trace_file_path", trace_file_path_);
    Put(writer, "output_result_path", output_result_path_);
    Put(writer, "eviction_params", eviction_config_);
    Put(writer, "instance_groups", instance_groups_);
    Put(writer, "enable_remote_matching", enable_remote_matching_);
}
} // namespace kv_cache_manager