#include "kv_cache_manager/optimizer/config/optimizer_config.h"

namespace kv_cache_manager {
bool OptimizerConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "trace_file_path", trace_file_path_);
    std::string trace_type_str;
    KVCM_JSON_GET_MACRO(rapid_value, "trace_type", trace_type_str);
    trace_type_ = ToTraceType(trace_type_str);
    KVCM_JSON_GET_MACRO(rapid_value, "output_result_path", output_result_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "eviction_params", eviction_config_);
    KVCM_JSON_GET_MACRO(rapid_value, "instance_groups", instance_groups_);
    return true;
};

void OptimizerConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "trace_file_path", trace_file_path_);
    Put(writer, "trace_type", ToString(trace_type_));
    Put(writer, "output_result_path", output_result_path_);
    Put(writer, "eviction_params", eviction_config_);
    Put(writer, "instance_groups", instance_groups_);
}
} // namespace kv_cache_manager