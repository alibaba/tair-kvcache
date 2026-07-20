#include "kv_cache_manager/optimizer/config/optimizer_lite_hit_config.h"

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

bool OptimizerLiteHitConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "trace_file_path", trace_file_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "output_result_path", output_result_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "instance_groups", instance_groups_);
    KVCM_JSON_GET_MACRO(rapid_value, "instances", instances_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "assume_time_sorted", assume_time_sorted_, true);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "override_instance_id", override_instance_id_, std::string());
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "aggregate_only", aggregate_only_, false);

    if (trace_file_path_.empty()) {
        KVCM_LOG_ERROR("lite_hit config: trace_file_path must not be empty");
        return false;
    }
    if (output_result_path_.empty()) {
        KVCM_LOG_ERROR("lite_hit config: output_result_path must not be empty");
        return false;
    }
    if (instance_groups_.empty()) {
        KVCM_LOG_ERROR("lite_hit config: instance_groups must not be empty");
        return false;
    }
    if (instances_.empty()) {
        KVCM_LOG_ERROR("lite_hit config: instances must not be empty");
        return false;
    }
    return true;
}

void OptimizerLiteHitConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "trace_file_path", trace_file_path_);
    Put(writer, "output_result_path", output_result_path_);
    Put(writer, "instance_groups", instance_groups_);
    Put(writer, "instances", instances_);
    Put(writer, "assume_time_sorted", assume_time_sorted_);
    Put(writer, "override_instance_id", override_instance_id_);
    Put(writer, "aggregate_only", aggregate_only_);
}

} // namespace kv_cache_manager
