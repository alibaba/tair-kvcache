#include "kv_cache_manager/optimizer/config/hierarchical_replay_config.h"

namespace kv_cache_manager {

bool L2L3StrategyConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    if (!rapid_value.IsObject()) {
        return false;
    }
    std::string write_mode_str = ToString(write_mode_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "write_mode", write_mode_str, write_mode_str);
    if (!IsValidTierWriteMode(write_mode_str)) {
        return false;
    }
    write_mode_ = ToTierWriteMode(write_mode_str);

    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "access_propagation_enabled", access_propagation_enabled_, access_propagation_enabled_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "promote_enabled", promote_enabled_, promote_enabled_);

    int64_t threshold = static_cast<int64_t>(selective_write_threshold_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "selective_write_threshold", threshold, threshold);
    if (threshold <= 0) {
        return false;
    }
    selective_write_threshold_ = static_cast<size_t>(threshold);
    return true;
}

void L2L3StrategyConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "write_mode", ToString(write_mode_));
    Put(writer, "access_propagation_enabled", access_propagation_enabled_);
    Put(writer, "promote_enabled", promote_enabled_);
    Put(writer, "selective_write_threshold", static_cast<int64_t>(selective_write_threshold_));
}

bool EngineToPoolMappingConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "engine_instance_id", engine_instance_id_);
    KVCM_JSON_GET_MACRO(rapid_value, "pool_instance_id", pool_instance_id_);
    return !engine_instance_id_.empty() && !pool_instance_id_.empty();
}

void EngineToPoolMappingConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "engine_instance_id", engine_instance_id_);
    Put(writer, "pool_instance_id", pool_instance_id_);
}

bool HierarchicalReplayConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "trace_file_path", trace_file_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "output_result_path", output_result_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "engine_config", engine_config_);
    KVCM_JSON_GET_MACRO(rapid_value, "pool_config", pool_config_);
    KVCM_JSON_GET_MACRO(rapid_value, "engine_to_pool", engine_to_pool_);
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "engine_scheduling_strategy", engine_scheduling_strategy_, std::string("preserve_trace"));
    if (engine_scheduling_strategy_ != "preserve_trace" && engine_scheduling_strategy_ != "round_robin" &&
        engine_scheduling_strategy_ != "prefix_hit") {
        return false;
    }
    if (rapid_value.HasMember("l2_l3_strategy") && !l2_l3_strategy_.FromRapidValue(rapid_value["l2_l3_strategy"])) {
        return false;
    }
    return !trace_file_path_.empty() && !output_result_path_.empty() && !engine_to_pool_.empty();
}

void HierarchicalReplayConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "trace_file_path", trace_file_path_);
    Put(writer, "output_result_path", output_result_path_);
    Put(writer, "engine_config", engine_config_);
    Put(writer, "pool_config", pool_config_);
    Put(writer, "engine_to_pool", engine_to_pool_);
    Put(writer, "engine_scheduling_strategy", engine_scheduling_strategy_);
    Put(writer, "l2_l3_strategy", l2_l3_strategy_);
}

} // namespace kv_cache_manager
