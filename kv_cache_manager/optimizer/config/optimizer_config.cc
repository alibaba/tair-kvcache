#include "kv_cache_manager/optimizer/config/optimizer_config.h"

namespace kv_cache_manager {
bool OptTraceReplayConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "write_delay_ns", write_delay_ns_, int64_t(1));
    return write_delay_ns_ > 0;
}

void OptTraceReplayConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "write_delay_ns", write_delay_ns_);
}

bool OptimizerConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "trace_file_path", trace_file_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "output_result_path", output_result_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "eviction_params", eviction_config_);
    trace_replay_config_ = OptTraceReplayConfig();
    if (rapid_value.HasMember("trace_replay")) {
        if (!rapid_value["trace_replay"].IsObject() ||
            !trace_replay_config_.FromRapidValue(rapid_value["trace_replay"])) {
            return false;
        }
    }
    KVCM_JSON_GET_MACRO(rapid_value, "instance_groups", instance_groups_);
    return true;
};

void OptimizerConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "trace_file_path", trace_file_path_);
    Put(writer, "output_result_path", output_result_path_);
    Put(writer, "eviction_params", eviction_config_);
    Put(writer, "trace_replay", trace_replay_config_);
    Put(writer, "instance_groups", instance_groups_);
}
} // namespace kv_cache_manager
