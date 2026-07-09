#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/replay_instance_group_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
namespace kv_cache_manager {

class OptTraceReplayConfig : public Jsonizable {
public:
    OptTraceReplayConfig() = default;
    ~OptTraceReplayConfig() override = default;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] int64_t write_delay_ns() const { return write_delay_ns_; }
    void set_write_delay_ns(int64_t delay_ns) { write_delay_ns_ = delay_ns; }

private:
    int64_t write_delay_ns_ = 1;
};

class OptimizerConfig : public Jsonizable {
public:
    OptimizerConfig() = default;
    ~OptimizerConfig() override = default;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

public:
    [[nodiscard]] const std::string &trace_file_path() const { return trace_file_path_; }
    [[nodiscard]] const std::string &output_result_path() const { return output_result_path_; }
    [[nodiscard]] const EvictionConfig &eviction_config() const { return eviction_config_; }
    [[nodiscard]] const OptTraceReplayConfig &trace_replay_config() const { return trace_replay_config_; }
    [[nodiscard]] const std::vector<OptimizerReplayInstanceGroupConfig> &instance_groups() const {
        return instance_groups_;
    }
    [[nodiscard]] std::vector<OptimizerReplayInstanceGroupConfig> &mutable_instance_groups() {
        return instance_groups_;
    }

    void set_trace_file_path(const std::string &path) { trace_file_path_ = path; }
    void set_output_result_path(const std::string &path) { output_result_path_ = path; }
    void set_eviction_params(const EvictionConfig &config) { eviction_config_ = config; }
    void set_trace_replay_config(const OptTraceReplayConfig &config) { trace_replay_config_ = config; }
    void set_instance_groups(const std::vector<OptimizerReplayInstanceGroupConfig> &groups) {
        instance_groups_ = groups;
    }

private:
    std::string trace_file_path_;
    std::string output_result_path_;
    EvictionConfig eviction_config_;
    OptTraceReplayConfig trace_replay_config_;
    std::vector<OptimizerReplayInstanceGroupConfig> instance_groups_;
};

} // namespace kv_cache_manager
