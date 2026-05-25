#pragma once

#include <string>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/config/types.h"

namespace kv_cache_manager {

class L2L3StrategyConfig : public Jsonizable {
public:
    L2L3StrategyConfig() = default;
    ~L2L3StrategyConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] TierWriteMode write_mode() const { return write_mode_; }
    [[nodiscard]] bool access_propagation_enabled() const { return access_propagation_enabled_; }
    [[nodiscard]] bool promote_enabled() const { return promote_enabled_; }
    [[nodiscard]] size_t selective_write_threshold() const { return selective_write_threshold_; }

    void set_write_mode(TierWriteMode mode) { write_mode_ = mode; }
    void set_access_propagation_enabled(bool enabled) { access_propagation_enabled_ = enabled; }
    void set_promote_enabled(bool enabled) { promote_enabled_ = enabled; }
    void set_selective_write_threshold(size_t threshold) { selective_write_threshold_ = threshold; }

private:
    TierWriteMode write_mode_ = TierWriteMode::WRITE_THROUGH;
    bool access_propagation_enabled_ = false;
    bool promote_enabled_ = false;
    size_t selective_write_threshold_ = 2;
};

class EngineToPoolMappingConfig : public Jsonizable {
public:
    EngineToPoolMappingConfig() = default;
    ~EngineToPoolMappingConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] const std::string &engine_instance_id() const { return engine_instance_id_; }
    [[nodiscard]] const std::string &pool_instance_id() const { return pool_instance_id_; }

    void set_engine_instance_id(const std::string &instance_id) { engine_instance_id_ = instance_id; }
    void set_pool_instance_id(const std::string &instance_id) { pool_instance_id_ = instance_id; }

private:
    std::string engine_instance_id_;
    std::string pool_instance_id_;
};

class HierarchicalReplayConfig : public Jsonizable {
public:
    HierarchicalReplayConfig() = default;
    ~HierarchicalReplayConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] const std::string &trace_file_path() const { return trace_file_path_; }
    [[nodiscard]] const std::string &output_result_path() const { return output_result_path_; }
    [[nodiscard]] const OptimizerConfig &engine_config() const { return engine_config_; }
    [[nodiscard]] const OptimizerConfig &pool_config() const { return pool_config_; }
    [[nodiscard]] const std::vector<EngineToPoolMappingConfig> &engine_to_pool() const { return engine_to_pool_; }
    [[nodiscard]] const std::string &engine_scheduling_strategy() const { return engine_scheduling_strategy_; }
    [[nodiscard]] const L2L3StrategyConfig &l2_l3_strategy() const { return l2_l3_strategy_; }

    void set_trace_file_path(const std::string &path) { trace_file_path_ = path; }
    void set_output_result_path(const std::string &path) { output_result_path_ = path; }
    void set_engine_config(const OptimizerConfig &config) { engine_config_ = config; }
    void set_pool_config(const OptimizerConfig &config) { pool_config_ = config; }
    void set_engine_to_pool(const std::vector<EngineToPoolMappingConfig> &mapping) { engine_to_pool_ = mapping; }
    void set_engine_scheduling_strategy(const std::string &strategy) { engine_scheduling_strategy_ = strategy; }
    void set_l2_l3_strategy(const L2L3StrategyConfig &strategy) { l2_l3_strategy_ = strategy; }

private:
    std::string trace_file_path_;
    std::string output_result_path_;
    OptimizerConfig engine_config_;
    OptimizerConfig pool_config_;
    std::vector<EngineToPoolMappingConfig> engine_to_pool_;
    std::string engine_scheduling_strategy_ = "preserve_trace";
    L2L3StrategyConfig l2_l3_strategy_;
};

} // namespace kv_cache_manager
