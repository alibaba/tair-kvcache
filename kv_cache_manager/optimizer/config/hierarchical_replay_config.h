#pragma once

#include <string>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/config/types.h"

namespace kv_cache_manager {

class StoragePoolFlowConfig : public Jsonizable {
public:
    StoragePoolFlowConfig() = default;
    ~StoragePoolFlowConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] TierWriteMode write_mode() const { return write_mode_; }
    [[nodiscard]] bool local_read_touch_enabled() const { return local_read_touch_enabled_; }
    [[nodiscard]] bool shadow_write_touch_enabled() const { return shadow_write_touch_enabled_; }
    [[nodiscard]] bool promote_enabled() const { return promote_enabled_; }
    [[nodiscard]] size_t selective_write_threshold() const { return selective_write_threshold_; }

    void set_write_mode(TierWriteMode mode) { write_mode_ = mode; }
    void set_local_read_touch_enabled(bool enabled) { local_read_touch_enabled_ = enabled; }
    void set_shadow_write_touch_enabled(bool enabled) { shadow_write_touch_enabled_ = enabled; }
    void set_promote_enabled(bool enabled) { promote_enabled_ = enabled; }
    void set_selective_write_threshold(size_t threshold) { selective_write_threshold_ = threshold; }

private:
    TierWriteMode write_mode_ = TierWriteMode::WRITE_THROUGH;
    bool local_read_touch_enabled_ = false;
    bool shadow_write_touch_enabled_ = false;
    bool promote_enabled_ = false;
    size_t selective_write_threshold_ = 2;
};

class EngineToStoragePoolMappingConfig : public Jsonizable {
public:
    EngineToStoragePoolMappingConfig() = default;
    ~EngineToStoragePoolMappingConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] const std::string &engine_instance_id() const { return engine_instance_id_; }
    [[nodiscard]] const std::string &storage_pool_instance_id() const { return storage_pool_instance_id_; }
    [[nodiscard]] const StoragePoolFlowConfig &storage_pool_flow() const { return storage_pool_flow_; }

    void set_engine_instance_id(const std::string &instance_id) { engine_instance_id_ = instance_id; }
    void set_storage_pool_instance_id(const std::string &instance_id) { storage_pool_instance_id_ = instance_id; }
    void set_storage_pool_flow(const StoragePoolFlowConfig &flow) { storage_pool_flow_ = flow; }

private:
    std::string engine_instance_id_;
    std::string storage_pool_instance_id_;
    StoragePoolFlowConfig storage_pool_flow_;
};

class HierarchicalModelConfig : public Jsonizable {
public:
    HierarchicalModelConfig() = default;
    ~HierarchicalModelConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] int32_t block_size() const { return block_size_; }
    [[nodiscard]] int64_t bytes_per_token() const { return bytes_per_token_; }
    [[nodiscard]] EvictionPolicyType eviction_policy_type() const { return eviction_policy_type_; }
    [[nodiscard]] const EvictionPolicyParam &eviction_policy_param() const { return eviction_policy_param_; }

private:
    int32_t block_size_ = 0;
    int64_t bytes_per_token_ = 0;
    EvictionPolicyType eviction_policy_type_ = EvictionPolicyType::POLICY_UNSPECIFIED;
    EvictionPolicyParam eviction_policy_param_;
};

class HierarchicalTierConfig : public Jsonizable {
public:
    HierarchicalTierConfig() = default;
    ~HierarchicalTierConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] const std::string &name() const { return name_; }
    [[nodiscard]] double capacity() const { return capacity_; }

private:
    std::string name_;
    double capacity_ = 0.0;
};

class InferClusterConfig : public Jsonizable {
public:
    InferClusterConfig() = default;
    ~InferClusterConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] const std::string &storage_pool_instance_id() const { return storage_pool_instance_id_; }
    [[nodiscard]] const HierarchicalModelConfig &model() const { return model_; }
    [[nodiscard]] const std::vector<std::string> &infer_ids() const { return infer_ids_; }
    [[nodiscard]] const OptTtlConfig &ttl_config() const { return ttl_config_; }
    [[nodiscard]] const std::vector<HierarchicalTierConfig> &tiers() const { return tiers_; }
    [[nodiscard]] const std::vector<OptTierFlowConfig> &tier_flows() const { return tier_flows_; }
    [[nodiscard]] const StoragePoolFlowConfig &storage_pool_flow() const { return storage_pool_flow_; }

private:
    std::string storage_pool_instance_id_;
    HierarchicalModelConfig model_;
    std::vector<std::string> infer_ids_;
    OptTtlConfig ttl_config_;
    std::vector<HierarchicalTierConfig> tiers_;
    std::vector<OptTierFlowConfig> tier_flows_;
    StoragePoolFlowConfig storage_pool_flow_;
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
    [[nodiscard]] const OptimizerConfig &storage_pool_config() const { return storage_pool_config_; }
    [[nodiscard]] const EvictionConfig &infer_eviction_config() const { return infer_eviction_config_; }
    [[nodiscard]] const std::vector<EngineToStoragePoolMappingConfig> &engine_to_storage_pool() const {
        return engine_to_storage_pool_;
    }
    [[nodiscard]] const std::string &infer_scheduling_strategy() const { return infer_scheduling_strategy_; }
    [[nodiscard]] bool enable_lifecycle_tracking() const { return enable_lifecycle_tracking_; }
    [[nodiscard]] const std::vector<InferClusterConfig> &infer_clusters() const { return infer_clusters_; }

    void set_trace_file_path(const std::string &path) { trace_file_path_ = path; }
    void set_output_result_path(const std::string &path) { output_result_path_ = path; }
    void set_infer_eviction_config(const EvictionConfig &config) { infer_eviction_config_ = config; }
    void set_engine_config(const OptimizerConfig &config) { engine_config_ = config; }
    void set_storage_pool_config(const OptimizerConfig &config) { storage_pool_config_ = config; }
    void set_engine_to_storage_pool(const std::vector<EngineToStoragePoolMappingConfig> &mapping) {
        engine_to_storage_pool_ = mapping;
    }
    void set_infer_scheduling_strategy(const std::string &strategy) { infer_scheduling_strategy_ = strategy; }
    void set_enable_lifecycle_tracking(bool enabled) { enable_lifecycle_tracking_ = enabled; }

private:
    bool BuildOptimizerConfigs();

    std::string trace_file_path_;
    std::string output_result_path_;
    EvictionConfig infer_eviction_config_;
    std::vector<InferClusterConfig> infer_clusters_;
    OptimizerConfig engine_config_;
    OptimizerConfig storage_pool_config_;
    std::vector<EngineToStoragePoolMappingConfig> engine_to_storage_pool_;
    std::string infer_scheduling_strategy_ = "preserve_trace";
    bool enable_lifecycle_tracking_ = false;
};

} // namespace kv_cache_manager
