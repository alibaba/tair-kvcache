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
    [[nodiscard]] size_t selective_write_threshold() const { return selective_write_threshold_; }

    void set_write_mode(TierWriteMode mode) { write_mode_ = mode; }
    void set_local_read_touch_enabled(bool enabled) { local_read_touch_enabled_ = enabled; }
    void set_shadow_write_touch_enabled(bool enabled) { shadow_write_touch_enabled_ = enabled; }
    void set_selective_write_threshold(size_t threshold) { selective_write_threshold_ = threshold; }

private:
    TierWriteMode write_mode_ = TierWriteMode::WRITE_THROUGH;
    bool local_read_touch_enabled_ = false;
    bool shadow_write_touch_enabled_ = false;
    size_t selective_write_threshold_ = 2;
};

class EngineToStoragePoolMappingConfig : public Jsonizable {
public:
    EngineToStoragePoolMappingConfig() = default;
    ~EngineToStoragePoolMappingConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] const std::string &engine_instance_id() const { return engine_instance_id_; }
    [[nodiscard]] const std::string &storage_pool_id() const { return storage_pool_id_; }
    [[nodiscard]] const std::string &engine_read_query_type() const { return engine_read_query_type_; }
    [[nodiscard]] const StoragePoolFlowConfig &storage_pool_flow() const { return storage_pool_flow_; }

    void set_engine_instance_id(const std::string &instance_id) { engine_instance_id_ = instance_id; }
    void set_storage_pool_id(const std::string &pool_id) { storage_pool_id_ = pool_id; }
    void set_engine_read_query_type(const std::string &query_type) { engine_read_query_type_ = query_type; }
    void set_storage_pool_flow(const StoragePoolFlowConfig &flow) { storage_pool_flow_ = flow; }

private:
    std::string engine_instance_id_;
    std::string storage_pool_id_;
    std::string engine_read_query_type_;
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

    void set_block_size(int32_t block_size) { block_size_ = block_size; }
    void set_bytes_per_token(int64_t bytes_per_token) { bytes_per_token_ = bytes_per_token; }
    void set_eviction_policy_type(EvictionPolicyType type) { eviction_policy_type_ = type; }
    void set_eviction_policy_param(const EvictionPolicyParam &param) { eviction_policy_param_ = param; }

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

    void set_name(const std::string &name) { name_ = name; }
    void set_capacity(double capacity) { capacity_ = capacity; }

private:
    std::string name_;
    double capacity_ = 0.0;
};

class HierarchicalStoragePoolEntryConfig : public Jsonizable {
public:
    HierarchicalStoragePoolEntryConfig() = default;
    ~HierarchicalStoragePoolEntryConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] const std::string &pool_id() const { return pool_id_; }
    [[nodiscard]] const HierarchicalModelConfig &model() const { return model_; }

    void set_pool_id(const std::string &pool_id) { pool_id_ = pool_id; }
    void set_model(const HierarchicalModelConfig &model) { model_ = model; }

private:
    std::string pool_id_;
    HierarchicalModelConfig model_;
};

class HierarchicalStoragePoolConfig : public Jsonizable {
public:
    HierarchicalStoragePoolConfig() = default;
    ~HierarchicalStoragePoolConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] const std::string &output_result_path() const { return output_result_path_; }
    [[nodiscard]] const std::string &storage_name() const { return storage_name_; }
    [[nodiscard]] double capacity() const { return capacity_; }
    [[nodiscard]] const EvictionConfig &eviction_config() const { return eviction_config_; }
    [[nodiscard]] const OptTtlConfig &ttl_config() const { return ttl_config_; }
    [[nodiscard]] const std::vector<HierarchicalStoragePoolEntryConfig> &pools() const { return pools_; }

    void set_output_result_path(const std::string &path) { output_result_path_ = path; }
    void set_storage_name(const std::string &name) { storage_name_ = name; }
    void set_capacity(double capacity) { capacity_ = capacity; }
    void set_eviction_config(const EvictionConfig &config) { eviction_config_ = config; }
    void set_ttl_config(const OptTtlConfig &config) { ttl_config_ = config; }
    void set_pools(const std::vector<HierarchicalStoragePoolEntryConfig> &pools) { pools_ = pools; }

private:
    std::string output_result_path_;
    std::string storage_name_;
    double capacity_ = 0.0;
    EvictionConfig eviction_config_;
    OptTtlConfig ttl_config_;
    std::vector<HierarchicalStoragePoolEntryConfig> pools_;
};

class P2PReadFlowConfig : public Jsonizable {
public:
    P2PReadFlowConfig() = default;
    ~P2PReadFlowConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] const std::string &tier() const { return tier_; }
    [[nodiscard]] bool peer_read_touch_enabled() const { return peer_read_touch_enabled_; }

    void set_tier(const std::string &tier) { tier_ = tier; }
    void set_peer_read_touch_enabled(bool enabled) { peer_read_touch_enabled_ = enabled; }

private:
    std::string tier_;
    bool peer_read_touch_enabled_ = true;
};

class InferActiveWindowConfig : public Jsonizable {
public:
    InferActiveWindowConfig() = default;
    ~InferActiveWindowConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] const std::string &infer_id() const { return infer_id_; }
    [[nodiscard]] int64_t start_ns() const { return start_ns_; }
    [[nodiscard]] int64_t end_ns() const { return end_ns_; }

    void set_infer_id(const std::string &infer_id) { infer_id_ = infer_id; }
    void set_start_ns(int64_t start_ns) { start_ns_ = start_ns; }
    void set_end_ns(int64_t end_ns) { end_ns_ = end_ns; }

private:
    std::string infer_id_;
    int64_t start_ns_ = 0;
    int64_t end_ns_ = 0;
};

class InferClusterConfig : public Jsonizable {
public:
    InferClusterConfig() = default;
    ~InferClusterConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] const std::string &storage_pool_id() const { return storage_pool_id_; }
    [[nodiscard]] const std::string &engine_read_query_type() const { return engine_read_query_type_; }
    [[nodiscard]] const HierarchicalModelConfig &model() const { return model_; }
    [[nodiscard]] const std::vector<std::string> &infer_ids() const { return infer_ids_; }
    [[nodiscard]] const OptTtlConfig &ttl_config() const { return ttl_config_; }
    [[nodiscard]] const std::vector<HierarchicalTierConfig> &tiers() const { return tiers_; }
    [[nodiscard]] const std::vector<OptTierFlowConfig> &tier_flows() const { return tier_flows_; }
    [[nodiscard]] const std::vector<P2PReadFlowConfig> &p2p_read_flows() const { return p2p_read_flows_; }
    [[nodiscard]] const std::vector<InferActiveWindowConfig> &active_windows() const { return active_windows_; }
    [[nodiscard]] const StoragePoolFlowConfig &storage_pool_flow() const { return storage_pool_flow_; }

    void set_storage_pool_id(const std::string &storage_pool_id) { storage_pool_id_ = storage_pool_id; }
    void set_engine_read_query_type(const std::string &query_type) { engine_read_query_type_ = query_type; }
    void set_model(const HierarchicalModelConfig &model) { model_ = model; }
    void set_infer_ids(const std::vector<std::string> &infer_ids) { infer_ids_ = infer_ids; }
    void set_ttl_config(const OptTtlConfig &ttl_config) { ttl_config_ = ttl_config; }
    void set_tiers(const std::vector<HierarchicalTierConfig> &tiers) { tiers_ = tiers; }
    void set_tier_flows(const std::vector<OptTierFlowConfig> &tier_flows) { tier_flows_ = tier_flows; }
    void set_p2p_read_flows(const std::vector<P2PReadFlowConfig> &flows) { p2p_read_flows_ = flows; }
    void set_active_windows(const std::vector<InferActiveWindowConfig> &windows) { active_windows_ = windows; }
    void set_storage_pool_flow(const StoragePoolFlowConfig &flow) { storage_pool_flow_ = flow; }

private:
    std::string storage_pool_id_;
    std::string engine_read_query_type_;
    HierarchicalModelConfig model_;
    std::vector<std::string> infer_ids_;
    OptTtlConfig ttl_config_;
    std::vector<HierarchicalTierConfig> tiers_;
    std::vector<OptTierFlowConfig> tier_flows_;
    std::vector<P2PReadFlowConfig> p2p_read_flows_;
    std::vector<InferActiveWindowConfig> active_windows_;
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
    [[nodiscard]] const HierarchicalStoragePoolConfig &storage_pool() const { return storage_pool_; }
    [[nodiscard]] const EvictionConfig &infer_eviction_config() const { return infer_eviction_config_; }
    [[nodiscard]] const OptTraceReplayConfig &trace_replay_config() const { return trace_replay_config_; }
    [[nodiscard]] const std::vector<EngineToStoragePoolMappingConfig> &engine_to_storage_pool() const {
        return engine_to_storage_pool_;
    }
    [[nodiscard]] const std::string &infer_scheduling_strategy() const { return infer_scheduling_strategy_; }
    [[nodiscard]] bool infer_active_windows_from_trace() const { return infer_active_windows_from_trace_; }
    [[nodiscard]] bool enable_lifecycle_tracking() const { return enable_lifecycle_tracking_; }
    [[nodiscard]] const std::vector<InferClusterConfig> &infer_clusters() const { return infer_clusters_; }

    void set_trace_file_path(const std::string &path) { trace_file_path_ = path; }
    void set_output_result_path(const std::string &path) { output_result_path_ = path; }
    void set_infer_eviction_config(const EvictionConfig &config) { infer_eviction_config_ = config; }
    void set_trace_replay_config(const OptTraceReplayConfig &config) { trace_replay_config_ = config; }
    void set_engine_config(const OptimizerConfig &config) { engine_config_ = config; }
    void set_storage_pool(const HierarchicalStoragePoolConfig &config) { storage_pool_ = config; }
    void set_infer_clusters(const std::vector<InferClusterConfig> &clusters) { infer_clusters_ = clusters; }
    void set_engine_to_storage_pool(const std::vector<EngineToStoragePoolMappingConfig> &mapping) {
        engine_to_storage_pool_ = mapping;
    }
    void set_infer_scheduling_strategy(const std::string &strategy) { infer_scheduling_strategy_ = strategy; }
    void set_infer_active_windows_from_trace(bool enabled) { infer_active_windows_from_trace_ = enabled; }
    void set_enable_lifecycle_tracking(bool enabled) { enable_lifecycle_tracking_ = enabled; }

private:
    bool BuildOptimizerConfigs();

    std::string trace_file_path_;
    std::string output_result_path_;
    EvictionConfig infer_eviction_config_;
    OptTraceReplayConfig trace_replay_config_;
    std::vector<InferClusterConfig> infer_clusters_;
    OptimizerConfig engine_config_;
    HierarchicalStoragePoolConfig storage_pool_;
    std::vector<EngineToStoragePoolMappingConfig> engine_to_storage_pool_;
    std::string infer_scheduling_strategy_ = "preserve_trace";
    bool infer_active_windows_from_trace_ = false;
    bool enable_lifecycle_tracking_ = false;
};

} // namespace kv_cache_manager
