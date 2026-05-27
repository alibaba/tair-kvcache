#include "kv_cache_manager/optimizer/config/hierarchical_replay_config.h"

#include <unordered_set>

namespace kv_cache_manager {
namespace {

constexpr double kBytesPerGb = static_cast<double>(1LL << 30);
constexpr int32_t kDefaultEvictionBatchSizePerInstance = 100;

int64_t GbToBytes(double gb) { return static_cast<int64_t>(gb * kBytesPerGb); }

std::string JoinPath(const std::string &base, const std::string &name) {
    if (base.empty() || base.back() == '/') {
        return base + name;
    }
    return base + "/" + name;
}

OptTierConfig BuildOptimizerTier(const std::string &name, double capacity, size_t priority) {
    OptTierConfig tier;
    tier.set_unique_name(name);
    tier.set_storage_type(DataStorageType::DATA_STORAGE_TYPE_DUMMY);
    tier.set_band_width_mbps(0);
    tier.set_priority(priority);
    tier.set_capacity(GbToBytes(capacity));
    return tier;
}

OptInstanceConfig BuildOptimizerInstance(const std::string &instance_id,
                                         const std::string &group_name,
                                         const HierarchicalModelConfig &model) {
    OptInstanceConfig instance;
    instance.set_instance_id(instance_id);
    instance.set_instance_group_name(group_name);
    instance.set_block_size(model.block_size());
    instance.set_bytes_per_token(model.bytes_per_token());
    instance.set_eviction_policy_type(model.eviction_policy_type());
    instance.set_eviction_policy_param(model.eviction_policy_param());
    return instance;
}

EvictionConfig BuildDefaultEvictionConfig() {
    EvictionConfig eviction_config;
    eviction_config.set_eviction_mode(EvictionMode::EVICTION_MODE_GROUP_ROUGH);
    eviction_config.set_eviction_batch_size_per_instance(kDefaultEvictionBatchSizePerInstance);
    return eviction_config;
}

} // namespace

bool L2L3StrategyConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    if (!rapid_value.IsObject()) {
        return false;
    }
    std::string write_mode_str;
    KVCM_JSON_GET_MACRO(rapid_value, "write_mode", write_mode_str);
    if (!IsValidTierWriteMode(write_mode_str)) {
        return false;
    }
    write_mode_ = ToTierWriteMode(write_mode_str);

    KVCM_JSON_GET_MACRO(rapid_value, "access_propagation_enabled", access_propagation_enabled_);
    KVCM_JSON_GET_MACRO(rapid_value, "write_propagation_enabled", write_propagation_enabled_);
    KVCM_JSON_GET_MACRO(rapid_value, "promote_enabled", promote_enabled_);

    int64_t threshold = 0;
    KVCM_JSON_GET_MACRO(rapid_value, "selective_write_threshold", threshold);
    if (threshold <= 0) {
        return false;
    }
    selective_write_threshold_ = static_cast<size_t>(threshold);
    return true;
}

void L2L3StrategyConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "write_mode", ToString(write_mode_));
    Put(writer, "access_propagation_enabled", access_propagation_enabled_);
    Put(writer, "write_propagation_enabled", write_propagation_enabled_);
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

bool HierarchicalModelConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "block_size", block_size_);
    KVCM_JSON_GET_MACRO(rapid_value, "bytes_per_token", bytes_per_token_);
    if (block_size_ <= 0 || bytes_per_token_ <= 0) {
        return false;
    }

    std::string eviction_policy_type_str;
    KVCM_JSON_GET_MACRO(rapid_value, "eviction_policy_type", eviction_policy_type_str);
    eviction_policy_type_ = ToEvictionPolicyType(eviction_policy_type_str);
    if (eviction_policy_type_ == EvictionPolicyType::POLICY_LRU ||
        eviction_policy_type_ == EvictionPolicyType::POLICY_LEAF_AWARE_LRU) {
        LruParams lru_params;
        KVCM_JSON_GET_MACRO(rapid_value, "eviction_policy_params", lru_params);
        eviction_policy_param_ = lru_params;
    } else if (eviction_policy_type_ == EvictionPolicyType::POLICY_RANDOM_LRU) {
        RandomLruParams random_lru_params;
        KVCM_JSON_GET_MACRO(rapid_value, "eviction_policy_params", random_lru_params);
        eviction_policy_param_ = random_lru_params;
    } else if (eviction_policy_type_ == EvictionPolicyType::POLICY_TTL) {
        TtlParams ttl_params;
        KVCM_JSON_GET_MACRO(rapid_value, "eviction_policy_params", ttl_params);
        eviction_policy_param_ = ttl_params;
    } else {
        return false;
    }
    return true;
}

void HierarchicalModelConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "block_size", block_size_);
    Put(writer, "bytes_per_token", bytes_per_token_);
    Put(writer, "eviction_policy_type", ToString(eviction_policy_type_));
    if (eviction_policy_type_ == EvictionPolicyType::POLICY_LRU ||
        eviction_policy_type_ == EvictionPolicyType::POLICY_LEAF_AWARE_LRU) {
        Put(writer, "eviction_policy_params", std::get<LruParams>(eviction_policy_param_));
    } else if (eviction_policy_type_ == EvictionPolicyType::POLICY_RANDOM_LRU) {
        Put(writer, "eviction_policy_params", std::get<RandomLruParams>(eviction_policy_param_));
    } else if (eviction_policy_type_ == EvictionPolicyType::POLICY_TTL) {
        Put(writer, "eviction_policy_params", std::get<TtlParams>(eviction_policy_param_));
    }
}

bool HierarchicalTierConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    if (rapid_value.HasMember("flow_to_next")) {
        return false;
    }
    KVCM_JSON_GET_MACRO(rapid_value, "name", name_);
    KVCM_JSON_GET_MACRO(rapid_value, "capacity", capacity_);
    return !name_.empty() && capacity_ > 0.0;
}

void HierarchicalTierConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "name", name_);
    Put(writer, "capacity", capacity_);
}

bool InferClusterConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    if (rapid_value.HasMember("tier_strategy") || rapid_value.HasMember("flow_to_pool") ||
        rapid_value.HasMember("engine_instance_ids") || rapid_value.HasMember("default_block_ttl_seconds") ||
        rapid_value.HasMember("ttl_refresh_on_read")) {
        return false;
    }
    KVCM_JSON_GET_MACRO(rapid_value, "pool_instance_id", pool_instance_id_);
    KVCM_JSON_GET_MACRO(rapid_value, "model", model_);
    KVCM_JSON_GET_MACRO(rapid_value, "instance_ids", instance_ids_);
    KVCM_JSON_GET_MACRO(rapid_value, "ttl_config", ttl_config_);
    KVCM_JSON_GET_MACRO(rapid_value, "tiers", tiers_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "tier_flows", tier_flows_, std::vector<OptTierFlowConfig>{});
    return !pool_instance_id_.empty() && !instance_ids_.empty() && !tiers_.empty();
}

void InferClusterConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "pool_instance_id", pool_instance_id_);
    Put(writer, "model", model_);
    Put(writer, "instance_ids", instance_ids_);
    Put(writer, "ttl_config", ttl_config_);
    Put(writer, "tiers", tiers_);
    if (!tier_flows_.empty()) {
        Put(writer, "tier_flows", tier_flows_);
    }
}

bool HierarchicalReplayConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    if (rapid_value.HasMember("engine_clusters") || rapid_value.HasMember("engine_scheduling_strategy") ||
        rapid_value.HasMember("infer_instance_groups")) {
        return false;
    }
    KVCM_JSON_GET_MACRO(rapid_value, "trace_file_path", trace_file_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "output_result_path", output_result_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "infer_clusters", infer_clusters_);
    KVCM_JSON_GET_MACRO(rapid_value, "pool_config", pool_config_);
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "infer_scheduling_strategy", infer_scheduling_strategy_, std::string("preserve_trace"));
    if (infer_scheduling_strategy_ != "preserve_trace" && infer_scheduling_strategy_ != "round_robin" &&
        infer_scheduling_strategy_ != "prefix_hit") {
        return false;
    }
    KVCM_JSON_GET_MACRO(rapid_value, "l2_l3_strategy", l2_l3_strategy_);
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "enable_lifecycle_tracking", enable_lifecycle_tracking_, enable_lifecycle_tracking_);
    return !trace_file_path_.empty() && !output_result_path_.empty() && BuildOptimizerConfigs();
}

void HierarchicalReplayConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "trace_file_path", trace_file_path_);
    Put(writer, "output_result_path", output_result_path_);
    Put(writer, "infer_scheduling_strategy", infer_scheduling_strategy_);
    Put(writer, "l2_l3_strategy", l2_l3_strategy_);
    Put(writer, "enable_lifecycle_tracking", enable_lifecycle_tracking_);
    Put(writer, "infer_clusters", infer_clusters_);
    Put(writer, "pool_config", pool_config_);
}

bool HierarchicalReplayConfig::BuildOptimizerConfigs() {
    std::unordered_set<std::string> pool_ids;
    for (const auto &group : pool_config_.instance_groups()) {
        for (const auto &instance : group.instances()) {
            if (!pool_ids.insert(instance.instance_id()).second) {
                return false;
            }
        }
    }
    if (pool_ids.empty()) {
        return false;
    }

    OptimizerConfig engine_config;
    engine_config.set_trace_file_path(trace_file_path_);
    engine_config.set_output_result_path(JoinPath(output_result_path_, "infer"));
    engine_config.set_eviction_params(BuildDefaultEvictionConfig());

    std::vector<OptInstanceGroupConfig> engine_groups;
    engine_to_pool_.clear();
    std::unordered_set<std::string> engine_ids;
    for (const auto &infer_group : infer_clusters_) {
        if (pool_ids.find(infer_group.pool_instance_id()) == pool_ids.end()) {
            return false;
        }

        std::vector<OptTierConfig> tiers;
        tiers.reserve(infer_group.tiers().size());
        double total_capacity = 0.0;
        for (size_t idx = 0; idx < infer_group.tiers().size(); ++idx) {
            const auto &tier = infer_group.tiers()[idx];
            tiers.push_back(BuildOptimizerTier(tier.name(), tier.capacity(), idx));
            total_capacity += tier.capacity();
        }
        OptTierFlowPolicyConfig tier_flow_policy =
            OptTierFlowPolicyConfig::FromTierFlows(tiers, infer_group.tier_flows());
        if (!tier_flow_policy.ValidateFlowConfigs(tiers)) {
            return false;
        }

        for (const auto &engine_instance_id : infer_group.instance_ids()) {
            if (engine_instance_id.empty() || !engine_ids.insert(engine_instance_id).second) {
                return false;
            }

            OptInstanceGroupConfig group;
            group.set_group_name(engine_instance_id);
            group.set_quota_capacity(GbToBytes(total_capacity));
            group.set_used_percentage(1.0);
            group.set_tier_flow_policy(tier_flow_policy);
            group.set_ttl_config(infer_group.ttl_config());
            group.set_storages(tiers);
            group.set_instances({BuildOptimizerInstance(engine_instance_id, engine_instance_id, infer_group.model())});
            engine_groups.push_back(group);

            EngineToPoolMappingConfig mapping;
            mapping.set_engine_instance_id(engine_instance_id);
            mapping.set_pool_instance_id(infer_group.pool_instance_id());
            engine_to_pool_.push_back(mapping);
        }
    }
    if (engine_groups.empty()) {
        return false;
    }
    engine_config.set_instance_groups(engine_groups);
    engine_config_ = engine_config;
    return true;
}

} // namespace kv_cache_manager
