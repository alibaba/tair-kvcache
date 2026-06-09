#include "kv_cache_manager/optimizer/config/hierarchical_replay_config.h"

#include <unordered_set>

#include "kv_cache_manager/optimizer/scheduler/infer_scheduling_strategy.h"

namespace kv_cache_manager {
namespace {

constexpr double kBytesPerGb = static_cast<double>(1LL << 30);

int64_t GbToBytes(double gb) { return static_cast<int64_t>(gb * kBytesPerGb); }

std::string JoinPath(const std::string &base, const std::string &name) {
    if (base.empty() || base.back() == '/') {
        return base + name;
    }
    return base + "/" + name;
}

OptTierConfig BuildOptimizerTier(const std::string &name, double capacity) {
    OptTierConfig tier;
    tier.set_unique_name(name);
    tier.set_storage_type(DataStorageType::DATA_STORAGE_TYPE_DUMMY);
    tier.set_band_width_mbps(0);
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

bool IsEngineReadQueryTypeValid(const std::string &query_type) {
    return query_type == "prefix_match" || query_type == "batch_get";
}

} // namespace

bool StoragePoolFlowConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    if (!rapid_value.IsObject()) {
        return false;
    }
    std::string write_mode_str;
    KVCM_JSON_GET_MACRO(rapid_value, "write_mode", write_mode_str);
    if (!IsValidTierWriteMode(write_mode_str)) {
        return false;
    }
    write_mode_ = ToTierWriteMode(write_mode_str);

    KVCM_JSON_GET_MACRO(rapid_value, "local_read_touch_enabled", local_read_touch_enabled_);
    KVCM_JSON_GET_MACRO(rapid_value, "shadow_write_touch_enabled", shadow_write_touch_enabled_);

    int64_t threshold = 0;
    KVCM_JSON_GET_MACRO(rapid_value, "selective_write_threshold", threshold);
    if (threshold <= 0) {
        return false;
    }
    selective_write_threshold_ = static_cast<size_t>(threshold);
    return true;
}

void StoragePoolFlowConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "write_mode", ToString(write_mode_));
    Put(writer, "local_read_touch_enabled", local_read_touch_enabled_);
    Put(writer, "shadow_write_touch_enabled", shadow_write_touch_enabled_);
    Put(writer, "selective_write_threshold", static_cast<int64_t>(selective_write_threshold_));
}

bool EngineToStoragePoolMappingConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "engine_instance_id", engine_instance_id_);
    KVCM_JSON_GET_MACRO(rapid_value, "storage_pool_id", storage_pool_id_);
    KVCM_JSON_GET_MACRO(rapid_value, "engine_read_query_type", engine_read_query_type_);
    KVCM_JSON_GET_MACRO(rapid_value, "storage_pool_flow", storage_pool_flow_);
    return !engine_instance_id_.empty() && !storage_pool_id_.empty() &&
           IsEngineReadQueryTypeValid(engine_read_query_type_);
}

void EngineToStoragePoolMappingConfig::ToRapidWriter(
    rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "engine_instance_id", engine_instance_id_);
    Put(writer, "storage_pool_id", storage_pool_id_);
    Put(writer, "engine_read_query_type", engine_read_query_type_);
    Put(writer, "storage_pool_flow", storage_pool_flow_);
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
    KVCM_JSON_GET_MACRO(rapid_value, "name", name_);
    KVCM_JSON_GET_MACRO(rapid_value, "capacity", capacity_);
    return !name_.empty() && capacity_ > 0.0;
}

void HierarchicalTierConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "name", name_);
    Put(writer, "capacity", capacity_);
}

bool HierarchicalStoragePoolEntryConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "pool_id", pool_id_);
    KVCM_JSON_GET_MACRO(rapid_value, "model", model_);
    return !pool_id_.empty();
}

void HierarchicalStoragePoolEntryConfig::ToRapidWriter(
    rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "pool_id", pool_id_);
    Put(writer, "model", model_);
}

bool HierarchicalStoragePoolConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "output_result_path", output_result_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "storage_name", storage_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "capacity", capacity_);
    KVCM_JSON_GET_MACRO(rapid_value, "eviction_params", eviction_config_);
    KVCM_JSON_GET_MACRO(rapid_value, "ttl_config", ttl_config_);
    KVCM_JSON_GET_MACRO(rapid_value, "pools", pools_);
    return !output_result_path_.empty() && !storage_name_.empty() && capacity_ > 0.0 && !pools_.empty();
}

void HierarchicalStoragePoolConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "output_result_path", output_result_path_);
    Put(writer, "storage_name", storage_name_);
    Put(writer, "capacity", capacity_);
    Put(writer, "eviction_params", eviction_config_);
    Put(writer, "ttl_config", ttl_config_);
    Put(writer, "pools", pools_);
}

bool P2PReadFlowConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "tier", tier_);
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "peer_read_touch_enabled", peer_read_touch_enabled_, peer_read_touch_enabled_);
    return !tier_.empty();
}

void P2PReadFlowConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "tier", tier_);
    Put(writer, "peer_read_touch_enabled", peer_read_touch_enabled_);
}

bool InferActiveWindowConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "infer_id", infer_id_);
    KVCM_JSON_GET_MACRO(rapid_value, "start_ns", start_ns_);
    KVCM_JSON_GET_MACRO(rapid_value, "end_ns", end_ns_);
    return !infer_id_.empty() && start_ns_ <= end_ns_;
}

void InferActiveWindowConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "infer_id", infer_id_);
    Put(writer, "start_ns", start_ns_);
    Put(writer, "end_ns", end_ns_);
}

bool InferClusterConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "storage_pool_id", storage_pool_id_);
    KVCM_JSON_GET_MACRO(rapid_value, "engine_read_query_type", engine_read_query_type_);
    KVCM_JSON_GET_MACRO(rapid_value, "model", model_);
    KVCM_JSON_GET_MACRO(rapid_value, "infer_ids", infer_ids_);
    KVCM_JSON_GET_MACRO(rapid_value, "ttl_config", ttl_config_);
    KVCM_JSON_GET_MACRO(rapid_value, "tiers", tiers_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "tier_flows", tier_flows_, std::vector<OptTierFlowConfig>{});
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "p2p_read_flows", p2p_read_flows_, std::vector<P2PReadFlowConfig>{});
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "active_windows", active_windows_, std::vector<InferActiveWindowConfig>{});
    KVCM_JSON_GET_MACRO(rapid_value, "storage_pool_flow", storage_pool_flow_);
    return !storage_pool_id_.empty() && IsEngineReadQueryTypeValid(engine_read_query_type_) && !infer_ids_.empty() &&
           !tiers_.empty();
}

void InferClusterConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "storage_pool_id", storage_pool_id_);
    Put(writer, "engine_read_query_type", engine_read_query_type_);
    Put(writer, "model", model_);
    Put(writer, "infer_ids", infer_ids_);
    Put(writer, "ttl_config", ttl_config_);
    Put(writer, "tiers", tiers_);
    if (!tier_flows_.empty()) {
        Put(writer, "tier_flows", tier_flows_);
    }
    if (!p2p_read_flows_.empty()) {
        Put(writer, "p2p_read_flows", p2p_read_flows_);
    }
    if (!active_windows_.empty()) {
        Put(writer, "active_windows", active_windows_);
    }
    Put(writer, "storage_pool_flow", storage_pool_flow_);
}

bool HierarchicalReplayConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "trace_file_path", trace_file_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "output_result_path", output_result_path_);
    KVCM_JSON_GET_MACRO(rapid_value, "infer_eviction_params", infer_eviction_config_);
    trace_replay_config_ = OptTraceReplayConfig();
    if (rapid_value.HasMember("trace_replay")) {
        if (!rapid_value["trace_replay"].IsObject() ||
            !trace_replay_config_.FromRapidValue(rapid_value["trace_replay"])) {
            return false;
        }
    }
    KVCM_JSON_GET_MACRO(rapid_value, "infer_clusters", infer_clusters_);
    KVCM_JSON_GET_MACRO(rapid_value, "storage_pool", storage_pool_);
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "infer_scheduling_strategy", infer_scheduling_strategy_, std::string("preserve_trace"));
    if (!IsSupportedInferSchedulingStrategy(infer_scheduling_strategy_)) {
        return false;
    }
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "infer_active_windows_from_trace", infer_active_windows_from_trace_, false);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "cache_drop_event_file", cache_drop_event_file_, std::string(""));
    if (infer_active_windows_from_trace_) {
        for (const auto &cluster : infer_clusters_) {
            if (!cluster.active_windows().empty()) {
                return false;
            }
        }
    }
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "enable_lifecycle_tracking", enable_lifecycle_tracking_, enable_lifecycle_tracking_);
    return !trace_file_path_.empty() && !output_result_path_.empty() && BuildOptimizerConfigs();
}

void HierarchicalReplayConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "trace_file_path", trace_file_path_);
    Put(writer, "output_result_path", output_result_path_);
    Put(writer, "infer_eviction_params", infer_eviction_config_);
    Put(writer, "trace_replay", trace_replay_config_);
    Put(writer, "infer_scheduling_strategy", infer_scheduling_strategy_);
    Put(writer, "infer_active_windows_from_trace", infer_active_windows_from_trace_);
    Put(writer, "enable_lifecycle_tracking", enable_lifecycle_tracking_);
    if (!cache_drop_event_file_.empty()) {
        Put(writer, "cache_drop_event_file", cache_drop_event_file_);
    }
    Put(writer, "infer_clusters", infer_clusters_);
    Put(writer, "storage_pool", storage_pool_);
}

bool HierarchicalReplayConfig::BuildOptimizerConfigs() {
    std::unordered_set<std::string> pool_ids;
    for (const auto &pool : storage_pool_.pools()) {
        if (!pool_ids.insert(pool.pool_id()).second) {
            return false;
        }
    }
    if (pool_ids.empty()) {
        return false;
    }

    OptimizerConfig engine_config;
    engine_config.set_trace_file_path(trace_file_path_);
    engine_config.set_output_result_path(JoinPath(output_result_path_, "infer"));
    engine_config.set_eviction_params(infer_eviction_config_);
    engine_config.set_trace_replay_config(trace_replay_config_);

    std::vector<OptInstanceGroupConfig> engine_groups;
    engine_to_storage_pool_.clear();
    std::unordered_set<std::string> engine_ids;
    for (const auto &infer_group : infer_clusters_) {
        if (pool_ids.find(infer_group.storage_pool_id()) == pool_ids.end()) {
            return false;
        }
        if (infer_active_windows_from_trace_ && !infer_group.active_windows().empty()) {
            return false;
        }
        std::unordered_set<std::string> tier_names;
        for (const auto &tier : infer_group.tiers()) {
            if (!tier_names.insert(tier.name()).second) {
                return false;
            }
        }
        std::unordered_set<std::string> p2p_tiers;
        for (const auto &flow : infer_group.p2p_read_flows()) {
            if (tier_names.find(flow.tier()) == tier_names.end() || !p2p_tiers.insert(flow.tier()).second) {
                return false;
            }
        }
        std::unordered_set<std::string> cluster_infer_ids(infer_group.infer_ids().begin(),
                                                          infer_group.infer_ids().end());
        for (const auto &window : infer_group.active_windows()) {
            if (cluster_infer_ids.find(window.infer_id()) == cluster_infer_ids.end()) {
                return false;
            }
        }

        std::vector<OptTierConfig> tiers;
        tiers.reserve(infer_group.tiers().size());
        double total_capacity = 0.0;
        for (size_t idx = 0; idx < infer_group.tiers().size(); ++idx) {
            const auto &tier = infer_group.tiers()[idx];
            tiers.push_back(BuildOptimizerTier(tier.name(), tier.capacity()));
            total_capacity += tier.capacity();
        }
        OptTierFlowPolicyConfig tier_flow_policy =
            OptTierFlowPolicyConfig::FromTierFlows(tiers, infer_group.tier_flows());
        if (!tier_flow_policy.ValidateFlowConfigs(tiers)) {
            return false;
        }

        for (const auto &infer_id : infer_group.infer_ids()) {
            if (infer_id.empty() || !engine_ids.insert(infer_id).second) {
                return false;
            }

            OptInstanceGroupConfig group;
            group.set_group_name(infer_id);
            group.set_quota_capacity(GbToBytes(total_capacity));
            group.set_used_percentage(1.0);
            group.set_tier_flow_policy(tier_flow_policy);
            group.set_ttl_config(infer_group.ttl_config());
            group.set_storages(tiers);
            group.set_instances({BuildOptimizerInstance(infer_id, infer_id, infer_group.model())});
            engine_groups.push_back(group);

            EngineToStoragePoolMappingConfig mapping;
            mapping.set_engine_instance_id(infer_id);
            mapping.set_storage_pool_id(infer_group.storage_pool_id());
            mapping.set_engine_read_query_type(infer_group.engine_read_query_type());
            mapping.set_storage_pool_flow(infer_group.storage_pool_flow());
            engine_to_storage_pool_.push_back(mapping);
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
