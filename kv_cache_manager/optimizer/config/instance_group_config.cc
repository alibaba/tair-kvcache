#include "kv_cache_manager/optimizer/config/instance_group_config.h"

#include <sstream>
#include <unordered_map>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

namespace {
std::string JoinTierNames(const std::vector<OptTierConfig> &storages) {
    std::ostringstream oss;
    for (size_t i = 0; i < storages.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << storages[i].unique_name();
    }
    return oss.str();
}

std::string JoinExpectedEdges(const std::vector<OptTierConfig> &storages) {
    std::ostringstream oss;
    for (size_t i = 0; i + 1 < storages.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << storages[i].unique_name() << "->" << storages[i + 1].unique_name();
    }
    return oss.str();
}

bool ParseStoragesByArrayOrder(const rapidjson::Value &rapid_value, std::vector<OptTierConfig> &storages) {
    if (!rapid_value.HasMember("storages") || !rapid_value["storages"].IsArray()) {
        KVCM_LOG_ERROR("instance_group storages must be an array");
        return false;
    }

    const auto &storage_array = rapid_value["storages"].GetArray();
    storages.clear();
    storages.reserve(storage_array.Size());
    for (rapidjson::SizeType idx = 0; idx < storage_array.Size(); ++idx) {
        OptTierConfig tier;
        if (!tier.FromRapidValue(storage_array[idx])) {
            KVCM_LOG_ERROR("failed to parse storages[%u]", idx);
            return false;
        }
        storages.push_back(tier);
    }
    return true;
}

} // namespace

bool OptTierFlowConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    if (!rapid_value.IsObject()) {
        KVCM_LOG_ERROR("tier_flows item must be an object");
        return false;
    }
    KVCM_JSON_GET_MACRO(rapid_value, "from_tier", from_tier_);
    KVCM_JSON_GET_MACRO(rapid_value, "to_tier", to_tier_);
    if (from_tier_.empty() || to_tier_.empty()) {
        KVCM_LOG_ERROR("tier_flows item has empty from_tier or to_tier");
        return false;
    }
    std::string write_mode_str;
    KVCM_JSON_GET_MACRO(rapid_value, "write_mode", write_mode_str);
    if (!IsValidTierWriteMode(write_mode_str)) {
        KVCM_LOG_ERROR("tier_flows edge %s->%s has invalid write_mode: %s",
                       from_tier_.c_str(),
                       to_tier_.c_str(),
                       write_mode_str.c_str());
        return false;
    }
    write_mode_ = ToTierWriteMode(write_mode_str);
    KVCM_JSON_GET_MACRO(rapid_value, "access_propagation_enabled", access_propagation_enabled_);
    KVCM_JSON_GET_MACRO(rapid_value, "write_propagation_enabled", write_propagation_enabled_);
    KVCM_JSON_GET_MACRO(rapid_value, "promote_enabled", promote_enabled_);
    KVCM_JSON_GET_MACRO(rapid_value, "selective_write_threshold", selective_write_threshold_);
    if (selective_write_threshold_ <= 0) {
        KVCM_LOG_ERROR("tier_flows edge %s->%s has invalid selective_write_threshold: %lld",
                       from_tier_.c_str(),
                       to_tier_.c_str(),
                       static_cast<long long>(selective_write_threshold_));
        return false;
    }
    return true;
}

void OptTierFlowConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "from_tier", from_tier_);
    Put(writer, "to_tier", to_tier_);
    Put(writer, "write_mode", ToString(write_mode_));
    Put(writer, "access_propagation_enabled", access_propagation_enabled_);
    Put(writer, "write_propagation_enabled", write_propagation_enabled_);
    Put(writer, "promote_enabled", promote_enabled_);
    Put(writer, "selective_write_threshold", selective_write_threshold_);
}

TierFlowStrategy OptTierFlowConfig::Resolve(const TierFlowStrategy &default_strategy) const {
    TierFlowStrategy strategy = default_strategy;
    strategy.write_mode = write_mode_;
    strategy.access_propagation_enabled = access_propagation_enabled_;
    strategy.write_propagation_enabled = write_propagation_enabled_;
    strategy.promote_enabled = promote_enabled_;
    strategy.selective_write_threshold = static_cast<size_t>(selective_write_threshold_);
    return strategy;
}

OptTierFlowPolicyConfig OptTierFlowPolicyConfig::FromTierFlows(const std::vector<OptTierConfig> &storages,
                                                               const std::vector<OptTierFlowConfig> &flows) {
    OptTierFlowPolicyConfig policy;
    policy.set_hierarchical_eviction_enabled(storages.size() > 1);
    policy.set_tier_flows(flows);
    if (!flows.empty()) {
        policy.set_write_mode(flows.front().write_mode());
        policy.set_access_propagation_enabled(flows.front().access_propagation_enabled());
        policy.set_write_propagation_enabled(flows.front().write_propagation_enabled());
        policy.set_promote_enabled(flows.front().promote_enabled());
        policy.set_selective_write_threshold(flows.front().selective_write_threshold());
    }
    return policy;
}

TierFlowStrategy OptTierFlowPolicyConfig::DefaultFlowStrategy() const {
    TierFlowStrategy strategy;
    strategy.write_mode = write_mode_;
    strategy.access_propagation_enabled = access_propagation_enabled_;
    strategy.write_propagation_enabled = write_propagation_enabled_;
    strategy.promote_enabled = promote_enabled_;
    strategy.selective_write_threshold = static_cast<size_t>(selective_write_threshold_);
    return strategy;
}

size_t OptTierFlowPolicyConfig::ResolveFlowEdgeIndex(const OptTierFlowConfig &flow,
                                                     const std::vector<OptTierConfig> &storages) const {
    if (storages.size() < 2) {
        return storages.size();
    }
    for (size_t i = 0; i + 1 < storages.size(); ++i) {
        if (storages[i].unique_name() == flow.from_tier() && storages[i + 1].unique_name() == flow.to_tier()) {
            return i;
        }
    }
    return storages.size();
}

bool OptTierFlowPolicyConfig::ValidateFlowConfigs(const std::vector<OptTierConfig> &storages) const {
    if (storages.size() < 2) {
        if (!tier_flows_.empty()) {
            KVCM_LOG_ERROR("tier_flows is configured but instance group has %zu storage tier(s)", storages.size());
            return false;
        }
        return true;
    }
    if (tier_flows_.size() != storages.size() - 1) {
        KVCM_LOG_ERROR(
            "tier_flows must define exactly %zu adjacent edge(s), got %zu", storages.size() - 1, tier_flows_.size());
        return false;
    }

    std::unordered_map<std::string, size_t> tier_index;
    for (size_t i = 0; i < storages.size(); ++i) {
        const auto &storage = storages[i];
        const auto name_insert_result = tier_index.emplace(storage.unique_name(), i);
        if (!name_insert_result.second) {
            KVCM_LOG_ERROR("storages contains duplicate unique_name '%s'; tier_flows cannot be matched unambiguously",
                           storage.unique_name().c_str());
            return false;
        }
    }

    std::vector<bool> seen(storages.size() - 1, false);
    for (const auto &flow : tier_flows_) {
        const auto from_it = tier_index.find(flow.from_tier());
        const auto to_it = tier_index.find(flow.to_tier());
        if (from_it == tier_index.end() || to_it == tier_index.end()) {
            KVCM_LOG_ERROR("tier_flows edge %s->%s references unknown tier; configured tiers by order: [%s]",
                           flow.from_tier().c_str(),
                           flow.to_tier().c_str(),
                           JoinTierNames(storages).c_str());
            return false;
        }

        const size_t edge_idx = from_it->second;
        if (edge_idx + 1 != to_it->second) {
            KVCM_LOG_ERROR("tier_flows edge %s->%s is not an adjacent tier edge; expected one of [%s]",
                           flow.from_tier().c_str(),
                           flow.to_tier().c_str(),
                           JoinExpectedEdges(storages).c_str());
            return false;
        }

        if (edge_idx >= seen.size()) {
            KVCM_LOG_ERROR("tier_flows edge %s->%s starts from the last storage tier; expected one of "
                           "[%s]",
                           flow.from_tier().c_str(),
                           flow.to_tier().c_str(),
                           JoinExpectedEdges(storages).c_str());
            return false;
        }

        if (seen[edge_idx]) {
            KVCM_LOG_ERROR(
                "tier_flows contains duplicate edge %s->%s", flow.from_tier().c_str(), flow.to_tier().c_str());
            return false;
        }
        seen[edge_idx] = true;
    }
    for (size_t idx = 0; idx < seen.size(); ++idx) {
        if (!seen[idx]) {
            KVCM_LOG_ERROR("tier_flows missing adjacent edge %s->%s",
                           storages[idx].unique_name().c_str(),
                           storages[idx + 1].unique_name().c_str());
            return false;
        }
    }
    return true;
}

std::vector<TierFlowStrategy>
OptTierFlowPolicyConfig::BuildFlowStrategies(const std::vector<OptTierConfig> &storages) const {
    if (storages.size() < 2) {
        return {};
    }
    std::vector<TierFlowStrategy> strategies(storages.size() - 1, DefaultFlowStrategy());
    for (const auto &flow : tier_flows_) {
        const size_t edge_idx = ResolveFlowEdgeIndex(flow, storages);
        if (edge_idx < strategies.size()) {
            strategies[edge_idx] = flow.Resolve(strategies[edge_idx]);
        }
    }
    return strategies;
}

bool OptTtlConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    if (!rapid_value.IsObject()) {
        KVCM_LOG_ERROR("ttl_config must be an object");
        return false;
    }
    KVCM_JSON_GET_MACRO(rapid_value, "default_block_ttl_seconds", default_block_ttl_seconds_);
    KVCM_JSON_GET_MACRO(rapid_value, "refresh_on_read", refresh_on_read_);
    if (default_block_ttl_seconds_ < 0) {
        KVCM_LOG_ERROR("ttl_config.default_block_ttl_seconds must be >= 0, got %lld",
                       static_cast<long long>(default_block_ttl_seconds_));
        return false;
    }
    return true;
}

void OptTtlConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "default_block_ttl_seconds", default_block_ttl_seconds_);
    Put(writer, "refresh_on_read", refresh_on_read_);
}

bool OptInstanceGroupConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "group_name", group_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "used_percentage", used_percentage_);
    KVCM_JSON_GET_MACRO(rapid_value, "ttl_config", ttl_config_);
    KVCM_JSON_GET_MACRO(rapid_value, "instances", instances_);
    // quota_capacity in config is in GB; convert to bytes
    double quota_capacity_gb = 0.0;
    KVCM_JSON_GET_MACRO(rapid_value, "quota_capacity", quota_capacity_gb);
    quota_capacity_ =
        quota_capacity_gb < 0 ? -1 : static_cast<int64_t>(quota_capacity_gb * static_cast<double>(1LL << 30));
    if (!ParseStoragesByArrayOrder(rapid_value, storages_)) {
        return false;
    }
    std::vector<OptTierFlowConfig> tier_flows;
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "tier_flows", tier_flows, std::vector<OptTierFlowConfig>{});
    tier_flow_policy_ = OptTierFlowPolicyConfig::FromTierFlows(storages_, tier_flows);
    if (!tier_flow_policy_.ValidateFlowConfigs(storages_)) {
        KVCM_LOG_ERROR("instance_group '%s' has invalid tier_flows", group_name_.c_str());
        return false;
    }
    return true;
};

void OptInstanceGroupConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "group_name", group_name_);
    // Write quota_capacity in GB
    const double quota_gb =
        quota_capacity_ < 0 ? -1.0 : static_cast<double>(quota_capacity_) / static_cast<double>(1LL << 30);
    Put(writer, "quota_capacity", quota_gb);
    Put(writer, "used_percentage", used_percentage_);
    if (!tier_flow_policy_.tier_flows().empty()) {
        Put(writer, "tier_flows", tier_flow_policy_.tier_flows());
    }
    Put(writer, "ttl_config", ttl_config_);
    Put(writer, "storages", storages_);
    Put(writer, "instances", instances_);
};
} // namespace kv_cache_manager
