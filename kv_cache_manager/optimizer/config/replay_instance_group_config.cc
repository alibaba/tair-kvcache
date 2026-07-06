#include "kv_cache_manager/optimizer/config/replay_instance_group_config.h"

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

namespace {
std::vector<OptTierConfig> SortStoragesByPriority(const std::vector<OptTierConfig> &storages) {
    std::vector<OptTierConfig> sorted_storages = storages;
    std::sort(sorted_storages.begin(), sorted_storages.end(), [](const OptTierConfig &a, const OptTierConfig &b) {
        return a.priority() < b.priority();
    });
    return sorted_storages;
}

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
} // namespace

bool OptTierFlowConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    if (!rapid_value.IsObject()) {
        KVCM_LOG_ERROR("tier_strategy.tier_flows item must be an object");
        return false;
    }
    KVCM_JSON_GET_MACRO(rapid_value, "from_tier", from_tier_);
    KVCM_JSON_GET_MACRO(rapid_value, "to_tier", to_tier_);
    if (from_tier_.empty() || to_tier_.empty()) {
        KVCM_LOG_ERROR("tier_strategy.tier_flows item has empty from_tier or to_tier");
        return false;
    }
    if (rapid_value.HasMember("write_mode")) {
        std::string write_mode_str;
        KVCM_JSON_GET_MACRO(rapid_value, "write_mode", write_mode_str);
        if (!IsValidTierWriteMode(write_mode_str)) {
            KVCM_LOG_ERROR("tier_strategy.tier_flows edge %s->%s has invalid write_mode: %s",
                           from_tier_.c_str(),
                           to_tier_.c_str(),
                           write_mode_str.c_str());
            return false;
        }
        write_mode_ = ToTierWriteMode(write_mode_str);
        has_write_mode_ = true;
    }
    if (rapid_value.HasMember("access_propagation_enabled")) {
        KVCM_JSON_GET_MACRO(rapid_value, "access_propagation_enabled", access_propagation_enabled_);
        has_access_propagation_enabled_ = true;
    }
    if (rapid_value.HasMember("promote_enabled")) {
        KVCM_JSON_GET_MACRO(rapid_value, "promote_enabled", promote_enabled_);
        has_promote_enabled_ = true;
    }
    if (rapid_value.HasMember("selective_write_threshold")) {
        KVCM_JSON_GET_MACRO(rapid_value, "selective_write_threshold", selective_write_threshold_);
        if (selective_write_threshold_ <= 0) {
            KVCM_LOG_ERROR("tier_strategy.tier_flows edge %s->%s has invalid selective_write_threshold: %lld",
                           from_tier_.c_str(),
                           to_tier_.c_str(),
                           static_cast<long long>(selective_write_threshold_));
            return false;
        }
        has_selective_write_threshold_ = true;
    }
    return true;
}

void OptTierFlowConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "from_tier", from_tier_);
    Put(writer, "to_tier", to_tier_);
    if (has_write_mode_) {
        Put(writer, "write_mode", ToString(write_mode_));
    }
    if (has_access_propagation_enabled_) {
        Put(writer, "access_propagation_enabled", access_propagation_enabled_);
    }
    if (has_promote_enabled_) {
        Put(writer, "promote_enabled", promote_enabled_);
    }
    if (has_selective_write_threshold_) {
        Put(writer, "selective_write_threshold", selective_write_threshold_);
    }
}

TierFlowStrategy OptTierFlowConfig::Resolve(const TierFlowStrategy &default_strategy) const {
    TierFlowStrategy strategy = default_strategy;
    if (has_write_mode_) {
        strategy.write_mode = write_mode_;
    }
    if (has_access_propagation_enabled_) {
        strategy.access_propagation_enabled = access_propagation_enabled_;
    }
    if (has_promote_enabled_) {
        strategy.promote_enabled = promote_enabled_;
    }
    if (has_selective_write_threshold_) {
        strategy.selective_write_threshold = static_cast<size_t>(selective_write_threshold_);
    }
    return strategy;
}

bool OptTierStrategyConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    if (!rapid_value.IsObject()) {
        KVCM_LOG_ERROR("tier_strategy must be an object");
        return false;
    }
    KVCM_JSON_GET_MACRO(rapid_value, "hierarchical_eviction_enabled", hierarchical_eviction_enabled_);
    std::string write_mode_str;
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "write_mode", write_mode_str, std::string("write_through"));
    if (!IsValidTierWriteMode(write_mode_str)) {
        KVCM_LOG_ERROR("tier_strategy.write_mode is invalid: %s", write_mode_str.c_str());
        return false;
    }
    write_mode_ = ToTierWriteMode(write_mode_str);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "access_propagation_enabled", access_propagation_enabled_, true);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "promote_enabled", promote_enabled_, true);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "selective_write_threshold", selective_write_threshold_, int64_t(2));
    if (selective_write_threshold_ <= 0) {
        KVCM_LOG_ERROR("tier_strategy.selective_write_threshold must be > 0, got %lld",
                       static_cast<long long>(selective_write_threshold_));
        return false;
    }
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "tier_flows", tier_flows_, std::vector<OptTierFlowConfig>{});
    return true;
}

void OptTierStrategyConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "hierarchical_eviction_enabled", hierarchical_eviction_enabled_);
    Put(writer, "write_mode", ToString(write_mode_));
    Put(writer, "access_propagation_enabled", access_propagation_enabled_);
    Put(writer, "promote_enabled", promote_enabled_);
    Put(writer, "selective_write_threshold", selective_write_threshold_);
    if (!tier_flows_.empty()) {
        Put(writer, "tier_flows", tier_flows_);
    }
}

TierFlowStrategy OptTierStrategyConfig::DefaultFlowStrategy() const {
    TierFlowStrategy strategy;
    strategy.write_mode = write_mode_;
    strategy.access_propagation_enabled = access_propagation_enabled_;
    strategy.promote_enabled = promote_enabled_;
    strategy.selective_write_threshold = static_cast<size_t>(selective_write_threshold_);
    return strategy;
}

size_t OptTierStrategyConfig::ResolveFlowEdgeIndex(const OptTierFlowConfig &flow,
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

bool OptTierStrategyConfig::ValidateFlowConfigs(const std::vector<OptTierConfig> &storages) const {
    if (tier_flows_.empty()) {
        return true;
    }
    const std::vector<OptTierConfig> sorted_storages = SortStoragesByPriority(storages);
    if (sorted_storages.size() < 2) {
        KVCM_LOG_ERROR("tier_strategy.tier_flows is configured but instance group has %zu storage tier(s); "
                       "at least 2 tiers are required",
                       sorted_storages.size());
        return false;
    }

    std::unordered_map<std::string, size_t> tier_index;
    std::unordered_set<size_t> priorities;
    for (size_t i = 0; i < sorted_storages.size(); ++i) {
        const auto &storage = sorted_storages[i];
        const auto name_insert_result = tier_index.emplace(storage.unique_name(), i);
        if (!name_insert_result.second) {
            KVCM_LOG_ERROR("storages contains duplicate unique_name '%s'; tier_strategy.tier_flows cannot be matched "
                           "unambiguously",
                           storage.unique_name().c_str());
            return false;
        }
        const auto priority_insert_result = priorities.emplace(storage.priority());
        if (!priority_insert_result.second) {
            KVCM_LOG_ERROR("storages contains duplicate priority %zu; tier_strategy.tier_flows edge order is "
                           "ambiguous",
                           storage.priority());
            return false;
        }
    }

    std::vector<bool> seen(sorted_storages.size() - 1, false);
    for (const auto &flow : tier_flows_) {
        const auto from_it = tier_index.find(flow.from_tier());
        const auto to_it = tier_index.find(flow.to_tier());
        if (from_it == tier_index.end() || to_it == tier_index.end()) {
            KVCM_LOG_ERROR("tier_strategy.tier_flows edge %s->%s references unknown tier; configured tiers by "
                           "priority: [%s]",
                           flow.from_tier().c_str(),
                           flow.to_tier().c_str(),
                           JoinTierNames(sorted_storages).c_str());
            return false;
        }

        const size_t edge_idx = from_it->second;
        if (edge_idx + 1 != to_it->second) {
            KVCM_LOG_ERROR("tier_strategy.tier_flows edge %s->%s is not an adjacent priority edge; expected one of "
                           "[%s]",
                           flow.from_tier().c_str(),
                           flow.to_tier().c_str(),
                           JoinExpectedEdges(sorted_storages).c_str());
            return false;
        }

        if (edge_idx >= seen.size()) {
            KVCM_LOG_ERROR("tier_strategy.tier_flows edge %s->%s starts from the last storage tier; expected one of "
                           "[%s]",
                           flow.from_tier().c_str(),
                           flow.to_tier().c_str(),
                           JoinExpectedEdges(sorted_storages).c_str());
            return false;
        }

        if (seen[edge_idx]) {
            KVCM_LOG_ERROR("tier_strategy.tier_flows contains duplicate edge %s->%s",
                           flow.from_tier().c_str(),
                           flow.to_tier().c_str());
            return false;
        }
        seen[edge_idx] = true;
    }
    return true;
}

std::vector<TierFlowStrategy>
OptTierStrategyConfig::BuildFlowStrategies(const std::vector<OptTierConfig> &storages) const {
    if (storages.size() < 2) {
        return {};
    }
    const std::vector<OptTierConfig> sorted_storages = SortStoragesByPriority(storages);
    std::vector<TierFlowStrategy> strategies(sorted_storages.size() - 1, DefaultFlowStrategy());
    for (const auto &flow : tier_flows_) {
        const size_t edge_idx = ResolveFlowEdgeIndex(flow, sorted_storages);
        if (edge_idx < strategies.size()) {
            strategies[edge_idx] = flow.Resolve(strategies[edge_idx]);
        }
    }
    return strategies;
}

bool OptimizerReplayInstanceGroupConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_MACRO(rapid_value, "group_name", group_name_);
    KVCM_JSON_GET_MACRO(rapid_value, "used_percentage", used_percentage_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "default_block_ttl_seconds", default_block_ttl_seconds_, int64_t(0));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "ttl_refresh_on_read", ttl_refresh_on_read_, true);
    KVCM_JSON_GET_MACRO(rapid_value, "instances", instances_);
    // quota_capacity in config is in GB; convert to bytes
    double quota_capacity_gb = 0.0;
    KVCM_JSON_GET_MACRO(rapid_value, "quota_capacity", quota_capacity_gb);
    quota_capacity_ =
        quota_capacity_gb < 0 ? -1 : static_cast<int64_t>(quota_capacity_gb * static_cast<double>(1LL << 30));
    // Parse storages; tier capacity is in GB in config, OptTierConfig::FromRapidValue handles conversion
    KVCM_JSON_GET_MACRO(rapid_value, "storages", storages_);
    if (!rapid_value.HasMember("tier_strategy")) {
        KVCM_LOG_ERROR("instance_group '%s' is missing required tier_strategy", group_name_.c_str());
        return false;
    }
    if (!tier_strategy_.FromRapidValue(rapid_value["tier_strategy"])) {
        KVCM_LOG_ERROR("instance_group '%s' has invalid tier_strategy", group_name_.c_str());
        return false;
    }
    if (!tier_strategy_.ValidateFlowConfigs(storages_)) {
        KVCM_LOG_ERROR("instance_group '%s' has invalid tier_strategy.tier_flows", group_name_.c_str());
        return false;
    }
    return true;
};

void OptimizerReplayInstanceGroupConfig::ToRapidWriter(
    rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "group_name", group_name_);
    // Write quota_capacity in GB
    const double quota_gb =
        quota_capacity_ < 0 ? -1.0 : static_cast<double>(quota_capacity_) / static_cast<double>(1LL << 30);
    Put(writer, "quota_capacity", quota_gb);
    Put(writer, "used_percentage", used_percentage_);
    Put(writer, "tier_strategy", tier_strategy_);
    Put(writer, "default_block_ttl_seconds", default_block_ttl_seconds_);
    Put(writer, "ttl_refresh_on_read", ttl_refresh_on_read_);
    Put(writer, "storages", storages_);
    Put(writer, "instances", instances_);
};
} // namespace kv_cache_manager
