#pragma once
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/optimizer/config/instance_config.h"
#include "kv_cache_manager/optimizer/config/tier_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
namespace kv_cache_manager {

class OptTierFlowConfig : public Jsonizable {
public:
    OptTierFlowConfig() = default;
    ~OptTierFlowConfig() override = default;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] const std::string &from_tier() const { return from_tier_; }
    [[nodiscard]] const std::string &to_tier() const { return to_tier_; }
    [[nodiscard]] TierFlowStrategy Resolve(const TierFlowStrategy &default_strategy) const;

private:
    std::string from_tier_;
    std::string to_tier_;
    bool has_write_mode_ = false;
    TierWriteMode write_mode_ = TierWriteMode::WRITE_THROUGH;
    bool has_access_propagation_enabled_ = false;
    bool access_propagation_enabled_ = true;
    bool has_promote_enabled_ = false;
    bool promote_enabled_ = true;
    bool has_selective_write_threshold_ = false;
    int64_t selective_write_threshold_ = 2;
};

class OptTierStrategyConfig : public Jsonizable {
public:
    OptTierStrategyConfig() = default;
    ~OptTierStrategyConfig() override = default;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    [[nodiscard]] bool hierarchical_eviction_enabled() const { return hierarchical_eviction_enabled_; }
    [[nodiscard]] TierWriteMode write_mode() const { return write_mode_; }
    [[nodiscard]] bool access_propagation_enabled() const { return access_propagation_enabled_; }
    [[nodiscard]] bool promote_enabled() const { return promote_enabled_; }
    [[nodiscard]] int64_t selective_write_threshold() const { return selective_write_threshold_; }
    [[nodiscard]] std::vector<TierFlowStrategy> BuildFlowStrategies(const std::vector<OptTierConfig> &storages) const;
    [[nodiscard]] bool ValidateFlowConfigs(const std::vector<OptTierConfig> &storages) const;

    void set_hierarchical_eviction_enabled(bool enabled) { hierarchical_eviction_enabled_ = enabled; }
    void set_write_mode(TierWriteMode mode) { write_mode_ = mode; }
    void set_access_propagation_enabled(bool enabled) { access_propagation_enabled_ = enabled; }
    void set_promote_enabled(bool enabled) { promote_enabled_ = enabled; }
    void set_selective_write_threshold(int64_t threshold) { selective_write_threshold_ = threshold; }

private:
    [[nodiscard]] TierFlowStrategy DefaultFlowStrategy() const;
    [[nodiscard]] size_t ResolveFlowEdgeIndex(const OptTierFlowConfig &flow,
                                              const std::vector<OptTierConfig> &storages) const;

    bool hierarchical_eviction_enabled_ = false;
    TierWriteMode write_mode_ = TierWriteMode::WRITE_THROUGH;
    bool access_propagation_enabled_ = true;
    bool promote_enabled_ = true;
    int64_t selective_write_threshold_ = 2;
    std::vector<OptTierFlowConfig> tier_flows_;
};

class OptInstanceGroupConfig : public Jsonizable {
public:
    OptInstanceGroupConfig() = default;
    ~OptInstanceGroupConfig() override = default;
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

public:
    [[nodiscard]] const std::string &group_name() const { return group_name_; }
    [[nodiscard]] int64_t quota_capacity() const { return quota_capacity_; }
    [[nodiscard]] double used_percentage() const { return used_percentage_; }
    [[nodiscard]] const OptTierStrategyConfig &tier_strategy() const { return tier_strategy_; }
    [[nodiscard]] bool hierarchical_eviction_enabled() const { return tier_strategy_.hierarchical_eviction_enabled(); }
    [[nodiscard]] TierWriteMode tier_write_mode() const { return tier_strategy_.write_mode(); }
    [[nodiscard]] bool tier_access_propagation_enabled() const { return tier_strategy_.access_propagation_enabled(); }
    [[nodiscard]] bool enable_promote() const { return tier_strategy_.promote_enabled(); }
    [[nodiscard]] int64_t selective_write_threshold() const { return tier_strategy_.selective_write_threshold(); }
    [[nodiscard]] std::vector<TierFlowStrategy> tier_flow_strategies() const {
        return tier_strategy_.BuildFlowStrategies(storages_);
    }
    [[nodiscard]] int64_t default_block_ttl_seconds() const { return default_block_ttl_seconds_; }
    [[nodiscard]] bool ttl_refresh_on_read() const { return ttl_refresh_on_read_; }
    [[nodiscard]] const std::vector<OptTierConfig> &storages() const { return storages_; }
    [[nodiscard]] const std::vector<OptInstanceConfig> &instances() const { return instances_; }
    [[nodiscard]] std::vector<OptTierConfig> &mutable_storages() { return storages_; }

    void set_group_name(const std::string &name) { group_name_ = name; }
    void set_quota_capacity(int64_t capacity) { quota_capacity_ = capacity; }
    void set_used_percentage(double percentage) { used_percentage_ = percentage; }
    void set_tier_strategy(const OptTierStrategyConfig &tier_strategy) { tier_strategy_ = tier_strategy; }
    void set_hierarchical_eviction_enabled(bool enabled) { tier_strategy_.set_hierarchical_eviction_enabled(enabled); }
    void set_tier_write_mode(TierWriteMode mode) { tier_strategy_.set_write_mode(mode); }
    void set_tier_access_propagation_enabled(bool enabled) { tier_strategy_.set_access_propagation_enabled(enabled); }
    void set_enable_promote(bool enable) { tier_strategy_.set_promote_enabled(enable); }
    void set_selective_write_threshold(int64_t threshold) { tier_strategy_.set_selective_write_threshold(threshold); }
    void set_default_block_ttl_seconds(int64_t ttl) { default_block_ttl_seconds_ = ttl; }
    void set_ttl_refresh_on_read(bool enabled) { ttl_refresh_on_read_ = enabled; }
    void set_storages(const std::vector<OptTierConfig> &storages) { storages_ = storages; }
    void set_instances(const std::vector<OptInstanceConfig> &instances) { instances_ = instances; }

private:
    std::string group_name_;
    int64_t quota_capacity_ = 0;
    double used_percentage_ = 0.0;
    OptTierStrategyConfig tier_strategy_;
    int64_t default_block_ttl_seconds_ = 0; // 0 = TTL 关闭
    bool ttl_refresh_on_read_ = true;

    std::vector<OptTierConfig> storages_;
    std::vector<OptInstanceConfig> instances_;
};

} // namespace kv_cache_manager
