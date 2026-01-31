#pragma once
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/optimizer/config/instance_config.h"
#include "kv_cache_manager/optimizer/config/tier_config.h"
namespace kv_cache_manager {

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
    [[nodiscard]] bool hierarchical_eviction_enabled() const { return hierarchical_eviction_enabled_; }
    [[nodiscard]] const std::vector<OptTierConfig> &storages() const { return storages_; }
    [[nodiscard]] const std::vector<OptInstanceConfig> &instances() const { return instances_; }
    [[nodiscard]] std::vector<OptTierConfig> &mutable_storages() { return storages_; }

    void set_group_name(const std::string &name) { group_name_ = name; }
    void set_quota_capacity(int64_t capacity) { quota_capacity_ = capacity; }
    void set_used_percentage(double percentage) { used_percentage_ = percentage; }
    void set_hierarchical_eviction_enabled(bool enabled) { hierarchical_eviction_enabled_ = enabled; }
    void set_storages(const std::vector<OptTierConfig> &storages) { storages_ = storages; }
    void set_instances(const std::vector<OptInstanceConfig> &instances) { instances_ = instances; }

private:
    std::string group_name_;
    int64_t quota_capacity_ = 0;
    double used_percentage_ = 0.0;
    bool hierarchical_eviction_enabled_ = false;

    std::vector<OptTierConfig> storages_;
    std::vector<OptInstanceConfig> instances_;
};

} // namespace kv_cache_manager
