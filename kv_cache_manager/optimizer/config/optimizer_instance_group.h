#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"

namespace kv_cache_manager {

class OptimizerInstanceGroup : public Jsonizable {
public:
    OptimizerInstanceGroup() = default;
    ~OptimizerInstanceGroup() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool ValidateRequiredFields(std::string &invalid_fields) const;

    const std::string &name() const { return name_; }
    const std::vector<double> &capacity_gb() const { return capacity_gb_; }
    const std::string &eviction_policy() const { return eviction_policy_; }
    bool shared_group_quota() const { return shared_group_quota_; }
    bool enable_theoretical_max_cache() const { return enable_theoretical_max_cache_; }
    int64_t ttl_seconds() const { return ttl_seconds_; }

    void set_name(const std::string &v) { name_ = v; }
    void set_capacity_gb(const std::vector<double> &v) {
        capacity_gb_ = v;
        SortCapacities();
    }
    void set_eviction_policy(const std::string &v) { eviction_policy_ = v; }
    void set_shared_group_quota(bool v) { shared_group_quota_ = v; }
    void set_enable_theoretical_max_cache(bool v) { enable_theoretical_max_cache_ = v; }
    void set_ttl_seconds(int64_t v) { ttl_seconds_ = v; }

private:
    void SortCapacities();

    std::string name_;
    std::vector<double> capacity_gb_;
    std::string eviction_policy_ = "lru";
    bool shared_group_quota_ = false;
    bool enable_theoretical_max_cache_ = false;
    int64_t ttl_seconds_ = 0;
};

} // namespace kv_cache_manager
