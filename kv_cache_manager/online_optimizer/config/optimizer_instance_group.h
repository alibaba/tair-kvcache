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
    bool enabled() const { return enabled_; }
    const std::vector<double> &capacity_gb() const { return capacity_gb_; }
    const std::string &indexer_type() const { return indexer_type_; }
    int64_t max_key_count() const { return max_key_count_; }
    int64_t ttl_seconds() const { return ttl_seconds_; }

    void set_name(const std::string &v) { name_ = v; }
    void set_enabled(bool v) { enabled_ = v; }
    void set_capacity_gb(const std::vector<double> &v) {
        capacity_gb_ = v;
        SortCapacities();
    }
    void set_indexer_type(const std::string &v) { indexer_type_ = v; }
    void set_max_key_count(int64_t v) { max_key_count_ = v; }
    void set_ttl_seconds(int64_t v) { ttl_seconds_ = v; }

private:
    void SortCapacities();

    std::string name_;
    bool enabled_ = false;
    std::vector<double> capacity_gb_;
    std::string indexer_type_ = "lru";
    int64_t max_key_count_ = 0;
    int64_t ttl_seconds_ = 0;
};

} // namespace kv_cache_manager
