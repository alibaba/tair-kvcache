#pragma once

#include <cstdint>
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
    int32_t primary_capacity_index() const { return primary_capacity_index_; }
    const std::string &indexer_type() const { return indexer_type_; }
    int64_t max_key_count() const { return max_key_count_; }

    void set_name(const std::string &v) { name_ = v; }
    void set_enabled(bool v) { enabled_ = v; }
    void set_capacity_gb(const std::vector<double> &v) { capacity_gb_ = v; }
    void set_primary_capacity_index(int32_t v) { primary_capacity_index_ = v; }
    void set_indexer_type(const std::string &v) { indexer_type_ = v; }
    void set_max_key_count(int64_t v) { max_key_count_ = v; }

private:
    std::string name_;
    bool enabled_ = false;
    std::vector<double> capacity_gb_;
    int32_t primary_capacity_index_ = 0;
    std::string indexer_type_ = "fenwick_lru";
    int64_t max_key_count_ = 0;
};

} // namespace kv_cache_manager
