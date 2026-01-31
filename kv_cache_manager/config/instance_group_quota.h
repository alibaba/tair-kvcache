#pragma once

#include <string>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/config/quota_config.h"

namespace kv_cache_manager {

class InstanceGroupQuota : public Jsonizable {
public:
    InstanceGroupQuota() = default;
    InstanceGroupQuota(int64_t capacity, const std::vector<QuotaConfig> &quota_config)
        : capacity_(capacity), quota_config_(quota_config) {}

    ~InstanceGroupQuota() override;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    bool ValidateRequiredFields(std::string &invalid_fields) const;
    // Getters
    int64_t capacity() const { return capacity_; }
    const std::vector<QuotaConfig> &quota_config() const { return quota_config_; }

    // Setters
    void set_capacity(int64_t capacity) { capacity_ = capacity; }
    void set_quota_config(const std::vector<QuotaConfig> &quota_config) { quota_config_ = quota_config; }

private:
    int64_t capacity_;
    std::vector<QuotaConfig> quota_config_;
};

} // namespace kv_cache_manager