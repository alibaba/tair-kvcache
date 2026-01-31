#pragma once

#include <string>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/instance_group_quota.h"

namespace kv_cache_manager {

class InstanceGroup : public Jsonizable {
public:
    ~InstanceGroup() override;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;
    bool ValidateRequiredFields(std::string &invalid_fields) const;

public:
    const std::string &name() const { return name_; }
    const std::vector<std::string> &storage_candidates() const { return storage_candidates_; }
    const std::string &global_quota_group_name() const { return global_quota_group_name_; }
    int64_t max_instance_count() const { return max_instance_count_; }
    const InstanceGroupQuota &quota() const { return quota_; }
    CacheConfigConstPtr cache_config() const { return cache_config_; }
    const std::string &user_data() const { return user_data_; }
    int64_t version() const { return version_; }
    // Setters
    void set_name(const std::string &name) { name_ = name; }
    void set_storage_candidates(const std::vector<std::string> &storage_candidates) {
        storage_candidates_ = storage_candidates;
    }
    void set_global_quota_group_name(const std::string &global_quota_group_name) {
        global_quota_group_name_ = global_quota_group_name;
    }
    void set_max_instance_count(int64_t max_instance_count) { max_instance_count_ = max_instance_count; }
    void set_quota(const InstanceGroupQuota &quota) { quota_ = quota; }
    void set_cache_config(const std::shared_ptr<CacheConfig> &cache_config) { cache_config_ = cache_config; }
    void set_user_data(const std::string &user_data) { user_data_ = user_data; }
    void set_version(int64_t version) { version_ = version; }

private:
    std::string name_;
    std::vector<std::string> storage_candidates_;
    std::string global_quota_group_name_;
    int64_t max_instance_count_;
    InstanceGroupQuota quota_;
    std::shared_ptr<CacheConfig> cache_config_;
    std::string user_data_;
    int64_t version_;
};

} // namespace kv_cache_manager