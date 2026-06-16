#pragma once

#include <string>
#include <vector>

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
    const std::string &extra_info() const { return extra_info_; }
    void set_extra_info(const std::string &extra_info) { extra_info_ = extra_info; }
    const std::vector<std::string> &event_reporting_storage_candidates() const {
        return event_reporting_storage_candidates_;
    }
    void set_event_reporting_storage_candidates(const std::vector<std::string> &candidates) {
        event_reporting_storage_candidates_ = candidates;
    }

    // Revisit interval histogram bucket boundaries (parsed, validated).
    // Empty vector means "use server-level default".
    const std::vector<double> &revisit_interval_buckets() const { return parsed_revisit_interval_buckets_; }

    // Set from raw config string. Parses and validates immediately.
    // On parse failure, logs a warning and stores empty vector (fallback to default).
    void set_revisit_interval_buckets(const std::string &buckets_str);

    // Raw string for JSON serialization (preserves original config text).
    const std::string &revisit_interval_buckets_raw() const { return revisit_interval_buckets_str_; }

private:
    std::string name_;
    std::vector<std::string> storage_candidates_;
    std::string global_quota_group_name_;
    int64_t max_instance_count_;
    InstanceGroupQuota quota_;
    std::shared_ptr<CacheConfig> cache_config_;
    std::string user_data_;
    int64_t version_;
    std::string extra_info_;
    std::vector<std::string> event_reporting_storage_candidates_;
    std::string revisit_interval_buckets_str_;               // raw string for JSON wire format
    std::vector<double> parsed_revisit_interval_buckets_;     // parsed, validated boundaries
};

} // namespace kv_cache_manager
