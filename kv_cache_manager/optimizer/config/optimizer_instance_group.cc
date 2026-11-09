#include "kv_cache_manager/optimizer/config/optimizer_instance_group.h"

#include <cmath>
#include <limits>
#include <unordered_set>

namespace kv_cache_manager {

namespace {

constexpr double kBytesPerGb = 1024.0 * 1024.0 * 1024.0;
constexpr double kMaxCapacityGbForInt64 = static_cast<double>(std::numeric_limits<int64_t>::max()) / kBytesPerGb;

} // namespace

bool OptimizerInstanceGroup::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "name", name_, std::string(""));
    KVCM_JSON_GET_MACRO(rapid_value, "capacity_gb", capacity_gb_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "eviction_policy", eviction_policy_, std::string("lru"));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "shared_group_quota", shared_group_quota_, false);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "enable_theoretical_max_cache", enable_theoretical_max_cache_, false);
    KVCM_JSON_GET_MACRO(rapid_value, "ttl_seconds_list", ttl_seconds_list_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "enable_prefix_hash", enable_prefix_hash_, false);
    SortCapacities();
    return true;
}

void OptimizerInstanceGroup::SortCapacities() { std::sort(capacity_gb_.begin(), capacity_gb_.end()); }

void OptimizerInstanceGroup::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "name", name_);
    Put(writer, "capacity_gb", capacity_gb_);
    Put(writer, "eviction_policy", eviction_policy_);
    Put(writer, "shared_group_quota", shared_group_quota_);
    Put(writer, "enable_theoretical_max_cache", enable_theoretical_max_cache_);
    Put(writer, "ttl_seconds_list", ttl_seconds_list_);
    Put(writer, "enable_prefix_hash", enable_prefix_hash_);
}

bool OptimizerInstanceGroup::ValidateRequiredFields(std::string &invalid_fields) const {
    bool valid = true;
    std::string local_invalid_fields;
    if (name_.empty()) {
        valid = false;
        local_invalid_fields += "{name}";
    }
    for (double cap : capacity_gb_) {
        if (!std::isfinite(cap) || cap <= 0.0) {
            valid = false;
            local_invalid_fields += "{capacity_gb: non-positive or non-finite value}";
            break;
        }
        if (cap > kMaxCapacityGbForInt64) {
            valid = false;
            local_invalid_fields += "{capacity_gb: value exceeds int64 byte range}";
            break;
        }
    }
    if (eviction_policy_ != "lru") {
        valid = false;
        local_invalid_fields += "{eviction_policy}";
    }
    std::unordered_set<int64_t> ttls;
    if (ttl_seconds_list_.empty()) {
        valid = false;
        local_invalid_fields += "{ttl_seconds_list}";
    }
    for (int64_t ttl : ttl_seconds_list_) {
        if (ttl < 0 || ttl > std::numeric_limits<int64_t>::max() / 1000000000LL || !ttls.insert(ttl).second) {
            valid = false;
            local_invalid_fields += "{ttl_seconds_list: invalid or duplicate value}";
        }
    }
    if (capacity_gb_.empty() && !enable_theoretical_max_cache_) {
        valid = false;
        local_invalid_fields += "{no analysis requested}";
    }
    if (!valid) {
        invalid_fields += "{OptimizerInstanceGroup: " + local_invalid_fields + "}";
    }
    return valid;
}

} // namespace kv_cache_manager
