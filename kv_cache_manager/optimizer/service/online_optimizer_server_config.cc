#include "kv_cache_manager/optimizer/service/online_optimizer_server_config.h"

#include <algorithm>
#include <unordered_set>

#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/string_util.h"

namespace kv_cache_manager {

bool KvcmEventSubscriptionConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "service_discovery_url", service_discovery_url_, std::string());
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "consumer_id", consumer_id_, std::string("online-optimizer"));
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "discovery_refresh_interval_ms", discovery_refresh_interval_ms_, int64_t(5000));
    return Validate();
}

void KvcmEventSubscriptionConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "service_discovery_url", service_discovery_url_);
    Put(writer, "consumer_id", consumer_id_);
    Put(writer, "discovery_refresh_interval_ms", discovery_refresh_interval_ms_);
}

bool KvcmEventSubscriptionConfig::Validate() const {
    return !service_discovery_url_.empty() && !consumer_id_.empty() && discovery_refresh_interval_ms_ > 0;
}

// clang-format off
std::unordered_map<std::string, OnlineOptimizerServerConfig::SettingFunction>
    OnlineOptimizerServerConfig::kSettingsMap = {
    {"kvcm_optimizer.rpc_port",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->rpc_port_ = std::stoi(value);
         return true;
     }},
    {"kvcm_optimizer.http_port",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->http_port_ = std::stoi(value);
         return true;
     }},
    {"kvcm_optimizer.registry_storage_uri",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->registry_storage_uri_ = value;
         return true;
     }},
    {"kvcm_optimizer.metrics_reporter_type",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->metrics_reporter_type_ = value;
         return true;
     }},
    {"kvcm_optimizer.metrics_report_interval_ms",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->metrics_report_interval_ms_ = std::stol(value);
         return true;
     }},
    {"kvcm_optimizer.enable_prometheus",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->enable_prometheus_ = value == "true";
         return true;
     }},
    {"kvcm_optimizer.prometheus_prefix",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->prometheus_prefix_ = value;
         return true;
     }},
    {"kvcm_optimizer.io_thread_num",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->io_thread_num_ = std::stoi(value);
         return true;
     }},
    {"kvcm_optimizer.kvcm_event_subscriptions",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         return Jsonizable::FromJsonString(value, config->kvcm_event_subscriptions_);
     }},
    {"kvcm_optimizer.quota_planner_enable",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->quota_planner_config_.enable = value == "true";
         return value == "true" || value == "false";
     }},
    {"kvcm_optimizer.quota_planner_enable_hard_resize",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->quota_planner_config_.enable_hard_resize = value == "true";
         return value == "true" || value == "false";
     }},
    {"kvcm_optimizer.quota_planner_period_seconds",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->quota_planner_config_.period_seconds = std::stol(value);
         return true;
     }},
    {"kvcm_optimizer.quota_planner_plan_ttl_seconds",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->quota_planner_config_.plan_ttl_seconds = std::stol(value);
         return true;
     }},
    {"kvcm_optimizer.quota_planner_release_timeout_seconds",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->quota_planner_config_.release_timeout_seconds = std::stol(value);
         return true;
     }},
    {"kvcm_optimizer.quota_planner_release_consecutive_samples",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->quota_planner_config_.release_consecutive_samples = std::stol(value);
         return true;
     }},
    {"kvcm_optimizer.quota_planner_pools",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         return Jsonizable::FromJsonString(value, config->quota_planner_config_.pools);
     }},
};
// clang-format on

bool OnlineOptimizerServerConfig::FromRapidValue(const rapidjson::Value &rapid_value) {
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "rpc_port", rpc_port_, int32_t(50052));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "http_port", http_port_, int32_t(8082));
    KVCM_JSON_GET_MACRO(rapid_value, "registry_storage_uri", registry_storage_uri_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "metrics_reporter_type", metrics_reporter_type_, std::string("local"));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "metrics_report_interval_ms", metrics_report_interval_ms_, int64_t(10000));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "enable_prometheus", enable_prometheus_, true);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "prometheus_prefix", prometheus_prefix_, std::string("kvcm_optimizer"));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "io_thread_num", io_thread_num_, int32_t(4));
    KVCM_JSON_GET_MACRO(rapid_value, "kvcm_event_subscriptions", kvcm_event_subscriptions_);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "quota_planner_enable", quota_planner_config_.enable, false);
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "quota_planner_enable_hard_resize", quota_planner_config_.enable_hard_resize, false);
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "quota_planner_period_seconds", quota_planner_config_.period_seconds, int64_t(300));
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "quota_planner_plan_ttl_seconds", quota_planner_config_.plan_ttl_seconds, int64_t(900));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value,
                                "quota_planner_release_timeout_seconds",
                                quota_planner_config_.release_timeout_seconds,
                                int64_t(1800));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value,
                                "quota_planner_release_consecutive_samples",
                                quota_planner_config_.release_consecutive_samples,
                                int64_t(3));
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "quota_planner_pools", quota_planner_config_.pools, std::vector<QuotaPoolConfig>());
    return ValidateKvcmEventSubscriptions() && ValidateQuotaPlanner();
}

void OnlineOptimizerServerConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "rpc_port", rpc_port_);
    Put(writer, "http_port", http_port_);
    Put(writer, "registry_storage_uri", registry_storage_uri_);
    Put(writer, "metrics_reporter_type", metrics_reporter_type_);
    Put(writer, "metrics_report_interval_ms", metrics_report_interval_ms_);
    Put(writer, "enable_prometheus", enable_prometheus_);
    Put(writer, "prometheus_prefix", prometheus_prefix_);
    Put(writer, "io_thread_num", io_thread_num_);
    Put(writer, "kvcm_event_subscriptions", kvcm_event_subscriptions_);
    Put(writer, "quota_planner_enable", quota_planner_config_.enable);
    Put(writer, "quota_planner_enable_hard_resize", quota_planner_config_.enable_hard_resize);
    Put(writer, "quota_planner_period_seconds", quota_planner_config_.period_seconds);
    Put(writer, "quota_planner_plan_ttl_seconds", quota_planner_config_.plan_ttl_seconds);
    Put(writer, "quota_planner_release_timeout_seconds", quota_planner_config_.release_timeout_seconds);
    Put(writer, "quota_planner_release_consecutive_samples", quota_planner_config_.release_consecutive_samples);
    Put(writer, "quota_planner_pools", quota_planner_config_.pools);
}

bool OnlineOptimizerServerConfig::OverrideFromEnviron(const EnvironMap &environ) {
    EnvironMap merged = environ;
    UpdateEnviron(merged);

    bool success = true;
    for (const auto &[k, v] : merged) {
        std::string key = k, val = v;
        StringUtil::Trim(key);
        StringUtil::Trim(val);
        auto setting_it = kSettingsMap.find(key);
        if (setting_it == kSettingsMap.end()) {
            fprintf(stderr, "Unknown optimizer config key: %s\n", key.c_str());
            continue;
        }
        try {
            if (!setting_it->second(val, this)) {
                fprintf(stderr, "Invalid value for optimizer config: %s = %s\n", key.c_str(), val.c_str());
                success = false;
            }
        } catch (...) {
            fprintf(stderr, "Invalid value for optimizer config: %s = %s\n", key.c_str(), val.c_str());
            success = false;
        }
    }
    return success && ValidateKvcmEventSubscriptions() && ValidateQuotaPlanner();
}

bool OnlineOptimizerServerConfig::ValidateKvcmEventSubscriptions() const {
    std::unordered_set<std::string> discovery_urls;
    for (const auto &subscription : kvcm_event_subscriptions_) {
        if (!subscription.Validate() || !discovery_urls.insert(subscription.service_discovery_url()).second) {
            return false;
        }
    }
    return true;
}

bool OnlineOptimizerServerConfig::ValidateQuotaPlanner() const {
    if (!quota_planner_config_.enable) {
        return !quota_planner_config_.enable_hard_resize;
    }
    if (quota_planner_config_.period_seconds <= 0 || quota_planner_config_.plan_ttl_seconds <= 0 ||
        quota_planner_config_.release_timeout_seconds <= 0 || quota_planner_config_.release_consecutive_samples <= 0 ||
        quota_planner_config_.pools.empty()) {
        return false;
    }
    if (quota_planner_config_.enable_hard_resize &&
        quota_planner_config_.plan_ttl_seconds <= quota_planner_config_.release_timeout_seconds) {
        return false;
    }
    std::unordered_set<std::string> pool_ids;
    for (const auto &pool : quota_planner_config_.pools) {
        std::string reason;
        if (!pool.Check(reason) || !pool_ids.insert(pool.pool_id).second) {
            return false;
        }
    }
    return true;
}

void OnlineOptimizerServerConfig::UpdateEnviron(EnvironMap &environ) {
    for (const auto &[key, _] : kSettingsMap) {
        std::string value = EnvUtil::GetEnv(key, std::string(""));
        if (value.empty()) {
            std::string underscore_key = key;
            std::replace(underscore_key.begin(), underscore_key.end(), '.', '_');
            if (underscore_key != key) {
                value = EnvUtil::GetEnv(underscore_key, std::string(""));
            }
        }
        if (!value.empty()) {
            environ[key] = value;
        }
    }
}

} // namespace kv_cache_manager
