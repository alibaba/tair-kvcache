#include "kv_cache_manager/optimizer/service/online_optimizer_server_config.h"

#include <algorithm>

#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/string_util.h"

namespace kv_cache_manager {

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
    return true;
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
    return success;
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
