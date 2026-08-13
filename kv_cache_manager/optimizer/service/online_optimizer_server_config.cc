#include "kv_cache_manager/optimizer/service/online_optimizer_server_config.h"

#include <algorithm>
#include <unordered_set>

#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/string_util.h"

namespace kv_cache_manager {
namespace {

class OnlineMrcGroupsConfig : public Jsonizable {
public:
    bool FromRapidValue(const rapidjson::Value &value) override {
        KVCM_JSON_GET_MACRO(value, "instance_groups", instance_groups);
        return true;
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "instance_groups", instance_groups);
    }

    std::vector<OptimizerInstanceGroup> instance_groups;
};

} // namespace

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
    {"optimizer.online_mrc.enable",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->online_mrc_config_.enable = value == "true";
         return true;
     }},
    {"optimizer.online_mrc.capacity_gb_grid",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         auto grid = StringUtil::ParseBucketBoundaries(value);
         if (grid.empty()) return false;
         config->online_mrc_config_.capacity_gb_grid = std::move(grid);
         return true;
     }},
    {"optimizer.online_mrc.report_interval_seconds",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->online_mrc_config_.report_interval_seconds = std::stol(value);
         return true;
     }},
    {"optimizer.online_mrc.instance_groups",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         OnlineMrcGroupsConfig parsed;
         if (!parsed.FromJsonString("{\"instance_groups\":" + value + "}") || parsed.instance_groups.empty()) {
             return false;
         }
         config->online_mrc_instance_groups_ = std::move(parsed.instance_groups);
         return true;
     }},
    {"optimizer.online_mrc.max_instances",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->online_mrc_config_.max_instances = std::stoi(value);
         return true;
     }},
    {"optimizer.online_mrc.receiver_queue_max_batches",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->online_mrc_config_.receiver_queue_max_batches = std::stol(value);
         return true;
     }},
    {"optimizer.online_mrc.kvcm_service_discovery_url",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->online_mrc_config_.kvcm_service_discovery_url = value;
         return true;
     }},
    {"optimizer.online_mrc.discovery_refresh_interval_ms",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->online_mrc_config_.discovery_refresh_interval_ms = std::stol(value);
         return true;
     }},
    {"optimizer.online_mrc.connect_timeout_ms",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->online_mrc_config_.connect_timeout_ms = std::stol(value);
         return true;
     }},
    {"optimizer.online_mrc.reconnect_interval_ms",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->online_mrc_config_.reconnect_interval_ms = std::stol(value);
         return true;
     }},
    {"optimizer.online_mrc.max_frame_bytes",
     [](const std::string &value, OnlineOptimizerServerConfig *config) {
         config->online_mrc_config_.max_frame_bytes = std::stol(value);
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
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value, "online_mrc_enable", online_mrc_config_.enable, false);
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value,
                                "online_mrc_capacity_gb_grid",
                                online_mrc_config_.capacity_gb_grid,
                                std::vector<double>({64, 128, 256, 340, 512, 1024}));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value,
                                "online_mrc_report_interval_seconds",
                                online_mrc_config_.report_interval_seconds,
                                int64_t(60));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value,
                                "online_mrc_instance_groups",
                                online_mrc_instance_groups_,
                                std::vector<OptimizerInstanceGroup>());
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "online_mrc_max_instances", online_mrc_config_.max_instances, int32_t(256));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value,
                                "online_mrc_receiver_queue_max_batches",
                                online_mrc_config_.receiver_queue_max_batches,
                                int64_t(1024));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value,
                                "online_mrc_kvcm_service_discovery_url",
                                online_mrc_config_.kvcm_service_discovery_url,
                                std::string(""));
    KVCM_JSON_GET_DEFAULT_MACRO(rapid_value,
                                "online_mrc_discovery_refresh_interval_ms",
                                online_mrc_config_.discovery_refresh_interval_ms,
                                int64_t(30000));
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "online_mrc_connect_timeout_ms", online_mrc_config_.connect_timeout_ms, int64_t(500));
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "online_mrc_reconnect_interval_ms", online_mrc_config_.reconnect_interval_ms, int64_t(1000));
    KVCM_JSON_GET_DEFAULT_MACRO(
        rapid_value, "online_mrc_max_frame_bytes", online_mrc_config_.max_frame_bytes, int64_t(8 * 1024 * 1024));
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
    Put(writer, "online_mrc_enable", online_mrc_config_.enable);
    Put(writer, "online_mrc_capacity_gb_grid", online_mrc_config_.capacity_gb_grid);
    Put(writer, "online_mrc_report_interval_seconds", online_mrc_config_.report_interval_seconds);
    Put(writer, "online_mrc_instance_groups", online_mrc_instance_groups_);
    Put(writer, "online_mrc_max_instances", online_mrc_config_.max_instances);
    Put(writer, "online_mrc_receiver_queue_max_batches", online_mrc_config_.receiver_queue_max_batches);
    Put(writer, "online_mrc_kvcm_service_discovery_url", online_mrc_config_.kvcm_service_discovery_url);
    Put(writer, "online_mrc_discovery_refresh_interval_ms", online_mrc_config_.discovery_refresh_interval_ms);
    Put(writer, "online_mrc_connect_timeout_ms", online_mrc_config_.connect_timeout_ms);
    Put(writer, "online_mrc_reconnect_interval_ms", online_mrc_config_.reconnect_interval_ms);
    Put(writer, "online_mrc_max_frame_bytes", online_mrc_config_.max_frame_bytes);
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

bool OnlineOptimizerServerConfig::Check() const {
    if (rpc_port_ < 1 || rpc_port_ > 65535 || http_port_ < 1 || http_port_ > 65535 ||
        metrics_report_interval_ms_ <= 0 || io_thread_num_ <= 0) {
        fprintf(stderr, "Invalid optimizer server port/thread/report settings\n");
        return false;
    }
    if (!online_mrc_config_.enable) {
        return true;
    }
    if (online_mrc_config_.report_interval_seconds <= 0 || online_mrc_config_.max_instances <= 0 ||
        online_mrc_config_.receiver_queue_max_batches <= 0 ||
        online_mrc_config_.capacity_gb_grid.empty() || online_mrc_instance_groups_.empty()) {
        fprintf(stderr, "Invalid optimizer online MRC bounds\n");
        return false;
    }
    if (online_mrc_config_.kvcm_service_discovery_url.empty() ||
        online_mrc_config_.discovery_refresh_interval_ms <= 0 || online_mrc_config_.connect_timeout_ms <= 0 ||
        online_mrc_config_.reconnect_interval_ms <= 0 || online_mrc_config_.max_frame_bytes <= 1) {
        fprintf(stderr, "Invalid optimizer-initiated online MRC stream settings\n");
        return false;
    }
    double previous = 0;
    for (const double capacity_gb : online_mrc_config_.capacity_gb_grid) {
        if (capacity_gb <= previous) {
            fprintf(stderr, "optimizer online MRC capacity grid must be positive and strictly increasing\n");
            return false;
        }
        previous = capacity_gb;
    }
    std::unordered_set<std::string> unique_groups;
    for (const auto &group : online_mrc_instance_groups_) {
        std::string invalid_fields;
        if (!group.ValidateRequiredFields(invalid_fields) || !group.enable_prefix_hash() ||
            !unique_groups.emplace(group.name()).second) {
            fprintf(stderr,
                    "optimizer online MRC instance groups must be valid, unique, and enable_prefix_hash=true\n");
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
