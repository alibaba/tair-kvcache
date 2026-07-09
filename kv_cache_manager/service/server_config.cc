#include "kv_cache_manager/service/server_config.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdio.h>

#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/string_util.h"

namespace kv_cache_manager {

std::vector<double> ServerConfig::ParseRevisitIntervalBuckets(const std::string &buckets_str) {
    auto boundaries = StringUtil::ParseBucketBoundaries(buckets_str);
    if (!buckets_str.empty() && boundaries.empty()) {
        fprintf(stderr, "Invalid revisit_interval_buckets: must be positive numbers in strictly ascending order\n");
    }
    return boundaries;
}

const std::vector<double> &ServerConfig::GetDefaultRevisitIntervalBuckets() {
    static const std::vector<double> kDefaults = {1, 5, 30, 60, 120, 180, 300, 600, 900, 1800, 3600, 21600, 86400};
    return kDefaults;
}

// clang-format off
std::unordered_map<std::string, ServerConfig::SettingFunction> ServerConfig::kSettingsMap = {
    {"kvcm.registry_storage.uri",
     [](const std::string &value, ServerConfig *config) {
         config->registry_storage_uri_ = value;
         return true;
     }},
    {"kvcm.coordination.uri",
     [](const std::string &value, ServerConfig *config) {
         config->coordination_uri_ = value;
         return true;
     }},
    {"kvcm.distributed_lock.uri",
     [](const std::string &value, ServerConfig *config) {
         // Backward compatibility: only set if not already set by kvcm.coordination.uri
         if (config->coordination_uri_.empty()) {
             config->coordination_uri_ = value;
         }
         return true;
     }},
    {"kvcm.leader_elector.node_id",
     [](const std::string &value, ServerConfig *config) {
         config->leader_elector_node_id_ = value;
         return true;
     }},
    {"kvcm.leader_elector.lease_ms",
     [](const std::string &value, ServerConfig *config) {
         config->leader_elector_lease_ms_ = std::stol(value);
         return true;
     }},
    {"kvcm.leader_elector.loop_interval_ms",
     [](const std::string &value, ServerConfig *config) {
         config->leader_elector_loop_interval_ms_ = std::stol(value);
         return true;
     }},
    {"kvcm.service.io_thread_num",
     [](const std::string &value, ServerConfig *config) {
         config->service_io_thread_num_ = std::stoi(value);
         return true;
     }},
    {"kvcm.service.rpc_port",
     [](const std::string &value, ServerConfig *config) {
         config->service_rpc_port_ = std::stoi(value);
         return true;
     }},
    {"kvcm.service.http_port",
     [](const std::string &value, ServerConfig *config) {
         config->service_http_port_ = std::stoi(value);
         return true;
     }},
    {"kvcm.service.admin_rpc_port",
     [](const std::string &value, ServerConfig *config) {
         config->service_admin_rpc_port_ = std::stoi(value);
         return true;
     }},
    {"kvcm.service.admin_http_port",
     [](const std::string &value, ServerConfig *config) {
         config->service_admin_http_port_ = std::stoi(value);
         return true;
     }},
    {"kvcm.service.enable_debug_service",
     [](const std::string &value, ServerConfig *config) {
         config->enable_debug_service_ = value == "true";
         return true;
     }},
    {"kvcm.logger.log_level",
     [](const std::string &value, ServerConfig *config) {
         // 0: auto, 1: fatal, 2: error, 3: warn, 4: info, 5: debug
         config->log_level_ = std::stoi(value);
         return true;
     }},
    {"kvcm.startup_config",
     [](const std::string &value, ServerConfig *config) {
         config->startup_config_ = value;
         return true;
     }},
    {"kvcm.schedule_plan_executor_thread_count",
     [](const std::string &value, ServerConfig *config) {
         config->schedule_plan_executor_thread_count_ = std::stoi(value);
         return true;
     }},
    {"kvcm.cache_reclaimer.key_sampling_size_total",
     [](const std::string &value, ServerConfig *config) {
         config->cache_reclaimer_key_sampling_size_total_ = std::stoull(value);
         return true;
     }},
    {"kvcm.cache_reclaimer.key_sampling_size_per_task",
     [](const std::string &value, ServerConfig *config) {
         config->cache_reclaimer_key_sampling_size_per_task_ = std::stoull(value);
         return true;
     }},
    {"kvcm.cache_reclaimer.del_batch_size",
     [](const std::string &value, ServerConfig *config) {
         config->cache_reclaimer_del_batch_size_ = std::stoull(value);
         return true;
     }},
    {"kvcm.cache_reclaimer.idle_interval_ms",
     [](const std::string &value, ServerConfig *config) {
         config->cache_reclaimer_idle_interval_ms_ = std::stol(value);
         return true;
     }},
    {"kvcm.cache_reclaimer.worker_size",
     [](const std::string &value, ServerConfig *config) {
         config->cache_reclaimer_worker_size_ = std::stol(value);
         return true;
    }},
    {"kvcm.metrics.reporter_type",
     [](const std::string &value, ServerConfig *config) {
         config->metrics_reporter_type_ = value;
         return true;
     }},
    {"kvcm.metrics.reporter_config",
     [](const std::string &value, ServerConfig *config) {
         config->metrics_reporter_config_ = value;
         return true;
     }},
    {"kvcm.metrics.report_interval_ms",
     [](const std::string &value, ServerConfig *config) {
         config->metrics_report_interval_ms_ = std::stol(value);
         return true;
     }},
    {"kvcm.event.event_publishers_configs", [](const std::string &value, ServerConfig *config) {
         config->event_publishers_configs_ = value;
         return true;
     }},
    {"kvcm.service.advertised_host",
     [](const std::string &value, ServerConfig *config) {
         config->advertised_host_ = value;
         return true;
     }},
    {"kvcm.service.custom_info",
     [](const std::string &value, ServerConfig *config) {
         config->custom_info_ = value;
         return true;
     }},
    {"kvcm.metrics.enable_prometheus",
     [](const std::string &value, ServerConfig *config) {
         config->enable_prometheus_ = value == "true";
         return true;
     }},
    {"kvcm.metrics.prometheus_prefix",
     [](const std::string &value, ServerConfig *config) {
         config->prometheus_prefix_ = value;
         return true;
     }},
    {"kvcm.metrics.revisit_interval_buckets",
     [](const std::string &value, ServerConfig *config) {
         config->revisit_interval_buckets_ = value;
         return true;
     }}};
// clang-format on

bool ServerConfig::Parse(const std::string &config_file, const EnvironMap &environ) {
    // 1. 默认配置
    // 2. 以文件的配置作为基础配置
    // 3. 使用环境变量覆盖基础配置
    UpdateDefaultConfig();
    if (!ParseFromFile(config_file)) {
        fprintf(stderr, "Parse config file [%s] failed\n", config_file.c_str());
        return false;
    }
    if (!ParseFromEnviron(environ)) {
        fprintf(stderr, "Parse config from environ failed\n");
        return false;
    }
    return true;
}

void ServerConfig::UpdateDefaultConfig() {
    metrics_report_interval_ms_ = 20000;
    leader_elector_lease_ms_ = 10000;
    leader_elector_loop_interval_ms_ = 100;
    cache_reclaimer_key_sampling_size_total_ = 1000;
    cache_reclaimer_key_sampling_size_per_task_ = 100;
    cache_reclaimer_del_batch_size_ = 100;
    cache_reclaimer_idle_interval_ms_ = 100;
    cache_reclaimer_worker_size_ = 16;
}

bool ServerConfig::ParseFromFile(const std::string &config_file) {
    if (config_file.empty()) {
        fprintf(stdout, "Config file is empty, ignored.\n");
        return true;
    }
    std::ifstream in(config_file);
    if (!in.is_open()) {
        fprintf(stderr, "Open config file %s failed.\n", config_file.c_str());
        return false;
    }
    bool success = true;
    std::string line;
    int32_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        // 去掉行内注释
        auto pos_hash = line.find('#');
        if (pos_hash != std::string::npos) {
            line.erase(pos_hash);
        };
        StringUtil::Trim(line);
        if (line.empty()) {
            continue;
        };
        auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            fprintf(stderr, "Invalid config line (no '=') at line %d: %s\n", line_no, line.c_str());
            success = false;
            continue;
        }
        std::string key = line.substr(0, eq_pos);
        std::string val = line.substr(eq_pos + 1);
        StringUtil::Trim(key);
        StringUtil::Trim(val);
        auto setting_it = kSettingsMap.find(key);
        if (setting_it == kSettingsMap.end()) {
            fprintf(stderr, "Unknown config key at line %d: %s\n", line_no, key.c_str());
            continue;
        }
        try {
            if (!setting_it->second(val, this)) {
                fprintf(stderr, "Parse value failed at line %d: %s = %s\n", line_no, key.c_str(), val.c_str());
                success = false;
            }
        } catch (...) {
            fprintf(stderr, "Invalid value for config: %s = %s\n", key.c_str(), val.c_str());
            success = false;
        }
    }
    return success;
}

bool ServerConfig::ParseFromEnviron(const EnvironMap &environ) {
    EnvironMap copy_environ = environ;
    UpdateEnviron(copy_environ);

    bool success = true;
    for (const auto &[k, v] : copy_environ) {
        std::string key = k, val = v;
        StringUtil::Trim(key);
        StringUtil::Trim(val);
        auto setting_it = kSettingsMap.find(key);
        if (setting_it == kSettingsMap.end()) {
            fprintf(stderr, "Unknown config key %s\n", key.c_str());
            continue;
        }
        if (!setting_it->second(val, this)) {
            fprintf(stderr, "Invalid value for config: %s = %s\n", key.c_str(), val.c_str());
            success = false;
            continue;
        }
    }
    return success;
}

void ServerConfig::UpdateEnviron(EnvironMap &environ) {
    // TODO 环境变量和原始ENV的覆盖关系
    // 系统环境变量具有最高优先级，覆盖了传入的environ中的同名配置项
    // 优先查带 `.` 的 key（如 kvcm.registry_storage.uri），查不到再尝试将 `.`
    // 替换为 `_` 后查（如 kvcm_registry_storage_uri），兼容 bash 等不支持
    // 变量名包含 `.` 的场景。
    for (const auto &[key, _] : kSettingsMap) {
        std::string value = EnvUtil::GetEnv(key, "");
        if (value.empty()) {
            std::string underscore_key = key;
            std::replace(underscore_key.begin(), underscore_key.end(), '.', '_');
            if (underscore_key != key) {
                value = EnvUtil::GetEnv(underscore_key, "");
            }
        }
        if (!value.empty()) {
            environ[key] = value;
        }
    }
}

bool ServerConfig::Check() {
    // registry_storage_uri is optional: when empty, RegistryStorageBackendFactory
    // falls back to local backend. No validation needed for empty value.

    // Validate revisit_interval_buckets if set
    if (!revisit_interval_buckets_.empty()) {
        auto boundaries = ParseRevisitIntervalBuckets(revisit_interval_buckets_);
        if (boundaries.empty()) {
            return false; // ParseRevisitIntervalBuckets already printed the error
        }
    }

    return true;
}

} // namespace kv_cache_manager
