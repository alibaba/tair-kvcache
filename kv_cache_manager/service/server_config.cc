#include "kv_cache_manager/service/server_config.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdio.h>

#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/string_util.h"

namespace kv_cache_manager {

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
    {"kvcm.raft.server_id",
     [](const std::string &value, ServerConfig *config) {
         config->raft_server_id_ = std::stoi(value);
         return true;
     }},
    {"kvcm.raft.host",
     [](const std::string &value, ServerConfig *config) {
         config->raft_host_ = value;
         return true;
     }},
    {"kvcm.raft.port",
     [](const std::string &value, ServerConfig *config) {
         config->raft_port_ = std::stoi(value);
         return true;
     }},
    {"kvcm.raft.peers",
     [](const std::string &value, ServerConfig *config) {
         // Format: "id1:host1:port1,id2:host2:port2,..."
         config->raft_peers_.clear();
         if (value.empty()) {
             return true;
         }
         std::istringstream ss(value);
         std::string token;
         while (std::getline(ss, token, ',')) {
             StringUtil::Trim(token);
             if (token.empty()) continue;
             // Split by ':'
             auto pos1 = token.find(':');
             auto pos2 = token.find(':', pos1 + 1);
             if (pos1 == std::string::npos || pos2 == std::string::npos) {
                 fprintf(stderr, "Invalid raft peer format: %s (expected id:host:port)\n", token.c_str());
                 return false;
             }
             ServerConfig::RaftPeer peer;
             peer.server_id = std::stoi(token.substr(0, pos1));
             peer.host = token.substr(pos1 + 1, pos2 - pos1 - 1);
             peer.port = std::stoi(token.substr(pos2 + 1));
             config->raft_peers_.push_back(peer);
         }
         return true;
     }},
    {"kvcm.raft.data_dir",
     [](const std::string &value, ServerConfig *config) {
         config->raft_data_dir_ = value;
         return true;
     }},
    {"kvcm.raft.snapshot_distance",
     [](const std::string &value, ServerConfig *config) {
         config->raft_snapshot_distance_ = std::stoi(value);
         return true;
     }},
    {"kvcm.raft.election_timeout_lower",
     [](const std::string &value, ServerConfig *config) {
         config->raft_election_timeout_lower_ = std::stoi(value);
         return true;
     }},
    {"kvcm.raft.election_timeout_upper",
     [](const std::string &value, ServerConfig *config) {
         config->raft_election_timeout_upper_ = std::stoi(value);
         return true;
     }},
    {"kvcm.raft.heart_beat_interval",
     [](const std::string &value, ServerConfig *config) {
         config->raft_heart_beat_interval_ = std::stoi(value);
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
    if (registry_storage_uri_.empty()) {
        fprintf(stderr, "registry_storage_uri must be set\n");
        return false;
    }
    if (IsRaftEnabled()) {
        if (raft_server_id_ <= 0) {
            fprintf(stderr, "kvcm.raft.server_id must be > 0 when raft is enabled\n");
            return false;
        }
        if (raft_port_ <= 0) {
            fprintf(stderr, "kvcm.raft.port must be > 0 when raft is enabled\n");
            return false;
        }
    }
    return true;
}

} // namespace kv_cache_manager
