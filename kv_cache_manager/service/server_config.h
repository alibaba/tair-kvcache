#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>

namespace kv_cache_manager {

class ServerConfig;

class ServerConfig {
private:
    using EnvironMap = std::unordered_map<std::string, std::string>;

public:
    bool Parse(const std::string &config_file, const EnvironMap &environ);
    bool Check();

public:
    const std::string &GetRegistryStorageUri() const { return registry_storage_uri_; }
    const std::string &GetDistributedLockUri() const { return distributed_lock_uri_; }
    const std::string &GetLeaderElectorNodeId() const { return leader_elector_node_id_; }
    int64_t GetLeaderElectorLeaseMs() const { return leader_elector_lease_ms_; }
    int64_t GetLeaderElectorLoopIntervalMs() const { return leader_elector_loop_interval_ms_; }
    int32_t GetServiceIoThreadNum() const { return service_io_thread_num_; }
    int32_t GetServiceRpcPort() const { return service_rpc_port_; }
    int32_t GetServiceHttpPort() const { return service_http_port_; }
    int32_t GetServiceAdminRpcPort() const { return service_admin_rpc_port_; }
    int32_t GetServiceAdminHttpPort() const { return service_admin_http_port_; }
    bool IsEnableDebugService() const { return enable_debug_service_; }
    uint32_t GetLogLevel() const { return log_level_; }
    const std::string &startup_config() { return startup_config_; }
    int32_t GetSchedulePlanExecutorThreadCount() { return schedule_plan_executor_thread_count_; }
    const std::string &metrics_reporter_type() { return metrics_reporter_type_; }
    const std::string &metrics_reporter_config() { return metrics_reporter_config_; }
    int64_t metrics_report_interval_ms() { return metrics_report_interval_ms_; }
    const std::string &event_publishers_configs() { return event_publishers_configs_; }

private:
    void UpdateDefaultConfig();
    bool ParseFromFile(const std::string &config_file);
    bool ParseFromEnviron(const EnvironMap &environ);
    void UpdateEnviron(EnvironMap &environ);

private:
    std::string registry_storage_uri_;
    std::string distributed_lock_uri_;
    std::string leader_elector_node_id_;
    int64_t leader_elector_lease_ms_ = 0;
    int64_t leader_elector_loop_interval_ms_ = 0;
    int32_t service_io_thread_num_ = 0;
    int32_t service_rpc_port_ = 0;
    int32_t service_http_port_ = 0;
    int32_t service_admin_rpc_port_ = 0;
    int32_t service_admin_http_port_ = 0;
    bool enable_debug_service_ = false;
    uint32_t log_level_ = 0;
    std::string startup_config_;
    int32_t schedule_plan_executor_thread_count_ = 0;
    std::string metrics_reporter_type_;
    std::string metrics_reporter_config_;
    int64_t metrics_report_interval_ms_ = 0;
    std::string event_publishers_configs_;

private:
    using SettingFunction = std::function<bool(const std::string &, ServerConfig *config)>;
    static std::unordered_map<std::string, SettingFunction> kSettingsMap;
};

} // namespace kv_cache_manager
