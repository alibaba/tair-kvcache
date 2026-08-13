#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/mrc/online_mrc_config.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_group.h"

namespace kv_cache_manager {

class OnlineOptimizerServerConfig : public Jsonizable {
private:
    using EnvironMap = std::unordered_map<std::string, std::string>;

public:
    OnlineOptimizerServerConfig() = default;
    ~OnlineOptimizerServerConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    bool OverrideFromEnviron(const EnvironMap &environ);
    bool Check() const;

    int32_t rpc_port() const { return rpc_port_; }
    int32_t http_port() const { return http_port_; }
    const std::string &registry_storage_uri() const { return registry_storage_uri_; }
    const std::string &metrics_reporter_type() const { return metrics_reporter_type_; }
    int64_t metrics_report_interval_ms() const { return metrics_report_interval_ms_; }
    bool enable_prometheus() const { return enable_prometheus_; }
    const std::string &prometheus_prefix() const { return prometheus_prefix_; }
    int32_t io_thread_num() const { return io_thread_num_; }
    const OnlineMrcConfig &online_mrc_config() const { return online_mrc_config_; }
    const std::vector<OptimizerInstanceGroup> &online_mrc_instance_groups() const {
        return online_mrc_instance_groups_;
    }

private:
    void UpdateEnviron(EnvironMap &environ);

    int32_t rpc_port_ = 50052;
    int32_t http_port_ = 8082;
    std::string registry_storage_uri_;
    std::string metrics_reporter_type_ = "local";
    int64_t metrics_report_interval_ms_ = 10000;
    bool enable_prometheus_ = true;
    std::string prometheus_prefix_ = "kvcm_optimizer";
    int32_t io_thread_num_ = 4;
    OnlineMrcConfig online_mrc_config_;
    std::vector<OptimizerInstanceGroup> online_mrc_instance_groups_;

    using SettingFunction = std::function<bool(const std::string &, OnlineOptimizerServerConfig *)>;
    static std::unordered_map<std::string, SettingFunction> kSettingsMap;
};

} // namespace kv_cache_manager
