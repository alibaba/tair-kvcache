#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/common/jsonizable.h"

namespace kv_cache_manager {

class KvcmEventSubscriptionConfig : public Jsonizable {
public:
    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    bool Validate() const;

    const std::string &service_discovery_url() const { return service_discovery_url_; }
    const std::string &consumer_id() const { return consumer_id_; }
    int64_t discovery_refresh_interval_ms() const { return discovery_refresh_interval_ms_; }
    const std::vector<double> &capacity_gb() const { return capacity_gb_; }

private:
    friend class OnlineOptimizerServerConfig;

    std::string service_discovery_url_;
    std::string consumer_id_ = "online-optimizer";
    int64_t discovery_refresh_interval_ms_ = 5000;
    std::vector<double> capacity_gb_;
};

class OnlineOptimizerServerConfig : public Jsonizable {
private:
    using EnvironMap = std::unordered_map<std::string, std::string>;

public:
    OnlineOptimizerServerConfig() = default;
    ~OnlineOptimizerServerConfig() override = default;

    bool FromRapidValue(const rapidjson::Value &rapid_value) override;
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override;

    bool OverrideFromEnviron(const EnvironMap &environ);

    int32_t rpc_port() const { return rpc_port_; }
    int32_t http_port() const { return http_port_; }
    const std::string &registry_storage_uri() const { return registry_storage_uri_; }
    const std::string &metrics_reporter_type() const { return metrics_reporter_type_; }
    int64_t metrics_report_interval_ms() const { return metrics_report_interval_ms_; }
    bool enable_prometheus() const { return enable_prometheus_; }
    const std::string &prometheus_prefix() const { return prometheus_prefix_; }
    int32_t io_thread_num() const { return io_thread_num_; }
    const std::vector<KvcmEventSubscriptionConfig> &kvcm_event_subscriptions() const {
        return kvcm_event_subscriptions_;
    }

private:
    void UpdateEnviron(EnvironMap &environ);
    bool ValidateKvcmEventSubscriptions() const;

    int32_t rpc_port_ = 50052;
    int32_t http_port_ = 8082;
    std::string registry_storage_uri_;
    std::string metrics_reporter_type_ = "local";
    int64_t metrics_report_interval_ms_ = 10000;
    bool enable_prometheus_ = true;
    std::string prometheus_prefix_ = "kvcm_optimizer";
    int32_t io_thread_num_ = 4;
    std::vector<KvcmEventSubscriptionConfig> kvcm_event_subscriptions_;

    using SettingFunction = std::function<bool(const std::string &, OnlineOptimizerServerConfig *)>;
    static std::unordered_map<std::string, SettingFunction> kSettingsMap;
};

} // namespace kv_cache_manager
