#pragma once

#include <istream>
#include <map>
#include <string>

namespace kv_cache_manager {

struct KmonParam {
public:
    KmonParam();

    bool Init();

    // An empty result delegates the default host and port to the KMonitor SDK.
    std::string ResolveSinkAddress() const;
    // sdk_host_name must be resolvable; host_info_ip must be a validated IPv4 address.
    std::string GetSinkAddress(const std::string &sdk_host_name, const std::string &host_info_ip = "") const;
    static std::string ParseHostInfoIp(std::istream &input);

private:
    static bool ParseKmonitorTags(const std::string &tags_str, std::map<std::string, std::string> &tags_map);
    // The default IP tag is not an explicitly configured receiver.
    bool has_hippo_slave_ip_ = false;

public:
    std::string hippo_slave_ip;
    // kmonitor
    std::string kmonitor_port;
    std::string kmonitor_service_name;
    std::string kmonitor_sink_address;
    bool kmonitor_enable_log_file_sink = false;
    bool kmonitor_enable_prometheus_sink = false;
    bool kmonitor_manually_mode = false;
    std::string kmonitor_tenant;
    std::string kmonitor_metrics_prefix;
    std::string kmonitor_metrics_reporter_cache_limit;
    std::map<std::string, std::string> kmonitor_tags;
    int kmonitor_normal_sample_period = 0;
};

} // namespace kv_cache_manager
