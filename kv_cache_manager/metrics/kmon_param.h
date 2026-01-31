#pragma once

#include <map>
#include <string>

namespace kv_cache_manager {

struct KmonParam {
public:
    KmonParam();

    bool Init();

private:
    static bool ParseKmonitorTags(const std::string &tags_str, std::map<std::string, std::string> &tags_map);

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
