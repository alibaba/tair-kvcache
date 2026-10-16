#include "kv_cache_manager/metrics/kmon_param.h"

#include <arpa/inet.h>
#include <fstream>

#include "kmonitor/client/common/Util.h"
#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"

namespace kv_cache_manager {

namespace {
static const std::string kKmonitorPort = "kmonitorPort";
static const std::string kKmonitorSinkAddress = "kmonitorSinkAddress";
static const std::string kKmonitorEnableLogFileSink = "kmonitorEnableLogFileSink";
static const std::string kKmonitorEnablePrometheusSink = "kmonitorEnablePrometheusSink";
static const std::string kKmonitorManuallyMode = "kmonitorManuallyMode";
static const std::string kKmonitorServiceName = "kmonitorServiceName";
static const std::string kKmonitorTenant = "kmonitorTenant";
static const std::string kKmonitorMetricsPrefix = "kmonitorMetricsPrefix";
static const std::string kKmonitorGlobalTableMetricsPrefix = "kmonitorGlobalTableMetricsPrefix";
static const std::string kKmonitorTableMetricsPrefix = "kmonitorTableMetricsPrefix";
static const std::string kKmonitorMetricsReporterCacheLimit = "kmonitorMetricsReporterCacheLimit";
static const std::string kKmonitorTags = "kmonitorTags";
static const std::string kKmonitorNormalSamplePeriod = "kmonitorNormalSamplePeriod";
static const std::string kKmonitorKeyvalueSep = "^";
static const std::string kKmonitorMultiSep = "@";
static const std::string kKmonitorTableNameWhitelist = "kmonitorTableNameWhiteList";

static const std::string kServiceName = "serviceName";
static const std::string kAmonitorPath = "amonitorPath";
static const std::string kPartId = "partId";
static const std::string kHippoSlaveIp = "HIPPO_SLAVE_IP";
static const std::string kRoleType = "roleType";
} // namespace

KmonParam::KmonParam()
    : kmonitor_enable_log_file_sink(false), kmonitor_manually_mode(false), kmonitor_normal_sample_period(1) {}

bool KmonParam::Init() {
    hippo_slave_ip = EnvUtil::GetEnv(kHippoSlaveIp, "127.0.0.1");
    has_hippo_slave_ip_ = !EnvUtil::GetEnv(kHippoSlaveIp, "").empty();
    /*** for kmon ***/
    // An explicitly empty port preserves a preformatted sink address.
    kmonitor_port = EnvUtil::GetEnv(kKmonitorPort, "4141");
    kmonitor_service_name = EnvUtil::GetEnv(kKmonitorServiceName, "kvcm_service");
    kmonitor_sink_address = EnvUtil::GetEnv(kKmonitorSinkAddress, "");
    kmonitor_enable_log_file_sink = EnvUtil::GetEnv(kKmonitorEnableLogFileSink, kmonitor_enable_log_file_sink);
    kmonitor_enable_prometheus_sink = EnvUtil::GetEnv(kKmonitorEnablePrometheusSink, kmonitor_enable_prometheus_sink);
    kmonitor_manually_mode = EnvUtil::GetEnv(kKmonitorManuallyMode, kmonitor_manually_mode);
    kmonitor_tenant = EnvUtil::GetEnv(kKmonitorTenant, "default");
    kmonitor_metrics_prefix = EnvUtil::GetEnv(kKmonitorMetricsPrefix, "kvcm");
    kmonitor_metrics_reporter_cache_limit = EnvUtil::GetEnv(kKmonitorMetricsReporterCacheLimit, "");
    std::string kmonitor_tags_str = EnvUtil::GetEnv(kKmonitorTags, "");
    if (!kmonitor_tags_str.empty() && !ParseKmonitorTags(kmonitor_tags_str, kmonitor_tags)) {
        return false;
    }
    kmonitor_normal_sample_period = EnvUtil::GetEnv(kKmonitorNormalSamplePeriod, 1);

    return true;
}

std::string KmonParam::ResolveSinkAddress() const {
    if (!kmonitor_sink_address.empty() || has_hippo_slave_ip_) {
        return GetSinkAddress("");
    }
    const auto sdk_host_name = kmonitor::Util::GetMachineName();
    std::string resolved_ip;
    if (!sdk_host_name.empty() && kmonitor::Util::HostnameToIp(sdk_host_name, resolved_ip)) {
        return GetSinkAddress(sdk_host_name);
    }
    std::ifstream host_info("/etc/hostinfo");
    const auto sink_address = GetSinkAddress("", ParseHostInfoIp(host_info));
    KVCM_LOG_WARN("KMonitor SDK agent host [%s] is unavailable or cannot be resolved; using [%s] instead.",
                  sdk_host_name.c_str(),
                  sink_address.c_str());
    return sink_address;
}

std::string KmonParam::GetSinkAddress(const std::string &sdk_host_name, const std::string &host_info_ip) const {
    if (!kmonitor_sink_address.empty()) {
        return kmonitor_port.empty() ? kmonitor_sink_address : kmonitor_sink_address + ":" + kmonitor_port;
    }
    if (has_hippo_slave_ip_) {
        return kmonitor_port.empty() ? hippo_slave_ip : hippo_slave_ip + ":" + kmonitor_port;
    }
    const auto port = kmonitor_port.empty() ? "4141" : kmonitor_port;
    if (!sdk_host_name.empty()) {
        // The SDK discovers the host through /etc/hostinfo and uses port 4141.
        // Keep the complete address empty to enable that path.
        return port == "4141" ? "" : sdk_host_name + ":" + port;
    }
    if (!host_info_ip.empty()) {
        return host_info_ip + ":" + port;
    }
    return "127.0.0.1:" + port;
}

std::string KmonParam::ParseHostInfoIp(std::istream &input) {
    std::string line;
    int nonempty_lines = 0;
    while (std::getline(input, line)) {
        StringUtil::Trim(line);
        if (line.empty()) {
            continue;
        }
        if (++nonempty_lines == 2) {
            in_addr address{};
            return inet_pton(AF_INET, line.c_str(), &address) == 1 ? line : "";
        }
    }
    return "";
}

bool KmonParam::ParseKmonitorTags(const std::string &tags_str, std::map<std::string, std::string> &tags_map) {
    auto tag_vec = StringUtil::Split(tags_str, kKmonitorMultiSep);
    for (const auto &tags : tag_vec) {
        auto kv_vec = StringUtil::Split(tags, kKmonitorKeyvalueSep);
        if (kv_vec.size() != 2) {
            KVCM_LOG_ERROR("parse kmonitor tags [%s] failed.", tags.c_str());
            return false;
        }
        StringUtil::Trim(kv_vec[0]);
        StringUtil::Trim(kv_vec[1]);
        tags_map[kv_vec[0]] = kv_vec[1];
    }
    return true;
}

} // namespace kv_cache_manager
