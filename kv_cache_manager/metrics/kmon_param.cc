#include "kv_cache_manager/metrics/kmon_param.h"

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
    /*** for kmon ***/
    kmonitor_port = EnvUtil::GetEnv(kKmonitorPort, "4141");
    kmonitor_service_name = EnvUtil::GetEnv(kKmonitorServiceName, "kvcm_service");
    kmonitor_sink_address = EnvUtil::GetEnv(kKmonitorSinkAddress, EnvUtil::GetEnv(kHippoSlaveIp, "127.0.0.1"));
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
