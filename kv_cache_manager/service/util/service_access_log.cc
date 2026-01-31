#include "kv_cache_manager/service/util/service_access_log.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"

namespace kv_cache_manager {

ServiceAccessLog::ServiceAccessLog(RequestContext *request_context) : request_context_(request_context) {}

ServiceAccessLog::~ServiceAccessLog() {
    // TODO  show request and response in debug mode
    // TODO Data Masking in request and response maybe
    std::string begin_time_str = TimestampUtil::FormatTimestampUs(request_context_->request_begin_time_us());
    auto cost_us = TimestampUtil::GetCurrentTimeUs() - request_context_->request_begin_time_us();
    std::string json_log = "{"
                           "\"request_begin_time\":\"" +
                           begin_time_str +
                           "\","
                           "\"client_ip\":\"" +
                           request_context_->client_ip() +
                           "\","
                           "\"trace_id\":\"" +
                           request_context_->trace_id() +
                           "\","
                           "\"request_id\":\"" +
                           request_context_->request_id() +
                           "\","
                           "\"api_name\":\"" +
                           request_context_->api_name() +
                           "\","
                           "\"status_code\":" +
                           std::to_string(request_context_->status_code()) +
                           ","
                           "\"request_cost_time_us\":" +
                           std::to_string(cost_us) +
                           ","
                           "\"request\":\"" +
                           request_context_->request_debug() +
                           "\","
                           "\"response\":\"" +
                           request_context_->response_debug() +
                           "\""
                           "}";
    KVCM_ACCESS_LOG(json_log);
}

} // namespace kv_cache_manager
