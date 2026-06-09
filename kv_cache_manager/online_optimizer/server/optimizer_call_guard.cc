#include "kv_cache_manager/online_optimizer/server/optimizer_call_guard.h"

#include <string>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/online_optimizer/metrics/optimizer_metrics_collector.h"
#include "kv_cache_manager/online_optimizer/metrics/optimizer_metrics_reporter.h"

namespace kv_cache_manager {

OptimizerCallGuard::OptimizerCallGuard(RequestContext *request_context,
                                       OptimizerMetricsReporter *metrics_reporter)
    : request_context_(request_context), metrics_reporter_(metrics_reporter) {
    auto *collector = dynamic_cast<OptimizerServiceMetricsCollector *>(request_context->metrics_collector());
    if (collector && !request_context->client_ip().empty()) {
        collector->set_client_ip(request_context->client_ip());
    }
    query_scope_ = KVCM_METRICS_COLLECTOR_CHRONO_SCOPE(collector, ServiceQuery);
}

OptimizerCallGuard::~OptimizerCallGuard() {
    query_scope_ = ChronoScopeGuard{};

    if (metrics_reporter_ && request_context_) {
        metrics_reporter_->ReportPerQuery(request_context_->metrics_collector());
    }

    if (request_context_) {
        int64_t cost_us = TimestampUtil::GetCurrentTimeUs() - request_context_->request_begin_time_us();
        std::string log = "{\"api_name\":\"" + request_context_->api_name() +
                          "\",\"trace_id\":\"" + request_context_->trace_id() +
                          "\",\"request_id\":\"" + request_context_->request_id() +
                          "\",\"client_ip\":\"" + request_context_->client_ip() +
                          "\",\"status_code\":" + std::to_string(request_context_->status_code()) +
                          ",\"cost_us\":" + std::to_string(cost_us) + "}";
        KVCM_ACCESS_LOG(log);
    }
}

} // namespace kv_cache_manager
