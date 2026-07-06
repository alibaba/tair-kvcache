#include "kv_cache_manager/optimizer/service/optimizer_call_guard.h"

#include <string>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/optimizer/service/metrics/optimizer_metrics_collector.h"
#include "kv_cache_manager/optimizer/service/metrics/optimizer_metrics_reporter.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace kv_cache_manager {

OptimizerCallGuard::OptimizerCallGuard(RequestContext *request_context, OptimizerMetricsReporter *metrics_reporter)
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
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        writer.StartObject();
        writer.Key("api_name");
        writer.String(request_context_->api_name());
        writer.Key("trace_id");
        writer.String(request_context_->trace_id());
        writer.Key("request_id");
        writer.String(request_context_->request_id());
        writer.Key("client_ip");
        writer.String(request_context_->client_ip());
        writer.Key("status_code");
        writer.Int(request_context_->status_code());
        writer.Key("cost_us");
        writer.Int64(cost_us);
        writer.EndObject();
        KVCM_ACCESS_LOG(buffer.GetString());
    }
}

} // namespace kv_cache_manager
