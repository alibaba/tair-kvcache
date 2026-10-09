#include "kv_cache_manager/service/util/service_call_guard.h"

#include <cassert>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_reporter.h"
#include "kv_cache_manager/service/util/access_log_writer.h"

namespace kv_cache_manager {

ServiceCallGuard::ServiceCallGuard(CacheManager *cache_manager,
                                   RequestContext *request_context,
                                   MetricsReporter *metrics_reporter)
    : cache_manager_(cache_manager), request_context_(request_context), metrics_reporter_(metrics_reporter) {
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    query_scope_ = KVCM_METRICS_COLLECTOR_CHRONO_SCOPE(service_metrics_collector, ServiceQuery);
}

ServiceCallGuard::ServiceCallGuard(CacheManager *cache_manager,
                                   RequestContext *request_context,
                                   MetricsReporter *metrics_reporter,
                                   std::function<void()> completion_callback)
    : cache_manager_(cache_manager)
    , request_context_(request_context)
    , metrics_reporter_(metrics_reporter)
    , completion_callback_(std::move(completion_callback)) {
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    query_scope_ = KVCM_METRICS_COLLECTOR_CHRONO_SCOPE(service_metrics_collector, ServiceQuery);
}

ServiceCallGuard::~ServiceCallGuard() {
    assert(cache_manager_);
    assert(request_context_);
    // Member destructors run after this function body, so we explicitly end the
    // scope here to flush query_rt_us into the Gauge; otherwise the subsequent
    // ReportPerQuery would read the stale value from the previous request.
    query_scope_ = ChronoScopeGuard{};
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context_->metrics_collector());
    const auto extra_collectors = request_context_->GetMetricsCollectorsVehicle().GetMetricsCollectors();
    const double request_rt_us =
        static_cast<double>(TimestampUtil::GetCurrentTimeUs() - request_context_->request_begin_time_us());
    // MetricsReporter treats any non-zero error_code sample as a failed
    // request, so this gauge is deliberately a 0/1 failure flag rather than
    // the wire enum value (where a successful request is non-zero).
    const double error_code = request_context_->status_code() == RequestContext::kOkStatusCode ? 0.0 : 1.0;
    // The response must be finalized before materialization. Doing this once
    // here lets the access log and the HTTP transport share the same bytes.
    request_context_->MaterializeResponseJson();
    if (completion_callback_) {
        completion_callback_();
    }
    if (metrics_reporter_) {
        metrics_reporter_->ReportPerQuery(service_metrics_collector);
        for (const auto &mc : extra_collectors) {
            if (auto *event_metrics_collector = dynamic_cast<EventReportMetricsCollector *>(mc.get())) {
                // The master ServiceMetricsCollector is cached per instance and
                // its gauges can be overwritten by concurrent requests. Source
                // event samples from this request's private context instead,
                // and write immediately before reporting to minimize the shared
                // registry gauge race window.
                SET_METRICS_(event_metrics_collector, service, query_rt_us, request_rt_us);
                SET_METRICS_(event_metrics_collector, service, error_code, error_code);
                event_metrics_collector->SetRequestSample(request_rt_us, error_code);
                if (event_metrics_collector->HasRequestKeyCountSample()) {
                    SET_METRICS_(event_metrics_collector,
                                 manager,
                                 request_key_count,
                                 event_metrics_collector->GetRequestKeyCountSample());
                }
            }
            metrics_reporter_->ReportPerQuery(mc.get());
        }
    }
    PrintAccessLog(request_context_);
}

void ServiceCallGuard::PrintAccessLog(RequestContext *request_context) {
    std::string access_log = AccessLogWriter::Build(*request_context);
    KVCM_ACCESS_LOG(access_log);
}

} // namespace kv_cache_manager
