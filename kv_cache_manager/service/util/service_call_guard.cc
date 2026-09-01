#include "kv_cache_manager/service/util/service_call_guard.h"

#include <cassert>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/metrics_reporter.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace {

/**
 * @brief Builds the access log entry as a valid JSON string.
 *
 * Purpose:
 *  - This access log is primarily used for tracing and observability.
 *  - It captures request/response details so they can be correlated across the system.
 *
 * The output is guaranteed to be standard JSON.
 * Fields `request` and `response` are embedded as nested JSON objects,
 * so external tools like `jq` can parse, pretty‑print, and query fields directly.
 *
 * Notes:
 *  - Any change to the log format should be made in this class only.
 */
class AccessLogJsonBuilder {
public:
    explicit AccessLogJsonBuilder(const kv_cache_manager::RequestContext *request_context)
        : request_context_(request_context) {}

    std::string Build() const {
        using namespace rapidjson;
        StringBuffer sb;
        Writer<StringBuffer> writer(sb);
        // TODO  show request and response in debug mode
        // TODO Data Masking in request and response maybe
        int64_t cost_us =
            kv_cache_manager::TimestampUtil::GetCurrentTimeUs() - request_context_->request_begin_time_us();

        writer.StartObject();
        writer.Key("request_begin_time");
        writer.String(
            kv_cache_manager::TimestampUtil::FormatTimestampUs(request_context_->request_begin_time_us()).c_str());

        writer.Key("client_ip");
        writer.String(request_context_->client_ip().c_str());

        writer.Key("trace_id");
        writer.String(request_context_->trace_id().c_str());

        writer.Key("request_id");
        writer.String(request_context_->request_id().c_str());

        writer.Key("api_name");
        writer.String(request_context_->api_name().c_str());

        writer.Key("status_code");
        writer.Int(request_context_->status_code());

        writer.Key("request_cost_time_us");
        writer.Int64(cost_us);

        // other fields

        // 嵌套 request json
        {
            Document req_doc;
            req_doc.Parse(request_context_->request_debug().c_str());
            if (req_doc.HasParseError()) {
                req_doc.SetObject();
            }
            writer.Key("request");
            req_doc.Accept(writer);
        }

        // 嵌套 response json
        {
            Document resp_doc;
            resp_doc.Parse(request_context_->response_debug().c_str());
            if (resp_doc.HasParseError()) {
                resp_doc.SetObject();
            }
            writer.Key("response");
            resp_doc.Accept(writer);
        }

        writer.EndObject();
        return sb.GetString();
    }

private:
    const kv_cache_manager::RequestContext *request_context_;
};

} // namespace

namespace kv_cache_manager {

void RecordServiceFinalResult(const std::shared_ptr<MetricsRegistry> &metrics_registry,
                              const RequestContext *request_context) {
    if (!metrics_registry || request_context == nullptr) {
        return;
    }
    try {
        const bool succeeded = request_context->status_code() == RequestContext::kOkStatusCode;
        ++metrics_registry->GetCounter(
            "service.final_result.requests_total",
            {{"api_name", request_context->api_name()}, {"result", succeeded ? "success" : "error"}});
    } catch (...) {
        // Monitoring must never change the service response.
    }
}

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
                                   std::function<void()> response_debug_setter)
    : cache_manager_(cache_manager)
    , request_context_(request_context)
    , metrics_reporter_(metrics_reporter)
    , response_debug_setter_(std::move(response_debug_setter)) {
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
    if (response_debug_setter_) {
        response_debug_setter_();
    }
    if (metrics_reporter_) {
        RecordServiceFinalResult(metrics_reporter_->GetMetricsRegistry(), request_context_);
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
    AccessLogJsonBuilder access_log_json_builder(request_context);
    std::string access_log = access_log_json_builder.Build();
    KVCM_ACCESS_LOG(access_log);
}

} // namespace kv_cache_manager
