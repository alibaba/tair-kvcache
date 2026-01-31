#include "kv_cache_manager/common/request_context.h"

#include <string>
#include <utility>

#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/metrics/metrics_collector.h"

namespace kv_cache_manager {

RequestContext::RequestContext(const std::string &trace_id) : RequestContext(trace_id, nullptr) {}

RequestContext::RequestContext(const std::string &trace_id, std::shared_ptr<MetricsCollector> metrics_collector)
    : trace_id_(trace_id), metrics_collector_(std::move(metrics_collector)) {
    request_id_ = trace_id + "_" + std::to_string(TimestampUtil::GetCurrentTimeUs());
    request_begin_time_us_ = TimestampUtil::GetCurrentTimeUs();
    query_tracer_.reset(new QueryTracer);
    need_span_tracer_ = StringUtil::EndsWith(trace_id, "__kvcm_need_span_tracer");
    if (need_span_tracer_) {
        root_span_tracer_ = std::make_shared<SpanTracer>(nullptr, trace_id_, trace_id_);
        parent_span_tracer_ = root_span_tracer_.get();
    }
    error_tracer_ = std::make_unique<ErrorTracer>();
    metrics_collectors_vehicle_.Init();
}

std::string RequestContext::EndAndGetSpanTracerDebugStr() const {
    if (root_span_tracer_) {
        return root_span_tracer_->EndAndGetTracerStr();
    }
    return "";
}

} // namespace kv_cache_manager
