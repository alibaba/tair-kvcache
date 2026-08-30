#include "kv_cache_manager/common/request_context.h"

#include <string>
#include <utility>

#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/metrics/metrics_collector.h"

namespace kv_cache_manager {

void RequestContext::LazyResponseJsonCache::SetGenerator(ResponseJsonGenerator generator, ResponseJsonKind kind) {
    fragment_ = {};
    generator_ = std::move(generator);
    materialized_ = false;
    kind_ = kind;
}

const RequestContext::JsonFragment &RequestContext::LazyResponseJsonCache::Get() const {
    if (!materialized_) {
        fragment_ = generator_ ? generator_() : JsonFragment{};
        generator_ = {};
        materialized_ = true;
    }
    return fragment_;
}

std::optional<std::string> RequestContext::LazyResponseJsonCache::TakeIfReusable() {
    Get();
    if (kind_ != ResponseJsonKind::kFullMessage || !fragment_.valid || fragment_.json.empty()) {
        return std::nullopt;
    }
    return std::move(fragment_.json);
}

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

void RequestContext::MaterializeResponseJson() const { response_debug_json_.Get(); }

std::optional<std::string> RequestContext::TakeReusableResponseJson() { return response_debug_json_.TakeIfReusable(); }

} // namespace kv_cache_manager
