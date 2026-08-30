#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "kv_cache_manager/common/tracer.h"
#include "kv_cache_manager/metrics/metrics_collector.h"

#define SPAN_TRACER(request_context_pointer)                                                                           \
    std::unique_ptr<SpanTracer> span_tracer;                                                                           \
    std::unique_ptr<SpanTracerHelper> span_tracer_helper;                                                              \
    if (request_context_pointer && request_context_pointer->need_span_tracer()) {                                      \
        static const char *span_tracer_file = __FILE__;                                                                \
        static const char *span_tracer_func = __func__;                                                                \
        SpanTracer *old_parent_tracer = request_context_pointer->parent_span_tracer();                                 \
        span_tracer = std::make_unique<SpanTracer>(old_parent_tracer, span_tracer_file, span_tracer_func);             \
        request_context_pointer->set_parent_span_tracer(span_tracer.get());                                            \
        span_tracer_helper = std::make_unique<SpanTracerHelper>(request_context_pointer, old_parent_tracer);           \
    }

namespace kv_cache_manager {

class RequestContext : std::enable_shared_from_this<RequestContext> {
public:
    struct JsonFragment {
        std::string json;
        // True only when the producer completed a full JSON object.
        bool valid = false;
    };

    // The response generator is request-local and may capture the response
    // protobuf object. It is evaluated at most once before that object leaves
    // the synchronous service call.
    using ResponseJsonGenerator = std::function<JsonFragment()>;

    enum class ResponseJsonKind {
        // Summaries are valid access-log fragments but not the full wire body.
        kAccessLogSummary,
        // The fragment is the canonical full protobuf JSON.
        kFullMessage,
    };

    // All public KVCM service protocols use numeric value 1 for their OK enum.
    // Keep the convention here so request-local observability does not mistake
    // proto3's default UNSPECIFIED=0 for success.
    static constexpr int kOkStatusCode = 1;

    RequestContext() = delete;
    explicit RequestContext(const std::string &trace_id);
    RequestContext(const std::string &trace_id, std::shared_ptr<MetricsCollector> metrics_collector);
    ~RequestContext() = default;

public:
    MetricsCollector *metrics_collector() { return metrics_collector_.get(); }
    const MetricsCollectors &GetMetricsCollectorsVehicle() const { return metrics_collectors_vehicle_; }
    const std::string &trace_id() const { return trace_id_; }
    const std::string &request_id() const { return request_id_; }
    const int64_t request_begin_time_us() const { return request_begin_time_us_; }
    const std::string &api_name() const { return api_name_; }
    const std::string &client_ip() const { return client_ip_; }
    const int status_code() const { return status_code_; }
    const JsonFragment &request_debug_json() const { return request_debug_json_; }
    const JsonFragment &response_debug_json() const { return response_debug_json_.Get(); }
    bool need_span_tracer() const { return need_span_tracer_; }
    SpanTracer *parent_span_tracer() const { return parent_span_tracer_; }
    std::string EndAndGetSpanTracerDebugStr() const;
    void set_api_name(const std::string &value) { api_name_ = value; }
    void set_client_ip(const std::string &value) { client_ip_ = value; }
    void set_status_code(int value) { status_code_ = value; }
    void set_request_debug_json(JsonFragment fragment) { request_debug_json_ = std::move(fragment); }
    void set_response_debug_json_generator(ResponseJsonGenerator generator,
                                           ResponseJsonKind kind = ResponseJsonKind::kAccessLogSummary) {
        response_debug_json_.SetGenerator(std::move(generator), kind);
    }
    void MaterializeResponseJson() const;
    // Must be called only after ServiceCallGuard has written the access log.
    std::optional<std::string> TakeReusableResponseJson();
    void set_parent_span_tracer(SpanTracer *tracer) const { parent_span_tracer_ = tracer; }
    ErrorTracer *error_tracer() { return error_tracer_.get(); }

private:
    class LazyResponseJsonCache {
    public:
        void SetGenerator(ResponseJsonGenerator generator, ResponseJsonKind kind);
        const JsonFragment &Get() const;
        std::optional<std::string> TakeIfReusable();

    private:
        mutable JsonFragment fragment_;
        mutable ResponseJsonGenerator generator_;
        mutable bool materialized_ = true;
        ResponseJsonKind kind_ = ResponseJsonKind::kAccessLogSummary;
    };

    bool need_span_tracer_ = false;
    std::string trace_id_;   // 用户传递的trace_id
    std::string request_id_; // 为每一次请求生成的request_id
    int64_t request_begin_time_us_;
    std::string api_name_; // 调用的接口名称
    std::string client_ip_;
    int status_code_{0};
    JsonFragment request_debug_json_;
    LazyResponseJsonCache response_debug_json_;
    const std::shared_ptr<MetricsCollector> metrics_collector_; // the master metrics collector for this request context
    MetricsCollectors metrics_collectors_vehicle_;              // for carrying other metrics collectors for reporting
    std::shared_ptr<QueryTracer> query_tracer_;                 // 和LOG结合，可以把LOG输出给用户
    std::unique_ptr<ErrorTracer> error_tracer_;
    std::shared_ptr<SpanTracer> root_span_tracer_;
    mutable SpanTracer *parent_span_tracer_ = nullptr;
};

class SpanTracerHelper {
public:
    SpanTracerHelper(RequestContext const *request_context, SpanTracer *old_parent_tracer)
        : request_context_(request_context), old_parent_tracer_(old_parent_tracer) {}
    ~SpanTracerHelper() {
        assert(request_context_);
        assert(old_parent_tracer_);
        request_context_->set_parent_span_tracer(old_parent_tracer_);
    }

private:
    RequestContext const *request_context_;
    SpanTracer *old_parent_tracer_;
};

}; // namespace kv_cache_manager
