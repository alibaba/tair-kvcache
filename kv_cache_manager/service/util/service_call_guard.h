#pragma once

#include <functional>

#include "kv_cache_manager/metrics/metrics_collector.h"

namespace kv_cache_manager {

class CacheManager;
class RequestContext;
class MetricsReporter;

class ServiceCallGuard {
public:
    ServiceCallGuard(CacheManager *cache_manager, RequestContext *request_context, MetricsReporter *metrics_reporter);
    ServiceCallGuard(CacheManager *cache_manager,
                     RequestContext *request_context,
                     MetricsReporter *metrics_reporter,
                     std::function<void()> response_debug_setter);

    ~ServiceCallGuard();
    void PrintAccessLog(RequestContext *request_context);

private:
    CacheManager *cache_manager_;
    RequestContext *request_context_;
    MetricsReporter *metrics_reporter_;
    std::function<void()> response_debug_setter_;
    // Records per-request begin timestamp; writes (now - begin) to
    // service.query_rt_us when the scope ends.  Explicitly reset in
    // ~ServiceCallGuard() before ReportPerQuery, so declaration order
    // does not matter.
    ChronoScopeGuard query_scope_;
};

} // namespace kv_cache_manager
