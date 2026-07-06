#pragma once

#include "kv_cache_manager/metrics/metrics_collector.h"

namespace kv_cache_manager {

class RequestContext;
class OptimizerMetricsReporter;

class OptimizerCallGuard {
public:
    OptimizerCallGuard(RequestContext *request_context, OptimizerMetricsReporter *metrics_reporter);
    ~OptimizerCallGuard();

    OptimizerCallGuard(const OptimizerCallGuard &) = delete;
    OptimizerCallGuard &operator=(const OptimizerCallGuard &) = delete;

private:
    RequestContext *request_context_;
    OptimizerMetricsReporter *metrics_reporter_;
    ChronoScopeGuard query_scope_;
};

} // namespace kv_cache_manager
