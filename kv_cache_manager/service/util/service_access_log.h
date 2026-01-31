#pragma once

namespace kv_cache_manager {

class RequestContext;

class ServiceAccessLog {
public:
    ServiceAccessLog(RequestContext *request_context);
    ~ServiceAccessLog();

private:
    RequestContext *request_context_;
};

} // namespace kv_cache_manager