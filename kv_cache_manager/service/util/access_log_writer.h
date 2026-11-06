#pragma once

#include <string>

namespace kv_cache_manager {

class RequestContext;

// Builds an access log containing only request metadata, status and elapsed
// time. Request and response payloads are neither read nor materialized here.
class AccessLogWriter {
public:
    static std::string Build(const RequestContext &request_context);
};

} // namespace kv_cache_manager
