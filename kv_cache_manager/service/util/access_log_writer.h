#pragma once

#include <string>

namespace kv_cache_manager {

class RequestContext;

// Builds the outer access-log object. Request and response fragments are
// produced exclusively by KVCM serializers and are embedded without reparsing.
class AccessLogWriter {
public:
    static std::string Build(const RequestContext &request_context);
};

} // namespace kv_cache_manager
