#include "kv_cache_manager/client/src/internal/async_rpc/async_rpc_client.h"

#include <algorithm>
#include <cctype>

namespace kv_cache_manager::async_rpc {

const char *TransportKindName(TransportKind kind) { return kind == TransportKind::kHttp ? "http" : "grpc"; }

const char *TransportErrorName(TransportError error) {
    switch (error) {
    case TransportError::kNone:
        return "none";
    case TransportError::kConnect:
        return "connect";
    case TransportError::kTimeout:
        return "timeout";
    case TransportError::kDisconnect:
        return "disconnect";
    case TransportError::kEncode:
        return "encode";
    case TransportError::kDecode:
        return "decode";
    case TransportError::kCancelled:
        return "cancelled";
    case TransportError::kNoPermit:
        return "no_permit";
    case TransportError::kUnsupported:
        return "unsupported";
    case TransportError::kOther:
        return "other";
    }
    return "unknown";
}

bool ValidateInsecureEndpoint(const std::string &endpoint, bool expect_http_scheme, std::string *error) {
    if (endpoint.empty()) {
        *error = "endpoint is empty";
        return false;
    }
    std::string lowered = endpoint;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lowered.rfind("https://", 0) == 0) {
        *error = "HTTPS/TLS endpoints are not supported and are never silently downgraded: " + endpoint;
        return false;
    }
    if (expect_http_scheme) {
        if (lowered.rfind("http://", 0) != 0) {
            *error = "HTTP endpoint must start with http://: " + endpoint;
            return false;
        }
        if (endpoint.size() <= 7) {
            *error = "HTTP endpoint has no host: " + endpoint;
            return false;
        }
        if (endpoint.back() == '/') {
            *error = "HTTP endpoint must not end with '/': " + endpoint;
            return false;
        }
        return true;
    }
    if (lowered.find("://") != std::string::npos) {
        *error = "gRPC endpoint must be host:port without a scheme: " + endpoint;
        return false;
    }
    const size_t colon = endpoint.rfind(':');
    if (colon == std::string::npos || colon + 1 >= endpoint.size()) {
        *error = "gRPC endpoint must be host:port: " + endpoint;
        return false;
    }
    for (size_t i = colon + 1; i < endpoint.size(); ++i) {
        if (std::isdigit(static_cast<unsigned char>(endpoint[i])) == 0) {
            *error = "gRPC endpoint port must be numeric: " + endpoint;
            return false;
        }
    }
    return true;
}

} // namespace kv_cache_manager::async_rpc
