#pragma once

#include <string>
#include <string_view>

namespace kv_cache_manager {

// outcome of an authorization attempt; mapped to RFC 6750 §3.1
// WWW-Authenticate `error` parameter values
enum class AuthOutcome {
    kOk,
    kMissingCredentials, // no Authorization header present
    kInvalidRequest,     // header malformed or scheme not Bearer
    kInvalidToken,       // scheme is Bearer but token not accepted
};

class TokenVerifier {
public:
    virtual ~TokenVerifier() = default;

    // verify the raw value of an HTTP Authorization header
    // (may be empty if the client sent no header)
    virtual AuthOutcome Verify(std::string_view authz_header) const = 0;

    // realm advertised in the WWW-Authenticate response header
    virtual std::string Realm() const { return "kvcm"; }
};

} // namespace kv_cache_manager
