#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "kv_cache_manager/service/http_service/auth/token_verifier.h"

namespace kv_cache_manager {

// verifies HTTP Authorization headers carrying a Bearer token (RFC
// 6750) against a fixed list of accepted tokens.  multiple tokens
// allow zero-downtime rotation: deploy with both old and new in the
// list, switch clients, then remove the old one
class StaticBearerTokenVerifier : public TokenVerifier {
public:
    explicit StaticBearerTokenVerifier(std::vector<std::string> accepted_tokens, std::string realm = "kvcm");

    AuthOutcome Verify(std::string_view authz_header) const override;
    std::string Realm() const override { return realm_; }

private:
    std::vector<std::string> tokens_;
    std::string realm_;
};

} // namespace kv_cache_manager
