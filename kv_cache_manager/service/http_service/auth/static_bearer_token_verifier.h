#pragma once

#include <shared_mutex>
#include <string>
#include <string_view>
#include <vector>

#include "kv_cache_manager/service/http_service/auth/token_verifier.h"

namespace kv_cache_manager {

// verifies HTTP Authorization headers carrying a Bearer token (RFC
// 6750) against an in-memory list of accepted tokens.  the list can
// be replaced at runtime via SetTokens, allowing online rotation and
// the open <-> enforcing transition.  multiple tokens allow zero-
// downtime rotation: install both old and new, switch clients, then
// remove the old one
//
// thread-safety: Verify and SnapshotTokens take a shared lock;
// SetTokens takes an exclusive lock.  contention is negligible
// because admin/debug QPS is low and writers are operator-driven
class StaticBearerTokenVerifier : public TokenVerifier {
public:
    explicit StaticBearerTokenVerifier(std::vector<std::string> accepted_tokens, std::string realm = "kvcm");

    AuthOutcome Verify(std::string_view authz_header) const override;
    std::string Realm() const override { return realm_; }

    // replace the accepted-token list atomically.  empty list means
    // "open mode" — Verify will accept everything, including absent
    // Authorization headers
    void SetTokens(std::vector<std::string> new_tokens);

    // copy of the current accepted-token list, taken under a shared
    // lock.  used by callers that need to inspect or read-modify-
    // write the list (e.g. RotateAdminAuthToken)
    std::vector<std::string> SnapshotTokens() const;

private:
    mutable std::shared_mutex mu_;
    std::vector<std::string> tokens_;
    std::string realm_;
};

} // namespace kv_cache_manager
