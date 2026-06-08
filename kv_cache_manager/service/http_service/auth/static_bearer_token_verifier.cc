#include "kv_cache_manager/service/http_service/auth/static_bearer_token_verifier.h"

#include <utility>

#include "kv_cache_manager/service/http_service/auth/auth_util.h"

namespace kv_cache_manager {

namespace {

// trim ASCII SP/HTAB at both ends
std::string_view TrimOWS(std::string_view sv) {
    while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) {
        sv.remove_prefix(1);
    }
    while (!sv.empty() && (sv.back() == ' ' || sv.back() == '\t')) {
        sv.remove_suffix(1);
    }
    return sv;
}

} // namespace

StaticBearerTokenVerifier::StaticBearerTokenVerifier(std::vector<std::string> accepted_tokens, std::string realm)
    : tokens_(std::move(accepted_tokens)), realm_(std::move(realm)) {}

AuthOutcome StaticBearerTokenVerifier::Verify(std::string_view authz_header) const {
    std::shared_lock<std::shared_mutex> lock(mu_);

    // open mode: an empty accepted-token list means auth is disabled.
    // we still go through the wrapper so it can be flipped on at
    // runtime by SetTokens; until then everything passes
    if (tokens_.empty()) {
        return AuthOutcome::kOk;
    }

    // RFC 7235 §4.2: an absent Authorization header means "no
    // credentials supplied"
    auto h = TrimOWS(authz_header);
    if (h.empty()) {
        return AuthOutcome::kMissingCredentials;
    }

    // scheme (RFC 7235 §2.1): scheme is case-insensitive, followed
    // by 1*SP and a token68
    constexpr std::string_view kScheme = "Bearer";
    if (h.size() < kScheme.size() + 1) {
        return AuthOutcome::kInvalidRequest;
    }
    if (!AuthUtil::ICaseEqualsAscii(h.substr(0, kScheme.size()), kScheme)) {
        return AuthOutcome::kInvalidRequest;
    }
    char sep = h[kScheme.size()];
    if (sep != ' ' && sep != '\t') {
        // adjacent token without separator (e.g. "BearerXYZ") is
        // not a valid Bearer credential
        return AuthOutcome::kInvalidRequest;
    }

    auto rest = h.substr(kScheme.size());
    // skip 1*SP (also tolerate HTAB; some clients use it)
    std::size_t i = 0;
    while (i < rest.size() && (rest[i] == ' ' || rest[i] == '\t')) {
        ++i;
    }
    if (i == 0 || i == rest.size()) {
        return AuthOutcome::kInvalidRequest;
    }
    auto token = rest.substr(i);
    // token68 has no internal whitespace; reject if any
    if (token.find_first_of(" \t") != std::string_view::npos) {
        return AuthOutcome::kInvalidRequest;
    }

    for (const auto &accepted : tokens_) {
        if (AuthUtil::ConstantTimeEquals(token, accepted)) {
            return AuthOutcome::kOk;
        }
    }
    return AuthOutcome::kInvalidToken;
}

void StaticBearerTokenVerifier::SetTokens(std::vector<std::string> new_tokens) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    tokens_ = std::move(new_tokens);
}

std::vector<std::string> StaticBearerTokenVerifier::SnapshotTokens() const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    return tokens_;
}

} // namespace kv_cache_manager
