#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/service/http_service/auth/static_bearer_token_verifier.h"

using namespace kv_cache_manager;

class StaticBearerTokenVerifierTest : public TESTBASE {};

TEST_F(StaticBearerTokenVerifierTest, MissingHeaderIsMissingCredentials) {
    StaticBearerTokenVerifier v({"secret"});
    ASSERT_EQ(AuthOutcome::kMissingCredentials, v.Verify(""));
    // OWS-only header still counts as missing
    ASSERT_EQ(AuthOutcome::kMissingCredentials, v.Verify("   "));
    ASSERT_EQ(AuthOutcome::kMissingCredentials, v.Verify("\t"));
}

TEST_F(StaticBearerTokenVerifierTest, NonBearerSchemeIsInvalidRequest) {
    StaticBearerTokenVerifier v({"secret"});
    ASSERT_EQ(AuthOutcome::kInvalidRequest, v.Verify("Basic dXNlcjpwYXNz"));
    ASSERT_EQ(AuthOutcome::kInvalidRequest, v.Verify("Digest realm=x"));
}

TEST_F(StaticBearerTokenVerifierTest, MalformedHeaderIsInvalidRequest) {
    StaticBearerTokenVerifier v({"secret"});
    // missing space between scheme and token
    ASSERT_EQ(AuthOutcome::kInvalidRequest, v.Verify("Bearersecret"));
    // scheme only, no token
    ASSERT_EQ(AuthOutcome::kInvalidRequest, v.Verify("Bearer"));
    ASSERT_EQ(AuthOutcome::kInvalidRequest, v.Verify("Bearer "));
    ASSERT_EQ(AuthOutcome::kInvalidRequest, v.Verify("Bearer    "));
    // internal whitespace inside the token portion
    ASSERT_EQ(AuthOutcome::kInvalidRequest, v.Verify("Bearer abc def"));
}

TEST_F(StaticBearerTokenVerifierTest, WrongTokenIsInvalidToken) {
    StaticBearerTokenVerifier v({"secret"});
    ASSERT_EQ(AuthOutcome::kInvalidToken, v.Verify("Bearer wrong"));
    ASSERT_EQ(AuthOutcome::kInvalidToken, v.Verify("Bearer SECRET")); // tokens are case-sensitive
}

TEST_F(StaticBearerTokenVerifierTest, AcceptsValidToken) {
    StaticBearerTokenVerifier v({"secret"});
    ASSERT_EQ(AuthOutcome::kOk, v.Verify("Bearer secret"));
    // scheme is case-insensitive (RFC 7235 §2.1)
    ASSERT_EQ(AuthOutcome::kOk, v.Verify("bearer secret"));
    ASSERT_EQ(AuthOutcome::kOk, v.Verify("BEARER secret"));
    ASSERT_EQ(AuthOutcome::kOk, v.Verify("BeArEr secret"));
    // multiple SP between scheme and token is allowed
    ASSERT_EQ(AuthOutcome::kOk, v.Verify("Bearer    secret"));
    // HTAB separator tolerated
    ASSERT_EQ(AuthOutcome::kOk, v.Verify("Bearer\tsecret"));
    // surrounding OWS tolerated (some proxies add them)
    ASSERT_EQ(AuthOutcome::kOk, v.Verify("  Bearer secret  "));
}

TEST_F(StaticBearerTokenVerifierTest, MultiTokenRotation) {
    StaticBearerTokenVerifier v({"old-token", "new-token"});
    ASSERT_EQ(AuthOutcome::kOk, v.Verify("Bearer old-token"));
    ASSERT_EQ(AuthOutcome::kOk, v.Verify("Bearer new-token"));
    ASSERT_EQ(AuthOutcome::kInvalidToken, v.Verify("Bearer other"));
}

TEST_F(StaticBearerTokenVerifierTest, EmptyAcceptedListRejectsEverything) {
    StaticBearerTokenVerifier v({});
    ASSERT_EQ(AuthOutcome::kInvalidToken, v.Verify("Bearer anything"));
    ASSERT_EQ(AuthOutcome::kMissingCredentials, v.Verify(""));
}

TEST_F(StaticBearerTokenVerifierTest, RealmDefaultsAndOverride) {
    StaticBearerTokenVerifier v_default({"x"});
    ASSERT_EQ("kvcm", v_default.Realm());

    StaticBearerTokenVerifier v_custom({"x"}, "admin-api");
    ASSERT_EQ("admin-api", v_custom.Realm());
}
