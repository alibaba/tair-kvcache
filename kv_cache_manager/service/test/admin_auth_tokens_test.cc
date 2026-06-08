#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/startup_config_loader.h"
#include "kv_cache_manager/metrics/dummy_metrics_reporter.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/admin_service.pb.h"
#include "kv_cache_manager/service/admin_service_impl.h"
#include "kv_cache_manager/service/http_service/auth/static_bearer_token_verifier.h"

using namespace kv_cache_manager;

class AdminAuthTokensTest : public TESTBASE {
public:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("", metrics_registry_);
        ASSERT_TRUE(registry_manager_->Init());
        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        ASSERT_TRUE(cache_manager_->Init());
        StartupConfigLoader loader;
        loader.Init(registry_manager_);
        loader.Load("");
        metrics_reporter_ = std::make_shared<DummyMetricsReporter>();
        metrics_reporter_->Init(cache_manager_, metrics_registry_, "");
        admin_impl_ = std::make_shared<AdminServiceImpl>(
            cache_manager_, metrics_reporter_, metrics_registry_, registry_manager_, /*leader_elector=*/nullptr);
    }

    void TearDown() override {
        admin_impl_.reset();
        metrics_reporter_.reset();
        cache_manager_.reset();
        registry_manager_.reset();
        metrics_registry_.reset();
    }

    static std::shared_ptr<RequestContext> MakeContext(const std::string &trace_id) {
        return std::make_shared<RequestContext>(trace_id);
    }

protected:
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<MetricsReporter> metrics_reporter_;
    std::shared_ptr<AdminServiceImpl> admin_impl_;
};

TEST_F(AdminAuthTokensTest, SetAdminAuthTokens_NoVerifier_Unsupported) {
    // verifier intentionally not injected
    proto::admin::SetAdminAuthTokensRequest req;
    req.set_trace_id("t-no-verifier");
    req.add_tokens("anything");
    proto::admin::CommonResponse resp;
    auto ctx = MakeContext(req.trace_id());

    admin_impl_->SetAdminAuthTokens(ctx.get(), &req, &resp);
    ASSERT_EQ(proto::admin::UNSUPPORTED, resp.header().status().code());
}

TEST_F(AdminAuthTokensTest, SetAdminAuthTokens_AppliesAndDropsEmpty) {
    auto verifier = std::make_shared<StaticBearerTokenVerifier>(std::vector<std::string>{});
    admin_impl_->SetTokenVerifier(verifier);

    proto::admin::SetAdminAuthTokensRequest req;
    req.set_trace_id("t-set-1");
    req.add_tokens("alpha");
    req.add_tokens("");      // dropped
    req.add_tokens("beta");
    proto::admin::CommonResponse resp;
    auto ctx = MakeContext(req.trace_id());

    admin_impl_->SetAdminAuthTokens(ctx.get(), &req, &resp);
    ASSERT_EQ(proto::admin::OK, resp.header().status().code());

    auto snap = verifier->SnapshotTokens();
    ASSERT_EQ(2u, snap.size());
    ASSERT_EQ("alpha", snap[0]);
    ASSERT_EQ("beta", snap[1]);
    // verify enforces with the new tokens
    ASSERT_EQ(AuthOutcome::kOk, verifier->Verify("Bearer alpha"));
    ASSERT_EQ(AuthOutcome::kInvalidToken, verifier->Verify("Bearer gamma"));
}

TEST_F(AdminAuthTokensTest, SetAdminAuthTokens_EmptyListReturnsToOpenMode) {
    auto verifier = std::make_shared<StaticBearerTokenVerifier>(std::vector<std::string>{"secret"});
    admin_impl_->SetTokenVerifier(verifier);
    ASSERT_EQ(AuthOutcome::kMissingCredentials, verifier->Verify(""));

    proto::admin::SetAdminAuthTokensRequest req;
    req.set_trace_id("t-set-empty");
    proto::admin::CommonResponse resp;
    auto ctx = MakeContext(req.trace_id());

    admin_impl_->SetAdminAuthTokens(ctx.get(), &req, &resp);
    ASSERT_EQ(proto::admin::OK, resp.header().status().code());
    ASSERT_TRUE(verifier->SnapshotTokens().empty());
    // open mode: every request passes
    ASSERT_EQ(AuthOutcome::kOk, verifier->Verify(""));
    ASSERT_EQ(AuthOutcome::kOk, verifier->Verify("Bearer anything"));
}

TEST_F(AdminAuthTokensTest, RotateAdminAuthToken_AddsNew) {
    auto verifier = std::make_shared<StaticBearerTokenVerifier>(std::vector<std::string>{"old"});
    admin_impl_->SetTokenVerifier(verifier);

    proto::admin::RotateAdminAuthTokenRequest req;
    req.set_trace_id("t-rot-add");
    req.set_new_token("new");
    proto::admin::CommonResponse resp;
    auto ctx = MakeContext(req.trace_id());

    admin_impl_->RotateAdminAuthToken(ctx.get(), &req, &resp);
    ASSERT_EQ(proto::admin::OK, resp.header().status().code());

    auto snap = verifier->SnapshotTokens();
    ASSERT_EQ(2u, snap.size());
    ASSERT_EQ("old", snap[0]);
    ASSERT_EQ("new", snap[1]);
    ASSERT_EQ(AuthOutcome::kOk, verifier->Verify("Bearer old"));
    ASSERT_EQ(AuthOutcome::kOk, verifier->Verify("Bearer new"));
}

TEST_F(AdminAuthTokensTest, RotateAdminAuthToken_SwapsOldForNew) {
    auto verifier = std::make_shared<StaticBearerTokenVerifier>(std::vector<std::string>{"old", "keep"});
    admin_impl_->SetTokenVerifier(verifier);

    proto::admin::RotateAdminAuthTokenRequest req;
    req.set_trace_id("t-rot-swap");
    req.set_old_token("old");
    req.set_new_token("fresh");
    proto::admin::CommonResponse resp;
    auto ctx = MakeContext(req.trace_id());

    admin_impl_->RotateAdminAuthToken(ctx.get(), &req, &resp);
    ASSERT_EQ(proto::admin::OK, resp.header().status().code());

    auto snap = verifier->SnapshotTokens();
    ASSERT_EQ(2u, snap.size());
    ASSERT_EQ("keep", snap[0]);
    ASSERT_EQ("fresh", snap[1]);
    ASSERT_EQ(AuthOutcome::kInvalidToken, verifier->Verify("Bearer old"));
    ASSERT_EQ(AuthOutcome::kOk, verifier->Verify("Bearer fresh"));
}

TEST_F(AdminAuthTokensTest, RotateAdminAuthToken_DuplicateNewIsIgnored) {
    auto verifier = std::make_shared<StaticBearerTokenVerifier>(std::vector<std::string>{"a", "b"});
    admin_impl_->SetTokenVerifier(verifier);

    proto::admin::RotateAdminAuthTokenRequest req;
    req.set_trace_id("t-rot-dup");
    req.set_new_token("a");
    proto::admin::CommonResponse resp;
    auto ctx = MakeContext(req.trace_id());

    admin_impl_->RotateAdminAuthToken(ctx.get(), &req, &resp);
    ASSERT_EQ(proto::admin::OK, resp.header().status().code());

    auto snap = verifier->SnapshotTokens();
    ASSERT_EQ(2u, snap.size());
    ASSERT_EQ("a", snap[0]);
    ASSERT_EQ("b", snap[1]);
}

TEST_F(AdminAuthTokensTest, RotateAdminAuthToken_OldNotFound_InvalidArgument) {
    auto verifier = std::make_shared<StaticBearerTokenVerifier>(std::vector<std::string>{"only"});
    admin_impl_->SetTokenVerifier(verifier);

    proto::admin::RotateAdminAuthTokenRequest req;
    req.set_trace_id("t-rot-missing");
    req.set_old_token("does-not-exist");
    req.set_new_token("new");
    proto::admin::CommonResponse resp;
    auto ctx = MakeContext(req.trace_id());

    admin_impl_->RotateAdminAuthToken(ctx.get(), &req, &resp);
    ASSERT_EQ(proto::admin::INVALID_ARGUMENT, resp.header().status().code());

    // verifier list unchanged
    auto snap = verifier->SnapshotTokens();
    ASSERT_EQ(1u, snap.size());
    ASSERT_EQ("only", snap[0]);
}

TEST_F(AdminAuthTokensTest, RotateAdminAuthToken_EmptyNewToken_InvalidArgument) {
    auto verifier = std::make_shared<StaticBearerTokenVerifier>(std::vector<std::string>{"keep"});
    admin_impl_->SetTokenVerifier(verifier);

    proto::admin::RotateAdminAuthTokenRequest req;
    req.set_trace_id("t-rot-empty-new");
    req.set_old_token("keep");
    // new_token left empty
    proto::admin::CommonResponse resp;
    auto ctx = MakeContext(req.trace_id());

    admin_impl_->RotateAdminAuthToken(ctx.get(), &req, &resp);
    ASSERT_EQ(proto::admin::INVALID_ARGUMENT, resp.header().status().code());

    auto snap = verifier->SnapshotTokens();
    ASSERT_EQ(1u, snap.size());
    ASSERT_EQ("keep", snap[0]);
}

TEST_F(AdminAuthTokensTest, ListAdminAuthTokens_OpenMode) {
    auto verifier = std::make_shared<StaticBearerTokenVerifier>(std::vector<std::string>{});
    admin_impl_->SetTokenVerifier(verifier);

    proto::admin::ListAdminAuthTokensRequest req;
    req.set_trace_id("t-list-open");
    proto::admin::ListAdminAuthTokensResponse resp;
    auto ctx = MakeContext(req.trace_id());

    admin_impl_->ListAdminAuthTokens(ctx.get(), &req, &resp);
    ASSERT_EQ(proto::admin::OK, resp.header().status().code());
    ASSERT_FALSE(resp.enforcing());
    ASSERT_EQ(0, resp.token_count());
    ASSERT_EQ(0, resp.fingerprints_size());
}

TEST_F(AdminAuthTokensTest, ListAdminAuthTokens_EnforcingWithFingerprints) {
    auto verifier = std::make_shared<StaticBearerTokenVerifier>(std::vector<std::string>{"alpha", "beta"});
    admin_impl_->SetTokenVerifier(verifier);

    proto::admin::ListAdminAuthTokensRequest req;
    req.set_trace_id("t-list-enf");
    proto::admin::ListAdminAuthTokensResponse resp;
    auto ctx = MakeContext(req.trace_id());

    admin_impl_->ListAdminAuthTokens(ctx.get(), &req, &resp);
    ASSERT_EQ(proto::admin::OK, resp.header().status().code());
    ASSERT_TRUE(resp.enforcing());
    ASSERT_EQ(2, resp.token_count());
    ASSERT_EQ(2, resp.fingerprints_size());
    // fingerprints are 8 lowercase hex chars
    for (const auto &fp : resp.fingerprints()) {
        ASSERT_EQ(8u, fp.size());
        for (char c : fp) {
            ASSERT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) << "non-hex char in fingerprint: " << fp;
        }
    }
    // distinct tokens produce distinct fingerprints
    ASSERT_NE(resp.fingerprints(0), resp.fingerprints(1));
}

TEST_F(AdminAuthTokensTest, ListAdminAuthTokens_FingerprintIsDeterministic) {
    auto v1 = std::make_shared<StaticBearerTokenVerifier>(std::vector<std::string>{"same-token"});
    admin_impl_->SetTokenVerifier(v1);
    proto::admin::ListAdminAuthTokensRequest req1;
    req1.set_trace_id("t-fp-1");
    proto::admin::ListAdminAuthTokensResponse resp1;
    auto ctx1 = MakeContext(req1.trace_id());
    admin_impl_->ListAdminAuthTokens(ctx1.get(), &req1, &resp1);
    ASSERT_EQ(1, resp1.fingerprints_size());

    auto v2 = std::make_shared<StaticBearerTokenVerifier>(std::vector<std::string>{"same-token"});
    admin_impl_->SetTokenVerifier(v2);
    proto::admin::ListAdminAuthTokensRequest req2;
    req2.set_trace_id("t-fp-2");
    proto::admin::ListAdminAuthTokensResponse resp2;
    auto ctx2 = MakeContext(req2.trace_id());
    admin_impl_->ListAdminAuthTokens(ctx2.get(), &req2, &resp2);
    ASSERT_EQ(1, resp2.fingerprints_size());

    ASSERT_EQ(resp1.fingerprints(0), resp2.fingerprints(0));
}
