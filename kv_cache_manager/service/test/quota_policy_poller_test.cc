#include <atomic>
#include <chrono>
#include <cstdio>
#include <grpcpp/grpcpp.h>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_group_quota.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/service/quota_policy_poller.h"

namespace kv_cache_manager {
namespace {

class FakeQuotaOptimizer final : public proto::optimizer::OptimizerService::Service {
public:
    grpc::Status PullQuotaAllocation(grpc::ServerContext *,
                                     const proto::optimizer::PullQuotaAllocationRequest *request,
                                     proto::optimizer::PullQuotaAllocationResponse *response) override {
        response->mutable_header()->mutable_status()->set_code(proto::optimizer::OK);
        response->set_pull_status(proto::optimizer::QUOTA_PULL_PLAN);
        response->set_plan_id("plan-1");
        response->set_plan_hash("hash-1");
        response->set_pool_id(request->pool_id());
        response->set_reason(hard_resize ? "two_phase_hard_resize" : "writes_quota=false");
        response->set_leader_epoch(11);
        response->set_allocation_epoch(22);
        response->set_valid_until_ns(NowNs() + 60'000'000'000LL);
        response->set_executable(hard_resize);
        response->set_writes_quota(hard_resize);
        response->set_execution_phase(hard_resize ? "DONOR_SHRINK" : "SHADOW");
        response->set_execution_revision(1);
        response->set_release_deadline_ns(NowNs() + 60'000'000'000LL);
        response->set_release_consecutive_samples(release_consecutive_samples);
        auto *allocation = response->mutable_allocation();
        allocation->set_quota_target_id(request->quota_target_id());
        allocation->set_instance_group("group-a");
        allocation->set_current_quota_bytes(1000);
        allocation->set_target_quota_bytes(800);
        allocation->set_min_quota_bytes(500);
        allocation->set_max_quota_bytes(1500);
        return grpc::Status::OK;
    }

    grpc::Status ReportQuotaResizeResult(grpc::ServerContext *,
                                         const proto::optimizer::ReportQuotaResizeResultRequest *request,
                                         proto::optimizer::ReportQuotaResizeResultResponse *response) override {
        ++result_count;
        last_status = request->status();
        last_used_bytes = request->observed_used_bytes();
        response->mutable_header()->mutable_status()->set_code(proto::optimizer::OK);
        return grpc::Status::OK;
    }

    static int64_t NowNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    std::atomic<int> result_count{0};
    bool hard_resize = false;
    int64_t release_consecutive_samples = 1;
    std::string last_status;
    int64_t last_used_bytes = -1;
};

class QuotaPolicyPollerTest : public TESTBASE {
protected:
    void SetUp() override {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(&service_);
        server_ = builder.BuildAndStart();
        ASSERT_NE(nullptr, server_);
        ASSERT_GT(port_, 0);

        metrics_ = std::make_shared<MetricsRegistry>();
        registry_ = std::make_shared<RegistryManager>("", metrics_);
        ASSERT_TRUE(registry_->Init());
        RequestContext context("quota-policy-poller-test");
        InstanceGroup group;
        group.set_name("group-a");
        group.set_max_instance_count(1);
        group.set_quota(InstanceGroupQuota(1000, {}));
        group.set_version(7);
        ASSERT_EQ(EC_OK, registry_->CreateInstanceGroup(&context, group));
        state_file_ = "/tmp/kvcm-quota-policy-poller-" + std::to_string(FakeQuotaOptimizer::NowNs());
    }

    void TearDown() override {
        std::remove(state_file_.c_str());
        server_->Shutdown();
    }

    QuotaPolicyPollerConfig MakeConfig(bool enable_hard_resize = false) const {
        QuotaPolicyPollerConfig config;
        config.enable = true;
        config.enable_hard_resize = enable_hard_resize;
        config.optimizer_service_discovery_url = "static://127.0.0.1:" + std::to_string(port_);
        config.pool_id = "pool-a";
        config.quota_target_id = "target-a";
        config.instance_group = "group-a";
        config.state_file = state_file_;
        config.rpc_timeout_ms = 1000;
        return config;
    }

    std::shared_ptr<const InstanceGroup> GetGroup() const {
        RequestContext context("quota-policy-poller-assert");
        return registry_->GetInstanceGroup(&context, "group-a").second;
    }

    FakeQuotaOptimizer service_;
    int port_ = 0;
    std::unique_ptr<grpc::Server> server_;
    std::shared_ptr<MetricsRegistry> metrics_;
    std::shared_ptr<RegistryManager> registry_;
    std::string state_file_;
};

} // namespace

TEST_F(QuotaPolicyPollerTest, LeaderPullsAndRecordsDryRunWithoutUpdatingQuota) {
    QuotaPolicyPoller poller(MakeConfig(), registry_, []() { return true; });
    ASSERT_TRUE(poller.Init());
    ASSERT_TRUE(poller.PollOnce());
    EXPECT_EQ(11u, poller.last_leader_epoch());
    EXPECT_EQ(22u, poller.last_allocation_epoch());
    EXPECT_EQ(1, service_.result_count.load());
    EXPECT_EQ("DRY_RUN_ACCEPTED", service_.last_status);
    const auto group = GetGroup();
    ASSERT_NE(nullptr, group);
    EXPECT_EQ(1000, group->quota().capacity());
    EXPECT_EQ(7, group->version());
}

TEST_F(QuotaPolicyPollerTest, HardShrinkUsesVersionCasAndConfirmsReleaseBeforeReporting) {
    service_.hard_resize = true;
    QuotaPolicyPoller poller(
        MakeConfig(true), registry_, []() { return true; }, nullptr, metrics_);
    ASSERT_TRUE(poller.Init());
    ASSERT_TRUE(poller.PollOnce());
    EXPECT_EQ("DONOR_RELEASE_CONFIRMED", service_.last_status);
    EXPECT_EQ(0, service_.last_used_bytes);
    const auto group = GetGroup();
    ASSERT_NE(nullptr, group);
    EXPECT_EQ(800, group->quota().capacity());
    EXPECT_EQ(8, group->version());
}

TEST_F(QuotaPolicyPollerTest, HardShrinkWaitsForConfiguredConsecutiveUsageSamples) {
    service_.hard_resize = true;
    service_.release_consecutive_samples = 2;
    QuotaPolicyPoller poller(MakeConfig(true), registry_, []() { return true; }, nullptr, metrics_);
    ASSERT_TRUE(poller.Init());

    ASSERT_TRUE(poller.PollOnce());
    EXPECT_EQ("DONOR_SHRINK_APPLIED", service_.last_status);
    EXPECT_EQ(0u, poller.last_execution_revision());
    auto group = GetGroup();
    ASSERT_NE(nullptr, group);
    EXPECT_EQ(800, group->quota().capacity());
    EXPECT_EQ(8, group->version());

    ASSERT_TRUE(poller.PollOnce());
    EXPECT_EQ("DONOR_RELEASE_CONFIRMED", service_.last_status);
    EXPECT_EQ(1u, poller.last_execution_revision());
    group = GetGroup();
    ASSERT_NE(nullptr, group);
    EXPECT_EQ(800, group->quota().capacity());
    EXPECT_EQ(8, group->version());
}

TEST_F(QuotaPolicyPollerTest, HardPlanIsRejectedWhenLocalHardResizeSwitchIsDisabled) {
    service_.hard_resize = true;
    QuotaPolicyPoller poller(MakeConfig(), registry_, []() { return true; });
    ASSERT_TRUE(poller.Init());
    ASSERT_TRUE(poller.PollOnce());
    EXPECT_EQ("HARD_RESIZE_REJECTED", service_.last_status);
    const auto group = GetGroup();
    ASSERT_NE(nullptr, group);
    EXPECT_EQ(1000, group->quota().capacity());
    EXPECT_EQ(7, group->version());
}

TEST_F(QuotaPolicyPollerTest, FollowerNeverCallsOptimizer) {
    QuotaPolicyPoller poller(MakeConfig(), registry_, []() { return false; });
    ASSERT_TRUE(poller.Init());
    EXPECT_FALSE(poller.PollOnce());
    EXPECT_EQ(0, service_.result_count.load());
}

TEST_F(QuotaPolicyPollerTest, LeadershipLossAfterPullPreventsHardQuotaCas) {
    service_.hard_resize = true;
    std::atomic<int> leadership_checks{0};
    QuotaPolicyPoller poller(MakeConfig(true), registry_, [&leadership_checks]() { return leadership_checks++ == 0; });
    ASSERT_TRUE(poller.Init());
    EXPECT_FALSE(poller.PollOnce());
    EXPECT_EQ(0, service_.result_count.load());
    EXPECT_EQ(0u, poller.last_execution_revision());
    const auto group = GetGroup();
    ASSERT_NE(nullptr, group);
    EXPECT_EQ(1000, group->quota().capacity());
    EXPECT_EQ(7, group->version());
}

TEST_F(QuotaPolicyPollerTest, MissingAuthoritativeUsageKeepsRevisionRetryable) {
    service_.hard_resize = true;
    RequestContext context("quota-policy-poller-usage-retry-test");
    ASSERT_EQ(EC_OK,
              registry_->RegisterInstance(
                  &context, "group-a", "instance-a", 1, {LocationSpecInfo("kv", 1)}, ModelDeployment{}));
    QuotaPolicyPoller poller(MakeConfig(true), registry_, []() { return true; });
    ASSERT_TRUE(poller.Init());
    ASSERT_TRUE(poller.PollOnce());
    EXPECT_EQ("USAGE_OBSERVATION_UNAVAILABLE", service_.last_status);
    EXPECT_EQ(0u, poller.last_leader_epoch());
    EXPECT_EQ(0u, poller.last_allocation_epoch());
    EXPECT_EQ(0u, poller.last_execution_revision());
}

} // namespace kv_cache_manager
