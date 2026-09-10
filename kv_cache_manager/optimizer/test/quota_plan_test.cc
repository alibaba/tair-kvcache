#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/quota_runtime/quota_plan.h"

namespace kv_cache_manager {
namespace {

constexpr int64_t kGiB = 1024LL * 1024 * 1024;

QuotaPoolMemberConfig Member(const std::string &id, const std::string &source) {
    QuotaPoolMemberConfig member;
    member.quota_target_id = id;
    member.source_id = source;
    member.instance_group = "group-" + id;
    member.quota_scope = "per_replica";
    member.current_quota_bytes = kGiB;
    member.min_quota_bytes = kGiB;
    member.configured_max_quota_bytes = 2 * kGiB;
    member.hardware_max_quota_bytes = 2 * kGiB;
    member.configured_max_source = "test_config";
    member.hardware_max_source = "test_hardware";
    return member;
}

OnlineMrcSourceSnapshot Source(const std::string &id, uint64_t hit_at_one, uint64_t hit_at_two) {
    OnlineMrcSourceSnapshot source;
    source.source_id = id;
    source.newest_event_time_ns = 99'000'000'000LL;
    source.accepted_facts = 10;
    source.curve = {{static_cast<uint64_t>(kGiB), 100, hit_at_one}, {static_cast<uint64_t>(2 * kGiB), 100, hit_at_two}};
    return source;
}

QuotaPlannerRuntimeConfig Config() {
    QuotaPlannerRuntimeConfig config;
    config.enable = true;
    config.plan_ttl_seconds = 60;
    QuotaPoolConfig pool;
    pool.pool_id = "pool-a";
    pool.quota_scope = "per_replica";
    pool.allocatable_bytes = 3 * kGiB;
    pool.allocatable_source = "test_pool_snapshot";
    pool.max_mrc_freshness_seconds = 10;
    pool.members = {Member("target-a", "source-a"), Member("target-b", "source-b")};
    config.pools = {pool};
    return config;
}

OnlineMrcDecisionSnapshot Snapshot() {
    OnlineMrcDecisionSnapshot snapshot;
    snapshot.snapshot_id = 7;
    snapshot.created_at_ns = 100'000'000'000LL;
    snapshot.sources = {Source("source-a", 10, 11), Source("source-b", 5, 20)};
    return snapshot;
}

} // namespace

TEST(ShadowQuotaPlannerTest, MaximizesHitTokensUnderFixedPoolBudget) {
    ShadowQuotaPlanner planner(Config());
    const auto plans = planner.BuildPlans(Snapshot());
    ASSERT_EQ(1u, plans.size());
    const auto &plan = *plans.front();
    EXPECT_EQ("SHADOW_READY", plan.status);
    EXPECT_FALSE(plan.executable);
    ASSERT_EQ(2u, plan.allocations.size());
    EXPECT_EQ(kGiB, plan.allocations[0].target_quota_bytes);
    EXPECT_EQ(2 * kGiB, plan.allocations[1].target_quota_bytes);
    EXPECT_EQ(10u, plan.allocations[0].baseline_hit_tokens);
    EXPECT_EQ(5u, plan.allocations[1].baseline_hit_tokens);
    EXPECT_DOUBLE_EQ(7.5, ExpectedHitRateGainPercentagePoints(plan));
    EXPECT_FALSE(plan.plan_hash.empty());
}

TEST(ShadowQuotaPlannerTest, UsesObservedQuotaAsBenefitBaseline) {
    ShadowQuotaPlanner planner(Config());
    InMemoryQuotaPlanStore::ObservedQuotaMap observed{{"pool-a", {{"target-a", 2 * kGiB}, {"target-b", kGiB}}}};
    const auto plan = planner.BuildPlans(Snapshot(), observed).front();
    ASSERT_EQ("SHADOW_READY", plan->status);
    ASSERT_EQ(2u, plan->allocations.size());
    EXPECT_EQ(11u, plan->allocations[0].baseline_hit_tokens);
    EXPECT_EQ(5u, plan->allocations[1].baseline_hit_tokens);
    EXPECT_DOUBLE_EQ(7.0, ExpectedHitRateGainPercentagePoints(*plan));
}

TEST(ShadowQuotaPlannerTest, FreezesWhenGrossBenefitIsBelowThreshold) {
    auto config = Config();
    config.pools[0].min_expected_hit_rate_gain_pp = 8.0;
    ShadowQuotaPlanner planner(config);
    const auto plan = planner.BuildPlans(Snapshot()).front();
    EXPECT_EQ("FROZEN", plan->status);
    EXPECT_EQ("expected_hit_rate_gain_below_threshold", plan->reason);
    EXPECT_DOUBLE_EQ(7.5, plan->expected_hit_rate_gain_pp);
}

TEST(ShadowQuotaPlannerTest, FreezesWhenBenefitPerTransferredTiBIsBelowThreshold) {
    auto config = Config();
    config.pools[0].min_gain_pp_per_tib_moved = 8000.0;
    ShadowQuotaPlanner planner(config);
    const auto plan = planner.BuildPlans(Snapshot()).front();
    EXPECT_EQ("FROZEN", plan->status);
    EXPECT_EQ("gain_per_tib_moved_below_threshold", plan->reason);
    EXPECT_EQ(static_cast<uint64_t>(kGiB), plan->quota_transfer_bytes);
    EXPECT_DOUBLE_EQ(7680.0, plan->gain_pp_per_tib_moved);
}

TEST(ShadowQuotaPlannerTest, AppliesPlanLevelTransferHysteresis) {
    auto config = Config();
    config.pools[0].min_quota_transfer_bytes = 2 * kGiB;
    ShadowQuotaPlanner planner(config);
    const auto plan = planner.BuildPlans(Snapshot()).front();
    EXPECT_EQ("FROZEN", plan->status);
    EXPECT_EQ("quota_transfer_below_hysteresis", plan->reason);
}

TEST(ShadowQuotaPlannerTest, MovementPenaltyCanKeepCurrentAllocation) {
    auto config = Config();
    config.pools[0].movement_penalty_pp_per_tib = 8000.0;
    ShadowQuotaPlanner planner(config);
    const auto plan = planner.BuildPlans(Snapshot()).front();
    EXPECT_EQ("NOOP", plan->status);
    ASSERT_EQ(2u, plan->allocations.size());
    EXPECT_EQ(kGiB, plan->allocations[0].target_quota_bytes);
    EXPECT_EQ(kGiB, plan->allocations[1].target_quota_bytes);
    EXPECT_EQ(0u, plan->quota_transfer_bytes);
}

TEST(ShadowQuotaPlannerTest, RequiresConsecutiveStableCandidates) {
    auto config = Config();
    config.pools[0].stability_required_plans = 3;
    ShadowQuotaPlanner planner(config);
    const auto first = planner.BuildPlans(Snapshot()).front();
    EXPECT_EQ("FROZEN", first->status);
    EXPECT_EQ(1, first->stability_confirmed_plans);
    const auto second = planner.BuildPlans(Snapshot()).front();
    EXPECT_EQ("FROZEN", second->status);
    EXPECT_EQ(2, second->stability_confirmed_plans);
    const auto third = planner.BuildPlans(Snapshot()).front();
    EXPECT_EQ("SHADOW_READY", third->status);
    EXPECT_EQ(3, third->stability_confirmed_plans);
    EXPECT_EQ(3, third->stability_required_plans);
}

TEST(ShadowQuotaPlannerTest, ReportsCapacitySavingAtConfiguredSla) {
    auto config = Config();
    config.pools[0].capacity_saving_sla_ratio = 0.25;
    ShadowQuotaPlanner planner(config);
    const auto plan = planner.BuildPlans(Snapshot()).front();
    EXPECT_EQ(2 * kGiB, plan->sla_required_capacity_bytes);
    EXPECT_EQ(kGiB, plan->sla_capacity_saving_bytes);
    EXPECT_EQ(0, plan->sla_capacity_deficit_bytes);
}

TEST(ShadowQuotaPlannerTest, FreezesWhenAnySourceHasNoFacts) {
    ShadowQuotaPlanner planner(Config());
    auto snapshot = Snapshot();
    snapshot.sources[1].accepted_facts = 0;
    const auto plan = planner.BuildPlans(snapshot).front();
    EXPECT_EQ("FROZEN", plan->status);
    EXPECT_NE(std::string::npos, plan->reason.find("empty_mrc_source"));
    EXPECT_TRUE(plan->allocations.empty());
}

TEST(InMemoryQuotaPlanStoreTest, AtomicallyPublishesLatestPlanPerPool) {
    InMemoryQuotaPlanStore store;
    auto first = std::make_shared<PoolQuotaPlan>();
    first->pool_id = "pool-a";
    first->allocation_epoch = 1;
    store.Publish(first);
    auto second = std::make_shared<PoolQuotaPlan>();
    second->pool_id = "pool-a";
    second->allocation_epoch = 2;
    store.Publish(second);
    ASSERT_NE(nullptr, store.Get("pool-a"));
    EXPECT_EQ(2u, store.Get("pool-a")->allocation_epoch);
    EXPECT_EQ(nullptr, store.Get("missing"));
}

TEST(InMemoryQuotaPlanStoreTest, ReleasesReceiverOnlyAfterEveryDonorConfirmsPhysicalRelease) {
    auto config = Config();
    config.enable_hard_resize = true;
    config.plan_ttl_seconds = 3600;
    config.release_timeout_seconds = 1800;
    config.release_consecutive_samples = 2;
    config.pools[0].members[0].current_quota_bytes = 2 * kGiB;
    ShadowQuotaPlanner planner(config);
    const auto built = planner.BuildPlans(Snapshot()).front();
    ASSERT_EQ("RECONCILE", built->execution_phase);
    ASSERT_TRUE(built->writes_quota);

    InMemoryQuotaPlanStore store;
    ASSERT_TRUE(store.Publish(built));
    QuotaResizeResult donor;
    donor.plan_id = built->plan_id;
    donor.plan_hash = built->plan_hash;
    donor.pool_id = built->pool_id;
    donor.quota_target_id = "target-a";
    donor.leader_epoch = built->leader_epoch;
    donor.allocation_epoch = built->allocation_epoch;
    donor.execution_revision = built->execution_revision;
    donor.status = "HOLD_ACKNOWLEDGED";
    donor.observed_quota_bytes = 2 * kGiB;
    donor.observed_used_bytes = 2 * kGiB;
    ASSERT_TRUE(store.RecordResizeResult(donor));
    QuotaResizeResult receiver = donor;
    receiver.quota_target_id = "target-b";
    receiver.observed_quota_bytes = kGiB;
    receiver.observed_used_bytes = kGiB;
    ASSERT_TRUE(store.RecordResizeResult(receiver));
    const auto donor_phase = store.Get("pool-a");
    ASSERT_EQ("DONOR_SHRINK", donor_phase->execution_phase);
    ASSERT_EQ(2u, donor_phase->execution_revision);

    donor.execution_revision = donor_phase->execution_revision;
    donor.status = "DONOR_RELEASE_CONFIRMED";
    ASSERT_TRUE(store.RecordResizeResult(donor));
    const auto receiver_phase = store.Get("pool-a");
    ASSERT_EQ("RECEIVER_GROW", receiver_phase->execution_phase);
    ASSERT_EQ(3u, receiver_phase->execution_revision);

    receiver.quota_target_id = "target-b";
    receiver.execution_revision = receiver_phase->execution_revision;
    receiver.status = "RECEIVER_GROW_APPLIED";
    ASSERT_TRUE(store.RecordResizeResult(receiver));
    EXPECT_EQ("COMPLETE", store.Get("pool-a")->execution_phase);
    EXPECT_FALSE(store.Get("pool-a")->executable);
}

TEST(InMemoryQuotaPlanStoreTest, TemporaryUsageUnavailabilityDoesNotFreezeTransfer) {
    auto plan = std::make_shared<PoolQuotaPlan>();
    plan->pool_id = "pool-a";
    plan->plan_id = "plan-a";
    plan->plan_hash = "hash-a";
    plan->writes_quota = true;
    plan->executable = true;
    plan->status = "EXECUTING";
    plan->execution_phase = "DONOR_SHRINK";
    plan->release_required_targets.insert("target-a");
    plan->allocations.push_back(QuotaAllocation{"target-a", "source-a", "group-a", 1000, 800, 500, 1500});
    InMemoryQuotaPlanStore store;
    ASSERT_TRUE(store.Publish(plan));

    QuotaResizeResult unavailable;
    unavailable.plan_id = plan->plan_id;
    unavailable.plan_hash = plan->plan_hash;
    unavailable.pool_id = plan->pool_id;
    unavailable.quota_target_id = "target-a";
    unavailable.execution_revision = 1;
    unavailable.status = "USAGE_OBSERVATION_UNAVAILABLE";
    unavailable.reason = "authoritative_group_usage_unavailable";
    unavailable.observed_quota_bytes = 1000;
    unavailable.observed_used_bytes = -1;

    ASSERT_TRUE(store.RecordResizeResult(unavailable));
    const auto current = store.Get("pool-a");
    ASSERT_NE(nullptr, current);
    EXPECT_EQ("DONOR_SHRINK", current->execution_phase);
    EXPECT_TRUE(current->executable);
    EXPECT_EQ(1u, current->execution_revision);
}

TEST(InMemoryQuotaPlanStoreTest, ReceiverWaitsUntilEveryDonorConfirmsRelease) {
    auto plan = std::make_shared<PoolQuotaPlan>();
    plan->pool_id = "pool-a";
    plan->plan_id = "plan-a";
    plan->plan_hash = "hash-a";
    plan->writes_quota = true;
    plan->executable = true;
    plan->status = "EXECUTING";
    plan->execution_phase = "DONOR_SHRINK";
    plan->release_required_targets = {"donor-a", "donor-b"};
    plan->allocations = {
        QuotaAllocation{"donor-a", "source-a", "group-a", 1000, 800, 500, 1500},
        QuotaAllocation{"donor-b", "source-b", "group-b", 1000, 900, 500, 1500},
        QuotaAllocation{"receiver", "source-c", "group-c", 500, 800, 500, 1500},
    };
    InMemoryQuotaPlanStore store;
    ASSERT_TRUE(store.Publish(plan));

    QuotaResizeResult result;
    result.plan_id = plan->plan_id;
    result.plan_hash = plan->plan_hash;
    result.pool_id = plan->pool_id;
    result.leader_epoch = plan->leader_epoch;
    result.allocation_epoch = plan->allocation_epoch;
    result.execution_revision = plan->execution_revision;
    result.status = "DONOR_RELEASE_CONFIRMED";
    result.quota_target_id = "donor-a";
    ASSERT_TRUE(store.RecordResizeResult(result));
    EXPECT_EQ("DONOR_SHRINK", store.Get("pool-a")->execution_phase);
    EXPECT_EQ(1u, store.Get("pool-a")->execution_revision);

    result.quota_target_id = "donor-b";
    ASSERT_TRUE(store.RecordResizeResult(result));
    EXPECT_EQ("RECEIVER_GROW", store.Get("pool-a")->execution_phase);
    EXPECT_EQ(2u, store.Get("pool-a")->execution_revision);
}

TEST(InMemoryQuotaPlanStoreTest, FreezesTransferOnResizeFailure) {
    auto plan = std::make_shared<PoolQuotaPlan>();
    plan->pool_id = "pool-a";
    plan->plan_id = "plan-a";
    plan->plan_hash = "hash-a";
    plan->writes_quota = true;
    plan->executable = true;
    plan->status = "EXECUTING";
    plan->execution_phase = "DONOR_SHRINK";
    plan->release_required_targets.insert("target-a");
    plan->allocations.push_back(QuotaAllocation{"target-a", "source-a", "group-a", 1000, 800, 500, 1500});
    InMemoryQuotaPlanStore store;
    ASSERT_TRUE(store.Publish(plan));
    QuotaResizeResult failed;
    failed.plan_id = plan->plan_id;
    failed.plan_hash = plan->plan_hash;
    failed.pool_id = plan->pool_id;
    failed.quota_target_id = "target-a";
    failed.execution_revision = 1;
    failed.status = "DONOR_SHRINK_FAILED";
    ASSERT_TRUE(store.RecordResizeResult(failed));
    EXPECT_EQ("FROZEN", store.Get("pool-a")->execution_phase);
    EXPECT_FALSE(store.Get("pool-a")->executable);
    auto replacement = std::make_shared<PoolQuotaPlan>();
    replacement->pool_id = "pool-a";
    EXPECT_FALSE(store.Publish(replacement));
}

} // namespace kv_cache_manager
