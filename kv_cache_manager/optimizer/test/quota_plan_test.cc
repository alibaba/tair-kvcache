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
    EXPECT_FALSE(plan.plan_hash.empty());
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
