#include <cmath>
#include <limits>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/effective_quota.h"

namespace kv_cache_manager {
namespace {
constexpr auto kDram = DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL;
constexpr auto kSsd = DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD;
constexpr int64_t kNow = 100000000;

QuotaConfig Auto(DataStorageType type, double ratio = 1.0) {
    QuotaConfig config(0, type);
    config.set_mode("auto");
    config.set_ratio(ratio);
    return config;
}

StorageCapacitySnapshot Capacity(DataStorageType type, int64_t capacity, std::string identity = "") {
    return {capacity, 1000, kNow, EC_OK, identity, type};
}

InstanceGroup Group(const std::vector<QuotaConfig> &configs = {}) {
    InstanceGroup group;
    group.set_storage_candidates({"dram", "ssd"});
    group.set_quota(InstanceGroupQuota(-1, configs));
    return group;
}
} // namespace

class EffectiveQuotaTest : public TESTBASE {};

TEST_F(EffectiveQuotaTest, MixedMediaExampleAndMonitoring) {
    const auto quota = ResolveEffectiveQuota(Group({Auto(kDram, 0.5), Auto(kSsd, 0.8)}),
                                             {{"dram", Capacity(kDram, 100000)}, {"ssd", Capacity(kSsd, 200000)}},
                                             kNow);
    EXPECT_EQ(-1, quota.ForAdmission().capacity());
    EXPECT_EQ(-1, quota.ForReclaim().capacity());
    EXPECT_EQ(50000, quota.types.at(kDram).capacity);
    EXPECT_EQ(160000, quota.types.at(kSsd).capacity);
    EXPECT_EQ(210000, quota.reference_capacity);
    EXPECT_NEAR(0.476190476, QuotaUsageRatio(100000, quota.reference_capacity), 1e-8);
    EXPECT_DOUBLE_EQ(0.4, QuotaUsageRatio(20000, quota.types.at(kDram).capacity));
    EXPECT_DOUBLE_EQ(0.5, QuotaUsageRatio(80000, quota.types.at(kSsd).capacity));
}

TEST_F(EffectiveQuotaTest, DefaultsAndMigrationOnlyTarget) {
    auto group = Group();
    group.set_storage_candidates({"dram", "dram"});
    auto strategy = std::make_shared<MigrationStrategy>();
    strategy->set_source_storage_name("dram");
    strategy->set_target_storage_name("ssd");
    auto cache = std::make_shared<CacheConfig>();
    cache->set_migration_strategies({strategy});
    group.set_cache_config(cache);
    const auto quota =
        ResolveEffectiveQuota(group, {{"dram", Capacity(kDram, 100)}, {"ssd", Capacity(kSsd, 200)}}, kNow);
    ASSERT_EQ(2u, quota.types.size());
    EXPECT_EQ(300, quota.reference_capacity);
    EXPECT_EQ("auto", quota.types.at(kSsd).config.mode());
}

TEST_F(EffectiveQuotaTest, DeduplicatesAliasesAndSumsIndependentPools) {
    auto group = Group({Auto(kDram, 0.5)});
    group.set_storage_candidates({"a", "a", "alias", "b"});
    const auto quota = ResolveEffectiveQuota(group,
                                             {{"a", Capacity(kDram, 100, "pool1")},
                                              {"alias", Capacity(kDram, 100, "pool1")},
                                              {"b", Capacity(kDram, 200, "pool2")}},
                                             kNow);
    EXPECT_EQ(300, quota.types.at(kDram).backend_capacity);
    EXPECT_EQ(150, quota.reference_capacity);
}

TEST_F(EffectiveQuotaTest, AliasWithUnsupportedAccountingCannotBorrowAnotherAliasesBudget) {
    auto group = Group({Auto(kDram)});
    group.set_storage_candidates({"strict", "auto"});
    auto unsupported = Capacity(kDram, -1, "same_pool");
    unsupported.refresh_code = EC_UNIMPLEMENTED;
    unsupported.last_success_steady_us = 0;
    const auto quota =
        ResolveEffectiveQuota(group, {{"strict", Capacity(kDram, 100, "same_pool")}, {"auto", unsupported}}, kNow);
    EXPECT_EQ("unsupported", quota.types.at(kDram).status);
    EXPECT_FALSE(quota.types.at(kDram).usable);
    EXPECT_TRUE(quota.ForReclaim().quota_config().empty());
}

TEST_F(EffectiveQuotaTest, ExplicitScopeDoesNotCountUnrelatedBackends) {
    auto config = Auto(kDram);
    config.set_storage_names({"dram", "dram"});
    auto group = Group({config});
    group.set_storage_candidates({"dram"});
    const auto quota = ResolveEffectiveQuota(
        group,
        {{"dram", Capacity(kDram, 100)}, {"ssd", Capacity(kSsd, 200)}, {"unrelated", Capacity(kDram, 10000)}},
        kNow);
    EXPECT_EQ(100, quota.reference_capacity);
}

TEST_F(EffectiveQuotaTest, UnlimitedPartialOverridesKeepOtherAssociatedTypesAutomatic) {
    const auto quota = ResolveEffectiveQuota(
        Group({QuotaConfig(50, kDram)}), {{"dram", Capacity(kDram, 100)}, {"ssd", Capacity(kSsd, 200)}}, kNow);
    EXPECT_EQ(250, quota.reference_capacity);
    EXPECT_EQ(50, quota.types.at(kDram).capacity);
    EXPECT_EQ("fixed", quota.types.at(kDram).config.mode());
    EXPECT_EQ(200, quota.types.at(kSsd).capacity);
    EXPECT_EQ("auto", quota.types.at(kSsd).config.mode());
}

TEST_F(EffectiveQuotaTest, FixedAndAutoMixAndExplicitTotalRemainIndependent) {
    auto group = Group({QuotaConfig(500, kDram), Auto(kSsd, 0.5)});
    group.quota_.set_capacity(100);
    const auto quota =
        ResolveEffectiveQuota(group, {{"dram", Capacity(kDram, 100)}, {"ssd", Capacity(kSsd, 200)}}, kNow);
    EXPECT_EQ(100, quota.ForAdmission().capacity());
    EXPECT_EQ(600, quota.reference_capacity);
    EXPECT_EQ(500, quota.types.at(kDram).capacity);
    EXPECT_EQ("fixed", quota.types.at(kDram).status);
}

TEST_F(EffectiveQuotaTest, FailureRetainsLastGoodAndExpiryNeverCreatesZeroEvictionQuota) {
    auto snapshot = Capacity(kDram, 100);
    snapshot.refresh_code = EC_ERROR;
    auto group = Group({Auto(kDram)});
    group.set_storage_candidates({"dram"});
    auto quota = ResolveEffectiveQuota(group, {{"dram", snapshot}}, kNow + 10000000);
    EXPECT_EQ("refresh_error", quota.types.at(kDram).status);
    EXPECT_EQ(100, quota.reference_capacity);
    EXPECT_EQ(100, quota.ForAdmission().quota_config().front().capacity());
    EXPECT_EQ(100, quota.ForReclaim().quota_config().front().capacity());
    quota = ResolveEffectiveQuota(group, {{"dram", snapshot}}, kNow + 30000001);
    EXPECT_EQ("stale", quota.types.at(kDram).status);
    EXPECT_TRUE(quota.types.at(kDram).known);
    EXPECT_FALSE(quota.types.at(kDram).usable);
    EXPECT_EQ(100, quota.reference_capacity);
    EXPECT_EQ(0, quota.ForAdmission().quota_config().front().capacity());
    EXPECT_TRUE(quota.ForReclaim().quota_config().empty());
    snapshot.capacity = 50;
    snapshot.last_success_steady_us = kNow + 40000000;
    snapshot.refresh_code = EC_OK;
    quota = ResolveEffectiveQuota(group, {{"dram", snapshot}}, kNow + 40000000);
    EXPECT_EQ("fresh", quota.types.at(kDram).status);
    EXPECT_EQ(50, quota.ForReclaim().quota_config().front().capacity());
}

TEST_F(EffectiveQuotaTest, FirstUnknownUnsupportedAndPartialUnknownAreNotPartialSums) {
    auto group = Group({Auto(kDram)});
    group.set_storage_candidates({"a", "b"});
    StorageCapacitySnapshot unknown;
    unknown.type = kDram;
    unknown.refresh_code = EC_UNIMPLEMENTED;
    const auto quota = ResolveEffectiveQuota(group, {{"a", Capacity(kDram, 100)}, {"b", unknown}}, kNow);
    EXPECT_EQ(-1, quota.reference_capacity);
    EXPECT_EQ(-1, quota.types.at(kDram).backend_capacity);
    EXPECT_EQ("unsupported", quota.types.at(kDram).status);
    EXPECT_TRUE(quota.ForReclaim().quota_config().empty());
    EXPECT_EQ(0, quota.ForAdmission().quota_config().front().capacity());
    EXPECT_TRUE(std::isnan(QuotaUsageRatio(10, quota.reference_capacity)));
}

TEST_F(EffectiveQuotaTest, PartialFailureRetainsFailedBackendContribution) {
    auto group = Group({Auto(kDram)});
    group.set_storage_candidates({"a", "b"});
    auto failed = Capacity(kDram, 100);
    failed.refresh_code = EC_ERROR;
    const auto quota = ResolveEffectiveQuota(group, {{"a", Capacity(kDram, 50)}, {"b", failed}}, kNow);
    EXPECT_EQ(150, quota.reference_capacity);
    EXPECT_EQ("refresh_error", quota.types.at(kDram).status);
}

TEST_F(EffectiveQuotaTest, MissingOrWrongTypeExplicitBackendBlocksWithoutEviction) {
    auto config = Auto(kDram);
    config.set_storage_names({"missing"});
    auto quota = ResolveEffectiveQuota(Group({config}), {{"dram", Capacity(kDram, 100)}}, kNow);
    EXPECT_EQ("missing_backend", quota.types.at(kDram).status);
    EXPECT_TRUE(quota.ForReclaim().quota_config().empty());
    config.set_storage_names({"ssd"});
    quota = ResolveEffectiveQuota(Group({config}), {{"ssd", Capacity(kSsd, 200)}}, kNow);
    EXPECT_FALSE(quota.types.at(kDram).known);
}

TEST_F(EffectiveQuotaTest, TrueZeroIsKnownAndEnforced) {
    auto group = Group({Auto(kDram)});
    group.set_storage_candidates({"dram"});
    const auto quota = ResolveEffectiveQuota(group, {{"dram", Capacity(kDram, 0)}}, kNow);
    EXPECT_EQ(0, quota.reference_capacity);
    EXPECT_TRUE(quota.types.at(kDram).usable);
    EXPECT_EQ(0, quota.ForReclaim().quota_config().front().capacity());
    EXPECT_DOUBLE_EQ(0, QuotaUsageRatio(0, 0));
    EXPECT_TRUE(std::isinf(QuotaUsageRatio(1, 0)));
}

TEST_F(EffectiveQuotaTest, DecimalRatiosUseRoundedProductAndFractionalBytesRoundDown) {
    auto group = Group({Auto(kDram, 0.3)});
    group.set_storage_candidates({"dram"});
    EXPECT_EQ(30, ResolveEffectiveQuota(group, {{"dram", Capacity(kDram, 100)}}, kNow).reference_capacity);
    group.set_quota(InstanceGroupQuota(-1, {Auto(kDram, 0.5)}));
    EXPECT_EQ(50, ResolveEffectiveQuota(group, {{"dram", Capacity(kDram, 101)}}, kNow).reference_capacity);
}

TEST_F(EffectiveQuotaTest, OverflowIsUnknownAndInt64MaxAtFullRatioIsExact) {
    auto group = Group({Auto(kDram)});
    group.set_storage_candidates({"dram"});
    auto quota = ResolveEffectiveQuota(group, {{"dram", Capacity(kDram, INT64_MAX)}}, kNow);
    EXPECT_EQ(INT64_MAX, quota.reference_capacity);
    group.set_storage_candidates({"dram", "b"});
    quota = ResolveEffectiveQuota(group, {{"dram", Capacity(kDram, INT64_MAX)}, {"b", Capacity(kDram, 1)}}, kNow);
    EXPECT_FALSE(quota.types.at(kDram).known);
    EXPECT_EQ("overflow", quota.types.at(kDram).status);
    EXPECT_TRUE(quota.ForReclaim().quota_config().empty());
    group.set_quota(InstanceGroupQuota(-1, {QuotaConfig(INT64_MAX, kDram), QuotaConfig(1, kSsd)}));
    quota = ResolveEffectiveQuota(group, {}, kNow);
    EXPECT_EQ(-1, quota.reference_capacity);
}

TEST_F(EffectiveQuotaTest, LegacyZeroAndFixedQuotasDoNotBecomeAuto) {
    auto group = Group();
    group.set_quota(InstanceGroupQuota(0, {}));
    auto quota = ResolveEffectiveQuota(group, {{"dram", Capacity(kDram, 100)}}, kNow);
    EXPECT_EQ(0, quota.ForAdmission().capacity());
    EXPECT_TRUE(quota.types.empty());
    group.set_quota(InstanceGroupQuota(100, {QuotaConfig(0, kDram)}));
    quota = ResolveEffectiveQuota(group, {}, kNow);
    EXPECT_EQ("fixed", quota.types.at(kDram).status);
    EXPECT_EQ(0, quota.ForReclaim().quota_config().front().capacity());
}

TEST_F(EffectiveQuotaTest, SwitchingModesRatioAndScopeDoesNotRewriteConfiguration) {
    auto group = Group({QuotaConfig(1000, kDram)});
    group.set_storage_candidates({"dram"});
    auto snapshot = Capacity(kDram, 100);
    EXPECT_EQ(1000, ResolveEffectiveQuota(group, {{"dram", snapshot}}, kNow).reference_capacity);
    group.set_quota(InstanceGroupQuota(-1, {Auto(kDram, 0.5)}));
    EXPECT_EQ(50, ResolveEffectiveQuota(group, {{"dram", snapshot}}, kNow).reference_capacity);
    snapshot.refresh_code = EC_ERROR;
    group.set_quota(InstanceGroupQuota(-1, {QuotaConfig(200, kDram)}));
    auto resolved = ResolveEffectiveQuota(group, {{"dram", snapshot}}, kNow + 40000000);
    EXPECT_EQ(200, resolved.ForAdmission().quota_config().front().capacity());
    EXPECT_EQ("fixed", resolved.types.at(kDram).status);
    group.set_quota(InstanceGroupQuota(-1, {Auto(kDram, 0.25)}));
    EXPECT_EQ(25, ResolveEffectiveQuota(group, {{"dram", snapshot}}, kNow).reference_capacity);
    auto config = Auto(kDram);
    config.set_storage_names({"new_pool"});
    group.set_quota(InstanceGroupQuota(-1, {config}));
    resolved = ResolveEffectiveQuota(group, {{"dram", snapshot}}, kNow);
    EXPECT_EQ(-1, resolved.reference_capacity);
    EXPECT_TRUE(resolved.ForReclaim().quota_config().empty());
    EXPECT_EQ(-1, group.quota().capacity());
    EXPECT_EQ(0, group.quota().quota_config().front().capacity());
    EXPECT_EQ("auto", group.quota().quota_config().front().mode());
}

TEST_F(EffectiveQuotaTest, ConfigurationValidationAndJsonRoundTrip) {
    QuotaConfig config;
    ASSERT_TRUE(config.FromJsonString(R"({"storage_type":"pace","mode":"auto","ratio":0.5})"));
    EXPECT_EQ(0, config.capacity());
    std::string invalid;
    EXPECT_TRUE(config.ValidateRequiredFields(invalid));
    QuotaConfig copy;
    ASSERT_TRUE(copy.FromJsonString(config.ToJsonString()));
    EXPECT_EQ("auto", copy.mode());
    EXPECT_DOUBLE_EQ(0.5, copy.ratio());
    for (const double ratio : {0.0, -0.1, 1.1, std::numeric_limits<double>::infinity()}) {
        config.set_ratio(ratio);
        EXPECT_FALSE(config.ValidateRequiredFields(invalid));
    }
    InstanceGroupQuota duplicate(-1, {Auto(kDram), QuotaConfig(100, kDram)});
    EXPECT_FALSE(duplicate.ValidateRequiredFields(invalid));
}
} // namespace kv_cache_manager
