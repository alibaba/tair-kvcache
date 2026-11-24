#include <cmath>
#include <limits>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/effective_quota.h"

namespace kv_cache_manager {
namespace {
constexpr auto kDram = DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL;
constexpr auto kSsd = DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD;

QuotaConfig Auto(DataStorageType type, double ratio = 1.0) {
    QuotaConfig config(0, type);
    config.set_mode("auto");
    config.set_ratio(ratio);
    return config;
}

StorageCapacitySnapshot Capacity(DataStorageType type, int64_t capacity) { return {capacity, 1000, EC_OK, type}; }

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
                                             {{"dram", Capacity(kDram, 100000)}, {"ssd", Capacity(kSsd, 200000)}});
    EXPECT_EQ(-1, quota.ForAdmission().capacity());
    EXPECT_EQ(-1, quota.ForReclaim().capacity());
    EXPECT_EQ(50000, quota.types.at(kDram).capacity);
    EXPECT_EQ(160000, quota.types.at(kSsd).capacity);
    EXPECT_EQ(210000, quota.reference_capacity);
    EXPECT_NEAR(0.476190476, QuotaUsageRatio(100000, quota.reference_capacity), 1e-8);
    EXPECT_DOUBLE_EQ(0.4, QuotaUsageRatio(20000, quota.types.at(kDram).capacity));
    EXPECT_DOUBLE_EQ(0.5, QuotaUsageRatio(80000, quota.types.at(kSsd).capacity));
}

TEST_F(EffectiveQuotaTest, ExplicitAutoIncludesMigrationOnlyTarget) {
    auto group = Group({Auto(kDram), Auto(kSsd)});
    group.set_storage_candidates({"dram", "dram"});
    auto strategy = std::make_shared<MigrationStrategy>();
    strategy->set_source_storage_name("dram");
    strategy->set_target_storage_name("ssd");
    auto cache = std::make_shared<CacheConfig>();
    cache->set_migration_strategies({strategy});
    group.set_cache_config(cache);
    const auto quota =
        ResolveEffectiveQuota(group, {{"dram", Capacity(kDram, 100)}, {"ssd", Capacity(kSsd, 200)}});
    ASSERT_EQ(2u, quota.types.size());
    EXPECT_EQ(300, quota.reference_capacity);
    EXPECT_EQ("auto", quota.types.at(kSsd).config.mode());
}


TEST_F(EffectiveQuotaTest, UnregisteredCandidatesDoNotContributeToAutoQuota) {
    const auto quota = ResolveEffectiveQuota(Group({Auto(kDram, 0.5), Auto(kSsd)}), {{"dram", Capacity(kDram, 100)}});
    EXPECT_EQ(50, quota.types.at(kDram).capacity);
    EXPECT_EQ(-1, quota.types.at(kSsd).capacity);
    const auto reclaim = quota.ForReclaim();
    ASSERT_EQ(1u, reclaim.quota_config().size());
    EXPECT_EQ(kDram, reclaim.quota_config().front().storage_spec());
}

TEST_F(EffectiveQuotaTest, PartialRefreshKeepsEachBackendsLastCapacity) {
    auto group = Group({Auto(kDram, 0.5), Auto(kSsd)});
    group.set_storage_candidates({"a", "b", "ssd"});
    auto samples = std::map<std::string, StorageCapacitySnapshot>{
        {"a", Capacity(kDram, 100)}, {"b", Capacity(kDram, 200)}, {"ssd", Capacity(kSsd, 20)}};
    samples["a"].refresh_code = EC_ERROR;
    auto quota = ResolveEffectiveQuota(group, samples);
    EXPECT_EQ(150, quota.types.at(kDram).capacity);
    EXPECT_FALSE(quota.types.at(kDram).refresh_success);
    EXPECT_EQ(2u, quota.ForReclaim().quota_config().size());

    samples["b"].capacity = 300;
    EXPECT_EQ(200, ResolveEffectiveQuota(group, samples).types.at(kDram).capacity);
    samples["a"].capacity = -1;
    quota = ResolveEffectiveQuota(group, samples);
    EXPECT_EQ(-1, quota.types.at(kDram).capacity);
    EXPECT_EQ(20, quota.types.at(kSsd).capacity);
    ASSERT_EQ(1u, quota.ForReclaim().quota_config().size());
    EXPECT_EQ(-1, quota.reference_capacity);

    samples["a"] = Capacity(kDram, 0);
    samples["b"] = Capacity(kDram, 0);
    quota = ResolveEffectiveQuota(group, samples);
    EXPECT_EQ(0, quota.types.at(kDram).capacity);
    EXPECT_EQ(2u, quota.ForReclaim().quota_config().size());
}

} // namespace kv_cache_manager
