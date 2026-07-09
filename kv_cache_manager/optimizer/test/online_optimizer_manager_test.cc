#include <climits>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/online_runtime/online_optimizer_manager.h"

namespace kv_cache_manager {

class OnlineOptimizerManagerTest : public TESTBASE {
protected:
    void SetUp() override {
        TESTBASE::SetUp();
        registry_ = std::make_shared<OptimizerRegistryManager>("");
        registry_->Init();
        mgr_ = std::make_shared<OnlineOptimizerManager>(registry_);
    }

    OptimizerInstanceGroup MakeGroup(const std::string &name = "g1",
                                     std::vector<double> caps = {1.0},
                                     const std::string &eviction_policy = "lru",
                                     bool enable_theoretical_max_cache = false,
                                     int64_t ttl_seconds = 0) {
        OptimizerInstanceGroup group;
        group.set_name(name);
        group.set_capacity_gb(caps);
        group.set_eviction_policy(eviction_policy);
        group.set_enable_theoretical_max_cache(enable_theoretical_max_cache);
        group.set_ttl_seconds(ttl_seconds);
        return group;
    }

    OptimizerInstanceInfo MakeInfo(const std::string &instance_id = "i1",
                                   const std::string &group_name = "g1",
                                   int32_t block_size = 16,
                                   int32_t linear_step = 1) {
        return OptimizerInstanceInfo(group_name,
                                     instance_id,
                                     block_size,
                                     MakeSpecs(),
                                     MakeGroups(),
                                     linear_step,
                                     OptimizerStateInfo("full", ""));
    }

    OptimizerInstanceInfo MakeHybridInfo(const std::string &instance_id = "i1",
                                         const std::string &group_name = "g1",
                                         int32_t block_size = 16,
                                         int32_t linear_step = 1) {
        return OptimizerInstanceInfo(group_name,
                                     instance_id,
                                     block_size,
                                     MakeHybridSpecs(),
                                     MakeHybridGroups(),
                                     linear_step,
                                     OptimizerStateInfo("F0", "L1"));
    }

    std::vector<LocationSpecInfo> MakeSpecs() { return {LocationSpecInfo("tp0", 8192), LocationSpecInfo("tp1", 8192)}; }

    std::vector<LocationSpecGroup> MakeGroups() { return {LocationSpecGroup("full", {"tp0", "tp1"})}; }

    std::vector<LocationSpecInfo> MakeHybridSpecs() {
        return {
            LocationSpecInfo("tp0_F0", 8192),
            LocationSpecInfo("tp1_F0", 8192),
            LocationSpecInfo("tp0_L1", 2048),
            LocationSpecInfo("tp1_L1", 2048),
        };
    }

    std::vector<LocationSpecGroup> MakeHybridGroups() {
        return {
            LocationSpecGroup("F0", {"tp0_F0", "tp1_F0"}),
            LocationSpecGroup("L1", {"tp0_L1", "tp1_L1"}),
        };
    }

    ErrorCode RegisterGroup(const OptimizerInstanceGroup &group) {
        if (registry_->GetInstanceGroup(group.name())) {
            return registry_->UpdateInstanceGroup(group);
        }
        return registry_->CreateInstanceGroup(group);
    }

    ErrorCode RegisterInstance(const OptimizerInstanceInfo &info,
                               const OptimizerInstanceGroup &group,
                               RegisterInstanceResult &result) {
        ErrorCode ec = RegisterGroup(group);
        if (ec != EC_OK) {
            return ec;
        }
        return mgr_->RegisterInstance(info, result);
    }

    static ErrorCode RegisterInstance(const std::shared_ptr<OptimizerRegistryManager> &registry,
                                      const std::shared_ptr<OnlineOptimizerManager> &mgr,
                                      const OptimizerInstanceInfo &info,
                                      const OptimizerInstanceGroup &group,
                                      RegisterInstanceResult &result) {
        ErrorCode ec = registry->CreateInstanceGroup(group);
        if (ec != EC_OK) {
            return ec;
        }
        return mgr->RegisterInstance(info, result);
    }

    std::shared_ptr<OptimizerRegistryManager> registry_;
    std::shared_ptr<OnlineOptimizerManager> mgr_;
};

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceBasic) {
    auto info = MakeInfo();
    auto group = MakeGroup();
    RegisterInstanceResult result;

    ErrorCode ec = RegisterInstance(info, group, result);
    EXPECT_EQ(EC_OK, ec);
    EXPECT_EQ(16384, result.size_full_only);
    EXPECT_EQ(16384, result.size_full_linear);
    EXPECT_EQ(1, result.estimated_capacity_blocks.size());
}

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceHybrid) {
    auto info = MakeHybridInfo("i1", "g1", 16, 3);
    auto group = MakeGroup("g1", {1.0});
    RegisterInstanceResult result;

    ErrorCode ec = RegisterInstance(info, group, result);
    EXPECT_EQ(EC_OK, ec);
    EXPECT_EQ(16384, result.size_full_only);
    EXPECT_EQ(20480, result.size_full_linear);
}

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceEmptyIdFails) {
    auto info = MakeInfo("");
    auto group = MakeGroup();
    RegisterInstanceResult result;
    EXPECT_EQ(EC_BADARGS, RegisterInstance(info, group, result));
}

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceEmptySpecsFails) {
    OptimizerInstanceInfo info("g1", "i1", 16, {}, {});
    auto group = MakeGroup();
    RegisterInstanceResult result;
    EXPECT_EQ(EC_BADARGS, RegisterInstance(info, group, result));
}

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceMissingOptimizerStateInfoFails) {
    OptimizerInstanceInfo info("g1", "i1", 16, MakeSpecs(), MakeGroups());
    auto group = MakeGroup();
    RegisterInstanceResult result;
    EXPECT_EQ(EC_BADARGS, RegisterInstance(info, group, result));
}

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceMissingFullGroupFails) {
    OptimizerInstanceInfo info("g1", "i1", 16, MakeSpecs(), MakeGroups(), 1, OptimizerStateInfo("missing", ""));
    auto group = MakeGroup();
    RegisterInstanceResult result;
    EXPECT_EQ(EC_BADARGS, RegisterInstance(info, group, result));
}

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceMissingSpecInStateGroupFails) {
    std::vector<LocationSpecGroup> groups = {LocationSpecGroup("full", {"tp0", "tp_missing"})};
    OptimizerInstanceInfo info("g1", "i1", 16, MakeSpecs(), groups, 1, OptimizerStateInfo("full", ""));
    auto group = MakeGroup();
    RegisterInstanceResult result;
    EXPECT_EQ(EC_BADARGS, RegisterInstance(info, group, result));
}

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceSharedGroupQuotaFails) {
    auto info = MakeInfo();
    auto group = MakeGroup();
    group.set_shared_group_quota(true);
    RegisterInstanceResult result;
    EXPECT_EQ(EC_BADARGS, RegisterInstance(info, group, result));
}

TEST_F(OnlineOptimizerManagerTest, RemoveInstance) {
    auto info = MakeInfo();
    auto group = MakeGroup();
    RegisterInstanceResult result;
    RegisterInstance(info, group, result);

    EXPECT_EQ(EC_OK, mgr_->RemoveInstance("i1"));
    EXPECT_EQ(EC_INSTANCE_NOT_EXIST, mgr_->RemoveInstance("i1"));
}

TEST_F(OnlineOptimizerManagerTest, TraceQueryBasic) {
    auto info = MakeInfo();
    auto group = MakeGroup();
    RegisterInstanceResult reg_result;
    RegisterInstance(info, group, reg_result);

    std::vector<int64_t> keys = {1, 2, 3, 4, 5};
    TraceQueryResult result;
    EXPECT_EQ(EC_OK, mgr_->TraceQuery("i1", keys, result));
    EXPECT_EQ(0, result.cache_hit_count);
    EXPECT_EQ(5, result.total_blocks);

    EXPECT_EQ(EC_OK, mgr_->TraceQuery("i1", keys, result));
    EXPECT_EQ(5, result.cache_hit_count);
    EXPECT_EQ(5, result.total_blocks);
}

TEST_F(OnlineOptimizerManagerTest, TraceQueryPrefixMatch) {
    auto info = MakeInfo();
    auto group = MakeGroup();
    RegisterInstanceResult reg_result;
    RegisterInstance(info, group, reg_result);

    TraceQueryResult dummy;
    mgr_->TraceQuery("i1", {1, 2, 3, 4, 5}, dummy);

    TraceQueryResult result;
    mgr_->TraceQuery("i1", {1, 2, 3, 100, 200}, result);
    EXPECT_EQ(3, result.cache_hit_count);
}

TEST_F(OnlineOptimizerManagerTest, TraceQueryNonExistentInstance) {
    TraceQueryResult result;
    EXPECT_EQ(EC_INSTANCE_NOT_EXIST, mgr_->TraceQuery("nonexistent", {1}, result));
}

TEST_F(OnlineOptimizerManagerTest, TraceQueryMultipleCapacities) {
    auto info = MakeInfo();
    auto group = MakeGroup("g1", {0.0001, 1.0});
    RegisterInstanceResult reg_result;
    RegisterInstance(info, group, reg_result);

    EXPECT_EQ(2, reg_result.estimated_capacity_blocks.size());

    std::vector<int64_t> init_keys;
    for (int64_t i = 0; i < 100; i++) {
        init_keys.push_back(i);
    }
    TraceQueryResult dummy;
    mgr_->TraceQuery("i1", init_keys, dummy);

    TraceQueryResult result;
    mgr_->TraceQuery("i1", init_keys, result);
    // cache_hit_count uses index 0 (smallest capacity ~6 blocks), prefix match starts at key 0
    // whose stack distance (99) exceeds the small capacity, so prefix hit = 0
    EXPECT_EQ(0, result.cache_hit_count);
    // Large capacity (index 1) should hit all 100 keys
    ASSERT_EQ(2, result.hit_count_per_capacity.size());
    EXPECT_EQ(100, result.hit_count_per_capacity[1]);
}

TEST_F(OnlineOptimizerManagerTest, ListInstances) {
    RegisterInstanceResult result;
    RegisterInstance(MakeInfo("i1", "g1"), MakeGroup("g1"), result);
    RegisterInstance(MakeInfo("i2", "g1"), MakeGroup("g1"), result);
    RegisterInstance(MakeInfo("i3", "g2"), MakeGroup("g2"), result);

    std::vector<InstanceSummary> summaries;
    mgr_->ListInstances("", summaries);
    EXPECT_EQ(3, summaries.size());

    mgr_->ListInstances("g1", summaries);
    EXPECT_EQ(2, summaries.size());

    mgr_->ListInstances("g2", summaries);
    EXPECT_EQ(1, summaries.size());
}

TEST_F(OnlineOptimizerManagerTest, ResetStats) {
    auto info = MakeInfo();
    auto group = MakeGroup();
    RegisterInstanceResult reg_result;
    RegisterInstance(info, group, reg_result);

    TraceQueryResult result;
    mgr_->TraceQuery("i1", {1, 2, 3}, result);
    mgr_->TraceQuery("i1", {1, 2, 3}, result);

    std::vector<InstanceSummary> summaries;
    mgr_->ListInstances("", summaries);
    EXPECT_EQ(1, summaries.size());
    EXPECT_EQ(2, summaries[0].total_queries);
    EXPECT_EQ(6, summaries[0].total_blocks_queried);

    EXPECT_EQ(EC_OK, mgr_->ResetStats("i1"));

    mgr_->ListInstances("", summaries);
    EXPECT_EQ(0, summaries[0].total_queries);
    EXPECT_EQ(0, summaries[0].total_blocks_queried);
    EXPECT_EQ(0, summaries[0].unique_keys);
}

TEST_F(OnlineOptimizerManagerTest, ResetStatsNonExistent) {
    EXPECT_EQ(EC_INSTANCE_NOT_EXIST, mgr_->ResetStats("nonexistent"));
}

TEST_F(OnlineOptimizerManagerTest, LruIndexerType) {
    auto info = MakeInfo();
    auto group = MakeGroup("g1", {1.0}, "lru");
    RegisterInstanceResult reg_result;
    RegisterInstance(info, group, reg_result);

    TraceQueryResult result;
    mgr_->TraceQuery("i1", {1, 2, 3, 4, 5}, result);
    EXPECT_EQ(0, result.cache_hit_count);
    EXPECT_EQ(5, result.total_blocks);
    EXPECT_EQ(5, result.current_unique_keys);

    mgr_->TraceQuery("i1", {1, 2, 3, 4, 5}, result);
    EXPECT_EQ(5, result.cache_hit_count);
    EXPECT_EQ(5, result.current_unique_keys);
}

TEST_F(OnlineOptimizerManagerTest, CapacityEvictionLimitsUniqueCount) {
    auto info = MakeInfo();
    auto group = MakeGroup("g1", {0.00005});
    RegisterInstanceResult reg_result;
    RegisterInstance(info, group, reg_result);

    TraceQueryResult result;
    mgr_->TraceQuery("i1", {1, 2, 3, 4, 5, 6, 7}, result);
    EXPECT_LE(result.current_unique_keys, 5);
}

TEST_F(OnlineOptimizerManagerTest, LargeCapacityNotTruncatedBySmallCapacity) {
    auto info = MakeInfo();
    auto group = MakeGroup("g1", {0.00005, 1.0});
    RegisterInstanceResult reg_result;
    RegisterInstance(info, group, reg_result);

    std::vector<int64_t> keys;
    for (int64_t i = 0; i < 100; i++) {
        keys.push_back(i);
    }
    TraceQueryResult dummy;
    mgr_->TraceQuery("i1", keys, dummy);

    TraceQueryResult result;
    mgr_->TraceQuery("i1", keys, result);
    // Large capacity (index 1) should hit all 100 keys even when a smaller capacity is present.
    ASSERT_EQ(2, result.hit_count_per_capacity.size());
    EXPECT_EQ(100, result.hit_count_per_capacity[1]);
}

TEST_F(OnlineOptimizerManagerTest, LruIndexerMaxKeyCountUnlimited) {
    auto info = MakeInfo();
    auto group = MakeGroup("g1", {1.0}, "lru", false);
    RegisterInstanceResult reg_result;
    RegisterInstance(info, group, reg_result);

    std::vector<int64_t> keys;
    for (int64_t i = 0; i < 200; i++) {
        keys.push_back(i);
    }
    TraceQueryResult result;
    mgr_->TraceQuery("i1", keys, result);
    EXPECT_EQ(200, result.current_unique_keys);

    mgr_->TraceQuery("i1", keys, result);
    EXPECT_EQ(200, result.cache_hit_count);
    EXPECT_EQ(200, result.current_unique_keys);
}

TEST_F(OnlineOptimizerManagerTest, ReRegisterReplacesPrevious) {
    auto info = MakeInfo("i1", "g1", 16);
    auto group = MakeGroup();
    RegisterInstanceResult result;
    RegisterInstance(info, group, result);

    TraceQueryResult tr;
    mgr_->TraceQuery("i1", {1, 2, 3}, tr);

    auto info2 = MakeInfo("i1", "g1", 32);
    RegisterInstance(info2, group, result);

    std::vector<InstanceSummary> summaries;
    mgr_->ListInstances("", summaries);
    EXPECT_EQ(1, summaries.size());
    EXPECT_EQ(0, summaries[0].total_queries);
    EXPECT_EQ(32, summaries[0].block_size);
}

TEST_F(OnlineOptimizerManagerTest, ListInstancesPerCapacityHitRates) {
    auto info = MakeInfo();
    auto group = MakeGroup("g1", {1.0, 5.0});
    RegisterInstanceResult reg_result;
    RegisterInstance(info, group, reg_result);

    TraceQueryResult result;
    mgr_->TraceQuery("i1", {1, 2, 3}, result);
    mgr_->TraceQuery("i1", {1, 2, 3}, result);

    std::vector<InstanceSummary> summaries;
    mgr_->ListInstances("", summaries);
    ASSERT_EQ(1, summaries.size());
    ASSERT_EQ(2, summaries[0].per_capacity_hit_rates.size());

    EXPECT_DOUBLE_EQ(1.0, summaries[0].per_capacity_hit_rates[0].capacity_gb);
    EXPECT_DOUBLE_EQ(5.0, summaries[0].per_capacity_hit_rates[1].capacity_gb);

    EXPECT_EQ(6, summaries[0].total_blocks_queried);
    EXPECT_GT(summaries[0].per_capacity_hit_rates[1].total_hits, 0);
    EXPECT_GE(summaries[0].per_capacity_hit_rates[1].hit_rate, 0.0);
    EXPECT_LE(summaries[0].per_capacity_hit_rates[1].hit_rate, 1.0);
}

TEST_F(OnlineOptimizerManagerTest, ReRegisterFailurePreservesOldRecord) {
    // Create a manager with a real registry_manager to test persistence rollback
    auto registry = std::make_shared<OptimizerRegistryManager>("");
    registry->Init();
    auto mgr = std::make_shared<OnlineOptimizerManager>(registry);

    // Register an instance successfully
    auto info = MakeInfo("i1", "g1", 16);
    auto group = MakeGroup();
    RegisterInstanceResult result;
    EXPECT_EQ(EC_OK, RegisterInstance(registry, mgr, info, group, result));

    // Verify instance is in registry
    auto saved = registry->GetInstanceInfo("i1");
    ASSERT_NE(nullptr, saved);
    EXPECT_EQ(16, saved->block_size());

    // Try to re-register with empty specs (should fail)
    OptimizerInstanceInfo bad_info("g1", "i1", 32, {}, {});
    RegisterInstanceResult result2;
    EXPECT_EQ(EC_BADARGS, mgr->RegisterInstance(bad_info, result2));

    // Verify old instance info is still in registry (not deleted)
    auto restored = registry->GetInstanceInfo("i1");
    ASSERT_NE(nullptr, restored);
    EXPECT_EQ(16, restored->block_size());

    // Verify in-memory state is still valid (can still TraceQuery)
    TraceQueryResult tr;
    EXPECT_EQ(EC_OK, mgr->TraceQuery("i1", {1, 2, 3}, tr));
}

} // namespace kv_cache_manager
