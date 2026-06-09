#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/online_optimizer/manager/online_optimizer_manager.h"

#include <climits>
#include <vector>

namespace kv_cache_manager {

class OnlineOptimizerManagerTest : public TESTBASE {
protected:
    void SetUp() override {
        TESTBASE::SetUp();
        mgr_ = std::make_shared<OnlineOptimizerManager>(nullptr);
    }

    OptimizerInstanceGroup MakeGroup(const std::string &name = "g1",
                                      std::vector<double> caps = {1.0},
                                      const std::string &indexer_type = "fenwick_lru",
                                      int64_t max_key_count = 0) {
        OptimizerInstanceGroup group;
        group.set_name(name);
        group.set_enabled(true);
        group.set_capacity_gb(caps);
        group.set_indexer_type(indexer_type);
        group.set_max_key_count(max_key_count);
        return group;
    }

    OptimizerInstanceInfo MakeInfo(const std::string &instance_id = "i1",
                                    const std::string &group_name = "g1",
                                    int32_t block_size = 16,
                                    int32_t linear_step = 1,
                                    const std::string &full_group_name = "") {
        return OptimizerInstanceInfo(group_name, instance_id, block_size, MakeSpecs(), {},
                                     linear_step, full_group_name);
    }

    OptimizerInstanceInfo MakeHybridInfo(const std::string &instance_id = "i1",
                                          const std::string &group_name = "g1",
                                          int32_t block_size = 16,
                                          int32_t linear_step = 1,
                                          const std::string &full_group_name = "") {
        return OptimizerInstanceInfo(group_name, instance_id, block_size, MakeHybridSpecs(), MakeHybridGroups(),
                                     linear_step, full_group_name);
    }

    std::vector<LocationSpecInfo> MakeSpecs() {
        return {LocationSpecInfo("tp0", 8192), LocationSpecInfo("tp1", 8192)};
    }

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
            LocationSpecGroup("F0L1", {"tp0_F0", "tp0_L1", "tp1_F0", "tp1_L1"}),
        };
    }

    std::shared_ptr<OnlineOptimizerManager> mgr_;
};

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceBasic) {
    auto info = MakeInfo();
    auto group = MakeGroup();
    RegisterInstanceResult result;

    ErrorCode ec = mgr_->RegisterInstance(info, group, result);
    EXPECT_EQ(EC_OK, ec);
    EXPECT_EQ(16384, result.avg_bytes_per_block);
    EXPECT_EQ(16384, result.size_full_only);
    EXPECT_EQ(16384, result.size_full_linear);
    EXPECT_EQ(1, result.capacity_blocks.size());
}

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceHybrid) {
    auto info = MakeHybridInfo("i1", "g1", 16, 3, "F0");
    auto group = MakeGroup("g1", {1.0});
    RegisterInstanceResult result;

    ErrorCode ec = mgr_->RegisterInstance(info, group, result);
    EXPECT_EQ(EC_OK, ec);
    EXPECT_EQ(16384, result.size_full_only);
    EXPECT_EQ(20480, result.size_full_linear);
    EXPECT_EQ(17749, result.avg_bytes_per_block);
}

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceEmptyIdFails) {
    auto info = MakeInfo("");
    auto group = MakeGroup();
    RegisterInstanceResult result;
    EXPECT_EQ(EC_BADARGS, mgr_->RegisterInstance(info, group, result));
}

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceEmptySpecsFails) {
    OptimizerInstanceInfo info("g1", "i1", 16, {}, {});
    auto group = MakeGroup();
    RegisterInstanceResult result;
    EXPECT_EQ(EC_BADARGS, mgr_->RegisterInstance(info, group, result));
}

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceNotEnabledFails) {
    auto info = MakeInfo();
    OptimizerInstanceGroup group;
    group.set_name("g1");
    group.set_enabled(false);
    RegisterInstanceResult result;
    EXPECT_EQ(EC_BADARGS, mgr_->RegisterInstance(info, group, result));
}

TEST_F(OnlineOptimizerManagerTest, RemoveInstance) {
    auto info = MakeInfo();
    auto group = MakeGroup();
    RegisterInstanceResult result;
    mgr_->RegisterInstance(info, group, result);

    EXPECT_EQ(EC_OK, mgr_->RemoveInstance("i1"));
    EXPECT_EQ(EC_INSTANCE_NOT_EXIST, mgr_->RemoveInstance("i1"));
}

TEST_F(OnlineOptimizerManagerTest, TraceQueryBasic) {
    auto info = MakeInfo();
    auto group = MakeGroup();
    RegisterInstanceResult reg_result;
    mgr_->RegisterInstance(info, group, reg_result);

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
    mgr_->RegisterInstance(info, group, reg_result);

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
    auto group = MakeGroup("g1", {0.0001, 1.0}, "fenwick_lru", 0);
    RegisterInstanceResult reg_result;
    mgr_->RegisterInstance(info, group, reg_result);

    EXPECT_EQ(2, reg_result.capacity_blocks.size());

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
    mgr_->RegisterInstance(MakeInfo("i1", "g1"), MakeGroup("g1"), result);
    mgr_->RegisterInstance(MakeInfo("i2", "g1"), MakeGroup("g1"), result);
    mgr_->RegisterInstance(MakeInfo("i3", "g2"), MakeGroup("g2"), result);

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
    mgr_->RegisterInstance(info, group, reg_result);

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

TEST_F(OnlineOptimizerManagerTest, BSTIndexerType) {
    auto info = MakeInfo();
    auto group = MakeGroup("g1", {1.0}, "bst_lru", 0);
    RegisterInstanceResult reg_result;
    mgr_->RegisterInstance(info, group, reg_result);

    TraceQueryResult result;
    mgr_->TraceQuery("i1", {1, 2, 3, 4, 5}, result);
    EXPECT_EQ(0, result.cache_hit_count);
    EXPECT_EQ(5, result.total_blocks);
    EXPECT_EQ(5, result.current_unique_keys);

    mgr_->TraceQuery("i1", {1, 2, 3, 4, 5}, result);
    EXPECT_EQ(5, result.cache_hit_count);
    EXPECT_EQ(5, result.current_unique_keys);

}

TEST_F(OnlineOptimizerManagerTest, MaxKeyCountEviction) {
    auto info = MakeInfo();
    auto group = MakeGroup("g1", {0.00005}, "fenwick_lru", 5);
    RegisterInstanceResult reg_result;
    mgr_->RegisterInstance(info, group, reg_result);

    TraceQueryResult result;
    mgr_->TraceQuery("i1", {1, 2, 3, 4, 5, 6, 7}, result);
    EXPECT_LE(result.current_unique_keys, 5);
}

TEST_F(OnlineOptimizerManagerTest, MaxKeyCountAutoRaised) {
    // max_key_count=5, but capacity 1.0 GB -> ~65536 blocks.
    // Auto-raise should lift max_key_count to 65536 so large capacity tier is not truncated.
    auto info = MakeInfo();
    auto group = MakeGroup("g1", {0.00005, 1.0}, "fenwick_lru", 5);
    RegisterInstanceResult reg_result;
    mgr_->RegisterInstance(info, group, reg_result);

    std::vector<int64_t> keys;
    for (int64_t i = 0; i < 100; i++) {
        keys.push_back(i);
    }
    TraceQueryResult dummy;
    mgr_->TraceQuery("i1", keys, dummy);

    TraceQueryResult result;
    mgr_->TraceQuery("i1", keys, result);
    // Large capacity (index 1) should hit all 100 keys — not truncated by max_key_count
    ASSERT_EQ(2, result.hit_count_per_capacity.size());
    EXPECT_EQ(100, result.hit_count_per_capacity[1]);
}

TEST_F(OnlineOptimizerManagerTest, LruIndexerMaxKeyCountUnlimited) {
    auto info = MakeInfo();
    auto group = MakeGroup("g1", {1.0}, "lru", 0);
    RegisterInstanceResult reg_result;
    mgr_->RegisterInstance(info, group, reg_result);

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
    mgr_->RegisterInstance(info, group, result);

    TraceQueryResult tr;
    mgr_->TraceQuery("i1", {1, 2, 3}, tr);

    auto info2 = MakeInfo("i1", "g1", 32);
    mgr_->RegisterInstance(info2, group, result);

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
    mgr_->RegisterInstance(info, group, reg_result);

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

} // namespace kv_cache_manager
