#include <climits>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/manager/online_runtime/online_optimizer_manager.h"

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
                                   int32_t linear_step = 0) {
        // Full-only specs: linear_step > 0 would require a Mamba spec group,
        // use MakeHybridInfo for linear instances.
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
                                         int32_t linear_step = -1) {
        if (linear_step < 0) {
            linear_step = block_size;
        }
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

    static double CapacityGbForBytes(uint64_t capacity_bytes) {
        constexpr double kBytesPerGb = 1024.0 * 1024.0 * 1024.0;
        return static_cast<double>(capacity_bytes) / kBytesPerGb;
    }

    static double FullCapacityGb(int64_t capacity_blocks) {
        constexpr uint64_t kFullBlockChargeBytes = 16384;
        return CapacityGbForBytes(static_cast<uint64_t>(capacity_blocks) * kFullBlockChargeBytes);
    }
};

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceBasic) {
    auto info = MakeInfo();
    auto group = MakeGroup();
    RegisterInstanceResult result;

    ErrorCode ec = RegisterInstance(info, group, result);
    EXPECT_EQ(EC_OK, ec);
    EXPECT_EQ(16384, result.full_charge_bytes);
    EXPECT_EQ(0, result.linear_charge_bytes);
    EXPECT_EQ(1, result.estimated_capacity_blocks.size());
}

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceHybrid) {
    // 48 tokens / 16 tokens-per-block = one Linear state every 3 blocks.
    auto info = MakeHybridInfo("i1", "g1", 16, 48);
    auto group = MakeGroup("g1", {1.0});
    RegisterInstanceResult result;

    ErrorCode ec = RegisterInstance(info, group, result);
    EXPECT_EQ(EC_OK, ec);
    EXPECT_EQ(16384, result.full_charge_bytes);
    EXPECT_EQ(4096, result.linear_charge_bytes);
    // Estimate only (hits run on the byte axis): a shared pool spends
    // 3 * 16384 + 4096 = 53248 bytes per 3 blocks, so 1 GB holds about
    // floor(1073741824 * 3 / 53248) blocks.
    ASSERT_EQ(1, result.estimated_capacity_blocks.size());
    EXPECT_EQ(60494, result.estimated_capacity_blocks[0]);
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
    OptimizerInstanceInfo info("g1", "i1", 16, MakeSpecs(), MakeGroups(), 16, OptimizerStateInfo("missing", ""));
    auto group = MakeGroup();
    RegisterInstanceResult result;
    EXPECT_EQ(EC_BADARGS, RegisterInstance(info, group, result));
}

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceMissingSpecInStateGroupFails) {
    std::vector<LocationSpecGroup> groups = {LocationSpecGroup("full", {"tp0", "tp_missing"})};
    OptimizerInstanceInfo info("g1", "i1", 16, MakeSpecs(), groups, 16, OptimizerStateInfo("full", ""));
    auto group = MakeGroup();
    RegisterInstanceResult result;
    EXPECT_EQ(EC_BADARGS, RegisterInstance(info, group, result));
}

TEST_F(OnlineOptimizerManagerTest, RegisterInstanceLinearStepNotTokenMultipleFails) {
    // linear_step counts tokens and must divide into whole blocks.
    auto info = MakeHybridInfo("i1", "g1", 16, /*linear_step tokens=*/24);
    auto group = MakeGroup();
    RegisterInstanceResult result;
    EXPECT_EQ(EC_BADARGS, RegisterInstance(info, group, result));

    auto ok_info = MakeHybridInfo("i1", "g1", 16, /*linear_step tokens=*/32);
    EXPECT_EQ(EC_OK, RegisterInstance(ok_info, group, result));

    // A linear instance without a Mamba spec group is rejected.
    auto no_mamba_group = MakeInfo("i2", "g1", 16, /*linear_step tokens=*/32);
    EXPECT_EQ(EC_BADARGS, RegisterInstance(no_mamba_group, group, result));
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
    EXPECT_EQ(0, result.hit_count_per_capacity.at(0));
    EXPECT_EQ(5, result.total_blocks);

    EXPECT_EQ(EC_OK, mgr_->TraceQuery("i1", keys, result));
    EXPECT_EQ(5, result.hit_count_per_capacity.at(0));
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
    EXPECT_EQ(3, result.hit_count_per_capacity.at(0));
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
    // Full-attention LiteHit path with tail-first commit: the chain head is
    // most recent, so the small capacity (~6 blocks) serves exactly its
    // capacity as prefix hits.
    ASSERT_EQ(2, result.hit_count_per_capacity.size());
    EXPECT_EQ(reg_result.estimated_capacity_blocks[0], result.hit_count_per_capacity.at(0));
    // Large capacity (index 1) should hit all 100 keys
    EXPECT_EQ(100, result.hit_count_per_capacity[1]);
}

TEST_F(OnlineOptimizerManagerTest, FullAttentionStoresCapacityInBytes) {
    constexpr uint64_t kFullChargeBytes = 16384;
    constexpr uint64_t kCapacityBytes = 2 * kFullChargeBytes - 1;
    auto info = MakeInfo("i1", "g1", 4, 0);
    auto group = MakeGroup("g1", {CapacityGbForBytes(kCapacityBytes)});
    RegisterInstanceResult reg_result;
    ASSERT_EQ(EC_OK, RegisterInstance(info, group, reg_result));
    EXPECT_EQ((std::vector<int64_t>{1}), reg_result.estimated_capacity_blocks);

    ASSERT_EQ(EC_OK, mgr_->GetInstanceState("i1", [&](const InstanceState &state) {
        EXPECT_EQ((std::vector<uint64_t>{kCapacityBytes}), state.capacity_bytes);
    }));

    TraceQueryResult result;
    ASSERT_EQ(EC_OK, mgr_->TraceQuery("i1", {1, 2}, 8, result));
    ASSERT_EQ(EC_OK, mgr_->TraceQuery("i1", {1, 2}, 8, result));
    EXPECT_EQ((std::vector<int64_t>{1}), result.hit_count_per_capacity);
    EXPECT_EQ((std::vector<int64_t>{1}), result.unique_keys_per_capacity);
}

TEST_F(OnlineOptimizerManagerTest, FullAttentionUsesLiteHitTokenRates) {
    auto info = MakeInfo("i1", "g1", 4, 0);
    auto group = MakeGroup("g1", {FullCapacityGb(2), FullCapacityGb(3)}, "lru", true);
    RegisterInstanceResult reg_result;
    ASSERT_EQ(EC_OK, RegisterInstance(info, group, reg_result));
    EXPECT_EQ((std::vector<int64_t>{2, 3}), reg_result.estimated_capacity_blocks);

    TraceQueryResult first;
    ASSERT_EQ(EC_OK, mgr_->TraceQuery("i1", {1, 2, 3}, 13, first));
    EXPECT_EQ((std::vector<int64_t>{0, 0}), first.hit_count_per_capacity);

    TraceQueryResult second;
    ASSERT_EQ(EC_OK, mgr_->TraceQuery("i1", {1, 2, 3}, 13, second));
    EXPECT_EQ((std::vector<int64_t>{2, 3}), second.hit_count_per_capacity);
    ASSERT_EQ(2, second.hit_rate_per_capacity.size());
    EXPECT_DOUBLE_EQ(8.0 / 13.0, second.hit_rate_per_capacity[0]);
    EXPECT_DOUBLE_EQ(12.0 / 13.0, second.hit_rate_per_capacity[1]);
    EXPECT_EQ(3, second.max_hit_count);
    EXPECT_DOUBLE_EQ(12.0 / 13.0, second.max_hit_rate);

    bool checked_state = false;
    ASSERT_EQ(EC_OK, mgr_->GetInstanceState("i1", [&](const InstanceState &state) {
        checked_state = true;
        EXPECT_NE(nullptr, state.lite_hit);
        EXPECT_EQ(2, state.total_queries);
        EXPECT_EQ(26, state.total_input_tokens);
    }));
    EXPECT_TRUE(checked_state);

    std::vector<InstanceSummary> summaries;
    ASSERT_EQ(EC_OK, mgr_->ListInstances("g1", summaries));
    ASSERT_EQ(1, summaries.size());
    EXPECT_EQ(2, summaries[0].total_queries);
    EXPECT_EQ(26, summaries[0].total_input_tokens);
    ASSERT_EQ(2, summaries[0].per_capacity_hit_rates.size());
    EXPECT_EQ(2, summaries[0].per_capacity_hit_rates[0].total_hits);
    EXPECT_DOUBLE_EQ(8.0 / 26.0, summaries[0].per_capacity_hit_rates[0].hit_rate);
    EXPECT_EQ(3, summaries[0].per_capacity_hit_rates[1].total_hits);
    EXPECT_DOUBLE_EQ(12.0 / 26.0, summaries[0].per_capacity_hit_rates[1].hit_rate);
    EXPECT_DOUBLE_EQ(12.0 / 26.0, summaries[0].max_hit_rate);
}

TEST_F(OnlineOptimizerManagerTest, MambaLinearUsesSharedLiteHit) {
    // block_size 16, linear_step 48 tokens -> one Linear state every 3 blocks
    // plus the forced last block. Hybrid specs: full charge 16384, mamba
    // charge 4096 per Linear state.
    auto info = MakeHybridInfo("i1", "g1", 16, 48);
    auto group = MakeGroup("g1", {1.0}, "lru", /*enable_theoretical_max_cache=*/true);
    RegisterInstanceResult reg_result;
    ASSERT_EQ(EC_OK, RegisterInstance(info, group, reg_result));

    // An empty working set has no per-resident-block average yet.
    {
        std::vector<InstanceSummary> empty_summaries;
        ASSERT_EQ(EC_OK, mgr_->ListInstances("g1", empty_summaries));
        ASSERT_EQ(1u, empty_summaries.size());
        EXPECT_DOUBLE_EQ(0.0, empty_summaries[0].bytes_per_block);
    }

    TraceQueryResult first;
    ASSERT_EQ(EC_OK, mgr_->TraceQuery("i1", {1, 2, 3, 4}, 70, first));
    ASSERT_EQ(1, first.hit_count_per_capacity.size());
    EXPECT_EQ(0, first.hit_count_per_capacity[0]);
    EXPECT_EQ(0, first.max_hit_count);

    TraceQueryResult second;
    ASSERT_EQ(EC_OK, mgr_->TraceQuery("i1", {1, 2, 3, 4}, 70, second));
    // 1 GB covers everything: the forced tail Linear state (position 3)
    // recovers all 4 complete blocks.
    EXPECT_EQ(4, second.hit_count_per_capacity[0]);
    EXPECT_DOUBLE_EQ(64.0 / 70.0, second.hit_rate_per_capacity[0]);
    EXPECT_EQ(4, second.max_hit_count);
    EXPECT_DOUBLE_EQ(64.0 / 70.0, second.max_hit_rate);
    EXPECT_EQ(4, second.theoretical_unique_keys); // Full objects only

    bool checked_state = false;
    ASSERT_EQ(EC_OK, mgr_->GetInstanceState("i1", [&](const InstanceState &state) {
        checked_state = true;
        ASSERT_NE(nullptr, state.lite_hit);
        EXPECT_TRUE(state.lite_hit->uses_linear());
        EXPECT_EQ((std::vector<uint64_t>{1ULL << 30}), state.capacity_bytes);
    }));
    EXPECT_TRUE(checked_state);

    std::vector<InstanceSummary> summaries;
    ASSERT_EQ(EC_OK, mgr_->ListInstances("g1", summaries));
    ASSERT_EQ(1, summaries.size());
    EXPECT_EQ(2, summaries[0].total_queries);
    EXPECT_EQ(140, summaries[0].total_input_tokens);
    EXPECT_EQ(4, summaries[0].unique_keys);
    // Working set: 4 Full * 16384 + 2 Linear states (positions 2 and 3) * 4096.
    EXPECT_EQ(4 * 16384 + 2 * 4096, summaries[0].kv_cache_usage_bytes);
    EXPECT_DOUBLE_EQ(static_cast<double>(4 * 16384 + 2 * 4096) / 4, summaries[0].bytes_per_block);
    ASSERT_EQ(1, summaries[0].per_capacity_hit_rates.size());
    EXPECT_EQ(4, summaries[0].per_capacity_hit_rates[0].total_hits);
    EXPECT_DOUBLE_EQ(64.0 / 140.0, summaries[0].per_capacity_hit_rates[0].hit_rate);

    ASSERT_EQ(EC_OK, mgr_->ResetStats("i1"));
    TraceQueryResult after_reset;
    ASSERT_EQ(EC_OK, mgr_->TraceQuery("i1", {1, 2, 3, 4}, 70, after_reset));
    EXPECT_EQ(0, after_reset.hit_count_per_capacity[0]);
}

TEST_F(OnlineOptimizerManagerTest, MambaCountsHistoricalForcedTailLinearState) {
    // block_size 16, linear_step 48 -> periodic Linear states every 3 blocks.
    auto info = MakeHybridInfo("i1", "g1", 16, 48);
    auto group = MakeGroup("g1", {1.0}, "lru", /*enable_theoretical_max_cache=*/true);
    RegisterInstanceResult reg_result;
    ASSERT_EQ(EC_OK, RegisterInstance(info, group, reg_result));

    TraceQueryResult first;
    ASSERT_EQ(EC_OK, mgr_->TraceQuery("i1", {1, 2}, 32, first));
    EXPECT_EQ(0, first.hit_count_per_capacity[0]);
    EXPECT_EQ(0, first.max_hit_count);

    TraceQueryResult second;
    ASSERT_EQ(EC_OK, mgr_->TraceQuery("i1", {1, 2, 3, 4}, 64, second));
    // key 2 has a Linear state solely as the first request's forced tail. It is
    // not scheduled in this request, but remains a valid
    // restore point and must contribute two hit blocks to both statistics.
    EXPECT_EQ(2, second.hit_count_per_capacity[0]);
    EXPECT_DOUBLE_EQ(0.5, second.hit_rate_per_capacity[0]);
    EXPECT_EQ(2, second.max_hit_count);
    EXPECT_DOUBLE_EQ(0.5, second.max_hit_rate);

    std::vector<InstanceSummary> summaries;
    ASSERT_EQ(EC_OK, mgr_->ListInstances("g1", summaries));
    ASSERT_EQ(1u, summaries.size());
    EXPECT_EQ(2, summaries[0].per_capacity_hit_rates[0].total_hits);
    EXPECT_DOUBLE_EQ(32.0 / 96.0, summaries[0].per_capacity_hit_rates[0].hit_rate);
    EXPECT_DOUBLE_EQ(32.0 / 96.0, summaries[0].max_hit_rate);
    // 4 Full blocks plus current Linear states 3/4 and historical forced-tail
    // Linear state 2: the real working-set average is (4F + 3M) / 4.
    EXPECT_EQ(4 * 16384 + 3 * 4096, summaries[0].kv_cache_usage_bytes);
    EXPECT_DOUBLE_EQ(static_cast<double>(4 * 16384 + 3 * 4096) / 4, summaries[0].bytes_per_block);
}

TEST_F(OnlineOptimizerManagerTest, FullAttentionRequiresConsistentInputTokenLength) {
    auto info = MakeInfo("i1", "g1", 4, 0);
    auto group = MakeGroup("g1", {FullCapacityGb(2)});
    RegisterInstanceResult reg_result;
    ASSERT_EQ(EC_OK, RegisterInstance(info, group, reg_result));

    TraceQueryResult result;
    EXPECT_EQ(EC_BADARGS, mgr_->TraceQuery("i1", {1}, 3, result));
    EXPECT_EQ(EC_BADARGS, mgr_->TraceQuery("i1", {}, 4, result));
    EXPECT_EQ(EC_OK, mgr_->TraceQuery("i1", {1}, 7, result));

    std::vector<InstanceSummary> summaries;
    mgr_->ListInstances("g1", summaries);
    ASSERT_EQ(1, summaries.size());
    EXPECT_EQ(1, summaries[0].total_queries);
    EXPECT_EQ(7, summaries[0].total_input_tokens);
}

TEST_F(OnlineOptimizerManagerTest, FullAttentionLayersGroupTtlOntoLiteHit) {
    auto info = MakeInfo("i1", "g1", 4, 0);
    auto group = MakeGroup("g1", {FullCapacityGb(2)}, "lru", /*enable_theoretical_max_cache=*/true, /*ttl=*/60);
    RegisterInstanceResult reg_result;
    ASSERT_EQ(EC_OK, RegisterInstance(info, group, reg_result));

    TraceQueryResult first;
    ASSERT_EQ(EC_OK, mgr_->TraceQuery("i1", {1, 2}, 8, first));
    EXPECT_EQ(0, first.max_hit_count);
    // The immediate re-query is well inside the 60s TTL: plain LRU behavior.
    TraceQueryResult second;
    ASSERT_EQ(EC_OK, mgr_->TraceQuery("i1", {1, 2}, 8, second));
    EXPECT_EQ(2, second.max_hit_count);
    EXPECT_EQ(2, second.hit_count_per_capacity.at(0));

    // Total-eviction contract: LiteHit has no capacity evictions, so the
    // summary total must always equal the TTL eviction count.
    std::vector<InstanceSummary> summaries;
    mgr_->ListInstances("g1", summaries);
    ASSERT_EQ(1, summaries.size());
    EXPECT_EQ(summaries[0].ttl_eviction_count, summaries[0].eviction_count);

    // Negative TTL is still rejected.
    auto bad_info = MakeInfo("i2", "g2", 4, 0);
    auto bad_group = MakeGroup("g2", {FullCapacityGb(2)}, "lru", false, /*ttl=*/-1);
    EXPECT_EQ(EC_BADARGS, RegisterInstance(bad_info, bad_group, reg_result));
}

TEST_F(OnlineOptimizerManagerTest, LinearInstanceLayersGroupTtlOntoSharedCore) {
    auto info = MakeHybridInfo("i1", "g1", 16, 48);
    auto ttl_group = MakeGroup("g1", {1.0}, "lru", /*enable_theoretical_max_cache=*/false, /*ttl=*/300);
    RegisterInstanceResult reg_result;
    ASSERT_EQ(EC_OK, RegisterInstance(info, ttl_group, reg_result));

    TraceQueryResult first;
    ASSERT_EQ(EC_OK, mgr_->TraceQuery("i1", {1}, 16, first));
    EXPECT_EQ(0, first.hit_count_per_capacity.at(0));
    TraceQueryResult second;
    ASSERT_EQ(EC_OK, mgr_->TraceQuery("i1", {1}, 16, second));
    EXPECT_EQ(1, second.hit_count_per_capacity.at(0));

    // Drive the shared watermark past the deadline without waiting in the UT.
    ASSERT_EQ(EC_OK,
              mgr_->GetInstanceState("i1", [](const InstanceState &state) { state.lite_hit->AdvanceTime(LLONG_MAX); }));
    std::vector<InstanceSummary> summaries;
    ASSERT_EQ(EC_OK, mgr_->ListInstances("g1", summaries));
    ASSERT_EQ(1u, summaries.size());
    EXPECT_EQ(0, summaries[0].unique_keys);
    EXPECT_EQ(0, summaries[0].kv_cache_usage_bytes);
    EXPECT_DOUBLE_EQ(0.0, summaries[0].bytes_per_block);
    EXPECT_EQ(1, summaries[0].ttl_eviction_count); // Full objects only
    EXPECT_EQ(summaries[0].ttl_eviction_count, summaries[0].eviction_count);
}

TEST_F(OnlineOptimizerManagerTest, ResetStatsResetsFullAttentionLiteHit) {
    auto info = MakeInfo("i1", "g1", 4, 0);
    auto group = MakeGroup("g1", {FullCapacityGb(2)});
    RegisterInstanceResult reg_result;
    ASSERT_EQ(EC_OK, RegisterInstance(info, group, reg_result));

    TraceQueryResult result;
    mgr_->TraceQuery("i1", {1, 2}, 9, result);
    mgr_->TraceQuery("i1", {1, 2}, 8, result);
    ASSERT_EQ(EC_OK, mgr_->ResetStats("i1"));

    std::vector<InstanceSummary> summaries;
    mgr_->ListInstances("g1", summaries);
    ASSERT_EQ(1, summaries.size());
    EXPECT_EQ(0, summaries[0].total_queries);
    EXPECT_EQ(0, summaries[0].total_input_tokens);
    EXPECT_EQ(0, summaries[0].total_blocks_queried);
    EXPECT_EQ(0, summaries[0].unique_keys);
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
    EXPECT_EQ(0, result.hit_count_per_capacity.at(0));
    EXPECT_EQ(5, result.total_blocks);
    EXPECT_EQ(5, result.unique_keys_per_capacity.at(0));

    mgr_->TraceQuery("i1", {1, 2, 3, 4, 5}, result);
    EXPECT_EQ(5, result.hit_count_per_capacity.at(0));
    EXPECT_EQ(5, result.unique_keys_per_capacity.at(0));
}

TEST_F(OnlineOptimizerManagerTest, CapacityEvictionLimitsUniqueCount) {
    auto info = MakeInfo();
    auto group = MakeGroup("g1", {0.00005});
    RegisterInstanceResult reg_result;
    RegisterInstance(info, group, reg_result);

    TraceQueryResult result;
    mgr_->TraceQuery("i1", {1, 2, 3, 4, 5, 6, 7}, result);
    EXPECT_LE(result.unique_keys_per_capacity.at(0), 5);
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
    EXPECT_EQ(200, result.unique_keys_per_capacity.at(0));

    mgr_->TraceQuery("i1", keys, result);
    EXPECT_EQ(200, result.hit_count_per_capacity.at(0));
    EXPECT_EQ(200, result.unique_keys_per_capacity.at(0));
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
