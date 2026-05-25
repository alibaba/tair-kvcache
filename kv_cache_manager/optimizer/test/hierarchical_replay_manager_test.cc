#include <filesystem>
#include <fstream>
#include <sstream>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/hierarchical_replay_config.h"
#include "kv_cache_manager/optimizer/config/instance_config.h"
#include "kv_cache_manager/optimizer/config/instance_group_config.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/config/tier_config.h"
#include "kv_cache_manager/optimizer/manager/hierarchical_replay_manager.h"

using namespace kv_cache_manager;

class HierarchicalReplayManagerTest : public TESTBASE {
protected:
    HierarchicalReplayConfig CreateHierarchicalConfig(const std::string &root) {
        HierarchicalReplayConfig config;
        config.set_trace_file_path(root + "/trace.jsonl");
        config.set_output_result_path(root + "/combined");
        config.set_engine_config(
            CreateEngineOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm", "dram"}, root + "/engine"));
        config.set_pool_config(CreateOptimizerConfig("pool_group", {"model_l3"}, {"l3"}, root + "/pool"));

        EngineToPoolMappingConfig map_a;
        map_a.set_engine_instance_id("engine_a");
        map_a.set_pool_instance_id("model_l3");
        EngineToPoolMappingConfig map_b;
        map_b.set_engine_instance_id("engine_b");
        map_b.set_pool_instance_id("model_l3");
        config.set_engine_to_pool({map_a, map_b});
        return config;
    }

    OptimizerConfig CreateEngineOptimizerConfig(const std::string &group_prefix,
                                                const std::vector<std::string> &instance_ids,
                                                const std::vector<std::string> &tier_names,
                                                const std::string &output_path,
                                                int64_t tier_capacity = 1LL << 28) {
        OptimizerConfig config;
        config.set_trace_file_path("/tmp/unused_trace.jsonl");
        config.set_output_result_path(output_path);
        config.set_eviction_params(CreateEvictionConfig());

        std::vector<OptInstanceGroupConfig> groups;
        groups.reserve(instance_ids.size());
        for (const auto &instance_id : instance_ids) {
            groups.push_back(
                CreateInstanceGroup(group_prefix + "_" + instance_id, {instance_id}, tier_names, tier_capacity));
        }
        config.set_instance_groups(groups);
        return config;
    }

    OptimizerConfig CreateOptimizerConfig(const std::string &group_name,
                                          const std::vector<std::string> &instance_ids,
                                          const std::vector<std::string> &tier_names,
                                          const std::string &output_path,
                                          int64_t tier_capacity = 1LL << 28) {
        OptimizerConfig config;
        config.set_trace_file_path("/tmp/unused_trace.jsonl");
        config.set_output_result_path(output_path);
        config.set_eviction_params(CreateEvictionConfig());
        config.set_instance_groups({CreateInstanceGroup(group_name, instance_ids, tier_names, tier_capacity)});
        return config;
    }

    EvictionConfig CreateEvictionConfig() {
        EvictionConfig eviction_config;
        eviction_config.set_eviction_mode(EvictionMode::EVICTION_MODE_INSTANCE_PRECISE);
        eviction_config.set_eviction_batch_size_per_instance(10);
        return eviction_config;
    }

    OptInstanceGroupConfig CreateInstanceGroup(const std::string &group_name,
                                               const std::vector<std::string> &instance_ids,
                                               const std::vector<std::string> &tier_names,
                                               int64_t tier_capacity) {
        OptInstanceGroupConfig group;
        group.set_group_name(group_name);
        group.set_quota_capacity(1LL << 30);
        group.set_used_percentage(1.0);
        group.set_hierarchical_eviction_enabled(tier_names.size() > 1);

        std::vector<OptTierConfig> tiers;
        for (size_t idx = 0; idx < tier_names.size(); ++idx) {
            OptTierConfig tier;
            tier.set_unique_name(tier_names[idx]);
            tier.set_capacity(tier_capacity);
            tier.set_storage_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
            tier.set_band_width_mbps(1000);
            tier.set_priority(static_cast<int32_t>(idx));
            tiers.push_back(tier);
        }
        group.set_storages(tiers);

        std::vector<OptInstanceConfig> instances;
        for (const auto &instance_id : instance_ids) {
            OptInstanceConfig instance;
            instance.set_instance_id(instance_id);
            instance.set_instance_group_name(group_name);
            instance.set_block_size(16);
            instance.set_bytes_per_token(1);
            LruParams params;
            params.sample_rate = 1.0;
            instance.set_eviction_policy_param(params);
            instance.set_eviction_policy_type(EvictionPolicyType::POLICY_LRU);
            instances.push_back(instance);
        }
        group.set_instances(instances);
        return group;
    }
};

TEST_F(HierarchicalReplayManagerTest, LinksEngineInstancesToSharedPool) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> keys = {101, 102};
    auto cold = manager.GetCacheLocation("engine_a", "cold", 1000, keys, 32);
    EXPECT_EQ(cold.engine_hit_length, 0);
    EXPECT_EQ(cold.pool_hit_length, 0);
    EXPECT_EQ(cold.total_hit_length, 0);

    manager.WriteCache("engine_a", "write_a", 1001, keys);

    auto pool_hit = manager.GetCacheLocation("engine_b", "pool_hit", 2000, keys, 32);
    EXPECT_EQ(pool_hit.engine_hit_length, 0);
    EXPECT_EQ(pool_hit.pool_hit_length, 2);
    EXPECT_EQ(pool_hit.total_hit_length, 2);

    manager.WriteCache("engine_b", "write_b", 2001, keys);

    auto engine_hit = manager.GetCacheLocation("engine_b", "engine_hit", 3000, keys, 32);
    EXPECT_EQ(engine_hit.engine_hit_length, 2);
    EXPECT_EQ(engine_hit.pool_hit_length, 0);
    EXPECT_EQ(engine_hit.total_hit_length, 2);

    manager.AnalyzeResults();
    EXPECT_TRUE(std::filesystem::exists(root + "/combined/hierarchical_hit_rates.csv"));
}

TEST_F(HierarchicalReplayManagerTest, SelectiveL2L3WriteUsesEngineHitThreshold) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_selective_l2_l3";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    L2L3StrategyConfig strategy;
    strategy.set_write_mode(TierWriteMode::WRITE_THROUGH_SELECTIVE);
    strategy.set_selective_write_threshold(2);
    config.set_l2_l3_strategy(strategy);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> keys = {501};
    manager.WriteCache("engine_a", "write_a", 1000, keys);

    auto before_hot = manager.GetCacheLocation("engine_b", "before_hot", 1100, keys, 16);
    EXPECT_EQ(before_hot.total_hit_length, 0);

    auto first_engine_hit = manager.GetCacheLocation("engine_a", "first_engine_hit", 1200, keys, 16);
    EXPECT_EQ(first_engine_hit.engine_hit_length, 1);
    auto second_engine_hit = manager.GetCacheLocation("engine_a", "second_engine_hit", 1300, keys, 16);
    EXPECT_EQ(second_engine_hit.engine_hit_length, 1);

    auto pool_hit_after_hot = manager.GetCacheLocation("engine_b", "pool_hit_after_hot", 1400, keys, 16);
    EXPECT_EQ(pool_hit_after_hot.engine_hit_length, 0);
    EXPECT_EQ(pool_hit_after_hot.pool_hit_length, 1);
}

TEST_F(HierarchicalReplayManagerTest, CascadingL2L3WriteMovesEngineEvictionsToPool) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_cascading_l2_l3";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    config.set_engine_config(
        CreateEngineOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm", "dram"}, root + "/engine", 16));
    L2L3StrategyConfig strategy;
    strategy.set_write_mode(TierWriteMode::CASCADING);
    config.set_l2_l3_strategy(strategy);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> first = {601};
    const std::vector<int64_t> second = {602};
    manager.WriteCache("engine_a", "write_first", 1000, first);

    auto before_eviction = manager.GetCacheLocation("engine_b", "before_eviction", 1100, first, 16);
    EXPECT_EQ(before_eviction.total_hit_length, 0);

    manager.WriteCache("engine_a", "write_second", 1200, second);
    auto pool_hit_after_eviction = manager.GetCacheLocation("engine_b", "pool_hit_after_eviction", 1300, first, 16);
    EXPECT_EQ(pool_hit_after_eviction.engine_hit_length, 0);
    EXPECT_EQ(pool_hit_after_eviction.pool_hit_length, 1);
}

TEST_F(HierarchicalReplayManagerTest, L3HitPromotesToEngineWhenEnabled) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_l3_promote";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    L2L3StrategyConfig strategy;
    strategy.set_promote_enabled(true);
    config.set_l2_l3_strategy(strategy);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> keys = {701};
    manager.WriteCache("engine_a", "write_a", 1000, keys);

    auto pool_hit = manager.GetCacheLocation("engine_b", "pool_hit", 2000, keys, 16);
    EXPECT_EQ(pool_hit.engine_hit_length, 0);
    EXPECT_EQ(pool_hit.pool_hit_length, 1);

    auto promoted_engine_hit = manager.GetCacheLocation("engine_b", "promoted_engine_hit", 3000, keys, 16);
    EXPECT_EQ(promoted_engine_hit.engine_hit_length, 1);
    EXPECT_EQ(promoted_engine_hit.pool_hit_length, 0);
}

TEST_F(HierarchicalReplayManagerTest, EngineHitPropagatesAccessToL3WhenEnabled) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_l3_access_propagation";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    OptimizerConfig pool_config = CreateOptimizerConfig("pool_group", {"model_l3"}, {"l3"}, root + "/pool");
    pool_config.mutable_instance_groups()[0].set_quota_capacity(32);
    config.set_pool_config(pool_config);
    L2L3StrategyConfig strategy;
    strategy.set_access_propagation_enabled(true);
    config.set_l2_l3_strategy(strategy);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> first = {801};
    const std::vector<int64_t> second = {802};
    const std::vector<int64_t> third = {803};
    manager.WriteCache("engine_a", "write_first", 1000, first);
    manager.WriteCache("engine_a", "write_second", 1001, second);

    auto touch_first = manager.GetCacheLocation("engine_a", "touch_first", 1100, first, 16);
    EXPECT_EQ(touch_first.engine_hit_length, 1);

    manager.WriteCache("engine_a", "write_third", 1200, third);
    auto first_still_in_l3 = manager.GetCacheLocation("engine_b", "first_still_in_l3", 1300, first, 16);
    EXPECT_EQ(first_still_in_l3.pool_hit_length, 1);
    auto second_evicted_from_l3 = manager.GetCacheLocation("engine_b", "second_evicted_from_l3", 1400, second, 16);
    EXPECT_EQ(second_evicted_from_l3.total_hit_length, 0);
}

TEST_F(HierarchicalReplayManagerTest, RejectsSharedEngineInstanceGroup) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_shared_engine_group";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    config.set_engine_config(
        CreateOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm", "dram"}, root + "/engine"));

    HierarchicalReplayManager manager(config);
    EXPECT_FALSE(manager.Init());
}

TEST_F(HierarchicalReplayManagerTest, ReplaysStandardTraceThroughEngineAndPool) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_direct";
    std::filesystem::create_directories(root);
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);

    std::ofstream trace(config.trace_file_path());
    trace
        << R"({"type":"get","instance_id":"engine_a","trace_id":"cold","timestamp_ns":1000,"keys":[101,102],"input_len":32,"block_mask":[]})"
        << "\n";
    trace << R"({"type":"write","instance_id":"engine_a","trace_id":"write_a","timestamp_ns":1001,"keys":[101,102]})"
          << "\n";
    trace
        << R"({"type":"get","instance_id":"engine_b","trace_id":"pool_hit","timestamp_ns":2000,"keys":[101,102],"input_len":32,"block_mask":[]})"
        << "\n";
    trace << R"({"type":"write","instance_id":"engine_b","trace_id":"write_b","timestamp_ns":2001,"keys":[101,102]})"
          << "\n";
    trace
        << R"({"type":"get","instance_id":"engine_b","trace_id":"engine_hit","timestamp_ns":3000,"keys":[101,102],"input_len":32,"block_mask":[]})"
        << "\n";
    trace.close();

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());
    manager.DirectRun();
    manager.AnalyzeResults();

    std::ifstream csv(root + "/combined/hierarchical_hit_rates.csv");
    ASSERT_TRUE(csv.is_open());
    std::ostringstream buffer;
    buffer << csv.rdbuf();
    const std::string content = buffer.str();
    EXPECT_THAT(content, HasSubstr("pool_hit,engine_b,model_l3,2,0,2,2"));
    EXPECT_THAT(content, HasSubstr("engine_hit,engine_b,model_l3,2,2,0,2"));
}

TEST_F(HierarchicalReplayManagerTest, RoundRobinSchedulesTraceRequestsToEngineInstances) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_round_robin";
    std::filesystem::create_directories(root);
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    config.set_engine_scheduling_strategy("round_robin");

    std::ofstream trace(config.trace_file_path());
    trace
        << R"({"type":"get","instance_id":"logical_source","trace_id":"cold","timestamp_ns":1000,"keys":[201],"input_len":16,"block_mask":[]})"
        << "\n";
    trace << R"({"type":"write","instance_id":"logical_source","trace_id":"write_a","timestamp_ns":1001,"keys":[201]})"
          << "\n";
    trace
        << R"({"type":"get","instance_id":"logical_source","trace_id":"pool_hit","timestamp_ns":2000,"keys":[201],"input_len":16,"block_mask":[]})"
        << "\n";
    trace.close();

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());
    manager.DirectRun();
    manager.AnalyzeResults();

    std::ifstream csv(root + "/combined/hierarchical_hit_rates.csv");
    ASSERT_TRUE(csv.is_open());
    std::ostringstream buffer;
    buffer << csv.rdbuf();
    const std::string content = buffer.str();
    EXPECT_THAT(content, HasSubstr("cold,engine_a,model_l3,1,0,0,0"));
    EXPECT_THAT(content, HasSubstr("pool_hit,engine_b,model_l3,1,0,1,1"));
}

TEST_F(HierarchicalReplayManagerTest, PrefixHitSchedulesToCachedEngineInstance) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_prefix_hit";
    std::filesystem::create_directories(root);
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    config.set_engine_scheduling_strategy("prefix_hit");

    std::ofstream trace(config.trace_file_path());
    trace
        << R"({"type":"get","instance_id":"logical_source","trace_id":"cold","timestamp_ns":1000,"keys":[401],"input_len":16,"block_mask":[]})"
        << "\n";
    trace << R"({"type":"write","instance_id":"logical_source","trace_id":"write_a","timestamp_ns":1001,"keys":[401]})"
          << "\n";
    trace
        << R"({"type":"get","instance_id":"logical_source","trace_id":"engine_hit","timestamp_ns":2000,"keys":[401],"input_len":16,"block_mask":[]})"
        << "\n";
    trace.close();

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());
    manager.DirectRun();
    manager.AnalyzeResults();

    std::ifstream csv(root + "/combined/hierarchical_hit_rates.csv");
    ASSERT_TRUE(csv.is_open());
    std::ostringstream buffer;
    buffer << csv.rdbuf();
    const std::string content = buffer.str();
    EXPECT_THAT(content, HasSubstr("cold,engine_a,model_l3,1,0,0,0"));
    EXPECT_THAT(content, HasSubstr("engine_hit,engine_a,model_l3,1,1,0,1"));
}

TEST_F(HierarchicalReplayManagerTest, SeparatesIndependentPoolInstances) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_multi_pool";
    HierarchicalReplayConfig config;
    config.set_trace_file_path(root + "/trace.jsonl");
    config.set_output_result_path(root + "/combined");
    config.set_engine_config(
        CreateEngineOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm", "dram"}, root + "/engine"));
    config.set_pool_config(CreateOptimizerConfig("pool_group", {"model_a_l3", "model_b_l3"}, {"l3"}, root + "/pool"));

    EngineToPoolMappingConfig map_a;
    map_a.set_engine_instance_id("engine_a");
    map_a.set_pool_instance_id("model_a_l3");
    EngineToPoolMappingConfig map_b;
    map_b.set_engine_instance_id("engine_b");
    map_b.set_pool_instance_id("model_b_l3");
    config.set_engine_to_pool({map_a, map_b});

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> keys = {301};
    manager.WriteCache("engine_a", "write_a", 1000, keys);

    auto miss_other_pool = manager.GetCacheLocation("engine_b", "miss_other_pool", 2000, keys, 16);
    EXPECT_EQ(miss_other_pool.engine_hit_length, 0);
    EXPECT_EQ(miss_other_pool.pool_hit_length, 0);
    EXPECT_EQ(miss_other_pool.total_hit_length, 0);

    auto hit_own_engine = manager.GetCacheLocation("engine_a", "hit_own_engine", 3000, keys, 16);
    EXPECT_EQ(hit_own_engine.engine_hit_length, 1);
    EXPECT_EQ(hit_own_engine.pool_hit_length, 0);
    EXPECT_EQ(hit_own_engine.total_hit_length, 1);
}
