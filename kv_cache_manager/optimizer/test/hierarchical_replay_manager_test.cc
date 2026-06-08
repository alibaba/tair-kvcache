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
#include "kv_cache_manager/optimizer/storage_pool/hash_storage_pool_manager.h"

using namespace kv_cache_manager;

class HierarchicalReplayManagerTest : public TESTBASE {
protected:
    HierarchicalReplayConfig CreateHierarchicalConfig(const std::string &root) {
        HierarchicalReplayConfig config;
        config.set_trace_file_path(root + "/trace.jsonl");
        config.set_output_result_path(root + "/combined");
        config.set_engine_config(
            CreateEngineOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm", "dram"}, root + "/infer"));
        config.set_storage_pool(CreateStoragePoolConfig({"model_l3"}, root + "/pool"));

        EngineToStoragePoolMappingConfig map_a;
        map_a.set_engine_instance_id("engine_a");
        map_a.set_storage_pool_id("model_l3");
        map_a.set_engine_read_query_type("batch_get");
        map_a.set_storage_pool_flow(CreateStoragePoolFlow());
        EngineToStoragePoolMappingConfig map_b;
        map_b.set_engine_instance_id("engine_b");
        map_b.set_storage_pool_id("model_l3");
        map_b.set_engine_read_query_type("batch_get");
        map_b.set_storage_pool_flow(CreateStoragePoolFlow());
        config.set_engine_to_storage_pool({map_a, map_b});
        return config;
    }

    StoragePoolFlowConfig CreateStoragePoolFlow() {
        StoragePoolFlowConfig flow;
        flow.set_write_mode(TierWriteMode::WRITE_THROUGH);
        flow.set_local_read_touch_enabled(false);
        flow.set_shadow_write_touch_enabled(false);
        flow.set_selective_write_threshold(2);
        return flow;
    }

    InferClusterConfig CreateInferClusterConfig(const StoragePoolFlowConfig &storage_pool_flow,
                                                const std::vector<P2PReadFlowConfig> &p2p_read_flows = {}) {
        LruParams params;
        params.sample_rate = 1.0;

        HierarchicalModelConfig model;
        model.set_block_size(16);
        model.set_bytes_per_token(1);
        model.set_eviction_policy_type(EvictionPolicyType::POLICY_LRU);
        model.set_eviction_policy_param(params);

        OptTtlConfig ttl_config;
        ttl_config.set_default_block_ttl_seconds(0);
        ttl_config.set_refresh_on_read(true);

        HierarchicalTierConfig hbm;
        hbm.set_name("hbm");
        hbm.set_capacity(1.0);
        HierarchicalTierConfig dram;
        dram.set_name("dram");
        dram.set_capacity(1.0);

        InferClusterConfig cluster;
        cluster.set_storage_pool_id("model_l3");
        cluster.set_engine_read_query_type("batch_get");
        cluster.set_model(model);
        cluster.set_infer_ids({"engine_a", "engine_b"});
        cluster.set_ttl_config(ttl_config);
        cluster.set_tiers({hbm, dram});
        cluster.set_p2p_read_flows(p2p_read_flows);
        cluster.set_storage_pool_flow(storage_pool_flow);
        return cluster;
    }

    InferActiveWindowConfig CreateInferActiveWindow(const std::string &infer_id, int64_t start_ns, int64_t end_ns) {
        InferActiveWindowConfig window;
        window.set_infer_id(infer_id);
        window.set_start_ns(start_ns);
        window.set_end_ns(end_ns);
        return window;
    }

    void SetStoragePoolFlow(HierarchicalReplayConfig &config, const StoragePoolFlowConfig &flow) {
        std::vector<EngineToStoragePoolMappingConfig> mappings = config.engine_to_storage_pool();
        for (auto &mapping : mappings) {
            mapping.set_storage_pool_flow(flow);
        }
        config.set_engine_to_storage_pool(mappings);
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

    HierarchicalStoragePoolConfig CreateStoragePoolConfig(const std::vector<std::string> &pool_ids,
                                                          const std::string &output_path,
                                                          int64_t capacity_bytes = 1LL << 30) {
        HierarchicalStoragePoolConfig config;
        config.set_output_result_path(output_path);
        config.set_storage_name("l3");
        config.set_capacity(static_cast<double>(capacity_bytes) / static_cast<double>(1LL << 30));
        config.set_eviction_config(CreateEvictionConfig());

        OptTtlConfig ttl_config;
        ttl_config.set_default_block_ttl_seconds(0);
        ttl_config.set_refresh_on_read(true);
        config.set_ttl_config(ttl_config);

        std::vector<HierarchicalStoragePoolEntryConfig> pools;
        pools.reserve(pool_ids.size());
        for (const auto &pool_id : pool_ids) {
            LruParams params;
            params.sample_rate = 1.0;
            HierarchicalModelConfig model;
            model.set_block_size(16);
            model.set_bytes_per_token(1);
            model.set_eviction_policy_type(EvictionPolicyType::POLICY_LRU);
            model.set_eviction_policy_param(params);

            HierarchicalStoragePoolEntryConfig pool;
            pool.set_pool_id(pool_id);
            pool.set_model(model);
            pools.push_back(pool);
        }
        config.set_pools(pools);
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
    EXPECT_EQ(cold.storage_pool_hit_length, 0);
    EXPECT_EQ(cold.total_hit_length, 0);

    manager.WriteCache("engine_a", "write_a", 1001, keys);

    auto pool_hit = manager.GetCacheLocation("engine_b", "pool_hit", 2000, keys, 32);
    EXPECT_EQ(pool_hit.engine_hit_length, 0);
    EXPECT_EQ(pool_hit.storage_pool_hit_length, 2);
    EXPECT_EQ(pool_hit.total_hit_length, 2);

    manager.WriteCache("engine_b", "write_b", 2001, keys);

    auto engine_hit = manager.GetCacheLocation("engine_b", "engine_hit", 3000, keys, 32);
    EXPECT_EQ(engine_hit.engine_hit_length, 2);
    EXPECT_EQ(engine_hit.storage_pool_hit_length, 0);
    EXPECT_EQ(engine_hit.total_hit_length, 2);

    manager.AnalyzeResults();
    EXPECT_TRUE(std::filesystem::exists(root + "/combined/hierarchical_hit_rates.csv"));
}

TEST_F(HierarchicalReplayManagerTest, P2PReadHitsPeerAndFillsCurrentEngine) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_p2p";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);

    StoragePoolFlowConfig flow = CreateStoragePoolFlow();
    flow.set_write_mode(TierWriteMode::CASCADING);
    SetStoragePoolFlow(config, flow);

    P2PReadFlowConfig p2p_flow;
    p2p_flow.set_tier("dram");
    p2p_flow.set_peer_read_touch_enabled(true);
    config.set_infer_clusters({CreateInferClusterConfig(flow, {p2p_flow})});

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> keys = {1001, 1002};
    manager.WriteCache("engine_a", "write_peer", 1000, keys);

    auto peer_hit = manager.GetCacheLocation("engine_b", "peer_hit", 2000, keys, 32, "batch_get");
    EXPECT_EQ(peer_hit.engine_hit_length, 0);
    EXPECT_EQ(peer_hit.peer_hit_length, 2);
    EXPECT_EQ(peer_hit.storage_pool_hit_length, 0);
    EXPECT_EQ(peer_hit.total_hit_length, 2);

    auto local_hit = manager.GetCacheLocation("engine_b", "local_after_peer", 3000, keys, 32, "batch_get");
    EXPECT_EQ(local_hit.engine_hit_length, 2);
    EXPECT_EQ(local_hit.peer_hit_length, 0);
    EXPECT_EQ(local_hit.storage_pool_hit_length, 0);
    EXPECT_EQ(local_hit.total_hit_length, 2);

    manager.AnalyzeResults();
    std::ifstream csv(root + "/combined/hierarchical_hit_rates.csv");
    ASSERT_TRUE(csv.is_open());
    std::ostringstream buffer;
    buffer << csv.rdbuf();
    const std::string content = buffer.str();
    EXPECT_THAT(content, HasSubstr("peer_hit,engine_b,model_l3,2,0,2,0,2"));
    EXPECT_THAT(content, HasSubstr("local_after_peer,engine_b,model_l3,2,2,0,0,2"));
}

TEST_F(HierarchicalReplayManagerTest, P2PReadSkipsInactivePeerFromTraceWindow) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_p2p_active_window";
    std::filesystem::create_directories(root);
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);

    StoragePoolFlowConfig flow = CreateStoragePoolFlow();
    flow.set_write_mode(TierWriteMode::CASCADING);
    SetStoragePoolFlow(config, flow);

    P2PReadFlowConfig p2p_flow;
    p2p_flow.set_tier("dram");
    p2p_flow.set_peer_read_touch_enabled(false);
    config.set_infer_clusters({CreateInferClusterConfig(flow, {p2p_flow})});

    std::ofstream trace(config.trace_file_path());
    trace << R"({"type":"write","instance_id":"engine_a","trace_id":"write_peer","timestamp_ns":100,"keys":[9001]})"
          << "\n";
    trace
        << R"({"type":"get","instance_id":"engine_b","trace_id":"read_after_scaledown","timestamp_ns":200,"keys":[9001],"input_len":16,"query_type":"batch_get","block_mask":[]})"
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
    EXPECT_THAT(content, HasSubstr("read_after_scaledown,engine_b,model_l3,1,0,0,0,0"));
}

TEST_F(HierarchicalReplayManagerTest, EngineReadUsesBatchGetAndPoolPrefixSkipsEngineHits) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_engine_batch_get";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);

    std::vector<EngineToStoragePoolMappingConfig> mappings = config.engine_to_storage_pool();
    for (auto &mapping : mappings) {
        if (mapping.engine_instance_id() == "engine_b") {
            StoragePoolFlowConfig flow = mapping.storage_pool_flow();
            flow.set_write_mode(TierWriteMode::CASCADING);
            mapping.set_storage_pool_flow(flow);
        }
    }
    config.set_engine_to_storage_pool(mappings);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    manager.WriteCache("engine_a", "pool_first", 1000, {811});
    manager.WriteCache("engine_a", "pool_last", 1010, {813});
    manager.WriteCache("engine_b", "engine_middle", 1020, {812});

    const std::vector<int64_t> keys = {811, 812, 813};
    auto mixed_hit = manager.GetCacheLocation("engine_b", "mixed_hit", 2000, keys, 48, "prefix_match");
    EXPECT_EQ(mixed_hit.engine_hit_length, 1);
    EXPECT_EQ(mixed_hit.storage_pool_hit_length, 2);
    EXPECT_EQ(mixed_hit.total_hit_length, 3);

    auto promoted = manager.GetCacheLocation("engine_b", "promoted", 3000, keys, 48, "prefix_match");
    EXPECT_EQ(promoted.engine_hit_length, 3);
    EXPECT_EQ(promoted.storage_pool_hit_length, 0);
    EXPECT_EQ(promoted.total_hit_length, 3);
}

TEST_F(HierarchicalReplayManagerTest, EngineReadQueryTypeCanUsePrefixMatch) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_engine_prefix_match";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);

    std::vector<EngineToStoragePoolMappingConfig> mappings = config.engine_to_storage_pool();
    for (auto &mapping : mappings) {
        if (mapping.engine_instance_id() == "engine_b") {
            mapping.set_engine_read_query_type("prefix_match");
            StoragePoolFlowConfig flow = mapping.storage_pool_flow();
            flow.set_write_mode(TierWriteMode::CASCADING);
            mapping.set_storage_pool_flow(flow);
        }
    }
    config.set_engine_to_storage_pool(mappings);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    manager.WriteCache("engine_b", "engine_first", 1000, {821});
    manager.WriteCache("engine_b", "engine_third", 1010, {823});

    auto prefix_hit = manager.GetCacheLocation("engine_b", "prefix_hit", 2000, {821, 822, 823}, 48, "prefix_match");
    EXPECT_EQ(prefix_hit.engine_hit_length, 1);
    EXPECT_EQ(prefix_hit.storage_pool_hit_length, 0);
    EXPECT_EQ(prefix_hit.total_hit_length, 1);
}

TEST_F(HierarchicalReplayManagerTest, ParsesCompactClusterConfig) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_compact_config";
    std::ostringstream json;
    json << R"({
        "trace_file_path": ")"
         << root << R"(/trace.jsonl",
        "output_result_path": ")"
         << root << R"(/output",
        "infer_eviction_params": {
            "eviction_mode": 3,
            "eviction_batch_size_per_instance": 10
        },
        "trace_replay": {
            "mode": "request",
            "write_delay_ns": 1000
        },
        "infer_scheduling_strategy": "preserve_trace",
        "enable_lifecycle_tracking": true,
        "infer_clusters": [
            {
                "storage_pool_id": "model_l3",
                "engine_read_query_type": "batch_get",
                "model": {
                    "block_size": 16,
                    "bytes_per_token": 1,
                    "eviction_policy_type": "lru",
                    "eviction_policy_params": {
                        "sample_rate": 1.0,
                        "shard_count": 1,
                        "sample_times": 32,
                        "eviction_amplification_factor": 1.0
                    }
                },
                "infer_ids": ["engine_a", "engine_b"],
                "ttl_config": {
                    "default_block_ttl_seconds": 0,
                    "refresh_on_read": true
                },
                "tiers": [
                    {"name": "hbm", "capacity": 1.0},
                    {"name": "dram", "capacity": 1.0}
                ],
                "tier_flows": [
                    {
                        "from_tier": "hbm",
                        "to_tier": "dram",
                        "write_mode": "write_through",
                        "access_propagation_enabled": false,
                        "write_propagation_enabled": false,
                        "selective_write_threshold": 2
                    }
                ],
                "storage_pool_flow": {
                    "write_mode": "cascading",
                    "local_read_touch_enabled": false,
                    "shadow_write_touch_enabled": false,
                    "selective_write_threshold": 2
                }
            }
        ],
        "storage_pool": {
            "output_result_path": ")"
         << root << R"(/output/pool",
            "storage_name": "l3",
            "capacity": 2.0,
            "eviction_params": {
                "eviction_mode": 3,
                "eviction_batch_size_per_instance": 10
            },
            "ttl_config": {
                "default_block_ttl_seconds": 0,
                "refresh_on_read": true
            },
            "pools": [
                {
                    "pool_id": "model_l3",
                    "model": {
                        "block_size": 16,
                        "bytes_per_token": 1,
                        "eviction_policy_type": "lru",
                        "eviction_policy_params": {
                            "sample_rate": 1.0,
                            "shard_count": 1,
                            "sample_times": 32,
                            "eviction_amplification_factor": 1.0
                        }
                    }
                }
            ]
        }
    })";

    HierarchicalReplayConfig config;
    ASSERT_TRUE(config.FromJsonString(json.str()));
    EXPECT_EQ(config.engine_config().instance_groups().size(), 2);
    EXPECT_EQ(config.storage_pool().pools().size(), 1);
    EXPECT_EQ(config.engine_to_storage_pool().size(), 2);
    EXPECT_EQ(config.engine_config().output_result_path(), root + "/output/infer");
    EXPECT_EQ(config.engine_config().eviction_config().eviction_mode(), EvictionMode::EVICTION_MODE_INSTANCE_PRECISE);
    EXPECT_EQ(config.engine_config().eviction_config().eviction_batch_size_per_instance(), 10);
    EXPECT_EQ(config.trace_replay_config().mode(), TraceReplayMode::REQUEST);
    EXPECT_EQ(config.trace_replay_config().write_delay_ns(), 1000);
    EXPECT_EQ(config.storage_pool().output_result_path(), root + "/output/pool");

    HierarchicalReplayManager manager(config);
    EXPECT_TRUE(manager.Init());
}

TEST_F(HierarchicalReplayManagerTest, ExportsLifecycleForEngineAndPoolWhenEnabled) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_lifecycle";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    config.set_enable_lifecycle_tracking(true);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> keys = {901, 902};
    manager.WriteCache("engine_a", "write_a", 1000, keys);
    manager.GetCacheLocation("engine_b", "pool_hit", 2000, keys, 32);
    manager.AnalyzeResults();

    EXPECT_TRUE(std::filesystem::exists(root + "/infer/engine_a_lifecycle.csv"));
    EXPECT_TRUE(std::filesystem::exists(root + "/pool/model_l3_lifecycle.csv"));
}

TEST_F(HierarchicalReplayManagerTest, SelectiveStoragePoolWriteIgnoresInferWritePropagation) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_selective_storage_pool";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    OptimizerConfig engine_config =
        CreateEngineOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm", "dram"}, root + "/infer");
    for (auto &group : engine_config.mutable_instance_groups()) {
        OptTierFlowPolicyConfig policy = group.tier_flow_policy();
        policy.set_write_propagation_enabled(true);
        group.set_tier_flow_policy(policy);
    }
    config.set_engine_config(engine_config);
    StoragePoolFlowConfig flow = CreateStoragePoolFlow();
    flow.set_write_mode(TierWriteMode::WRITE_THROUGH_SELECTIVE);
    flow.set_selective_write_threshold(2);
    SetStoragePoolFlow(config, flow);

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

    auto before_write_touch = manager.GetCacheLocation("engine_b", "before_write_touch", 1350, keys, 16);
    EXPECT_EQ(before_write_touch.total_hit_length, 0);

    manager.WriteCache("engine_a", "first_write_touch", 1400, keys);
    auto after_first_write_touch = manager.GetCacheLocation("engine_b", "after_first_write_touch", 1450, keys, 16);
    EXPECT_EQ(after_first_write_touch.engine_hit_length, 0);
    EXPECT_EQ(after_first_write_touch.storage_pool_hit_length, 0);

    manager.WriteCache("engine_a", "second_write_touch", 1500, keys);
    auto pool_hit_after_hot = manager.GetCacheLocation("engine_b", "pool_hit_after_hot", 1600, keys, 16);
    EXPECT_EQ(pool_hit_after_hot.engine_hit_length, 0);
    EXPECT_EQ(pool_hit_after_hot.storage_pool_hit_length, 0);
}

TEST_F(HierarchicalReplayManagerTest, SelectiveStoragePoolWriteUsesUntieredInferBlockState) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_selective_untiered_infer";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    config.set_engine_config(
        CreateEngineOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm"}, root + "/infer"));
    StoragePoolFlowConfig flow = CreateStoragePoolFlow();
    flow.set_write_mode(TierWriteMode::WRITE_THROUGH_SELECTIVE);
    flow.set_selective_write_threshold(2);
    SetStoragePoolFlow(config, flow);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> keys = {521};
    manager.WriteCache("engine_a", "write_a", 1000, keys);

    auto before_write_touch = manager.GetCacheLocation("engine_b", "before_write_touch", 1100, keys, 16);
    EXPECT_EQ(before_write_touch.total_hit_length, 0);

    manager.WriteCache("engine_a", "write_touch", 1200, keys);
    auto after_write_touch = manager.GetCacheLocation("engine_b", "after_write_touch", 1300, keys, 16);
    EXPECT_EQ(after_write_touch.engine_hit_length, 0);
    EXPECT_EQ(after_write_touch.storage_pool_hit_length, 1);
}

TEST_F(HierarchicalReplayManagerTest, SelectiveStoragePoolWriteRequiresInferSourceLayerTouch) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_selective_source_layer";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    StoragePoolFlowConfig flow = CreateStoragePoolFlow();
    flow.set_write_mode(TierWriteMode::WRITE_THROUGH_SELECTIVE);
    flow.set_selective_write_threshold(2);
    SetStoragePoolFlow(config, flow);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> keys = {531};
    manager.WriteCache("engine_a", "write_a", 1000, keys);
    manager.WriteCache("engine_a", "write_touch_hbm_only", 1100, keys);

    auto from_other_engine = manager.GetCacheLocation("engine_b", "from_other_engine", 1200, keys, 16);
    EXPECT_EQ(from_other_engine.total_hit_length, 0);
}

TEST_F(HierarchicalReplayManagerTest, SelectiveStoragePoolWriteIgnoresDemotedSourceWithoutWriteTouch) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_selective_source_keys";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    OptimizerConfig engine_config =
        CreateEngineOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm", "dram"}, root + "/infer", 16);
    for (auto &group : engine_config.mutable_instance_groups()) {
        OptTierFlowPolicyConfig policy = group.tier_flow_policy();
        policy.set_write_mode(TierWriteMode::CASCADING);
        group.set_tier_flow_policy(policy);
    }
    config.set_engine_config(engine_config);
    StoragePoolFlowConfig flow = CreateStoragePoolFlow();
    flow.set_write_mode(TierWriteMode::WRITE_THROUGH_SELECTIVE);
    flow.set_selective_write_threshold(1);
    SetStoragePoolFlow(config, flow);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> first = {541};
    const std::vector<int64_t> second = {542};
    manager.WriteCache("engine_a", "write_first", 1000, first);
    manager.WriteCache("engine_a", "write_second", 1100, second);

    auto from_other_engine = manager.GetCacheLocation("engine_b", "from_other_engine", 1200, first, 16);
    EXPECT_EQ(from_other_engine.engine_hit_length, 0);
    EXPECT_EQ(from_other_engine.storage_pool_hit_length, 0);
}

TEST_F(HierarchicalReplayManagerTest, CascadingStoragePoolWriteMovesInferEvictionsToPool) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_cascading_storage_pool";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    config.set_engine_config(
        CreateEngineOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm", "dram"}, root + "/infer", 16));
    StoragePoolFlowConfig flow = CreateStoragePoolFlow();
    flow.set_write_mode(TierWriteMode::CASCADING);
    SetStoragePoolFlow(config, flow);

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
    EXPECT_EQ(pool_hit_after_eviction.storage_pool_hit_length, 1);
}

TEST_F(HierarchicalReplayManagerTest, CascadingStoragePoolWriteUsesInferSourceTierEvictions) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_cascading_source_tier_eviction";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    OptimizerConfig engine_config =
        CreateEngineOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm", "dram"}, root + "/infer", 16);
    for (auto &group : engine_config.mutable_instance_groups()) {
        auto &tiers = group.mutable_storages();
        ASSERT_EQ(tiers.size(), 2);
        tiers[0].set_capacity(32);
        tiers[1].set_capacity(16);
    }
    config.set_engine_config(engine_config);
    StoragePoolFlowConfig flow = CreateStoragePoolFlow();
    flow.set_write_mode(TierWriteMode::CASCADING);
    SetStoragePoolFlow(config, flow);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> first = {611};
    const std::vector<int64_t> second = {612};
    manager.WriteCache("engine_a", "write_first", 1000, first);
    manager.WriteCache("engine_a", "evict_first_from_source_tier", 1100, second);

    auto engine_a_hit = manager.GetCacheLocation("engine_a", "engine_a_still_has_first", 1200, first, 16);
    EXPECT_EQ(engine_a_hit.engine_hit_length, 1);
    EXPECT_EQ(engine_a_hit.storage_pool_hit_length, 0);

    auto pool_hit = manager.GetCacheLocation("engine_b", "pool_hit_after_source_tier_eviction", 1300, first, 16);
    EXPECT_EQ(pool_hit.engine_hit_length, 0);
    EXPECT_EQ(pool_hit.storage_pool_hit_length, 1);
}

TEST_F(HierarchicalReplayManagerTest, CascadingStoragePoolWriteTouchExistingFollowsConfig) {
    auto first_survives_after_pool_pressure = [this](bool shadow_write_touch_enabled) {
        const std::string root = GetTestTempRootPath() + "/hierarchical_replay_cascading_shadow_touch_" +
                                 (shadow_write_touch_enabled ? "enabled" : "disabled");
        HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
        OptimizerConfig engine_config =
            CreateEngineOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm"}, root + "/infer", 16);
        for (auto &group : engine_config.mutable_instance_groups()) {
            group.set_quota_capacity(16);
        }
        config.set_engine_config(engine_config);
        config.set_storage_pool(CreateStoragePoolConfig({"model_l3"}, root + "/pool", 32));
        StoragePoolFlowConfig flow = CreateStoragePoolFlow();
        flow.set_write_mode(TierWriteMode::CASCADING);
        flow.set_shadow_write_touch_enabled(shadow_write_touch_enabled);
        SetStoragePoolFlow(config, flow);

        HierarchicalReplayManager manager(config);
        EXPECT_TRUE(manager.Init());

        const std::vector<int64_t> first = {621};
        manager.WriteCache("engine_a", "write_first", 1000, first);
        manager.WriteCache("engine_a", "evict_first_to_pool", 1100, {622});
        manager.WriteCache("engine_a", "evict_second_to_pool", 1200, {623});
        manager.WriteCache("engine_b", "write_existing_first_locally", 1300, first);
        manager.WriteCache("engine_b", "evict_first_to_existing_pool_block", 1400, {624});
        manager.WriteCache("engine_b", "insert_new_pool_block", 1500, {625});

        auto first_read = manager.GetCacheLocation("engine_b", "read_first", 1600, first, 16);
        return first_read.storage_pool_hit_length == 1;
    };

    EXPECT_FALSE(first_survives_after_pool_pressure(false));
    EXPECT_TRUE(first_survives_after_pool_pressure(true));
}

TEST_F(HierarchicalReplayManagerTest, WriteThroughStoragePoolUsesInferLastTierWriteEvents) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_write_through_source_events";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    OptimizerConfig engine_config =
        CreateEngineOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm", "dram"}, root + "/infer", 16);
    for (auto &group : engine_config.mutable_instance_groups()) {
        OptTierFlowPolicyConfig policy = group.tier_flow_policy();
        policy.set_write_mode(TierWriteMode::CASCADING);
        group.set_tier_flow_policy(policy);
    }
    config.set_engine_config(engine_config);
    StoragePoolFlowConfig flow = CreateStoragePoolFlow();
    flow.set_write_mode(TierWriteMode::WRITE_THROUGH);
    SetStoragePoolFlow(config, flow);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> first = {631};
    const std::vector<int64_t> second = {632};
    manager.WriteCache("engine_a", "write_first", 1000, first);
    auto before_demote = manager.GetCacheLocation("engine_b", "before_demote", 1050, first, 16);
    EXPECT_EQ(before_demote.total_hit_length, 0);

    manager.WriteCache("engine_a", "write_second", 1100, second);
    auto after_demote = manager.GetCacheLocation("engine_b", "after_demote", 1200, first, 16);
    EXPECT_EQ(after_demote.engine_hit_length, 0);
    EXPECT_EQ(after_demote.storage_pool_hit_length, 1);
}

TEST_F(HierarchicalReplayManagerTest, L3HitPromotesToEngine) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_l3_promote";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> keys = {701};
    manager.WriteCache("engine_a", "write_a", 1000, keys);

    auto pool_hit = manager.GetCacheLocation("engine_b", "pool_hit", 2000, keys, 16);
    EXPECT_EQ(pool_hit.engine_hit_length, 0);
    EXPECT_EQ(pool_hit.storage_pool_hit_length, 1);

    auto promoted_engine_hit = manager.GetCacheLocation("engine_b", "promoted_engine_hit", 3000, keys, 16);
    EXPECT_EQ(promoted_engine_hit.engine_hit_length, 1);
    EXPECT_EQ(promoted_engine_hit.storage_pool_hit_length, 0);
}

TEST_F(HierarchicalReplayManagerTest, PoolPromoteEvictionWritesBackToPoolWhenCascading) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_pool_promote_writeback";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    OptimizerConfig engine_config =
        CreateEngineOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm"}, root + "/infer", 16);
    for (auto &group : engine_config.mutable_instance_groups()) {
        group.set_quota_capacity(16);
    }
    config.set_engine_config(engine_config);
    StoragePoolFlowConfig flow = CreateStoragePoolFlow();
    flow.set_write_mode(TierWriteMode::CASCADING);
    SetStoragePoolFlow(config, flow);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> remote = {711};
    const std::vector<int64_t> filler = {712};
    const std::vector<int64_t> victim = {713};
    manager.WriteCache("engine_a", "write_remote", 1000, remote);
    manager.WriteCache("engine_a", "evict_remote_to_pool", 1100, filler);
    manager.WriteCache("engine_b", "write_victim", 1200, victim);

    auto remote_pool_hit = manager.GetCacheLocation("engine_b", "remote_pool_hit", 1300, remote, 16);
    EXPECT_EQ(remote_pool_hit.engine_hit_length, 0);
    EXPECT_EQ(remote_pool_hit.storage_pool_hit_length, 1);

    auto victim_check = manager.GetCacheLocation("engine_a", "victim_check", 1400, victim, 16);
    EXPECT_EQ(victim_check.engine_hit_length, 0);
    EXPECT_EQ(victim_check.storage_pool_hit_length, 1);
}

TEST_F(HierarchicalReplayManagerTest, EngineReadPromoteSourceTierEvictionWritesBackToPoolWhenCascading) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_engine_promote_source_writeback";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    OptimizerConfig engine_config =
        CreateEngineOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm", "dram"}, root + "/infer", 16);
    for (auto &group : engine_config.mutable_instance_groups()) {
        OptTierFlowPolicyConfig policy = group.tier_flow_policy();
        policy.set_write_mode(TierWriteMode::CASCADING);
        group.set_tier_flow_policy(policy);
    }
    config.set_engine_config(engine_config);
    StoragePoolFlowConfig flow = CreateStoragePoolFlow();
    flow.set_write_mode(TierWriteMode::CASCADING);
    SetStoragePoolFlow(config, flow);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> first = {721};
    const std::vector<int64_t> second = {722};
    manager.WriteCache("engine_a", "write_first", 1000, first);
    manager.WriteCache("engine_a", "demote_first_to_source", 1100, second);

    auto first_engine_hit = manager.GetCacheLocation("engine_a", "promote_first", 1200, first, 16);
    EXPECT_EQ(first_engine_hit.engine_hit_length, 1);
    EXPECT_EQ(first_engine_hit.storage_pool_hit_length, 0);

    auto first_pool_hit = manager.GetCacheLocation("engine_b", "read_first_from_pool", 1300, first, 16);
    auto second_pool_hit = manager.GetCacheLocation("engine_b", "read_second_from_pool", 1400, second, 16);
    EXPECT_GE(first_pool_hit.storage_pool_hit_length + second_pool_hit.storage_pool_hit_length, 1);
}

TEST_F(HierarchicalReplayManagerTest, StoragePoolUsesFullKeysAfterEnginePrefixHit) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_full_keys_to_pool";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> full_keys = {731, 732};
    const std::vector<int64_t> prefix = {731};
    manager.WriteCache("engine_a", "write_full_to_pool", 1000, full_keys);
    manager.WriteCache("engine_b", "write_engine_prefix", 1100, prefix);

    auto read_full = manager.GetCacheLocation("engine_b", "read_full", 1200, full_keys, 32);
    EXPECT_EQ(read_full.engine_hit_length, 1);
    EXPECT_EQ(read_full.storage_pool_hit_length, 1);
    EXPECT_EQ(read_full.total_hit_length, 2);
}

TEST_F(HierarchicalReplayManagerTest, HashStoragePoolReadsFromEngineMissOffset) {
    const std::string root = GetTestTempRootPath() + "/hash_storage_pool_prefix_read";
    HierarchicalStoragePoolConfig pool_config = CreateStoragePoolConfig({"model_l3"}, root + "/pool");

    HashStoragePoolManager pool(pool_config);
    ASSERT_TRUE(pool.Init());

    const std::vector<int64_t> keys = {741, 742, 743};
    pool.WriteKeys("model_l3", "write_tail_only", 1000, {743}, 0, false);

    auto cold_prefix =
        pool.Read(HashStoragePoolReadRequest("model_l3", "cold_prefix", 1100, keys, {}, 48, "prefix_match", false));
    EXPECT_EQ(cold_prefix.hit_blocks, 0);

    auto suffix_after_engine_prefix =
        pool.Read(HashStoragePoolReadRequest("model_l3", "suffix", 1200, keys, {0, 1}, 48, "prefix_match", false));
    EXPECT_EQ(suffix_after_engine_prefix.hit_blocks, 1);

    auto batch_tail =
        pool.Read(HashStoragePoolReadRequest("model_l3", "batch_tail", 1300, keys, {}, 48, "batch_get", false));
    EXPECT_EQ(batch_tail.hit_blocks, 1);
    ASSERT_EQ(batch_tail.hit_indices.size(), 1);
    EXPECT_EQ(batch_tail.hit_indices[0], 2);
}

TEST_F(HierarchicalReplayManagerTest, BatchGetSkipsEngineHitIndicesWhenReadingPool) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_batch_get";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> full_keys = {751, 752};
    manager.WriteCache("engine_a", "write_full_to_pool", 1000, full_keys);
    manager.WriteCache("engine_b", "write_engine_tail", 1100, {752});

    auto batch_read = manager.GetCacheLocation("engine_b", "batch_read", 1200, full_keys, 32, "batch_get");
    EXPECT_EQ(batch_read.engine_hit_length, 1);
    EXPECT_EQ(batch_read.storage_pool_hit_length, 1);
    EXPECT_EQ(batch_read.total_hit_length, 2);
}

TEST_F(HierarchicalReplayManagerTest, EngineHitUpdatesStoragePoolAccess) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_l3_access_propagation";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    config.set_storage_pool(CreateStoragePoolConfig({"model_l3"}, root + "/pool", 32));
    StoragePoolFlowConfig flow = CreateStoragePoolFlow();
    flow.set_local_read_touch_enabled(true);
    SetStoragePoolFlow(config, flow);

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
    EXPECT_EQ(first_still_in_l3.storage_pool_hit_length, 1);
    auto second_evicted_from_l3 = manager.GetCacheLocation("engine_b", "second_evicted_from_l3", 1400, second, 16);
    EXPECT_EQ(second_evicted_from_l3.total_hit_length, 0);
}

TEST_F(HierarchicalReplayManagerTest, EngineHitDoesNotUpdateStoragePoolAccessWhenDisabled) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_l3_access_no_propagation";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    config.set_storage_pool(CreateStoragePoolConfig({"model_l3"}, root + "/pool", 32));

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> first = {804};
    const std::vector<int64_t> second = {805};
    const std::vector<int64_t> third = {806};
    manager.WriteCache("engine_a", "write_first", 1000, first);
    manager.WriteCache("engine_a", "write_second", 1001, second);

    auto touch_first = manager.GetCacheLocation("engine_a", "touch_first", 1100, first, 16);
    EXPECT_EQ(touch_first.engine_hit_length, 1);

    manager.WriteCache("engine_a", "write_third", 1200, third);
    auto first_evicted_from_l3 = manager.GetCacheLocation("engine_b", "first_evicted_from_l3", 1300, first, 16);
    EXPECT_EQ(first_evicted_from_l3.total_hit_length, 0);
    auto second_still_in_l3 = manager.GetCacheLocation("engine_b", "second_still_in_l3", 1400, second, 16);
    EXPECT_EQ(second_still_in_l3.storage_pool_hit_length, 1);
}

TEST_F(HierarchicalReplayManagerTest, WriteThroughDoesNotTouchExistingL3WhenDisabled) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_l3_write_no_propagation";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    config.set_storage_pool(CreateStoragePoolConfig({"model_l3"}, root + "/pool", 32));
    StoragePoolFlowConfig flow = CreateStoragePoolFlow();
    flow.set_write_mode(TierWriteMode::WRITE_THROUGH);
    flow.set_shadow_write_touch_enabled(false);
    SetStoragePoolFlow(config, flow);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> first = {811};
    const std::vector<int64_t> second = {812};
    const std::vector<int64_t> third = {813};
    manager.WriteCache("engine_a", "write_first", 1000, first);
    manager.WriteCache("engine_a", "write_second", 1100, second);
    manager.WriteCache("engine_a", "rewrite_first", 1200, first);
    manager.WriteCache("engine_a", "write_third", 1300, third);

    auto first_from_other_engine = manager.GetCacheLocation("engine_b", "read_first", 1400, first, 16);
    EXPECT_EQ(first_from_other_engine.storage_pool_hit_length, 0);
}

TEST_F(HierarchicalReplayManagerTest, WritePropagationDoesNotTouchExistingL3) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_l3_write_propagation";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    OptimizerConfig engine_config =
        CreateEngineOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm", "dram"}, root + "/infer");
    for (auto &group : engine_config.mutable_instance_groups()) {
        OptTierFlowPolicyConfig policy = group.tier_flow_policy();
        policy.set_write_propagation_enabled(true);
        group.set_tier_flow_policy(policy);
    }
    config.set_engine_config(engine_config);
    config.set_storage_pool(CreateStoragePoolConfig({"model_l3"}, root + "/pool", 32));
    StoragePoolFlowConfig flow = CreateStoragePoolFlow();
    flow.set_write_mode(TierWriteMode::WRITE_THROUGH);
    flow.set_shadow_write_touch_enabled(true);
    SetStoragePoolFlow(config, flow);

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> first = {821};
    const std::vector<int64_t> second = {822};
    const std::vector<int64_t> third = {823};
    manager.WriteCache("engine_a", "write_first", 1000, first);
    manager.WriteCache("engine_a", "write_second", 1100, second);
    manager.WriteCache("engine_a", "rewrite_first", 1200, first);
    manager.WriteCache("engine_a", "write_third", 1300, third);

    auto first_from_other_engine = manager.GetCacheLocation("engine_b", "read_first", 1400, first, 16);
    EXPECT_EQ(first_from_other_engine.storage_pool_hit_length, 0);
}

TEST_F(HierarchicalReplayManagerTest, RejectsSharedEngineInstanceGroup) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_shared_engine_group";
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    config.set_engine_config(
        CreateOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm", "dram"}, root + "/infer"));

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
    EXPECT_THAT(content, HasSubstr("LocalHitBlocks,PeerHitBlocks,RemoteHitBlocks,HitBlocks"));
    EXPECT_THAT(content, HasSubstr("AccLocalHitTokens,AccPeerHitTokens,AccRemoteHitTokens,AccHitTokens"));
    EXPECT_THAT(content, HasSubstr("pool_hit,engine_b,model_l3,2,0,0,2,2"));
    EXPECT_THAT(content, HasSubstr("engine_hit,engine_b,model_l3,2,2,0,0,2"));
}

TEST_F(HierarchicalReplayManagerTest, ReplaysRequestTraceWithDelayedWrite) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_request";
    std::filesystem::create_directories(root);
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    OptTraceReplayConfig trace_replay_config;
    trace_replay_config.set_mode(TraceReplayMode::REQUEST);
    config.set_trace_replay_config(trace_replay_config);

    std::ofstream trace(config.trace_file_path());
    trace
        << R"({"type":"request","instance_id":"engine_a","trace_id":"cold","timestamp_ns":1000,"keys":[101,102],"input_len":32,"block_mask":[]})"
        << "\n";
    trace
        << R"({"type":"request","instance_id":"engine_a","trace_id":"engine_hit","timestamp_ns":2000,"keys":[101,102],"input_len":32,"block_mask":[]})"
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
    EXPECT_THAT(content, HasSubstr("cold,engine_a,model_l3,2,0,0,0,0"));
    EXPECT_THAT(content, HasSubstr("engine_hit,engine_a,model_l3,2,2,0,0,2"));
}

TEST_F(HierarchicalReplayManagerTest, DirectRunSortsTraceByTimestamp) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_sort";
    std::filesystem::create_directories(root);
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);

    std::ofstream trace(config.trace_file_path());
    trace
        << R"({"type":"get","instance_id":"engine_a","trace_id":"late_get","timestamp_ns":2000,"keys":[501],"input_len":16,"block_mask":[]})"
        << "\n";
    trace << R"({"type":"write","instance_id":"engine_a","trace_id":"early_write","timestamp_ns":1000,"keys":[501]})"
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
    EXPECT_THAT(buffer.str(), HasSubstr("late_get,engine_a,model_l3,1,1,0,0,1"));
}

TEST_F(HierarchicalReplayManagerTest, RoundRobinSchedulesTraceRequestsToEngineInstances) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_round_robin";
    std::filesystem::create_directories(root);
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    config.set_infer_scheduling_strategy("round_robin");

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
    EXPECT_THAT(content, HasSubstr("cold,engine_a,model_l3,1,0,0,0,0"));
    EXPECT_THAT(content, HasSubstr("pool_hit,engine_b,model_l3,1,0,0,1,1"));
}

TEST_F(HierarchicalReplayManagerTest, RoundRobinUsesConfiguredActiveWindows) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_round_robin_windows";
    std::filesystem::create_directories(root);
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    config.set_infer_scheduling_strategy("round_robin");

    StoragePoolFlowConfig flow = CreateStoragePoolFlow();
    auto cluster = CreateInferClusterConfig(flow);
    cluster.set_active_windows(
        {CreateInferActiveWindow("engine_a", 0, 999), CreateInferActiveWindow("engine_b", 1000, 3000)});
    config.set_infer_clusters({cluster});

    std::ofstream trace(config.trace_file_path());
    trace
        << R"({"type":"get","instance_id":"logical_source","trace_id":"first","timestamp_ns":100,"keys":[501],"input_len":16,"block_mask":[]})"
        << "\n";
    trace
        << R"({"type":"get","instance_id":"logical_source","trace_id":"second","timestamp_ns":200,"keys":[502],"input_len":16,"block_mask":[]})"
        << "\n";
    trace
        << R"({"type":"get","instance_id":"logical_source","trace_id":"third","timestamp_ns":1200,"keys":[503],"input_len":16,"block_mask":[]})"
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
    EXPECT_THAT(content, HasSubstr("first,engine_a,model_l3,1,0,0,0,0"));
    EXPECT_THAT(content, HasSubstr("second,engine_a,model_l3,1,0,0,0,0"));
    EXPECT_THAT(content, HasSubstr("third,engine_b,model_l3,1,0,0,0,0"));
}

TEST_F(HierarchicalReplayManagerTest, RoundRobinCanUseTraceActiveWindows) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_round_robin_trace_windows";
    std::filesystem::create_directories(root);
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    config.set_infer_scheduling_strategy("round_robin");
    config.set_infer_active_windows_from_trace(true);

    std::ofstream trace(config.trace_file_path());
    trace
        << R"({"type":"get","instance_id":"engine_a","trace_id":"early_a","timestamp_ns":100,"keys":[601],"input_len":16,"block_mask":[]})"
        << "\n";
    trace
        << R"({"type":"get","instance_id":"engine_a","trace_id":"early_b","timestamp_ns":200,"keys":[602],"input_len":16,"block_mask":[]})"
        << "\n";
    trace
        << R"({"type":"get","instance_id":"engine_b","trace_id":"late","timestamp_ns":1200,"keys":[603],"input_len":16,"block_mask":[]})"
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
    EXPECT_THAT(content, HasSubstr("early_a,engine_a,model_l3,1,0,0,0,0"));
    EXPECT_THAT(content, HasSubstr("early_b,engine_a,model_l3,1,0,0,0,0"));
    EXPECT_THAT(content, HasSubstr("late,engine_b,model_l3,1,0,0,0,0"));
}

TEST_F(HierarchicalReplayManagerTest, PrefixHitSchedulesToCachedEngineInstance) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_prefix_hit";
    std::filesystem::create_directories(root);
    HierarchicalReplayConfig config = CreateHierarchicalConfig(root);
    config.set_infer_scheduling_strategy("prefix_hit");

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
    EXPECT_THAT(content, HasSubstr("cold,engine_a,model_l3,1,0,0,0,0"));
    EXPECT_THAT(content, HasSubstr("engine_hit,engine_a,model_l3,1,1,0,0,1"));
}

TEST_F(HierarchicalReplayManagerTest, SeparatesIndependentPoolInstances) {
    const std::string root = GetTestTempRootPath() + "/hierarchical_replay_multi_pool";
    HierarchicalReplayConfig config;
    config.set_trace_file_path(root + "/trace.jsonl");
    config.set_output_result_path(root + "/combined");
    config.set_engine_config(
        CreateEngineOptimizerConfig("engine_group", {"engine_a", "engine_b"}, {"hbm", "dram"}, root + "/infer"));
    config.set_storage_pool(CreateStoragePoolConfig({"model_a_l3", "model_b_l3"}, root + "/pool"));

    EngineToStoragePoolMappingConfig map_a;
    map_a.set_engine_instance_id("engine_a");
    map_a.set_storage_pool_id("model_a_l3");
    map_a.set_engine_read_query_type("batch_get");
    map_a.set_storage_pool_flow(CreateStoragePoolFlow());
    EngineToStoragePoolMappingConfig map_b;
    map_b.set_engine_instance_id("engine_b");
    map_b.set_storage_pool_id("model_b_l3");
    map_b.set_engine_read_query_type("batch_get");
    map_b.set_storage_pool_flow(CreateStoragePoolFlow());
    config.set_engine_to_storage_pool({map_a, map_b});

    HierarchicalReplayManager manager(config);
    ASSERT_TRUE(manager.Init());

    const std::vector<int64_t> keys = {301};
    manager.WriteCache("engine_a", "write_a", 1000, keys);

    auto miss_other_pool = manager.GetCacheLocation("engine_b", "miss_other_pool", 2000, keys, 16);
    EXPECT_EQ(miss_other_pool.engine_hit_length, 0);
    EXPECT_EQ(miss_other_pool.storage_pool_hit_length, 0);
    EXPECT_EQ(miss_other_pool.total_hit_length, 0);

    auto hit_own_engine = manager.GetCacheLocation("engine_a", "hit_own_engine", 3000, keys, 16);
    EXPECT_EQ(hit_own_engine.engine_hit_length, 1);
    EXPECT_EQ(hit_own_engine.storage_pool_hit_length, 0);
    EXPECT_EQ(hit_own_engine.total_hit_length, 1);
}
