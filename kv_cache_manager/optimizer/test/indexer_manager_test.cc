#include <memory>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/instance_config.h"
#include "kv_cache_manager/optimizer/config/instance_group_config.h"
#include "kv_cache_manager/optimizer/config/tier_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/eviction_policy/lru.h"
#include "kv_cache_manager/optimizer/manager/eviction_manager.h"
#include "kv_cache_manager/optimizer/manager/indexer_manager.h"

using namespace kv_cache_manager;

class OptIndexerManagerTest : public TESTBASE {
public:
    void SetUp() override {
        eviction_manager_ = std::make_shared<OptEvictionManager>();
        EvictionConfig eviction_config;
        eviction_config.set_eviction_batch_size_per_instance(10);
        eviction_config.set_eviction_mode(EvictionMode::EVICTION_MODE_INSTANCE_PRECISE);
        ASSERT_TRUE(eviction_manager_->Init(eviction_config));

        indexer_manager_ = std::make_shared<OptIndexerManager>(eviction_manager_);
    }

protected:
    std::shared_ptr<OptIndexerManager> indexer_manager_;
    std::shared_ptr<OptEvictionManager> eviction_manager_;
    OptInstanceConfig CreateTestInstanceConfig(const std::string &instance_id);
    std::vector<OptTierConfig> CreateTestTierConfigs();
    OptInstanceGroupConfig CreateTestInstanceGroupConfig();
};

OptInstanceConfig OptIndexerManagerTest::CreateTestInstanceConfig(const std::string &instance_id) {
    OptInstanceConfig config;
    config.set_instance_id(instance_id);
    config.set_instance_group_name("test_group");
    config.set_block_size(1024);
    LruParams params;
    params.sample_rate = 1.0;
    EvictionPolicyParam policy_param;
    policy_param = params;
    config.set_eviction_policy_param(policy_param);
    config.set_eviction_policy_type(EvictionPolicyType::POLICY_LRU);

    return config;
}

std::vector<OptTierConfig> OptIndexerManagerTest::CreateTestTierConfigs() {
    std::vector<OptTierConfig> configs;

    OptTierConfig tier1;
    tier1.set_unique_name("tier1");
    tier1.set_capacity(1024 * 1024 * 10); // 10MB
    tier1.set_storage_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    tier1.set_band_width_mbps(1000);
    tier1.set_priority(1);
    configs.push_back(tier1);

    return configs;
}

OptInstanceGroupConfig OptIndexerManagerTest::CreateTestInstanceGroupConfig() {
    OptInstanceGroupConfig config;
    config.set_group_name("test_group");
    config.set_quota_capacity(1024 * 1024 * 100); // 100MB
    config.set_used_percentage(0.0);
    config.set_hierarchical_eviction_enabled(false);

    OptTierConfig tier1;
    tier1.set_unique_name("tier1");
    tier1.set_capacity(1024 * 1024 * 10);
    tier1.set_storage_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    tier1.set_band_width_mbps(1000);
    tier1.set_priority(1);
    config.set_storages({tier1});

    return config;
}

TEST_F(OptIndexerManagerTest, BasicInitialization) {
    EXPECT_NE(indexer_manager_, nullptr);
    EXPECT_NE(eviction_manager_, nullptr);
}

TEST_F(OptIndexerManagerTest, CreateOptIndexer) {
    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();

    EXPECT_TRUE(indexer_manager_->CreateOptIndexer(instance_config, tier_configs, false));

    auto indexer = indexer_manager_->GetOptIndexer("instance1");
    EXPECT_NE(indexer, nullptr);

    // 验证索引器可以插入和查询数据
    std::vector<int64_t> block_keys = {1, 2, 3, 4, 5};
    auto inserted = indexer->InsertOnly(block_keys, 1000);
    EXPECT_EQ(inserted.size(), 5);

    std::vector<std::vector<int64_t>> hits;
    auto inserted2 = indexer->InsertWithQuery(block_keys, 2000, hits);
    EXPECT_EQ(inserted2.size(), 0); // 已存在
    EXPECT_EQ(hits.size(), 1);      // 命中
}

TEST_F(OptIndexerManagerTest, CreateMultipleOptIndexers) {
    auto instance_config1 = CreateTestInstanceConfig("instance1");
    auto instance_config2 = CreateTestInstanceConfig("instance2");
    auto tier_configs = CreateTestTierConfigs();

    EXPECT_TRUE(indexer_manager_->CreateOptIndexer(instance_config1, tier_configs, false));
    EXPECT_TRUE(indexer_manager_->CreateOptIndexer(instance_config2, tier_configs, false));

    auto indexer1 = indexer_manager_->GetOptIndexer("instance1");
    auto indexer2 = indexer_manager_->GetOptIndexer("instance2");

    EXPECT_NE(indexer1, nullptr);
    EXPECT_NE(indexer2, nullptr);
}

TEST_F(OptIndexerManagerTest, GetOptIndexerSize) {
    auto instance_config1 = CreateTestInstanceConfig("instance1");
    auto instance_config2 = CreateTestInstanceConfig("instance2");
    auto tier_configs = CreateTestTierConfigs();

    EXPECT_EQ(indexer_manager_->GetOptIndexerSize(), 0);

    indexer_manager_->CreateOptIndexer(instance_config1, tier_configs, false);
    EXPECT_EQ(indexer_manager_->GetOptIndexerSize(), 1);

    indexer_manager_->CreateOptIndexer(instance_config2, tier_configs, false);
    EXPECT_EQ(indexer_manager_->GetOptIndexerSize(), 2);
}

TEST_F(OptIndexerManagerTest, GetAllOptIndexers) {
    auto instance_config1 = CreateTestInstanceConfig("instance1");
    auto instance_config2 = CreateTestInstanceConfig("instance2");
    auto tier_configs = CreateTestTierConfigs();

    indexer_manager_->CreateOptIndexer(instance_config1, tier_configs, false);
    indexer_manager_->CreateOptIndexer(instance_config2, tier_configs, false);

    auto all_indexers = indexer_manager_->GetAllOptIndexers();
    EXPECT_EQ(all_indexers.size(), 2);
    EXPECT_NE(all_indexers.find("instance1"), all_indexers.end());
    EXPECT_NE(all_indexers.find("instance2"), all_indexers.end());
}

TEST_F(OptIndexerManagerTest, RegisterInstanceGroups) {
    std::unordered_map<std::string, OptInstanceGroupConfig> instance_groups;
    instance_groups["test_group"] = CreateTestInstanceGroupConfig();

    indexer_manager_->RegisterInstanceGroups(instance_groups);

    // 不应该崩溃
    SUCCEED();
}

TEST_F(OptIndexerManagerTest, RegisterInstances) {
    std::unordered_map<std::string, OptInstanceConfig> instances;
    instances["instance1"] = CreateTestInstanceConfig("instance1");
    instances["instance2"] = CreateTestInstanceConfig("instance2");

    indexer_manager_->RegisterInstances(instances);

    // 不应该崩溃
    SUCCEED();
}

TEST_F(OptIndexerManagerTest, CheckAndEvict) {
    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();

    indexer_manager_->CreateOptIndexer(instance_config, tier_configs, false);

    // 检查并触发驱逐
    indexer_manager_->CheckAndEvict("instance1");

    // 不应该崩溃
    SUCCEED();
}

TEST_F(OptIndexerManagerTest, GetCurrentInstanceUsage) {
    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();

    indexer_manager_->CreateOptIndexer(instance_config, tier_configs, false);

    // 初始使用量为0
    auto usage = indexer_manager_->GetCurrentInstanceUsage("instance1");
    EXPECT_EQ(usage, 0);

    // 插入数据后使用量应该增加
    auto indexer = indexer_manager_->GetOptIndexer("instance1");
    std::vector<int64_t> block_keys = {1, 2, 3, 4, 5};
    indexer->InsertOnly(block_keys, 1000);

    usage = indexer_manager_->GetCurrentInstanceUsage("instance1");
    EXPECT_GT(usage, 0);
}

TEST_F(OptIndexerManagerTest, GetNonExistentIndexer) {
    auto indexer = indexer_manager_->GetOptIndexer("non_existent_instance");
    EXPECT_EQ(indexer, nullptr);
}

TEST_F(OptIndexerManagerTest, HierarchicalEvictionEnabled) {
    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();

    // 启用分层驱逐
    EXPECT_TRUE(indexer_manager_->CreateOptIndexer(instance_config, tier_configs, true));

    auto indexer = indexer_manager_->GetOptIndexer("instance1");
    EXPECT_NE(indexer, nullptr);
}

TEST_F(OptIndexerManagerTest, RegisterInstanceGroupsAndInstances) {
    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();

    // 注册实例组和实例
    std::unordered_map<std::string, OptInstanceGroupConfig> instance_groups;
    instance_groups["test_group"] = CreateTestInstanceGroupConfig();
    indexer_manager_->RegisterInstanceGroups(instance_groups);

    std::unordered_map<std::string, OptInstanceConfig> instances;
    instances["instance1"] = instance_config;
    indexer_manager_->RegisterInstances(instances);

    // 创建索引器
    EXPECT_TRUE(indexer_manager_->CreateOptIndexer(instance_config, tier_configs, false));

    auto indexer = indexer_manager_->GetOptIndexer("instance1");
    EXPECT_NE(indexer, nullptr);
}

TEST_F(OptIndexerManagerTest, CheckAndEvictNonExistentInstance) {
    // 检查不存在的实例
    indexer_manager_->CheckAndEvict("non_existent_instance");

    // 不应该崩溃
    SUCCEED();
}

TEST_F(OptIndexerManagerTest, GetCurrentInstanceUsageNonExistent) {
    auto usage = indexer_manager_->GetCurrentInstanceUsage("non_existent_instance");
    EXPECT_EQ(usage, 0);
}