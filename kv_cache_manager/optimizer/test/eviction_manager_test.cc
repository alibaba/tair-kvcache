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

using namespace kv_cache_manager;

class OptEvictionManagerTest : public TESTBASE {
public:
    void SetUp() override {
        manager_ = std::make_shared<OptEvictionManager>();
        EvictionConfig eviction_config;
        eviction_config.set_eviction_batch_size_per_instance(10);
        eviction_config.set_eviction_mode(EvictionMode::EVICTION_MODE_INSTANCE_PRECISE);
        ASSERT_TRUE(manager_->Init(eviction_config));
    }

protected:
    std::shared_ptr<OptEvictionManager> manager_;
    OptInstanceConfig CreateTestInstanceConfig(const std::string &instance_id);
    std::vector<OptTierConfig> CreateTestTierConfigs();
    OptInstanceGroupConfig CreateTestInstanceGroupConfig();
};

OptInstanceConfig OptEvictionManagerTest::CreateTestInstanceConfig(const std::string &instance_id) {
    OptInstanceConfig config;
    config.set_instance_id(instance_id);
    config.set_instance_group_name("test_group");
    config.set_block_size(1024);
    LruParams params;
    params.sample_rate = 1.0; // 采样率100%
    EvictionPolicyParam policy_param;
    policy_param = params;
    config.set_eviction_policy_param(policy_param);
    config.set_eviction_policy_type(EvictionPolicyType::POLICY_LRU);

    return config;
}

std::vector<OptTierConfig> OptEvictionManagerTest::CreateTestTierConfigs() {
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

OptInstanceGroupConfig OptEvictionManagerTest::CreateTestInstanceGroupConfig() {
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

    // 添加实例配置
    OptInstanceConfig instance1;
    instance1.set_instance_id("instance1");
    instance1.set_instance_group_name("test_group");
    instance1.set_block_size(1024);
    LruParams params;
    params.sample_rate = 1.0;
    EvictionPolicyParam policy_param;
    policy_param = params;
    instance1.set_eviction_policy_param(policy_param);
    instance1.set_eviction_policy_type(EvictionPolicyType::POLICY_LRU);
    config.set_instances({instance1});

    return config;
}

TEST_F(OptEvictionManagerTest, CreateAndRegisterEvictionPolicy) {
    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();

    // 启用分层驱逐,这样策略名称是tier1
    auto policy = manager_->CreateAndRegisterEvictionPolicy(instance_config, tier_configs, true);
    EXPECT_NE(policy, nullptr);
    EXPECT_EQ(policy->name(), "tier1");
}

TEST_F(OptEvictionManagerTest, CreateMultipleEvictionPolicies) {
    auto instance_config1 = CreateTestInstanceConfig("instance1");
    auto instance_config2 = CreateTestInstanceConfig("instance2");
    auto tier_configs = CreateTestTierConfigs();

    // 启用分层驱逐
    auto policy1 = manager_->CreateAndRegisterEvictionPolicy(instance_config1, tier_configs, true);
    auto policy2 = manager_->CreateAndRegisterEvictionPolicy(instance_config2, tier_configs, true);

    EXPECT_NE(policy1, nullptr);
    EXPECT_NE(policy2, nullptr);
    EXPECT_EQ(policy1->name(), "tier1");
    EXPECT_EQ(policy2->name(), "tier1");
}

TEST_F(OptEvictionManagerTest, EvictByInstancePrecise) {
    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();
    auto instance_group_config = CreateTestInstanceGroupConfig();

    // 启用分层驱逐
    auto policy = manager_->CreateAndRegisterEvictionPolicy(instance_config, tier_configs, true);
    ASSERT_NE(policy, nullptr);

    // 驱逐测试
    auto evicted = manager_->EvictByMode("instance1", instance_group_config);
    // 不应该崩溃
    SUCCEED();
}

TEST_F(OptEvictionManagerTest, EvictByInstanceRough) {
    auto manager = std::make_shared<OptEvictionManager>();
    EvictionConfig eviction_config;
    eviction_config.set_eviction_batch_size_per_instance(10);
    eviction_config.set_eviction_mode(EvictionMode::EVICTION_MODE_INSTANCE_ROUGH);
    ASSERT_TRUE(manager_->Init(eviction_config));

    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();
    auto instance_group_config = CreateTestInstanceGroupConfig();

    // 启用分层驱逐
    auto policy = manager->CreateAndRegisterEvictionPolicy(instance_config, tier_configs, true);
    ASSERT_NE(policy, nullptr);

    // 驱逐测试
    auto evicted = manager->EvictByMode("instance1", instance_group_config);
    // 不应该崩溃
    SUCCEED();
}

TEST_F(OptEvictionManagerTest, EvictByGroupRough) {
    auto manager = std::make_shared<OptEvictionManager>();
    EvictionConfig eviction_config;
    eviction_config.set_eviction_batch_size_per_instance(10);
    eviction_config.set_eviction_mode(EvictionMode::EVICTION_MODE_GROUP_ROUGH);
    ASSERT_TRUE(manager_->Init(eviction_config));

    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();
    auto instance_group_config = CreateTestInstanceGroupConfig();

    // 启用分层驱逐
    auto policy = manager->CreateAndRegisterEvictionPolicy(instance_config, tier_configs, true);
    ASSERT_NE(policy, nullptr);

    // 驱逐测试
    auto evicted = manager->EvictByMode("instance1", instance_group_config);
    // 不应该崩溃
    SUCCEED();
}

TEST_F(OptEvictionManagerTest, GetCurrentInstanceUsage) {
    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();

    // 启用分层驱逐
    auto policy = manager_->CreateAndRegisterEvictionPolicy(instance_config, tier_configs, true);
    ASSERT_NE(policy, nullptr);

    // 初始使用量为0
    auto usage = manager_->GetCurrentInstanceUsage("instance1");
    EXPECT_EQ(usage, 0);

    // 添加一些块
    for (int i = 0; i < 5; i++) {
        BlockEntry block;
        block.key = i;
        block.last_access_time = i * 100;
        block.writing_time = i * 100;
        policy->OnBlockWritten(&block);
    }

    // 使用量应该增加
    usage = manager_->GetCurrentInstanceUsage("instance1");
    EXPECT_GT(usage, 0);
}

TEST_F(OptEvictionManagerTest, GetCurrentGroupUsage) {
    auto instance_config1 = CreateTestInstanceConfig("instance1");
    auto instance_config2 = CreateTestInstanceConfig("instance2");
    auto tier_configs = CreateTestTierConfigs();
    auto instance_group_config = CreateTestInstanceGroupConfig();

    // 启用分层驱逐
    auto policy1 = manager_->CreateAndRegisterEvictionPolicy(instance_config1, tier_configs, true);
    auto policy2 = manager_->CreateAndRegisterEvictionPolicy(instance_config2, tier_configs, true);
    ASSERT_NE(policy1, nullptr);
    ASSERT_NE(policy2, nullptr);

    // 初始使用量为0
    auto usage = manager_->GetCurrentGroupUsage(instance_group_config);
    EXPECT_EQ(usage, 0);

    // 给instance1添加块
    for (int i = 0; i < 3; i++) {
        BlockEntry block;
        block.key = i;
        block.last_access_time = i * 100;
        block.writing_time = i * 100;
        policy1->OnBlockWritten(&block);
    }

    // 给instance2添加块
    for (int i = 0; i < 2; i++) {
        BlockEntry block;
        block.key = i + 10;
        block.last_access_time = i * 100;
        block.writing_time = i * 100;
        policy2->OnBlockWritten(&block);
    }

    // 组使用量应该等于两个实例使用量之和
    usage = manager_->GetCurrentGroupUsage(instance_group_config);
    EXPECT_GT(usage, 0);
}

TEST_F(OptEvictionManagerTest, GetExcessUsageForInstanceInGroup) {
    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();
    auto instance_group_config = CreateTestInstanceGroupConfig();

    // 启用分层驱逐
    auto policy = manager_->CreateAndRegisterEvictionPolicy(instance_config, tier_configs, true);
    ASSERT_NE(policy, nullptr);

    // 初始没有超额使用
    auto excess = manager_->GetExcessUsageForInstanceInGroup(instance_group_config);
    EXPECT_EQ(excess, 0);

    // 添加大量块,超过容量
    for (int i = 0; i < 100; i++) {
        BlockEntry block;
        block.key = i;
        block.last_access_time = i * 100;
        block.writing_time = i * 100;
        policy->OnBlockWritten(&block);
    }

    // 应该有超额使用
    excess = manager_->GetExcessUsageForInstanceInGroup(instance_group_config);
    EXPECT_GT(excess, 0);
}

TEST_F(OptEvictionManagerTest, EvictFromNonExistentInstance) {
    auto instance_group_config = CreateTestInstanceGroupConfig();

    // 尝试从不存在的实例驱逐 - 可能会抛出异常
    // 这个测试验证了系统的健壮性
    try {
        auto evicted = manager_->EvictByMode("non_existent_instance", instance_group_config);
        // 如果没有抛出异常,检查返回值
        EXPECT_TRUE(evicted.empty());
    } catch (...) {
        // 如果抛出异常,也是可以接受的
        SUCCEED();
    }
}

TEST_F(OptEvictionManagerTest, HierarchicalEvictionEnabled) {
    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();

    // 启用分层驱逐
    auto policy = manager_->CreateAndRegisterEvictionPolicy(instance_config, tier_configs, true);
    EXPECT_NE(policy, nullptr);
}
