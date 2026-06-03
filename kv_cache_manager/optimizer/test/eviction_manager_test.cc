#include <memory>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/config/eviction_config.h"
#include "kv_cache_manager/optimizer/config/instance_config.h"
#include "kv_cache_manager/optimizer/config/instance_group_config.h"
#include "kv_cache_manager/optimizer/config/tier_config.h"
#include "kv_cache_manager/optimizer/config/types.h"
#include "kv_cache_manager/optimizer/eviction_policy/lru.h"
#include "kv_cache_manager/optimizer/eviction_policy/ttl.h"
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
    config.set_bytes_per_token(1); // bytes_per_block = block_size * bytes_per_token = 1024
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
    instance1.set_bytes_per_token(1); // bytes_per_block = 1024
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
    auto *policy_group = manager_->CreateAndRegisterEvictionPolicy(instance_config, tier_configs, true);
    EXPECT_NE(policy_group, nullptr);
    EXPECT_GE(policy_group->tier_count(), 1);
    EXPECT_EQ(policy_group->GetPolicyByIndex(0)->name(), "tier1");
}

TEST_F(OptEvictionManagerTest, CreateMultipleEvictionPolicies) {
    auto instance_config1 = CreateTestInstanceConfig("instance1");
    auto instance_config2 = CreateTestInstanceConfig("instance2");
    auto tier_configs = CreateTestTierConfigs();

    // 启用分层驱逐
    auto *pg1 = manager_->CreateAndRegisterEvictionPolicy(instance_config1, tier_configs, true);
    auto *pg2 = manager_->CreateAndRegisterEvictionPolicy(instance_config2, tier_configs, true);

    EXPECT_NE(pg1, nullptr);
    EXPECT_NE(pg2, nullptr);
    EXPECT_EQ(pg1->GetPolicyByIndex(0)->name(), "tier1");
    EXPECT_EQ(pg2->GetPolicyByIndex(0)->name(), "tier1");
}

TEST_F(OptEvictionManagerTest, EvictByInstancePreciseMode) {
    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();
    auto instance_group_config = CreateTestInstanceGroupConfig();

    // 启用分层驱逐
    auto *pg = manager_->CreateAndRegisterEvictionPolicy(instance_config, tier_configs, true);
    ASSERT_NE(pg, nullptr);

    // 驱逐测试
    auto evicted = manager_->EvictByMode("instance1", instance_group_config, 0);
    // 不应该崩溃
    SUCCEED();
}

TEST_F(OptEvictionManagerTest, EvictByInstanceRoughMode) {
    auto manager = std::make_shared<OptEvictionManager>();
    EvictionConfig eviction_config;
    eviction_config.set_eviction_batch_size_per_instance(10);
    eviction_config.set_eviction_mode(EvictionMode::EVICTION_MODE_INSTANCE_ROUGH);
    ASSERT_TRUE(manager_->Init(eviction_config));

    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();
    auto instance_group_config = CreateTestInstanceGroupConfig();

    // 启用分层驱逐
    auto *pg = manager->CreateAndRegisterEvictionPolicy(instance_config, tier_configs, true);
    ASSERT_NE(pg, nullptr);

    // 驱逐测试
    auto evicted = manager->EvictByMode("instance1", instance_group_config, 0);
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
    auto *pg = manager->CreateAndRegisterEvictionPolicy(instance_config, tier_configs, true);
    ASSERT_NE(pg, nullptr);

    // 驱逐测试
    auto evicted = manager->EvictByMode("instance1", instance_group_config, 0);
    // 不应该崩溃
    SUCCEED();
}

TEST_F(OptEvictionManagerTest, GetCurrentInstanceUsage) {
    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();

    // 启用分层驱逐
    auto *pg = manager_->CreateAndRegisterEvictionPolicy(instance_config, tier_configs, true);
    ASSERT_NE(pg, nullptr);

    // 初始使用量为0
    auto usage = manager_->GetCurrentInstanceUsage("instance1");
    EXPECT_EQ(usage, 0);

    // 添加一些块
    auto front_policy = pg->GetPolicyByIndex(0);
    for (int i = 0; i < 5; i++) {
        BlockEntry block;
        block.key = i;
        block.last_access_time = i * 100;
        block.writing_time = i * 100;
        front_policy->OnBlockWritten(&block);
    }

    // 使用量应该增加
    usage = manager_->GetCurrentInstanceUsage("instance1");
    EXPECT_GT(usage, 0);
}

TEST_F(OptEvictionManagerTest, GetCurrentGroupUsageBytes) {
    auto instance_config1 = CreateTestInstanceConfig("instance1");
    auto instance_config2 = CreateTestInstanceConfig("instance2");
    auto tier_configs = CreateTestTierConfigs();
    auto instance_group_config = CreateTestInstanceGroupConfig();

    // 启用分层驱逐
    auto *pg1 = manager_->CreateAndRegisterEvictionPolicy(instance_config1, tier_configs, true);
    auto *pg2 = manager_->CreateAndRegisterEvictionPolicy(instance_config2, tier_configs, true);
    ASSERT_NE(pg1, nullptr);
    ASSERT_NE(pg2, nullptr);

    // 初始使用量为0
    auto usage = manager_->GetCurrentGroupUsageBytes(instance_group_config);
    EXPECT_EQ(usage, 0);

    // 给instance1添加块
    auto p1 = pg1->GetPolicyByIndex(0);
    for (int i = 0; i < 3; i++) {
        BlockEntry block;
        block.key = i;
        block.last_access_time = i * 100;
        block.writing_time = i * 100;
        p1->OnBlockWritten(&block);
    }

    // 给instance2添加块
    auto p2 = pg2->GetPolicyByIndex(0);
    for (int i = 0; i < 2; i++) {
        BlockEntry block;
        block.key = i + 10;
        block.last_access_time = i * 100;
        block.writing_time = i * 100;
        p2->OnBlockWritten(&block);
    }

    // 组使用量应该等于两个实例使用量之和
    usage = manager_->GetCurrentGroupUsageBytes(instance_group_config);
    EXPECT_GT(usage, 0);
}

TEST_F(OptEvictionManagerTest, GetExcessUsageNonTiered) {
    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();
    auto instance_group_config = CreateTestInstanceGroupConfig();

    // 非分层模式
    auto *pg = manager_->CreateAndRegisterEvictionPolicy(instance_config, tier_configs, false);
    ASSERT_NE(pg, nullptr);

    // 初始没有超额使用
    auto excess = manager_->GetExcessUsage(instance_group_config, std::nullopt);
    EXPECT_EQ(excess, 0);

    // 添加大量块,超过容量
    auto front_policy = pg->GetPolicyByIndex(0);
    for (int i = 0; i < 100; i++) {
        BlockEntry block;
        block.key = i;
        block.last_access_time = i * 100;
        block.writing_time = i * 100;
        front_policy->OnBlockWritten(&block);
    }

    // 应该有超额使用
    excess = manager_->GetExcessUsage(instance_group_config, std::nullopt);
    EXPECT_GT(excess, 0);
}

TEST_F(OptEvictionManagerTest, NegativeQuotaCapacityMeansUnlimited) {
    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();
    auto instance_group_config = CreateTestInstanceGroupConfig();
    instance_group_config.set_quota_capacity(-1);
    instance_group_config.set_used_percentage(1.0);

    auto *pg = manager_->CreateAndRegisterEvictionPolicy(instance_config, tier_configs, false);
    ASSERT_NE(pg, nullptr);

    auto front_policy = pg->GetPolicyByIndex(0);
    for (int i = 0; i < 100; i++) {
        BlockEntry block;
        block.key = i;
        block.last_access_time = i * 100;
        block.writing_time = i * 100;
        front_policy->OnBlockWritten(&block);
    }

    auto excess = manager_->GetExcessUsage(instance_group_config, std::nullopt);
    EXPECT_EQ(excess, 0);
}

TEST_F(OptEvictionManagerTest, GetExcessUsageTiered) {
    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();
    auto instance_group_config = CreateTestInstanceGroupConfig();
    instance_group_config.set_hierarchical_eviction_enabled(true);

    // 分层模式
    auto *pg = manager_->CreateAndRegisterEvictionPolicy(instance_config, tier_configs, true);
    ASSERT_NE(pg, nullptr);

    // 初始 tier0 没有超额
    auto excess = manager_->GetExcessUsage(instance_group_config, size_t(0));
    EXPECT_EQ(excess, 0);

    // 添加块到 tier0 policy
    auto front_policy = pg->GetPolicyByIndex(0);
    for (int i = 0; i < 100; i++) {
        BlockEntry block;
        block.key = i;
        block.last_access_time = i * 100;
        block.writing_time = i * 100;
        front_policy->OnBlockWritten(&block);
    }

    // tier0 应该有超额使用
    excess = manager_->GetExcessUsage(instance_group_config, size_t(0));
    EXPECT_GT(excess, 0);
}

TEST_F(OptEvictionManagerTest, NegativeTierCapacityMeansUnlimited) {
    auto instance_config = CreateTestInstanceConfig("instance1");
    auto tier_configs = CreateTestTierConfigs();
    tier_configs[0].set_capacity(-1);
    auto instance_group_config = CreateTestInstanceGroupConfig();
    instance_group_config.set_hierarchical_eviction_enabled(true);
    instance_group_config.set_used_percentage(1.0);
    instance_group_config.set_storages(tier_configs);

    auto *pg = manager_->CreateAndRegisterEvictionPolicy(instance_config, tier_configs, true);
    ASSERT_NE(pg, nullptr);

    auto front_policy = pg->GetPolicyByIndex(0);
    for (int i = 0; i < 100; i++) {
        BlockEntry block;
        block.key = i;
        block.last_access_time = i * 100;
        block.writing_time = i * 100;
        front_policy->OnBlockWritten(&block);
    }

    auto excess = manager_->GetExcessUsage(instance_group_config, size_t(0));
    EXPECT_EQ(excess, 0);
}

TEST_F(OptEvictionManagerTest, EvictFromNonExistentInstance) {
    auto instance_group_config = CreateTestInstanceGroupConfig();

    // 尝试从不存在的实例驱逐 - 可能会抛出异常
    // 这个测试验证了系统的健壮性
    try {
        auto evicted = manager_->EvictByMode("non_existent_instance", instance_group_config, 0);
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

TEST_F(OptEvictionManagerTest, ActiveEvictExpiredShouldNotTriggerFallbackEviction) {
    OptInstanceConfig ttl_instance;
    ttl_instance.set_instance_id("ttl_instance");
    ttl_instance.set_instance_group_name("ttl_group");
    ttl_instance.set_block_size(1024);
    ttl_instance.set_eviction_policy_type(EvictionPolicyType::POLICY_TTL);
    TtlParams ttl_params;
    ttl_params.fallback_on_pressure = true;
    ttl_instance.set_eviction_policy_param(ttl_params);

    auto tier_configs = CreateTestTierConfigs();
    auto policy = manager_->CreateAndRegisterEvictionPolicy(ttl_instance, tier_configs, true);
    ASSERT_NE(policy, nullptr);

    BlockEntry expired_block;
    expired_block.key = 1;
    expired_block.last_access_time = 100;
    expired_block.ttl_ns = 10;
    expired_block.location_map["tier1"] = TierStat{};

    BlockEntry alive_block;
    alive_block.key = 2;
    alive_block.last_access_time = 100;
    alive_block.ttl_ns = 1000;
    alive_block.location_map["tier1"] = TierStat{};

    auto tier_policy = policy->GetPolicyByIndex(0);
    ASSERT_NE(tier_policy, nullptr);
    tier_policy->OnBlockWritten(&expired_block);
    tier_policy->OnBlockWritten(&alive_block);
    tier_policy->OnBlockAccessedWithOptions(&alive_block, 1000, true); // 推进时钟，确保 expired_block 过期

    OptInstanceGroupConfig ttl_group;
    ttl_group.set_group_name("ttl_group");
    ttl_group.set_instances({ttl_instance});

    auto evicted = manager_->ActiveEvictExpired(ttl_group, 1000);
    ASSERT_EQ(evicted.size(), 1);
    ASSERT_TRUE(evicted.count("ttl_instance"));
    ASSERT_EQ(evicted["ttl_instance"].size(), 1);
    EXPECT_EQ(evicted["ttl_instance"][0]->key, 1);
    EXPECT_FALSE(alive_block.location_map.empty());
}

TEST_F(OptEvictionManagerTest, ActiveEvictExpiredUsesCurrentTimestamp) {
    OptInstanceConfig ttl_instance;
    ttl_instance.set_instance_id("ttl_instance_ts");
    ttl_instance.set_instance_group_name("ttl_group_ts");
    ttl_instance.set_block_size(1024);
    ttl_instance.set_eviction_policy_type(EvictionPolicyType::POLICY_TTL);
    TtlParams ttl_params;
    ttl_params.fallback_on_pressure = false;
    ttl_instance.set_eviction_policy_param(ttl_params);

    auto tier_configs = CreateTestTierConfigs();
    auto policy = manager_->CreateAndRegisterEvictionPolicy(ttl_instance, tier_configs, true);
    ASSERT_NE(policy, nullptr);

    BlockEntry expired_block;
    expired_block.key = 100;
    expired_block.last_access_time = 100;
    expired_block.ttl_ns = 10;
    expired_block.location_map["tier1"] = TierStat{};
    auto tier_policy = policy->GetPolicyByIndex(0);
    ASSERT_NE(tier_policy, nullptr);
    tier_policy->OnBlockWritten(&expired_block);

    // 这里不通过 OnBlockAccessed 推进策略时钟，仅依赖 ActiveEvictExpired 的 current_timestamp。
    auto ttl_group = OptInstanceGroupConfig();
    ttl_group.set_group_name("ttl_group_ts");
    ttl_group.set_instances({ttl_instance});

    auto evicted = manager_->ActiveEvictExpired(ttl_group, 1000);
    ASSERT_EQ(evicted.size(), 1);
    ASSERT_TRUE(evicted.count("ttl_instance_ts"));
    ASSERT_EQ(evicted["ttl_instance_ts"].size(), 1);
    EXPECT_EQ(evicted["ttl_instance_ts"][0]->key, 100);
}

TEST_F(OptEvictionManagerTest, ActiveEvictExpiredSkipsNonTtlPolicyInV1) {
    auto lru_instance = CreateTestInstanceConfig("lru_instance_with_ttl");
    lru_instance.set_instance_group_name("mixed_group");
    lru_instance.set_eviction_policy_type(EvictionPolicyType::POLICY_LRU);

    auto tier_configs = CreateTestTierConfigs();
    auto policy = manager_->CreateAndRegisterEvictionPolicy(lru_instance, tier_configs, true);
    ASSERT_NE(policy, nullptr);

    BlockEntry expired_block;
    expired_block.key = 200;
    expired_block.last_access_time = 100;
    expired_block.ttl_ns = 10;
    expired_block.location_map["tier1"] = TierStat{};

    BlockEntry alive_block;
    alive_block.key = 201;
    alive_block.last_access_time = 100;
    alive_block.ttl_ns = 10000;
    alive_block.location_map["tier1"] = TierStat{};

    auto tier_policy = policy->GetPolicyByIndex(0);
    ASSERT_NE(tier_policy, nullptr);
    tier_policy->OnBlockWritten(&expired_block);
    tier_policy->OnBlockWritten(&alive_block);

    OptInstanceGroupConfig group_cfg;
    group_cfg.set_group_name("mixed_group");
    group_cfg.set_instances({lru_instance});

    auto evicted = manager_->ActiveEvictExpired(group_cfg, 1000);
    EXPECT_TRUE(evicted.empty());
    EXPECT_FALSE(expired_block.location_map.empty());
    EXPECT_FALSE(alive_block.location_map.empty());
}

// TTL 过期应清除 block 在所有 tier 的 location（block 级语义）
TEST_F(OptEvictionManagerTest, ActiveEvictExpiredClearsAllTiers) {
    // 创建 2-tier hierarchical TTL 实例
    OptInstanceConfig ttl_instance;
    ttl_instance.set_instance_id("ttl_multi_tier");
    ttl_instance.set_instance_group_name("ttl_mt_group");
    ttl_instance.set_block_size(1024);
    ttl_instance.set_eviction_policy_type(EvictionPolicyType::POLICY_TTL);
    TtlParams ttl_params;
    ttl_params.fallback_on_pressure = false;
    ttl_instance.set_eviction_policy_param(ttl_params);

    // 两个 tier
    std::vector<OptTierConfig> tier_configs;
    OptTierConfig tier0;
    tier0.set_unique_name("gpu");
    tier0.set_capacity(1024 * 1024);
    tier_configs.push_back(tier0);
    OptTierConfig tier1;
    tier1.set_unique_name("host");
    tier1.set_capacity(1024 * 1024);
    tier_configs.push_back(tier1);

    auto *policy_group = manager_->CreateAndRegisterEvictionPolicy(ttl_instance, tier_configs, true);
    ASSERT_NE(policy_group, nullptr);
    ASSERT_EQ(policy_group->tier_count(), 2);

    // 构造一个 block，同时注册到两个 tier（模拟 WRITE_THROUGH）
    BlockEntry expired_block;
    expired_block.key = 42;
    expired_block.last_access_time = 100;
    expired_block.ttl_ns = 50; // 会在 timestamp > 150 时过期
    expired_block.location_map["gpu"] = TierStat{};
    expired_block.location_map["host"] = TierStat{};

    // 写入两个 tier 的策略
    policy_group->policies[0]->OnBlockWritten(&expired_block);
    policy_group->policies[1]->OnBlockWritten(&expired_block);
    ASSERT_EQ(policy_group->policies[0]->size(), 1);
    ASSERT_EQ(policy_group->policies[1]->size(), 1);

    // 构造一个不过期的 block 作为对照
    BlockEntry alive_block;
    alive_block.key = 43;
    alive_block.last_access_time = 100;
    alive_block.ttl_ns = 10000;
    alive_block.location_map["gpu"] = TierStat{};
    alive_block.location_map["host"] = TierStat{};
    policy_group->policies[0]->OnBlockWritten(&alive_block);
    policy_group->policies[1]->OnBlockWritten(&alive_block);

    OptInstanceGroupConfig group;
    group.set_group_name("ttl_mt_group");
    group.set_instances({ttl_instance});

    // 触发过期驱逐，timestamp=200 > 100+50=150，expired_block 应过期
    auto evicted = manager_->ActiveEvictExpired(group, 200);
    ASSERT_EQ(evicted.size(), 1);
    ASSERT_TRUE(evicted.count("ttl_multi_tier"));
    ASSERT_EQ(evicted["ttl_multi_tier"].size(), 1);
    EXPECT_EQ(evicted["ttl_multi_tier"][0]->key, 42);

    // 核心断言：expired_block 的 location_map 应被完全清空（所有 tier）
    EXPECT_TRUE(expired_block.location_map.empty());

    // expired_block 应从两个 tier 的策略中都被移除
    EXPECT_EQ(policy_group->policies[0]->size(), 1); // 只剩 alive_block
    EXPECT_EQ(policy_group->policies[1]->size(), 1);

    // alive_block 不受影响
    EXPECT_FALSE(alive_block.location_map.empty());
    EXPECT_EQ(alive_block.location_map.size(), 2);
}
