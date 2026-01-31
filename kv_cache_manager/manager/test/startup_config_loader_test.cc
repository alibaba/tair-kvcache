#include <gtest/gtest.h>
#include <memory>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/startup_config_loader.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;

class StartupConfigLoaderTest : public TESTBASE {
public:
    void SetUp() override {}
    void TearDown() override {}

private:
    InstanceGroup MakeInstanceGroup();
    StorageConfig MakeStorageConfig();
};

// TODO 只测试了NFS，其他类型再加吧
TEST_F(StartupConfigLoaderTest, TestStartupConfigJsonize) {
    StartupConfig startup_config;
    startup_config.set_instance_group(MakeInstanceGroup());
    startup_config.set_storage_config(MakeStorageConfig());
    auto json = startup_config.ToJsonString();

    StartupConfig startup_config2;
    startup_config2.FromJsonString(json);

    // ASSERT
    // Compare InstanceGroup
    const InstanceGroup &ig1 = startup_config.instance_group();
    const InstanceGroup &ig2 = startup_config2.instance_group();

    ASSERT_EQ(ig1.name(), ig2.name());
    ASSERT_EQ(ig1.storage_candidates(), ig2.storage_candidates());
    ASSERT_EQ(ig1.global_quota_group_name(), ig2.global_quota_group_name());
    ASSERT_EQ(ig1.max_instance_count(), ig2.max_instance_count());
    ASSERT_EQ(ig1.user_data(), ig2.user_data());
    ASSERT_EQ(ig1.version(), ig2.version());

    // Compare InstanceGroupQuota
    const InstanceGroupQuota &quota1 = ig1.quota();
    const InstanceGroupQuota &quota2 = ig2.quota();
    ASSERT_EQ(quota1.capacity(), quota2.capacity());

    const std::vector<QuotaConfig> &quota_configs1 = quota1.quota_config();
    const std::vector<QuotaConfig> &quota_configs2 = quota2.quota_config();
    ASSERT_EQ(quota_configs1.size(), quota_configs2.size());

    if (!quota_configs1.empty() && !quota_configs2.empty()) {
        ASSERT_EQ(quota_configs1[0].capacity(), quota_configs2[0].capacity());
        ASSERT_EQ(quota_configs1[0].storage_spec(), quota_configs2[0].storage_spec());
    }

    // Compare CacheConfig
    ASSERT_NE(ig1.cache_config(), nullptr);
    ASSERT_NE(ig2.cache_config(), nullptr);

    const CacheConfig &cache_config1 = *(ig1.cache_config());
    const CacheConfig &cache_config2 = *(ig2.cache_config());

    ASSERT_EQ(cache_config1.cache_prefer_strategy(), cache_config2.cache_prefer_strategy());

    // Compare CacheReclaimStrategy
    const CacheReclaimStrategy &reclaim1 = *(cache_config1.reclaim_strategy());
    const CacheReclaimStrategy &reclaim2 = *(cache_config2.reclaim_strategy());

    ASSERT_EQ(reclaim1.storage_unique_name(), reclaim2.storage_unique_name());
    ASSERT_EQ(reclaim1.reclaim_policy(), reclaim2.reclaim_policy());
    ASSERT_EQ(reclaim1.trigger_period_seconds(), reclaim2.trigger_period_seconds());
    ASSERT_EQ(reclaim1.reclaim_step_size(), reclaim2.reclaim_step_size());
    ASSERT_EQ(reclaim1.reclaim_step_percentage(), reclaim2.reclaim_step_percentage());
    ASSERT_EQ(reclaim1.delay_before_delete_ms(), reclaim2.delay_before_delete_ms());

    // Compare TriggerStrategy
    const TriggerStrategy &trigger1 = reclaim1.trigger_strategy();
    const TriggerStrategy &trigger2 = reclaim2.trigger_strategy();

    ASSERT_EQ(trigger1.used_size(), trigger2.used_size());
    ASSERT_EQ(trigger1.used_percentage(), trigger2.used_percentage());

    // Compare MetaIndexerConfig
    ASSERT_NE(cache_config1.meta_indexer_config(), nullptr);
    ASSERT_NE(cache_config2.meta_indexer_config(), nullptr);

    const MetaIndexerConfig &meta1 = *(cache_config1.meta_indexer_config());
    const MetaIndexerConfig &meta2 = *(cache_config2.meta_indexer_config());

    ASSERT_EQ(meta1.GetMaxKeyCount(), meta2.GetMaxKeyCount());
    ASSERT_EQ(meta1.GetMutexShardNum(), meta2.GetMutexShardNum());
    ASSERT_EQ(meta1.GetBatchKeySize(), meta2.GetBatchKeySize());

    // Compare StorageConfig
    const StorageConfig &sc1 = startup_config.storage_config();
    const StorageConfig &sc2 = startup_config2.storage_config();

    ASSERT_EQ(sc1.type(), sc2.type());
    ASSERT_EQ(sc1.global_unique_name(), sc2.global_unique_name());

    // For NFS storage spec
    ASSERT_NE(sc1.storage_spec(), nullptr);
    ASSERT_NE(sc2.storage_spec(), nullptr);

    std::shared_ptr<NfsStorageSpec> nfs_spec1 = std::dynamic_pointer_cast<NfsStorageSpec>(sc1.storage_spec());
    std::shared_ptr<NfsStorageSpec> nfs_spec2 = std::dynamic_pointer_cast<NfsStorageSpec>(sc2.storage_spec());

    ASSERT_NE(nfs_spec1, nullptr);
    ASSERT_NE(nfs_spec2, nullptr);

    ASSERT_EQ(nfs_spec1->root_path(), nfs_spec2->root_path());
    ASSERT_EQ(nfs_spec1->key_count_per_file(), nfs_spec2->key_count_per_file());
}

TEST_F(StartupConfigLoaderTest, TestLoad) {
    {
        StartupConfigLoader loader;
        std::shared_ptr<RegistryManager> registry_manager(
            new RegistryManager("local://fake", std::make_shared<MetricsRegistry>()));
        auto request_context1 = std::make_shared<RequestContext>("traceid_123").get();
        auto request_context2 = std::make_shared<RequestContext>("traceid_456").get();
        registry_manager->Init();
        loader.Init(registry_manager);
        loader.Load("");
        auto [ec, ig] = registry_manager->GetInstanceGroup(request_context1, "default");
        ASSERT_EQ(ErrorCode::EC_OK, ec);
        ASSERT_TRUE(ig);
        auto [ec2, config_vec] = registry_manager->ListStorage(request_context2);
        ASSERT_EQ(ErrorCode::EC_OK, ec2);
        ASSERT_EQ(1, config_vec.size());
    }

    {
        StartupConfigLoader loader;
        std::shared_ptr<RegistryManager> registry_manager(
            new RegistryManager("local://fake", std::make_shared<MetricsRegistry>()));
        auto request_context1 = std::make_shared<RequestContext>("traceid_123").get();
        auto request_context2 = std::make_shared<RequestContext>("traceid_456").get();
        registry_manager->Init();
        loader.Init(registry_manager);
        loader.Load(GetPrivateTestDataPath() + "test_startup_config_load.json");
        auto [ec, ig] = registry_manager->GetInstanceGroup(request_context1, "default");
        ASSERT_EQ(ErrorCode::EC_OK, ec);
        ASSERT_TRUE(ig);
        auto [ec2, config_vec] = registry_manager->ListStorage(request_context2);
        ASSERT_EQ(ErrorCode::EC_OK, ec2);
        ASSERT_EQ(1, config_vec.size());
    }
}

InstanceGroup StartupConfigLoaderTest::MakeInstanceGroup() {
    InstanceGroup instance_group;
    // Set basic instance group properties
    instance_group.set_name("default");
    instance_group.set_storage_candidates({"nfs_01"});
    instance_group.set_global_quota_group_name("default_quota_group");
    instance_group.set_max_instance_count(100);
    instance_group.set_user_data("{\"description\": \"Default instance group for KV Cache Manager\"}");
    instance_group.set_version(1);

    // Set quota configuration
    QuotaConfig quota_config;
    quota_config.set_capacity(10737418240LL); // 10GB
    quota_config.set_storage_type(DataStorageType::DATA_STORAGE_TYPE_NFS);

    InstanceGroupQuota quota;
    quota.set_capacity(10737418240LL); // 10GB
    quota.set_quota_config({quota_config});
    instance_group.set_quota(quota);

    // Set cache configuration
    // Create trigger strategy
    TriggerStrategy trigger_strategy;
    trigger_strategy.set_used_size(1073741824); // 1GB (int32_t max is ~2GB)
    trigger_strategy.set_used_percentage(0.8);

    // Create reclaim strategy
    std::shared_ptr<CacheReclaimStrategy> reclaim_strategy = std::make_shared<CacheReclaimStrategy>();
    reclaim_strategy->set_storage_unique_name("nfs_01");
    reclaim_strategy->set_reclaim_policy(ReclaimPolicy::POLICY_LRU);
    reclaim_strategy->set_trigger_strategy(trigger_strategy);
    reclaim_strategy->set_trigger_period_seconds(60);
    reclaim_strategy->set_reclaim_step_size(1073741824); // 1GB
    reclaim_strategy->set_reclaim_step_percentage(10);
    reclaim_strategy->set_delay_before_delete_ms(1000);

    // Create meta storage backend config
    auto meta_storage_backend_config = std::make_shared<MetaStorageBackendConfig>();
    meta_storage_backend_config->SetStorageType(META_LOCAL_BACKEND_TYPE_STR);
    meta_storage_backend_config->SetStorageUri("file:///tmp/meta_storage");

    // Create meta cache policy config
    auto meta_cache_policy_config = std::make_shared<MetaCachePolicyConfig>();
    meta_cache_policy_config->SetCapacity(10000);
    meta_cache_policy_config->SetType("LRU");

    // Create meta indexer config
    auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
    meta_indexer_config->SetMaxKeyCount(1000000);
    meta_indexer_config->SetMutexShardNum(16);
    meta_indexer_config->SetBatchKeySize(16);
    meta_indexer_config->SetMetaStorageBackendConfig(meta_storage_backend_config);
    meta_indexer_config->SetMetaCachePolicyConfig(meta_cache_policy_config);

    // Create cache config
    auto cache_config = std::make_shared<CacheConfig>();
    cache_config->set_reclaim_strategy(reclaim_strategy);
    cache_config->set_cache_prefer_strategy(CachePreferStrategy::CPS_PREFER_3FS);
    cache_config->set_meta_indexer_config(meta_indexer_config);

    instance_group.set_cache_config(cache_config);

    return instance_group;
}

StorageConfig StartupConfigLoaderTest::MakeStorageConfig() {
    std::shared_ptr<NfsStorageSpec> nfs_spec_ptr(new NfsStorageSpec());
    auto &nfs_spec = *nfs_spec_ptr;
    nfs_spec.set_root_path("/mnt/nfs");
    nfs_spec.set_key_count_per_file(5);
    StorageConfig config(DataStorageType::DATA_STORAGE_TYPE_NFS, "nfs_01", nfs_spec_ptr);
    return config;
}
