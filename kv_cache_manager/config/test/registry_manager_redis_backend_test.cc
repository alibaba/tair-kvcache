#include <filesystem>
#include <memory>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/account.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

class RegistryManagerRedisBackendTest : public TESTBASE {
public:
    void SetUp() override;
    void TearDown() override {}

public:
    bool InitRegistryManager();
    void AddStorage(const std::string &global_unique_name);
    void CreateInstanceGroup(const std::string &name);
    void RegisterInstance(const std::string &instance_group, const std::string &instance_id);
    std::shared_ptr<ThreeFSStorageSpec> GetDefault3fsStorageSpec();

private:
    std::shared_ptr<RequestContext> request_context_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

void RegistryManagerRedisBackendTest::SetUp() {
    metrics_registry_ = std::make_shared<MetricsRegistry>();
    request_context_ = std::make_shared<RequestContext>("test");
    registry_manager_ = std::make_shared<RegistryManager>("test", metrics_registry_);
}

bool RegistryManagerRedisBackendTest::InitRegistryManager() {
    std::string uri =
        "redis://test_redis_user:test_redis_password@localhost:6379/?timeout_ms=1000&retry_count=3&cluster_name=test";
    registry_manager_ = std::make_shared<RegistryManager>(uri, metrics_registry_);
    return registry_manager_->Init();
}

void RegistryManagerRedisBackendTest::AddStorage(const std::string &global_unique_name) {
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(1);
    spec->set_root_path("/data/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, global_unique_name, spec);
    auto ec = registry_manager_->AddStorage(request_context_.get(), storage_config);
    ASSERT_EQ(EC_OK, ec);
}

void RegistryManagerRedisBackendTest::CreateInstanceGroup(const std::string &name) {
    std::shared_ptr<InstanceGroup> instance_group = std::make_shared<InstanceGroup>();
    auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
    auto cache_config = std::make_shared<CacheConfig>();
    cache_config->meta_indexer_config_ = meta_indexer_config;
    cache_config->cache_prefer_strategy_ = CachePreferStrategy::CPS_PREFER_3FS;
    auto cache_reclaim_stratrgy = std::make_shared<CacheReclaimStrategy>();
    cache_config->set_reclaim_strategy(cache_reclaim_stratrgy);
    instance_group->set_cache_config(cache_config);
    instance_group->set_name(name);
    instance_group->set_version(0);
    instance_group->set_user_data("test");
    auto ec = registry_manager_->CreateInstanceGroup(request_context_.get(), *instance_group);
    ASSERT_EQ(EC_OK, ec);
}

void RegistryManagerRedisBackendTest::RegisterInstance(const std::string &instance_group,
                                                       const std::string &instance_id) {
    LocationSpecInfo info;
    ModelDeployment model_deployment;
    auto ec = registry_manager_->RegisterInstance(
        request_context_.get(), instance_group, instance_id, 1024, {info}, model_deployment);
    ASSERT_EQ(EC_OK, ec);
}

TEST_F(RegistryManagerRedisBackendTest, TestInit) {
    // test success
    ASSERT_TRUE(InitRegistryManager());

    // fail back to local
    std::string uri = "";
    registry_manager_ = std::make_shared<RegistryManager>(uri, metrics_registry_);
    ASSERT_TRUE(registry_manager_->Init());

    // invalid storage type
    uri = "test://";
    registry_manager_ = std::make_shared<RegistryManager>(uri, metrics_registry_);
    ASSERT_FALSE(registry_manager_->Init());

    // no cluster name
    uri = "redis://test_redis_user:test_redis_password@localhost:6379/?timeout_ms=1000&retry_count=3";
    registry_manager_ = std::make_shared<RegistryManager>(uri, metrics_registry_);
    ASSERT_FALSE(registry_manager_->Init());
}

TEST_F(RegistryManagerRedisBackendTest, TestRecover) {
    // do init
    ASSERT_TRUE(InitRegistryManager());
    AddStorage("storage1");
    AddStorage("storage2");
    AddStorage("storage3");
    ASSERT_EQ(EC_OK, registry_manager_->DisableStorage(request_context_.get(), "storage2"));
    ASSERT_EQ(EC_OK, registry_manager_->DisableStorage(request_context_.get(), "storage3"));
    CreateInstanceGroup("group1");
    CreateInstanceGroup("group2");
    RegisterInstance("group1", "instance1");
    RegisterInstance("group1", "instance2");
    ASSERT_EQ(EC_OK, registry_manager_->AddAccount(request_context_.get(), "user1", "pwd1", AccountRole::ROLE_USER));
    ASSERT_EQ(EC_OK, registry_manager_->AddAccount(request_context_.get(), "user2", "pwd2", AccountRole::ROLE_USER));

    // do recover
    ASSERT_TRUE(InitRegistryManager());
    {
        ASSERT_EQ(EC_OK, registry_manager_->DoRecover());
        auto storage1 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage1");
        ASSERT_TRUE(storage1);
        ASSERT_TRUE(storage1->Available());
        const auto storage_config1 = storage1->GetStorageConfig();
        ASSERT_EQ("storage1", storage_config1.global_unique_name());
        ASSERT_TRUE(storage_config1.is_available());
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, storage_config1.type());
        auto storage2 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage2");
        ASSERT_TRUE(storage2);
        ASSERT_FALSE(storage2->Available());
        const auto storage_config2 = storage2->GetStorageConfig();
        ASSERT_EQ("storage2", storage_config2.global_unique_name());
        ASSERT_FALSE(storage_config2.is_available());
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, storage_config2.type());
        auto storage3 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage3");
        ASSERT_TRUE(storage3);
        ASSERT_FALSE(storage3->Available());
        const auto storage_config3 = storage3->GetStorageConfig();
        ASSERT_EQ("storage3", storage_config3.global_unique_name());
        ASSERT_FALSE(storage_config3.is_available());
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, storage_config3.type());

        auto [ec1, instance_group1] = registry_manager_->GetInstanceGroup(request_context_.get(), "group1");
        ASSERT_EQ(EC_OK, ec1);
        ASSERT_TRUE(instance_group1);
        ASSERT_EQ("group1", instance_group1->name());
        ASSERT_EQ(0, instance_group1->version());
        ASSERT_EQ("test", instance_group1->user_data());
        auto [ec2, instance_group2] = registry_manager_->GetInstanceGroup(request_context_.get(), "group2");
        ASSERT_EQ(EC_OK, ec2);
        ASSERT_TRUE(instance_group2);
        ASSERT_EQ("group2", instance_group2->name());
        ASSERT_EQ(0, instance_group2->version());
        ASSERT_EQ("test", instance_group2->user_data());

        auto instance_info1 = registry_manager_->GetInstanceInfo(request_context_.get(), "instance1");
        ASSERT_TRUE(instance_info1);
        ASSERT_EQ("instance1", instance_info1->instance_id());
        ASSERT_EQ("group1", instance_info1->instance_group_name());
        auto instance_info2 = registry_manager_->GetInstanceInfo(request_context_.get(), "instance2");
        ASSERT_TRUE(instance_info2);
        ASSERT_EQ("instance2", instance_info2->instance_id());
        ASSERT_EQ("group1", instance_info2->instance_group_name());

        auto account_iter1 = registry_manager_->accounts_.find("user1");
        ASSERT_NE(registry_manager_->accounts_.end(), account_iter1);
        ASSERT_EQ("user1", account_iter1->second->user_name());
        ASSERT_EQ(AccountRole::ROLE_USER, account_iter1->second->role());
        auto account_iter2 = registry_manager_->accounts_.find("user2");
        ASSERT_NE(registry_manager_->accounts_.end(), account_iter2);
        ASSERT_EQ("user2", account_iter2->second->user_name());
        ASSERT_EQ(AccountRole::ROLE_USER, account_iter2->second->role());
    }

    // do some update
    ASSERT_EQ(EC_OK, registry_manager_->DisableStorage(request_context_.get(), "storage1"));
    ASSERT_EQ(EC_OK, registry_manager_->EnableStorage(request_context_.get(), "storage2"));
    ASSERT_EQ(EC_OK, registry_manager_->RemoveStorage(request_context_.get(), "storage3"));
    ASSERT_EQ(EC_OK, registry_manager_->RemoveInstanceGroup(request_context_.get(), "group2"));
    ASSERT_EQ(EC_OK, registry_manager_->RemoveInstance(request_context_.get(), "group1", "instance2"));
    ASSERT_EQ(EC_OK, registry_manager_->DeleteAccount(request_context_.get(), "user2"));

    // do recover again
    ASSERT_EQ(EC_OK, registry_manager_->DoCleanup());
    ASSERT_EQ(EC_OK, registry_manager_->DoRecover());
    {
        auto storage1 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage1");
        ASSERT_TRUE(storage1);
        ASSERT_FALSE(storage1->Available());
        const auto storage_config1 = storage1->GetStorageConfig();
        ASSERT_EQ("storage1", storage_config1.global_unique_name());
        ASSERT_FALSE(storage_config1.is_available());
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, storage_config1.type());
        auto storage2 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage2");
        ASSERT_TRUE(storage2);
        ASSERT_TRUE(storage2->Available());
        const auto storage_config2 = storage2->GetStorageConfig();
        ASSERT_EQ("storage2", storage_config2.global_unique_name());
        ASSERT_TRUE(storage_config2.is_available());
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, storage_config2.type());
        auto storage3 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage3");
        ASSERT_FALSE(storage3);

        auto [ec1, instance_group1] = registry_manager_->GetInstanceGroup(request_context_.get(), "group1");
        ASSERT_EQ(EC_OK, ec1);
        ASSERT_TRUE(instance_group1);
        ASSERT_EQ("group1", instance_group1->name());
        ASSERT_EQ(0, instance_group1->version());
        ASSERT_EQ("test", instance_group1->user_data());
        auto [ec2, instance_group2] = registry_manager_->GetInstanceGroup(request_context_.get(), "group2");
        ASSERT_EQ(EC_NOENT, ec2);
        ASSERT_FALSE(instance_group2);

        auto instance_info1 = registry_manager_->GetInstanceInfo(request_context_.get(), "instance1");
        ASSERT_TRUE(instance_info1);
        ASSERT_EQ("instance1", instance_info1->instance_id());
        ASSERT_EQ("group1", instance_info1->instance_group_name());
        auto instance_info2 = registry_manager_->GetInstanceInfo(request_context_.get(), "instance2");
        ASSERT_FALSE(instance_info2);

        auto account_iter1 = registry_manager_->accounts_.find("user1");
        ASSERT_NE(registry_manager_->accounts_.end(), account_iter1);
        ASSERT_EQ("user1", account_iter1->second->user_name());
        ASSERT_EQ(AccountRole::ROLE_USER, account_iter1->second->role());
        auto account_iter2 = registry_manager_->accounts_.find("user2");
        ASSERT_EQ(registry_manager_->accounts_.end(), account_iter2);
    }

    // remove all and recover
    ASSERT_EQ(EC_OK, registry_manager_->RemoveStorage(request_context_.get(), "storage1"));
    ASSERT_EQ(EC_OK, registry_manager_->RemoveStorage(request_context_.get(), "storage2"));
    ASSERT_EQ(EC_OK, registry_manager_->RemoveInstance(request_context_.get(), "group1", "instance1"));
    ASSERT_EQ(EC_OK, registry_manager_->RemoveInstanceGroup(request_context_.get(), "group1"));
    ASSERT_EQ(EC_OK, registry_manager_->DeleteAccount(request_context_.get(), "user1"));
    ASSERT_TRUE(InitRegistryManager());
    {
        auto storage1 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage1");
        ASSERT_FALSE(storage1);
        auto storage2 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage2");
        ASSERT_FALSE(storage2);

        auto [ec1, instance_group1] = registry_manager_->GetInstanceGroup(request_context_.get(), "group1");
        ASSERT_EQ(EC_NOENT, ec1);
        ASSERT_FALSE(instance_group1);

        auto instance_info1 = registry_manager_->GetInstanceInfo(request_context_.get(), "instance1");
        ASSERT_FALSE(instance_info1);

        auto account_iter1 = registry_manager_->accounts_.find("user1");
        ASSERT_EQ(registry_manager_->accounts_.end(), account_iter1);
    }
}

std::shared_ptr<ThreeFSStorageSpec> RegistryManagerRedisBackendTest::GetDefault3fsStorageSpec() {
    auto root_path = GetPrivateTestRuntimeDataPath();
    std::filesystem::path p(root_path);
    auto parent = p.parent_path().parent_path();
    std::string root_dir = p.lexically_relative(parent);
    std::string mountpoint = parent.string();

    std::shared_ptr<ThreeFSStorageSpec> spec(new ThreeFSStorageSpec);
    spec->set_cluster_name("test");
    spec->set_mountpoint(mountpoint);
    spec->set_root_dir(root_dir);
    return spec;
}

TEST_F(RegistryManagerRedisBackendTest, TestUpdate) {
    // do init
    ASSERT_TRUE(InitRegistryManager());
    AddStorage("storage");
    CreateInstanceGroup("group3");
    RegisterInstance("group3", "instance3");
    ASSERT_EQ(EC_OK, registry_manager_->AddAccount(request_context_.get(), "user3", "pwd3", AccountRole::ROLE_USER));

    // do recover
    ASSERT_TRUE(InitRegistryManager());
    ASSERT_EQ(EC_OK, registry_manager_->DoRecover());
    {
        auto storage = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage");
        ASSERT_TRUE(storage);
        ASSERT_TRUE(storage->Available());
        const auto storage_config = storage->GetStorageConfig();
        ASSERT_EQ("storage", storage_config.global_unique_name());
        ASSERT_TRUE(storage_config.is_available());
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, storage_config.type());

        auto [ec, instance_group] = registry_manager_->GetInstanceGroup(request_context_.get(), "group3");
        ASSERT_EQ(EC_OK, ec);
        ASSERT_TRUE(instance_group);
        ASSERT_EQ("group3", instance_group->name());
        ASSERT_EQ(0, instance_group->version());
        ASSERT_EQ("test", instance_group->user_data());

        auto instance_info = registry_manager_->GetInstanceInfo(request_context_.get(), "instance3");
        ASSERT_TRUE(instance_info);
        ASSERT_EQ("instance3", instance_info->instance_id());
        ASSERT_EQ("group3", instance_info->instance_group_name());

        auto account_iter = registry_manager_->accounts_.find("user3");
        ASSERT_NE(registry_manager_->accounts_.end(), account_iter);
        ASSERT_EQ("user3", account_iter->second->user_name());
        ASSERT_EQ(AccountRole::ROLE_USER, account_iter->second->role());
    }

    // do update
    StorageConfig new_storage_config(DataStorageType::DATA_STORAGE_TYPE_HF3FS, "storage", GetDefault3fsStorageSpec());
    ASSERT_EQ(EC_OK, registry_manager_->UpdateStorage(request_context_.get(), new_storage_config, true));
    auto [ec, instance_group] = registry_manager_->GetInstanceGroup(request_context_.get(), "group3");
    ASSERT_EQ(EC_OK, ec);
    ASSERT_TRUE(instance_group);
    auto new_instance_group = *(instance_group);
    new_instance_group.set_user_data("new_test");
    new_instance_group.set_version(2);
    ASSERT_EQ(EC_OK, registry_manager_->UpdateInstanceGroup(request_context_.get(), new_instance_group, 0));

    // do recover again
    ASSERT_TRUE(InitRegistryManager());
    ASSERT_EQ(EC_OK, registry_manager_->DoRecover());
    {
        auto storage = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage");
        ASSERT_TRUE(storage);
        ASSERT_TRUE(storage->Available());
        const auto storage_config = storage->GetStorageConfig();
        ASSERT_EQ("storage", storage_config.global_unique_name());
        ASSERT_TRUE(storage_config.is_available());
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_HF3FS, storage_config.type());

        auto [ec, instance_group] = registry_manager_->GetInstanceGroup(request_context_.get(), "group3");
        ASSERT_EQ(EC_OK, ec);
        ASSERT_TRUE(instance_group);
        ASSERT_EQ("group3", instance_group->name());
        ASSERT_EQ(2, instance_group->version());
        ASSERT_EQ("new_test", instance_group->user_data());

        auto instance_info = registry_manager_->GetInstanceInfo(request_context_.get(), "instance3");
        ASSERT_TRUE(instance_info);
        ASSERT_EQ("instance3", instance_info->instance_id());
        ASSERT_EQ("group3", instance_info->instance_group_name());

        auto account_iter = registry_manager_->accounts_.find("user3");
        ASSERT_NE(registry_manager_->accounts_.end(), account_iter);
        ASSERT_EQ("user3", account_iter->second->user_name());
        ASSERT_EQ(AccountRole::ROLE_USER, account_iter->second->role());
    }
}

} // namespace kv_cache_manager
