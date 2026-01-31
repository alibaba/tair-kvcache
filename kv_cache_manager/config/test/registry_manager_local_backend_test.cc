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

class RegistryManagerLocalBackendTest : public TESTBASE {
public:
    void SetUp() override;
    void TearDown() override {}

public:
    bool InitRegistryManager(const std::string &uri);
    void AddNfsStorage(const std::string &global_unique_name, int key_count_per_file);
    void CreateInstanceGroup(const std::string &name);
    void RegisterInstance(const std::string &instance_group, const std::string &instance_id);
    std::shared_ptr<NfsStorageSpec> GetDefaultNfsStorageSpec();

private:
    std::shared_ptr<RequestContext> request_context_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

void RegistryManagerLocalBackendTest::SetUp() {
    metrics_registry_ = std::make_shared<MetricsRegistry>();
    request_context_ = std::make_shared<RequestContext>("test");
}

bool RegistryManagerLocalBackendTest::InitRegistryManager(const std::string &uri) {
    registry_manager_ = std::make_shared<RegistryManager>(uri, metrics_registry_);
    return registry_manager_->Init();
}

void RegistryManagerLocalBackendTest::AddNfsStorage(const std::string &global_unique_name, int key_count_per_file) {
    std::shared_ptr<NfsStorageSpec> spec = GetDefaultNfsStorageSpec();
    spec->set_key_count_per_file(key_count_per_file);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, global_unique_name, spec);
    auto ec = registry_manager_->AddStorage(request_context_.get(), storage_config);
    ASSERT_EQ(EC_OK, ec);
}

void RegistryManagerLocalBackendTest::CreateInstanceGroup(const std::string &name) {
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
    instance_group->set_user_data("test test\ttest"); // support space, \t, etc.
    auto ec = registry_manager_->CreateInstanceGroup(request_context_.get(), *instance_group);
    ASSERT_EQ(EC_OK, ec);
}

void RegistryManagerLocalBackendTest::RegisterInstance(const std::string &instance_group,
                                                       const std::string &instance_id) {
    LocationSpecInfo info;
    ModelDeployment model_deployment;
    auto ec = registry_manager_->RegisterInstance(
        request_context_.get(), instance_group, instance_id, 1024, {info}, model_deployment);
    ASSERT_EQ(EC_OK, ec);
}

std::shared_ptr<NfsStorageSpec> RegistryManagerLocalBackendTest::GetDefaultNfsStorageSpec() {
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(1);
    spec->set_root_path(GetPrivateTestRuntimeDataPath() + "/nfs_root/");
    return spec;
}

TEST_F(RegistryManagerLocalBackendTest, TestInit) {
    // test success without local path
    ASSERT_TRUE(InitRegistryManager(/*uri*/ "local://?cluster_name=test"));

    // test success with local path
    std::string local_path = GetPrivateTestRuntimeDataPath() + "_registry_local_backend_file1";
    std::string uri = "local://" + local_path + "?cluster_name=test";
    ASSERT_TRUE(InitRegistryManager(uri));
}

TEST_F(RegistryManagerLocalBackendTest, TestStorageManagement) {
    std::string local_path = GetPrivateTestRuntimeDataPath() + "_registry_local_backend_storage_test";
    std::string uri = "local://" + local_path + "?cluster_name=test";
    ASSERT_TRUE(InitRegistryManager(uri));

    // Test AddStorage
    AddNfsStorage("storage1", 1);
    AddNfsStorage("storage2", 2); // Different key count per file

    // Verify storage was added
    auto storage1 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage1");
    ASSERT_TRUE(storage1);
    ASSERT_TRUE(storage1->Available());
    ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, storage1->GetStorageConfig().type());

    auto storage2 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage2");
    ASSERT_TRUE(storage2);
    ASSERT_TRUE(storage2->Available());
    ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, storage2->GetStorageConfig().type());

    // Test ListStorage
    auto [ec_list, storage_configs] = registry_manager_->ListStorage(request_context_.get());
    ASSERT_EQ(EC_OK, ec_list);
    ASSERT_EQ(2, storage_configs.size());

    // Test DisableStorage
    auto ec_disable = registry_manager_->DisableStorage(request_context_.get(), "storage1");
    ASSERT_EQ(EC_OK, ec_disable);

    storage1 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage1");
    ASSERT_TRUE(storage1);
    ASSERT_FALSE(storage1->Available());

    // Test EnableStorage
    auto ec_enable = registry_manager_->EnableStorage(request_context_.get(), "storage1");
    ASSERT_EQ(EC_OK, ec_enable);

    storage1 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage1");
    ASSERT_TRUE(storage1);
    ASSERT_TRUE(storage1->Available());

    // Test UpdateStorage
    std::shared_ptr<NfsStorageSpec> new_spec = GetDefaultNfsStorageSpec();
    new_spec->set_root_path(GetPrivateTestRuntimeDataPath() + "/new_nfs_root/");
    StorageConfig new_storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "storage1", new_spec);
    auto ec_update = registry_manager_->UpdateStorage(request_context_.get(), new_storage_config, true);
    ASSERT_EQ(EC_OK, ec_update);

    // Check that the new root path is updated
    auto updated_storage1 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage1");
    ASSERT_TRUE(updated_storage1);
    ASSERT_EQ(
        GetPrivateTestRuntimeDataPath() + "/new_nfs_root/",
        std::dynamic_pointer_cast<NfsStorageSpec>(updated_storage1->GetStorageConfig().storage_spec())->root_path());

    // Test RemoveStorage
    auto ec_remove = registry_manager_->RemoveStorage(request_context_.get(), "storage2");
    ASSERT_EQ(EC_OK, ec_remove);

    storage2 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage2");
    ASSERT_FALSE(storage2);
}

TEST_F(RegistryManagerLocalBackendTest, TestInstanceGroupManagement) {
    std::string local_path = GetPrivateTestRuntimeDataPath() + "_registry_local_backend_group_test";
    std::string uri = "local://" + local_path + "?cluster_name=test";
    ASSERT_TRUE(InitRegistryManager(uri));

    // Test CreateInstanceGroup
    CreateInstanceGroup("group1");
    CreateInstanceGroup("group2");

    // Test GetInstanceGroup
    auto [ec_get, instance_group] = registry_manager_->GetInstanceGroup(request_context_.get(), "group1");
    ASSERT_EQ(EC_OK, ec_get);
    ASSERT_TRUE(instance_group);
    ASSERT_EQ("group1", instance_group->name());
    ASSERT_EQ(0, instance_group->version());
    ASSERT_EQ("test test\ttest", instance_group->user_data());

    // Test ListInstanceGroup
    auto [ec_list, instance_groups] = registry_manager_->ListInstanceGroup(request_context_.get());
    ASSERT_EQ(EC_OK, ec_list);
    ASSERT_EQ(2, instance_groups.size());

    // Test UpdateInstanceGroup
    auto new_instance_group = *(instance_group);
    new_instance_group.set_user_data("updated_test");
    new_instance_group.set_version(1);
    auto ec_update = registry_manager_->UpdateInstanceGroup(request_context_.get(), new_instance_group, 0);
    ASSERT_EQ(EC_OK, ec_update);

    auto [ec_get_updated, updated_instance_group] =
        registry_manager_->GetInstanceGroup(request_context_.get(), "group1");
    ASSERT_EQ(EC_OK, ec_get_updated);
    ASSERT_TRUE(updated_instance_group);
    ASSERT_EQ("group1", updated_instance_group->name());
    ASSERT_EQ(1, updated_instance_group->version());
    ASSERT_EQ("updated_test", updated_instance_group->user_data());

    // Test RemoveInstanceGroup
    auto ec_remove = registry_manager_->RemoveInstanceGroup(request_context_.get(), "group2");
    ASSERT_EQ(EC_OK, ec_remove);

    auto [ec_get_removed, removed_instance_group] =
        registry_manager_->GetInstanceGroup(request_context_.get(), "group2");
    ASSERT_EQ(EC_NOENT, ec_get_removed);
    ASSERT_FALSE(removed_instance_group);

    auto [ec_list_after_remove, instance_groups_after_remove] =
        registry_manager_->ListInstanceGroup(request_context_.get());
    ASSERT_EQ(EC_OK, ec_list_after_remove);
    ASSERT_EQ(1, instance_groups_after_remove.size());
}

TEST_F(RegistryManagerLocalBackendTest, TestInstanceRegistration) {
    std::string local_path = GetPrivateTestRuntimeDataPath() + "_registry_local_backend_instance_test";
    std::string uri = "local://" + local_path + "?cluster_name=test";
    ASSERT_TRUE(InitRegistryManager(uri));

    // Setup prerequisite
    AddNfsStorage("storage1", 1);
    CreateInstanceGroup("group1");

    // Test RegisterInstance
    RegisterInstance("group1", "instance1");
    RegisterInstance("group1", "instance2");

    // Test GetInstanceInfo
    auto instance_info1 = registry_manager_->GetInstanceInfo(request_context_.get(), "instance1");
    ASSERT_TRUE(instance_info1);
    ASSERT_EQ("instance1", instance_info1->instance_id());
    ASSERT_EQ("group1", instance_info1->instance_group_name());

    auto instance_info2 = registry_manager_->GetInstanceInfo(request_context_.get(), "instance2");
    ASSERT_TRUE(instance_info2);
    ASSERT_EQ("instance2", instance_info2->instance_id());
    ASSERT_EQ("group1", instance_info2->instance_group_name());

    // Test ListInstanceInfo
    auto [ec_list, instance_infos] = registry_manager_->ListInstanceInfo(request_context_.get(), "group1");
    ASSERT_EQ(EC_OK, ec_list);
    ASSERT_EQ(2, instance_infos.size());

    // Test RemoveInstance
    auto ec_remove = registry_manager_->RemoveInstance(request_context_.get(), "group1", "instance2");
    ASSERT_EQ(EC_OK, ec_remove);

    auto instance_info_removed = registry_manager_->GetInstanceInfo(request_context_.get(), "instance2");
    ASSERT_FALSE(instance_info_removed);

    auto [ec_list_after_remove, instance_infos_after_remove] =
        registry_manager_->ListInstanceInfo(request_context_.get(), "group1");
    ASSERT_EQ(EC_OK, ec_list_after_remove);
    ASSERT_EQ(1, instance_infos_after_remove.size());
}

TEST_F(RegistryManagerLocalBackendTest, TestAccountManagement) {
    std::string local_path = GetPrivateTestRuntimeDataPath() + "_registry_local_backend_account_test";
    std::string uri = "local://" + local_path + "?cluster_name=test";
    ASSERT_TRUE(InitRegistryManager(uri));

    // Test AddAccount
    auto ec_add1 = registry_manager_->AddAccount(request_context_.get(), "user1", "pwd1", AccountRole::ROLE_USER);
    ASSERT_EQ(EC_OK, ec_add1);

    auto ec_add2 = registry_manager_->AddAccount(request_context_.get(), "admin1", "pwd2", AccountRole::ROLE_ADMIN);
    ASSERT_EQ(EC_OK, ec_add2);

    // Test ListAccount
    auto [ec_list, accounts] = registry_manager_->ListAccount(request_context_.get());
    ASSERT_EQ(EC_OK, ec_list);
    ASSERT_EQ(2, accounts.size());

    // Verify account details
    bool found_user1 = false;
    bool found_admin1 = false;
    for (const auto &account : accounts) {
        if (account->user_name() == "user1") {
            ASSERT_EQ(AccountRole::ROLE_USER, account->role());
            found_user1 = true;
        } else if (account->user_name() == "admin1") {
            ASSERT_EQ(AccountRole::ROLE_ADMIN, account->role());
            found_admin1 = true;
        }
    }
    ASSERT_TRUE(found_user1);
    ASSERT_TRUE(found_admin1);

    // Test DeleteAccount
    auto ec_delete = registry_manager_->DeleteAccount(request_context_.get(), "user1");
    ASSERT_EQ(EC_OK, ec_delete);

    auto [ec_list_after_delete, accounts_after_delete] = registry_manager_->ListAccount(request_context_.get());
    ASSERT_EQ(EC_OK, ec_list_after_delete);
    ASSERT_EQ(1, accounts_after_delete.size());
    ASSERT_EQ("admin1", accounts_after_delete[0]->user_name());
}

TEST_F(RegistryManagerLocalBackendTest, TestRecover) {
    std::string local_path = GetPrivateTestRuntimeDataPath() + "_registry_local_backend_recover_test";
    std::string uri = "local://" + local_path + "?cluster_name=test";

    // Setup initial data
    {
        ASSERT_TRUE(InitRegistryManager(uri));
        AddNfsStorage("storage1", 1);
        AddNfsStorage("storage2", 2);
        ASSERT_EQ(EC_OK, registry_manager_->DisableStorage(request_context_.get(), "storage2"));
        CreateInstanceGroup("group1");
        CreateInstanceGroup("group2");
        RegisterInstance("group1", "instance1");
        RegisterInstance("group1", "instance2");
        ASSERT_EQ(EC_OK,
                  registry_manager_->AddAccount(request_context_.get(), "user1", "pwd1", AccountRole::ROLE_USER));
        ASSERT_EQ(EC_OK,
                  registry_manager_->AddAccount(request_context_.get(), "user2", "pwd2", AccountRole::ROLE_USER));
    }

    // Test recover
    {
        ASSERT_TRUE(InitRegistryManager(uri));
        ASSERT_EQ(EC_OK, registry_manager_->DoRecover());

        // Verify storage recovery
        auto storage1 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage1");
        ASSERT_TRUE(storage1);
        ASSERT_TRUE(storage1->Available());
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, storage1->GetStorageConfig().type());

        auto storage2 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage2");
        ASSERT_TRUE(storage2);
        ASSERT_FALSE(storage2->Available());
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, storage2->GetStorageConfig().type());

        // Verify instance group recovery
        auto [ec_group1, instance_group1] = registry_manager_->GetInstanceGroup(request_context_.get(), "group1");
        ASSERT_EQ(EC_OK, ec_group1);
        ASSERT_TRUE(instance_group1);
        ASSERT_EQ("group1", instance_group1->name());

        auto [ec_group2, instance_group2] = registry_manager_->GetInstanceGroup(request_context_.get(), "group2");
        ASSERT_EQ(EC_OK, ec_group2);
        ASSERT_TRUE(instance_group2);
        ASSERT_EQ("group2", instance_group2->name());

        // Verify instance info recovery
        auto instance_info1 = registry_manager_->GetInstanceInfo(request_context_.get(), "instance1");
        ASSERT_TRUE(instance_info1);
        ASSERT_EQ("instance1", instance_info1->instance_id());
        ASSERT_EQ("group1", instance_info1->instance_group_name());

        auto instance_info2 = registry_manager_->GetInstanceInfo(request_context_.get(), "instance2");
        ASSERT_TRUE(instance_info2);
        ASSERT_EQ("instance2", instance_info2->instance_id());
        ASSERT_EQ("group1", instance_info2->instance_group_name());

        // Verify account recovery
        auto [ec_accounts, accounts] = registry_manager_->ListAccount(request_context_.get());
        ASSERT_EQ(EC_OK, ec_accounts);
        ASSERT_EQ(2, accounts.size());

        // Test DoCleanup then DoRecover scenario
        // 1. Cleanup memory state
        auto ec_cleanup = registry_manager_->DoCleanup();
        ASSERT_EQ(EC_OK, ec_cleanup);

        // 2. Verify memory state is empty after cleanup
        auto [ec_groups_after_cleanup, groups_after_cleanup] =
            registry_manager_->ListInstanceGroup(request_context_.get());
        ASSERT_EQ(EC_OK, ec_groups_after_cleanup);
        ASSERT_TRUE(groups_after_cleanup.empty()) << "instance_group_configs_ should be empty after cleanup";

        auto [ec_accounts_after_cleanup, accounts_after_cleanup] =
            registry_manager_->ListAccount(request_context_.get());
        ASSERT_EQ(EC_OK, ec_accounts_after_cleanup);
        ASSERT_TRUE(accounts_after_cleanup.empty()) << "accounts_ should be empty after cleanup";

        // 3. Recover again from persistent storage
        auto ec_recover_again = registry_manager_->DoRecover();
        ASSERT_EQ(EC_OK, ec_recover_again);

        // 4. Verify data can be recovered again
        auto [ec_groups_recover_again, groups_recover_again] =
            registry_manager_->ListInstanceGroup(request_context_.get());
        ASSERT_EQ(EC_OK, ec_groups_recover_again);
        ASSERT_EQ(2, groups_recover_again.size()) << "Should recover 2 instance groups again";

        auto [ec_accounts_recover_again, accounts_recover_again] =
            registry_manager_->ListAccount(request_context_.get());
        ASSERT_EQ(EC_OK, ec_accounts_recover_again);
        ASSERT_EQ(2, accounts_recover_again.size()) << "Should recover 2 accounts again";

        // Verify storage recovery again
        auto storage1_again = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage1");
        ASSERT_TRUE(storage1_again);
        ASSERT_TRUE(storage1_again->Available());

        auto storage2_again = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage2");
        ASSERT_TRUE(storage2_again);
        ASSERT_FALSE(storage2_again->Available());

        // Verify instance info recovery again
        auto instance_info1_again = registry_manager_->GetInstanceInfo(request_context_.get(), "instance1");
        ASSERT_TRUE(instance_info1_again);
        ASSERT_EQ("instance1", instance_info1_again->instance_id());

        auto instance_info2_again = registry_manager_->GetInstanceInfo(request_context_.get(), "instance2");
        ASSERT_TRUE(instance_info2_again);
        ASSERT_EQ("instance2", instance_info2_again->instance_id());
    }
}

TEST_F(RegistryManagerLocalBackendTest, TestConfigSnapshot) {
    std::string local_path = GetPrivateTestRuntimeDataPath() + "_registry_local_backend_snapshot_test";
    std::string uri = "local://" + local_path + "?cluster_name=test";
    ASSERT_TRUE(InitRegistryManager(uri));
    // TODO add test after snapshot is implemented
    auto [ec_gen, snapshot] = registry_manager_->GenConfigSnapshot(request_context_.get());
    ASSERT_EQ(EC_UNIMPLEMENTED, ec_gen);
    auto ec_load = registry_manager_->LoadConfigSnapshot(request_context_.get(), snapshot);
    ASSERT_EQ(EC_UNIMPLEMENTED, ec_load);

    // // Setup test data
    // AddNfsStorage("storage1", 1);
    // CreateInstanceGroup("group1");
    // RegisterInstance("group1", "instance1");
    // ASSERT_EQ(EC_OK, registry_manager_->AddAccount(request_context_.get(), "user1", "pwd1", AccountRole::ROLE_USER));

    // // Test GenConfigSnapshot
    // auto [ec_gen, snapshot] = registry_manager_->GenConfigSnapshot(request_context_.get());
    // ASSERT_EQ(EC_OK, ec_gen);
    // // Snapshot should not be empty
    // ASSERT_FALSE(snapshot.empty());

    // // Test LoadConfigSnapshot
    // auto ec_load = registry_manager_->LoadConfigSnapshot(request_context_.get(), snapshot);
    // ASSERT_EQ(EC_OK, ec_load);
}

TEST_F(RegistryManagerLocalBackendTest, TestDoCleanupCompleteness) {
    std::string local_path = GetPrivateTestRuntimeDataPath() + "_registry_local_backend_cleanup_test";
    std::string uri = "local://" + local_path + "?cluster_name=test";

    // 1. Initialize and populate various data
    ASSERT_TRUE(InitRegistryManager(uri));

    // Add storage
    AddNfsStorage("storage1", 1);
    AddNfsStorage("storage2", 2);
    ASSERT_EQ(EC_OK, registry_manager_->DisableStorage(request_context_.get(), "storage2"));

    // Create instance groups
    CreateInstanceGroup("group1");
    CreateInstanceGroup("group2");

    // Register instances
    RegisterInstance("group1", "instance1");
    RegisterInstance("group1", "instance2");
    RegisterInstance("group2", "instance3");

    // Add accounts
    ASSERT_EQ(EC_OK, registry_manager_->AddAccount(request_context_.get(), "user1", "pwd1", AccountRole::ROLE_USER));
    ASSERT_EQ(EC_OK, registry_manager_->AddAccount(request_context_.get(), "admin1", "pwd2", AccountRole::ROLE_ADMIN));

    // 2. Verify data has been successfully added
    auto [ec_storage, storage_configs] = registry_manager_->ListStorage(request_context_.get());
    ASSERT_EQ(EC_OK, ec_storage);
    ASSERT_EQ(2, storage_configs.size());

    auto [ec_groups, groups] = registry_manager_->ListInstanceGroup(request_context_.get());
    ASSERT_EQ(EC_OK, ec_groups);
    ASSERT_EQ(2, groups.size());

    auto [ec_instances, instances] = registry_manager_->ListInstanceInfo(request_context_.get(), "group1");
    ASSERT_EQ(EC_OK, ec_instances);
    ASSERT_EQ(2, instances.size());

    auto [ec_accounts, accounts] = registry_manager_->ListAccount(request_context_.get());
    ASSERT_EQ(EC_OK, ec_accounts);
    ASSERT_EQ(2, accounts.size());

    // 3. Perform cleanup
    auto ec_cleanup = registry_manager_->DoCleanup();
    ASSERT_EQ(EC_OK, ec_cleanup);

    // 4. Verify all containers are empty
    // Verify instance group list is empty
    auto [ec_groups_after, groups_after] = registry_manager_->ListInstanceGroup(request_context_.get());
    ASSERT_EQ(EC_OK, ec_groups_after);
    ASSERT_TRUE(groups_after.empty()) << "instance_group_configs_ should be empty after cleanup";

    // Verify instance list is empty
    auto [ec_instances_after, instances_after] = registry_manager_->ListInstanceInfo(request_context_.get(), "group1");
    ASSERT_EQ(EC_OK, ec_instances_after);
    ASSERT_TRUE(instances_after.empty()) << "instance_infos_ should be empty after cleanup";

    // Verify account list is empty
    auto [ec_accounts_after, accounts_after] = registry_manager_->ListAccount(request_context_.get());
    ASSERT_EQ(EC_OK, ec_accounts_after);
    ASSERT_TRUE(accounts_after.empty()) << "accounts_ should be empty after cleanup";

    // 5. Verify storage has been cleaned up (via ListStorage check)
    // Note: DoCleanup() calls UnRegisterStorage, but storage_map may not be directly accessible
    // Check if storage has been removed via ListStorage
    auto [ec_storage_after, storage_configs_after] = registry_manager_->ListStorage(request_context_.get());
    ASSERT_EQ(EC_OK, ec_storage_after);
    // Expect storage list to be empty as UnRegisterStorage should have cleaned up all storages
    ASSERT_TRUE(storage_configs_after.empty()) << "data_storage_manager_ should have no storages after cleanup";
}

} // namespace kv_cache_manager
