#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/account.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_local_backend.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

class FaultInjectRegistryLocalBackend : public RegistryLocalBackend {
public:
    explicit FaultInjectRegistryLocalBackend(std::string instance_id) : instance_id_(std::move(instance_id)) {}

    ErrorCode
    UpsertField(const std::string &key, const std::string &field, const std::string &value) noexcept override {
        if (fail_next_marker_upsert && key == "instance" && field == instance_id_ &&
            value == instance_id_ + ":retired") {
            fail_next_marker_upsert = false;
            return EC_ERROR;
        }
        if (fail_next_info_upsert && key == instance_id_ && field == instance_id_) {
            fail_next_info_upsert = false;
            return EC_ERROR;
        }
        return RegistryLocalBackend::UpsertField(key, field, value);
    }

    ErrorCode DeleteField(const std::string &key, const std::string &field) noexcept override {
        const ErrorCode ec = RegistryLocalBackend::DeleteField(key, field);
        if (lose_index_delete_reply && ec == EC_OK && key == "instance" && field == instance_id_) {
            lose_index_delete_reply = false;
            return EC_ERROR;
        }
        return ec;
    }

    bool fail_next_info_upsert = false;
    bool fail_next_marker_upsert = false;
    bool lose_index_delete_reply = false;

private:
    std::string instance_id_;
};

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

TEST_F(RegistryManagerLocalBackendTest, TestRemovingInstanceIntentSurvivesRecover) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "_removing_instance_registry";
    ASSERT_TRUE(InitRegistryManager("local://" + path + "?cluster_name=test"));
    CreateInstanceGroup("group1");
    RegisterInstance("group1", "__kvcm_internal_removing_instance_registry__");
    ASSERT_EQ(EC_OK,
              registry_manager_->RemoveInstance(
                  request_context_.get(), "group1", "__kvcm_internal_removing_instance_registry__"));
    RegisterInstance("group1", "instance1");
    std::map<std::string, std::string> fields;
    ASSERT_EQ(EC_OK, registry_manager_->storage_->Load("instance", fields));
    EXPECT_EQ("instance1", fields.at("instance1"));

    ASSERT_EQ(EC_OK, registry_manager_->BeginRemoveInstance(request_context_.get(), "instance1"));
    EXPECT_EQ(EC_NOENT, registry_manager_->storage_->Load("__kvcm_internal_removing_instance_registry__", fields));
    ASSERT_EQ(EC_OK, registry_manager_->storage_->Load("instance1", fields));
    InstanceInfo deleting;
    ASSERT_TRUE(deleting.FromJsonString(fields.at("instance1")));
    EXPECT_EQ(InstanceInfo::State::kDeleting, deleting.state());
    ASSERT_EQ(EC_OK, registry_manager_->storage_->Load("instance", fields));
    EXPECT_EQ("instance1", fields.at("instance1"));
    EXPECT_EQ(nullptr, registry_manager_->GetInstanceInfo(request_context_.get(), "instance1"));
    ASSERT_NE(nullptr, registry_manager_->GetRemovingInstanceInfo("instance1"));
    EXPECT_NE(EC_OK, registry_manager_->RemoveInstanceGroup(request_context_.get(), "group1"));
    auto [group_ec, group] = registry_manager_->GetInstanceGroup(request_context_.get(), "group1");
    ASSERT_EQ(EC_OK, group_ec);
    auto updated_group = *group;
    updated_group.set_version(group->version() + 1);
    EXPECT_NE(EC_OK, registry_manager_->UpdateInstanceGroup(request_context_.get(), updated_group, group->version()));
    EXPECT_NE(EC_OK,
              registry_manager_->RegisterInstance(
                  request_context_.get(), "group1", "instance1", 1024, {LocationSpecInfo{}}, ModelDeployment{}));

    ASSERT_EQ(EC_OK, registry_manager_->DoCleanup());
    ASSERT_EQ(EC_OK, registry_manager_->DoRecoverOnce());
    ASSERT_NE(nullptr, registry_manager_->GetRemovingInstanceInfo("instance1"));
    EXPECT_EQ(nullptr, registry_manager_->GetInstanceInfo(request_context_.get(), "instance1"));
    ASSERT_EQ(EC_OK, registry_manager_->BeginRemoveInstance(request_context_.get(), "instance1"));
    ASSERT_EQ(EC_OK, registry_manager_->CompleteRemoveInstance("instance1"));
    EXPECT_EQ(nullptr, registry_manager_->GetRemovingInstanceInfo("instance1"));
    EXPECT_EQ(EC_NOENT, registry_manager_->storage_->Load("instance1", fields));
    EXPECT_EQ(EC_NOENT, registry_manager_->storage_->Load("instance", fields));
    RegisterInstance("group1", "instance1");
}

TEST_F(RegistryManagerLocalBackendTest, TestRetiredInstanceRecordRecovery) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "_retired_instance_registry";
    ASSERT_TRUE(InitRegistryManager("local://" + path + "?cluster_name=test"));
    CreateInstanceGroup("group1");
    RegisterInstance("group1", "retired_with_info");
    RegisterInstance("group1", "retired_without_info");
    RegisterInstance("group1", "serving_instance");

    ASSERT_EQ(EC_OK, registry_manager_->BeginRemoveInstance(request_context_.get(), "retired_with_info"));
    ASSERT_EQ(EC_OK, registry_manager_->BeginRemoveInstance(request_context_.get(), "retired_without_info"));
    ASSERT_EQ(EC_OK,
              registry_manager_->storage_->UpsertField("instance", "retired_with_info", "retired_with_info:retired"));
    ASSERT_EQ(EC_OK,
              registry_manager_->storage_->UpsertField(
                  "instance", "retired_without_info", "retired_without_info:retired"));
    ASSERT_EQ(EC_OK, registry_manager_->storage_->DeleteField("retired_without_info", "retired_without_info"));

    ASSERT_EQ(EC_OK, registry_manager_->DoCleanup());
    ASSERT_EQ(EC_OK, registry_manager_->DoRecoverOnce());
    std::map<std::string, std::string> fields;
    EXPECT_EQ(EC_NOENT, registry_manager_->storage_->Load("retired_with_info", fields));
    EXPECT_EQ(EC_NOENT, registry_manager_->storage_->Load("retired_without_info", fields));
    ASSERT_EQ(EC_OK, registry_manager_->storage_->Load("instance", fields));
    EXPECT_EQ(0, fields.count("retired_with_info"));
    EXPECT_EQ(0, fields.count("retired_without_info"));
    EXPECT_EQ(1, fields.count("serving_instance"));
    EXPECT_EQ(nullptr, registry_manager_->GetRemovingInstanceInfo("retired_with_info"));
    EXPECT_EQ(nullptr, registry_manager_->GetRemovingInstanceInfo("retired_without_info"));
}

TEST_F(RegistryManagerLocalBackendTest, TestAmbiguousFinalIndexDeleteDoesNotRecreateInstanceInfo) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "_ambiguous_final_index_delete";
    const std::string uri = "local://" + path + "?cluster_name=test";
    const std::string instance_id = "instance1";
    ASSERT_TRUE(InitRegistryManager(uri));
    CreateInstanceGroup("group1");
    RegisterInstance("group1", instance_id);

    auto backend = std::make_unique<FaultInjectRegistryLocalBackend>(instance_id);
    ASSERT_EQ(EC_OK, backend->Init(StandardUri::FromUri(uri)));
    auto *fault_backend = backend.get();
    backend->fail_next_info_upsert = true;
    backend->lose_index_delete_reply = true;
    registry_manager_->storage_ = std::move(backend);

    EXPECT_EQ(EC_ERROR, registry_manager_->BeginRemoveInstance(request_context_.get(), instance_id));
    EXPECT_EQ(nullptr, registry_manager_->GetInstanceInfo(request_context_.get(), instance_id));
    ASSERT_EQ(EC_OK, registry_manager_->BeginRemoveInstance(request_context_.get(), instance_id));
    std::map<std::string, std::string> fields;
    ASSERT_EQ(EC_OK, registry_manager_->storage_->Load(instance_id, fields));
    InstanceInfo deleting;
    ASSERT_TRUE(deleting.FromJsonString(fields.at(instance_id)));
    EXPECT_EQ(InstanceInfo::State::kDeleting, deleting.state());
    fault_backend->fail_next_info_upsert = true;
    EXPECT_EQ(EC_ERROR, registry_manager_->BeginRemoveInstance(request_context_.get(), instance_id));
    EXPECT_EQ(EC_OK, registry_manager_->BeginRemoveInstance(request_context_.get(), instance_id));
    EXPECT_EQ(EC_ERROR, registry_manager_->CompleteRemoveInstance(instance_id));

    EXPECT_EQ(EC_NOENT, registry_manager_->storage_->Load(instance_id, fields));
    EXPECT_EQ(EC_NOENT, registry_manager_->storage_->Load("instance", fields));
    ASSERT_EQ(EC_OK, registry_manager_->BeginRemoveInstance(request_context_.get(), instance_id));
    EXPECT_EQ(EC_NOENT, registry_manager_->storage_->Load(instance_id, fields));

    ASSERT_TRUE(InitRegistryManager(uri)); // Simulate a restart before the next CompleteRemoveInstance.
    ASSERT_EQ(EC_OK, registry_manager_->DoRecoverOnce());
    EXPECT_EQ(nullptr, registry_manager_->GetRemovingInstanceInfo(instance_id));
    EXPECT_EQ(EC_NOENT, registry_manager_->storage_->Load(instance_id, fields));
    RegisterInstance("group1", instance_id);
}

TEST_F(RegistryManagerLocalBackendTest, TestRetiredRemovalPhaseSurvivesRegistryRecoveryRetry) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "_retired_removal_phase";
    const std::string uri = "local://" + path + "?cluster_name=test";
    const std::string instance_id = "instance1";
    ASSERT_TRUE(InitRegistryManager(uri));
    CreateInstanceGroup("group1");
    RegisterInstance("group1", instance_id);
    ASSERT_EQ(EC_OK, registry_manager_->BeginRemoveInstance(request_context_.get(), instance_id));

    auto backend = std::make_unique<FaultInjectRegistryLocalBackend>(instance_id);
    ASSERT_EQ(EC_OK, backend->Init(StandardUri::FromUri(uri)));
    backend->fail_next_marker_upsert = true;
    registry_manager_->storage_ = std::move(backend);
    EXPECT_EQ(EC_ERROR, registry_manager_->CompleteRemoveInstance(instance_id));
    EXPECT_TRUE(registry_manager_->IsInstanceRemovalRetired(instance_id));

    ASSERT_EQ(EC_OK, registry_manager_->DoRecoverOnce());
    EXPECT_TRUE(registry_manager_->IsInstanceRemovalRetired(instance_id));
    EXPECT_EQ(EC_OK, registry_manager_->CompleteRemoveInstance(instance_id));
    EXPECT_EQ(nullptr, registry_manager_->GetRemovingInstanceInfo(instance_id));
}

TEST_F(RegistryManagerLocalBackendTest, TestMalformedDeletingRecordDoesNotHideServingInstances) {
    const std::string path = GetPrivateTestRuntimeDataPath() + "_malformed_deleting_registry";
    ASSERT_TRUE(InitRegistryManager("local://" + path + "?cluster_name=test"));
    CreateInstanceGroup("group1");
    RegisterInstance("group1", "healthy_instance");
    ASSERT_EQ(EC_OK, registry_manager_->storage_->UpsertField("instance", "broken_instance", "broken_instance"));
    ASSERT_EQ(EC_OK,
              registry_manager_->storage_->UpsertField("broken_instance", "broken_instance", "invalid_json"));

    ASSERT_EQ(EC_OK, registry_manager_->DoCleanup());
    EXPECT_EQ(EC_ERROR, registry_manager_->DoRecoverOnce());
    EXPECT_NE(nullptr, registry_manager_->GetInstanceInfo(request_context_.get(), "healthy_instance"));
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

    // TairMempool type is persisted in CacheLocation and usage accounting; reject conversion in place.
    StorageConfig changed_type_config(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD, "storage1", nullptr);
    ASSERT_EQ(EC_BADARGS, registry_manager_->UpdateStorage(request_context_.get(), changed_type_config, true));
    updated_storage1 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage1");
    ASSERT_TRUE(updated_storage1);
    ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, updated_storage1->GetStorageConfig().type());

    // Test RemoveStorage
    auto ec_remove = registry_manager_->RemoveStorage(request_context_.get(), "storage2");
    ASSERT_EQ(EC_OK, ec_remove);

    storage2 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage2");
    ASSERT_FALSE(storage2);
}

TEST_F(RegistryManagerLocalBackendTest, TestTairMempoolMediaTypeIsImmutable) {
    const std::string local_path = GetPrivateTestRuntimeDataPath() + "_registry_local_backend_tair_media_test";
    ASSERT_TRUE(InitRegistryManager("local://" + local_path + "?cluster_name=test"));

    auto verify_media_type_change_rejected = [this](const std::string &storage_name,
                                                    uint16_t current_media_type,
                                                    uint16_t requested_media_type) {
        AddNfsStorage(storage_name, 1);
        const auto backend = registry_manager_->data_storage_manager()->GetDataStorageBackend(storage_name);
        ASSERT_TRUE(backend);

        auto current_spec = std::make_shared<TairMemPoolStorageSpec>();
        current_spec->set_domain("pace.meta");
        current_spec->set_timeout(5000);
        current_spec->set_media_type(current_media_type);
        // This open-source test uses an NFS backend as a configured-backend holder because the
        // open-source TairMempool backend is intentionally non-functional.
        backend->config_ =
            StorageConfig(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, storage_name, current_spec);

        auto requested_spec = std::make_shared<TairMemPoolStorageSpec>(*current_spec);
        requested_spec->set_media_type(requested_media_type);
        StorageConfig requested_config(
            DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, storage_name, requested_spec);
        ASSERT_EQ(EC_BADARGS, registry_manager_->UpdateStorage(request_context_.get(), requested_config, true));

        const auto unchanged_backend =
            registry_manager_->data_storage_manager()->GetDataStorageBackend(storage_name);
        ASSERT_EQ(backend, unchanged_backend);
        const auto unchanged_spec =
            std::dynamic_pointer_cast<TairMemPoolStorageSpec>(unchanged_backend->GetStorageConfig().storage_spec());
        ASSERT_TRUE(unchanged_spec);
        ASSERT_EQ(current_media_type, unchanged_spec->media_type());
    };

    verify_media_type_change_rejected(
        "pace_dram", kTairMemPoolMediaTypeDram, kTairMemPoolMediaTypeSsd);
    verify_media_type_change_rejected(
        "pace_legacy_ssd", kTairMemPoolMediaTypeSsd, kTairMemPoolMediaTypeDram);
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

TEST_F(RegistryManagerLocalBackendTest, DuplicateInstanceCannotMoveToAnotherGroup) {
    std::string local_path = GetPrivateTestRuntimeDataPath() + "_registry_local_backend_group_mismatch";
    ASSERT_TRUE(InitRegistryManager("local://" + local_path + "?cluster_name=test"));
    AddNfsStorage("storage1", 1);
    CreateInstanceGroup("group1");
    CreateInstanceGroup("group2");
    RegisterInstance("group1", "shared_instance");

    LocationSpecInfo info;
    ModelDeployment model_deployment;
    EXPECT_EQ(EC_DUPLICATE_ENTITY,
              registry_manager_->RegisterInstance(
                  request_context_.get(), "group2", "shared_instance", 1024, {info}, model_deployment));
    const auto existing = registry_manager_->GetInstanceInfo(request_context_.get(), "shared_instance");
    ASSERT_NE(nullptr, existing);
    EXPECT_EQ("group1", existing->instance_group_name());
    EXPECT_NE(std::string::npos, request_context_->error_tracer()->ToJsonString().find("instance_group_name"));
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

TEST_F(RegistryManagerLocalBackendTest, TestDoRecoverOnceIdempotent) {
    std::string local_path = GetPrivateTestRuntimeDataPath() + "_registry_local_backend_recover_idempotent_test";
    std::string uri = "local://" + local_path + "?cluster_name=test";

    // Setup initial data
    {
        ASSERT_TRUE(InitRegistryManager(uri));
        AddNfsStorage("storage1", 1);
        AddNfsStorage("storage2", 2);
        ASSERT_EQ(EC_OK, registry_manager_->DisableStorage(request_context_.get(), "storage2"));
        CreateInstanceGroup("group1");
        RegisterInstance("group1", "instance1");
        ASSERT_EQ(EC_OK,
                  registry_manager_->AddAccount(request_context_.get(), "user1", "pwd1", AccountRole::ROLE_USER));
    }

    // Re-init and call DoRecoverOnce multiple times - should be idempotent
    {
        ASSERT_TRUE(InitRegistryManager(uri));
        ASSERT_EQ(EC_OK, registry_manager_->DoRecoverOnce());

        // Second call should also succeed (skip already-recovered items)
        ASSERT_EQ(EC_OK, registry_manager_->DoRecoverOnce());

        // Third call
        ASSERT_EQ(EC_OK, registry_manager_->DoRecoverOnce());

        // Verify data is correct (not duplicated)
        auto storage1 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage1");
        ASSERT_TRUE(storage1);
        ASSERT_TRUE(storage1->Available());

        auto storage2 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage2");
        ASSERT_TRUE(storage2);
        ASSERT_FALSE(storage2->Available());

        auto [ec_groups, groups] = registry_manager_->ListInstanceGroup(request_context_.get());
        ASSERT_EQ(EC_OK, ec_groups);
        ASSERT_EQ(1, groups.size());

        auto instance_info = registry_manager_->GetInstanceInfo(request_context_.get(), "instance1");
        ASSERT_TRUE(instance_info);
        ASSERT_EQ("instance1", instance_info->instance_id());

        auto [ec_accounts, accounts] = registry_manager_->ListAccount(request_context_.get());
        ASSERT_EQ(EC_OK, ec_accounts);
        ASSERT_EQ(1, accounts.size());
    }
}

TEST_F(RegistryManagerLocalBackendTest, TestDoRecoverOncePartialFailure) {
    std::string local_path = GetPrivateTestRuntimeDataPath() + "_recover_partial_fail_test";
    std::string uri = "local://" + local_path + "?cluster_name=test";

    // Setup valid data first
    ASSERT_TRUE(InitRegistryManager(uri));
    AddNfsStorage("storage1", 1);
    CreateInstanceGroup("group1");
    RegisterInstance("group1", "instance1");
    ASSERT_EQ(EC_OK, registry_manager_->AddAccount(request_context_.get(), "user1", "pwd1", AccountRole::ROLE_USER));

    // Inject invalid entries directly via storage_ backend
    std::map<std::string, std::string> storage_map;
    registry_manager_->storage_->Load("storage", storage_map);
    storage_map["bad_storage"] = "this_is_not_valid_json";
    registry_manager_->storage_->Save("storage", storage_map);

    std::map<std::string, std::string> account_map;
    registry_manager_->storage_->Load("account", account_map);
    account_map["bad_account"] = "invalid_json";
    registry_manager_->storage_->Save("account", account_map);

    // Re-init from persistent storage containing both valid and invalid entries
    ASSERT_TRUE(InitRegistryManager(uri));
    auto ec = registry_manager_->DoRecoverOnce();
    ASSERT_EQ(EC_ERROR, ec) << "Should report error due to invalid entries";

    // Valid data should still be recovered
    auto storage1 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage1");
    ASSERT_TRUE(storage1) << "Valid storage1 should be recovered";
    ASSERT_TRUE(storage1->Available());

    auto bad_storage = registry_manager_->data_storage_manager()->GetDataStorageBackend("bad_storage");
    ASSERT_FALSE(bad_storage) << "Invalid bad_storage should not be recovered";

    auto [ec_groups, groups] = registry_manager_->ListInstanceGroup(request_context_.get());
    ASSERT_EQ(EC_OK, ec_groups);
    ASSERT_EQ(1, groups.size());

    auto instance_info = registry_manager_->GetInstanceInfo(request_context_.get(), "instance1");
    ASSERT_TRUE(instance_info);

    auto [ec_accounts, accounts] = registry_manager_->ListAccount(request_context_.get());
    ASSERT_EQ(EC_OK, ec_accounts);
    ASSERT_EQ(1, accounts.size()) << "Valid user1 recovered, bad_account skipped";

    // Call again - idempotent (valid items skipped, bad items still fail)
    ec = registry_manager_->DoRecoverOnce();
    ASSERT_EQ(EC_ERROR, ec);
    storage1 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage1");
    ASSERT_TRUE(storage1);
}

TEST_F(RegistryManagerLocalBackendTest, TestDoRecoverFixedAfterRetry) {
    std::string local_path = GetPrivateTestRuntimeDataPath() + "_recover_fix_after_retry_test";
    std::string uri = "local://" + local_path + "?cluster_name=test";

    // Setup valid data
    ASSERT_TRUE(InitRegistryManager(uri));
    AddNfsStorage("storage1", 1);

    // Inject invalid entry
    std::map<std::string, std::string> storage_map;
    registry_manager_->storage_->Load("storage", storage_map);
    storage_map["bad_storage"] = "invalid";
    registry_manager_->storage_->Save("storage", storage_map);

    // Cleanup and recover - partial failure
    ASSERT_EQ(EC_OK, registry_manager_->DoCleanup());
    auto ec = registry_manager_->DoRecoverOnce();
    ASSERT_EQ(EC_ERROR, ec);

    // Fix the bad entry by removing it from backend
    registry_manager_->storage_->Load("storage", storage_map);
    storage_map.erase("bad_storage");
    registry_manager_->storage_->Save("storage", storage_map);

    // Retry should now succeed
    ec = registry_manager_->DoRecoverOnce();
    ASSERT_EQ(EC_OK, ec);
}

TEST_F(RegistryManagerLocalBackendTest, TestDoRecoverStartsRetryOnFailure) {
    std::string local_path = GetPrivateTestRuntimeDataPath() + "_recover_starts_retry_test";
    std::string uri = "local://" + local_path + "?cluster_name=test";

    // Setup valid data + inject bad entry
    ASSERT_TRUE(InitRegistryManager(uri));
    AddNfsStorage("storage1", 1);

    std::map<std::string, std::string> storage_map;
    registry_manager_->storage_->Load("storage", storage_map);
    storage_map["bad_storage"] = "invalid";
    registry_manager_->storage_->Save("storage", storage_map);

    // Cleanup memory, then DoRecover - should return EC_OK but start retry loop
    ASSERT_EQ(EC_OK, registry_manager_->DoCleanup());
    auto ec = registry_manager_->DoRecover();
    ASSERT_EQ(EC_OK, ec) << "DoRecover tolerates errors and returns OK";

    // Valid storage should still be recovered
    auto storage1 = registry_manager_->data_storage_manager()->GetDataStorageBackend("storage1");
    ASSERT_TRUE(storage1);

    // Stop the retry thread
    registry_manager_->StopRecoverRetryLoop();
}

TEST_F(RegistryManagerLocalBackendTest, TestRecoverRetryLoopLifecycle) {
    std::string local_path = GetPrivateTestRuntimeDataPath() + "_recover_retry_lifecycle_test";
    std::string uri = "local://" + local_path + "?cluster_name=test";

    // StopRecoverRetryLoop should be safe to call even without active thread
    {
        ASSERT_TRUE(InitRegistryManager(uri));
        registry_manager_->StopRecoverRetryLoop();
        registry_manager_->StopRecoverRetryLoop(); // double-stop should not crash
    }

    // StartRecoverRetryLoop + StopRecoverRetryLoop lifecycle
    {
        ASSERT_TRUE(InitRegistryManager(uri));
        registry_manager_->StartRecoverRetryLoop();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        registry_manager_->StopRecoverRetryLoop(); // should join within ~100ms
    }

    // Destructor should properly stop retry thread via DoCleanup
    {
        auto rm = std::make_shared<RegistryManager>(uri, std::make_shared<MetricsRegistry>());
        ASSERT_TRUE(rm->Init());
        rm->StartRecoverRetryLoop();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        rm.reset(); // destructor -> DoCleanup -> StopRecoverRetryLoop
    }
}

} // namespace kv_cache_manager
