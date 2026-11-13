#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/dummy_backend.h"
#include "kv_cache_manager/data_storage/nfs_backend.h"

using namespace kv_cache_manager;

class RegistryManagerTest : public TESTBASE {
public:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(RegistryManagerTest, TestSimple) {
    // RegistryManager requires a valid storage backend; basic construction test only.
    RegistryManager registry_manager("fake_uri", nullptr);
}

namespace {
class CapacityDummyBackend : public DummyBackend {
public:
    using DummyBackend::DummyBackend;
    bool SupportsStorageCapacity() const override { return true; }
    StorageCapacitySnapshot GetStorageCapacity() const override { return snapshot; }
    StorageCapacitySnapshot snapshot;
};
}

class RegistryManagerQuotaTest : public TESTBASE {
public:
    void SetUp() override {
        metrics = std::make_shared<MetricsRegistry>();
        registry = std::make_shared<RegistryManager>("", metrics);
        ASSERT_TRUE(registry->Init());
        capacity_backend = std::make_shared<CapacityDummyBackend>(metrics);
        auto manager = registry->data_storage_manager();
        manager->storage_map_["supported"] = capacity_backend;
        manager->storage_map_["unsupported"] = std::make_shared<DummyBackend>(metrics);
        manager->storage_map_["nfs"] = std::make_shared<NfsBackend>(metrics);
    }

    InstanceGroup Group(bool automatic = true) {
        InstanceGroup group;
        group.set_name("quota_test");
        group.set_version(1);
        group.set_storage_candidates({"unsupported"});
        group.set_cache_config(std::make_shared<CacheConfig>());
        QuotaConfig quota(100, DataStorageType::DATA_STORAGE_TYPE_DUMMY);
        if (automatic) {
            quota.set_mode("auto");
        }
        group.set_quota(InstanceGroupQuota(-1, {quota}));
        return group;
    }

    std::shared_ptr<MetricsRegistry> metrics;
    std::shared_ptr<RegistryManager> registry;
    std::shared_ptr<CapacityDummyBackend> capacity_backend;
    RequestContext context{"quota_validation"};
};

TEST_F(RegistryManagerQuotaTest, RejectsUnsupportedAutoBeforePersistingButAllowsFixed) {
    auto group = Group();
    std::string error;
    EXPECT_FALSE(registry->ValidateAutomaticQuota(group, error));
    EXPECT_NE(std::string::npos, error.find("unsupported"));
    EXPECT_NE(std::string::npos, error.find("fixed"));
    EXPECT_EQ(EC_BADARGS, registry->CreateInstanceGroup(&context, group));
    EXPECT_EQ(EC_NOENT, registry->GetInstanceGroup(&context, group.name()).first);
    group = Group(false);
    EXPECT_EQ(EC_OK, registry->CreateInstanceGroup(&context, group));
    auto updated = Group();
    updated.set_version(2);
    EXPECT_EQ(EC_BADARGS, registry->UpdateInstanceGroup(&context, updated, 1));
    const auto stored = registry->GetInstanceGroup(&context, group.name()).second;
    ASSERT_NE(nullptr, stored);
    EXPECT_EQ(1, stored->version());
    EXPECT_FALSE(stored->quota().quota_config().front().automatic());
}

TEST_F(RegistryManagerQuotaTest, RejectsImplicitAutoAndSupportsMixedFixedOverrides) {
    auto group = Group();
    group.set_quota(InstanceGroupQuota(-1, {}));
    EXPECT_EQ(EC_BADARGS, registry->CreateInstanceGroup(&context, group));
    group.set_storage_candidates({"supported", "nfs"});
    group.set_quota(InstanceGroupQuota(-1, {QuotaConfig(100, DataStorageType::DATA_STORAGE_TYPE_NFS)}));
    EXPECT_EQ(EC_OK, registry->CreateInstanceGroup(&context, group));
}

TEST_F(RegistryManagerQuotaTest, CapacityFailuresDoNotChangeSupportOrBlockConfiguration) {
    auto group = Group();
    group.set_storage_candidates({"supported"});
    EXPECT_EQ(EC_OK, registry->CreateInstanceGroup(&context, group));
    EXPECT_FALSE(registry->ResolveQuota(group).types.begin()->second.usable);
    EXPECT_TRUE(registry->ResolveQuota(group).ForReclaim().quota_config().empty());
    capacity_backend->snapshot = {100, 1, 1, EC_ERROR};
    group.set_version(2);
    EXPECT_EQ(EC_OK, registry->UpdateInstanceGroup(&context, group, 1));
    EXPECT_FALSE(registry->ResolveQuota(group).types.begin()->second.usable);
}

TEST_F(RegistryManagerQuotaTest, ValidatesMigrationScopeButExcludesEventReportAndExplicitlyExcludedBackends) {
    auto group = Group();
    group.set_storage_candidates({"supported"});
    group.set_event_report_storage_candidates({"unsupported"});
    std::string error;
    EXPECT_TRUE(registry->ValidateAutomaticQuota(group, error));
    auto migration = std::make_shared<MigrationStrategy>();
    migration->set_source_storage_name("supported");
    migration->set_target_storage_name("unsupported");
    auto cache = std::make_shared<CacheConfig>();
    cache->set_migration_strategies({migration});
    group.set_cache_config(cache);
    EXPECT_FALSE(registry->ValidateAutomaticQuota(group, error));
    auto quota = group.quota().quota_config().front();
    quota.set_storage_names({"supported"});
    group.set_quota(InstanceGroupQuota(-1, {quota}));
    EXPECT_TRUE(registry->ValidateAutomaticQuota(group, error));
    group.set_storage_candidates({"not_registered_yet"});
    group.set_cache_config(std::make_shared<CacheConfig>());
    quota.set_storage_names({"not_registered_yet"});
    group.set_quota(InstanceGroupQuota(-1, {quota}));
    EXPECT_EQ(EC_OK, registry->CreateInstanceGroup(&context, group));
    EXPECT_EQ(-1, registry->ResolveQuota(group).reference_capacity);
}
