#include <memory>
#include <set>
#include <string>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/migration_strategy.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "rapidjson/document.h"

namespace kv_cache_manager {

class StorageConfigQueryTest : public TESTBASE {
protected:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("local://", metrics_registry_);
        ASSERT_TRUE(registry_manager_->Init());
        request_context_ = std::make_shared<RequestContext>("storage-config-query");
        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);

        RegisterNfsStorage("hot");
        RegisterNfsStorage("migration_source");
        RegisterNfsStorage("migration_target");
        RegisterNfsStorage("other_group_only");
        RegisterDummyStorage("server_only");

        auto migration = std::make_shared<MigrationStrategy>();
        migration->set_source_storage_name("migration_source");
        migration->set_target_storage_name("migration_target");
        auto cache_config = std::make_shared<CacheConfig>();
        cache_config->set_migration_strategies({migration});

        InstanceGroup group_a;
        group_a.set_name("group_a");
        group_a.set_storage_candidates({"hot", "server_only"});
        group_a.set_cache_config(cache_config);
        ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(request_context_.get(), group_a));

        InstanceGroup group_b;
        group_b.set_name("group_b");
        group_b.set_storage_candidates({"other_group_only"});
        ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(request_context_.get(), group_b));

        InstanceGroup empty_group;
        empty_group.set_name("empty_group");
        empty_group.set_storage_candidates({"server_only"});
        ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(request_context_.get(), empty_group));
    }

    void RegisterNfsStorage(const std::string &name) {
        auto spec = std::make_shared<NfsStorageSpec>();
        spec->set_root_path(GetPrivateTestRuntimeDataPath() + name + "/");
        spec->set_key_count_per_file(1);
        StorageConfig config(DataStorageType::DATA_STORAGE_TYPE_NFS, name, spec);
        ASSERT_EQ(EC_OK,
                  registry_manager_->data_storage_manager()->RegisterStorage(request_context_.get(), name, config));
    }

    void RegisterDummyStorage(const std::string &name) {
        auto spec = std::make_shared<DummyStorageSpec>();
        spec->set_root_path(GetPrivateTestRuntimeDataPath() + name + "/");
        spec->set_key_count_per_file(1);
        StorageConfig config(DataStorageType::DATA_STORAGE_TYPE_DUMMY, name, spec);
        ASSERT_EQ(EC_OK,
                  registry_manager_->data_storage_manager()->RegisterStorage(request_context_.get(), name, config));
    }

    static std::set<std::string> ParseStorageNames(const std::string &json) {
        rapidjson::Document document;
        document.Parse(json.c_str());
        EXPECT_FALSE(document.HasParseError());
        EXPECT_TRUE(document.IsArray());
        std::set<std::string> names;
        if (!document.IsArray()) {
            return names;
        }
        for (const auto &config : document.GetArray()) {
            EXPECT_TRUE(config.HasMember("global_unique_name"));
            if (config.HasMember("global_unique_name") && config["global_unique_name"].IsString()) {
                names.insert(config["global_unique_name"].GetString());
            }
        }
        return names;
    }

    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<RequestContext> request_context_;
    std::shared_ptr<CacheManager> cache_manager_;
};

TEST_F(StorageConfigQueryTest, ReturnsOnlyCandidateAndMigrationStorages) {
    auto [ec, json] = cache_manager_->GetStorageConfigsByInstanceGroup(request_context_.get(), "group_a");

    EXPECT_EQ(EC_OK, ec);
    EXPECT_EQ((std::set<std::string>{"hot", "migration_source", "migration_target"}), ParseStorageNames(json));
}

TEST_F(StorageConfigQueryTest, MissingGroupIsNotConfusedWithEmptyStorageList) {
    auto [missing_ec, missing_json] =
        cache_manager_->GetStorageConfigsByInstanceGroup(request_context_.get(), "missing");
    EXPECT_EQ(EC_NOENT, missing_ec);
    EXPECT_TRUE(missing_json.empty());

    auto [empty_ec, empty_json] =
        cache_manager_->GetStorageConfigsByInstanceGroup(request_context_.get(), "empty_group");
    EXPECT_EQ(EC_OK, empty_ec);
    EXPECT_EQ("[]", empty_json);
}

} // namespace kv_cache_manager
