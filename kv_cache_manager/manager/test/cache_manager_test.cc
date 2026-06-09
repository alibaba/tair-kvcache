#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_backend.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/manager/cache_location_view.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
#include "kv_cache_manager/manager/meta_searcher_manager.h"
#include "kv_cache_manager/manager/reclaimer_task_supervisor.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/manager/startup_config_loader.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace {
static const std::string default_storage_configs(
    "[{\"type\":\"file\",\"is_available\":true,\"global_unique_name\":\"nfs_01\",\"storage_spec\":{"
    "\"root_path\":\"/tmp/nfs/\",\"key_count_per_file\":8}}]");
} // namespace

namespace kv_cache_manager {

class MockDataStorageBackend : public DataStorageBackend {
public:
    explicit MockDataStorageBackend(std::shared_ptr<MetricsRegistry> mr) : DataStorageBackend(std::move(mr)) {}
    MOCK_METHOD(DataStorageType, GetType, (), (override));
    MOCK_METHOD(bool, Available, (), (override));
    MOCK_METHOD(double, GetStorageUsageRatio, (const std::string &), (const, override));
    MOCK_METHOD(ErrorCode, DoOpen, (const StorageConfig &, const std::string &), (override));
    MOCK_METHOD(ErrorCode, Close, (), (override));
    MOCK_METHOD((std::vector<std::pair<ErrorCode, DataStorageUri>>),
                Create,
                (const std::vector<std::string> &, size_t, const std::string &, std::function<void()>),
                (override));
    MOCK_METHOD(std::vector<ErrorCode>,
                Delete,
                (const std::vector<DataStorageUri> &, const std::string &, std::function<void()>),
                (override));
    MOCK_METHOD(std::vector<bool>, Exist, (const std::vector<DataStorageUri> &), (override));
    MOCK_METHOD(std::vector<bool>, MightExist, (const std::vector<DataStorageUri> &), (override));
    MOCK_METHOD(std::vector<ErrorCode>, Lock, (const std::vector<DataStorageUri> &), (override));
    MOCK_METHOD(std::vector<ErrorCode>, UnLock, (const std::vector<DataStorageUri> &), (override));
};

// wraps a real DataStorageBackend but intercepts MightExist() with a
// user-provided function; all other calls are delegated to the real
// backend
class MightExistInterceptor : public DataStorageBackend {
public:
    using MightExistFunc = std::function<std::vector<bool>(const std::vector<DataStorageUri> &)>;

    MightExistInterceptor(std::shared_ptr<DataStorageBackend> delegate, MightExistFunc fn)
        : DataStorageBackend(delegate->metrics_registry_), delegate_(std::move(delegate)), fn_(std::move(fn)) {
        SetOpen(delegate_->IsOpen());
        SetAvailable(true);
    }

    DataStorageType GetType() override { return delegate_->GetType(); }
    bool Available() override { return delegate_->Available(); }
    double GetStorageUsageRatio(const std::string &t) const override { return delegate_->GetStorageUsageRatio(t); }
    const StorageConfig &GetStorageConfig() override { return delegate_->GetStorageConfig(); }
    ErrorCode DoOpen(const StorageConfig &c, const std::string &t) override { return delegate_->DoOpen(c, t); }
    ErrorCode Close() override { return delegate_->Close(); }
    std::vector<std::pair<ErrorCode, DataStorageUri>>
    Create(const std::vector<std::string> &k, size_t s, const std::string &t, std::function<void()> cb) override {
        return delegate_->Create(k, s, t, std::move(cb));
    }
    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &u, const std::string &t, std::function<void()> cb) override {
        return delegate_->Delete(u, t, std::move(cb));
    }
    std::vector<bool> Exist(const std::vector<DataStorageUri> &u) override { return delegate_->Exist(u); }
    std::vector<bool> MightExist(const std::vector<DataStorageUri> &u) override { return fn_(u); }
    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &u) override { return delegate_->Lock(u); }
    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &u) override { return delegate_->UnLock(u); }

private:
    std::shared_ptr<DataStorageBackend> delegate_;
    MightExistFunc fn_;
};

class CacheManagerTest : public TESTBASE {
public:
    void SetUp() override {
        cache_manager_ = createCacheManager();
        request_context_.reset(new RequestContext("fake_trace_id"));
    }

    void TearDown() override {}

    std::unique_ptr<CacheManager> createCacheManager() {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        std::shared_ptr<MetricsRegistry> &metrics_registry = metrics_registry_;
        registry_manager_ = std::make_shared<RegistryManager>("", metrics_registry);
        std::shared_ptr<InstanceGroup> instance_group = std::make_shared<InstanceGroup>();
        auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
        instance_group->cache_config_ = std::make_shared<CacheConfig>();
        instance_group->cache_config_->meta_indexer_config_ = meta_indexer_config;
        instance_group->cache_config_->cache_prefer_strategy_ = CachePreferStrategy::CPS_PREFER_3FS;
        auto backend_config = std::make_shared<MetaStorageBackendConfig>();
        backend_config->storage_type_ = META_LOCAL_BACKEND_TYPE_STR;
        auto cache_policy_config = std::make_shared<MetaCachePolicyConfig>();
        meta_indexer_config->meta_storage_backend_config_ = backend_config;
        meta_indexer_config->meta_cache_policy_config_ = cache_policy_config;

        std::shared_ptr<InstanceInfo> instance_info = std::make_shared<InstanceInfo>(
            "test_quota_group", "default", "test_instance", 64, createLocationSpecInfos(), createModelDeployment());
        registry_manager_->instance_group_configs_["test_group"] = instance_group;
        registry_manager_->instance_infos_["test_instance"] = instance_info;
        registry_manager_->Init();
        // Mark registry as recovered since data was injected directly (not via backend)
        registry_manager_->recover_complete_.store(true);
        std::unique_ptr<CacheManager> cache_manager =
            std::make_unique<CacheManager>(metrics_registry, registry_manager_);

        EXPECT_TRUE(cache_manager->Init());

        // load first because we need default group
        // in real usage, we load startup config after recover
        StartupConfigLoader loader;
        loader.Init(registry_manager_);
        loader.Load("");

        EXPECT_EQ(EC_OK, cache_manager->DoRecover());

        return cache_manager;
    }

    ModelDeployment createModelDeployment() {
        ModelDeployment model_deployment;
        model_deployment.set_model_name("fake model");
        model_deployment.set_use_mla(false);
        model_deployment.set_tp_size(4);
        model_deployment.set_dp_size(0);
        model_deployment.set_pp_size(1);
        model_deployment.set_extra("");
        model_deployment.set_user_data("");
        return model_deployment;
    }

    std::vector<LocationSpecInfo> createLocationSpecInfos() {
        std::vector<LocationSpecInfo> location_spec_infos = {
            LocationSpecInfo("tp0", 512),
            LocationSpecInfo("tp1", 512),
            LocationSpecInfo("tp2", 512),
            LocationSpecInfo("tp3", 512),
        };
        return location_spec_infos;
    }

    void expectEmptySpec(const CacheLocationView::LocationSpecViewVec &specs) {
        for (auto &spec : specs) {
            EXPECT_EQ("", spec.uri());
        }
    }
    void expectNonEmptySpec(const CacheLocationView::LocationSpecViewVec &specs) {
        for (auto &spec : specs) {
            EXPECT_NE("", spec.uri());
        }
    }

    std::unique_ptr<CacheManager> cache_manager_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<RequestContext> request_context_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

TEST_F(CacheManagerTest, TestRegisterInstance) {
    // register same instance in each round
    {
        size_t round = 20;
        for (int i = 0; i < round; ++i) {
            const int num_threads = 2;
            auto instance_id = std::to_string(rand() % 1000);
            ;
            std::vector<std::thread> threads;
            int32_t block_size = 64;
            for (int j = 0; j < num_threads; ++j) {
                threads.emplace_back([this, &instance_id, block_size]() {
                    auto request_context =
                        std::make_unique<RequestContext>("fake_trace_" + std::to_string(rand() % 1000));
                    auto ret = cache_manager_->RegisterInstance(request_context.get(),
                                                                "default",
                                                                instance_id,
                                                                block_size,
                                                                createLocationSpecInfos(),
                                                                createModelDeployment(),
                                                                std::vector<LocationSpecGroup>());
                    EXPECT_EQ(EC_OK, ret.first);
                });
            }
            for (auto &t : threads) {
                t.join();
            }
        }
    }
    // register diff instance in each round
    {
        size_t round = 20;
        size_t success_count = 0;
        size_t error_count = 0;
        for (int i = 0; i < round; ++i) {
            const int num_threads = 2;
            auto instance_id = std::to_string(rand() % 1000);
            ;
            std::vector<std::thread> threads;
            int32_t block_size = 64;
            for (int j = 0; j < num_threads; ++j) {
                block_size += j;
                threads.emplace_back([&, block_size]() {
                    auto request_context =
                        std::make_unique<RequestContext>("fake_trace_" + std::to_string(rand() % 1000));
                    auto ret = cache_manager_->RegisterInstance(request_context.get(),
                                                                "default",
                                                                instance_id,
                                                                block_size,
                                                                createLocationSpecInfos(),
                                                                createModelDeployment(),
                                                                std::vector<LocationSpecGroup>());
                    if (ret.first == EC_OK) {
                        ++success_count;
                    } else {
                        ++error_count;
                    }
                });
            }
            for (auto &t : threads) {
                t.join();
            }
            std::cout << error_count << std::endl;
            EXPECT_EQ(true, success_count == error_count);
        }
    }
}

TEST_F(CacheManagerTest, TestRemoveInstance) {
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "placeholder_id",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    {
        auto [ec, ptr] = cache_manager_->GetInstanceInfo(request_context_.get(), "placeholder_id");
        ASSERT_EQ(ErrorCode::EC_OK, ec);
        ASSERT_NE(nullptr, ptr);
    }

    std::vector<std::int64_t> keys;
    for (std::int64_t i = 0; i < 65; ++i) {
        keys.push_back(i);
    }

    auto [ec0, _info] =
        cache_manager_->StartWriteCache(request_context_.get(), "placeholder_id", keys, {}, {}, 100000000);

    auto ec1 = cache_manager_->RemoveInstance(request_context_.get(), "default", "placeholder_id");
    ASSERT_EQ(ErrorCode::EC_OK, ec1);
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    BlockMask block_mask = static_cast<std::size_t>(0);

    {
        auto [ec, ptr] = cache_manager_->GetInstanceInfo(request_context_.get(), "placeholder_id");
        ASSERT_NE(ErrorCode::EC_OK, ec);
        ASSERT_EQ(nullptr, ptr);
    }

    auto [ec2, cache_metas] =
        cache_manager_->GetCacheMeta(request_context_.get(), "placeholder_id", keys, {}, block_mask, 0);
    const auto &cache_locations_view = cache_metas.cache_locations_view();
    const auto &metas = cache_metas.metas();
    ASSERT_EQ(65, cache_locations_view.size());
    ASSERT_EQ(65, metas.size());
    for (int i = 0; i < 65; ++i) {
        std::map<std::string, std::string> meta;
        ASSERT_TRUE(Jsonizable::FromJsonString(metas[i], meta));
        ASSERT_EQ(CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_NOT_FOUND), meta.at("status"));
    }
}

TEST_F(CacheManagerTest, TestRecover) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_TRUE(meta_searcher);
    ASSERT_TRUE(meta_searcher->meta_indexer_);
    ASSERT_EQ("test_instance", meta_searcher->meta_indexer_->instance_id_);
    auto meta_indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer("test_instance");
    ASSERT_TRUE(meta_indexer);
    ASSERT_EQ("test_instance", meta_indexer->instance_id_);
}

TEST_F(CacheManagerTest, TestCleanup) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(EC_OK, cache_manager_->DoCleanup());
    ASSERT_EQ(EC_OK, cache_manager_->DoRecover());
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_TRUE(meta_searcher);
    ASSERT_TRUE(meta_searcher->meta_indexer_);
    ASSERT_EQ("test_instance", meta_searcher->meta_indexer_->instance_id_);
    auto meta_indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer("test_instance");
    ASSERT_TRUE(meta_indexer);
    ASSERT_EQ("test_instance", meta_indexer->instance_id_);
}

TEST_F(CacheManagerTest, TestStartWriteCache) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));
    std::vector<int64_t> keys{1, 2, 3};
    auto [ec, start_write_cache_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 1000);
    ASSERT_EQ(EC_OK, ec);
    const auto &cache_locations_view = start_write_cache_info.locations().cache_locations_view();
    ASSERT_EQ(0, std::get<BlockMaskOffset>(start_write_cache_info.block_mask()));
    ASSERT_EQ(3, cache_locations_view.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const auto &cache_location = cache_locations_view[i];
        ASSERT_EQ(kDefaultStorageType, cache_location.type());
        ASSERT_EQ(4, cache_location.spec_size());
        const auto &location_specs = cache_location.location_specs();
        ASSERT_EQ(4, location_specs.size());
        for (int j = 0; j < 4; ++j) {
            ASSERT_EQ(std::string("tp") + std::to_string(j), location_specs[j].name());
            // std::string expected = std::string("3fs://") + std::to_string(i) + "/" + std::to_string(j);
            // ASSERT_EQ(expected, location_specs[j].location());
        }
    }
}

TEST_F(CacheManagerTest, TestStartWriteDuplicateCache) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));
    {
        std::vector<int64_t> keys{1, 2};
        auto [ec, start_write_cache_info] =
            cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = start_write_cache_info.locations().cache_locations_view();
        ASSERT_EQ(0, std::get<BlockMaskOffset>(start_write_cache_info.block_mask()));
        ASSERT_EQ(2, cache_locations_view.size());
    }
    {
        std::vector<int64_t> keys{1, 2, 3, 4};
        auto [ec, start_write_cache_info] =
            cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = start_write_cache_info.locations().cache_locations_view();
        ASSERT_EQ(2, std::get<BlockMaskOffset>(start_write_cache_info.block_mask()));
        ASSERT_EQ(2, cache_locations_view.size());
    }
    {
        std::vector<int64_t> keys{1, 11, 12, 2};
        auto [ec, start_write_cache_info] =
            cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = start_write_cache_info.locations().cache_locations_view();
        ASSERT_EQ(BlockMaskVector({true, false, false, true}),
                  std::get<BlockMaskVector>(start_write_cache_info.block_mask()));
        ASSERT_EQ(2, cache_locations_view.size());
    }
}

TEST_F(CacheManagerTest, TestStartWriteCacheWithLocationSpecGroup) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    std::vector<LocationSpecInfo> location_spec_infos = {
        LocationSpecInfo("tp0_F0", 512),
        LocationSpecInfo("tp1_F0", 512),
        LocationSpecInfo("tp0_L1", 512),
        LocationSpecInfo("tp1_L1", 512),
    };
    std::vector<LocationSpecGroup> location_spec_groups = {
        LocationSpecGroup("F0L1", {"tp0_F0", "tp1_F0", "tp0_L1", "tp1_L1"}),
        LocationSpecGroup("F0", {"tp0_F0", "tp1_F0"}),
    };

    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance2",
                                               64,
                                               location_spec_infos,
                                               createModelDeployment(),
                                               location_spec_groups));
    // have been sorted
    ASSERT_EQ(
        std::string("F0"),
        cache_manager_->registry_manager_->instance_infos_.at("test_instance2")->location_spec_groups().at(0).name());
    ASSERT_EQ(std::vector<std::string>({"tp0_F0", "tp0_L1", "tp1_F0", "tp1_L1"}),
              cache_manager_->registry_manager_->instance_infos_.at("test_instance2")
                  ->location_spec_groups()
                  .at(1)
                  .spec_names());
    {
        std::vector<int64_t> keys{1, 2, 3};
        auto [ec, start_write_cache_info] =
            cache_manager_->StartWriteCache(request_context_.get(), "test_instance2", keys, {}, {}, 1000);
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = start_write_cache_info.locations().cache_locations_view();
        ASSERT_EQ(0, std::get<BlockMaskOffset>(start_write_cache_info.block_mask()));
        ASSERT_EQ(3, cache_locations_view.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            const auto &cache_location = cache_locations_view[i];
            ASSERT_EQ(kDefaultStorageType, cache_location.type());
            ASSERT_EQ(4, cache_location.spec_size());
            const auto &location_specs = cache_location.location_specs();
            ASSERT_EQ(4, location_specs.size());
        }
    }
    {
        std::vector<int64_t> keys{11, 12, 13};
        auto [ec, start_write_cache_info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_instance2", keys, {}, {"F0", "F0", "F0L1"}, 1000);
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = start_write_cache_info.locations().cache_locations_view();
        ASSERT_EQ(0, std::get<BlockMaskOffset>(start_write_cache_info.block_mask()));
        ASSERT_EQ(3, cache_locations_view.size());
        for (size_t i = 0; i < 2; ++i) {
            const auto &cache_location = cache_locations_view[i];
            ASSERT_EQ(2, cache_location.spec_size());
            const auto &location_specs = cache_location.location_specs();
            ASSERT_EQ(2, location_specs.size());
            ASSERT_EQ(std::string("tp0_F0"), location_specs[0].name());
            ASSERT_EQ(std::string("tp1_F0"), location_specs[1].name());
        }
        {
            const auto &cache_location = cache_locations_view[2];
            ASSERT_EQ(4, cache_location.spec_size());
            const auto &location_specs = cache_location.location_specs();
            ASSERT_EQ(4, location_specs.size());
            ASSERT_EQ(std::string("tp0_F0"), location_specs[0].name());
            ASSERT_EQ(std::string("tp1_F0"), location_specs[1].name());
            ASSERT_EQ(std::string("tp0_L1"), location_specs[2].name());
            ASSERT_EQ(std::string("tp1_L1"), location_specs[3].name());
        }
    }
    {
        std::vector<int64_t> keys{22, 22, 23, 24};
        {
            auto [ec, start_write_cache_info] = cache_manager_->StartWriteCache(
                request_context_.get(), "test_instance2", keys, {}, {"F0L1", "F0", "F0L1"}, 1000);
            ASSERT_EQ(EC_ERROR, ec);
        }
        {
            auto [ec, start_write_cache_info] = cache_manager_->StartWriteCache(
                request_context_.get(), "test_instance2", keys, {}, {"F0L1", "F0", "F0L1_notexist"}, 1000);
            ASSERT_EQ(EC_ERROR, ec);
        }
        {
            auto [ec, start_write_cache_info] = cache_manager_->StartWriteCache(
                request_context_.get(), "test_instance2", keys, {}, {"F0", "F0L1", "F0", "F0L1"}, 1000);
            ASSERT_EQ(EC_OK, ec);
            const auto &cache_locations_view = start_write_cache_info.locations().cache_locations_view();
            ASSERT_EQ(0, std::get<BlockMaskOffset>(start_write_cache_info.block_mask()));
            ASSERT_EQ(4, cache_locations_view.size());
            for (size_t i : std::vector<size_t>({0, 2})) {
                const auto &cache_location = cache_locations_view[i];
                ASSERT_EQ(2, cache_location.spec_size());
                const auto &location_specs = cache_location.location_specs();
                ASSERT_EQ(2, location_specs.size());
                ASSERT_EQ(std::string("tp0_F0"), location_specs[0].name());
                ASSERT_EQ(std::string("tp1_F0"), location_specs[1].name());
            }
            for (size_t i : std::vector<size_t>({1, 3})) {
                const auto &cache_location = cache_locations_view[i];
                ASSERT_EQ(4, cache_location.spec_size());
                const auto &location_specs = cache_location.location_specs();
                ASSERT_EQ(4, location_specs.size());
                ASSERT_EQ(std::string("tp0_F0"), location_specs[0].name());
                ASSERT_EQ(std::string("tp1_F0"), location_specs[1].name());
                ASSERT_EQ(std::string("tp0_L1"), location_specs[2].name());
                ASSERT_EQ(std::string("tp1_L1"), location_specs[3].name());
            }
        }
    }
}

TEST_F(CacheManagerTest, TestWriteCacheTimeout) {
    cache_manager_->reclaimer_task_supervisor_->Stop();
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));
    std::vector<int64_t> keys{1, 2};
    auto [ec, start_write_cache_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 1);
    ASSERT_EQ(EC_OK, ec);
    const auto &cache_locations_view = start_write_cache_info.locations().cache_locations_view();
    ASSERT_EQ(0, std::get<BlockMaskOffset>(start_write_cache_info.block_mask()));
    ASSERT_EQ(2, cache_locations_view.size());
    std::this_thread::sleep_for(std::chrono::seconds(6));
    {
        BlockMask block_mask = static_cast<size_t>(2);
        auto ec = cache_manager_->FinishWriteCache(
            request_context_.get(), "test_instance", start_write_cache_info.write_session_id(), block_mask);
        ASSERT_EQ(EC_ERROR, ec);
    }
    ASSERT_EQ(1, cache_manager_->reclaimer_task_supervisor_->cell_queue_.Size());
}

TEST_F(CacheManagerTest, TestGetCacheLocationPrefixMatch) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));
    std::vector<int64_t> keys{1, 2, 3};
    auto [ec1, start_write_cache_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);

    {
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                      "test_instance",
                                                                      CacheManager::QueryType::QT_PREFIX_MATCH,
                                                                      keys,
                                                                      {},
                                                                      block_mask,
                                                                      0,
                                                                      {});
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_locations.cache_locations_view();
        ASSERT_EQ(0, cache_locations_view.size());
    }
    {
        BlockMask block_mask = static_cast<size_t>(3);
        auto ec = cache_manager_->FinishWriteCache(
            request_context_.get(), "test_instance", start_write_cache_info.write_session_id(), block_mask);
        ASSERT_EQ(EC_OK, ec);
    }
    {
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                      "test_instance",
                                                                      CacheManager::QueryType::QT_PREFIX_MATCH,
                                                                      keys,
                                                                      {},
                                                                      block_mask,
                                                                      0,
                                                                      {});
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_locations.cache_locations_view();
        ASSERT_EQ(3, cache_locations_view.size());
    }
    {
        std::vector<int64_t> keys{1, 2, 4, 3};
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                      "test_instance",
                                                                      CacheManager::QueryType::QT_PREFIX_MATCH,
                                                                      keys,
                                                                      {},
                                                                      block_mask,
                                                                      0,
                                                                      {});
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_locations.cache_locations_view();
        ASSERT_EQ(2, cache_locations_view.size());
    }
    {
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                      "test_instance",
                                                                      CacheManager::QueryType::QT_PREFIX_MATCH,
                                                                      keys,
                                                                      {},
                                                                      block_mask,
                                                                      0,
                                                                      {"tp0", "tp1", "tp2"});
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_locations.cache_locations_view();
        ASSERT_EQ(3, cache_locations_view.size());
        for (auto &cache_location_view : cache_locations_view) {
            ASSERT_EQ(3, cache_location_view.location_specs().size());
        }
    }
}

TEST_F(CacheManagerTest, TestGetCacheLocationHitRateCounters) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    // Write blocks {1, 2, 3}
    std::vector<int64_t> write_keys{1, 2, 3};
    auto [ec1, write_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", write_keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    BlockMask finish_mask = static_cast<size_t>(3); // all 3 blocks succeeded (offset semantics)
    ASSERT_EQ(EC_OK,
              cache_manager_->FinishWriteCache(
                  request_context_.get(), "test_instance", write_info.write_session_id(), finish_mask));

    // Create a RequestContext with a ServiceMetricsCollector so counters get incremented
    auto svc_collector = std::make_shared<ServiceMetricsCollector>(metrics_registry_);
    ASSERT_TRUE(svc_collector->Init());
    auto metrics_ctx = std::make_unique<RequestContext>("hit_rate_test", svc_collector);

    // Query 1: PrefixMatch with keys {1, 2, 4} → hits {1, 2}, miss at 4
    {
        std::vector<int64_t> query_keys{1, 2, 4};
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, result] = cache_manager_->GetCacheLocation(metrics_ctx.get(),
                                                             "test_instance",
                                                             CacheManager::QueryType::QT_PREFIX_MATCH,
                                                             query_keys,
                                                             {},
                                                             block_mask,
                                                             0,
                                                             {});
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(2u, result.cache_locations_view().size()); // 2 hits
    }

    // Query 2: PrefixMatch with keys {1, 2, 3} → all 3 hit
    {
        std::vector<int64_t> query_keys{1, 2, 3};
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, result] = cache_manager_->GetCacheLocation(metrics_ctx.get(),
                                                             "test_instance",
                                                             CacheManager::QueryType::QT_PREFIX_MATCH,
                                                             query_keys,
                                                             {},
                                                             block_mask,
                                                             0,
                                                             {});
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(3u, result.cache_locations_view().size()); // 3 hits
    }

    // Verify cumulative counters: query 3+3=6, hit 2+3=5
    Counter query_counter, hit_counter;
    COPY_METRICS_(svc_collector.get(), manager, get_cache_location_query_block_counter, query_counter);
    COPY_METRICS_(svc_collector.get(), manager, get_cache_location_hit_block_counter, hit_counter);
    EXPECT_EQ(6u, query_counter.Get());
    EXPECT_EQ(5u, hit_counter.Get());
}

TEST_F(CacheManagerTest, TestGetCacheLocationHitRateCounters_BatchGet) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    // Write blocks {1, 2, 3}
    std::vector<int64_t> write_keys{1, 2, 3};
    auto [ec1, write_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", write_keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    BlockMask finish_mask = static_cast<size_t>(3);
    ASSERT_EQ(EC_OK,
              cache_manager_->FinishWriteCache(
                  request_context_.get(), "test_instance", write_info.write_session_id(), finish_mask));

    auto svc_collector = std::make_shared<ServiceMetricsCollector>(metrics_registry_);
    ASSERT_TRUE(svc_collector->Init());
    auto metrics_ctx = std::make_unique<RequestContext>("batch_get_hit_rate_test", svc_collector);

    // BatchGet query: keys {1, 2, 99} → 2 hits (1,2), 1 miss (99)
    {
        std::vector<int64_t> query_keys{1, 2, 99};
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, result] = cache_manager_->GetCacheLocation(metrics_ctx.get(),
                                                             "test_instance",
                                                             CacheManager::QueryType::QT_BATCH_GET,
                                                             query_keys,
                                                             {},
                                                             block_mask,
                                                             0,
                                                             {});
        ASSERT_EQ(EC_OK, ec);
        // BatchGet returns all keys; hits have non-empty id, misses have empty id
        ASSERT_EQ(3u, result.cache_locations_view().size());
    }

    Counter query_counter, hit_counter;
    COPY_METRICS_(svc_collector.get(), manager, get_cache_location_query_block_counter, query_counter);
    COPY_METRICS_(svc_collector.get(), manager, get_cache_location_hit_block_counter, hit_counter);
    EXPECT_EQ(3u, query_counter.Get());
    EXPECT_EQ(2u, hit_counter.Get());
}

TEST_F(CacheManagerTest, TestGetCacheLocationHitRateCounters_ErrorPath) {
    auto svc_collector = std::make_shared<ServiceMetricsCollector>(metrics_registry_);
    ASSERT_TRUE(svc_collector->Init());
    auto metrics_ctx = std::make_unique<RequestContext>("error_path_test", svc_collector);

    // Query non-existent instance → should fail and NOT increment counters
    std::vector<int64_t> query_keys{1, 2, 3};
    BlockMask block_mask = static_cast<size_t>(0);
    auto [ec, result] = cache_manager_->GetCacheLocation(metrics_ctx.get(),
                                                         "nonexistent_instance",
                                                         CacheManager::QueryType::QT_PREFIX_MATCH,
                                                         query_keys,
                                                         {},
                                                         block_mask,
                                                         0,
                                                         {});
    EXPECT_NE(EC_OK, ec);

    Counter query_counter, hit_counter;
    COPY_METRICS_(svc_collector.get(), manager, get_cache_location_query_block_counter, query_counter);
    COPY_METRICS_(svc_collector.get(), manager, get_cache_location_hit_block_counter, hit_counter);
    EXPECT_EQ(0u, query_counter.Get());
    EXPECT_EQ(0u, hit_counter.Get());
}

TEST_F(CacheManagerTest, TestGetCacheLocationBatchGet) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));
    std::vector<int64_t> keys{1, 2, 3, 4};
    auto [ec1, start_write_cache_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    {
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                      "test_instance",
                                                                      CacheManager::QueryType::QT_BATCH_GET,
                                                                      keys,
                                                                      {},
                                                                      block_mask,
                                                                      0,
                                                                      {});
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_locations.cache_locations_view();
        ASSERT_EQ(4, cache_locations_view.size());
        for (auto &cache_location : cache_locations_view) {
            ASSERT_EQ(4, cache_location.location_specs().size());
        }
    }
    {
        BlockMask block_mask = static_cast<size_t>(3);
        auto ec = cache_manager_->FinishWriteCache(
            request_context_.get(), "test_instance", start_write_cache_info.write_session_id(), block_mask);
        ASSERT_EQ(EC_OK, ec);
    }
    {
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                      "test_instance",
                                                                      CacheManager::QueryType::QT_BATCH_GET,
                                                                      {1, 2, 3, 4},
                                                                      {},
                                                                      block_mask,
                                                                      0,
                                                                      {});
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_locations.cache_locations_view();
        ASSERT_EQ(4, cache_locations_view.size());
        ASSERT_EQ(4, cache_locations_view[0].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[1].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[2].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[3].location_specs().size());
        for (size_t i = 0; i < 3; i++) {
            expectNonEmptySpec(cache_locations_view[i].location_specs());
        }
        expectEmptySpec(cache_locations_view[3].location_specs());
    }
    {
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                      "test_instance",
                                                                      CacheManager::QueryType::QT_BATCH_GET,
                                                                      {1, 2, 111, 4},
                                                                      {},
                                                                      block_mask,
                                                                      0,
                                                                      {});
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_locations.cache_locations_view();
        ASSERT_EQ(4, cache_locations_view.size());
        ASSERT_EQ(4, cache_locations_view[0].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[1].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[2].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[3].location_specs().size());
        for (size_t i = 0; i < 2; i++) {
            expectNonEmptySpec(cache_locations_view[i].location_specs());
        }
        for (size_t i = 2; i < 4; i++) {
            expectEmptySpec(cache_locations_view[i].location_specs());
        }
    }
}

TEST_F(CacheManagerTest, TestGetCacheLocationReverseRollSlideWindowMatch) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));
    std::vector<int64_t> keys{1, 2, 3, 4, 5, 6};
    auto [ec1, start_write_cache_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    {
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                      "test_instance",
                                                                      CacheManager::QueryType::QT_REVERSE_ROLL_SW_MATCH,
                                                                      keys,
                                                                      {},
                                                                      block_mask,
                                                                      2,
                                                                      {});
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_locations.cache_locations_view();
        ASSERT_EQ(6, cache_locations_view.size());
        for (auto &cache_location : cache_locations_view) {
            ASSERT_EQ(4, cache_location.location_specs().size());
            for (auto &spec : cache_location.location_specs()) {
                ASSERT_EQ("", spec.uri());
            }
        }
    }
    {
        BlockMask block_mask = static_cast<size_t>(5);
        auto ec = cache_manager_->FinishWriteCache(
            request_context_.get(), "test_instance", start_write_cache_info.write_session_id(), block_mask);
        ASSERT_EQ(EC_OK, ec);
    }
    {
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                      "test_instance",
                                                                      CacheManager::QueryType::QT_REVERSE_ROLL_SW_MATCH,
                                                                      keys,
                                                                      {},
                                                                      block_mask,
                                                                      3,
                                                                      {});
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_locations.cache_locations_view();
        ASSERT_EQ(6, cache_locations_view.size());
        ASSERT_EQ(4, cache_locations_view[0].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[1].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[2].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[3].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[4].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[5].location_specs().size());
    }
    {
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                      "test_instance",
                                                                      CacheManager::QueryType::QT_REVERSE_ROLL_SW_MATCH,
                                                                      {1, 2, 3, 4, 5, 6, 7},
                                                                      {},
                                                                      block_mask,
                                                                      2,
                                                                      {});
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_locations.cache_locations_view();
        ASSERT_EQ(7, cache_locations_view.size());
        ASSERT_EQ(4, cache_locations_view[0].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[1].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[2].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[3].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[4].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[5].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[6].location_specs().size());

        expectEmptySpec(cache_locations_view[0].location_specs());
        expectEmptySpec(cache_locations_view[1].location_specs());
        expectEmptySpec(cache_locations_view[2].location_specs());
        expectNonEmptySpec(cache_locations_view[3].location_specs());
        expectNonEmptySpec(cache_locations_view[4].location_specs());
        expectEmptySpec(cache_locations_view[5].location_specs());
        expectEmptySpec(cache_locations_view[6].location_specs());
    }
    {
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                      "test_instance",
                                                                      CacheManager::QueryType::QT_REVERSE_ROLL_SW_MATCH,
                                                                      {1, 2, 3, 10, 5, 6, 7},
                                                                      {},
                                                                      block_mask,
                                                                      2,
                                                                      {});
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_locations.cache_locations_view();
        ASSERT_EQ(7, cache_locations_view.size());
        ASSERT_EQ(4, cache_locations_view[0].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[1].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[2].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[3].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[4].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[5].location_specs().size());
        ASSERT_EQ(4, cache_locations_view[6].location_specs().size());

        expectEmptySpec(cache_locations_view[0].location_specs());
        expectNonEmptySpec(cache_locations_view[1].location_specs());
        expectNonEmptySpec(cache_locations_view[2].location_specs());
        expectEmptySpec(cache_locations_view[3].location_specs());
        expectEmptySpec(cache_locations_view[4].location_specs());
        expectEmptySpec(cache_locations_view[5].location_specs());
        expectEmptySpec(cache_locations_view[6].location_specs());
    }
}

TEST_F(CacheManagerTest, TestGetCacheNotExistLocation) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));
    std::vector<int64_t> keys{1, 2, 3};
    auto [ec1, start_write_cache_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    {
        BlockMask block_mask = static_cast<size_t>(3);
        auto ec = cache_manager_->FinishWriteCache(
            request_context_.get(), "test_instance", start_write_cache_info.write_session_id(), block_mask);
        ASSERT_EQ(EC_OK, ec);
    }
    {
        std::vector<int64_t> keys{1, 2, 3, 12212};
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                      "test_instance",
                                                                      CacheManager::QueryType::QT_PREFIX_MATCH,
                                                                      keys,
                                                                      {},
                                                                      block_mask,
                                                                      0,
                                                                      {});
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_locations.cache_locations_view();
        ASSERT_EQ(3, cache_locations_view.size());
    }
}

TEST_F(CacheManagerTest, TestFinishWriteCacheWithBlockMask) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));
    {
        std::vector<int64_t> keys{1, 2, 3};
        auto [ec1, start_write_cache_info] =
            cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
        ASSERT_EQ(EC_OK, ec1);

        {
            BlockMask block_mask = static_cast<size_t>(0);
            auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                          "test_instance",
                                                                          CacheManager::QueryType::QT_PREFIX_MATCH,
                                                                          keys,
                                                                          {},
                                                                          block_mask,
                                                                          0,
                                                                          {});
            ASSERT_EQ(EC_OK, ec);
            const auto &cache_locations_view = cache_locations.cache_locations_view();
            ASSERT_EQ(0, cache_locations_view.size());
        }

        {
            BlockMask block_mask = static_cast<size_t>(2);
            auto ec = cache_manager_->FinishWriteCache(
                request_context_.get(), "test_instance", start_write_cache_info.write_session_id(), block_mask);
            ASSERT_EQ(EC_OK, ec);
        }

        {
            BlockMask block_mask = static_cast<size_t>(0);
            auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                          "test_instance",
                                                                          CacheManager::QueryType::QT_PREFIX_MATCH,
                                                                          keys,
                                                                          {},
                                                                          block_mask,
                                                                          0,
                                                                          {});
            ASSERT_EQ(EC_OK, ec);
            const auto &cache_locations_view = cache_locations.cache_locations_view();
            ASSERT_EQ(2, cache_locations_view.size());
        }

        {
            BlockMask block_mask = static_cast<size_t>(0);
            auto [ec, cache_metas] =
                cache_manager_->GetCacheMeta(request_context_.get(), "test_instance", keys, {}, block_mask, 0);
            ASSERT_EQ(EC_OK, ec);
            const auto &cache_locations_view = cache_metas.cache_locations_view();
            const auto &metas = cache_metas.metas();
            ASSERT_EQ(3, cache_locations_view.size());
            std::map<std::string, std::string> meta;
            ASSERT_TRUE(Jsonizable::FromJsonString(metas[2], meta));
            ASSERT_TRUE(
                CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_DELETING) == meta.at("status") ||
                CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_NOT_FOUND) == meta.at("status"));
        }
    }

    {
        std::vector<int64_t> keys{4, 5, 6, 7};
        auto [ec1, start_write_cache_info] =
            cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
        ASSERT_EQ(EC_OK, ec1);

        {
            BlockMask block_mask = static_cast<size_t>(0);
            auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                          "test_instance",
                                                                          CacheManager::QueryType::QT_PREFIX_MATCH,
                                                                          keys,
                                                                          {},
                                                                          block_mask,
                                                                          0,
                                                                          {});
            ASSERT_EQ(EC_OK, ec);
            const auto &cache_locations_view = cache_locations.cache_locations_view();
            ASSERT_EQ(0, cache_locations_view.size());
        }

        {
            BlockMask block_mask = BlockMaskVector({true, true, false, true});
            auto ec = cache_manager_->FinishWriteCache(
                request_context_.get(), "test_instance", start_write_cache_info.write_session_id(), block_mask);
            ASSERT_EQ(EC_OK, ec);
        }

        {
            BlockMask block_mask = static_cast<size_t>(0);
            auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                          "test_instance",
                                                                          CacheManager::QueryType::QT_PREFIX_MATCH,
                                                                          keys,
                                                                          {},
                                                                          block_mask,
                                                                          0,
                                                                          {});
            ASSERT_EQ(EC_OK, ec);
            const auto &cache_locations_view = cache_locations.cache_locations_view();
            ASSERT_EQ(2, cache_locations_view.size());
        }

        {
            BlockMask block_mask = static_cast<size_t>(0);
            auto [ec, cache_metas] =
                cache_manager_->GetCacheMeta(request_context_.get(), "test_instance", keys, {}, block_mask, 0);
            ASSERT_EQ(EC_OK, ec);
            const auto &cache_locations_view = cache_metas.cache_locations_view();
            const auto &metas = cache_metas.metas();
            ASSERT_EQ(4, cache_locations_view.size());
            std::map<std::string, std::string> meta;
            std::vector<int> pos_vec = {0, 1, 3};
            for (int pos : pos_vec) {
                ASSERT_TRUE(Jsonizable::FromJsonString(metas[pos], meta));
                ASSERT_EQ(CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_SERVING),
                          meta.at("status"));
            }
            ASSERT_TRUE(Jsonizable::FromJsonString(metas[2], meta));
            ASSERT_TRUE(
                CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_DELETING) == meta.at("status") ||
                CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_NOT_FOUND) == meta.at("status"));
        }
    }
}

TEST_F(CacheManagerTest, TestGetCacheMeta) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));
    std::vector<int64_t> keys{1, 2, 3};
    auto [ec1, start_write_cache_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);

    {
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_metas] =
            cache_manager_->GetCacheMeta(request_context_.get(), "test_instance", keys, {}, block_mask, 0);
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_metas.cache_locations_view();
        const auto &metas = cache_metas.metas();
        ASSERT_EQ(3, cache_locations_view.size());
        ASSERT_EQ(3, metas.size());
        for (int i = 0; i < 3; ++i) {
            std::map<std::string, std::string> meta;
            ASSERT_TRUE(Jsonizable::FromJsonString(metas[i], meta));
            ASSERT_EQ(CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_WRITING), meta.at("status"));
        }
    }
    {
        BlockMask block_mask = static_cast<size_t>(3);
        auto ec = cache_manager_->FinishWriteCache(
            request_context_.get(), "test_instance", start_write_cache_info.write_session_id(), block_mask);
        ASSERT_EQ(EC_OK, ec);
    }
    {
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_metas] =
            cache_manager_->GetCacheMeta(request_context_.get(), "test_instance", keys, {}, block_mask, 0);
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_metas.cache_locations_view();
        const auto &metas = cache_metas.metas();
        ASSERT_EQ(3, cache_locations_view.size());
        ASSERT_EQ(3, metas.size());
        for (int i = 0; i < 3; ++i) {
            std::map<std::string, std::string> meta;
            ASSERT_TRUE(Jsonizable::FromJsonString(metas[i], meta));
            ASSERT_EQ(CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_SERVING), meta.at("status"));
        }
    }
}

TEST_F(CacheManagerTest, TestGetNotExistCacheMeta) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));
    std::vector<int64_t> keys{1, 2, 3, 4};
    auto [ec1, start_write_cache_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);

    {
        std::vector<int64_t> keys{1, 2, 3, 111111, 4};
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_metas] =
            cache_manager_->GetCacheMeta(request_context_.get(), "test_instance", keys, {}, block_mask, 0);
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_metas.cache_locations_view();
        const auto &metas = cache_metas.metas();
        ASSERT_EQ(5, cache_locations_view.size());
        ASSERT_EQ(5, metas.size());
        std::vector<int> pos_vec = {0, 1, 2, 4};
        std::map<std::string, std::string> meta;
        ASSERT_TRUE(Jsonizable::FromJsonString(metas[3], meta));
        ASSERT_EQ(CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_NOT_FOUND), meta.at("status"));
        for (int pos : pos_vec) {
            ASSERT_TRUE(Jsonizable::FromJsonString(metas[pos], meta));
            ASSERT_EQ(CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_WRITING), meta.at("status"));
        }
    }
}

TEST_F(CacheManagerTest, TestRemoveCache) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "placeholder_id",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));
    std::vector<int64_t> keys{1, 2, 3};
    auto [ec1, start_write_cache_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "placeholder_id", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    BlockMask block_mask = static_cast<size_t>(0);

    {
        auto [ec, cache_metas] =
            cache_manager_->GetCacheMeta(request_context_.get(), "placeholder_id", keys, {}, block_mask, 0);
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_metas.cache_locations_view();
        const auto &metas = cache_metas.metas();
        ASSERT_EQ(3, cache_locations_view.size());
        ASSERT_EQ(3, metas.size());
    }

    {
        auto ec = cache_manager_->RemoveCache(request_context_.get(), "placeholder_id", keys, {}, block_mask);
        ASSERT_EQ(EC_OK, ec);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    {
        auto [ec, cache_metas] =
            cache_manager_->GetCacheMeta(request_context_.get(), "placeholder_id", keys, {}, block_mask, 0);
        ASSERT_EQ(EC_OK, ec);
        const auto &cache_locations_view = cache_metas.cache_locations_view();
        const auto &metas = cache_metas.metas();
        ASSERT_EQ(3, cache_locations_view.size());
        ASSERT_EQ(3, metas.size());
        for (int i = 0; i < 3; ++i) {
            std::map<std::string, std::string> meta;
            ASSERT_TRUE(Jsonizable::FromJsonString(metas[i], meta));
            ASSERT_EQ(CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_NOT_FOUND),
                      meta.at("status"));
        }
    }
}

TEST_F(CacheManagerTest, TestTrimCache) {
    {
        ASSERT_EQ(
            ErrorCode::EC_UNIMPLEMENTED,
            cache_manager_->TrimCache(request_context_.get(), "ins_id_00", proto::meta::TrimStrategy::TS_UNSPECIFIED));
        ASSERT_EQ(ErrorCode::EC_UNIMPLEMENTED,
                  cache_manager_->TrimCache(
                      request_context_.get(), "ins_id_00", proto::meta::TrimStrategy::TS_REMOVE_ALL_META));
        ASSERT_EQ(
            ErrorCode::EC_UNIMPLEMENTED,
            cache_manager_->TrimCache(request_context_.get(), "ins_id_00", proto::meta::TrimStrategy::TS_TIMESTAMP));
    }

    {
        cache_manager_->RegisterInstance(request_context_.get(),
                                         "default",
                                         "placeholder_id",
                                         64,
                                         createLocationSpecInfos(),
                                         createModelDeployment(),
                                         std::vector<LocationSpecGroup>());
        std::vector<std::int64_t> keys{1, 2, 3};
        auto [ec0, _info] =
            cache_manager_->StartWriteCache(request_context_.get(), "placeholder_id", keys, {}, {}, 100000000);

        auto ec1 = cache_manager_->TrimCache(
            request_context_.get(), "placeholder_id", proto::meta::TrimStrategy::TS_REMOVE_ALL_CACHE);
        ASSERT_EQ(ErrorCode::EC_OK, ec1);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        BlockMask block_mask = static_cast<std::size_t>(0);

        auto [ec2, cache_metas] =
            cache_manager_->GetCacheMeta(request_context_.get(), "placeholder_id", keys, {}, block_mask, 0);
        const auto &cache_locations_view = cache_metas.cache_locations_view();
        const auto &metas = cache_metas.metas();
        ASSERT_EQ(3, cache_locations_view.size());
        ASSERT_EQ(3, metas.size());
        for (int i = 0; i < 3; ++i) {
            std::map<std::string, std::string> meta;
            ASSERT_TRUE(Jsonizable::FromJsonString(metas[i], meta));
            ASSERT_EQ(CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_NOT_FOUND),
                      meta.at("status"));
        }
    }

    {
        cache_manager_->RegisterInstance(request_context_.get(),
                                         "default",
                                         "placeholder_id",
                                         64,
                                         createLocationSpecInfos(),
                                         createModelDeployment(),
                                         std::vector<LocationSpecGroup>());
        std::vector<std::int64_t> keys;
        for (std::int64_t i = 0; i < 65; ++i) {
            keys.push_back(i);
        }

        auto [ec0, _info] =
            cache_manager_->StartWriteCache(request_context_.get(), "placeholder_id", keys, {}, {}, 100000000);

        auto ec1 = cache_manager_->TrimCache(
            request_context_.get(), "placeholder_id", proto::meta::TrimStrategy::TS_REMOVE_ALL_CACHE);
        ASSERT_EQ(ErrorCode::EC_OK, ec1);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        BlockMask block_mask = static_cast<std::size_t>(0);

        auto [ec2, cache_metas] =
            cache_manager_->GetCacheMeta(request_context_.get(), "placeholder_id", keys, {}, block_mask, 0);
        const auto &cache_locations_view = cache_metas.cache_locations_view();
        const auto &metas = cache_metas.metas();
        ASSERT_EQ(65, cache_locations_view.size());
        ASSERT_EQ(65, metas.size());
        for (int i = 0; i < 65; ++i) {
            std::map<std::string, std::string> meta;
            ASSERT_TRUE(Jsonizable::FromJsonString(metas[i], meta));
            ASSERT_EQ(CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_NOT_FOUND),
                      meta.at("status"));
        }
    }
}

TEST_F(CacheManagerTest, TestUnavailableStorage) {
    auto registry_manager = cache_manager_->registry_manager_;
    RequestContext context("TestUnavailableStorage");
    { // nfs_test_01
        std::string config_str =
            R"({"type":"file","global_unique_name":"nfs_test_01","storage_spec":{"root_path":"/tmp/nfs_test_01/","timeout":1000}})";
        StorageConfig config;
        ASSERT_TRUE(config.FromJsonString(config_str));
        ASSERT_EQ(EC_OK, registry_manager->AddStorage(&context, config));
    }
    { // nfs_test_02
        std::string config_str =
            R"({"type":"file","global_unique_name":"nfs_test_02","storage_spec":{"root_path":"/tmp/nfs_test_01/","timeout":1000}})";
        StorageConfig config;
        ASSERT_TRUE(config.FromJsonString(config_str));
        ASSERT_EQ(EC_OK, registry_manager->AddStorage(&context, config));
    }
    { // 3fs_test_01
        std::string config_str =
            R"({"type":"hf3fs","global_unique_name":"3fs_test_01","storage_spec":{"cluster_name":"test_cluster_name","mountpoint":"/3fs/test_mountpoint","root_dir":"test_root_dir","key_count_per_file":2}})";
        StorageConfig config;
        ASSERT_TRUE(config.FromJsonString(config_str));
        ASSERT_EQ(EC_OK, registry_manager->AddStorage(&context, config));
    }
    { // registry instance group
        InstanceGroup instance_group;
        std::string instance_group_str = R"(
{
    "name": "test_group2",
    "storage_candidates":
    [
        "nfs_test_01",
        "nfs_test_02",
        "3fs_test_01"
    ],
    "global_quota_group_name": "test_quota_group2",
    "max_instance_count": 100,
    "quota":
    {
        "capacity": 10737418240,
        "quota_config":
        [
            {
                "capacity": 10737418240,
                "storage_type": "file"
            }
        ]
    },
    "cache_config":
    {
        "reclaim_strategy":
        {
            "storage_unique_name": "",
            "reclaim_policy": 1,
            "trigger_strategy":
            {
                "used_size": 1073741824,
                "used_percentage": 0.8
            },
            "trigger_period_seconds": 60,
            "reclaim_step_size": 1073741824,
            "reclaim_step_percentage": 10,
            "delay_before_delete_ms": 1000
        },
        "cache_prefer_strategy": 2,
        "meta_indexer_config": {}
    },
    "user_data": "{\"description\": \"Test instance group\"}",
    "version": 1
}
)";
        instance_group.FromJsonString(instance_group_str);
        ASSERT_EQ(EC_OK, registry_manager->CreateInstanceGroup(request_context_.get(), instance_group));
    }
    auto expected = std::pair<ErrorCode, std::string>(
        EC_OK,
        "[{\"type\":\"hf3fs\",\"is_available\":true,\"global_unique_name\":\"3fs_test_01\",\"storage_spec\":{\"cluster_"
        "name\":\"test_cluster_name\",\"mountpoint\":\"/3fs/"
        "test_mountpoint\",\"root_dir\":\"test_root_dir\",\"key_count_per_file\":2}},{\"type\":\"file\",\"is_"
        "available\":true,\"global_unique_name\":\"nfs_test_01\",\"storage_spec\":{\"root_path\":\"/tmp/nfs_test_01/"
        "\",\"key_count_per_file\":1}},{\"type\":\"file\",\"is_available\":true,\"global_unique_name\":\"nfs_test_02\","
        "\"storage_spec\":{\"root_path\":\"/tmp/nfs_test_01/\",\"key_count_per_file\":1}}]");
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "test_group2",
                                               "test_group2_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    auto test_write_and_find_location = [this](int start,
                                               DataStorageType expect_type,
                                               const std::string &expect_sub_path) {
        for (int i = start; i < start + 10; ++i) {
            std::vector<int64_t> keys{i * 10 + 1, i * 10 + 2, i * 10 + 3, i * 10 + 4};
            auto [ec1, start_write_cache_info] = cache_manager_->StartWriteCache(
                request_context_.get(), "test_group2_instance", keys, {}, {}, 100000000);
            ASSERT_EQ(EC_OK, ec1);
            ASSERT_EQ(4, start_write_cache_info.locations().cache_locations_view().size());
            for (const auto &start_write_location : start_write_cache_info.locations().cache_locations_view()) {
                ASSERT_EQ(expect_type, start_write_location.type());
                const auto &location_spec = start_write_location.location_specs().front();
                ASSERT_THAT(location_spec.uri(), HasSubstr(expect_sub_path));
            }
            {
                BlockMask block_mask = static_cast<size_t>(0);
                auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                              "test_group2_instance",
                                                                              CacheManager::QueryType::QT_PREFIX_MATCH,
                                                                              keys,
                                                                              {},
                                                                              block_mask,
                                                                              0,
                                                                              {});
                ASSERT_EQ(EC_OK, ec);
                const auto &cache_locations_view = cache_locations.cache_locations_view();
                ASSERT_EQ(0, cache_locations_view.size());
            }
            {
                BlockMask block_mask = static_cast<size_t>(4);
                auto ec = cache_manager_->FinishWriteCache(request_context_.get(),
                                                           "test_group2_instance",
                                                           start_write_cache_info.write_session_id(),
                                                           block_mask);
                ASSERT_EQ(EC_OK, ec);
            }
            {
                BlockMask block_mask = static_cast<size_t>(0);
                auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                              "test_group2_instance",
                                                                              CacheManager::QueryType::QT_PREFIX_MATCH,
                                                                              keys,
                                                                              {},
                                                                              block_mask,
                                                                              0,
                                                                              {});
                ASSERT_EQ(EC_OK, ec);
                const auto &cache_locations_view = cache_locations.cache_locations_view();
                ASSERT_EQ(4, cache_locations_view.size());
            }
        }
    };

    auto test_match_location = [this](int start, size_t expect_location_size, const std::string &expect_sub_path = "") {
        for (int i = start; i < start + 10; ++i) {
            std::vector<int64_t> keys{i * 10 + 1, i * 10 + 2, i * 10 + 3, i * 10 + 4};
            BlockMask block_mask = static_cast<size_t>(0);
            auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                          "test_group2_instance",
                                                                          CacheManager::QueryType::QT_PREFIX_MATCH,
                                                                          keys,
                                                                          {},
                                                                          block_mask,
                                                                          0,
                                                                          {});
            ASSERT_EQ(EC_OK, ec);
            const auto &cache_locations_view = cache_locations.cache_locations_view();
            ASSERT_EQ(expect_location_size, cache_locations_view.size());
            for (const auto &cache_location : cache_locations_view) {
                for (const auto &location : cache_location.location_specs()) {
                    ASSERT_THAT(location.uri(), HasSubstr(expect_sub_path));
                }
            }
        }
    };

    // PREFER_3FS, use 3fs_test_01
    test_write_and_find_location(0, DataStorageType::DATA_STORAGE_TYPE_HF3FS, "3fs_test_01");
    test_match_location(0, 4, "3fs_test_01");
    // 3fs_test_01 unavailable
    ASSERT_EQ(EC_OK, registry_manager->DisableStorage(request_context_.get(), "3fs_test_01"));
    test_match_location(0, 0); // not match available location
    // use nfs_test_01
    test_write_and_find_location(0, DataStorageType::DATA_STORAGE_TYPE_NFS, "nfs_test_01");
    test_write_and_find_location(10, DataStorageType::DATA_STORAGE_TYPE_NFS, "nfs_test_01");
    test_match_location(0, 4, "nfs_test_01");
    test_match_location(10, 4, "nfs_test_01");
    // nfs_test_01 unavailable
    ASSERT_EQ(EC_OK, registry_manager->DisableStorage(request_context_.get(), "nfs_test_01"));
    test_match_location(0, 0);  // not match available location
    test_match_location(10, 0); // not match available location
    // use nfs_test_02
    test_write_and_find_location(0, DataStorageType::DATA_STORAGE_TYPE_NFS, "nfs_test_02");
    test_write_and_find_location(20, DataStorageType::DATA_STORAGE_TYPE_NFS, "nfs_test_02");
    test_match_location(0, 4, "nfs_test_02");
    test_match_location(10, 0); // not match available location
    test_match_location(20, 4, "nfs_test_02");
    // nfs_test_01 available again
    ASSERT_EQ(EC_OK, registry_manager->EnableStorage(request_context_.get(), "nfs_test_01"));
    test_match_location(10, 4, "nfs_test_01"); // match available location
}

TEST_F(CacheManagerTest, TestStartWriteCacheWithNoAvailableStorage) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    ASSERT_EQ(EC_OK, registry_manager_->DisableStorage(request_context_.get(), "nfs_01"));

    std::vector<int64_t> keys{1, 2, 3, 4};
    auto [ec, start_write_cache_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    EXPECT_EQ(EC_ERROR, ec);
    EXPECT_EQ(0, start_write_cache_info.locations().cache_locations_view().size());
}

TEST_F(CacheManagerTest, TestGetCacheLocationLen) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    std::vector<int64_t> keys{1, 2, 3, 4, 5, 6, 7};
    auto [ec1, start_write_cache_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);

    {
        BlockMask block_mask = static_cast<size_t>(5);
        auto ec = cache_manager_->FinishWriteCache(
            request_context_.get(), "test_instance", start_write_cache_info.write_session_id(), block_mask);
        ASSERT_EQ(EC_OK, ec);
    }

    // Test QT_PREFIX_MATCH
    {
        std::vector<int64_t> keys{1, 2, 8, 4, 5, 6};
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_location_len] = cache_manager_->GetCacheLocationLen(
            request_context_.get(), "test_instance", CacheManager::QueryType::QT_PREFIX_MATCH, keys, {}, 0);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(2, cache_location_len);
    }

    // Test QT_BATCH_GET
    {
        std::vector<int64_t> keys{1, 2, 8, 4, 5, 9, 6};
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_location_len] = cache_manager_->GetCacheLocationLen(
            request_context_.get(), "test_instance", CacheManager::QueryType::QT_BATCH_GET, keys, {}, 0);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(4, cache_location_len);
    }

    // Test QT_REVERSE_ROLL_SW_MATCH
    {
        std::vector<int64_t> keys{1, 2, 3, 8, 5, 9, 6};
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_location_len] = cache_manager_->GetCacheLocationLen(
            request_context_.get(), "test_instance", CacheManager::QueryType::QT_REVERSE_ROLL_SW_MATCH, keys, {}, 2);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(2, cache_location_len);
    }
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_NullRegistryManager) {
    // when registry_manager_ is null, the functor should return true
    // (assume data exists as a safe fallback)
    auto saved = cache_manager_->registry_manager_;
    cache_manager_->registry_manager_ = nullptr;

    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_location_specs({LocationSpec("tp0", "file://mock_store/path")});
    ASSERT_TRUE(func(loc));

    cache_manager_->registry_manager_ = saved;
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_NullDataStorageManager) {
    // when data_storage_manager() is null, the functor should return
    // true
    auto saved = registry_manager_->data_storage_manager_;
    registry_manager_->data_storage_manager_ = nullptr;

    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_location_specs({LocationSpec("tp0", "file://mock_store/path")});
    ASSERT_TRUE(func(loc));

    registry_manager_->data_storage_manager_ = saved;
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_EmptyLocationSpecs) {
    // no location specs -> no URIs to check -> returns true
    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    ASSERT_TRUE(func(loc));
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_InvalidUri) {
    // invalid URI string (no protocol) -> DataStorageUri::Valid() is
    // false -> no valid URIs collected -> returns true
    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_location_specs({LocationSpec("tp0", "no_protocol_here")});
    ASSERT_TRUE(func(loc));
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_AllExist) {
    // inject a mock backend where MightExist returns all true;
    // the functor should return true
    auto metrics_registry = cache_manager_->metrics_registry_;
    auto mock_backend = std::make_shared<MockDataStorageBackend>(metrics_registry);
    EXPECT_CALL(*mock_backend, MightExist(_)).WillOnce([](const std::vector<DataStorageUri> &uris) {
        return std::vector<bool>(uris.size(), true);
    });

    auto dsm = registry_manager_->data_storage_manager_;
    dsm->storage_map_["mock_store"] = mock_backend;

    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_location_specs(
        {LocationSpec("tp0", "file://mock_store/path_a"), LocationSpec("tp1", "file://mock_store/path_b")});
    ASSERT_TRUE(func(loc));

    dsm->storage_map_.erase("mock_store");
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_NoneExist) {
    // inject a mock backend where MightExist returns all false;
    // the functor should return false
    auto metrics_registry = cache_manager_->metrics_registry_;
    auto mock_backend = std::make_shared<MockDataStorageBackend>(metrics_registry);
    EXPECT_CALL(*mock_backend, MightExist(_)).WillOnce([](const std::vector<DataStorageUri> &uris) {
        return std::vector<bool>(uris.size(), false);
    });

    auto dsm = registry_manager_->data_storage_manager_;
    dsm->storage_map_["mock_store"] = mock_backend;

    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_location_specs(
        {LocationSpec("tp0", "file://mock_store/path_a"), LocationSpec("tp1", "file://mock_store/path_b")});
    ASSERT_FALSE(func(loc));

    dsm->storage_map_.erase("mock_store");
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_PartialExist) {
    // inject a mock backend where MightExist returns mixed results;
    // std::all_of requires all true, so the functor should return false
    auto metrics_registry = cache_manager_->metrics_registry_;
    auto mock_backend = std::make_shared<MockDataStorageBackend>(metrics_registry);
    EXPECT_CALL(*mock_backend, MightExist(_)).WillOnce([](const std::vector<DataStorageUri> &uris) {
        std::vector<bool> result(uris.size(), true);
        // mark the last URI as non-existent
        result.back() = false;
        return result;
    });

    auto dsm = registry_manager_->data_storage_manager_;
    dsm->storage_map_["mock_store"] = mock_backend;

    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_location_specs(
        {LocationSpec("tp0", "file://mock_store/path_a"), LocationSpec("tp1", "file://mock_store/path_b")});
    ASSERT_FALSE(func(loc));

    dsm->storage_map_.erase("mock_store");
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_VerifiesUriPassthrough) {
    // verify that the functor passes the correct parsed URIs to
    // MightExist and uses the hostname from the first URI for backend
    // lookup
    auto metrics_registry = cache_manager_->metrics_registry_;
    auto mock_backend = std::make_shared<MockDataStorageBackend>(metrics_registry);
    EXPECT_CALL(*mock_backend, MightExist(_)).WillOnce([](const std::vector<DataStorageUri> &uris) {
        // should receive exactly 2 valid URIs (invalid ones
        // filtered out)
        EXPECT_EQ(2u, uris.size());
        EXPECT_EQ("mock_store", uris[0].GetHostName());
        EXPECT_EQ("mock_store", uris[1].GetHostName());
        return std::vector<bool>{true, true};
    });

    auto dsm = registry_manager_->data_storage_manager_;
    dsm->storage_map_["mock_store"] = mock_backend;

    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    // mix of invalid and valid URIs; only valid ones should reach
    // MightExist
    loc.set_location_specs({LocationSpec("tp0", "no_protocol"),
                            LocationSpec("tp1", "file://mock_store/path_a"),
                            LocationSpec("tp2", "file://mock_store/path_b")});
    ASSERT_TRUE(func(loc));

    dsm->storage_map_.erase("mock_store");
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_UnregisteredBackend) {
    // valid URIs whose hostname does not match any registered backend;
    // DataStorageManager::Exist returns an empty vector, and
    // std::all_of on an empty range is true -> functor returns true
    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_location_specs({LocationSpec("tp0", "file://nonexistent_backend/path")});
    ASSERT_TRUE(func(loc));
}

TEST_F(CacheManagerTest, TestGetSubmitDelReqFunc_NullExecutor) {
    // when schedule_plan_executor_ is null, calling the functor should
    // not crash
    auto saved = cache_manager_->schedule_plan_executor_;
    cache_manager_->schedule_plan_executor_ = nullptr;

    auto func = cache_manager_->GetSubmitDelReqFunc("test_instance");
    func({1, 2, 3}, {{"loc_a"}, {"loc_b"}, {"loc_c"}});

    cache_manager_->schedule_plan_executor_ = saved;
}

TEST_F(CacheManagerTest, TestGetSubmitDelReqFunc_SubmitsToExecutor) {
    // verify that the functor actually submits a task to the
    // schedule_plan_executor_ by checking that the internal task queue
    // grows
    auto &executor = cache_manager_->schedule_plan_executor_;

    // stop worker threads so tasks accumulate in the queue without
    // being consumed
    executor->stop_.store(true);
    executor->condition_.notify_all();
    for (auto &w : executor->workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    executor->workers_.clear();
    // reset stop_ so SubmitRaw accepts new tasks
    executor->stop_.store(false);

    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        executor->tasks_.clear();
    }

    auto func = cache_manager_->GetSubmitDelReqFunc("test_instance");
    func({100, 200}, {{"loc_a"}, {"loc_b"}});

    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        ASSERT_EQ(1u, executor->tasks_.size());
    }

    // submit a second request and verify count increases
    func({300}, {{"loc_c"}});
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        ASSERT_EQ(2u, executor->tasks_.size());
    }

    // clean up: clear tasks so executor destructor is clean
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        executor->tasks_.clear();
    }
}

TEST_F(CacheManagerTest, TestGetSubmitDelReqFunc_DeletesLocationMetadata) {
    // end-to-end: write cache entries, then use the del functor to
    // request deletion; verify the location status changes
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    std::vector<std::int64_t> keys{1001, 1002};
    auto [ec1, start_write_cache_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);

    {
        BlockMask block_mask = static_cast<std::size_t>(2);
        auto ec = cache_manager_->FinishWriteCache(
            request_context_.get(), "test_instance", start_write_cache_info.write_session_id(), block_mask);
        ASSERT_EQ(EC_OK, ec);
    }

    // collect location IDs from metadata
    BlockMask block_mask = static_cast<std::size_t>(0);
    std::vector<std::vector<std::string>> loc_ids;
    {
        auto [ec, cache_metas] =
            cache_manager_->GetCacheMeta(request_context_.get(), "test_instance", keys, {}, block_mask, 0);
        ASSERT_EQ(EC_OK, ec);
        const auto &views = cache_metas.cache_locations_view();
        ASSERT_EQ(2u, views.size());
        for (const auto &view : views) {
            std::map<std::string, std::string> meta;
            ASSERT_TRUE(Jsonizable::FromJsonString(cache_metas.metas()[&view - &views[0]], meta));
            ASSERT_EQ(CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_SERVING), meta.at("status"));
            loc_ids.push_back({view.cache_location_.id()});
        }
    }

    // use GetSubmitDelReqFunc to submit a deletion request
    auto del_func = cache_manager_->GetSubmitDelReqFunc("test_instance");
    del_func(keys, loc_ids);

    // wait for the async executor to process the request
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // verify location statuses changed to DELETING or NOT_FOUND
    {
        auto [ec, cache_metas] =
            cache_manager_->GetCacheMeta(request_context_.get(), "test_instance", keys, {}, block_mask, 0);
        ASSERT_EQ(EC_OK, ec);
        const auto &metas = cache_metas.metas();
        ASSERT_EQ(2u, metas.size());
        for (int i = 0; i < 2; ++i) {
            std::map<std::string, std::string> meta;
            ASSERT_TRUE(Jsonizable::FromJsonString(metas[i], meta));
            auto status = meta.at("status");
            ASSERT_TRUE(status == CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_DELETING) ||
                        status == CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_NOT_FOUND))
                << "expected DELETING or NOT_FOUND, got: " << status;
        }
    }
}

// ---------------------------------------------------------------------
// FilterWriteCache tests: verify the aggressive prune policy feature
// where stale locations (data no longer on storage) are detected via
// MightExist, excluded from the block mask, and submitted for deletion.
// ---------------------------------------------------------------------

TEST_F(CacheManagerTest, TestFilterWriteCache_NoStaleLocations) {
    // baseline: all existing locations have valid data, so behaviour
    // should be the same as before the aggressive prune feature
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    // write keys {1,2} and finish them as CLS_SERVING
    std::vector<std::int64_t> write_keys{1, 2};
    auto [ec1, swci1] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", write_keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    {
        BlockMask bm = static_cast<std::size_t>(2);
        ASSERT_EQ(
            EC_OK,
            cache_manager_->FinishWriteCache(request_context_.get(), "test_instance", swci1.write_session_id(), bm));
    }

    // install interceptor that returns all-true (data exists)
    auto dsm = registry_manager_->data_storage_manager_;
    auto original = dsm->storage_map_["nfs_01"];
    std::atomic<int> might_exist_calls{0};
    dsm->storage_map_["nfs_01"] = std::make_shared<MightExistInterceptor>(
        original, [&might_exist_calls](const std::vector<DataStorageUri> &uris) {
            might_exist_calls.fetch_add(1);
            return std::vector<bool>(uris.size(), true);
        });

    // StartWriteCache with {1, 2, 3, 4}; keys 1,2 exist, 3,4 need
    // new writes
    std::vector<std::int64_t> keys{1, 2, 3, 4};
    auto [ec2, swci2] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec2);

    // contiguous prefix → BlockMaskOffset
    ASSERT_EQ(2u, std::get<BlockMaskOffset>(swci2.block_mask()));
    ASSERT_EQ(2u, swci2.locations().cache_locations_view().size());

    // MightExist should have been called for the 2 existing
    // CLS_SERVING locations
    ASSERT_EQ(2, might_exist_calls.load());

    dsm->storage_map_["nfs_01"] = original;
}

TEST_F(CacheManagerTest, TestFilterWriteCache_StaleBreaksPrefix) {
    // one stale location in the middle of what would otherwise be a
    // contiguous prefix; the block mask should become a vector
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    // write keys {1,2,3} and finish as CLS_SERVING
    std::vector<std::int64_t> write_keys{1, 2, 3};
    auto [ec1, swci1] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", write_keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    {
        BlockMask bm = static_cast<std::size_t>(3);
        ASSERT_EQ(
            EC_OK,
            cache_manager_->FinishWriteCache(request_context_.get(), "test_instance", swci1.write_session_id(), bm));
    }

    // stop executor workers so we can inspect queued tasks
    auto &executor = cache_manager_->schedule_plan_executor_;
    executor->stop_.store(true);
    executor->condition_.notify_all();
    for (auto &w : executor->workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    executor->workers_.clear();
    executor->stop_.store(false);
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        executor->tasks_.clear();
    }

    // install interceptor: key 1 exists, key 2 stale, key 3 exists
    auto dsm = registry_manager_->data_storage_manager_;
    auto original = dsm->storage_map_["nfs_01"];
    std::atomic<int> call_idx{0};
    dsm->storage_map_["nfs_01"] =
        std::make_shared<MightExistInterceptor>(original, [&call_idx](const std::vector<DataStorageUri> &uris) {
            int idx = call_idx.fetch_add(1);
            if (idx == 1) {
                // second CLS_SERVING location (key 2): stale
                return std::vector<bool>(uris.size(), false);
            }
            return std::vector<bool>(uris.size(), true);
        });

    // StartWriteCache with {1, 2, 3, 4}
    std::vector<std::int64_t> keys{1, 2, 3, 4};
    auto [ec2, swci2] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec2);

    // key 2 stale breaks the contiguous prefix → BlockMaskVector
    auto mask = std::get<BlockMaskVector>(swci2.block_mask());
    ASSERT_EQ(4u, mask.size());
    ASSERT_TRUE(mask[0]);  // key 1: valid
    ASSERT_FALSE(mask[1]); // key 2: stale
    ASSERT_TRUE(mask[2]);  // key 3: valid
    ASSERT_FALSE(mask[3]); // key 4: not written yet
    // 2 new write locations: for keys 2 and 4
    ASSERT_EQ(2u, swci2.locations().cache_locations_view().size());

    // a deletion request should have been submitted for the stale
    // location
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        ASSERT_EQ(1u, executor->tasks_.size());
    }

    // clean up
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        executor->tasks_.clear();
    }
    dsm->storage_map_["nfs_01"] = original;
}

TEST_F(CacheManagerTest, TestFilterWriteCache_AllStale) {
    // all existing locations are stale; block mask should indicate all
    // keys need new writes (offset 0); deletion requests submitted for
    // all stale keys
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    // write keys {1,2} and finish as CLS_SERVING
    std::vector<std::int64_t> write_keys{1, 2};
    auto [ec1, swci1] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", write_keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    {
        BlockMask bm = static_cast<std::size_t>(2);
        ASSERT_EQ(
            EC_OK,
            cache_manager_->FinishWriteCache(request_context_.get(), "test_instance", swci1.write_session_id(), bm));
    }

    // stop executor workers
    auto &executor = cache_manager_->schedule_plan_executor_;
    executor->stop_.store(true);
    executor->condition_.notify_all();
    for (auto &w : executor->workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    executor->workers_.clear();
    executor->stop_.store(false);
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        executor->tasks_.clear();
    }

    // install interceptor: all stale
    auto dsm = registry_manager_->data_storage_manager_;
    auto original = dsm->storage_map_["nfs_01"];
    dsm->storage_map_["nfs_01"] = std::make_shared<MightExistInterceptor>(
        original, [](const std::vector<DataStorageUri> &uris) { return std::vector<bool>(uris.size(), false); });

    // StartWriteCache with {1, 2, 3}
    std::vector<std::int64_t> keys{1, 2, 3};
    auto [ec2, swci2] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec2);

    // all stale + one new key → contiguous prefix of not-existing from
    // offset 0
    ASSERT_EQ(0u, std::get<BlockMaskOffset>(swci2.block_mask()));
    // all 3 keys need new write locations
    ASSERT_EQ(3u, swci2.locations().cache_locations_view().size());

    // deletion request submitted for the 2 stale keys
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        ASSERT_EQ(1u, executor->tasks_.size());
    }

    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        executor->tasks_.clear();
    }
    dsm->storage_map_["nfs_01"] = original;
}

TEST_F(CacheManagerTest, TestFilterWriteCache_StaleSuffix) {
    // stale locations are in the suffix (after the first valid prefix);
    // since all entries from first_empty onward are not-existing, the
    // contiguous prefix optimisation (BlockMaskOffset) still applies
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    // write keys {1,2,3} and finish as CLS_SERVING
    std::vector<std::int64_t> write_keys{1, 2, 3};
    auto [ec1, swci1] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", write_keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    {
        BlockMask bm = static_cast<std::size_t>(3);
        ASSERT_EQ(
            EC_OK,
            cache_manager_->FinishWriteCache(request_context_.get(), "test_instance", swci1.write_session_id(), bm));
    }

    // stop executor workers
    auto &executor = cache_manager_->schedule_plan_executor_;
    executor->stop_.store(true);
    executor->condition_.notify_all();
    for (auto &w : executor->workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    executor->workers_.clear();
    executor->stop_.store(false);
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        executor->tasks_.clear();
    }

    // install interceptor: key 1 valid, key 2 stale, key 3 stale
    auto dsm = registry_manager_->data_storage_manager_;
    auto original = dsm->storage_map_["nfs_01"];
    std::atomic<int> call_idx{0};
    dsm->storage_map_["nfs_01"] =
        std::make_shared<MightExistInterceptor>(original, [&call_idx](const std::vector<DataStorageUri> &uris) {
            int idx = call_idx.fetch_add(1);
            if (idx == 0) {
                // first CLS_SERVING location (key 1): valid
                return std::vector<bool>(uris.size(), true);
            }
            // keys 2, 3: stale
            return std::vector<bool>(uris.size(), false);
        });

    // StartWriteCache with {1, 2, 3}; key 1 exists, keys 2,3 stale
    std::vector<std::int64_t> keys{1, 2, 3};
    auto [ec2, swci2] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec2);

    // from offset 1 onward all are not-existing → BlockMaskOffset
    ASSERT_EQ(1u, std::get<BlockMaskOffset>(swci2.block_mask()));
    // 2 new write locations for keys 2,3
    ASSERT_EQ(2u, swci2.locations().cache_locations_view().size());

    // deletion request submitted for stale keys 2,3
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        ASSERT_EQ(1u, executor->tasks_.size());
    }

    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        executor->tasks_.clear();
    }
    dsm->storage_map_["nfs_01"] = original;
}

TEST_F(CacheManagerTest, TestFilterWriteCache_StaleVsNonEmptyBlockMask) {
    // verify that the block mask uses exists_results (data existence)
    // rather than the old !location_maps[i].empty() check; a location
    // map can be non-empty (has CLS_SERVING metadata) yet the data is
    // stale, so it should be marked false in the mask
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    // write keys {1,2,3,4} and finish as CLS_SERVING
    std::vector<std::int64_t> write_keys{1, 2, 3, 4};
    auto [ec1, swci1] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", write_keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    {
        BlockMask bm = static_cast<std::size_t>(4);
        ASSERT_EQ(
            EC_OK,
            cache_manager_->FinishWriteCache(request_context_.get(), "test_instance", swci1.write_session_id(), bm));
    }

    // install interceptor: keys 1,3 valid; keys 2,4 stale
    auto dsm = registry_manager_->data_storage_manager_;
    auto original = dsm->storage_map_["nfs_01"];
    std::atomic<int> call_idx{0};
    dsm->storage_map_["nfs_01"] =
        std::make_shared<MightExistInterceptor>(original, [&call_idx](const std::vector<DataStorageUri> &uris) {
            int idx = call_idx.fetch_add(1);
            if (idx == 0 || idx == 2) {
                // keys 1 and 3: valid
                return std::vector<bool>(uris.size(), true);
            }
            // keys 2 and 4: stale
            return std::vector<bool>(uris.size(), false);
        });

    // StartWriteCache with {1, 2, 3, 4}; re-request same keys
    std::vector<std::int64_t> keys{1, 2, 3, 4};
    auto [ec2, swci2] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec2);

    // stale entries produce a non-contiguous pattern → BlockMaskVector
    auto mask = std::get<BlockMaskVector>(swci2.block_mask());
    ASSERT_EQ(4u, mask.size());
    ASSERT_TRUE(mask[0]);  // key 1: valid data
    ASSERT_FALSE(mask[1]); // key 2: stale → needs rewrite
    ASSERT_TRUE(mask[2]);  // key 3: valid data
    ASSERT_FALSE(mask[3]); // key 4: stale → needs rewrite
    // 2 new write locations for stale keys 2 and 4
    ASSERT_EQ(2u, swci2.locations().cache_locations_view().size());

    dsm->storage_map_["nfs_01"] = original;
}

TEST_F(CacheManagerTest, TestStartWriteCacheSpecGroupDedup) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    std::vector<LocationSpecInfo> location_spec_infos = {
        LocationSpecInfo("tp0_F0", 512),
        LocationSpecInfo("tp1_F0", 512),
        LocationSpecInfo("tp0_L1", 512),
        LocationSpecInfo("tp1_L1", 512),
    };
    std::vector<LocationSpecGroup> location_spec_groups = {
        LocationSpecGroup("F0", {"tp0_F0", "tp1_F0"}),
        LocationSpecGroup("L1", {"tp0_L1", "tp1_L1"}),
    };
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_dedup",
                                               64,
                                               location_spec_infos,
                                               createModelDeployment(),
                                               location_spec_groups));
    std::vector<int64_t> keys{1, 2, 3};

    // First write: F0 group
    std::string write_session_id_1;
    {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_dedup", keys, {}, {"F0", "F0", "F0"}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        write_session_id_1 = info.write_session_id();
        ASSERT_EQ(0, std::get<BlockMaskOffset>(info.block_mask()));
        const auto &views = info.locations().cache_locations_view();
        ASSERT_EQ(3, views.size());
        for (size_t i = 0; i < views.size(); ++i) {
            ASSERT_EQ(2, views[i].spec_size()) << "F0 group should have 2 specs at key index " << i;
        }
    }
    // Finish first write successfully
    {
        BlockMask block_mask = static_cast<size_t>(3); // all 3 blocks succeed
        auto ec =
            cache_manager_->FinishWriteCache(request_context_.get(), "test_dedup", write_session_id_1, block_mask);
        ASSERT_EQ(EC_OK, ec);
    }

    // Second write: L1 group for same keys → should NOT be deduped
    std::string write_session_id_2;
    {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_dedup", keys, {}, {"L1", "L1", "L1"}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        write_session_id_2 = info.write_session_id();
        ASSERT_EQ(0, std::get<BlockMaskOffset>(info.block_mask()))
            << "L1 write should not be deduped by existing F0 locations";
        const auto &views = info.locations().cache_locations_view();
        ASSERT_EQ(3, views.size());
        for (size_t i = 0; i < views.size(); ++i) {
            ASSERT_EQ(2, views[i].spec_size()) << "L1 group should have 2 specs at key index " << i;
        }
    }
    // Finish second write
    {
        BlockMask block_mask = static_cast<size_t>(3);
        auto ec =
            cache_manager_->FinishWriteCache(request_context_.get(), "test_dedup", write_session_id_2, block_mask);
        ASSERT_EQ(EC_OK, ec);
    }

    // Third write: F0 group again → should be fully deduped
    {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_dedup", keys, {}, {"F0", "F0", "F0"}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(3, std::get<BlockMaskOffset>(info.block_mask())) << "F0 re-write should be fully deduped";
        ASSERT_EQ(0, info.locations().cache_locations_view().size());
    }
}

TEST_F(CacheManagerTest, TestWriteThenReadRoundTripWithSpecGroups) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    std::vector<LocationSpecInfo> location_spec_infos = {
        LocationSpecInfo("tp0_F0", 512),
        LocationSpecInfo("tp1_F0", 512),
        LocationSpecInfo("tp0_L1", 512),
        LocationSpecInfo("tp1_L1", 512),
    };
    std::vector<LocationSpecGroup> location_spec_groups = {
        LocationSpecGroup("F0", {"tp0_F0", "tp1_F0"}),
        LocationSpecGroup("L1", {"tp0_L1", "tp1_L1"}),
    };
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_roundtrip",
                                               64,
                                               location_spec_infos,
                                               createModelDeployment(),
                                               location_spec_groups));
    std::vector<int64_t> keys{100, 200, 300};

    // Write F0 group
    {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_roundtrip", keys, {}, {"F0", "F0", "F0"}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        BlockMask block_mask = static_cast<size_t>(3);
        ec = cache_manager_->FinishWriteCache(
            request_context_.get(), "test_roundtrip", info.write_session_id(), block_mask);
        ASSERT_EQ(EC_OK, ec);
    }

    // Write L1 group
    {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_roundtrip", keys, {}, {"L1", "L1", "L1"}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        BlockMask block_mask = static_cast<size_t>(3);
        ec = cache_manager_->FinishWriteCache(
            request_context_.get(), "test_roundtrip", info.write_session_id(), block_mask);
        ASSERT_EQ(EC_OK, ec);
    }

    // Read back via QT_PREFIX_MATCH
    {
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                      "test_roundtrip",
                                                                      CacheManager::QueryType::QT_PREFIX_MATCH,
                                                                      keys,
                                                                      {},
                                                                      block_mask,
                                                                      0,
                                                                      {});
        ASSERT_EQ(EC_OK, ec);
        const auto &views = cache_locations.cache_locations_view();
        ASSERT_EQ(3, views.size());
        for (size_t i = 0; i < views.size(); ++i) {
            // Both F0 and L1 groups are on the same storage type (NFS),
            // so specs from both locations are merged → 4 specs total
            ASSERT_EQ(4, views[i].location_specs().size())
                << "key index " << i << " should have 4 specs merged from same storage type";
            for (const auto &spec : views[i].location_specs()) {
                EXPECT_FALSE(spec.uri().empty());
            }
        }
    }

    // Read with spec name filter: only F0 specs
    {
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                      "test_roundtrip",
                                                                      CacheManager::QueryType::QT_PREFIX_MATCH,
                                                                      keys,
                                                                      {},
                                                                      block_mask,
                                                                      0,
                                                                      {"tp0_F0", "tp1_F0"});
        ASSERT_EQ(EC_OK, ec);
        const auto &views = cache_locations.cache_locations_view();
        ASSERT_EQ(3, views.size());
        for (size_t i = 0; i < views.size(); ++i) {
            // All specs come from NFS (same type), so filtering by F0 spec names
            // should always yield 2 specs
            ASSERT_EQ(2, views[i].location_specs().size())
                << "key index " << i << " should have 2 F0 specs after filtering";
        }
    }
}

TEST_F(CacheManagerTest, TestDoRecoverAfterCleanup) {
    // Cleanup then recover
    ASSERT_EQ(EC_OK, cache_manager_->DoCleanup());
    ASSERT_EQ(EC_OK, cache_manager_->DoRecoverOnce());

    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_TRUE(meta_searcher);
    ASSERT_TRUE(meta_searcher->meta_indexer_);
    ASSERT_EQ("test_instance", meta_searcher->meta_indexer_->instance_id_);

    // Call again - should be idempotent
    ASSERT_EQ(EC_OK, cache_manager_->DoRecoverOnce());
    meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_TRUE(meta_searcher);
    ASSERT_EQ("test_instance", meta_searcher->meta_indexer_->instance_id_);
}

TEST_F(CacheManagerTest, TestDoRecoverOnceWithRegistryPartialFailureThenFix) {
    // Scenario:
    // 1. RegistryManager has instance_group + test_instance recovered, but a second instance
    //    "missing_instance" is expected but not in instance_infos_ (simulates partial recover failure)
    // 2. RegistryManager::IsRecoverComplete() returns false
    // 3. CacheManager::DoRecoverOnce() succeeds for test_instance but overall returns ERROR
    // 4. "Fix" RegistryManager by adding missing_instance and marking recover complete
    // 5. CacheManager::DoRecoverOnce() now succeeds and missing_instance gets its MetaSearcher

    // Cleanup to start fresh
    ASSERT_EQ(EC_OK, cache_manager_->DoCleanup());

    // Simulate RegistryManager partial failure: recover_complete_ is false
    // test_instance is already in instance_infos_ (from SetUp), but mark as incomplete
    registry_manager_->recover_complete_.store(false);

    // CacheManager DoRecoverOnce - should return ERROR because RegistryManager is incomplete
    auto ec = cache_manager_->DoRecoverOnce();
    ASSERT_EQ(EC_ERROR, ec);

    // test_instance MetaSearcher should still have been created (partial progress is retained)
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_TRUE(meta_searcher);
    ASSERT_EQ("test_instance", meta_searcher->meta_indexer_->instance_id_);

    // "missing_instance" doesn't exist yet - no MetaSearcher
    meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("missing_instance");
    ASSERT_FALSE(meta_searcher);

    // Now "fix" RegistryManager: add missing_instance to instance_infos_ and mark complete
    auto missing_instance_info = std::make_shared<InstanceInfo>(
        "test_quota_group", "default", "missing_instance", 64, createLocationSpecInfos(), createModelDeployment());
    registry_manager_->instance_infos_["missing_instance"] = missing_instance_info;
    registry_manager_->recover_complete_.store(true);

    // CacheManager DoRecoverOnce - should now succeed
    ec = cache_manager_->DoRecoverOnce();
    ASSERT_EQ(EC_OK, ec);

    // Both instances should have MetaSearcher
    meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_TRUE(meta_searcher);
    ASSERT_EQ("test_instance", meta_searcher->meta_indexer_->instance_id_);

    meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("missing_instance");
    ASSERT_TRUE(meta_searcher);
    ASSERT_EQ("missing_instance", meta_searcher->meta_indexer_->instance_id_);
}

TEST_F(CacheManagerTest, TestRecoverRetryLoopLifecycle) {
    // StopRecoverRetryLoop should be safe without active thread
    cache_manager_->StopRecoverRetryLoop();
    cache_manager_->StopRecoverRetryLoop(); // double-stop should not crash

    // StartRecoverRetryLoop + StopRecoverRetryLoop
    cache_manager_->StartRecoverRetryLoop();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    cache_manager_->StopRecoverRetryLoop(); // should join within ~100ms

    // DoCleanup should stop retry thread
    cache_manager_->StartRecoverRetryLoop();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ASSERT_EQ(EC_OK, cache_manager_->DoCleanup());
}

} // namespace kv_cache_manager
