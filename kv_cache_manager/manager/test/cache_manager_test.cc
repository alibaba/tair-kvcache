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
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace {
static const std::string default_storage_configs(
    "[{\"type\":\"file\",\"is_available\":true,\"global_unique_name\":\"nfs_01\",\"storage_spec\":{"
    "\"root_path\":\"/tmp/nfs/\",\"key_count_per_file\":8}}]");
} // namespace

namespace kv_cache_manager {

class CacheManagerTest : public TESTBASE {
public:
    void SetUp() override {
        cache_manager_ = createCacheManager();
        request_context_.reset(new RequestContext("fake_trace_id"));
    }

    void TearDown() override {}

    std::unique_ptr<CacheManager> createCacheManager() {
        std::shared_ptr<MetricsRegistry> metrics_registry = std::make_shared<MetricsRegistry>();
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
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
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

} // namespace kv_cache_manager
