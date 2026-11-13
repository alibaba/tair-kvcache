#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/meta_searcher_manager.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;

namespace {

class SearcherInitializationGate {
public:
    void Enter() {
        std::unique_lock lock(mutex_);
        ++entered_;
        cv_.notify_all();
        cv_.wait(lock, [this] { return released_; });
    }

    bool WaitForEntered(size_t count) {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(5), [this, count] { return entered_ >= count; });
    }

    void Release() {
        std::lock_guard lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t entered_ = 0;
    bool released_ = false;
};

class ControlledSearchIndexerManager : public MetaIndexerManager {
public:
    std::atomic<size_t> init_count{0};
    std::function<ErrorCode(const std::string &)> initialize;

protected:
    ErrorCode InitMetaIndexer(const std::shared_ptr<MetaIndexer> &indexer,
                              const std::string &instance_id,
                              const std::shared_ptr<MetaIndexerConfig> &config) override {
        init_count.fetch_add(1);
        const auto ec = initialize ? initialize(instance_id) : EC_OK;
        return ec == EC_OK ? MetaIndexerManager::InitMetaIndexer(indexer, instance_id, config) : ec;
    }
};

CheckLocDataExistFunc dummy_check_loc_data_exist = [](const CacheLocation &) -> bool { return true; };
SubmitDelReqFunc dummy_submit_del_req = [](const std::vector<std::int64_t> &,
                                           const std::vector<std::vector<std::string>> &,
                                           const std::vector<std::vector<std::string>> &,
                                           bool) -> void {};
} // namespace

class MetaSearcherManagerTest : public TESTBASE {
public:
    void SetUp() override {
        registry_manager_ = std::make_shared<RegistryManager>("local://fake", std::make_shared<MetricsRegistry>());
        std::vector<LocationSpecInfo> location_spec_infos = {LocationSpecInfo("tp0", 512),
                                                             LocationSpecInfo("tp1", 512)};
        auto instance_info = std::make_shared<InstanceInfo>(
            "test_quota_group", "test_group", "test_instance", 64, location_spec_infos, ModelDeployment());
        registry_manager_->instance_infos_["test_instance"] = instance_info;
        auto instance_group = std::make_shared<InstanceGroup>();
        registry_manager_->instance_group_configs_["test_group"] = instance_group;
        auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
        instance_group->cache_config_ = std::make_shared<CacheConfig>();
        instance_group->cache_config_->meta_indexer_config_ = meta_indexer_config;
        auto backend_config = std::make_shared<MetaStorageBackendConfig>();
        backend_config->storage_type_ = META_LOCAL_BACKEND_TYPE_STR;
        meta_indexer_config->meta_storage_backend_config_ = backend_config;
        meta_searcher_manager_ =
            std::make_unique<MetaSearcherManager>(registry_manager_, std::make_shared<MetaIndexerManager>());
    }

    void TearDown() override {}

    std::shared_ptr<ControlledSearchIndexerManager> UseControlledIndexerManager() {
        auto indexers = std::make_shared<ControlledSearchIndexerManager>();
        meta_searcher_manager_ = std::make_unique<MetaSearcherManager>(registry_manager_, indexers);
        return indexers;
    }

    std::shared_ptr<MetaSearcher> CreateSearcher(const std::string &id) {
        RequestContext context("create_" + id);
        return meta_searcher_manager_->TryCreateMetaSearcher(
            &context, id, dummy_check_loc_data_exist, dummy_submit_del_req);
    }

private:
    std::shared_ptr<RegistryManager> registry_manager_;
    std::unique_ptr<MetaSearcherManager> meta_searcher_manager_;
};

TEST_F(MetaSearcherManagerTest, TestDoCleanup) {
    // 1. 创建 MetaSearcher
    std::shared_ptr<RequestContext> request_context(new RequestContext("test_trace_id"));
    auto searcher = meta_searcher_manager_->TryCreateMetaSearcher(
        request_context.get(), "test_instance", dummy_check_loc_data_exist, dummy_submit_del_req);
    ASSERT_NE(searcher, nullptr);

    // 验证可以获取到创建的 MetaSearcher
    auto retrieved_searcher = meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_EQ(retrieved_searcher, searcher);

    // 2. 调用 DoCleanup
    meta_searcher_manager_->DoCleanup();

    // 3. 验证清理后获取不到 MetaSearcher
    auto after_cleanup_searcher = meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_EQ(after_cleanup_searcher, nullptr);

    // 4. 再次调用 DoCleanup 应该正常工作
    meta_searcher_manager_->DoCleanup();

    // 5. 清理掉MetaIndexer
    meta_searcher_manager_->meta_indexer_manager_->DoCleanup();

    // 6. 重新创建 MetaSearcher
    auto new_searcher = meta_searcher_manager_->TryCreateMetaSearcher(
        request_context.get(), "test_instance", dummy_check_loc_data_exist, dummy_submit_del_req);
    ASSERT_NE(new_searcher, nullptr);

    // 7. 验证可以获取到新创建的 MetaSearcher
    retrieved_searcher = meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_EQ(retrieved_searcher, new_searcher);
}

TEST_F(MetaSearcherManagerTest, DifferentInstancesInitializeConcurrentlyAndReadersStayAvailable) {
    auto indexers = UseControlledIndexerManager();
    SearcherInitializationGate gate;
    indexers->initialize = [&](const std::string &) {
        gate.Enter();
        return EC_OK;
    };
    registry_manager_->instance_infos_["second"] = std::make_shared<InstanceInfo>(
        "test_quota_group", "test_group", "second", 64, std::vector<LocationSpecInfo>{}, ModelDeployment());
    auto first = std::async(std::launch::async, [&] { return CreateSearcher("test_instance"); });
    const bool first_entered = gate.WaitForEntered(1);
    auto second = std::async(std::launch::async, [&] { return CreateSearcher("second"); });
    const bool both_entered = gate.WaitForEntered(2);
    auto read =
        std::async(std::launch::async, [&] { return meta_searcher_manager_->GetMetaSearcher("test_instance"); });
    const auto read_status = read.wait_for(std::chrono::seconds(1));
    gate.Release();
    EXPECT_TRUE(first_entered);
    EXPECT_TRUE(both_entered);
    EXPECT_EQ(std::future_status::ready, read_status);
    EXPECT_EQ(nullptr, read.get());
    EXPECT_NE(nullptr, first.get());
    EXPECT_NE(nullptr, second.get());
    EXPECT_EQ(2, indexers->init_count.load());
}

TEST_F(MetaSearcherManagerTest, SameInstanceInitializesOnceAndReturnsSameSearcher) {
    auto indexers = UseControlledIndexerManager();
    SearcherInitializationGate gate;
    indexers->initialize = [&](const std::string &) {
        gate.Enter();
        return EC_OK;
    };
    auto first = std::async(std::launch::async, [&] { return CreateSearcher("test_instance"); });
    const bool entered = gate.WaitForEntered(1);
    auto second = std::async(std::launch::async, [&] { return CreateSearcher("test_instance"); });
    const auto second_status = second.wait_for(std::chrono::milliseconds(50));
    EXPECT_EQ(nullptr, meta_searcher_manager_->GetMetaSearcher("test_instance"));
    gate.Release();
    EXPECT_TRUE(entered);
    EXPECT_EQ(std::future_status::timeout, second_status);
    auto first_searcher = first.get();
    EXPECT_NE(nullptr, first_searcher);
    EXPECT_EQ(first_searcher, second.get());
    EXPECT_EQ(1, indexers->init_count.load());
}

TEST_F(MetaSearcherManagerTest, CleanupWaitsForInFlightInitialization) {
    auto indexers = UseControlledIndexerManager();
    SearcherInitializationGate gate;
    indexers->initialize = [&](const std::string &) {
        gate.Enter();
        return EC_OK;
    };
    auto create = std::async(std::launch::async, [&] { return CreateSearcher("test_instance"); });
    const bool entered = gate.WaitForEntered(1);
    std::promise<void> cleanup_started;
    auto cleanup = std::async(std::launch::async, [&] {
        cleanup_started.set_value();
        meta_searcher_manager_->DoCleanup();
    });
    cleanup_started.get_future().wait();
    const auto cleanup_status = cleanup.wait_for(std::chrono::milliseconds(50));
    gate.Release();
    EXPECT_TRUE(entered);
    EXPECT_EQ(std::future_status::timeout, cleanup_status);
    EXPECT_NE(nullptr, create.get());
    cleanup.get();
    EXPECT_EQ(nullptr, meta_searcher_manager_->GetMetaSearcher("test_instance"));
}

TEST_F(MetaSearcherManagerTest, ExtractWaitsForInFlightInitialization) {
    auto indexers = UseControlledIndexerManager();
    SearcherInitializationGate gate;
    indexers->initialize = [&](const std::string &) {
        gate.Enter();
        return EC_OK;
    };
    auto create = std::async(std::launch::async, [&] { return CreateSearcher("test_instance"); });
    const bool entered = gate.WaitForEntered(1);
    std::promise<void> extract_started;
    auto extract = std::async(std::launch::async, [&] {
        extract_started.set_value();
        return meta_searcher_manager_->ExtractMetaSearcher("test_instance");
    });
    extract_started.get_future().wait();
    const auto extract_status = extract.wait_for(std::chrono::milliseconds(50));
    gate.Release();
    EXPECT_TRUE(entered);
    EXPECT_EQ(std::future_status::timeout, extract_status);
    auto created = create.get();
    EXPECT_NE(nullptr, created);
    EXPECT_EQ(created, extract.get());
    EXPECT_EQ(nullptr, meta_searcher_manager_->GetMetaSearcher("test_instance"));
}

TEST_F(MetaSearcherManagerTest, FailedInitializationDoesNotPublishAndCanRetry) {
    auto indexers = UseControlledIndexerManager();
    indexers->initialize = [](const std::string &) { return EC_ERROR; };
    EXPECT_EQ(nullptr, CreateSearcher("test_instance"));
    EXPECT_EQ(nullptr, meta_searcher_manager_->GetMetaSearcher("test_instance"));
    EXPECT_EQ(nullptr, indexers->GetMetaIndexer("test_instance"));
    indexers->initialize = [](const std::string &) { return EC_OK; };
    EXPECT_NE(nullptr, CreateSearcher("test_instance"));
    EXPECT_EQ(2, indexers->init_count.load());
}

TEST_F(MetaSearcherManagerTest, SearcherCleanupReusesInitializedIndexer) {
    auto indexers = UseControlledIndexerManager();
    auto original = CreateSearcher("test_instance");
    ASSERT_NE(nullptr, original);
    auto indexer = indexers->GetMetaIndexer("test_instance");
    ASSERT_NE(nullptr, indexer);
    ASSERT_NE(nullptr, indexer->backend_manager_);
    EXPECT_EQ(indexer, original->meta_indexer_);
    EXPECT_EQ(1, indexers->init_count.load());

    meta_searcher_manager_->DoCleanup();
    EXPECT_EQ(nullptr, meta_searcher_manager_->GetMetaSearcher("test_instance"));
    EXPECT_EQ(indexer, indexers->GetMetaIndexer("test_instance"));
    EXPECT_EQ(1, indexers->init_count.load());

    auto searcher = CreateSearcher("test_instance");
    ASSERT_NE(nullptr, searcher);
    EXPECT_NE(original, searcher);
    EXPECT_EQ(indexer, searcher->meta_indexer_);
    EXPECT_EQ(indexer, indexers->GetMetaIndexer("test_instance"));
    EXPECT_EQ(1, indexers->init_count.load());
}

TEST_F(MetaSearcherManagerTest, RemovedInstancesDoNotAccumulateMutexEntries) {
    auto indexers = UseControlledIndexerManager();
    for (int i = 0; i < 16; ++i) {
        const auto id = "removed_instance_" + std::to_string(i);
        registry_manager_->instance_infos_[id] = std::make_shared<InstanceInfo>(
            "test_quota_group", "test_group", id, 64, std::vector<LocationSpecInfo>{}, ModelDeployment());
        auto searcher = CreateSearcher(id);
        ASSERT_NE(nullptr, searcher);
        auto indexer = indexers->GetMetaIndexer(id);
        ASSERT_NE(nullptr, indexer);
        EXPECT_EQ(searcher, meta_searcher_manager_->ExtractMetaSearcher(id));
        EXPECT_EQ(indexer, indexers->ExtractMetaIndexer(id));
        registry_manager_->instance_infos_.erase(id);
    }
    EXPECT_LE(meta_searcher_manager_->instance_mutexes_.size(), 1);
    EXPECT_LE(indexers->instance_mutexes_.size(), 1);
}
