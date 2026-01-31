#include <thread>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/meta/meta_indexer.h"
using namespace kv_cache_manager;

namespace {
// Helper class to create test data
class MetaSearcherRedisTestHelper {
public:
    static LocationSpec CreateLocationSpec(const std::string &name = "", const std::string &uri = "") {
        LocationSpec spec(name, uri);
        return spec;
    }

    static CacheLocation CreateCacheLocation(DataStorageType type = DataStorageType::DATA_STORAGE_TYPE_NFS,
                                             size_t spec_size = 1,
                                             const std::vector<LocationSpec> &specs = {}) {
        return CacheLocation(type, spec_size, specs);
    }

    static std::vector<LocationSpec> CreateDefaultLocationSpecs() {
        LocationSpec spec = CreateLocationSpec();
        return {spec};
    }
};
} // namespace

class MetaSearcherRealServiceTest : public TESTBASE {
public:
    void SetUp() override {
        meta_indexer_ = CreateMetaIndexer();
        meta_searcher_ = std::make_shared<MetaSearcher>(meta_indexer_);
        request_context_.reset(new RequestContext("fake_trace_id"));
    }

    void TearDown() override {}

    std::shared_ptr<MetaStorageBackendConfig> ConstructMetaStorageBackendConfig() {
        auto meta_storage_backend_config = std::make_shared<MetaStorageBackendConfig>();
        meta_storage_backend_config->SetStorageType(META_REDIS_BACKEND_TYPE_STR);
        meta_storage_backend_config->SetStorageUri("redis://@localhost:6379/");
        return meta_storage_backend_config;
    }

    std::shared_ptr<MetaIndexer> CreateMetaIndexer() {
        auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
        auto backend_config = ConstructMetaStorageBackendConfig();
        meta_indexer_config->SetMetaStorageBackendConfig(backend_config);
        meta_indexer_config->SetMutexShardNum(32);
        meta_indexer_config->SetMaxKeyCount(10000);
        auto indexer = std::make_shared<MetaIndexer>();
        auto metaCachePolicyConfig = std::make_shared<MetaCachePolicyConfig>();
        metaCachePolicyConfig->SetCapacity(10000);
        meta_indexer_config->SetMetaCachePolicyConfig(metaCachePolicyConfig);
        auto ec = indexer->Init(/*instance_id*/ "meta_searcher_real_service_test", meta_indexer_config);
        if (ec != ErrorCode::EC_OK) {
            KVCM_LOG_ERROR("Init meta indexer failed");
            return nullptr;
        }
        return indexer;
    }

    std::shared_ptr<MetaIndexer> meta_indexer_;
    std::shared_ptr<MetaSearcher> meta_searcher_;
    std::shared_ptr<RequestContext> request_context_;
    StaticWeightSLPolicy policy_;
};

TEST_F(MetaSearcherRealServiceTest, TestBatchAddAndGetLocation) {
    // 准备测试数据
    MetaSearcher::KeyVector keys = {1, 2, 3};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherRedisTestHelper::CreateDefaultLocationSpecs();
    CacheLocation location1 =
        MetaSearcherRedisTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocation location2 =
        MetaSearcherRedisTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
    CacheLocation location3 = MetaSearcherRedisTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

    CacheLocationVector locations = {location1, location2, location3};

    // 调用BatchAddLocation
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);

    // 验证结果
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    // 验证添加的位置信息可以被检索到
    std::vector<CacheLocationMap> out_location_maps;
    BlockMask mask; // 空mask，不跳过任何元素

    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 3);

    for (const auto &location_map : out_location_maps) {
        EXPECT_FALSE(location_map.empty());
        // 每个map应该只有一个location（我们刚添加的）
        EXPECT_EQ(location_map.size(), 1);
    }
}

TEST_F(MetaSearcherRealServiceTest, TestBatchUpdateLocationStatus) {
    // 首先添加一些测试数据
    MetaSearcher::KeyVector keys = {1000, 2000, 3000};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherRedisTestHelper::CreateDefaultLocationSpecs();
    CacheLocation location1 =
        MetaSearcherRedisTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocation location2 =
        MetaSearcherRedisTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
    CacheLocation location3 = MetaSearcherRedisTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

    CacheLocationVector locations = {location1, location2, location3};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    // 准备更新状态的数据
    std::vector<CacheLocationStatus> new_statuses = {CLS_SERVING, CLS_DELETING, CLS_NEW};

    // 构建批量任务，每个key对应一个任务
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> batch_tasks;
    batch_tasks.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        tasks.push_back({out_location_ids[i], new_statuses[i]});
        batch_tasks.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> out_batch_results;
    // 调用BatchUpdateLocationStatus
    ec = meta_searcher_->BatchUpdateLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);

    // 验证状态已更新
    std::vector<CacheLocationMap> out_location_maps;
    BlockMask mask; // 空mask，不跳过任何元素

    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 3);

    for (size_t i = 0; i < out_location_maps.size(); i++) {
        const auto &location_map = out_location_maps[i];
        EXPECT_FALSE(location_map.empty());
        EXPECT_EQ(location_map.size(), 1);

        // 验证location信息
        auto it = location_map.find(out_location_ids[i]);
        EXPECT_NE(it, location_map.end());

        if (it != location_map.end()) {
            const CacheLocation &location = it->second;
            EXPECT_EQ(location.id(), out_location_ids[i]);
            EXPECT_EQ(location.status(), new_statuses[i]); // 状态应已更新
            EXPECT_EQ(location.type(), locations[i].type());
        }
    }
}

TEST_F(MetaSearcherRealServiceTest, TestPrefixMatchWithServingStatus) {
    // 首先添加一些测试数据
    MetaSearcher::KeyVector keys = {10, 20, 30};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherRedisTestHelper::CreateDefaultLocationSpecs();
    CacheLocation location1 =
        MetaSearcherRedisTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocation location2 =
        MetaSearcherRedisTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
    CacheLocation location3 = MetaSearcherRedisTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

    CacheLocationVector locations = {location1, location2, location3};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    // 将状态更新为SERVING以便可以被PrefixMatch找到
    std::vector<CacheLocationStatus> new_status(out_location_ids.size(), CacheLocationStatus::CLS_SERVING);

    // 构建批量任务，每个key对应一个任务
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> batch_tasks;
    batch_tasks.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        tasks.push_back({out_location_ids[i], new_status[i]});
        batch_tasks.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> out_batch_results;
    ec = meta_searcher_->BatchUpdateLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);

    // 测试PrefixMatch
    CacheLocationVector out_locations;
    BlockMask mask; // 空mask，不跳过任何元素

    ec = meta_searcher_->PrefixMatch(request_context_.get(), keys, mask, out_locations, &policy_);

    // 验证结果
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_locations.size(), 3);

    // 验证返回的locations与添加的locations匹配
    for (size_t i = 0; i < out_locations.size(); i++) {
        EXPECT_EQ(out_locations[i].id(), out_location_ids[i]);
        EXPECT_EQ(out_locations[i].status(), CLS_SERVING);
        EXPECT_EQ(out_locations[i].type(), locations[i].type());
    }
}

TEST_F(MetaSearcherRealServiceTest, TestConcurrentOperations) {
    const int NUM_THREADS = 10;
    const int OPERATIONS_PER_THREAD = 10;

    auto worker = [this](int thread_id) {
        for (int i = 0; i < OPERATIONS_PER_THREAD; ++i) {
            int base_key = thread_id * OPERATIONS_PER_THREAD + i;
            MetaSearcher::KeyVector keys = {base_key, base_key + 1000, base_key + 2000};

            // 创建CacheLocation对象
            auto location_specs = MetaSearcherRedisTestHelper::CreateDefaultLocationSpecs();
            CacheLocation location1 = MetaSearcherRedisTestHelper::CreateCacheLocation(
                DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
            CacheLocation location2 = MetaSearcherRedisTestHelper::CreateCacheLocation(
                DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
            CacheLocation location3 = MetaSearcherRedisTestHelper::CreateCacheLocation(
                DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

            CacheLocationVector locations = {location1, location2, location3};

            // 添加位置信息
            std::vector<std::string> out_location_ids;
            ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
            ASSERT_EQ(ec, ErrorCode::EC_OK);
            ASSERT_EQ(out_location_ids.size(), 3);

            // 更新状态为SERVING
            std::vector<CacheLocationStatus> new_statuses = {CLS_SERVING, CLS_SERVING, CLS_SERVING};

            // 构建批量任务，每个key对应一个任务
            std::vector<std::vector<MetaSearcher::LocationUpdateTask>> batch_tasks;
            batch_tasks.reserve(keys.size());
            for (size_t i = 0; i < keys.size(); ++i) {
                std::vector<MetaSearcher::LocationUpdateTask> tasks;
                tasks.push_back({out_location_ids[i], new_statuses[i]});
                batch_tasks.push_back(tasks);
            }

            std::vector<std::vector<ErrorCode>> out_batch_results;
            ec =
                meta_searcher_->BatchUpdateLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
            ASSERT_EQ(ec, ErrorCode::EC_OK);

            // 测试PrefixMatch
            CacheLocationVector out_locations;
            BlockMask mask;
            ec = meta_searcher_->PrefixMatch(request_context_.get(), keys, mask, out_locations, &policy_);
            ASSERT_EQ(ec, ErrorCode::EC_OK);
            ASSERT_EQ(out_locations.size(), 3);
        }
    };

    // 创建并启动多个线程
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(worker, i);
    }

    // 等待所有线程完成
    for (auto &t : threads) {
        t.join();
    }
}

TEST_F(MetaSearcherRealServiceTest, TestBatchVsSequentialPerformance) {
    const size_t num_keys = 100;
    MetaSearcher::KeyVector keys;
    CacheLocationVector locations;

    // 准备测试数据
    for (size_t i = 0; i < num_keys; i++) {
        keys.push_back(10000 + i);

        auto location_specs = MetaSearcherRedisTestHelper::CreateDefaultLocationSpecs();
        CacheLocation location =
            MetaSearcherRedisTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
        locations.push_back(location);
    }

    // 批量添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);

    // 将所有位置更新为SERVING状态
    std::vector<CacheLocationStatus> statuses(out_location_ids.size(), CacheLocationStatus::CLS_SERVING);

    // 构建批量任务，每个key对应一个任务
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> batch_tasks;
    batch_tasks.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        tasks.push_back({out_location_ids[i], statuses[i]});
        batch_tasks.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> out_batch_results;
    ec = meta_searcher_->BatchUpdateLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);

    // 测试批量PrefixMatch性能
    auto batch_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 100; ++i) {
        CacheLocationVector out_locations;
        BlockMask mask; // 空mask，不跳过任何元素
        ec = meta_searcher_->PrefixMatch(request_context_.get(), keys, mask, out_locations, &policy_);
        // EXPECT_EQ(ec, ErrorCode::EC_OK);
        // EXPECT_EQ(out_locations.size(), num_keys);
    }
    auto batch_end = std::chrono::high_resolution_clock::now();
    auto batch_duration = std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start);

    KVCM_LOG_INFO("Batch PrefixMatch duration for 100 runs: %ld ms for %ld", batch_duration.count() / 1000, num_keys);
    KVCM_LOG_INFO("Average per run: %.2f ms", batch_duration.count() / 100000.0);
}
