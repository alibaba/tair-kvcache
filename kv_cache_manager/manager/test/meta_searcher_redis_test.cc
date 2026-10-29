#include <thread>

#include "kv_cache_manager/common/test/redis_test_environment.h"
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

CheckLocDataExistFunc dummy_check_loc_data_exist = [](const CacheLocation &) -> bool { return true; };
SubmitDelReqFunc dummy_submit_del_req = [](const std::vector<std::int64_t> &,
                                           const std::vector<std::vector<std::string>> &,
                                           const std::vector<std::vector<std::string>> &,
                                           bool) -> void {};

ErrorCode BatchAddLocationForTest(MetaSearcher *meta_searcher,
                                  RequestContext *request_context,
                                  const KeyVector &keys,
                                  const CacheLocationVector &locations,
                                  std::vector<std::string> &out_location_ids) {
    std::vector<MetaSearcher::AddLocationResult> results;
    const ErrorCode ec = meta_searcher->BatchAddLocation(request_context, keys, locations, results);
    out_location_ids.clear();
    out_location_ids.reserve(results.size());
    for (const auto &result : results) {
        out_location_ids.push_back(result.location_id);
    }
    return ec;
}
} // namespace

class MetaSearcherRealServiceTest : public TESTBASE {
public:
    void SetUp() override {
        meta_indexer_ = CreateMetaIndexer();
        meta_searcher_ =
            std::make_shared<MetaSearcher>(meta_indexer_, dummy_check_loc_data_exist, dummy_submit_del_req);
        request_context_.reset(new RequestContext("fake_trace_id"));
    }

    void TearDown() override {}

    std::shared_ptr<MetaStorageBackendConfig> ConstructMetaStorageBackendConfig() {
        auto meta_storage_backend_config = std::make_shared<MetaStorageBackendConfig>();
        meta_storage_backend_config->SetStorageType(META_REDIS_BACKEND_TYPE_STR);
        meta_storage_backend_config->SetStorageUri(
            redis_test::Uri("test_user:test_password", redis_test::kMetaSearcherDb));
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
    auto location1 = std::make_shared<const CacheLocation>(
        MetaSearcherRedisTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs));
    auto location2 = std::make_shared<const CacheLocation>(
        MetaSearcherRedisTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs));
    auto location3 = std::make_shared<const CacheLocation>(MetaSearcherRedisTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs));

    CacheLocationVector locations = {location1, location2, location3};

    // 调用BatchAddLocation
    std::vector<std::string> out_location_ids;
    ErrorCode ec =
        BatchAddLocationForTest(meta_searcher_.get(), request_context_.get(), keys, locations, out_location_ids);

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
    auto location1 = std::make_shared<const CacheLocation>(
        MetaSearcherRedisTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs));
    auto location2 = std::make_shared<const CacheLocation>(
        MetaSearcherRedisTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs));
    auto location3 = std::make_shared<const CacheLocation>(MetaSearcherRedisTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs));

    CacheLocationVector locations = {location1, location2, location3};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec =
        BatchAddLocationForTest(meta_searcher_.get(), request_context_.get(), keys, locations, out_location_ids);
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
            EXPECT_EQ(it->second->status(), new_statuses[i]); // 状态应已更新
        }
    }
}

TEST_F(MetaSearcherRealServiceTest, TestPrefixMatchWithServingStatus) {
    // 首先添加一些测试数据
    MetaSearcher::KeyVector keys = {10, 20, 30};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherRedisTestHelper::CreateDefaultLocationSpecs();
    auto location1 = std::make_shared<const CacheLocation>(
        MetaSearcherRedisTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs));
    auto location2 = std::make_shared<const CacheLocation>(
        MetaSearcherRedisTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs));
    auto location3 = std::make_shared<const CacheLocation>(MetaSearcherRedisTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs));

    CacheLocationVector locations = {location1, location2, location3};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec =
        BatchAddLocationForTest(meta_searcher_.get(), request_context_.get(), keys, locations, out_location_ids);
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

    // 验证返回的 locations（PrefixMatch 合并视图的 id 无单一元数据语义，不测 id）
    for (size_t i = 0; i < out_locations.size(); i++) {
        EXPECT_EQ(out_locations[i]->status(), CLS_SERVING);
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
            auto location1 = std::make_shared<const CacheLocation>(MetaSearcherRedisTestHelper::CreateCacheLocation(
                DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs));
            auto location2 = std::make_shared<const CacheLocation>(MetaSearcherRedisTestHelper::CreateCacheLocation(
                DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs));
            auto location3 = std::make_shared<const CacheLocation>(MetaSearcherRedisTestHelper::CreateCacheLocation(
                DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs));

            CacheLocationVector locations = {location1, location2, location3};

            // 添加位置信息
            std::vector<std::string> out_location_ids;
            ErrorCode ec = BatchAddLocationForTest(
                meta_searcher_.get(), request_context_.get(), keys, locations, out_location_ids);
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
        auto location = std::make_shared<const CacheLocation>(MetaSearcherRedisTestHelper::CreateCacheLocation(
            DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs));
        locations.push_back(location);
    }

    // 批量添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec =
        BatchAddLocationForTest(meta_searcher_.get(), request_context_.get(), keys, locations, out_location_ids);
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

class MetaSearcherDuplicateDeleteRedisTest : public MetaSearcherRealServiceTest,
                                             public ::testing::WithParamInterface<std::string> {
public:
    void SetUp() override {
        auto backend_config = ConstructMetaStorageBackendConfig();
        if (GetParam() != "Redis") {
            backend_config->SetStorageType(META_CACHED_BACKEND_TYPE_STR);
            const std::string persistent_type = GetParam() == "CachedAsyncRedis" ? "async_redis" : "redis";
            backend_config->SetStorageUri(redis_test::Uri("test_user:test_password",
                                                          redis_test::kMetaSearcherDb,
                                                          "cache_type=local&persistent_type=" + persistent_type +
                                                              "&capacity=16&num_shard_bits=2&async_queue_count=1"));
        }

        auto indexer_config = std::make_shared<MetaIndexerConfig>();
        indexer_config->SetMetaStorageBackendConfig(backend_config);
        indexer_config->SetMutexShardNum(32);
        indexer_config->SetMaxKeyCount(10000);
        auto cache_policy = std::make_shared<MetaCachePolicyConfig>();
        cache_policy->SetCapacity(16);
        indexer_config->SetMetaCachePolicyConfig(cache_policy);
        meta_indexer_ = std::make_shared<MetaIndexer>();
        // Isolate each mode and repeated test run without flushing the shared Redis database.
        const std::string instance_id = GetCurrentTestName() + "_" + std::to_string(TimestampUtil::GetCurrentTimeUs());
        ASSERT_EQ(EC_OK, meta_indexer_->Init(instance_id, indexer_config));
        meta_searcher_ =
            std::make_shared<MetaSearcher>(meta_indexer_, dummy_check_loc_data_exist, dummy_submit_del_req);
        request_context_ = std::make_shared<RequestContext>("duplicate_delete_redis");
    }

    void CheckDuplicateDeleteAccounting(bool compare_status) {
        constexpr KeyType healthy_key = 92001;
        constexpr KeyType dead_key = 92002;
        constexpr auto storage_type = DataStorageType::DATA_STORAGE_TYPE_NFS;
        const KeyVector seed_keys{healthy_key, dead_key};
        CacheLocationVector locations{
            std::make_shared<CacheLocation>(
                storage_type, 1, std::vector<LocationSpec>{LocationSpec("healthy", "file:///tmp/healthy?size=101")}),
            std::make_shared<CacheLocation>(
                storage_type, 1, std::vector<LocationSpec>{LocationSpec("dead", "file:///tmp/dead?size=17")}),
        };
        std::vector<std::string> ids;
        ASSERT_EQ(EC_OK,
                  BatchAddLocationForTest(meta_searcher_.get(), request_context_.get(), seed_keys, locations, ids));
        ASSERT_EQ(2u, ids.size());
        std::vector<std::vector<MetaSearcher::LocationUpdateTask>> status_tasks{{{ids[0], CLS_SERVING}},
                                                                                {{ids[1], CLS_DELETING}}};
        std::vector<std::vector<ErrorCode>> results;
        ASSERT_EQ(EC_OK,
                  meta_searcher_->BatchUpdateLocationStatus(request_context_.get(), seed_keys, status_tasks, results));
        ASSERT_EQ((std::vector<std::vector<ErrorCode>>{{EC_OK}, {EC_OK}}), results);
        ASSERT_TRUE(meta_indexer_->Sync(seed_keys));
        ASSERT_EQ(2u, meta_indexer_->GetKeyCount());
        ASSERT_EQ(118u, meta_indexer_->GetStorageUsage());
        ASSERT_EQ(118u, meta_indexer_->GetStorageUsageByType(storage_type));

        CacheLocationMapVector before;
        const auto before_result = meta_indexer_->GetLocationsFromPrimary(request_context_.get(), seed_keys, before);
        ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK}), before_result.error_codes);
        ASSERT_EQ(2u, before.size());
        ASSERT_EQ(1u, before[0].count(ids[0]));
        ASSERT_EQ(1u, before[1].count(ids[1]));
        const std::string healthy_snapshot = before[0].at(ids[0])->ToJsonString();
        const std::string dead_snapshot = before[1].at(ids[1])->ToJsonString();

        // Duplicate the target both within one row and across repeated keys.
        const KeyVector keys{dead_key, dead_key, dead_key};
        const LocationIdsPerKey location_ids{{ids[1], ids[1]}, {ids[1]}, {ids[1], ids[1]}};
        const std::vector<std::vector<MetaSearcher::LocationCADTask>> cad_tasks{
            {{ids[1], CLS_DELETING}, {ids[1], CLS_DELETING}},
            {{ids[1], CLS_DELETING}},
            {{ids[1], CLS_DELETING}, {ids[1], CLS_DELETING}},
        };
        const std::vector<std::vector<std::string>> expected_values{
            {dead_snapshot, dead_snapshot}, {dead_snapshot}, {dead_snapshot, dead_snapshot}};

        for (size_t attempt = 0; attempt < 2; ++attempt) {
            SCOPED_TRACE(attempt);
            if (compare_status) {
                ASSERT_EQ(EC_OK,
                          meta_searcher_->BatchCADLocationStatus(request_context_.get(), keys, cad_tasks, results));
            } else {
                ASSERT_EQ(EC_OK,
                          meta_searcher_->BatchDeleteLocations(
                              request_context_.get(), keys, location_ids, results, expected_values));
            }
            const ErrorCode expected_ec = attempt == 0 ? EC_OK : EC_NOENT;
            const std::vector<std::vector<ErrorCode>> expected_results{
                {expected_ec, expected_ec}, {expected_ec}, {expected_ec, expected_ec}};
            EXPECT_EQ(expected_results, results);
            ASSERT_TRUE(meta_indexer_->Sync(seed_keys));
            EXPECT_EQ(1u, meta_indexer_->GetKeyCount());
            EXPECT_EQ(101u, meta_indexer_->GetStorageUsage());
            EXPECT_EQ(101u, meta_indexer_->GetStorageUsageByType(storage_type));

            CacheLocationMapVector current;
            const auto current_result = meta_indexer_->GetLocations(request_context_.get(), seed_keys, current);
            ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_NOENT}), current_result.error_codes);
            ASSERT_EQ(2u, current.size());
            ASSERT_EQ(1u, current[0].size());
            ASSERT_EQ(1u, current[0].count(ids[0]));
            EXPECT_EQ(healthy_snapshot, current[0].at(ids[0])->ToJsonString());
            EXPECT_TRUE(current[1].empty());

            // A cache hit alone is insufficient: verify the real Redis state after the async barrier too.
            CacheLocationMapVector persistent;
            const auto persistent_result =
                meta_indexer_->GetLocationsFromPrimary(request_context_.get(), seed_keys, persistent);
            ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_NOENT}), persistent_result.error_codes);
            ASSERT_EQ(2u, persistent.size());
            ASSERT_EQ(1u, persistent[0].size());
            ASSERT_EQ(1u, persistent[0].count(ids[0]));
            EXPECT_EQ(healthy_snapshot, persistent[0].at(ids[0])->ToJsonString());
            EXPECT_TRUE(persistent[1].empty());
        }
    }
};

TEST_P(MetaSearcherDuplicateDeleteRedisTest, BatchCADLocationStatusDuplicateTargetsAccountOnce) {
    CheckDuplicateDeleteAccounting(true);
}

TEST_P(MetaSearcherDuplicateDeleteRedisTest, BatchDeleteLocationsDuplicateTargetsAccountOnce) {
    CheckDuplicateDeleteAccounting(false);
}

INSTANTIATE_TEST_SUITE_P(RealRedisBackends,
                         MetaSearcherDuplicateDeleteRedisTest,
                         ::testing::Values(std::string("Redis"),
                                           std::string("CachedRedis"),
                                           std::string("CachedAsyncRedis")),
                         [](const ::testing::TestParamInfo<std::string> &info) { return info.param; });
