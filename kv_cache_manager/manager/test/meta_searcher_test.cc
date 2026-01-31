#include <filesystem>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/meta/meta_indexer.h"

using namespace kv_cache_manager;

namespace {
// Helper class to create test data
class MetaSearcherTestHelper {
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

class MetaSearcherTest : public TESTBASE {
public:
    void SetUp() override {
        meta_indexer_ = CreateMetaIndexer();
        meta_searcher_ = std::make_shared<MetaSearcher>(meta_indexer_);
        request_context_ = std::make_shared<RequestContext>("test_trace_id");
    }

    void TearDown() override {}

    std::shared_ptr<MetaStorageBackendConfig> ConstructMetaStorageBackendConfig() {
        auto meta_storage_backend_config = std::make_shared<MetaStorageBackendConfig>();
        std::string local_path = GetPrivateTestRuntimeDataPath() + "meta_local_backend_file_1";
        meta_storage_backend_config->SetStorageUri("file://" + local_path);
        std::error_code ec;
        bool exists = std::filesystem::exists(local_path, ec);
        EXPECT_FALSE(ec) << local_path; // false means correct
        if (exists) {
            std::remove(local_path.c_str());
        }
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
        metaCachePolicyConfig->SetCapacity(0);
        meta_indexer_config->SetMetaCachePolicyConfig(metaCachePolicyConfig);
        auto ec = indexer->Init(/*instance_id*/ "test", meta_indexer_config);
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

TEST_F(MetaSearcherTest, TestBatchAddLocation) {
    // 准备测试数据
    MetaSearcher::KeyVector keys = {1, 2, 3};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocation location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocation location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
    CacheLocation location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

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

TEST_F(MetaSearcherTest, TestBatchAddLocation2) {
    // 准备测试数据
    MetaSearcher::KeyVector keys = {1, 2, 3};

    // 创建CacheLocation对象
    std::vector<LocationSpec> specs1(
        1, MetaSearcherTestHelper::CreateLocationSpec("nfs", "file:///tmp/test1/test.txt?offset=1&length=2&size=3"));
    CacheLocation location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, specs1);

    std::vector<LocationSpec> specs2(
        1, MetaSearcherTestHelper::CreateLocationSpec("hf3fs", "hf3fs:///tmp/test1/test.txt?offset=1&length=2&size=4"));
    CacheLocation location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, specs2);

    std::vector<LocationSpec> specs3(2,
                                     MetaSearcherTestHelper::CreateLocationSpec(
                                         "mooncake", "mooncake:///tmp/test1/test.txt?offset=1&length=2&size=5"));
    CacheLocation location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, specs3);

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

    EXPECT_EQ(
        3,
        meta_indexer_->storage_usage_array_[static_cast<std::uint8_t>(DataStorageType::DATA_STORAGE_TYPE_NFS)].load());
    EXPECT_EQ(4,
              meta_indexer_->storage_usage_array_[static_cast<std::uint8_t>(DataStorageType::DATA_STORAGE_TYPE_HF3FS)]
                  .load());
    EXPECT_EQ(
        10,
        meta_indexer_->storage_usage_array_[static_cast<std::uint8_t>(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE)]
            .load());
}

TEST_F(MetaSearcherTest, TestPrefixMatch) {
    // 首先添加一些测试数据
    MetaSearcher::KeyVector keys = {10, 20, 30};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocation location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocation location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
    CacheLocation location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

    CacheLocationVector locations = {location1, location2, location3};
    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    {
        // 改成serving之前match不到
        CacheLocationVector out_locations;
        BlockMask mask; // 空mask，不跳过任何元素

        ec = meta_searcher_->PrefixMatch(request_context_.get(), keys, mask, out_locations, &policy_);
        EXPECT_EQ(ec, ErrorCode::EC_OK);
        EXPECT_EQ(out_locations.size(), 0);
    }

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

    struct PrefixMatchTestData {
        MetaSearcher::KeyVector keys;
        size_t result_length;
    };
    std::vector<PrefixMatchTestData> test_datas;
    test_datas.push_back({{10, 20, 30}, 3});
    test_datas.push_back({{10, 20, 999}, 2});
    test_datas.push_back({{10, 999, 999}, 1});
    test_datas.push_back({{999, 999, 999}, 0});
    test_datas.push_back({{}, 0});
    test_datas.push_back({{10}, 1});
    test_datas.push_back({{10, 20}, 2});
    test_datas.push_back({{10, 999, 30}, 1});

    for (auto &test_data : test_datas) {

        // 测试PrefixMatch
        CacheLocationVector out_locations;
        BlockMask mask; // 空mask，不跳过任何元素

        ec = meta_searcher_->PrefixMatch(request_context_.get(), test_data.keys, mask, out_locations, &policy_);

        // 验证结果
        EXPECT_EQ(ec, ErrorCode::EC_OK);
        EXPECT_EQ(out_locations.size(), test_data.result_length);

        // 验证返回的locations与添加的locations匹配
        for (size_t i = 0; i < out_locations.size(); i++) {
            EXPECT_EQ(out_locations[i].id(), out_location_ids[i]);
            EXPECT_EQ(out_locations[i].status(), CLS_SERVING);
            EXPECT_EQ(out_locations[i].type(), locations[i].type());
        }
    }

    // 测试带mask的PrefixMatch
    std::vector<BlockMask> mask_vectors;
    mask_vectors.emplace_back(BlockMaskVector{true, false, false}); // 跳过第一个元素
    mask_vectors.emplace_back(BlockMaskOffset{1});                  // 跳过第一个元素

    for (auto &block_mask : mask_vectors) {
        CacheLocationVector out_locations;
        out_locations.clear();
        ec = meta_searcher_->PrefixMatch(request_context_.get(), keys, block_mask, out_locations, &policy_);

        // 验证结果 - 应该只返回后两个元素
        EXPECT_EQ(ec, ErrorCode::EC_OK);
        EXPECT_EQ(out_locations.size(), 2);

        // 验证返回的locations与添加的locations匹配（跳过第一个）
        for (size_t i = 0; i < out_locations.size(); i++) {
            EXPECT_EQ(out_locations[i].id(), out_location_ids[i + 1]);
            EXPECT_EQ(out_locations[i].status(), CLS_SERVING);
            EXPECT_EQ(out_locations[i].type(), locations[i + 1].type());
        }
    }
}

TEST_F(MetaSearcherTest, TestBatchGet) {
    // 首先添加一些测试数据
    MetaSearcher::KeyVector keys = {100, 200, 300};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocation location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocation location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
    CacheLocation location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

    CacheLocationVector locations = {location1, location2, location3};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    // 测试BatchGetLocation
    std::vector<CacheLocationMap> out_location_maps;
    BlockMask mask; // 空mask，不跳过任何元素

    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);

    // 验证结果
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 3);

    // 验证返回的location maps
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
            EXPECT_EQ(location.status(), CLS_WRITING);
            EXPECT_EQ(location.type(), locations[i].type());
        }
    }

    // 测试带mask的PrefixMatch
    std::vector<BlockMask> mask_vectors;
    mask_vectors.emplace_back(BlockMaskVector{false, true, true}); // 跳过后二个元素
    mask_vectors.emplace_back(BlockMaskOffset{2});                 // 跳过前两个元素

    for (auto &block_mask : mask_vectors) {
        out_location_maps.clear();
        ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, block_mask, out_location_maps);

        // 验证结果 - 应该返回1个元素
        EXPECT_EQ(ec, ErrorCode::EC_OK);
        EXPECT_EQ(out_location_maps.size(), 1);

        // 应该有数据
        for (size_t idx = 0; idx < 1; idx++) {
            EXPECT_FALSE(out_location_maps[idx].empty());
            EXPECT_EQ(out_location_maps[idx].size(), 1);
        }
    }
}

TEST_F(MetaSearcherTest, TestBatchUpdateLocationStatus) {
    // 首先添加一些测试数据
    MetaSearcher::KeyVector keys = {1000, 2000, 3000};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocation location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocation location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
    CacheLocation location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

    CacheLocationVector locations = {location1, location2, location3};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    // 验证初始状态为CLS_WRITING
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
            EXPECT_EQ(location.status(), CLS_WRITING); // 初始状态应为CLS_WRITING
            EXPECT_EQ(location.type(), locations[i].type());
        }
    }

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
    ec = meta_searcher_->BatchUpdateLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);

    // 验证状态已更新
    out_location_maps.clear();
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

    // 测试错误情况：key、location_id和status数量不匹配
    std::vector<std::string> mismatched_location_ids = {out_location_ids[0], out_location_ids[1]}; // 只有两个
    std::vector<CacheLocationStatus> mismatched_statuses = {CLS_SERVING, CLS_DELETING, CLS_NEW, CLS_WRITING}; // 四个

    // 测试错误情况：keys和batch_tasks大小不匹配
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> mismatched_batch_tasks;
    // 只为前两个keys创建任务，但keys有三个，应该返回EC_BADARGS
    for (size_t i = 0; i < 2; ++i) { // 只为前两个key创建任务
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        tasks.push_back({out_location_ids[i], new_statuses[i]});
        mismatched_batch_tasks.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> out_batch_results1;
    ec = meta_searcher_->BatchUpdateLocationStatus(
        request_context_.get(), keys, mismatched_batch_tasks, out_batch_results1);
    EXPECT_EQ(ec, ErrorCode::EC_BADARGS);

    // 为三个keys创建任务，但keys只有两个，应该返回EC_BADARGS
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> mismatched_batch_tasks2;
    for (size_t i = 0; i < 3; ++i) { // 为三个key创建任务
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        tasks.push_back({out_location_ids[i], new_statuses[i]});
        mismatched_batch_tasks2.push_back(tasks);
    }

    std::vector<MetaSearcher::KeyVector::value_type> mismatched_keys = {keys[0], keys[1]}; // 只有两个key
    std::vector<std::vector<ErrorCode>> out_batch_results2;
    ec = meta_searcher_->BatchUpdateLocationStatus(
        request_context_.get(), mismatched_keys, mismatched_batch_tasks2, out_batch_results2);
    EXPECT_EQ(ec, ErrorCode::EC_BADARGS);
}

TEST_F(MetaSearcherTest, TestBlockKeyWithMultipleLocations) {
    // 准备测试数据 - 使用相同的key添加多个location
    MetaSearcher::KeyVector keys = {12345}; // 只使用一个key

    // 创建多个不同的CacheLocation对象
    std::vector<LocationSpec> location_specs1 = {MetaSearcherTestHelper::CreateLocationSpec("tp0", "uri1")};
    std::vector<LocationSpec> location_specs2 = {MetaSearcherTestHelper::CreateLocationSpec("tp1", "uri2")};
    std::vector<LocationSpec> location_specs3 = {MetaSearcherTestHelper::CreateLocationSpec("tp2", "uri3")};

    CacheLocation location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs1);
    CacheLocation location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs2);
    CacheLocation location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs3);

    // 将三个location添加到同一个key
    CacheLocationVector locations1 = {location1};
    CacheLocationVector locations2 = {location2};
    CacheLocationVector locations3 = {location3};

    // 分别调用三次BatchAddLocation，为同一个key添加三个不同的location
    std::vector<std::string> out_location_ids1, out_location_ids2, out_location_ids3;

    ErrorCode ec1 = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations1, out_location_ids1);
    ErrorCode ec2 = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations2, out_location_ids2);
    ErrorCode ec3 = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations3, out_location_ids3);

    // 验证结果
    EXPECT_EQ(ec1, ErrorCode::EC_OK);
    EXPECT_EQ(ec2, ErrorCode::EC_OK);
    EXPECT_EQ(ec3, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids1.size(), 1);
    EXPECT_EQ(out_location_ids2.size(), 1);
    EXPECT_EQ(out_location_ids3.size(), 1);

    // 验证添加的位置信息可以被检索到
    std::vector<CacheLocationMap> out_location_maps;
    BlockMask mask; // 空mask，不跳过任何元素

    ErrorCode ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 1); // 只有一个key

    // 验证该key对应的location map包含三个location
    const auto &location_map = out_location_maps[0];
    EXPECT_EQ(location_map.size(), 3); // 应该有三个location

    // 验证三个location都存在且信息正确
    EXPECT_NE(location_map.find(out_location_ids1[0]), location_map.end());
    EXPECT_NE(location_map.find(out_location_ids2[0]), location_map.end());
    EXPECT_NE(location_map.find(out_location_ids3[0]), location_map.end());

    // 验证每个location的信息
    const CacheLocation &retrieved_location1 = location_map.at(out_location_ids1[0]);
    EXPECT_EQ(retrieved_location1.type(), DataStorageType::DATA_STORAGE_TYPE_NFS);
    EXPECT_EQ(retrieved_location1.spec_size(), 1);
    EXPECT_EQ(retrieved_location1.location_specs().size(), 1);

    const CacheLocation &retrieved_location2 = location_map.at(out_location_ids2[0]);
    EXPECT_EQ(retrieved_location2.type(), DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    EXPECT_EQ(retrieved_location2.spec_size(), 2);
    EXPECT_EQ(retrieved_location2.location_specs().size(), 1);

    const CacheLocation &retrieved_location3 = location_map.at(out_location_ids3[0]);
    EXPECT_EQ(retrieved_location3.type(), DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
    EXPECT_EQ(retrieved_location3.spec_size(), 3);
    EXPECT_EQ(retrieved_location3.location_specs().size(), 1);
}

TEST_F(MetaSearcherTest, TestBatchDeleteLocation) {
    // 首先添加一些测试数据
    MetaSearcher::KeyVector keys = {5000, 6000, 7000};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocation location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocation location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
    CacheLocation location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

    CacheLocationVector locations = {location1, location2, location3};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
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

    // 准备删除数据 - 删除前两个location
    std::vector<std::string> delete_location_ids = {out_location_ids[0], out_location_ids[1], "non_existent_id"};
    MetaSearcher::KeyVector delete_keys = {5000, 6000, 7000}; // 对应的keys

    // 调用BatchDeleteLocation
    std::vector<ErrorCode> delete_results;
    ec = meta_searcher_->BatchDeleteLocation(request_context_.get(), delete_keys, delete_location_ids, delete_results);

    // 验证结果
    EXPECT_EQ(ec, ErrorCode::EC_PARTIAL_OK);
    EXPECT_EQ(delete_results.size(), 3);
    EXPECT_EQ(delete_results[0], ErrorCode::EC_OK);    // 成功删除第一个
    EXPECT_EQ(delete_results[1], ErrorCode::EC_OK);    // 成功删除第二个
    EXPECT_EQ(delete_results[2], ErrorCode::EC_NOENT); // 删除不存在的location应返回EC_NOENT

    // 验证删除后的状态
    out_location_maps.clear();
    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 3);

    // 第一个key应该没有location了
    EXPECT_TRUE(out_location_maps[0].empty());

    // 第二个key应该没有location了
    EXPECT_TRUE(out_location_maps[1].empty());

    // 第三个key应该还有location
    EXPECT_FALSE(out_location_maps[2].empty());
    EXPECT_EQ(out_location_maps[2].size(), 1);

    // 验证第三个location仍然存在且信息正确
    auto it = out_location_maps[2].find(out_location_ids[2]);
    EXPECT_NE(it, out_location_maps[2].end());

    if (it != out_location_maps[2].end()) {
        const CacheLocation &location = it->second;
        EXPECT_EQ(location.id(), out_location_ids[2]);
        EXPECT_EQ(location.status(), CLS_WRITING);
        EXPECT_EQ(location.type(), locations[2].type());
    }

    // 测试错误情况：key和location_id数量不匹配
    MetaSearcher::KeyVector mismatched_keys = {5000, 6000}; // 只有两个
    std::vector<std::string> mismatched_location_ids = {
        out_location_ids[0], out_location_ids[1], out_location_ids[2]}; // 三个

    ec = meta_searcher_->BatchDeleteLocation(
        request_context_.get(), mismatched_keys, mismatched_location_ids, delete_results);
    EXPECT_EQ(ec, ErrorCode::EC_BADARGS);
}

TEST_F(MetaSearcherTest, TestBatchDeleteLocation2) {
    // 首先添加一些测试数据
    MetaSearcher::KeyVector keys = {5000, 6000, 7000};

    // 创建CacheLocation对象
    std::vector<LocationSpec> specs1(
        1, MetaSearcherTestHelper::CreateLocationSpec("nfs", "file:///tmp/test1/test.txt?offset=1&length=2&size=3"));
    CacheLocation location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, specs1);

    std::vector<LocationSpec> specs2(
        1, MetaSearcherTestHelper::CreateLocationSpec("hf3fs", "hf3fs:///tmp/test1/test.txt?offset=1&length=2&size=4"));
    CacheLocation location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, specs2);

    std::vector<LocationSpec> specs3(2,
                                     MetaSearcherTestHelper::CreateLocationSpec(
                                         "mooncake", "mooncake:///tmp/test1/test.txt?offset=1&length=2&size=5"));
    CacheLocation location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, specs3);

    CacheLocationVector locations = {location1, location2, location3};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    // 准备删除数据 - 删除前两个location
    std::vector<std::string> delete_location_ids = {out_location_ids[0], out_location_ids[1], "non_existent_id"};
    MetaSearcher::KeyVector delete_keys = {5000, 6000, 7000}; // 对应的keys

    // 调用BatchDeleteLocation
    std::vector<ErrorCode> delete_results;
    ec = meta_searcher_->BatchDeleteLocation(request_context_.get(), delete_keys, delete_location_ids, delete_results);

    // 验证结果
    EXPECT_EQ(
        0,
        meta_indexer_->storage_usage_array_[static_cast<std::uint8_t>(DataStorageType::DATA_STORAGE_TYPE_NFS)].load());
    EXPECT_EQ(0,
              meta_indexer_->storage_usage_array_[static_cast<std::uint8_t>(DataStorageType::DATA_STORAGE_TYPE_HF3FS)]
                  .load());
    EXPECT_EQ(
        10,
        meta_indexer_->storage_usage_array_[static_cast<std::uint8_t>(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE)]
            .load());
}

TEST_F(MetaSearcherTest, TestBatchVsSequentialPerformance) {
    const size_t num_keys = 100;
    MetaSearcher::KeyVector keys;
    CacheLocationVector locations;

    for (size_t i = 0; i < num_keys; i++) {
        keys.push_back(10000 + i);

        auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
        CacheLocation location =
            MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
        locations.push_back(location);
    }

    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);

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
    auto batch_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i <= 100; ++i) {
        CacheLocationVector out_locations;
        BlockMask mask; // 空mask，不跳过任何元素
        ec = meta_searcher_->PrefixMatch(request_context_.get(), keys, mask, out_locations, &policy_);
        EXPECT_EQ(ec, ErrorCode::EC_OK);
        EXPECT_EQ(out_locations.size(), num_keys);
    }
    auto batch_end = std::chrono::high_resolution_clock::now();
    auto batch_duration = std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start);
    KVCM_LOG_INFO("Batch PrefixMatch duration for 100 runs: %ld ms", batch_duration.count() / 1000);
    KVCM_LOG_INFO("Average per run: %.2f ms", batch_duration.count() / 100000.0);

    auto sequential_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i <= 100; ++i) {
        CacheLocationVector out_locations;
        BlockMask mask;
        ErrorCode result_ec = ErrorCode::EC_OK;
        for (size_t j = 0; j < keys.size(); ++j) {
            MetaSearcher::KeyVector single_key = {keys[j]};
            result_ec = meta_searcher_->PrefixMatch(request_context_.get(), single_key, mask, out_locations, &policy_);
            if (result_ec != ErrorCode::EC_OK) {
                break;
            }
        }
        EXPECT_EQ(result_ec, ErrorCode::EC_OK);
    }
    auto sequential_end = std::chrono::high_resolution_clock::now();
    auto sequential_duration = std::chrono::duration_cast<std::chrono::microseconds>(sequential_end - sequential_start);
    KVCM_LOG_INFO("Sequential PrefixMatch duration for 100 runs: %ld ms", sequential_duration.count() / 1000);
    KVCM_LOG_INFO("Average per run: %.2f ms", sequential_duration.count() / 100000.0);
}

TEST_F(MetaSearcherTest, TestBatchCASLocationStatus) {
    // 准备测试数据
    MetaSearcher::KeyVector keys = {100, 200, 300};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocation location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocation location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
    CacheLocation location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

    CacheLocationVector locations = {location1, location2, location3};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    // 验证初始状态为CLS_WRITING
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
            EXPECT_EQ(location.status(), CLS_WRITING); // 初始状态应为CLS_WRITING
        }
    }

    // 准备CAS任务：将状态从WRITING更新为SERVING
    std::vector<std::vector<MetaSearcher::LocationCASTask>> batch_tasks;
    for (size_t i = 0; i < keys.size(); i++) {
        std::vector<MetaSearcher::LocationCASTask> tasks;
        tasks.push_back({out_location_ids[i], CLS_WRITING, CLS_SERVING}); // 从WRITING状态更新为SERVING状态
        batch_tasks.push_back(tasks);
    }

    // 调用BatchCASLocationStatus
    std::vector<std::vector<ErrorCode>> out_batch_results;
    ec = meta_searcher_->BatchCASLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_batch_results.size(), 3);

    // 验证每个任务的结果
    for (const auto &results : out_batch_results) {
        EXPECT_EQ(results.size(), 1);            // 每个key只有一个任务
        EXPECT_EQ(results[0], ErrorCode::EC_OK); // 应该成功
    }

    // 验证状态已更新
    out_location_maps.clear();
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
            EXPECT_EQ(location.status(), CLS_SERVING); // 状态应已更新为SERVING
            EXPECT_EQ(location.type(), locations[i].type());
        }
    }

    // 再次尝试CAS操作，这次应该失败，因为状态已经不是WRITING了
    std::vector<std::vector<MetaSearcher::LocationCASTask>> batch_tasks_fail;
    for (size_t i = 0; i < keys.size(); i++) {
        std::vector<MetaSearcher::LocationCASTask> tasks;
        tasks.push_back(
            {out_location_ids[i], CLS_WRITING, CLS_DELETING}); // 从WRITING状态更新为DELETING状态，但实际是SERVING
        batch_tasks_fail.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> out_batch_results_fail;
    ec = meta_searcher_->BatchCASLocationStatus(request_context_.get(), keys, batch_tasks_fail, out_batch_results_fail);
    EXPECT_EQ(ec, ErrorCode::EC_OK); // 整体操作成功，但单个任务会失败

    // 验证CAS失败的结果
    for (const auto &results : out_batch_results_fail) {
        EXPECT_EQ(results.size(), 1);                  // 每个key只有一个任务
        EXPECT_EQ(results[0], ErrorCode::EC_MISMATCH); // 应该失败，因为状态不匹配
    }
}

TEST_F(MetaSearcherTest, TestBatchCADLocationStatus) {
    // 准备测试数据
    MetaSearcher::KeyVector keys = {400, 500, 600};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocation location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocation location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, location_specs);
    CacheLocation location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, location_specs);

    CacheLocationVector locations = {location1, location2, location3};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    // 首先将状态更新为DELETING
    std::vector<CacheLocationStatus> new_statuses = {CLS_DELETING, CLS_DELETING, CLS_DELETING};
    // 构建批量任务，每个key对应一个任务
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> update_batch_tasks;
    update_batch_tasks.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        tasks.push_back({out_location_ids[i], new_statuses[i]});
        update_batch_tasks.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> update_out_batch_results;
    ec = meta_searcher_->BatchUpdateLocationStatus(
        request_context_.get(), keys, update_batch_tasks, update_out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);

    // 准备CAD任务：删除状态为DELETING的位置
    std::vector<std::vector<MetaSearcher::LocationCADTask>> batch_tasks;
    for (size_t i = 0; i < keys.size(); i++) {
        std::vector<MetaSearcher::LocationCADTask> tasks;
        tasks.push_back({out_location_ids[i], CLS_DELETING}); // 删除状态为DELETING的位置
        batch_tasks.push_back(tasks);
    }

    // 调用BatchCADLocationStatus
    std::vector<std::vector<ErrorCode>> out_batch_results;
    ec = meta_searcher_->BatchCADLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_batch_results.size(), 3);

    // 验证每个任务的结果
    for (const auto &results : out_batch_results) {
        EXPECT_EQ(results.size(), 1);            // 每个key只有一个任务
        EXPECT_EQ(results[0], ErrorCode::EC_OK); // 应该成功
    }

    // 验证位置已被删除
    std::vector<CacheLocationMap> out_location_maps;
    BlockMask mask; // 空mask，不跳过任何元素

    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 3);

    for (const auto &location_map : out_location_maps) {
        // 由于位置已被删除，location_map应该是空的
        EXPECT_TRUE(location_map.empty());
    }

    // 再次尝试CAD操作，这次应该失败，因为位置已经不存在了
    std::vector<std::vector<MetaSearcher::LocationCADTask>> batch_tasks_fail;
    for (size_t i = 0; i < keys.size(); i++) {
        std::vector<MetaSearcher::LocationCADTask> tasks;
        tasks.push_back({out_location_ids[i], CLS_DELETING}); // 尝试删除已不存在的位置
        batch_tasks_fail.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> out_batch_results_fail;
    ec = meta_searcher_->BatchCADLocationStatus(request_context_.get(), keys, batch_tasks_fail, out_batch_results_fail);
    EXPECT_EQ(ec, ErrorCode::EC_ERROR); // location删空了，key也不存在了，所以会error。

    // 验证CAD失败的结果
    for (const auto &results : out_batch_results_fail) {
        EXPECT_EQ(results.size(), 1); // 每个key只有一个任务
        // 位置不存在时，可能返回EC_NOENT或EC_IO_ERROR
        EXPECT_TRUE(results[0] == ErrorCode::EC_NOENT); // 应该失败，因为位置不存在
    }
}

TEST_F(MetaSearcherTest, TestBatchCADLocationStatus2) {
    // 准备测试数据
    MetaSearcher::KeyVector keys = {400, 500, 600};

    // 创建CacheLocation对象
    std::vector<LocationSpec> specs1(
        1, MetaSearcherTestHelper::CreateLocationSpec("nfs", "file:///tmp/test1/test.txt?offset=1&length=2&size=3"));
    CacheLocation location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, specs1);

    std::vector<LocationSpec> specs2(
        1, MetaSearcherTestHelper::CreateLocationSpec("hf3fs", "hf3fs:///tmp/test1/test.txt?offset=1&length=2&size=4"));
    CacheLocation location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 2, specs2);

    std::vector<LocationSpec> specs3(2,
                                     MetaSearcherTestHelper::CreateLocationSpec(
                                         "mooncake", "mooncake:///tmp/test1/test.txt?offset=1&length=2&size=5"));
    CacheLocation location3 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 3, specs3);

    CacheLocationVector locations = {location1, location2, location3};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 3);

    // 首先将状态更新为DELETING
    std::vector<CacheLocationStatus> new_statuses = {CLS_DELETING, CLS_DELETING, CLS_DELETING};
    // 构建批量任务，每个key对应一个任务
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> update_batch_tasks;
    update_batch_tasks.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        tasks.push_back({out_location_ids[i], new_statuses[i]});
        update_batch_tasks.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> update_out_batch_results;
    ec = meta_searcher_->BatchUpdateLocationStatus(
        request_context_.get(), keys, update_batch_tasks, update_out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);

    // 准备CAD任务：删除状态为DELETING的位置
    std::vector<std::vector<MetaSearcher::LocationCADTask>> batch_tasks;
    for (size_t i = 0; i < keys.size(); i++) {
        std::vector<MetaSearcher::LocationCADTask> tasks;
        tasks.push_back({out_location_ids[i], CLS_DELETING}); // 删除状态为DELETING的位置
        batch_tasks.push_back(tasks);
    }

    // 调用BatchCADLocationStatus
    std::vector<std::vector<ErrorCode>> out_batch_results;
    ec = meta_searcher_->BatchCADLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_batch_results.size(), 3);

    EXPECT_EQ(
        0,
        meta_indexer_->storage_usage_array_[static_cast<std::uint8_t>(DataStorageType::DATA_STORAGE_TYPE_NFS)].load());
    EXPECT_EQ(0,
              meta_indexer_->storage_usage_array_[static_cast<std::uint8_t>(DataStorageType::DATA_STORAGE_TYPE_HF3FS)]
                  .load());
    EXPECT_EQ(
        0,
        meta_indexer_->storage_usage_array_[static_cast<std::uint8_t>(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE)]
            .load());
}

TEST_F(MetaSearcherTest, TestBatchCASLocationStatusMultipleTasksPerKey) {
    // 测试一个key对应多个CAS任务的情况
    MetaSearcher::KeyVector keys = {700};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocation location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);

    CacheLocationVector locations = {location1};

    // 添加位置信息
    std::vector<std::string> out_location_ids;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations, out_location_ids);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids.size(), 1);

    // 添加第二个位置到同一个key
    CacheLocation location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 1, location_specs);
    std::vector<std::string> out_location_ids2;
    ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, {location2}, out_location_ids2);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids2.size(), 1);

    // 验证现在这个key有两个位置
    std::vector<CacheLocationMap> out_location_maps;
    BlockMask mask;
    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 1);
    EXPECT_EQ(out_location_maps[0].size(), 2); // 应该有两个位置

    // 准备CAS任务：对同一个key的两个位置进行状态更新
    std::vector<std::vector<MetaSearcher::LocationCASTask>> batch_tasks;
    std::vector<MetaSearcher::LocationCASTask> tasks;
    tasks.push_back({out_location_ids[0], CLS_WRITING, CLS_SERVING});   // 更新第一个位置
    tasks.push_back({out_location_ids2[0], CLS_WRITING, CLS_DELETING}); // 更新第二个位置
    batch_tasks.push_back(tasks);

    // 调用BatchCASLocationStatus
    std::vector<std::vector<ErrorCode>> out_batch_results;
    ec = meta_searcher_->BatchCASLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_batch_results.size(), 1);
    EXPECT_EQ(out_batch_results[0].size(), 2); // 应该有两个结果

    // 验证结果
    EXPECT_EQ(out_batch_results[0][0], ErrorCode::EC_OK); // 第一个位置更新成功
    EXPECT_EQ(out_batch_results[0][1], ErrorCode::EC_OK); // 第二个位置更新成功

    // 验证状态已更新
    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 1);
    EXPECT_EQ(out_location_maps[0].size(), 2); // 仍然有两个位置

    // 检查每个位置的状态
    auto it1 = out_location_maps[0].find(out_location_ids[0]);
    auto it2 = out_location_maps[0].find(out_location_ids2[0]);
    EXPECT_NE(it1, out_location_maps[0].end());
    EXPECT_NE(it2, out_location_maps[0].end());

    if (it1 != out_location_maps[0].end()) {
        EXPECT_EQ(it1->second.status(), CLS_SERVING);
    }
    if (it2 != out_location_maps[0].end()) {
        EXPECT_EQ(it2->second.status(), CLS_DELETING);
    }
}

TEST_F(MetaSearcherTest, TestBatchCADLocationStatusMultipleTasksPerKey) {
    // 测试一个key对应多个CAD任务的情况
    MetaSearcher::KeyVector keys = {800};

    // 创建CacheLocation对象
    auto location_specs = MetaSearcherTestHelper::CreateDefaultLocationSpecs();
    CacheLocation location1 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_NFS, 1, location_specs);
    CacheLocation location2 =
        MetaSearcherTestHelper::CreateCacheLocation(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 1, location_specs);

    // 添加第一个位置信息
    CacheLocationVector locations1 = {location1};
    std::vector<std::string> out_location_ids1;
    ErrorCode ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations1, out_location_ids1);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids1.size(), 1);

    // 添加第二个位置到同一个key
    CacheLocationVector locations2 = {location2};
    std::vector<std::string> out_location_ids2;
    ec = meta_searcher_->BatchAddLocation(request_context_.get(), keys, locations2, out_location_ids2);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_ids2.size(), 1);

    // 合并位置ID
    std::vector<std::string> out_location_ids = {out_location_ids1[0], out_location_ids2[0]};

    // 验证位置已添加
    std::vector<CacheLocationMap> out_location_maps;
    BlockMask mask;
    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 1);
    EXPECT_EQ(out_location_maps[0].size(), 2); // 应该有两个位置
    for (const auto &loc_pair : out_location_maps[0]) {
        EXPECT_EQ(loc_pair.second.status(), CLS_WRITING);
    }

    // 将两个位置的状态都更新为DELETING
    std::vector<CacheLocationStatus> new_statuses1 = {CLS_DELETING};
    // 构建批量任务，每个key对应一个任务
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> batch_tasks1;
    batch_tasks1.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        if (i < out_location_ids1.size()) {
            tasks.push_back({out_location_ids1[i], new_statuses1[0]});
        }
        batch_tasks1.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> out_batch_results1;
    ec = meta_searcher_->BatchUpdateLocationStatus(request_context_.get(), keys, batch_tasks1, out_batch_results1);
    EXPECT_EQ(ec, ErrorCode::EC_OK);

    std::vector<CacheLocationStatus> new_statuses2 = {CLS_DELETING};
    // 构建批量任务，每个key对应一个任务
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> batch_tasks2;
    batch_tasks2.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<MetaSearcher::LocationUpdateTask> tasks;
        if (i < out_location_ids2.size()) {
            tasks.push_back({out_location_ids2[i], new_statuses2[0]});
        }
        batch_tasks2.push_back(tasks);
    }

    std::vector<std::vector<ErrorCode>> out_batch_results2;
    ec = meta_searcher_->BatchUpdateLocationStatus(request_context_.get(), keys, batch_tasks2, out_batch_results2);
    EXPECT_EQ(ec, ErrorCode::EC_OK);

    // 验证状态已更新
    out_location_maps.clear();
    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 1);
    EXPECT_EQ(out_location_maps[0].size(), 2); // 仍然有两个位置
    for (const auto &loc_pair : out_location_maps[0]) {
        EXPECT_EQ(loc_pair.second.status(), CLS_DELETING);
    }

    // 准备CAD任务：删除同一个key的两个位置
    std::vector<std::vector<MetaSearcher::LocationCADTask>> batch_tasks;
    std::vector<MetaSearcher::LocationCADTask> tasks;
    tasks.push_back({out_location_ids[0], CLS_DELETING}); // 删除第一个位置
    tasks.push_back({out_location_ids[1], CLS_DELETING}); // 删除第二个位置
    batch_tasks.push_back(tasks);

    // 调用BatchCADLocationStatus
    std::vector<std::vector<ErrorCode>> out_batch_results;
    ec = meta_searcher_->BatchCADLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_batch_results.size(), 1);
    EXPECT_EQ(out_batch_results[0].size(), 2); // 应该有两个结果

    // 验证结果
    EXPECT_EQ(out_batch_results[0][0], ErrorCode::EC_OK); // 第一个位置删除成功
    EXPECT_EQ(out_batch_results[0][1], ErrorCode::EC_OK); // 第二个位置删除成功

    // 验证位置已被删除
    out_location_maps.clear();
    ec = meta_searcher_->BatchGetLocation(request_context_.get(), keys, mask, out_location_maps);
    EXPECT_EQ(ec, ErrorCode::EC_OK);
    EXPECT_EQ(out_location_maps.size(), 1);
    EXPECT_TRUE(out_location_maps[0].empty()); // 应该为空
}

TEST_F(MetaSearcherTest, TestBatchCASLocationStatusErrorCases) {
    // 测试错误情况：keys和batch_tasks大小不匹配
    MetaSearcher::KeyVector keys = {900, 901};
    std::vector<std::vector<MetaSearcher::LocationCASTask>> batch_tasks;
    std::vector<MetaSearcher::LocationCASTask> tasks;
    tasks.push_back({"location_id_1", CLS_NEW, CLS_SERVING});
    batch_tasks.push_back(tasks);
    // 注意：这里batch_tasks只有一个元素，而keys有两个元素，应该返回EC_BADARGS

    std::vector<std::vector<ErrorCode>> out_batch_results;
    ErrorCode ec = meta_searcher_->BatchCASLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_BADARGS);
}

TEST_F(MetaSearcherTest, TestBatchCADLocationStatusErrorCases) {
    // 测试错误情况：keys和batch_tasks大小不匹配
    MetaSearcher::KeyVector keys = {1000, 1001};
    std::vector<std::vector<MetaSearcher::LocationCADTask>> batch_tasks;
    std::vector<MetaSearcher::LocationCADTask> tasks;
    tasks.push_back({"location_id_1", CLS_DELETING});
    batch_tasks.push_back(tasks);
    // 注意：这里batch_tasks只有一个元素，而keys有两个元素，应该返回EC_BADARGS

    std::vector<std::vector<ErrorCode>> out_batch_results;
    ErrorCode ec = meta_searcher_->BatchCADLocationStatus(request_context_.get(), keys, batch_tasks, out_batch_results);
    EXPECT_EQ(ec, ErrorCode::EC_BADARGS);
}
