#include <filesystem>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/manager/cache_location.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
using namespace kv_cache_manager;
namespace {
class SchedulePlanExecutorTestHelper {
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
class SchedulePlanExecutorTest : public TESTBASE {
public:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        meta_manager_ = std::make_shared<MetaIndexerManager>();
        data_storage_manager_ = std::make_shared<DataStorageManager>(metrics_registry_);
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

    bool CreateMetaIndexer(const std::string &instance_id, const std::string &storage_type) {
        auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
        auto backend_config = ConstructMetaStorageBackendConfig();
        meta_indexer_config->meta_storage_backend_config_ = backend_config;
        meta_indexer_config->mutex_shard_num_ = 32;
        meta_indexer_config->max_key_count_ = 10000;
        return meta_manager_->CreateMetaIndexer(instance_id, meta_indexer_config);
    }

    bool CreateDataStorage() {
        auto nfs_storage_spec = std::make_shared<NfsStorageSpec>();
        nfs_storage_spec->set_root_path("/mnt/nfs");
        nfs_storage_spec->set_key_count_per_file(5);

        StorageConfig nfs_storage_config;
        nfs_storage_config.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
        nfs_storage_config.set_global_unique_name("nfs_01");
        nfs_storage_config.set_storage_spec(nfs_storage_spec);
        auto request_context = std::make_shared<RequestContext>("test_trace_id");
        return data_storage_manager_->RegisterStorage(request_context.get(), "nfs_01", nfs_storage_config);
    }
    std::shared_ptr<MetaIndexerManager> meta_manager_;
    std::shared_ptr<DataStorageManager> data_storage_manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    const std::string kTestInstanceName = "test_instance";
};

TEST_F(SchedulePlanExecutorTest, TestSubmit) {
    CreateMetaIndexer(kTestInstanceName, "local");
    CreateDataStorage();
    SchedulePlanExecutor executor(2, meta_manager_, data_storage_manager_, metrics_registry_);
    auto request_context = std::make_shared<RequestContext>("test_trace_id");

    int num_task = 10;
    for (int i = 0; i < num_task; i++) {
        // 使用MetaSearcher添加location
        MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));

        // 创建CacheLocation对象
        CacheLocation location1 = SchedulePlanExecutorTestHelper::CreateCacheLocation(
            DataStorageType::DATA_STORAGE_TYPE_NFS,
            1,
            {SchedulePlanExecutorTestHelper::CreateLocationSpec("TP0", "nfs://nfs_01/block" + std::to_string(i * 2))});
        CacheLocation location2 = SchedulePlanExecutorTestHelper::CreateCacheLocation(
            DataStorageType::DATA_STORAGE_TYPE_NFS,
            1,
            {SchedulePlanExecutorTestHelper::CreateLocationSpec("TP0",
                                                                "nfs://nfs_01/block" + std::to_string(i * 2 + 1))});

        // 添加location
        std::vector<std::string> location_ids1, location_ids2;
        ASSERT_EQ(ErrorCode::EC_OK,
                  meta_searcher.BatchAddLocation(request_context.get(), {i * 2}, {location1}, location_ids1));
        ASSERT_EQ(ErrorCode::EC_OK,
                  meta_searcher.BatchAddLocation(request_context.get(), {i * 2 + 1}, {location2}, location_ids2));

        // 验证数据已添加
        std::vector<CacheLocationMap> location_maps;
        BlockMask empty_mask;
        ASSERT_EQ(ErrorCode::EC_OK,
                  meta_searcher.BatchGetLocation(request_context.get(), {i * 2}, empty_mask, location_maps));
        ASSERT_FALSE(location_maps.empty());
        ASSERT_EQ(1, location_maps[0].size());
        location_maps.clear();
        ASSERT_EQ(ErrorCode::EC_OK,
                  meta_searcher.BatchGetLocation(request_context.get(), {i * 2 + 1}, empty_mask, location_maps));
        ASSERT_FALSE(location_maps.empty());
        ASSERT_EQ(1, location_maps[0].size());
    }

    std::vector<std::future<PlanExecuteResult>> futures;
    for (int i = 0; i < num_task; i++) {
        CacheMetaDelRequest request{
            .instance_id = "test_instance",
            .block_keys = {i * 2, i * 2 + 1},
        };
        auto future = executor.Submit(request);
        futures.push_back(std::move(future));
    }
    // 等待所有任务完成
    for (int i = 0; i < num_task; i++) {
        auto result = futures[i].get();
        ASSERT_TRUE(result.status == ErrorCode::EC_OK || result.status == ErrorCode::EC_PARTIAL_OK)
            << result.error_message;
    }

    for (int i = 0; i < num_task; i++) {
        MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));
        std::vector<CacheLocationMap> location_maps;
        BlockMask empty_mask;
        // BatchGetLocation总是返回EC_OK，即使key不存在，所以检查location_maps是否为空
        ASSERT_EQ(ErrorCode::EC_OK,
                  meta_searcher.BatchGetLocation(request_context.get(), {i * 2}, empty_mask, location_maps));
        ASSERT_TRUE(location_maps.empty() || location_maps[0].empty());
        location_maps.clear();
        ASSERT_EQ(ErrorCode::EC_OK,
                  meta_searcher.BatchGetLocation(request_context.get(), {i * 2 + 1}, empty_mask, location_maps));
        ASSERT_TRUE(location_maps.empty() || location_maps[0].empty());
    }
}

TEST_F(SchedulePlanExecutorTest, TestNonExistInstance) {
    CreateMetaIndexer(kTestInstanceName, "local");
    CreateDataStorage();

    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);

    CacheMetaDelRequest request{
        .instance_id = "test_instance_non_exist",
        .block_keys = {1, 2},
    };
    auto result = executor.Submit(request).get();
    ASSERT_EQ(ErrorCode::EC_NOENT, result.status);
}

TEST_F(SchedulePlanExecutorTest, TestStop) {
    CreateMetaIndexer(kTestInstanceName, "local");
    CreateDataStorage();

    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);
    executor.Stop();

    CacheMetaDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = {1, 2},
    };
    auto future = executor.Submit(request);
    ASSERT_EQ(ErrorCode::EC_ERROR, future.get().status);
}
// 测试设置状态为DELETING功能
TEST_F(SchedulePlanExecutorTest, TestSetStatusToDeleting) {
    // 创建 MetaIndexer
    CreateMetaIndexer(kTestInstanceName, "local");
    CreateDataStorage();
    // 创建 SchedulePlanExecutor (DataStorageManager可以为nullptr，因为我们只测试状态设置)
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);

    // 添加测试数据
    auto request_context = std::make_shared<RequestContext>("test_trace_id");

    // 使用MetaSearcher添加location
    MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));

    // 创建CacheLocation对象
    CacheLocation new_location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("test_loc", "nfs://nfs_01/block200")});

    // 添加location
    std::vector<std::string> location_ids;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchAddLocation(request_context.get(), {200}, {new_location}, location_ids));

    // 验证数据已添加
    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {200}, empty_mask, location_maps));
    ASSERT_FALSE(location_maps.empty());

    // 提交删除任务
    CacheMetaDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = {200},
    };

    // 提交任务后立即检查状态是否变为DELETING (Submit方法会同步设置状态)
    auto future = executor.Submit(request);

    // 检查状态是否已更新为DELETING
    location_maps.clear();
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {200}, empty_mask, location_maps));
    ASSERT_FALSE(location_maps.empty());

    for (const auto &location_map : location_maps) {
        for (const auto &loc_kv : location_map) {
            const auto &location = loc_kv.second;
            ASSERT_EQ(CacheLocationStatus::CLS_DELETING, location.status())
                << "Location status should be CLS_DELETING after Submit";
        }
    }
    // 等待任务完成 (即使DataStorageManager为nullptr，任务也会完成，只是存储删除会失败)
    future.get();
}
// 测试一个block_key对应多个location的情况
TEST_F(SchedulePlanExecutorTest, TestMultipleLocationsPerBlockKey) {
    // 创建 MetaIndexer
    CreateMetaIndexer(kTestInstanceName, "local");
    CreateDataStorage();
    // 创建 SchedulePlanExecutor
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);

    // 添加测试数据，一个block_key对应多个location
    auto request_context = std::make_shared<RequestContext>("test_trace_id");

    // 使用MetaSearcher添加多个location
    MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));

    // 创建多个CacheLocation对象
    CacheLocation location1 = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("TP0", "nfs://nfs_01/block1")});
    CacheLocation location2 = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("TP0", "nfs://nfs_01/block2")});
    CacheLocation location3 = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("TP0", "nfs://nfs_01/block3")});

    // 分别添加location到同一个block_key
    std::vector<std::string> location_ids1, location_ids2, location_ids3;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchAddLocation(request_context.get(), {400}, {location1}, location_ids1));
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchAddLocation(request_context.get(), {400}, {location2}, location_ids2));
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchAddLocation(request_context.get(), {400}, {location3}, location_ids3));

    // 验证数据已添加
    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {400}, empty_mask, location_maps));
    ASSERT_FALSE(location_maps.empty());
    ASSERT_EQ(1, location_maps.size());    // 应该只有一个block_key
    ASSERT_EQ(3, location_maps[0].size()); // 但包含三个location

    // 提交删除任务
    CacheMetaDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = {400},
    };

    // 提交任务
    auto future = executor.Submit(request);

    // 检查状态是否已更新为DELETING
    location_maps.clear();
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {400}, empty_mask, location_maps));
    ASSERT_FALSE(location_maps.empty());
    ASSERT_EQ(1, location_maps.size());    // 应该只有一个block_key
    ASSERT_EQ(3, location_maps[0].size()); // 但包含三个location

    // 检查所有location的状态是否都更新为DELETING
    for (const auto &loc_kv : location_maps[0]) {
        const auto &location = loc_kv.second;
        KVCM_LOG_INFO("Location ID: %s, Status: %d", loc_kv.first.c_str(), location.status());
        ASSERT_EQ(CacheLocationStatus::CLS_DELETING, location.status())
            << "Location status should be CLS_DELETING after Submit";
    }

    // 等待任务完成
    auto result = future.get();
    ASSERT_TRUE(result.status == ErrorCode::EC_OK || result.status == ErrorCode::EC_PARTIAL_OK)
        << "Error message: " << result.error_message;
}
// 测试存储删除功能
TEST_F(SchedulePlanExecutorTest, TestStorageDelete) {
    // 创建 NFS 存储配置
    CreateDataStorage();

    // 创建 MetaIndexer
    CreateMetaIndexer(kTestInstanceName, "local");

    // 创建 SchedulePlanExecutor
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);

    // 添加测试数据，使用NFS URI格式
    auto request_context = std::make_shared<RequestContext>("test_trace_id");

    // 使用MetaSearcher添加location
    MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));

    // 创建CacheLocation对象
    CacheLocation location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("test_loc", "nfs://nfs_01/test_block_for_storage_delete")});

    // 添加location
    std::vector<std::string> location_ids;
    ASSERT_EQ(ErrorCode::EC_OK, meta_searcher.BatchAddLocation(request_context.get(), {300}, {location}, location_ids));

    // 验证数据已添加
    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {300}, empty_mask, location_maps));
    ASSERT_FALSE(location_maps.empty());
    ASSERT_EQ(1, location_maps[0].size());

    // 提交删除任务
    CacheMetaDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = {300},
    };

    auto future = executor.Submit(request);
    auto result = future.get();

    // 验证删除结果 (应该成功，因为我们正确配置了存储)
    ASSERT_TRUE(result.status == ErrorCode::EC_OK || result.status == ErrorCode::EC_PARTIAL_OK)
        << "Error message: " << result.error_message;

    // 验证数据已被删除
    location_maps.clear();
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {300}, empty_mask, location_maps));
    ASSERT_TRUE(location_maps.empty() || location_maps[0].empty());
}

// 测试延迟执行功能
TEST_F(SchedulePlanExecutorTest, TestDelayExecution) {
    // 创建 MetaIndexer
    CreateMetaIndexer(kTestInstanceName, "local");
    CreateDataStorage();

    // 创建 SchedulePlanExecutor
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);

    // 添加测试数据
    auto request_context = std::make_shared<RequestContext>("test_trace_id");

    // 使用MetaSearcher添加location
    MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));

    // 创建CacheLocation对象
    CacheLocation location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("test_loc", "nfs://nfs_01/test_block_for_delay")});

    // 添加location
    std::vector<std::string> location_ids;
    ASSERT_EQ(ErrorCode::EC_OK, meta_searcher.BatchAddLocation(request_context.get(), {500}, {location}, location_ids));

    // 验证数据已添加
    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {500}, empty_mask, location_maps));
    ASSERT_FALSE(location_maps.empty());
    ASSERT_EQ(1, location_maps[0].size());

    // 记录开始时间
    auto start_time = std::chrono::steady_clock::now();

    // 提交一个延迟500毫秒的任务
    std::chrono::microseconds delay = std::chrono::milliseconds(500);
    CacheMetaDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = {500},
        .delay = delay,
    };

    auto future = executor.Submit(request);

    // 等待任务完成
    auto result = future.get();

    // 验证删除结果
    ASSERT_TRUE(result.status == ErrorCode::EC_OK || result.status == ErrorCode::EC_PARTIAL_OK)
        << "Error message: " << result.error_message;

    // 验证数据已被删除
    location_maps.clear();
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {500}, empty_mask, location_maps));
    ASSERT_TRUE(location_maps.empty() || location_maps[0].empty());

    // 验证任务确实延迟执行了
    auto end_time = std::chrono::steady_clock::now();
    auto execution_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // 由于系统调度等因素，允许一定的误差范围（如100毫秒）
    ASSERT_GE(execution_duration.count(), delay.count() / 1000 - 10)
        << "Task executed too early, expected delay: " << delay.count() / 1000 - 10
        << "ms, actual execution time: " << execution_duration.count() << "ms";
}

// 测试多个延迟任务的执行顺序
TEST_F(SchedulePlanExecutorTest, TestMultipleDelayedTasksExecutionOrder) {
    // 创建 MetaIndexer
    CreateMetaIndexer(kTestInstanceName, "local");
    CreateDataStorage();

    // 创建 SchedulePlanExecutor
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);

    // 添加测试数据
    auto request_context = std::make_shared<RequestContext>("test_trace_id");

    // 使用MetaSearcher添加location
    MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));

    for (int i = 0; i < 3; i++) {
        // 创建CacheLocation对象
        CacheLocation location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
            DataStorageType::DATA_STORAGE_TYPE_NFS,
            1,
            {SchedulePlanExecutorTestHelper::CreateLocationSpec(
                "test_loc", "nfs://nfs_01/test_block_for_delay_order_" + std::to_string(i))});

        // 添加location
        std::vector<std::string> location_ids;
        ASSERT_EQ(ErrorCode::EC_OK,
                  meta_searcher.BatchAddLocation(request_context.get(), {600 + i}, {location}, location_ids));

        // 验证数据已添加
        std::vector<CacheLocationMap> location_maps;
        BlockMask empty_mask;
        ASSERT_EQ(ErrorCode::EC_OK,
                  meta_searcher.BatchGetLocation(request_context.get(), {600 + i}, empty_mask, location_maps));
        ASSERT_FALSE(location_maps.empty());
        ASSERT_EQ(1, location_maps[0].size());
    }

    // 记录开始时间
    auto start_time = std::chrono::steady_clock::now();

    // 提交多个不同延迟的任务
    std::vector<std::chrono::microseconds> delays = {
        std::chrono::milliseconds(600), // 最晚执行
        std::chrono::milliseconds(300), // 中间执行
        std::chrono::milliseconds(100)  // 最先执行
    };

    std::vector<std::future<PlanExecuteResult>> futures;
    for (int i = 0; i < 3; i++) {
        CacheMetaDelRequest request{
            .instance_id = kTestInstanceName,
            .block_keys = {600 + i},
            .delay = delays[i],
        };
        futures.push_back(executor.Submit(request));
    }

    // 等待所有任务完成
    for (int i = 0; i < 3; i++) {
        auto result = futures[i].get();
        ASSERT_TRUE(result.status == ErrorCode::EC_OK || result.status == ErrorCode::EC_PARTIAL_OK)
            << "Error message: " << result.error_message;
    }

    // 验证所有数据都已被删除
    for (int i = 0; i < 3; i++) {
        std::vector<CacheLocationMap> location_maps;
        BlockMask empty_mask;
        ASSERT_EQ(ErrorCode::EC_OK,
                  meta_searcher.BatchGetLocation(request_context.get(), {600 + i}, empty_mask, location_maps));
        ASSERT_TRUE(location_maps.empty() || location_maps[0].empty());
    }

    // 验证总执行时间至少等于最长的延迟时间
    auto end_time = std::chrono::steady_clock::now();
    auto execution_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // 最长的延迟是600ms，加上一些误差范围
    ASSERT_GE(execution_duration.count(), 590)
        << "Tasks executed too quickly, expected at least 590ms, actual: " << execution_duration.count() << "ms";
}

// 测试CacheLocationDelRequest的Submit方法
TEST_F(SchedulePlanExecutorTest, TestSubmitLocationDelRequest) {
    // 创建 MetaIndexer
    CreateMetaIndexer(kTestInstanceName, "local");
    CreateDataStorage();

    // 创建 SchedulePlanExecutor
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);

    // 添加测试数据，一个block_key对应多个location
    auto request_context = std::make_shared<RequestContext>("test_trace_id");

    // 使用MetaSearcher添加多个location
    MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));

    // 创建多个CacheLocation对象
    CacheLocation location1 = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("TP0", "nfs://nfs_01/block100")});
    CacheLocation location2 = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("TP0", "nfs://nfs_01/block101")});
    CacheLocation location3 = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("TP0", "nfs://nfs_01/block102")});

    // 分别添加location到同一个block_key
    std::vector<std::string> location_ids1, location_ids2, location_ids3;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchAddLocation(request_context.get(), {700}, {location1}, location_ids1));
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchAddLocation(request_context.get(), {700}, {location2}, location_ids2));
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchAddLocation(request_context.get(), {700}, {location3}, location_ids3));

    // 验证数据已添加
    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {700}, empty_mask, location_maps));
    ASSERT_FALSE(location_maps.empty());
    ASSERT_EQ(1, location_maps.size());    // 应该只有一个block_key
    ASSERT_EQ(3, location_maps[0].size()); // 但包含三个location

    // 记录原始location IDs
    std::vector<std::string> original_location_ids;
    for (const auto &loc_kv : location_maps[0]) {
        original_location_ids.push_back(loc_kv.first);
    }
    ASSERT_EQ(3, original_location_ids.size());

    // 提交删除特定location的任务
    CacheLocationDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = {700},
        .location_ids = {std::vector<std::string>{original_location_ids[0],
                                                  original_location_ids[1]}}, // 删除前两个location
        .delay = std::chrono::milliseconds(0),                                // 立即执行
    };

    // 提交任务
    auto future = executor.Submit(request);

    // 检查状态是否已更新为DELETING
    location_maps.clear();
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {700}, empty_mask, location_maps));
    ASSERT_FALSE(location_maps.empty());
    ASSERT_EQ(1, location_maps.size());    // 应该只有一个block_key
    ASSERT_EQ(3, location_maps[0].size()); // 但包含三个location

    // 检查被标记为删除的location的状态是否都更新为DELETING
    int deleting_count = 0;
    for (const auto &loc_kv : location_maps[0]) {
        const auto &location = loc_kv.second;
        if (loc_kv.first == original_location_ids[0] || loc_kv.first == original_location_ids[1]) {
            // 这两个应该被标记为DELETING
            ASSERT_EQ(CacheLocationStatus::CLS_DELETING, location.status())
                << "Location " << loc_kv.first << " status should be CLS_DELETING after Submit";
            deleting_count++;
        } else {
            // 其他location应该保持原状态
            ASSERT_NE(CacheLocationStatus::CLS_DELETING, location.status())
                << "Location " << loc_kv.first << " status should not be CLS_DELETING";
        }
    }
    ASSERT_EQ(2, deleting_count); // 应该有两个location被标记为删除

    // 等待任务完成
    auto result = future.get();
    ASSERT_TRUE(result.status == ErrorCode::EC_OK || result.status == ErrorCode::EC_PARTIAL_OK)
        << "Error message: " << result.error_message;

    // 验证被删除的location确实被删除了
    location_maps.clear();
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {700}, empty_mask, location_maps));
    ASSERT_FALSE(location_maps.empty());
    ASSERT_EQ(1, location_maps.size());    // 应该只有一个block_key
    ASSERT_EQ(1, location_maps[0].size()); // 只剩一个location，因为删除了两个

    // 确认剩下的location是第三个
    for (const auto &loc_kv : location_maps[0]) {
        ASSERT_EQ(original_location_ids[2], loc_kv.first) << "Only the third location should remain after deletion";
    }
}

// 测试CacheLocationDelRequest的Submit方法 - 非存在实例
TEST_F(SchedulePlanExecutorTest, TestSubmitLocationDelRequestNonExistInstance) {
    CreateMetaIndexer(kTestInstanceName, "local");
    CreateDataStorage();

    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);

    CacheLocationDelRequest request{
        .instance_id = "test_instance_non_exist",
        .block_keys = {1, 2},
        .location_ids = {{"loc1"}, {"loc2"}},
        .delay = std::chrono::milliseconds(0),
    };
    auto future = executor.Submit(request);
    auto result = future.get();
    ASSERT_EQ(ErrorCode::EC_NOENT, result.status);
}

// 测试CacheLocationDelRequest的Submit方法 - 延迟执行
TEST_F(SchedulePlanExecutorTest, TestSubmitLocationDelRequestWithDelay) {
    // 创建 MetaIndexer
    CreateMetaIndexer(kTestInstanceName, "local");
    CreateDataStorage();

    // 创建 SchedulePlanExecutor
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);

    // 添加测试数据
    auto request_context = std::make_shared<RequestContext>("test_trace_id");

    // 使用MetaSearcher添加location
    MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));

    // 创建CacheLocation对象
    CacheLocation location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("test_loc", "nfs://nfs_01/test_block_for_location_delay")});

    // 添加location
    std::vector<std::string> location_ids;
    ASSERT_EQ(ErrorCode::EC_OK, meta_searcher.BatchAddLocation(request_context.get(), {800}, {location}, location_ids));

    // 验证数据已添加
    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {800}, empty_mask, location_maps));
    ASSERT_FALSE(location_maps.empty());
    ASSERT_EQ(1, location_maps[0].size());

    // 记录原始location ID
    std::string original_location_id = location_maps[0].begin()->first;

    // 记录开始时间
    auto start_time = std::chrono::steady_clock::now();

    // 提交一个延迟500毫秒的任务
    std::chrono::microseconds delay = std::chrono::milliseconds(500);
    CacheLocationDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = {800},
        .location_ids = {{original_location_id}}, // 删除特定location
        .delay = delay,
    };

    auto future = executor.Submit(request);

    // 等待任务完成
    auto result = future.get();

    // 验证删除结果
    ASSERT_TRUE(result.status == ErrorCode::EC_OK || result.status == ErrorCode::EC_PARTIAL_OK)
        << "Error message: " << result.error_message;

    // 验证数据已被删除
    location_maps.clear();
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {800}, empty_mask, location_maps));
    ASSERT_TRUE(location_maps.empty() || location_maps[0].empty());

    // 验证任务确实延迟执行了
    auto end_time = std::chrono::steady_clock::now();
    auto execution_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // 由于系统调度等因素，允许一定的误差范围（如100毫秒）
    ASSERT_GE(execution_duration.count(), delay.count() / 1000 - 100)
        << "Task executed too early, expected delay: " << delay.count() / 1000 - 100
        << "ms, actual execution time: " << execution_duration.count() << "ms";
}
