#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "stub.h"
using namespace kv_cache_manager;
namespace {
std::atomic<bool> sync_entered{false};
std::atomic<bool> sync_completed{false};
std::atomic<bool> release_sync{true};
std::atomic<std::int64_t> sync_delay_ms{0};
std::atomic<std::size_t> sync_thread_hash{0};

bool MetaIndexer_Sync_stub(void *obj, const KeyVector &keys) noexcept {
    (void)obj;
    (void)keys;
    sync_thread_hash.store(std::hash<std::thread::id>{}(std::this_thread::get_id()), std::memory_order_relaxed);
    sync_entered.store(true, std::memory_order_release);
    while (!release_sync.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sync_delay_ms.load(std::memory_order_relaxed)));
    sync_completed.store(true, std::memory_order_release);
    return true;
}

std::shared_ptr<MetaIndexer> MetaIndexerManager_GetMetaIndexer_throw_stub(void *obj, const std::string &instance_id) {
    (void)obj;
    (void)instance_id;
    throw std::runtime_error("injected GetMetaIndexer exception");
}

class SchedulePlanExecutorTestHelper {
public:
    static LocationSpec CreateLocationSpec(const std::string &name = "", const std::string &uri = "") {
        LocationSpec spec(name, uri);
        return spec;
    }

    static CacheLocationConstPtr CreateCacheLocation(DataStorageType type = DataStorageType::DATA_STORAGE_TYPE_NFS,
                                                     size_t spec_size = 1,
                                                     const std::vector<LocationSpec> &specs = {}) {
        return std::make_shared<CacheLocation>(type, spec_size, specs);
    }

    static std::vector<LocationSpec> CreateDefaultLocationSpecs() {
        LocationSpec spec = CreateLocationSpec();
        return {spec};
    }
};

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

    ErrorCode CreateMetaIndexer(const std::string &instance_id, const std::string &storage_type) {
        auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
        auto backend_config = ConstructMetaStorageBackendConfig();
        meta_indexer_config->meta_storage_backend_config_ = backend_config;
        meta_indexer_config->mutex_shard_num_ = 32;
        meta_indexer_config->max_key_count_ = 10000;
        return meta_manager_->CreateMetaIndexer(instance_id, meta_indexer_config);
    }

    ErrorCode CreateCachedMetaIndexer(const std::string &instance_id, const std::string &path) {
        std::filesystem::remove(path);
        auto backend_config = std::make_shared<MetaStorageBackendConfig>();
        backend_config->SetStorageType(META_CACHED_BACKEND_TYPE_STR);
        backend_config->SetStorageUri("file://" + path + "?persistent_type=dummy&cache_type=local");
        auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
        meta_indexer_config->meta_storage_backend_config_ = std::move(backend_config);
        meta_indexer_config->mutex_shard_num_ = 32;
        meta_indexer_config->max_key_count_ = 10000;
        return meta_manager_->CreateMetaIndexer(instance_id, meta_indexer_config);
    }

    static void WaitForCachedMetaIndexerRunning(const std::shared_ptr<MetaIndexer> &indexer) {
        ASSERT_TRUE(indexer);
        for (int i = 0; i < 100; ++i) {
            if (indexer->backend_manager_->GetRecoverState() == MetaStorageBackendManager::RecoverState::kRunning) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        FAIL() << "cached metadata recovery did not finish in time";
    }

    ErrorCode CreateDataStorage() {
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

    std::shared_ptr<EventReportBackend> CreateEventReportStorage(const std::string &name) {
        auto spec = std::make_shared<EventReportStorageSpec>();
        spec->set_snapshot_min_interval_ms(0);
        spec->set_heartbeat_timeout_ms(60000);
        spec->set_cleanup_grace_ms(60000);
        StorageConfig config(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, name, spec);
        RequestContext context("create_event_report_storage");
        EXPECT_EQ(EC_OK, data_storage_manager_->RegisterStorage(&context, name, config));
        return std::dynamic_pointer_cast<EventReportBackend>(data_storage_manager_->GetDataStorageBackend(name));
    }
    // 注册一个 Dummy storage（基于文件系统），供 copy 任务做真实文件复制
    bool CreateDummyStorage(const std::string &name, const std::string &root) {
        auto spec = std::make_shared<DummyStorageSpec>();
        spec->set_root_path(root);
        spec->set_key_count_per_file(1);
        StorageConfig config;
        config.set_type(DataStorageType::DATA_STORAGE_TYPE_DUMMY);
        config.set_global_unique_name(name);
        config.set_storage_spec(spec);
        auto request_context = std::make_shared<RequestContext>("test_trace_id");
        return data_storage_manager_->RegisterStorage(request_context.get(), name, config) == ErrorCode::EC_OK;
    }
    static DataStorageUri MakeUri(const std::string &path) {
        DataStorageUri uri;
        uri.SetProtocol("dummy");
        uri.SetPath(path);
        return uri;
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
        CacheLocationConstPtr location1 = SchedulePlanExecutorTestHelper::CreateCacheLocation(
            DataStorageType::DATA_STORAGE_TYPE_NFS,
            1,
            {SchedulePlanExecutorTestHelper::CreateLocationSpec("TP0", "nfs://nfs_01/block" + std::to_string(i * 2))});
        CacheLocationConstPtr location2 = SchedulePlanExecutorTestHelper::CreateCacheLocation(
            DataStorageType::DATA_STORAGE_TYPE_NFS,
            1,
            {SchedulePlanExecutorTestHelper::CreateLocationSpec("TP0",
                                                                "nfs://nfs_01/block" + std::to_string(i * 2 + 1))});

        // 添加location
        std::vector<std::string> location_ids1, location_ids2;
        ASSERT_EQ(ErrorCode::EC_OK,
                  BatchAddLocationForTest(&meta_searcher, request_context.get(), {i * 2}, {location1}, location_ids1));
        ASSERT_EQ(
            ErrorCode::EC_OK,
            BatchAddLocationForTest(&meta_searcher, request_context.get(), {i * 2 + 1}, {location2}, location_ids2));

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
    CacheLocationConstPtr new_location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("test_loc", "nfs://nfs_01/block200")});

    // 添加location
    std::vector<std::string> location_ids;
    ASSERT_EQ(ErrorCode::EC_OK,
              BatchAddLocationForTest(&meta_searcher, request_context.get(), {200}, {new_location}, location_ids));

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
            const auto &location = *loc_kv.second;
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
    CacheLocationConstPtr location1 = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("TP0", "nfs://nfs_01/block1")});
    CacheLocationConstPtr location2 = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("TP0", "nfs://nfs_01/block2")});
    CacheLocationConstPtr location3 = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("TP0", "nfs://nfs_01/block3")});

    // 分别添加location到同一个block_key
    std::vector<std::string> location_ids1, location_ids2, location_ids3;
    ASSERT_EQ(ErrorCode::EC_OK,
              BatchAddLocationForTest(&meta_searcher, request_context.get(), {400}, {location1}, location_ids1));
    ASSERT_EQ(ErrorCode::EC_OK,
              BatchAddLocationForTest(&meta_searcher, request_context.get(), {400}, {location2}, location_ids2));
    ASSERT_EQ(ErrorCode::EC_OK,
              BatchAddLocationForTest(&meta_searcher, request_context.get(), {400}, {location3}, location_ids3));

    // 验证数据已添加
    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {400}, empty_mask, location_maps));
    ASSERT_FALSE(location_maps.empty());
    ASSERT_EQ(1, location_maps.size());    // 应该只有一个block_key
    ASSERT_EQ(3, location_maps[0].size()); // 但包含三个location

    // 提交删除任务，设置延迟确保在检查状态时任务还没开始执行
    CacheMetaDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = {400},
        .delay = std::chrono::milliseconds(1000),
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
        const auto &location = *loc_kv.second;
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
    CacheLocationConstPtr location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("test_loc", "nfs://nfs_01/test_block_for_storage_delete")});

    // 添加location
    std::vector<std::string> location_ids;
    ASSERT_EQ(ErrorCode::EC_OK,
              BatchAddLocationForTest(&meta_searcher, request_context.get(), {300}, {location}, location_ids));

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
    CacheLocationConstPtr location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("test_loc", "nfs://nfs_01/test_block_for_delay")});

    // 添加location
    std::vector<std::string> location_ids;
    ASSERT_EQ(ErrorCode::EC_OK,
              BatchAddLocationForTest(&meta_searcher, request_context.get(), {500}, {location}, location_ids));

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
        CacheLocationConstPtr location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
            DataStorageType::DATA_STORAGE_TYPE_NFS,
            1,
            {SchedulePlanExecutorTestHelper::CreateLocationSpec(
                "test_loc", "nfs://nfs_01/test_block_for_delay_order_" + std::to_string(i))});

        // 添加location
        std::vector<std::string> location_ids;
        ASSERT_EQ(ErrorCode::EC_OK,
                  BatchAddLocationForTest(&meta_searcher, request_context.get(), {600 + i}, {location}, location_ids));

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
    CacheLocationConstPtr location1 = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("TP0", "nfs://nfs_01/block100")});
    CacheLocationConstPtr location2 = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("TP0", "nfs://nfs_01/block101")});
    CacheLocationConstPtr location3 = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("TP0", "nfs://nfs_01/block102")});

    // 分别添加location到同一个block_key
    std::vector<std::string> location_ids1, location_ids2, location_ids3;
    ASSERT_EQ(ErrorCode::EC_OK,
              BatchAddLocationForTest(&meta_searcher, request_context.get(), {700}, {location1}, location_ids1));
    ASSERT_EQ(ErrorCode::EC_OK,
              BatchAddLocationForTest(&meta_searcher, request_context.get(), {700}, {location2}, location_ids2));
    ASSERT_EQ(ErrorCode::EC_OK,
              BatchAddLocationForTest(&meta_searcher, request_context.get(), {700}, {location3}, location_ids3));

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
    const CacheLocationStatus untouched_location_status = location_maps[0].at(original_location_ids[2])->status();

    // 提交删除特定location的任务
    CacheLocationDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = {700},
        .location_ids = {std::vector<std::string>{original_location_ids[0],
                                                  original_location_ids[1]}}, // 删除前两个location
        .delay = std::chrono::milliseconds(1000),                             // 添加延迟方便DELETING检查
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
        const auto &location = *loc_kv.second;
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

    // Snapshot cleanup carries the serialized location observed by its scan.
    // If the stable location has since been replaced, the stale cleanup must
    // not mark the refreshed value as deleting.
    CacheLocationDelRequest stale_conditional_request{
        .instance_id = kTestInstanceName,
        .block_keys = {700},
        .location_ids = {{original_location_ids[2]}},
        .delay = std::chrono::seconds(0),
        .expected_location_values = {{"value-observed-before-refresh"}},
    };
    auto stale_result = executor.Submit(stale_conditional_request).get();
    ASSERT_EQ(ErrorCode::EC_OK, stale_result.status);

    location_maps.clear();
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {700}, empty_mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    ASSERT_EQ(1u, location_maps[0].size());
    ASSERT_EQ(untouched_location_status, location_maps[0].at(original_location_ids[2])->status());
}

TEST_F(SchedulePlanExecutorTest, TestMetadataOnlyLocationDeleteSkipsPhysicalBackend) {
    ASSERT_EQ(EC_OK, CreateMetaIndexer(kTestInstanceName, "local"));

    class CountingDeleteBackend : public DataStorageBackend {
    public:
        explicit CountingDeleteBackend(std::atomic<size_t> &delete_calls)
            : DataStorageBackend(nullptr), delete_calls_(delete_calls) {
            config_.set_type(DataStorageType::DATA_STORAGE_TYPE_DUMMY);
            config_.set_global_unique_name("external_cache");
            SetOpen(true);
            SetAvailable(true);
        }
        DataStorageType GetType() override { return DataStorageType::DATA_STORAGE_TYPE_DUMMY; }
        bool Available() override { return true; }
        double GetStorageUsageRatio(const std::string &) const override { return 0.0; }
        const StorageConfig &GetStorageConfig() override { return config_; }
        ErrorCode DoOpen(const StorageConfig &, const std::string &) override { return EC_OK; }
        ErrorCode Close() override { return EC_OK; }
        std::vector<std::pair<ErrorCode, DataStorageUri>>
        Create(const std::vector<std::string> &, size_t, const std::string &, std::function<void()>) override {
            return {};
        }
        std::vector<ErrorCode>
        Delete(const std::vector<DataStorageUri> &uris, const std::string &, std::function<void()>) override {
            ++delete_calls_;
            return std::vector<ErrorCode>(uris.size(), EC_OK);
        }
        std::vector<bool> Exist(const std::vector<DataStorageUri> &uris) override {
            return std::vector<bool>(uris.size(), true);
        }
        std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &uris) override {
            return std::vector<ErrorCode>(uris.size(), EC_OK);
        }
        std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &uris) override {
            return std::vector<ErrorCode>(uris.size(), EC_OK);
        }

    private:
        std::atomic<size_t> &delete_calls_;
        StorageConfig config_;
    };

    std::atomic<size_t> delete_calls{0};
    data_storage_manager_->storage_map_["external_cache"] = std::make_shared<CountingDeleteBackend>(delete_calls);

    auto request_context = std::make_shared<RequestContext>("metadata_only_delete");
    MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));
    const int64_t block_key = 701;
    auto location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec(
            "tp0", "event_report://external_cache/cache/block701?s_version=11111111111111111111111111111111")});
    std::vector<std::string> location_ids;
    ASSERT_EQ(EC_OK,
              BatchAddLocationForTest(&meta_searcher, request_context.get(), {block_key}, {location}, location_ids));
    ASSERT_EQ(1u, location_ids.size());

    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);
    CacheLocationDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = {block_key},
        .location_ids = {{location_ids.front()}},
        .metadata_only = true,
    };
    const auto result = executor.Submit(request).get();
    ASSERT_EQ(EC_OK, result.status);
    EXPECT_EQ(0u, delete_calls.load());

    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    ASSERT_EQ(EC_OK, meta_searcher.BatchGetLocation(request_context.get(), {block_key}, empty_mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    EXPECT_TRUE(location_maps.front().empty());

    const int64_t control_block_key = 702;
    std::vector<std::string> control_location_ids;
    ASSERT_EQ(EC_OK,
              BatchAddLocationForTest(
                  &meta_searcher, request_context.get(), {control_block_key}, {location}, control_location_ids));
    ASSERT_EQ(1u, control_location_ids.size());
    CacheLocationDelRequest control_request{
        .instance_id = kTestInstanceName,
        .block_keys = {control_block_key},
        .location_ids = {{control_location_ids.front()}},
    };
    const auto control_result = executor.Submit(control_request).get();
    ASSERT_EQ(EC_OK, control_result.status);
    EXPECT_EQ(1u, delete_calls.load());
}

TEST_F(SchedulePlanExecutorTest, TestEventReportMetadataDeleteRevalidatesTokenOnWorker) {
    ASSERT_EQ(EC_OK, CreateMetaIndexer(kTestInstanceName, "local"));
    auto backend = CreateEventReportStorage("event_report_l2");
    ASSERT_TRUE(backend);

    const ReporterSnapshotKey reporter{kTestInstanceName, "127.0.0.1:8080"};
    ASSERT_EQ(EC_OK, backend->RegisterNode(reporter.instance_id, reporter.host_ip_port, {"mem"}));
    uint64_t lifecycle_generation = 0;
    ASSERT_EQ(EC_OK,
              backend->UnregisterNodeForHostDown(
                  reporter.instance_id, reporter.host_ip_port, lifecycle_generation));

    const KeyVector keys{703};
    const std::string location_id = backend->BuildLocationId("mem", reporter.host_ip_port);
    MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));
    RequestContext context("event_report_executor_test");
    std::vector<std::vector<MetaSearcher::ReplaceLocationSpecsTask>> replace_tasks = {{
        {location_id,
         backend->GetStorageType(),
         CLS_SERVING,
         {LocationSpec("tp0", "event_report://127.0.0.1:8080/mem?size=11")}},
    }};
    std::vector<ErrorCode> per_key_ec;
    ASSERT_EQ(EC_OK,
              meta_searcher.BatchReplaceLocationSpecs(
                  &context, keys, replace_tasks, per_key_ec));

    std::vector<CacheLocationMap> locations;
    BlockMask empty_mask;
    ASSERT_EQ(EC_OK, meta_searcher.BatchGetLocation(&context, keys, empty_mask, locations));
    const std::string expected_value = locations.front().at(location_id)->ToJsonString();

    EventReportMetadataDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = keys,
        .targets = {{EventReportMetadataDeleteTarget{
            .location_id = location_id,
            .expected_location_value = expected_value,
            .backend_unique_name = "event_report_l2",
            .storage_type = backend->GetStorageType(),
            .expected_backend = backend,
            .cleanup_token = EventReportBackend::MaintenanceCleanupToken{
                .reason = EventReportBackend::MaintenanceCleanupReason::kDownHost,
                .reporter_key = reporter,
                .lifecycle_generation = lifecycle_generation,
            },
        }}},
    };

    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);
    std::promise<void> blocker_started;
    std::promise<void> release_blocker;
    const auto release_future = release_blocker.get_future().share();
    ASSERT_TRUE(executor.SubmitTask([&blocker_started, release_future]() {
        blocker_started.set_value();
        release_future.wait();
    }));
    ASSERT_EQ(std::future_status::ready, blocker_started.get_future().wait_for(std::chrono::seconds(1)));
    auto submit_result = executor.SubmitAsync(request);
    ASSERT_TRUE(submit_result.accepted);
    ASSERT_TRUE(submit_result.future.valid());
    EXPECT_EQ(std::future_status::timeout, submit_result.future.wait_for(std::chrono::milliseconds(10)));
    release_blocker.set_value();
    EXPECT_EQ(EC_OK, submit_result.future.get().status);

    locations.clear();
    ASSERT_EQ(EC_OK, meta_searcher.BatchGetLocation(&context, keys, empty_mask, locations));
    ASSERT_EQ(1u, locations.size());
    EXPECT_TRUE(locations.front().empty());
    EXPECT_EQ(1,
              metrics_registry_
                  ->GetCounter("cache_gc.event_report_delete_location_count",
                               {{"reason", "down_host"}, {"status", "deleted"}})
                  .Get());
}

TEST_F(SchedulePlanExecutorTest, TestEventReportMetadataDeleteSkipsStaleLifecycleToken) {
    ASSERT_EQ(EC_OK, CreateMetaIndexer(kTestInstanceName, "local"));
    auto backend = CreateEventReportStorage("event_report_l2_stale");
    ASSERT_TRUE(backend);

    const ReporterSnapshotKey reporter{kTestInstanceName, "127.0.0.2:8080"};
    ASSERT_EQ(EC_OK, backend->RegisterNode(reporter.instance_id, reporter.host_ip_port, {"mem"}));
    uint64_t old_generation = 0;
    ASSERT_EQ(EC_OK,
              backend->UnregisterNodeForHostDown(reporter.instance_id, reporter.host_ip_port, old_generation));

    const KeyVector keys{704};
    const std::string location_id = backend->BuildLocationId("mem", reporter.host_ip_port);
    MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));
    RequestContext context("event_report_executor_stale_test");
    std::vector<std::vector<MetaSearcher::ReplaceLocationSpecsTask>> replace_tasks = {{
        {location_id,
         backend->GetStorageType(),
         CLS_SERVING,
         {LocationSpec("tp0", "event_report://127.0.0.2:8080/mem?size=13")}},
    }};
    std::vector<ErrorCode> per_key_ec;
    ASSERT_EQ(EC_OK,
              meta_searcher.BatchReplaceLocationSpecs(
                  &context, keys, replace_tasks, per_key_ec));
    std::vector<CacheLocationMap> locations;
    BlockMask empty_mask;
    ASSERT_EQ(EC_OK, meta_searcher.BatchGetLocation(&context, keys, empty_mask, locations));
    const std::string expected_value = locations.front().at(location_id)->ToJsonString();

    EventReportMetadataDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = keys,
        .targets = {{EventReportMetadataDeleteTarget{
            .location_id = location_id,
            .expected_location_value = expected_value,
            .backend_unique_name = "event_report_l2_stale",
            .storage_type = backend->GetStorageType(),
            .expected_backend = backend,
            .cleanup_token = EventReportBackend::MaintenanceCleanupToken{
                .reason = EventReportBackend::MaintenanceCleanupReason::kDownHost,
                .reporter_key = reporter,
                .lifecycle_generation = old_generation,
            },
        }}},
    };

    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);
    std::promise<void> blocker_started;
    std::promise<void> release_blocker;
    const auto release_future = release_blocker.get_future().share();
    ASSERT_TRUE(executor.SubmitTask([&blocker_started, release_future]() {
        blocker_started.set_value();
        release_future.wait();
    }));
    ASSERT_EQ(std::future_status::ready, blocker_started.get_future().wait_for(std::chrono::seconds(1)));
    auto submit_result = executor.SubmitAsync(request);
    ASSERT_TRUE(submit_result.accepted);
    ASSERT_EQ(EC_OK, backend->RegisterNode(reporter.instance_id, reporter.host_ip_port, {"mem"}));
    release_blocker.set_value();
    EXPECT_EQ(EC_OK, submit_result.future.get().status);

    locations.clear();
    ASSERT_EQ(EC_OK, meta_searcher.BatchGetLocation(&context, keys, empty_mask, locations));
    ASSERT_EQ(1u, locations.size());
    EXPECT_EQ(1u, locations.front().count(location_id));
    EXPECT_EQ(1,
              metrics_registry_
                  ->GetCounter("cache_gc.event_report_delete_location_count",
                               {{"reason", "down_host"}, {"status", "mismatch"}})
                  .Get());
}

TEST_F(SchedulePlanExecutorTest, TestEventReportMetadataDeleteRejectsInvalidRequestWithoutFuture) {
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);
    EventReportMetadataDelRequest request{.instance_id = kTestInstanceName, .block_keys = {1}, .targets = {}};
    auto submit_result = executor.SubmitAsync(request);
    EXPECT_FALSE(submit_result.accepted);
    EXPECT_FALSE(submit_result.future.valid());
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
    CacheLocationConstPtr location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("test_loc", "nfs://nfs_01/test_block_for_location_delay")});

    // 添加location
    std::vector<std::string> location_ids;
    ASSERT_EQ(ErrorCode::EC_OK,
              BatchAddLocationForTest(&meta_searcher, request_context.get(), {800}, {location}, location_ids));

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

TEST_F(SchedulePlanExecutorTest, TestSubmitAsyncAdmissionRunsOnWorkerAndReturnsQuickly) {
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer(kTestInstanceName, "local"));
    ASSERT_EQ(ErrorCode::EC_OK, CreateDataStorage());

    auto request_context = std::make_shared<RequestContext>("async_admission_test");
    MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));
    auto location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("test_loc", "nfs://nfs_01/async_admission?size=1")});
    std::vector<std::string> location_ids;
    ASSERT_EQ(ErrorCode::EC_OK,
              BatchAddLocationForTest(&meta_searcher, request_context.get(), {900}, {location}, location_ids));
    ASSERT_EQ(1, location_ids.size());
    std::vector<CacheLocationMap> initial_location_maps;
    BlockMask empty_mask;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {900}, empty_mask, initial_location_maps));
    ASSERT_EQ(1, initial_location_maps.size());
    const auto initial_status = initial_location_maps.front().at(location_ids.front())->status();

    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);
    std::promise<void> blocker_started;
    std::promise<void> release_blocker;
    auto release_blocker_future = release_blocker.get_future().share();
    ASSERT_TRUE(executor.SubmitTask([&blocker_started, release_blocker_future] {
        blocker_started.set_value();
        release_blocker_future.wait();
    }));
    ASSERT_EQ(std::future_status::ready, blocker_started.get_future().wait_for(std::chrono::seconds(1)));

    CacheLocationDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = {900},
        .location_ids = {{location_ids.front()}},
    };
    const auto begin = std::chrono::steady_clock::now();
    auto submit_result = executor.SubmitAsync(request);
    const auto submit_cost = std::chrono::steady_clock::now() - begin;
    ASSERT_TRUE(submit_result.accepted);
    ASSERT_TRUE(submit_result.future.valid());
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(submit_cost).count(), 100);

    std::vector<CacheLocationMap> location_maps;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {900}, empty_mask, location_maps));
    ASSERT_EQ(initial_status, location_maps.front().at(location_ids.front())->status());

    release_blocker.set_value();
    const auto result = submit_result.future.get();
    EXPECT_TRUE(result.status == ErrorCode::EC_OK || result.status == ErrorCode::EC_PARTIAL_OK) << result.error_message;
}

TEST_F(SchedulePlanExecutorTest, TestSubmitAsyncMetaSelectsAllLocationsAndSkipsDeleting) {
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer(kTestInstanceName, "local"));
    ASSERT_EQ(ErrorCode::EC_OK, CreateDataStorage());

    auto request_context = std::make_shared<RequestContext>("async_meta_selection_test");
    MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));
    std::vector<std::string> location_ids;
    for (int location_idx = 0; location_idx < 2; ++location_idx) {
        auto location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
            DataStorageType::DATA_STORAGE_TYPE_NFS,
            1,
            {SchedulePlanExecutorTestHelper::CreateLocationSpec("test_loc_" + std::to_string(location_idx),
                                                                "nfs://nfs_01/async_meta_selection_" +
                                                                    std::to_string(location_idx) + "?size=1")});
        std::vector<std::string> added_location_ids;
        ASSERT_EQ(
            ErrorCode::EC_OK,
            BatchAddLocationForTest(&meta_searcher, request_context.get(), {903}, {location}, added_location_ids));
        ASSERT_EQ(1, added_location_ids.size());
        location_ids.push_back(added_location_ids.front());
    }

    Stub stub;
    sync_entered.store(false);
    sync_completed.store(false);
    release_sync.store(true);
    sync_delay_ms.store(0);
    sync_thread_hash.store(0);
    stub.set(ADDR(MetaIndexer, Sync), MetaIndexer_Sync_stub);

    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);
    CacheMetaDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = {903},
        .delay = std::chrono::seconds(10),
    };
    const auto caller_thread_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto first_submit_result = executor.SubmitAsync(request);
    ASSERT_TRUE(first_submit_result.accepted);

    const auto sync_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!sync_completed.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < sync_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(sync_completed.load(std::memory_order_acquire));
    EXPECT_NE(caller_thread_hash, sync_thread_hash.load(std::memory_order_relaxed));
    EXPECT_EQ(std::future_status::timeout, first_submit_result.future.wait_for(std::chrono::milliseconds(10)));

    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {903}, empty_mask, location_maps));
    ASSERT_EQ(1, location_maps.size());
    ASSERT_EQ(2, location_maps.front().size());
    for (const auto &location_id : location_ids) {
        ASSERT_EQ(CacheLocationStatus::CLS_DELETING, location_maps.front().at(location_id)->status());
    }

    sync_entered.store(false, std::memory_order_release);
    auto second_submit_result = executor.SubmitAsync(request);
    ASSERT_TRUE(second_submit_result.accepted);
    ASSERT_EQ(std::future_status::ready, second_submit_result.future.wait_for(std::chrono::seconds(1)));
    EXPECT_EQ(ErrorCode::EC_OK, second_submit_result.future.get().status);
    EXPECT_FALSE(sync_entered.load(std::memory_order_acquire));

    executor.Stop();
    ASSERT_EQ(std::future_status::ready, first_submit_result.future.wait_for(std::chrono::seconds(1)));
    EXPECT_EQ(ErrorCode::EC_ERROR, first_submit_result.future.get().status);
    stub.reset(ADDR(MetaIndexer, Sync));
}

TEST_F(SchedulePlanExecutorTest, TestSubmitAsyncMetaAdmissionCancelledOnStop) {
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);
    std::promise<void> blocker_started;
    std::promise<void> release_blocker;
    auto release_blocker_future = release_blocker.get_future().share();
    ASSERT_TRUE(executor.SubmitTask([&blocker_started, release_blocker_future] {
        blocker_started.set_value();
        release_blocker_future.wait();
    }));
    ASSERT_EQ(std::future_status::ready, blocker_started.get_future().wait_for(std::chrono::seconds(1)));

    const auto submit_begin = std::chrono::steady_clock::now();
    auto submit_result = executor.SubmitAsync(CacheMetaDelRequest{
        .instance_id = kTestInstanceName,
        .block_keys = {904},
    });
    const auto submit_cost = std::chrono::steady_clock::now() - submit_begin;
    ASSERT_TRUE(submit_result.accepted);
    ASSERT_TRUE(submit_result.future.valid());
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(submit_cost).count(), 100);

    std::thread stop_thread([&executor]() { executor.Stop(); });
    const auto future_status = submit_result.future.wait_for(std::chrono::seconds(1));
    release_blocker.set_value();
    stop_thread.join();

    ASSERT_EQ(std::future_status::ready, future_status);
    const auto result = submit_result.future.get();
    EXPECT_EQ(ErrorCode::EC_ERROR, result.status);
    EXPECT_NE(std::string::npos, result.error_message.find("before delete admission"));
}

TEST_F(SchedulePlanExecutorTest, TestSubmitAsyncDelayStartsAfterSyncAndDoesNotOccupyWorker) {
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer(kTestInstanceName, "local"));
    ASSERT_EQ(ErrorCode::EC_OK, CreateDataStorage());

    auto request_context = std::make_shared<RequestContext>("async_delay_test");
    MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));
    auto location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("test_loc", "nfs://nfs_01/async_delay?size=1")});
    std::vector<std::string> location_ids;
    ASSERT_EQ(ErrorCode::EC_OK,
              BatchAddLocationForTest(&meta_searcher, request_context.get(), {901}, {location}, location_ids));

    Stub stub;
    sync_entered.store(false);
    sync_completed.store(false);
    release_sync.store(true);
    sync_delay_ms.store(150);
    sync_thread_hash.store(0);
    stub.set(ADDR(MetaIndexer, Sync), MetaIndexer_Sync_stub);

    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);
    CacheLocationDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = {901},
        .location_ids = {{location_ids.front()}},
        .delay = std::chrono::milliseconds(250),
    };
    const auto caller_thread_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
    const auto begin = std::chrono::steady_clock::now();
    auto submit_result = executor.SubmitAsync(request);
    const auto submit_cost = std::chrono::steady_clock::now() - begin;
    ASSERT_TRUE(submit_result.accepted);
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(submit_cost).count(), 100);

    const auto sync_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!sync_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < sync_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(sync_entered.load(std::memory_order_acquire));
    EXPECT_NE(caller_thread_hash, sync_thread_hash.load(std::memory_order_relaxed));

    const auto deleting_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    bool deleting = false;
    while (std::chrono::steady_clock::now() < deleting_deadline) {
        std::vector<CacheLocationMap> location_maps;
        BlockMask empty_mask;
        ASSERT_EQ(ErrorCode::EC_OK,
                  meta_searcher.BatchGetLocation(request_context.get(), {901}, empty_mask, location_maps));
        deleting = !location_maps.empty() && !location_maps.front().empty() &&
                   location_maps.front().at(location_ids.front())->status() == CacheLocationStatus::CLS_DELETING;
        if (deleting) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(deleting);

    const auto sync_completed_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!sync_completed.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < sync_completed_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(sync_completed.load(std::memory_order_acquire));

    std::promise<void> immediate_task_done;
    auto immediate_task_future = immediate_task_done.get_future();
    ASSERT_TRUE(executor.SubmitTask([&immediate_task_done] { immediate_task_done.set_value(); }));
    EXPECT_EQ(std::future_status::ready, immediate_task_future.wait_for(std::chrono::milliseconds(100)));

    const auto result = submit_result.future.get();
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    EXPECT_TRUE(result.status == ErrorCode::EC_OK || result.status == ErrorCode::EC_PARTIAL_OK) << result.error_message;
    EXPECT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 380);
    stub.reset(ADDR(MetaIndexer, Sync));
}

TEST_F(SchedulePlanExecutorTest, TestSubmitAsyncCompletesFailureAndExceptionPaths) {
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);
    CacheLocationDelRequest request{
        .instance_id = "missing_instance",
        .block_keys = {1},
        .location_ids = {{"location"}},
    };

    auto missing_result = executor.SubmitAsync(request);
    ASSERT_TRUE(missing_result.accepted);
    ASSERT_EQ(std::future_status::ready, missing_result.future.wait_for(std::chrono::seconds(1)));
    EXPECT_EQ(ErrorCode::EC_NOENT, missing_result.future.get().status);

    auto missing_meta_result = executor.SubmitAsync(CacheMetaDelRequest{
        .instance_id = "missing_instance",
        .block_keys = {1},
    });
    ASSERT_TRUE(missing_meta_result.accepted);
    ASSERT_EQ(std::future_status::ready, missing_meta_result.future.wait_for(std::chrono::seconds(1)));
    EXPECT_EQ(ErrorCode::EC_NOENT, missing_meta_result.future.get().status);

    Stub stub;
    stub.set(ADDR(MetaIndexerManager, GetMetaIndexer), MetaIndexerManager_GetMetaIndexer_throw_stub);
    auto exception_result = executor.SubmitAsync(request);
    ASSERT_TRUE(exception_result.accepted);
    ASSERT_EQ(std::future_status::ready, exception_result.future.wait_for(std::chrono::seconds(1)));
    const auto result = exception_result.future.get();
    EXPECT_EQ(ErrorCode::EC_ERROR, result.status);
    EXPECT_NE(std::string::npos, result.error_message.find("injected GetMetaIndexer exception"));

    auto meta_exception_result = executor.SubmitAsync(CacheMetaDelRequest{
        .instance_id = "missing_instance",
        .block_keys = {1},
    });
    ASSERT_TRUE(meta_exception_result.accepted);
    ASSERT_EQ(std::future_status::ready, meta_exception_result.future.wait_for(std::chrono::seconds(1)));
    const auto meta_result = meta_exception_result.future.get();
    EXPECT_EQ(ErrorCode::EC_ERROR, meta_result.status);
    EXPECT_NE(std::string::npos, meta_result.error_message.find("injected GetMetaIndexer exception"));
    stub.reset(ADDR(MetaIndexerManager, GetMetaIndexer));
}

TEST_F(SchedulePlanExecutorTest, TestSubmitAsyncSecondEnqueueFailureCompletesFuture) {
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer(kTestInstanceName, "local"));
    ASSERT_EQ(ErrorCode::EC_OK, CreateDataStorage());

    auto request_context = std::make_shared<RequestContext>("async_second_enqueue_test");
    MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));
    auto location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("test_loc", "nfs://nfs_01/second_enqueue?size=1")});
    std::vector<std::string> location_ids;
    ASSERT_EQ(ErrorCode::EC_OK,
              BatchAddLocationForTest(&meta_searcher, request_context.get(), {902}, {location}, location_ids));

    Stub stub;
    sync_entered.store(false);
    release_sync.store(false);
    sync_delay_ms.store(0);
    stub.set(ADDR(MetaIndexer, Sync), MetaIndexer_Sync_stub);

    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);
    CacheLocationDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = {902},
        .location_ids = {{location_ids.front()}},
        .delay = std::chrono::milliseconds(10),
    };
    auto submit_result = executor.SubmitAsync(request);
    ASSERT_TRUE(submit_result.accepted);

    const auto sync_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!sync_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < sync_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(sync_entered.load(std::memory_order_acquire));
    executor.stop_.store(true);
    executor.condition_.notify_all();
    release_sync.store(true, std::memory_order_release);

    ASSERT_EQ(std::future_status::ready, submit_result.future.wait_for(std::chrono::seconds(1)));
    const auto result = submit_result.future.get();
    EXPECT_EQ(ErrorCode::EC_ERROR, result.status);
    EXPECT_NE(std::string::npos, result.error_message.find("physical delete task"));
    executor.Stop();
    stub.reset(ADDR(MetaIndexer, Sync));
}

TEST_F(SchedulePlanExecutorTest, TestSubmitAsyncRejectedHasNoFuture) {
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);
    executor.Stop();
    const auto location_submit_result = executor.SubmitAsync(CacheLocationDelRequest{});
    EXPECT_FALSE(location_submit_result.accepted);
    EXPECT_FALSE(location_submit_result.future.valid());
    const auto meta_submit_result = executor.SubmitAsync(CacheMetaDelRequest{});
    EXPECT_FALSE(meta_submit_result.accepted);
    EXPECT_FALSE(meta_submit_result.future.valid());
}

// ===== CacheLocationCopyRequest（多层存储迁移 copy 任务）=====

TEST_F(SchedulePlanExecutorTest, TestCopyTaskSuccess) {
    std::string root = GetPrivateTestRuntimeDataPath() + "copy_dummy/";
    ASSERT_TRUE(CreateDummyStorage("dummy_01", root));
    SchedulePlanExecutor executor(2, meta_manager_, data_storage_manager_, metrics_registry_);

    // 准备源文件（含内容），目标父目录尚不存在
    std::string src = root + "src_block";
    std::string dst = root + "cold/dst_block";
    {
        std::ofstream ofs(src);
        ofs << "kvcache-bytes";
    }
    ASSERT_TRUE(std::filesystem::exists(src));

    CacheLocationCopyRequest req{
        .instance_id = kTestInstanceName,
        .block_key = 1001,
        .exec_storage_name = "dummy_01",
        .src_uris = {MakeUri(src)},
        .dst_uris = {MakeUri(dst)},
    };
    auto result = executor.Submit(req).get();
    ASSERT_EQ(ErrorCode::EC_OK, result.status) << result.error_message;
    ASSERT_TRUE(std::filesystem::exists(dst));
    ASSERT_EQ(std::filesystem::file_size(src), std::filesystem::file_size(dst));
}

TEST_F(SchedulePlanExecutorTest, TestCopyTaskSourceMissing) {
    std::string root = GetPrivateTestRuntimeDataPath() + "copy_dummy_miss/";
    ASSERT_TRUE(CreateDummyStorage("dummy_02", root));
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);

    CacheLocationCopyRequest req{
        .instance_id = kTestInstanceName,
        .block_key = 1002,
        .exec_storage_name = "dummy_02",
        .src_uris = {MakeUri(root + "no_such_src")},
        .dst_uris = {MakeUri(root + "dst")},
    };
    auto result = executor.Submit(req).get();
    // 源端缺失 -> 部分失败（新 PlanExecuteResult 仅 status）
    ASSERT_EQ(ErrorCode::EC_PARTIAL_OK, result.status);
    ASSERT_FALSE(std::filesystem::exists(root + "dst"));
}

TEST_F(SchedulePlanExecutorTest, TestCopyTaskSizeMismatch) {
    std::string root = GetPrivateTestRuntimeDataPath() + "copy_dummy_mismatch/";
    ASSERT_TRUE(CreateDummyStorage("dummy_03", root));
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);

    CacheLocationCopyRequest req{
        .instance_id = kTestInstanceName,
        .block_key = 1003,
        .exec_storage_name = "dummy_03",
        .src_uris = {MakeUri(root + "a"), MakeUri(root + "b")},
        .dst_uris = {MakeUri(root + "a_dst")}, // 数量不匹配
    };
    auto result = executor.Submit(req).get();
    ASSERT_EQ(ErrorCode::EC_BADARGS, result.status);
}

// Copy 后端返回短 vector（违反 postcondition）时，DoCopyTask 应整体判为失败，
// 防止 MigrationManager promote 不完整目标 location。
TEST_F(SchedulePlanExecutorTest, TestCopyTaskShortResultVector) {
    std::string root = GetPrivateTestRuntimeDataPath() + "copy_dummy_short/";
    ASSERT_TRUE(CreateDummyStorage("dummy_short", root));
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);

    // 注入一个 Copy 返回短 vector 的 mock 后端（覆盖真实 dummy backend）。
    class ShortCopyBackend : public DataStorageBackend {
    public:
        ShortCopyBackend() : DataStorageBackend(nullptr) {
            SetOpen(true);
            SetAvailable(true);
        }
        DataStorageType GetType() override { return DataStorageType::DATA_STORAGE_TYPE_DUMMY; }
        bool Available() override { return true; }
        double GetStorageUsageRatio(const std::string &) const override { return 0.0; }
        const StorageConfig &GetStorageConfig() override { return config_; }
        ErrorCode DoOpen(const StorageConfig &, const std::string &) override { return EC_OK; }
        ErrorCode Close() override { return EC_OK; }
        std::vector<std::pair<ErrorCode, DataStorageUri>>
        Create(const std::vector<std::string> &, size_t, const std::string &, std::function<void()>) override {
            return {};
        }
        std::vector<ErrorCode>
        Delete(const std::vector<DataStorageUri> &, const std::string &, std::function<void()>) override {
            return {};
        }
        std::vector<bool> Exist(const std::vector<DataStorageUri> &u) override {
            return std::vector<bool>(u.size(), true);
        }
        std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &u) override {
            return std::vector<ErrorCode>(u.size(), EC_OK);
        }
        std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &u) override {
            return std::vector<ErrorCode>(u.size(), EC_OK);
        }
        std::vector<ErrorCode> Copy(const std::vector<DataStorageUri> &src,
                                    const std::vector<DataStorageUri> &,
                                    const std::string &) override {
            // 故意只返回 1 个结果（输入可能 2 或更多）——违反 postcondition
            return {ErrorCode::EC_OK};
        }
        StorageConfig config_;
    };
    data_storage_manager_->storage_map_["dummy_short"] = std::make_shared<ShortCopyBackend>();

    std::string src1 = root + "src1", src2 = root + "src2";
    std::string dst1 = root + "dst1", dst2 = root + "dst2";
    CacheLocationCopyRequest req{
        .instance_id = kTestInstanceName,
        .block_key = 1005,
        .exec_storage_name = "dummy_short",
        .src_uris = {MakeUri(src1), MakeUri(src2)},
        .dst_uris = {MakeUri(dst1), MakeUri(dst2)},
    };
    auto result = executor.Submit(req).get();
    // 短 vector → 整体失败，不应是 EC_OK
    ASSERT_NE(ErrorCode::EC_OK, result.status);
    ASSERT_FALSE(result.error_message.empty());
}

TEST_F(SchedulePlanExecutorTest, TestCopyTaskStopped) {
    std::string root = GetPrivateTestRuntimeDataPath() + "copy_dummy_stop/";
    ASSERT_TRUE(CreateDummyStorage("dummy_04", root));
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);
    executor.Stop();

    CacheLocationCopyRequest req{
        .instance_id = kTestInstanceName,
        .block_key = 1004,
        .exec_storage_name = "dummy_04",
        .src_uris = {MakeUri(root + "x")},
        .dst_uris = {MakeUri(root + "y")},
    };
    auto result = executor.Submit(req).get();
    ASSERT_EQ(ErrorCode::EC_ERROR, result.status);
}

TEST_F(SchedulePlanExecutorTest, TestQueuedCopyTaskCompletesOnStop) {
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);
    CacheLocationCopyRequest req{
        .instance_id = kTestInstanceName,
        .block_key = 1006,
        .exec_storage_name = "unused_storage",
        .src_uris = {MakeUri("unused_src")},
        .dst_uris = {MakeUri("unused_dst")},
        .delay = std::chrono::hours(1),
    };

    auto future = executor.Submit(req);
    ASSERT_EQ(std::future_status::timeout, future.wait_for(std::chrono::milliseconds(10)));
    executor.Stop();

    ASSERT_EQ(std::future_status::ready, future.wait_for(std::chrono::seconds(1)));
    const auto result = future.get();
    EXPECT_EQ(ErrorCode::EC_ERROR, result.status);
    EXPECT_NE(std::string::npos, result.error_message.find("before copy task execution"));
}

TEST_F(SchedulePlanExecutorTest, TestMigrationBudgetLeavesWorkerForReclaim) {
    SchedulePlanExecutor executor(4, meta_manager_, data_storage_manager_, metrics_registry_, 2);

    std::mutex mutex;
    std::condition_variable cv;
    bool release_migrations = false;
    int running_migrations = 0;
    int completed_migrations = 0;
    int max_running_migrations = 0;

    bool all_migrations_accepted = true;
    for (int i = 0; i < 4; ++i) {
        all_migrations_accepted &= executor.SubmitTask(ScheduleTaskClass::kMigrationPrepare, [&]() {
            std::unique_lock<std::mutex> lock(mutex);
            ++running_migrations;
            max_running_migrations = std::max(max_running_migrations, running_migrations);
            cv.notify_all();
            cv.wait(lock, [&]() { return release_migrations; });
            --running_migrations;
            ++completed_migrations;
            cv.notify_all();
        });
    }

    bool budget_filled = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        budget_filled = cv.wait_for(lock, std::chrono::seconds(3), [&]() { return running_migrations == 2; });
    }

    std::promise<void> reclaim_ran;
    auto reclaim_future = reclaim_ran.get_future();
    const bool reclaim_accepted = executor.SubmitTask(ScheduleTaskClass::kReclaim, [&]() { reclaim_ran.set_value(); });
    const bool reclaim_completed =
        reclaim_accepted && reclaim_future.wait_for(std::chrono::seconds(3)) == std::future_status::ready;

    {
        std::lock_guard<std::mutex> lock(mutex);
        release_migrations = true;
    }
    cv.notify_all();
    bool all_migrations_completed = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        all_migrations_completed =
            cv.wait_for(lock, std::chrono::seconds(3), [&]() { return completed_migrations == 4; });
    }

    EXPECT_TRUE(budget_filled);
    EXPECT_TRUE(all_migrations_accepted);
    EXPECT_TRUE(reclaim_accepted);
    EXPECT_TRUE(reclaim_completed);
    EXPECT_TRUE(all_migrations_completed);
    EXPECT_EQ(2, max_running_migrations);
}

TEST_F(SchedulePlanExecutorTest, TestMigrationBudgetClampAppliesAcrossPrepareAndContinuation) {
    auto run_case = [&](unsigned int requested_budget, int expected_running) {
        auto executor = std::make_unique<SchedulePlanExecutor>(
            2, meta_manager_, data_storage_manager_, metrics_registry_, requested_budget);

        std::mutex mutex;
        std::condition_variable cv;
        bool release_migrations = false;
        int running_migrations = 0;
        int completed_migrations = 0;
        int max_running_migrations = 0;
        auto migration_task = [&]() {
            std::unique_lock<std::mutex> lock(mutex);
            ++running_migrations;
            max_running_migrations = std::max(max_running_migrations, running_migrations);
            cv.notify_all();
            cv.wait(lock, [&]() { return release_migrations; });
            --running_migrations;
            ++completed_migrations;
            cv.notify_all();
        };

        const bool prepare_accepted = executor->SubmitTask(ScheduleTaskClass::kMigrationPrepare, migration_task);
        const bool continuation_accepted =
            executor->SubmitTask(ScheduleTaskClass::kMigrationContinuation, migration_task);

        bool expected_concurrency_reached = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            expected_concurrency_reached =
                cv.wait_for(lock, std::chrono::seconds(3), [&]() { return running_migrations == expected_running; });
        }

        // With a clamped budget of one, the second worker must remain available for reclaim
        // even though one Prepare and one Continuation are both ready.
        bool reclaim_completed = true;
        if (expected_running == 1) {
            std::promise<void> reclaim_ran;
            auto reclaim_future = reclaim_ran.get_future();
            const bool reclaim_accepted =
                executor->SubmitTask(ScheduleTaskClass::kReclaim, [&]() { reclaim_ran.set_value(); });
            reclaim_completed =
                reclaim_accepted && reclaim_future.wait_for(std::chrono::seconds(3)) == std::future_status::ready;
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            release_migrations = true;
        }
        cv.notify_all();
        bool all_completed = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            all_completed = cv.wait_for(lock, std::chrono::seconds(3), [&]() { return completed_migrations == 2; });
        }

        EXPECT_TRUE(prepare_accepted);
        EXPECT_TRUE(continuation_accepted);
        EXPECT_TRUE(expected_concurrency_reached);
        EXPECT_TRUE(reclaim_completed);
        EXPECT_TRUE(all_completed);
        EXPECT_EQ(expected_running, max_running_migrations);
    };

    run_case(/*requested_budget*/ 0, /*expected_running*/ 1);
    run_case(/*requested_budget*/ 99, /*expected_running*/ 2);
    run_case(std::numeric_limits<unsigned int>::max(), /*expected_running*/ 2);
}

TEST_F(SchedulePlanExecutorTest, TestMigrationContinuationPrecedesNewPrepare) {
    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_, 1);

    std::mutex mutex;
    std::condition_variable cv;
    bool system_started = false;
    bool release_system = false;
    std::vector<std::string> execution_order;

    ASSERT_TRUE(executor.SubmitTask([&]() {
        std::unique_lock<std::mutex> lock(mutex);
        system_started = true;
        cv.notify_all();
        cv.wait(lock, [&]() { return release_system; });
    }));
    bool started = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        started = cv.wait_for(lock, std::chrono::seconds(3), [&]() { return system_started; });
    }

    const bool prepare_accepted = executor.SubmitTask(ScheduleTaskClass::kMigrationPrepare, [&]() {
        std::lock_guard<std::mutex> lock(mutex);
        execution_order.emplace_back("prepare");
        cv.notify_all();
    });
    const bool continuation_accepted = executor.SubmitTask(ScheduleTaskClass::kMigrationContinuation, [&]() {
        std::lock_guard<std::mutex> lock(mutex);
        execution_order.emplace_back("continuation");
        cv.notify_all();
    });

    {
        std::lock_guard<std::mutex> lock(mutex);
        release_system = true;
    }
    cv.notify_all();

    bool both_completed = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        both_completed = cv.wait_for(lock, std::chrono::seconds(3), [&]() { return execution_order.size() == 2; });
    }
    EXPECT_TRUE(started);
    EXPECT_TRUE(prepare_accepted);
    EXPECT_TRUE(continuation_accepted);
    ASSERT_TRUE(both_completed);
    ASSERT_EQ(2u, execution_order.size());
    EXPECT_EQ("continuation", execution_order[0]);
    EXPECT_EQ("prepare", execution_order[1]);
}

TEST_F(SchedulePlanExecutorTest, TestMigrationExceptionReleasesBudget) {
    SchedulePlanExecutor executor(2, meta_manager_, data_storage_manager_, metrics_registry_, 1);

    std::promise<void> second_ran;
    auto second_future = second_ran.get_future();
    ASSERT_TRUE(executor.SubmitTask(ScheduleTaskClass::kMigrationPrepare,
                                    []() { throw std::runtime_error("injected migration task failure"); }));
    ASSERT_TRUE(executor.SubmitTask(ScheduleTaskClass::kMigrationPrepare, [&]() { second_ran.set_value(); }));

    EXPECT_EQ(std::future_status::ready, second_future.wait_for(std::chrono::seconds(3)));
}

TEST_F(SchedulePlanExecutorTest, TestLocationRefreshWinsBeforeConditionalAsyncAdmission) {
    ASSERT_EQ(ErrorCode::EC_OK, CreateMetaIndexer(kTestInstanceName, "local"));
    ASSERT_EQ(ErrorCode::EC_OK, CreateDataStorage());

    auto request_context = std::make_shared<RequestContext>("finish_before_gc_admission_test");
    MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kTestInstanceName));
    auto location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("test_loc",
                                                            "nfs://nfs_01/finish_before_gc_admission?size=1")});
    std::vector<std::string> location_ids;
    ASSERT_EQ(ErrorCode::EC_OK,
              BatchAddLocationForTest(&meta_searcher, request_context.get(), {912}, {location}, location_ids));
    ASSERT_EQ(1, location_ids.size());

    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {912}, empty_mask, location_maps));
    ASSERT_EQ(1, location_maps.size());
    ASSERT_EQ(1, location_maps.front().size());
    const std::string expected_location_value = location_maps.front().at(location_ids.front())->ToJsonString();

    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);
    std::promise<void> blocker_started;
    std::promise<void> release_blocker;
    const auto release_future = release_blocker.get_future().share();
    ASSERT_TRUE(executor.SubmitTask([&blocker_started, release_future]() {
        blocker_started.set_value();
        release_future.wait_for(std::chrono::seconds(2));
    }));
    ASSERT_EQ(std::future_status::ready, blocker_started.get_future().wait_for(std::chrono::seconds(1)));

    CacheLocationDelRequest request{
        .instance_id = kTestInstanceName,
        .block_keys = {912},
        .location_ids = {{location_ids.front()}},
        .expected_location_values = {{expected_location_value}},
    };
    auto submit_result = executor.SubmitAsync(request);
    ASSERT_TRUE(submit_result.accepted);
    ASSERT_TRUE(submit_result.future.valid());

    std::vector<std::vector<ErrorCode>> update_results;
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchUpdateLocationStatus(request_context.get(),
                                                      {912},
                                                      {{{location_ids.front(), CacheLocationStatus::CLS_SERVING}}},
                                                      update_results));

    release_blocker.set_value();
    const auto result = submit_result.future.get();
    EXPECT_EQ(ErrorCode::EC_OK, result.status) << result.error_message;

    location_maps.clear();
    ASSERT_EQ(ErrorCode::EC_OK,
              meta_searcher.BatchGetLocation(request_context.get(), {912}, empty_mask, location_maps));
    ASSERT_EQ(1, location_maps.size());
    ASSERT_EQ(1, location_maps.front().size());
    EXPECT_EQ(CacheLocationStatus::CLS_SERVING, location_maps.front().at(location_ids.front())->status());
}

TEST_F(SchedulePlanExecutorTest, TestAuthoritativeAdmissionRefreshesCachedMetadataBeforeConditionalDelete) {
    const std::string instance_id = "cached_authoritative_delete";
    const std::string persistent_path = GetPrivateTestRuntimeDataPath() + "schedule_plan_executor_cached_authoritative";
    ASSERT_EQ(EC_OK, CreateCachedMetaIndexer(instance_id, persistent_path));
    const auto indexer = meta_manager_->GetMetaIndexer(instance_id);
    WaitForCachedMetaIndexerRunning(indexer);

    auto request_context = std::make_shared<RequestContext>("cached_authoritative_delete_test");
    MetaSearcher meta_searcher(indexer);
    const int64_t block_key = 913;
    auto location = SchedulePlanExecutorTestHelper::CreateCacheLocation(
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        1,
        {SchedulePlanExecutorTestHelper::CreateLocationSpec("default",
                                                            "nfs://nfs_01/cached_authoritative_delete?size=1")});
    std::vector<std::string> location_ids;
    ASSERT_EQ(EC_OK,
              BatchAddLocationForTest(&meta_searcher, request_context.get(), {block_key}, {location}, location_ids));
    ASSERT_EQ(1u, location_ids.size());

    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    ASSERT_EQ(EC_OK, meta_searcher.BatchGetLocation(request_context.get(), {block_key}, empty_mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    const std::string expected_value = location_maps.front().at(location_ids.front())->ToJsonString();

    // Simulate a Running dual-backend instance whose authoritative metadata
    // still contains the key after the local hot-cache entry was evicted.
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), indexer->backend_manager_->cache_backend_->Delete(nullptr, {block_key}));
    location_maps.clear();
    ASSERT_EQ(EC_OK, meta_searcher.BatchGetLocation(request_context.get(), {block_key}, empty_mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    ASSERT_TRUE(location_maps.front().empty());

    SchedulePlanExecutor executor(1, meta_manager_, data_storage_manager_, metrics_registry_);
    CacheLocationDelRequest request{
        .instance_id = instance_id,
        .block_keys = {block_key},
        .location_ids = {{location_ids.front()}},
        .expected_location_values = {{expected_value}},
        .metadata_only = true,
        .authoritative_read = true,
    };
    const PlanExecuteResult result = executor.Submit(request).get();
    ASSERT_EQ(EC_OK, result.status) << result.error_message;

    CacheLocationMapVector persistent_locations;
    const auto persistent_results =
        indexer->backend_manager_->GetLocationsFromPersistent(request_context.get(), {block_key}, persistent_locations);
    ASSERT_EQ(1u, persistent_results.size());
    ASSERT_EQ(1u, persistent_locations.size());
    EXPECT_TRUE(persistent_results.front() == EC_NOENT || persistent_locations.front().empty());
}
