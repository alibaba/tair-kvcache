#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/event/event_publisher.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/manager/migration_manager.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;

class CaptureEventPublisher : public EventPublisher {
public:
    bool Init(const std::string & /*config*/) override { return true; }
    bool Publish(const std::shared_ptr<BaseEvent> &event) override {
        events.push_back(event);
        return true;
    }
    bool Stop() override { return true; }

    std::vector<std::shared_ptr<BaseEvent>> events;
};

class MigrationManagerTest : public TESTBASE {
public:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        meta_manager_ = std::make_shared<MetaIndexerManager>();
        data_storage_manager_ = std::make_shared<DataStorageManager>(metrics_registry_);
        schedule_plan_executor_ =
            std::make_shared<SchedulePlanExecutor>(2, meta_manager_, data_storage_manager_, metrics_registry_);
    }

    void TearDown() override {}

    std::shared_ptr<MetaStorageBackendConfig> ConstructMetaStorageBackendConfig() {
        auto cfg = std::make_shared<MetaStorageBackendConfig>();
        std::string local_path = GetPrivateTestRuntimeDataPath() + "migration_meta_backend";
        cfg->SetStorageUri("file://" + local_path);
        std::error_code ec;
        if (std::filesystem::exists(local_path, ec)) {
            std::remove(local_path.c_str());
        }
        return cfg;
    }

    bool CreateMetaIndexer(const std::string &instance_id) {
        auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
        meta_indexer_config->meta_storage_backend_config_ = ConstructMetaStorageBackendConfig();
        meta_indexer_config->mutex_shard_num_ = 32;
        meta_indexer_config->max_key_count_ = 10000;
        return meta_manager_->CreateMetaIndexer(instance_id, meta_indexer_config) == ErrorCode::EC_OK;
    }

    bool CreateDummyStorage(const std::string &name, const std::string &root) {
        auto spec = std::make_shared<DummyStorageSpec>();
        spec->set_root_path(root);
        spec->set_key_count_per_file(1);
        StorageConfig config;
        config.set_type(DataStorageType::DATA_STORAGE_TYPE_DUMMY);
        config.set_global_unique_name(name);
        config.set_storage_spec(spec);
        auto rc = std::make_shared<RequestContext>("test_trace_id");
        return data_storage_manager_->RegisterStorage(rc.get(), name, config) == ErrorCode::EC_OK;
    }

    // 在 hot 存储上创建一个 SERVING 的源 location，可选写入真实源文件内容。
    // 返回 src_location_id。
    std::string CreateSourceLocation(int64_t block_key,
                                     const std::string &hot_storage,
                                     bool write_file,
                                     const std::string &content) {
        auto rc = std::make_shared<RequestContext>("create_source");
        std::string key = kInstance + "/TP0/" + StringUtil::Uint64ToHex(block_key);
        auto results = data_storage_manager_->Create(rc.get(), hot_storage, {key}, content.size(), nullptr);
        EXPECT_EQ(1u, results.size());
        EXPECT_EQ(ErrorCode::EC_OK, results[0].first);
        DataStorageUri src_uri = results[0].second;

        if (write_file) {
            std::filesystem::path p = src_uri.GetPath();
            std::error_code ec;
            std::filesystem::create_directories(p.parent_path(), ec);
            std::ofstream ofs(p);
            ofs << content;
        }

        MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kInstance));
        auto loc = std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                                   1,
                                                   std::vector<LocationSpec>{LocationSpec("TP0", src_uri.ToUriString())});
        std::vector<std::string> ids;
        EXPECT_EQ(ErrorCode::EC_OK, meta_searcher.BatchAddLocation(rc.get(), {block_key}, {loc}, ids));
        EXPECT_EQ(1u, ids.size());
        // BatchAddLocation 写入 CLS_WRITING，CAS 到 SERVING 模拟一个已就绪的源副本。
        std::vector<std::vector<MetaSearcher::LocationCASTask>> cas{
            {MetaSearcher::LocationCASTask{ids[0], CLS_WRITING, CLS_SERVING}}};
        std::vector<std::vector<ErrorCode>> cas_results;
        EXPECT_EQ(ErrorCode::EC_OK, meta_searcher.BatchCASLocationStatus(rc.get(), {block_key}, cas, cas_results));
        return ids[0];
    }

    // 查询某 block_key 下某 location_id 的状态，不存在返回 CLS_NOT_FOUND。
    CacheLocationStatus GetLocationStatus(int64_t block_key, const std::string &location_id) {
        auto rc = std::make_shared<RequestContext>("get_status");
        MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kInstance));
        std::vector<CacheLocationMap> maps;
        BlockMask empty_mask;
        meta_searcher.BatchGetLocation(rc.get(), {block_key}, empty_mask, maps);
        if (maps.empty()) {
            return CLS_NOT_FOUND;
        }
        auto it = maps[0].find(location_id);
        return (it == maps[0].end() || !it->second) ? CLS_NOT_FOUND : it->second->status();
    }

    size_t LocationCount(int64_t block_key) {
        auto rc = std::make_shared<RequestContext>("loc_count");
        MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kInstance));
        std::vector<CacheLocationMap> maps;
        BlockMask empty_mask;
        meta_searcher.BatchGetLocation(rc.get(), {block_key}, empty_mask, maps);
        return maps.empty() ? 0 : maps[0].size();
    }

    std::string GetRawTieredWriteTarget(int64_t block_key) {
        auto indexer = meta_manager_->GetMetaIndexer(kInstance);
        if (indexer == nullptr) {
            return {};
        }
        RequestContext rc("get_raw_tiered_write_target");
        PropertyMapVector props;
        indexer->GetProperties(&rc, {block_key}, {MigrationManager::PROPERTY_TIERED_WRITE_TARGET}, props);
        if (props.empty()) {
            return {};
        }
        auto iter = props[0].find(MigrationManager::PROPERTY_TIERED_WRITE_TARGET);
        return iter == props[0].end() ? std::string() : iter->second;
    }

    void DeleteLocationMeta(int64_t block_key, const std::string &location_id) {
        auto rc = std::make_shared<RequestContext>("delete_location_meta");
        MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kInstance));
        std::vector<std::vector<ErrorCode>> results;
        ASSERT_EQ(ErrorCode::EC_OK, meta_searcher.BatchDeleteLocations(rc.get(), {block_key}, {{location_id}}, results));
    }

    template <typename Pred>
    bool WaitFor(Pred pred, int timeout_ms = 5000) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (pred()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return pred();
    }

    std::shared_ptr<MetaIndexerManager> meta_manager_;
    std::shared_ptr<DataStorageManager> data_storage_manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<SchedulePlanExecutor> schedule_plan_executor_;
    const std::string kInstance = "test_instance";
};

// ============ Mark 路径 ============

TEST_F(MigrationManagerTest, TestMarkLifecycle) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "ml_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "ml_cold/"));
    // 持久化方案：打标前 block 必须已存在（有 location），否则 MA_SKIP 不打标。
    CreateSourceLocation(1, "hot_01", false, "a");
    CreateSourceLocation(2, "hot_01", false, "b");
    CreateSourceLocation(3, "hot_01", false, "c");

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);

    ASSERT_FALSE(mgr.IsMarkedForTieredWrite(kInstance, 1));

    ASSERT_EQ(ErrorCode::EC_OK, mgr.MarkForTieredWrite(kInstance, {1, 2, 3}, "cold_01"));
    ASSERT_TRUE(mgr.IsMarkedForTieredWrite(kInstance, 1));
    ASSERT_TRUE(mgr.IsMarkedForTieredWrite(kInstance, 2));
    ASSERT_EQ("cold_01", mgr.GetTieredWriteTarget(kInstance, 2));
    ASSERT_EQ("cold_01", mgr.GetTieredWriteTarget(kInstance, 3));

    // property-only RMW 合并语义回归：打标只写属性，不破坏已有 location。
    ASSERT_EQ(1u, LocationCount(1));
    ASSERT_EQ(1u, LocationCount(2));
    ASSERT_EQ(1u, LocationCount(3));

    mgr.ClearTieredWriteMark(kInstance, 2);
    ASSERT_FALSE(mgr.IsMarkedForTieredWrite(kInstance, 2));
    ASSERT_TRUE(mgr.IsMarkedForTieredWrite(kInstance, 1));
    ASSERT_EQ("", mgr.GetTieredWriteTarget(kInstance, 2));
    ASSERT_EQ(1u, LocationCount(2)); // 清标同样不破坏 location

    auto stats = mgr.GetStats();
    ASSERT_EQ(3u, stats.marks_added);
    ASSERT_EQ(1u, stats.marks_cleared);
    ASSERT_EQ(2u, stats.active_marks); // best-effort = added - cleared
}

// F-02: 打标到未注册的 target storage 应失败（EC_NOENT）且不写入任何 mark，
// 避免产生"永不被满足、只能等超时"的孤儿标记。覆盖 reclaimer 打标路径。
TEST_F(MigrationManagerTest, TestMarkForTieredWriteRejectsUnregisteredTarget) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "mark_badtarget_hot/"));
    CreateSourceLocation(1, "hot_01", false, "a");

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    ASSERT_EQ(ErrorCode::EC_NOENT, mgr.MarkForTieredWrite(kInstance, {1}, "unregistered_cold"));
    ASSERT_FALSE(mgr.IsMarkedForTieredWrite(kInstance, 1));
    ASSERT_EQ("", mgr.GetTieredWriteTarget(kInstance, 1));

    auto stats = mgr.GetStats();
    ASSERT_EQ(0u, stats.marks_added);
}

TEST_F(MigrationManagerTest, TestMarkConsumedEventIncludesTargetStorage) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "mark_event_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "mark_event_cold/"));
    CreateSourceLocation(4, "hot_01", false, "d");

    auto event_manager = std::make_shared<EventManager>();
    ASSERT_TRUE(event_manager->Init());
    auto publisher = std::make_shared<CaptureEventPublisher>();
    ASSERT_TRUE(event_manager->RegisterPublisher("capture", publisher));
    MigrationManager mgr(schedule_plan_executor_,
                         meta_manager_,
                         data_storage_manager_,
                         nullptr,
                         event_manager);

    ASSERT_EQ(ErrorCode::EC_OK, mgr.MarkForTieredWrite(kInstance, {4}, "cold_01"));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    mgr.ClearTieredWriteMark(kInstance, 4);

    bool found = false;
    for (const auto &event : publisher->events) {
        if (!event || event->event_type() != "MigrationMarkConsumed") {
            continue;
        }
        found = true;
        rapidjson::Document doc;
        doc.Parse(event->ToJsonString().c_str());
        ASSERT_FALSE(doc.HasParseError());
        ASSERT_TRUE(doc.HasMember("dst_storage"));
        ASSERT_TRUE(doc["dst_storage"].IsString());
        ASSERT_STREQ("cold_01", doc["dst_storage"].GetString());
    }
    ASSERT_TRUE(found);
}

TEST_F(MigrationManagerTest, TestMarkExpiresLazilyOnLookup) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "mark_lazy_expire_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "mark_lazy_expire_cold/"));
    CreateSourceLocation(5, "hot_01", false, "e");

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    ASSERT_EQ(ErrorCode::EC_OK, mgr.MarkForTieredWrite(kInstance, {5}, "cold_01", /*timeout_ms*/ 1));
    ASSERT_TRUE(mgr.IsMarkedForTieredWrite(kInstance, 5));
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    ASSERT_FALSE(mgr.IsMarkedForTieredWrite(kInstance, 5));
    ASSERT_EQ("", GetRawTieredWriteTarget(5));
}

TEST_F(MigrationManagerTest, TestMarkExpiresByBackgroundCleanup) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "mark_bg_expire_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "mark_bg_expire_cold/"));
    CreateSourceLocation(6, "hot_01", false, "f");

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.Start();
    ASSERT_EQ(ErrorCode::EC_OK, mgr.MarkForTieredWrite(kInstance, {6}, "cold_01", /*timeout_ms*/ 10));
    ASSERT_TRUE(WaitFor([&]() { return GetRawTieredWriteTarget(6).empty(); }, 1000));
    ASSERT_FALSE(mgr.IsMarkedForTieredWrite(kInstance, 6));
    mgr.Stop();
}

TEST_F(MigrationManagerTest, TestExpiredCleanupDoesNotClearRefreshedMark) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "mark_refresh_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "mark_refresh_cold/"));
    CreateSourceLocation(7, "hot_01", false, "g");

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.Start();
    ASSERT_EQ(ErrorCode::EC_OK, mgr.MarkForTieredWrite(kInstance, {7}, "cold_01", /*timeout_ms*/ 1));
    ASSERT_EQ(ErrorCode::EC_OK, mgr.MarkForTieredWrite(kInstance, {7}, "cold_01", /*timeout_ms*/ 10000));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_TRUE(mgr.IsMarkedForTieredWrite(kInstance, 7));
    ASSERT_EQ("cold_01", mgr.GetTieredWriteTarget(kInstance, 7));
    mgr.Stop();
}

// ============ Submit 参数校验 ============

TEST_F(MigrationManagerTest, TestSubmitNonExistInstance) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    MigrationManager::MigrationRequest req;
    req.instance_id = "no_such_instance";
    req.block_key = 1;
    req.src_location_id = "loc";
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    ASSERT_EQ(ErrorCode::EC_INSTANCE_NOT_EXIST, mgr.Submit("t", req));
}

TEST_F(MigrationManagerTest, TestSubmitBadArgs) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = 1;
    // 缺 src_location_id / storages
    ASSERT_EQ(ErrorCode::EC_BADARGS, mgr.Submit("t", req));
}

TEST_F(MigrationManagerTest, TestSubmitSourceNotFound) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "snf_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "snf_cold/"));
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = 999;
    req.src_location_id = "not_exist";
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    ASSERT_EQ(ErrorCode::EC_NOENT, mgr.Submit("t", req));
}

TEST_F(MigrationManagerTest, TestSubmitRejectedBeforeStart) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "before_start_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "before_start_cold/"));

    const int64_t block_key = 99;
    const std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "before-start-copy");
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);

    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";

    ASSERT_EQ(ErrorCode::EC_ERROR, mgr.Submit("before_start_submit", req));
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
    ASSERT_EQ(1u, LocationCount(block_key));
    ASSERT_EQ(CLS_SERVING, GetLocationStatus(block_key, src_loc));
}

// ============ 状态流转（手动驱动 OnTaskSuccess/OnTaskFailed） ============

TEST_F(MigrationManagerTest, TestSubmitThenSuccessDeleteSource) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "succ_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "succ_cold/"));

    int64_t block_key = 100;
    std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "kvcache-bytes");
    ASSERT_FALSE(src_loc.empty());
    ASSERT_EQ(1u, LocationCount(block_key)); // 仅源 location

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    ASSERT_EQ(ErrorCode::EC_OK, mgr.MarkForTieredWrite(kInstance, {block_key}, "cold_01"));
    ASSERT_TRUE(mgr.IsMarkedForTieredWrite(kInstance, block_key));

    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    req.retention = MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE;

    ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", req));

    // 提交后：活跃任务存在，目标 location 已建（CLS_WRITING）。
    ASSERT_TRUE(mgr.HasMigrationTask(kInstance, block_key));
    ASSERT_EQ(1u, mgr.ActiveTaskCount());
    std::string dst_loc = mgr.GetActiveTaskDstLocation(kInstance, block_key);
    ASSERT_FALSE(dst_loc.empty());
    ASSERT_EQ(2u, LocationCount(block_key)); // 源 + 目标
    ASSERT_EQ(CLS_WRITING, GetLocationStatus(block_key, dst_loc));
    ASSERT_EQ(CLS_SERVING, GetLocationStatus(block_key, src_loc));

    // 手动驱动成功（监控线程未启动，状态流转可控）。
    mgr.OnTaskSuccess(kInstance, block_key);

    // 目标提升为 SERVING，活跃任务清除。
    ASSERT_EQ(CLS_SERVING, GetLocationStatus(block_key, dst_loc));
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
    ASSERT_FALSE(mgr.IsMarkedForTieredWrite(kInstance, block_key));

    // delete_source：源端删除任务异步提交，等待源 location 消失。
    ASSERT_TRUE(WaitFor([&]() { return GetLocationStatus(block_key, src_loc) == CLS_NOT_FOUND; }));

    auto stats = mgr.GetStats();
    ASSERT_EQ(1u, stats.copy_submitted);
    ASSERT_EQ(1u, stats.copy_completed);
}

TEST_F(MigrationManagerTest, TestSubmitThenSuccessKeepBoth) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "kb_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "kb_cold/"));

    int64_t block_key = 200;
    std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "data");
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    req.retention = MigrationRetention::MIGRATION_RETENTION_KEEP_BOTH;
    ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", req));
    std::string dst_loc = mgr.GetActiveTaskDstLocation(kInstance, block_key);

    mgr.OnTaskSuccess(kInstance, block_key);
    ASSERT_EQ(CLS_SERVING, GetLocationStatus(block_key, dst_loc));
    // keep_both：源端不删，双副本保留。
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ASSERT_EQ(CLS_SERVING, GetLocationStatus(block_key, src_loc));
    ASSERT_EQ(2u, LocationCount(block_key));
}

TEST_F(MigrationManagerTest, TestSubmitThenSuccessSourceLost) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "lost_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "lost_cold/"));

    int64_t block_key = 250;
    std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "data");
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    req.retention = MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE;
    ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", req));
    std::string dst_loc = mgr.GetActiveTaskDstLocation(kInstance, block_key);
    ASSERT_EQ(CLS_WRITING, GetLocationStatus(block_key, dst_loc));

    // Copy 字节完成后、收尾前源 location 已被其他路径删掉，目标副本应作废。
    DeleteLocationMeta(block_key, src_loc);
    ASSERT_EQ(CLS_NOT_FOUND, GetLocationStatus(block_key, src_loc));

    mgr.OnTaskSuccess(kInstance, block_key);

    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
    ASSERT_TRUE(WaitFor([&]() { return GetLocationStatus(block_key, dst_loc) == CLS_NOT_FOUND; }));
    auto stats = mgr.GetStats();
    ASSERT_EQ(1u, stats.copy_submitted);
    ASSERT_EQ(0u, stats.copy_completed);
    ASSERT_EQ(1u, stats.copy_failed);
}

TEST_F(MigrationManagerTest, TestSubmitThenFail) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "fail_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "fail_cold/"));

    int64_t block_key = 300;
    std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "data");
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    ASSERT_EQ(ErrorCode::EC_OK, mgr.MarkForTieredWrite(kInstance, {block_key}, "cold_01"));
    ASSERT_TRUE(mgr.IsMarkedForTieredWrite(kInstance, block_key));
    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", req));
    std::string dst_loc = mgr.GetActiveTaskDstLocation(kInstance, block_key);

    mgr.OnTaskFailed(kInstance, block_key, ErrorCode::EC_IO_ERROR);

    // 活跃任务清除；目标半成品异步删除；源端保留。
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
    ASSERT_TRUE(WaitFor([&]() { return GetLocationStatus(block_key, dst_loc) == CLS_NOT_FOUND; }));
    ASSERT_EQ(CLS_SERVING, GetLocationStatus(block_key, src_loc));
    ASSERT_TRUE(mgr.IsMarkedForTieredWrite(kInstance, block_key));
    ASSERT_EQ(1u, mgr.GetStats().copy_failed);
}

// ============ 端到端 copy 链路（启动监控线程 + DummyBackend 实测） ============

TEST_F(MigrationManagerTest, TestEndToEndCopySuccess) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    std::string hot_root = GetPrivateTestRuntimeDataPath() + "e2e_hot/";
    std::string cold_root = GetPrivateTestRuntimeDataPath() + "e2e_cold/";
    ASSERT_TRUE(CreateDummyStorage("hot_01", hot_root));
    ASSERT_TRUE(CreateDummyStorage("cold_01", cold_root));

    int64_t block_key = 400;
    std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "real-kvcache-payload");

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.Start();

    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    req.retention = MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE;
    ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", req));

    // 监控线程驱动：copy 完成 -> 目标 SERVING -> 删源 -> 移除活跃任务。
    ASSERT_TRUE(WaitFor([&]() { return mgr.GetStats().copy_completed == 1 && mgr.ActiveTaskCount() == 0; }));

    // 源 location 已删，目标 location SERVING 且 dst 文件已落地。
    ASSERT_TRUE(WaitFor([&]() { return GetLocationStatus(block_key, src_loc) == CLS_NOT_FOUND; }));
    ASSERT_EQ(1u, LocationCount(block_key));

    // 校验目标文件存在且内容一致。
    auto rc = std::make_shared<RequestContext>("verify");
    MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kInstance));
    std::vector<CacheLocationMap> maps;
    BlockMask empty_mask;
    meta_searcher.BatchGetLocation(rc.get(), {block_key}, empty_mask, maps);
    ASSERT_EQ(1u, maps[0].size());
    const auto &dst_location = maps[0].begin()->second;
    ASSERT_EQ(CLS_SERVING, dst_location->status());
    DataStorageUri dst_uri(dst_location->location_specs()[0].uri());
    ASSERT_TRUE(std::filesystem::exists(dst_uri.GetPath()));

    mgr.Stop();
}

TEST_F(MigrationManagerTest, TestEndToEndCopySourceMissing) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "miss_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "miss_cold/"));

    int64_t block_key = 500;
    // 源 location 元数据存在并 SERVING，但不写真实源文件 -> copy 时 EC_NOENT。
    std::string src_loc = CreateSourceLocation(block_key, "hot_01", false, "phantom");

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.Start();

    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", req));

    ASSERT_TRUE(WaitFor([&]() { return mgr.GetStats().copy_failed == 1 && mgr.ActiveTaskCount() == 0; }));
    // 源端 location 保留（失败路径不删源）。
    ASSERT_EQ(CLS_SERVING, GetLocationStatus(block_key, src_loc));
    mgr.Stop();
}

// ============ 重复提交 ============

TEST_F(MigrationManagerTest, TestSubmitDuplicate) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "dup_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "dup_cold/"));

    int64_t block_key = 700;
    std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "data");
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", req));
    // 同 block 重复提交被拒绝。
    ASSERT_EQ(ErrorCode::EC_EXIST, mgr.Submit("t", req));
    ASSERT_EQ(1u, mgr.ActiveTaskCount());
}

TEST_F(MigrationManagerTest, TestBatchSubmit) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "batch_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "batch_cold/"));

    std::vector<MigrationManager::MigrationRequest> reqs;
    for (int64_t bk = 800; bk < 803; ++bk) {
        std::string src_loc = CreateSourceLocation(bk, "hot_01", true, "data");
        MigrationManager::MigrationRequest req;
        req.instance_id = kInstance;
        req.block_key = bk;
        req.src_location_id = src_loc;
        req.src_storage_name = "hot_01";
        req.dst_storage_name = "cold_01";
        reqs.push_back(req);
    }
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    auto results = mgr.BatchSubmit("t", reqs);
    ASSERT_EQ(3u, results.size());
    for (auto ec : results) {
        ASSERT_EQ(ErrorCode::EC_OK, ec);
    }
    ASSERT_EQ(3u, mgr.ActiveTaskCount());
}

// ===== 可观测：metrics 计数/gauge + events 发布 smoke =====

TEST_F(MigrationManagerTest, TestMetricsAndEvents) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "m_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "m_cold/"));

    auto event_manager = std::make_shared<EventManager>();
    event_manager->Init();
    // 注入 metrics_registry_ + event_manager（启用可观测）
    MigrationManager mgr(schedule_plan_executor_,
                         meta_manager_,
                         data_storage_manager_,
                         metrics_registry_,
                         event_manager);

    // ---- Mark 指标 ----（持久化方案：先建出 block 1/2）
    CreateSourceLocation(1, "hot_01", false, "a");
    CreateSourceLocation(2, "hot_01", false, "b");
    mgr.MarkForTieredWrite(kInstance, {1, 2}, "cold_01");
    ASSERT_DOUBLE_EQ(2.0, metrics_registry_->GetGauge("migration.marks_active").Get());
    mgr.ClearTieredWriteMark(kInstance, 1);
    ASSERT_EQ(1u, metrics_registry_->GetCounter("migration.marks_consumed_total").Get());
    ASSERT_DOUBLE_EQ(1.0, metrics_registry_->GetGauge("migration.marks_active").Get());

    // ---- Copy 指标：submit + success ----
    mgr.DebugEnableCopySubmissionsForTest();
    const int64_t block_key = 1000;
    const std::string content = "abcdefghij"; // 10 字节
    std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, content);
    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", req));
    ASSERT_EQ(1u, metrics_registry_->GetCounter("migration.tasks_submitted_total").Get());
    ASSERT_DOUBLE_EQ(1.0, metrics_registry_->GetGauge("migration.tasks_active").Get());

    mgr.OnTaskSuccess(kInstance, block_key);
    ASSERT_EQ(1u,
              metrics_registry_->GetCounter("migration.tasks_completed_total", {{"status", "success"}}).Get());
    ASSERT_DOUBLE_EQ(0.0, metrics_registry_->GetGauge("migration.tasks_active").Get());
    ASSERT_EQ(static_cast<uint64_t>(content.size()),
              metrics_registry_->GetCounter("migration.copy_bytes_total").Get());

    // ---- 失败计数 ----
    const int64_t block_key2 = 1001;
    std::string src_loc2 = CreateSourceLocation(block_key2, "hot_01", true, "xyz");
    MigrationManager::MigrationRequest req2 = req;
    req2.block_key = block_key2;
    req2.src_location_id = src_loc2;
    ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", req2));
    mgr.OnTaskFailed(kInstance, block_key2, ErrorCode::EC_IO_ERROR);
    ASSERT_EQ(1u, metrics_registry_->GetCounter("migration.tasks_completed_total", {{"status", "failed"}}).Get());
    ASSERT_DOUBLE_EQ(0.0, metrics_registry_->GetGauge("migration.tasks_active").Get());
}

// ============ Copy 准入策略：CheckCopyAdmission ============

namespace {

CacheLocationConstPtr MakeLocation(const std::string &id,
                                   const std::string &storage_name,
                                   CacheLocationStatus status) {
    std::string uri = "dummy://" + storage_name + "/blk?size=8";
    auto loc = std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                               1,
                                               std::vector<LocationSpec>{LocationSpec("TP0", uri)});
    loc->set_id(id);
    loc->set_status(status);
    return loc;
}

CacheLocationConstPtr MakeLocationWithSpecs(const std::string &id,
                                            const std::string &storage_name,
                                            CacheLocationStatus status,
                                            const std::vector<std::string> &spec_names) {
    std::vector<LocationSpec> specs;
    specs.reserve(spec_names.size());
    for (const auto &spec_name : spec_names) {
        specs.emplace_back(spec_name, "dummy://" + storage_name + "/blk/" + spec_name + "?size=8");
    }
    auto loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY, static_cast<int64_t>(specs.size()), specs);
    loc->set_id(id);
    loc->set_status(status);
    return loc;
}

} // namespace

TEST_F(MigrationManagerTest, TestCheckCopyAdmission) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    const std::string src = "hot_01";
    const std::string dst = "cold_01";

    // kAccept：源 storage 上有 SERVING 副本，目标 storage 上没有副本。
    {
        CacheLocationMap loc_map;
        auto src_loc = MakeLocation("loc_src", src, CLS_SERVING);
        loc_map[src_loc->id()] = src_loc;
        const auto adm = mgr.CheckCopyAdmission(kInstance, /*block_key*/ 1, loc_map, src, dst);
        ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kAccept, adm.status);
        ASSERT_NE(nullptr, adm.src_location);
        ASSERT_EQ("loc_src", adm.src_location->id());
    }

    // kAlreadyMigrating：block_key 已有活跃 Copy 任务。
    {
        CacheLocationMap loc_map;
        auto src_loc = MakeLocation("loc_src", src, CLS_SERVING);
        loc_map[src_loc->id()] = src_loc;
        mgr.DebugInsertActiveCopyTask(kInstance, /*block_key*/ 2, "loc_dst");
        const auto adm = mgr.CheckCopyAdmission(kInstance, /*block_key*/ 2, loc_map, src, dst);
        ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kAlreadyMigrating, adm.status);
        ASSERT_EQ(nullptr, adm.src_location);
    }

    // kTargetServingExists：目标 storage 已存在 SERVING 副本（即使源也有）。
    {
        CacheLocationMap loc_map;
        auto src_loc = MakeLocation("loc_src", src, CLS_SERVING);
        auto dst_loc = MakeLocation("loc_dst", dst, CLS_SERVING);
        loc_map[src_loc->id()] = src_loc;
        loc_map[dst_loc->id()] = dst_loc;
        const auto adm = mgr.CheckCopyAdmission(kInstance, /*block_key*/ 3, loc_map, src, dst);
        ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kTargetServingExists, adm.status);
        ASSERT_EQ(nullptr, adm.src_location);
    }

    // kTargetWritingExists：目标 storage 上存在 WRITING 副本（可能是其他迁移半成品）。
    {
        CacheLocationMap loc_map;
        auto src_loc = MakeLocation("loc_src", src, CLS_SERVING);
        auto dst_loc = MakeLocation("loc_dst", dst, CLS_WRITING);
        loc_map[src_loc->id()] = src_loc;
        loc_map[dst_loc->id()] = dst_loc;
        const auto adm = mgr.CheckCopyAdmission(kInstance, /*block_key*/ 4, loc_map, src, dst);
        ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kTargetWritingExists, adm.status);
        ASSERT_EQ(nullptr, adm.src_location);
    }

    // kSourceServingNotFound：源 storage 上没有 SERVING 副本（仅 WRITING）。
    {
        CacheLocationMap loc_map;
        auto src_loc = MakeLocation("loc_src", src, CLS_WRITING);
        loc_map[src_loc->id()] = src_loc;
        const auto adm = mgr.CheckCopyAdmission(kInstance, /*block_key*/ 5, loc_map, src, dst);
        ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kSourceServingNotFound, adm.status);
        ASSERT_EQ(nullptr, adm.src_location);
    }
}

TEST_F(MigrationManagerTest, TestCheckCopyAdmissionAllowsPartialTarget) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    const std::string src = "hot_01";
    const std::string dst = "cold_01";

    {
        CacheLocationMap loc_map;
        auto src_l1_loc = MakeLocationWithSpecs("loc_src_l1", src, CLS_SERVING, {"tp0_L1", "tp1_L1"});
        auto dst_f0_loc = MakeLocationWithSpecs("loc_dst_f0", dst, CLS_SERVING, {"tp0_F0", "tp1_F0"});
        loc_map[src_l1_loc->id()] = src_l1_loc;
        loc_map[dst_f0_loc->id()] = dst_f0_loc;

        const auto adm = mgr.CheckCopyAdmission(kInstance, /*block_key*/ 6, loc_map, src, dst);
        ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kAccept, adm.status);
        ASSERT_NE(nullptr, adm.src_location);
        ASSERT_EQ("loc_src_l1", adm.src_location->id());
    }

    {
        CacheLocationMap loc_map;
        auto src_l1_loc = MakeLocationWithSpecs("loc_src_l1", src, CLS_SERVING, {"tp0_L1", "tp1_L1"});
        auto dst_l1_loc = MakeLocationWithSpecs("loc_dst_l1", dst, CLS_SERVING, {"tp0_L1", "tp1_L1"});
        loc_map[src_l1_loc->id()] = src_l1_loc;
        loc_map[dst_l1_loc->id()] = dst_l1_loc;

        const auto adm = mgr.CheckCopyAdmission(kInstance, /*block_key*/ 7, loc_map, src, dst);
        ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kTargetServingExists, adm.status);
        ASSERT_EQ(nullptr, adm.src_location);
    }
}

TEST_F(MigrationManagerTest, TestCheckCopyAdmissionTriesNextSourceLocation) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    const std::string src = "hot_01";
    const std::string dst = "cold_01";

    CacheLocationMap loc_map;
    auto src_f0_loc = MakeLocationWithSpecs("loc_src_f0", src, CLS_SERVING, {"tp0_F0", "tp1_F0"});
    auto src_l1_loc = MakeLocationWithSpecs("loc_src_l1", src, CLS_SERVING, {"tp0_L1", "tp1_L1"});
    auto dst_f0_loc = MakeLocationWithSpecs("loc_dst_f0", dst, CLS_SERVING, {"tp0_F0", "tp1_F0"});
    loc_map[src_f0_loc->id()] = src_f0_loc;
    loc_map[src_l1_loc->id()] = src_l1_loc;
    loc_map[dst_f0_loc->id()] = dst_f0_loc;

    const auto adm = mgr.CheckCopyAdmission(kInstance, /*block_key*/ 8, loc_map, src, dst);
    ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kAccept, adm.status);
    ASSERT_NE(nullptr, adm.src_location);
    ASSERT_EQ("loc_src_l1", adm.src_location->id());
}

TEST_F(MigrationManagerTest, TestActiveTasksScopedByInstance) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    const std::string other_instance = "other_instance";
    const std::string third_instance = "third_instance";
    const int64_t block_key = 2001;

    mgr.DebugInsertActiveCopyTask(kInstance, block_key, "loc_dst_a");
    ASSERT_TRUE(mgr.HasMigrationTask(kInstance, block_key));
    ASSERT_FALSE(mgr.HasMigrationTask(other_instance, block_key));
    ASSERT_EQ("loc_dst_a", mgr.GetActiveTaskDstLocation(kInstance, block_key));
    ASSERT_EQ("", mgr.GetActiveTaskDstLocation(other_instance, block_key));

    mgr.DebugInsertActiveCopyTask(other_instance, block_key, "loc_dst_b");
    ASSERT_TRUE(mgr.HasMigrationTask(kInstance, block_key));
    ASSERT_TRUE(mgr.HasMigrationTask(other_instance, block_key));
    ASSERT_EQ("loc_dst_a", mgr.GetActiveTaskDstLocation(kInstance, block_key));
    ASSERT_EQ("loc_dst_b", mgr.GetActiveTaskDstLocation(other_instance, block_key));
    ASSERT_EQ(2u, mgr.ActiveTaskCount());

    CacheLocationMap loc_map;
    auto src_loc = MakeLocation("loc_src", "hot_01", CLS_SERVING);
    loc_map[src_loc->id()] = src_loc;
    const auto same_instance_adm = mgr.CheckCopyAdmission(kInstance, block_key, loc_map, "hot_01", "cold_01");
    ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kAlreadyMigrating, same_instance_adm.status);
    const auto third_instance_adm = mgr.CheckCopyAdmission(third_instance, block_key, loc_map, "hot_01", "cold_01");
    ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kAccept, third_instance_adm.status);
}

// Stop 必须清空活跃任务表，否则 Leader 重新 Start 后这些 stale 条目会让对应 block
// 永远卡在 HasMigrationTask=true、目标 WRITING location 永远被 HasActiveCopyTargetLocation 保护。
TEST_F(MigrationManagerTest, TestStopDropsActiveTasks) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.Start();

    mgr.DebugInsertActiveCopyTask(kInstance, /*block_key*/ 1001, "loc_dst_a");
    mgr.DebugInsertActiveCopyTask(kInstance, /*block_key*/ 1002, "loc_dst_b");
    ASSERT_TRUE(mgr.HasMigrationTask(kInstance, 1001));
    ASSERT_TRUE(mgr.HasActiveCopyTargetLocation("loc_dst_a"));
    ASSERT_EQ(2u, mgr.ActiveTaskCount());

    mgr.Stop();

    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, 1001));
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, 1002));
    ASSERT_FALSE(mgr.HasActiveCopyTargetLocation("loc_dst_a"));
    ASSERT_FALSE(mgr.HasActiveCopyTargetLocation("loc_dst_b"));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());

    // 重新 Start 后这些 block_key 应该可以再次被识别为"无活跃任务"。
    mgr.Start();
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, 1001));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
    mgr.Stop();
}

TEST_F(MigrationManagerTest, TestSubmitRejectedAfterStop) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "stopped_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "stopped_cold/"));

    const int64_t block_key = 1003;
    const std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "stopped-copy");
    ASSERT_EQ(1u, LocationCount(block_key));

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.Start();
    mgr.Stop();

    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    req.retention = MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE;

    ASSERT_EQ(ErrorCode::EC_ERROR, mgr.Submit("stopped_submit", req));
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
    ASSERT_EQ(1u, LocationCount(block_key));
    ASSERT_EQ(CLS_SERVING, GetLocationStatus(block_key, src_loc));
}
