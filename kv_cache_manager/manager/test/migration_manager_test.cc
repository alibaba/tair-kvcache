#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
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
#include "stub.h"

using namespace kv_cache_manager;

// ---- F-10 orphan-leak 测试用 DataStorageManager::Create/Delete 存根 ----
// 复现"异构 size spec 分散到多个 create_group、某 group 失败使 block ineligible、
// 后处理 group 已成功分配的 URI 被跳过 → rollback 漏删 → orphan"的场景。
namespace f10_orphan_stub {
std::vector<DataStorageUri> g_created_uris;
std::vector<DataStorageUri> g_deleted_uris;
int g_create_call_count = 0;

std::vector<std::pair<ErrorCode, DataStorageUri>> Create_stub(void * /*obj*/,
                                                              RequestContext * /*rc*/,
                                                              const std::string & /*name*/,
                                                              const std::vector<std::string> &keys,
                                                              size_t size,
                                                              std::function<void()> /*cb*/) {
    ++g_create_call_count;
    const bool fail = (g_create_call_count == 1); // 第一个被处理的 group 失败(与迭代序无关地触发泄漏路径)
    std::vector<std::pair<ErrorCode, DataStorageUri>> results;
    for (const auto &k : keys) {
        if (fail) {
            results.emplace_back(ErrorCode::EC_ERROR, DataStorageUri{});
            continue;
        }
        DataStorageUri uri("dummy://cold_01/" + k + "?size=" + std::to_string(size));
        g_created_uris.push_back(uri);
        results.emplace_back(ErrorCode::EC_OK, uri);
    }
    return results;
}

std::vector<ErrorCode> Delete_stub(void * /*obj*/,
                                   RequestContext * /*rc*/,
                                   const std::string & /*name*/,
                                   const std::vector<DataStorageUri> &uris,
                                   std::function<void()> /*cb*/) {
    for (const auto &u : uris) {
        g_deleted_uris.push_back(u);
    }
    return std::vector<ErrorCode>(uris.size(), ErrorCode::EC_OK);
}
} // namespace f10_orphan_stub

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
                                     const std::string &content,
                                     int64_t create_time = 0) {
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
        if (create_time != 0) {
            loc->set_create_time(create_time);
        }
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

    CacheLocationConstPtr GetLocation(int64_t block_key, const std::string &location_id) {
        auto rc = std::make_shared<RequestContext>("get_location");
        MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kInstance));
        std::vector<CacheLocationMap> maps;
        BlockMask empty_mask;
        meta_searcher.BatchGetLocation(rc.get(), {block_key}, empty_mask, maps);
        if (maps.empty()) {
            return nullptr;
        }
        auto it = maps[0].find(location_id);
        return (it == maps[0].end()) ? nullptr : it->second;
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

    // F-15: 旧的无条件 ClearTieredWriteMark 已删除，测试用 helper 模拟：
    // 先查 mark 拿到 target+deadline，再做 match-clear。
    void ClearMarkForTest(MigrationManager &mgr, const std::string &instance_id, int64_t block_key) {
        std::vector<MigrationManager::MarkQueryResult> results;
        mgr.BatchGetTieredWriteTargets(instance_id, {block_key}, results);
        if (!results.empty() && !results[0].target.empty()) {
            mgr.ClearTieredWriteMarkIfMatch(instance_id, block_key, results[0].target, results[0].deadline_ms);
        }
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

    ClearMarkForTest(mgr, kInstance, 2);
    ASSERT_FALSE(mgr.IsMarkedForTieredWrite(kInstance, 2));
    ASSERT_TRUE(mgr.IsMarkedForTieredWrite(kInstance, 1));
    ASSERT_EQ("", mgr.GetTieredWriteTarget(kInstance, 2));
    ASSERT_EQ(1u, LocationCount(2)); // 清标同样不破坏 location

    auto stats = mgr.GetStats();
    ASSERT_EQ(3u, stats.marks_added);
    ASSERT_EQ(1u, stats.marks_cleared);
    ASSERT_EQ(2u, stats.active_marks); // best-effort = added - cleared
}

// F-15: match-clear 不会清掉同 block 上后来打的新 mark（不同 target 或 deadline）。
TEST_F(MigrationManagerTest, TestClearMarkDoesNotClobberNewerMark) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "clobber_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "clobber_cold_01/"));
    ASSERT_TRUE(CreateDummyStorage("cold_02", GetPrivateTestRuntimeDataPath() + "clobber_cold_02/"));
    CreateSourceLocation(1, "hot_01", false, "x");

    auto event_publisher = std::make_shared<CaptureEventPublisher>();
    auto event_manager = std::make_shared<EventManager>();
    ASSERT_TRUE(event_manager->Init());
    ASSERT_TRUE(event_manager->RegisterPublisher("capture", event_publisher));
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_, metrics_registry_, event_manager);
    mgr.Start();

    // 打 mark A: target=cold_01
    ASSERT_EQ(ErrorCode::EC_OK, mgr.MarkForTieredWrite(kInstance, {1}, "cold_01"));
    ASSERT_TRUE(mgr.IsMarkedForTieredWrite(kInstance, 1));
    // 快照 mark A 的 target+deadline（模拟 Copy 提交时的快照）
    std::vector<MigrationManager::MarkQueryResult> snap;
    mgr.BatchGetTieredWriteTargets(kInstance, {1}, snap);
    ASSERT_EQ("cold_01", snap[0].target);
    const auto old_deadline = snap[0].deadline_ms;
    ASSERT_GT(old_deadline, 0);

    // 覆盖：打 mark B: target=cold_02（模拟新一轮迁移打了不同目标的 mark）
    ASSERT_EQ(ErrorCode::EC_OK, mgr.MarkForTieredWrite(kInstance, {1}, "cold_02"));
    ASSERT_EQ("cold_02", mgr.GetTieredWriteTarget(kInstance, 1));

    // 用 mark A 的快照做 match-clear → 不应匹配（当前 mark 是 B）
    ASSERT_FALSE(mgr.ClearTieredWriteMarkIfMatch(kInstance, 1, "cold_01", old_deadline));
    // mark B 应该存活
    ASSERT_TRUE(mgr.IsMarkedForTieredWrite(kInstance, 1));
    ASSERT_EQ("cold_02", mgr.GetTieredWriteTarget(kInstance, 1));

    mgr.Stop();
}

// F-14: 部分 key 因 block 不存在被 MA_SKIP 时，marks_added / expiry / event 只计实际成功数，
// 不按 request 全量计——否则 active_marks(added-cleared) 会单调膨胀。
TEST_F(MigrationManagerTest, TestMarkStatsOnlyCountActualSuccess) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "mark_stats_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "mark_stats_cold/"));
    // 只创建 key 1 和 3 的 block；key 2 不存在 → MA_SKIP
    CreateSourceLocation(1, "hot_01", false, "a");
    CreateSourceLocation(3, "hot_01", false, "c");

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    ASSERT_EQ(ErrorCode::EC_OK, mgr.MarkForTieredWrite(kInstance, {1, 2, 3}, "cold_01"));

    // key 1, 3 成功打标；key 2 被 MA_SKIP
    ASSERT_TRUE(mgr.IsMarkedForTieredWrite(kInstance, 1));
    ASSERT_FALSE(mgr.IsMarkedForTieredWrite(kInstance, 2)); // 不存在，未打标
    ASSERT_TRUE(mgr.IsMarkedForTieredWrite(kInstance, 3));

    auto stats = mgr.GetStats();
    ASSERT_EQ(2u, stats.marks_added);      // actual=2, not request=3
    ASSERT_EQ(0u, stats.marks_cleared);
    ASSERT_EQ(2u, stats.active_marks);     // 2-0=2, 不会有幽灵 +1

    // 清掉全部两个成功的 mark → active 归零
    ClearMarkForTest(mgr, kInstance, 1);
    ClearMarkForTest(mgr, kInstance, 3);
    stats = mgr.GetStats();
    ASSERT_EQ(2u, stats.marks_added);
    ASSERT_EQ(2u, stats.marks_cleared);
    ASSERT_EQ(0u, stats.active_marks);     // 完全收敛，无漂移
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
    ClearMarkForTest(mgr, kInstance, 4);

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
    ASSERT_EQ(ErrorCode::EC_OK, mgr.MarkForTieredWrite(kInstance, {5}, "cold_01", /*timeout_ms*/ 50));
    ASSERT_TRUE(mgr.IsMarkedForTieredWrite(kInstance, 5));
    std::this_thread::sleep_for(std::chrono::milliseconds(80));

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

// R2-03: Copy 只允许消费与自身 destination 一致的 mark。写往 cold_02 的 Copy
// 即使成功，也不能清掉仍要求写往 cold_01 的独立迁移意图。
TEST_F(MigrationManagerTest, TestSubmitSuccessDoesNotClearMarkForDifferentTarget) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "mark_mismatch_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "mark_mismatch_cold_01/"));
    ASSERT_TRUE(CreateDummyStorage("cold_02", GetPrivateTestRuntimeDataPath() + "mark_mismatch_cold_02/"));

    const int64_t block_key = 150;
    const std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "data");
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();

    ASSERT_EQ(ErrorCode::EC_OK, mgr.MarkForTieredWrite(kInstance, {block_key}, "cold_01"));
    ASSERT_EQ("cold_01", mgr.GetTieredWriteTarget(kInstance, block_key));

    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_02";
    req.retention = MigrationRetention::MIGRATION_RETENTION_KEEP_BOTH;
    ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("mark_target_mismatch", req));

    const std::string dst_loc = mgr.GetActiveTaskDstLocation(kInstance, block_key);
    ASSERT_FALSE(dst_loc.empty());
    mgr.OnTaskSuccess(kInstance, block_key);

    ASSERT_EQ(CLS_SERVING, GetLocationStatus(block_key, dst_loc));
    ASSERT_EQ("cold_01", mgr.GetTieredWriteTarget(kInstance, block_key));
    ASSERT_EQ(0u, mgr.GetStats().marks_cleared);
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

// F-08: Submit 时 PrepareCopyTask 应记录 src_create_time。当源 location 的 create_time 与记录不匹配时
// (id 复用场景),OnTaskSuccess 应按 source_lost 处理并清理目标半成品。
// 验证方式:创建带非零 create_time 的源,Submit,保留源但手动篡改活跃任务中的 src_create_time
// 使其不匹配→OnTaskSuccess 应判 source_lost。
TEST_F(MigrationManagerTest, TestSubmitThenSuccessSourceCreateTimeMismatch) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "reuse_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "reuse_cold/"));

    int64_t block_key = 260;
    const int64_t original_create_time = 1000;
    std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "data", original_create_time);
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

    // 源仍在(id 命中 + SERVING),但篡改活跃任务的 src_create_time 使其不匹配——
    // 模拟"id 被复用、新 location 的 create_time 和记录不同"。
    {
        std::lock_guard<std::mutex> lock(mgr.task_mutex_);
        auto &tasks = mgr.active_tasks_by_instance_[kInstance];
        auto it = tasks.find(block_key);
        ASSERT_TRUE(it != tasks.end());
        it->second.src_create_time = original_create_time + 9999; // 篡改：不等于源实际的 1000
    }

    mgr.OnTaskSuccess(kInstance, block_key);

    // IsSourceLocationServing 找到了 src_loc(id + SERVING),但 create_time 不匹配 → source_lost。
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
    ASSERT_TRUE(WaitFor([&]() { return GetLocationStatus(block_key, dst_loc) == CLS_NOT_FOUND; }));
    // 源不应被删(retention DELETE_SOURCE 只在 promote 成功后执行,这里走 source_lost)。
    ASSERT_EQ(CLS_SERVING, GetLocationStatus(block_key, src_loc));
    ASSERT_EQ(1u, mgr.GetStats().copy_failed);
}

// F-17: create_time=0 边界——源和 ctx 都默认 0,0==0 应视为匹配,正常 promote。
TEST_F(MigrationManagerTest, TestSubmitThenSuccessSourceCreateTimeZeroMatches) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "ct0_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "ct0_cold/"));

    int64_t block_key = 270;
    // create_time=0(默认,不设)
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

    mgr.OnTaskSuccess(kInstance, block_key);

    // create_time 两边都 0 → 匹配 → 正常 promote,不走 source_lost
    std::string dst_loc = mgr.GetActiveTaskDstLocation(kInstance, block_key);
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
    ASSERT_EQ(1u, mgr.GetStats().copy_completed);
    ASSERT_EQ(0u, mgr.GetStats().copy_failed);
    ASSERT_EQ(CLS_SERVING, GetLocationStatus(block_key, src_loc));
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

// ============ F-11：Cancel 任务状态机（认领 + 延迟收尾） ============

// Cancel 只标 cancelling、不立即清理；任务与 WRITING 目标保留（slot 保护）；待完成回调收尾。
TEST_F(MigrationManagerTest, TestCancelDefersCleanupAndKeepsSlot) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "cancel_defer_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "cancel_defer_cold/"));

    const int64_t block_key = 700;
    const std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "data");
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", req));
    const std::string dst_loc = mgr.GetActiveTaskDstLocation(kInstance, block_key);
    ASSERT_EQ(CLS_WRITING, GetLocationStatus(block_key, dst_loc));

    // Cancel：标记 cancelling，不删不移除。
    ASSERT_EQ(ErrorCode::EC_OK, mgr.Cancel(kInstance, block_key));
    ASSERT_TRUE(mgr.HasMigrationTask(kInstance, block_key));            // slot 保留
    ASSERT_TRUE(mgr.HasActiveCopyTargetLocation(kInstance, block_key, dst_loc)); // 目标仍被保护
    ASSERT_EQ(CLS_WRITING, GetLocationStatus(block_key, dst_loc));      // 目标未删

    // cancelling 期间同 block 不接受新提交（slot 占用）。
    ASSERT_EQ(ErrorCode::EC_EXIST, mgr.Submit("t2", req));

    // 幂等：重复 Cancel 仍 OK。
    ASSERT_EQ(ErrorCode::EC_OK, mgr.Cancel(kInstance, block_key));
}

// Cancel 后 copy 成功：不 promote、不删源，删 WRITING 目标，记 cancelled 终态。
TEST_F(MigrationManagerTest, TestCancelThenSuccessDiscardsTarget) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "cancel_succ_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "cancel_succ_cold/"));

    const int64_t block_key = 701;
    const std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "data");
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
    const std::string dst_loc = mgr.GetActiveTaskDstLocation(kInstance, block_key);

    ASSERT_EQ(ErrorCode::EC_OK, mgr.Cancel(kInstance, block_key));
    // 完成回调认领到 cancelling → 走取消收尾。
    mgr.OnTaskSuccess(kInstance, block_key);

    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
    // 目标未提升为 SERVING，而是被删（WRITING->DELETING，异步消失）。
    ASSERT_TRUE(WaitFor([&]() { return GetLocationStatus(block_key, dst_loc) == CLS_NOT_FOUND; }));
    ASSERT_NE(CLS_SERVING, GetLocationStatus(block_key, dst_loc));
    // 源端不动。
    ASSERT_EQ(CLS_SERVING, GetLocationStatus(block_key, src_loc));
    // 记 cancelled 终态，非 completed。
    const auto stats = mgr.GetStats();
    ASSERT_EQ(1u, stats.copy_cancelled);
    ASSERT_EQ(0u, stats.copy_completed);
}

// Cancel 后 copy 失败：仍按 cancelled 收尾（用户意图优先），不计 failed。
TEST_F(MigrationManagerTest, TestCancelThenFailCountsCancelled) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "cancel_fail_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "cancel_fail_cold/"));

    const int64_t block_key = 702;
    const std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "data");
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", req));
    const std::string dst_loc = mgr.GetActiveTaskDstLocation(kInstance, block_key);

    ASSERT_EQ(ErrorCode::EC_OK, mgr.Cancel(kInstance, block_key));
    mgr.OnTaskFailed(kInstance, block_key, ErrorCode::EC_IO_ERROR);

    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
    ASSERT_TRUE(WaitFor([&]() { return GetLocationStatus(block_key, dst_loc) == CLS_NOT_FOUND; }));
    const auto stats = mgr.GetStats();
    ASSERT_EQ(1u, stats.copy_cancelled);
    ASSERT_EQ(0u, stats.copy_failed);
}

// 完成已认领(kCompleting)后 Cancel 太晚：返回 EC_EXIST，迁移照常完成。
TEST_F(MigrationManagerTest, TestCancelWhileCompletingIsTooLate) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "cancel_late_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "cancel_late_cold/"));

    const int64_t block_key = 703;
    const std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "data");
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
    const std::string dst_loc = mgr.GetActiveTaskDstLocation(kInstance, block_key);

    // 直接把状态置 kCompleting，模拟 monitor 正在收尾的窗口（测试经 -fno-access-control 访问私有）。
    {
        std::lock_guard<std::mutex> lock(mgr.task_mutex_);
        mgr.active_tasks_by_instance_[kInstance][block_key].state =
            MigrationManager::CopyTaskState::kCompleting;
    }
    ASSERT_EQ(ErrorCode::EC_EXIST, mgr.Cancel(kInstance, block_key)); // 太晚

    // 复位为 kRunning 让完成路径正常收尾（验证迁移照常完成）。
    {
        std::lock_guard<std::mutex> lock(mgr.task_mutex_);
        mgr.active_tasks_by_instance_[kInstance][block_key].state =
            MigrationManager::CopyTaskState::kRunning;
    }
    mgr.OnTaskSuccess(kInstance, block_key);
    ASSERT_EQ(CLS_SERVING, GetLocationStatus(block_key, dst_loc)); // 正常提升
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
    ASSERT_EQ(1u, mgr.GetStats().copy_completed);
}

// Cancel 不存在的任务返回 EC_NOENT。
TEST_F(MigrationManagerTest, TestCancelNonExistentReturnsNoent) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    ASSERT_EQ(ErrorCode::EC_NOENT, mgr.Cancel(kInstance, 999999));
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

// R2-03: 覆盖带 src_specs 的 BatchSubmit 主路径，确保批量 Copy 到 cold_02
// 不会消费 block 上仍指向 cold_01 的 mark。
TEST_F(MigrationManagerTest, TestBatchSubmitSuccessDoesNotClearMarkForDifferentTarget) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "batch_mark_mismatch_hot/"));
    ASSERT_TRUE(
        CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "batch_mark_mismatch_cold_01/"));
    ASSERT_TRUE(
        CreateDummyStorage("cold_02", GetPrivateTestRuntimeDataPath() + "batch_mark_mismatch_cold_02/"));

    const int64_t block_key = 850;
    const std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "batch-data");
    const auto src_location = GetLocation(block_key, src_loc);
    ASSERT_NE(nullptr, src_location);

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    ASSERT_EQ(ErrorCode::EC_OK, mgr.MarkForTieredWrite(kInstance, {block_key}, "cold_01"));

    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_create_time = src_location->create_time();
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_02";
    req.retention = MigrationRetention::MIGRATION_RETENTION_KEEP_BOTH;
    req.src_specs = src_location->location_specs(); // 非空：强制走 BatchSubmit 真批量路径。

    auto results = mgr.BatchSubmit("batch_mark_target_mismatch", {req});
    ASSERT_EQ(1u, results.size());
    ASSERT_EQ(ErrorCode::EC_OK, results[0]);

    const std::string dst_loc = mgr.GetActiveTaskDstLocation(kInstance, block_key);
    ASSERT_FALSE(dst_loc.empty());
    mgr.OnTaskSuccess(kInstance, block_key);

    ASSERT_EQ(CLS_SERVING, GetLocationStatus(block_key, dst_loc));
    ASSERT_EQ("cold_01", mgr.GetTieredWriteTarget(kInstance, block_key));
    ASSERT_EQ(0u, mgr.GetStats().marks_cleared);
}

// F-17: BatchSubmit partial failure — 部分请求有坏 src_location_id，其他成功的仍被跟踪。
TEST_F(MigrationManagerTest, TestBatchSubmitPartialFailure) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "batch_pf_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "batch_pf_cold/"));

    std::string good_loc_0 = CreateSourceLocation(900, "hot_01", true, "data0");
    std::string good_loc_2 = CreateSourceLocation(902, "hot_01", true, "data2");

    std::vector<MigrationManager::MigrationRequest> reqs;
    // req 0: good
    {
        MigrationManager::MigrationRequest r;
        r.instance_id = kInstance;
        r.block_key = 900;
        r.src_location_id = good_loc_0;
        r.src_storage_name = "hot_01";
        r.dst_storage_name = "cold_01";
        reqs.push_back(std::move(r));
    }
    // req 1: bad src_location_id
    {
        MigrationManager::MigrationRequest r;
        r.instance_id = kInstance;
        r.block_key = 901;
        r.src_location_id = "nonexistent_loc";
        r.src_storage_name = "hot_01";
        r.dst_storage_name = "cold_01";
        reqs.push_back(std::move(r));
    }
    // req 2: good
    {
        MigrationManager::MigrationRequest r;
        r.instance_id = kInstance;
        r.block_key = 902;
        r.src_location_id = good_loc_2;
        r.src_storage_name = "hot_01";
        r.dst_storage_name = "cold_01";
        reqs.push_back(std::move(r));
    }

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    auto results = mgr.BatchSubmit("t", reqs);
    ASSERT_EQ(3u, results.size());
    ASSERT_EQ(ErrorCode::EC_OK, results[0]);
    ASSERT_NE(ErrorCode::EC_OK, results[1]);
    ASSERT_EQ(ErrorCode::EC_OK, results[2]);
    ASSERT_EQ(2u, mgr.ActiveTaskCount());
    ASSERT_TRUE(mgr.HasMigrationTask(kInstance, 900));
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, 901));
    ASSERT_TRUE(mgr.HasMigrationTask(kInstance, 902));
}

// F-10 泄漏修复: 异构 size spec 分散到多个 create_group,某 group 的 Create 失败使 block
// ineligible;后处理 group 里同 block 已成功分配的 URI 不能被跳过泄漏,必须 Delete。
TEST_F(MigrationManagerTest, TestBatchSubmitHeteroSpecCreateFailNoOrphan) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "orphan_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "orphan_cold/"));

    f10_orphan_stub::g_created_uris.clear();
    f10_orphan_stub::g_deleted_uris.clear();
    f10_orphan_stub::g_create_call_count = 0;
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), f10_orphan_stub::Create_stub);
    stub.set(ADDR(DataStorageManager, Delete), f10_orphan_stub::Delete_stub);

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();

    // 单 block,两个不同 size 的 spec → 落到两个 create_group。
    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = 700;
    req.src_location_id = "src_700";
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    req.src_specs = {
        LocationSpec("tp0", "dummy://hot_01/src_700_tp0?size=111"),
        LocationSpec("tp1", "dummy://hot_01/src_700_tp1?size=222"),
    };
    std::vector<MigrationManager::MigrationRequest> reqs;
    reqs.push_back(std::move(req));

    auto results = mgr.BatchSubmit("t", reqs);
    ASSERT_EQ(1u, results.size());
    ASSERT_NE(ErrorCode::EC_OK, results[0]); // 整块失败(一个 group Create 失败)

    // 一个 group 失败、一个成功 → 恰好 1 个 URI 被成功 Create。
    ASSERT_EQ(1u, f10_orphan_stub::g_created_uris.size());
    // 关键: 成功 Create 的 URI 必须被 Delete,不能 orphan(修复前会被跳过泄漏)。
    for (const auto &created : f10_orphan_stub::g_created_uris) {
        const bool deleted =
            std::any_of(f10_orphan_stub::g_deleted_uris.begin(),
                        f10_orphan_stub::g_deleted_uris.end(),
                        [&](const DataStorageUri &d) { return d.ToUriString() == created.ToUriString(); });
        ASSERT_TRUE(deleted) << "created URI leaked (not deleted): " << created.ToUriString();
    }
}

// F-03: ActiveTaskCountForInstances 按 instance 集合过滤，跨 group 不抢 slot。
TEST_F(MigrationManagerTest, TestActiveTaskCountForInstancesIsolation) {
    const std::string kInstanceA = "group_a_inst_01";
    const std::string kInstanceB = "group_b_inst_01";

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_, metrics_registry_, nullptr);

    // DebugInsertActiveCopyTask 直接往活跃表插 entry，不走真实 copy 流程，
    // 故无需 storage / Start / DebugEnable —— ActiveTaskCount* 只读内存表。
    mgr.DebugInsertActiveCopyTask(kInstanceA, 100, "dst_a1");
    mgr.DebugInsertActiveCopyTask(kInstanceA, 200, "dst_a2");

    // 全局 = 2
    ASSERT_EQ(2u, mgr.ActiveTaskCount());
    // group A 的 instance 集合 = 2
    ASSERT_EQ(2u, mgr.ActiveTaskCountForInstances({kInstanceA}));
    // group B 的 instance 集合 = 0（B 没有任何 task）
    ASSERT_EQ(0u, mgr.ActiveTaskCountForInstances({kInstanceB}));
    // 两个 group 合 = 全局
    ASSERT_EQ(2u, mgr.ActiveTaskCountForInstances({kInstanceA, kInstanceB}));
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
    ClearMarkForTest(mgr, kInstance, 1);
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

// F-09: 多个 target location 分别覆盖不同 specs，联合覆盖完整 → 不应 kAccept。
TEST_F(MigrationManagerTest, TestCheckCopyAdmissionUnionCoverage) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    const std::string src = "hot_01";
    const std::string dst = "cold_01";

    // source 含 tp0, tp1；cold 上 loc_a(tp0, SERVING) + loc_b(tp1, SERVING) 联合覆盖。
    {
        CacheLocationMap loc_map;
        auto src_loc = MakeLocationWithSpecs("loc_src", src, CLS_SERVING, {"tp0", "tp1"});
        auto dst_a = MakeLocationWithSpecs("loc_dst_a", dst, CLS_SERVING, {"tp0"});
        auto dst_b = MakeLocationWithSpecs("loc_dst_b", dst, CLS_SERVING, {"tp1"});
        loc_map[src_loc->id()] = src_loc;
        loc_map[dst_a->id()] = dst_a;
        loc_map[dst_b->id()] = dst_b;
        const auto adm = mgr.CheckCopyAdmission(kInstance, 9, loc_map, src, dst);
        ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kTargetServingExists, adm.status);
    }
    // WRITING 联合覆盖 → kTargetWritingExists
    {
        CacheLocationMap loc_map;
        auto src_loc = MakeLocationWithSpecs("loc_src", src, CLS_SERVING, {"tp0", "tp1"});
        auto dst_a = MakeLocationWithSpecs("loc_dst_a", dst, CLS_WRITING, {"tp0"});
        auto dst_b = MakeLocationWithSpecs("loc_dst_b", dst, CLS_WRITING, {"tp1"});
        loc_map[src_loc->id()] = src_loc;
        loc_map[dst_a->id()] = dst_a;
        loc_map[dst_b->id()] = dst_b;
        const auto adm = mgr.CheckCopyAdmission(kInstance, 10, loc_map, src, dst);
        ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kTargetWritingExists, adm.status);
    }
    // 混合：tp0 SERVING + tp1 WRITING → kTargetWritingExists（SERVING∪WRITING 联合覆盖）
    {
        CacheLocationMap loc_map;
        auto src_loc = MakeLocationWithSpecs("loc_src", src, CLS_SERVING, {"tp0", "tp1"});
        auto dst_a = MakeLocationWithSpecs("loc_dst_a", dst, CLS_SERVING, {"tp0"});
        auto dst_b = MakeLocationWithSpecs("loc_dst_b", dst, CLS_WRITING, {"tp1"});
        loc_map[src_loc->id()] = src_loc;
        loc_map[dst_a->id()] = dst_a;
        loc_map[dst_b->id()] = dst_b;
        const auto adm = mgr.CheckCopyAdmission(kInstance, 11, loc_map, src, dst);
        ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kTargetWritingExists, adm.status);
    }
    // 部分覆盖（只有 tp0，缺 tp1）→ 仍 kAccept
    {
        CacheLocationMap loc_map;
        auto src_loc = MakeLocationWithSpecs("loc_src", src, CLS_SERVING, {"tp0", "tp1"});
        auto dst_a = MakeLocationWithSpecs("loc_dst_a", dst, CLS_SERVING, {"tp0"});
        loc_map[src_loc->id()] = src_loc;
        loc_map[dst_a->id()] = dst_a;
        const auto adm = mgr.CheckCopyAdmission(kInstance, 12, loc_map, src, dst);
        ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kAccept, adm.status);
    }
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
    ASSERT_TRUE(mgr.HasActiveCopyTargetLocation(kInstance, 1001, "loc_dst_a"));
    ASSERT_EQ(2u, mgr.ActiveTaskCount());

    mgr.Stop();

    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, 1001));
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, 1002));
    ASSERT_FALSE(mgr.HasActiveCopyTargetLocation(kInstance, 1001, "loc_dst_a"));
    ASSERT_FALSE(mgr.HasActiveCopyTargetLocation(kInstance, 1002, "loc_dst_b"));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());

    // 重新 Start 后这些 block_key 应该可以再次被识别为"无活跃任务"。
    mgr.Start();
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, 1001));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
    mgr.Stop();
}

// F-12: GetActiveBlockKeysForInstance 返回指定 instance 的活跃 block_key 列表（用于 drain 前 BatchCancel）。
TEST_F(MigrationManagerTest, TestGetActiveBlockKeysForInstance) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    const std::string other_instance = "other_instance";

    // 空 instance → 空列表
    ASSERT_TRUE(mgr.GetActiveBlockKeysForInstance(kInstance).empty());

    mgr.DebugInsertActiveCopyTask(kInstance, 1001, "loc_a");
    mgr.DebugInsertActiveCopyTask(kInstance, 1002, "loc_b");
    mgr.DebugInsertActiveCopyTask(other_instance, 2001, "loc_c");

    auto keys = mgr.GetActiveBlockKeysForInstance(kInstance);
    ASSERT_EQ(2u, keys.size());
    std::sort(keys.begin(), keys.end());
    ASSERT_EQ(1001, keys[0]);
    ASSERT_EQ(1002, keys[1]);

    // other_instance 独立
    auto other_keys = mgr.GetActiveBlockKeysForInstance(other_instance);
    ASSERT_EQ(1u, other_keys.size());
    ASSERT_EQ(2001, other_keys[0]);

    // 不存在的 instance → 空列表
    ASSERT_TRUE(mgr.GetActiveBlockKeysForInstance("no_such").empty());
}

// F-12: BatchCancel + GetActiveBlockKeysForInstance 配合实现 drain：
// cancel 后 copy 完成回调清理活跃任务，block keys 列表变空。
TEST_F(MigrationManagerTest, TestDrainInstanceViaBatchCancelAndPoll) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "drain_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "drain_cold/"));

    const int64_t bk1 = 801, bk2 = 802;
    const std::string src1 = CreateSourceLocation(bk1, "hot_01", true, "d1");
    const std::string src2 = CreateSourceLocation(bk2, "hot_01", true, "d2");

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();

    auto submit = [&](int64_t bk, const std::string &src_loc) {
        MigrationManager::MigrationRequest req;
        req.instance_id = kInstance;
        req.block_key = bk;
        req.src_location_id = src_loc;
        req.src_storage_name = "hot_01";
        req.dst_storage_name = "cold_01";
        return mgr.Submit("t", req);
    };
    ASSERT_EQ(ErrorCode::EC_OK, submit(bk1, src1));
    ASSERT_EQ(ErrorCode::EC_OK, submit(bk2, src2));
    ASSERT_EQ(2u, mgr.GetActiveBlockKeysForInstance(kInstance).size());

    // drain: 取 keys → BatchCancel
    const auto keys = mgr.GetActiveBlockKeysForInstance(kInstance);
    mgr.BatchCancel(kInstance, keys);

    // 任务标为 cancelling，仍在活跃表
    ASSERT_EQ(2u, mgr.GetActiveBlockKeysForInstance(kInstance).size());

    // 模拟 copy 完成回调（monitor 线程未启动，手动驱动）
    mgr.OnTaskSuccess(kInstance, bk1);
    mgr.OnTaskFailed(kInstance, bk2, ErrorCode::EC_IO_ERROR);

    // drain 完成：活跃表空
    ASSERT_TRUE(mgr.GetActiveBlockKeysForInstance(kInstance).empty());
    ASSERT_EQ(2u, mgr.GetStats().copy_cancelled);
}

// F-12: draining gate 阻止该 instance 的新提交（Submit + BatchSubmit 两路），
// EndDraining 后恢复；其他 instance 不受影响。
TEST_F(MigrationManagerTest, TestDrainingInstanceGateRejectsSubmit) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    const std::string kOther = "other_instance";
    ASSERT_TRUE(CreateMetaIndexer(kOther));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "draingate_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "draingate_cold/"));

    const std::string src1 = CreateSourceLocation(901, "hot_01", true, "d1");

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();

    auto make_req = [](const std::string &inst, int64_t bk, const std::string &src) {
        MigrationManager::MigrationRequest req;
        req.instance_id = inst;
        req.block_key = bk;
        req.src_location_id = src;
        req.src_storage_name = "hot_01";
        req.dst_storage_name = "cold_01";
        return req;
    };

    mgr.BeginDrainingInstance(kInstance);

    // draining 中：Submit 被拒，且不建活跃任务
    ASSERT_EQ(ErrorCode::EC_ERROR, mgr.Submit("t", make_req(kInstance, 901, src1)));
    ASSERT_TRUE(mgr.GetActiveBlockKeysForInstance(kInstance).empty());

    // draining 中：BatchSubmit 也被拒（覆盖 admin/reclaimer 两路的共同下游 BatchSubmit）
    std::vector<MigrationManager::MigrationRequest> batch;
    batch.push_back(make_req(kInstance, 902, src1));
    auto batch_res = mgr.BatchSubmit("t", batch);
    ASSERT_EQ(1u, batch_res.size());
    ASSERT_NE(ErrorCode::EC_OK, batch_res[0]);
    ASSERT_TRUE(mgr.GetActiveBlockKeysForInstance(kInstance).empty());

    // 另一个未 drain 的 instance 提交不受影响（draining 是 per-instance）
    ASSERT_EQ(0u, mgr.GetActiveBlockKeysForInstance(kOther).size());
    // kOther 未 draining：Submit gate 不拒（源 location 缺失会在更后阶段失败，此处只验证未被 draining 挡下——
    // 用 BatchSubmit 空 src_specs 走 fallback 到 PrepareCopyTask，返回非 draining 的错误码即可区分）。

    // EndDraining 后：kInstance 恢复可提交
    mgr.EndDrainingInstance(kInstance);
    ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", make_req(kInstance, 901, src1)));
    ASSERT_EQ(1u, mgr.GetActiveBlockKeysForInstance(kInstance).size());
}

// F-18: HasActiveCopyTargetLocation 按 (instance_id, block_key) 作用域判断——
// 两个不同 instance/block 用相同 dst_location_id 时，只有匹配 scope 的那个才应被保护。
TEST_F(MigrationManagerTest, TestHasActiveCopyTargetLocationIsScoped) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    const std::string other_instance = "other_instance";
    const std::string shared_dst = "shared_dst_loc"; // 两个 task 复用同一 dst_location_id

    // instance A / block 1 → shared_dst；instance B / block 2 → shared_dst
    mgr.DebugInsertActiveCopyTask(kInstance, /*block_key*/ 1, shared_dst);
    mgr.DebugInsertActiveCopyTask(other_instance, /*block_key*/ 2, shared_dst);

    // 精确匹配 (instance, block, loc) 才为 true
    ASSERT_TRUE(mgr.HasActiveCopyTargetLocation(kInstance, 1, shared_dst));
    ASSERT_TRUE(mgr.HasActiveCopyTargetLocation(other_instance, 2, shared_dst));

    // 同 loc_id 但 scope 不匹配 → false（旧的裸 id 实现在此会误报 true）
    ASSERT_FALSE(mgr.HasActiveCopyTargetLocation(kInstance, 2, shared_dst));         // 错的 block
    ASSERT_FALSE(mgr.HasActiveCopyTargetLocation(other_instance, 1, shared_dst));    // 错的 block
    ASSERT_FALSE(mgr.HasActiveCopyTargetLocation("nonexistent_instance", 1, shared_dst)); // 错的 instance
    ASSERT_FALSE(mgr.HasActiveCopyTargetLocation(kInstance, 1, "different_loc"));    // 错的 loc_id
    ASSERT_FALSE(mgr.HasActiveCopyTargetLocation(kInstance, 1, ""));                 // 空 loc_id
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
