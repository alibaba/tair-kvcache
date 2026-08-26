#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/config/registry_manager.h"
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

namespace {
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

std::atomic<int> g_guard_sync_call_count{0};

bool FailFirstGuardSyncStub(void * /*obj*/, const KeyVector & /*keys*/) noexcept {
    return g_guard_sync_call_count.fetch_add(1, std::memory_order_acq_rel) > 0;
}
} // namespace

namespace async_guard_recovery_stub {
std::vector<std::shared_ptr<const InstanceGroup>> groups;
std::vector<std::shared_ptr<const InstanceInfo>> instances;
ErrorCode list_groups_result = EC_OK;
ErrorCode list_instances_result = EC_OK;
size_t resume_calls = 0;
std::string resumed_storage;
std::string resumed_operation_id;
std::vector<std::string> resumed_task_ids;

std::pair<ErrorCode, std::vector<std::shared_ptr<const InstanceGroup>>>
ListInstanceGroupStub(void * /*obj*/, RequestContext * /*request_context*/) {
    return {list_groups_result, groups};
}

std::pair<ErrorCode, std::vector<std::shared_ptr<const InstanceInfo>>>
ListInstanceInfoStub(void * /*obj*/, RequestContext * /*request_context*/, const std::string &group_name) {
    std::vector<std::shared_ptr<const InstanceInfo>> filtered;
    for (const auto &instance : instances) {
        if (instance && instance->instance_group_name() == group_name) {
            filtered.push_back(instance);
        }
    }
    return {list_instances_result, filtered};
}

AsyncCopyExecuteSubmitResult ResumeAsyncCopyStub(void * /*obj*/,
                                                 const std::string &storage_name,
                                                 const std::vector<std::string> &backend_task_ids,
                                                 size_t /*expected_items*/,
                                                 const std::string &operation_id,
                                                 const AsyncCopyOptions & /*options*/) {
    ++resume_calls;
    resumed_storage = storage_name;
    resumed_operation_id = operation_id;
    resumed_task_ids = backend_task_ids;
    AsyncCopyExecuteSubmitResult result;
    result.submit_result.status = EC_OK;
    result.submit_result.accepted = true;
    result.submit_result.operation_id = operation_id;
    result.submit_result.backend_task_ids = backend_task_ids;
    auto completion = std::make_shared<std::promise<PlanExecuteResult>>();
    result.future = completion->get_future();
    completion->set_value(PlanExecuteResult{EC_OK, "", true, true});
    return result;
}
} // namespace async_guard_recovery_stub

// ---- orphan cleanup 测试用 DataStorageManager::Create/Delete 存根 ----
// 复现"异构 size spec 分散到多个 create_group、某 group 失败使 block ineligible、
// 后处理 group 已成功分配的 URI 被跳过 → rollback 漏删 → orphan"的场景。
namespace orphan_cleanup_stub {
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
} // namespace orphan_cleanup_stub

// ---- batch Create 返回 shape 测试存根 ----
// DataStorageBackend::Create 的契约是 keys/results 按下标一一对应；这里分别制造短返回和长返回，
// 并记录所有成功返回的 URI，验证 shape 异常时调用方不会采用任何结果且会全部回滚。
namespace create_result_shape_stub {
enum class Shape {
    kShort,
    kLong,
};

Shape g_shape = Shape::kShort;
std::vector<DataStorageUri> g_created_uris;
std::vector<DataStorageUri> g_deleted_uris;

void Reset(Shape shape) {
    g_shape = shape;
    g_created_uris.clear();
    g_deleted_uris.clear();
}

std::vector<std::pair<ErrorCode, DataStorageUri>> Create_stub(void * /*obj*/,
                                                              RequestContext * /*rc*/,
                                                              const std::string &name,
                                                              const std::vector<std::string> &keys,
                                                              size_t size,
                                                              std::function<void()> /*cb*/) {
    const std::size_t result_count = g_shape == Shape::kShort ? (keys.empty() ? 0 : keys.size() - 1) : keys.size() + 1;
    std::vector<std::pair<ErrorCode, DataStorageUri>> results;
    results.reserve(result_count);
    for (std::size_t i = 0; i < result_count; ++i) {
        const std::string key = i < keys.size() ? keys[i] : "unexpected_extra_" + std::to_string(i);
        DataStorageUri uri("dummy://" + name + "/" + key + "?size=" + std::to_string(size));
        g_created_uris.push_back(uri);
        results.emplace_back(EC_OK, std::move(uri));
    }
    return results;
}

std::vector<ErrorCode> Delete_stub(void * /*obj*/,
                                   RequestContext * /*rc*/,
                                   const std::string & /*name*/,
                                   const std::vector<DataStorageUri> &uris,
                                   std::function<void()> /*cb*/) {
    g_deleted_uris.insert(g_deleted_uris.end(), uris.begin(), uris.end());
    return std::vector<ErrorCode>(uris.size(), EC_OK);
}

bool WasDeleted(const DataStorageUri &created) {
    return std::any_of(g_deleted_uris.begin(), g_deleted_uris.end(), [&](const DataStorageUri &deleted) {
        return deleted.ToUriString() == created.ToUriString();
    });
}
} // namespace create_result_shape_stub

// ---- prepared BatchSubmit 逐项 Create 失败测试存根 ----
// 返回数量严格对齐，但第二项失败，验证普通 per-item failure 不会扩大成整组失败。
namespace prepared_batch_partial_create_stub {
int g_create_calls = 0;
std::vector<std::size_t> g_create_key_counts;

void Reset() {
    g_create_calls = 0;
    g_create_key_counts.clear();
}

std::vector<std::pair<ErrorCode, DataStorageUri>> Create_stub(void * /*obj*/,
                                                              RequestContext * /*rc*/,
                                                              const std::string &name,
                                                              const std::vector<std::string> &keys,
                                                              size_t size,
                                                              std::function<void()> /*cb*/) {
    ++g_create_calls;
    g_create_key_counts.push_back(keys.size());
    std::vector<std::pair<ErrorCode, DataStorageUri>> results;
    results.reserve(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (i == 1) {
            results.emplace_back(EC_ERROR, DataStorageUri{});
            continue;
        }
        results.emplace_back(
            EC_OK, DataStorageUri("dummy://" + name + "/" + keys[i] + "?size=" + std::to_string(size)));
    }
    return results;
}
} // namespace prepared_batch_partial_create_stub

namespace mark_query_result_stub {
ErrorCode g_aggregate_ec = EC_OK;
std::vector<ErrorCode> g_per_key_ecs;
PropertyMapVector g_properties;

void Reset() {
    g_aggregate_ec = EC_OK;
    g_per_key_ecs.clear();
    g_properties.clear();
}

MetaIndexer::Result GetProperties_stub(void * /*obj*/,
                                       RequestContext * /*rc*/,
                                       const KeyVector & /*keys*/,
                                       const std::vector<std::string> & /*property_names*/,
                                       PropertyMapVector &out_properties) noexcept {
    out_properties = g_properties;
    MetaIndexer::Result result(g_aggregate_ec);
    result.error_codes = g_per_key_ecs;
    return result;
}
} // namespace mark_query_result_stub

namespace preparing_reservation_stub {
MigrationManager *g_manager = nullptr;
std::vector<std::pair<std::string, int64_t>> g_expected_reservations;
bool g_all_reserved_before_create = true;
bool g_all_block_scoped_targets_protected = true;
bool g_cancel_first_reservation_on_create = false;
std::optional<ErrorCode> g_cancel_result;
int g_create_calls = 0;
std::vector<std::size_t> g_create_key_counts;
std::vector<DataStorageUri> g_deleted_uris;
std::vector<std::shared_ptr<std::promise<PlanExecuteResult>>> g_pending_copy_promises;

void Reset() {
    g_manager = nullptr;
    g_expected_reservations.clear();
    g_all_reserved_before_create = true;
    g_all_block_scoped_targets_protected = true;
    g_cancel_first_reservation_on_create = false;
    g_cancel_result.reset();
    g_create_calls = 0;
    g_create_key_counts.clear();
    g_deleted_uris.clear();
    g_pending_copy_promises.clear();
}

std::vector<std::pair<ErrorCode, DataStorageUri>> Create_stub(void * /*obj*/,
                                                              RequestContext * /*rc*/,
                                                              const std::string & /*name*/,
                                                              const std::vector<std::string> &keys,
                                                              size_t size,
                                                              std::function<void()> /*cb*/) {
    ++g_create_calls;
    g_create_key_counts.push_back(keys.size());
    for (const auto &[instance_id, block_key] : g_expected_reservations) {
        g_all_reserved_before_create =
            g_all_reserved_before_create && g_manager != nullptr && g_manager->HasMigrationTask(instance_id, block_key);
        // Create 发生在 BatchAddLocation 之前，此时 reservation 还没有 id；任意非空候选 id 都应受
        // (instance, block) 临时保护，否则 publish->bind 窗口仍会被 Reclaimer 穿透。
        g_all_block_scoped_targets_protected =
            g_all_block_scoped_targets_protected && g_manager != nullptr &&
            g_manager->HasActiveCopyTargetLocation(instance_id, block_key, "future_location_probe");
    }
    if (g_cancel_first_reservation_on_create && g_create_calls == 1 && g_manager != nullptr &&
        !g_expected_reservations.empty()) {
        const auto &[instance_id, block_key] = g_expected_reservations.front();
        g_cancel_result = g_manager->Cancel(instance_id, block_key);
    }

    std::vector<std::pair<ErrorCode, DataStorageUri>> results;
    results.reserve(keys.size());
    for (const auto &key : keys) {
        results.emplace_back(EC_OK, DataStorageUri("dummy://cold_01/" + key + "?size=" + std::to_string(size)));
    }
    return results;
}

std::vector<ErrorCode> Delete_stub(void * /*obj*/,
                                   RequestContext * /*rc*/,
                                   const std::string & /*name*/,
                                   const std::vector<DataStorageUri> &uris,
                                   std::function<void()> /*cb*/) {
    g_deleted_uris.insert(g_deleted_uris.end(), uris.begin(), uris.end());
    return std::vector<ErrorCode>(uris.size(), EC_OK);
}

MetaIndexer::Result ReadModifyWriteBlockFail_stub(void * /*obj*/,
                                                  RequestContext * /*rc*/,
                                                  const KeyVector &keys,
                                                  const BlockIdsOnlyModifierFunc & /*modifier*/) noexcept {
    MetaIndexer::Result result(EC_ERROR);
    result.error_codes.assign(keys.size(), EC_ERROR);
    return result;
}

using CopySubmitLocation = std::future<PlanExecuteResult> (SchedulePlanExecutor::*)(const CacheLocationCopyRequest &);

std::future<PlanExecuteResult> CopySubmitInvalid_stub(void * /*obj*/, const CacheLocationCopyRequest & /*request*/) {
    return {};
}

std::future<PlanExecuteResult> CopySubmitPending_stub(void * /*obj*/, const CacheLocationCopyRequest & /*request*/) {
    auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    auto future = promise->get_future();
    g_pending_copy_promises.push_back(std::move(promise));
    return future;
}
} // namespace preparing_reservation_stub

// ---- submission 并发与锁域测试存根 ----
// 第一笔 migration Create 可由测试精确卡住；后续 Create 仍可进入，用来区分“共享 lifecycle
// barrier + 短准入锁”和旧的全函数互斥锁。所有状态都在同一 mutex 下访问，避免测试自身 data race。
namespace submission_concurrency_stub {
std::mutex g_mutex;
std::condition_variable g_cv;
bool g_block_first_create = false;
bool g_first_create_entered = false;
bool g_release_first_create = false;
int g_create_calls = 0;

void Reset(bool block_first_create = true) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_block_first_create = block_first_create;
    g_first_create_entered = false;
    g_release_first_create = false;
    g_create_calls = 0;
}

bool WaitForFirstCreate(std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
    std::unique_lock<std::mutex> lock(g_mutex);
    return g_cv.wait_for(lock, timeout, []() { return g_first_create_entered; });
}

void ReleaseFirstCreate() {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_release_first_create = true;
    }
    g_cv.notify_all();
}

int CreateCallCount() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_create_calls;
}

std::vector<std::pair<ErrorCode, DataStorageUri>> Create_stub(void * /*obj*/,
                                                              RequestContext * /*rc*/,
                                                              const std::string &name,
                                                              const std::vector<std::string> &keys,
                                                              size_t size,
                                                              std::function<void()> /*cb*/) {
    {
        std::unique_lock<std::mutex> lock(g_mutex);
        const int call_index = ++g_create_calls;
        if (g_block_first_create && call_index == 1) {
            g_first_create_entered = true;
            g_cv.notify_all();
            g_cv.wait(lock, []() { return g_release_first_create; });
        }
    }

    std::vector<std::pair<ErrorCode, DataStorageUri>> results;
    results.reserve(keys.size());
    for (const auto &key : keys) {
        results.emplace_back(EC_OK, DataStorageUri("dummy://" + name + "/" + key + "?size=" + std::to_string(size)));
    }
    return results;
}
} // namespace submission_concurrency_stub

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

    // 在指定 instance 的 hot 存储上创建一个 SERVING 源 location，可选写入真实源文件内容。
    std::string CreateSourceLocationForInstance(const std::string &instance_id,
                                                int64_t block_key,
                                                const std::string &hot_storage,
                                                bool write_file,
                                                const std::string &content,
                                                int64_t create_time = 0) {
        auto rc = std::make_shared<RequestContext>("create_source");
        std::string key = instance_id + "/TP0/" + StringUtil::Uint64ToHex(block_key);
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

        MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(instance_id));
        auto loc =
            std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                            1,
                                            std::vector<LocationSpec>{LocationSpec("TP0", src_uri.ToUriString())});
        if (create_time != 0) {
            loc->set_create_time(create_time);
        }
        std::vector<std::string> ids;
        EXPECT_EQ(
            ErrorCode::EC_OK, BatchAddLocationForTest(&meta_searcher, rc.get(), {block_key}, {loc}, ids));
        EXPECT_EQ(1u, ids.size());
        // BatchAddLocation 写入 CLS_WRITING，CAS 到 SERVING 模拟一个已就绪的源副本。
        std::vector<std::vector<MetaSearcher::LocationCASTask>> cas{
            {MetaSearcher::LocationCASTask{ids[0], CLS_WRITING, CLS_SERVING}}};
        std::vector<std::vector<ErrorCode>> cas_results;
        EXPECT_EQ(ErrorCode::EC_OK, meta_searcher.BatchCASLocationStatus(rc.get(), {block_key}, cas, cas_results));
        return ids[0];
    }

    // 绝大多数用例使用默认测试 instance，保留原 helper 以免测试调用面膨胀。
    std::string CreateSourceLocation(int64_t block_key,
                                     const std::string &hot_storage,
                                     bool write_file,
                                     const std::string &content,
                                     int64_t create_time = 0) {
        return CreateSourceLocationForInstance(kInstance, block_key, hot_storage, write_file, content, create_time);
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

    void SetRawMark(int64_t block_key, const std::string &target, std::optional<std::string> deadline) {
        auto indexer = meta_manager_->GetMetaIndexer(kInstance);
        ASSERT_TRUE(indexer);
        auto modifier = [&target, &deadline](const LocationIdVector & /*existing*/,
                                             ErrorCode get_ec,
                                             size_t /*idx*/,
                                             PropertyMap &upsert_property_map,
                                             CacheLocationMap & /*out_new*/) -> ModifierResult {
            if (get_ec != EC_OK) {
                return {MA_FAIL, get_ec};
            }
            upsert_property_map[MigrationManager::PROPERTY_TIERED_WRITE_TARGET] = target;
            if (deadline.has_value()) {
                upsert_property_map[MigrationManager::PROPERTY_TIERED_WRITE_DEADLINE_MS] = *deadline;
            }
            return {MA_OK, EC_OK};
        };
        RequestContext rc("set_raw_tiered_write_mark");
        ASSERT_EQ(EC_OK, indexer->ReadModifyWriteBlock(&rc, {block_key}, modifier).ec);
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

    // 旧的无条件 ClearTieredWriteMark 已删除，测试用 helper 模拟：
    // 先查 mark 拿到 target+deadline，再做 match-clear。
    void ClearMarkForTest(MigrationManager &mgr, const std::string &instance_id, int64_t block_key) {
        std::vector<MigrationManager::MarkQueryResult> results;
        mgr.BatchGetTieredWriteTargets(instance_id, {block_key}, results);
        if (!results.empty() && results[0].HasValidMark()) {
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

TEST_F(MigrationManagerTest, TestSelectExplicitMigrationKeysDeduplicatesInOrder) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    RequestContext rc("select_explicit_keys");

    auto [ec, keys] = mgr.SelectMigrationCandidateKeys(
        &rc, "select_explicit_keys", {11, 12, 11, 13, 12}, 0, /*meta_indexer*/ nullptr);

    ASSERT_EQ(ErrorCode::EC_OK, ec);
    ASSERT_EQ((std::vector<int64_t>{11, 12, 13}), keys);
}

TEST_F(MigrationManagerTest, TestBatchGetTieredWriteTargetsPreservesPartialResults) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    mark_query_result_stub::Reset();
    mark_query_result_stub::g_aggregate_ec = EC_PARTIAL_OK;
    mark_query_result_stub::g_per_key_ecs = {EC_OK, EC_ERROR, EC_OK};
    mark_query_result_stub::g_properties = {
        {{MigrationManager::PROPERTY_TIERED_WRITE_TARGET, "cold_01"},
         {MigrationManager::PROPERTY_TIERED_WRITE_DEADLINE_MS, "9999999999999"}},
        {},
        {},
    };
    Stub stub;
    stub.set(ADDR(MetaIndexer, GetProperties), mark_query_result_stub::GetProperties_stub);

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_, metrics_registry_);
    std::vector<MigrationManager::MarkQueryResult> results;
    ASSERT_EQ(EC_PARTIAL_OK, mgr.BatchGetTieredWriteTargets(kInstance, {1, 2, 3}, results));

    ASSERT_EQ(3u, results.size());
    ASSERT_EQ(MigrationManager::MarkQueryState::kValid, results[0].state);
    ASSERT_EQ(EC_OK, results[0].ec);
    ASSERT_EQ("cold_01", results[0].target);
    ASSERT_EQ(9999999999999LL, results[0].deadline_ms);
    ASSERT_EQ(MigrationManager::MarkQueryState::kReadError, results[1].state);
    ASSERT_EQ(EC_ERROR, results[1].ec);
    ASSERT_TRUE(results[1].target.empty());
    ASSERT_EQ(MigrationManager::MarkQueryState::kNoMark, results[2].state);
    ASSERT_EQ(EC_OK, results[2].ec);
    ASSERT_TRUE(results[2].target.empty());
    ASSERT_EQ(1u, metrics_registry_->GetCounter("migration.mark_query_errors_total").Get());
}

TEST_F(MigrationManagerTest, TestBatchGetTieredWriteTargetsReportsAllReadFailures) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    mark_query_result_stub::Reset();
    mark_query_result_stub::g_aggregate_ec = EC_ERROR;
    mark_query_result_stub::g_per_key_ecs = {EC_ERROR, EC_ERROR};
    mark_query_result_stub::g_properties = {{}, {}};
    Stub stub;
    stub.set(ADDR(MetaIndexer, GetProperties), mark_query_result_stub::GetProperties_stub);

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_, metrics_registry_);
    std::vector<MigrationManager::MarkQueryResult> results;
    ASSERT_EQ(EC_ERROR, mgr.BatchGetTieredWriteTargets(kInstance, {1, 2}, results));

    ASSERT_EQ(2u, results.size());
    for (const auto &result : results) {
        ASSERT_EQ(MigrationManager::MarkQueryState::kReadError, result.state);
        ASSERT_EQ(EC_ERROR, result.ec);
        ASSERT_TRUE(result.target.empty());
    }
    ASSERT_EQ(2u, metrics_registry_->GetCounter("migration.mark_query_errors_total").Get());
}

TEST_F(MigrationManagerTest, TestBatchGetTieredWriteTargetsDistinguishesBlockNotFound) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    mark_query_result_stub::Reset();
    // MetaIndexer 把 EC_NOENT 计入 aggregate partial；mark query 将其显式保留为状态，
    // 但不视为存储读取故障，因此对调用方返回 EC_OK。
    mark_query_result_stub::g_aggregate_ec = EC_PARTIAL_OK;
    mark_query_result_stub::g_per_key_ecs = {EC_NOENT, EC_OK};
    mark_query_result_stub::g_properties = {{}, {}};
    Stub stub;
    stub.set(ADDR(MetaIndexer, GetProperties), mark_query_result_stub::GetProperties_stub);

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_, metrics_registry_);
    std::vector<MigrationManager::MarkQueryResult> results;
    ASSERT_EQ(EC_OK, mgr.BatchGetTieredWriteTargets(kInstance, {1, 2}, results));

    ASSERT_EQ(2u, results.size());
    ASSERT_EQ(MigrationManager::MarkQueryState::kBlockNotFound, results[0].state);
    ASSERT_EQ(EC_NOENT, results[0].ec);
    ASSERT_EQ(MigrationManager::MarkQueryState::kNoMark, results[1].state);
    ASSERT_EQ(EC_OK, results[1].ec);
    ASSERT_EQ(0u, metrics_registry_->GetCounter("migration.mark_query_errors_total").Get());
}

TEST_F(MigrationManagerTest, TestBatchGetTieredWriteTargetsRejectsMisalignedOutput) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    mark_query_result_stub::Reset();
    mark_query_result_stub::g_aggregate_ec = EC_OK;
    mark_query_result_stub::g_per_key_ecs = {EC_OK, EC_OK};
    mark_query_result_stub::g_properties = {{{MigrationManager::PROPERTY_TIERED_WRITE_TARGET, "stale"}}};
    Stub stub;
    stub.set(ADDR(MetaIndexer, GetProperties), mark_query_result_stub::GetProperties_stub);

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_, metrics_registry_);
    MigrationManager::MarkQueryResult stale;
    stale.state = MigrationManager::MarkQueryState::kValid;
    stale.target = "must_be_cleared";
    std::vector<MigrationManager::MarkQueryResult> results(2, stale);
    ASSERT_EQ(EC_ERROR, mgr.BatchGetTieredWriteTargets(kInstance, {1, 2}, results));

    ASSERT_EQ(2u, results.size());
    for (const auto &result : results) {
        ASSERT_EQ(MigrationManager::MarkQueryState::kReadError, result.state);
        ASSERT_EQ(EC_ERROR, result.ec);
        ASSERT_TRUE(result.target.empty());
    }
    ASSERT_EQ(2u, metrics_registry_->GetCounter("migration.mark_query_errors_total").Get());
}

TEST_F(MigrationManagerTest, TestMalformedDeadlinesAreTreatedAsNoMarkAndCleared) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "malformed_mark_hot/"));
    const std::vector<std::optional<std::string>> malformed_deadlines = {
        std::nullopt,
        std::string(),
        "not-a-number",
        "0",
        "-1",
        "99999999999999999999",
        "-99999999999999999999",
    };
    std::vector<int64_t> block_keys;
    for (size_t i = 0; i < malformed_deadlines.size(); ++i) {
        const int64_t block_key = static_cast<int64_t>(100 + i);
        CreateSourceLocation(block_key, "hot_01", false, "x");
        SetRawMark(block_key, "cold_01", malformed_deadlines[i]);
        block_keys.push_back(block_key);
    }

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    std::vector<MigrationManager::MarkQueryResult> results;
    ASSERT_EQ(EC_OK, mgr.BatchGetTieredWriteTargets(kInstance, block_keys, results));

    ASSERT_EQ(block_keys.size(), results.size());
    for (size_t i = 0; i < block_keys.size(); ++i) {
        EXPECT_EQ(MigrationManager::MarkQueryState::kNoMark, results[i].state);
        EXPECT_EQ(EC_OK, results[i].ec);
        EXPECT_TRUE(results[i].target.empty());
        EXPECT_EQ(0, results[i].deadline_ms);
        EXPECT_TRUE(GetRawTieredWriteTarget(block_keys[i]).empty());
    }
}

TEST_F(MigrationManagerTest, TestMalformedCleanupDoesNotClearValidRefresh) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "malformed_refresh_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "malformed_refresh_cold/"));
    CreateSourceLocation(200, "hot_01", false, "x");
    SetRawMark(200, "cold_01", std::string("invalid"));

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    ASSERT_FALSE(mgr.ClearTieredWriteMarkIfMatch(kInstance, 200, "cold_01", 0));
    ASSERT_EQ("cold_01", GetRawTieredWriteTarget(200));
    // 模拟查询已拿到 malformed target 后，另一线程把它刷新成同 target 的合法 Mark。
    ASSERT_EQ(EC_OK, mgr.MarkForTieredWrite(kInstance, {200}, "cold_01", 10000));
    ASSERT_FALSE(mgr.ClearTieredWriteMarkIfMatchInternal(kInstance, 200, "cold_01", 0));

    std::vector<MigrationManager::MarkQueryResult> results;
    ASSERT_EQ(EC_OK, mgr.BatchGetTieredWriteTargets(kInstance, {200}, results));
    ASSERT_EQ(1u, results.size());
    ASSERT_TRUE(results[0].HasValidMark());
    ASSERT_EQ("cold_01", results[0].target);
    ASSERT_GT(results[0].deadline_ms, 0);
}

TEST_F(MigrationManagerTest, TestMarkForTieredWriteRejectsDeadlineOverflow) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "mark_overflow_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "mark_overflow_cold/"));
    CreateSourceLocation(201, "hot_01", false, "x");

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    ASSERT_EQ(EC_BADARGS, mgr.MarkForTieredWrite(kInstance, {201}, "cold_01", std::numeric_limits<int64_t>::max()));
    ASSERT_TRUE(GetRawTieredWriteTarget(201).empty());
    ASSERT_EQ(0u, mgr.GetStats().marks_added);
}

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

// match-clear 不会清掉同 block 上后来打的新 mark（不同 target 或 deadline）。
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

// 部分 key 因 block 不存在被 MA_SKIP 时，marks_added / expiry / event 只计实际成功数，
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

// 打标到未注册的 target storage 应失败（EC_NOENT）且不写入任何 mark，
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
    ASSERT_FALSE(mgr.HasMigrationTask(req.instance_id, req.block_key));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
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
    ASSERT_FALSE(mgr.HasMigrationTask(req.instance_id, req.block_key));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
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
    ASSERT_FALSE(mgr.HasMigrationTask(req.instance_id, req.block_key));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
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

// Copy 只允许消费与自身 destination 一致的 mark。写往 cold_02 的 Copy
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

// Submit 时 PrepareCopyTask 应记录 src_create_time。当源 location 的 create_time 与记录不匹配时
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

// create_time=0 边界：源和 ctx 都默认 0，0==0 应视为匹配并正常 promote。
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
    ASSERT_EQ(1u, mgr.GetStats().source_failures_recorded);
    ASSERT_EQ(1u, mgr.GetStats().source_failure_entries);
}

// ============ Cancel 任务状态机（认领 + 延迟收尾） ============

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

// preparing 与现有 active task 共用一张表，并显式覆盖去重、Reclaimer、
// completion/cancel 以及 preparing->running 的状态转换。
TEST_F(MigrationManagerTest, TestPreparingReservationStateMachine) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    MigrationManager::MigrationRequest req;
    req.instance_group_name = "group_a";
    req.instance_id = kInstance;
    req.block_key = 680;
    req.src_location_id = "src_680";
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";

    {
        std::lock_guard<std::mutex> lock(mgr.task_mutex_);
        ASSERT_TRUE(mgr.ReservePreparingTaskLocked(req));
        ASSERT_FALSE(mgr.ReservePreparingTaskLocked(req)) << "reserve must atomically reject duplicates";
    }
    ASSERT_TRUE(mgr.HasMigrationTask(kInstance, req.block_key));
    ASSERT_EQ(1u, mgr.ActiveTaskCount());
    ASSERT_EQ(1u, mgr.ActiveTaskCountForGroup("group_a"));
    ASSERT_TRUE(mgr.HasActiveCopyTargetLocation(kInstance, req.block_key, "not_bound_yet"));

    // premature completion 不能认领未提交任务；Cancel 只转取消态，由 Prepare 所有者安全清理。
    mgr.OnTaskSuccess(kInstance, req.block_key);
    mgr.OnTaskFailed(kInstance, req.block_key, EC_ERROR);
    ASSERT_EQ(EC_OK, mgr.Cancel(kInstance, req.block_key));
    ASSERT_EQ(EC_OK, mgr.Cancel(kInstance, req.block_key)) << "preparing cancel must be idempotent";
    ASSERT_TRUE(mgr.HasMigrationTask(kInstance, req.block_key));
    ASSERT_TRUE(mgr.HasActiveCopyTargetLocation(kInstance, req.block_key, "not_bound_yet"));
    {
        std::lock_guard<std::mutex> lock(mgr.task_mutex_);
        ASSERT_EQ(MigrationManager::CopyTaskState::kPrepareCancelling,
                  mgr.active_tasks_by_instance_[kInstance][req.block_key].state);
        MigrationManager::CopyTaskContext ignored;
        ignored.instance_id = req.instance_id;
        ignored.block_key = req.block_key;
        ASSERT_FALSE(mgr.UpdatePreparingTaskLocked(ignored)) << "cancelled prepare must not be overwritten";
        ASSERT_FALSE(mgr.MarkTaskRunningLocked(kInstance, req.block_key));
        ASSERT_TRUE(mgr.RemovePreparingTaskLocked(kInstance, req.block_key));
    }
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, req.block_key));

    MigrationManager::CopyTaskContext prepared;
    prepared.instance_group_name = req.instance_group_name;
    prepared.instance_id = req.instance_id;
    prepared.block_key = req.block_key;
    prepared.src_location_id = req.src_location_id;
    prepared.src_storage_name = req.src_storage_name;
    prepared.dst_storage_name = req.dst_storage_name;
    prepared.dst_location_id = "dst_680";
    {
        std::lock_guard<std::mutex> lock(mgr.task_mutex_);
        ASSERT_TRUE(mgr.ReservePreparingTaskLocked(req));
        ASSERT_TRUE(mgr.UpdatePreparingTaskLocked(prepared));
    }
    ASSERT_TRUE(mgr.HasActiveCopyTargetLocation(kInstance, req.block_key, "dst_680"));
    ASSERT_FALSE(mgr.HasActiveCopyTargetLocation(kInstance, req.block_key, "other_dst"));

    {
        std::lock_guard<std::mutex> lock(mgr.task_mutex_);
        ASSERT_TRUE(mgr.MarkTaskRunningLocked(kInstance, req.block_key));
        ASSERT_FALSE(mgr.MarkTaskRunningLocked(kInstance, req.block_key));
    }
    ASSERT_EQ(EC_OK, mgr.Cancel(kInstance, req.block_key));
    {
        std::lock_guard<std::mutex> lock(mgr.task_mutex_);
        ASSERT_EQ(MigrationManager::CopyTaskState::kCancelling,
                  mgr.active_tasks_by_instance_[kInstance][req.block_key].state);
        ASSERT_TRUE(mgr.RemoveActiveTaskLocked(kInstance, req.block_key));
    }
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
}

// 单条 Submit 在第一次目标 Create 前就必须让 preparing 对 active-target 查询可见。
TEST_F(MigrationManagerTest, TestSubmitReservesPreparingBeforeCreate) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "reserve_single_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "reserve_single_cold/"));

    const int64_t block_key = 681;
    const std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "single-reserve");
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();

    preparing_reservation_stub::Reset();
    preparing_reservation_stub::g_manager = &mgr;
    preparing_reservation_stub::g_expected_reservations = {{kInstance, block_key}};
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), preparing_reservation_stub::Create_stub);

    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    ASSERT_EQ(EC_OK, mgr.Submit("preparing_reservation", req));

    ASSERT_GT(preparing_reservation_stub::g_create_calls, 0);
    ASSERT_TRUE(preparing_reservation_stub::g_all_reserved_before_create);
    ASSERT_TRUE(preparing_reservation_stub::g_all_block_scoped_targets_protected);
    const std::string dst_loc = mgr.GetActiveTaskDstLocation(kInstance, block_key);
    ASSERT_FALSE(dst_loc.empty());
    ASSERT_TRUE(mgr.HasActiveCopyTargetLocation(kInstance, block_key, dst_loc));
    ASSERT_FALSE(mgr.HasActiveCopyTargetLocation(kInstance, block_key, "future_location_probe"));
}

// Cancel 与同步 Prepare I/O 并发时只标记 reservation；提交线程返回后必须清目标、释放
// reservation，且绝不能把取消态覆盖成 kRunning 或向 executor 提交 copy。
TEST_F(MigrationManagerTest, TestSubmitCancelDuringPrepareStopsBeforeCopy) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "reserve_cancel_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "reserve_cancel_cold/"));

    const int64_t block_key = 683;
    const std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "cancel-during-prepare");
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();

    preparing_reservation_stub::Reset();
    preparing_reservation_stub::g_manager = &mgr;
    preparing_reservation_stub::g_expected_reservations = {{kInstance, block_key}};
    preparing_reservation_stub::g_cancel_first_reservation_on_create = true;
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), preparing_reservation_stub::Create_stub);

    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    ASSERT_EQ(EC_ERROR, mgr.Submit("cancel_preparing_reservation", req));
    ASSERT_EQ(std::optional<ErrorCode>(EC_OK), preparing_reservation_stub::g_cancel_result);
    ASSERT_EQ(0u, mgr.GetStats().copy_submitted);
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
    ASSERT_TRUE(WaitFor([&]() { return LocationCount(block_key) == 1u; }));
}

// executor 接收失败发生在 kPreparing->kRunning 之后；失败路径仍须删除目标并移除 active entry。
TEST_F(MigrationManagerTest, TestSubmitExecutorFailureReleasesActiveTask) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "reserve_exec_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "reserve_exec_cold/"));

    const int64_t block_key = 682;
    const std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "executor-failure");
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    Stub stub;
    stub.set(static_cast<preparing_reservation_stub::CopySubmitLocation>(ADDR(SchedulePlanExecutor, Submit)),
             preparing_reservation_stub::CopySubmitInvalid_stub);

    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    ASSERT_EQ(EC_ERROR, mgr.Submit("copy_executor_submit_failure", req));
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
    ASSERT_TRUE(WaitFor([&]() { return LocationCount(block_key) == 1u; }));
}

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

// success 用例必须显式证明走了真批量主路径。同尺寸的三个 prepared request 应合并为
// 一次三 key Create，并一次发布三个 WRITING location；不能只凭最终 EC_OK 间接判断。
TEST_F(MigrationManagerTest, TestBatchSubmit) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "batch_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "batch_cold/"));

    std::vector<MigrationManager::MigrationRequest> reqs;
    for (int64_t bk = 800; bk < 803; ++bk) {
        std::string src_loc = CreateSourceLocation(bk, "hot_01", true, "data");
        const auto src = GetLocation(bk, src_loc);
        ASSERT_NE(nullptr, src);
        MigrationManager::MigrationRequest req;
        req.instance_id = kInstance;
        req.block_key = bk;
        req.src_location_id = src_loc;
        req.src_create_time = src->create_time();
        req.src_storage_name = "hot_01";
        req.dst_storage_name = "cold_01";
        req.src_specs = src->location_specs();
        reqs.push_back(req);
    }
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();

    preparing_reservation_stub::Reset();
    preparing_reservation_stub::g_manager = &mgr;
    preparing_reservation_stub::g_expected_reservations = {{kInstance, 800}, {kInstance, 801}, {kInstance, 802}};
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), preparing_reservation_stub::Create_stub);

    auto results = mgr.BatchSubmit("t", reqs);
    ASSERT_EQ(3u, results.size());
    for (auto ec : results) {
        ASSERT_EQ(ErrorCode::EC_OK, ec);
    }
    ASSERT_EQ(1, preparing_reservation_stub::g_create_calls);
    ASSERT_EQ((std::vector<std::size_t>{3}), preparing_reservation_stub::g_create_key_counts);
    ASSERT_EQ(3u, mgr.ActiveTaskCount());
    for (int64_t block_key = 800; block_key < 803; ++block_key) {
        const std::string dst_location_id = mgr.GetActiveTaskDstLocation(kInstance, block_key);
        ASSERT_FALSE(dst_location_id.empty());
        ASSERT_EQ(2u, LocationCount(block_key));
        ASSERT_EQ(CLS_WRITING, GetLocationStatus(block_key, dst_location_id));
    }
}

// private prepared-request API 不兼容空 src_specs fallback。违反 contract 时必须在
// reservation/backend/meta I/O 前整批返回 EC_BADARGS。
TEST_F(MigrationManagerTest, TestBatchSubmitRejectsEmptySourceSpecsBeforeIo) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    preparing_reservation_stub::Reset();
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), preparing_reservation_stub::Create_stub);

    MigrationManager::MigrationRequest request;
    request.instance_id = kInstance;
    request.block_key = 804;
    request.src_location_id = "src_804";
    request.src_storage_name = "hot_01";
    request.dst_storage_name = "cold_01";

    const auto results = mgr.BatchSubmit("reject_empty_source_specs", {request});
    ASSERT_EQ(1u, results.size());
    ASSERT_EQ(EC_BADARGS, results[0]);
    ASSERT_EQ(0, preparing_reservation_stub::g_create_calls);
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
}

// 实现只持有一份 first_req indexer/target，混合 instance/target 的 batch 不能进入
// Create 或 reservation；private contract 被内部误用时也要安全失败。
TEST_F(MigrationManagerTest, TestBatchSubmitRejectsMixedPreparedBatchBeforeIo) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    preparing_reservation_stub::Reset();
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), preparing_reservation_stub::Create_stub);

    MigrationManager::MigrationRequest first;
    first.instance_id = kInstance;
    first.block_key = 805;
    first.src_location_id = "src_805";
    first.src_storage_name = "hot_01";
    first.dst_storage_name = "cold_01";
    first.src_specs = {LocationSpec("TP0", "dummy://hot_01/src_805?size=4")};
    auto second = first;
    second.instance_id = "other_instance";
    second.block_key = 806;
    second.dst_storage_name = "cold_02";

    const auto results = mgr.BatchSubmit("reject_mixed_prepared_batch", {first, second});
    ASSERT_EQ(2u, results.size());
    ASSERT_EQ(EC_BADARGS, results[0]);
    ASSERT_EQ(EC_BADARGS, results[1]);
    ASSERT_EQ(0, preparing_reservation_stub::g_create_calls);
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
}

// 真批量路径必须在第一次 batch Create 前 reserve 所有 eligible block；批内重复逐项失败，
// 不能中止其他项，也不能多占一个并发 slot。
TEST_F(MigrationManagerTest, TestBatchReservesAllBeforeCreateAndDeduplicates) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "reserve_batch_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "reserve_batch_cold/"));

    auto make_prepared_req = [&](int64_t block_key) {
        const std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "batch-reserve");
        const auto src = GetLocation(block_key, src_loc);
        EXPECT_NE(nullptr, src);
        MigrationManager::MigrationRequest req;
        req.instance_group_name = "group_a";
        req.instance_id = kInstance;
        req.block_key = block_key;
        req.src_location_id = src_loc;
        req.src_create_time = src == nullptr ? 0 : src->create_time();
        req.src_storage_name = "hot_01";
        req.dst_storage_name = "cold_01";
        if (src != nullptr) {
            req.src_specs = src->location_specs();
        }
        return req;
    };

    auto first = make_prepared_req(810);
    auto duplicate = first;
    auto second = make_prepared_req(811);
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();

    preparing_reservation_stub::Reset();
    preparing_reservation_stub::g_manager = &mgr;
    preparing_reservation_stub::g_expected_reservations = {{kInstance, 810}, {kInstance, 811}};
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), preparing_reservation_stub::Create_stub);

    auto results = mgr.BatchSubmit(
        "batch_preparing_reservation", {first, duplicate, second}, MigrationManager::CopyConcurrencyLimit{"group_a", 2});
    ASSERT_EQ(3u, results.size());
    ASSERT_EQ(EC_OK, results[0]);
    ASSERT_EQ(EC_EXIST, results[1]);
    ASSERT_EQ(EC_OK, results[2]);
    ASSERT_GT(preparing_reservation_stub::g_create_calls, 0);
    ASSERT_TRUE(preparing_reservation_stub::g_all_reserved_before_create);
    ASSERT_TRUE(preparing_reservation_stub::g_all_block_scoped_targets_protected);
    ASSERT_EQ(2u, mgr.ActiveTaskCount());
    ASSERT_TRUE(mgr.HasMigrationTask(kInstance, 810));
    ASSERT_TRUE(mgr.HasMigrationTask(kInstance, 811));
    ASSERT_FALSE(mgr.GetActiveTaskDstLocation(kInstance, 810).empty());
    ASSERT_FALSE(mgr.GetActiveTaskDstLocation(kInstance, 811).empty());
}

// batch AddLocation 失败时，目标 URI 必须按逐 key 结果回滚，所有尚未运行的 reservation 必须释放。
TEST_F(MigrationManagerTest, TestBatchAddLocationFailureReleasesPreparingReservations) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "reserve_add_fail_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "reserve_add_fail_cold/"));

    const int64_t block_key = 812;
    const std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "add-failure");
    const auto src = GetLocation(block_key, src_loc);
    ASSERT_NE(nullptr, src);
    MigrationManager::MigrationRequest req;
    req.instance_group_name = "group_a";
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_create_time = src->create_time();
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    req.src_specs = src->location_specs();

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    Stub stub;
    stub.set(ADDR(MetaIndexer, ReadModifyWriteBlock), preparing_reservation_stub::ReadModifyWriteBlockFail_stub);

    const auto results = mgr.BatchSubmit("batch_add_location_failure", {req});
    ASSERT_EQ(1u, results.size());
    ASSERT_NE(EC_OK, results[0]);
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
    ASSERT_EQ(1u, LocationCount(block_key));
}

// batch AddLocation 部分成功时，成功项继续进入 copy，失败项按其 rollback-only ID 精确清理；
// 不能再把整个 batch 一并判失败或删除成功项的 URI。
TEST_F(MigrationManagerTest, TestBatchAddLocationPartialFailureKeepsSuccessfulCopyAndRollsBackFailedItem) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "reserve_partial_add_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "reserve_partial_add_cold/"));

    auto indexer = meta_manager_->GetMetaIndexer(kInstance);
    ASSERT_NE(nullptr, indexer);
    indexer->batch_key_size_ = 1;
    indexer->max_key_count_ = 1;

    std::vector<int64_t> block_keys{814, 815};
    // batch_key_size is a soft limit: every key in one mutex shard remains
    // in the same batch. Select two shards using this indexer's runtime hash
    // seed so max_key_count=1 deterministically exercises one successful
    // batch followed by one capacity failure.
    while (indexer->GetMutexShardIndex(block_keys[0]) == indexer->GetMutexShardIndex(block_keys[1])) {
        ++block_keys[1];
    }
    auto make_request = [&](int64_t block_key) {
        MigrationManager::MigrationRequest req;
        req.instance_group_name = "group_a";
        req.instance_id = kInstance;
        req.block_key = block_key;
        req.src_location_id = "src_" + std::to_string(block_key);
        req.src_storage_name = "hot_01";
        req.dst_storage_name = "cold_01";
        req.src_specs = {
            LocationSpec("TP0", "dummy://hot_01/src_" + std::to_string(block_key) + "?size=4")};
        return req;
    };

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_, metrics_registry_, nullptr);
    mgr.DebugEnableCopySubmissionsForTest();
    preparing_reservation_stub::Reset();
    preparing_reservation_stub::g_manager = &mgr;
    preparing_reservation_stub::g_expected_reservations = {{kInstance, block_keys[0]}, {kInstance, block_keys[1]}};
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), preparing_reservation_stub::Create_stub);
    stub.set(ADDR(DataStorageManager, Delete), preparing_reservation_stub::Delete_stub);
    stub.set(static_cast<preparing_reservation_stub::CopySubmitLocation>(ADDR(SchedulePlanExecutor, Submit)),
             preparing_reservation_stub::CopySubmitPending_stub);

    const auto results =
        mgr.BatchSubmit("batch_add_location_partial", {make_request(block_keys[0]), make_request(block_keys[1])});
    ASSERT_EQ(2u, results.size());
    ASSERT_TRUE((results[0] == EC_OK && results[1] == EC_NOSPC) ||
                (results[1] == EC_OK && results[0] == EC_NOSPC));

    const std::size_t success_index = results[0] == EC_OK ? 0 : 1;
    const std::size_t failed_index = 1 - success_index;
    EXPECT_TRUE(mgr.HasMigrationTask(kInstance, block_keys[success_index]));
    EXPECT_FALSE(mgr.HasMigrationTask(kInstance, block_keys[failed_index]));
    EXPECT_EQ(1u, mgr.ActiveTaskCount());
    EXPECT_EQ(1u, preparing_reservation_stub::g_pending_copy_promises.size());
    EXPECT_EQ(1u, LocationCount(block_keys[success_index]));
    EXPECT_EQ(0u, LocationCount(block_keys[failed_index]));
    ASSERT_EQ(1u, preparing_reservation_stub::g_deleted_uris.size());
    EXPECT_NE(std::string::npos,
              preparing_reservation_stub::g_deleted_uris.front().ToUriString().find(
                  StringUtil::Uint64ToHex(block_keys[failed_index])));

    // 统一口径：add 成功项已提交 executor（stub pending future），计入其 4 字节；
    // add 失败项（EC_NOSPC）已回滚且未提交，不计入。
    EXPECT_EQ(4u,
              metrics_registry_->GetCounter("data_storage.write_bytes_dispatched_total",
                                            {{"type", ToString(DataStorageType::DATA_STORAGE_TYPE_DUMMY)},
                                             {"unique_name", "cold_01"}})
                  .Get());
}

// 真批量路径在 Create 阶段收到取消后，同样不能覆盖取消态或提交 copy。
TEST_F(MigrationManagerTest, TestBatchCancelDuringPrepareStopsBeforeCopy) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "reserve_batch_cancel_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "reserve_batch_cancel_cold/"));

    const int64_t block_key = 813;
    const std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "batch-cancel-prepare");
    const auto src = GetLocation(block_key, src_loc);
    ASSERT_NE(nullptr, src);
    MigrationManager::MigrationRequest req;
    req.instance_group_name = "group_a";
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src_loc;
    req.src_create_time = src->create_time();
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";
    req.src_specs = src->location_specs();

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    preparing_reservation_stub::Reset();
    preparing_reservation_stub::g_manager = &mgr;
    preparing_reservation_stub::g_expected_reservations = {{kInstance, block_key}};
    preparing_reservation_stub::g_cancel_first_reservation_on_create = true;
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), preparing_reservation_stub::Create_stub);

    const auto results = mgr.BatchSubmit("batch_cancel_preparing", {req});
    ASSERT_EQ(1u, results.size());
    ASSERT_EQ(EC_ERROR, results[0]);
    ASSERT_EQ(std::optional<ErrorCode>(EC_OK), preparing_reservation_stub::g_cancel_result);
    ASSERT_EQ(0u, mgr.GetStats().copy_submitted);
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
    ASSERT_TRUE(WaitFor([&]() { return LocationCount(block_key) == 1u; }));
}

// 一笔 Submit 卡在同步 Create 时，另一个 instance 的 Submit 仍可进入自己的 Create；
// lifecycle shared lock 只与 Stop 互斥，不能重新退化成 Submit 之间的全局串行锁。
TEST_F(MigrationManagerTest, TestSlowCreateDoesNotSerializeOtherInstanceSubmit) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    const std::string other_instance = "parallel_submit_other_instance";
    ASSERT_TRUE(CreateMetaIndexer(other_instance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "parallel_submit_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "parallel_submit_cold/"));

    const std::string first_src = CreateSourceLocation(821, "hot_01", true, "parallel-first");
    const std::string second_src =
        CreateSourceLocationForInstance(other_instance, 822, "hot_01", true, "parallel-second");
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();

    auto make_req = [](const std::string &instance_id, int64_t block_key, const std::string &src_location_id) {
        MigrationManager::MigrationRequest req;
        req.instance_id = instance_id;
        req.block_key = block_key;
        req.src_location_id = src_location_id;
        req.src_storage_name = "hot_01";
        req.dst_storage_name = "cold_01";
        return req;
    };

    submission_concurrency_stub::Reset();
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), submission_concurrency_stub::Create_stub);
    auto first = std::async(std::launch::async,
                            [&]() { return mgr.Submit("parallel_submit_first", make_req(kInstance, 821, first_src)); });
    const bool first_entered = submission_concurrency_stub::WaitForFirstCreate();
    if (!first_entered) {
        submission_concurrency_stub::ReleaseFirstCreate();
        first.wait();
        EXPECT_TRUE(first_entered);
        return;
    }

    auto second = std::async(std::launch::async, [&]() {
        return mgr.Submit("parallel_submit_second", make_req(other_instance, 822, second_src));
    });
    const auto second_status_before_release = second.wait_for(std::chrono::seconds(5));
    submission_concurrency_stub::ReleaseFirstCreate();
    const auto first_ec = first.get();
    const auto second_ec = second.get();

    ASSERT_EQ(std::future_status::ready, second_status_before_release)
        << "second Submit was serialized behind an unrelated slow Create";
    ASSERT_EQ(EC_OK, first_ec);
    ASSERT_EQ(EC_OK, second_ec);
    ASSERT_EQ(2, submission_concurrency_stub::CreateCallCount());
}

// drain 在慢 Create 期间必须快速关 gate，并用已可见的 preparing reservation 做一次 Cancel。
// Cancel 持久化为 kPrepareCancelling，Create 返回后提交线程应回滚，绝不能进入 executor。
TEST_F(MigrationManagerTest, TestDrainCancelsReservationDuringSlowCreate) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "drain_during_create_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "drain_during_create_cold/"));

    const int64_t block_key = 823;
    const std::string src = CreateSourceLocation(block_key, "hot_01", true, "drain-slow-create");
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();

    MigrationManager::MigrationRequest req;
    req.instance_id = kInstance;
    req.block_key = block_key;
    req.src_location_id = src;
    req.src_storage_name = "hot_01";
    req.dst_storage_name = "cold_01";

    submission_concurrency_stub::Reset();
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), submission_concurrency_stub::Create_stub);
    auto submit = std::async(std::launch::async, [&]() { return mgr.Submit("drain_during_create", req); });
    const bool create_entered = submission_concurrency_stub::WaitForFirstCreate();
    if (!create_entered) {
        submission_concurrency_stub::ReleaseFirstCreate();
        submit.wait();
        EXPECT_TRUE(create_entered);
        return;
    }

    auto drain = std::async(std::launch::async, [&]() {
        mgr.BeginDrainingInstance(kInstance);
        const auto keys = mgr.GetActiveBlockKeysForInstance(kInstance);
        return std::make_pair(keys, mgr.BatchCancel(kInstance, keys));
    });
    const auto drain_status_before_release = drain.wait_for(std::chrono::seconds(5));
    submission_concurrency_stub::ReleaseFirstCreate();
    const auto drain_result = drain.get();
    const auto submit_ec = submit.get();
    mgr.EndDrainingInstance(kInstance);

    ASSERT_EQ(std::future_status::ready, drain_status_before_release)
        << "BeginDraining was blocked by slow prepare I/O";
    ASSERT_EQ((std::vector<int64_t>{block_key}), drain_result.first);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), drain_result.second);
    ASSERT_EQ(EC_ERROR, submit_ec);
    ASSERT_EQ(0u, mgr.GetStats().copy_submitted);
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
}

// Stop 先关闭 accepting，再通过 lifecycle unique lock 等待已准入 Submit 完整返回。
// 等待期间新请求应快速失败；Stop 不得在慢 Create owner 仍可能回写 active/pending 时提前清表。
TEST_F(MigrationManagerTest, TestStopWaitsForSlowSubmitLifecycle) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "stop_inflight_submit_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "stop_inflight_submit_cold/"));

    const std::string first_src = CreateSourceLocation(824, "hot_01", true, "stop-first");
    const std::string rejected_src = CreateSourceLocation(825, "hot_01", true, "stop-rejected");
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.Start();

    auto make_req = [&](int64_t block_key, const std::string &src_location_id) {
        MigrationManager::MigrationRequest req;
        req.instance_id = kInstance;
        req.block_key = block_key;
        req.src_location_id = src_location_id;
        req.src_storage_name = "hot_01";
        req.dst_storage_name = "cold_01";
        return req;
    };

    submission_concurrency_stub::Reset();
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), submission_concurrency_stub::Create_stub);
    auto submit =
        std::async(std::launch::async, [&]() { return mgr.Submit("stop_inflight_submit", make_req(824, first_src)); });
    const bool create_entered = submission_concurrency_stub::WaitForFirstCreate();
    if (!create_entered) {
        submission_concurrency_stub::ReleaseFirstCreate();
        submit.wait();
        mgr.Stop();
        EXPECT_TRUE(create_entered);
        return;
    }

    auto stop = std::async(std::launch::async, [&]() { mgr.Stop(); });
    const bool accepting_closed =
        WaitFor([&]() { return !mgr.accepting_copy_submissions_.load(std::memory_order_acquire); });
    const auto stop_status_before_release = stop.wait_for(std::chrono::milliseconds(200));
    const auto rejected_ec = mgr.Submit("stop_rejects_new_submit", make_req(825, rejected_src));
    submission_concurrency_stub::ReleaseFirstCreate();
    const auto submit_ec = submit.get();
    stop.get();

    ASSERT_TRUE(accepting_closed);
    ASSERT_EQ(std::future_status::timeout, stop_status_before_release)
        << "Stop returned while an admitted Submit still owned slow prepare I/O";
    ASSERT_EQ(EC_ERROR, rejected_ec);
    ASSERT_EQ(EC_OK, submit_ec);
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
}

// 缩锁后两个 BatchSubmit 可并发进入准入，但 group slot 的统计与 reservation 必须仍原子；
// 第一批卡在 Create 时，第二批应立即 EC_OUT_OF_LIMIT，且不能进入第二次 Create。
TEST_F(MigrationManagerTest, TestConcurrentBatchSubmitDoesNotOverissueGroupLimit) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "concurrent_limit_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "concurrent_limit_cold/"));

    auto make_req = [&](int64_t block_key) {
        const std::string src = CreateSourceLocation(block_key, "hot_01", true, "group-limit");
        const auto src_location = GetLocation(block_key, src);
        EXPECT_NE(nullptr, src_location);
        MigrationManager::MigrationRequest req;
        req.instance_group_name = "concurrent_limit_group";
        req.instance_id = kInstance;
        req.block_key = block_key;
        req.src_location_id = src;
        req.src_create_time = src_location == nullptr ? 0 : src_location->create_time();
        req.src_storage_name = "hot_01";
        req.dst_storage_name = "cold_01";
        if (src_location != nullptr) {
            req.src_specs = src_location->location_specs();
        }
        return req;
    };
    const auto first_req = make_req(826);
    const auto second_req = make_req(827);
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    const MigrationManager::CopyConcurrencyLimit limit{"concurrent_limit_group", 1};

    submission_concurrency_stub::Reset();
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), submission_concurrency_stub::Create_stub);
    auto first =
        std::async(std::launch::async, [&]() { return mgr.BatchSubmit("concurrent_limit_first", {first_req}, limit); });
    const bool first_entered = submission_concurrency_stub::WaitForFirstCreate();
    if (!first_entered) {
        submission_concurrency_stub::ReleaseFirstCreate();
        first.wait();
        EXPECT_TRUE(first_entered);
        return;
    }

    auto second =
        std::async(std::launch::async, [&]() { return mgr.BatchSubmit("concurrent_limit_second", {second_req}, limit); });
    const auto second_status_before_release = second.wait_for(std::chrono::seconds(5));
    const int create_calls_before_release = submission_concurrency_stub::CreateCallCount();
    submission_concurrency_stub::ReleaseFirstCreate();
    const auto first_results = first.get();
    const auto second_results = second.get();

    ASSERT_EQ(std::future_status::ready, second_status_before_release)
        << "group admission was serialized behind slow Create";
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), first_results);
    ASSERT_EQ((std::vector<ErrorCode>{EC_OUT_OF_LIMIT}), second_results);
    ASSERT_EQ(1, create_calls_before_release);
    ASSERT_EQ(1, submission_concurrency_stub::CreateCallCount());
}

// 覆盖带 src_specs 的 BatchSubmit 主路径，确保批量 Copy 到 cold_02
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

// prepared batch 中每个 item 必须绑定自己的 Mark 快照。三个相邻 item 分别携带
// matching target、different target 和 no-mark，完成回调只能清除第一项，不能按平行下标串位。
TEST_F(MigrationManagerTest, TestBatchSubmitMarkSnapshotsStayAlignedPerItem) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "batch_mark_align_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "batch_mark_align_cold_01/"));
    ASSERT_TRUE(CreateDummyStorage("cold_02", GetPrivateTestRuntimeDataPath() + "batch_mark_align_cold_02/"));

    auto make_prepared_request = [&](int64_t block_key) {
        const std::string src_location_id =
            CreateSourceLocation(block_key, "hot_01", true, "batch-mark-align");
        const auto src_location = GetLocation(block_key, src_location_id);
        EXPECT_NE(nullptr, src_location);

        MigrationManager::MigrationRequest request;
        request.instance_id = kInstance;
        request.block_key = block_key;
        request.src_location_id = src_location_id;
        request.src_storage_name = "hot_01";
        request.dst_storage_name = "cold_01";
        request.retention = MigrationRetention::MIGRATION_RETENTION_KEEP_BOTH;
        if (src_location != nullptr) {
            request.src_create_time = src_location->create_time();
            request.src_specs = src_location->location_specs();
        }
        return request;
    };

    std::vector<MigrationManager::MigrationRequest> requests;
    for (int64_t block_key : {851, 852, 853}) {
        requests.push_back(make_prepared_request(block_key));
    }

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    ASSERT_EQ(EC_OK, mgr.MarkForTieredWrite(kInstance, {851}, "cold_01")); // 与 Copy 目标匹配。
    ASSERT_EQ(EC_OK, mgr.MarkForTieredWrite(kInstance, {852}, "cold_02")); // 独立迁移意图。

    const auto results = mgr.BatchSubmit("prepared_mark_alignment", std::move(requests));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK, EC_OK, EC_OK}), results);

    std::vector<std::string> dst_location_ids;
    for (int64_t block_key : {851, 852, 853}) {
        dst_location_ids.push_back(mgr.GetActiveTaskDstLocation(kInstance, block_key));
        ASSERT_FALSE(dst_location_ids.back().empty());
    }
    for (int64_t block_key : {851, 852, 853}) {
        mgr.OnTaskSuccess(kInstance, block_key);
    }

    ASSERT_EQ(CLS_SERVING, GetLocationStatus(851, dst_location_ids[0]));
    ASSERT_EQ(CLS_SERVING, GetLocationStatus(852, dst_location_ids[1]));
    ASSERT_EQ(CLS_SERVING, GetLocationStatus(853, dst_location_ids[2]));
    ASSERT_EQ("", mgr.GetTieredWriteTarget(kInstance, 851));
    ASSERT_EQ("cold_02", mgr.GetTieredWriteTarget(kInstance, 852));
    ASSERT_EQ("", mgr.GetTieredWriteTarget(kInstance, 853));
    ASSERT_EQ(1u, mgr.GetStats().marks_cleared);
}

// prepared BatchSubmit 的 Create 逐项失败不影响其他 request，且测试必须进入
// 真批量路径，不能再通过空 src_specs fallback 覆盖单条流程。
TEST_F(MigrationManagerTest, TestBatchSubmitPartialFailure) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "batch_pf_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "batch_pf_cold/"));

    std::vector<MigrationManager::MigrationRequest> reqs;
    for (int64_t block_key = 900; block_key <= 902; ++block_key) {
        const std::string src_location_id = CreateSourceLocation(block_key, "hot_01", true, "data0");
        const auto src = GetLocation(block_key, src_location_id);
        ASSERT_NE(nullptr, src);
        MigrationManager::MigrationRequest r;
        r.instance_id = kInstance;
        r.block_key = block_key;
        r.src_location_id = src_location_id;
        r.src_create_time = src->create_time();
        r.src_storage_name = "hot_01";
        r.dst_storage_name = "cold_01";
        r.src_specs = src->location_specs();
        reqs.push_back(std::move(r));
    }

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    prepared_batch_partial_create_stub::Reset();
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), prepared_batch_partial_create_stub::Create_stub);
    auto results = mgr.BatchSubmit("t", reqs);
    ASSERT_EQ(3u, results.size());
    ASSERT_EQ(ErrorCode::EC_OK, results[0]);
    ASSERT_NE(ErrorCode::EC_OK, results[1]);
    ASSERT_EQ(ErrorCode::EC_OK, results[2]);
    ASSERT_EQ(1, prepared_batch_partial_create_stub::g_create_calls);
    ASSERT_EQ((std::vector<std::size_t>{3}), prepared_batch_partial_create_stub::g_create_key_counts);
    ASSERT_EQ(2u, mgr.ActiveTaskCount());
    ASSERT_TRUE(mgr.HasMigrationTask(kInstance, 900));
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, 901));
    ASSERT_TRUE(mgr.HasMigrationTask(kInstance, 902));
    ASSERT_EQ(2u, LocationCount(900));
    ASSERT_EQ(1u, LocationCount(901));
    ASSERT_EQ(2u, LocationCount(902));
}

// 异构 size spec 分散到多个 create_group，某 group 的 Create 失败使 block
// ineligible;后处理 group 里同 block 已成功分配的 URI 不能被跳过泄漏,必须 Delete。
TEST_F(MigrationManagerTest, TestBatchSubmitHeteroSpecCreateFailNoOrphan) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "orphan_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "orphan_cold/"));

    orphan_cleanup_stub::g_created_uris.clear();
    orphan_cleanup_stub::g_deleted_uris.clear();
    orphan_cleanup_stub::g_create_call_count = 0;
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), orphan_cleanup_stub::Create_stub);
    stub.set(ADDR(DataStorageManager, Delete), orphan_cleanup_stub::Delete_stub);

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
    ASSERT_EQ(1u, orphan_cleanup_stub::g_created_uris.size());
    // 关键: 成功 Create 的 URI 必须被 Delete,不能 orphan(修复前会被跳过泄漏)。
    for (const auto &created : orphan_cleanup_stub::g_created_uris) {
        const bool deleted =
            std::any_of(orphan_cleanup_stub::g_deleted_uris.begin(),
                        orphan_cleanup_stub::g_deleted_uris.end(),
                        [&](const DataStorageUri &d) { return d.ToUriString() == created.ToUriString(); });
        ASSERT_TRUE(deleted) << "created URI leaked (not deleted): " << created.ToUriString();
    }
    ASSERT_FALSE(mgr.HasMigrationTask(kInstance, 700));
    ASSERT_EQ(0u, mgr.ActiveTaskCount()) << "failed batch item leaked its preparing reservation";
}

// Batch Create 短返回时无法建立完整的位置映射。整组 request 都必须失败，且实际
// 返回的成功 URI 必须全部删除，不能依赖后面的 per-block spec 数量检查间接兜底。
TEST_F(MigrationManagerTest, TestBatchSubmitShortCreateResultRollsBackWholeGroup) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "shape_short_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "shape_short_cold/"));

    std::vector<MigrationManager::MigrationRequest> requests;
    for (int64_t block_key : {830, 831}) {
        const std::string src_location_id = CreateSourceLocation(block_key, "hot_01", true, "same-size");
        const auto src_location = GetLocation(block_key, src_location_id);
        ASSERT_NE(nullptr, src_location);
        MigrationManager::MigrationRequest request;
        request.instance_id = kInstance;
        request.block_key = block_key;
        request.src_location_id = src_location_id;
        request.src_storage_name = "hot_01";
        request.dst_storage_name = "cold_01";
        request.src_specs = src_location->location_specs();
        requests.push_back(std::move(request));
    }

    create_result_shape_stub::Reset(create_result_shape_stub::Shape::kShort);
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), create_result_shape_stub::Create_stub);
    stub.set(ADDR(DataStorageManager, Delete), create_result_shape_stub::Delete_stub);

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    const auto results = mgr.BatchSubmit("short_create_result", std::move(requests));

    ASSERT_EQ(2u, results.size());
    ASSERT_EQ(EC_ERROR, results[0]);
    ASSERT_EQ(EC_ERROR, results[1]);
    ASSERT_EQ(1u, create_result_shape_stub::g_created_uris.size());
    ASSERT_EQ(create_result_shape_stub::g_created_uris.size(), create_result_shape_stub::g_deleted_uris.size());
    for (const auto &created : create_result_shape_stub::g_created_uris) {
        ASSERT_TRUE(create_result_shape_stub::WasDeleted(created));
    }
    ASSERT_EQ(1u, LocationCount(830));
    ASSERT_EQ(1u, LocationCount(831));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
}

// Batch Create 长返回中的额外 URI 没有对应 CreateEntry。不能只采用前 N 个结果；
// 本组全部成功返回（包括 extra）都必须删除，且不得发布任何 WRITING location。
TEST_F(MigrationManagerTest, TestBatchSubmitLongCreateResultDeletesEveryReturnedUri) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "shape_long_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "shape_long_cold/"));

    std::vector<MigrationManager::MigrationRequest> requests;
    for (int64_t block_key : {832, 833}) {
        const std::string src_location_id = CreateSourceLocation(block_key, "hot_01", true, "same-size");
        const auto src_location = GetLocation(block_key, src_location_id);
        ASSERT_NE(nullptr, src_location);
        MigrationManager::MigrationRequest request;
        request.instance_id = kInstance;
        request.block_key = block_key;
        request.src_location_id = src_location_id;
        request.src_storage_name = "hot_01";
        request.dst_storage_name = "cold_01";
        request.src_specs = src_location->location_specs();
        requests.push_back(std::move(request));
    }

    create_result_shape_stub::Reset(create_result_shape_stub::Shape::kLong);
    Stub stub;
    stub.set(ADDR(DataStorageManager, Create), create_result_shape_stub::Create_stub);
    stub.set(ADDR(DataStorageManager, Delete), create_result_shape_stub::Delete_stub);

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    mgr.DebugEnableCopySubmissionsForTest();
    const auto results = mgr.BatchSubmit("long_create_result", std::move(requests));

    ASSERT_EQ(2u, results.size());
    ASSERT_EQ(EC_ERROR, results[0]);
    ASSERT_EQ(EC_ERROR, results[1]);
    ASSERT_EQ(3u, create_result_shape_stub::g_created_uris.size());
    ASSERT_EQ(create_result_shape_stub::g_created_uris.size(), create_result_shape_stub::g_deleted_uris.size());
    for (const auto &created : create_result_shape_stub::g_created_uris) {
        ASSERT_TRUE(create_result_shape_stub::WasDeleted(created));
    }
    ASSERT_EQ(1u, LocationCount(832));
    ASSERT_EQ(1u, LocationCount(833));
    ASSERT_EQ(0u, mgr.ActiveTaskCount());
}

// ActiveTaskCountForInstances 按 instance 集合过滤，跨 group 不抢 slot。
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

TEST_F(MigrationManagerTest, TestCopyWriteBytesRecorded) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "m_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "m_cold/"));

    MigrationManager mgr(schedule_plan_executor_,
                         meta_manager_,
                         data_storage_manager_,
                         metrics_registry_,
                         nullptr);

    mgr.DebugEnableCopySubmissionsForTest();

    auto get_write_bytes = [&]() {
        return metrics_registry_->GetCounter("data_storage.write_bytes_dispatched_total",
                              {{"type", ToString(DataStorageType::DATA_STORAGE_TYPE_DUMMY)},
                               {"unique_name", "cold_01"}}).Get();
    };

    // 场景1：成功复制
    {
        int64_t block_key = 1000;
        std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "abcdefghij");

        MigrationManager::MigrationRequest req;
        req.instance_id = kInstance;
        req.block_key = block_key;
        req.src_location_id = src_loc;
        req.src_storage_name = "hot_01";
        req.dst_storage_name = "cold_01";
        ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", req));
        // 统一口径：任务成功提交 executor 即计入（Submit 返回时已发生）
        ASSERT_EQ(10u, get_write_bytes());
        mgr.OnTaskSuccess(kInstance, block_key);
        // 完成回调不再重复计入
        ASSERT_EQ(10u, get_write_bytes());
        ASSERT_EQ(10u, metrics_registry_->GetCounter("migration.copy_bytes_total").Get());
    }

    // 场景2：取消仍计入
    {
        int64_t block_key = 1001;
        std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "abcdefghij");

        MigrationManager::MigrationRequest req;
        req.instance_id = kInstance;
        req.block_key = block_key;
        req.src_location_id = src_loc;
        req.src_storage_name = "hot_01";
        req.dst_storage_name = "cold_01";
        ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", req));
        ASSERT_EQ(20u, get_write_bytes()); // 提交 executor 即计入，累计：场景1的10 + 本次10
        // copy 进行中用户取消：任务标记 kCancelling，仍留在活跃表
        ASSERT_EQ(ErrorCode::EC_OK, mgr.Cancel(kInstance, block_key));
        // 完成回调认领到 kWasCancelling：不计入也不走主路径（计数在提交时已发生）
        mgr.OnTaskSuccess(kInstance, block_key);
        ASSERT_EQ(20u, get_write_bytes());
        ASSERT_EQ(10u, metrics_registry_->GetCounter("migration.copy_bytes_total").Get()); // 取消不 promote，不涨
    }

    // 场景3: Copy 失败
    {
        int64_t block_key = 1002;
        std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "abcdefghij");

        MigrationManager::MigrationRequest req;
        req.instance_id = kInstance;
        req.block_key = block_key;
        req.src_location_id = src_loc;
        req.src_storage_name = "hot_01";
        req.dst_storage_name = "cold_01";
        ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", req));
        // copy 失败仍计入：任务已提交 executor，失败属于派发未兑现（累计 20 + 10）
        ASSERT_EQ(30u, get_write_bytes());
        mgr.OnTaskFailed(kInstance, block_key, ErrorCode::EC_IO_ERROR);
        ASSERT_EQ(30u, get_write_bytes()); // 失败回调不再影响计数
        ASSERT_EQ(10u, metrics_registry_->GetCounter("migration.copy_bytes_total").Get());
    }

    // 场景4: 源丢失，仍计入
    {
        int64_t block_key = 1003;
        std::string src_loc = CreateSourceLocation(block_key, "hot_01", true, "abcdefghij");

        MigrationManager::MigrationRequest req;
        req.instance_id = kInstance;
        req.block_key = block_key;
        req.src_location_id = src_loc;
        req.src_storage_name = "hot_01";
        req.dst_storage_name = "cold_01";
        ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", req));
        ASSERT_EQ(40u, get_write_bytes()); // 提交 executor 即计入，累计：30 + 10

        // 破坏源：把源 location 从 SERVING CAS 成 DELETING，模拟 copy 期间源被回收
        auto rc = std::make_shared<RequestContext>("break_source");
        MetaSearcher meta_searcher(meta_manager_->GetMetaIndexer(kInstance));
        std::vector<std::vector<MetaSearcher::LocationCASTask>> cas{
            {MetaSearcher::LocationCASTask{src_loc, CLS_SERVING, CLS_DELETING}}};
        std::vector<std::vector<ErrorCode>> cas_results;
        ASSERT_EQ(ErrorCode::EC_OK, meta_searcher.BatchCASLocationStatus(rc.get(), {block_key}, cas, cas_results));

        // 完成回调：计数已在提交时发生；此处走 source_lost 失败收尾，不再影响计数
        mgr.OnTaskSuccess(kInstance, block_key);
        ASSERT_EQ(40u, get_write_bytes());
        ASSERT_EQ(10u, metrics_registry_->GetCounter("migration.copy_bytes_total").Get()); // 源丢失按失败收尾，不涨
    }

}

// ============ Copy 准入策略：CheckCopyAdmission ============

namespace {

CacheLocationConstPtr MakeLocation(const std::string &id,
                                   const std::string &storage_name,
                                   CacheLocationStatus status,
                                   int64_t create_time = 0) {
    std::string uri = "dummy://" + storage_name + "/blk?size=8";
    auto loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY, 1, std::vector<LocationSpec>{LocationSpec("TP0", uri)});
    loc->set_id(id);
    loc->set_status(status);
    loc->set_create_time(create_time);
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

TEST_F(MigrationManagerTest, TestDispatchSkipsMarkWhenDedupQueryFails) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    mark_query_result_stub::Reset();
    mark_query_result_stub::g_aggregate_ec = EC_ERROR;
    mark_query_result_stub::g_per_key_ecs = {EC_ERROR};
    mark_query_result_stub::g_properties = {{}};
    Stub stub;
    stub.set(ADDR(MetaIndexer, GetProperties), mark_query_result_stub::GetProperties_stub);

    CacheLocationMap loc_map;
    auto src_loc = MakeLocation("loc_src", "hot_01", CLS_SERVING);
    loc_map[src_loc->id()] = src_loc;
    MigrationManager::DispatchBatchParams params;
    params.do_mark = true;
    params.dedup_marks = true;

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_, metrics_registry_);
    mgr.DebugEnableCopySubmissionsForTest();
    const auto result =
        mgr.DispatchMigrationBatch("mark_query_error", kInstance, "hot_01", "cold_01", {1}, {loc_map}, params);

    ASSERT_EQ(0, result.mark_submitted);
    ASSERT_EQ(0u, mgr.GetStats().marks_added);
    ASSERT_EQ(1u, metrics_registry_->GetCounter("migration.mark_query_errors_total").Get());
}

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

TEST_F(MigrationManagerTest, TestCopyFailureBackoffSwitchesToHealthySource) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_, metrics_registry_);
    const std::string src = "hot_01";
    const std::string dst = "cold_01";
    const int64_t block_key = 81;

    auto failed_source = MakeLocation("loc_failed", src, CLS_SERVING, 100);
    auto healthy_source = MakeLocation("loc_healthy", src, CLS_SERVING, 200);

    MigrationManager::CopyTaskContext failed_ctx;
    failed_ctx.instance_id = kInstance;
    failed_ctx.block_key = block_key;
    failed_ctx.src_location_id = failed_source->id();
    failed_ctx.src_create_time = failed_source->create_time();
    failed_ctx.src_storage_name = src;
    failed_ctx.dst_storage_name = dst;
    mgr.RecordCopySourceFailure(failed_ctx, EC_PARTIAL_OK);

    CacheLocationMap loc_map;
    loc_map[failed_source->id()] = failed_source;
    loc_map[healthy_source->id()] = healthy_source;
    const auto admission = mgr.CheckCopyAdmission(kInstance, block_key, loc_map, src, dst);

    ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kAccept, admission.status);
    ASSERT_NE(nullptr, admission.src_location);
    EXPECT_EQ(healthy_source->id(), admission.src_location->id());
    const auto stats = mgr.GetStats();
    EXPECT_EQ(1u, stats.source_failures_recorded);
    EXPECT_EQ(1u, stats.source_retries_suppressed);
    EXPECT_EQ(1u, stats.source_switches);
    EXPECT_EQ(1u, stats.source_failure_entries);
    EXPECT_EQ(0u, stats.no_usable_source);
}

TEST_F(MigrationManagerTest, TestTaskFailureHooksSourceBackoffBeforeRemovingActiveTask) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    auto failed_source = MakeLocation("loc_failed_hook", "hot_01", CLS_SERVING, 100);
    auto healthy_source = MakeLocation("loc_healthy_hook", "hot_01", CLS_SERVING, 200);
    MigrationManager::CopyTaskContext ctx;
    ctx.instance_id = kInstance;
    ctx.block_key = 810;
    ctx.src_location_id = failed_source->id();
    ctx.src_create_time = failed_source->create_time();
    ctx.src_storage_name = "hot_01";
    ctx.dst_storage_name = "cold_01";
    ctx.state = MigrationManager::CopyTaskState::kRunning;
    ctx.submit_time = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(mgr.task_mutex_);
        ASSERT_TRUE(mgr.InsertActiveTaskLocked(ctx));
    }

    mgr.OnTaskFailed(kInstance, ctx.block_key, EC_NOENT);
    EXPECT_FALSE(mgr.HasMigrationTask(kInstance, ctx.block_key));

    CacheLocationMap loc_map;
    loc_map.emplace(failed_source->id(), failed_source);
    loc_map.emplace(healthy_source->id(), healthy_source);
    const auto admission =
        mgr.CheckCopyAdmission(kInstance, ctx.block_key, loc_map, ctx.src_storage_name, ctx.dst_storage_name);
    ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kAccept, admission.status);
    ASSERT_NE(nullptr, admission.src_location);
    EXPECT_EQ(healthy_source->id(), admission.src_location->id());
}

TEST_F(MigrationManagerTest, TestFailedSourceStaysLowerPriorityAfterBackoffExpires) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_, metrics_registry_);
    const std::string src = "hot_01";
    const std::string dst = "cold_01";
    const int64_t block_key = 85;

    auto failed_source = MakeLocation("loc_failed", src, CLS_SERVING, 100);
    auto healthy_source = MakeLocation("loc_healthy", src, CLS_SERVING, 200);

    MigrationManager::CopyTaskContext failed_ctx;
    failed_ctx.instance_id = kInstance;
    failed_ctx.block_key = block_key;
    failed_ctx.src_location_id = failed_source->id();
    failed_ctx.src_create_time = failed_source->create_time();
    failed_ctx.src_storage_name = src;
    failed_ctx.dst_storage_name = dst;
    mgr.RecordCopySourceFailure(failed_ctx, EC_PARTIAL_OK);

    // Reclaimer 的采样周期可能长于首次 5s backoff。即使退避窗口已经结束，存在从未失败的
    // 完整副本时也应优先换源，而不是因为坏源再次 eligible 就回到 unordered_map 的随机顺序。
    const auto failed_key = mgr.MakeCopySourceFailureKey(failed_ctx);
    {
        std::lock_guard<std::mutex> lock(mgr.copy_source_failure_mutex_);
        auto iter = mgr.copy_source_failures_.find(failed_key);
        ASSERT_NE(mgr.copy_source_failures_.end(), iter);
        iter->second.retry_after = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
    }

    CacheLocationMap loc_map;
    loc_map[failed_source->id()] = failed_source;
    loc_map[healthy_source->id()] = healthy_source;
    const auto admission = mgr.CheckCopyAdmission(kInstance, block_key, loc_map, src, dst);

    ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kAccept, admission.status);
    ASSERT_NE(nullptr, admission.src_location);
    EXPECT_EQ(healthy_source->id(), admission.src_location->id());
    const auto stats = mgr.GetStats();
    EXPECT_EQ(0u, stats.source_retries_suppressed);
    EXPECT_EQ(1u, stats.source_switches);
}

TEST_F(MigrationManagerTest, TestCopyFailureBackoffStopsRetryWhenNoAlternativeSource) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_, metrics_registry_);
    const std::string src = "hot_01";
    const std::string dst = "cold_01";
    const int64_t block_key = 82;
    auto failed_source = MakeLocation("loc_failed", src, CLS_SERVING, 100);

    MigrationManager::CopyTaskContext failed_ctx;
    failed_ctx.instance_id = kInstance;
    failed_ctx.block_key = block_key;
    failed_ctx.src_location_id = failed_source->id();
    failed_ctx.src_create_time = failed_source->create_time();
    failed_ctx.src_storage_name = src;
    failed_ctx.dst_storage_name = dst;
    mgr.RecordCopySourceFailure(failed_ctx, EC_PARTIAL_OK);

    CacheLocationMap loc_map;
    loc_map[failed_source->id()] = failed_source;
    const auto admission = mgr.CheckCopyAdmission(kInstance, block_key, loc_map, src, dst);

    EXPECT_EQ(MigrationManager::CopyAdmissionStatus::kSourceRetrySuppressed, admission.status);
    EXPECT_EQ(nullptr, admission.src_location);
    const auto stats = mgr.GetStats();
    EXPECT_EQ(1u, stats.source_retries_suppressed);
    EXPECT_EQ(1u, stats.no_usable_source);
}

TEST_F(MigrationManagerTest, TestCopyFailureIdentityDoesNotPoisonRecreatedLocationOrOtherTarget) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    const std::string src = "hot_01";
    const int64_t block_key = 83;
    auto old_source = MakeLocation("loc_reused", src, CLS_SERVING, 100);

    MigrationManager::CopyTaskContext failed_ctx;
    failed_ctx.instance_id = kInstance;
    failed_ctx.block_key = block_key;
    failed_ctx.src_location_id = old_source->id();
    failed_ctx.src_create_time = old_source->create_time();
    failed_ctx.src_storage_name = src;
    failed_ctx.dst_storage_name = "cold_01";
    mgr.RecordCopySourceFailure(failed_ctx, EC_PARTIAL_OK);

    CacheLocationMap recreated_map;
    auto recreated_source = MakeLocation("loc_reused", src, CLS_SERVING, 101);
    recreated_map[recreated_source->id()] = recreated_source;
    const auto recreated = mgr.CheckCopyAdmission(kInstance, block_key, recreated_map, src, "cold_01");
    ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kAccept, recreated.status);
    ASSERT_EQ(recreated_source.get(), recreated.src_location);

    CacheLocationMap old_map;
    old_map[old_source->id()] = old_source;
    const auto other_target = mgr.CheckCopyAdmission(kInstance, block_key, old_map, src, "archive_01");
    ASSERT_EQ(MigrationManager::CopyAdmissionStatus::kAccept, other_target.status);
    ASSERT_EQ(old_source.get(), other_target.src_location);
}

TEST_F(MigrationManagerTest, TestCopySuccessClearsSourceFailureBackoff) {
    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    const std::string src = "hot_01";
    const std::string dst = "cold_01";
    const int64_t block_key = 84;
    auto source = MakeLocation("loc_source", src, CLS_SERVING, 100);

    MigrationManager::CopyTaskContext ctx;
    ctx.instance_id = kInstance;
    ctx.block_key = block_key;
    ctx.src_location_id = source->id();
    ctx.src_create_time = source->create_time();
    ctx.src_storage_name = src;
    ctx.dst_storage_name = dst;
    mgr.RecordCopySourceFailure(ctx, EC_PARTIAL_OK);

    CacheLocationMap loc_map;
    loc_map[source->id()] = source;
    EXPECT_EQ(MigrationManager::CopyAdmissionStatus::kSourceRetrySuppressed,
              mgr.CheckCopyAdmission(kInstance, block_key, loc_map, src, dst).status);

    mgr.ClearCopySourceFailure(ctx);
    const auto admission = mgr.CheckCopyAdmission(kInstance, block_key, loc_map, src, dst);
    EXPECT_EQ(MigrationManager::CopyAdmissionStatus::kAccept, admission.status);
    EXPECT_EQ(0u, mgr.GetStats().source_failure_entries);
}

TEST_F(MigrationManagerTest, TestGuardRecoveryRebuildsQuarantineAndFailsBeforeClearingOnScanSetupError) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    constexpr int64_t block_key = 859;
    constexpr uint64_t total_bytes = 7;
    const std::string operation_id = "recovered-unknown-operation";

    MigrationCopyGuard guard;
    guard.set_schema_version(MigrationCopyGuard::kCurrentSchemaVersion);
    guard.set_state(MigrationCopyGuardState::MCGS_UNKNOWN);
    guard.set_operation_id(operation_id);
    guard.set_source_location_id("recovered-source");
    guard.set_source_location_create_time(1);
    guard.set_source_storage_name("hot_01");
    guard.set_target_storage_name("cold_01");
    guard.set_total_bytes(total_bytes);
    guard.set_backend_task_ids({"pace-recovered-task"});

    MetaSearcher searcher(meta_manager_->GetMetaIndexer(kInstance));
    auto rc = std::make_shared<RequestContext>("async_guard_recovery_target");
    auto target = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://cold/recovered?size=7")});
    target->set_migration_copy_guard(guard);
    std::vector<std::string> target_ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(&searcher, rc.get(), {block_key}, {target}, target_ids));

    auto group = std::make_shared<InstanceGroup>();
    group->set_name("recovery-group");
    group->set_cache_config(std::make_shared<CacheConfig>());
    auto instance = std::make_shared<InstanceInfo>();
    instance->set_instance_id(kInstance);
    instance->set_instance_group_name(group->name());
    async_guard_recovery_stub::groups = {group};
    async_guard_recovery_stub::instances = {instance};
    async_guard_recovery_stub::list_groups_result = EC_OK;
    async_guard_recovery_stub::list_instances_result = EC_OK;

    Stub stub;
    stub.set(ADDR(RegistryManager, ListInstanceGroup), async_guard_recovery_stub::ListInstanceGroupStub);
    stub.set(ADDR(RegistryManager, ListInstanceInfo), async_guard_recovery_stub::ListInstanceInfoStub);
    auto registry = std::make_shared<RegistryManager>("", metrics_registry_);
    MigrationManager mgr(
        schedule_plan_executor_, meta_manager_, data_storage_manager_, metrics_registry_, nullptr, registry);

    ASSERT_TRUE(mgr.RecoverAsyncCopyGuards());
    const auto quarantined = mgr.ListAsyncCopyQuarantine(group->name());
    ASSERT_EQ(1u, quarantined.size());
    EXPECT_EQ(operation_id, quarantined[0].guard.operation_id());
    EXPECT_EQ(total_bytes, mgr.GetStats().async_copy_quarantine_bytes);
    EXPECT_EQ(0u, mgr.GetStats().async_copy_inflight_operations);

    // A later setup/list failure must return before replacing the previously
    // rebuilt accounting tables; leader code will demote instead of serving a
    // false-empty quarantine view.
    async_guard_recovery_stub::list_groups_result = EC_ERROR;
    EXPECT_FALSE(mgr.RecoverAsyncCopyGuards());
    EXPECT_EQ(1u, mgr.ListAsyncCopyQuarantine(group->name()).size());

    async_guard_recovery_stub::groups.clear();
    async_guard_recovery_stub::instances.clear();
    async_guard_recovery_stub::list_groups_result = EC_OK;
    async_guard_recovery_stub::list_instances_result = EC_OK;
}

TEST_F(MigrationManagerTest, TestGuardRecoveryResumesActiveTaskWithPersistedHandles) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    constexpr int64_t block_key = 860;
    const std::string operation_id = "recovered-active-operation";
    const std::vector<std::string> task_ids = {"pace-recovered-active-task"};

    MigrationCopyGuard guard;
    guard.set_schema_version(MigrationCopyGuard::kCurrentSchemaVersion);
    guard.set_state(MigrationCopyGuardState::MCGS_ACTIVE);
    guard.set_operation_id(operation_id);
    guard.set_source_location_id("recovered-active-source");
    guard.set_source_location_create_time(1);
    guard.set_source_storage_name("hot_01");
    guard.set_target_storage_name("cold_01");
    guard.set_total_bytes(7);
    guard.set_backend_task_ids(task_ids);
    guard.set_create_time_us(TimestampUtil::GetCurrentTimeUs());

    MetaSearcher searcher(meta_manager_->GetMetaIndexer(kInstance));
    auto rc = std::make_shared<RequestContext>("async_guard_recovery_active_target");
    auto target = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://cold/recovered-active?size=7")});
    target->set_migration_copy_guard(guard);
    std::vector<std::string> target_ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(&searcher, rc.get(), {block_key}, {target}, target_ids));

    auto group = std::make_shared<InstanceGroup>();
    group->set_name("recovery-active-group");
    group->set_cache_config(std::make_shared<CacheConfig>());
    auto instance = std::make_shared<InstanceInfo>();
    instance->set_instance_id(kInstance);
    instance->set_instance_group_name(group->name());
    async_guard_recovery_stub::groups = {group};
    async_guard_recovery_stub::instances = {instance};
    async_guard_recovery_stub::list_groups_result = EC_OK;
    async_guard_recovery_stub::list_instances_result = EC_OK;
    async_guard_recovery_stub::resume_calls = 0;
    async_guard_recovery_stub::resumed_storage.clear();
    async_guard_recovery_stub::resumed_operation_id.clear();
    async_guard_recovery_stub::resumed_task_ids.clear();

    Stub stub;
    stub.set(ADDR(RegistryManager, ListInstanceGroup), async_guard_recovery_stub::ListInstanceGroupStub);
    stub.set(ADDR(RegistryManager, ListInstanceInfo), async_guard_recovery_stub::ListInstanceInfoStub);
    stub.set(ADDR(SchedulePlanExecutor, ResumeAsyncCopy), async_guard_recovery_stub::ResumeAsyncCopyStub);
    auto registry = std::make_shared<RegistryManager>("", metrics_registry_);
    MigrationManager mgr(
        schedule_plan_executor_, meta_manager_, data_storage_manager_, metrics_registry_, nullptr, registry);

    ASSERT_TRUE(mgr.RecoverAsyncCopyGuards());
    EXPECT_EQ(1u, async_guard_recovery_stub::resume_calls);
    EXPECT_EQ("hot_01", async_guard_recovery_stub::resumed_storage);
    EXPECT_EQ(operation_id, async_guard_recovery_stub::resumed_operation_id);
    EXPECT_EQ(task_ids, async_guard_recovery_stub::resumed_task_ids);
    EXPECT_TRUE(mgr.HasMigrationTask(kInstance, block_key));
    EXPECT_TRUE(mgr.ListAsyncCopyQuarantine(group->name()).empty());
    EXPECT_EQ(1u, mgr.GetStats().async_copy_inflight_operations);

    async_guard_recovery_stub::groups.clear();
    async_guard_recovery_stub::instances.clear();
}

TEST_F(MigrationManagerTest, TestRemoteAsyncAcceptancePersistsActiveTaskHandles) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "async_accept_hot/"));
    const int64_t block_key = 86;
    const std::string source_id = CreateSourceLocation(block_key, "hot_01", false, "payload");
    const auto source = GetLocation(block_key, source_id);
    ASSERT_TRUE(source);

    MetaSearcher searcher(meta_manager_->GetMetaIndexer(kInstance));
    auto rc = std::make_shared<RequestContext>("async_accept_target");
    auto target = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://cold/target?size=7")});
    std::vector<std::string> target_ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(&searcher, rc.get(), {block_key}, {target}, target_ids));
    ASSERT_EQ(1u, target_ids.size());

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    MigrationManager::CopyTaskContext ctx;
    ctx.instance_id = kInstance;
    ctx.block_key = block_key;
    ctx.src_location_id = source_id;
    ctx.src_create_time = source->create_time();
    ctx.src_storage_name = "hot_01";
    ctx.dst_storage_name = "cold_01";
    ctx.dst_location_id = target_ids[0];
    ctx.copy_execution_mode = MigrationCopyExecutionMode::ASYNC_REQUIRED;
    ctx.async_operation_id = "async-operation-86";
    ctx.async_guard_create_time_us = 100;
    ctx.total_bytes = 7;
    ctx.state = MigrationManager::CopyTaskState::kRunning;
    ASSERT_EQ(MigrationManager::GuardMutationResult::kAppliedDurably,
              mgr.UpdateAsyncCopyGuard(
                  ctx, MigrationCopyGuardState::MCGS_NONE, MigrationCopyGuardState::MCGS_SUBMITTING, {}, ""));
    {
        std::lock_guard<std::mutex> lock(mgr.task_mutex_);
        ASSERT_TRUE(mgr.InsertActiveTaskLocked(ctx));
    }

    AsyncCopyRemoteSubmitResult remote;
    remote.status = EC_OK;
    remote.accepted = true;
    remote.operation_id = ctx.async_operation_id;
    remote.backend_task_ids = {"pace-task-86"};
    MigrationManager::CopyTaskContext accepted_ctx;
    bool cancel_requested = true;
    ASSERT_TRUE(mgr.PersistRemoteAsyncCopyAcceptance(
        kInstance, block_key, remote, accepted_ctx, cancel_requested));
    EXPECT_FALSE(cancel_requested);
    EXPECT_EQ(remote.backend_task_ids, accepted_ctx.async_backend_task_ids);

    const auto guarded_target = GetLocation(block_key, target_ids[0]);
    ASSERT_TRUE(guarded_target);
    ASSERT_TRUE(guarded_target->has_migration_copy_guard());
    EXPECT_EQ(MigrationCopyGuardState::MCGS_ACTIVE, guarded_target->migration_copy_guard().state());
    EXPECT_EQ(remote.backend_task_ids, guarded_target->migration_copy_guard().backend_task_ids());
}

TEST_F(MigrationManagerTest, TestCancellationBeforeRemoteAcceptanceCannotBeOverwrittenByActive) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "async_cancel_hot/"));
    const int64_t block_key = 87;
    const std::string source_id = CreateSourceLocation(block_key, "hot_01", false, "payload");
    const auto source = GetLocation(block_key, source_id);
    ASSERT_TRUE(source);

    MetaSearcher searcher(meta_manager_->GetMetaIndexer(kInstance));
    auto rc = std::make_shared<RequestContext>("async_cancel_target");
    auto target = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://cold/target?size=7")});
    std::vector<std::string> target_ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(&searcher, rc.get(), {block_key}, {target}, target_ids));

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    MigrationManager::CopyTaskContext ctx;
    ctx.instance_id = kInstance;
    ctx.block_key = block_key;
    ctx.src_location_id = source_id;
    ctx.src_create_time = source->create_time();
    ctx.src_storage_name = "hot_01";
    ctx.dst_storage_name = "cold_01";
    ctx.dst_location_id = target_ids[0];
    ctx.copy_execution_mode = MigrationCopyExecutionMode::ASYNC_REQUIRED;
    ctx.async_operation_id = "async-operation-87";
    ctx.async_guard_create_time_us = 100;
    ctx.total_bytes = 7;
    ctx.state = MigrationManager::CopyTaskState::kRunning;
    ASSERT_EQ(MigrationManager::GuardMutationResult::kAppliedDurably,
              mgr.UpdateAsyncCopyGuard(
                  ctx, MigrationCopyGuardState::MCGS_NONE, MigrationCopyGuardState::MCGS_SUBMITTING, {}, ""));
    {
        std::lock_guard<std::mutex> lock(mgr.task_mutex_);
        ASSERT_TRUE(mgr.InsertActiveTaskLocked(ctx));
    }

    ASSERT_EQ(EC_OK, mgr.Cancel(kInstance, block_key));
    AsyncCopyRemoteSubmitResult remote;
    remote.status = EC_OK;
    remote.accepted = true;
    remote.operation_id = ctx.async_operation_id;
    remote.backend_task_ids = {"pace-task-87"};
    MigrationManager::CopyTaskContext accepted_ctx;
    bool cancel_requested = false;
    ASSERT_TRUE(mgr.PersistRemoteAsyncCopyAcceptance(
        kInstance, block_key, remote, accepted_ctx, cancel_requested));
    EXPECT_TRUE(cancel_requested);

    // Every non-initial guard transition checks both operation-id and state.
    // A stale ACTIVE writer must not regress the durable CANCELLING intent.
    EXPECT_EQ(MigrationManager::GuardMutationResult::kNotApplied,
              mgr.UpdateAsyncCopyGuard(accepted_ctx,
                                       MigrationCopyGuardState::MCGS_ACTIVE,
                                       MigrationCopyGuardState::MCGS_UNKNOWN,
                                       remote.backend_task_ids,
                                       "stale-active-writer"));

    const auto guarded_target = GetLocation(block_key, target_ids[0]);
    ASSERT_TRUE(guarded_target);
    ASSERT_TRUE(guarded_target->has_migration_copy_guard());
    EXPECT_EQ(MigrationCopyGuardState::MCGS_CANCELLING, guarded_target->migration_copy_guard().state());
    EXPECT_EQ(remote.backend_task_ids, guarded_target->migration_copy_guard().backend_task_ids());
}

TEST_F(MigrationManagerTest, TestCancelAndCompletionHaveExactlyOneTerminalOwner) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "async_race_cold/"));
    constexpr int64_t block_key = 871;

    MetaSearcher searcher(meta_manager_->GetMetaIndexer(kInstance));
    auto rc = std::make_shared<RequestContext>("async_cancel_completion_race");
    auto target =
        std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                        1,
                                        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://cold_01/race?size=7")});
    std::vector<std::string> target_ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(&searcher, rc.get(), {block_key}, {target}, target_ids));

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    MigrationManager::CopyTaskContext ctx;
    ctx.instance_group_name = "group-race";
    ctx.instance_id = kInstance;
    ctx.block_key = block_key;
    ctx.src_location_id = "source-race";
    ctx.src_create_time = 1;
    ctx.src_storage_name = "hot_01";
    ctx.dst_storage_name = "cold_01";
    ctx.dst_location_id = target_ids[0];
    ctx.copy_execution_mode = MigrationCopyExecutionMode::ASYNC_REQUIRED;
    ctx.async_operation_id = "async-operation-race";
    ctx.async_backend_task_ids = {"pace-task-race"};
    ctx.async_credit_active = true;
    ctx.total_bytes = 7;
    ctx.submit_time = std::chrono::steady_clock::now();
    ctx.state = MigrationManager::CopyTaskState::kRunning;
    ASSERT_EQ(MigrationManager::GuardMutationResult::kAppliedDurably,
              mgr.UpdateAsyncCopyGuard(ctx,
                                       MigrationCopyGuardState::MCGS_NONE,
                                       MigrationCopyGuardState::MCGS_ACTIVE,
                                       ctx.async_backend_task_ids,
                                       ""));
    {
        std::lock_guard<std::mutex> lock(mgr.task_mutex_);
        ASSERT_TRUE(mgr.InsertActiveTaskLocked(ctx));
    }
    ASSERT_TRUE(mgr.RestoreAsyncCopyInflightCredit(ctx));

    const auto transition_mutex = mgr.GetAsyncTransitionMutex(kInstance, block_key);
    ASSERT_TRUE(transition_mutex);
    std::unique_lock<std::mutex> hold_transition(*transition_mutex);
    std::atomic<int> ready{0};
    auto cancel = std::async(std::launch::async, [&]() {
        ready.fetch_add(1, std::memory_order_release);
        return mgr.Cancel(kInstance, block_key);
    });
    auto completion = std::async(std::launch::async, [&]() {
        ready.fetch_add(1, std::memory_order_release);
        return mgr.OnTaskFailed(kInstance, block_key, EC_ERROR);
    });
    while (ready.load(std::memory_order_acquire) != 2) {
        std::this_thread::yield();
    }
    hold_transition.unlock();

    const ErrorCode cancel_result = cancel.get();
    EXPECT_TRUE(cancel_result == EC_OK || cancel_result == EC_EXIST || cancel_result == EC_NOENT);
    EXPECT_TRUE(completion.get());
    EXPECT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
    const auto stats = mgr.GetStats();
    EXPECT_EQ(1u, stats.copy_failed + stats.copy_cancelled);
    EXPECT_EQ(0u, stats.async_copy_inflight_operations);
    EXPECT_EQ(0u, stats.async_copy_quarantine_operations);

    const auto persisted = GetLocation(block_key, target_ids[0]);
    if (persisted) {
        EXPECT_EQ(CLS_DELETING, persisted->status());
        EXPECT_FALSE(persisted->has_migration_copy_guard());
    }
}

TEST_F(MigrationManagerTest, TestAppliedFinalizationRetriesOnlySyncWithoutPhantomQuarantine) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "async_sync_retry_hot/"));
    constexpr int64_t block_key = 872;
    const std::string source_id = CreateSourceLocation(block_key, "hot_01", false, "payload");
    const auto source = GetLocation(block_key, source_id);
    ASSERT_TRUE(source);

    MetaSearcher searcher(meta_manager_->GetMetaIndexer(kInstance));
    auto rc = std::make_shared<RequestContext>("async_guard_sync_retry");
    auto target = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://cold/sync-retry?size=7")});
    std::vector<std::string> target_ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(&searcher, rc.get(), {block_key}, {target}, target_ids));

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    MigrationManager::CopyTaskContext ctx;
    ctx.instance_group_name = "group-sync-retry";
    ctx.instance_id = kInstance;
    ctx.block_key = block_key;
    ctx.src_location_id = source_id;
    ctx.src_create_time = source->create_time();
    ctx.src_storage_name = "hot_01";
    ctx.dst_storage_name = "cold_01";
    ctx.dst_location_id = target_ids[0];
    ctx.retention = MigrationRetention::MIGRATION_RETENTION_KEEP_BOTH;
    ctx.copy_execution_mode = MigrationCopyExecutionMode::ASYNC_REQUIRED;
    ctx.async_operation_id = "async-operation-sync-retry";
    ctx.async_backend_task_ids = {"pace-task-sync-retry"};
    ctx.async_credit_active = true;
    ctx.total_bytes = 7;
    ctx.submit_time = std::chrono::steady_clock::now();
    ctx.state = MigrationManager::CopyTaskState::kRunning;
    ASSERT_EQ(MigrationManager::GuardMutationResult::kAppliedDurably,
              mgr.UpdateAsyncCopyGuard(ctx,
                                       MigrationCopyGuardState::MCGS_NONE,
                                       MigrationCopyGuardState::MCGS_ACTIVE,
                                       ctx.async_backend_task_ids,
                                       ""));
    {
        std::lock_guard<std::mutex> lock(mgr.task_mutex_);
        ASSERT_TRUE(mgr.InsertActiveTaskLocked(ctx));
    }
    ASSERT_TRUE(mgr.RestoreAsyncCopyInflightCredit(ctx));

    Stub stub;
    g_guard_sync_call_count.store(0, std::memory_order_release);
    stub.set(ADDR(MetaIndexer, Sync), FailFirstGuardSyncStub);

    EXPECT_FALSE(mgr.OnTaskSuccess(kInstance, block_key));
    const auto applied = GetLocation(block_key, target_ids[0]);
    ASSERT_TRUE(applied);
    EXPECT_EQ(CLS_SERVING, applied->status());
    EXPECT_FALSE(applied->has_migration_copy_guard());
    EXPECT_TRUE(mgr.HasMigrationTask(kInstance, block_key));
    EXPECT_EQ(1u, mgr.GetStats().async_copy_inflight_operations);
    EXPECT_EQ(0u, mgr.GetStats().async_copy_quarantine_operations);

    EXPECT_TRUE(mgr.OnTaskSuccess(kInstance, block_key));
    EXPECT_FALSE(mgr.HasMigrationTask(kInstance, block_key));
    const auto stats = mgr.GetStats();
    EXPECT_EQ(1u, stats.copy_completed);
    EXPECT_EQ(0u, stats.copy_failed);
    EXPECT_EQ(0u, stats.async_copy_unknown);
    EXPECT_EQ(0u, stats.async_copy_inflight_operations);
    EXPECT_EQ(0u, stats.async_copy_quarantine_operations);
}

TEST_F(MigrationManagerTest, TestDefinitePreSubmitFailureDoesNotEnterRemoteUnknownQuarantine) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    constexpr int64_t block_key = 88;
    MetaSearcher searcher(meta_manager_->GetMetaIndexer(kInstance));
    auto rc = std::make_shared<RequestContext>("pre_submit_failure_target");
    auto target = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://cold/target?size=7")});
    MigrationCopyGuard other_guard;
    other_guard.set_schema_version(MigrationCopyGuard::kCurrentSchemaVersion);
    other_guard.set_state(MigrationCopyGuardState::MCGS_SUBMITTING);
    other_guard.set_operation_id("persisted-other-operation");
    target->set_migration_copy_guard(other_guard);
    std::vector<std::string> target_ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(&searcher, rc.get(), {block_key}, {target}, target_ids));

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    MigrationManager::CopyTaskContext ctx;
    ctx.instance_group_name = "group-88";
    ctx.instance_id = kInstance;
    ctx.block_key = block_key;
    ctx.src_location_id = "source-88";
    ctx.src_create_time = 88;
    ctx.src_storage_name = "hot_01";
    ctx.dst_storage_name = "cold_01";
    ctx.dst_location_id = target_ids[0];
    ctx.copy_execution_mode = MigrationCopyExecutionMode::ASYNC_REQUIRED;
    ctx.async_operation_id = "never-submitted-operation";
    ctx.async_credit_active = true;
    ctx.total_bytes = 7;
    ctx.submit_time = std::chrono::steady_clock::now();
    ctx.state = MigrationManager::CopyTaskState::kRunning;
    {
        std::lock_guard<std::mutex> lock(mgr.task_mutex_);
        ASSERT_TRUE(mgr.InsertActiveTaskLocked(ctx));
    }
    mgr.RestoreAsyncCopyInflightCredit(ctx);
    ASSERT_EQ(1u, mgr.GetStats().async_copy_inflight_operations);

    // The exact-owner finalize intentionally fails because the persisted
    // guard belongs to another operation.  Since no backend handoff happened,
    // this must release local admission credit without inventing a remote
    // UNKNOWN operation or deleting the other owner's target.
    mgr.CompleteCopyTaskAsFailed(ctx, "guard_install_failed_before_submit", false);
    const auto stats = mgr.GetStats();
    EXPECT_EQ(0u, stats.async_copy_inflight_operations);
    EXPECT_EQ(0u, stats.async_copy_quarantine_operations);
    EXPECT_EQ(0u, stats.async_copy_unknown);
    EXPECT_EQ(0u, stats.active_copy_tasks);

    const auto persisted = GetLocation(block_key, target_ids[0]);
    ASSERT_TRUE(persisted);
    ASSERT_TRUE(persisted->has_migration_copy_guard());
    EXPECT_EQ("persisted-other-operation", persisted->migration_copy_guard().operation_id());
    EXPECT_EQ(CLS_WRITING, persisted->status());
}

TEST_F(MigrationManagerTest, TestBreakGlassReleaseRequiresUnknownGuardAndClearsQuarantine) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "break_glass_cold/"));
    constexpr int64_t block_key = 89;
    constexpr uint64_t total_bytes = 7;
    const std::string operation_id = "async-operation-89";
    const std::string group_name = "group-89";

    MigrationCopyGuard guard;
    guard.set_schema_version(MigrationCopyGuard::kCurrentSchemaVersion);
    guard.set_state(MigrationCopyGuardState::MCGS_UNKNOWN);
    guard.set_operation_id(operation_id);
    guard.set_source_location_id("source-89");
    guard.set_source_location_create_time(89);
    guard.set_source_storage_name("hot_01");
    guard.set_target_storage_name("cold_01");
    guard.set_total_bytes(total_bytes);
    guard.set_backend_task_ids({"pace-task-89"});
    guard.set_last_error("remote_terminal_state_unknown");

    MetaSearcher searcher(meta_manager_->GetMetaIndexer(kInstance));
    auto rc = std::make_shared<RequestContext>("break_glass_target");
    auto target = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://cold_01/target-89?size=7")});
    target->set_migration_copy_guard(guard);
    std::vector<std::string> target_ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(&searcher, rc.get(), {block_key}, {target}, target_ids));
    ASSERT_EQ(1u, target_ids.size());

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    {
        std::lock_guard<std::mutex> lock(mgr.async_copy_usage_mutex_);
        mgr.async_copy_quarantine_by_operation_.emplace(
            operation_id,
            MigrationManager::AsyncCopyQuarantineRecord{
                group_name, kInstance, block_key, target_ids[0], guard});
        auto &usage = mgr.async_copy_usage_by_group_[group_name];
        usage.quarantine_operations = 1;
        usage.quarantine_bytes = total_bytes;
    }

    ASSERT_EQ(1u, mgr.ListAsyncCopyQuarantine(group_name).size());
    EXPECT_EQ(EC_BADARGS, mgr.BreakGlassReleaseAsyncCopy(operation_id, "", "ticket-89"));
    EXPECT_EQ(EC_NOENT, mgr.BreakGlassReleaseAsyncCopy("missing-operation", "operator-89", "ticket-89"));
    EXPECT_EQ(EC_OK, mgr.BreakGlassReleaseAsyncCopy(operation_id, "operator-89", "ticket-89"));
    EXPECT_TRUE(mgr.ListAsyncCopyQuarantine(group_name).empty());
    const auto stats = mgr.GetStats();
    EXPECT_EQ(0u, stats.async_copy_quarantine_operations);
    EXPECT_EQ(0u, stats.async_copy_quarantine_bytes);

    // The exact-owner CAS must clear the durable guard before the asynchronous
    // target deletion is handed to SchedulePlanExecutor.  Depending on worker
    // timing the location is either already gone or observably DELETING, but it
    // must never remain a reusable WRITING location with a silently erased guard.
    const auto released = GetLocation(block_key, target_ids[0]);
    if (released) {
        EXPECT_EQ(CLS_DELETING, released->status());
        EXPECT_FALSE(released->has_migration_copy_guard());
    }
}

TEST_F(MigrationManagerTest, TestBreakGlassRejectsPersistentStateOrOperationMismatch) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    MetaSearcher searcher(meta_manager_->GetMetaIndexer(kInstance));
    auto rc = std::make_shared<RequestContext>("break_glass_mismatch_targets");

    auto make_guard = [](const std::string &operation_id, MigrationCopyGuardState state) {
        MigrationCopyGuard guard;
        guard.set_schema_version(MigrationCopyGuard::kCurrentSchemaVersion);
        guard.set_state(state);
        guard.set_operation_id(operation_id);
        guard.set_source_location_id("source-mismatch");
        guard.set_source_location_create_time(1);
        guard.set_source_storage_name("hot_01");
        guard.set_target_storage_name("cold_01");
        guard.set_total_bytes(7);
        guard.set_backend_task_ids({"pace-task-mismatch"});
        return guard;
    };
    auto active_target =
        std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                        1,
                                        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://cold/active?size=7")});
    active_target->set_migration_copy_guard(make_guard("active-state-operation", MigrationCopyGuardState::MCGS_ACTIVE));
    auto other_owner_target = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://cold/other-owner?size=7")});
    other_owner_target->set_migration_copy_guard(
        make_guard("persistent-other-operation", MigrationCopyGuardState::MCGS_UNKNOWN));

    std::vector<std::string> target_ids;
    ASSERT_EQ(
        EC_OK,
        BatchAddLocationForTest(&searcher, rc.get(), {890, 891}, {active_target, other_owner_target}, target_ids));
    ASSERT_EQ(2u, target_ids.size());

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    const auto active_record_guard = make_guard("active-state-operation", MigrationCopyGuardState::MCGS_UNKNOWN);
    const auto wrong_owner_record_guard = make_guard("recorded-operation", MigrationCopyGuardState::MCGS_UNKNOWN);
    {
        std::lock_guard<std::mutex> lock(mgr.async_copy_usage_mutex_);
        mgr.async_copy_quarantine_by_operation_.emplace(
            "active-state-operation",
            MigrationManager::AsyncCopyQuarantineRecord{
                "group-mismatch", kInstance, 890, target_ids[0], active_record_guard});
        mgr.async_copy_quarantine_by_operation_.emplace(
            "recorded-operation",
            MigrationManager::AsyncCopyQuarantineRecord{
                "group-mismatch", kInstance, 891, target_ids[1], wrong_owner_record_guard});
        auto &usage = mgr.async_copy_usage_by_group_["group-mismatch"];
        usage.quarantine_operations = 2;
        usage.quarantine_bytes = 14;
    }

    EXPECT_EQ(EC_MISMATCH,
              mgr.BreakGlassReleaseAsyncCopy("active-state-operation", "operator", "external-terminal-proof"));
    EXPECT_EQ(EC_MISMATCH, mgr.BreakGlassReleaseAsyncCopy("recorded-operation", "operator", "external-terminal-proof"));
    EXPECT_EQ(2u, mgr.ListAsyncCopyQuarantine("group-mismatch").size());
    EXPECT_EQ(MigrationCopyGuardState::MCGS_ACTIVE, GetLocation(890, target_ids[0])->migration_copy_guard().state());
    EXPECT_EQ("persistent-other-operation", GetLocation(891, target_ids[1])->migration_copy_guard().operation_id());
}

// 多个 target location 分别覆盖不同 specs，联合覆盖完整时不应 kAccept。
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

TEST_F(MigrationManagerTest, TestStopDetachesRecoverableAsyncCopyWithoutQuarantine) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    constexpr int64_t block_key = 1003;
    constexpr uint64_t total_bytes = 7;
    const std::string operation_id = "stop-detach-operation";

    MetaSearcher searcher(meta_manager_->GetMetaIndexer(kInstance));
    auto rc = std::make_shared<RequestContext>("stop_detach_target");
    auto target = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://cold/stop-detach?size=7")});
    std::vector<std::string> target_ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(&searcher, rc.get(), {block_key}, {target}, target_ids));
    ASSERT_EQ(1u, target_ids.size());

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    ASSERT_EQ(EC_OK, mgr.Start());
    MigrationManager::CopyTaskContext ctx;
    ctx.instance_group_name = "stop-detach-group";
    ctx.instance_id = kInstance;
    ctx.block_key = block_key;
    ctx.src_location_id = "stop-detach-source";
    ctx.src_create_time = 1;
    ctx.src_storage_name = "hot_01";
    ctx.dst_storage_name = "cold_01";
    ctx.dst_location_id = target_ids[0];
    ctx.copy_execution_mode = MigrationCopyExecutionMode::ASYNC_REQUIRED;
    ctx.async_operation_id = operation_id;
    ctx.async_backend_task_ids = {"pace-stop-detach-task"};
    ctx.async_credit_active = true;
    ctx.async_guard_create_time_us = TimestampUtil::GetCurrentTimeUs();
    ctx.total_bytes = total_bytes;
    ctx.submit_time = std::chrono::steady_clock::now();
    ctx.state = MigrationManager::CopyTaskState::kRunning;
    ASSERT_EQ(MigrationManager::GuardMutationResult::kAppliedDurably,
              mgr.UpdateAsyncCopyGuard(ctx,
                                       MigrationCopyGuardState::MCGS_NONE,
                                       MigrationCopyGuardState::MCGS_ACTIVE,
                                       ctx.async_backend_task_ids,
                                       ""));
    {
        std::lock_guard<std::mutex> lock(mgr.task_mutex_);
        ASSERT_TRUE(mgr.InsertActiveTaskLocked(ctx));
    }
    ASSERT_TRUE(mgr.RestoreAsyncCopyInflightCredit(ctx));

    mgr.Stop();

    EXPECT_EQ(0u, mgr.ActiveTaskCount());
    EXPECT_TRUE(mgr.ListAsyncCopyQuarantine(ctx.instance_group_name).empty());
    const auto persisted = GetLocation(block_key, target_ids[0]);
    ASSERT_TRUE(persisted);
    ASSERT_TRUE(persisted->has_migration_copy_guard());
    EXPECT_EQ(MigrationCopyGuardState::MCGS_ACTIVE, persisted->migration_copy_guard().state());
    EXPECT_EQ(operation_id, persisted->migration_copy_guard().operation_id());
    EXPECT_EQ(ctx.async_backend_task_ids, persisted->migration_copy_guard().backend_task_ids());
}

TEST_F(MigrationManagerTest, TestStopQuarantinesSubmittingAsyncCopyWithoutTaskIds) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    constexpr int64_t block_key = 1004;

    MetaSearcher searcher(meta_manager_->GetMetaIndexer(kInstance));
    auto rc = std::make_shared<RequestContext>("stop_submitting_target");
    auto target = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://cold/stop-submitting?size=7")});
    std::vector<std::string> target_ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(&searcher, rc.get(), {block_key}, {target}, target_ids));
    ASSERT_EQ(1u, target_ids.size());

    MigrationManager mgr(schedule_plan_executor_, meta_manager_, data_storage_manager_);
    ASSERT_EQ(EC_OK, mgr.Start());
    MigrationManager::CopyTaskContext ctx;
    ctx.instance_group_name = "stop-submitting-group";
    ctx.instance_id = kInstance;
    ctx.block_key = block_key;
    ctx.src_location_id = "stop-submitting-source";
    ctx.src_create_time = 1;
    ctx.src_storage_name = "hot_01";
    ctx.dst_storage_name = "cold_01";
    ctx.dst_location_id = target_ids[0];
    ctx.copy_execution_mode = MigrationCopyExecutionMode::ASYNC_REQUIRED;
    ctx.async_operation_id = "stop-submitting-operation";
    ctx.async_credit_active = true;
    ctx.async_guard_create_time_us = TimestampUtil::GetCurrentTimeUs();
    ctx.total_bytes = 7;
    ctx.submit_time = std::chrono::steady_clock::now();
    ctx.state = MigrationManager::CopyTaskState::kRunning;
    ASSERT_EQ(MigrationManager::GuardMutationResult::kAppliedDurably,
              mgr.UpdateAsyncCopyGuard(ctx,
                                       MigrationCopyGuardState::MCGS_NONE,
                                       MigrationCopyGuardState::MCGS_SUBMITTING,
                                       {},
                                       ""));
    {
        std::lock_guard<std::mutex> lock(mgr.task_mutex_);
        ASSERT_TRUE(mgr.InsertActiveTaskLocked(ctx));
    }
    ASSERT_TRUE(mgr.RestoreAsyncCopyInflightCredit(ctx));

    mgr.Stop();

    const auto quarantined = mgr.ListAsyncCopyQuarantine(ctx.instance_group_name);
    ASSERT_EQ(1u, quarantined.size());
    EXPECT_EQ(ctx.async_operation_id, quarantined[0].guard.operation_id());
    const auto persisted = GetLocation(block_key, target_ids[0]);
    ASSERT_TRUE(persisted);
    ASSERT_TRUE(persisted->has_migration_copy_guard());
    EXPECT_EQ(MigrationCopyGuardState::MCGS_UNKNOWN, persisted->migration_copy_guard().state());
}

// GetActiveBlockKeysForInstance 返回指定 instance 的活跃 block_key 列表（用于 drain 前 BatchCancel）。
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

// BatchCancel + GetActiveBlockKeysForInstance 配合实现 drain：
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

// draining gate 阻止该 instance 的新提交（Submit + BatchSubmit 两路），
// EndDraining 后恢复；其他 instance 不受影响。
TEST_F(MigrationManagerTest, TestDrainingInstanceGateRejectsSubmit) {
    ASSERT_TRUE(CreateMetaIndexer(kInstance));
    const std::string kOther = "other_instance";
    ASSERT_TRUE(CreateMetaIndexer(kOther));
    ASSERT_TRUE(CreateDummyStorage("hot_01", GetPrivateTestRuntimeDataPath() + "draingate_hot/"));
    ASSERT_TRUE(CreateDummyStorage("cold_01", GetPrivateTestRuntimeDataPath() + "draingate_cold/"));

    const std::string src1 = CreateSourceLocation(901, "hot_01", true, "d1");
    const std::string src2 = CreateSourceLocation(902, "hot_01", true, "d2");
    const std::string other_src = CreateSourceLocationForInstance(kOther, 903, "hot_01", true, "d3");

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
    auto prepared = make_req(kInstance, 902, src2);
    const auto src2_location = GetLocation(902, src2);
    ASSERT_NE(nullptr, src2_location);
    prepared.src_create_time = src2_location->create_time();
    prepared.src_specs = src2_location->location_specs();
    auto batch_res = mgr.BatchSubmit("t", {prepared});
    ASSERT_EQ(1u, batch_res.size());
    ASSERT_NE(ErrorCode::EC_OK, batch_res[0]);
    ASSERT_TRUE(mgr.GetActiveBlockKeysForInstance(kInstance).empty());

    // 另一个未 drain 的 instance 提交不受影响（draining 是 per-instance）
    ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", make_req(kOther, 903, other_src)));
    ASSERT_EQ(1u, mgr.GetActiveBlockKeysForInstance(kOther).size());

    // EndDraining 后：kInstance 恢复可提交
    mgr.EndDrainingInstance(kInstance);
    ASSERT_EQ(ErrorCode::EC_OK, mgr.Submit("t", make_req(kInstance, 901, src1)));
    ASSERT_EQ(1u, mgr.GetActiveBlockKeysForInstance(kInstance).size());
}

// HasActiveCopyTargetLocation 按 (instance_id, block_key) 作用域判断：
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
