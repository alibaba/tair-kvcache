#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <thread>
#include <tuple>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/migration_strategy.h"
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_backend.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/event_report_backend.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/manager/cache_location_view.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/manager/meta_searcher_manager.h"
#include "kv_cache_manager/manager/migration_manager.h"
#include "kv_cache_manager/manager/reclaimer_task_supervisor.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/manager/startup_config_loader.h"
#include "kv_cache_manager/manager/write_location_manager.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/utils.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "stub.h"

namespace {
static const std::string default_storage_configs(
    "[{\"type\":\"file\",\"is_available\":true,\"global_unique_name\":\"nfs_01\",\"storage_spec\":{"
    "\"root_path\":\"/tmp/nfs/\",\"key_count_per_file\":8}}]");
} // namespace

namespace kv_cache_manager {

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
} // namespace

namespace mark_query_read_error_stub {
ErrorCode ReadError_stub(void * /*obj*/,
                         const std::string & /*instance_id*/,
                         const std::vector<int64_t> &block_keys,
                         std::vector<MigrationManager::MarkQueryResult> &out) {
    out.assign(block_keys.size(), MigrationManager::MarkQueryResult{});
    for (auto &result : out) {
        result.state = MigrationManager::MarkQueryState::kReadError;
        result.ec = EC_ERROR;
        // 故意携带 stale target，证明调用方依据 state 而不是 target 是否为空做判断。
        result.target = "cold_01";
    }
    return EC_ERROR;
}
} // namespace mark_query_read_error_stub

namespace remove_instance_reclaimer_state_stub {
CacheReclaimer *reclaimer = nullptr;
bool called = false;
bool observed_paused = false;

void Reset(CacheReclaimer *value) {
    reclaimer = value;
    called = false;
    observed_paused = false;
}

ErrorCode RemoveInstance_stub(void * /*obj*/,
                              RequestContext * /*request_context*/,
                              const std::string & /*instance_group*/,
                              const std::string & /*instance_id*/) {
    called = true;
    if (reclaimer != nullptr) {
        observed_paused = reclaimer->IsPaused();
    }
    return EC_ERROR;
}
} // namespace remove_instance_reclaimer_state_stub

class MockDataStorageBackend : public DataStorageBackend {
public:
    explicit MockDataStorageBackend(std::shared_ptr<MetricsRegistry> mr) : DataStorageBackend(std::move(mr)) {}
    MOCK_METHOD(DataStorageType, GetType, (), (override));
    MOCK_METHOD(bool, Available, (), (override));
    MOCK_METHOD(double, GetStorageUsageRatio, (const std::string &), (const, override));
    MOCK_METHOD(ErrorCode, DoOpen, (const StorageConfig &, const std::string &), (override));
    MOCK_METHOD(ErrorCode, Close, (), (override));
    MOCK_METHOD((std::vector<std::pair<ErrorCode, DataStorageUri>>),
                Create,
                (const std::vector<std::string> &, size_t, const std::string &, std::function<void()>),
                (override));
    MOCK_METHOD(std::vector<ErrorCode>,
                Delete,
                (const std::vector<DataStorageUri> &, const std::string &, std::function<void()>),
                (override));
    MOCK_METHOD(std::vector<bool>, Exist, (const std::vector<DataStorageUri> &), (override));
    MOCK_METHOD(std::vector<bool>, MightExist, (const std::vector<DataStorageUri> &), (override));
    MOCK_METHOD(std::vector<ErrorCode>, Lock, (const std::vector<DataStorageUri> &), (override));
    MOCK_METHOD(std::vector<ErrorCode>, UnLock, (const std::vector<DataStorageUri> &), (override));
};

// wraps a real DataStorageBackend but intercepts MightExist() with a
// user-provided function; all other calls are delegated to the real
// backend
class MightExistInterceptor : public DataStorageBackend {
public:
    using MightExistFunc = std::function<std::vector<bool>(const std::vector<DataStorageUri> &)>;

    MightExistInterceptor(std::shared_ptr<DataStorageBackend> delegate, MightExistFunc fn)
        : DataStorageBackend(delegate->metrics_registry_), delegate_(std::move(delegate)), fn_(std::move(fn)) {
        SetOpen(delegate_->IsOpen());
        SetAvailable(true);
    }

    DataStorageType GetType() override { return delegate_->GetType(); }
    bool Available() override { return IsAvailable() && delegate_->Available(); }
    double GetStorageUsageRatio(const std::string &t) const override { return delegate_->GetStorageUsageRatio(t); }
    const StorageConfig &GetStorageConfig() override { return delegate_->GetStorageConfig(); }
    ErrorCode DoOpen(const StorageConfig &c, const std::string &t) override { return delegate_->DoOpen(c, t); }
    ErrorCode Close() override { return delegate_->Close(); }
    std::vector<std::pair<ErrorCode, DataStorageUri>>
    Create(const std::vector<std::string> &k, size_t s, const std::string &t, std::function<void()> cb) override {
        return delegate_->Create(k, s, t, std::move(cb));
    }
    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &u, const std::string &t, std::function<void()> cb) override {
        return delegate_->Delete(u, t, std::move(cb));
    }
    std::vector<bool> Exist(const std::vector<DataStorageUri> &u) override { return delegate_->Exist(u); }
    std::vector<bool> MightExist(const std::vector<DataStorageUri> &u) override { return fn_(u); }
    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &u) override { return delegate_->Lock(u); }
    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &u) override { return delegate_->UnLock(u); }

private:
    std::shared_ptr<DataStorageBackend> delegate_;
    MightExistFunc fn_;
};

// A deterministic write gate for ReportEvent ordering tests.  Sleeping in a
// test cannot prove which request acquired the snapshot fence first; blocking
// the persistent Upsert lets the test observe and release that exact point.
class ControllableMetaLocalBackend : public MetaLocalBackend {
public:
    void BlockNextLocationRead() {
        std::lock_guard<std::mutex> lock(control_mutex_);
        block_next_location_read_ = true;
        location_read_entered_ = false;
        release_location_read_ = false;
    }

    bool WaitUntilLocationReadEntered(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(control_mutex_);
        return control_cv_.wait_for(lock, timeout, [&] { return location_read_entered_; });
    }

    void ReleaseLocationRead() {
        std::lock_guard<std::mutex> lock(control_mutex_);
        release_location_read_ = true;
        control_cv_.notify_all();
    }

    void BlockNextUpsert() {
        std::lock_guard<std::mutex> lock(control_mutex_);
        block_next_upsert_ = true;
        upsert_entered_ = false;
        release_upsert_ = false;
    }

    bool WaitUntilUpsertEntered(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(control_mutex_);
        return control_cv_.wait_for(lock, timeout, [&] { return upsert_entered_; });
    }

    void ReleaseUpsert() {
        std::lock_guard<std::mutex> lock(control_mutex_);
        release_upsert_ = true;
        control_cv_.notify_all();
    }

    void FailKeyOnNextUpsert(int64_t key) {
        std::lock_guard<std::mutex> lock(control_mutex_);
        fail_key_on_next_upsert_ = key;
    }

    size_t GetSyncCallCount() {
        std::lock_guard<std::mutex> lock(control_mutex_);
        return sync_call_count_;
    }

    std::vector<ErrorCode> Upsert(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  const CacheLocationMapVector &locations,
                                  const PropertyMapVector &properties) noexcept override {
        MaybeBlockUpsert();

        std::optional<int64_t> failed_key;
        {
            std::lock_guard<std::mutex> lock(control_mutex_);
            failed_key = fail_key_on_next_upsert_;
            fail_key_on_next_upsert_.reset();
        }
        if (!failed_key.has_value()) {
            return MetaLocalBackend::Upsert(request_context, keys, locations, properties);
        }

        std::vector<ErrorCode> results(keys.size(), EC_ERROR);
        for (size_t i = 0; i < keys.size(); ++i) {
            if (keys[i] == failed_key.value()) {
                continue;
            }
            const auto one_result = MetaLocalBackend::Upsert(request_context,
                                                             KeyTypeVec{keys[i]},
                                                             CacheLocationMapVector{locations[i]},
                                                             PropertyMapVector{properties[i]});
            results[i] = one_result.empty() ? EC_ERROR : one_result.front();
        }
        return results;
    }

    std::vector<std::vector<ErrorCode>> GetLocations(RequestContext *request_context,
                                                     const KeyTypeVec &keys,
                                                     const LocationIdsPerKey &location_ids,
                                                     LocationsPerKey &out_locations) noexcept override {
        MaybeBlockLocationRead();
        return MetaLocalBackend::GetLocations(request_context, keys, location_ids, out_locations);
    }

    std::vector<std::vector<ErrorCode>>
    GetLocationsWithKeyStatus(RequestContext *request_context,
                              const KeyTypeVec &keys,
                              const LocationIdsPerKey &location_ids,
                              LocationsPerKey &out_locations,
                              std::vector<ErrorCode> &out_key_error_codes) noexcept override {
        MaybeBlockLocationRead();
        return MetaLocalBackend::GetLocationsWithKeyStatus(
            request_context, keys, location_ids, out_locations, out_key_error_codes);
    }

    std::vector<ErrorCode> GetLocations(RequestContext *request_context,
                                        const KeyTypeVec &keys,
                                        CacheLocationMapVector &out_locations) noexcept override {
        MaybeBlockLocationRead();
        return MetaLocalBackend::GetLocations(request_context, keys, out_locations);
    }

    std::vector<ErrorCode> GetLocationValues(RequestContext *request_context,
                                             const KeyTypeVec &keys,
                                             LocationsPerKey &out_locations) noexcept override {
        MaybeBlockLocationRead();
        return MetaLocalBackend::GetLocationValues(request_context, keys, out_locations);
    }

    std::vector<ErrorCode> GetLocationValuesCompact(RequestContext *request_context,
                                                    const KeyType *keys,
                                                    size_t key_count,
                                                    CompactLocationsPerKey &out_locations) noexcept override {
        MaybeBlockLocationRead();
        return MetaLocalBackend::GetLocationValuesCompact(request_context, keys, key_count, out_locations);
    }

    bool Sync(const KeyTypeVec &keys) noexcept override {
        {
            std::lock_guard<std::mutex> lock(control_mutex_);
            ++sync_call_count_;
        }
        return MetaLocalBackend::Sync(keys);
    }

private:
    void MaybeBlockLocationRead() {
        std::unique_lock<std::mutex> lock(control_mutex_);
        if (!block_next_location_read_) {
            return;
        }
        block_next_location_read_ = false;
        location_read_entered_ = true;
        control_cv_.notify_all();
        control_cv_.wait(lock, [&] { return release_location_read_; });
    }

    void MaybeBlockUpsert() {
        std::unique_lock<std::mutex> lock(control_mutex_);
        if (!block_next_upsert_) {
            return;
        }
        block_next_upsert_ = false;
        upsert_entered_ = true;
        control_cv_.notify_all();
        control_cv_.wait(lock, [&] { return release_upsert_; });
    }

    std::mutex control_mutex_;
    std::condition_variable control_cv_;
    bool block_next_upsert_ = false;
    bool upsert_entered_ = false;
    bool release_upsert_ = false;
    bool block_next_location_read_ = false;
    bool location_read_entered_ = false;
    bool release_location_read_ = false;
    std::optional<int64_t> fail_key_on_next_upsert_;
    size_t sync_call_count_ = 0;
};

class DeleteRecordingBackend : public DataStorageBackend {
public:
    explicit DeleteRecordingBackend(std::shared_ptr<DataStorageBackend> delegate)
        : DataStorageBackend(delegate->metrics_registry_), delegate_(std::move(delegate)) {
        SetOpen(delegate_->IsOpen());
        SetAvailable(true);
    }

    DataStorageType GetType() override { return delegate_->GetType(); }
    bool Available() override { return delegate_->Available(); }
    double GetStorageUsageRatio(const std::string &trace_id) const override {
        return delegate_->GetStorageUsageRatio(trace_id);
    }
    const StorageConfig &GetStorageConfig() override { return delegate_->GetStorageConfig(); }
    ErrorCode DoOpen(const StorageConfig &config, const std::string &trace_id) override {
        return delegate_->DoOpen(config, trace_id);
    }
    ErrorCode Close() override { return delegate_->Close(); }
    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &trace_id,
                                                             std::function<void()> cb) override {
        return delegate_->Create(keys, size_per_key, trace_id, std::move(cb));
    }
    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &uris, const std::string &trace_id, std::function<void()> cb) override {
        deleted_uri_count_.fetch_add(uris.size(), std::memory_order_relaxed);
        return delegate_->Delete(uris, trace_id, std::move(cb));
    }
    std::vector<bool> Exist(const std::vector<DataStorageUri> &uris) override { return delegate_->Exist(uris); }
    std::vector<bool> MightExist(const std::vector<DataStorageUri> &uris) override {
        return delegate_->MightExist(uris);
    }
    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &uris) override { return delegate_->Lock(uris); }
    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &uris) override { return delegate_->UnLock(uris); }

    size_t DeletedUriCount() const { return deleted_uri_count_.load(std::memory_order_relaxed); }

private:
    std::shared_ptr<DataStorageBackend> delegate_;
    std::atomic<size_t> deleted_uri_count_{0};
};

class CacheManagerTest : public TESTBASE {
public:
    void SetUp() override {
        cache_manager_ = createCacheManager();
        request_context_.reset(new RequestContext("fake_trace_id"));
    }

    void TearDown() override {}

    std::unique_ptr<CacheManager> createCacheManager() {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        std::shared_ptr<MetricsRegistry> &metrics_registry = metrics_registry_;
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
        // Mark registry as recovered since data was injected directly (not via backend)
        registry_manager_->recover_complete_.store(true);
        std::unique_ptr<CacheManager> cache_manager =
            std::make_unique<CacheManager>(metrics_registry, registry_manager_);

        EXPECT_TRUE(cache_manager->Init());

        // load first because we need default group
        // in real usage, we load startup config after recover
        StartupConfigLoader loader;
        loader.Init(registry_manager_);
        loader.Load("");

        EXPECT_EQ(EC_OK, cache_manager->DoRecover());

        // 注册 tiered 测试用的冷热 dummy 后端。MarkForTieredWrite 要求 target 已注册，
        // stale location 测试需要真实 backend 以便用 MightExistInterceptor 构造"meta SERVING 但数据已丢"。
        RegisterDummyStorage("hot_01");
        RegisterDummyStorage("cold_01");

        return cache_manager;
    }

    bool RegisterDummyStorage(const std::string &name) {
        auto spec = std::make_shared<DummyStorageSpec>();
        spec->set_root_path(GetPrivateTestRuntimeDataPath() + name + "/");
        spec->set_key_count_per_file(1);
        StorageConfig config;
        config.set_type(DataStorageType::DATA_STORAGE_TYPE_DUMMY);
        config.set_global_unique_name(name);
        config.set_storage_spec(spec);
        auto rc = std::make_shared<RequestContext>("reg_storage");
        auto dsm = registry_manager_->data_storage_manager();
        if (dsm->RegisterStorage(rc.get(), name, config) != EC_OK) {
            return false;
        }
        // 默认让数据 MightExist=true，使仅建 meta location(未真正写数据文件)的 tiered 测试仍视其为有效；
        // 需要模拟"数据已丢"的用例可在测试内覆盖为返回 false 的 interceptor。
        auto original = dsm->storage_map_[name];
        dsm->storage_map_[name] = std::make_shared<MightExistInterceptor>(
            original, [](const std::vector<DataStorageUri> &uris) { return std::vector<bool>(uris.size(), true); });
        return true;
    }

    bool RegisterNfsStorage(const std::string &name) {
        auto spec = std::make_shared<NfsStorageSpec>();
        spec->set_root_path(GetPrivateTestRuntimeDataPath() + name + "/");
        spec->set_key_count_per_file(1);
        StorageConfig config;
        config.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
        config.set_global_unique_name(name);
        config.set_storage_spec(spec);
        auto rc = std::make_shared<RequestContext>("reg_nfs_storage");
        return registry_manager_->data_storage_manager()->RegisterStorage(rc.get(), name, config) == EC_OK;
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

    ModelDeployment createModelDeploymentWithEaglePop() {
        ModelDeployment model_deployment = createModelDeployment();
        model_deployment.set_use_eagle_pop(true);
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

    void EnableTieredMigrationStrategy(const std::string &group_name = "default",
                                       const std::string &source_storage = "hot_01",
                                       const std::string &target_storage = "cold_01",
                                       int64_t mark_timeout_ms = MigrationMarkMethod::kDefaultTimeoutMs) {
        auto iter = registry_manager_->instance_group_configs_.find(group_name);
        ASSERT_TRUE(iter != registry_manager_->instance_group_configs_.end());
        ASSERT_TRUE(iter->second != nullptr);
        ASSERT_TRUE(iter->second->cache_config_ != nullptr);

        auto strategy = std::make_shared<MigrationStrategy>();
        strategy->set_source_storage_name(source_storage);
        strategy->set_target_storage_name(target_storage);
        strategy->set_trigger_threshold(0.01);
        MigrationMethods methods;
        methods.mutable_mark().set_enabled(true);
        methods.mutable_mark().set_timeout_ms(mark_timeout_ms);
        strategy->set_methods(methods);
        strategy->set_retention(MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE);
        iter->second->cache_config_->set_migration_strategies({strategy});
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

    std::shared_ptr<EventReportBackend>
    InstallEventReportBackend(const std::string &storage_name = "event_report_default") {
        const std::string group_name = registry_manager_->GetInstanceGroupName("test_instance");
        auto group = registry_manager_->instance_group_configs_.at(group_name);
        group->set_event_report_storage_candidates({storage_name});

        auto backend = std::make_shared<EventReportBackend>(metrics_registry_);
        StorageConfig config;
        config.set_global_unique_name(storage_name);
        config.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2);
        auto spec = std::make_shared<EventReportStorageSpec>();
        spec->set_liveness_check_interval_ms(10);
        config.set_storage_spec(spec);
        if (backend->Open(config, "report_event_ordering_test") != EC_OK) {
            return nullptr;
        }
        backend->SetSnapshotMinIntervalMsForTest(0);
        registry_manager_->data_storage_manager_->storage_map_[storage_name] = backend;
        return backend;
    }

    std::string InitializeEventReporter(const std::string &instance_id,
                                        const std::string &host,
                                        proto::meta::StorageType storage_type) {
        proto::meta::ReportEventRequest register_request;
        register_request.set_instance_id(instance_id);
        register_request.set_host_ip_port(host);
        register_request.set_storage_type(storage_type);
        auto *register_event = register_request.add_events();
        register_event->set_event_type(proto::meta::EVENT_NODE_REGISTER);
        register_event->mutable_node_register()->add_mediums("mem");
        register_event->mutable_node_register()->add_mediums("disk");
        proto::meta::ReportEventResponse register_response;
        EXPECT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &register_request, &register_response));

        proto::meta::ReportEventRequest snapshot_request;
        snapshot_request.set_instance_id(instance_id);
        snapshot_request.set_host_ip_port(host);
        snapshot_request.set_storage_type(storage_type);
        auto *snapshot_event = snapshot_request.add_events();
        snapshot_event->set_event_type(proto::meta::EVENT_BLOCK_SNAPSHOT);
        snapshot_event->mutable_block_snapshot();
        proto::meta::ReportEventResponse snapshot_response;
        EXPECT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &snapshot_request, &snapshot_response));
        EXPECT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(snapshot_response.committed_snapshot_version()));
        return snapshot_response.committed_snapshot_version();
    }

    ControllableMetaLocalBackend *InstallControllableMetaBackend() {
        auto indexer = cache_manager_->meta_indexer_manager_->GetMetaIndexer("test_instance");
        if (!indexer) {
            return nullptr;
        }
        auto config = std::make_shared<MetaStorageBackendConfig>();
        auto controlled = std::make_unique<ControllableMetaLocalBackend>();
        if (controlled->Init("test_instance", config) != EC_OK || controlled->Open() != EC_OK) {
            return nullptr;
        }
        auto *controlled_raw = controlled.get();
        indexer->backend_manager_->persistent_backend_->Close();
        indexer->backend_manager_->persistent_backend_ = std::move(controlled);
        indexer->backend_manager_->cache_backend_.reset();
        return controlled_raw;
    }

    static proto::meta::ReportEventRequest
    MakeSnapshotRequest(const std::string &host, const std::vector<std::pair<int64_t, std::string>> &key_sources) {
        proto::meta::ReportEventRequest request;
        request.set_instance_id("test_instance");
        request.set_host_ip_port(host);
        request.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
        auto *event = request.add_events();
        event->set_event_type(proto::meta::EVENT_BLOCK_SNAPSHOT);
        auto *snapshot = event->mutable_block_snapshot();
        for (const auto &[key, source] : key_sources) {
            auto *block = snapshot->add_blocks();
            block->set_block_key(std::to_string(key));
            block->set_medium("mem");
            auto *spec = block->add_specs();
            spec->set_name("tp0");
            spec->set_uri("event_report://" + host + "/mem?source=" + source);
        }
        return request;
    }

    static proto::meta::ReportEventRequest
    MakeAddRequest(const std::string &host, int64_t key, const std::string &source) {
        proto::meta::ReportEventRequest request;
        request.set_instance_id("test_instance");
        request.set_host_ip_port(host);
        request.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
        auto *event = request.add_events();
        event->set_event_type(proto::meta::EVENT_BLOCK_ADD);
        event->mutable_block_add()->set_block_key(std::to_string(key));
        event->mutable_block_add()->set_medium("mem");
        auto *spec = event->mutable_block_add()->add_specs();
        spec->set_name("tp0");
        spec->set_uri("event_report://" + host + "/mem?source=" + source);
        return request;
    }

    std::pair<ErrorCode, proto::meta::ReportEventResponse>
    CallReportEvent(const proto::meta::ReportEventRequest &request, const std::string &trace_id) {
        RequestContext context(trace_id);
        proto::meta::ReportEventResponse response;
        const ErrorCode ec = cache_manager_->ReportEvent(&context, &request, &response);
        return {ec, std::move(response)};
    }

    std::vector<std::string> QueryEventReportUris(const std::vector<int64_t> &keys) {
        RequestContext context("query_report_event_ordering");
        auto [ec, locations] = cache_manager_->GetCacheLocation(
            &context, "test_instance", CacheManager::QueryType::QT_BATCH_GET, keys, {}, BlockMask{}, 0, {});
        EXPECT_EQ(EC_OK, ec);
        std::vector<std::string> uris;
        for (const auto &location : locations.cache_locations_view()) {
            for (const auto &spec : location.location_specs()) {
                if (spec.uri().rfind("event_report://", 0) == 0) {
                    uris.push_back(spec.uri());
                }
            }
        }
        return uris;
    }

    std::vector<std::string> QueryRawEventReportUris(int64_t key) {
        MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
        if (!meta_searcher) {
            return {};
        }
        RequestContext context("query_raw_report_event");
        std::vector<CacheLocationMap> location_maps;
        BlockMask mask;
        const ErrorCode ec = meta_searcher->BatchGetLocation(&context, {key}, mask, location_maps);
        if ((ec != EC_OK && ec != EC_PARTIAL_OK) || location_maps.empty()) {
            return {};
        }
        std::vector<std::string> uris;
        for (const auto &[location_id, location] : location_maps.front()) {
            (void)location_id;
            if (!location || location->type() != DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2) {
                continue;
            }
            for (const auto &spec : location->location_specs()) {
                uris.push_back(spec.uri());
            }
        }
        return uris;
    }

    std::unique_ptr<CacheManager> cache_manager_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<RequestContext> request_context_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

TEST_F(CacheManagerTest, TestInitRejectsInvalidMigrationWorkerBudget) {
    const std::vector<std::pair<int32_t, uint32_t>> invalid_configs{
        {1, 1}, // at least one worker must remain available outside migration
        {2, 0},
        {2, 2},
        {2, 3},
    };
    for (const auto &[worker_count, migration_budget] : invalid_configs) {
        auto manager = std::make_unique<CacheManager>(metrics_registry_, registry_manager_);
        EXPECT_FALSE(manager->Init(worker_count,
                                   /*cache_reclaimer_key_sampling_size_total*/ 1000,
                                   /*cache_reclaimer_key_sampling_size_per_task*/ 100,
                                   /*cache_reclaimer_del_batch_size*/ 100,
                                   /*cache_reclaimer_idle_interval_ms*/ 100,
                                   /*cache_reclaimer_worker_size*/ 16,
                                   CacheReclaimerAsyncDeleteConfig{},
                                   migration_budget))
            << "worker_count=" << worker_count << " migration_budget=" << migration_budget;
    }
}

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

TEST_F(CacheManagerTest, TestRegisterInstanceRejectsDifferentInstanceGroup) {
    auto [ec, storage_configs] = cache_manager_->RegisterInstance(request_context_.get(),
                                                                  "different_group",
                                                                  "test_instance",
                                                                  64,
                                                                  createLocationSpecInfos(),
                                                                  createModelDeployment(),
                                                                  {});
    EXPECT_EQ(EC_DUPLICATE_ENTITY, ec);
    EXPECT_TRUE(storage_configs.empty());
    const auto existing = registry_manager_->GetInstanceInfo(request_context_.get(), "test_instance");
    ASSERT_NE(nullptr, existing);
    EXPECT_EQ("default", existing->instance_group_name());
    EXPECT_NE(std::string::npos, request_context_->error_tracer()->ToJsonString().find("instance_group_name"));
}

TEST_F(CacheManagerTest, TestRegisterInstanceReturnsTieredMigrationStorageConfigs) {
    const std::string migration_source = "nfs_migration_source";
    const std::string migration_target = "nfs_migration_target";
    ASSERT_TRUE(RegisterNfsStorage(migration_source));
    ASSERT_TRUE(RegisterNfsStorage(migration_target));

    auto [group_ec, instance_group] = registry_manager_->GetInstanceGroup(request_context_.get(), "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_NE(nullptr, instance_group);
    const auto original_storage_candidates = instance_group->storage_candidates();
    ASSERT_EQ(std::vector<std::string>({"nfs_01"}), original_storage_candidates);

    EnableTieredMigrationStrategy("default", migration_source, migration_target);
    auto [register_ec, storage_configs] = cache_manager_->RegisterInstance(request_context_.get(),
                                                                           "default",
                                                                           "tiered_sdk_config_instance",
                                                                           64,
                                                                           createLocationSpecInfos(),
                                                                           createModelDeployment(),
                                                                           std::vector<LocationSpecGroup>());
    ASSERT_EQ(EC_OK, register_ec);

    std::vector<std::shared_ptr<StorageConfig>> returned_configs;
    ASSERT_TRUE(Jsonizable::FromJsonString(storage_configs, returned_configs));
    std::set<std::string> returned_storage_names;
    for (const auto &config : returned_configs) {
        ASSERT_NE(nullptr, config);
        returned_storage_names.insert(config->global_unique_name());
    }
    EXPECT_EQ((std::set<std::string>{"nfs_01", migration_source, migration_target}), returned_storage_names);
    EXPECT_EQ(original_storage_candidates, instance_group->storage_candidates());
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
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
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

// RemoveInstance 的 per-instance draining 不得读写全局 Reclaimer pause 状态。
// Registry stub 在 drain 与删除的边界观察中间状态：false 场景验证删除 A 不会暂停
// 其他 instance；true 场景验证错误返回不会 Resume 掉 Server 生命周期的既有暂停。
TEST_F(CacheManagerTest, TestRemoveInstanceDoesNotChangeGlobalReclaimerPauseState) {
    Stub stub;
    stub.set(ADDR(RegistryManager, RemoveInstance), remove_instance_reclaimer_state_stub::RemoveInstance_stub);

    auto verify_pause_state = [&](bool initially_paused) {
        if (initially_paused) {
            cache_manager_->PauseReclaimer();
        } else {
            cache_manager_->ResumeReclaimer();
        }
        ASSERT_EQ(initially_paused, cache_manager_->cache_reclaimer()->IsPaused());

        remove_instance_reclaimer_state_stub::Reset(cache_manager_->cache_reclaimer().get());
        RequestContext ctx(initially_paused ? "remove_instance_paused" : "remove_instance_running");
        EXPECT_EQ(EC_ERROR, cache_manager_->RemoveInstance(&ctx, "default", "test_instance"));
        EXPECT_TRUE(remove_instance_reclaimer_state_stub::called);
        EXPECT_EQ(initially_paused, remove_instance_reclaimer_state_stub::observed_paused);
        EXPECT_EQ(initially_paused, cache_manager_->cache_reclaimer()->IsPaused());
    };

    verify_pause_state(false);
    verify_pause_state(true);
    cache_manager_->ResumeReclaimer();
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

TEST_F(CacheManagerTest, TestStartWriteCacheRollsBackPartialBatchAdd) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    auto meta_indexer = cache_manager_->meta_indexer_manager_->GetMetaIndexer("test_instance");
    ASSERT_TRUE(meta_indexer);
    meta_indexer->batch_key_size_ = 1;
    meta_indexer->max_key_count_ = 1;

    auto data_storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(data_storage_manager);
    auto original_backend = data_storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original_backend);
    auto recording_backend = std::make_shared<DeleteRecordingBackend>(original_backend);
    {
        std::unique_lock<std::shared_mutex> lock(data_storage_manager->rw_lock_);
        data_storage_manager->storage_map_["nfs_01"] = recording_backend;
    }

    std::vector<int64_t> keys{1001, 1002};
    while (GetShardIndex(keys[0], meta_indexer->mutex_shard_mask_) ==
           GetShardIndex(keys[1], meta_indexer->mutex_shard_mask_)) {
        ++keys[1];
    }
    auto [ec, start_write_cache_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 1000);
    EXPECT_EQ(EC_PARTIAL_OK, ec);
    EXPECT_TRUE(start_write_cache_info.locations().cache_locations_view().empty());

    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_TRUE(meta_searcher);
    std::vector<CacheLocationMap> location_maps;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    bool rollback_completed = false;
    do {
        location_maps.clear();
        BlockMask empty_mask;
        const ErrorCode get_ec =
            meta_searcher->BatchGetLocation(request_context_.get(), keys, empty_mask, location_maps);
        const bool metadata_removed = get_ec == EC_OK && location_maps.size() == keys.size() &&
                                      std::all_of(location_maps.begin(),
                                                  location_maps.end(),
                                                  [](const auto &locations) { return locations.empty(); });
        rollback_completed = metadata_removed && meta_indexer->GetStorageUsage() == 0 &&
                             recording_backend->DeletedUriCount() >= keys.size() * createLocationSpecInfos().size();
        if (!rollback_completed) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    } while (!rollback_completed && std::chrono::steady_clock::now() < deadline);

    EXPECT_TRUE(rollback_completed);
    EXPECT_EQ(keys.size() * createLocationSpecInfos().size(), recording_backend->DeletedUriCount());
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

// StartWriteCache with min_replica_count > 1 requires n_total >= min_replica_count to skip,
// whereas min_replica_count=1 (default) skips with any 1 replica.
// Also tests spec-group-aware filtering with location_spec_group_names.
TEST_F(CacheManagerTest, TestStartWriteCacheWithMinReplica) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    std::vector<int64_t> keys{1};

    // Scenario A: 0 replicas -> evict writes a new replica.
    std::string evict_session_a;
    {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_instance", keys, {}, {}, 100000000, /*min_replica_count=*/2);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(0u, std::get<BlockMaskOffset>(info.block_mask()));
        ASSERT_EQ(1u, info.locations().cache_locations_view().size());
        evict_session_a = info.write_session_id();
    }
    {
        BlockMask bm = static_cast<std::size_t>(1);
        ASSERT_EQ(EC_OK,
                  cache_manager_->FinishWriteCache(request_context_.get(), "test_instance", evict_session_a, bm));
    }
    // Scenario B: 1 replica -> StartWriteCache(min=1) skips, StartWriteCache(min=2) still writes.
    {
        auto [ec, info] =
            cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(1u, std::get<BlockMaskOffset>(info.block_mask()));
        ASSERT_EQ(0u, info.locations().cache_locations_view().size());
    }
    std::string evict_session_b;
    {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_instance", keys, {}, {}, 100000000, /*min_replica_count=*/2);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(0u, std::get<BlockMaskOffset>(info.block_mask()));
        ASSERT_EQ(1u, info.locations().cache_locations_view().size());
        evict_session_b = info.write_session_id();
    }
    {
        BlockMask bm = static_cast<std::size_t>(1);
        ASSERT_EQ(EC_OK,
                  cache_manager_->FinishWriteCache(request_context_.get(), "test_instance", evict_session_b, bm));
    }

    // Scenario C: 2 replicas -> both skip; min<=0 defaults to 2.
    for (int32_t min_replica_count : {2, 0}) {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_instance", keys, {}, {}, 100000000, min_replica_count);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(1u, std::get<BlockMaskOffset>(info.block_mask()));
        ASSERT_EQ(0u, info.locations().cache_locations_view().size());
    }

    // --- Spec-group-aware filtering ---

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
                                               "test_instance3",
                                               64,
                                               location_spec_infos,
                                               createModelDeployment(),
                                               location_spec_groups));

    // Scenario D: fresh key, evict with group "F0", min=2 -> needs write; returned location has only F0 specs.
    std::vector<int64_t> keys_sg2{100};
    {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_instance3", keys_sg2, {}, {"F0"}, 100000000, 2);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(1u, info.locations().cache_locations_view().size());
        const auto &loc = info.locations().cache_locations_view()[0];
        ASSERT_EQ(2, loc.spec_size());
        BlockMask bm = static_cast<std::size_t>(1);
        ASSERT_EQ(
            EC_OK,
            cache_manager_->FinishWriteCache(request_context_.get(), "test_instance3", info.write_session_id(), bm));
    }

    // Scenario E: empty group name falls back to block-level check.
    // key 100 now has 1 replica. evict with empty group, min=2 -> needs 1 more.
    {
        auto [ec, info] =
            cache_manager_->StartWriteCache(request_context_.get(), "test_instance3", keys_sg2, {}, {}, 100000000, 2);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(1u, info.locations().cache_locations_view().size());
    }

    // Scenario F: replicas cover F0 only, query F0L1 -> spec coverage insufficient.
    // Write 2 replicas with group "F0" for key 200 (each covers tp0_F0, tp1_F0 only).
    std::vector<int64_t> keys_sg3{200};
    {
        auto [ec, info] =
            cache_manager_->StartWriteCache(request_context_.get(), "test_instance3", keys_sg3, {}, {"F0"}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(1u, info.locations().cache_locations_view().size());
        ASSERT_EQ(2, info.locations().cache_locations_view()[0].spec_size());
        BlockMask bm = static_cast<std::size_t>(1);
        ASSERT_EQ(
            EC_OK,
            cache_manager_->FinishWriteCache(request_context_.get(), "test_instance3", info.write_session_id(), bm));
    }
    {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_instance3", keys_sg3, {}, {"F0"}, 100000000, 2);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(1u, info.locations().cache_locations_view().size());
        ASSERT_EQ(2, info.locations().cache_locations_view()[0].spec_size());
        BlockMask bm = static_cast<std::size_t>(1);
        ASSERT_EQ(
            EC_OK,
            cache_manager_->FinishWriteCache(request_context_.get(), "test_instance3", info.write_session_id(), bm));
    }
    // Now key 200 has 2 replicas, each covering F0 (tp0_F0, tp1_F0).
    // evict with group "F0", min=2 -> satisfied, no write needed.
    {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_instance3", keys_sg3, {}, {"F0"}, 100000000, 2);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(0u, info.locations().cache_locations_view().size());
    }
    // evict with group "F0L1", min=2 -> NOT satisfied.
    // Both replicas only cover tp0_F0 and tp1_F0, missing tp0_L1 and tp1_L1.
    // Even though total replica count is 2 >= min, spec coverage is insufficient.
    {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_instance3", keys_sg3, {}, {"F0L1"}, 100000000, 2);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(1u, info.locations().cache_locations_view().size());
        const auto &loc = info.locations().cache_locations_view()[0];
        ASSERT_EQ(4, loc.spec_size());
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
    cache_manager_->reclaimer_task_supervisor_->Stop();
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
    ASSERT_EQ(1, cache_manager_->reclaimer_task_supervisor_->cell_queue_.Size());
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

TEST_F(CacheManagerTest, TestGetCacheLocationHitRateCounters) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    // Write blocks {1, 2, 3}
    std::vector<int64_t> write_keys{1, 2, 3};
    auto [ec1, write_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", write_keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    BlockMask finish_mask = static_cast<size_t>(3); // all 3 blocks succeeded (offset semantics)
    ASSERT_EQ(EC_OK,
              cache_manager_->FinishWriteCache(
                  request_context_.get(), "test_instance", write_info.write_session_id(), finish_mask));

    // Create a RequestContext with a ServiceMetricsCollector so counters get incremented
    auto svc_collector = std::make_shared<ServiceMetricsCollector>(metrics_registry_);
    ASSERT_TRUE(svc_collector->Init());
    auto metrics_ctx = std::make_unique<RequestContext>("hit_rate_test", svc_collector);

    // Query 1: PrefixMatch with keys {1, 2, 4} → hits {1, 2}, miss at 4
    {
        std::vector<int64_t> query_keys{1, 2, 4};
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, result] = cache_manager_->GetCacheLocation(metrics_ctx.get(),
                                                             "test_instance",
                                                             CacheManager::QueryType::QT_PREFIX_MATCH,
                                                             query_keys,
                                                             {},
                                                             block_mask,
                                                             0,
                                                             {});
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(2u, result.cache_locations_view().size()); // 2 hits
    }

    // Query 2: PrefixMatch with keys {1, 2, 3} → all 3 hit
    {
        std::vector<int64_t> query_keys{1, 2, 3};
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, result] = cache_manager_->GetCacheLocation(metrics_ctx.get(),
                                                             "test_instance",
                                                             CacheManager::QueryType::QT_PREFIX_MATCH,
                                                             query_keys,
                                                             {},
                                                             block_mask,
                                                             0,
                                                             {});
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(3u, result.cache_locations_view().size()); // 3 hits
    }

    // Verify cumulative counters: query 3+3=6, hit 2+3=5
    Counter query_counter, hit_counter;
    COPY_METRICS_(svc_collector.get(), manager, get_cache_location_query_block_counter, query_counter);
    COPY_METRICS_(svc_collector.get(), manager, get_cache_location_hit_block_counter, hit_counter);
    EXPECT_EQ(6u, query_counter.Get());
    EXPECT_EQ(5u, hit_counter.Get());
}

TEST_F(CacheManagerTest, TestGetCacheLocationHitRateCounters_BatchGet) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    // Write blocks {1, 2, 3}
    std::vector<int64_t> write_keys{1, 2, 3};
    auto [ec1, write_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", write_keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    BlockMask finish_mask = static_cast<size_t>(3);
    ASSERT_EQ(EC_OK,
              cache_manager_->FinishWriteCache(
                  request_context_.get(), "test_instance", write_info.write_session_id(), finish_mask));

    auto svc_collector = std::make_shared<ServiceMetricsCollector>(metrics_registry_);
    ASSERT_TRUE(svc_collector->Init());
    auto metrics_ctx = std::make_unique<RequestContext>("batch_get_hit_rate_test", svc_collector);

    // BatchGet query: keys {1, 2, 99} → 2 hits (1,2), 1 miss (99)
    {
        std::vector<int64_t> query_keys{1, 2, 99};
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, result] = cache_manager_->GetCacheLocation(metrics_ctx.get(),
                                                             "test_instance",
                                                             CacheManager::QueryType::QT_BATCH_GET,
                                                             query_keys,
                                                             {},
                                                             block_mask,
                                                             0,
                                                             {});
        ASSERT_EQ(EC_OK, ec);
        // BatchGet returns all keys; hits have non-empty id, misses have empty id
        ASSERT_EQ(3u, result.cache_locations_view().size());
    }

    Counter query_counter, hit_counter;
    COPY_METRICS_(svc_collector.get(), manager, get_cache_location_query_block_counter, query_counter);
    COPY_METRICS_(svc_collector.get(), manager, get_cache_location_hit_block_counter, hit_counter);
    EXPECT_EQ(3u, query_counter.Get());
    EXPECT_EQ(2u, hit_counter.Get());
}

TEST_F(CacheManagerTest, TestGetCacheLocationHitRateCounters_ErrorPath) {
    auto svc_collector = std::make_shared<ServiceMetricsCollector>(metrics_registry_);
    ASSERT_TRUE(svc_collector->Init());
    auto metrics_ctx = std::make_unique<RequestContext>("error_path_test", svc_collector);

    // Query non-existent instance → should fail and NOT increment counters
    std::vector<int64_t> query_keys{1, 2, 3};
    BlockMask block_mask = static_cast<size_t>(0);
    auto [ec, result] = cache_manager_->GetCacheLocation(metrics_ctx.get(),
                                                         "nonexistent_instance",
                                                         CacheManager::QueryType::QT_PREFIX_MATCH,
                                                         query_keys,
                                                         {},
                                                         block_mask,
                                                         0,
                                                         {});
    EXPECT_NE(EC_OK, ec);

    Counter query_counter, hit_counter;
    COPY_METRICS_(svc_collector.get(), manager, get_cache_location_query_block_counter, query_counter);
    COPY_METRICS_(svc_collector.get(), manager, get_cache_location_hit_block_counter, hit_counter);
    EXPECT_EQ(0u, query_counter.Get());
    EXPECT_EQ(0u, hit_counter.Get());
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
        ASSERT_EQ(keys.size(), cache_locations_view.size());
        ASSERT_EQ(keys.size(), metas.size());
        for (std::size_t i = 0; i < metas.size(); ++i) {
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
        ASSERT_EQ(keys.size(), cache_locations_view.size());
        ASSERT_EQ(keys.size(), metas.size());
        for (std::size_t i = 0; i < metas.size(); ++i) {
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
        ASSERT_EQ(keys.size(), cache_locations_view.size());
        ASSERT_EQ(keys.size(), metas.size());
        std::vector<std::size_t> pos_vec = {0, 1, 2, 4};
        std::map<std::string, std::string> meta;
        ASSERT_TRUE(Jsonizable::FromJsonString(metas[3], meta));
        ASSERT_EQ(CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_NOT_FOUND), meta.at("status"));
        for (std::size_t pos : pos_vec) {
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
        for (int offset = 0; offset < 10; ++offset) {
            const int64_t i = static_cast<int64_t>(start) + offset;
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
        for (int offset = 0; offset < 10; ++offset) {
            const int64_t i = static_cast<int64_t>(start) + offset;
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

TEST_F(CacheManagerTest, TestGetCacheLocationLen) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    std::vector<int64_t> keys{1, 2, 3, 4, 5, 6, 7};
    auto [ec1, start_write_cache_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);

    {
        BlockMask block_mask = static_cast<size_t>(5);
        auto ec = cache_manager_->FinishWriteCache(
            request_context_.get(), "test_instance", start_write_cache_info.write_session_id(), block_mask);
        ASSERT_EQ(EC_OK, ec);
    }

    // Test QT_PREFIX_MATCH
    {
        std::vector<int64_t> keys{1, 2, 8, 4, 5, 6};
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_location_len] = cache_manager_->GetCacheLocationLen(
            request_context_.get(), "test_instance", CacheManager::QueryType::QT_PREFIX_MATCH, keys, {}, 0);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(2, cache_location_len);
    }

    // Test QT_BATCH_GET
    {
        std::vector<int64_t> keys{1, 2, 8, 4, 5, 9, 6};
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_location_len] = cache_manager_->GetCacheLocationLen(
            request_context_.get(), "test_instance", CacheManager::QueryType::QT_BATCH_GET, keys, {}, 0);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(4, cache_location_len);
    }

    // Test QT_REVERSE_ROLL_SW_MATCH
    {
        std::vector<int64_t> keys{1, 2, 3, 8, 5, 9, 6};
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_location_len] = cache_manager_->GetCacheLocationLen(
            request_context_.get(), "test_instance", CacheManager::QueryType::QT_REVERSE_ROLL_SW_MATCH, keys, {}, 2);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(2, cache_location_len);
    }
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_NullRegistryManager) {
    auto saved = cache_manager_->registry_manager_;
    cache_manager_->registry_manager_ = nullptr;

    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_location_specs({LocationSpec("tp0", "file://mock_store/path")});
    ASSERT_EQ(func(loc), true);

    cache_manager_->registry_manager_ = saved;
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_NullDataStorageManager) {
    // when data_storage_manager() is null, the functor should return true
    auto saved = registry_manager_->data_storage_manager_;
    registry_manager_->data_storage_manager_ = nullptr;

    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_location_specs({LocationSpec("tp0", "file://mock_store/path")});
    ASSERT_EQ(func(loc), true);

    registry_manager_->data_storage_manager_ = saved;
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_EmptyLocationSpecs) {
    // no location specs -> no URIs to check -> returns true
    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    ASSERT_EQ(func(loc), true);
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_InvalidUri) {
    // invalid URI string (no protocol) -> DataStorageUri::Valid() is
    // false -> no valid URIs collected -> returns true
    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_location_specs({LocationSpec("tp0", "no_protocol_here")});
    ASSERT_EQ(func(loc), true);
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_AllExist) {
    // inject a mock backend where MightExist returns all true;
    // the functor should return true
    auto metrics_registry = cache_manager_->metrics_registry_;
    auto mock_backend = std::make_shared<MockDataStorageBackend>(metrics_registry);
    EXPECT_CALL(*mock_backend, MightExist(_)).WillOnce([](const std::vector<DataStorageUri> &uris) {
        return std::vector<bool>(uris.size(), true);
    });

    auto dsm = registry_manager_->data_storage_manager_;
    dsm->storage_map_["mock_store"] = mock_backend;

    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_location_specs(
        {LocationSpec("tp0", "file://mock_store/path_a"), LocationSpec("tp1", "file://mock_store/path_b")});
    ASSERT_EQ(func(loc), true);

    dsm->storage_map_.erase("mock_store");
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_NoneExist) {
    // inject a mock backend where MightExist returns all false;
    // the functor should return false
    auto metrics_registry = cache_manager_->metrics_registry_;
    auto mock_backend = std::make_shared<MockDataStorageBackend>(metrics_registry);
    EXPECT_CALL(*mock_backend, MightExist(_)).WillOnce([](const std::vector<DataStorageUri> &uris) {
        return std::vector<bool>(uris.size(), false);
    });

    auto dsm = registry_manager_->data_storage_manager_;
    dsm->storage_map_["mock_store"] = mock_backend;

    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_location_specs(
        {LocationSpec("tp0", "file://mock_store/path_a"), LocationSpec("tp1", "file://mock_store/path_b")});
    ASSERT_EQ(func(loc), false);

    dsm->storage_map_.erase("mock_store");
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_PartialExist) {
    // inject a mock backend where MightExist returns mixed results;
    // std::all_of requires all true, so the functor should return false
    auto metrics_registry = cache_manager_->metrics_registry_;
    auto mock_backend = std::make_shared<MockDataStorageBackend>(metrics_registry);
    EXPECT_CALL(*mock_backend, MightExist(_)).WillOnce([](const std::vector<DataStorageUri> &uris) {
        std::vector<bool> result(uris.size(), true);
        // mark the last URI as non-existent
        result.back() = false;
        return result;
    });

    auto dsm = registry_manager_->data_storage_manager_;
    dsm->storage_map_["mock_store"] = mock_backend;

    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_location_specs(
        {LocationSpec("tp0", "file://mock_store/path_a"), LocationSpec("tp1", "file://mock_store/path_b")});
    ASSERT_EQ(func(loc), false);

    dsm->storage_map_.erase("mock_store");
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_VerifiesUriPassthrough) {
    // verify that the functor passes the correct parsed URIs to
    // MightExist and uses the hostname from the first URI for backend
    // lookup
    auto metrics_registry = cache_manager_->metrics_registry_;
    auto mock_backend = std::make_shared<MockDataStorageBackend>(metrics_registry);
    EXPECT_CALL(*mock_backend, MightExist(_)).WillOnce([](const std::vector<DataStorageUri> &uris) {
        // should receive exactly 2 valid URIs (invalid ones
        // filtered out)
        EXPECT_EQ(2u, uris.size());
        EXPECT_EQ("mock_store", uris[0].GetHostName());
        EXPECT_EQ("mock_store", uris[1].GetHostName());
        return std::vector<bool>{true, true};
    });

    auto dsm = registry_manager_->data_storage_manager_;
    dsm->storage_map_["mock_store"] = mock_backend;

    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    // mix of invalid and valid URIs; only valid ones should reach
    // MightExist
    loc.set_location_specs({LocationSpec("tp0", "no_protocol"),
                            LocationSpec("tp1", "file://mock_store/path_a"),
                            LocationSpec("tp2", "file://mock_store/path_b")});
    ASSERT_EQ(func(loc), true);

    dsm->storage_map_.erase("mock_store");
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_UnregisteredBackend) {
    // A missing backend returns no per-URI result and must fail closed.
    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_location_specs({LocationSpec("tp0", "file://nonexistent_backend/path")});
    EXPECT_FALSE(func(loc));
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_ShortBackendResultFailsClosed) {
    auto mock_backend = std::make_shared<MockDataStorageBackend>(cache_manager_->metrics_registry_);
    EXPECT_CALL(*mock_backend, MightExist(_)).WillOnce([](const std::vector<DataStorageUri> &) {
        return std::vector<bool>{true};
    });
    auto dsm = registry_manager_->data_storage_manager_;
    dsm->storage_map_["short_result_store"] = mock_backend;

    const auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");
    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_location_specs({LocationSpec("tp0", "file://short_result_store/path_a"),
                            LocationSpec("tp1", "file://short_result_store/path_b")});
    EXPECT_FALSE(func(loc));

    dsm->storage_map_.erase("short_result_store");
}

TEST(ReportEventContractTest, SnapshotAndResponseFieldNumbersMatchContract) {
    EXPECT_EQ(2, proto::meta::BlockSnapshotItem::descriptor()->FindFieldByName("medium")->number());
    EXPECT_EQ(3, proto::meta::BlockSnapshotItem::descriptor()->FindFieldByName("specs")->number());
    EXPECT_EQ(1, proto::meta::BlockSnapshotEventParams::descriptor()->FindFieldByName("medium")->number());
    EXPECT_EQ(2, proto::meta::BlockSnapshotEventParams::descriptor()->FindFieldByName("blocks")->number());
    EXPECT_EQ(3,
              proto::meta::ReportEventResponse::descriptor()->FindFieldByName("committed_snapshot_version")->number());
    EXPECT_EQ(4, proto::meta::ReportEventResponse::descriptor()->FindFieldByName("retry_after_ms")->number());
    EXPECT_EQ(5, proto::meta::ReportEventResponse::descriptor()->FindFieldByName("snapshot_required")->number());
    EXPECT_EQ(6, proto::meta::ReportEventResponse::descriptor()->FindFieldByName("extra_info")->number());
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_MissingEventReportBackendFailsClosed) {
    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_id("kvs#event_report_l1p5#mem#192.168.1.100:8080");
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5);
    loc.set_location_specs(
        {LocationSpec("tp0", "event_report://cache-node/mem?s_version=0123456789abcdef0123456789abcdef")});

    EXPECT_FALSE(func(loc));
}

TEST_F(CacheManagerTest, TestReportEventDeltaAlreadyAdmittedThenSnapshotWins) {
    const std::string host = "192.168.10.1:8080";
    const int64_t key = 9400;
    auto event_backend = InstallEventReportBackend();
    auto *meta_backend = InstallControllableMetaBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_NE(nullptr, meta_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    auto [baseline_ec, baseline_response] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "baseline"}}), "ordering_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    const std::string baseline_token = baseline_response.committed_snapshot_version();
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(baseline_token));

    meta_backend->BlockNextUpsert();
    auto delta_future = std::async(std::launch::async, [this, host, key] {
        return CallReportEvent(MakeAddRequest(host, key, "delta_before_snapshot"), "ordering_delta_first");
    });
    const bool delta_entered = meta_backend->WaitUntilUpsertEntered(std::chrono::seconds(1));
    if (!delta_entered) {
        meta_backend->ReleaseUpsert();
    }
    ASSERT_TRUE(delta_entered);

    auto snapshot_future = std::async(std::launch::async, [this, host, key] {
        return CallReportEvent(MakeSnapshotRequest(host, {{key, "snapshot_after_delta"}}), "ordering_snapshot_second");
    });
    EXPECT_EQ(std::future_status::timeout, snapshot_future.wait_for(std::chrono::milliseconds(30)));
    EXPECT_EQ(baseline_token, event_backend->GetSnapshotVersion({"test_instance", host}));

    meta_backend->ReleaseUpsert();
    ASSERT_EQ(std::future_status::ready, delta_future.wait_for(std::chrono::seconds(2)));
    const auto [delta_ec, delta_response] = delta_future.get();
    ASSERT_EQ(EC_OK, delta_ec);
    EXPECT_EQ(baseline_token, delta_response.committed_snapshot_version());

    ASSERT_EQ(std::future_status::ready, snapshot_future.wait_for(std::chrono::seconds(2)));
    const auto [snapshot_ec, snapshot_response] = snapshot_future.get();
    ASSERT_EQ(EC_OK, snapshot_ec);
    const std::string snapshot_token = snapshot_response.committed_snapshot_version();
    EXPECT_NE(baseline_token, snapshot_token);

    const auto uris = QueryEventReportUris({key});
    ASSERT_EQ(1u, uris.size());
    EXPECT_NE(std::string::npos, uris[0].find("source=snapshot_after_delta"));
    EXPECT_NE(std::string::npos, uris[0].find("s_version=" + snapshot_token));
}

TEST_F(CacheManagerTest, TestReportEventSnapshotGateThenDeltaInheritsNewTokenAndWins) {
    const std::string host = "192.168.10.2:8080";
    const int64_t key = 9410;
    auto event_backend = InstallEventReportBackend();
    auto *meta_backend = InstallControllableMetaBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_NE(nullptr, meta_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    auto [baseline_ec, baseline_response] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "baseline"}}), "gate_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    const std::string baseline_token = baseline_response.committed_snapshot_version();

    meta_backend->BlockNextUpsert();
    auto snapshot_future = std::async(std::launch::async, [this, host, key] {
        return CallReportEvent(MakeSnapshotRequest(host, {{key, "snapshot_in_flight"}}), "gate_snapshot_first");
    });
    const bool snapshot_entered = meta_backend->WaitUntilUpsertEntered(std::chrono::seconds(1));
    if (!snapshot_entered) {
        meta_backend->ReleaseUpsert();
    }
    ASSERT_TRUE(snapshot_entered);
    EXPECT_EQ(baseline_token, event_backend->GetSnapshotVersion({"test_instance", host}));

    auto delta_future = std::async(std::launch::async, [this, host, key] {
        return CallReportEvent(MakeAddRequest(host, key, "delta_after_snapshot_gate"), "gate_delta_second");
    });
    EXPECT_EQ(std::future_status::timeout, delta_future.wait_for(std::chrono::milliseconds(30)));

    meta_backend->ReleaseUpsert();
    ASSERT_EQ(std::future_status::ready, snapshot_future.wait_for(std::chrono::seconds(2)));
    const auto [snapshot_ec, snapshot_response] = snapshot_future.get();
    ASSERT_EQ(EC_OK, snapshot_ec);
    const std::string new_token = snapshot_response.committed_snapshot_version();
    EXPECT_NE(baseline_token, new_token);

    ASSERT_EQ(std::future_status::ready, delta_future.wait_for(std::chrono::seconds(2)));
    const auto [delta_ec, delta_response] = delta_future.get();
    ASSERT_EQ(EC_OK, delta_ec);
    EXPECT_EQ(new_token, delta_response.committed_snapshot_version());

    const auto uris = QueryEventReportUris({key});
    ASSERT_EQ(1u, uris.size());
    EXPECT_NE(std::string::npos, uris[0].find("source=delta_after_snapshot_gate"));
    EXPECT_NE(std::string::npos, uris[0].find("s_version=" + new_token));
}

TEST_F(CacheManagerTest, TestReportEventDeltaGateTimeoutReturnsSnapshotInProgressAndCanRetry) {
    const std::string host = "192.168.10.48:8080";
    const int64_t snapshot_key = 94'480;
    const int64_t delta_key = 94'481;
    auto event_backend = InstallEventReportBackend();
    auto *meta_backend = InstallControllableMetaBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_NE(nullptr, meta_backend);
    event_backend->SetSnapshotDeltaDrainTimeoutMsForTest(20);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    std::string baseline_token;
    ASSERT_EQ(EC_OK, event_backend->BeginDeltaMutation({"test_instance", host}, baseline_token));
    event_backend->EndDeltaMutation({"test_instance", host});
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(baseline_token));

    meta_backend->BlockNextUpsert();
    auto snapshot_future = std::async(std::launch::async, [this, host, snapshot_key] {
        return CallReportEvent(MakeSnapshotRequest(host, {{snapshot_key, "snapshot_in_flight"}}),
                               "delta_timeout_snapshot");
    });
    const bool snapshot_entered = meta_backend->WaitUntilUpsertEntered(std::chrono::seconds(1));
    if (!snapshot_entered) {
        meta_backend->ReleaseUpsert();
    }
    ASSERT_TRUE(snapshot_entered);

    const auto [delta_ec, timed_out_delta] =
        CallReportEvent(MakeAddRequest(host, delta_key, "timed_out_delta"), "delta_timeout_request");
    EXPECT_EQ(EC_PARTIAL_OK, delta_ec);
    EXPECT_EQ(proto::meta::SNAPSHOT_IN_PROGRESS, timed_out_delta.header().status().code());
    ASSERT_EQ(1, timed_out_delta.item_results_size());
    EXPECT_EQ(proto::meta::SNAPSHOT_IN_PROGRESS, timed_out_delta.item_results(0));
    EXPECT_EQ(0u, timed_out_delta.retry_after_ms());
    EXPECT_EQ(baseline_token, timed_out_delta.committed_snapshot_version());

    meta_backend->ReleaseUpsert();
    ASSERT_EQ(std::future_status::ready, snapshot_future.wait_for(std::chrono::seconds(2)));
    const auto [snapshot_ec, snapshot_response] = snapshot_future.get();
    ASSERT_EQ(EC_OK, snapshot_ec);
    const std::string snapshot_token = snapshot_response.committed_snapshot_version();
    EXPECT_NE(baseline_token, snapshot_token);
    EXPECT_TRUE(QueryEventReportUris({delta_key}).empty());

    const auto [retry_ec, retried_delta] =
        CallReportEvent(MakeAddRequest(host, delta_key, "retried_delta"), "delta_timeout_retry");
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ(snapshot_token, retried_delta.committed_snapshot_version());
    const auto uris = QueryEventReportUris({delta_key});
    ASSERT_EQ(1u, uris.size());
    EXPECT_NE(std::string::npos, uris.front().find("source=retried_delta"));
    EXPECT_NE(std::string::npos, uris.front().find("s_version=" + snapshot_token));
}

TEST_F(CacheManagerTest, TestHostDownCancelsSnapshotAlreadyWritingMetadata) {
    const std::string host = "192.168.10.43:8080";
    const int64_t key = 94'420;
    auto event_backend = InstallEventReportBackend();
    auto *meta_backend = InstallControllableMetaBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_NE(nullptr, meta_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));
    const auto [baseline_ec, baseline] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "baseline"}}), "host_down_snapshot_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    const std::string baseline_token = baseline.committed_snapshot_version();
    ASSERT_EQ(1u, QueryEventReportUris({key}).size());

    meta_backend->BlockNextLocationRead();
    auto snapshot_future = std::async(std::launch::async, [this, host, key] {
        return CallReportEvent(MakeSnapshotRequest(host, {{key, "must_not_commit"}}), "host_down_snapshot_in_flight");
    });
    const bool snapshot_entered = meta_backend->WaitUntilLocationReadEntered(std::chrono::seconds(1));
    if (!snapshot_entered) {
        meta_backend->ReleaseLocationRead();
    }
    ASSERT_TRUE(snapshot_entered);

    proto::meta::ReportEventRequest host_down;
    host_down.set_instance_id("test_instance");
    host_down.set_host_ip_port(host);
    host_down.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
    auto *host_down_event = host_down.add_events();
    host_down_event->set_event_type(proto::meta::EVENT_HOST_DOWN);
    host_down_event->mutable_host_down();
    const auto [host_down_ec, host_down_response] = CallReportEvent(host_down, "host_down_during_snapshot");
    ASSERT_EQ(EC_OK, host_down_ec);
    EXPECT_EQ(proto::meta::OK, host_down_response.header().status().code());
    EXPECT_FALSE(event_backend->IsNodeRegistered("test_instance", host));
    EXPECT_TRUE(event_backend->GetSnapshotVersion({"test_instance", host}).empty());

    meta_backend->ReleaseLocationRead();
    ASSERT_EQ(std::future_status::ready, snapshot_future.wait_for(std::chrono::seconds(2)));
    const auto [snapshot_ec, snapshot_response] = snapshot_future.get();
    EXPECT_EQ(EC_PARTIAL_OK, snapshot_ec);
    EXPECT_EQ(proto::meta::NODE_NOT_REGISTERED, snapshot_response.header().status().code());
    EXPECT_TRUE(snapshot_response.committed_snapshot_version().empty());
    EXPECT_TRUE(snapshot_response.snapshot_required());
    EXPECT_TRUE(QueryEventReportUris({key}).empty());

    std::string baseline_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://physical-cache:9600/mem", baseline_token, baseline_uri));
    EXPECT_EQ((std::vector<bool>{false}), event_backend->MightExist({DataStorageUri(baseline_uri)}));
}

TEST_F(CacheManagerTest, TestHostDownMakesAlreadyAdmittedDeltaInvisibleWithoutDeadlock) {
    const std::string host = "192.168.10.44:8080";
    const int64_t key = 94'421;
    auto event_backend = InstallEventReportBackend();
    auto *meta_backend = InstallControllableMetaBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_NE(nullptr, meta_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));
    const auto [baseline_ec, baseline] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "baseline"}}), "host_down_delta_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(baseline.committed_snapshot_version()));

    meta_backend->BlockNextLocationRead();
    auto delta_future = std::async(std::launch::async, [this, host, key] {
        return CallReportEvent(MakeAddRequest(host, key, "admitted_before_host_down"), "host_down_delta_in_flight");
    });
    const bool delta_entered = meta_backend->WaitUntilLocationReadEntered(std::chrono::seconds(1));
    if (!delta_entered) {
        meta_backend->ReleaseLocationRead();
    }
    ASSERT_TRUE(delta_entered);

    proto::meta::ReportEventRequest host_down;
    host_down.set_instance_id("test_instance");
    host_down.set_host_ip_port(host);
    host_down.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
    auto *host_down_event = host_down.add_events();
    host_down_event->set_event_type(proto::meta::EVENT_HOST_DOWN);
    host_down_event->mutable_host_down();
    const auto [host_down_ec, host_down_response] = CallReportEvent(host_down, "host_down_during_delta");
    ASSERT_EQ(EC_OK, host_down_ec);
    EXPECT_EQ(proto::meta::OK, host_down_response.header().status().code());

    meta_backend->ReleaseLocationRead();
    ASSERT_EQ(std::future_status::ready, delta_future.wait_for(std::chrono::seconds(2)));
    const auto [delta_ec, delta_response] = delta_future.get();
    EXPECT_EQ(EC_PARTIAL_OK, delta_ec);
    EXPECT_EQ(proto::meta::NODE_NOT_REGISTERED, delta_response.header().status().code());
    EXPECT_TRUE(delta_response.committed_snapshot_version().empty());
    EXPECT_TRUE(delta_response.snapshot_required());
    EXPECT_FALSE(event_backend->IsNodeRegistered("test_instance", host));
    EXPECT_TRUE(QueryEventReportUris({key}).empty());
}

TEST_F(CacheManagerTest, TestEventCleanupTasksDrainAndRemainFencedAcrossReactivation) {
    const std::string host = "192.168.10.78:8080";
    const int64_t key = 94'478;
    auto event_backend = InstallEventReportBackend();
    auto *meta_backend = InstallControllableMetaBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_NE(nullptr, meta_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));
    ASSERT_EQ(EC_OK, CallReportEvent(MakeSnapshotRequest(host, {{key, "baseline"}}), "cleanup_gate_baseline").first);
    ASSERT_EQ(1u, QueryRawEventReportUris(key).size());
    ASSERT_TRUE(event_backend->IsCleanupCallbackSet());

    // Stop executor workers so the test can deterministically take ownership
    // of the cleanup closure admitted by the backend callback.
    cache_manager_->reclaimer_task_supervisor_->Stop();
    auto &executor = cache_manager_->schedule_plan_executor_;
    executor->stop_.store(true);
    executor->condition_.notify_all();
    for (auto &worker : executor->workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    executor->workers_.clear();
    executor->stop_.store(false);
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        for (auto &queue : executor->task_queues_) {
            queue.clear();
        }
    }

    EventReportBackend::CleanupCallback callback;
    {
        std::lock_guard<std::mutex> lock(event_backend->cleanup_cb_mutex_);
        callback = event_backend->cleanup_callback_;
    }
    ASSERT_TRUE(callback);

    auto take_only_queued_task = [&]() {
        std::function<void()> task;
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        EXPECT_EQ(1u, executor->WaitingTaskCountLocked());
        for (auto &queue : executor->task_queues_) {
            if (!queue.empty()) {
                task = queue.begin()->task;
                queue.clear();
            }
        }
        return task;
    };

    uint64_t cleanup_generation = 0;
    ASSERT_EQ(EC_OK, event_backend->UnregisterNodeForHostDown("test_instance", host, cleanup_generation));
    callback("test_instance", host, cleanup_generation);
    auto running_cleanup = take_only_queued_task();
    ASSERT_TRUE(running_cleanup);

    // A cleanup already executing must hold the lifetime lease until its
    // metadata access completes. Deactivation therefore waits rather than
    // racing manager cleanup against the raw-this task.
    meta_backend->BlockNextLocationRead();
    auto cleanup_future = std::async(std::launch::async, std::move(running_cleanup));
    ASSERT_TRUE(meta_backend->WaitUntilLocationReadEntered(std::chrono::seconds(1)));
    auto deactivate_future =
        std::async(std::launch::async, [this] { cache_manager_->DeactivateEventCleanupCallbacks(); });
    EXPECT_EQ(std::future_status::timeout, deactivate_future.wait_for(std::chrono::milliseconds(50)));
    meta_backend->ReleaseLocationRead();
    ASSERT_EQ(std::future_status::ready, cleanup_future.wait_for(std::chrono::seconds(2)));
    cleanup_future.get();
    ASSERT_EQ(std::future_status::ready, deactivate_future.wait_for(std::chrono::seconds(2)));
    deactivate_future.get();
    EXPECT_TRUE(QueryRawEventReportUris(key).empty());

    cache_manager_->ActivateEventCleanupCallbacks();
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));
    ASSERT_EQ(EC_OK, CallReportEvent(MakeAddRequest(host, key, "after_reactivate"), "cleanup_gate_delta").first);
    ASSERT_EQ(1u, QueryRawEventReportUris(key).size());

    // A task queued in the old epoch must remain inert after reactivation; a
    // boolean-only gate would incorrectly make it live again here.
    uint64_t stale_cleanup_generation = 0;
    ASSERT_EQ(EC_OK, event_backend->UnregisterNodeForHostDown("test_instance", host, stale_cleanup_generation));
    callback("test_instance", host, stale_cleanup_generation);
    auto stale_queued_cleanup = take_only_queued_task();
    ASSERT_TRUE(stale_queued_cleanup);
    cache_manager_->DeactivateEventCleanupCallbacks();
    cache_manager_->ActivateEventCleanupCallbacks();
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));
    stale_queued_cleanup();
    EXPECT_EQ(1u, QueryRawEventReportUris(key).size());
}

TEST_F(CacheManagerTest, TestOldDeltaCannotCrossReporterLifecycleAfterReregisterAndSnapshot) {
    const std::string host = "192.168.10.45:8080";
    const int64_t key = 94'422;
    const int64_t new_key = 94'423;
    auto event_backend = InstallEventReportBackend();
    auto *meta_backend = InstallControllableMetaBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_NE(nullptr, meta_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));
    ASSERT_EQ(EC_OK, CallReportEvent(MakeSnapshotRequest(host, {{key, "baseline"}}), "lifecycle_baseline").first);

    // Pause the old request after it entered metadata read I/O but before its
    // modifier can acquire the generation-pinned write lease.
    meta_backend->BlockNextLocationRead();
    auto old_delta = std::async(std::launch::async, [this, host, key] {
        return CallReportEvent(MakeAddRequest(host, key, "stale_old_lifecycle"), "old_lifecycle_delta");
    });
    ASSERT_TRUE(meta_backend->WaitUntilLocationReadEntered(std::chrono::seconds(1)));

    proto::meta::ReportEventRequest host_down;
    host_down.set_instance_id("test_instance");
    host_down.set_host_ip_port(host);
    host_down.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
    auto *host_down_event = host_down.add_events();
    host_down_event->set_event_type(proto::meta::EVENT_HOST_DOWN);
    host_down_event->mutable_host_down();
    ASSERT_EQ(EC_OK, CallReportEvent(host_down, "old_lifecycle_host_down").first);

    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));
    const auto [snapshot_ec, snapshot_response] =
        CallReportEvent(MakeSnapshotRequest(host, {{new_key, "new_lifecycle"}}), "new_lifecycle_snapshot");
    ASSERT_EQ(EC_OK, snapshot_ec);
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(snapshot_response.committed_snapshot_version()));

    meta_backend->ReleaseLocationRead();
    ASSERT_EQ(std::future_status::ready, old_delta.wait_for(std::chrono::seconds(2)));
    const auto [old_ec, old_response] = old_delta.get();
    EXPECT_EQ(EC_PARTIAL_OK, old_ec);
    EXPECT_EQ(proto::meta::NODE_NOT_REGISTERED, old_response.header().status().code());

    for (const auto &uri : QueryRawEventReportUris(key)) {
        EXPECT_EQ(std::string::npos, uri.find("source=stale_old_lifecycle"));
    }
    const auto new_uris = QueryRawEventReportUris(new_key);
    ASSERT_EQ(1u, new_uris.size());
    EXPECT_NE(std::string::npos, new_uris.front().find("source=new_lifecycle"));
}

TEST_F(CacheManagerTest, TestReportEventPartialSnapshotFailureKeepsCacheReadableAndRetryConverges) {
    const std::string host = "192.168.10.3:8080";
    const int64_t key_a = 9420;
    const int64_t key_b = 9421;
    auto event_backend = InstallEventReportBackend();
    auto *meta_backend = InstallControllableMetaBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_NE(nullptr, meta_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    auto [baseline_ec, baseline_response] =
        CallReportEvent(MakeSnapshotRequest(host, {{key_a, "baseline_a"}, {key_b, "baseline_b"}}), "partial_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    const std::string baseline_token = baseline_response.committed_snapshot_version();

    meta_backend->FailKeyOnNextUpsert(key_b);
    const auto [failed_ec, failed_response] = CallReportEvent(
        MakeSnapshotRequest(host, {{key_a, "partial_a"}, {key_b, "partial_b"}}), "partial_injected_failure");
    EXPECT_NE(EC_OK, failed_ec);
    EXPECT_EQ(baseline_token, failed_response.committed_snapshot_version());
    EXPECT_EQ(baseline_token, event_backend->GetSnapshotVersion({"test_instance", host}));

    const auto [delta_ec, delta_response] =
        CallReportEvent(MakeAddRequest(host, key_b, "delta_after_failed_snapshot"), "partial_delta_after_abort");
    ASSERT_EQ(EC_OK, delta_ec);
    EXPECT_EQ(baseline_token, delta_response.committed_snapshot_version());

    // Event-report metadata is a soft cache index. Both the partially written
    // snapshot data and the following delta remain useful candidates.
    const auto visible_after_failure = QueryEventReportUris({key_a, key_b});
    ASSERT_EQ(2u, visible_after_failure.size());
    EXPECT_TRUE(std::any_of(visible_after_failure.begin(), visible_after_failure.end(), [](const std::string &uri) {
        return uri.find("source=partial_a") != std::string::npos;
    }));
    EXPECT_TRUE(std::any_of(visible_after_failure.begin(), visible_after_failure.end(), [](const std::string &uri) {
        return uri.find("source=delta_after_failed_snapshot") != std::string::npos;
    }));

    const auto [retry_ec, retry_response] =
        CallReportEvent(MakeSnapshotRequest(host, {{key_a, "retry_a"}, {key_b, "retry_b"}}), "partial_full_retry");
    ASSERT_EQ(EC_OK, retry_ec);
    const std::string retry_token = retry_response.committed_snapshot_version();
    EXPECT_NE(baseline_token, retry_token);

    const auto visible_after_retry = QueryEventReportUris({key_a, key_b});
    ASSERT_EQ(2u, visible_after_retry.size());
    std::set<std::string> retry_sources;
    for (const auto &uri : visible_after_retry) {
        EXPECT_NE(std::string::npos, uri.find("s_version=" + retry_token));
        if (uri.find("source=retry_a") != std::string::npos) {
            retry_sources.insert("retry_a");
        }
        if (uri.find("source=retry_b") != std::string::npos) {
            retry_sources.insert("retry_b");
        }
    }
    EXPECT_EQ((std::set<std::string>{"retry_a", "retry_b"}), retry_sources);
}

TEST_F(CacheManagerTest, TestReportEventSnapshotCommitsWithoutWaitingForPersistentSync) {
    const std::string host = "192.168.10.30:8080";
    const int64_t key = 9425;
    auto event_backend = InstallEventReportBackend();
    auto *meta_backend = InstallControllableMetaBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_NE(nullptr, meta_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    ASSERT_EQ(0u, meta_backend->GetSyncCallCount());
    const auto [snapshot_ec, snapshot_response] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "async_snapshot"}}), "async_snapshot_no_sync");
    ASSERT_EQ(EC_OK, snapshot_ec);
    EXPECT_EQ(proto::meta::OK, snapshot_response.header().status().code());
    EXPECT_EQ(0, snapshot_response.item_results_size());
    const std::string committed = snapshot_response.committed_snapshot_version();
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(committed));
    EXPECT_FALSE(snapshot_response.snapshot_required());
    EXPECT_EQ(committed, event_backend->GetSnapshotVersion({"test_instance", host}));
    EXPECT_EQ(0u, meta_backend->GetSyncCallCount());

    const auto visible = QueryEventReportUris({key});
    ASSERT_EQ(1u, visible.size());
    EXPECT_NE(std::string::npos, visible.front().find("source=async_snapshot"));
    EXPECT_NE(std::string::npos, visible.front().find("s_version=" + committed));
}

TEST_F(CacheManagerTest, TestReportEventSameRequestDeltaOrderUsesLastOperationPerSpec) {
    const std::string host = "192.168.10.31:8080";
    const int64_t key = 9426;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    const auto [baseline_ec, baseline] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "baseline"}}), "delta_order_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    const std::string token = baseline.committed_snapshot_version();

    auto make_delete = [&] {
        auto request = MakeAddRequest(host, key, "placeholder");
        request.clear_events();
        auto *event = request.add_events();
        event->set_event_type(proto::meta::EVENT_BLOCK_DELETE);
        event->mutable_block_delete()->set_block_key(std::to_string(key));
        event->mutable_block_delete()->set_medium("mem");
        event->mutable_block_delete()->add_spec_names("tp0");
        return request;
    };

    auto delete_then_add = make_delete();
    auto add_event = MakeAddRequest(host, key, "last_add").events(0);
    *delete_then_add.add_events() = add_event;
    const auto [delete_add_ec, delete_add_response] = CallReportEvent(delete_then_add, "delta_order_delete_then_add");
    ASSERT_EQ(EC_OK, delete_add_ec);
    EXPECT_EQ(token, delete_add_response.committed_snapshot_version());
    auto visible = QueryEventReportUris({key});
    ASSERT_EQ(1u, visible.size());
    EXPECT_NE(std::string::npos, visible.front().find("source=last_add"));

    auto add_then_delete = MakeAddRequest(host, key, "add_before_delete");
    *add_then_delete.add_events() = make_delete().events(0);
    const auto [add_delete_ec, add_delete_response] = CallReportEvent(add_then_delete, "delta_order_add_then_delete");
    ASSERT_EQ(EC_OK, add_delete_ec);
    EXPECT_EQ(token, add_delete_response.committed_snapshot_version());
    EXPECT_TRUE(QueryEventReportUris({key}).empty());

    auto add_delete_add = MakeAddRequest(host, key, "first_add");
    *add_delete_add.add_events() = make_delete().events(0);
    *add_delete_add.add_events() = MakeAddRequest(host, key, "final_add").events(0);
    const auto [toggle_ec, toggle_response] = CallReportEvent(add_delete_add, "delta_order_add_delete_add");
    ASSERT_EQ(EC_OK, toggle_ec);
    EXPECT_EQ(token, toggle_response.committed_snapshot_version());
    visible = QueryEventReportUris({key});
    ASSERT_EQ(1u, visible.size());
    EXPECT_NE(std::string::npos, visible.front().find("source=final_add"));
    EXPECT_NE(std::string::npos, visible.front().find("s_version=" + token));
}

TEST_F(CacheManagerTest, TestReportEventFlatFoldMatchesReferenceAcrossBlocksMediaAndSpecs) {
    const std::string host = "192.168.10.81:8080";
    constexpr int64_t first_key = 96'000;
    constexpr size_t key_count = 48;
    constexpr size_t event_count = 768;
    std::vector<std::string> mediums;
    for (size_t i = 0; i < 17; ++i) {
        mediums.push_back("medium_" + std::to_string(i));
    }
    const std::array<std::string, 4> spec_names{"tp0", "tp1", "tp2", "tp3"};

    struct ExpectedSpec {
        std::string raw_uri;
        std::uint64_t size = 0;
    };
    std::map<int64_t, std::map<std::string, std::map<std::string, ExpectedSpec>>> expected;

    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, mediums));

    proto::meta::ReportEventRequest request;
    request.set_instance_id("test_instance");
    request.set_host_ip_port(host);
    request.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);

    std::uint64_t random_state = 0x9e3779b97f4a7c15ULL;
    auto next_random = [&random_state] {
        random_state = random_state * 6364136223846793005ULL + 1442695040888963407ULL;
        return random_state;
    };
    for (size_t event_index = 0; event_index < event_count; ++event_index) {
        const int64_t key = first_key + static_cast<int64_t>(next_random() % key_count);
        const std::string &medium = mediums[next_random() % mediums.size()];
        const size_t first_spec_index = next_random() % spec_names.size();
        const bool is_add = next_random() % 4 != 0;
        auto *event = request.add_events();
        if (is_add) {
            event->set_event_type(proto::meta::EVENT_BLOCK_ADD);
            auto *params = event->mutable_block_add();
            params->set_block_key(std::to_string(key));
            params->set_medium(medium);
            auto add_spec = [&](size_t spec_index) {
                const std::string &name = spec_names[spec_index];
                const std::uint64_t size = next_random() % 97 + 1;
                const std::string raw_uri = "event_report://" + host + "/" + medium + "?source=model_" +
                                            std::to_string(event_index) + "_" + name + "&size=" + std::to_string(size);
                auto *spec = params->add_specs();
                spec->set_name(name);
                spec->set_uri(raw_uri);
                expected[key][medium][name] = ExpectedSpec{raw_uri, size};
            };
            add_spec(first_spec_index);
            if (next_random() % 7 == 0) {
                add_spec((first_spec_index + 1) % spec_names.size());
            }
        } else {
            event->set_event_type(proto::meta::EVENT_BLOCK_DELETE);
            auto *params = event->mutable_block_delete();
            params->set_block_key(std::to_string(key));
            params->set_medium(medium);
            params->add_spec_names(spec_names[first_spec_index]);
            expected[key][medium].erase(spec_names[first_spec_index]);
            if (next_random() % 7 == 0) {
                const std::string &second_name = spec_names[(first_spec_index + 1) % spec_names.size()];
                params->add_spec_names(second_name);
                expected[key][medium].erase(second_name);
            }
        }
    }

    const auto [ec, response] = CallReportEvent(request, "flat_fold_reference_model");
    ASSERT_EQ(EC_OK, ec);
    EXPECT_EQ(proto::meta::OK, response.header().status().code());
    EXPECT_EQ(0, response.item_results_size());
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(response.committed_snapshot_version()));

    std::vector<int64_t> keys;
    keys.reserve(key_count);
    for (size_t i = 0; i < key_count; ++i) {
        keys.push_back(first_key + static_cast<int64_t>(i));
    }
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_NE(nullptr, meta_searcher);
    std::vector<CacheLocationMap> location_maps;
    BlockMask mask;
    ASSERT_EQ(EC_OK, meta_searcher->BatchGetLocation(request_context_.get(), keys, mask, location_maps));
    ASSERT_EQ(keys.size(), location_maps.size());

    std::uint64_t expected_total_size = 0;
    for (size_t key_index = 0; key_index < keys.size(); ++key_index) {
        using FlattenedSpecs = std::map<std::pair<std::string, std::string>, std::string>;
        FlattenedSpecs expected_specs;
        FlattenedSpecs actual_specs;
        size_t expected_location_count = 0;
        for (const auto &medium : mediums) {
            const auto expected_medium = expected[keys[key_index]].find(medium);
            if (expected_medium != expected[keys[key_index]].end()) {
                expected_location_count += static_cast<size_t>(!expected_medium->second.empty());
                for (const auto &[name, expected_spec] : expected_medium->second) {
                    std::string versioned_uri;
                    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
                        expected_spec.raw_uri, response.committed_snapshot_version(), versioned_uri));
                    expected_specs[{medium, name}] = std::move(versioned_uri);
                    expected_total_size += expected_spec.size;
                }
            }

            const auto location_it = location_maps[key_index].find(event_backend->BuildLocationId(medium, host));
            if (location_it == location_maps[key_index].end()) {
                continue;
            }
            ASSERT_TRUE(location_it->second);
            EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, location_it->second->type());
            EXPECT_EQ(location_it->second->location_specs().size(), location_it->second->spec_size());
            EXPECT_TRUE(std::is_sorted(location_it->second->location_specs().begin(),
                                       location_it->second->location_specs().end(),
                                       [](const auto &lhs, const auto &rhs) { return lhs.name() < rhs.name(); }));
            for (const auto &spec : location_it->second->location_specs()) {
                actual_specs[{medium, spec.name()}] = spec.uri();
            }
        }
        EXPECT_EQ(expected_location_count, location_maps[key_index].size()) << "block key " << keys[key_index];
        EXPECT_EQ(expected_specs, actual_specs) << "block key " << keys[key_index];
    }

    auto meta_indexer = cache_manager_->meta_indexer_manager_->GetMetaIndexer("test_instance");
    ASSERT_NE(nullptr, meta_indexer);
    EXPECT_EQ(expected_total_size,
              meta_indexer->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2));
}

TEST_F(CacheManagerTest, TestReportEventFoldedTotalSizeOverflowFailsWithoutMetadata) {
    const std::string host = "192.168.10.82:8080";
    constexpr int64_t key = 96'100;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    proto::meta::ReportEventRequest request;
    request.set_instance_id("test_instance");
    request.set_host_ip_port(host);
    request.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
    auto add_event = [&](const std::string &name, const std::string &size) {
        auto *event = request.add_events();
        event->set_event_type(proto::meta::EVENT_BLOCK_ADD);
        auto *params = event->mutable_block_add();
        params->set_block_key(std::to_string(key));
        params->set_medium("mem");
        auto *spec = params->add_specs();
        spec->set_name(name);
        spec->set_uri("event_report://" + host + "/mem?size=" + size);
    };
    add_event("tp0", "18446744073709551615");
    add_event("tp1", "1");

    const auto [ec, response] = CallReportEvent(request, "folded_total_size_overflow");
    EXPECT_EQ(EC_PARTIAL_OK, ec);
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
    ASSERT_EQ(2, response.item_results_size());
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.item_results(0));
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.item_results(1));
    EXPECT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(response.committed_snapshot_version()));
    EXPECT_TRUE(response.snapshot_required());
    EXPECT_TRUE(QueryRawEventReportUris(key).empty());

    auto meta_indexer = cache_manager_->meta_indexer_manager_->GetMetaIndexer("test_instance");
    ASSERT_NE(nullptr, meta_indexer);
    EXPECT_EQ(0u, meta_indexer->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2));
}

TEST_F(CacheManagerTest, TestReportEventRejectsMergeThatOverflowsExistingLocation) {
    const std::string host = "192.168.10.83:8080";
    constexpr int64_t key = 96'101;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    auto make_add = [&](const std::string &name, const std::string &size) {
        proto::meta::ReportEventRequest request;
        request.set_instance_id("test_instance");
        request.set_host_ip_port(host);
        request.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
        auto *event = request.add_events();
        event->set_event_type(proto::meta::EVENT_BLOCK_ADD);
        auto *params = event->mutable_block_add();
        params->set_block_key(std::to_string(key));
        params->set_medium("mem");
        auto *spec = params->add_specs();
        spec->set_name(name);
        spec->set_uri("event_report://" + host + "/mem?size=" + size);
        return request;
    };

    const auto [initial_ec, initial_response] =
        CallReportEvent(make_add("tp0", "18446744073709551615"), "existing_size_max");
    ASSERT_EQ(EC_OK, initial_ec);
    ASSERT_EQ(proto::meta::OK, initial_response.header().status().code());

    const auto [overflow_ec, overflow_response] =
        CallReportEvent(make_add("tp1", "1"), "existing_plus_new_size_overflow");
    EXPECT_EQ(EC_PARTIAL_OK, overflow_ec);
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, overflow_response.header().status().code());
    ASSERT_EQ(1, overflow_response.item_results_size());
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, overflow_response.item_results(0));

    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_NE(nullptr, meta_searcher);
    std::vector<CacheLocationMap> location_maps;
    BlockMask mask;
    ASSERT_EQ(EC_OK, meta_searcher->BatchGetLocation(request_context_.get(), {key}, mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    ASSERT_EQ(1u, location_maps[0].size());
    const auto &stored_specs = location_maps[0].begin()->second->location_specs();
    ASSERT_EQ(1u, stored_specs.size());
    EXPECT_EQ("tp0", stored_specs[0].name());

    auto meta_indexer = cache_manager_->meta_indexer_manager_->GetMetaIndexer("test_instance");
    ASSERT_NE(nullptr, meta_indexer);
    EXPECT_EQ(std::numeric_limits<std::uint64_t>::max(),
              meta_indexer->GetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2));
}

TEST_F(CacheManagerTest, TestReportEventFoldedDeltaEventsShareFinalWriteFailure) {
    const std::string host = "192.168.10.52:8080";
    const int64_t key = 94'440;
    auto event_backend = InstallEventReportBackend();
    auto *meta_backend = InstallControllableMetaBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_NE(nullptr, meta_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    auto request = MakeAddRequest(host, key, "final_add");
    auto final_add = request.events(0);
    request.clear_events();
    auto *absorbed_delete = request.add_events();
    absorbed_delete->set_event_type(proto::meta::EVENT_BLOCK_DELETE);
    absorbed_delete->mutable_block_delete()->set_block_key(std::to_string(key));
    absorbed_delete->mutable_block_delete()->set_medium("mem");
    absorbed_delete->mutable_block_delete()->add_spec_names("tp0");
    *request.add_events() = final_add;

    meta_backend->FailKeyOnNextUpsert(key);
    const auto [ec, response] = CallReportEvent(request, "folded_delta_final_write_failure");
    EXPECT_EQ(EC_PARTIAL_OK, ec);
    EXPECT_EQ(proto::meta::INTERNAL_ERROR, response.header().status().code());
    ASSERT_EQ(2, response.item_results_size());
    EXPECT_EQ(proto::meta::INTERNAL_ERROR, response.item_results(0));
    EXPECT_EQ(proto::meta::INTERNAL_ERROR, response.item_results(1));
    EXPECT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(response.committed_snapshot_version()));
    EXPECT_TRUE(response.snapshot_required());
    EXPECT_TRUE(QueryEventReportUris({key}).empty());
}

TEST_F(CacheManagerTest, TestReportEventDeltaFailureMarksSafeRetryDependencyClosure) {
    const std::string host = "192.168.10.72:8080";
    const int64_t key = 94'472;
    auto event_backend = InstallEventReportBackend();
    auto *meta_backend = InstallControllableMetaBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_NE(nullptr, meta_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    ASSERT_EQ(EC_OK, CallReportEvent(MakeAddRequest(host, key, "baseline_a"), "retry_closure_baseline").first);

    auto request = MakeAddRequest(host, key, "readd_a");
    auto *spec_b = request.mutable_events(0)->mutable_block_add()->add_specs();
    spec_b->set_name("tp1");
    spec_b->set_uri("event_report://" + host + "/mem?source=add_b");
    auto *delete_a = request.add_events();
    delete_a->set_event_type(proto::meta::EVENT_BLOCK_DELETE);
    delete_a->mutable_block_delete()->set_block_key(std::to_string(key));
    delete_a->mutable_block_delete()->set_medium("mem");
    delete_a->mutable_block_delete()->add_spec_names("tp0");

    // The final ADD phase writes only tp1; tp0's final operation is DELETE.
    // Fail that ADD, then let DELETE succeed. Retrying only event 0 would
    // otherwise resurrect tp0, so both dependent events must be reported as
    // failed and retried in their original order.
    meta_backend->FailKeyOnNextUpsert(key);
    const auto [failed_ec, failed_response] = CallReportEvent(request, "retry_closure_injected_failure");
    EXPECT_EQ(EC_PARTIAL_OK, failed_ec);
    ASSERT_EQ(2, failed_response.item_results_size());
    EXPECT_EQ(proto::meta::INTERNAL_ERROR, failed_response.item_results(0));
    EXPECT_EQ(proto::meta::INTERNAL_ERROR, failed_response.item_results(1));
    EXPECT_TRUE(QueryRawEventReportUris(key).empty());

    const auto [retry_ec, retry_response] = CallReportEvent(request, "retry_closure_retry_all_failed_items");
    EXPECT_EQ(EC_OK, retry_ec);
    EXPECT_EQ(0, retry_response.item_results_size());
    const auto final_uris = QueryRawEventReportUris(key);
    ASSERT_EQ(1u, final_uris.size());
    EXPECT_NE(std::string::npos, final_uris.front().find("source=add_b"));
}

TEST_F(CacheManagerTest, TestReportEventLazilyRestoresReporterWithoutRegisterOrSnapshot) {
    const std::string host = "192.168.10.32:8080";
    const std::string snapshot_host = "192.168.10.132:8080";
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);

    proto::meta::ReportEventRequest heartbeat;
    heartbeat.set_instance_id("test_instance");
    heartbeat.set_host_ip_port(host);
    heartbeat.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
    heartbeat.add_events()->set_event_type(proto::meta::EVENT_HEARTBEAT);
    heartbeat.mutable_events(0)->mutable_heartbeat();
    const auto [heartbeat_ec, heartbeat_response] = CallReportEvent(heartbeat, "unregistered_heartbeat");
    EXPECT_EQ(EC_OK, heartbeat_ec);
    EXPECT_EQ(proto::meta::OK, heartbeat_response.header().status().code());
    EXPECT_EQ(0, heartbeat_response.item_results_size());
    EXPECT_TRUE(heartbeat_response.committed_snapshot_version().empty());
    EXPECT_TRUE(heartbeat_response.snapshot_required());
    EXPECT_TRUE(event_backend->IsNodeRegistered("test_instance", host));
    EXPECT_TRUE(event_backend->IsNodeAvailable("test_instance", host));

    const auto [first_delta_ec, first_delta] =
        CallReportEvent(MakeAddRequest(host, 9427, "first_delta"), "delta_without_register_or_snapshot");
    ASSERT_EQ(EC_OK, first_delta_ec);
    EXPECT_EQ(proto::meta::OK, first_delta.header().status().code());
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(first_delta.committed_snapshot_version()));
    EXPECT_TRUE(first_delta.snapshot_required());
    auto visible = QueryEventReportUris({9427});
    ASSERT_EQ(1u, visible.size());
    EXPECT_NE(std::string::npos, visible.front().find("source=first_delta"));
    EXPECT_NE(std::string::npos, visible.front().find("s_version=" + first_delta.committed_snapshot_version()));

    const auto [first_snapshot_ec, first_snapshot] =
        CallReportEvent(MakeSnapshotRequest(snapshot_host, {}), "snapshot_without_register");
    ASSERT_EQ(EC_OK, first_snapshot_ec);
    EXPECT_EQ(proto::meta::OK, first_snapshot.header().status().code());
    EXPECT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(first_snapshot.committed_snapshot_version()));
    EXPECT_FALSE(first_snapshot.snapshot_required());
    EXPECT_TRUE(event_backend->IsNodeRegistered("test_instance", snapshot_host));
    EXPECT_TRUE(event_backend->IsNodeAvailable("test_instance", snapshot_host));

    const auto [baseline_ec, baseline] = CallReportEvent(MakeSnapshotRequest(host, {}), "registered_empty_snapshot");
    ASSERT_EQ(EC_OK, baseline_ec);
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(baseline.committed_snapshot_version()));
    EXPECT_NE(first_delta.committed_snapshot_version(), baseline.committed_snapshot_version());
    const auto cleanup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < cleanup_deadline && !QueryEventReportUris({9427}).empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_TRUE(QueryEventReportUris({9427}).empty());

    event_backend->SetNodeUnavailable("test_instance", host);
    const auto [unavailable_delta_ec, unavailable_delta] =
        CallReportEvent(MakeAddRequest(host, 9427, "registered_but_unavailable"), "registered_unavailable_delta");
    ASSERT_EQ(EC_OK, unavailable_delta_ec);
    EXPECT_EQ(baseline.committed_snapshot_version(), unavailable_delta.committed_snapshot_version());
    EXPECT_TRUE(QueryEventReportUris({9427}).empty());

    ASSERT_EQ(EC_OK, event_backend->OnHeartbeat("test_instance", host, {}));
    visible = QueryEventReportUris({9427});
    ASSERT_EQ(1u, visible.size());
    EXPECT_NE(std::string::npos, visible.front().find("source=registered_but_unavailable"));
}

TEST_F(CacheManagerTest, TestReportEventHeartbeatFailureDoesNotOverwriteMalformedItems) {
    const std::string host = "192.168.10.79:8080";
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));
    ASSERT_EQ(EC_OK, event_backend->UnregisterNode("test_instance", host));

    proto::meta::ReportEventRequest request;
    request.set_instance_id("test_instance");
    request.set_host_ip_port(host);
    request.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
    // This item is structurally invalid and must keep INVALID_ARGUMENT even
    // when the valid heartbeat in the same request fails at the backend.
    request.add_events()->set_event_type(proto::meta::EVENT_HEARTBEAT);
    auto *valid_heartbeat = request.add_events();
    valid_heartbeat->set_event_type(proto::meta::EVENT_HEARTBEAT);
    valid_heartbeat->mutable_heartbeat();

    const auto [ec, response] = CallReportEvent(request, "mixed_invalid_and_tombstoned_heartbeat");
    EXPECT_EQ(EC_PARTIAL_OK, ec);
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
    ASSERT_EQ(2, response.item_results_size());
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.item_results(0));
    EXPECT_EQ(proto::meta::NODE_NOT_REGISTERED, response.item_results(1));
    EXPECT_FALSE(event_backend->IsNodeRegistered("test_instance", host));
}

TEST_F(CacheManagerTest, TestReportEventSnapshotRequiredOnlyForGenerationCreatingDelta) {
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);

    const std::string add_host = "192.168.10.133:8080";
    proto::meta::ReportEventRequest heartbeat;
    heartbeat.set_instance_id("test_instance");
    heartbeat.set_host_ip_port(add_host);
    heartbeat.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
    auto *heartbeat_event = heartbeat.add_events();
    heartbeat_event->set_event_type(proto::meta::EVENT_HEARTBEAT);
    heartbeat_event->mutable_heartbeat();
    const auto [heartbeat_ec, heartbeat_response] = CallReportEvent(heartbeat, "snapshot_required_heartbeat");
    ASSERT_EQ(EC_OK, heartbeat_ec);
    EXPECT_TRUE(heartbeat_response.committed_snapshot_version().empty());
    EXPECT_TRUE(heartbeat_response.snapshot_required());

    const auto [first_add_ec, first_add] =
        CallReportEvent(MakeAddRequest(add_host, 94'427, "first_add"), "snapshot_required_first_add");
    ASSERT_EQ(EC_OK, first_add_ec);
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(first_add.committed_snapshot_version()));
    EXPECT_TRUE(first_add.snapshot_required());

    const auto [second_add_ec, second_add] =
        CallReportEvent(MakeAddRequest(add_host, 94'428, "second_add"), "snapshot_required_second_add");
    ASSERT_EQ(EC_OK, second_add_ec);
    EXPECT_EQ(first_add.committed_snapshot_version(), second_add.committed_snapshot_version());
    EXPECT_FALSE(second_add.snapshot_required());

    const std::string delete_host = "192.168.10.134:8080";
    auto first_delete_request = MakeAddRequest(delete_host, 94'429, "placeholder");
    first_delete_request.clear_events();
    auto *delete_event = first_delete_request.add_events();
    delete_event->set_event_type(proto::meta::EVENT_BLOCK_DELETE);
    delete_event->mutable_block_delete()->set_block_key("94429");
    delete_event->mutable_block_delete()->set_medium("mem");
    delete_event->mutable_block_delete()->add_spec_names("tp0");
    const auto [first_delete_ec, first_delete] =
        CallReportEvent(first_delete_request, "snapshot_required_first_delete");
    ASSERT_EQ(EC_OK, first_delete_ec);
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(first_delete.committed_snapshot_version()));
    EXPECT_TRUE(first_delete.snapshot_required());

    const std::string snapshot_host = "192.168.10.135:8080";
    const auto [snapshot_ec, snapshot] =
        CallReportEvent(MakeSnapshotRequest(snapshot_host, {}), "snapshot_required_first_snapshot");
    ASSERT_EQ(EC_OK, snapshot_ec);
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(snapshot.committed_snapshot_version()));
    EXPECT_FALSE(snapshot.snapshot_required());
}

TEST_F(CacheManagerTest, TestReportEventSnapshotWhileUnavailableCommitsButStaysHiddenUntilHeartbeat) {
    const std::string host = "192.168.10.53:8080";
    const int64_t key = 94'441;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    const auto [baseline_ec, baseline] =
        CallReportEvent(MakeAddRequest(host, key, "before_unavailable"), "snapshot_unavailable_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    const std::string baseline_generation = baseline.committed_snapshot_version();
    ASSERT_EQ(1u, QueryEventReportUris({key}).size());

    event_backend->SetNodeUnavailable("test_instance", host);
    ASSERT_FALSE(event_backend->IsNodeAvailable("test_instance", host));
    const auto [snapshot_ec, snapshot] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "snapshot_while_unavailable"}}), "snapshot_while_unavailable");
    ASSERT_EQ(EC_OK, snapshot_ec);
    const std::string snapshot_generation = snapshot.committed_snapshot_version();
    EXPECT_NE(baseline_generation, snapshot_generation);
    EXPECT_FALSE(snapshot.snapshot_required());
    EXPECT_TRUE(QueryEventReportUris({key}).empty());

    proto::meta::ReportEventRequest heartbeat;
    heartbeat.set_instance_id("test_instance");
    heartbeat.set_host_ip_port(host);
    heartbeat.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
    auto *heartbeat_event = heartbeat.add_events();
    heartbeat_event->set_event_type(proto::meta::EVENT_HEARTBEAT);
    heartbeat_event->mutable_heartbeat();
    const auto [heartbeat_ec, heartbeat_response] =
        CallReportEvent(heartbeat, "snapshot_unavailable_recovery_heartbeat");
    ASSERT_EQ(EC_OK, heartbeat_ec);
    EXPECT_EQ(snapshot_generation, heartbeat_response.committed_snapshot_version());

    const auto visible = QueryEventReportUris({key});
    ASSERT_EQ(1u, visible.size());
    EXPECT_NE(std::string::npos, visible.front().find("source=snapshot_while_unavailable"));
    EXPECT_NE(std::string::npos, visible.front().find("s_version=" + snapshot_generation));
}

TEST_F(CacheManagerTest, TestReportEventHeartbeatRecoveryCarriesSameRequestMutationsIntoNewLifecycle) {
    const std::string host = "192.168.10.73:8080";
    const int64_t add_key = 94'473;
    const int64_t delete_key = 94'474;
    const int64_t snapshot_key = 94'475;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));
    ASSERT_EQ(EC_OK,
              CallReportEvent(MakeAddRequest(host, delete_key, "delete_baseline"), "recovery_batch_baseline").first);

    event_backend->SetNodeUnavailable("test_instance", host);
    const uint64_t add_old_generation = event_backend->GetNodeGeneration("test_instance", host);
    auto heartbeat_then_add = MakeAddRequest(host, add_key, "heartbeat_then_add");
    const auto add_event = heartbeat_then_add.events(0);
    heartbeat_then_add.clear_events();
    auto *heartbeat = heartbeat_then_add.add_events();
    heartbeat->set_event_type(proto::meta::EVENT_HEARTBEAT);
    (*heartbeat->mutable_heartbeat()->mutable_system_status())["phase"] = "recover_add";
    *heartbeat_then_add.add_events() = add_event;
    const auto [add_ec, add_response] = CallReportEvent(heartbeat_then_add, "recovery_batch_heartbeat_then_add");
    ASSERT_EQ(EC_OK, add_ec);
    EXPECT_EQ(0, add_response.item_results_size());
    EXPECT_GT(event_backend->GetNodeGeneration("test_instance", host), add_old_generation);
    ASSERT_EQ(1u, QueryEventReportUris({add_key}).size());

    event_backend->SetNodeUnavailable("test_instance", host);
    proto::meta::ReportEventRequest delete_then_heartbeat;
    delete_then_heartbeat.set_instance_id("test_instance");
    delete_then_heartbeat.set_host_ip_port(host);
    delete_then_heartbeat.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
    auto *delete_event = delete_then_heartbeat.add_events();
    delete_event->set_event_type(proto::meta::EVENT_BLOCK_DELETE);
    delete_event->mutable_block_delete()->set_block_key(std::to_string(delete_key));
    delete_event->mutable_block_delete()->set_medium("mem");
    delete_event->mutable_block_delete()->add_spec_names("tp0");
    heartbeat = delete_then_heartbeat.add_events();
    heartbeat->set_event_type(proto::meta::EVENT_HEARTBEAT);
    heartbeat->mutable_heartbeat();
    const auto [delete_ec, delete_response] =
        CallReportEvent(delete_then_heartbeat, "recovery_batch_delete_then_heartbeat");
    ASSERT_EQ(EC_OK, delete_ec);
    EXPECT_EQ(0, delete_response.item_results_size());
    EXPECT_TRUE(QueryEventReportUris({delete_key}).empty());

    event_backend->SetNodeUnavailable("test_instance", host);
    auto heartbeat_then_snapshot = MakeSnapshotRequest(host, {{snapshot_key, "heartbeat_then_snapshot"}});
    const auto snapshot_event = heartbeat_then_snapshot.events(0);
    heartbeat_then_snapshot.clear_events();
    heartbeat = heartbeat_then_snapshot.add_events();
    heartbeat->set_event_type(proto::meta::EVENT_HEARTBEAT);
    heartbeat->mutable_heartbeat();
    *heartbeat_then_snapshot.add_events() = snapshot_event;
    const auto [snapshot_ec, snapshot_response] =
        CallReportEvent(heartbeat_then_snapshot, "recovery_batch_heartbeat_then_snapshot");
    ASSERT_EQ(EC_OK, snapshot_ec);
    EXPECT_EQ(0, snapshot_response.item_results_size());
    EXPECT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(snapshot_response.committed_snapshot_version()));
    ASSERT_EQ(1u, QueryEventReportUris({snapshot_key}).size());
}

TEST_F(CacheManagerTest, TestReportEventRegisterThenFirstDeltaInSameRequest) {
    const std::string host = "192.168.10.48:8080";
    const int64_t key = 94'433;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);

    auto request = MakeAddRequest(host, key, "same_request_first_delta");
    const auto add_event = request.events(0);
    request.clear_events();
    auto *register_event = request.add_events();
    register_event->set_event_type(proto::meta::EVENT_NODE_REGISTER);
    register_event->mutable_node_register()->add_mediums("mem");
    *request.add_events() = add_event;
    auto *heartbeat_event = request.add_events();
    heartbeat_event->set_event_type(proto::meta::EVENT_HEARTBEAT);
    (*heartbeat_event->mutable_heartbeat()->mutable_system_status())["phase"] = "boot";

    const auto [ec, response] = CallReportEvent(request, "register_and_first_delta_same_request");
    ASSERT_EQ(EC_OK, ec);
    EXPECT_EQ(proto::meta::OK, response.header().status().code());
    EXPECT_EQ(0, response.item_results_size());
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(response.committed_snapshot_version()));
    EXPECT_TRUE(response.snapshot_required());
    EXPECT_TRUE(event_backend->IsNodeRegistered("test_instance", host));
    EXPECT_TRUE(event_backend->IsNodeAvailable("test_instance", host));

    const auto visible = QueryEventReportUris({key});
    ASSERT_EQ(1u, visible.size());
    EXPECT_NE(std::string::npos, visible.front().find("source=same_request_first_delta"));
    EXPECT_NE(std::string::npos, visible.front().find("s_version=" + response.committed_snapshot_version()));
}

TEST_F(CacheManagerTest, TestReportEventDeltaBeforeExplicitRegisterSucceedsInSameRequest) {
    const std::string host = "192.168.10.50:8080";
    const int64_t key = 94'436;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);

    auto request = MakeAddRequest(host, key, "delta_before_register");
    auto *register_event = request.add_events();
    register_event->set_event_type(proto::meta::EVENT_NODE_REGISTER);
    register_event->mutable_node_register()->add_mediums("mem");
    auto *heartbeat_event = request.add_events();
    heartbeat_event->set_event_type(proto::meta::EVENT_HEARTBEAT);
    (*heartbeat_event->mutable_heartbeat()->mutable_system_status())["phase"] = "boot";

    const auto [ec, response] = CallReportEvent(request, "delta_before_register_same_request");
    ASSERT_EQ(EC_OK, ec);
    EXPECT_EQ(proto::meta::OK, response.header().status().code());
    EXPECT_EQ(0, response.item_results_size());
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(response.committed_snapshot_version()));
    EXPECT_TRUE(response.snapshot_required());
    EXPECT_TRUE(event_backend->IsNodeRegistered("test_instance", host));
    EXPECT_TRUE(event_backend->IsNodeAvailable("test_instance", host));
    const auto visible = QueryEventReportUris({key});
    ASSERT_EQ(1u, visible.size());
    EXPECT_NE(std::string::npos, visible.front().find("source=delta_before_register"));
    EXPECT_NE(std::string::npos, visible.front().find("s_version=" + response.committed_snapshot_version()));
}

TEST_F(CacheManagerTest, TestReportEventAdmissionFailurePropagatesToLaterRelatedMutation) {
    const std::string host = "192.168.10.81:8080";
    const int64_t key = 94'481;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));
    ASSERT_EQ(EC_OK, event_backend->UnregisterNode("test_instance", host));

    auto request = MakeAddRequest(host, key, "must_be_deleted_after_retry");
    auto *register_event = request.add_events();
    register_event->set_event_type(proto::meta::EVENT_NODE_REGISTER);
    register_event->mutable_node_register()->add_mediums("mem");
    auto *delete_event = request.add_events();
    delete_event->set_event_type(proto::meta::EVENT_BLOCK_DELETE);
    delete_event->mutable_block_delete()->set_block_key(std::to_string(key));
    delete_event->mutable_block_delete()->set_medium("mem");
    delete_event->mutable_block_delete()->add_spec_names("tp0");

    const auto [ec, response] = CallReportEvent(request, "admission_failure_dependency_closure");
    EXPECT_EQ(EC_PARTIAL_OK, ec);
    EXPECT_EQ(proto::meta::NODE_NOT_REGISTERED, response.header().status().code());
    ASSERT_EQ(3, response.item_results_size());
    EXPECT_EQ(proto::meta::NODE_NOT_REGISTERED, response.item_results(0));
    EXPECT_EQ(proto::meta::OK, response.item_results(1));
    // The DELETE physically succeeded, but it shares the retry dependency
    // group with the earlier failed ADD. Returning success here would let a
    // caller retry only ADD and reverse the request's last-operation-wins
    // result.
    EXPECT_EQ(proto::meta::NODE_NOT_REGISTERED, response.item_results(2));
    EXPECT_TRUE(QueryEventReportUris({key}).empty());

    proto::meta::ReportEventRequest retry = request;
    retry.clear_events();
    *retry.add_events() = request.events(0);
    *retry.add_events() = request.events(2);
    const auto [retry_ec, retry_response] = CallReportEvent(retry, "admission_failure_dependency_retry");
    EXPECT_EQ(EC_OK, retry_ec);
    EXPECT_EQ(0, retry_response.item_results_size());
    EXPECT_TRUE(QueryEventReportUris({key}).empty());
}

TEST_F(CacheManagerTest, TestReportEventValidatesMultipleRegisterItemsIndependently) {
    const std::string host = "192.168.10.74:8080";
    const int64_t key = 94'476;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);

    auto request = MakeAddRequest(host, key, "valid_register_survives_invalid_sibling");
    const auto add_event = request.events(0);
    request.clear_events();
    auto *valid_register = request.add_events();
    valid_register->set_event_type(proto::meta::EVENT_NODE_REGISTER);
    valid_register->mutable_node_register()->add_mediums("mem");
    auto *invalid_register = request.add_events();
    invalid_register->set_event_type(proto::meta::EVENT_NODE_REGISTER);
    invalid_register->mutable_node_register()->add_mediums("bad#medium");
    *request.add_events() = add_event;

    const auto [ec, response] = CallReportEvent(request, "multiple_register_item_validation");
    EXPECT_EQ(EC_PARTIAL_OK, ec);
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
    ASSERT_EQ(3, response.item_results_size());
    EXPECT_EQ(proto::meta::OK, response.item_results(0));
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.item_results(1));
    EXPECT_EQ(proto::meta::OK, response.item_results(2));
    EXPECT_TRUE(event_backend->IsNodeRegistered("test_instance", host));
    ASSERT_EQ(1u, QueryEventReportUris({key}).size());
}

TEST_F(CacheManagerTest, TestReportEventCoalescesMultipleValidRegistersIntoOneLifecycle) {
    const std::string host = "192.168.10.82:8080";
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);

    proto::meta::ReportEventRequest request;
    request.set_instance_id("test_instance");
    request.set_host_ip_port(host);
    request.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
    auto *mem_register = request.add_events();
    mem_register->set_event_type(proto::meta::EVENT_NODE_REGISTER);
    mem_register->mutable_node_register()->add_mediums("mem");
    auto *disk_register = request.add_events();
    disk_register->set_event_type(proto::meta::EVENT_NODE_REGISTER);
    disk_register->mutable_node_register()->add_mediums("disk");
    disk_register->mutable_node_register()->add_mediums("mem");

    const auto [ec, response] = CallReportEvent(request, "multiple_valid_registers_one_lifecycle");
    ASSERT_EQ(EC_OK, ec);
    EXPECT_EQ(proto::meta::OK, response.header().status().code());
    EXPECT_EQ(0, response.item_results_size());
    EXPECT_EQ(1u, event_backend->GetNodeGeneration("test_instance", host));

    const auto [retry_ec, retry_response] = CallReportEvent(request, "multiple_valid_registers_next_request");
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ(0, retry_response.item_results_size());
    EXPECT_EQ(2u, event_backend->GetNodeGeneration("test_instance", host));
}

TEST_F(CacheManagerTest, TestReportEventDisabledBackendRejectsReportsAndHidesExistingLocations) {
    const std::string host = "192.168.10.75:8080";
    const int64_t baseline_key = 94'477;
    const int64_t rejected_key = 94'478;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK,
              CallReportEvent(MakeAddRequest(host, baseline_key, "before_disable"), "disable_backend_baseline").first);
    ASSERT_EQ(1u, QueryEventReportUris({baseline_key}).size());

    event_backend->SetAvailable(false);
    EXPECT_TRUE(QueryEventReportUris({baseline_key}).empty());
    const auto [rejected_ec, rejected_response] =
        CallReportEvent(MakeAddRequest(host, rejected_key, "must_not_write"), "disable_backend_reject_report");
    EXPECT_EQ(EC_INSTANCE_NOT_EXIST, rejected_ec);
    EXPECT_EQ(proto::meta::INSTANCE_NOT_EXIST, rejected_response.header().status().code());
    EXPECT_TRUE(QueryRawEventReportUris(rejected_key).empty());

    event_backend->SetAvailable(true);
    ASSERT_EQ(1u, QueryEventReportUris({baseline_key}).size());
}

TEST_F(CacheManagerTest, TestHostCleanupCannotCrossEventBackendIncarnations) {
    const std::string host = "192.168.10.76:8080";
    const int64_t key = 94'479;
    auto old_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, old_backend);
    ASSERT_EQ(EC_OK, CallReportEvent(MakeAddRequest(host, key, "old_incarnation"), "old_incarnation_add").first);
    const uint64_t old_generation = old_backend->GetNodeGeneration("test_instance", host);
    ASSERT_NE(0u, old_generation);
    ASSERT_EQ(EC_OK, old_backend->Close());

    auto new_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, new_backend);
    ASSERT_EQ(EC_OK, CallReportEvent(MakeAddRequest(host, key, "new_incarnation"), "new_incarnation_add").first);
    ASSERT_EQ(old_generation, new_backend->GetNodeGeneration("test_instance", host));

    cache_manager_->CleanupHostLocations(
        "test_instance", host, old_generation, DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, old_backend);
    const auto uris = QueryRawEventReportUris(key);
    ASSERT_EQ(1u, uris.size());
    EXPECT_NE(std::string::npos, uris.front().find("source=new_incarnation"));
}

TEST_F(CacheManagerTest, TestHostCleanupUsesTheBackendThatCurrentlyWinsCandidateRouting) {
    const std::string host = "192.168.10.77:8080";
    const int64_t key = 94'480;
    auto old_backend = InstallEventReportBackend("event_report_old_candidate");
    ASSERT_NE(nullptr, old_backend);
    ASSERT_EQ(EC_OK, CallReportEvent(MakeAddRequest(host, key, "old_candidate"), "old_candidate_add").first);
    const uint64_t old_generation = old_backend->GetNodeGeneration("test_instance", host);
    ASSERT_NE(0u, old_generation);

    auto current_backend = InstallEventReportBackend("event_report_current_candidate");
    ASSERT_NE(nullptr, current_backend);
    const std::string group_name = registry_manager_->GetInstanceGroupName("test_instance");
    registry_manager_->instance_group_configs_.at(group_name)
        ->set_event_report_storage_candidates({"event_report_current_candidate", "event_report_old_candidate"});
    ASSERT_EQ(EC_OK, CallReportEvent(MakeAddRequest(host, key, "current_candidate"), "current_candidate_add").first);
    ASSERT_EQ(old_generation, current_backend->GetNodeGeneration("test_instance", host));

    cache_manager_->CleanupHostLocations(
        "test_instance", host, old_generation, DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, old_backend);
    const auto uris = QueryRawEventReportUris(key);
    ASSERT_EQ(1u, uris.size());
    EXPECT_NE(std::string::npos, uris.front().find("source=current_candidate"));
}

TEST_F(CacheManagerTest, TestSnapshotCleanupCannotCrossBackendReplacementDuringScan) {
    const std::string host = "192.168.10.78:8080";
    const int64_t key = 94'482;
    auto old_backend = InstallEventReportBackend();
    auto *meta_backend = InstallControllableMetaBackend();
    ASSERT_NE(nullptr, old_backend);
    ASSERT_NE(nullptr, meta_backend);

    const auto [baseline_ec, baseline] =
        CallReportEvent(MakeAddRequest(host, key, "old_snapshot"), "old_snapshot_for_cleanup");
    ASSERT_EQ(EC_OK, baseline_ec);
    const ReporterSnapshotKey reporter_key{"test_instance", host};
    const uint64_t cleanup_generation = old_backend->GetNodeGeneration("test_instance", host);

    std::string cleanup_version;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, old_backend->BeginSnapshot(reporter_key, cleanup_version, retry_after_ms));
    ASSERT_NE(baseline.committed_snapshot_version(), cleanup_version);
    ASSERT_TRUE(old_backend->CommitSnapshotVersion(reporter_key, cleanup_version));
    const uint64_t cleanup_epoch = old_backend->GetSnapshotAttemptEpoch(reporter_key);

    meta_backend->BlockNextLocationRead();
    auto cleanup = std::async(std::launch::async, [&] {
        return cache_manager_->CleanupStaleSnapshotLocations(reporter_key,
                                                             cleanup_version,
                                                             DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                                                             old_backend,
                                                             cleanup_epoch,
                                                             cleanup_generation);
    });
    ASSERT_TRUE(meta_backend->WaitUntilLocationReadEntered(std::chrono::seconds(2)));

    ASSERT_EQ(EC_OK, old_backend->Close());
    auto new_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, new_backend);
    ASSERT_EQ(EC_OK, CallReportEvent(MakeAddRequest(host, key, "new_backend_value"), "new_backend_value_add").first);

    meta_backend->ReleaseLocationRead();
    ASSERT_EQ(std::future_status::ready, cleanup.wait_for(std::chrono::seconds(2)));
    EXPECT_EQ(EC_OK, cleanup.get());
    const auto uris = QueryRawEventReportUris(key);
    ASSERT_EQ(1u, uris.size());
    EXPECT_NE(std::string::npos, uris.front().find("source=new_backend_value"));
}

TEST_F(CacheManagerTest, TestReportEventInvalidFirstDeltaDoesNotCreateVersionButPartialBatchDoes) {
    const std::string host = "192.168.10.49:8080";
    const int64_t invalid_key = 94'434;
    const int64_t valid_key = 94'435;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);

    auto invalid_request = MakeAddRequest(host, invalid_key, "invalid_without_specs");
    invalid_request.mutable_events(0)->mutable_block_add()->clear_specs();
    const auto [invalid_ec, invalid_response] =
        CallReportEvent(invalid_request, "invalid_first_delta_without_snapshot");
    ASSERT_EQ(EC_PARTIAL_OK, invalid_ec);
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, invalid_response.header().status().code());
    ASSERT_EQ(1, invalid_response.item_results_size());
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, invalid_response.item_results(0));
    EXPECT_TRUE(invalid_response.committed_snapshot_version().empty());
    EXPECT_TRUE(invalid_response.snapshot_required());
    EXPECT_TRUE(event_backend->GetSnapshotVersion({"test_instance", host}).empty());
    EXPECT_FALSE(event_backend->IsNodeRegistered("test_instance", host));
    EXPECT_TRUE(QueryEventReportUris({invalid_key}).empty());

    auto partial_request = invalid_request;
    *partial_request.add_events() = MakeAddRequest(host, valid_key, "valid_in_partial_first_delta").events(0);
    const auto [partial_ec, partial_response] =
        CallReportEvent(partial_request, "partial_first_delta_without_snapshot");
    ASSERT_EQ(EC_PARTIAL_OK, partial_ec);
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, partial_response.header().status().code());
    ASSERT_EQ(2, partial_response.item_results_size());
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, partial_response.item_results(0));
    EXPECT_EQ(proto::meta::OK, partial_response.item_results(1));
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(partial_response.committed_snapshot_version()));
    EXPECT_TRUE(partial_response.snapshot_required());
    EXPECT_EQ(partial_response.committed_snapshot_version(),
              event_backend->GetSnapshotVersion({"test_instance", host}));
    EXPECT_TRUE(event_backend->IsNodeRegistered("test_instance", host));

    EXPECT_TRUE(QueryEventReportUris({invalid_key}).empty());
    const auto visible = QueryEventReportUris({valid_key});
    ASSERT_EQ(1u, visible.size());
    EXPECT_NE(std::string::npos, visible.front().find("source=valid_in_partial_first_delta"));
    EXPECT_NE(std::string::npos, visible.front().find("s_version=" + partial_response.committed_snapshot_version()));
}

TEST_F(CacheManagerTest, TestReportEventFirstDeleteWithoutSnapshotCreatesReusableVersion) {
    const std::string host = "192.168.10.46:8080";
    const int64_t key = 94'430;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);

    auto delete_request = MakeAddRequest(host, key, "placeholder");
    delete_request.clear_events();
    auto *delete_event = delete_request.add_events();
    delete_event->set_event_type(proto::meta::EVENT_BLOCK_DELETE);
    delete_event->mutable_block_delete()->set_block_key(std::to_string(key));
    delete_event->mutable_block_delete()->set_medium("mem");
    delete_event->mutable_block_delete()->add_spec_names("tp0");

    const auto [delete_ec, delete_response] = CallReportEvent(delete_request, "first_delete_without_snapshot");
    ASSERT_EQ(EC_OK, delete_ec);
    const std::string version = delete_response.committed_snapshot_version();
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(version));
    EXPECT_TRUE(delete_response.snapshot_required());
    EXPECT_TRUE(event_backend->IsNodeRegistered("test_instance", host));
    EXPECT_TRUE(QueryEventReportUris({key}).empty());

    const auto [add_ec, add_response] =
        CallReportEvent(MakeAddRequest(host, key, "after_first_delete"), "delta_after_first_delete");
    ASSERT_EQ(EC_OK, add_ec);
    EXPECT_EQ(version, add_response.committed_snapshot_version());
    const auto visible = QueryEventReportUris({key});
    ASSERT_EQ(1u, visible.size());
    EXPECT_NE(std::string::npos, visible.front().find("source=after_first_delete"));
    EXPECT_NE(std::string::npos, visible.front().find("s_version=" + version));
}

TEST_F(CacheManagerTest, TestReportEventMissingBlockDeletesRemainSuccessfulAsOneBatch) {
    const std::string host = "192.168.10.51:8080";
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    auto request = MakeAddRequest(host, 94'437, "placeholder");
    request.clear_events();
    for (const int64_t key : {94'437, 94'438, 94'439}) {
        auto *delete_event = request.add_events();
        delete_event->set_event_type(proto::meta::EVENT_BLOCK_DELETE);
        delete_event->mutable_block_delete()->set_block_key(std::to_string(key));
        delete_event->mutable_block_delete()->set_medium("mem");
        delete_event->mutable_block_delete()->add_spec_names("tp0");
    }

    const auto [ec, response] = CallReportEvent(request, "missing_block_delete_batch");
    ASSERT_EQ(EC_OK, ec);
    EXPECT_EQ(proto::meta::OK, response.header().status().code());
    EXPECT_EQ(0, response.item_results_size());
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(response.committed_snapshot_version()));
    EXPECT_TRUE(response.snapshot_required());
    EXPECT_TRUE(QueryEventReportUris({94'437, 94'438, 94'439}).empty());
}

TEST_F(CacheManagerTest, TestReportEventLargeDeltaBatchAcrossRepeatedMediums) {
    const std::string host = "192.168.10.56:8080";
    constexpr int64_t first_key = 95'000;
    constexpr size_t event_count = 512;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);

    auto request = MakeAddRequest(host, first_key, "batch_0");
    for (size_t i = 1; i < event_count; ++i) {
        auto event_request = MakeAddRequest(host, first_key + static_cast<int64_t>(i), "batch_" + std::to_string(i));
        if (i % 2 != 0) {
            event_request.mutable_events(0)->mutable_block_add()->set_medium("disk");
            event_request.mutable_events(0)->mutable_block_add()->mutable_specs(0)->set_uri(
                "event_report://" + host + "/disk?source=batch_" + std::to_string(i));
        }
        *request.add_events() = event_request.events(0);
    }

    const auto [ec, response] = CallReportEvent(request, "large_repeated_medium_delta_batch");
    ASSERT_EQ(EC_OK, ec);
    EXPECT_EQ(proto::meta::OK, response.header().status().code());
    EXPECT_EQ(0, response.item_results_size());
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(response.committed_snapshot_version()));
    EXPECT_TRUE(response.snapshot_required());

    const auto visible = QueryEventReportUris({first_key,
                                               first_key + static_cast<int64_t>(event_count / 2),
                                               first_key + static_cast<int64_t>(event_count - 1)});
    ASSERT_EQ(3u, visible.size());
    EXPECT_TRUE(std::any_of(visible.begin(), visible.end(), [](const auto &uri) {
        return uri.find("source=batch_0") != std::string::npos;
    }));
    EXPECT_TRUE(std::any_of(visible.begin(), visible.end(), [](const auto &uri) {
        return uri.find("source=batch_256") != std::string::npos;
    }));
    EXPECT_TRUE(std::any_of(visible.begin(), visible.end(), [](const auto &uri) {
        return uri.find("source=batch_511") != std::string::npos;
    }));
}

TEST_F(CacheManagerTest, TestReportEventRestartKeepsHistoricalCacheAndAcceptsDeltaWithoutSnapshot) {
    const std::string host = "192.168.10.47:8080";
    const int64_t historical_key = 94'431;
    const int64_t realtime_key = 94'432;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    const auto [before_ec, before] =
        CallReportEvent(MakeAddRequest(host, historical_key, "before_restart"), "delta_before_restart");
    ASSERT_EQ(EC_OK, before_ec);
    const std::string previous_version = before.committed_snapshot_version();
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(previous_version));
    ASSERT_EQ(1u, QueryEventReportUris({historical_key}).size());

    const StorageConfig storage_config = event_backend->GetStorageConfig();
    ASSERT_EQ(EC_OK, event_backend->Close());
    // A process/configuration restart creates a fresh backend incarnation. A
    // closed object deliberately cannot be reopened because queued callbacks
    // and lifecycle fences are scoped to exactly one incarnation.
    event_backend = std::make_shared<EventReportBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, event_backend->Open(storage_config, "delta_only_restart"));
    event_backend->SetSnapshotMinIntervalMsForTest(0);
    registry_manager_->data_storage_manager_->storage_map_[storage_config.global_unique_name()] = event_backend;
    EXPECT_TRUE(event_backend->GetSnapshotVersion({"test_instance", host}).empty());
    EXPECT_TRUE(QueryEventReportUris({historical_key}).empty());

    const auto [delta_ec, delta] =
        CallReportEvent(MakeAddRequest(host, realtime_key, "after_restart"), "first_delta_after_restart");
    ASSERT_EQ(EC_OK, delta_ec);
    EXPECT_EQ(proto::meta::OK, delta.header().status().code());
    EXPECT_TRUE(delta.snapshot_required());
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(delta.committed_snapshot_version()));
    EXPECT_NE(previous_version, delta.committed_snapshot_version());
    EXPECT_TRUE(event_backend->IsNodeRegistered("test_instance", host));
    EXPECT_TRUE(event_backend->IsNodeAvailable("test_instance", host));

    const int64_t followup_key = 94'433;
    const auto [followup_ec, followup] =
        CallReportEvent(MakeAddRequest(host, followup_key, "second_after_restart"), "second_delta_after_restart");
    ASSERT_EQ(EC_OK, followup_ec);
    EXPECT_FALSE(followup.snapshot_required());
    EXPECT_EQ(delta.committed_snapshot_version(), followup.committed_snapshot_version());

    const auto visible = QueryEventReportUris({historical_key, realtime_key, followup_key});
    ASSERT_EQ(3u, visible.size());
    EXPECT_TRUE(std::any_of(visible.begin(), visible.end(), [](const std::string &uri) {
        return uri.find("source=before_restart") != std::string::npos;
    }));
    EXPECT_TRUE(std::any_of(visible.begin(), visible.end(), [](const std::string &uri) {
        return uri.find("source=after_restart") != std::string::npos;
    }));
    EXPECT_TRUE(std::any_of(visible.begin(), visible.end(), [](const std::string &uri) {
        return uri.find("source=second_after_restart") != std::string::npos;
    }));
}

TEST_F(CacheManagerTest, TestReportEventRejectsCanonicalDuplicateSnapshotKeysButAllowsDifferentMedia) {
    const std::string host = "192.168.10.33:8080";
    const int64_t key = 9428;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem", "disk"}));
    const auto [baseline_ec, baseline] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "baseline"}}), "canonical_duplicate_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    const std::string baseline_token = baseline.committed_snapshot_version();

    auto make_snapshot_with_keys = [&](const std::string &first_key,
                                       const std::string &first_medium,
                                       const std::string &second_key,
                                       const std::string &second_medium) {
        proto::meta::ReportEventRequest request;
        request.set_instance_id("test_instance");
        request.set_host_ip_port(host);
        request.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
        auto *snapshot = request.add_events()->mutable_block_snapshot();
        request.mutable_events(0)->set_event_type(proto::meta::EVENT_BLOCK_SNAPSHOT);
        for (const auto &[block_key, medium, spec_name] :
             {std::tuple<std::string, std::string, std::string>{first_key, first_medium, "tp0"},
              std::tuple<std::string, std::string, std::string>{second_key, second_medium, "tp1"}}) {
            auto *block = snapshot->add_blocks();
            block->set_block_key(block_key);
            block->set_medium(medium);
            auto *spec = block->add_specs();
            spec->set_name(spec_name);
            spec->set_uri("event_report://" + host + "/" + medium + "?source=" + spec_name);
        }
        return request;
    };

    const auto [duplicate_ec, duplicate_response] =
        CallReportEvent(make_snapshot_with_keys(std::to_string(key), "mem", "0" + std::to_string(key), "mem"),
                        "canonical_duplicate_snapshot");
    EXPECT_EQ(EC_PARTIAL_OK, duplicate_ec);
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, duplicate_response.header().status().code());
    EXPECT_EQ(baseline_token, duplicate_response.committed_snapshot_version());
    EXPECT_FALSE(duplicate_response.snapshot_required());
    EXPECT_EQ(baseline_token, event_backend->GetSnapshotVersion({"test_instance", host}));
    const auto visible_after_rejection = QueryEventReportUris({key});
    ASSERT_EQ(1u, visible_after_rejection.size());
    EXPECT_NE(std::string::npos, visible_after_rejection.front().find("source=baseline"));
    EXPECT_EQ(1u, QueryRawEventReportUris(key).size());

    const auto [different_media_ec, different_media_response] =
        CallReportEvent(make_snapshot_with_keys(std::to_string(key), "mem", "0" + std::to_string(key), "disk"),
                        "same_key_different_media_snapshot");
    ASSERT_EQ(EC_OK, different_media_ec);
    const std::string token = different_media_response.committed_snapshot_version();
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(token));
    const auto visible = QueryEventReportUris({key});
    ASSERT_EQ(2u, visible.size());
    for (const auto &uri : visible) {
        EXPECT_NE(std::string::npos, uri.find("s_version=" + token));
    }
}

TEST_F(CacheManagerTest, TestReportEventSnapshotReplacesCompleteSpecSetPerBlock) {
    const std::string host = "192.168.10.34:8080";
    const int64_t key = 9429;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    auto make_snapshot = [&](const std::vector<std::pair<std::string, std::string>> &spec_sources) {
        proto::meta::ReportEventRequest request;
        request.set_instance_id("test_instance");
        request.set_host_ip_port(host);
        request.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
        auto *event = request.add_events();
        event->set_event_type(proto::meta::EVENT_BLOCK_SNAPSHOT);
        auto *block = event->mutable_block_snapshot()->add_blocks();
        block->set_block_key(std::to_string(key));
        block->set_medium("mem");
        for (const auto &[spec_name, source] : spec_sources) {
            auto *spec = block->add_specs();
            spec->set_name(spec_name);
            spec->set_uri("event_report://" + host + "/mem?source=" + source);
        }
        return request;
    };

    const auto [baseline_ec, baseline] =
        CallReportEvent(make_snapshot({{"tp0", "old_tp0"}, {"tp1", "old_tp1"}}), "complete_specs_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    const std::string baseline_token = baseline.committed_snapshot_version();
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(baseline_token));

    const auto [replace_ec, replacement] =
        CallReportEvent(make_snapshot({{"tp1", "new_tp1"}, {"tp2", "new_tp2"}}), "complete_specs_replace");
    ASSERT_EQ(EC_OK, replace_ec);
    const std::string replacement_token = replacement.committed_snapshot_version();
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(replacement_token));
    EXPECT_NE(baseline_token, replacement_token);

    const auto visible = QueryEventReportUris({key});
    ASSERT_EQ(2u, visible.size());
    bool found_tp1 = false;
    bool found_tp2 = false;
    for (const auto &uri : visible) {
        EXPECT_EQ(std::string::npos, uri.find("source=old_tp0"));
        EXPECT_EQ(std::string::npos, uri.find("source=old_tp1"));
        EXPECT_NE(std::string::npos, uri.find("s_version=" + replacement_token));
        found_tp1 = found_tp1 || uri.find("source=new_tp1") != std::string::npos;
        found_tp2 = found_tp2 || uri.find("source=new_tp2") != std::string::npos;
    }
    EXPECT_TRUE(found_tp1);
    EXPECT_TRUE(found_tp2);
    EXPECT_EQ(2u, QueryRawEventReportUris(key).size());
}

TEST_F(CacheManagerTest, TestReportEventRejectsDuplicatePhysicalSnapshotItemsWithoutStateChange) {
    const std::string host = "192.168.10.35:8080";
    const int64_t key = 9433;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    const auto [baseline_ec, baseline] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "baseline"}}), "physical_duplicate_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    const std::string baseline_token = baseline.committed_snapshot_version();

    auto duplicate = MakeSnapshotRequest(host, {});
    auto *snapshot = duplicate.mutable_events(0)->mutable_block_snapshot();
    for (int i = 0; i < 2; ++i) {
        auto *block = snapshot->add_blocks();
        block->set_block_key(std::to_string(key));
        block->set_medium("mem");
        auto *spec = block->add_specs();
        spec->set_name("tp0");
        spec->set_uri("event_report://" + host + "/mem?source=physical_duplicate");
    }

    const auto [duplicate_ec, duplicate_response] = CallReportEvent(duplicate, "physical_duplicate_snapshot");
    EXPECT_EQ(EC_PARTIAL_OK, duplicate_ec);
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, duplicate_response.header().status().code());
    ASSERT_EQ(1, duplicate_response.item_results_size());
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, duplicate_response.item_results(0));
    EXPECT_EQ(baseline_token, duplicate_response.committed_snapshot_version());
    EXPECT_FALSE(duplicate_response.snapshot_required());
    EXPECT_EQ(baseline_token, event_backend->GetSnapshotVersion({"test_instance", host}));

    const auto visible_after_rejection = QueryEventReportUris({key});
    ASSERT_EQ(1u, visible_after_rejection.size());
    EXPECT_NE(std::string::npos, visible_after_rejection.front().find("source=baseline"));
    EXPECT_NE(std::string::npos, visible_after_rejection.front().find("s_version=" + baseline_token));
    EXPECT_EQ(1u, QueryRawEventReportUris(key).size());

    const auto [retry_ec, retry] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "deduplicated"}}), "physical_duplicate_retry");
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_NE(baseline_token, retry.committed_snapshot_version());
    const auto visible_after_retry = QueryEventReportUris({key});
    ASSERT_EQ(1u, visible_after_retry.size());
    EXPECT_NE(std::string::npos, visible_after_retry.front().find("source=deduplicated"));
}

TEST_F(CacheManagerTest, TestReportEventRejectsDuplicateSpecNamesWithinSnapshotBlock) {
    const std::string host = "192.168.10.36:8080";
    const int64_t key = 9434;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    const auto [baseline_ec, baseline] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "baseline"}}), "duplicate_spec_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    const std::string baseline_token = baseline.committed_snapshot_version();

    for (const bool same_uri : {true, false}) {
        SCOPED_TRACE(same_uri ? "same spec name and URI" : "same spec name with conflicting URIs");
        auto duplicate = MakeSnapshotRequest(host, {});
        auto *block = duplicate.mutable_events(0)->mutable_block_snapshot()->add_blocks();
        block->set_block_key(std::to_string(key));
        block->set_medium("mem");
        for (int i = 0; i < 2; ++i) {
            auto *spec = block->add_specs();
            spec->set_name("tp0");
            const std::string source = same_uri || i == 0 ? "duplicate_a" : "duplicate_b";
            spec->set_uri("event_report://" + host + "/mem?source=" + source);
        }

        const auto [duplicate_ec, duplicate_response] =
            CallReportEvent(duplicate, same_uri ? "duplicate_spec_same_uri" : "duplicate_spec_conflicting_uri");
        EXPECT_EQ(EC_PARTIAL_OK, duplicate_ec);
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, duplicate_response.header().status().code());
        EXPECT_EQ(baseline_token, duplicate_response.committed_snapshot_version());
        EXPECT_FALSE(duplicate_response.snapshot_required());
        EXPECT_EQ(baseline_token, event_backend->GetSnapshotVersion({"test_instance", host}));
        EXPECT_EQ(1u, QueryRawEventReportUris(key).size());
    }

    const auto visible = QueryEventReportUris({key});
    ASSERT_EQ(1u, visible.size());
    EXPECT_NE(std::string::npos, visible.front().find("source=baseline"));
    EXPECT_NE(std::string::npos, visible.front().find("s_version=" + baseline_token));
}

TEST_F(CacheManagerTest, TestReportEventStoresSameSpecAcrossMediaAndDeduplicatesQueryByName) {
    const std::string host = "192.168.10.37:8080";
    const int64_t key = 9435;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem", "disk"}));

    auto snapshot_request = MakeSnapshotRequest(host, {});
    auto *snapshot = snapshot_request.mutable_events(0)->mutable_block_snapshot();
    for (const auto &medium : {"mem", "disk"}) {
        auto *block = snapshot->add_blocks();
        block->set_block_key(std::to_string(key));
        block->set_medium(medium);
        auto *spec = block->add_specs();
        spec->set_name("tp0");
        spec->set_uri("event_report://" + host + "/" + medium + "?source=" + medium);
    }

    const auto [snapshot_ec, response] = CallReportEvent(snapshot_request, "same_key_spec_across_media");
    ASSERT_EQ(EC_OK, snapshot_ec);
    const std::string token = response.committed_snapshot_version();
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(token));

    const auto raw = QueryRawEventReportUris(key);
    ASSERT_EQ(2u, raw.size());
    bool stored_mem = false;
    bool stored_disk = false;
    for (const auto &uri : raw) {
        EXPECT_NE(std::string::npos, uri.find("s_version=" + token));
        stored_mem = stored_mem || uri.find("/mem?") != std::string::npos;
        stored_disk = stored_disk || uri.find("/disk?") != std::string::npos;
    }
    EXPECT_TRUE(stored_mem);
    EXPECT_TRUE(stored_disk);

    // The query path merges locations from the same backend and exposes one
    // value per spec name. The two media are alternative physical copies, not
    // two logical tp0 components.
    const auto visible = QueryEventReportUris({key});
    ASSERT_EQ(1u, visible.size());
    EXPECT_NE(std::string::npos, visible.front().find("s_version=" + token));
    EXPECT_TRUE(visible.front().find("/mem?") != std::string::npos ||
                visible.front().find("/disk?") != std::string::npos);
}

TEST_F(CacheManagerTest, TestReportEventRepeatedPhysicalDeltaUsesSetNotReferenceCountSemantics) {
    const std::string host = "192.168.10.38:8080";
    const int64_t key = 9436;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    const auto [baseline_ec, baseline] = CallReportEvent(MakeSnapshotRequest(host, {}), "physical_delta_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    const std::string token = baseline.committed_snapshot_version();

    const auto [first_add_ec, first_add] =
        CallReportEvent(MakeAddRequest(host, key, "physical_0"), "physical_delta_first_add");
    ASSERT_EQ(EC_OK, first_add_ec);
    EXPECT_EQ(token, first_add.committed_snapshot_version());

    const auto [second_add_ec, second_add] =
        CallReportEvent(MakeAddRequest(host, key, "physical_1"), "physical_delta_second_add");
    ASSERT_EQ(EC_OK, second_add_ec);
    EXPECT_EQ(token, second_add.committed_snapshot_version());
    auto visible = QueryEventReportUris({key});
    ASSERT_EQ(1u, visible.size());
    EXPECT_NE(std::string::npos, visible.front().find("source=physical_1"));
    EXPECT_EQ(1u, QueryRawEventReportUris(key).size());

    auto delete_request = MakeAddRequest(host, key, "unused");
    delete_request.clear_events();
    auto *delete_event = delete_request.add_events();
    delete_event->set_event_type(proto::meta::EVENT_BLOCK_DELETE);
    delete_event->mutable_block_delete()->set_block_key(std::to_string(key));
    delete_event->mutable_block_delete()->set_medium("mem");
    delete_event->mutable_block_delete()->add_spec_names("tp0");
    const auto [delete_ec, deletion] = CallReportEvent(delete_request, "physical_delta_single_delete");
    ASSERT_EQ(EC_OK, delete_ec);
    EXPECT_EQ(token, deletion.committed_snapshot_version());
    EXPECT_TRUE(QueryEventReportUris({key}).empty());
    EXPECT_TRUE(QueryRawEventReportUris(key).empty());
}

TEST_F(CacheManagerTest, TestReportEventSnapshotCommitReclaimsOnlyStaleReporterLocations) {
    const std::string host_a = "192.168.10.4:8080";
    const std::string host_b = "192.168.10.5:8080";
    const int64_t stale_key = 9430;
    const int64_t current_key = 9431;
    const int64_t other_host_key = 9432;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host_a, {"mem"}));
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host_b, {"mem"}));

    const auto [host_a_baseline_ec, host_a_baseline] =
        CallReportEvent(MakeSnapshotRequest(host_a, {{stale_key, "host_a_stale"}, {current_key, "host_a_old_current"}}),
                        "reclaimer_host_a_baseline");
    ASSERT_EQ(EC_OK, host_a_baseline_ec);
    const std::string host_a_old_token = host_a_baseline.committed_snapshot_version();

    const auto [host_b_baseline_ec, host_b_baseline] =
        CallReportEvent(MakeSnapshotRequest(host_b, {{other_host_key, "host_b_current"}}), "reclaimer_host_b_baseline");
    ASSERT_EQ(EC_OK, host_b_baseline_ec);
    const std::string host_b_token = host_b_baseline.committed_snapshot_version();
    ASSERT_NE(host_a_old_token, host_b_token);

    ASSERT_EQ(1u, QueryRawEventReportUris(stale_key).size());
    ASSERT_EQ(1u, QueryRawEventReportUris(current_key).size());
    ASSERT_EQ(1u, QueryRawEventReportUris(other_host_key).size());

    const auto [reconcile_ec, reconcile] = CallReportEvent(
        MakeSnapshotRequest(host_a, {{current_key, "host_a_new_current"}}), "reclaimer_host_a_reconcile");
    ASSERT_EQ(EC_OK, reconcile_ec);
    const std::string host_a_new_token = reconcile.committed_snapshot_version();
    ASSERT_NE(host_a_old_token, host_a_new_token);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline && !QueryRawEventReportUris(stale_key).empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_TRUE(QueryRawEventReportUris(stale_key).empty());
    EXPECT_TRUE(QueryEventReportUris({stale_key}).empty());

    // The same cleanup scan must preserve the reporter's current token and
    // every location belonging to a different reporter.
    const auto current_uris = QueryRawEventReportUris(current_key);
    ASSERT_EQ(1u, current_uris.size());
    EXPECT_NE(std::string::npos, current_uris.front().find("source=host_a_new_current"));
    EXPECT_NE(std::string::npos, current_uris.front().find("s_version=" + host_a_new_token));
    const auto other_host_uris = QueryRawEventReportUris(other_host_key);
    ASSERT_EQ(1u, other_host_uris.size());
    EXPECT_NE(std::string::npos, other_host_uris.front().find("source=host_b_current"));
    EXPECT_NE(std::string::npos, other_host_uris.front().find("s_version=" + host_b_token));

    // Re-scanning after the stale location has already gone is idempotent.
    EXPECT_EQ(EC_OK,
              cache_manager_->CleanupStaleSnapshotLocations({"test_instance", host_a},
                                                            host_a_new_token,
                                                            DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                                                            event_backend));
    EXPECT_TRUE(QueryRawEventReportUris(stale_key).empty());
    EXPECT_EQ(1u, QueryRawEventReportUris(current_key).size());
    EXPECT_EQ(1u, QueryRawEventReportUris(other_host_key).size());
}

TEST_F(CacheManagerTest, TestSnapshotCleanupPreservesPostCommitDeltaOnMixedGenerationLocation) {
    const std::string host = "192.168.10.33:8080";
    const int64_t key = 9434;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    auto baseline_request = MakeSnapshotRequest(host, {});
    auto *block = baseline_request.mutable_events(0)->mutable_block_snapshot()->add_blocks();
    block->set_block_key(std::to_string(key));
    block->set_medium("mem");
    auto *tp0 = block->add_specs();
    tp0->set_name("tp0");
    tp0->set_uri("event_report://" + host + "/mem?source=old_tp0");
    auto *tp1 = block->add_specs();
    tp1->set_name("tp1");
    tp1->set_uri("event_report://" + host + "/mem?source=old_tp1");

    const auto [baseline_ec, baseline] = CallReportEvent(baseline_request, "mixed_generation_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    const std::string old_token = baseline.committed_snapshot_version();

    // Publish the next complete generation without touching this block. This
    // models the interval after commit but before the asynchronous cleanup
    // scan has reclaimed an omitted location.
    std::string current_token;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, event_backend->BeginSnapshot({"test_instance", host}, current_token, retry_after_ms));
    ASSERT_NE(old_token, current_token);
    ASSERT_TRUE(event_backend->CommitSnapshotVersion({"test_instance", host}, current_token));

    // A post-commit delta refreshes one spec in the stable location. The other
    // spec legitimately retains its older reconciliation tag.
    const auto [delta_ec, delta] = CallReportEvent(MakeAddRequest(host, key, "new_tp0"), "mixed_generation_delta");
    ASSERT_EQ(EC_OK, delta_ec);
    ASSERT_EQ(current_token, delta.committed_snapshot_version());
    auto before_cleanup = QueryRawEventReportUris(key);
    ASSERT_EQ(2u, before_cleanup.size());

    // Cleanup is location-granular. It must not delete the whole location just
    // because one sibling spec still has the previous generation.
    ASSERT_EQ(
        EC_OK,
        cache_manager_->CleanupStaleSnapshotLocations(
            {"test_instance", host}, current_token, DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, event_backend));
    auto cleanup_latency = metrics_registry_->GetMetricsData("event_report.snapshot_cleanup_scan_latency_ms");
    ASSERT_NE(nullptr, cleanup_latency);
    const MetricsTags cleanup_tags = {
        {"instance_id", "test_instance"},
        {"host", host},
        {"type", "event_report_l2"},
    };
    EXPECT_GE(cleanup_latency->GetOrCreateGauge(cleanup_tags).Get(), 0.0);

    const auto after_cleanup = QueryRawEventReportUris(key);
    ASSERT_EQ(2u, after_cleanup.size());
    EXPECT_EQ(1u, std::count_if(after_cleanup.begin(), after_cleanup.end(), [](const std::string &uri) {
                  return uri.find("source=new_tp0") != std::string::npos;
              }));
    EXPECT_EQ(1u, std::count_if(after_cleanup.begin(), after_cleanup.end(), [](const std::string &uri) {
                  return uri.find("source=old_tp1") != std::string::npos;
              }));
    EXPECT_EQ(2u, QueryEventReportUris({key}).size());
}

TEST_F(CacheManagerTest, TestSnapshotCleanupPreservesCurrentDeltaBesideLegacySpec) {
    const std::string host = "192.168.10.32:8080";
    const int64_t key = 9433;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    std::string current_token;
    ASSERT_EQ(EC_OK, event_backend->BeginDeltaMutation({"test_instance", host}, current_token));
    event_backend->EndDeltaMutation({"test_instance", host});

    std::string current_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://" + host + "/mem?source=current", current_token, current_uri));
    const std::string legacy_uri = "event_report://" + host + "/mem?source=legacy";

    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_NE(nullptr, meta_searcher);
    std::vector<ErrorCode> merge_results;
    ASSERT_EQ(EC_OK,
              meta_searcher->BatchMergeLocationSpecs(request_context_.get(),
                                                     {key},
                                                     {{{event_backend->BuildLocationId("mem", host),
                                                        DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                                                        CacheLocationStatus::CLS_SERVING,
                                                        {LocationSpec("tp1", legacy_uri)}}}},
                                                     merge_results));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), merge_results);
    ASSERT_EQ(EC_OK,
              meta_searcher->BatchMergeLocationSpecs(request_context_.get(),
                                                     {key},
                                                     {{{event_backend->BuildLocationId("mem", host),
                                                        DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                                                        CacheLocationStatus::CLS_SERVING,
                                                        {LocationSpec("tp0", current_uri)}}}},
                                                     merge_results));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), merge_results);

    ASSERT_EQ(
        EC_OK,
        cache_manager_->CleanupStaleSnapshotLocations(
            {"test_instance", host}, current_token, DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, event_backend));
    EXPECT_EQ(2u, QueryRawEventReportUris(key).size());
    EXPECT_EQ(2u, QueryEventReportUris({key}).size());
}

TEST_F(CacheManagerTest, TestSnapshotCleanupPreservesInFlightStableLocationUntilAbort) {
    const std::string host = "192.168.10.34:8080";
    const int64_t key = 9435;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    const auto [baseline_ec, baseline] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "baseline"}}), "cleanup_in_flight_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    const std::string committed = baseline.committed_snapshot_version();

    std::string in_flight;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, event_backend->BeginSnapshot({"test_instance", host}, in_flight, retry_after_ms));
    ASSERT_NE(committed, in_flight);
    std::string in_flight_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://" + host + "/mem?source=in_flight", in_flight, in_flight_uri));

    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_NE(nullptr, meta_searcher);
    std::vector<ErrorCode> replace_results;
    ASSERT_EQ(EC_OK,
              meta_searcher->BatchReplaceLocationSpecs(request_context_.get(),
                                                       {key},
                                                       {{{event_backend->BuildLocationId("mem", host),
                                                          DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                                                          CacheLocationStatus::CLS_SERVING,
                                                          {LocationSpec("tp0", in_flight_uri)}}}},
                                                       replace_results));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), replace_results);
    ASSERT_EQ(1u, QueryRawEventReportUris(key).size());
    EXPECT_TRUE(QueryEventReportUris({key}).empty());

    ASSERT_EQ(
        EC_OK,
        cache_manager_->CleanupStaleSnapshotLocations(
            {"test_instance", host}, committed, DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, event_backend));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ASSERT_EQ(1u, QueryRawEventReportUris(key).size());

    event_backend->AbortSnapshotVersion({"test_instance", host}, in_flight);
    const auto failed_candidate_visible = QueryEventReportUris({key});
    ASSERT_EQ(1u, failed_candidate_visible.size());
    EXPECT_NE(std::string::npos, failed_candidate_visible.front().find("source=in_flight"));
    ASSERT_EQ(
        EC_OK,
        cache_manager_->CleanupStaleSnapshotLocations(
            {"test_instance", host}, committed, DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, event_backend));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline && !QueryRawEventReportUris(key).empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_TRUE(QueryRawEventReportUris(key).empty());
}

TEST_F(CacheManagerTest, TestOldSnapshotCleanupDoesNotDeleteLaterAbortedAttemptWrites) {
    const std::string host = "192.168.10.36:8080";
    const int64_t key = 9437;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    const auto [baseline_ec, baseline] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "baseline"}}), "cleanup_attempt_epoch_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    const ReporterSnapshotKey reporter_key{"test_instance", host};
    const std::string committed = baseline.committed_snapshot_version();
    const uint64_t cleanup_attempt_epoch = event_backend->GetSnapshotAttemptEpoch(reporter_key);
    ASSERT_GT(cleanup_attempt_epoch, 0u);

    std::string failed_attempt;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, event_backend->BeginSnapshot(reporter_key, failed_attempt, retry_after_ms));
    ASSERT_GT(event_backend->GetSnapshotAttemptEpoch(reporter_key), cleanup_attempt_epoch);

    std::string failed_attempt_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://" + host + "/mem?source=partial_failed_attempt", failed_attempt, failed_attempt_uri));
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_NE(nullptr, meta_searcher);
    std::vector<ErrorCode> replace_results;
    ASSERT_EQ(EC_OK,
              meta_searcher->BatchReplaceLocationSpecs(request_context_.get(),
                                                       {key},
                                                       {{{event_backend->BuildLocationId("mem", host),
                                                          DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                                                          CacheLocationStatus::CLS_SERVING,
                                                          {LocationSpec("tp0", failed_attempt_uri)}}}},
                                                       replace_results));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), replace_results);
    event_backend->AbortSnapshotVersion(reporter_key, failed_attempt);

    // The cleanup task belongs to the baseline attempt. Even though the later
    // attempt has aborted and no longer has an in-flight token, its epoch must
    // permanently fence this older cleanup from deleting the partial write.
    ASSERT_EQ(EC_OK,
              cache_manager_->CleanupStaleSnapshotLocations(reporter_key,
                                                            committed,
                                                            DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                                                            event_backend,
                                                            cleanup_attempt_epoch));
    const auto uris = QueryRawEventReportUris(key);
    ASSERT_EQ(1u, uris.size());
    EXPECT_NE(std::string::npos, uris.front().find("source=partial_failed_attempt"));
}

TEST_F(CacheManagerTest, TestSnapshotCleanupCASPreservesLocationRefreshedAfterScan) {
    const std::string host = "192.168.10.35:8080";
    const int64_t key = 9436;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    const auto [baseline_ec, baseline] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "baseline"}}), "cleanup_cas_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    const std::string baseline_token = baseline.committed_snapshot_version();

    // Publish a new authoritative token without touching this key, leaving the
    // baseline location stale for the cleanup scan below.
    std::string cleanup_token;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, event_backend->BeginSnapshot({"test_instance", host}, cleanup_token, retry_after_ms));
    ASSERT_NE(baseline_token, cleanup_token);
    ASSERT_TRUE(event_backend->CommitSnapshotVersion({"test_instance", host}, cleanup_token));

    std::vector<int64_t> captured_keys;
    std::vector<std::vector<std::string>> captured_location_ids;
    std::vector<std::vector<std::string>> captured_expected_values;
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer("test_instance");
    ASSERT_NE(nullptr, indexer);
    MetaSearcher paused_cleanup_searcher(
        indexer,
        [](const CacheLocation &) { return true; },
        [&](const std::vector<int64_t> &keys,
            const std::vector<std::vector<std::string>> &location_ids,
            const std::vector<std::vector<std::string>> &expected_values,
            bool metadata_only) {
            EXPECT_TRUE(metadata_only);
            captured_keys = keys;
            captured_location_ids = location_ids;
            captured_expected_values = expected_values;
        });
    ASSERT_EQ(EC_OK,
              paused_cleanup_searcher.CleanupLocationsByPredicate(
                  request_context_.get(),
                  DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                  1,
                  [&](int64_t block_key, const std::string &location_id, const CacheLocation &location) {
                      if (block_key != key || location_id != event_backend->BuildLocationId("mem", host)) {
                          return false;
                      }
                      SnapshotUriInfo info;
                      return location.location_specs().size() == 1 &&
                             SnapshotUriUtils::ParseSnapshotUriInfo(location.location_specs().front().uri(), info) &&
                             info.version == baseline_token;
                  }));
    ASSERT_EQ((std::vector<int64_t>{key}), captured_keys);
    ASSERT_EQ(1u, captured_location_ids.size());
    ASSERT_EQ(1u, captured_location_ids.front().size());
    ASSERT_EQ(1u, captured_expected_values.size());
    ASSERT_EQ(1u, captured_expected_values.front().size());

    // Before the paused cleanup reaches its conditional delete, the next
    // snapshot refreshes the same stable location with a newer token.
    std::string refreshed_token;
    ASSERT_EQ(EC_OK, event_backend->BeginSnapshot({"test_instance", host}, refreshed_token, retry_after_ms));
    ASSERT_NE(cleanup_token, refreshed_token);
    std::string refreshed_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://" + host + "/mem?source=refreshed", refreshed_token, refreshed_uri));
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_NE(nullptr, meta_searcher);
    std::vector<ErrorCode> replace_results;
    ASSERT_EQ(EC_OK,
              meta_searcher->BatchReplaceLocationSpecs(request_context_.get(),
                                                       {key},
                                                       {{{event_backend->BuildLocationId("mem", host),
                                                          DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                                                          CacheLocationStatus::CLS_SERVING,
                                                          {LocationSpec("tp0", refreshed_uri)}}}},
                                                       replace_results));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), replace_results);
    ASSERT_TRUE(event_backend->CommitSnapshotVersion({"test_instance", host}, refreshed_token));

    CacheLocationDelRequest stale_cleanup_request{
        .instance_id = "test_instance",
        .block_keys = captured_keys,
        .location_ids = captured_location_ids,
        .delay = std::chrono::seconds(0),
        .expected_location_values = captured_expected_values,
        .metadata_only = true,
    };
    const auto cleanup_result = cache_manager_->schedule_plan_executor_->Submit(stale_cleanup_request).get();
    ASSERT_EQ(EC_OK, cleanup_result.status);

    const auto raw_uris = QueryRawEventReportUris(key);
    ASSERT_EQ(1u, raw_uris.size());
    EXPECT_NE(std::string::npos, raw_uris.front().find("source=refreshed"));
    EXPECT_NE(std::string::npos, raw_uris.front().find("s_version=" + refreshed_token));
    ASSERT_EQ(1u, QueryEventReportUris({key}).size());
}

TEST_F(CacheManagerTest, TestReportEventEmptySnapshotCommitsAndReclaimsPreviousBlocks) {
    const std::string host = "192.168.10.6:8080";
    const int64_t key = 9440;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    const auto [baseline_ec, baseline] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "baseline"}}), "empty_snapshot_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(baseline.committed_snapshot_version()));
    ASSERT_EQ(1u, QueryEventReportUris({key}).size());

    const auto [empty_ec, empty_snapshot] = CallReportEvent(MakeSnapshotRequest(host, {}), "empty_snapshot_commit");
    ASSERT_EQ(EC_OK, empty_ec);
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(empty_snapshot.committed_snapshot_version()));
    EXPECT_NE(baseline.committed_snapshot_version(), empty_snapshot.committed_snapshot_version());

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline && !QueryRawEventReportUris(key).empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_TRUE(QueryRawEventReportUris(key).empty());
    EXPECT_TRUE(QueryEventReportUris({key}).empty());
}

TEST_F(CacheManagerTest, TestSuccessfulSnapshotImmediatelyFencesOmittedOldVersionBeforeCleanup) {
    const std::string host = "192.168.10.60:8080";
    const int64_t old_key = 94'460;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    const auto [baseline_ec, baseline] =
        CallReportEvent(MakeSnapshotRequest(host, {{old_key, "baseline"}}), "visibility_fence_baseline");
    ASSERT_EQ(EC_OK, baseline_ec);
    ASSERT_EQ(1u, QueryRawEventReportUris(old_key).size());
    ASSERT_EQ(1u, QueryEventReportUris({old_key}).size());

    // Keep the old metadata physically present so this test verifies query
    // fencing rather than the timing of asynchronous cleanup.
    auto saved_executor = cache_manager_->schedule_plan_executor_;
    cache_manager_->schedule_plan_executor_.reset();
    const auto [empty_ec, empty_snapshot] =
        CallReportEvent(MakeSnapshotRequest(host, {}), "visibility_fence_empty_snapshot");
    cache_manager_->schedule_plan_executor_ = std::move(saved_executor);

    ASSERT_EQ(EC_OK, empty_ec);
    ASSERT_NE(baseline.committed_snapshot_version(), empty_snapshot.committed_snapshot_version());
    ASSERT_EQ(1u, QueryRawEventReportUris(old_key).size());
    EXPECT_TRUE(QueryEventReportUris({old_key}).empty());
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_EventReportFallbackLookup) {
    // Event report URI hostname is a node IP, not the global_unique_name.
    // The functor should look up the backend via event_report_storage_candidates.
    const std::string instance_id = "my_cluster";
    const std::string instance_group = "my_group";
    const std::string event_report_storage_name = "event_report_" + instance_group;
    const std::string node_host = "192.168.1.100:8080";

    // Register instance so GetInstanceGroupName works
    auto instance_info = std::make_shared<InstanceInfo>(
        "test_quota_group", instance_group, instance_id, 64, createLocationSpecInfos(), createModelDeployment());
    registry_manager_->instance_infos_[instance_id] = instance_info;

    // Create InstanceGroup with event_report_storage_candidates
    auto ig = std::make_shared<InstanceGroup>();
    ig->set_name(instance_group);
    ig->set_storage_candidates({event_report_storage_name});
    ig->set_event_report_storage_candidates({event_report_storage_name});
    ig->set_global_quota_group_name("test_quota_group");
    ig->set_max_instance_count(10);
    ig->set_version(1);
    registry_manager_->instance_group_configs_[instance_group] = ig;

    auto metrics_registry = cache_manager_->metrics_registry_;
    auto event_report_backend = std::make_shared<EventReportBackend>(metrics_registry);

    StorageConfig config;
    config.set_global_unique_name(event_report_storage_name);
    config.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5);
    auto spec = std::make_shared<EventReportStorageSpec>();
    config.set_storage_spec(spec);
    event_report_backend->Open(config, "test_trace");

    event_report_backend->RegisterNode(instance_id, node_host, {"mem"});

    auto dsm = registry_manager_->data_storage_manager_;
    dsm->storage_map_[event_report_storage_name] = event_report_backend;

    auto func = cache_manager_->GetCheckLocDataExistFunc(instance_id);

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5);
    loc.set_location_specs({LocationSpec("tp0", "event_report://192.168.1.100:8080/mem?gpu=A100")});
    ASSERT_EQ(func(loc), false);

    dsm->storage_map_.erase(event_report_storage_name);
    registry_manager_->instance_group_configs_.erase(instance_group);
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFuncFencesVersionsAfterSuccessfulSnapshot) {
    const std::string host = "192.168.10.35:8080";
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    const std::string unknown_token = "ffffffffffffffffffffffffffffffff";
    const std::string legacy_uri = "event_report://physical-cache:9600/mem?source=legacy";
    std::string unknown_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://physical-cache:9600/mem?source=unknown", unknown_token, unknown_uri));

    CacheLocation location;
    location.set_id(event_backend->BuildLocationId("mem", host));
    location.set_status(CLS_SERVING);
    location.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2);
    auto check = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    // Before this process has committed a complete snapshot, queries remain
    // soft so restart recovery and realtime-only reporters keep working.
    location.set_location_specs({LocationSpec("tp0", legacy_uri)});
    EXPECT_TRUE(check(location));
    location.set_location_specs({LocationSpec("tp0", unknown_uri)});
    EXPECT_TRUE(check(location));

    const auto [snapshot_ec, snapshot_response] =
        CallReportEvent(MakeSnapshotRequest(host, {}), "query_filter_baseline");
    ASSERT_EQ(EC_OK, snapshot_ec);
    const std::string token = snapshot_response.committed_snapshot_version();
    std::string current_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://physical-cache:9600/mem?source=current", token, current_uri));

    location.set_location_specs({LocationSpec("tp0", current_uri), LocationSpec("tp1", current_uri)});
    EXPECT_TRUE(check(location));
    location.set_location_specs({LocationSpec("tp0", current_uri), LocationSpec("tp1", unknown_uri)});
    EXPECT_TRUE(check(location));
    location.set_location_specs({LocationSpec("tp0", current_uri), LocationSpec("tp1", legacy_uri)});
    EXPECT_TRUE(check(location));
    location.set_location_specs({LocationSpec("tp0", legacy_uri)});
    EXPECT_FALSE(check(location));
    location.set_location_specs({LocationSpec("tp0", unknown_uri)});
    EXPECT_FALSE(check(location));
    location.set_location_specs(
        {LocationSpec("tp0", current_uri), LocationSpec("tp1", "event_report://raw/mem?s_version=bad")});
    EXPECT_FALSE(check(location));
    location.set_location_specs({});
    EXPECT_FALSE(check(location));

    // During an admitted snapshot, strict mode continues to recognize only
    // the last committed token. Candidate-only metadata becomes visible on
    // commit, or through soft fallback if the attempt fails.
    const ReporterSnapshotKey reporter_key{"test_instance", host};
    std::string candidate;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, event_backend->BeginSnapshot(reporter_key, candidate, retry_after_ms));
    std::string candidate_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://physical-cache:9600/mem?source=candidate", candidate, candidate_uri));
    location.set_location_specs({LocationSpec("tp0", current_uri)});
    EXPECT_TRUE(check(location));
    location.set_location_specs({LocationSpec("tp0", candidate_uri)});
    EXPECT_FALSE(check(location));
    location.set_location_specs({LocationSpec("tp0", current_uri), LocationSpec("tp1", candidate_uri)});
    EXPECT_TRUE(check(location));
    location.set_location_specs({LocationSpec("tp0", unknown_uri)});
    EXPECT_FALSE(check(location));

    // An admitted snapshot failure falls back to soft visibility because
    // candidate writes may already have replaced committed metadata in place.
    event_backend->AbortSnapshotVersion(reporter_key, candidate);
    location.set_location_specs({LocationSpec("tp0", unknown_uri)});
    EXPECT_TRUE(check(location));
    location.set_location_specs({LocationSpec("tp0", legacy_uri)});
    EXPECT_TRUE(check(location));

    // The next successful complete snapshot restores strict visibility and
    // immediately fences every older generation.
    std::string recovered;
    ASSERT_EQ(EC_OK, event_backend->BeginSnapshot(reporter_key, recovered, retry_after_ms));
    ASSERT_TRUE(event_backend->CommitSnapshotVersion(reporter_key, recovered));
    std::string recovered_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://physical-cache:9600/mem?source=recovered", recovered, recovered_uri));
    location.set_location_specs({LocationSpec("tp0", recovered_uri)});
    EXPECT_TRUE(check(location));
    location.set_location_specs({LocationSpec("tp0", current_uri)});
    EXPECT_FALSE(check(location));
    location.set_location_specs({LocationSpec("tp0", candidate_uri)});
    EXPECT_FALSE(check(location));
    location.set_location_specs({LocationSpec("tp0", unknown_uri)});
    EXPECT_FALSE(check(location));

    event_backend->SetNodeUnavailable("test_instance", host);
    location.set_location_specs({LocationSpec("tp0", recovered_uri)});
    EXPECT_FALSE(check(location));
    ASSERT_EQ(EC_OK, event_backend->OnHeartbeat("test_instance", host, {}));
    EXPECT_TRUE(check(location));
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFuncEventReportUriValidationMatrix) {
    const std::string instance_id = "test_instance";
    const std::string host = "192.168.10.37:8080";
    const std::string token = "0123456789abcdef0123456789abcdef";
    const std::string upper_token = "ABCDEF0123456789ABCDEF0123456789";
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode(instance_id, host, {"mem"}));

    CacheLocation location;
    location.set_id(event_backend->BuildLocationId("mem", host));
    location.set_status(CLS_SERVING);
    location.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2);
    const auto check = cache_manager_->GetCheckLocDataExistFunc(instance_id);

    struct Case {
        const char *name;
        std::vector<std::string> uris;
        bool readable;
    };
    const std::vector<Case> cases{
        {"empty_specs", {}, false},
        {"legacy_without_query", {"event_report://physical-cache:9600/mem"}, true},
        {"legacy_with_other_params", {"event_report://physical-cache:9600/mem?size=4096&offset=0"}, true},
        {"valid_lowercase_version", {"event_report://physical-cache:9600/mem?s_version=" + token}, true},
        {"valid_uppercase_version", {"event_report://physical-cache:9600/mem?s_version=" + upper_token}, true},
        {"near_match_param", {"event_report://physical-cache:9600/mem?xs_version=bad&s_version_suffix=bad"}, true},
        {"empty_version", {"event_report://physical-cache:9600/mem?s_version="}, false},
        {"version_without_equals", {"event_report://physical-cache:9600/mem?s_version"}, false},
        {"short_version", {"event_report://physical-cache:9600/mem?s_version=" + std::string(31, 'a')}, false},
        {"long_version", {"event_report://physical-cache:9600/mem?s_version=" + std::string(33, 'a')}, false},
        {"non_hex_version",
         {"event_report://physical-cache:9600/mem?s_version=0123456789abcdef0123456789abcdeg"},
         false},
        {"duplicate_same_version",
         {"event_report://physical-cache:9600/mem?s_version=" + token + "&s_version=" + token},
         false},
        {"duplicate_different_version",
         {"event_report://physical-cache:9600/mem?s_version=" + token + "&s_version=" + upper_token},
         false},
        {"invalid_uri", {"not-a-uri"}, false},
        {"mixed_versioned_and_legacy",
         {"event_report://physical-cache:9600/mem?s_version=" + token,
          "event_report://physical-cache:9600/mem?source=legacy"},
         true},
        {"mixed_valid_and_malformed",
         {"event_report://physical-cache:9600/mem?s_version=" + token,
          "event_report://physical-cache:9600/mem?s_version=bad"},
         false},
        {"mixed_valid_and_duplicate",
         {"event_report://physical-cache:9600/mem?s_version=" + token,
          "event_report://physical-cache:9600/mem?s_version=" + token + "&s_version=" + token},
         false},
    };

    for (const auto &test_case : cases) {
        SCOPED_TRACE(test_case.name);
        std::vector<LocationSpec> specs;
        specs.reserve(test_case.uris.size());
        for (size_t i = 0; i < test_case.uris.size(); ++i) {
            specs.emplace_back("tp" + std::to_string(i), test_case.uris[i]);
        }
        location.set_location_specs(std::move(specs));
        EXPECT_EQ(test_case.readable, check(location));
    }

    location.set_location_specs({LocationSpec("tp0", "event_report://physical-cache:9600/mem?s_version=" + token)});
    location.set_id("kvs#event_report_l2#mem");
    EXPECT_FALSE(check(location));
    location.set_id("kvs#event_report_l1p5#mem#" + host);
    EXPECT_FALSE(check(location));
    location.set_id(event_backend->BuildLocationId("mem", "192.168.10.38:8080"));
    EXPECT_FALSE(check(location));
}

TEST_F(CacheManagerTest, TestGetCacheLocationEnforcesReporterLifecycleAndBatchOrdering) {
    const std::string host_a = "192.168.10.41:8080";
    const std::string host_b = "192.168.10.42:8080";
    const int64_t current_key = 94'410;
    const int64_t old_key = 94'411;
    const int64_t down_key = 94'412;
    const int64_t missing_key = 94'413;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host_a, {"mem"}));
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host_b, {"mem"}));

    const auto [first_a_ec, first_a] = CallReportEvent(
        MakeSnapshotRequest(host_a, {{current_key, "a_first"}, {old_key, "a_old"}}), "query_lifecycle_a_first");
    ASSERT_EQ(EC_OK, first_a_ec);
    const std::string first_a_token = first_a.committed_snapshot_version();
    const auto [first_b_ec, first_b] =
        CallReportEvent(MakeSnapshotRequest(host_b, {{down_key, "b_current"}}), "query_lifecycle_b_first");
    ASSERT_EQ(EC_OK, first_b_ec);
    const std::string first_b_token = first_b.committed_snapshot_version();

    auto query_visibility = [&](const std::vector<int64_t> &keys) {
        RequestContext context("query_reporter_lifecycle");
        auto [ec, locations] = cache_manager_->GetCacheLocation(
            &context, "test_instance", CacheManager::QueryType::QT_BATCH_GET, keys, {}, BlockMask{}, 0, {});
        EXPECT_EQ(EC_OK, ec);
        std::vector<std::vector<std::string>> uris_by_key;
        for (const auto &location : locations.cache_locations_view()) {
            auto &uris = uris_by_key.emplace_back();
            for (const auto &spec : location.location_specs()) {
                if (spec.uri().rfind("event_report://", 0) == 0) {
                    uris.push_back(spec.uri());
                }
            }
        }
        return uris_by_key;
    };

    // Positive baseline first: all three committed-token locations are
    // visible through the real GetCacheLocation entry point.
    auto visible = query_visibility({current_key, old_key, down_key, missing_key});
    ASSERT_EQ(4u, visible.size());
    ASSERT_EQ(1u, visible[0].size());
    ASSERT_EQ(1u, visible[1].size());
    ASSERT_EQ(1u, visible[2].size());
    EXPECT_TRUE(visible[3].empty());
    EXPECT_NE(std::string::npos, visible[0][0].find("s_version=" + first_a_token));
    EXPECT_NE(std::string::npos, visible[2][0].find("s_version=" + first_b_token));

    // A successful snapshot immediately fences omitted old versions; metadata
    // reclamation remains asynchronous and does not affect query correctness.
    const auto [second_a_ec, second_a] =
        CallReportEvent(MakeSnapshotRequest(host_a, {{current_key, "a_second"}}), "query_lifecycle_a_second");
    ASSERT_EQ(EC_OK, second_a_ec);
    const std::string second_a_token = second_a.committed_snapshot_version();
    ASSERT_NE(first_a_token, second_a_token);
    visible = query_visibility({current_key, old_key, down_key, missing_key});
    ASSERT_EQ(4u, visible.size());
    ASSERT_EQ(1u, visible[0].size());
    EXPECT_TRUE(visible[1].empty());
    ASSERT_EQ(1u, visible[2].size());
    EXPECT_TRUE(visible[3].empty());
    const auto cleanup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < cleanup_deadline && !QueryRawEventReportUris(old_key).empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(QueryRawEventReportUris(old_key).empty());
    visible = query_visibility({current_key, old_key, down_key, missing_key});
    ASSERT_EQ(4u, visible.size());
    ASSERT_EQ(1u, visible[0].size());
    EXPECT_TRUE(visible[1].empty());
    ASSERT_EQ(1u, visible[2].size());
    EXPECT_TRUE(visible[3].empty());
    EXPECT_NE(std::string::npos, visible[0][0].find("s_version=" + second_a_token));

    // Only host B is hidden; host A and the batch result ordering are
    // unaffected. Heartbeat recovery restores B's original committed token.
    event_backend->SetNodeUnavailable("test_instance", host_b);
    visible = query_visibility({current_key, old_key, down_key, missing_key});
    ASSERT_EQ(4u, visible.size());
    ASSERT_EQ(1u, visible[0].size());
    EXPECT_TRUE(visible[1].empty());
    EXPECT_TRUE(visible[2].empty());
    EXPECT_TRUE(visible[3].empty());
    ASSERT_EQ(EC_OK, event_backend->OnHeartbeat("test_instance", host_b, {}));
    visible = query_visibility({current_key, old_key, down_key, missing_key});
    ASSERT_EQ(1u, visible[2].size());
    EXPECT_NE(std::string::npos, visible[2][0].find("s_version=" + first_b_token));

    // Same-process unregister creates a tombstone: late events cannot revive
    // the reporter until an explicit REGISTER clears it.
    ASSERT_EQ(EC_OK, event_backend->UnregisterNode("test_instance", host_a));
    visible = query_visibility({current_key, down_key});
    ASSERT_EQ(2u, visible.size());
    EXPECT_TRUE(visible[0].empty());
    ASSERT_EQ(1u, visible[1].size());
    EXPECT_EQ(EC_NODE_NOT_REGISTERED, event_backend->OnHeartbeat("test_instance", host_a, {}));
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host_a, {"mem"}));
    visible = query_visibility({current_key});
    ASSERT_EQ(1u, visible.size());
    ASSERT_EQ(1u, visible[0].size());
    EXPECT_NE(std::string::npos, visible[0][0].find("source=a_second"));

    const auto [delta_ec, delta_response] =
        CallReportEvent(MakeAddRequest(host_a, current_key, "new_lifecycle_delta"), "query_lifecycle_first_delta");
    ASSERT_EQ(EC_OK, delta_ec);
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(delta_response.committed_snapshot_version()));
    EXPECT_NE(second_a_token, delta_response.committed_snapshot_version());
    visible = query_visibility({current_key});
    ASSERT_EQ(1u, visible.size());
    ASSERT_EQ(1u, visible[0].size());
    EXPECT_NE(std::string::npos, visible[0][0].find("source=new_lifecycle_delta"));
    const auto [third_a_ec, third_a] =
        CallReportEvent(MakeSnapshotRequest(host_a, {{current_key, "a_third"}}), "query_lifecycle_a_third");
    ASSERT_EQ(EC_OK, third_a_ec);
    ASSERT_NE(second_a_token, third_a.committed_snapshot_version());
    visible = query_visibility({current_key});
    ASSERT_EQ(1u, visible.size());
    ASSERT_EQ(1u, visible[0].size());
    EXPECT_NE(std::string::npos, visible[0][0].find("source=a_third"));
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_EventReportNodeUnavailable) {
    // Start from a known-readable location, then change only node liveness.
    const std::string instance_id = "test_instance";
    const std::string node_host = "192.168.1.200:8080";
    auto event_report_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_report_backend);
    ASSERT_EQ(EC_OK, event_report_backend->RegisterNode(instance_id, node_host, {"mem"}));
    const auto [snapshot_ec, snapshot_response] =
        CallReportEvent(MakeSnapshotRequest(node_host, {}), "unavailable_valid_snapshot");
    ASSERT_EQ(EC_OK, snapshot_ec);
    const std::string token = snapshot_response.committed_snapshot_version();
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(token));

    std::string current_uri;
    ASSERT_TRUE(
        SnapshotUriUtils::AddSnapshotVersionToUri("event_report://physical-cache:9600/mem", token, current_uri));
    auto func = cache_manager_->GetCheckLocDataExistFunc(instance_id);
    CacheLocation loc;
    loc.set_id(event_report_backend->BuildLocationId("mem", node_host));
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2);
    loc.set_location_specs({LocationSpec("tp0", current_uri)});

    ASSERT_TRUE(func(loc));
    event_report_backend->SetNodeUnavailable(instance_id, node_host);
    EXPECT_FALSE(func(loc));
    ASSERT_EQ(EC_OK, event_report_backend->OnHeartbeat(instance_id, node_host, {}));
    EXPECT_TRUE(func(loc));
}

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFunc_EventReportNodeUnregistered) {
    const std::string instance_id = "test_instance";
    const std::string node_host = "192.168.1.200:8080";
    auto event_report_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_report_backend);
    ASSERT_EQ(EC_OK, event_report_backend->RegisterNode(instance_id, node_host, {"mem"}));
    const auto [first_snapshot_ec, first_snapshot] =
        CallReportEvent(MakeSnapshotRequest(node_host, {}), "unregistered_valid_snapshot");
    ASSERT_EQ(EC_OK, first_snapshot_ec);
    const std::string first_token = first_snapshot.committed_snapshot_version();
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(first_token));

    std::string first_uri;
    ASSERT_TRUE(
        SnapshotUriUtils::AddSnapshotVersionToUri("event_report://physical-cache:9600/mem", first_token, first_uri));
    auto func = cache_manager_->GetCheckLocDataExistFunc(instance_id);
    CacheLocation loc;
    loc.set_id(event_report_backend->BuildLocationId("mem", node_host));
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2);
    loc.set_location_specs({LocationSpec("tp0", first_uri)});
    ASSERT_TRUE(func(loc));

    ASSERT_EQ(EC_OK, event_report_backend->UnregisterNode(instance_id, node_host));
    EXPECT_FALSE(func(loc));

    // Same-process unregister is tombstoned, so only an explicit REGISTER can
    // reuse persisted soft cache metadata. The first delta then establishes
    // a fresh reconciliation generation.
    ASSERT_EQ(EC_NODE_NOT_REGISTERED, event_report_backend->OnHeartbeat(instance_id, node_host, {}));
    ASSERT_EQ(EC_OK, event_report_backend->RegisterNode(instance_id, node_host, {"mem"}));
    EXPECT_TRUE(func(loc));
    std::string committed = "must-be-cleared";
    ASSERT_EQ(EC_OK, event_report_backend->BeginDeltaMutation({instance_id, node_host}, committed));
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(committed));
    EXPECT_NE(first_token, committed);
    event_report_backend->EndDeltaMutation({instance_id, node_host});

    const auto [second_snapshot_ec, second_snapshot] =
        CallReportEvent(MakeSnapshotRequest(node_host, {}), "reregister_new_snapshot");
    ASSERT_EQ(EC_OK, second_snapshot_ec);
    const std::string second_token = second_snapshot.committed_snapshot_version();
    ASSERT_NE(first_token, second_token);
    std::string second_uri;
    ASSERT_TRUE(
        SnapshotUriUtils::AddSnapshotVersionToUri("event_report://physical-cache:9600/mem", second_token, second_uri));
    loc.set_location_specs({LocationSpec("tp0", second_uri)});
    EXPECT_TRUE(func(loc));
    // The successful replacement snapshot restores strict visibility for the
    // new lifecycle, so the old lifecycle's token is fenced immediately.
    loc.set_location_specs({LocationSpec("tp0", first_uri)});
    EXPECT_FALSE(func(loc));
}

TEST_F(CacheManagerTest, TestGetSubmitDelReqFunc_NullExecutor) {
    // when schedule_plan_executor_ is null, calling the functor should
    // not crash
    auto saved = cache_manager_->schedule_plan_executor_;
    cache_manager_->schedule_plan_executor_ = nullptr;

    auto func = cache_manager_->GetSubmitDelReqFunc("test_instance");
    func({1, 2, 3}, {{"loc_a"}, {"loc_b"}, {"loc_c"}}, {}, false);

    cache_manager_->schedule_plan_executor_ = saved;
}

TEST_F(CacheManagerTest, TestGetSubmitDelReqFunc_DeletesLocationMetadata) {
    // end-to-end: write cache entries, then use the del functor to
    // request deletion; verify the location status changes
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    std::vector<std::int64_t> keys{1001, 1002};
    auto [ec1, start_write_cache_info] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);

    {
        BlockMask block_mask = static_cast<std::size_t>(2);
        auto ec = cache_manager_->FinishWriteCache(
            request_context_.get(), "test_instance", start_write_cache_info.write_session_id(), block_mask);
        ASSERT_EQ(EC_OK, ec);
    }

    // collect location IDs from metadata
    BlockMask block_mask = static_cast<std::size_t>(0);
    std::vector<std::vector<std::string>> loc_ids;
    {
        auto [ec, cache_metas] =
            cache_manager_->GetCacheMeta(request_context_.get(), "test_instance", keys, {}, block_mask, 0);
        ASSERT_EQ(EC_OK, ec);
        const auto &views = cache_metas.cache_locations_view();
        ASSERT_EQ(2u, views.size());
        for (const auto &view : views) {
            std::map<std::string, std::string> meta;
            ASSERT_TRUE(Jsonizable::FromJsonString(cache_metas.metas()[&view - &views[0]], meta));
            ASSERT_EQ(CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_SERVING), meta.at("status"));
            loc_ids.push_back({view.cache_location_.id()});
        }
    }

    // use GetSubmitDelReqFunc to submit a deletion request
    auto del_func = cache_manager_->GetSubmitDelReqFunc("test_instance");
    del_func(keys, loc_ids, {}, false);

    // wait for the async executor to process the request
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // verify location statuses changed to DELETING or NOT_FOUND
    {
        auto [ec, cache_metas] =
            cache_manager_->GetCacheMeta(request_context_.get(), "test_instance", keys, {}, block_mask, 0);
        ASSERT_EQ(EC_OK, ec);
        const auto &metas = cache_metas.metas();
        ASSERT_EQ(2u, metas.size());
        for (int i = 0; i < 2; ++i) {
            std::map<std::string, std::string> meta;
            ASSERT_TRUE(Jsonizable::FromJsonString(metas[i], meta));
            auto status = meta.at("status");
            ASSERT_TRUE(status == CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_DELETING) ||
                        status == CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_NOT_FOUND))
                << "expected DELETING or NOT_FOUND, got: " << status;
        }
    }
}

// ---------------------------------------------------------------------
// FilterWriteCache tests: verify the aggressive prune policy feature
// where stale locations (data no longer on storage) are detected via
// MightExist, excluded from the block mask, and submitted for deletion.
// ---------------------------------------------------------------------

TEST_F(CacheManagerTest, TestFilterWriteCache_NoStaleLocations) {
    // baseline: all existing locations have valid data, so behaviour
    // should be the same as before the aggressive prune feature
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    // write keys {1,2} and finish them as CLS_SERVING
    std::vector<std::int64_t> write_keys{1, 2};
    auto [ec1, swci1] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", write_keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    {
        BlockMask bm = static_cast<std::size_t>(2);
        ASSERT_EQ(
            EC_OK,
            cache_manager_->FinishWriteCache(request_context_.get(), "test_instance", swci1.write_session_id(), bm));
    }

    // install interceptor that returns all-true (data exists)
    auto dsm = registry_manager_->data_storage_manager_;
    auto original = dsm->storage_map_["nfs_01"];
    std::atomic<int> might_exist_calls{0};
    dsm->storage_map_["nfs_01"] = std::make_shared<MightExistInterceptor>(
        original, [&might_exist_calls](const std::vector<DataStorageUri> &uris) {
            might_exist_calls.fetch_add(1);
            return std::vector<bool>(uris.size(), true);
        });

    // StartWriteCache with {1, 2, 3, 4}; keys 1,2 exist, 3,4 need
    // new writes
    std::vector<std::int64_t> keys{1, 2, 3, 4};
    auto [ec2, swci2] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec2);

    // contiguous prefix → BlockMaskOffset
    ASSERT_EQ(2u, std::get<BlockMaskOffset>(swci2.block_mask()));
    ASSERT_EQ(2u, swci2.locations().cache_locations_view().size());

    // MightExist should have been called for the 2 existing
    // CLS_SERVING locations
    ASSERT_EQ(2, might_exist_calls.load());

    dsm->storage_map_["nfs_01"] = original;
}

TEST_F(CacheManagerTest, TestFilterWriteCache_StaleBreaksPrefix) {
    // one stale location in the middle of what would otherwise be a
    // contiguous prefix; the block mask should become a vector
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    // write keys {1,2,3} and finish as CLS_SERVING
    std::vector<std::int64_t> write_keys{1, 2, 3};
    auto [ec1, swci1] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", write_keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    {
        BlockMask bm = static_cast<std::size_t>(3);
        ASSERT_EQ(
            EC_OK,
            cache_manager_->FinishWriteCache(request_context_.get(), "test_instance", swci1.write_session_id(), bm));
    }

    // stop executor workers so we can inspect queued tasks
    auto &executor = cache_manager_->schedule_plan_executor_;
    executor->stop_.store(true);
    executor->condition_.notify_all();
    for (auto &w : executor->workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    executor->workers_.clear();
    executor->stop_.store(false);
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        for (auto &queue : executor->task_queues_) {
            queue.clear();
        }
    }

    // install interceptor: key 1 exists, key 2 stale, key 3 exists
    auto dsm = registry_manager_->data_storage_manager_;
    auto original = dsm->storage_map_["nfs_01"];
    std::atomic<int> call_idx{0};
    dsm->storage_map_["nfs_01"] =
        std::make_shared<MightExistInterceptor>(original, [&call_idx](const std::vector<DataStorageUri> &uris) {
            int idx = call_idx.fetch_add(1);
            if (idx == 1) {
                // second CLS_SERVING location (key 2): stale
                return std::vector<bool>(uris.size(), false);
            }
            return std::vector<bool>(uris.size(), true);
        });

    // StartWriteCache with {1, 2, 3, 4}
    std::vector<std::int64_t> keys{1, 2, 3, 4};
    auto [ec2, swci2] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec2);

    // key 2 stale breaks the contiguous prefix → BlockMaskVector
    auto mask = std::get<BlockMaskVector>(swci2.block_mask());
    ASSERT_EQ(4u, mask.size());
    ASSERT_TRUE(mask[0]);  // key 1: valid
    ASSERT_FALSE(mask[1]); // key 2: stale
    ASSERT_TRUE(mask[2]);  // key 3: valid
    ASSERT_FALSE(mask[3]); // key 4: not written yet
    // 2 new write locations: for keys 2 and 4
    ASSERT_EQ(2u, swci2.locations().cache_locations_view().size());

    // a deletion request should have been submitted for the stale
    // location
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        ASSERT_EQ(1u, executor->WaitingTaskCountLocked());
    }

    // clean up
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        for (auto &queue : executor->task_queues_) {
            queue.clear();
        }
    }
    dsm->storage_map_["nfs_01"] = original;
}

TEST_F(CacheManagerTest, TestFilterWriteCache_AllStale) {
    // all existing locations are stale; block mask should indicate all
    // keys need new writes (offset 0); deletion requests submitted for
    // all stale keys
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    // This test inspects the queued delete request directly. Stop the
    // supervisor so it cannot concurrently consume a future whose scheduled
    // task is deliberately removed below.
    cache_manager_->reclaimer_task_supervisor_->Stop();

    // write keys {1,2} and finish as CLS_SERVING
    std::vector<std::int64_t> write_keys{1, 2};
    auto [ec1, swci1] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", write_keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    {
        BlockMask bm = static_cast<std::size_t>(2);
        ASSERT_EQ(
            EC_OK,
            cache_manager_->FinishWriteCache(request_context_.get(), "test_instance", swci1.write_session_id(), bm));
    }

    // stop executor workers
    auto &executor = cache_manager_->schedule_plan_executor_;
    executor->stop_.store(true);
    executor->condition_.notify_all();
    for (auto &w : executor->workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    executor->workers_.clear();
    executor->stop_.store(false);
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        for (auto &queue : executor->task_queues_) {
            queue.clear();
        }
    }

    // install interceptor: all stale
    auto dsm = registry_manager_->data_storage_manager_;
    auto original = dsm->storage_map_["nfs_01"];
    dsm->storage_map_["nfs_01"] = std::make_shared<MightExistInterceptor>(
        original, [](const std::vector<DataStorageUri> &uris) { return std::vector<bool>(uris.size(), false); });

    // StartWriteCache with {1, 2, 3}
    std::vector<std::int64_t> keys{1, 2, 3};
    auto [ec2, swci2] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec2);

    // all stale + one new key → contiguous prefix of not-existing from
    // offset 0
    ASSERT_EQ(0u, std::get<BlockMaskOffset>(swci2.block_mask()));
    // all 3 keys need new write locations
    ASSERT_EQ(3u, swci2.locations().cache_locations_view().size());

    // deletion request submitted for the 2 stale keys
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        ASSERT_EQ(1u, executor->WaitingTaskCountLocked());
    }

    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        for (auto &queue : executor->task_queues_) {
            queue.clear();
        }
    }
    dsm->storage_map_["nfs_01"] = original;
}

TEST_F(CacheManagerTest, TestFilterWriteCache_StaleSuffix) {
    // stale locations are in the suffix (after the first valid prefix);
    // since all entries from first_empty onward are not-existing, the
    // contiguous prefix optimisation (BlockMaskOffset) still applies
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    // write keys {1,2,3} and finish as CLS_SERVING
    std::vector<std::int64_t> write_keys{1, 2, 3};
    auto [ec1, swci1] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", write_keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    {
        BlockMask bm = static_cast<std::size_t>(3);
        ASSERT_EQ(
            EC_OK,
            cache_manager_->FinishWriteCache(request_context_.get(), "test_instance", swci1.write_session_id(), bm));
    }

    // stop executor workers
    auto &executor = cache_manager_->schedule_plan_executor_;
    executor->stop_.store(true);
    executor->condition_.notify_all();
    for (auto &w : executor->workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    executor->workers_.clear();
    executor->stop_.store(false);
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        for (auto &queue : executor->task_queues_) {
            queue.clear();
        }
    }

    // install interceptor: key 1 valid, key 2 stale, key 3 stale
    auto dsm = registry_manager_->data_storage_manager_;
    auto original = dsm->storage_map_["nfs_01"];
    std::atomic<int> call_idx{0};
    dsm->storage_map_["nfs_01"] =
        std::make_shared<MightExistInterceptor>(original, [&call_idx](const std::vector<DataStorageUri> &uris) {
            int idx = call_idx.fetch_add(1);
            if (idx == 0) {
                // first CLS_SERVING location (key 1): valid
                return std::vector<bool>(uris.size(), true);
            }
            // keys 2, 3: stale
            return std::vector<bool>(uris.size(), false);
        });

    // StartWriteCache with {1, 2, 3}; key 1 exists, keys 2,3 stale
    std::vector<std::int64_t> keys{1, 2, 3};
    auto [ec2, swci2] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec2);

    // from offset 1 onward all are not-existing → BlockMaskOffset
    ASSERT_EQ(1u, std::get<BlockMaskOffset>(swci2.block_mask()));
    // 2 new write locations for keys 2,3
    ASSERT_EQ(2u, swci2.locations().cache_locations_view().size());

    // deletion request submitted for stale keys 2,3
    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        ASSERT_EQ(1u, executor->WaitingTaskCountLocked());
    }

    {
        std::lock_guard<std::mutex> lock(executor->queue_mutex_);
        for (auto &queue : executor->task_queues_) {
            queue.clear();
        }
    }
    dsm->storage_map_["nfs_01"] = original;
}

TEST_F(CacheManagerTest, TestFilterWriteCache_StaleVsNonEmptyBlockMask) {
    // verify that the block mask uses exists_results (data existence)
    // rather than the old !location_maps[i].empty() check; a location
    // map can be non-empty (has CLS_SERVING metadata) yet the data is
    // stale, so it should be marked false in the mask
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    // write keys {1,2,3,4} and finish as CLS_SERVING
    std::vector<std::int64_t> write_keys{1, 2, 3, 4};
    auto [ec1, swci1] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", write_keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec1);
    {
        BlockMask bm = static_cast<std::size_t>(4);
        ASSERT_EQ(
            EC_OK,
            cache_manager_->FinishWriteCache(request_context_.get(), "test_instance", swci1.write_session_id(), bm));
    }

    // install interceptor: keys 1,3 valid; keys 2,4 stale
    auto dsm = registry_manager_->data_storage_manager_;
    auto original = dsm->storage_map_["nfs_01"];
    std::atomic<int> call_idx{0};
    dsm->storage_map_["nfs_01"] =
        std::make_shared<MightExistInterceptor>(original, [&call_idx](const std::vector<DataStorageUri> &uris) {
            int idx = call_idx.fetch_add(1);
            if (idx == 0 || idx == 2) {
                // keys 1 and 3: valid
                return std::vector<bool>(uris.size(), true);
            }
            // keys 2 and 4: stale
            return std::vector<bool>(uris.size(), false);
        });

    // StartWriteCache with {1, 2, 3, 4}; re-request same keys
    std::vector<std::int64_t> keys{1, 2, 3, 4};
    auto [ec2, swci2] =
        cache_manager_->StartWriteCache(request_context_.get(), "test_instance", keys, {}, {}, 100000000);
    ASSERT_EQ(EC_OK, ec2);

    // stale entries produce a non-contiguous pattern → BlockMaskVector
    auto mask = std::get<BlockMaskVector>(swci2.block_mask());
    ASSERT_EQ(4u, mask.size());
    ASSERT_TRUE(mask[0]);  // key 1: valid data
    ASSERT_FALSE(mask[1]); // key 2: stale → needs rewrite
    ASSERT_TRUE(mask[2]);  // key 3: valid data
    ASSERT_FALSE(mask[3]); // key 4: stale → needs rewrite
    // 2 new write locations for stale keys 2 and 4
    ASSERT_EQ(2u, swci2.locations().cache_locations_view().size());

    dsm->storage_map_["nfs_01"] = original;
}

TEST_F(CacheManagerTest, TestStartWriteCacheSpecGroupDedup) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    std::vector<LocationSpecInfo> location_spec_infos = {
        LocationSpecInfo("tp0_F0", 512),
        LocationSpecInfo("tp1_F0", 512),
        LocationSpecInfo("tp0_L1", 512),
        LocationSpecInfo("tp1_L1", 512),
    };
    std::vector<LocationSpecGroup> location_spec_groups = {
        LocationSpecGroup("F0", {"tp0_F0", "tp1_F0"}),
        LocationSpecGroup("L1", {"tp0_L1", "tp1_L1"}),
    };
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_dedup",
                                               64,
                                               location_spec_infos,
                                               createModelDeployment(),
                                               location_spec_groups));
    std::vector<int64_t> keys{1, 2, 3};

    // First write: F0 group
    std::string write_session_id_1;
    {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_dedup", keys, {}, {"F0", "F0", "F0"}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        write_session_id_1 = info.write_session_id();
        ASSERT_EQ(0, std::get<BlockMaskOffset>(info.block_mask()));
        const auto &views = info.locations().cache_locations_view();
        ASSERT_EQ(3, views.size());
        for (size_t i = 0; i < views.size(); ++i) {
            ASSERT_EQ(2, views[i].spec_size()) << "F0 group should have 2 specs at key index " << i;
        }
    }
    // Finish first write successfully
    {
        BlockMask block_mask = static_cast<size_t>(3); // all 3 blocks succeed
        auto ec =
            cache_manager_->FinishWriteCache(request_context_.get(), "test_dedup", write_session_id_1, block_mask);
        ASSERT_EQ(EC_OK, ec);
    }

    // Second write: L1 group for same keys → should NOT be deduped
    std::string write_session_id_2;
    {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_dedup", keys, {}, {"L1", "L1", "L1"}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        write_session_id_2 = info.write_session_id();
        ASSERT_EQ(0, std::get<BlockMaskOffset>(info.block_mask()))
            << "L1 write should not be deduped by existing F0 locations";
        const auto &views = info.locations().cache_locations_view();
        ASSERT_EQ(3, views.size());
        for (size_t i = 0; i < views.size(); ++i) {
            ASSERT_EQ(2, views[i].spec_size()) << "L1 group should have 2 specs at key index " << i;
        }
    }
    // Finish second write
    {
        BlockMask block_mask = static_cast<size_t>(3);
        auto ec =
            cache_manager_->FinishWriteCache(request_context_.get(), "test_dedup", write_session_id_2, block_mask);
        ASSERT_EQ(EC_OK, ec);
    }

    // Third write: F0 group again → should be fully deduped
    {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_dedup", keys, {}, {"F0", "F0", "F0"}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(3, std::get<BlockMaskOffset>(info.block_mask())) << "F0 re-write should be fully deduped";
        ASSERT_EQ(0, info.locations().cache_locations_view().size());
    }
}

TEST_F(CacheManagerTest, TestWriteThenReadRoundTripWithSpecGroups) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    std::vector<LocationSpecInfo> location_spec_infos = {
        LocationSpecInfo("tp0_F0", 512),
        LocationSpecInfo("tp1_F0", 512),
        LocationSpecInfo("tp0_L1", 512),
        LocationSpecInfo("tp1_L1", 512),
    };
    std::vector<LocationSpecGroup> location_spec_groups = {
        LocationSpecGroup("F0", {"tp0_F0", "tp1_F0"}),
        LocationSpecGroup("L1", {"tp0_L1", "tp1_L1"}),
    };
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_roundtrip",
                                               64,
                                               location_spec_infos,
                                               createModelDeployment(),
                                               location_spec_groups));
    std::vector<int64_t> keys{100, 200, 300};

    // Write F0 group
    {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_roundtrip", keys, {}, {"F0", "F0", "F0"}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        BlockMask block_mask = static_cast<size_t>(3);
        ec = cache_manager_->FinishWriteCache(
            request_context_.get(), "test_roundtrip", info.write_session_id(), block_mask);
        ASSERT_EQ(EC_OK, ec);
    }

    // Write L1 group
    {
        auto [ec, info] = cache_manager_->StartWriteCache(
            request_context_.get(), "test_roundtrip", keys, {}, {"L1", "L1", "L1"}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        BlockMask block_mask = static_cast<size_t>(3);
        ec = cache_manager_->FinishWriteCache(
            request_context_.get(), "test_roundtrip", info.write_session_id(), block_mask);
        ASSERT_EQ(EC_OK, ec);
    }

    // Read back via QT_PREFIX_MATCH
    {
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                      "test_roundtrip",
                                                                      CacheManager::QueryType::QT_PREFIX_MATCH,
                                                                      keys,
                                                                      {},
                                                                      block_mask,
                                                                      0,
                                                                      {});
        ASSERT_EQ(EC_OK, ec);
        const auto &views = cache_locations.cache_locations_view();
        ASSERT_EQ(3, views.size());
        for (size_t i = 0; i < views.size(); ++i) {
            // Both F0 and L1 groups are on the same storage type (NFS),
            // so specs from both locations are merged → 4 specs total
            ASSERT_EQ(4, views[i].location_specs().size())
                << "key index " << i << " should have 4 specs merged from same storage type";
            for (const auto &spec : views[i].location_specs()) {
                EXPECT_FALSE(spec.uri().empty());
            }
        }
    }

    // Read with spec name filter: only F0 specs
    {
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, cache_locations] = cache_manager_->GetCacheLocation(request_context_.get(),
                                                                      "test_roundtrip",
                                                                      CacheManager::QueryType::QT_PREFIX_MATCH,
                                                                      keys,
                                                                      {},
                                                                      block_mask,
                                                                      0,
                                                                      {"tp0_F0", "tp1_F0"});
        ASSERT_EQ(EC_OK, ec);
        const auto &views = cache_locations.cache_locations_view();
        ASSERT_EQ(3, views.size());
        for (size_t i = 0; i < views.size(); ++i) {
            // All specs come from NFS (same type), so filtering by F0 spec names
            // should always yield 2 specs
            ASSERT_EQ(2, views[i].location_specs().size())
                << "key index " << i << " should have 2 F0 specs after filtering";
        }
    }
}

TEST_F(CacheManagerTest, TestDoRecoverAfterCleanup) {
    // Cleanup then recover
    ASSERT_EQ(EC_OK, cache_manager_->DoCleanup());
    ASSERT_EQ(EC_OK, cache_manager_->DoRecoverOnce());

    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_TRUE(meta_searcher);
    ASSERT_TRUE(meta_searcher->meta_indexer_);
    ASSERT_EQ("test_instance", meta_searcher->meta_indexer_->instance_id_);

    // Call again - should be idempotent
    ASSERT_EQ(EC_OK, cache_manager_->DoRecoverOnce());
    meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_TRUE(meta_searcher);
    ASSERT_EQ("test_instance", meta_searcher->meta_indexer_->instance_id_);
}

TEST_F(CacheManagerTest, TestDoRecoverPreservesRegisteredDefaultQueryType) {
    registry_manager_->instance_infos_["test_instance"]->set_default_query_type(
        static_cast<int32_t>(CacheManager::QueryType::QT_PREFIX_MATCH));

    ASSERT_EQ(EC_OK, cache_manager_->DoCleanup());
    ASSERT_EQ(EC_OK, cache_manager_->DoRecoverOnce());

    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_TRUE(meta_searcher);
    ASSERT_TRUE(meta_searcher->meta_indexer_);
    ASSERT_EQ("test_instance", meta_searcher->meta_indexer_->instance_id_);

    CacheManager::KeyVector keys = {1};
    auto [ec, hosts] = cache_manager_->GetHostCacheState(
        request_context_.get(), "test_instance", CacheManager::QueryType::QT_UNSPECIFIED, keys);
    EXPECT_EQ(EC_OK, ec);
    EXPECT_TRUE(hosts.empty());
}

TEST_F(CacheManagerTest, TestDoRecoverOnceWithRegistryPartialFailureThenFix) {
    // Scenario:
    // 1. RegistryManager has instance_group + test_instance recovered, but a second instance
    //    "missing_instance" is expected but not in instance_infos_ (simulates partial recover failure)
    // 2. RegistryManager::IsRecoverComplete() returns false
    // 3. CacheManager::DoRecoverOnce() succeeds for test_instance but overall returns ERROR
    // 4. "Fix" RegistryManager by adding missing_instance and marking recover complete
    // 5. CacheManager::DoRecoverOnce() now succeeds and missing_instance gets its MetaSearcher

    // Cleanup to start fresh
    ASSERT_EQ(EC_OK, cache_manager_->DoCleanup());

    // Simulate RegistryManager partial failure: recover_complete_ is false
    // test_instance is already in instance_infos_ (from SetUp), but mark as incomplete
    registry_manager_->recover_complete_.store(false);

    // CacheManager DoRecoverOnce - should return ERROR because RegistryManager is incomplete
    auto ec = cache_manager_->DoRecoverOnce();
    ASSERT_EQ(EC_ERROR, ec);

    // test_instance MetaSearcher should still have been created (partial progress is retained)
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_TRUE(meta_searcher);
    ASSERT_EQ("test_instance", meta_searcher->meta_indexer_->instance_id_);

    // "missing_instance" doesn't exist yet - no MetaSearcher
    meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("missing_instance");
    ASSERT_FALSE(meta_searcher);

    // Now "fix" RegistryManager: add missing_instance to instance_infos_ and mark complete
    auto missing_instance_info = std::make_shared<InstanceInfo>(
        "test_quota_group", "default", "missing_instance", 64, createLocationSpecInfos(), createModelDeployment());
    registry_manager_->instance_infos_["missing_instance"] = missing_instance_info;
    registry_manager_->recover_complete_.store(true);

    // CacheManager DoRecoverOnce - should now succeed
    ec = cache_manager_->DoRecoverOnce();
    ASSERT_EQ(EC_OK, ec);

    // Both instances should have MetaSearcher
    meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_TRUE(meta_searcher);
    ASSERT_EQ("test_instance", meta_searcher->meta_indexer_->instance_id_);

    meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("missing_instance");
    ASSERT_TRUE(meta_searcher);
    ASSERT_EQ("missing_instance", meta_searcher->meta_indexer_->instance_id_);
}

TEST_F(CacheManagerTest, TestRecoverRetryLoopLifecycle) {
    // StopRecoverRetryLoop should be safe without active thread
    cache_manager_->StopRecoverRetryLoop();
    cache_manager_->StopRecoverRetryLoop(); // double-stop should not crash

    // StartRecoverRetryLoop + StopRecoverRetryLoop
    cache_manager_->StartRecoverRetryLoop();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    cache_manager_->StopRecoverRetryLoop(); // should join within ~100ms

    // DoCleanup should stop retry thread
    cache_manager_->StartRecoverRetryLoop();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ASSERT_EQ(EC_OK, cache_manager_->DoCleanup());
}

/* ------------ InvalidateInstanceMetrics tests ------------ */

TEST_F(CacheManagerTest, InvalidateInstanceMetricsEmptyIdIsNoOp) {
    auto metrics_registry = std::make_shared<MetricsRegistry>();
    auto rm_registry = std::make_shared<MetricsRegistry>(); // separate for RM internals
    auto rm = std::make_shared<RegistryManager>("", rm_registry);
    rm->Init();
    auto cm = std::make_unique<CacheManager>(metrics_registry, rm);

    metrics_registry->GetCounter("test.counter", {{"instance_id", "inst1"}});
    ASSERT_EQ(1, metrics_registry->GetSize());

    // empty id should be a no-op
    cm->InvalidateInstanceMetrics("");
    ASSERT_EQ(1, metrics_registry->GetSize());
}

TEST_F(CacheManagerTest, InvalidateInstanceMetricsNoCallbackDoesNotCrash) {
    auto metrics_registry = std::make_shared<MetricsRegistry>();
    auto rm_registry = std::make_shared<MetricsRegistry>();
    auto rm = std::make_shared<RegistryManager>("", rm_registry);
    rm->Init();
    auto cm = std::make_unique<CacheManager>(metrics_registry, rm);

    metrics_registry->GetCounter("test.counter", {{"instance_id", "inst1"}});
    ASSERT_EQ(1, metrics_registry->GetSize());

    // no callback set — should not crash, and still purge registry
    ASSERT_NO_FATAL_FAILURE(cm->InvalidateInstanceMetrics("inst1"));
    ASSERT_EQ(0, metrics_registry->GetSize());
}

TEST_F(CacheManagerTest, InvalidateInstanceMetricsPurgesRegistry) {
    auto metrics_registry = std::make_shared<MetricsRegistry>();
    auto rm_registry = std::make_shared<MetricsRegistry>();
    auto rm = std::make_shared<RegistryManager>("", rm_registry);
    rm->Init();
    auto cm = std::make_unique<CacheManager>(metrics_registry, rm);

    metrics_registry->GetCounter("m1", {{"instance_id", "inst1"}});
    metrics_registry->GetGauge("m2", {{"instance_id", "inst1"}, {"extra", "tag"}});
    metrics_registry->GetCounter("m1", {{"instance_id", "inst2"}});
    ASSERT_EQ(3, metrics_registry->GetSize());

    cm->InvalidateInstanceMetrics("inst1");

    // only inst2 remains
    ASSERT_EQ(1, metrics_registry->GetSize());
    std::vector<MetricsRegistry::metrics_tuple_t> all;
    metrics_registry->GetAllMetrics(all);
    ASSERT_EQ(1, all.size());
    auto &[name, tags, _] = all[0];
    ASSERT_EQ("m1", name);
    ASSERT_EQ("inst2", tags.at("instance_id"));
}

TEST_F(CacheManagerTest, InvalidateInstanceMetricsInvokesCallback) {
    auto metrics_registry = std::make_shared<MetricsRegistry>();
    auto rm_registry = std::make_shared<MetricsRegistry>();
    auto rm = std::make_shared<RegistryManager>("", rm_registry);
    rm->Init();
    auto cm = std::make_unique<CacheManager>(metrics_registry, rm);

    std::string received_id;
    int call_count = 0;
    cm->SetOnInstanceRemoved([&](const std::string &id) {
        received_id = id;
        ++call_count;
    });

    cm->InvalidateInstanceMetrics("inst42");
    ASSERT_EQ(1, call_count);
    ASSERT_EQ("inst42", received_id);

    // empty id should not invoke callback
    cm->InvalidateInstanceMetrics("");
    ASSERT_EQ(1, call_count);
}

TEST_F(CacheManagerTest, TestReportEventMutationValidationMatrixHasNoSideEffects) {
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);

    using ConfigureEvent = std::function<void(proto::meta::EventItem *, int64_t, const std::string &)>;
    struct TestCase {
        const char *name;
        ConfigureEvent configure;
    };
    auto configure_valid_add = [](proto::meta::EventItem *event, int64_t key, const std::string &host) {
        event->set_event_type(proto::meta::EVENT_BLOCK_ADD);
        auto *params = event->mutable_block_add();
        params->set_block_key(std::to_string(key));
        params->set_medium("mem");
        auto *spec = params->add_specs();
        spec->set_name("tp0");
        spec->set_uri("event_report://" + host + "/mem");
        return params;
    };
    auto configure_valid_delete = [](proto::meta::EventItem *event, int64_t key) {
        event->set_event_type(proto::meta::EVENT_BLOCK_DELETE);
        auto *params = event->mutable_block_delete();
        params->set_block_key(std::to_string(key));
        params->set_medium("mem");
        params->add_spec_names("tp0");
        return params;
    };
    auto configure_valid_snapshot = [](proto::meta::EventItem *event, int64_t key, const std::string &host) {
        event->set_event_type(proto::meta::EVENT_BLOCK_SNAPSHOT);
        auto *block = event->mutable_block_snapshot()->add_blocks();
        block->set_block_key(std::to_string(key));
        block->set_medium("mem");
        auto *spec = block->add_specs();
        spec->set_name("tp0");
        spec->set_uri("event_report://" + host + "/mem");
        return block;
    };

    const std::vector<TestCase> cases = {
        {"add_non_numeric_key",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_add(event, key, host)->set_block_key("not-a-number");
         }},
        {"add_positive_overflow_key",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_add(event, key, host)->set_block_key("18446744073709551616");
         }},
        {"add_negative_overflow_key",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_add(event, key, host)->set_block_key("-9223372036854775809");
         }},
        {"add_negative_uint64_alias_key",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_add(event, key, host)->set_block_key("-18446744073709551615");
         }},
        {"add_leading_plus_key",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_add(event, key, host)->set_block_key("+1");
         }},
        {"add_leading_space_key",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_add(event, key, host)->set_block_key(" 1");
         }},
        {"add_trailing_space_key",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_add(event, key, host)->set_block_key("1 ");
         }},
        {"add_empty_medium",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_add(event, key, host)->clear_medium();
         }},
        {"add_medium_with_separator",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_add(event, key, host)->set_medium("mem#bad");
         }},
        {"add_deprecated_uri_without_specs",
         [](auto *event, int64_t key, const std::string &host) {
             event->set_event_type(proto::meta::EVENT_BLOCK_ADD);
             auto *params = event->mutable_block_add();
             params->set_block_key(std::to_string(key));
             params->set_medium("mem");
             params->set_uri("event_report://" + host + "/deprecated");
         }},
        {"add_empty_spec_name",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_add(event, key, host)->mutable_specs(0)->clear_name();
         }},
        {"add_duplicate_spec_name",
         [=](auto *event, int64_t key, const std::string &host) {
             auto *params = configure_valid_add(event, key, host);
             *params->add_specs() = params->specs(0);
         }},
        {"add_invalid_uri",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_add(event, key, host)->mutable_specs(0)->set_uri("not-a-uri");
         }},
        {"add_invalid_uri_port",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_add(event, key, host)
                 ->mutable_specs(0)
                 ->set_uri("event_report://cache-node:not-a-port/mem");
         }},
        {"add_total_size_overflow",
         [=](auto *event, int64_t key, const std::string &host) {
             auto *params = configure_valid_add(event, key, host);
             params->mutable_specs(0)->set_uri("event_report://" + host + "/mem?size=18446744073709551615");
             auto *second = params->add_specs();
             second->set_name("tp1");
             second->set_uri("event_report://" + host + "/mem?size=1");
         }},
        {"add_client_snapshot_version",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_add(event, key, host)
                 ->mutable_specs(0)
                 ->set_uri("event_report://" + host + "/mem?s_version=" + std::string(32, 'a'));
         }},
        {"delete_non_numeric_key",
         [=](auto *event, int64_t key, const std::string &) {
             configure_valid_delete(event, key)->set_block_key("not-a-number");
         }},
        {"delete_empty_medium",
         [=](auto *event, int64_t key, const std::string &) { configure_valid_delete(event, key)->clear_medium(); }},
        {"delete_medium_with_separator",
         [=](auto *event, int64_t key, const std::string &) {
             configure_valid_delete(event, key)->set_medium("mem#bad");
         }},
        {"delete_empty_spec_names",
         [=](auto *event, int64_t key, const std::string &) {
             configure_valid_delete(event, key)->clear_spec_names();
         }},
        {"delete_empty_spec_name",
         [=](auto *event, int64_t key, const std::string &) {
             configure_valid_delete(event, key)->set_spec_names(0, "");
         }},
        {"delete_duplicate_spec_name",
         [=](auto *event, int64_t key, const std::string &) {
             configure_valid_delete(event, key)->add_spec_names("tp0");
         }},
        {"snapshot_non_numeric_key",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_snapshot(event, key, host)->set_block_key("not-a-number");
         }},
        {"snapshot_empty_medium",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_snapshot(event, key, host)->clear_medium();
         }},
        {"snapshot_medium_with_separator",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_snapshot(event, key, host)->set_medium("mem#bad");
         }},
        {"snapshot_empty_specs",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_snapshot(event, key, host)->clear_specs();
         }},
        {"snapshot_empty_spec_name",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_snapshot(event, key, host)->mutable_specs(0)->clear_name();
         }},
        {"snapshot_duplicate_spec_name",
         [=](auto *event, int64_t key, const std::string &host) {
             auto *block = configure_valid_snapshot(event, key, host);
             *block->add_specs() = block->specs(0);
         }},
        {"snapshot_invalid_uri",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_snapshot(event, key, host)->mutable_specs(0)->set_uri("not-a-uri");
         }},
        {"snapshot_invalid_uri_port",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_snapshot(event, key, host)
                 ->mutable_specs(0)
                 ->set_uri("event_report://cache-node:not-a-port/mem");
         }},
        {"snapshot_total_size_overflow",
         [=](auto *event, int64_t key, const std::string &host) {
             auto *block = configure_valid_snapshot(event, key, host);
             block->mutable_specs(0)->set_uri("event_report://" + host + "/mem?size=18446744073709551615");
             auto *second = block->add_specs();
             second->set_name("tp1");
             second->set_uri("event_report://" + host + "/mem?size=1");
         }},
        {"snapshot_client_snapshot_version",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_snapshot(event, key, host)
                 ->mutable_specs(0)
                 ->set_uri("event_report://" + host + "/mem?s_version=" + std::string(32, 'a'));
         }},
        {"snapshot_duplicate_block_and_medium",
         [=](auto *event, int64_t key, const std::string &host) {
             configure_valid_snapshot(event, key, host);
             auto *duplicate = event->mutable_block_snapshot()->add_blocks();
             *duplicate = event->block_snapshot().blocks(0);
         }},
    };

    for (size_t index = 0; index < cases.size(); ++index) {
        const auto &test_case = cases[index];
        SCOPED_TRACE(test_case.name);
        const std::string host = "192.168.12." + std::to_string(index + 1) + ":8080";
        const int64_t key = 95'000 + static_cast<int64_t>(index);
        ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

        proto::meta::ReportEventRequest request;
        request.set_instance_id("test_instance");
        request.set_host_ip_port(host);
        request.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
        test_case.configure(request.add_events(), key, host);

        const auto [ec, response] = CallReportEvent(request, test_case.name);
        EXPECT_EQ(EC_PARTIAL_OK, ec);
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
        ASSERT_EQ(1, response.item_results_size());
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.item_results(0));
        EXPECT_TRUE(response.committed_snapshot_version().empty());
        EXPECT_TRUE(response.snapshot_required());
        EXPECT_TRUE(event_backend->GetSnapshotVersion({"test_instance", host}).empty());
        EXPECT_TRUE(QueryRawEventReportUris(key).empty());
    }
}

TEST_F(CacheManagerTest, TestReportEventAcceptsSignedAndUnsignedBlockKeySpellingsWithoutChangingBitPattern) {
    const std::string host = "192.168.12.100:8080";
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    struct TestCase {
        const char *text;
        int64_t expected_key;
    };
    const std::vector<TestCase> cases = {
        {"0", 0},
        {"9223372036854775807", std::numeric_limits<int64_t>::max()},
        {"-9223372036854775808", std::numeric_limits<int64_t>::min()},
        {"9223372036854775808", std::numeric_limits<int64_t>::min()},
        {"-1", -1},
        {"18446744073709551615", -1},
    };

    for (size_t index = 0; index < cases.size(); ++index) {
        SCOPED_TRACE(cases[index].text);
        auto request = MakeAddRequest(host, 0, "block_key_boundary_" + std::to_string(index));
        request.mutable_events(0)->mutable_block_add()->set_block_key(cases[index].text);
        const auto [ec, response] = CallReportEvent(request, "block_key_boundary");
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(proto::meta::OK, response.header().status().code());
        EXPECT_FALSE(QueryRawEventReportUris(cases[index].expected_key).empty());
    }
}

TEST_F(CacheManagerTest, TestReportEventRejectsMismatchedPayloadsWithoutSideEffects) {
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);

    using ConfigurePayload = std::function<void(proto::meta::EventItem *)>;
    struct TestCase {
        const char *name;
        proto::meta::ReportEventType event_type;
        ConfigurePayload configure_payload;
        bool pre_register;
    };
    const std::vector<TestCase> cases = {
        {"unspecified_without_payload", proto::meta::EVENT_UNSPECIFIED, {}, true},
        {"register_without_payload", proto::meta::EVENT_NODE_REGISTER, {}, false},
        {"register_with_heartbeat_payload",
         proto::meta::EVENT_NODE_REGISTER,
         [](auto *event) { event->mutable_heartbeat(); },
         false},
        {"heartbeat_without_payload", proto::meta::EVENT_HEARTBEAT, {}, true},
        {"heartbeat_with_register_payload",
         proto::meta::EVENT_HEARTBEAT,
         [](auto *event) { event->mutable_node_register()->add_mediums("mem"); },
         true},
        {"host_down_without_payload", proto::meta::EVENT_HOST_DOWN, {}, true},
        {"host_down_with_heartbeat_payload",
         proto::meta::EVENT_HOST_DOWN,
         [](auto *event) { event->mutable_heartbeat(); },
         true},
        {"add_without_payload", proto::meta::EVENT_BLOCK_ADD, {}, true},
        {"delete_without_payload", proto::meta::EVENT_BLOCK_DELETE, {}, true},
        {"snapshot_without_payload", proto::meta::EVENT_BLOCK_SNAPSHOT, {}, true},
    };

    for (size_t index = 0; index < cases.size(); ++index) {
        const auto &test_case = cases[index];
        SCOPED_TRACE(test_case.name);
        const std::string host = "192.168.11." + std::to_string(index + 1) + ":8080";
        if (test_case.pre_register) {
            ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));
        }

        proto::meta::ReportEventRequest request;
        request.set_instance_id("test_instance");
        request.set_host_ip_port(host);
        request.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
        auto *event = request.add_events();
        event->set_event_type(test_case.event_type);
        if (test_case.configure_payload) {
            test_case.configure_payload(event);
        }

        const auto [ec, response] = CallReportEvent(request, test_case.name);
        EXPECT_EQ(EC_PARTIAL_OK, ec);
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
        ASSERT_EQ(1, response.item_results_size());
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.item_results(0));
        EXPECT_TRUE(response.committed_snapshot_version().empty());
        EXPECT_TRUE(response.snapshot_required());
        EXPECT_TRUE(event_backend->GetSnapshotVersion({"test_instance", host}).empty());
        EXPECT_EQ(test_case.pre_register, event_backend->IsNodeRegistered("test_instance", host));
    }

    // A malformed REGISTER is isolated to that item. Registration is not a
    // data-plane prerequisite, so a later valid mutation still lazily
    // restores the reporter and is applied.
    const std::string mixed_host = "192.168.11.200:8080";
    const int64_t mixed_key = 95'900;
    proto::meta::ReportEventRequest mixed_request;
    mixed_request.set_instance_id("test_instance");
    mixed_request.set_host_ip_port(mixed_host);
    mixed_request.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
    auto *bad_register = mixed_request.add_events();
    bad_register->set_event_type(proto::meta::EVENT_NODE_REGISTER);
    bad_register->mutable_heartbeat();
    *mixed_request.add_events() = MakeAddRequest(mixed_host, mixed_key, "must_not_write").events(0);

    const auto [mixed_ec, mixed_response] = CallReportEvent(mixed_request, "mismatched_register_then_delta");
    EXPECT_EQ(EC_PARTIAL_OK, mixed_ec);
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, mixed_response.header().status().code());
    ASSERT_EQ(2, mixed_response.item_results_size());
    EXPECT_EQ(proto::meta::INVALID_ARGUMENT, mixed_response.item_results(0));
    EXPECT_EQ(proto::meta::OK, mixed_response.item_results(1));
    EXPECT_TRUE(event_backend->IsNodeRegistered("test_instance", mixed_host));
    EXPECT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(
        event_backend->GetSnapshotVersion({"test_instance", mixed_host})));
    EXPECT_FALSE(QueryRawEventReportUris(mixed_key).empty());
}

TEST_F(CacheManagerTest, TestReportEventRejectsRequestShapeBeforeAnySideEffect) {
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    const std::string host = "192.168.13.1:8080";
    const int64_t key = 96'001;
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    auto expect_request_rejected = [&](proto::meta::ReportEventRequest request, const char *trace_id) {
        SCOPED_TRACE(trace_id);
        const auto [ec, response] = CallReportEvent(request, trace_id);
        EXPECT_EQ(EC_BADARGS, ec);
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
        EXPECT_EQ(0, response.item_results_size());
        EXPECT_TRUE(event_backend->GetSnapshotVersion({"test_instance", host}).empty());
        EXPECT_TRUE(QueryRawEventReportUris(key).empty());
    };

    auto snapshot_and_delta = MakeSnapshotRequest(host, {{key, "snapshot"}});
    *snapshot_and_delta.add_events() = MakeAddRequest(host, key, "delta").events(0);
    expect_request_rejected(std::move(snapshot_and_delta), "snapshot_and_delta");

    auto two_snapshots = MakeSnapshotRequest(host, {});
    *two_snapshots.add_events() = MakeSnapshotRequest(host, {}).events(0);
    expect_request_rejected(std::move(two_snapshots), "two_snapshots");

    proto::meta::ReportEventRequest host_down_and_heartbeat;
    host_down_and_heartbeat.set_instance_id("test_instance");
    host_down_and_heartbeat.set_host_ip_port(host);
    host_down_and_heartbeat.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
    auto *host_down = host_down_and_heartbeat.add_events();
    host_down->set_event_type(proto::meta::EVENT_HOST_DOWN);
    host_down->mutable_host_down();
    auto *heartbeat = host_down_and_heartbeat.add_events();
    heartbeat->set_event_type(proto::meta::EVENT_HEARTBEAT);
    heartbeat->mutable_heartbeat();
    expect_request_rejected(std::move(host_down_and_heartbeat), "host_down_and_heartbeat");
    EXPECT_TRUE(event_backend->IsNodeRegistered("test_instance", host));
    EXPECT_TRUE(event_backend->IsNodeAvailable("test_instance", host));
}

TEST_F(CacheManagerTest, TestReportEventFirstDeltaMetadataFailureReportsFailureAndReusesGeneration) {
    const std::string host = "192.168.13.2:8080";
    const int64_t key = 96'002;
    auto event_backend = InstallEventReportBackend();
    auto *meta_backend = InstallControllableMetaBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_NE(nullptr, meta_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    meta_backend->FailKeyOnNextUpsert(key);
    const auto [failed_ec, failed] =
        CallReportEvent(MakeAddRequest(host, key, "first_write_fails"), "first_delta_metadata_failure");
    EXPECT_EQ(EC_PARTIAL_OK, failed_ec);
    EXPECT_EQ(proto::meta::INTERNAL_ERROR, failed.header().status().code());
    ASSERT_EQ(1, failed.item_results_size());
    EXPECT_EQ(proto::meta::INTERNAL_ERROR, failed.item_results(0));
    const std::string generation = failed.committed_snapshot_version();
    EXPECT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(generation));
    EXPECT_TRUE(failed.snapshot_required());
    EXPECT_EQ(generation, event_backend->GetSnapshotVersion({"test_instance", host}));
    EXPECT_TRUE(QueryEventReportUris({key}).empty());

    const auto [retry_ec, retry] =
        CallReportEvent(MakeAddRequest(host, key, "first_write_retry"), "first_delta_metadata_retry");
    ASSERT_EQ(EC_OK, retry_ec);
    EXPECT_EQ(generation, retry.committed_snapshot_version());
    EXPECT_FALSE(retry.snapshot_required());
    const auto visible = QueryEventReportUris({key});
    ASSERT_EQ(1u, visible.size());
    EXPECT_NE(std::string::npos, visible.front().find("source=first_write_retry"));
    EXPECT_NE(std::string::npos, visible.front().find("s_version=" + generation));
}

TEST_F(CacheManagerTest, TestReportEventRejectsInvalidRequestsAndMapsItemErrors) {
    auto add_register_event = [](proto::meta::ReportEventRequest &request) {
        auto *event = request.add_events();
        event->set_event_type(proto::meta::EVENT_NODE_REGISTER);
        event->mutable_node_register()->add_mediums("mem");
    };

    {
        proto::meta::ReportEventRequest request;
        request.set_host_ip_port("10.0.0.30:8080");
        request.set_storage_type(proto::meta::ST_EVENT_REPORT_L1P5);
        add_register_event(request);

        proto::meta::ReportEventResponse response;
        EXPECT_EQ(EC_BADARGS, cache_manager_->ReportEvent(request_context_.get(), &request, &response));
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
        EXPECT_EQ(0, response.item_results_size());
    }

    {
        proto::meta::ReportEventRequest request;
        request.set_instance_id("test_instance");
        request.set_host_ip_port("10.0.0.30:8080");
        add_register_event(request);

        proto::meta::ReportEventResponse response;
        EXPECT_EQ(EC_BADARGS, cache_manager_->ReportEvent(request_context_.get(), &request, &response));
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
        EXPECT_EQ("storage_type is required", response.header().status().message());
    }

    {
        proto::meta::ReportEventRequest request;
        request.set_instance_id("test_instance");
        request.set_host_ip_port("10.0.0.30:8080");
        request.set_storage_type(proto::meta::ST_NFS);
        add_register_event(request);

        proto::meta::ReportEventResponse response;
        EXPECT_EQ(EC_BADARGS, cache_manager_->ReportEvent(request_context_.get(), &request, &response));
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
    }

    {
        proto::meta::ReportEventRequest request;
        request.set_instance_id("test_instance");
        request.set_host_ip_port("10.0.0.30:8080");
        // 263 has the same low byte as ST_EVENT_REPORT_L1P5 (7). It must remain
        // an unsupported open-enum value rather than being truncated and
        // routed to a real backend.
        request.set_storage_type(static_cast<proto::meta::StorageType>(263));
        add_register_event(request);

        proto::meta::ReportEventResponse response;
        EXPECT_EQ(EC_BADARGS, cache_manager_->ReportEvent(request_context_.get(), &request, &response));
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
        EXPECT_EQ("unsupported event-report storage_type: 263", response.header().status().message());
    }

    {
        proto::meta::ReportEventRequest request;
        request.set_instance_id("missing_instance");
        request.set_host_ip_port("10.0.0.30:8080");
        request.set_storage_type(proto::meta::ST_EVENT_REPORT_L1P5);
        add_register_event(request);

        proto::meta::ReportEventResponse response;
        EXPECT_EQ(EC_INSTANCE_NOT_EXIST, cache_manager_->ReportEvent(request_context_.get(), &request, &response));
        EXPECT_EQ(proto::meta::INSTANCE_NOT_EXIST, response.header().status().code());
    }

    auto event_backend = std::make_shared<EventReportBackend>(cache_manager_->metrics_registry_);
    StorageConfig config;
    config.set_global_unique_name("event_backend_errors");
    config.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5);
    config.set_storage_spec(std::make_shared<EventReportStorageSpec>());
    ASSERT_EQ(EC_OK, event_backend->Open(config, "test_trace"));
    registry_manager_->data_storage_manager_->storage_map_["event_backend_errors"] = event_backend;
    registry_manager_->instance_group_configs_["default"]->set_event_report_storage_candidates(
        {"event_backend_errors"});

    {
        proto::meta::ReportEventRequest request;
        request.set_instance_id("test_instance");
        request.set_host_ip_port("10.0.0.30#8080");
        request.set_storage_type(proto::meta::ST_EVENT_REPORT_L1P5);
        add_register_event(request);

        proto::meta::ReportEventResponse response;
        EXPECT_EQ(EC_BADARGS, cache_manager_->ReportEvent(request_context_.get(), &request, &response));
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
        EXPECT_FALSE(event_backend->IsNodeRegistered("test_instance", "10.0.0.30#8080"));
    }

    {
        proto::meta::ReportEventRequest request;
        request.set_instance_id("test_instance");
        request.set_host_ip_port("10.0.0.30:8080");
        request.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
        add_register_event(request);

        proto::meta::ReportEventResponse response;
        EXPECT_EQ(EC_INSTANCE_NOT_EXIST, cache_manager_->ReportEvent(request_context_.get(), &request, &response));
        EXPECT_EQ(proto::meta::INSTANCE_NOT_EXIST, response.header().status().code());
        EXPECT_FALSE(event_backend->IsNodeAvailable("test_instance", "10.0.0.30:8080"));
    }

    {
        proto::meta::ReportEventRequest request;
        request.set_instance_id("test_instance");
        request.set_host_ip_port("10.0.0.30:8080");
        request.set_storage_type(proto::meta::ST_EVENT_REPORT_L1P5);
        add_register_event(request);

        auto *invalid_add = request.add_events();
        invalid_add->set_event_type(proto::meta::EVENT_BLOCK_ADD);
        invalid_add->mutable_block_add()->set_block_key("100");
        invalid_add->mutable_block_add()->set_medium("mem");

        proto::meta::ReportEventResponse response;
        EXPECT_EQ(EC_PARTIAL_OK, cache_manager_->ReportEvent(request_context_.get(), &request, &response));
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
        ASSERT_EQ(2, response.item_results_size());
        EXPECT_EQ(proto::meta::OK, response.item_results(0));
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.item_results(1));
        EXPECT_TRUE(event_backend->IsNodeAvailable("test_instance", "10.0.0.30:8080"));
    }

    {
        proto::meta::ReportEventRequest request;
        request.set_instance_id("test_instance");
        request.set_host_ip_port("10.0.0.30:8080");
        request.set_storage_type(proto::meta::ST_EVENT_REPORT_L1P5);
        auto *invalid_add = request.add_events();
        invalid_add->set_event_type(proto::meta::EVENT_BLOCK_ADD);
        invalid_add->mutable_block_add()->set_block_key("101");
        invalid_add->mutable_block_add()->set_medium("mem#invalid");
        auto *spec = invalid_add->mutable_block_add()->add_specs();
        spec->set_name("tp0");
        spec->set_uri("event_report://10.0.0.30:8080/mem");

        proto::meta::ReportEventResponse response;
        EXPECT_EQ(EC_PARTIAL_OK, cache_manager_->ReportEvent(request_context_.get(), &request, &response));
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
        EXPECT_TRUE(event_backend->GetSnapshotVersion({"test_instance", "10.0.0.30:8080"}).empty());
    }

    {
        proto::meta::ReportEventRequest request;
        request.set_instance_id("test_instance");
        request.set_host_ip_port("10.0.0.30:8080");
        request.set_storage_type(proto::meta::ST_EVENT_REPORT_L1P5);
        auto *snapshot_event = request.add_events();
        snapshot_event->set_event_type(proto::meta::EVENT_BLOCK_SNAPSHOT);
        auto *block = snapshot_event->mutable_block_snapshot()->add_blocks();
        block->set_block_key("102");
        block->set_medium("mem#invalid");
        auto *spec = block->add_specs();
        spec->set_name("tp0");
        spec->set_uri("event_report://10.0.0.30:8080/mem");

        proto::meta::ReportEventResponse response;
        EXPECT_EQ(EC_PARTIAL_OK, cache_manager_->ReportEvent(request_context_.get(), &request, &response));
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
        EXPECT_TRUE(event_backend->GetSnapshotVersion({"test_instance", "10.0.0.30:8080"}).empty());
    }

    ASSERT_EQ(EC_OK, event_backend->Close());
}

TEST_F(CacheManagerTest, TestReportEventBlockAddMergesLocationSpecs) {
    auto expected_reg = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    const std::string instance_id = "test_report_event_merge";
    std::vector<LocationSpecInfo> location_spec_infos = {
        LocationSpecInfo("linear_0", 512),
        LocationSpecInfo("linear_1", 512),
        LocationSpecInfo("full_3", 512),
    };
    ASSERT_EQ(expected_reg,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               instance_id,
                                               64,
                                               location_spec_infos,
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    auto event_backend = std::make_shared<EventReportBackend>(cache_manager_->metrics_registry_);
    {
        StorageConfig cfg;
        cfg.set_global_unique_name("event_backend_default");
        cfg.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5);
        cfg.set_storage_spec(std::make_shared<EventReportStorageSpec>());
        event_backend->Open(cfg, "test_trace");
    }
    registry_manager_->data_storage_manager_->storage_map_["event_backend_default"] = event_backend;
    registry_manager_->instance_group_configs_["default"]->set_event_report_storage_candidates(
        {"event_backend_default"});

    const std::string host = "10.0.0.9:8080";
    const std::string snapshot_version = InitializeEventReporter(instance_id, host, proto::meta::ST_EVENT_REPORT_L1P5);
    auto report_specs =
        [&](int64_t key, const std::string &medium, const std::vector<std::vector<LocationSpec>> &spec_groups) {
            proto::meta::ReportEventRequest req;
            req.set_instance_id(instance_id);
            req.set_host_ip_port(host);
            req.set_storage_type(proto::meta::ST_EVENT_REPORT_L1P5);

            for (const auto &specs : spec_groups) {
                auto *ev = req.add_events();
                ev->set_event_type(proto::meta::EVENT_BLOCK_ADD);
                auto *ba = ev->mutable_block_add();
                ba->set_block_key(std::to_string(key));
                ba->set_medium(medium);
                for (const auto &input_spec : specs) {
                    auto *spec = ba->add_specs();
                    spec->set_name(input_spec.name());
                    spec->set_uri(input_spec.uri());
                }
            }

            proto::meta::ReportEventResponse resp;
            ASSERT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &req, &resp));
        };

    auto *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher(instance_id);
    ASSERT_NE(nullptr, meta_searcher);

    auto get_location_map = [&](int64_t key) {
        std::vector<CacheLocationMap> location_maps;
        BlockMask mask = static_cast<size_t>(0);
        EXPECT_EQ(EC_OK, meta_searcher->BatchGetLocation(request_context_.get(), {key}, mask, location_maps));
        EXPECT_EQ(1u, location_maps.size());
        return location_maps.empty() ? CacheLocationMap() : location_maps[0];
    };
    auto get_spec_uris = [](const CacheLocationConstPtr &location) {
        std::map<std::string, std::string> spec_uris;
        if (!location) {
            return spec_uris;
        }
        for (const auto &spec : location->location_specs()) {
            spec_uris[spec.name()] = spec.uri();
        }
        return spec_uris;
    };
    auto versioned_uri = [&](const std::string &raw_uri) {
        std::string result;
        EXPECT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(raw_uri, snapshot_version, result));
        return result;
    };
    const std::string mem_uri = versioned_uri("event_report://10.0.0.9:8080/mem");
    const std::string disk_uri = versioned_uri("event_report://10.0.0.9:8080/disk");

    // Case 1: one BlockAdd can create one CacheLocation with multiple specs.
    const int64_t multi_spec_key = 9001;
    report_specs(multi_spec_key,
                 "mem",
                 {{LocationSpec("linear_0", "event_report://10.0.0.9:8080/mem"),
                   LocationSpec("linear_1", "event_report://10.0.0.9:8080/mem")}});
    {
        auto location_map = get_location_map(multi_spec_key);
        ASSERT_EQ(1u, location_map.size());
        const std::string location_id = event_backend->BuildLocationId("mem", host);
        auto loc_it = location_map.find(location_id);
        ASSERT_NE(location_map.end(), loc_it);
        ASSERT_TRUE(loc_it->second);
        EXPECT_EQ(2u, loc_it->second->spec_size());
        auto spec_uris = get_spec_uris(loc_it->second);
        ASSERT_EQ(2u, spec_uris.size());
        EXPECT_EQ(mem_uri, spec_uris["linear_0"]);
        EXPECT_EQ(mem_uri, spec_uris["linear_1"]);
    }

    // Case 2: later reports append new specs and overwrite same-name specs.
    report_specs(multi_spec_key, "mem", {{LocationSpec("linear_0", "event_report://10.0.0.9:8080/mem")}});
    report_specs(multi_spec_key, "mem", {{LocationSpec("full_3", "event_report://10.0.0.9:8080/mem")}});
    {
        auto location_map = get_location_map(multi_spec_key);
        ASSERT_EQ(1u, location_map.size());
        const std::string location_id = event_backend->BuildLocationId("mem", host);
        auto loc_it = location_map.find(location_id);
        ASSERT_NE(location_map.end(), loc_it);
        ASSERT_TRUE(loc_it->second);
        EXPECT_EQ(3u, loc_it->second->spec_size());
        auto spec_uris = get_spec_uris(loc_it->second);
        ASSERT_EQ(3u, spec_uris.size());
        EXPECT_EQ(mem_uri, spec_uris["linear_0"]);
        EXPECT_EQ(mem_uri, spec_uris["linear_1"]);
        EXPECT_EQ(mem_uri, spec_uris["full_3"]);
    }

    // Case 3: multiple BlockAdd events for the same key in one request are merged before writing meta.
    const int64_t same_request_key = 9002;
    report_specs(same_request_key,
                 "mem",
                 {{LocationSpec("linear_0", "event_report://10.0.0.9:8080/mem")},
                  {LocationSpec("linear_1", "event_report://10.0.0.9:8080/mem")},
                  {LocationSpec("linear_0", "event_report://10.0.0.9:8080/mem")}});
    {
        auto location_map = get_location_map(same_request_key);
        ASSERT_EQ(1u, location_map.size());
        const std::string location_id = event_backend->BuildLocationId("mem", host);
        auto loc_it = location_map.find(location_id);
        ASSERT_NE(location_map.end(), loc_it);
        ASSERT_TRUE(loc_it->second);
        EXPECT_EQ(2u, loc_it->second->spec_size());
        auto spec_uris = get_spec_uris(loc_it->second);
        ASSERT_EQ(2u, spec_uris.size());
        EXPECT_EQ(mem_uri, spec_uris["linear_0"]);
        EXPECT_EQ(mem_uri, spec_uris["linear_1"]);
    }

    // Case 4: same key with different medium uses different location_id and does not merge into one CacheLocation.
    const int64_t multi_medium_key = 9003;
    report_specs(multi_medium_key, "mem", {{LocationSpec("linear_0", "event_report://10.0.0.9:8080/mem")}});
    report_specs(multi_medium_key, "disk", {{LocationSpec("linear_1", "event_report://10.0.0.9:8080/disk")}});
    {
        auto location_map = get_location_map(multi_medium_key);
        ASSERT_EQ(2u, location_map.size());

        const std::string mem_location_id = event_backend->BuildLocationId("mem", host);
        const std::string disk_location_id = event_backend->BuildLocationId("disk", host);
        auto mem_it = location_map.find(mem_location_id);
        auto disk_it = location_map.find(disk_location_id);
        ASSERT_NE(location_map.end(), mem_it);
        ASSERT_NE(location_map.end(), disk_it);
        ASSERT_TRUE(mem_it->second);
        ASSERT_TRUE(disk_it->second);

        auto mem_specs = get_spec_uris(mem_it->second);
        auto disk_specs = get_spec_uris(disk_it->second);
        ASSERT_EQ(1u, mem_specs.size());
        ASSERT_EQ(1u, disk_specs.size());
        EXPECT_EQ(mem_uri, mem_specs["linear_0"]);
        EXPECT_EQ(disk_uri, disk_specs["linear_1"]);
    }
}

TEST_F(CacheManagerTest, TestReportEventL1P5L2BlockAddAreIsolated) {
    auto expected_reg = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    const std::string instance_id = "test_report_event_l1p5_l2";
    std::vector<LocationSpecInfo> location_spec_infos = {
        LocationSpecInfo("linear_0", 512),
        LocationSpecInfo("linear_1", 512),
    };
    ASSERT_EQ(expected_reg,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               instance_id,
                                               64,
                                               location_spec_infos,
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    auto make_backend = [&](const std::string &name, DataStorageType type) {
        auto backend = std::make_shared<EventReportBackend>(cache_manager_->metrics_registry_);
        StorageConfig cfg;
        cfg.set_global_unique_name(name);
        cfg.set_type(type);
        cfg.set_storage_spec(std::make_shared<EventReportStorageSpec>());
        return std::make_pair(backend, backend->Open(cfg, "test_trace"));
    };

    auto l1p5_backend_result = make_backend("event_backend_l1p5", DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5);
    ASSERT_EQ(EC_OK, l1p5_backend_result.second);
    auto l2_backend_result = make_backend("event_backend_l2", DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2);
    ASSERT_EQ(EC_OK, l2_backend_result.second);
    auto l1p5_backend = l1p5_backend_result.first;
    auto l2_backend = l2_backend_result.first;
    registry_manager_->data_storage_manager_->storage_map_["event_backend_l1p5"] = l1p5_backend;
    registry_manager_->data_storage_manager_->storage_map_["event_backend_l2"] = l2_backend;
    registry_manager_->instance_group_configs_["default"]->set_event_report_storage_candidates(
        {"event_backend_l1p5", "event_backend_l2"});

    const std::string host = "10.0.0.11:8080";
    const int64_t key = 9201;
    const std::string l1p5_version = InitializeEventReporter(instance_id, host, proto::meta::ST_EVENT_REPORT_L1P5);
    const std::string l2_version = InitializeEventReporter(instance_id, host, proto::meta::ST_EVENT_REPORT_L2);
    auto report_one = [&](proto::meta::StorageType storage_type, const std::string &spec_name, const std::string &uri) {
        proto::meta::ReportEventRequest req;
        req.set_instance_id(instance_id);
        req.set_host_ip_port(host);
        req.set_storage_type(storage_type);
        auto *ev = req.add_events();
        ev->set_event_type(proto::meta::EVENT_BLOCK_ADD);
        auto *ba = ev->mutable_block_add();
        ba->set_block_key(std::to_string(key));
        ba->set_medium("mem");
        auto *spec = ba->add_specs();
        spec->set_name(spec_name);
        spec->set_uri(uri);
        proto::meta::ReportEventResponse resp;
        ASSERT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &req, &resp));
    };

    report_one(proto::meta::ST_EVENT_REPORT_L1P5, "linear_0", "event_report://10.0.0.11:8080/l1p5");
    report_one(proto::meta::ST_EVENT_REPORT_L2, "linear_1", "event_report://10.0.0.11:8080/l2");

    auto *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher(instance_id);
    ASSERT_NE(nullptr, meta_searcher);
    std::vector<CacheLocationMap> location_maps;
    BlockMask mask = static_cast<size_t>(0);
    ASSERT_EQ(EC_OK, meta_searcher->BatchGetLocation(request_context_.get(), {key}, mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    const auto &location_map = location_maps[0];
    ASSERT_EQ(2u, location_map.size());

    const std::string l1p5_location_id = l1p5_backend->BuildLocationId("mem", host);
    const std::string l2_location_id = l2_backend->BuildLocationId("mem", host);
    ASSERT_NE(l1p5_location_id, l2_location_id);
    auto l1p5_it = location_map.find(l1p5_location_id);
    auto l2_it = location_map.find(l2_location_id);
    ASSERT_NE(location_map.end(), l1p5_it);
    ASSERT_NE(location_map.end(), l2_it);
    ASSERT_TRUE(l1p5_it->second);
    ASSERT_TRUE(l2_it->second);

    EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5, l1p5_it->second->type());
    EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, l2_it->second->type());
    ASSERT_EQ(1u, l1p5_it->second->location_specs().size());
    ASSERT_EQ(1u, l2_it->second->location_specs().size());
    EXPECT_EQ("linear_0", l1p5_it->second->location_specs()[0].name());
    std::string expected_l1p5_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://10.0.0.11:8080/l1p5", l1p5_version, expected_l1p5_uri));
    EXPECT_EQ(expected_l1p5_uri, l1p5_it->second->location_specs()[0].uri());
    EXPECT_EQ("linear_1", l2_it->second->location_specs()[0].name());
    std::string expected_l2_uri;
    ASSERT_TRUE(
        SnapshotUriUtils::AddSnapshotVersionToUri("event_report://10.0.0.11:8080/l2", l2_version, expected_l2_uri));
    EXPECT_EQ(expected_l2_uri, l2_it->second->location_specs()[0].uri());
}

TEST_F(CacheManagerTest, TestReportEventBlockDeleteRemovesLocationSpecs) {
    auto expected_reg = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    const std::string instance_id = "test_report_event_delete_specs";
    std::vector<LocationSpecInfo> location_spec_infos = {
        LocationSpecInfo("linear_0", 512),
        LocationSpecInfo("linear_1", 512),
        LocationSpecInfo("full_3", 512),
    };
    ASSERT_EQ(expected_reg,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               instance_id,
                                               64,
                                               location_spec_infos,
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    auto event_backend = std::make_shared<EventReportBackend>(cache_manager_->metrics_registry_);
    {
        StorageConfig cfg;
        cfg.set_global_unique_name("event_backend_delete");
        cfg.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5);
        cfg.set_storage_spec(std::make_shared<EventReportStorageSpec>());
        event_backend->Open(cfg, "test_trace");
    }
    registry_manager_->data_storage_manager_->storage_map_["event_backend_delete"] = event_backend;
    registry_manager_->instance_group_configs_["default"]->set_event_report_storage_candidates(
        {"event_backend_delete"});

    const std::string host = "10.0.0.10:8080";
    InitializeEventReporter(instance_id, host, proto::meta::ST_EVENT_REPORT_L1P5);
    auto report_add = [&](int64_t key, const std::string &medium, const std::vector<LocationSpec> &specs) {
        proto::meta::ReportEventRequest req;
        req.set_instance_id(instance_id);
        req.set_host_ip_port(host);
        req.set_storage_type(proto::meta::ST_EVENT_REPORT_L1P5);
        auto *ev = req.add_events();
        ev->set_event_type(proto::meta::EVENT_BLOCK_ADD);
        auto *ba = ev->mutable_block_add();
        ba->set_block_key(std::to_string(key));
        ba->set_medium(medium);
        for (const auto &input_spec : specs) {
            auto *spec = ba->add_specs();
            spec->set_name(input_spec.name());
            spec->set_uri(input_spec.uri());
        }
        proto::meta::ReportEventResponse resp;
        ASSERT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &req, &resp));
    };
    auto report_delete =
        [&](int64_t key, const std::string &medium, const std::vector<std::vector<std::string>> &spec_name_groups) {
            proto::meta::ReportEventRequest req;
            req.set_instance_id(instance_id);
            req.set_host_ip_port(host);
            req.set_storage_type(proto::meta::ST_EVENT_REPORT_L1P5);
            for (const auto &spec_names : spec_name_groups) {
                auto *ev = req.add_events();
                ev->set_event_type(proto::meta::EVENT_BLOCK_DELETE);
                auto *bd = ev->mutable_block_delete();
                bd->set_block_key(std::to_string(key));
                bd->set_medium(medium);
                for (const auto &spec_name : spec_names) {
                    bd->add_spec_names(spec_name);
                }
            }
            proto::meta::ReportEventResponse resp;
            ASSERT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &req, &resp));
        };

    auto *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher(instance_id);
    ASSERT_NE(nullptr, meta_searcher);
    auto get_location_map = [&](int64_t key) {
        std::vector<CacheLocationMap> location_maps;
        BlockMask mask = static_cast<size_t>(0);
        EXPECT_EQ(EC_OK, meta_searcher->BatchGetLocation(request_context_.get(), {key}, mask, location_maps));
        EXPECT_EQ(1u, location_maps.size());
        return location_maps.empty() ? CacheLocationMap() : location_maps[0];
    };
    auto get_spec_names = [](const CacheLocationConstPtr &location) {
        std::set<std::string> spec_names;
        if (!location) {
            return spec_names;
        }
        for (const auto &spec : location->location_specs()) {
            spec_names.insert(spec.name());
        }
        return spec_names;
    };

    const int64_t partial_delete_key = 9101;
    report_add(partial_delete_key,
               "mem",
               {LocationSpec("linear_0", "event_report://10.0.0.10:8080/mem"),
                LocationSpec("linear_1", "event_report://10.0.0.10:8080/mem"),
                LocationSpec("full_3", "event_report://10.0.0.10:8080/mem")});

    report_delete(partial_delete_key, "mem", {{"linear_0"}});
    {
        auto location_map = get_location_map(partial_delete_key);
        ASSERT_EQ(1u, location_map.size());
        const auto location_id = event_backend->BuildLocationId("mem", host);
        auto loc_it = location_map.find(location_id);
        ASSERT_NE(location_map.end(), loc_it);
        EXPECT_EQ((std::set<std::string>{"linear_1", "full_3"}), get_spec_names(loc_it->second));
        EXPECT_EQ(2u, loc_it->second->spec_size());
    }

    report_delete(partial_delete_key, "mem", {{"linear_1"}, {"full_3"}});
    {
        auto location_map = get_location_map(partial_delete_key);
        EXPECT_TRUE(location_map.empty());
    }

    const int64_t multi_medium_key = 9103;
    report_add(multi_medium_key, "mem", {LocationSpec("linear_0", "event_report://10.0.0.10:8080/mem")});
    report_add(multi_medium_key, "disk", {LocationSpec("linear_1", "event_report://10.0.0.10:8080/disk")});
    report_delete(multi_medium_key, "mem", {{"linear_0"}});
    {
        auto location_map = get_location_map(multi_medium_key);
        ASSERT_EQ(1u, location_map.size());
        const auto disk_location_id = event_backend->BuildLocationId("disk", host);
        auto disk_it = location_map.find(disk_location_id);
        ASSERT_NE(location_map.end(), disk_it);
        EXPECT_EQ((std::set<std::string>{"linear_1"}), get_spec_names(disk_it->second));
    }
}

TEST_F(CacheManagerTest, TestGetCacheLocationsByBackend) {
    auto expected_reg = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    const std::string instance_id = "test_backend_selectors";
    auto location_spec_infos = createLocationSpecInfos();
    location_spec_infos.emplace_back("full_0", 512);
    location_spec_infos.emplace_back("linear_1", 512);
    location_spec_infos.emplace_back("linear_2", 512);
    location_spec_infos.emplace_back("linear_3", 512);
    ASSERT_EQ(expected_reg,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               instance_id,
                                               64,
                                               location_spec_infos,
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    std::vector<int64_t> all_keys{300, 400, 500, 600, 700};

    // Write NFS locations for a subset of keys (non-contiguous: 300, 500, 700)
    std::vector<int64_t> nfs_keys{300, 500, 700};
    {
        auto [ec, swci] =
            cache_manager_->StartWriteCache(request_context_.get(), instance_id, nfs_keys, {}, {}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        BlockMask bm = static_cast<size_t>(nfs_keys.size());
        ASSERT_EQ(EC_OK,
                  cache_manager_->FinishWriteCache(request_context_.get(), instance_id, swci.write_session_id(), bm));
    }

    // Set up EventReportBackend
    auto metrics_registry = cache_manager_->metrics_registry_;
    auto event_report_backend = std::make_shared<EventReportBackend>(metrics_registry);
    {
        StorageConfig er_config;
        er_config.set_global_unique_name("event_report_default");
        er_config.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2);
        er_config.set_storage_spec(std::make_shared<EventReportStorageSpec>());
        event_report_backend->Open(er_config, "test_trace");
    }
    auto dsm = registry_manager_->data_storage_manager_;
    dsm->storage_map_["event_report_default"] = event_report_backend;

    // Configure event_report_storage_candidates for the "default" instance group
    registry_manager_->instance_group_configs_["default"]->set_event_report_storage_candidates(
        {"event_report_default"});

    // Inject event report locations via ReportEvent
    struct PeerKeys {
        std::string host;
        std::vector<int64_t> keys;
    };
    std::vector<PeerKeys> peer_data = {
        {"192.168.1.1:8080", {300, 400, 600}},
        {"192.168.1.2:8080", {300, 400, 500, 600}},
        {"192.168.1.3:8080", {300}},
    };
    for (const auto &pd : peer_data) {
        InitializeEventReporter(instance_id, pd.host, proto::meta::ST_EVENT_REPORT_L2);
        proto::meta::ReportEventRequest req;
        req.set_instance_id(instance_id);
        req.set_host_ip_port(pd.host);
        req.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);

        for (int64_t key : pd.keys) {
            auto *ev = req.add_events();
            ev->set_event_type(proto::meta::EVENT_BLOCK_ADD);
            auto *ba = ev->mutable_block_add();
            ba->set_block_key(std::to_string(key));
            ba->set_medium("mem");
            auto *spec = ba->add_specs();
            spec->set_name("tp0");
            spec->set_uri("event_report://" + pd.host + "/mem");
        }

        proto::meta::ReportEventResponse resp;
        ASSERT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &req, &resp));
    }

    // --- Test 1: empty backend_selectors → returns EC_BADARGS ---
    {
        BlockMask bm = static_cast<size_t>(0);
        auto [ec, locs] = cache_manager_->GetCacheLocationsByBackend(
            request_context_.get(), instance_id, CacheManager::QueryType::QT_BATCH_GET, all_keys, {}, bm, 0, {}, {});
        ASSERT_EQ(EC_BADARGS, ec);
    }

    // Invalid, incompatible, and duplicate selectors fail closed instead of
    // silently behaving like weighted-random selection or duplicating output.
    for (const auto &selectors : std::vector<std::vector<BackendSelector>>{
             {{DataStorageType::DATA_STORAGE_TYPE_UNKNOWN, LocationSelectStrategy::LSS_WEIGHTED_RANDOM}},
             {{DataStorageType::DATA_STORAGE_TYPE_NFS, LocationSelectStrategy::LSS_UNSPECIFIED}},
             {{DataStorageType::DATA_STORAGE_TYPE_NFS, LocationSelectStrategy::LSS_V6D_PREFIX}},
             {{DataStorageType::DATA_STORAGE_TYPE_NFS, LocationSelectStrategy::LSS_WEIGHTED_RANDOM},
              {DataStorageType::DATA_STORAGE_TYPE_NFS, LocationSelectStrategy::LSS_WEIGHTED_RANDOM}},
         }) {
        BlockMask bm = static_cast<size_t>(0);
        auto [ec, locs] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                     instance_id,
                                                                     CacheManager::QueryType::QT_BATCH_GET,
                                                                     all_keys,
                                                                     {},
                                                                     bm,
                                                                     0,
                                                                     {},
                                                                     selectors);
        EXPECT_EQ(EC_BADARGS, ec);
        EXPECT_TRUE(locs.empty());
    }

    // Invalid masks must fail closed. An omitted protobuf mask is represented
    // as an empty bool vector and remains backward-compatible with no mask.
    const std::vector<BackendSelector> nfs_selectors = {
        {DataStorageType::DATA_STORAGE_TYPE_NFS, LocationSelectStrategy::LSS_WEIGHTED_RANDOM},
    };
    for (const BlockMask &invalid_mask : std::vector<BlockMask>{
             BlockMaskOffset{all_keys.size() + 1},
             BlockMaskVector{true, false},
         }) {
        auto [ec, locs] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                     instance_id,
                                                                     CacheManager::QueryType::QT_BATCH_GET,
                                                                     all_keys,
                                                                     {},
                                                                     invalid_mask,
                                                                     0,
                                                                     {},
                                                                     nfs_selectors);
        EXPECT_EQ(EC_BADARGS, ec);
        EXPECT_TRUE(locs.empty());
    }
    {
        const BlockMask implicit_empty_mask = BlockMaskVector{};
        auto [ec, locs] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                     instance_id,
                                                                     CacheManager::QueryType::QT_BATCH_GET,
                                                                     all_keys,
                                                                     {},
                                                                     implicit_empty_mask,
                                                                     0,
                                                                     {},
                                                                     nfs_selectors);
        EXPECT_EQ(EC_OK, ec);
        EXPECT_EQ(all_keys.size(), locs.size());
    }

    // Masked entries stay positionally aligned but do not expose a remote
    // location. Both vector and prefix-offset forms are supported.
    for (const BlockMask &mask : std::vector<BlockMask>{
             BlockMaskVector{true, false, true, false, true},
             BlockMaskOffset{2},
         }) {
        auto [ec, locs] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                     instance_id,
                                                                     CacheManager::QueryType::QT_BATCH_GET,
                                                                     all_keys,
                                                                     {},
                                                                     mask,
                                                                     0,
                                                                     {},
                                                                     nfs_selectors);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(all_keys.size(), locs.size());
        for (size_t i = 0; i < all_keys.size(); ++i) {
            if (IsIndexInMaskRange(mask, i)) {
                EXPECT_TRUE(locs[i].cache_locations_view().empty());
            }
        }
    }

    // --- Test 2: EVENT_REPORT PREFIX + NFS (NFS on 300,500,700 should not affect event report peer selection) ---
    {
        std::vector<BackendSelector> selectors = {
            {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_PREFIX},
            {DataStorageType::DATA_STORAGE_TYPE_NFS, LocationSelectStrategy::LSS_WEIGHTED_RANDOM},
        };
        BlockMask bm = static_cast<size_t>(0);
        auto [ec, locs] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                     instance_id,
                                                                     CacheManager::QueryType::QT_BATCH_GET,
                                                                     all_keys,
                                                                     {},
                                                                     bm,
                                                                     0,
                                                                     {},
                                                                     selectors);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(5u, locs.size());

        // peer_B wins with prefix=4 (keys 300,400,500,600)
        // key 300 (index 0): event report + NFS = 2
        {
            const auto &kl = locs[0].cache_locations_view();
            ASSERT_EQ(2u, kl.size());
            EXPECT_NE(std::string::npos, kl[0].location_specs()[0].uri().find("192.168.1.2"));
        }
        // key 400 (index 1): event report only = 1 (no NFS for 400)
        {
            const auto &kl = locs[1].cache_locations_view();
            ASSERT_EQ(1u, kl.size());
            EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, kl[0].type());
            EXPECT_NE(std::string::npos, kl[0].location_specs()[0].uri().find("192.168.1.2"));
        }
        // key 500 (index 2): event report + NFS = 2
        {
            const auto &kl = locs[2].cache_locations_view();
            ASSERT_EQ(2u, kl.size());
            EXPECT_NE(std::string::npos, kl[0].location_specs()[0].uri().find("192.168.1.2"));
        }
        // key 600 (index 3): event report only = 1 (no NFS for 600)
        {
            const auto &kl = locs[3].cache_locations_view();
            ASSERT_EQ(1u, kl.size());
            EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, kl[0].type());
            EXPECT_NE(std::string::npos, kl[0].location_specs()[0].uri().find("192.168.1.2"));
        }
        // key 700 (index 4): NFS only = 1 (no event report peer has this key)
        {
            const auto &kl = locs[4].cache_locations_view();
            ASSERT_EQ(1u, kl.size());
            EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, kl[0].type());
        }
    }

    // --- Test 3: EVENT_REPORT COVERAGE + NFS (NFS presence does not affect event report coverage selection) ---
    {
        std::vector<BackendSelector> selectors = {
            {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_COVERAGE},
            {DataStorageType::DATA_STORAGE_TYPE_NFS, LocationSelectStrategy::LSS_WEIGHTED_RANDOM},
        };
        BlockMask bm = static_cast<size_t>(0);
        auto [ec, locs] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                     instance_id,
                                                                     CacheManager::QueryType::QT_BATCH_GET,
                                                                     all_keys,
                                                                     {},
                                                                     bm,
                                                                     0,
                                                                     {},
                                                                     selectors);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(5u, locs.size());

        // peer_B covers most keys (300,400,500,600) = 4
        // key 300 (index 0): event report + NFS = 2
        ASSERT_EQ(2u, locs[0].cache_locations_view().size());
        EXPECT_NE(std::string::npos, locs[0].cache_locations_view()[0].location_specs()[0].uri().find("192.168.1.2"));
        // key 400 (index 1): event report only = 1
        ASSERT_EQ(1u, locs[1].cache_locations_view().size());
        EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, locs[1].cache_locations_view()[0].type());
        // key 500 (index 2): event report + NFS = 2
        ASSERT_EQ(2u, locs[2].cache_locations_view().size());
        // key 600 (index 3): event report only = 1
        ASSERT_EQ(1u, locs[3].cache_locations_view().size());
        EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, locs[3].cache_locations_view()[0].type());
        // key 700 (index 4): NFS only = 1
        ASSERT_EQ(1u, locs[4].cache_locations_view().size());
        EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, locs[4].cache_locations_view()[0].type());
    }

    // --- Test 4: NFS WEIGHTED_RANDOM only (only keys 300,500,700 have NFS) ---
    {
        std::vector<BackendSelector> selectors = {
            {DataStorageType::DATA_STORAGE_TYPE_NFS, LocationSelectStrategy::LSS_WEIGHTED_RANDOM},
        };
        BlockMask bm = static_cast<size_t>(0);
        auto [ec, locs] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                     instance_id,
                                                                     CacheManager::QueryType::QT_BATCH_GET,
                                                                     all_keys,
                                                                     {},
                                                                     bm,
                                                                     0,
                                                                     {},
                                                                     selectors);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(5u, locs.size());
        // key 300 (index 0): has NFS
        ASSERT_EQ(1u, locs[0].cache_locations_view().size());
        EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, locs[0].cache_locations_view()[0].type());
        // key 400 (index 1): no NFS
        EXPECT_TRUE(locs[1].cache_locations_view().empty());
        // key 500 (index 2): has NFS
        ASSERT_EQ(1u, locs[2].cache_locations_view().size());
        EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, locs[2].cache_locations_view()[0].type());
        // key 600 (index 3): no NFS
        EXPECT_TRUE(locs[3].cache_locations_view().empty());
        // key 700 (index 4): has NFS
        ASSERT_EQ(1u, locs[4].cache_locations_view().size());
        EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, locs[4].cache_locations_view()[0].type());
    }

    // --- Test 5: PREFIX stops when first key has no event report, but NFS still works ---
    // keys = {700, 300, 400}; NFS exists for 700 and 300, not for 400
    {
        std::vector<int64_t> keys_no_er_first = {700, 300, 400};
        std::vector<BackendSelector> selectors = {
            {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_PREFIX},
            {DataStorageType::DATA_STORAGE_TYPE_NFS, LocationSelectStrategy::LSS_WEIGHTED_RANDOM},
        };
        BlockMask bm = static_cast<size_t>(0);
        auto [ec, locs] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                     instance_id,
                                                                     CacheManager::QueryType::QT_BATCH_GET,
                                                                     keys_no_er_first,
                                                                     {},
                                                                     bm,
                                                                     0,
                                                                     {},
                                                                     selectors);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(3u, locs.size());
        // EVENT_REPORT PREFIX stops at key 700 → no event report for any key
        // key 700 (index 0): NFS only = 1
        ASSERT_EQ(1u, locs[0].cache_locations_view().size());
        EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, locs[0].cache_locations_view()[0].type());
        // key 300 (index 1): NFS only = 1 (event report blocked by prefix)
        ASSERT_EQ(1u, locs[1].cache_locations_view().size());
        EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, locs[1].cache_locations_view()[0].type());
        // key 400 (index 2): nothing (no event report from prefix, no NFS written)
        EXPECT_TRUE(locs[2].cache_locations_view().empty());
    }

    // --- Test 6: COVERAGE skips keys with no event report, NFS fills gaps independently ---
    // keys = {700, 300, 400}; NFS exists for 700 and 300, not for 400
    {
        std::vector<int64_t> keys_gap = {700, 300, 400};
        std::vector<BackendSelector> selectors = {
            {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_COVERAGE},
            {DataStorageType::DATA_STORAGE_TYPE_NFS, LocationSelectStrategy::LSS_WEIGHTED_RANDOM},
        };
        BlockMask bm = static_cast<size_t>(0);
        auto [ec, locs] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                     instance_id,
                                                                     CacheManager::QueryType::QT_BATCH_GET,
                                                                     keys_gap,
                                                                     {},
                                                                     bm,
                                                                     0,
                                                                     {},
                                                                     selectors);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(3u, locs.size());
        // key 700 (index 0): NFS only = 1 (no event report peer)
        ASSERT_EQ(1u, locs[0].cache_locations_view().size());
        EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, locs[0].cache_locations_view()[0].type());
        // key 300 (index 1): event report + NFS = 2
        ASSERT_EQ(2u, locs[1].cache_locations_view().size());
        // key 400 (index 2): event report only = 1 (no NFS for 400)
        ASSERT_EQ(1u, locs[2].cache_locations_view().size());
        EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, locs[2].cache_locations_view()[0].type());
    }

    // --- Test 7: nonexistent keys → all empty ---
    {
        std::vector<int64_t> bad_keys = {99998, 99999};
        std::vector<BackendSelector> selectors = {
            {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_PREFIX},
            {DataStorageType::DATA_STORAGE_TYPE_NFS, LocationSelectStrategy::LSS_WEIGHTED_RANDOM},
        };
        BlockMask bm = static_cast<size_t>(0);
        auto [ec, locs] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                     instance_id,
                                                                     CacheManager::QueryType::QT_BATCH_GET,
                                                                     bad_keys,
                                                                     {},
                                                                     bm,
                                                                     0,
                                                                     {},
                                                                     selectors);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(2u, locs.size());
        EXPECT_TRUE(locs[0].cache_locations_view().empty());
        EXPECT_TRUE(locs[1].cache_locations_view().empty());
    }

    // --- Test 8: non-empty location_spec_names must align one-to-one with query keys ---
    {
        std::vector<BackendSelector> selectors = {
            {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_COVERAGE},
        };
        BlockMask bm = static_cast<size_t>(0);
        auto [size_ec, size_locs] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                               instance_id,
                                                                               CacheManager::QueryType::QT_BATCH_GET,
                                                                               {300, 400, 300},
                                                                               {},
                                                                               bm,
                                                                               0,
                                                                               {"full_0", "linear_1"},
                                                                               selectors);
        EXPECT_EQ(EC_BADARGS, size_ec);
        EXPECT_TRUE(size_locs.empty());

        auto [empty_ec, empty_locs] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                                 instance_id,
                                                                                 CacheManager::QueryType::QT_BATCH_GET,
                                                                                 {300, 400, 300},
                                                                                 {},
                                                                                 bm,
                                                                                 0,
                                                                                 {"full_0", "", "linear_1"},
                                                                                 selectors);
        EXPECT_EQ(EC_BADARGS, empty_ec);
        EXPECT_TRUE(empty_locs.empty());
    }

    // --- Test 9: non-hybrid attention sends one full spec per key and uses prefix selection ---
    // Current group-aware Vineyard represents each FullAttention object as
    // (block key, full_0), preserving the original query order.
    {
        auto report_full_keys = [&](const std::string &host, const std::vector<int64_t> &keys) {
            proto::meta::ReportEventRequest req;
            req.set_instance_id(instance_id);
            req.set_host_ip_port(host);
            req.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
            for (int64_t key : keys) {
                auto *ev = req.add_events();
                ev->set_event_type(proto::meta::EVENT_BLOCK_ADD);
                auto *ba = ev->mutable_block_add();
                ba->set_block_key(std::to_string(key));
                ba->set_medium("mem");
                auto *spec = ba->add_specs();
                spec->set_name("full_0");
                spec->set_uri("event_report://" + host + "/mem");
            }
            proto::meta::ReportEventResponse resp;
            ASSERT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &req, &resp));
        };

        const std::string full_prefix_host = "192.168.2.3:8080";
        const std::string short_full_host = "192.168.2.4:8080";
        InitializeEventReporter(instance_id, full_prefix_host, proto::meta::ST_EVENT_REPORT_L2);
        InitializeEventReporter(instance_id, short_full_host, proto::meta::ST_EVENT_REPORT_L2);
        report_full_keys(full_prefix_host, {900, 901});
        report_full_keys(short_full_host, {900});

        const std::vector<BackendSelector> selectors = {
            {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_PREFIX},
        };
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, locs] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                     instance_id,
                                                                     CacheManager::QueryType::QT_BATCH_GET,
                                                                     {900, 901, 902},
                                                                     {},
                                                                     block_mask,
                                                                     0,
                                                                     {"full_0", "full_0", "full_0"},
                                                                     selectors);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(3u, locs.size());
        for (size_t i = 0; i < 2; ++i) {
            const auto &key_locations = locs[i].cache_locations_view();
            ASSERT_EQ(1u, key_locations.size()) << "query index=" << i;
            ASSERT_EQ(1u, key_locations[0].location_specs().size()) << "query index=" << i;
            EXPECT_EQ(1u, key_locations[0].spec_size()) << "query index=" << i;
            EXPECT_EQ("full_0", key_locations[0].location_specs()[0].name()) << "query index=" << i;
            EXPECT_NE(std::string::npos, key_locations[0].location_specs()[0].uri().find(full_prefix_host))
                << "query index=" << i;
        }
        EXPECT_TRUE(locs[2].cache_locations_view().empty());
    }

    // --- Test 10: mixed-attention Mamba groups share one ordered best-effort query ---
    // Current Vineyard sends all group-aware objects from one lookup in their
    // original order. Different Mamba groups can therefore repeat the same block
    // key and are distinguished only by per-position location_spec_names.
    {
        auto report_specs = [&](const std::string &host, int64_t key, const std::vector<std::string> &spec_names) {
            proto::meta::ReportEventRequest req;
            req.set_instance_id(instance_id);
            req.set_host_ip_port(host);
            req.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
            auto *ev = req.add_events();
            ev->set_event_type(proto::meta::EVENT_BLOCK_ADD);
            auto *ba = ev->mutable_block_add();
            ba->set_block_key(std::to_string(key));
            ba->set_medium("mem");
            for (const auto &spec_name : spec_names) {
                auto *spec = ba->add_specs();
                spec->set_name(spec_name);
                spec->set_uri("event_report://" + host + "/mem");
            }
            proto::meta::ReportEventResponse resp;
            ASSERT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &req, &resp));
        };

        const std::string linear_1_host = "192.168.2.1:8080";
        const std::string remaining_groups_host = "192.168.2.2:8080";
        InitializeEventReporter(instance_id, linear_1_host, proto::meta::ST_EVENT_REPORT_L2);
        InitializeEventReporter(instance_id, remaining_groups_host, proto::meta::ST_EVENT_REPORT_L2);
        report_specs(linear_1_host, 800, {"linear_1"});
        report_specs(linear_1_host, 801, {"linear_1"});
        report_specs(remaining_groups_host, 800, {"linear_2", "linear_3"});
        report_specs(remaining_groups_host, 801, {"linear_3"});

        const std::vector<BackendSelector> selectors = {
            {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_COVERAGE},
        };
        BlockMask block_mask = static_cast<size_t>(0);
        auto [ec, locs] =
            cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                       instance_id,
                                                       CacheManager::QueryType::QT_BATCH_GET,
                                                       {800, 801, 800, 800, 801},
                                                       {},
                                                       block_mask,
                                                       0,
                                                       {"linear_1", "linear_1", "linear_2", "linear_3", "linear_3"},
                                                       selectors);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(5u, locs.size());
        EXPECT_TRUE(locs[0].cache_locations_view().empty());
        EXPECT_TRUE(locs[1].cache_locations_view().empty());

        const std::vector<std::string> expected_specs = {"linear_2", "linear_3", "linear_3"};
        for (size_t i = 0; i < expected_specs.size(); ++i) {
            const auto &key_locations = locs[i + 2].cache_locations_view();
            ASSERT_EQ(1u, key_locations.size()) << "query index=" << i + 2;
            ASSERT_EQ(1u, key_locations[0].location_specs().size()) << "query index=" << i + 2;
            EXPECT_EQ(1u, key_locations[0].spec_size()) << "query index=" << i + 2;
            EXPECT_EQ(expected_specs[i], key_locations[0].location_specs()[0].name()) << "query index=" << i + 2;
            EXPECT_NE(std::string::npos, key_locations[0].location_specs()[0].uri().find(remaining_groups_host))
                << "query index=" << i + 2;
        }
    }

    // --- Test 11: backend-selected queries enforce reporter liveness ---
    {
        for (const auto &peer : peer_data) {
            event_report_backend->SetNodeUnavailable(instance_id, peer.host);
        }
        const std::vector<BackendSelector> selectors = {
            {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_PREFIX},
        };
        BlockMask block_mask = static_cast<size_t>(0);
        auto [hidden_ec, hidden] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                              instance_id,
                                                                              CacheManager::QueryType::QT_BATCH_GET,
                                                                              all_keys,
                                                                              {},
                                                                              block_mask,
                                                                              0,
                                                                              {},
                                                                              selectors);
        ASSERT_EQ(EC_OK, hidden_ec);
        ASSERT_EQ(all_keys.size(), hidden.size());
        for (const auto &location : hidden) {
            EXPECT_TRUE(location.cache_locations_view().empty());
        }

        for (const auto &peer : peer_data) {
            ASSERT_EQ(EC_OK, event_report_backend->OnHeartbeat(instance_id, peer.host, {}));
        }
        auto [restored_ec, restored] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                                  instance_id,
                                                                                  CacheManager::QueryType::QT_BATCH_GET,
                                                                                  all_keys,
                                                                                  {},
                                                                                  block_mask,
                                                                                  0,
                                                                                  {},
                                                                                  selectors);
        ASSERT_EQ(EC_OK, restored_ec);
        ASSERT_EQ(all_keys.size(), restored.size());
        EXPECT_FALSE(restored[0].cache_locations_view().empty());
        EXPECT_FALSE(restored[1].cache_locations_view().empty());
        EXPECT_FALSE(restored[2].cache_locations_view().empty());
        EXPECT_FALSE(restored[3].cache_locations_view().empty());
        EXPECT_TRUE(restored[4].cache_locations_view().empty());
    }

    dsm->storage_map_.erase("event_report_default");
}

// =============================================================
// GetHostCacheState — per-host prefix match length
// =============================================================
//
// Data layout:
//   3 hosts: A (10.0.0.1:8080), B (10.0.0.2:8080), C (10.0.0.3:8080)
//   key 100: host_A, host_B, host_C
//   key 200: host_A, host_B
//   key 300: host_B only
//   key 400: host_A, host_B
//   key 500: no host
//
// Query keys = {100, 200, 300, 400, 500}
//   host_A: 100→200→(miss 300) → prefix=2
//   host_B: 100→200→300→400→(miss 500) → prefix=4
//   host_C: 100→(miss 200) → prefix=1
//
TEST_F(CacheManagerTest, TestGetHostCacheStateSnapshotsHostLivenessAfterMetadataRead) {
    auto event_backend = InstallEventReportBackend();
    ASSERT_TRUE(event_backend);
    auto *meta_backend = InstallControllableMetaBackend();
    ASSERT_TRUE(meta_backend);

    const std::string instance_id = "test_instance";
    const std::string host = "10.0.9.1:8080";
    InitializeEventReporter(instance_id, host, proto::meta::ST_EVENT_REPORT_L2);
    proto::meta::ReportEventRequest report;
    report.set_instance_id(instance_id);
    report.set_host_ip_port(host);
    report.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
    auto *event = report.add_events();
    event->set_event_type(proto::meta::EVENT_BLOCK_ADD);
    event->mutable_block_add()->set_block_key("9001");
    event->mutable_block_add()->set_medium("mem");
    auto *spec = event->mutable_block_add()->add_specs();
    spec->set_name("tp0");
    spec->set_uri("event_report://" + host + "/mem");
    proto::meta::ReportEventResponse report_response;
    ASSERT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &report, &report_response));

    meta_backend->BlockNextLocationRead();
    auto query = std::async(std::launch::async, [&] {
        RequestContext context("host_liveness_after_meta_read");
        return cache_manager_->GetHostCacheState(
            &context, instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, {9001});
    });
    const bool read_entered = meta_backend->WaitUntilLocationReadEntered(std::chrono::seconds(2));
    if (!read_entered) {
        meta_backend->ReleaseLocationRead();
        (void)query.get();
        FAIL() << "GetHostCacheState did not enter the controlled metadata read";
    }

    event_backend->SetNodeUnavailable(instance_id, host);
    meta_backend->ReleaseLocationRead();
    auto [ec, hosts] = query.get();
    EXPECT_EQ(EC_OK, ec);
    EXPECT_TRUE(hosts.empty());
}

TEST_F(CacheManagerTest, TestGetHostCacheStateConcurrentWithReportEventAndHostDown) {
    auto event_backend = InstallEventReportBackend();
    ASSERT_TRUE(event_backend);
    const std::string instance_id = "test_instance";
    const std::string host = "10.0.9.2:8080";
    constexpr std::size_t kBlockCount = 384;
    InitializeEventReporter(instance_id, host, proto::meta::ST_EVENT_REPORT_L2);

    auto make_add_request = [&](std::size_t round) {
        proto::meta::ReportEventRequest request;
        request.set_instance_id(instance_id);
        request.set_host_ip_port(host);
        request.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
        for (std::size_t i = 0; i < kBlockCount; ++i) {
            auto *event = request.add_events();
            event->set_event_type(proto::meta::EVENT_BLOCK_ADD);
            auto *add = event->mutable_block_add();
            add->set_block_key(std::to_string(10000 + i));
            add->set_medium("mem");
            auto *spec = add->add_specs();
            spec->set_name("tp0");
            spec->set_uri("event_report://" + host + "/mem?round=" + std::to_string(round));
        }
        return request;
    };

    auto setup_request = make_add_request(0);
    proto::meta::ReportEventResponse setup_response;
    ASSERT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &setup_request, &setup_response));
    CacheManager::KeyVector keys;
    keys.reserve(kBlockCount);
    for (std::size_t i = 0; i < kBlockCount; ++i) {
        keys.push_back(10000 + i);
    }

    std::atomic<bool> start{false};
    std::atomic<bool> writer_done{false};
    std::atomic<std::size_t> failures{0};
    std::thread writer([&] {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (std::size_t round = 1; round <= 12; ++round) {
            auto request = make_add_request(round);
            proto::meta::ReportEventResponse response;
            RequestContext context("concurrent_report_" + std::to_string(round));
            if (cache_manager_->ReportEvent(&context, &request, &response) != EC_OK) {
                failures.fetch_add(1, std::memory_order_relaxed);
            }
        }
        writer_done.store(true, std::memory_order_release);
    });
    std::thread reader([&] {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::size_t query_count = 0;
        while (!writer_done.load(std::memory_order_acquire) || query_count < 24) {
            RequestContext context("concurrent_host_query_" + std::to_string(query_count));
            auto [ec, hosts] = cache_manager_->GetHostCacheState(
                &context, instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, keys, {"mem"});
            if (ec != EC_OK || hosts.size() != 1 || hosts.front().host_ip_port != host ||
                hosts.front().local != static_cast<int64_t>(kBlockCount)) {
                failures.fetch_add(1, std::memory_order_relaxed);
            }
            ++query_count;
        }
    });
    start.store(true, std::memory_order_release);
    writer.join();
    reader.join();
    EXPECT_EQ(0u, failures.load(std::memory_order_relaxed));

    proto::meta::ReportEventRequest host_down;
    host_down.set_instance_id(instance_id);
    host_down.set_host_ip_port(host);
    host_down.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
    auto *host_down_event = host_down.add_events();
    host_down_event->set_event_type(proto::meta::EVENT_HOST_DOWN);
    host_down_event->mutable_host_down();
    proto::meta::ReportEventResponse host_down_response;
    ASSERT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &host_down, &host_down_response));
    auto [ec, hosts] = cache_manager_->GetHostCacheState(
        request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, keys);
    EXPECT_EQ(EC_OK, ec);
    EXPECT_TRUE(hosts.empty());
}

TEST_F(CacheManagerTest, TestGetHostCacheState) {
    auto expected_reg = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    const std::string instance_id = "test_host_cache_state_prefix";
    ASSERT_EQ(expected_reg,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               instance_id,
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>(),
                                               CacheManager::QueryType::QT_PREFIX_MATCH));

    // Set up EventReportBackend so that location_ids carry host_ip_port
    auto metrics_registry = cache_manager_->metrics_registry_;
    auto event_backend = std::make_shared<EventReportBackend>(metrics_registry);
    {
        StorageConfig cfg;
        cfg.set_global_unique_name("event_backend_default");
        cfg.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5);
        cfg.set_storage_spec(std::make_shared<EventReportStorageSpec>());
        event_backend->Open(cfg, "test_trace");
    }
    auto dsm = registry_manager_->data_storage_manager_;
    dsm->storage_map_["event_backend_default"] = event_backend;
    registry_manager_->instance_group_configs_["default"]->set_event_report_storage_candidates(
        {"event_backend_default"});

    // Inject cache locations via ReportEvent — each host reports a subset of keys
    struct HostKeys {
        std::string host;
        std::vector<int64_t> keys;
    };
    std::vector<HostKeys> host_data = {
        {"10.0.0.1:8080", {100, 200, 400}},
        {"10.0.0.2:8080", {100, 200, 300, 400}},
        {"10.0.0.3:8080", {100}},
        {"10.0.0.4:8080", {200, 300}},
    };
    for (const auto &hd : host_data) {
        InitializeEventReporter(instance_id, hd.host, proto::meta::ST_EVENT_REPORT_L1P5);
        proto::meta::ReportEventRequest req;
        req.set_instance_id(instance_id);
        req.set_host_ip_port(hd.host);
        req.set_storage_type(proto::meta::ST_EVENT_REPORT_L1P5);

        for (int64_t key : hd.keys) {
            auto *ev = req.add_events();
            ev->set_event_type(proto::meta::EVENT_BLOCK_ADD);
            auto *ba = ev->mutable_block_add();
            ba->set_block_key(std::to_string(key));
            ba->set_medium("mem");
            auto *spec = ba->add_specs();
            spec->set_name("tp0");
            spec->set_uri("event_report://" + hd.host + "/mem");
        }

        proto::meta::ReportEventResponse resp;
        ASSERT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &req, &resp));
    }

    // Helper: find a host's local in the result
    auto find_prefix = [](const std::vector<CacheManager::HostCacheMatch> &hosts, const std::string &host) -> int64_t {
        for (const auto &h : hosts) {
            if (h.host_ip_port == host) {
                return h.local;
            }
        }
        return -1; // not found
    };

    // --- Test 1: full query — different prefix lengths per host ---
    // keys = {100, 200, 300, 400, 500}
    //   host_A: prefix=2 (100,200; miss 300)
    //   host_B: prefix=4 (100,200,300,400; miss 500)
    //   host_C: prefix=1 (100; miss 200)
    {
        CacheManager::KeyVector keys = {100, 200, 300, 400, 500};
        auto [ec, hosts] = cache_manager_->GetHostCacheState(
            request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, keys);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(3, hosts.size());

        EXPECT_EQ(2, find_prefix(hosts, "10.0.0.1:8080"));
        EXPECT_EQ(4, find_prefix(hosts, "10.0.0.2:8080"));
        EXPECT_EQ(1, find_prefix(hosts, "10.0.0.3:8080"));
    }

    // --- Test 1b: unspecified query type falls back to RegisterInstance.default_query_type ---
    {
        CacheManager::KeyVector keys = {100, 200, 300, 400, 500};
        auto [ec, hosts] = cache_manager_->GetHostCacheState(
            request_context_.get(), instance_id, CacheManager::QueryType::QT_UNSPECIFIED, keys);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(3, hosts.size());

        EXPECT_EQ(2, find_prefix(hosts, "10.0.0.1:8080"));
        EXPECT_EQ(4, find_prefix(hosts, "10.0.0.2:8080"));
        EXPECT_EQ(1, find_prefix(hosts, "10.0.0.3:8080"));
    }

    // --- Test 2: all keys cached by host_B → prefix = full length ---
    // keys = {100, 200, 300, 400}
    //   host_A: prefix=2 (miss 300)
    //   host_B: prefix=4 (all matched)
    //   host_C: prefix=1 (miss 200)
    {
        CacheManager::KeyVector keys = {100, 200, 300, 400};
        auto [ec, hosts] = cache_manager_->GetHostCacheState(
            request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, keys);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(3, hosts.size());

        EXPECT_EQ(2, find_prefix(hosts, "10.0.0.1:8080"));
        EXPECT_EQ(4, find_prefix(hosts, "10.0.0.2:8080"));
        EXPECT_EQ(1, find_prefix(hosts, "10.0.0.3:8080"));
    }

    // --- Test 3: single key — all hosts have prefix=1 ---
    {
        CacheManager::KeyVector keys = {100};
        auto [ec, hosts] = cache_manager_->GetHostCacheState(
            request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, keys);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(3, hosts.size());

        EXPECT_EQ(1, find_prefix(hosts, "10.0.0.1:8080"));
        EXPECT_EQ(1, find_prefix(hosts, "10.0.0.2:8080"));
        EXPECT_EQ(1, find_prefix(hosts, "10.0.0.3:8080"));
    }

    // --- Test 4: first key not cached by any host → empty response ---
    // keys = {999, 100, 200}
    {
        CacheManager::KeyVector keys = {999, 100, 200};
        auto [ec, hosts] = cache_manager_->GetHostCacheState(
            request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, keys);
        ASSERT_EQ(EC_OK, ec);
        EXPECT_EQ(0, hosts.size());
    }

    // --- Test 5: medium filter (only "mem") — should not change results ---
    {
        CacheManager::KeyVector keys = {100, 200, 300, 400, 500};
        auto [ec, hosts] = cache_manager_->GetHostCacheState(
            request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, keys, {"mem"});
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(3, hosts.size());

        EXPECT_EQ(2, find_prefix(hosts, "10.0.0.1:8080"));
        EXPECT_EQ(4, find_prefix(hosts, "10.0.0.2:8080"));
        EXPECT_EQ(1, find_prefix(hosts, "10.0.0.3:8080"));
    }

    // --- Test 6: medium filter (non-existent medium "ssd") → no hosts ---
    {
        CacheManager::KeyVector keys = {100, 200};
        auto [ec, hosts] = cache_manager_->GetHostCacheState(
            request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, keys, {"ssd"});
        ASSERT_EQ(EC_OK, ec);
        EXPECT_EQ(0, hosts.size());
    }

    // --- Test 7: middle miss stops prefix; later hits do not extend any host ---
    {
        CacheManager::KeyVector keys = {100, 500, 400};
        auto [ec, hosts] = cache_manager_->GetHostCacheState(
            request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, keys);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(3, hosts.size());

        EXPECT_EQ(1, find_prefix(hosts, "10.0.0.1:8080"));
        EXPECT_EQ(1, find_prefix(hosts, "10.0.0.2:8080"));
        EXPECT_EQ(1, find_prefix(hosts, "10.0.0.3:8080"));
        EXPECT_EQ(-1, find_prefix(hosts, "10.0.0.4:8080"));
    }

    // --- Test 8: hosts absent from the first key are not returned with prefix=0 ---
    {
        CacheManager::KeyVector keys = {100, 200, 300};
        auto [ec, hosts] = cache_manager_->GetHostCacheState(
            request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, keys);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(3, hosts.size());

        EXPECT_EQ(2, find_prefix(hosts, "10.0.0.1:8080"));
        EXPECT_EQ(3, find_prefix(hosts, "10.0.0.2:8080"));
        EXPECT_EQ(1, find_prefix(hosts, "10.0.0.3:8080"));
        EXPECT_EQ(-1, find_prefix(hosts, "10.0.0.4:8080"));
    }

    // --- Test 9: requests above the parallel threshold preserve ordering and
    // prefix semantics. Repeated keys also stress concurrent reads of the same
    // local-cache item rather than only independent LRU shards. ---
    {
        CacheManager::KeyVector keys;
        keys.reserve(384);
        for (std::size_t i = 0; i < 96; ++i) {
            keys.insert(keys.end(), {100, 200, 300, 400});
        }
        auto [ec, hosts] = cache_manager_->GetHostCacheState(
            request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, keys);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(3u, hosts.size());
        EXPECT_EQ("10.0.0.1:8080", hosts[0].host_ip_port);
        EXPECT_EQ("10.0.0.2:8080", hosts[1].host_ip_port);
        EXPECT_EQ("10.0.0.3:8080", hosts[2].host_ip_port);
        EXPECT_EQ(2, find_prefix(hosts, "10.0.0.1:8080"));
        EXPECT_EQ(384, find_prefix(hosts, "10.0.0.2:8080"));
        EXPECT_EQ(1, find_prefix(hosts, "10.0.0.3:8080"));
    }

    // --- Test 10: unavailable host is filtered even before metadata cleanup ---
    {
        event_backend->SetNodeUnavailable(instance_id, "10.0.0.2:8080");
        CacheManager::KeyVector keys = {100, 200, 300, 400};
        auto [ec, hosts] = cache_manager_->GetHostCacheState(
            request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, keys);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(2, hosts.size());

        EXPECT_EQ(2, find_prefix(hosts, "10.0.0.1:8080"));
        EXPECT_EQ(-1, find_prefix(hosts, "10.0.0.2:8080"));
        EXPECT_EQ(1, find_prefix(hosts, "10.0.0.3:8080"));
    }

    dsm->storage_map_.erase("event_backend_default");
}

TEST_F(CacheManagerTest, TestGetHostCacheStateP2P) {
    auto expected_reg = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    const std::string instance_id = "test_host_cache_state_single_p2p";
    ASSERT_EQ(expected_reg,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               instance_id,
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>(),
                                               CacheManager::QueryType::QT_PREFIX_MATCH));

    auto make_backend = [&](const std::string &name, DataStorageType type) {
        auto backend = std::make_shared<EventReportBackend>(cache_manager_->metrics_registry_);
        StorageConfig cfg;
        cfg.set_global_unique_name(name);
        cfg.set_type(type);
        cfg.set_storage_spec(std::make_shared<EventReportStorageSpec>());
        return std::make_pair(backend, backend->Open(cfg, "test_trace"));
    };
    auto subscriber_backend_result =
        make_backend("host_state_subscriber", DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5);
    auto vineyard_backend_result =
        make_backend("host_state_vineyard", DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2);
    ASSERT_EQ(EC_OK, subscriber_backend_result.second);
    ASSERT_EQ(EC_OK, vineyard_backend_result.second);

    auto dsm = registry_manager_->data_storage_manager_;
    dsm->storage_map_["host_state_subscriber"] = subscriber_backend_result.first;
    dsm->storage_map_["host_state_vineyard"] = vineyard_backend_result.first;
    registry_manager_->instance_group_configs_["default"]->set_event_report_storage_candidates(
        {"host_state_subscriber", "host_state_vineyard"});

    auto report_keys = [&](proto::meta::StorageType type, const std::string &host, const std::vector<int64_t> &keys) {
        InitializeEventReporter(instance_id, host, type);
        proto::meta::ReportEventRequest req;
        req.set_instance_id(instance_id);
        req.set_host_ip_port(host);
        req.set_storage_type(type);
        for (int64_t key : keys) {
            auto *event = req.add_events();
            event->set_event_type(proto::meta::EVENT_BLOCK_ADD);
            auto *block_add = event->mutable_block_add();
            block_add->set_block_key(std::to_string(key));
            block_add->set_medium("mem");
            auto *spec = block_add->add_specs();
            spec->set_name("tp0");
            spec->set_uri("event_report://" + host + "/mem");
        }
        proto::meta::ReportEventResponse resp;
        ASSERT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &req, &resp));
    };

    const std::string host_a = "10.0.2.1:8080";
    const std::string host_b = "10.0.2.2:8080";
    const std::string host_c = "10.0.2.3:8080";
    report_keys(proto::meta::ST_EVENT_REPORT_L1P5, host_a, {100, 300});
    report_keys(proto::meta::ST_EVENT_REPORT_L2, host_b, {100, 200, 400});
    report_keys(proto::meta::ST_EVENT_REPORT_L2, host_c, {100, 200, 300});

    auto [ec, hosts] = cache_manager_->GetHostCacheState(
        request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, {100, 200, 300, 400, 500});
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(3u, hosts.size());

    // host A selects B for local-miss keys {200, 400}; host B selects C for {300};
    // host C selects B for {400}. All three stop at the uncached key 500.
    auto expect_match = [&](const std::string &host, int64_t local, int64_t p2p_1_fetch, int64_t p2p_1_total_match) {
        auto it =
            std::find_if(hosts.begin(), hosts.end(), [&](const auto &match) { return match.host_ip_port == host; });
        ASSERT_NE(hosts.end(), it);
        EXPECT_EQ(local, it->local);
        EXPECT_EQ(p2p_1_fetch, it->p2p_1_fetch);
        EXPECT_EQ(p2p_1_total_match, it->p2p_1_total_match);
    };
    expect_match(host_a, 1, 2, 4);
    expect_match(host_b, 2, 1, 4);
    expect_match(host_c, 3, 1, 4);

    // Only the five hosts with the largest local prefix compute P2P. Hosts
    // after the cutoff are still returned in host order with local-only totals.
    const std::vector<int64_t> top5_keys = {1000, 1001, 1002, 1003, 1004, 1005, 1006};
    const std::vector<std::pair<std::string, size_t>> top5_hosts = {
        {"10.0.5.1:8080", 7},
        {"10.0.5.2:8080", 6},
        {"10.0.5.3:8080", 5},
        {"10.0.5.4:8080", 4},
        {"10.0.5.5:8080", 3},
        {"10.0.5.6:8080", 3},
        {"10.0.5.7:8080", 1},
    };
    for (size_t i = 0; i < top5_hosts.size(); ++i) {
        const auto &[host, prefix_len] = top5_hosts[i];
        report_keys(i == 0 ? proto::meta::ST_EVENT_REPORT_L2 : proto::meta::ST_EVENT_REPORT_L1P5,
                    host,
                    std::vector<int64_t>(top5_keys.begin(), top5_keys.begin() + prefix_len));
    }
    const std::string zero_local_host = "10.0.5.8:8080";
    report_keys(proto::meta::ST_EVENT_REPORT_L1P5, zero_local_host, {top5_keys[1]});

    auto [top5_ec, top5_matches] = cache_manager_->GetHostCacheState(
        request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, top5_keys);
    ASSERT_EQ(EC_OK, top5_ec);
    ASSERT_EQ(top5_hosts.size(), top5_matches.size());
    const std::vector<std::tuple<int64_t, int64_t, int64_t>> expected_top5_matches = {
        {7, 0, 7},
        {6, 1, 7},
        {5, 2, 7},
        {4, 3, 7},
        {3, 4, 7},
        {3, 0, 3},
        {1, 0, 1},
    };
    for (size_t i = 0; i < top5_hosts.size(); ++i) {
        EXPECT_EQ(top5_hosts[i].first, top5_matches[i].host_ip_port);
        EXPECT_EQ(std::get<0>(expected_top5_matches[i]), top5_matches[i].local);
        EXPECT_EQ(std::get<1>(expected_top5_matches[i]), top5_matches[i].p2p_1_fetch);
        EXPECT_EQ(std::get<2>(expected_top5_matches[i]), top5_matches[i].p2p_1_total_match);
    }
    EXPECT_EQ(top5_matches.end(), std::find_if(top5_matches.begin(), top5_matches.end(), [&](const auto &match) {
                  return match.host_ip_port == zero_local_host;
              }));

    // Prefix match with Mamba: local and P2P specs jointly complete the blocks,
    // then the merged prefix is evaluated with the Mamba state and Eagle POP rules.
    const std::string mamba_instance_id = "test_host_cache_state_mamba_p2p";
    std::vector<LocationSpecInfo> mamba_location_spec_infos = {
        LocationSpecInfo("full_0", 512),
        LocationSpecInfo("linear_0", 512),
        LocationSpecInfo("linear_1", 512),
    };
    std::vector<LocationSpecGroup> mamba_location_spec_groups = {
        LocationSpecGroup("full_0", {"full_0"}),
        LocationSpecGroup("linear_0", {"linear_0"}),
        LocationSpecGroup("linear_1", {"linear_1"}),
    };
    ASSERT_EQ(expected_reg,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               mamba_instance_id,
                                               64,
                                               mamba_location_spec_infos,
                                               createModelDeploymentWithEaglePop(),
                                               mamba_location_spec_groups,
                                               CacheManager::QueryType::QT_PREFIX_MATCH_WITH_MAMBA));

    const std::string mamba_host = "10.0.3.1:8080";
    const std::string mamba_p2p_host = "10.0.3.2:8080";
    InitializeEventReporter(mamba_instance_id, mamba_host, proto::meta::ST_EVENT_REPORT_L1P5);
    InitializeEventReporter(mamba_instance_id, mamba_p2p_host, proto::meta::ST_EVENT_REPORT_L2);

    auto report_mamba_specs = [&](proto::meta::StorageType type,
                                  const std::string &host,
                                  int64_t key,
                                  const std::vector<std::string> &spec_names) {
        proto::meta::ReportEventRequest req;
        req.set_instance_id(mamba_instance_id);
        req.set_host_ip_port(host);
        req.set_storage_type(type);
        auto *event = req.add_events();
        event->set_event_type(proto::meta::EVENT_BLOCK_ADD);
        auto *block_add = event->mutable_block_add();
        block_add->set_block_key(std::to_string(key));
        block_add->set_medium("mem");
        for (const auto &spec_name : spec_names) {
            auto *spec = block_add->add_specs();
            spec->set_name(spec_name);
            spec->set_uri("event_report://" + host + "/mem");
        }
        proto::meta::ReportEventResponse resp;
        ASSERT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &req, &resp));
    };

    report_mamba_specs(proto::meta::ST_EVENT_REPORT_L1P5, mamba_host, 100, {"full_0", "linear_0", "linear_1"});
    report_mamba_specs(proto::meta::ST_EVENT_REPORT_L1P5, mamba_host, 200, {"full_0"});
    report_mamba_specs(proto::meta::ST_EVENT_REPORT_L1P5, mamba_host, 300, {"full_0", "linear_0", "linear_1"});
    report_mamba_specs(proto::meta::ST_EVENT_REPORT_L1P5, mamba_host, 400, {"full_0", "linear_0"});
    report_mamba_specs(proto::meta::ST_EVENT_REPORT_L2, mamba_p2p_host, 200, {"linear_0", "linear_1"});
    report_mamba_specs(proto::meta::ST_EVENT_REPORT_L2, mamba_p2p_host, 400, {"linear_1"});
    report_mamba_specs(proto::meta::ST_EVENT_REPORT_L2, mamba_p2p_host, 500, {"full_0", "linear_0", "linear_1"});

    auto [mamba_ec, mamba_hosts] =
        cache_manager_->GetHostCacheState(request_context_.get(),
                                          mamba_instance_id,
                                          CacheManager::QueryType::QT_PREFIX_MATCH_WITH_MAMBA,
                                          {100, 200, 300, 400, 500});
    ASSERT_EQ(EC_OK, mamba_ec);
    ASSERT_EQ(1u, mamba_hosts.size());
    EXPECT_EQ(mamba_host, mamba_hosts[0].host_ip_port);
    EXPECT_EQ(3, mamba_hosts[0].local);
    EXPECT_EQ(3, mamba_hosts[0].p2p_1_fetch);
    EXPECT_EQ(4, mamba_hosts[0].p2p_1_total_match);

    // Hybrid P2P uses coverage across missing spec positions. Peer A owns the
    // first two linear_1 positions, while peer B owns four later positions on
    // only three distinct block keys. Coverage must select B and fetch=3.
    const std::string coverage_instance_id = "test_host_cache_state_mamba_coverage";
    std::vector<LocationSpecInfo> coverage_spec_infos = {
        LocationSpecInfo("full_0", 512),
        LocationSpecInfo("linear_1", 512),
        LocationSpecInfo("linear_2", 512),
        LocationSpecInfo("linear_3", 512),
    };
    std::vector<LocationSpecGroup> coverage_spec_groups = {
        LocationSpecGroup("full_0", {"full_0"}),
        LocationSpecGroup("linear_1", {"linear_1"}),
        LocationSpecGroup("linear_2", {"linear_2"}),
        LocationSpecGroup("linear_3", {"linear_3"}),
    };
    ASSERT_EQ(expected_reg,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               coverage_instance_id,
                                               64,
                                               coverage_spec_infos,
                                               createModelDeployment(),
                                               coverage_spec_groups,
                                               CacheManager::QueryType::QT_PREFIX_MATCH_WITH_MAMBA));

    const std::string coverage_host = "10.0.4.1:8080";
    const std::string prefix_peer = "10.0.4.2:8080";
    const std::string coverage_peer = "10.0.4.3:8080";
    InitializeEventReporter(coverage_instance_id, coverage_host, proto::meta::ST_EVENT_REPORT_L1P5);
    InitializeEventReporter(coverage_instance_id, prefix_peer, proto::meta::ST_EVENT_REPORT_L2);
    InitializeEventReporter(coverage_instance_id, coverage_peer, proto::meta::ST_EVENT_REPORT_L2);

    auto report_coverage_specs = [&](proto::meta::StorageType type,
                                     const std::string &host,
                                     int64_t key,
                                     const std::vector<std::string> &spec_names) {
        proto::meta::ReportEventRequest req;
        req.set_instance_id(coverage_instance_id);
        req.set_host_ip_port(host);
        req.set_storage_type(type);
        auto *event = req.add_events();
        event->set_event_type(proto::meta::EVENT_BLOCK_ADD);
        auto *block_add = event->mutable_block_add();
        block_add->set_block_key(std::to_string(key));
        block_add->set_medium("mem");
        for (const auto &spec_name : spec_names) {
            auto *spec = block_add->add_specs();
            spec->set_name(spec_name);
            spec->set_uri("event_report://" + host + "/mem");
        }
        proto::meta::ReportEventResponse resp;
        ASSERT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &req, &resp));
    };

    report_coverage_specs(
        proto::meta::ST_EVENT_REPORT_L1P5, coverage_host, 100, {"full_0", "linear_1", "linear_2", "linear_3"});
    report_coverage_specs(proto::meta::ST_EVENT_REPORT_L1P5, coverage_host, 200, {"full_0"});
    report_coverage_specs(proto::meta::ST_EVENT_REPORT_L1P5, coverage_host, 300, {"full_0"});
    report_coverage_specs(proto::meta::ST_EVENT_REPORT_L1P5, coverage_host, 400, {"full_0", "linear_1"});
    report_coverage_specs(proto::meta::ST_EVENT_REPORT_L2, prefix_peer, 200, {"linear_1"});
    report_coverage_specs(proto::meta::ST_EVENT_REPORT_L2, prefix_peer, 300, {"linear_1"});
    report_coverage_specs(proto::meta::ST_EVENT_REPORT_L2, coverage_peer, 200, {"linear_2"});
    report_coverage_specs(proto::meta::ST_EVENT_REPORT_L2, coverage_peer, 300, {"linear_2"});
    report_coverage_specs(proto::meta::ST_EVENT_REPORT_L2, coverage_peer, 400, {"linear_2", "linear_3"});

    auto [coverage_ec, coverage_hosts] =
        cache_manager_->GetHostCacheState(request_context_.get(),
                                          coverage_instance_id,
                                          CacheManager::QueryType::QT_PREFIX_MATCH_WITH_MAMBA,
                                          {100, 200, 300, 400});
    ASSERT_EQ(EC_OK, coverage_ec);
    ASSERT_EQ(1u, coverage_hosts.size());
    EXPECT_EQ(coverage_host, coverage_hosts[0].host_ip_port);
    EXPECT_EQ(1, coverage_hosts[0].local);
    EXPECT_EQ(3, coverage_hosts[0].p2p_1_fetch);
    EXPECT_EQ(4, coverage_hosts[0].p2p_1_total_match);

    const std::vector<int64_t> mamba_top5_keys = {2000, 2001, 2002, 2003, 2004, 2005, 2006};
    const std::vector<std::pair<std::string, size_t>> mamba_top5_hosts = {
        {"10.0.6.1:8080", 7},
        {"10.0.6.2:8080", 6},
        {"10.0.6.3:8080", 5},
        {"10.0.6.4:8080", 4},
        {"10.0.6.5:8080", 3},
        {"10.0.6.6:8080", 3},
        {"10.0.6.7:8080", 1},
    };
    const std::vector<std::string> all_coverage_specs = {"full_0", "linear_1", "linear_2", "linear_3"};
    for (size_t i = 0; i < mamba_top5_hosts.size(); ++i) {
        const auto &[host, prefix_len] = mamba_top5_hosts[i];
        const auto storage_type = i == 0 ? proto::meta::ST_EVENT_REPORT_L2 : proto::meta::ST_EVENT_REPORT_L1P5;
        InitializeEventReporter(coverage_instance_id, host, storage_type);
        for (size_t key_index = 0; key_index < prefix_len; ++key_index) {
            report_coverage_specs(storage_type, host, mamba_top5_keys[key_index], all_coverage_specs);
        }
    }
    const std::string mamba_zero_local_host = "10.0.6.8:8080";
    InitializeEventReporter(coverage_instance_id, mamba_zero_local_host, proto::meta::ST_EVENT_REPORT_L1P5);
    report_coverage_specs(
        proto::meta::ST_EVENT_REPORT_L1P5, mamba_zero_local_host, mamba_top5_keys[1], all_coverage_specs);

    auto [mamba_top5_ec, mamba_top5_matches] =
        cache_manager_->GetHostCacheState(request_context_.get(),
                                          coverage_instance_id,
                                          CacheManager::QueryType::QT_PREFIX_MATCH_WITH_MAMBA,
                                          mamba_top5_keys);
    ASSERT_EQ(EC_OK, mamba_top5_ec);
    ASSERT_EQ(mamba_top5_hosts.size(), mamba_top5_matches.size());
    const std::vector<std::tuple<int64_t, int64_t, int64_t>> expected_mamba_top5_matches = {
        {7, 0, 7},
        {6, 1, 7},
        {5, 2, 7},
        {4, 3, 7},
        {3, 4, 7},
        {3, 0, 3},
        {1, 0, 1},
    };
    for (size_t i = 0; i < mamba_top5_hosts.size(); ++i) {
        EXPECT_EQ(mamba_top5_hosts[i].first, mamba_top5_matches[i].host_ip_port);
        EXPECT_EQ(std::get<0>(expected_mamba_top5_matches[i]), mamba_top5_matches[i].local);
        EXPECT_EQ(std::get<1>(expected_mamba_top5_matches[i]), mamba_top5_matches[i].p2p_1_fetch);
        EXPECT_EQ(std::get<2>(expected_mamba_top5_matches[i]), mamba_top5_matches[i].p2p_1_total_match);
    }
    EXPECT_EQ(mamba_top5_matches.end(),
              std::find_if(mamba_top5_matches.begin(), mamba_top5_matches.end(), [&](const auto &match) {
                  return match.host_ip_port == mamba_zero_local_host;
              }));

    dsm->storage_map_.erase("host_state_subscriber");
    dsm->storage_map_.erase("host_state_vineyard");
}

TEST_F(CacheManagerTest, TestGetHostCacheStateUnspecifiedWithoutRegisteredQueryType) {
    auto expected_reg = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    const std::string instance_id = "test_host_cache_state_no_query_type";
    ASSERT_EQ(expected_reg,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               instance_id,
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    CacheManager::KeyVector keys = {100};
    auto [ec, hosts] = cache_manager_->GetHostCacheState(
        request_context_.get(), instance_id, CacheManager::QueryType::QT_UNSPECIFIED, keys);
    EXPECT_EQ(EC_ERROR, ec);
    EXPECT_TRUE(hosts.empty());
}

TEST_F(CacheManagerTest, TestGetHostCacheStatePrefixMatchWithMamba) {
    auto expected_reg = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    const std::string instance_id = "test_host_cache_state_mamba";
    std::vector<LocationSpecInfo> location_spec_infos = {
        LocationSpecInfo("full_0", 512),
        LocationSpecInfo("linear_0", 512),
        LocationSpecInfo("linear_1", 512),
    };
    std::vector<LocationSpecGroup> location_spec_groups = {
        LocationSpecGroup("full_0", {"full_0"}),
        LocationSpecGroup("linear_0", {"linear_0"}),
        LocationSpecGroup("linear_1", {"linear_1"}),
    };
    ASSERT_EQ(expected_reg,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               instance_id,
                                               64,
                                               location_spec_infos,
                                               createModelDeploymentWithEaglePop(),
                                               location_spec_groups,
                                               CacheManager::QueryType::QT_PREFIX_MATCH_WITH_MAMBA));

    auto metrics_registry = cache_manager_->metrics_registry_;
    auto event_backend = std::make_shared<EventReportBackend>(metrics_registry);
    {
        StorageConfig cfg;
        cfg.set_global_unique_name("event_backend_mamba");
        cfg.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5);
        cfg.set_storage_spec(std::make_shared<EventReportStorageSpec>());
        event_backend->Open(cfg, "test_trace");
    }
    auto dsm = registry_manager_->data_storage_manager_;
    dsm->storage_map_["event_backend_mamba"] = event_backend;
    registry_manager_->instance_group_configs_["default"]->set_event_report_storage_candidates({"event_backend_mamba"});

    auto report_specs = [&](const std::string &host, int64_t key, const std::vector<std::string> &spec_names) {
        proto::meta::ReportEventRequest req;
        req.set_instance_id(instance_id);
        req.set_host_ip_port(host);
        req.set_storage_type(proto::meta::ST_EVENT_REPORT_L1P5);

        auto *ev = req.add_events();
        ev->set_event_type(proto::meta::EVENT_BLOCK_ADD);
        auto *ba = ev->mutable_block_add();
        ba->set_block_key(std::to_string(key));
        ba->set_medium("mem");
        for (const auto &spec_name : spec_names) {
            auto *spec = ba->add_specs();
            spec->set_name(spec_name);
            spec->set_uri("event_report://" + host + "/mem");
        }

        proto::meta::ReportEventResponse resp;
        ASSERT_EQ(EC_OK, cache_manager_->ReportEvent(request_context_.get(), &req, &resp))
            << "host=" << host << " key=" << key << " response=" << resp.DebugString();
    };

    const std::string host_a = "10.0.1.1:8080";
    const std::string host_b = "10.0.1.2:8080";
    const std::string host_c = "10.0.1.3:8080";
    const std::string host_e = "10.0.1.5:8080";
    InitializeEventReporter(instance_id, host_a, proto::meta::ST_EVENT_REPORT_L1P5);
    InitializeEventReporter(instance_id, host_b, proto::meta::ST_EVENT_REPORT_L1P5);
    InitializeEventReporter(instance_id, host_c, proto::meta::ST_EVENT_REPORT_L1P5);
    InitializeEventReporter(instance_id, host_e, proto::meta::ST_EVENT_REPORT_L1P5);

    report_specs(host_a, 100, {"full_0", "linear_0", "linear_1"});
    report_specs(host_a, 200, {"full_0"});
    report_specs(host_a, 300, {"full_0"});
    report_specs(host_a, 300, {"linear_0", "linear_1"});
    report_specs(host_a, 400, {"full_0", "linear_0"});

    report_specs(host_b, 100, {"full_0"});
    report_specs(host_b, 200, {"full_0"});
    report_specs(host_b, 300, {"full_0"});
    report_specs(host_b, 400, {"full_0", "linear_0", "linear_1"});

    report_specs(host_c, 100, {"full_0"});
    report_specs(host_c, 200, {"full_0"});

    report_specs(host_e, 100, {"full_0", "linear_0", "linear_1"});
    report_specs(host_e, 200, {"full_0", "linear_0", "linear_1"});
    report_specs(host_e, 300, {"full_0", "linear_0", "linear_1"});

    const std::string host_d = "10.0.1.4:8080";
    InitializeEventReporter(instance_id, host_d, proto::meta::ST_EVENT_REPORT_L1P5);
    report_specs(host_d, 200, {"full_0", "linear_0", "linear_1"});
    report_specs(host_d, 300, {"full_0", "linear_0", "linear_1"});

    auto find_prefix = [](const std::vector<CacheManager::HostCacheMatch> &hosts, const std::string &host) -> int64_t {
        for (const auto &h : hosts) {
            if (h.host_ip_port == host) {
                return h.local;
            }
        }
        return -1;
    };

    CacheManager::KeyVector keys = {100, 200, 300, 400, 500};
    auto [ec, hosts] = cache_manager_->GetHostCacheState(
        request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH_WITH_MAMBA, keys);
    ASSERT_EQ(EC_OK, ec);

    auto expect_mamba_matches = [&](const std::vector<CacheManager::HostCacheMatch> &matches) {
        EXPECT_EQ(3, find_prefix(matches, host_a));
        EXPECT_EQ(-1, find_prefix(matches, host_b));
        EXPECT_EQ(-1, find_prefix(matches, host_c));
        EXPECT_EQ(-1, find_prefix(matches, host_d));
        EXPECT_EQ(2, find_prefix(matches, host_e));
    };
    expect_mamba_matches(hosts);

    // An explicit request query type takes precedence over the registered default.
    auto [explicit_ec, explicit_hosts] = cache_manager_->GetHostCacheState(
        request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, keys);
    ASSERT_EQ(EC_OK, explicit_ec);
    EXPECT_EQ(3, find_prefix(explicit_hosts, host_a));
    EXPECT_EQ(3, find_prefix(explicit_hosts, host_b));
    EXPECT_EQ(1, find_prefix(explicit_hosts, host_c));
    EXPECT_EQ(-1, find_prefix(explicit_hosts, host_d));
    EXPECT_EQ(2, find_prefix(explicit_hosts, host_e));

    auto [fallback_ec, fallback_hosts] = cache_manager_->GetHostCacheState(
        request_context_.get(), instance_id, CacheManager::QueryType::QT_UNSPECIFIED, keys);
    ASSERT_EQ(EC_OK, fallback_ec);
    expect_mamba_matches(fallback_hosts);

    {
        CacheManager::KeyVector break_keys = {100, 500, 400};
        auto [break_ec, break_hosts] = cache_manager_->GetHostCacheState(
            request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH_WITH_MAMBA, break_keys);
        ASSERT_EQ(EC_OK, break_ec);
        EXPECT_EQ(-1, find_prefix(break_hosts, host_a));
        EXPECT_EQ(-1, find_prefix(break_hosts, host_b));
        EXPECT_EQ(-1, find_prefix(break_hosts, host_c));
        EXPECT_EQ(-1, find_prefix(break_hosts, host_d));
        EXPECT_EQ(-1, find_prefix(break_hosts, host_e));
    }

    {
        CacheManager::KeyVector keys_without_host_d_first = {100, 200, 300};
        auto [absent_ec, absent_hosts] =
            cache_manager_->GetHostCacheState(request_context_.get(),
                                              instance_id,
                                              CacheManager::QueryType::QT_PREFIX_MATCH_WITH_MAMBA,
                                              keys_without_host_d_first);
        ASSERT_EQ(EC_OK, absent_ec);
        EXPECT_EQ(1, find_prefix(absent_hosts, host_a));
        EXPECT_EQ(-1, find_prefix(absent_hosts, host_b));
        EXPECT_EQ(-1, find_prefix(absent_hosts, host_c));
        EXPECT_EQ(-1, find_prefix(absent_hosts, host_d));
        EXPECT_EQ(2, find_prefix(absent_hosts, host_e));
    }

    {
        CacheManager::KeyVector large_keys;
        large_keys.reserve(384);
        for (std::size_t i = 0; i < 128; ++i) {
            large_keys.insert(large_keys.end(), {100, 200, 300});
        }
        auto [large_ec, large_hosts] = cache_manager_->GetHostCacheState(
            request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH_WITH_MAMBA, large_keys);
        ASSERT_EQ(EC_OK, large_ec);
        EXPECT_EQ(382, find_prefix(large_hosts, host_a));
        EXPECT_EQ(-1, find_prefix(large_hosts, host_b));
        EXPECT_EQ(-1, find_prefix(large_hosts, host_c));
        EXPECT_EQ(-1, find_prefix(large_hosts, host_d));
        EXPECT_EQ(383, find_prefix(large_hosts, host_e));
    }

    dsm->storage_map_.erase("event_backend_mamba");
}
// ===== 多层存储 Mark 消费（写路径）=====

// FilterWriteCache 统一入口：命中 mark 的 block 记录目标冷 storage（未命中为空）
TEST_F(CacheManagerTest, TestFilterWriteCacheTieredMarkPropagation) {
    EnableTieredMigrationStrategy();
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "placeholder_id",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("placeholder_id");
    ASSERT_TRUE(meta_searcher);

    // 持久化打标要求 block 先存在：给 block 1 建一个 location。
    {
        auto loc =
            std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                            1,
                                            std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot/blk1?size=1")});
        std::vector<std::string> ids;
        ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {loc}, ids));
    }
    cache_manager_->migration_manager()->MarkForTieredWrite("placeholder_id", {1}, "cold_01");

    CacheManager::KeyVector keys = {1, 2};
    CacheManager::KeyVector new_keys;
    std::vector<std::string_view> new_sgn;
    BlockMask block_mask;
    std::vector<std::string> new_targets;
    auto ec = cache_manager_->FilterWriteCache(request_context_.get(),
                                               "placeholder_id",
                                               meta_searcher,
                                               keys,
                                               new_keys,
                                               {},
                                               new_sgn,
                                               block_mask,
                                               1,
                                               new_targets);
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(2u, new_keys.size());
    ASSERT_EQ(new_keys.size(), new_targets.size());
    for (size_t i = 0; i < new_keys.size(); ++i) {
        if (new_keys[i] == 1) {
            ASSERT_EQ("cold_01", new_targets[i]); // 命中 mark -> 目标冷 storage
        } else {
            ASSERT_EQ("", new_targets[i]); // 未命中 -> 空（走默认）
        }
    }
}

TEST_F(CacheManagerTest, TestFilterWriteCacheFallsBackToOrdinaryPolicyOnMarkReadError) {
    EnableTieredMigrationStrategy();
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "mark_read_error",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("mark_read_error");
    ASSERT_TRUE(meta_searcher);

    auto hot_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot_01/mark_read_error?size=1")});
    std::vector<std::string> ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {hot_loc}, ids));
    ASSERT_EQ(1u, ids.size());
    std::vector<std::vector<MetaSearcher::LocationCASTask>> cas_tasks{
        {MetaSearcher::LocationCASTask{ids[0], CLS_WRITING, CLS_SERVING}}};
    std::vector<std::vector<ErrorCode>> cas_results;
    ASSERT_EQ(EC_OK, meta_searcher->BatchCASLocationStatus(request_context_.get(), {1}, cas_tasks, cas_results));

    Stub stub;
    stub.set(ADDR(MigrationManager, BatchGetTieredWriteTargets), mark_query_read_error_stub::ReadError_stub);
    CacheManager::KeyVector new_keys;
    std::vector<std::string_view> new_sgn;
    BlockMask block_mask;
    std::vector<std::string> new_targets;
    ASSERT_EQ(EC_OK,
              cache_manager_->FilterWriteCache(request_context_.get(),
                                               "mark_read_error",
                                               meta_searcher,
                                               {1},
                                               new_keys,
                                               {},
                                               new_sgn,
                                               block_mask,
                                               1,
                                               new_targets));

    ASSERT_TRUE(new_keys.empty());
    ASSERT_TRUE(new_targets.empty());
}

TEST_F(CacheManagerTest, TestFilterWriteCacheInvalidTieredTargetUsesOrdinaryPolicy) {
    EnableTieredMigrationStrategy();
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "invalid_tiered_target",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("invalid_tiered_target");
    ASSERT_TRUE(meta_searcher);

    auto hot_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot_01/invalid_target?size=1")});
    std::vector<std::string> ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {hot_loc}, ids));
    ASSERT_EQ(1u, ids.size());
    std::vector<std::vector<MetaSearcher::LocationCASTask>> cas_tasks{
        {MetaSearcher::LocationCASTask{ids[0], CLS_WRITING, CLS_SERVING}}};
    std::vector<std::vector<ErrorCode>> cas_results;
    ASSERT_EQ(EC_OK, meta_searcher->BatchCASLocationStatus(request_context_.get(), {1}, cas_tasks, cas_results));
    ASSERT_EQ(EC_OK, cache_manager_->migration_manager()->MarkForTieredWrite("invalid_tiered_target", {1}, "cold_01"));
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("invalid_tiered_target", 1));

    // Mark 创建后 target backend 被注销；hot 副本已满足普通策略，不应 fallback 再写一份 hot。
    ASSERT_EQ(EC_OK, registry_manager_->data_storage_manager()->UnRegisterStorage("cold_01"));
    CacheManager::KeyVector new_keys;
    std::vector<std::string_view> new_sgn;
    BlockMask block_mask;
    std::vector<std::string> new_targets;
    ASSERT_EQ(EC_OK,
              cache_manager_->FilterWriteCache(request_context_.get(),
                                               "invalid_tiered_target",
                                               meta_searcher,
                                               {1},
                                               new_keys,
                                               {},
                                               new_sgn,
                                               block_mask,
                                               1,
                                               new_targets));

    ASSERT_TRUE(new_keys.empty());
    ASSERT_TRUE(new_targets.empty());
    ASSERT_FALSE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("invalid_tiered_target", 1));
}

TEST_F(CacheManagerTest, TestFilterWriteCacheUnavailableTieredTargetUsesOrdinaryPolicyAndKeepsMark) {
    EnableTieredMigrationStrategy();
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "unavailable_tiered_target",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("unavailable_tiered_target");
    ASSERT_TRUE(meta_searcher);

    auto hot_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot_01/unavailable_target?size=1")});
    std::vector<std::string> ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {hot_loc}, ids));
    std::vector<std::vector<MetaSearcher::LocationCASTask>> cas_tasks{
        {MetaSearcher::LocationCASTask{ids[0], CLS_WRITING, CLS_SERVING}}};
    std::vector<std::vector<ErrorCode>> cas_results;
    ASSERT_EQ(EC_OK, meta_searcher->BatchCASLocationStatus(request_context_.get(), {1}, cas_tasks, cas_results));
    ASSERT_EQ(EC_OK,
              cache_manager_->migration_manager()->MarkForTieredWrite("unavailable_tiered_target", {1}, "cold_01"));
    ASSERT_EQ(EC_OK, registry_manager_->DisableStorage(request_context_.get(), "cold_01"));

    CacheManager::KeyVector new_keys;
    std::vector<std::string_view> new_sgn;
    BlockMask block_mask;
    std::vector<std::string> new_targets;
    ASSERT_EQ(EC_OK,
              cache_manager_->FilterWriteCache(request_context_.get(),
                                               "unavailable_tiered_target",
                                               meta_searcher,
                                               {1},
                                               new_keys,
                                               {},
                                               new_sgn,
                                               block_mask,
                                               1,
                                               new_targets));
    ASSERT_TRUE(new_keys.empty());
    ASSERT_TRUE(new_targets.empty());
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("unavailable_tiered_target", 1));
}

TEST_F(CacheManagerTest, TestMigrationTargetsRespectGroupQuota) {
    EnableTieredMigrationStrategy();
    cache_manager_->StartMigrationManager();
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "migration_target_quota",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    auto default_group = registry_manager_->instance_group_configs_["default"];
    ASSERT_NE(nullptr, default_group);
    default_group->set_quota(InstanceGroupQuota(0, {}));

    EXPECT_EQ(EC_NOSPC,
              cache_manager_->migration_manager()->MarkForTieredWrite("migration_target_quota", {1}, "cold_01"));
    EXPECT_FALSE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("migration_target_quota", 1));

    MigrationManager::MigrationRequest request;
    request.instance_group_name = "default";
    request.instance_id = "migration_target_quota";
    request.block_key = 1;
    request.src_location_id = "source_location";
    request.src_storage_name = "hot_01";
    request.dst_storage_name = "cold_01";
    request.src_specs = {LocationSpec("tp0", "dummy://hot_01/source?size=1")};
    const auto results = cache_manager_->migration_manager()->BatchSubmit("target-quota", {request});
    ASSERT_EQ(1u, results.size());
    EXPECT_EQ(EC_NOSPC, results[0]);
    EXPECT_EQ(0u, cache_manager_->migration_manager()->GetStats().active_copy_tasks);
}

TEST_F(CacheManagerTest, TestFilterWriteCacheWithMinReplicaFallsBackOnMarkReadError) {
    EnableTieredMigrationStrategy();
    ASSERT_TRUE(RegisterDummyStorage("hot_02"));
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "min_replica_mark_read_error",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    MetaSearcher *meta_searcher =
        cache_manager_->meta_searcher_manager_->GetMetaSearcher("min_replica_mark_read_error");
    ASSERT_TRUE(meta_searcher);

    auto make_hot_loc = [](const std::string &uri) {
        return std::make_shared<CacheLocation>(
            DataStorageType::DATA_STORAGE_TYPE_DUMMY, 1, std::vector<LocationSpec>{LocationSpec("tp0", uri)});
    };
    for (const auto &uri : {"dummy://hot_01/mark_read_error_a?size=1", "dummy://hot_02/mark_read_error_b?size=1"}) {
        std::vector<std::string> ids;
        ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {make_hot_loc(uri)}, ids));
        ASSERT_EQ(1u, ids.size());
        std::vector<std::vector<MetaSearcher::LocationCASTask>> cas_tasks{
            {MetaSearcher::LocationCASTask{ids[0], CLS_WRITING, CLS_SERVING}}};
        std::vector<std::vector<ErrorCode>> cas_results;
        ASSERT_EQ(EC_OK, meta_searcher->BatchCASLocationStatus(request_context_.get(), {1}, cas_tasks, cas_results));
    }

    Stub stub;
    stub.set(ADDR(MigrationManager, BatchGetTieredWriteTargets), mark_query_read_error_stub::ReadError_stub);
    CacheManager::KeyVector new_keys;
    std::vector<std::string_view> new_sgn;
    BlockMask block_mask;
    std::vector<std::string> new_targets;
    ASSERT_EQ(EC_OK,
              cache_manager_->FilterWriteCache(request_context_.get(),
                                               "min_replica_mark_read_error",
                                               meta_searcher,
                                               {1},
                                               new_keys,
                                               {},
                                               new_sgn,
                                               block_mask,
                                               2,
                                               new_targets));

    ASSERT_TRUE(new_keys.empty());
    ASSERT_TRUE(new_targets.empty());
}

TEST_F(CacheManagerTest, TestFilterWriteCacheWithMinReplicaInvalidTieredTargetUsesOrdinaryPolicy) {
    EnableTieredMigrationStrategy();
    ASSERT_TRUE(RegisterDummyStorage("hot_02"));
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "min_replica_invalid_target",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("min_replica_invalid_target");
    ASSERT_TRUE(meta_searcher);

    auto add_serving_location = [&](int64_t block_key, const std::string &uri) {
        auto loc = std::make_shared<CacheLocation>(
            DataStorageType::DATA_STORAGE_TYPE_DUMMY, 1, std::vector<LocationSpec>{LocationSpec("tp0", uri)});
        std::vector<std::string> ids;
        ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {block_key}, {loc}, ids));
        ASSERT_EQ(1u, ids.size());
        std::vector<std::vector<MetaSearcher::LocationCASTask>> cas_tasks{
            {MetaSearcher::LocationCASTask{ids[0], CLS_WRITING, CLS_SERVING}}};
        std::vector<std::vector<ErrorCode>> cas_results;
        ASSERT_EQ(EC_OK,
                  meta_searcher->BatchCASLocationStatus(request_context_.get(), {block_key}, cas_tasks, cas_results));
    };
    // block 1 已有两个普通副本，block 2 只有一个；min_replica_count=2。
    add_serving_location(1, "dummy://hot_01/min_replica_invalid_1a?size=1");
    add_serving_location(1, "dummy://hot_02/min_replica_invalid_1b?size=1");
    add_serving_location(2, "dummy://hot_01/min_replica_invalid_2?size=1");
    ASSERT_EQ(EC_OK,
              cache_manager_->migration_manager()->MarkForTieredWrite("min_replica_invalid_target", {1, 2}, "cold_01"));

    ASSERT_EQ(EC_OK, registry_manager_->data_storage_manager()->UnRegisterStorage("cold_01"));
    CacheManager::KeyVector new_keys;
    std::vector<std::string_view> new_sgn;
    BlockMask block_mask;
    std::vector<std::string> new_targets;
    ASSERT_EQ(EC_OK,
              cache_manager_->FilterWriteCache(request_context_.get(),
                                               "min_replica_invalid_target",
                                               meta_searcher,
                                               {1, 2},
                                               new_keys,
                                               {},
                                               new_sgn,
                                               block_mask,
                                               2,
                                               new_targets));

    // 失效 Mark 不再强制写：已满足的 block 1 跳过；未满足的 block 2 按普通策略走默认层。
    ASSERT_EQ((CacheManager::KeyVector{2}), new_keys);
    ASSERT_EQ(1u, new_targets.size());
    ASSERT_TRUE(new_targets[0].empty());
    ASSERT_FALSE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("min_replica_invalid_target", 1));
    ASSERT_FALSE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("min_replica_invalid_target", 2));
}

TEST_F(CacheManagerTest, TestFilterWriteCacheSkipsTieredMarkWhenMigrationDisabled) {
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "tiered_disabled",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("tiered_disabled");
    ASSERT_TRUE(meta_searcher);

    auto hot_loc =
        std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                        1,
                                        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot_01/blk1?size=1")});
    std::vector<std::string> ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {hot_loc}, ids));
    ASSERT_EQ(EC_OK, cache_manager_->migration_manager()->MarkForTieredWrite("tiered_disabled", {1}, "cold_01"));

    CacheManager::KeyVector new_keys;
    std::vector<std::string_view> new_sgn;
    BlockMask block_mask;
    std::vector<std::string> new_targets;
    auto ec = cache_manager_->FilterWriteCache(request_context_.get(),
                                               "tiered_disabled",
                                               meta_searcher,
                                               {1},
                                               new_keys,
                                               {},
                                               new_sgn,
                                               block_mask,
                                               1,
                                               new_targets);
    ASSERT_EQ(EC_OK, ec);
    ASSERT_TRUE(new_keys.empty());
    ASSERT_TRUE(new_targets.empty());
}

TEST_F(CacheManagerTest, TestFilterWriteCacheTieredMarkSkipsExistingTarget) {
    EnableTieredMigrationStrategy();
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "placeholder_id",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("placeholder_id");
    ASSERT_TRUE(meta_searcher);

    auto writing_loc = std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                                       4,
                                                       std::vector<LocationSpec>{
                                                           LocationSpec("tp0", "dummy://cold_01/blk1/tp0?size=1"),
                                                           LocationSpec("tp1", "dummy://cold_01/blk1/tp1?size=1"),
                                                           LocationSpec("tp2", "dummy://cold_01/blk1/tp2?size=1"),
                                                           LocationSpec("tp3", "dummy://cold_01/blk1/tp3?size=1"),
                                                       });
    auto serving_loc = std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                                       4,
                                                       std::vector<LocationSpec>{
                                                           LocationSpec("tp0", "dummy://cold_01/blk2/tp0?size=1"),
                                                           LocationSpec("tp1", "dummy://cold_01/blk2/tp1?size=1"),
                                                           LocationSpec("tp2", "dummy://cold_01/blk2/tp2?size=1"),
                                                           LocationSpec("tp3", "dummy://cold_01/blk2/tp3?size=1"),
                                                       });
    std::vector<std::string> ids;
    ASSERT_EQ(EC_OK,
              BatchAddLocationForTest(meta_searcher, request_context_.get(), {1, 2}, {writing_loc, serving_loc}, ids));
    ASSERT_EQ(2u, ids.size());
    std::vector<std::vector<MetaSearcher::LocationCASTask>> cas_tasks{
        {MetaSearcher::LocationCASTask{ids[1], CLS_WRITING, CLS_SERVING}}};
    std::vector<std::vector<ErrorCode>> cas_results;
    ASSERT_EQ(EC_OK, meta_searcher->BatchCASLocationStatus(request_context_.get(), {2}, cas_tasks, cas_results));
    cache_manager_->migration_manager()->MarkForTieredWrite("placeholder_id", {1, 2}, "cold_01");

    CacheManager::KeyVector keys = {1, 2};
    CacheManager::KeyVector new_keys;
    std::vector<std::string_view> new_sgn;
    BlockMask block_mask;
    std::vector<std::string> new_targets;
    auto ec = cache_manager_->FilterWriteCache(request_context_.get(),
                                               "placeholder_id",
                                               meta_searcher,
                                               keys,
                                               new_keys,
                                               {},
                                               new_sgn,
                                               block_mask,
                                               1,
                                               new_targets);
    ASSERT_EQ(EC_OK, ec);
    ASSERT_TRUE(new_keys.empty());
    ASSERT_TRUE(new_targets.empty());
}

// 冷层 target 有 CLS_SERVING location 但数据已丢（MightExist=false）时，marked write 不应因
// meta 仍 SERVING 就跳过；应视 target 未满足 → block 进待写集并路由回 cold_01。stale 判断复用普通路径
// 的 prune 结果(同一次 Exist)，不额外查后端。
TEST_F(CacheManagerTest, TestFilterWriteCacheTieredStaleTargetTriggersRewrite) {
    EnableTieredMigrationStrategy();
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "stale_tier_instance",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("stale_tier_instance");
    ASSERT_TRUE(meta_searcher);

    // block 1: 冷层 cold_01 上有一个覆盖全 spec 的 SERVING location。
    auto cold_loc = std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                                    4,
                                                    std::vector<LocationSpec>{
                                                        LocationSpec("tp0", "dummy://cold_01/blk1/tp0?size=1"),
                                                        LocationSpec("tp1", "dummy://cold_01/blk1/tp1?size=1"),
                                                        LocationSpec("tp2", "dummy://cold_01/blk1/tp2?size=1"),
                                                        LocationSpec("tp3", "dummy://cold_01/blk1/tp3?size=1"),
                                                    });
    std::vector<std::string> ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {cold_loc}, ids));
    ASSERT_EQ(1u, ids.size());
    std::vector<std::vector<MetaSearcher::LocationCASTask>> cas_tasks{
        {MetaSearcher::LocationCASTask{ids[0], CLS_WRITING, CLS_SERVING}}};
    std::vector<std::vector<ErrorCode>> cas_results;
    ASSERT_EQ(EC_OK, meta_searcher->BatchCASLocationStatus(request_context_.get(), {1}, cas_tasks, cas_results));
    cache_manager_->migration_manager()->MarkForTieredWrite("stale_tier_instance", {1}, "cold_01");
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("stale_tier_instance", 1));

    // 让 cold_01 的数据 MightExist=false(模拟数据被驱逐/丢失)——此时 meta 仍 SERVING，但数据不在。
    auto dsm = registry_manager_->data_storage_manager_;
    auto original = dsm->storage_map_["cold_01"];
    dsm->storage_map_["cold_01"] = std::make_shared<MightExistInterceptor>(
        original, [](const std::vector<DataStorageUri> &uris) { return std::vector<bool>(uris.size(), false); });

    CacheManager::KeyVector keys = {1};
    CacheManager::KeyVector new_keys;
    std::vector<std::string_view> new_sgn;
    BlockMask block_mask;
    std::vector<std::string> new_targets;
    auto ec = cache_manager_->FilterWriteCache(request_context_.get(),
                                               "stale_tier_instance",
                                               meta_searcher,
                                               keys,
                                               new_keys,
                                               {},
                                               new_sgn,
                                               block_mask,
                                               1,
                                               new_targets);
    dsm->storage_map_["cold_01"] = original;

    ASSERT_EQ(EC_OK, ec);
    // 关键:stale 冷层 target 被视为未满足 → block 1 需重写且路由回 cold_01(而非误判已满足跳过)。
    ASSERT_EQ(1u, new_keys.size());
    ASSERT_EQ(1, new_keys[0]);
    ASSERT_EQ(1u, new_targets.size());
    ASSERT_EQ("cold_01", new_targets[0]);
}

TEST_F(CacheManagerTest, TestFilterWriteCacheWithMinReplicaUsesTieredMarkTarget) {
    EnableTieredMigrationStrategy();
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "min_replica_tiered",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("min_replica_tiered");
    ASSERT_TRUE(meta_searcher);

    auto hot_loc =
        std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                        1,
                                        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot_01/blk1?size=1")});
    std::vector<std::string> ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {hot_loc}, ids));
    cache_manager_->migration_manager()->MarkForTieredWrite("min_replica_tiered", {1}, "cold_01");

    CacheManager::KeyVector new_keys;
    std::vector<std::string_view> new_sgn;
    BlockMask block_mask;
    std::vector<std::string> new_targets;
    auto ec = cache_manager_->FilterWriteCache(request_context_.get(),
                                               "min_replica_tiered",
                                               meta_searcher,
                                               {1},
                                               new_keys,
                                               {},
                                               new_sgn,
                                               block_mask,
                                               2,
                                               new_targets);
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(1u, new_keys.size());
    ASSERT_EQ(1, new_keys[0]);
    ASSERT_EQ(1u, new_targets.size());
    ASSERT_EQ("cold_01", new_targets[0]);
}

TEST_F(CacheManagerTest, TestFilterWriteCacheWithMinReplicaHonorsTieredMarkWhenReplicaSatisfied) {
    EnableTieredMigrationStrategy();
    ASSERT_TRUE(RegisterDummyStorage("hot_02"));
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "min_replica_satisfied_tiered",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    MetaSearcher *meta_searcher =
        cache_manager_->meta_searcher_manager_->GetMetaSearcher("min_replica_satisfied_tiered");
    ASSERT_TRUE(meta_searcher);

    auto make_hot_loc = [](const std::string &uri) {
        return std::make_shared<CacheLocation>(
            DataStorageType::DATA_STORAGE_TYPE_DUMMY, 1, std::vector<LocationSpec>{LocationSpec("tp0", uri)});
    };
    std::vector<std::string> ids;
    ASSERT_EQ(EC_OK,
              BatchAddLocationForTest(
                  meta_searcher, request_context_.get(), {1}, {make_hot_loc("dummy://hot_01/blk1_a?size=1")}, ids));
    ASSERT_EQ(1u, ids.size());
    std::vector<std::vector<MetaSearcher::LocationCASTask>> cas_tasks{
        {MetaSearcher::LocationCASTask{ids[0], CLS_WRITING, CLS_SERVING}}};
    std::vector<std::vector<ErrorCode>> cas_results;
    ASSERT_EQ(EC_OK, meta_searcher->BatchCASLocationStatus(request_context_.get(), {1}, cas_tasks, cas_results));

    ids.clear();
    ASSERT_EQ(EC_OK,
              BatchAddLocationForTest(
                  meta_searcher, request_context_.get(), {1}, {make_hot_loc("dummy://hot_02/blk1_b?size=1")}, ids));
    ASSERT_EQ(1u, ids.size());
    cas_tasks = {{MetaSearcher::LocationCASTask{ids[0], CLS_WRITING, CLS_SERVING}}};
    cas_results.clear();
    ASSERT_EQ(EC_OK, meta_searcher->BatchCASLocationStatus(request_context_.get(), {1}, cas_tasks, cas_results));

    cache_manager_->migration_manager()->MarkForTieredWrite("min_replica_satisfied_tiered", {1}, "cold_01");

    CacheManager::KeyVector new_keys;
    std::vector<std::string_view> new_sgn;
    BlockMask block_mask;
    std::vector<std::string> new_targets;
    auto ec = cache_manager_->FilterWriteCache(request_context_.get(),
                                               "min_replica_satisfied_tiered",
                                               meta_searcher,
                                               {1},
                                               new_keys,
                                               {},
                                               new_sgn,
                                               block_mask,
                                               2,
                                               new_targets);
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(1u, new_keys.size());
    ASSERT_EQ(1, new_keys[0]);
    ASSERT_EQ(1u, new_targets.size());
    ASSERT_EQ("cold_01", new_targets[0]);
}

TEST_F(CacheManagerTest, TestFilterWriteCacheTieredMarkChecksSpecGroupOnTarget) {
    EnableTieredMigrationStrategy();
    std::vector<LocationSpecInfo> location_spec_infos = {
        LocationSpecInfo("tp0_F0", 512),
        LocationSpecInfo("tp1_F0", 512),
        LocationSpecInfo("tp0_L1", 512),
        LocationSpecInfo("tp1_L1", 512),
    };
    std::vector<LocationSpecGroup> location_spec_groups = {
        LocationSpecGroup("F0", {"tp0_F0", "tp1_F0"}),
        LocationSpecGroup("L1", {"tp0_L1", "tp1_L1"}),
    };
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "tiered_spec_group",
                                               64,
                                               location_spec_infos,
                                               createModelDeployment(),
                                               location_spec_groups));
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("tiered_spec_group");
    ASSERT_TRUE(meta_searcher);

    auto cold_f0_loc = std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                                       2,
                                                       std::vector<LocationSpec>{
                                                           LocationSpec("tp0_F0", "dummy://cold_01/blk1/tp0_F0?size=1"),
                                                           LocationSpec("tp1_F0", "dummy://cold_01/blk1/tp1_F0?size=1"),
                                                       });
    std::vector<std::string> ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {cold_f0_loc}, ids));
    ASSERT_EQ(1u, ids.size());
    cache_manager_->migration_manager()->MarkForTieredWrite("tiered_spec_group", {1}, "cold_01");

    {
        CacheManager::KeyVector new_keys;
        const std::vector<std::string> location_spec_group_names = {"F0"};
        std::vector<std::string_view> new_sgn;
        BlockMask block_mask;
        std::vector<std::string> new_targets;
        auto ec = cache_manager_->FilterWriteCache(request_context_.get(),
                                                   "tiered_spec_group",
                                                   meta_searcher,
                                                   {1},
                                                   new_keys,
                                                   location_spec_group_names,
                                                   new_sgn,
                                                   block_mask,
                                                   1,
                                                   new_targets);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_TRUE(new_keys.empty());
        ASSERT_TRUE(new_targets.empty());
    }

    {
        CacheManager::KeyVector new_keys;
        const std::vector<std::string> location_spec_group_names = {"L1"};
        std::vector<std::string_view> new_sgn;
        BlockMask block_mask;
        std::vector<std::string> new_targets;
        auto ec = cache_manager_->FilterWriteCache(request_context_.get(),
                                                   "tiered_spec_group",
                                                   meta_searcher,
                                                   {1},
                                                   new_keys,
                                                   location_spec_group_names,
                                                   new_sgn,
                                                   block_mask,
                                                   1,
                                                   new_targets);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(1u, new_keys.size());
        ASSERT_EQ(1, new_keys[0]);
        ASSERT_EQ(1u, new_sgn.size());
        ASSERT_EQ("L1", new_sgn[0]);
        ASSERT_EQ(1u, new_targets.size());
        ASSERT_EQ("cold_01", new_targets[0]);
    }
}

// GenWriteLocation 按 block 路由：marked block 的 location 落在目标冷 storage
TEST_F(CacheManagerTest, TestGenWriteLocationTieredRouting) {
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "placeholder_id",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    // cold_01 由 fixture 注册（dummy + MightExist=true）。

    CacheManager::KeyVector new_keys = {1, 2};
    std::vector<std::string_view> new_sgn;
    std::vector<std::string> tiered_targets = {"", "cold_01"}; // block 2 -> 冷层
    CacheLocationVector new_locations;
    auto ec = cache_manager_->GenWriteLocation(
        request_context_.get(), "placeholder_id", new_keys, new_sgn, tiered_targets, new_locations);
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(2u, new_locations.size());
    // block 2 被路由到 cold_01（DUMMY 类型）；block 1 走默认 storage（非 DUMMY）
    ASSERT_TRUE(new_locations[1] != nullptr && new_locations[0] != nullptr);
    ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_DUMMY, new_locations[1]->type());
    ASSERT_NE(DataStorageType::DATA_STORAGE_TYPE_DUMMY, new_locations[0]->type());
}

// 全部 block 都有 tiered target 时，不应因为默认 hot storage 不可选而阻断冷层写入。
TEST_F(CacheManagerTest, TestGenWriteLocationAllTieredDoesNotRequireDefaultStorage) {
    auto expected = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "all_tiered_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));
    // cold_01 由 fixture 注册（dummy + MightExist=true）。
    ASSERT_EQ(EC_OK, registry_manager_->DisableStorage(request_context_.get(), "nfs_01"));

    CacheManager::KeyVector new_keys = {1, 2};
    std::vector<std::string_view> new_sgn;
    std::vector<std::string> tiered_targets = {"cold_01", "cold_01"};
    CacheLocationVector new_locations;
    auto ec = cache_manager_->GenWriteLocation(
        request_context_.get(), "all_tiered_instance", new_keys, new_sgn, tiered_targets, new_locations);

    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(2u, new_locations.size());
    for (const auto &location : new_locations) {
        ASSERT_NE(nullptr, location);
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_DUMMY, location->type());
        for (const auto &spec : location->location_specs()) {
            ASSERT_THAT(spec.uri(), HasSubstr("dummy://cold_01/"));
        }
    }
}

TEST_F(CacheManagerTest, TestGenWriteLocationMissingTieredTargetDoesNotFallback) {
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "missing_tiered_target",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    ASSERT_EQ(EC_OK, registry_manager_->data_storage_manager()->UnRegisterStorage("cold_01"));

    CacheLocationVector new_locations;
    ASSERT_EQ(EC_NOENT,
              cache_manager_->GenWriteLocation(
                  request_context_.get(), "missing_tiered_target", {1}, {}, {"cold_01"}, new_locations));
    ASSERT_TRUE(new_locations.empty());
}

// FinishWriteCache 成功把本次 target CacheLocation 置为 SERVING 后清除 tiered-write mark
TEST_F(CacheManagerTest, TestFinishWriteCacheClearsTieredMark) {
    // 清标是 tiered migration 行为，仅对启用了 migration_strategies 的 group 生效。
    EnableTieredMigrationStrategy();
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "placeholder_id",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("placeholder_id");
    ASSERT_TRUE(meta_searcher);
    std::vector<std::string> source_ids;
    {
        auto loc =
            std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                            1,
                                            std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot/blk1?size=1")});
        ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {loc}, source_ids));
    }
    ASSERT_EQ(1u, source_ids.size());
    cache_manager_->migration_manager()->MarkForTieredWrite("placeholder_id", {1}, "cold_01");
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("placeholder_id", 1));

    auto target_loc =
        std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                        1,
                                        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://cold_01/blk1?size=1")});
    std::vector<std::string> target_ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {target_loc}, target_ids));
    ASSERT_EQ(1u, target_ids.size());

    auto info = std::make_unique<WriteLocationManager::WriteLocationInfo>();
    info->keys = {1};
    info->location_ids = {target_ids[0]};
    BlockMask success_mask = static_cast<BlockMaskOffset>(1); // 全部成功
    auto ec = cache_manager_->FinishWriteCache(
        request_context_.get(), "placeholder_id", "sess_p5", success_mask, std::move(info));
    ASSERT_EQ(EC_OK, ec);
    ASSERT_FALSE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("placeholder_id", 1));

    std::vector<CacheLocationMap> location_maps;
    BlockMask empty_mask;
    ASSERT_EQ(EC_OK, meta_searcher->BatchGetLocation(request_context_.get(), {1}, empty_mask, location_maps));
    ASSERT_EQ(1u, location_maps.size());
    ASSERT_EQ(2u, location_maps[0].size());
    EXPECT_NE(location_maps[0].end(), location_maps[0].find(source_ids[0]));
    EXPECT_NE(location_maps[0].end(), location_maps[0].find(target_ids[0]));
}

TEST_F(CacheManagerTest, TestAdminMarkUsesMatchingStrategyTimeout) {
    constexpr int64_t kTimeoutMs = 3000;
    EnableTieredMigrationStrategy("default", "hot_01", "cold_01", kTimeoutMs);
    cache_manager_->StartMigrationManager();
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "admin_mark_timeout_instance",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    auto *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("admin_mark_timeout_instance");
    ASSERT_TRUE(meta_searcher);
    auto source_loc =
        std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                        1,
                                        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot_01/blk1?size=1")});
    std::vector<std::string> source_ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {source_loc}, source_ids));
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> status_tasks = {
        {{source_ids[0], CacheLocationStatus::CLS_SERVING}}};
    std::vector<std::vector<ErrorCode>> status_results;
    ASSERT_EQ(EC_OK,
              meta_searcher->BatchUpdateLocationStatus(request_context_.get(), {1}, status_tasks, status_results));

    const auto before = std::chrono::system_clock::now();
    const auto result = cache_manager_->MigrateCache(request_context_.get(),
                                                     "admin-mark-timeout",
                                                     "admin_mark_timeout_instance",
                                                     "hot_01",
                                                     "cold_01",
                                                     false,
                                                     true,
                                                     {1},
                                                     0);
    const auto after = std::chrono::system_clock::now();
    ASSERT_EQ(EC_OK, result.ec);
    ASSERT_EQ(1, result.accepted);
    ASSERT_EQ(0, result.rejected);

    std::vector<MigrationManager::MarkQueryResult> marks;
    ASSERT_EQ(
        EC_OK,
        cache_manager_->migration_manager()->BatchGetTieredWriteTargets("admin_mark_timeout_instance", {1}, marks));
    ASSERT_EQ(1u, marks.size());
    ASSERT_TRUE(marks[0].HasValidMark());
    const auto before_deadline =
        std::chrono::duration_cast<std::chrono::milliseconds>(before.time_since_epoch()).count() + kTimeoutMs;
    const auto after_deadline =
        std::chrono::duration_cast<std::chrono::milliseconds>(after.time_since_epoch()).count() + kTimeoutMs;
    EXPECT_GE(marks[0].deadline_ms, before_deadline);
    EXPECT_LE(marks[0].deadline_ms, after_deadline);
}

TEST_F(CacheManagerTest, TestAdminMarkAllowsUnmatchedTargetWithDefaultTimeout) {
    EnableTieredMigrationStrategy("default", "hot_01", "cold_01", 3000);
    cache_manager_->StartMigrationManager();
    ASSERT_TRUE(RegisterDummyStorage("cold_02"));
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "admin_mark_unmatched_target",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    auto *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("admin_mark_unmatched_target");
    ASSERT_TRUE(meta_searcher);
    auto source_loc =
        std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                        1,
                                        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot_01/blk1?size=1")});
    std::vector<std::string> source_ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {source_loc}, source_ids));
    ASSERT_EQ(1u, source_ids.size());
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> status_tasks = {
        {{source_ids[0], CacheLocationStatus::CLS_SERVING}}};
    std::vector<std::vector<ErrorCode>> status_results;
    ASSERT_EQ(EC_OK,
              meta_searcher->BatchUpdateLocationStatus(request_context_.get(), {1}, status_tasks, status_results));

    const auto before = std::chrono::system_clock::now();
    const auto result = cache_manager_->MigrateCache(request_context_.get(),
                                                     "admin-mark-unmatched-target",
                                                     "admin_mark_unmatched_target",
                                                     "hot_01",
                                                     "cold_02",
                                                     false,
                                                     true,
                                                     {1},
                                                     0);
    const auto after = std::chrono::system_clock::now();
    ASSERT_EQ(EC_OK, result.ec);
    ASSERT_EQ(1, result.accepted);
    ASSERT_EQ(0, result.rejected);

    std::vector<MigrationManager::MarkQueryResult> marks;
    ASSERT_EQ(
        EC_OK,
        cache_manager_->migration_manager()->BatchGetTieredWriteTargets("admin_mark_unmatched_target", {1}, marks));
    ASSERT_EQ(1u, marks.size());
    ASSERT_TRUE(marks[0].HasValidMark());
    EXPECT_EQ("cold_02", marks[0].target);
    const auto before_deadline =
        std::chrono::duration_cast<std::chrono::milliseconds>(before.time_since_epoch()).count() +
        MigrationMarkMethod::kDefaultTimeoutMs;
    const auto after_deadline =
        std::chrono::duration_cast<std::chrono::milliseconds>(after.time_since_epoch()).count() +
        MigrationMarkMethod::kDefaultTimeoutMs;
    EXPECT_GE(marks[0].deadline_ms, before_deadline);
    EXPECT_LE(marks[0].deadline_ms, after_deadline);
}

TEST_F(CacheManagerTest, TestFinishWriteCacheFullBlockPolicyKeepsPartialMark) {
    auto default_group = registry_manager_->instance_group_configs_["default"];
    ASSERT_TRUE(default_group != nullptr);
    default_group->cache_config_->set_migration_mark_clear_policy(
        MigrationMarkClearPolicy::CLEAR_ON_FULL_BLOCK_COVERED);
    // 清标只对启用了 migration_strategies 的 group 生效。
    EnableTieredMigrationStrategy();

    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "full_policy_instance",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("full_policy_instance");
    ASSERT_TRUE(meta_searcher);

    auto hot_loc =
        std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                        1,
                                        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot_01/blk1?size=1")});
    std::vector<std::string> hot_ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {hot_loc}, hot_ids));
    cache_manager_->migration_manager()->MarkForTieredWrite("full_policy_instance", {1}, "cold_01");
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("full_policy_instance", 1));

    auto partial_cold_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://cold_01/blk1/tp0?size=1")});
    std::vector<std::string> partial_ids;
    ASSERT_EQ(EC_OK,
              BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {partial_cold_loc}, partial_ids));
    auto partial_info = std::make_unique<WriteLocationManager::WriteLocationInfo>();
    partial_info->keys = {1};
    partial_info->location_ids = {partial_ids[0]};
    ASSERT_EQ(EC_OK,
              cache_manager_->FinishWriteCache(request_context_.get(),
                                               "full_policy_instance",
                                               "sess_partial",
                                               static_cast<BlockMaskOffset>(1),
                                               std::move(partial_info)));
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("full_policy_instance", 1));

    auto remaining_cold_loc =
        std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                        3,
                                        std::vector<LocationSpec>{
                                            LocationSpec("tp1", "dummy://cold_01/blk1/tp1?size=1"),
                                            LocationSpec("tp2", "dummy://cold_01/blk1/tp2?size=1"),
                                            LocationSpec("tp3", "dummy://cold_01/blk1/tp3?size=1"),
                                        });
    std::vector<std::string> remaining_ids;
    ASSERT_EQ(EC_OK,
              BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {remaining_cold_loc}, remaining_ids));
    auto remaining_info = std::make_unique<WriteLocationManager::WriteLocationInfo>();
    remaining_info->keys = {1};
    remaining_info->location_ids = {remaining_ids[0]};
    ASSERT_EQ(EC_OK,
              cache_manager_->FinishWriteCache(request_context_.get(),
                                               "full_policy_instance",
                                               "sess_remaining",
                                               static_cast<BlockMaskOffset>(1),
                                               std::move(remaining_info)));
    ASSERT_FALSE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("full_policy_instance", 1));
}

// FinishWriteCache 的 mark 清理只对启用了 tiered migration 的 instance group 生效，
// 与 FilterWriteCache 的 mark 消费入口对称。未配置 migration_strategies 的 group（例如 admin
// 旁路直接打标）不应在 finish 时清标——这类 mark 由 MigrationManager 的超时线程兜底清理。
// 旧实现用 `migration_manager_ != nullptr`（恒真）当门，会错误地对无策略 group 也清标。
TEST_F(CacheManagerTest, TestFinishWriteCacheSkipsTieredMarkWhenMigrationDisabled) {
    // 注意：默认 "default" group 未启用 migration 策略（未调用 EnableTieredMigrationStrategy）。
    cache_manager_->RegisterInstance(request_context_.get(),
                                     "default",
                                     "tiered_disabled_finish",
                                     64,
                                     createLocationSpecInfos(),
                                     createModelDeployment(),
                                     std::vector<LocationSpecGroup>());
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("tiered_disabled_finish");
    ASSERT_TRUE(meta_searcher);

    auto hot_loc =
        std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                        1,
                                        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot/blk1?size=1")});
    std::vector<std::string> hot_ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {hot_loc}, hot_ids));

    // 通过 admin 旁路直接打标（该 group 无策略）。
    cache_manager_->migration_manager()->MarkForTieredWrite("tiered_disabled_finish", {1}, "cold_01");
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("tiered_disabled_finish", 1));

    // 构造一个 finish 后会 SERVING 且覆盖 spec 的冷层 target：旧代码（判空恒真）据此清标，
    // 新代码因该 group 未启用 migration 而跳过整段，mark 应保留。
    auto target_loc =
        std::make_shared<CacheLocation>(DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                        1,
                                        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://cold_01/blk1?size=1")});
    std::vector<std::string> target_ids;
    ASSERT_EQ(EC_OK, BatchAddLocationForTest(meta_searcher, request_context_.get(), {1}, {target_loc}, target_ids));
    ASSERT_EQ(1u, target_ids.size());

    auto info = std::make_unique<WriteLocationManager::WriteLocationInfo>();
    info->keys = {1};
    info->location_ids = {target_ids[0]};
    BlockMask success_mask = static_cast<BlockMaskOffset>(1);
    auto ec = cache_manager_->FinishWriteCache(
        request_context_.get(), "tiered_disabled_finish", "sess_disabled", success_mask, std::move(info));
    ASSERT_EQ(EC_OK, ec);
    // 关键断言：未启用 migration 的 group，finish 不清标。
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("tiered_disabled_finish", 1));
}

} // namespace kv_cache_manager
