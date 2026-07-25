#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
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
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "stub.h"

namespace {
static const std::string default_storage_configs(
    "[{\"type\":\"file\",\"is_available\":true,\"global_unique_name\":\"nfs_01\",\"storage_spec\":{"
    "\"root_path\":\"/tmp/nfs/\",\"key_count_per_file\":8}}]");
} // namespace

namespace kv_cache_manager {

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

    void FailNextSync() {
        std::lock_guard<std::mutex> lock(control_mutex_);
        fail_next_sync_ = true;
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
            const auto one_result = MetaLocalBackend::Upsert(
                request_context,
                KeyTypeVec{keys[i]},
                CacheLocationMapVector{locations[i]},
                PropertyMapVector{properties[i]});
            results[i] = one_result.empty() ? EC_ERROR : one_result.front();
        }
        return results;
    }

    bool Sync(const KeyTypeVec &keys) noexcept override {
        {
            std::lock_guard<std::mutex> lock(control_mutex_);
            if (fail_next_sync_) {
                fail_next_sync_ = false;
                return false;
            }
        }
        return MetaLocalBackend::Sync(keys);
    }

private:
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
    std::optional<int64_t> fail_key_on_next_upsert_;
    bool fail_next_sync_ = false;
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

    std::shared_ptr<EventReportBackend> InstallEventReportBackend() {
        const std::string storage_name = "event_report_default";
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
        EXPECT_EQ(EC_OK,
                  cache_manager_->ReportEvent(request_context_.get(), &register_request, &register_response));

        proto::meta::ReportEventRequest snapshot_request;
        snapshot_request.set_instance_id(instance_id);
        snapshot_request.set_host_ip_port(host);
        snapshot_request.set_storage_type(storage_type);
        auto *snapshot_event = snapshot_request.add_events();
        snapshot_event->set_event_type(proto::meta::EVENT_BLOCK_SNAPSHOT);
        snapshot_event->mutable_block_snapshot();
        proto::meta::ReportEventResponse snapshot_response;
        EXPECT_EQ(EC_OK,
                  cache_manager_->ReportEvent(request_context_.get(), &snapshot_request, &snapshot_response));
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

    static proto::meta::ReportEventRequest MakeSnapshotRequest(
        const std::string &host,
        const std::vector<std::pair<int64_t, std::string>> &key_sources) {
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

    static proto::meta::ReportEventRequest MakeAddRequest(const std::string &host,
                                                           int64_t key,
                                                           const std::string &source) {
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
        auto [ec, locations] = cache_manager_->GetCacheLocation(&context,
                                                                "test_instance",
                                                                CacheManager::QueryType::QT_BATCH_GET,
                                                                keys,
                                                                {},
                                                                BlockMask{},
                                                                0,
                                                                {});
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
        ASSERT_EQ(3, cache_locations_view.size());
        ASSERT_EQ(3, metas.size());
        for (int i = 0; i < 3; ++i) {
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
        ASSERT_EQ(3, cache_locations_view.size());
        ASSERT_EQ(3, metas.size());
        for (int i = 0; i < 3; ++i) {
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
        ASSERT_EQ(5, cache_locations_view.size());
        ASSERT_EQ(5, metas.size());
        std::vector<int> pos_vec = {0, 1, 2, 4};
        std::map<std::string, std::string> meta;
        ASSERT_TRUE(Jsonizable::FromJsonString(metas[3], meta));
        ASSERT_EQ(CacheLocation::CacheLocationStatusToString(CacheLocationStatus::CLS_NOT_FOUND), meta.at("status"));
        for (int pos : pos_vec) {
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
        for (int i = start; i < start + 10; ++i) {
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
        for (int i = start; i < start + 10; ++i) {
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
    // valid URIs whose hostname does not match any registered backend;
    // DataStorageManager::Exist returns an empty vector, and
    // std::all_of on an empty range is true -> functor returns true
    auto func = cache_manager_->GetCheckLocDataExistFunc("test_instance");

    CacheLocation loc;
    loc.set_status(CLS_SERVING);
    loc.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    loc.set_location_specs({LocationSpec("tp0", "file://nonexistent_backend/path")});
    ASSERT_EQ(func(loc), true);
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
        return CallReportEvent(
            MakeSnapshotRequest(host, {{key, "snapshot_after_delta"}}), "ordering_snapshot_second");
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

    meta_backend->BlockNextUpsert();
    auto snapshot_future = std::async(std::launch::async, [this, host, key] {
        return CallReportEvent(MakeSnapshotRequest(host, {{key, "must_not_commit"}}), "host_down_snapshot_in_flight");
    });
    const bool snapshot_entered = meta_backend->WaitUntilUpsertEntered(std::chrono::seconds(1));
    if (!snapshot_entered) {
        meta_backend->ReleaseUpsert();
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

    meta_backend->ReleaseUpsert();
    ASSERT_EQ(std::future_status::ready, snapshot_future.wait_for(std::chrono::seconds(2)));
    const auto [snapshot_ec, snapshot_response] = snapshot_future.get();
    EXPECT_EQ(EC_PARTIAL_OK, snapshot_ec);
    EXPECT_EQ(proto::meta::INTERNAL_ERROR, snapshot_response.header().status().code());
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

    meta_backend->BlockNextUpsert();
    auto delta_future = std::async(std::launch::async, [this, host, key] {
        return CallReportEvent(MakeAddRequest(host, key, "admitted_before_host_down"), "host_down_delta_in_flight");
    });
    const bool delta_entered = meta_backend->WaitUntilUpsertEntered(std::chrono::seconds(1));
    if (!delta_entered) {
        meta_backend->ReleaseUpsert();
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

    meta_backend->ReleaseUpsert();
    ASSERT_EQ(std::future_status::ready, delta_future.wait_for(std::chrono::seconds(2)));
    const auto [delta_ec, delta_response] = delta_future.get();
    EXPECT_EQ(EC_OK, delta_ec);
    EXPECT_TRUE(delta_response.committed_snapshot_version().empty());
    EXPECT_TRUE(delta_response.snapshot_required());
    EXPECT_FALSE(event_backend->IsNodeRegistered("test_instance", host));
    EXPECT_TRUE(QueryEventReportUris({key}).empty());
}

TEST_F(CacheManagerTest, TestReportEventPartialSnapshotFailureIsFailClosedAndRetryConverges) {
    const std::string host = "192.168.10.3:8080";
    const int64_t key_a = 9420;
    const int64_t key_b = 9421;
    auto event_backend = InstallEventReportBackend();
    auto *meta_backend = InstallControllableMetaBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_NE(nullptr, meta_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    auto [baseline_ec, baseline_response] = CallReportEvent(
        MakeSnapshotRequest(host, {{key_a, "baseline_a"}, {key_b, "baseline_b"}}), "partial_baseline");
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

    // key_a was physically overwritten with an uncommitted token and must be
    // hidden; a delta admitted after abort inherits the old committed token
    // and keeps the reporter usable until its full retry.
    const auto visible_after_failure = QueryEventReportUris({key_a, key_b});
    ASSERT_EQ(1u, visible_after_failure.size());
    EXPECT_NE(std::string::npos, visible_after_failure[0].find("source=delta_after_failed_snapshot"));
    EXPECT_NE(std::string::npos, visible_after_failure[0].find("s_version=" + baseline_token));

    const auto [retry_ec, retry_response] = CallReportEvent(
        MakeSnapshotRequest(host, {{key_a, "retry_a"}, {key_b, "retry_b"}}), "partial_full_retry");
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

TEST_F(CacheManagerTest, TestReportEventSyncFailureDoesNotPublishAndImmediateFullRetryConverges) {
    const std::string host = "192.168.10.30:8080";
    const int64_t key = 9425;
    auto event_backend = InstallEventReportBackend();
    auto *meta_backend = InstallControllableMetaBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_NE(nullptr, meta_backend);
    event_backend->SetSnapshotMinIntervalMsForTest(30'000);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));

    meta_backend->FailNextSync();
    const auto [failed_ec, failed_response] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "sync_failure"}}), "sync_failure_snapshot");
    EXPECT_EQ(EC_PARTIAL_OK, failed_ec);
    EXPECT_EQ(proto::meta::INTERNAL_ERROR, failed_response.header().status().code());
    ASSERT_EQ(1, failed_response.item_results_size());
    EXPECT_EQ(proto::meta::INTERNAL_ERROR, failed_response.item_results(0));
    EXPECT_TRUE(failed_response.committed_snapshot_version().empty());
    EXPECT_TRUE(failed_response.snapshot_required());
    EXPECT_TRUE(event_backend->GetSnapshotVersion({"test_instance", host}).empty());
    EXPECT_TRUE(QueryEventReportUris({key}).empty());
    ASSERT_EQ(1u, QueryRawEventReportUris(key).size());

    // A failed attempt must not start the rate-limit interval.
    const auto [retry_ec, retry_response] =
        CallReportEvent(MakeSnapshotRequest(host, {{key, "sync_retry"}}), "sync_failure_immediate_retry");
    ASSERT_EQ(EC_OK, retry_ec);
    const std::string committed = retry_response.committed_snapshot_version();
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(committed));
    EXPECT_FALSE(retry_response.snapshot_required());

    const auto visible = QueryEventReportUris({key});
    ASSERT_EQ(1u, visible.size());
    EXPECT_NE(std::string::npos, visible.front().find("source=sync_retry"));
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

TEST_F(CacheManagerTest, TestReportEventDistinguishesUnregisteredNodeFromMissingSnapshot) {
    const std::string host = "192.168.10.32:8080";
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);

    proto::meta::ReportEventRequest heartbeat;
    heartbeat.set_instance_id("test_instance");
    heartbeat.set_host_ip_port(host);
    heartbeat.set_storage_type(proto::meta::ST_EVENT_REPORT_L2);
    heartbeat.add_events()->set_event_type(proto::meta::EVENT_HEARTBEAT);
    heartbeat.mutable_events(0)->mutable_heartbeat();
    const auto [heartbeat_ec, heartbeat_response] = CallReportEvent(heartbeat, "unregistered_heartbeat");
    EXPECT_EQ(EC_PARTIAL_OK, heartbeat_ec);
    EXPECT_EQ(proto::meta::NODE_NOT_REGISTERED, heartbeat_response.header().status().code());
    ASSERT_EQ(1, heartbeat_response.item_results_size());
    EXPECT_EQ(proto::meta::NODE_NOT_REGISTERED, heartbeat_response.item_results(0));

    const auto [unregistered_delta_ec, unregistered_delta] =
        CallReportEvent(MakeAddRequest(host, 9427, "unregistered_delta"), "unregistered_delta");
    EXPECT_EQ(EC_PARTIAL_OK, unregistered_delta_ec);
    EXPECT_EQ(proto::meta::NODE_NOT_REGISTERED, unregistered_delta.header().status().code());
    EXPECT_TRUE(unregistered_delta.committed_snapshot_version().empty());
    EXPECT_TRUE(unregistered_delta.snapshot_required());

    const auto [unregistered_snapshot_ec, unregistered_snapshot] =
        CallReportEvent(MakeSnapshotRequest(host, {}), "unregistered_snapshot");
    EXPECT_EQ(EC_PARTIAL_OK, unregistered_snapshot_ec);
    EXPECT_EQ(proto::meta::NODE_NOT_REGISTERED, unregistered_snapshot.header().status().code());
    EXPECT_TRUE(unregistered_snapshot.committed_snapshot_version().empty());
    EXPECT_TRUE(unregistered_snapshot.snapshot_required());

    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));
    const auto [missing_snapshot_ec, missing_snapshot] =
        CallReportEvent(MakeAddRequest(host, 9427, "missing_snapshot"), "registered_without_snapshot");
    EXPECT_EQ(EC_PARTIAL_OK, missing_snapshot_ec);
    EXPECT_EQ(proto::meta::SNAPSHOT_REQUIRED, missing_snapshot.header().status().code());
    EXPECT_TRUE(missing_snapshot.committed_snapshot_version().empty());
    EXPECT_TRUE(missing_snapshot.snapshot_required());

    const auto [baseline_ec, baseline] = CallReportEvent(MakeSnapshotRequest(host, {}), "registered_empty_snapshot");
    ASSERT_EQ(EC_OK, baseline_ec);
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(baseline.committed_snapshot_version()));

    event_backend->SetNodeUnavailable("test_instance", host);
    const auto [unavailable_delta_ec, unavailable_delta] =
        CallReportEvent(MakeAddRequest(host, 9427, "registered_but_unavailable"), "registered_unavailable_delta");
    ASSERT_EQ(EC_OK, unavailable_delta_ec);
    EXPECT_EQ(baseline.committed_snapshot_version(), unavailable_delta.committed_snapshot_version());
    EXPECT_TRUE(QueryEventReportUris({9427}).empty());

    ASSERT_EQ(EC_OK, event_backend->OnHeartbeat("test_instance", host, {}));
    const auto visible = QueryEventReportUris({9427});
    ASSERT_EQ(1u, visible.size());
    EXPECT_NE(std::string::npos, visible.front().find("source=registered_but_unavailable"));
}

TEST_F(CacheManagerTest, TestReportEventRejectsCanonicalDuplicateSnapshotKeysButAllowsDifferentMedia) {
    const std::string host = "192.168.10.33:8080";
    const int64_t key = 9428;
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem", "disk"}));

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
    EXPECT_TRUE(duplicate_response.committed_snapshot_version().empty());
    EXPECT_TRUE(duplicate_response.snapshot_required());
    EXPECT_TRUE(QueryRawEventReportUris(key).empty());

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

    const auto [host_a_baseline_ec, host_a_baseline] = CallReportEvent(
        MakeSnapshotRequest(host_a, {{stale_key, "host_a_stale"}, {current_key, "host_a_old_current"}}),
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

    // Query filtering must hide the omitted block immediately, before making
    // any assumption about asynchronous physical reclamation.
    EXPECT_TRUE(QueryEventReportUris({stale_key}).empty());

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline && !QueryRawEventReportUris(stale_key).empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_TRUE(QueryRawEventReportUris(stale_key).empty());

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
    ASSERT_EQ(EC_OK,
              event_backend->BeginSnapshot({"test_instance", host}, cleanup_token, retry_after_ms));
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
            const std::vector<std::vector<std::string>> &expected_values) {
            captured_keys = keys;
            captured_location_ids = location_ids;
            captured_expected_values = expected_values;
        });
    ASSERT_EQ(
        EC_OK,
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
    ASSERT_EQ(EC_OK,
              event_backend->BeginSnapshot({"test_instance", host}, refreshed_token, retry_after_ms));
    ASSERT_NE(cleanup_token, refreshed_token);
    std::string refreshed_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://" + host + "/mem?source=refreshed", refreshed_token, refreshed_uri));
    MetaSearcher *meta_searcher = cache_manager_->meta_searcher_manager_->GetMetaSearcher("test_instance");
    ASSERT_NE(nullptr, meta_searcher);
    std::vector<ErrorCode> replace_results;
    ASSERT_EQ(EC_OK,
              meta_searcher->BatchReplaceLocationSpecs(
                  request_context_.get(),
                  {key},
                  {{{event_backend->BuildLocationId("mem", host),
                     DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2,
                     CacheLocationStatus::CLS_SERVING,
                     {LocationSpec("tp0", refreshed_uri)}}}},
                  replace_results));
    ASSERT_EQ((std::vector<ErrorCode>{EC_OK}), replace_results);
    ASSERT_TRUE(meta_searcher->Sync({key}));
    ASSERT_TRUE(event_backend->CommitSnapshotVersion({"test_instance", host}, refreshed_token));

    CacheLocationDelRequest stale_cleanup_request{
        .instance_id = "test_instance",
        .block_keys = captured_keys,
        .location_ids = captured_location_ids,
        .delay = std::chrono::seconds(0),
        .expected_location_values = captured_expected_values,
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
    EXPECT_TRUE(QueryEventReportUris({key}).empty());

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline && !QueryRawEventReportUris(key).empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_TRUE(QueryRawEventReportUris(key).empty());
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

TEST_F(CacheManagerTest, TestGetCheckLocDataExistFuncRequiresAllSpecsToMatchCurrentReporterToken) {
    const std::string host = "192.168.10.35:8080";
    auto event_backend = InstallEventReportBackend();
    ASSERT_NE(nullptr, event_backend);
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host, {"mem"}));
    const auto [snapshot_ec, snapshot_response] =
        CallReportEvent(MakeSnapshotRequest(host, {}), "query_filter_baseline");
    ASSERT_EQ(EC_OK, snapshot_ec);
    const std::string token = snapshot_response.committed_snapshot_version();

    std::string current_uri;
    std::string unknown_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://physical-cache:9600/mem?source=current", token, current_uri));
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://physical-cache:9600/mem?source=unknown", "ffffffffffffffffffffffffffffffff", unknown_uri));

    CacheLocation location;
    location.set_id(event_backend->BuildLocationId("mem", host));
    location.set_status(CLS_SERVING);
    location.set_type(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2);
    location.set_location_specs({LocationSpec("tp0", current_uri), LocationSpec("tp1", current_uri)});
    auto check = cache_manager_->GetCheckLocDataExistFunc("test_instance");
    EXPECT_TRUE(check(location));

    location.set_location_specs({LocationSpec("tp0", current_uri), LocationSpec("tp1", unknown_uri)});
    EXPECT_FALSE(check(location));
    location.set_location_specs({LocationSpec("tp0", current_uri), LocationSpec("tp1", "event_report://raw/mem")});
    EXPECT_FALSE(check(location));
    location.set_location_specs({});
    EXPECT_FALSE(check(location));

    location.set_location_specs({LocationSpec("tp0", current_uri)});
    event_backend->SetNodeUnavailable("test_instance", host);
    EXPECT_FALSE(check(location));
    ASSERT_EQ(EC_OK, event_backend->OnHeartbeat("test_instance", host, {}));
    EXPECT_TRUE(check(location));

    // A token is opaque, but its owner must still match the reporter encoded
    // by the stable location id. A healthy token belonging to another host or
    // another instance must fail closed.
    const std::string other_host = "192.168.10.36:8080";
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", other_host, {"mem"}));
    const auto [other_host_ec, other_host_snapshot] =
        CallReportEvent(MakeSnapshotRequest(other_host, {}), "query_filter_other_host");
    ASSERT_EQ(EC_OK, other_host_ec);
    std::string other_host_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://physical-cache:9600/mem", other_host_snapshot.committed_snapshot_version(), other_host_uri));
    location.set_location_specs({LocationSpec("tp0", other_host_uri)});
    EXPECT_FALSE(check(location));

    const ReporterSnapshotKey other_instance{"other_instance", host};
    ASSERT_EQ(EC_OK, event_backend->RegisterNode(other_instance.instance_id, host, {"mem"}));
    std::string other_instance_token;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, event_backend->BeginSnapshot(other_instance, other_instance_token, retry_after_ms));
    ASSERT_TRUE(event_backend->CommitSnapshotVersion(other_instance, other_instance_token));
    std::string other_instance_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://physical-cache:9600/mem", other_instance_token, other_instance_uri));
    location.set_location_specs({LocationSpec("tp0", other_instance_uri)});
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

    // A new authoritative snapshot makes the omitted block's old token
    // invisible immediately, even before its physical cleanup completes.
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

    // Once host A is unregistered, neither heartbeat nor registration alone
    // may resurrect its old token. A new full snapshot is mandatory.
    ASSERT_EQ(EC_OK, event_backend->UnregisterNode("test_instance", host_a));
    visible = query_visibility({current_key, down_key});
    ASSERT_EQ(2u, visible.size());
    EXPECT_TRUE(visible[0].empty());
    ASSERT_EQ(1u, visible[1].size());
    EXPECT_EQ(EC_NODE_NOT_REGISTERED, event_backend->OnHeartbeat("test_instance", host_a, {}));
    ASSERT_EQ(EC_OK, event_backend->RegisterNode("test_instance", host_a, {"mem"}));
    visible = query_visibility({current_key});
    ASSERT_EQ(1u, visible.size());
    EXPECT_TRUE(visible[0].empty());

    const auto [delta_ec, delta_response] =
        CallReportEvent(MakeAddRequest(host_a, current_key, "must_be_rejected"), "query_lifecycle_delta_rejected");
    EXPECT_EQ(EC_PARTIAL_OK, delta_ec);
    EXPECT_EQ(proto::meta::SNAPSHOT_REQUIRED, delta_response.header().status().code());
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
    // Start from a valid committed snapshot. Otherwise an unversioned URI
    // would be rejected as stale before this test ever reached the node
    // availability branch.
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

    // Registration alone starts a fresh reporter lifecycle. It must not
    // resurrect the previous token.
    ASSERT_EQ(EC_OK, event_report_backend->RegisterNode(instance_id, node_host, {"mem"}));
    EXPECT_FALSE(func(loc));
    std::string committed = "must-be-cleared";
    EXPECT_EQ(EC_SNAPSHOT_REQUIRED, event_report_backend->BeginDeltaMutation({instance_id, node_host}, committed));
    EXPECT_TRUE(committed.empty());

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
    loc.set_location_specs({LocationSpec("tp0", first_uri)});
    EXPECT_FALSE(func(loc));
}

TEST_F(CacheManagerTest, TestGetSubmitDelReqFunc_NullExecutor) {
    // when schedule_plan_executor_ is null, calling the functor should
    // not crash
    auto saved = cache_manager_->schedule_plan_executor_;
    cache_manager_->schedule_plan_executor_ = nullptr;

    auto func = cache_manager_->GetSubmitDelReqFunc("test_instance");
    func({1, 2, 3}, {{"loc_a"}, {"loc_b"}, {"loc_c"}}, {});

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
    del_func(keys, loc_ids, {});

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
TEST_F(CacheManagerTest, TestReportEventRejectsInvalidRequestsAndMapsItemErrors) {
    auto add_register_event = [](proto::meta::ReportEventRequest &request) {
        auto *event = request.add_events();
        event->set_event_type(proto::meta::EVENT_NODE_REGISTER);
        event->mutable_node_register()->add_mediums("mem");
    };

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
    const std::string snapshot_version =
        InitializeEventReporter(instance_id, host, proto::meta::ST_EVENT_REPORT_L1P5);
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
    const std::string l1p5_version =
        InitializeEventReporter(instance_id, host, proto::meta::ST_EVENT_REPORT_L1P5);
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


TEST_F(CacheManagerTest, TestGetCacheLocationsByBackendWithBackendSelectors) {
    auto expected_reg = std::pair<ErrorCode, std::string>(EC_OK, default_storage_configs);
    ASSERT_EQ(expected_reg,
              cache_manager_->RegisterInstance(request_context_.get(),
                                               "default",
                                               "test_instance",
                                               64,
                                               createLocationSpecInfos(),
                                               createModelDeployment(),
                                               std::vector<LocationSpecGroup>()));

    std::vector<int64_t> all_keys{300, 400, 500, 600, 700};

    // Write NFS locations for a subset of keys (non-contiguous: 300, 500, 700)
    std::vector<int64_t> nfs_keys{300, 500, 700};
    {
        auto [ec, swci] =
            cache_manager_->StartWriteCache(request_context_.get(), "test_instance", nfs_keys, {}, {}, 100000000);
        ASSERT_EQ(EC_OK, ec);
        BlockMask bm = static_cast<size_t>(nfs_keys.size());
        ASSERT_EQ(
            EC_OK,
            cache_manager_->FinishWriteCache(request_context_.get(), "test_instance", swci.write_session_id(), bm));
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
        InitializeEventReporter("test_instance", pd.host, proto::meta::ST_EVENT_REPORT_L2);
        proto::meta::ReportEventRequest req;
        req.set_instance_id("test_instance");
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
        auto [ec, locs] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                     "test_instance",
                                                                     CacheManager::QueryType::QT_BATCH_GET,
                                                                     all_keys,
                                                                     {},
                                                                     bm,
                                                                     0,
                                                                     {},
                                                                     {});
        ASSERT_EQ(EC_BADARGS, ec);
    }

    // --- Test 2: EVENT_REPORT PREFIX + NFS (NFS on 300,500,700 should not affect event report peer selection) ---
    {
        std::vector<BackendSelector> selectors = {
            {DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2, LocationSelectStrategy::LSS_V6D_PREFIX},
            {DataStorageType::DATA_STORAGE_TYPE_NFS, LocationSelectStrategy::LSS_WEIGHTED_RANDOM},
        };
        BlockMask bm = static_cast<size_t>(0);
        auto [ec, locs] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                     "test_instance",
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
                                                                     "test_instance",
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
                                                                     "test_instance",
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
                                                                     "test_instance",
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
                                                                     "test_instance",
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
                                                                     "test_instance",
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

    // --- Test 8: location_spec_names filter still works with backend_selectors ---
    {
        std::vector<BackendSelector> selectors = {
            {DataStorageType::DATA_STORAGE_TYPE_NFS, LocationSelectStrategy::LSS_WEIGHTED_RANDOM},
        };
        BlockMask bm = static_cast<size_t>(0);
        auto [ec, locs] = cache_manager_->GetCacheLocationsByBackend(request_context_.get(),
                                                                     "test_instance",
                                                                     CacheManager::QueryType::QT_BATCH_GET,
                                                                     {300},
                                                                     {},
                                                                     bm,
                                                                     0,
                                                                     {"tp0", "tp2"},
                                                                     selectors);
        ASSERT_EQ(EC_OK, ec);
        ASSERT_EQ(1u, locs.size());
        const auto &kl = locs[0].cache_locations_view();
        ASSERT_EQ(1u, kl.size());
        EXPECT_EQ(2u, kl[0].location_specs().size());
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

    // Helper: find a host's prefix_match_blocks in the result
    auto find_prefix = [](const std::vector<CacheManager::HostCacheMatch> &hosts, const std::string &host) -> int64_t {
        for (const auto &h : hosts) {
            if (h.host_ip_port == host) {
                return h.prefix_match_blocks;
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

    // --- Test 9: unavailable host is filtered even before metadata cleanup ---
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
                                               createModelDeployment(),
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
    InitializeEventReporter(instance_id, host_a, proto::meta::ST_EVENT_REPORT_L1P5);
    InitializeEventReporter(instance_id, host_b, proto::meta::ST_EVENT_REPORT_L1P5);
    InitializeEventReporter(instance_id, host_c, proto::meta::ST_EVENT_REPORT_L1P5);

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

    const std::string host_d = "10.0.1.4:8080";
    InitializeEventReporter(instance_id, host_d, proto::meta::ST_EVENT_REPORT_L1P5);
    report_specs(host_d, 200, {"full_0", "linear_0", "linear_1"});
    report_specs(host_d, 300, {"full_0", "linear_0", "linear_1"});

    auto find_prefix = [](const std::vector<CacheManager::HostCacheMatch> &hosts, const std::string &host) -> int64_t {
        for (const auto &h : hosts) {
            if (h.host_ip_port == host) {
                return h.prefix_match_blocks;
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
        EXPECT_EQ(4, find_prefix(matches, host_b));
        EXPECT_EQ(-1, find_prefix(matches, host_c));
        EXPECT_EQ(-1, find_prefix(matches, host_d));
    };
    expect_mamba_matches(hosts);

    // An explicit request query type takes precedence over the registered default.
    auto [explicit_ec, explicit_hosts] = cache_manager_->GetHostCacheState(
        request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH, keys);
    ASSERT_EQ(EC_OK, explicit_ec);
    EXPECT_EQ(4, find_prefix(explicit_hosts, host_a));
    EXPECT_EQ(4, find_prefix(explicit_hosts, host_b));
    EXPECT_EQ(2, find_prefix(explicit_hosts, host_c));
    EXPECT_EQ(-1, find_prefix(explicit_hosts, host_d));

    auto [fallback_ec, fallback_hosts] = cache_manager_->GetHostCacheState(
        request_context_.get(), instance_id, CacheManager::QueryType::QT_UNSPECIFIED, keys);
    ASSERT_EQ(EC_OK, fallback_ec);
    expect_mamba_matches(fallback_hosts);

    {
        CacheManager::KeyVector break_keys = {100, 500, 400};
        auto [break_ec, break_hosts] = cache_manager_->GetHostCacheState(
            request_context_.get(), instance_id, CacheManager::QueryType::QT_PREFIX_MATCH_WITH_MAMBA, break_keys);
        ASSERT_EQ(EC_OK, break_ec);
        EXPECT_EQ(1, find_prefix(break_hosts, host_a));
        EXPECT_EQ(-1, find_prefix(break_hosts, host_b));
        EXPECT_EQ(-1, find_prefix(break_hosts, host_c));
        EXPECT_EQ(-1, find_prefix(break_hosts, host_d));
    }

    {
        CacheManager::KeyVector keys_without_host_d_first = {100, 200, 300};
        auto [absent_ec, absent_hosts] =
            cache_manager_->GetHostCacheState(request_context_.get(),
                                              instance_id,
                                              CacheManager::QueryType::QT_PREFIX_MATCH_WITH_MAMBA,
                                              keys_without_host_d_first);
        ASSERT_EQ(EC_OK, absent_ec);
        EXPECT_EQ(3, find_prefix(absent_hosts, host_a));
        EXPECT_EQ(-1, find_prefix(absent_hosts, host_b));
        EXPECT_EQ(-1, find_prefix(absent_hosts, host_c));
        EXPECT_EQ(-1, find_prefix(absent_hosts, host_d));
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
        auto loc = std::make_shared<CacheLocation>(
            DataStorageType::DATA_STORAGE_TYPE_DUMMY,
            1,
            std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot/blk1?size=1")});
        std::vector<std::string> ids;
        ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {loc}, ids));
    }
    cache_manager_->migration_manager()->MarkForTieredWrite("placeholder_id", {1}, "cold_01");

    CacheManager::KeyVector keys = {1, 2};
    CacheManager::KeyVector new_keys;
    std::vector<std::string_view> new_sgn;
    BlockMask block_mask;
    std::vector<std::string> new_targets;
    auto ec = cache_manager_->FilterWriteCache(
        request_context_.get(), "placeholder_id", meta_searcher, keys, new_keys, {}, new_sgn, block_mask, 1, new_targets);
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
    ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {hot_loc}, ids));
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
    ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {hot_loc}, ids));
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
    MetaSearcher *meta_searcher =
        cache_manager_->meta_searcher_manager_->GetMetaSearcher("unavailable_tiered_target");
    ASSERT_TRUE(meta_searcher);

    auto hot_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot_01/unavailable_target?size=1")});
    std::vector<std::string> ids;
    ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {hot_loc}, ids));
    std::vector<std::vector<MetaSearcher::LocationCASTask>> cas_tasks{
        {MetaSearcher::LocationCASTask{ids[0], CLS_WRITING, CLS_SERVING}}};
    std::vector<std::vector<ErrorCode>> cas_results;
    ASSERT_EQ(EC_OK, meta_searcher->BatchCASLocationStatus(request_context_.get(), {1}, cas_tasks, cas_results));
    ASSERT_EQ(EC_OK,
              cache_manager_->migration_manager()->MarkForTieredWrite(
                  "unavailable_tiered_target", {1}, "cold_01"));
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
              cache_manager_->migration_manager()->MarkForTieredWrite(
                  "migration_target_quota", {1}, "cold_01"));
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
        ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {make_hot_loc(uri)}, ids));
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
        ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {block_key}, {loc}, ids));
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

    auto hot_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot_01/blk1?size=1")});
    std::vector<std::string> ids;
    ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {hot_loc}, ids));
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

    auto writing_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        4,
        std::vector<LocationSpec>{
            LocationSpec("tp0", "dummy://cold_01/blk1/tp0?size=1"),
            LocationSpec("tp1", "dummy://cold_01/blk1/tp1?size=1"),
            LocationSpec("tp2", "dummy://cold_01/blk1/tp2?size=1"),
            LocationSpec("tp3", "dummy://cold_01/blk1/tp3?size=1"),
        });
    auto serving_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        4,
        std::vector<LocationSpec>{
            LocationSpec("tp0", "dummy://cold_01/blk2/tp0?size=1"),
            LocationSpec("tp1", "dummy://cold_01/blk2/tp1?size=1"),
            LocationSpec("tp2", "dummy://cold_01/blk2/tp2?size=1"),
            LocationSpec("tp3", "dummy://cold_01/blk2/tp3?size=1"),
        });
    std::vector<std::string> ids;
    ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1, 2}, {writing_loc, serving_loc}, ids));
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
    auto ec = cache_manager_->FilterWriteCache(
        request_context_.get(), "placeholder_id", meta_searcher, keys, new_keys, {}, new_sgn, block_mask, 1, new_targets);
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
    auto cold_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        4,
        std::vector<LocationSpec>{
            LocationSpec("tp0", "dummy://cold_01/blk1/tp0?size=1"),
            LocationSpec("tp1", "dummy://cold_01/blk1/tp1?size=1"),
            LocationSpec("tp2", "dummy://cold_01/blk1/tp2?size=1"),
            LocationSpec("tp3", "dummy://cold_01/blk1/tp3?size=1"),
        });
    std::vector<std::string> ids;
    ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {cold_loc}, ids));
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

    auto hot_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot_01/blk1?size=1")});
    std::vector<std::string> ids;
    ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {hot_loc}, ids));
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
            DataStorageType::DATA_STORAGE_TYPE_DUMMY,
            1,
            std::vector<LocationSpec>{LocationSpec("tp0", uri)});
    };
    std::vector<std::string> ids;
    ASSERT_EQ(EC_OK,
              meta_searcher->BatchAddLocation(
                  request_context_.get(), {1}, {make_hot_loc("dummy://hot_01/blk1_a?size=1")}, ids));
    ASSERT_EQ(1u, ids.size());
    std::vector<std::vector<MetaSearcher::LocationCASTask>> cas_tasks{
        {MetaSearcher::LocationCASTask{ids[0], CLS_WRITING, CLS_SERVING}}};
    std::vector<std::vector<ErrorCode>> cas_results;
    ASSERT_EQ(EC_OK, meta_searcher->BatchCASLocationStatus(request_context_.get(), {1}, cas_tasks, cas_results));

    ids.clear();
    ASSERT_EQ(EC_OK,
              meta_searcher->BatchAddLocation(
                  request_context_.get(), {1}, {make_hot_loc("dummy://hot_02/blk1_b?size=1")}, ids));
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

    auto cold_f0_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        2,
        std::vector<LocationSpec>{
            LocationSpec("tp0_F0", "dummy://cold_01/blk1/tp0_F0?size=1"),
            LocationSpec("tp1_F0", "dummy://cold_01/blk1/tp1_F0?size=1"),
        });
    std::vector<std::string> ids;
    ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {cold_f0_loc}, ids));
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
        auto loc = std::make_shared<CacheLocation>(
            DataStorageType::DATA_STORAGE_TYPE_DUMMY,
            1,
            std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot/blk1?size=1")});
        ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {loc}, source_ids));
    }
    ASSERT_EQ(1u, source_ids.size());
    cache_manager_->migration_manager()->MarkForTieredWrite("placeholder_id", {1}, "cold_01");
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("placeholder_id", 1));

    auto target_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://cold_01/blk1?size=1")});
    std::vector<std::string> target_ids;
    ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {target_loc}, target_ids));
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
    auto *meta_searcher =
        cache_manager_->meta_searcher_manager_->GetMetaSearcher("admin_mark_timeout_instance");
    ASSERT_TRUE(meta_searcher);
    auto source_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot_01/blk1?size=1")});
    std::vector<std::string> source_ids;
    ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {source_loc}, source_ids));
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> status_tasks = {
        {{source_ids[0], CacheLocationStatus::CLS_SERVING}}};
    std::vector<std::vector<ErrorCode>> status_results;
    ASSERT_EQ(EC_OK,
              meta_searcher->BatchUpdateLocationStatus(
                  request_context_.get(), {1}, status_tasks, status_results));

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
    ASSERT_EQ(EC_OK,
              cache_manager_->migration_manager()->BatchGetTieredWriteTargets(
                  "admin_mark_timeout_instance", {1}, marks));
    ASSERT_EQ(1u, marks.size());
    ASSERT_TRUE(marks[0].HasValidMark());
    const auto before_deadline = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     before.time_since_epoch())
                                     .count() +
                                 kTimeoutMs;
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
    auto *meta_searcher =
        cache_manager_->meta_searcher_manager_->GetMetaSearcher("admin_mark_unmatched_target");
    ASSERT_TRUE(meta_searcher);
    auto source_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot_01/blk1?size=1")});
    std::vector<std::string> source_ids;
    ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {source_loc}, source_ids));
    ASSERT_EQ(1u, source_ids.size());
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> status_tasks = {
        {{source_ids[0], CacheLocationStatus::CLS_SERVING}}};
    std::vector<std::vector<ErrorCode>> status_results;
    ASSERT_EQ(EC_OK,
              meta_searcher->BatchUpdateLocationStatus(
                  request_context_.get(), {1}, status_tasks, status_results));

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
    ASSERT_EQ(EC_OK,
              cache_manager_->migration_manager()->BatchGetTieredWriteTargets(
                  "admin_mark_unmatched_target", {1}, marks));
    ASSERT_EQ(1u, marks.size());
    ASSERT_TRUE(marks[0].HasValidMark());
    EXPECT_EQ("cold_02", marks[0].target);
    const auto before_deadline = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     before.time_since_epoch())
                                     .count() +
                                 MigrationMarkMethod::kDefaultTimeoutMs;
    const auto after_deadline = std::chrono::duration_cast<std::chrono::milliseconds>(after.time_since_epoch()).count() +
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

    auto hot_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot_01/blk1?size=1")});
    std::vector<std::string> hot_ids;
    ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {hot_loc}, hot_ids));
    cache_manager_->migration_manager()->MarkForTieredWrite("full_policy_instance", {1}, "cold_01");
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("full_policy_instance", 1));

    auto partial_cold_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://cold_01/blk1/tp0?size=1")});
    std::vector<std::string> partial_ids;
    ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {partial_cold_loc}, partial_ids));
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

    auto remaining_cold_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        3,
        std::vector<LocationSpec>{
            LocationSpec("tp1", "dummy://cold_01/blk1/tp1?size=1"),
            LocationSpec("tp2", "dummy://cold_01/blk1/tp2?size=1"),
            LocationSpec("tp3", "dummy://cold_01/blk1/tp3?size=1"),
        });
    std::vector<std::string> remaining_ids;
    ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {remaining_cold_loc}, remaining_ids));
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

    auto hot_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://hot/blk1?size=1")});
    std::vector<std::string> hot_ids;
    ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {hot_loc}, hot_ids));

    // 通过 admin 旁路直接打标（该 group 无策略）。
    cache_manager_->migration_manager()->MarkForTieredWrite("tiered_disabled_finish", {1}, "cold_01");
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite("tiered_disabled_finish", 1));

    // 构造一个 finish 后会 SERVING 且覆盖 spec 的冷层 target：旧代码（判空恒真）据此清标，
    // 新代码因该 group 未启用 migration 而跳过整段，mark 应保留。
    auto target_loc = std::make_shared<CacheLocation>(
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("tp0", "dummy://cold_01/blk1?size=1")});
    std::vector<std::string> target_ids;
    ASSERT_EQ(EC_OK, meta_searcher->BatchAddLocation(request_context_.get(), {1}, {target_loc}, target_ids));
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
