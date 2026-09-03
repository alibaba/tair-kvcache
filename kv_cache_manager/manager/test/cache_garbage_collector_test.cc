#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <map>
#include <memory>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/event_report_backend.h"
#include "kv_cache_manager/data_storage/snapshot_uri_utils.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/cache_garbage_collector.h"
#include "kv_cache_manager/manager/migration_manager.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/meta/types.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "stub.h"

namespace kv_cache_manager {
namespace {

using InstanceGroups = std::vector<std::shared_ptr<const InstanceGroup>>;
using InstanceInfos = std::vector<std::shared_ptr<const InstanceInfo>>;
using SubmitAsyncLocation = AsyncDeleteSubmitResult (SchedulePlanExecutor::*)(const CacheLocationDelRequest &);
using SubmitAsyncEventReport =
    AsyncDeleteSubmitResult (SchedulePlanExecutor::*)(const EventReportMetadataDelRequest &);

struct ScanResponse {
    ErrorCode ec{EC_OK};
    MaintenanceScanBatch batch;
};

struct MightExistCall {
    std::string storage_name;
    std::vector<DataStorageUri> uris;
    bool fastpath{false};
};

enum class SubmitMode {
    kReadyOk,
    kReadyPartial,
    kPending,
    kRejected,
    kAcceptedInvalid,
    kFutureException,
    kThrow,
};

class ProbeTestBackend : public DataStorageBackend {
public:
    explicit ProbeTestBackend(DataStorageType type) : DataStorageBackend(nullptr), type_(type) {}

    DataStorageType GetType() override { return type_; }
    bool Available() override { return true; }
    double GetStorageUsageRatio(const std::string &) const override { return 0.0; }
    ErrorCode DoOpen(const StorageConfig &, const std::string &) override { return EC_OK; }
    ErrorCode Close() override { return EC_OK; }
    std::vector<std::pair<ErrorCode, DataStorageUri>>
    Create(const std::vector<std::string> &keys, size_t, const std::string &, std::function<void()>) override {
        return std::vector<std::pair<ErrorCode, DataStorageUri>>(keys.size(), {EC_OK, DataStorageUri{}});
    }
    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &uris, const std::string &, std::function<void()>) override {
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
    DataStorageType type_;
};

ErrorCode list_groups_ec = EC_OK;
InstanceGroups instance_groups;
std::map<std::string, std::pair<ErrorCode, InstanceInfos>> instances_by_group;
std::map<std::string, std::vector<ScanResponse>> scan_responses;
std::map<std::string, size_t> scan_positions;
std::vector<std::pair<std::string, std::string>> scan_calls;
std::set<std::string> missing_indexers;
std::string active_instance_id;
std::shared_ptr<MetaIndexer> dummy_indexer;
std::vector<CacheLocationDelRequest> submitted_requests;
std::vector<EventReportMetadataDelRequest> submitted_event_report_requests;
SubmitMode submit_mode = SubmitMode::kReadyOk;
std::shared_ptr<std::promise<PlanExecuteResult>> pending_delete_promise;
std::vector<std::shared_ptr<std::promise<PlanExecuteResult>>> pending_delete_promises;
size_t list_groups_call_count = 0;
std::atomic<size_t> scan_call_count{0};
std::atomic<bool> block_scan{false};
std::atomic<bool> scan_block_entered{false};
std::atomic<bool> release_scan{true};
std::vector<MightExistCall> might_exist_calls;
std::map<std::string, bool> might_exist_by_uri;
std::map<std::string, std::vector<bool>> might_exist_result_overrides;
std::set<std::string> might_exist_throw_storages;
std::set<std::string> missing_probe_storages;
std::map<std::string, DataStorageType> probe_storage_type_overrides;
std::map<std::string, std::shared_ptr<DataStorageBackend>> storage_backends;
std::map<std::string, std::shared_ptr<const InstanceGroup>> group_configs;
std::map<std::string, std::string> group_by_instance;
std::atomic<bool> block_might_exist{false};
std::atomic<bool> might_exist_block_entered{false};
std::atomic<bool> release_might_exist{true};

std::pair<ErrorCode, InstanceGroups> ListInstanceGroupStub(void *, RequestContext *) {
    ++list_groups_call_count;
    return {list_groups_ec, instance_groups};
}

std::pair<ErrorCode, InstanceInfos> ListInstanceInfoStub(void *, RequestContext *, const std::string &instance_group) {
    const auto it = instances_by_group.find(instance_group);
    if (it == instances_by_group.end()) {
        return {EC_OK, {}};
    }
    return it->second;
}

std::shared_ptr<MetaIndexer> GetMetaIndexerStub(void *, const std::string &instance_id) {
    active_instance_id = instance_id;
    if (missing_indexers.count(instance_id) > 0) {
        return nullptr;
    }
    return dummy_indexer;
}

std::shared_ptr<DataStorageBackend> GetDataStorageBackendStub(void *, const std::string &storage_name) {
    if (const auto backend_it = storage_backends.find(storage_name); backend_it != storage_backends.end()) {
        return backend_it->second;
    }
    if (missing_probe_storages.count(storage_name) > 0) {
        return nullptr;
    }
    const auto it = probe_storage_type_overrides.find(storage_name);
    const auto type = it == probe_storage_type_overrides.end() ? DataStorageType::DATA_STORAGE_TYPE_DUMMY : it->second;
    return std::make_shared<ProbeTestBackend>(type);
}

std::shared_ptr<const InstanceGroup> GetInstanceGroupConfigStub(void *, const std::string &group_name) {
    const auto it = group_configs.find(group_name);
    return it == group_configs.end() ? nullptr : it->second;
}

std::string GetInstanceGroupNameStub(void *, const std::string &instance_id) {
    const auto it = group_by_instance.find(instance_id);
    return it == group_by_instance.end() ? std::string{} : it->second;
}

ErrorCode ScanLocationsForMaintenanceStub(
    void *, RequestContext *, const std::string &cursor, size_t, MaintenanceScanBatch &out) noexcept {
    scan_calls.emplace_back(active_instance_id, cursor);
    scan_call_count.fetch_add(1, std::memory_order_release);
    if (block_scan.load(std::memory_order_acquire)) {
        scan_block_entered.store(true, std::memory_order_release);
        while (!release_scan.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    auto &position = scan_positions[active_instance_id];
    const auto responses_it = scan_responses.find(active_instance_id);
    if (responses_it == scan_responses.end() || position >= responses_it->second.size()) {
        out.Clear();
        out.next_cursor = SCAN_BASE_CURSOR;
        return EC_OK;
    }
    const ScanResponse &response = responses_it->second[position++];
    if (response.ec != EC_OK) {
        out.Clear();
        return response.ec;
    }
    out = response.batch;
    return EC_OK;
}

AsyncDeleteSubmitResult SubmitAsyncStub(void *, const CacheLocationDelRequest &request) {
    submitted_requests.push_back(request);
    if (submit_mode == SubmitMode::kThrow) {
        throw std::runtime_error("injected SubmitAsync exception");
    }
    if (submit_mode == SubmitMode::kRejected) {
        return {};
    }
    if (submit_mode == SubmitMode::kAcceptedInvalid) {
        return {true, {}};
    }

    auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    auto future = promise->get_future();
    if (submit_mode == SubmitMode::kPending) {
        pending_delete_promise = promise;
        pending_delete_promises.push_back(promise);
    } else if (submit_mode == SubmitMode::kFutureException) {
        promise->set_exception(std::make_exception_ptr(std::runtime_error("injected Future exception")));
    } else if (submit_mode == SubmitMode::kReadyPartial) {
        promise->set_value({EC_PARTIAL_OK, "injected partial"});
    } else {
        promise->set_value({EC_OK, ""});
    }
    return {true, std::move(future)};
}

AsyncDeleteSubmitResult SubmitAsyncEventReportStub(void *, const EventReportMetadataDelRequest &request) {
    submitted_event_report_requests.push_back(request);
    if (submit_mode == SubmitMode::kThrow) {
        throw std::runtime_error("injected SubmitAsync exception");
    }
    if (submit_mode == SubmitMode::kRejected) {
        return {};
    }
    if (submit_mode == SubmitMode::kAcceptedInvalid) {
        return {true, {}};
    }

    auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    auto future = promise->get_future();
    if (submit_mode == SubmitMode::kPending) {
        pending_delete_promise = promise;
        pending_delete_promises.push_back(promise);
    } else if (submit_mode == SubmitMode::kFutureException) {
        promise->set_exception(std::make_exception_ptr(std::runtime_error("injected Future exception")));
    } else if (submit_mode == SubmitMode::kReadyPartial) {
        promise->set_value({EC_PARTIAL_OK, "injected partial"});
    } else {
        promise->set_value({EC_OK, ""});
    }
    return {true, std::move(future)};
}

MetaIndexer::Result
GetLocationsFromPersistentErrorStub(void *, RequestContext *, const KeyVector &, CacheLocationMapVector &) noexcept {
    return MetaIndexer::Result(EC_ERROR);
}

std::vector<bool>
DataStorageExistStub(void *, const std::string &storage_name, const std::vector<DataStorageUri> &uris, bool fastpath) {
    might_exist_calls.push_back({storage_name, uris, fastpath});
    if (block_might_exist.load(std::memory_order_acquire)) {
        might_exist_block_entered.store(true, std::memory_order_release);
        while (!release_might_exist.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    if (might_exist_throw_storages.count(storage_name) > 0) {
        throw std::runtime_error("injected MightExist exception");
    }
    if (const auto override_it = might_exist_result_overrides.find(storage_name);
        override_it != might_exist_result_overrides.end()) {
        return override_it->second;
    }
    std::vector<bool> result;
    result.reserve(uris.size());
    for (const auto &uri : uris) {
        const auto it = might_exist_by_uri.find(uri.ToUriString());
        result.push_back(it == might_exist_by_uri.end() || it->second);
    }
    return result;
}

std::shared_ptr<CacheLocation> MakeLocation(const std::string &id, CacheLocationStatus status, int64_t create_time_us) {
    auto location = std::make_shared<CacheLocation>();
    location->set_id(id);
    location->set_status(status);
    location->set_create_time(create_time_us);
    return location;
}

std::shared_ptr<CacheLocation> MakeStoredLocation(const std::string &id,
                                                  CacheLocationStatus status,
                                                  DataStorageType type,
                                                  const std::vector<std::string> &uris) {
    auto location = MakeLocation(id, status, TimestampUtil::GetCurrentTimeUs());
    location->set_type(type);
    location->set_spec_size(uris.size());
    std::vector<LocationSpec> specs;
    specs.reserve(uris.size());
    for (size_t i = 0; i < uris.size(); ++i) {
        specs.emplace_back("spec_" + std::to_string(i), uris[i]);
    }
    location->set_location_specs(std::move(specs));
    return location;
}

MaintenanceScanBatch MakeBatch(const std::string &next_cursor,
                               const KeyVector &keys,
                               CacheLocationMapVector locations,
                               std::vector<ErrorCode> results = {}) {
    MaintenanceScanBatch batch;
    batch.next_cursor = next_cursor;
    batch.keys = keys;
    batch.locations = std::move(locations);
    batch.location_results = results.empty() ? std::vector<ErrorCode>(batch.keys.size(), EC_OK) : std::move(results);
    return batch;
}

} // namespace

class CacheGarbageCollectorTest : public TESTBASE {
public:
    void SetUp() override {
        stub_.set(ADDR(RegistryManager, ListInstanceGroup), ListInstanceGroupStub);
        stub_.set(ADDR(RegistryManager, ListInstanceInfo), ListInstanceInfoStub);
        stub_.set(ADDR(MetaIndexerManager, GetMetaIndexer), GetMetaIndexerStub);
        stub_.set(ADDR(MetaIndexer, ScanLocationsForMaintenance), ScanLocationsForMaintenanceStub);
        stub_.set(ADDR(DataStorageManager, GetDataStorageBackend), GetDataStorageBackendStub);
        stub_.set(ADDR(DataStorageManager, Exist), DataStorageExistStub);
        stub_.set(ADDR(RegistryManager, GetInstanceGroupConfig), GetInstanceGroupConfigStub);
        stub_.set(ADDR(RegistryManager, GetInstanceGroupName), GetInstanceGroupNameStub);
        stub_.set(static_cast<SubmitAsyncLocation>(ADDR(SchedulePlanExecutor, SubmitAsync)), SubmitAsyncStub);
        stub_.set(static_cast<SubmitAsyncEventReport>(ADDR(SchedulePlanExecutor, SubmitAsync)),
                  SubmitAsyncEventReportStub);

        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("", metrics_registry_);
        meta_indexer_manager_ = std::make_shared<MetaIndexerManager>();
        data_storage_manager_ = std::make_shared<DataStorageManager>(metrics_registry_);
        executor_ =
            std::make_shared<SchedulePlanExecutor>(1, meta_indexer_manager_, data_storage_manager_, metrics_registry_);
        migration_manager_ = std::make_shared<MigrationManager>(
            executor_, meta_indexer_manager_, data_storage_manager_, metrics_registry_);
        dummy_indexer = std::make_shared<MetaIndexer>();

        list_groups_ec = EC_OK;
        instance_groups.clear();
        instances_by_group.clear();
        scan_responses.clear();
        scan_positions.clear();
        scan_calls.clear();
        missing_indexers.clear();
        active_instance_id.clear();
        submitted_requests.clear();
        submitted_event_report_requests.clear();
        submit_mode = SubmitMode::kReadyOk;
        pending_delete_promise.reset();
        pending_delete_promises.clear();
        list_groups_call_count = 0;
        scan_call_count.store(0, std::memory_order_relaxed);
        block_scan.store(false, std::memory_order_relaxed);
        scan_block_entered.store(false, std::memory_order_relaxed);
        release_scan.store(true, std::memory_order_relaxed);
        might_exist_calls.clear();
        might_exist_by_uri.clear();
        might_exist_result_overrides.clear();
        might_exist_throw_storages.clear();
        missing_probe_storages.clear();
        probe_storage_type_overrides.clear();
        storage_backends.clear();
        group_configs.clear();
        group_by_instance.clear();
        block_might_exist.store(false, std::memory_order_relaxed);
        might_exist_block_entered.store(false, std::memory_order_relaxed);
        release_might_exist.store(true, std::memory_order_relaxed);
        AddInstance("group_a", "instance_a");
    }

    void TearDown() override {
        release_scan.store(true, std::memory_order_release);
        release_might_exist.store(true, std::memory_order_release);
        migration_manager_.reset();
        executor_.reset();
        dummy_indexer.reset();
        pending_delete_promise.reset();
        pending_delete_promises.clear();
    }

    void AddInstance(const std::string &group_name, const std::string &instance_id) {
        auto group_it = std::find_if(instance_groups.begin(), instance_groups.end(), [&](const auto &group) {
            return group && group->name() == group_name;
        });
        if (group_it == instance_groups.end()) {
            auto group = std::make_shared<InstanceGroup>();
            group->set_name(group_name);
            instance_groups.push_back(group);
            group_configs[group_name] = group;
        }
        auto instance = std::make_shared<InstanceInfo>();
        instance->set_instance_group_name(group_name);
        instance->set_instance_id(instance_id);
        instances_by_group[group_name].first = EC_OK;
        instances_by_group[group_name].second.push_back(instance);
        group_by_instance[instance_id] = group_name;
    }

    CacheGarbageCollector::Config DefaultConfig() const {
        CacheGarbageCollector::Config config;
        config.enabled = true;
        config.scan_interval_ms = 1000;
        config.round_pause_ms = 1000;
        config.scan_batch_size = 2;
        // Most fixture cases exercise regular GC only. EventReport-specific
        // cases opt in explicitly even though the production default is on.
        config.event_report_cleanup_enabled = false;
        config.event_report_action_batch_size = 2;
        config.orphan_writing_grace_period_ms = kMinCacheGcOrphanWritingGracePeriodMs;
        return config;
    }

    std::unique_ptr<CacheGarbageCollector> MakeGc(const CacheGarbageCollector::Config &config) {
        return std::make_unique<CacheGarbageCollector>(config,
                                                       registry_manager_,
                                                       meta_indexer_manager_,
                                                       data_storage_manager_,
                                                       executor_,
                                                       metrics_registry_,
                                                       migration_manager_);
    }

    void PrepareForSingleStep(CacheGarbageCollector &gc) {
        gc.RegisterMetrics();
        gc.ResetWorkerState();
        gc.stop_requested_.store(false);
    }

    std::shared_ptr<EventReportBackend>
    AddEventReportBackend(const std::string &storage_name,
                          DataStorageType type = DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5) {
        auto spec = std::make_shared<EventReportStorageSpec>();
        spec->set_snapshot_min_interval_ms(0);
        // Keep liveness from changing fixture state during deterministic GC
        // steps. Recovery-grace tests move the deadline explicitly.
        spec->set_heartbeat_timeout_ms(60000);
        spec->set_cleanup_grace_ms(60000);
        auto backend = std::make_shared<EventReportBackend>(metrics_registry_);
        EXPECT_EQ(EC_OK, backend->Open(StorageConfig(type, storage_name, spec), "test"));
        storage_backends[storage_name] = backend;
        auto group = std::const_pointer_cast<InstanceGroup>(group_configs.at("group_a"));
        group->set_event_report_storage_candidates({storage_name});
        return backend;
    }

    bool WaitForScanCallCount(size_t expected) const {
        for (size_t i = 0; i < 100; ++i) {
            if (scan_call_count.load(std::memory_order_acquire) >= expected) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    }

    int64_t OldCreateTimeUs(const CacheGarbageCollector::Config &config, int64_t extra_us = 0) const {
        return TimestampUtil::GetCurrentTimeUs() - config.orphan_writing_grace_period_ms * 1000 - extra_us;
    }

protected:
    Stub stub_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<MetaIndexerManager> meta_indexer_manager_;
    std::shared_ptr<DataStorageManager> data_storage_manager_;
    std::shared_ptr<SchedulePlanExecutor> executor_;
    std::shared_ptr<MigrationManager> migration_manager_;
};

TEST_F(CacheGarbageCollectorTest, ConfigAndRestartableLifecycle) {
    auto disabled_config = DefaultConfig();
    disabled_config.enabled = false;
    disabled_config.event_report_cleanup_enabled = true;
    auto disabled_gc = MakeGc(disabled_config);
    EXPECT_EQ(EC_OK, disabled_gc->Start());
    EXPECT_FALSE(disabled_gc->IsRunning());
    EXPECT_FALSE(disabled_gc->IsEventReportCleanupEnabled());

    auto invalid_config = DefaultConfig();
    invalid_config.orphan_writing_grace_period_ms = kMinCacheGcOrphanWritingGracePeriodMs - 1;
    auto invalid_gc = MakeGc(invalid_config);
    EXPECT_EQ(EC_CONFIG_ERROR, invalid_gc->Validate());

    invalid_config = DefaultConfig();
    invalid_config.max_inflight_delete_requests = 0;
    invalid_gc = MakeGc(invalid_config);
    EXPECT_EQ(EC_CONFIG_ERROR, invalid_gc->Validate());

    auto zero_round_pause_config = DefaultConfig();
    zero_round_pause_config.round_pause_ms = 0;
    EXPECT_EQ(EC_OK, MakeGc(zero_round_pause_config)->Validate());

    invalid_config = DefaultConfig();
    invalid_config.round_pause_ms = -1;
    invalid_gc = MakeGc(invalid_config);
    EXPECT_EQ(EC_CONFIG_ERROR, invalid_gc->Validate());

    invalid_config = DefaultConfig();
    invalid_config.event_report_cleanup_enabled = true;
    invalid_config.event_report_action_batch_size = 0;
    invalid_gc = MakeGc(invalid_config);
    EXPECT_EQ(EC_CONFIG_ERROR, invalid_gc->Validate());

    invalid_config = DefaultConfig();
    invalid_config.event_report_cleanup_enabled = true;
    invalid_config.event_report_action_batch_size = invalid_config.scan_batch_size + 1;
    invalid_gc = MakeGc(invalid_config);
    EXPECT_EQ(EC_OK, invalid_gc->Validate());

    auto gc = MakeGc(DefaultConfig());
    ASSERT_EQ(EC_OK, gc->Start());
    EXPECT_TRUE(gc->IsRunning());
    EXPECT_EQ(EC_OK, gc->Start());
    gc->RequestStop();
    gc->RequestStop();
    gc->Join();
    gc->Join();
    EXPECT_FALSE(gc->IsRunning());

    ASSERT_EQ(EC_OK, gc->Start());
    EXPECT_TRUE(gc->IsRunning());
    gc->Stop();
    EXPECT_FALSE(gc->IsRunning());
}

TEST_F(CacheGarbageCollectorTest, StartRestoresFullEventReportRecoveryGraceAfterLeaderRecovery) {
    auto spec = std::make_shared<EventReportStorageSpec>();
    spec->set_heartbeat_timeout_ms(60000);
    spec->set_cleanup_grace_ms(60000);
    spec->set_liveness_check_interval_ms(60000);
    const std::string storage_name = "event_report_recovery_grace";
    RequestContext context("register_event_report_recovery_grace");
    ASSERT_EQ(EC_OK,
              data_storage_manager_->RegisterStorage(
                  &context,
                  storage_name,
                  StorageConfig(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5, storage_name, spec)));

    std::shared_ptr<EventReportBackend> backend;
    for (const auto &storage : data_storage_manager_->GetAvailableStorages()) {
        if (storage && storage->GetStorageConfig().global_unique_name() == storage_name) {
            backend = std::dynamic_pointer_cast<EventReportBackend>(storage);
            break;
        }
    }
    ASSERT_NE(nullptr, backend);
    // Simulate a Backend opened before a long leader recovery. GC Start must
    // grant Reporters a fresh registration/heartbeat window.
    backend->maintenance_recovery_deadline_ms_.store(0, std::memory_order_release);
    EXPECT_EQ(0, backend->GetMaintenanceRecoveryGraceRemainingMs());

    auto config = DefaultConfig();
    config.event_report_cleanup_enabled = true;
    config.scan_interval_ms = 60000;
    auto gc = MakeGc(config);
    ASSERT_EQ(EC_OK, gc->Start());
    EXPECT_GT(backend->GetMaintenanceRecoveryGraceRemainingMs(), 0);
    gc->Stop();

    ASSERT_EQ(EC_OK, data_storage_manager_->UnRegisterStorage(storage_name));
}

TEST_F(CacheGarbageCollectorTest, EventReportLookupSkipsMissingCandidateAndRejectsAmbiguousOrUnavailableOwner) {
    auto config = DefaultConfig();
    config.event_report_cleanup_enabled = true;
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    auto first = AddEventReportBackend("event_report_first");
    auto second = AddEventReportBackend("event_report_second", DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2);
    auto duplicate_second =
        AddEventReportBackend("event_report_second_duplicate", DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2);
    auto group = std::const_pointer_cast<InstanceGroup>(group_configs.at("group_a"));
    group->set_event_report_storage_candidates({"event_report_second", "event_report_second_duplicate"});
    auto route = gc->LookupEventReportBackend("instance_a", second->GetStorageType());
    EXPECT_EQ(CacheGarbageCollector::EventReportBackendRouteStatus::kAmbiguous, route.status);

    group->set_event_report_storage_candidates({"missing_owner", "event_report_second"});
    route = gc->LookupEventReportBackend("instance_a", second->GetStorageType());
    ASSERT_EQ(CacheGarbageCollector::EventReportBackendRouteStatus::kResolved, route.status);
    EXPECT_EQ(second.get(), route.backend.get());

    group->set_event_report_storage_candidates({"event_report_first"});
    first->SetAvailable(false);
    route = gc->LookupEventReportBackend("instance_a", first->GetStorageType());
    EXPECT_EQ(CacheGarbageCollector::EventReportBackendRouteStatus::kUnavailable, route.status);
    first->Close();
    second->Close();
    duplicate_second->Close();
}

TEST_F(CacheGarbageCollectorTest, SharedScanDeletesOnlyStaleSnapshotAndMalformedIsUnknown) {
    auto config = DefaultConfig();
    config.event_report_cleanup_enabled = true;
    config.scan_batch_size = 8;
    config.event_report_action_batch_size = 4;
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    auto backend = AddEventReportBackend("event_report_l1p5");
    const ReporterSnapshotKey reporter{"instance_a", "10.0.0.1:9000"};
    ASSERT_EQ(EC_OK, backend->RegisterNode(reporter.instance_id, reporter.host_ip_port, {"hbm"}));

    uint64_t retry_after_ms = 0;
    std::string old_version;
    ASSERT_EQ(EC_OK, backend->BeginSnapshot(reporter, old_version, retry_after_ms));
    ASSERT_TRUE(backend->CommitSnapshotVersion(reporter, old_version));
    std::string committed_version;
    ASSERT_EQ(EC_OK, backend->BeginSnapshot(reporter, committed_version, retry_after_ms));
    ASSERT_TRUE(backend->CommitSnapshotVersion(reporter, committed_version));

    std::string old_uri;
    std::string committed_uri;
    ASSERT_TRUE(
        SnapshotUriUtils::AddSnapshotVersionToUri("event_report://event_report_l1p5/old", old_version, old_uri));
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://event_report_l1p5/current", committed_version, committed_uri));
    const std::string malformed_uri = "event_report://event_report_l1p5/malformed?s_version=first&s_version=second";
    const std::string location_id = backend->BuildLocationId("hbm", reporter.host_ip_port);

    CacheLocationMap old_locations;
    old_locations.emplace(location_id,
                          MakeStoredLocation(location_id, CLS_SERVING, backend->GetStorageType(), {old_uri}));
    CacheLocationMap current_locations;
    current_locations.emplace(location_id,
                              MakeStoredLocation(location_id, CLS_SERVING, backend->GetStorageType(), {committed_uri}));
    CacheLocationMap malformed_locations;
    malformed_locations.emplace(
        location_id, MakeStoredLocation(location_id, CLS_SERVING, backend->GetStorageType(), {malformed_uri}));
    scan_responses[reporter.instance_id] = {
        {EC_OK, MakeBatch(SCAN_BASE_CURSOR, {101, 102, 103}, {old_locations, current_locations, malformed_locations})}};

    gc->RunOneTick();

    ASSERT_EQ(1u, submitted_event_report_requests.size());
    EXPECT_EQ((KeyVector{101}), submitted_event_report_requests.front().block_keys);
    ASSERT_EQ(1u, submitted_event_report_requests.front().targets.size());
    ASSERT_EQ(1u, submitted_event_report_requests.front().targets.front().size());
    EXPECT_EQ(location_id, submitted_event_report_requests.front().targets.front().front().location_id);
    EXPECT_TRUE(submitted_requests.empty());
    EXPECT_EQ(
        1, metrics_registry_->GetCounter("cache_gc.event_report_probe_unknown_count", {{"cause", "malformed"}}).Get());
    backend->Close();
}

TEST_F(CacheGarbageCollectorTest, DownHostsShareOneRegularScanAndUseKeyBoundedActions) {
    auto config = DefaultConfig();
    config.event_report_cleanup_enabled = true;
    config.scan_batch_size = 8;
    config.event_report_action_batch_size = 4;
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    auto backend = AddEventReportBackend("event_report_l1p5");
    CacheLocationMap first_locations;
    CacheLocationMap second_locations;
    const std::vector<std::string> hosts{"10.0.0.1:9000", "10.0.0.2:9000"};
    for (size_t i = 0; i < hosts.size(); ++i) {
        ASSERT_EQ(EC_OK, backend->RegisterNode("instance_a", hosts[i], {"hbm"}));
        uint64_t generation = 0;
        ASSERT_EQ(EC_OK, backend->UnregisterNodeForHostDown("instance_a", hosts[i], generation));
        const std::string location_id = backend->BuildLocationId("hbm", hosts[i]);
        auto location = MakeStoredLocation(location_id,
                                           CLS_SERVING,
                                           backend->GetStorageType(),
                                           {"event_report://event_report_l1p5/object" + std::to_string(i)});
        (i == 0 ? first_locations : second_locations).emplace(location_id, std::move(location));
    }
    scan_responses["instance_a"] = {
        {EC_OK, MakeBatch(SCAN_BASE_CURSOR, {101, 202}, {first_locations, second_locations})}};

    gc->RunOneTick();

    ASSERT_EQ(1u, scan_calls.size());
    ASSERT_EQ(1u, submitted_event_report_requests.size());
    EXPECT_EQ((KeyVector{101, 202}), submitted_event_report_requests.front().block_keys);
    EXPECT_TRUE(submitted_requests.empty());
    backend->Close();
}

TEST_F(CacheGarbageCollectorTest, UnifiedBudgetPrioritizesBlockingGarbageOverEventReport) {
    auto config = DefaultConfig();
    config.event_report_cleanup_enabled = true;
    config.scan_batch_size = 2;
    config.event_report_action_batch_size = 1;
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    auto backend = AddEventReportBackend("event_report_l1p5");
    const ReporterSnapshotKey reporter{"instance_a", "10.0.0.1:9000"};
    ASSERT_EQ(EC_OK, backend->RegisterNode(reporter.instance_id, reporter.host_ip_port, {"hbm"}));
    uint64_t retry_after_ms = 0;
    std::string old_version;
    ASSERT_EQ(EC_OK, backend->BeginSnapshot(reporter, old_version, retry_after_ms));
    ASSERT_TRUE(backend->CommitSnapshotVersion(reporter, old_version));
    std::string committed_version;
    ASSERT_EQ(EC_OK, backend->BeginSnapshot(reporter, committed_version, retry_after_ms));
    ASSERT_TRUE(backend->CommitSnapshotVersion(reporter, committed_version));
    std::string old_uri;
    ASSERT_TRUE(
        SnapshotUriUtils::AddSnapshotVersionToUri("event_report://event_report_l1p5/old", old_version, old_uri));

    const std::string event_location_id = backend->BuildLocationId("hbm", reporter.host_ip_port);
    CacheLocationMap event_locations;
    event_locations.emplace(event_location_id,
                            MakeStoredLocation(event_location_id, CLS_SERVING, backend->GetStorageType(), {old_uri}));

    const std::string missing_uri = "dummy://dummy/missing?size=1";
    might_exist_by_uri[missing_uri] = false;
    CacheLocationMap missing_locations;
    missing_locations.emplace(
        "missing", MakeStoredLocation("missing", CLS_SERVING, DataStorageType::DATA_STORAGE_TYPE_DUMMY, {missing_uri}));

    CacheLocationMap orphan_locations;
    orphan_locations.emplace("orphan", MakeLocation("orphan", CLS_WRITING, OldCreateTimeUs(config, 1)));
    scan_responses["instance_a"] = {
        {EC_OK, MakeBatch(SCAN_BASE_CURSOR, {1, 100, 200}, {event_locations, missing_locations, orphan_locations})}};

    gc->RunOneTick();

    ASSERT_EQ(1u, submitted_requests.size());
    EXPECT_EQ((KeyVector{100, 200}), submitted_requests.front().block_keys);
    EXPECT_TRUE(submitted_event_report_requests.empty());
    EXPECT_EQ(1,
              metrics_registry_
                  ->GetCounter("cache_gc.candidate_dropped_count",
                               {{"reason", "event_report_stale_snapshot"}, {"cause", "total_budget"}})
                  .Get());
    backend->Close();
}

TEST_F(CacheGarbageCollectorTest, FullPhysicalInflightWindowPausesSharedEventReportScan) {
    auto config = DefaultConfig();
    config.event_report_cleanup_enabled = true;
    config.max_inflight_delete_requests = 1;
    config.scan_batch_size = 4;
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    auto backend = AddEventReportBackend("event_report_l1p5");
    const std::string host = "10.0.0.1:9000";
    ASSERT_EQ(EC_OK, backend->RegisterNode("instance_a", host, {"hbm"}));
    uint64_t generation = 0;
    ASSERT_EQ(EC_OK, backend->UnregisterNodeForHostDown("instance_a", host, generation));
    const std::string event_location_id = backend->BuildLocationId("hbm", host);
    CacheLocationMap event_locations;
    event_locations.emplace(
        event_location_id,
        MakeStoredLocation(
            event_location_id, CLS_SERVING, backend->GetStorageType(), {"event_report://event_report_l1p5/object"}));

    CacheLocationMap orphan_locations;
    orphan_locations.emplace("orphan", MakeLocation("orphan", CLS_WRITING, OldCreateTimeUs(config, 1)));
    scan_responses["instance_a"] = {
        {EC_OK, MakeBatch("next", {100}, {orphan_locations})},
        {EC_OK, MakeBatch(SCAN_BASE_CURSOR, {200}, {event_locations})},
    };

    submit_mode = SubmitMode::kPending;
    gc->RunOneTick();
    ASSERT_EQ(1u, gc->inflight_deletes_.size());
    ASSERT_EQ(1u, scan_calls.size());

    gc->RunOneTick();
    EXPECT_EQ(1u, scan_calls.size());
    EXPECT_TRUE(submitted_event_report_requests.empty());

    ASSERT_TRUE(pending_delete_promise);
    pending_delete_promise->set_value({EC_OK, ""});
    gc->RunOneTick();
    EXPECT_EQ(2u, scan_calls.size());
    ASSERT_EQ(1u, submitted_event_report_requests.size());
    EXPECT_EQ((KeyVector{200}), submitted_event_report_requests.front().block_keys);
    backend->Close();
}

TEST_F(CacheGarbageCollectorTest, EventReportActionBudgetCountsKeysAndKeepsLocationsForSelectedKey) {
    auto config = DefaultConfig();
    config.event_report_cleanup_enabled = true;
    config.scan_batch_size = 8;
    config.event_report_action_batch_size = 1;
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    auto backend = AddEventReportBackend("event_report_l1p5");
    const std::vector<std::string> hosts{"10.0.0.1:9000", "10.0.0.2:9000", "10.0.0.3:9000"};
    CacheLocationMap first_key_locations;
    CacheLocationMap second_key_locations;
    for (size_t i = 0; i < hosts.size(); ++i) {
        ASSERT_EQ(EC_OK, backend->RegisterNode("instance_a", hosts[i], {"hbm"}));
        uint64_t generation = 0;
        ASSERT_EQ(EC_OK, backend->UnregisterNodeForHostDown("instance_a", hosts[i], generation));
        const std::string location_id = backend->BuildLocationId("hbm", hosts[i]);
        auto location = MakeStoredLocation(location_id,
                                           CLS_SERVING,
                                           backend->GetStorageType(),
                                           {"event_report://event_report_l1p5/object" + std::to_string(i)});
        (i < 2 ? first_key_locations : second_key_locations).emplace(location_id, std::move(location));
    }
    scan_responses["instance_a"] = {
        {EC_OK, MakeBatch(SCAN_BASE_CURSOR, {100, 200}, {first_key_locations, second_key_locations})}};

    gc->RunOneTick();

    ASSERT_EQ(1u, submitted_event_report_requests.size());
    EXPECT_EQ((KeyVector{100}), submitted_event_report_requests.front().block_keys);
    ASSERT_EQ(1u, submitted_event_report_requests.front().targets.size());
    EXPECT_EQ(2u, submitted_event_report_requests.front().targets.front().size());
    EXPECT_EQ(1,
              metrics_registry_
                  ->GetCounter("cache_gc.candidate_dropped_count",
                               {{"reason", "event_report_down_host"}, {"cause", "event_report_budget"}})
                  .Get());
    backend->Close();
}

TEST_F(CacheGarbageCollectorTest, EventReportActionBudgetIsIndependentAndObservable) {
    auto config = DefaultConfig();
    config.event_report_cleanup_enabled = true;
    config.scan_batch_size = 4;
    config.event_report_action_batch_size = 1;
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    auto backend = AddEventReportBackend("event_report_l1p5");
    const ReporterSnapshotKey reporter{"instance_a", "10.0.0.1:9000"};
    ASSERT_EQ(EC_OK, backend->RegisterNode(reporter.instance_id, reporter.host_ip_port, {"hbm"}));
    uint64_t retry_after_ms = 0;
    std::string old_version;
    ASSERT_EQ(EC_OK, backend->BeginSnapshot(reporter, old_version, retry_after_ms));
    ASSERT_TRUE(backend->CommitSnapshotVersion(reporter, old_version));
    std::string committed_version;
    ASSERT_EQ(EC_OK, backend->BeginSnapshot(reporter, committed_version, retry_after_ms));
    ASSERT_TRUE(backend->CommitSnapshotVersion(reporter, committed_version));
    std::string old_uri;
    ASSERT_TRUE(
        SnapshotUriUtils::AddSnapshotVersionToUri("event_report://event_report_l1p5/old", old_version, old_uri));
    const std::string location_id = backend->BuildLocationId("hbm", reporter.host_ip_port);
    CacheLocationMap first;
    CacheLocationMap second;
    first.emplace(location_id, MakeStoredLocation(location_id, CLS_SERVING, backend->GetStorageType(), {old_uri}));
    second.emplace(location_id, MakeStoredLocation(location_id, CLS_SERVING, backend->GetStorageType(), {old_uri}));
    scan_responses["instance_a"] = {{EC_OK, MakeBatch(SCAN_BASE_CURSOR, {101, 202}, {first, second})}};

    gc->RunOneTick();

    ASSERT_EQ(1u, submitted_event_report_requests.size());
    EXPECT_EQ((KeyVector{101}), submitted_event_report_requests.front().block_keys);
    EXPECT_EQ(1,
              metrics_registry_
                  ->GetCounter("cache_gc.candidate_dropped_count",
                               {{"reason", "event_report_stale_snapshot"}, {"cause", "event_report_budget"}})
                  .Get());
    backend->Close();
}

TEST_F(CacheGarbageCollectorTest, RejectedEventReportSubmissionDoesNotCreatePendingOrInflight) {
    auto config = DefaultConfig();
    config.event_report_cleanup_enabled = true;
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    auto backend = AddEventReportBackend("event_report_l1p5");
    const std::string host = "10.0.0.1:9000";
    ASSERT_EQ(EC_OK, backend->RegisterNode("instance_a", host, {"hbm"}));
    uint64_t generation = 0;
    ASSERT_EQ(EC_OK, backend->UnregisterNodeForHostDown("instance_a", host, generation));
    const std::string location_id = backend->BuildLocationId("hbm", host);
    CacheLocationMap locations;
    locations.emplace(
        location_id,
        MakeStoredLocation(
            location_id, CLS_SERVING, backend->GetStorageType(), {"event_report://event_report_l1p5/object"}));
    scan_responses["instance_a"] = {{EC_OK, MakeBatch(SCAN_BASE_CURSOR, {101}, {locations})}};

    submit_mode = SubmitMode::kRejected;
    gc->RunOneTick();

    ASSERT_EQ(1u, submitted_event_report_requests.size());
    EXPECT_TRUE(gc->inflight_deletes_.empty());
    EXPECT_TRUE(gc->pending_locations_.empty());
    EXPECT_EQ(0, metrics_registry_->GetCounter("cache_gc.operation_error_count", {{"stage", "submit_rejected"}}).Get());
    backend->Close();
}

TEST_F(CacheGarbageCollectorTest, PhysicalAndEventReportActionsShareGcInflightWindow) {
    auto config = DefaultConfig();
    config.event_report_cleanup_enabled = true;
    config.scan_batch_size = 4;
    config.max_inflight_delete_requests = 2;
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    auto backend = AddEventReportBackend("event_report_l1p5");
    const std::string host = "10.0.0.1:9000";
    ASSERT_EQ(EC_OK, backend->RegisterNode("instance_a", host, {"hbm"}));
    uint64_t generation = 0;
    ASSERT_EQ(EC_OK, backend->UnregisterNodeForHostDown("instance_a", host, generation));
    const std::string event_location_id = backend->BuildLocationId("hbm", host);
    CacheLocationMap locations;
    locations.emplace(
        event_location_id,
        MakeStoredLocation(
            event_location_id, CLS_SERVING, backend->GetStorageType(), {"event_report://event_report_l1p5/object"}));
    locations.emplace("orphan", MakeLocation("orphan", CLS_WRITING, OldCreateTimeUs(config, 1)));
    scan_responses["instance_a"] = {{EC_OK, MakeBatch(SCAN_BASE_CURSOR, {101}, {locations})}};

    submit_mode = SubmitMode::kPending;
    gc->RunOneTick();

    ASSERT_EQ(1u, submitted_requests.size());
    ASSERT_EQ(1u, submitted_event_report_requests.size());
    EXPECT_EQ(2u, gc->inflight_deletes_.size());
    EXPECT_EQ(2u, gc->pending_locations_.size());
    backend->Close();
}

TEST_F(CacheGarbageCollectorTest, BackendRecoveryGraceDoesNotPauseOrdinaryGcCandidates) {
    auto config = DefaultConfig();
    config.event_report_cleanup_enabled = true;
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    auto backend = AddEventReportBackend("event_report_l1p5");
    const std::string host = "10.0.0.1:9000";
    const std::string location_id = backend->BuildLocationId("hbm", host);
    CacheLocationMap locations;
    locations.emplace(
        location_id,
        MakeStoredLocation(
            location_id, CLS_SERVING, backend->GetStorageType(), {"event_report://event_report_l1p5/object"}));
    locations.emplace("orphan", MakeLocation("orphan", CLS_WRITING, OldCreateTimeUs(config, 1)));
    scan_responses["instance_a"] = {{EC_OK, MakeBatch(SCAN_BASE_CURSOR, {101}, {locations})}};

    gc->RunOneTick();

    ASSERT_EQ(1u, scan_calls.size());
    ASSERT_EQ(1u, submitted_requests.size());
    EXPECT_EQ((std::vector<std::vector<std::string>>{{"orphan"}}), submitted_requests.front().location_ids);
    EXPECT_TRUE(submitted_event_report_requests.empty());
    EXPECT_EQ(1,
              metrics_registry_
                  ->GetCounter("cache_gc.event_report_probe_unknown_count", {{"cause", "recovery_grace"}})
                  .Get());
    backend->Close();
}

TEST_F(CacheGarbageCollectorTest, RestartBeginsScanningFromBaseCursor) {
    auto config = DefaultConfig();
    config.scan_interval_ms = 60000;
    scan_responses["instance_a"] = {
        {EC_OK, MakeBatch("cursor_after_first_start", {}, {})},
        {EC_OK, MakeBatch(SCAN_BASE_CURSOR, {}, {})},
    };

    auto gc = MakeGc(config);
    ASSERT_EQ(EC_OK, gc->Start());
    ASSERT_TRUE(WaitForScanCallCount(1));
    gc->Stop();

    ASSERT_EQ(EC_OK, gc->Start());
    ASSERT_TRUE(WaitForScanCallCount(2));
    gc->Stop();

    ASSERT_EQ(2, scan_calls.size());
    EXPECT_EQ(SCAN_BASE_CURSOR, scan_calls[0].second);
    EXPECT_EQ(SCAN_BASE_CURSOR, scan_calls[1].second);
}

TEST_F(CacheGarbageCollectorTest, FixedPredicateIsFailClosedAndRequestIsBounded) {
    auto config = DefaultConfig();
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    const int64_t now_us = TimestampUtil::GetCurrentTimeUs();
    const int64_t grace_us = config.orphan_writing_grace_period_ms * 1000;
    CacheLocationMap first_locations;
    first_locations["old"] = MakeLocation("old", CLS_WRITING, now_us - grace_us);
    first_locations["young"] = MakeLocation("young", CLS_WRITING, now_us - grace_us + 1);
    first_locations["serving"] = MakeLocation("serving", CLS_SERVING, now_us - grace_us - 1);
    first_locations["future"] = MakeLocation("future", CLS_WRITING, now_us + 1);
    first_locations["zero"] = MakeLocation("zero", CLS_WRITING, 0);
    first_locations["map_id"] = MakeLocation("different_id", CLS_WRITING, now_us - grace_us - 1);
    CacheLocationMap second_locations;
    second_locations["another"] = MakeLocation("another", CLS_WRITING, now_us - grace_us - 1);
    CacheLocationMap third_locations;
    third_locations["third"] = MakeLocation("third", CLS_WRITING, now_us - grace_us - 1);
    CacheLocationMap duplicate_locations;
    duplicate_locations["old"] = MakeLocation("old", CLS_WRITING, now_us - grace_us - 1);

    const auto batch = MakeBatch(
        SCAN_BASE_CURSOR, {10, 20, 30, 10}, {first_locations, second_locations, third_locations, duplicate_locations});
    const CacheLocationDelRequest request = gc->BuildDeleteActions("instance_a", batch, now_us).executor_request;
    ASSERT_EQ(2, request.block_keys.size());
    EXPECT_EQ((KeyVector{10, 20}), request.block_keys);
    EXPECT_EQ((std::vector<std::vector<std::string>>{{"old"}, {"another"}}), request.location_ids);
    EXPECT_EQ((std::vector<std::vector<std::string>>{{first_locations.at("old")->ToJsonString()},
                                                     {second_locations.at("another")->ToJsonString()}}),
              request.expected_location_values);
    EXPECT_TRUE(request.authoritative_read);
    EXPECT_EQ(3, gc->get_cache_gc_candidate_count_metrics());

    MaintenanceScanBatch broken_batch = batch;
    broken_batch.location_results.pop_back();
    EXPECT_TRUE(gc->BuildDeleteActions("instance_a", broken_batch, now_us).executor_request.block_keys.empty());
    EXPECT_TRUE(gc->BuildDeleteActions("", batch, now_us).executor_request.block_keys.empty());
}

TEST_F(CacheGarbageCollectorTest, ServingMissingSpecIsBatchedAndSubmittedWithExactSnapshot) {
    auto config = DefaultConfig();
    config.scan_batch_size = 10;
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    const std::string missing_a_uri = "dummy://storage_a/missing_a?size=1";
    const std::string existing_a_uri = "dummy://storage_a/existing_a?size=1";
    const std::string existing_b_uri = "dummy://storage_a/existing_b?size=1";
    const std::string missing_b_uri = "dummy://storage_b/missing_b?size=1";
    might_exist_by_uri[missing_a_uri] = false;
    might_exist_by_uri[missing_b_uri] = false;

    CacheLocationMap first_locations;
    first_locations["old_writing"] = MakeLocation("old_writing", CLS_WRITING, OldCreateTimeUs(config, /*extra_us=*/1));
    first_locations["serving_missing"] = MakeStoredLocation(
        "serving_missing", CLS_SERVING, DataStorageType::DATA_STORAGE_TYPE_DUMMY, {missing_a_uri, existing_a_uri});
    first_locations["serving_existing"] =
        MakeStoredLocation("serving_existing", CLS_SERVING, DataStorageType::DATA_STORAGE_TYPE_DUMMY, {existing_b_uri});
    first_locations["event_report_missing"] = MakeStoredLocation("event_report_missing",
                                                                 CLS_SERVING,
                                                                 DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5,
                                                                 {"event_report://event_storage/missing"});
    first_locations["unknown_type"] = MakeStoredLocation(
        "unknown_type", CLS_SERVING, DataStorageType::DATA_STORAGE_TYPE_UNKNOWN, {"dummy://unknown_storage/missing"});
    first_locations["malformed_uri"] =
        MakeStoredLocation("malformed_uri", CLS_SERVING, DataStorageType::DATA_STORAGE_TYPE_DUMMY, {"not-a-uri"});

    CacheLocationMap second_locations;
    second_locations["second_missing"] =
        MakeStoredLocation("second_missing", CLS_SERVING, DataStorageType::DATA_STORAGE_TYPE_DUMMY, {missing_b_uri});

    const auto batch = MakeBatch(SCAN_BASE_CURSOR, {10, 20}, {first_locations, second_locations});
    const CacheLocationDelRequest request =
        gc->BuildDeleteActions("instance_a", batch, TimestampUtil::GetCurrentTimeUs()).executor_request;

    EXPECT_EQ((KeyVector{10, 20}), request.block_keys);
    EXPECT_EQ((std::vector<std::vector<std::string>>{{"old_writing", "serving_missing"}, {"second_missing"}}),
              request.location_ids);
    EXPECT_EQ((std::vector<std::vector<std::string>>{{first_locations.at("old_writing")->ToJsonString(),
                                                      first_locations.at("serving_missing")->ToJsonString()},
                                                     {second_locations.at("second_missing")->ToJsonString()}}),
              request.expected_location_values);

    ASSERT_EQ(2, might_exist_calls.size());
    const auto storage_a_call =
        std::find_if(might_exist_calls.begin(), might_exist_calls.end(), [](const MightExistCall &call) {
            return call.storage_name == "storage_a";
        });
    const auto storage_b_call =
        std::find_if(might_exist_calls.begin(), might_exist_calls.end(), [](const MightExistCall &call) {
            return call.storage_name == "storage_b";
        });
    ASSERT_NE(might_exist_calls.end(), storage_a_call);
    ASSERT_NE(might_exist_calls.end(), storage_b_call);
    EXPECT_TRUE(storage_a_call->fastpath);
    EXPECT_TRUE(storage_b_call->fastpath);
    EXPECT_EQ(3, storage_a_call->uris.size());
    EXPECT_EQ(1, storage_b_call->uris.size());

    EXPECT_EQ(1, gc->get_cache_gc_candidate_count_metrics());
    EXPECT_EQ(
        2, metrics_registry_->GetCounter("cache_gc.candidate_count", MetricsTags{{"reason", "storage_missing"}}).Get());
}

TEST_F(CacheGarbageCollectorTest, ServingProbeErrorsAreUnknownButOtherDefinitiveMissingStillWins) {
    auto config = DefaultConfig();
    config.scan_batch_size = 10;
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    might_exist_result_overrides["shape_storage"] = {};
    might_exist_throw_storages.insert("throw_storage");
    might_exist_by_uri["dummy://missing_storage/missing?size=1"] = false;

    CacheLocationMap locations;
    locations["shape_only"] = MakeStoredLocation(
        "shape_only", CLS_SERVING, DataStorageType::DATA_STORAGE_TYPE_DUMMY, {"dummy://shape_storage/value?size=1"});
    locations["throw_only"] = MakeStoredLocation(
        "throw_only", CLS_SERVING, DataStorageType::DATA_STORAGE_TYPE_DUMMY, {"dummy://throw_storage/value?size=1"});
    locations["missing_and_unknown"] =
        MakeStoredLocation("missing_and_unknown",
                           CLS_SERVING,
                           DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                           {"dummy://shape_storage/unknown?size=1", "dummy://missing_storage/missing?size=1"});

    const CacheLocationDelRequest request = gc->BuildDeleteActions("instance_a",
                                                                   MakeBatch(SCAN_BASE_CURSOR, {10}, {locations}),
                                                                   TimestampUtil::GetCurrentTimeUs())
                                                .executor_request;

    EXPECT_EQ((KeyVector{10}), request.block_keys);
    EXPECT_EQ((std::vector<std::vector<std::string>>{{"missing_and_unknown"}}), request.location_ids);
    EXPECT_EQ((std::vector<std::vector<std::string>>{{locations.at("missing_and_unknown")->ToJsonString()}}),
              request.expected_location_values);
    EXPECT_EQ(
        1,
        metrics_registry_->GetCounter("cache_gc.operation_error_count", MetricsTags{{"stage", "might_exist_shape"}})
            .Get());
    EXPECT_EQ(
        1,
        metrics_registry_->GetCounter("cache_gc.operation_error_count", MetricsTags{{"stage", "might_exist_exception"}})
            .Get());
}

TEST_F(CacheGarbageCollectorTest, ServingProbeBatchesBoundEachMightExistCall) {
    auto config = DefaultConfig();
    config.scan_batch_size = 1;
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    std::vector<std::string> uris;
    uris.reserve(513);
    for (size_t i = 0; i < 513; ++i) {
        uris.push_back("dummy://storage_a/object_" + std::to_string(i) + "?size=1");
    }
    CacheLocationMap locations;
    locations["serving"] = MakeStoredLocation("serving", CLS_SERVING, DataStorageType::DATA_STORAGE_TYPE_DUMMY, uris);

    const CacheLocationDelRequest request = gc->BuildDeleteActions("instance_a",
                                                                   MakeBatch(SCAN_BASE_CURSOR, {10}, {locations}),
                                                                   TimestampUtil::GetCurrentTimeUs())
                                                .executor_request;

    EXPECT_TRUE(request.block_keys.empty());
    ASSERT_EQ(2, might_exist_calls.size());
    EXPECT_EQ(512, might_exist_calls[0].uris.size());
    EXPECT_EQ(1, might_exist_calls[1].uris.size());
}

TEST_F(CacheGarbageCollectorTest, StopAfterActiveProbeSkipsRemainingMightExistBatches) {
    auto config = DefaultConfig();
    config.scan_interval_ms = 60000;

    std::vector<std::string> uris;
    uris.reserve(513);
    for (size_t i = 0; i < 513; ++i) {
        uris.push_back("dummy://storage_a/object_" + std::to_string(i) + "?size=1");
    }
    CacheLocationMap locations;
    locations["serving"] = MakeStoredLocation("serving", CLS_SERVING, DataStorageType::DATA_STORAGE_TYPE_DUMMY, uris);
    scan_responses["instance_a"] = {{EC_OK, MakeBatch(SCAN_BASE_CURSOR, {10}, {locations})}};
    block_might_exist.store(true, std::memory_order_release);
    release_might_exist.store(false, std::memory_order_release);

    auto gc = MakeGc(config);
    ASSERT_EQ(EC_OK, gc->Start());
    for (size_t i = 0; i < 100 && !might_exist_block_entered.load(std::memory_order_acquire); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!might_exist_block_entered.load(std::memory_order_acquire)) {
        release_might_exist.store(true, std::memory_order_release);
        gc->Stop();
        FAIL() << "MightExist probe did not start";
    }

    gc->RequestStop();
    release_might_exist.store(true, std::memory_order_release);
    gc->Join();

    ASSERT_EQ(1, might_exist_calls.size());
    EXPECT_EQ(512, might_exist_calls.front().uris.size());
    EXPECT_TRUE(submitted_requests.empty());
}

TEST_F(CacheGarbageCollectorTest, MissingOrMismatchedStorageIsUnknownAndClassifiedSeparately) {
    auto config = DefaultConfig();
    config.scan_batch_size = 10;
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    missing_probe_storages.insert("unregistered_storage");
    probe_storage_type_overrides["wrong_type_storage"] = DataStorageType::DATA_STORAGE_TYPE_NFS;
    const std::string missing_uri = "dummy://healthy_storage/missing?size=1";
    might_exist_by_uri[missing_uri] = false;

    CacheLocationMap locations;
    locations["unregistered"] = MakeStoredLocation("unregistered",
                                                   CLS_SERVING,
                                                   DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                                   {"dummy://unregistered_storage/value?size=1"});
    locations["wrong_type"] = MakeStoredLocation("wrong_type",
                                                 CLS_SERVING,
                                                 DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                                 {"dummy://wrong_type_storage/value?size=1"});
    locations["definitive_missing"] =
        MakeStoredLocation("definitive_missing", CLS_SERVING, DataStorageType::DATA_STORAGE_TYPE_DUMMY, {missing_uri});

    const CacheLocationDelRequest request = gc->BuildDeleteActions("instance_a",
                                                                   MakeBatch(SCAN_BASE_CURSOR, {10}, {locations}),
                                                                   TimestampUtil::GetCurrentTimeUs())
                                                .executor_request;

    EXPECT_EQ((KeyVector{10}), request.block_keys);
    EXPECT_EQ((std::vector<std::vector<std::string>>{{"definitive_missing"}}), request.location_ids);
    ASSERT_EQ(1, might_exist_calls.size());
    EXPECT_EQ("healthy_storage", might_exist_calls.front().storage_name);
    EXPECT_EQ(
        1,
        metrics_registry_
            ->GetCounter("cache_gc.operation_error_count", MetricsTags{{"stage", "might_exist_storage_not_found"}})
            .Get());
    EXPECT_EQ(
        1,
        metrics_registry_
            ->GetCounter("cache_gc.operation_error_count", MetricsTags{{"stage", "might_exist_storage_type_mismatch"}})
            .Get());
}

TEST_F(CacheGarbageCollectorTest, ActiveMigrationCopyTargetIsNotCollected) {
    auto config = DefaultConfig();
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    MigrationManager::CopyTaskContext task;
    task.instance_group_name = "group_a";
    task.instance_id = "instance_a";
    task.block_key = 10;
    task.dst_location_id = "migration_target";
    task.state = MigrationManager::CopyTaskState::kRunning;
    migration_manager_->active_tasks_by_instance_["instance_a"].emplace(task.block_key, task);

    const int64_t now_us = TimestampUtil::GetCurrentTimeUs();
    CacheLocationMap migration_locations;
    migration_locations["migration_target"] =
        MakeLocation("migration_target", CLS_WRITING, now_us - config.orphan_writing_grace_period_ms * 1000 - 1);
    CacheLocationMap orphan_locations;
    orphan_locations["migration_target"] =
        MakeLocation("migration_target", CLS_WRITING, now_us - config.orphan_writing_grace_period_ms * 1000 - 1);

    const auto batch = MakeBatch(SCAN_BASE_CURSOR, {10, 20}, {migration_locations, orphan_locations});
    const CacheLocationDelRequest same_instance_request =
        gc->BuildDeleteActions("instance_a", batch, now_us).executor_request;
    const CacheLocationDelRequest other_instance_request =
        gc->BuildDeleteActions("instance_b", MakeBatch(SCAN_BASE_CURSOR, {10}, {migration_locations}), now_us)
            .executor_request;

    EXPECT_EQ((KeyVector{20}), same_instance_request.block_keys);
    EXPECT_EQ((std::vector<std::vector<std::string>>{{"migration_target"}}), same_instance_request.location_ids);
    EXPECT_EQ((KeyVector{10}), other_instance_request.block_keys);
    EXPECT_EQ((std::vector<std::vector<std::string>>{{"migration_target"}}), other_instance_request.location_ids);
    EXPECT_EQ(2, gc->get_cache_gc_candidate_count_metrics());
}

TEST_F(CacheGarbageCollectorTest, PersistentCopyGuardFencesTargetAndExactSourceWithoutActiveTask) {
    auto config = DefaultConfig();
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    const int64_t now_us = TimestampUtil::GetCurrentTimeUs();
    const int64_t source_create_time = now_us - 100;
    const std::string source_uri = "dummy://pace_dram/source?size=1";
    might_exist_by_uri[source_uri] = false;

    auto source = MakeStoredLocation(
        "source_location", CLS_SERVING, DataStorageType::DATA_STORAGE_TYPE_DUMMY, {source_uri});
    source->set_create_time(source_create_time);
    auto target = MakeLocation(
        "target_location", CLS_WRITING, now_us - config.orphan_writing_grace_period_ms * 1000 - 1);
    MigrationCopyGuard guard;
    guard.set_schema_version(MigrationCopyGuard::kCurrentSchemaVersion);
    guard.set_state(MigrationCopyGuardState::MCGS_UNKNOWN);
    guard.set_operation_id("operation-after-restart");
    guard.set_source_location_id(source->id());
    guard.set_source_location_create_time(source_create_time);
    guard.set_source_storage_name("pace_dram");
    guard.set_target_storage_name("pace_ssd");
    guard.set_total_bytes(1);
    target->set_migration_copy_guard(guard);

    CacheLocationMap locations;
    locations[source->id()] = source;
    locations[target->id()] = target;
    const auto request =
        gc->BuildDeleteActions("instance_a", MakeBatch(SCAN_BASE_CURSOR, {10}, {locations}), now_us)
            .executor_request;

    EXPECT_TRUE(request.block_keys.empty());
    EXPECT_TRUE(might_exist_calls.empty());
    EXPECT_FALSE(gc->IsOrphanWriting(target->id(), *target, now_us));
}

TEST_F(CacheGarbageCollectorTest, MissingScannedKeyIsNotCountedAsOperationError) {
    auto gc = MakeGc(DefaultConfig());
    PrepareForSingleStep(*gc);

    const auto batch =
        MakeBatch(SCAN_BASE_CURSOR, {1, 2}, {CacheLocationMap{}, CacheLocationMap{}}, {EC_NOENT, EC_ERROR});
    EXPECT_TRUE(gc->BuildDeleteActions("instance_a", batch, TimestampUtil::GetCurrentTimeUs())
                    .executor_request.block_keys.empty());

    EXPECT_EQ(
        1, metrics_registry_->GetCounter("cache_gc.operation_error_count", MetricsTags{{"stage", "scan_key"}}).Get());
}

TEST_F(CacheGarbageCollectorTest, TickScansOneBatchAndSubmitsConditionalAsyncDelete) {
    auto config = DefaultConfig();
    const int64_t old_time = OldCreateTimeUs(config, 1);
    CacheLocationMap locations;
    locations["loc"] = MakeLocation("loc", CLS_WRITING, old_time);
    scan_responses["instance_a"] = {{EC_OK, MakeBatch(SCAN_BASE_CURSOR, {100}, {locations})}};

    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);
    gc->RunOneTick();

    ASSERT_EQ(1, scan_calls.size());
    EXPECT_EQ(std::make_pair(std::string("instance_a"), SCAN_BASE_CURSOR), scan_calls.front());
    ASSERT_EQ(1, submitted_requests.size());
    EXPECT_EQ("instance_a", submitted_requests.front().instance_id);
    EXPECT_EQ((KeyVector{100}), submitted_requests.front().block_keys);
    EXPECT_EQ((std::vector<std::vector<std::string>>{{"loc"}}), submitted_requests.front().location_ids);
    EXPECT_EQ((std::vector<std::vector<std::string>>{{locations.at("loc")->ToJsonString()}}),
              submitted_requests.front().expected_location_values);
    EXPECT_TRUE(submitted_requests.front().authoritative_read);
    EXPECT_EQ(1, gc->inflight_deletes_.size());
    EXPECT_EQ(1, gc->pending_locations_.size());
}

TEST_F(CacheGarbageCollectorTest, ScanFailureRetriesSameCursorWithoutBusyLoop) {
    scan_responses["instance_a"] = {
        {EC_ERROR, {}},
        {EC_OK, MakeBatch("next", {}, {})},
    };
    auto gc = MakeGc(DefaultConfig());
    PrepareForSingleStep(*gc);

    gc->RunOneTick();
    ASSERT_EQ(1, scan_calls.size());
    ASSERT_EQ(1u, gc->instances_.size());
    EXPECT_EQ(SCAN_BASE_CURSOR, gc->instances_.front().cursor);
    EXPECT_TRUE(gc->inflight_deletes_.empty());

    gc->RunOneTick();
    ASSERT_EQ(2, scan_calls.size());
    EXPECT_EQ(SCAN_BASE_CURSOR, scan_calls[0].second);
    EXPECT_EQ(SCAN_BASE_CURSOR, scan_calls[1].second);
    EXPECT_EQ("next", gc->instances_.front().cursor);
}

TEST_F(CacheGarbageCollectorTest, PersistentlyFailingInstanceIsSkippedAfterBoundedRetriesAndNextRoundProceeds) {
    AddInstance("group_a", "instance_b");
    scan_responses["instance_a"] = {
        {EC_ERROR, {}},
        {EC_ERROR, {}},
        {EC_ERROR, {}},
        {EC_OK, MakeBatch(SCAN_BASE_CURSOR, {}, {})},
    };
    scan_responses["instance_b"] = {
        {EC_OK, MakeBatch(SCAN_BASE_CURSOR, {}, {})},
        {EC_OK, MakeBatch(SCAN_BASE_CURSOR, {}, {})},
    };
    auto gc = MakeGc(DefaultConfig());
    PrepareForSingleStep(*gc);

    for (size_t i = 0; i < CacheGarbageCollector::kMaxScanFailuresPerInstancePerRound + 1; ++i) {
        gc->RunOneTick();
    }

    EXPECT_EQ((std::vector<std::pair<std::string, std::string>>{
                  {"instance_a", SCAN_BASE_CURSOR},
                  {"instance_b", SCAN_BASE_CURSOR},
                  {"instance_a", SCAN_BASE_CURSOR},
                  {"instance_a", SCAN_BASE_CURSOR},
              }),
              scan_calls);
    EXPECT_FALSE(gc->round_active_);
    EXPECT_EQ(1, gc->get_cache_gc_scan_round_count_metrics());
    EXPECT_EQ(
        1, metrics_registry_->GetCounter("cache_gc.operation_error_count", {{"stage", "scan_retry_exhausted"}}).Get());

    AddInstance("group_a", "instance_c");
    gc->next_round_at_ = CacheGarbageCollector::Clock::now();
    for (size_t i = 0; i < 3; ++i) {
        gc->RunOneTick();
    }

    ASSERT_GE(scan_calls.size(), 3u);
    const std::vector<std::pair<std::string, std::string>> second_round_calls(
        scan_calls.end() - 3, scan_calls.end());
    EXPECT_EQ((std::vector<std::pair<std::string, std::string>>{
                  {"instance_a", SCAN_BASE_CURSOR},
                  {"instance_b", SCAN_BASE_CURSOR},
                  {"instance_c", SCAN_BASE_CURSOR},
              }),
              second_round_calls);
    EXPECT_FALSE(gc->round_active_);
    EXPECT_EQ(2, gc->get_cache_gc_scan_round_count_metrics());
}

TEST_F(CacheGarbageCollectorTest, ActiveRoundRotatesOneBatchPerInstance) {
    AddInstance("group_a", "instance_b");
    scan_responses["instance_a"] = {
        {EC_OK, MakeBatch("a_next", {}, {})},
        {EC_OK, MakeBatch(SCAN_BASE_CURSOR, {}, {})},
    };
    scan_responses["instance_b"] = {
        {EC_OK, MakeBatch("b_next", {}, {})},
        {EC_OK, MakeBatch(SCAN_BASE_CURSOR, {}, {})},
    };
    auto gc = MakeGc(DefaultConfig());
    PrepareForSingleStep(*gc);

    for (size_t i = 0; i < 4; ++i) {
        gc->RunOneTick();
    }

    EXPECT_EQ((std::vector<std::pair<std::string, std::string>>{
                  {"instance_a", SCAN_BASE_CURSOR},
                  {"instance_b", SCAN_BASE_CURSOR},
                  {"instance_a", "a_next"},
                  {"instance_b", "b_next"},
              }),
              scan_calls);
    EXPECT_FALSE(gc->round_active_);
}

TEST_F(CacheGarbageCollectorTest, RegistryFailureDiscardsIncompleteSnapshot) {
    AddInstance("group_b", "instance_b");
    instances_by_group["group_b"].first = EC_ERROR;
    auto gc = MakeGc(DefaultConfig());
    PrepareForSingleStep(*gc);

    gc->RunOneTick();
    EXPECT_FALSE(gc->round_active_);
    EXPECT_TRUE(gc->instances_.empty());
    EXPECT_TRUE(scan_calls.empty());

    instances_by_group["group_b"].first = EC_OK;
    gc->RunOneTick();
    EXPECT_TRUE(gc->round_active_);
    ASSERT_EQ(2, gc->instances_.size());
    ASSERT_EQ(1, scan_calls.size());
    EXPECT_EQ("instance_a", scan_calls.front().first);
}

TEST_F(CacheGarbageCollectorTest, InflightLimitBlocksScanAndReadyFutureResumes) {
    auto config = DefaultConfig();
    config.max_inflight_delete_requests = 1;
    const int64_t old_time = OldCreateTimeUs(config, 1);
    CacheLocationMap first_locations;
    first_locations["loc_1"] = MakeLocation("loc_1", CLS_WRITING, old_time);
    scan_responses["instance_a"] = {
        {EC_OK, MakeBatch("next", {1}, {first_locations})},
        {EC_OK, MakeBatch(SCAN_BASE_CURSOR, {}, {})},
    };
    submit_mode = SubmitMode::kPending;

    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);
    gc->RunOneTick();
    ASSERT_EQ(1, scan_calls.size());
    ASSERT_EQ(1, gc->inflight_deletes_.size());

    gc->RunOneTick();
    EXPECT_EQ(1, scan_calls.size());
    ASSERT_TRUE(pending_delete_promise);
    pending_delete_promise->set_value({EC_OK, ""});

    gc->RunOneTick();
    EXPECT_EQ(2, scan_calls.size());
    EXPECT_TRUE(gc->inflight_deletes_.empty());
    EXPECT_TRUE(gc->pending_locations_.empty());
    EXPECT_EQ(1, gc->get_cache_gc_scan_round_count_metrics());
}

TEST_F(CacheGarbageCollectorTest, BoundedInflightWindowProgressesAroundOneStuckDelete) {
    auto config = DefaultConfig();
    config.max_inflight_delete_requests = 2;
    const int64_t old_time = OldCreateTimeUs(config, 1);
    CacheLocationMap first_locations;
    first_locations["loc_1"] = MakeLocation("loc_1", CLS_WRITING, old_time);
    CacheLocationMap second_locations;
    second_locations["loc_2"] = MakeLocation("loc_2", CLS_WRITING, old_time);
    CacheLocationMap third_locations;
    third_locations["loc_3"] = MakeLocation("loc_3", CLS_WRITING, old_time);
    scan_responses["instance_a"] = {
        {EC_OK, MakeBatch("cursor_2", {1}, {first_locations})},
        {EC_OK, MakeBatch("cursor_3", {2}, {second_locations})},
        {EC_OK, MakeBatch(SCAN_BASE_CURSOR, {3}, {third_locations})},
    };

    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);
    submit_mode = SubmitMode::kPending;
    gc->RunOneTick();
    ASSERT_EQ(1, pending_delete_promises.size());
    ASSERT_EQ(1, gc->inflight_deletes_.size());

    submit_mode = SubmitMode::kReadyOk;
    gc->RunOneTick();
    ASSERT_EQ(2, gc->inflight_deletes_.size());
    EXPECT_EQ(2, scan_calls.size());

    gc->RunOneTick();
    EXPECT_EQ(3, scan_calls.size());
    EXPECT_EQ(3, submitted_requests.size());
    EXPECT_EQ(2, gc->inflight_deletes_.size());
    EXPECT_EQ(2, gc->pending_locations_.size());
    EXPECT_EQ(2, gc->get_cache_gc_inflight_delete_count_metrics());

    pending_delete_promises.front()->set_value({EC_OK, ""});
    gc->RunOneTick();
    EXPECT_TRUE(gc->inflight_deletes_.empty());
    EXPECT_TRUE(gc->pending_locations_.empty());
    EXPECT_EQ(0, gc->get_cache_gc_inflight_delete_count_metrics());
    EXPECT_EQ(1, gc->get_cache_gc_scan_round_count_metrics());
}

TEST_F(CacheGarbageCollectorTest, PendingTargetDeduplicatesBeforeCasAndKeepsInstanceIsolation) {
    auto config = DefaultConfig();
    config.max_inflight_delete_requests = 2;
    const int64_t old_time = OldCreateTimeUs(config, 1);
    CacheLocationMap locations;
    locations["same_location"] = MakeLocation("same_location", CLS_WRITING, old_time);
    const auto batch = MakeBatch("next", {9}, {locations});
    scan_responses["instance_a"] = {
        {EC_OK, batch},
        {EC_OK, MakeBatch(SCAN_BASE_CURSOR, {9}, {locations})},
    };

    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);
    submit_mode = SubmitMode::kPending;
    gc->RunOneTick();
    ASSERT_EQ(1, gc->inflight_deletes_.size());
    ASSERT_EQ(1, gc->pending_locations_.size());

    gc->RunOneTick();
    EXPECT_EQ(2, scan_calls.size());
    EXPECT_EQ(1, submitted_requests.size());
    EXPECT_EQ(1, gc->inflight_deletes_.size());

    EXPECT_TRUE(gc->BuildDeleteActions("instance_a", batch, TimestampUtil::GetCurrentTimeUs())
                    .executor_request.block_keys.empty());
    EXPECT_FALSE(gc->BuildDeleteActions("instance_b", batch, TimestampUtil::GetCurrentTimeUs())
                     .executor_request.block_keys.empty());

    pending_delete_promises.front()->set_value({EC_OK, ""});
    gc->PollInflightDeletes();
    EXPECT_TRUE(gc->inflight_deletes_.empty());
    EXPECT_TRUE(gc->pending_locations_.empty());
    EXPECT_FALSE(gc->BuildDeleteActions("instance_a", batch, TimestampUtil::GetCurrentTimeUs())
                     .executor_request.block_keys.empty());
}

TEST_F(CacheGarbageCollectorTest, PendingTargetRemainsDeduplicatedAcrossRoundBoundary) {
    auto config = DefaultConfig();
    config.max_inflight_delete_requests = 2;
    const int64_t old_time = OldCreateTimeUs(config, 1);
    CacheLocationMap locations;
    locations["same_location"] = MakeLocation("same_location", CLS_WRITING, old_time);
    const auto batch = MakeBatch(SCAN_BASE_CURSOR, {9}, {locations});
    scan_responses["instance_a"] = {{EC_OK, batch}, {EC_OK, batch}};

    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);
    submit_mode = SubmitMode::kPending;
    gc->RunOneTick();
    ASSERT_FALSE(gc->round_active_);
    ASSERT_EQ(1, gc->inflight_deletes_.size());
    ASSERT_EQ(1, gc->pending_locations_.size());
    ASSERT_EQ(1, submitted_requests.size());

    gc->next_round_at_ = CacheGarbageCollector::Clock::now();
    gc->RunOneTick();
    EXPECT_EQ(2, scan_calls.size());
    EXPECT_EQ(1, submitted_requests.size());
    EXPECT_EQ(1, gc->inflight_deletes_.size());
    EXPECT_EQ(1, gc->pending_locations_.size());

    pending_delete_promises.front()->set_value({EC_OK, ""});
}

TEST_F(CacheGarbageCollectorTest, RealExecutorBacklogDoesNotBlockGcSubmitAndStopsFurtherScan) {
    auto config = DefaultConfig();
    config.max_inflight_delete_requests = 1;
    const int64_t old_time = OldCreateTimeUs(config, 1);
    CacheLocationMap locations;
    locations["loc_1"] = MakeLocation("loc_1", CLS_WRITING, old_time);
    scan_responses["instance_a"] = {
        {EC_OK, MakeBatch("next", {1}, {locations})},
        {EC_OK, MakeBatch(SCAN_BASE_CURSOR, {}, {})},
    };

    stub_.reset(static_cast<SubmitAsyncLocation>(ADDR(SchedulePlanExecutor, SubmitAsync)));
    stub_.set(ADDR(MetaIndexer, GetLocationsFromPersistent), GetLocationsFromPersistentErrorStub);

    std::promise<void> blocker_started;
    std::promise<void> release_blocker;
    const auto release_future = release_blocker.get_future().share();
    ASSERT_TRUE(executor_->SubmitTask([&blocker_started, release_future]() {
        blocker_started.set_value();
        release_future.wait_for(std::chrono::seconds(2));
    }));
    ASSERT_EQ(std::future_status::ready, blocker_started.get_future().wait_for(std::chrono::seconds(1)));

    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);
    const auto begin = std::chrono::steady_clock::now();
    gc->RunOneTick();
    const auto submit_cost = std::chrono::steady_clock::now() - begin;
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(submit_cost).count(), 100);
    ASSERT_EQ(1, gc->inflight_deletes_.size());
    ASSERT_EQ(1, scan_calls.size());

    gc->RunOneTick();
    EXPECT_EQ(1, scan_calls.size());

    release_blocker.set_value();
    ASSERT_EQ(std::future_status::ready, gc->inflight_deletes_.front().future.wait_for(std::chrono::seconds(1)));
    gc->RunOneTick();
    EXPECT_TRUE(gc->inflight_deletes_.empty());
    EXPECT_EQ(2, scan_calls.size());
}

TEST_F(CacheGarbageCollectorTest, RejectedAndBrokenSubmissionsDoNotOccupySlot) {
    auto config = DefaultConfig();
    const int64_t old_time = OldCreateTimeUs(config, 1);
    CacheLocationMap locations;
    locations["loc"] = MakeLocation("loc", CLS_WRITING, old_time);
    scan_responses["instance_a"] = {{EC_OK, MakeBatch("next", {1}, {locations})}};

    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);
    submit_mode = SubmitMode::kRejected;
    gc->RunOneTick();
    EXPECT_TRUE(gc->inflight_deletes_.empty());
    EXPECT_TRUE(gc->pending_locations_.empty());

    scan_responses["instance_a"].push_back({EC_OK, MakeBatch("last", {2}, {locations})});
    submit_mode = SubmitMode::kAcceptedInvalid;
    gc->RunOneTick();
    EXPECT_TRUE(gc->inflight_deletes_.empty());
    EXPECT_TRUE(gc->pending_locations_.empty());

    scan_responses["instance_a"].push_back({EC_OK, MakeBatch(SCAN_BASE_CURSOR, {3}, {locations})});
    submit_mode = SubmitMode::kThrow;
    gc->RunOneTick();
    EXPECT_TRUE(gc->inflight_deletes_.empty());
    EXPECT_TRUE(gc->pending_locations_.empty());
    EXPECT_EQ(3, submitted_requests.size());
    EXPECT_EQ(
        1,
        metrics_registry_->GetCounter("cache_gc.operation_error_count", MetricsTags{{"stage", "submit_exception"}})
            .Get());
    EXPECT_EQ(0,
              metrics_registry_->GetCounter("cache_gc.operation_error_count", MetricsTags{{"stage", "tick_exception"}})
                  .Get());
}

TEST_F(CacheGarbageCollectorTest, FuturePartialAndExceptionAreTerminal) {
    auto config = DefaultConfig();
    const int64_t old_time = OldCreateTimeUs(config, 1);
    CacheLocationMap locations;
    locations["loc"] = MakeLocation("loc", CLS_WRITING, old_time);
    scan_responses["instance_a"] = {
        {EC_OK, MakeBatch("next", {1}, {locations})},
        {EC_OK, MakeBatch("last", {2}, {locations})},
        {EC_OK, MakeBatch(SCAN_BASE_CURSOR, {}, {})},
    };
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    submit_mode = SubmitMode::kReadyPartial;
    gc->RunOneTick();
    ASSERT_EQ(1, gc->inflight_deletes_.size());
    submit_mode = SubmitMode::kFutureException;
    gc->RunOneTick();
    ASSERT_EQ(1, gc->inflight_deletes_.size());
    gc->RunOneTick();
    EXPECT_TRUE(gc->inflight_deletes_.empty());
    EXPECT_TRUE(gc->pending_locations_.empty());
    EXPECT_EQ(3, scan_calls.size());
}

TEST_F(CacheGarbageCollectorTest, SnapshotPreservesInstanceIsolationAndCooldown) {
    AddInstance("group_a", "instance_b");
    auto config = DefaultConfig();
    const int64_t old_time = OldCreateTimeUs(config, 1);
    CacheLocationMap locations;
    locations["same_location"] = MakeLocation("same_location", CLS_WRITING, old_time);
    scan_responses["instance_a"] = {{EC_OK, MakeBatch(SCAN_BASE_CURSOR, {9}, {locations})}};
    scan_responses["instance_b"] = {{EC_OK, MakeBatch(SCAN_BASE_CURSOR, {9}, {locations})}};

    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);
    gc->RunOneTick();
    gc->RunOneTick();
    gc->RunOneTick();
    ASSERT_EQ(2, submitted_requests.size());
    EXPECT_EQ("instance_a", submitted_requests[0].instance_id);
    EXPECT_EQ("instance_b", submitted_requests[1].instance_id);
    EXPECT_EQ(1, gc->get_cache_gc_scan_round_count_metrics());

    const size_t snapshot_count = list_groups_call_count;
    gc->RunOneTick();
    EXPECT_EQ(snapshot_count, list_groups_call_count);
}

TEST_F(CacheGarbageCollectorTest, MissingIndexerAdvancesAndEmptyRequestIsNotSubmitted) {
    AddInstance("group_a", "instance_b");
    missing_indexers.insert("instance_a");
    scan_responses["instance_b"] = {{EC_OK, MakeBatch(SCAN_BASE_CURSOR, {1}, {CacheLocationMap{}})}};

    auto gc = MakeGc(DefaultConfig());
    PrepareForSingleStep(*gc);
    gc->RunOneTick();
    EXPECT_TRUE(scan_calls.empty());
    gc->RunOneTick();
    EXPECT_EQ(1, scan_calls.size());
    EXPECT_TRUE(submitted_requests.empty());
    EXPECT_EQ(1, gc->get_cache_gc_scan_round_count_metrics());
}

TEST_F(CacheGarbageCollectorTest, StopInterruptsLongTickWait) {
    auto config = DefaultConfig();
    config.scan_interval_ms = 60000;
    auto gc = MakeGc(config);
    ASSERT_EQ(EC_OK, gc->Start());
    const auto begin = std::chrono::steady_clock::now();
    gc->RequestStop();
    gc->Join();
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count(), 1);
}

TEST_F(CacheGarbageCollectorTest, JoinDetachesPendingDeleteWithoutWaitingForFuture) {
    auto config = DefaultConfig();
    const int64_t old_time = OldCreateTimeUs(config, 1);
    CacheLocationMap locations;
    locations["loc_1"] = MakeLocation("loc_1", CLS_WRITING, old_time);
    scan_responses["instance_a"] = {{EC_OK, MakeBatch(SCAN_BASE_CURSOR, {1}, {locations})}};
    submit_mode = SubmitMode::kPending;

    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);
    gc->RunOneTick();
    ASSERT_EQ(1, gc->inflight_deletes_.size());
    ASSERT_EQ(1, gc->pending_locations_.size());
    ASSERT_TRUE(pending_delete_promise);

    const auto begin = std::chrono::steady_clock::now();
    gc->Join();
    const auto elapsed = std::chrono::steady_clock::now() - begin;

    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count(), 1);
    EXPECT_TRUE(gc->inflight_deletes_.empty());
    EXPECT_TRUE(gc->pending_locations_.empty());
    EXPECT_EQ(0, gc->get_cache_gc_inflight_delete_count_metrics());
    pending_delete_promise->set_value({EC_OK, ""});
}

TEST_F(CacheGarbageCollectorTest, JoinDetachesPendingEventReportActionWithoutWaitingForFuture) {
    auto config = DefaultConfig();
    config.event_report_cleanup_enabled = true;
    auto gc = MakeGc(config);
    PrepareForSingleStep(*gc);

    auto backend = AddEventReportBackend("event_report_l1p5");
    const std::string host = "10.0.0.1:9000";
    ASSERT_EQ(EC_OK, backend->RegisterNode("instance_a", host, {"hbm"}));
    uint64_t generation = 0;
    ASSERT_EQ(EC_OK, backend->UnregisterNodeForHostDown("instance_a", host, generation));
    const std::string location_id = backend->BuildLocationId("hbm", host);
    CacheLocationMap locations;
    locations.emplace(
        location_id,
        MakeStoredLocation(
            location_id, CLS_SERVING, backend->GetStorageType(), {"event_report://event_report_l1p5/object"}));
    scan_responses["instance_a"] = {{EC_OK, MakeBatch(SCAN_BASE_CURSOR, {101}, {locations})}};

    submit_mode = SubmitMode::kPending;
    gc->RunOneTick();
    ASSERT_EQ(1u, submitted_event_report_requests.size());
    ASSERT_EQ(1u, gc->inflight_deletes_.size());
    ASSERT_EQ(1u, gc->pending_locations_.size());
    ASSERT_TRUE(pending_delete_promise);

    const auto begin = std::chrono::steady_clock::now();

    gc->Join();
    const auto elapsed = std::chrono::steady_clock::now() - begin;

    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count(), 1);
    EXPECT_TRUE(gc->inflight_deletes_.empty());
    EXPECT_TRUE(gc->pending_locations_.empty());
    pending_delete_promise->set_value({EC_OK, ""});
    backend->Close();
}

TEST_F(CacheGarbageCollectorTest, JoinWaitsForActiveMaintenanceScan) {
    auto config = DefaultConfig();
    config.scan_interval_ms = 60000;
    block_scan.store(true, std::memory_order_release);
    release_scan.store(false, std::memory_order_release);

    auto gc = MakeGc(config);
    ASSERT_EQ(EC_OK, gc->Start());
    for (size_t i = 0; i < 100 && !scan_block_entered.load(std::memory_order_acquire); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!scan_block_entered.load(std::memory_order_acquire)) {
        release_scan.store(true, std::memory_order_release);
        gc->Stop();
        FAIL() << "maintenance scan did not start";
    }

    auto join_future = std::async(std::launch::async, [&gc]() {
        gc->RequestStop();
        gc->Join();
    });
    EXPECT_EQ(std::future_status::timeout, join_future.wait_for(std::chrono::milliseconds(50)));

    release_scan.store(true, std::memory_order_release);
    ASSERT_EQ(std::future_status::ready, join_future.wait_for(std::chrono::seconds(1)));
    join_future.get();
    EXPECT_FALSE(gc->IsRunning());
}

} // namespace kv_cache_manager
