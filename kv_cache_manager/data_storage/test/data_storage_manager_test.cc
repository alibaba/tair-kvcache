#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <unistd.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/dummy_backend.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;

namespace {

class RecordingCreateBackend : public DummyBackend {
public:
    using DummyBackend::DummyBackend;

    DataStorageCreateResult Create(const std::vector<std::string> &keys,
                                   size_t size_per_key,
                                   const std::string &trace_id,
                                   std::function<void()> cb) override {
        requests.push_back({keys, size_per_key});
        trace_ids.push_back(trace_id);
        DataStorageCreateResult result;
        if (requests.size() <= scripted_results.size()) {
            result = scripted_results[requests.size() - 1];
        } else {
            for (const auto &key : keys) {
                result.emplace_back(EC_OK,
                                    DataStorageUri("file://backend/" + key + "?size=" + std::to_string(size_per_key)));
            }
        }
        if (cb) {
            cb();
        }
        return result;
    }

    std::vector<DataStorageCreateRequest> requests;
    std::vector<std::string> trace_ids;
    std::vector<DataStorageCreateResult> scripted_results;
};

class RecordingCleanupBackend : public RecordingCreateBackend {
public:
    using RecordingCreateBackend::RecordingCreateBackend;

    DataStorageCreateOutcome CreateWithCleanup(const std::vector<std::string> &keys,
                                               size_t size,
                                               const std::string &trace,
                                               std::function<void()> cb) override {
        return {Create(keys, size, trace, std::move(cb)), {DataStorageUri("file://backend/unmapped?size=128")}};
    }
};

class RecordingBatchBackend : public RecordingCreateBackend {
public:
    using RecordingCreateBackend::RecordingCreateBackend;

    bool Writable() override { return writable && Available(); }

    DataStorageBatchCreateResult CreateBatch(const std::vector<DataStorageCreateRequest> &requests,
                                             const std::string &trace_id,
                                             std::function<void()> cb) override {
        ++batch_call_count;
        batch_requests = requests;
        batch_trace_id = trace_id;
        if (cb) {
            cb();
        }
        return batch_result;
    }

    size_t batch_call_count = 0;
    bool writable = true;
    std::vector<DataStorageCreateRequest> batch_requests;
    std::string batch_trace_id;
    DataStorageBatchCreateResult batch_result;
};

class BlockingBatchBackend : public RecordingCreateBackend {
public:
    using RecordingCreateBackend::RecordingCreateBackend;

    DataStorageBatchCreateResult CreateBatch(const std::vector<DataStorageCreateRequest> &requests,
                                             const std::string &trace_id,
                                             std::function<void()> cb) override {
        auto worker = std::async(std::launch::async, [this, release = release_worker.get_future()]() {
            worker_started.set_value();
            return release.wait_for(std::chrono::seconds(5));
        });
        worker_released = worker.get() == std::future_status::ready;
        return DataStorageBackend::CreateBatch(requests, trace_id, std::move(cb));
    }

    std::promise<void> worker_started;
    std::promise<void> release_worker;
    bool worker_released = false;
};

} // namespace

class DataStorageManagerTest : public TESTBASE {
public:
    void SetUp() override { metrics_registry_ = std::make_shared<MetricsRegistry>(); }
    void TearDown() override {}
    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

TEST_F(DataStorageManagerTest, TestSimple) {
    DataStorageManager data_storage_manager(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(1);
    spec->set_root_path("/data/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "storage1", spec);
    RequestContext request_context("test");
    // register storage
    ASSERT_EQ(EC_OK, data_storage_manager.RegisterStorage(&request_context, "storage1", storage_config));
    ASSERT_EQ(EC_EXIST, data_storage_manager.RegisterStorage(&request_context, "storage1", storage_config));

    // get all storage name list
    std::vector<std::string> data_storage_names = data_storage_manager.GetAllStorageNames();
    ASSERT_EQ(1, data_storage_names.size());
    ASSERT_EQ("storage1", data_storage_names[0]);

    // get available storages
    std::vector<std::shared_ptr<DataStorageBackend>> data_storage_backends =
        data_storage_manager.GetAvailableStorages();
    ASSERT_EQ(1, data_storage_backends.size());

    // get storage by name
    std::shared_ptr<DataStorageBackend> data_storage_backend = data_storage_manager.GetDataStorageBackend("storage1");
    ASSERT_NE(nullptr, data_storage_backend);
    ASSERT_EQ(nullptr, data_storage_manager.GetDataStorageBackend("storage2"));

    // disable storage
    ASSERT_EQ(EC_OK, data_storage_manager.DisableStorage("storage1"));
    EXPECT_FALSE(data_storage_backend->Available());
    ASSERT_EQ(EC_NOENT, data_storage_manager.DisableStorage("storage2"));
    auto disabled_create = data_storage_manager.Create(&request_context, "storage1", {"disabled_key"}, 128, []() {});
    ASSERT_EQ(1u, disabled_create.results.size());
    EXPECT_EQ(EC_NOENT, disabled_create.results[0].first);
    EXPECT_TRUE(disabled_create.cleanup_uris.empty());

    // enable storage
    ASSERT_EQ(EC_OK, data_storage_manager.EnableStorage("storage1"));
    EXPECT_TRUE(data_storage_backend->Available());
    ASSERT_EQ(EC_NOENT, data_storage_manager.EnableStorage("storage2"));

    // create exist delete
    DataStorageUri storage_uri1("file://storage1/data/key1?size=128");
    // ASSERT_FALSE(data_storage_manager.Exist("storage1", {storage_uri1})[0]);
    RequestContext requesst_context("test");
    auto uris = data_storage_manager.Create(&requesst_context, "storage1", {"key1"}, 128, []() {});
    ASSERT_EQ(1, uris.results.size());
    ASSERT_EQ(EC_OK, uris.results[0].first);
    ASSERT_EQ(storage_uri1.ToUriString(), uris.results[0].second.ToUriString());
    EXPECT_TRUE(uris.cleanup_uris.empty());
    // unique name not exist
    uris = data_storage_manager.Create(&requesst_context, "storage2", {"key1"}, 128, []() {});
    ASSERT_EQ(0, uris.results.size());

    // unregister storage
    ASSERT_EQ(EC_OK, data_storage_manager.UnRegisterStorage("storage1"));
    ASSERT_EQ(EC_NOENT, data_storage_manager.UnRegisterStorage("storage2"));
    data_storage_backends = data_storage_manager.GetAvailableStorages();
    ASSERT_EQ(0, data_storage_backends.size());
}

TEST_F(DataStorageManagerTest, TestCopyRejectsMismatchedUris) {
    DataStorageManager data_storage_manager(metrics_registry_);
    RequestContext request_context("test");
    std::vector<DataStorageUri> src_uris = {DataStorageUri("file://storage1/data/src1?size=128"),
                                            DataStorageUri("file://storage1/data/src2?size=128")};
    std::vector<DataStorageUri> dst_uris = {DataStorageUri("file://storage1/data/dst1?size=128")};

    const auto results = data_storage_manager.Copy(&request_context, "storage1", src_uris, dst_uris);
    ASSERT_EQ(src_uris.size(), results.size());
    for (const auto ec : results) {
        ASSERT_EQ(EC_BADARGS, ec);
    }
}

TEST_F(DataStorageManagerTest, TestOptionalBackendsFollowBuildConfig) {
    DataStorageManager data_storage_manager(metrics_registry_);

#ifdef ENABLE_MOONCAKE
    EXPECT_NE(nullptr, data_storage_manager.CreateStorageBackend(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE));
#else
    EXPECT_EQ(nullptr, data_storage_manager.CreateStorageBackend(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE));
#endif

#ifdef ENABLE_VCNS
    EXPECT_NE(nullptr, data_storage_manager.CreateStorageBackend(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
#else
    EXPECT_EQ(nullptr, data_storage_manager.CreateStorageBackend(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
#endif

    auto pace_ssd_backend =
        data_storage_manager.CreateStorageBackend(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD);
    ASSERT_NE(nullptr, pace_ssd_backend);
    auto pace_ssd_spec = std::make_shared<TairMemPoolStorageSpec>();
    pace_ssd_spec->set_domain("pace.meta");
    pace_ssd_spec->set_timeout(5000);
    pace_ssd_spec->set_media_type(kTairMemPoolMediaTypeSsd);
    StorageConfig pace_ssd_config(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD, "pace_ssd_1", pace_ssd_spec);
    // Do not assert Open(): the open-source stub intentionally returns EC_ERROR,
    // while the internal PACE backend can initialize successfully.
    pace_ssd_backend->config_ = pace_ssd_config;
    EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD, pace_ssd_backend->GetType());
}

TEST_F(DataStorageManagerTest, TestConfirmedMissingDeletePolicies) {
    DataStorageManager manager(metrics_registry_);
    std::vector<DataStorageType> skip_types = {
        DataStorageType::DATA_STORAGE_TYPE_NFS,
        DataStorageType::DATA_STORAGE_TYPE_HF3FS,
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
    };
#ifdef ENABLE_MOONCAKE
    skip_types.push_back(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
#endif
#ifdef ENABLE_VCNS
    skip_types.push_back(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS);
#endif
    for (const auto type : skip_types) {
        const auto backend = manager.CreateStorageBackend(type);
        ASSERT_NE(nullptr, backend);
        EXPECT_TRUE(backend->ShouldSkipConfirmedMissingBackendDelete()) << static_cast<int>(type);
        EXPECT_FALSE(backend->DataStorageBackend::ShouldSkipConfirmedMissingBackendDelete());
    }
    for (const auto type :
         {DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD}) {
        const auto backend = manager.CreateStorageBackend(type);
        ASSERT_NE(nullptr, backend);
        EXPECT_FALSE(backend->ShouldSkipConfirmedMissingBackendDelete());
    }
}

TEST_F(DataStorageManagerTest, TestDefaultCreateBatchPreservesGroupsAndOrder) {
    RecordingCreateBackend backend(metrics_registry_);
    const std::vector<DataStorageCreateRequest> requests = {
        {{"first", "second"}, 128}, {{}, 256}, {{"third"}, 256}, {{"fourth"}, 128}};
    size_t callback_count = 0;

    const auto batch_result = backend.CreateBatch(requests, "batch_trace", [&]() { ++callback_count; });

    const auto &results = batch_result.results;
    EXPECT_EQ(1u, callback_count);
    EXPECT_TRUE(batch_result.cleanup_uris.empty());
    ASSERT_EQ(3u, backend.requests.size());
    EXPECT_EQ(requests[0].keys, backend.requests[0].keys);
    EXPECT_EQ(requests[2].keys, backend.requests[1].keys);
    EXPECT_EQ(requests[3].keys, backend.requests[2].keys);
    EXPECT_EQ(128u, backend.requests[0].size_per_key);
    EXPECT_EQ(256u, backend.requests[1].size_per_key);
    EXPECT_EQ(128u, backend.requests[2].size_per_key);
    EXPECT_EQ(std::vector<std::string>(3, "batch_trace"), backend.trace_ids);
    ASSERT_EQ(requests.size(), results.size());
    for (size_t group = 0; group < requests.size(); ++group) {
        ASSERT_EQ(requests[group].keys.size(), results[group].size());
        for (size_t key = 0; key < requests[group].keys.size(); ++key) {
            EXPECT_EQ(EC_OK, results[group][key].first);
            EXPECT_EQ("/" + requests[group].keys[key], results[group][key].second.GetPath());
            EXPECT_EQ(std::to_string(requests[group].size_per_key), results[group][key].second.GetParam("size"));
        }
    }
}

TEST_F(DataStorageManagerTest, TestDefaultCreateBatchStopsOnFailureAndPreservesAllocatedUris) {
    RecordingCreateBackend backend(metrics_registry_);
    backend.scripted_results = {{{EC_OK, DataStorageUri("file://backend/first")}},
                                {{EC_OK, DataStorageUri("file://backend/second")}, {EC_IO_ERROR, DataStorageUri{}}}};
    const std::vector<DataStorageCreateRequest> requests = {
        {{"first"}, 128}, {{}, 256}, {{"second", "failed"}, 256}, {{}, 512}, {{"skipped1", "skipped2"}, 512}};
    size_t callback_count = 0;

    const auto batch_result = backend.CreateBatch(requests, "batch_trace", [&]() { ++callback_count; });

    const auto &results = batch_result.results;
    EXPECT_EQ(1u, callback_count);
    EXPECT_TRUE(batch_result.cleanup_uris.empty());
    EXPECT_EQ(2u, backend.requests.size());
    ASSERT_EQ(requests.size(), results.size());
    ASSERT_EQ(1u, results[0].size());
    EXPECT_EQ(EC_OK, results[0][0].first);
    EXPECT_EQ("/first", results[0][0].second.GetPath());
    EXPECT_TRUE(results[1].empty());
    ASSERT_EQ(2u, results[2].size());
    EXPECT_EQ(EC_OK, results[2][0].first);
    EXPECT_EQ("/second", results[2][0].second.GetPath());
    EXPECT_EQ(EC_IO_ERROR, results[2][1].first);
    EXPECT_TRUE(results[3].empty());
    ASSERT_EQ(2u, results[4].size());
    for (const auto &[ec, uri] : results[4]) {
        EXPECT_EQ(EC_ERROR, ec);
        EXPECT_FALSE(uri.Valid());
    }
}

TEST_F(DataStorageManagerTest, TestDefaultCreateBatchSeparatesMalformedResponseAllocationsForCleanup) {
    for (const size_t returned_count : {1u, 3u}) {
        SCOPED_TRACE(returned_count);
        RecordingCreateBackend backend(metrics_registry_);
        backend.scripted_results = {
            {{EC_OK, DataStorageUri("file://backend/earlier")}},
            DataStorageCreateResult(returned_count, {EC_OK, DataStorageUri("file://backend/allocated")})};
        const std::vector<DataStorageCreateRequest> requests = {
            {{"earlier"}, 64}, {{"first", "second"}, 128}, {{"skipped"}, 256}};
        size_t callback_count = 0;

        const auto batch_result = backend.CreateBatch(requests, "batch_trace", [&]() { ++callback_count; });

        const auto &results = batch_result.results;
        EXPECT_EQ(1u, callback_count);
        EXPECT_EQ(2u, backend.requests.size());
        ASSERT_EQ(3u, results.size());
        ASSERT_EQ(1u, results[0].size());
        EXPECT_EQ(EC_OK, results[0][0].first);
        EXPECT_EQ("/earlier", results[0][0].second.GetPath());
        ASSERT_EQ(2u, results[1].size());
        for (const auto &[ec, uri] : results[1]) {
            EXPECT_EQ(EC_ERROR, ec);
            EXPECT_FALSE(uri.Valid());
        }
        ASSERT_EQ(1u, results[2].size());
        EXPECT_EQ(EC_ERROR, results[2][0].first);
        ASSERT_EQ(returned_count, batch_result.cleanup_uris.size());
        for (const auto &uri : batch_result.cleanup_uris) {
            EXPECT_EQ("/allocated", uri.GetPath());
        }
    }
}

TEST_F(DataStorageManagerTest, TestDefaultCreateBatchEmptyRequestsAvoidCreate) {
    RecordingCreateBackend backend(metrics_registry_);
    size_t callback_count = 0;

    EXPECT_TRUE(backend.CreateBatch({}, "batch_trace", [&]() { ++callback_count; }).results.empty());
    const auto batch_result = backend.CreateBatch({{{}, 128}, {{}, 256}}, "batch_trace", [&]() { ++callback_count; });

    const auto &results = batch_result.results;
    ASSERT_EQ(2u, results.size());
    EXPECT_TRUE(results[0].empty());
    EXPECT_TRUE(results[1].empty());
    EXPECT_TRUE(backend.requests.empty());
    EXPECT_EQ(2u, callback_count);
    EXPECT_TRUE(backend.CreateBatch({}, "batch_trace", nullptr).results.empty());
}

TEST_F(DataStorageManagerTest, TestCreateOutcomeRetainsCleanupAndDefaultBatchStops) {
    DataStorageManager manager(metrics_registry_);
    auto backend = std::make_shared<RecordingCleanupBackend>(metrics_registry_);
    auto spec = std::make_shared<DummyStorageSpec>();
    spec->set_root_path("memory://");
    ASSERT_EQ(EC_OK,
              backend->Open(StorageConfig(DataStorageType::DATA_STORAGE_TYPE_DUMMY, "storage1", spec), "open_trace"));
    manager.storage_map_["storage1"] = backend;
    RequestContext context("cleanup_outcome");
    size_t callbacks = 0;
    const auto outcome = manager.Create(&context, "storage1", {"first"}, 128, [&]() { ++callbacks; });
    EXPECT_EQ(callbacks, 1u);
    ASSERT_EQ(outcome.results.size(), 1u);
    EXPECT_EQ(outcome.results[0].first, EC_OK);
    EXPECT_EQ(outcome.results[0].second.GetHostName(), "storage1");
    ASSERT_EQ(outcome.cleanup_uris.size(), 1u);
    EXPECT_EQ(outcome.cleanup_uris[0].ToUriString(), "file://storage1/unmapped?size=128");

    backend->requests.clear();
    const auto batch =
        manager.CreateBatch(&context, "storage1", {{{"first"}, 128}, {{"second"}, 256}}, [&]() { ++callbacks; });
    EXPECT_EQ(callbacks, 2u);
    ASSERT_EQ(backend->requests.size(), 1u);
    ASSERT_EQ(batch.results.size(), 2u);
    ASSERT_EQ(batch.results[0].size(), 1u);
    EXPECT_EQ(batch.results[0][0].first, EC_OK);
    ASSERT_EQ(batch.results[1].size(), 1u);
    EXPECT_EQ(batch.results[1][0].first, EC_ERROR);
    ASSERT_EQ(batch.cleanup_uris.size(), 1u);
    EXPECT_EQ(batch.cleanup_uris[0].ToUriString(), "file://storage1/unmapped?size=128");
}

TEST_F(DataStorageManagerTest, TestCreateBatchDelegatesToBackendAndRetainsRequestObservability) {
    DataStorageManager manager(metrics_registry_);
    auto backend = std::make_shared<RecordingBatchBackend>(metrics_registry_);
    auto spec = std::make_shared<DummyStorageSpec>();
    spec->set_root_path("memory://");
    ASSERT_EQ(EC_OK,
              backend->Open(StorageConfig(DataStorageType::DATA_STORAGE_TYPE_DUMMY, "storage1", spec), "open_trace"));
    manager.storage_map_["storage1"] = backend;
    backend->batch_result.results = {
        {{EC_OK, DataStorageUri("file://backend/first")}, {EC_OK, DataStorageUri("file://backend/second")}},
        {},
        {{EC_IO_ERROR, DataStorageUri("file://failed/third")}},
        {{EC_OK, DataStorageUri("file://backend/fourth")}}};
    backend->batch_result.cleanup_uris = {DataStorageUri("file://backend/unmapped?size=128")};
    const std::vector<DataStorageCreateRequest> requests = {
        {{"first", "second"}, 128}, {{}, 256}, {{"third"}, 256}, {{"fourth"}, 128}};
    RequestContext request_context("batch__kvcm_need_span_tracer");
    auto *parent_span = request_context.parent_span_tracer();
    const auto caller_thread = std::this_thread::get_id();
    const auto collector = backend->GetMetricsCollector();
    ASSERT_NE(nullptr, collector);
    collector->set_data_storage_create_time_us_metrics(-1);
    size_t callback_count = 0;

    const auto batch_result = manager.CreateBatch(&request_context, "storage1", requests, [&]() {
        ++callback_count;
        EXPECT_EQ(caller_thread, std::this_thread::get_id());
        EXPECT_NE(parent_span, request_context.parent_span_tracer());
    });

    const auto &results = batch_result.results;
    EXPECT_EQ(1u, callback_count);
    EXPECT_EQ(1u, backend->batch_call_count);
    EXPECT_TRUE(backend->requests.empty());
    ASSERT_EQ(requests.size(), backend->batch_requests.size());
    for (size_t group = 0; group < requests.size(); ++group) {
        EXPECT_EQ(requests[group].keys, backend->batch_requests[group].keys);
        EXPECT_EQ(requests[group].size_per_key, backend->batch_requests[group].size_per_key);
    }
    EXPECT_EQ(request_context.trace_id(), backend->batch_trace_id);
    ASSERT_EQ(requests.size(), results.size());
    for (size_t group = 0; group < results.size(); ++group) {
        ASSERT_EQ(backend->batch_result.results[group].size(), results[group].size());
        for (size_t key = 0; key < results[group].size(); ++key) {
            const auto &[ec, uri] = results[group][key];
            EXPECT_EQ(backend->batch_result.results[group][key].first, ec);
            EXPECT_EQ(backend->batch_result.results[group][key].second.GetPath(), uri.GetPath());
            EXPECT_EQ(ec == EC_OK ? "storage1" : "failed", uri.GetHostName());
        }
    }
    ASSERT_EQ(1u, batch_result.cleanup_uris.size());
    EXPECT_EQ("file://storage1/unmapped?size=128", batch_result.cleanup_uris[0].ToUriString());
    EXPECT_DOUBLE_EQ(4, collector->get_data_storage_create_keys_qps_metrics());
    EXPECT_GE(collector->get_data_storage_create_time_us_metrics(), 0);
    const auto collectors = request_context.GetMetricsCollectorsVehicle().GetMetricsCollectors();
    ASSERT_EQ(1u, collectors.size());
    EXPECT_EQ(collector, collectors[0]);
    EXPECT_EQ(parent_span, request_context.parent_span_tracer());
    const auto span_text = request_context.EndAndGetSpanTracerDebugStr();
    const auto batch_span = span_text.find("CreateBatch");
    ASSERT_NE(std::string::npos, batch_span);
    EXPECT_EQ(std::string::npos, span_text.find("CreateBatch", batch_span + 1));
}

TEST_F(DataStorageManagerTest, TestCreateBatchRejectsUnwritableBackend) {
    DataStorageManager manager(metrics_registry_);
    auto backend = std::make_shared<RecordingBatchBackend>(metrics_registry_);
    auto spec = std::make_shared<DummyStorageSpec>();
    spec->set_root_path("memory://");
    ASSERT_EQ(EC_OK,
              backend->Open(StorageConfig(DataStorageType::DATA_STORAGE_TYPE_DUMMY, "storage1", spec), "open_trace"));
    manager.storage_map_["storage1"] = backend;
    ASSERT_EQ(EC_OK, manager.DisableStorage("storage1"));
    RequestContext request_context("batch_trace");
    const std::vector<DataStorageCreateRequest> requests = {{{"first", "second"}, 128}, {{}, 256}, {{"third"}, 256}};

    for (const bool read_only : {false, true}) {
        SCOPED_TRACE(read_only);
        if (read_only) {
            ASSERT_EQ(EC_OK, manager.EnableStorage("storage1"));
            ASSERT_TRUE(backend->Available());
            backend->writable = false;
        }
        size_t callback_count = 0;
        const auto batch_result =
            manager.CreateBatch(&request_context, "storage1", requests, [&]() { ++callback_count; });

        const auto &results = batch_result.results;
        EXPECT_EQ(1u, callback_count);
        EXPECT_EQ(0u, backend->batch_call_count);
        EXPECT_TRUE(backend->requests.empty());
        ASSERT_EQ(requests.size(), results.size());
        for (size_t group = 0; group < requests.size(); ++group) {
            ASSERT_EQ(requests[group].keys.size(), results[group].size());
            for (const auto &[ec, uri] : results[group]) {
                EXPECT_EQ(EC_NOENT, ec);
                EXPECT_FALSE(uri.Valid());
            }
        }
        EXPECT_TRUE(request_context.GetMetricsCollectorsVehicle().GetMetricsCollectors().empty());
    }
}

TEST_F(DataStorageManagerTest, TestCreateBatchEmptyRequestsAvoidBackend) {
    DataStorageManager manager(metrics_registry_);
    auto backend = std::make_shared<RecordingBatchBackend>(metrics_registry_);
    manager.storage_map_["storage1"] = backend;
    RequestContext request_context("batch_trace");
    size_t callback_count = 0;

    EXPECT_TRUE(manager.CreateBatch(&request_context, "storage1", {}, [&]() { ++callback_count; }).results.empty());
    const auto batch_result =
        manager.CreateBatch(&request_context, "storage1", {{{}, 128}, {{}, 256}}, [&]() { ++callback_count; });

    const auto &results = batch_result.results;
    ASSERT_EQ(2u, results.size());
    EXPECT_TRUE(results[0].empty());
    EXPECT_TRUE(results[1].empty());
    EXPECT_EQ(0u, backend->batch_call_count);
    EXPECT_TRUE(backend->requests.empty());
    EXPECT_EQ(2u, callback_count);
    EXPECT_TRUE(request_context.GetMetricsCollectorsVehicle().GetMetricsCollectors().empty());
}

TEST_F(DataStorageManagerTest, TestCreateBatchCallbackCanReenterStorageMutations) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    enum class Outcome {
        Success,
        PartialFailure,
        Malformed,
        Missing,
        Disabled,
        ReadOnly,
        EmptyRequests,
        EmptyGroups
    };
    for (const auto outcome : {Outcome::Success,
                               Outcome::PartialFailure,
                               Outcome::Malformed,
                               Outcome::Missing,
                               Outcome::Disabled,
                               Outcome::ReadOnly,
                               Outcome::EmptyRequests,
                               Outcome::EmptyGroups}) {
        for (const bool unregister : {false, true}) {
            SCOPED_TRACE(static_cast<int>(outcome));
            SCOPED_TRACE(unregister ? "UnRegisterStorage" : "DisableStorage");
            const auto check_case = [&]() {
                DataStorageManager manager(metrics_registry_);
                auto backend = std::make_shared<RecordingBatchBackend>(metrics_registry_);
                auto spec = std::make_shared<DummyStorageSpec>();
                spec->set_root_path("memory://");
                ASSERT_EQ(
                    EC_OK,
                    backend->Open(StorageConfig(DataStorageType::DATA_STORAGE_TYPE_DUMMY, "storage1", spec), "open"));
                if (outcome != Outcome::Missing) {
                    manager.storage_map_["storage1"] = backend;
                }
                if (outcome == Outcome::Disabled) {
                    ASSERT_EQ(EC_OK, manager.DisableStorage("storage1"));
                }
                backend->writable = outcome != Outcome::ReadOnly;
                backend->batch_result.results = {{{EC_OK, DataStorageUri("file://backend/first")}},
                                                 {{EC_OK, DataStorageUri("file://backend/second")}}};
                if (outcome == Outcome::PartialFailure) {
                    backend->batch_result.results[1][0] = {EC_IO_ERROR, DataStorageUri{}};
                } else if (outcome == Outcome::Malformed) {
                    backend->batch_result.results[1][0] = {EC_ERROR, DataStorageUri{}};
                    backend->batch_result.cleanup_uris = {DataStorageUri("file://backend/unmapped")};
                }
                std::vector<DataStorageCreateRequest> requests = {{{"first"}, 128}, {{"second"}, 256}};
                if (outcome == Outcome::EmptyRequests) {
                    requests.clear();
                } else if (outcome == Outcome::EmptyGroups) {
                    requests = {{{}, 128}, {{}, 256}};
                }
                const bool allocated =
                    outcome == Outcome::Success || outcome == Outcome::PartialFailure || outcome == Outcome::Malformed;
                const bool rejected =
                    outcome == Outcome::Missing || outcome == Outcome::Disabled || outcome == Outcome::ReadOnly;
                RequestContext context("reentrant_batch");
                const auto calling_thread = std::this_thread::get_id();
                size_t callback_count = 0;
                const auto result = manager.CreateBatch(&context, "storage1", requests, [&]() {
                    ++callback_count;
                    EXPECT_EQ(calling_thread, std::this_thread::get_id());
                    EXPECT_EQ(allocated ? 1u : 0u, context.GetMetricsCollectorsVehicle().GetMetricsCollectors().size());
                    if (allocated) {
                        EXPECT_DOUBLE_EQ(2, backend->GetMetricsCollector()->get_data_storage_create_keys_qps_metrics());
                    }
                    const auto ec =
                        unregister ? manager.UnRegisterStorage("storage1") : manager.DisableStorage("storage1");
                    EXPECT_EQ(outcome == Outcome::Missing ? EC_NOENT : EC_OK, ec);
                });
                EXPECT_EQ(1u, callback_count);
                EXPECT_EQ(allocated ? 1u : 0u, backend->batch_call_count);
                EXPECT_EQ(unregister || outcome == Outcome::Missing,
                          manager.GetDataStorageBackend("storage1") == nullptr);
                if (outcome != Outcome::Missing) {
                    EXPECT_FALSE(backend->Available());
                }
                ASSERT_EQ(requests.size(), result.results.size());
                for (size_t group = 0; group < requests.size(); ++group) {
                    ASSERT_EQ(requests[group].keys.size(), result.results[group].size());
                    for (const auto &[ec, uri] : result.results[group]) {
                        EXPECT_EQ(rejected ? EC_NOENT : backend->batch_result.results[group][0].first, ec);
                        if (ec == EC_OK) {
                            EXPECT_EQ("storage1", uri.GetHostName());
                            EXPECT_EQ(backend->batch_result.results[group][0].second.GetPath(), uri.GetPath());
                        }
                    }
                }
                ASSERT_EQ(outcome == Outcome::Malformed ? 1u : 0u, result.cleanup_uris.size());
                if (!result.cleanup_uris.empty()) {
                    EXPECT_EQ("file://storage1/unmapped", result.cleanup_uris[0].ToUriString());
                }
            };
            // A callback invoked under the allocation lock deadlocks on these mutations.
            // Bound the regression in a subprocess so no blocked thread outlives its test data.
            ASSERT_EXIT(
                {
                    ::alarm(5);
                    check_case();
                    ::_exit(::testing::Test::HasFailure() ? 1 : 0);
                },
                ::testing::ExitedWithCode(0),
                "");
        }
    }
}

TEST_F(DataStorageManagerTest, TestCreateBatchKeepsStorageLockedUntilBackendWorkersFinish) {
    for (const bool unregister : {false, true}) {
        SCOPED_TRACE(unregister ? "UnRegisterStorage" : "DisableStorage");
        DataStorageManager manager(metrics_registry_);
        auto backend = std::make_shared<BlockingBatchBackend>(metrics_registry_);
        auto spec = std::make_shared<DummyStorageSpec>();
        spec->set_root_path("memory://");
        ASSERT_EQ(EC_OK,
                  backend->Open(StorageConfig(DataStorageType::DATA_STORAGE_TYPE_DUMMY, "storage1", spec), "open"));
        manager.storage_map_["storage1"] = backend;
        auto worker_started = backend->worker_started.get_future();
        size_t callback_count = 0;
        std::thread::id allocation_thread;
        std::thread::id callback_thread;
        bool callback_after_worker = false;
        auto allocation = std::async(std::launch::async, [&]() {
            allocation_thread = std::this_thread::get_id();
            RequestContext request_context("batch_trace");
            return manager.CreateBatch(&request_context, "storage1", {{{"first"}, 128}, {{"second"}, 256}}, [&]() {
                ++callback_count;
                callback_thread = std::this_thread::get_id();
                callback_after_worker = backend->worker_released;
            });
        });
        EXPECT_EQ(std::future_status::ready, worker_started.wait_for(std::chrono::seconds(5)));

        std::promise<void> mutation_started;
        auto mutation_started_future = mutation_started.get_future();
        auto mutation = std::async(std::launch::async, [&]() {
            mutation_started.set_value();
            return unregister ? manager.UnRegisterStorage("storage1") : manager.DisableStorage("storage1");
        });
        EXPECT_EQ(std::future_status::ready, mutation_started_future.wait_for(std::chrono::seconds(5)));
        EXPECT_EQ(std::future_status::timeout, mutation.wait_for(std::chrono::milliseconds(100)));

        backend->release_worker.set_value();
        EXPECT_EQ(std::future_status::ready, allocation.wait_for(std::chrono::seconds(5)));
        EXPECT_EQ(std::future_status::ready, mutation.wait_for(std::chrono::seconds(5)));
        const auto batch_result = allocation.get();
        EXPECT_EQ(EC_OK, mutation.get());
        const auto &results = batch_result.results;
        EXPECT_EQ(1u, callback_count);
        EXPECT_EQ(allocation_thread, callback_thread);
        EXPECT_TRUE(callback_after_worker);
        ASSERT_EQ(2u, results.size());
        for (const auto &result : results) {
            ASSERT_EQ(1u, result.size());
            EXPECT_EQ(EC_OK, result[0].first);
            EXPECT_EQ("storage1", result[0].second.GetHostName());
        }
        EXPECT_FALSE(backend->Available());
        EXPECT_EQ(unregister, manager.GetDataStorageBackend("storage1") == nullptr);
    }
}
