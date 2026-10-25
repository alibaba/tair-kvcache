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
    ASSERT_EQ(1u, disabled_create.size());
    EXPECT_EQ(EC_NOENT, disabled_create[0].first);

    // enable storage
    ASSERT_EQ(EC_OK, data_storage_manager.EnableStorage("storage1"));
    EXPECT_TRUE(data_storage_backend->Available());
    ASSERT_EQ(EC_NOENT, data_storage_manager.EnableStorage("storage2"));

    // create exist delete
    DataStorageUri storage_uri1("file://storage1/data/key1?size=128");
    // ASSERT_FALSE(data_storage_manager.Exist("storage1", {storage_uri1})[0]);
    RequestContext requesst_context("test");
    auto uris = data_storage_manager.Create(&requesst_context, "storage1", {"key1"}, 128, []() {});
    ASSERT_EQ(1, uris.size());
    ASSERT_EQ(EC_OK, uris[0].first);
    ASSERT_EQ(storage_uri1.ToUriString(), uris[0].second.ToUriString());
    // unique name not exist
    uris = data_storage_manager.Create(&requesst_context, "storage2", {"key1"}, 128, []() {});
    ASSERT_EQ(0, uris.size());

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
    size_t callbacks = 0;
    const auto results = backend.CreateBatch(requests, "batch_trace", [&] { ++callbacks; });

    EXPECT_EQ(1u, callbacks);
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
    EXPECT_TRUE(backend.CreateBatch({}, "empty", [&] { ++callbacks; }).empty());
    EXPECT_EQ(2u, callbacks);
    EXPECT_EQ(3u, backend.requests.size());
}

TEST_F(DataStorageManagerTest, TestDefaultCreateBatchStopsAndPreservesFailureResponses) {
    // Partial failure, short response and excess allocations all stop later
    // requests while preserving every returned URI for the caller's rollback.
    for (size_t returned_count : {1u, 2u, 3u}) {
        SCOPED_TRACE(returned_count);
        RecordingCreateBackend backend(metrics_registry_);
        backend.scripted_results = {
            {{EC_OK, DataStorageUri("file://backend/earlier")}},
            DataStorageCreateResult(returned_count, {EC_OK, DataStorageUri("file://backend/allocated")})};
        if (returned_count == 2) {
            backend.scripted_results[1][1] = {EC_IO_ERROR, DataStorageUri{}};
        }
        size_t callbacks = 0;
        const auto results = backend.CreateBatch(
            {{{"earlier"}, 64}, {{"first", "second"}, 128}, {{"skipped"}, 256}}, "batch", [&] { ++callbacks; });
        EXPECT_EQ(1u, callbacks);
        EXPECT_EQ(2u, backend.requests.size());
        ASSERT_EQ(3u, results.size());
        ASSERT_EQ(1u, results[0].size());
        EXPECT_EQ(EC_OK, results[0][0].first);
        EXPECT_EQ("/earlier", results[0][0].second.GetPath());
        ASSERT_EQ(returned_count, results[1].size());
        for (size_t index = 0; index < returned_count; ++index) {
            EXPECT_EQ(backend.scripted_results[1][index].first, results[1][index].first);
            EXPECT_EQ(backend.scripted_results[1][index].second.ToUriString(), results[1][index].second.ToUriString());
        }
        EXPECT_TRUE(results[2].empty());
    }
}

TEST_F(DataStorageManagerTest, TestCreateBatchForwardsRequestsAndAnnotatesSuccessfulUris) {
    DataStorageManager manager(metrics_registry_);
    auto backend = std::make_shared<RecordingBatchBackend>(metrics_registry_);
    auto spec = std::make_shared<DummyStorageSpec>();
    spec->set_root_path("memory://");
    ASSERT_EQ(EC_OK, backend->Open(StorageConfig(DataStorageType::DATA_STORAGE_TYPE_DUMMY, "storage1", spec), "open"));
    manager.storage_map_["storage1"] = backend;
    backend->batch_result = {
        {{EC_OK, DataStorageUri("file://backend/first")}, {EC_OK, DataStorageUri("file://backend/extra")}},
        {},
        {{EC_IO_ERROR, DataStorageUri("file://failed/second")}}};
    const std::vector<DataStorageCreateRequest> requests = {{{"first"}, 128}, {{}, 256}, {{"second"}, 256}};
    RequestContext context("batch__kvcm_need_span_tracer");
    const auto caller = std::this_thread::get_id();
    size_t callbacks = 0;
    const auto results = manager.CreateBatch(&context, "storage1", requests, [&] {
        ++callbacks;
        EXPECT_EQ(caller, std::this_thread::get_id());
    });

    EXPECT_EQ(1u, callbacks);
    EXPECT_EQ(1u, backend->batch_call_count);
    EXPECT_TRUE(backend->requests.empty());
    EXPECT_EQ(context.trace_id(), backend->batch_trace_id);
    ASSERT_EQ(requests.size(), backend->batch_requests.size());
    ASSERT_EQ(backend->batch_result.size(), results.size());
    for (size_t group = 0; group < requests.size(); ++group) {
        EXPECT_EQ(requests[group].keys, backend->batch_requests[group].keys);
        EXPECT_EQ(requests[group].size_per_key, backend->batch_requests[group].size_per_key);
        ASSERT_EQ(backend->batch_result[group].size(), results[group].size());
        for (size_t key = 0; key < results[group].size(); ++key) {
            const auto &[ec, uri] = results[group][key];
            EXPECT_EQ(backend->batch_result[group][key].first, ec);
            EXPECT_EQ(backend->batch_result[group][key].second.GetPath(), uri.GetPath());
            EXPECT_EQ(ec == EC_OK ? "storage1" : "failed", uri.GetHostName());
        }
    }
    const auto collector = backend->GetMetricsCollector();
    ASSERT_NE(nullptr, collector);
    EXPECT_DOUBLE_EQ(2, collector->get_data_storage_create_keys_qps_metrics());
    EXPECT_GE(collector->get_data_storage_create_time_us_metrics(), 0);
    const auto collectors = context.GetMetricsCollectorsVehicle().GetMetricsCollectors();
    ASSERT_EQ(1u, collectors.size());
    EXPECT_EQ(collector, collectors[0]);
    EXPECT_NE(std::string::npos, context.EndAndGetSpanTracerDebugStr().find("CreateBatch"));
}

TEST_F(DataStorageManagerTest, TestCreateBatchCallbackCanReenterStorageMutations) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    enum class Outcome {
        Success,
        PartialFailure,
        Missing,
        Disabled,
        EmptyRequests
    };
    for (const auto outcome :
         {Outcome::Success, Outcome::PartialFailure, Outcome::Missing, Outcome::Disabled, Outcome::EmptyRequests}) {
        SCOPED_TRACE(static_cast<int>(outcome));
        const auto check_case = [&] {
            DataStorageManager manager(metrics_registry_);
            auto backend = std::make_shared<RecordingBatchBackend>(metrics_registry_);
            auto spec = std::make_shared<DummyStorageSpec>();
            spec->set_root_path("memory://");
            ASSERT_EQ(EC_OK,
                      backend->Open(StorageConfig(DataStorageType::DATA_STORAGE_TYPE_DUMMY, "storage1", spec), "open"));
            if (outcome != Outcome::Missing) {
                manager.storage_map_["storage1"] = backend;
            }
            if (outcome == Outcome::Disabled) {
                ASSERT_EQ(EC_OK, manager.DisableStorage("storage1"));
            }
            backend->batch_result = {{{EC_OK, DataStorageUri("file://backend/first")}},
                                     {{EC_IO_ERROR, DataStorageUri{}}}};
            if (outcome == Outcome::Success) {
                backend->batch_result[1][0] = {EC_OK, DataStorageUri("file://backend/second")};
            }
            std::vector<DataStorageCreateRequest> requests = {{{"first"}, 128}, {{"second"}, 256}};
            if (outcome == Outcome::EmptyRequests) {
                requests = {{{}, 128}, {{}, 256}};
            }
            RequestContext context("reentrant_batch");
            size_t callbacks = 0;
            const auto results = manager.CreateBatch(&context, "storage1", requests, [&] {
                ++callbacks;
                EXPECT_EQ(outcome == Outcome::Missing ? EC_NOENT : EC_OK, manager.UnRegisterStorage("storage1"));
            });
            EXPECT_EQ(1u, callbacks);
            EXPECT_EQ(outcome == Outcome::Success || outcome == Outcome::PartialFailure ? 1u : 0u,
                      backend->batch_call_count);
            EXPECT_EQ(nullptr, manager.GetDataStorageBackend("storage1"));
            ASSERT_EQ(requests.size(), results.size());
            for (size_t group = 0; group < requests.size(); ++group) {
                ASSERT_EQ(requests[group].keys.size(), results[group].size());
                for (const auto &[ec, uri] : results[group]) {
                    const auto expected_ec = outcome == Outcome::Missing || outcome == Outcome::Disabled
                                                 ? EC_NOENT
                                                 : backend->batch_result[group][0].first;
                    EXPECT_EQ(expected_ec, ec);
                }
            }
        };
        // Bound a reentrant-lock regression without leaving a blocked test thread.
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
        const auto results = allocation.get();
        EXPECT_EQ(EC_OK, mutation.get());
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
