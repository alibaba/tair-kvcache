#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <memory>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/dummy_backend.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;

namespace {
struct AsyncCopyLifetimeState {
    AsyncCopyCompletion completion;
    int close_count = 0;
    int destroy_count = 0;
    int copy_count = 0;
};

class LifetimeAsyncBackend : public DummyBackend {
public:
    LifetimeAsyncBackend(const std::shared_ptr<MetricsRegistry> &metrics, std::shared_ptr<AsyncCopyLifetimeState> state)
        : DummyBackend(metrics), state_(std::move(state)) {}
    ~LifetimeAsyncBackend() override { ++state_->destroy_count; }
    bool Available() override { return true; }
    bool SupportsAsyncCopy() const override { return true; }
    ErrorCode Close() override {
        ++state_->close_count;
        return EC_OK;
    }
    AsyncCopySubmitResult CopyAsync(const std::vector<DataStorageUri> &,
                                    const std::vector<DataStorageUri> &,
                                    const std::string &operation_id,
                                    const std::string &,
                                    const AsyncCopyOptions &,
                                    AsyncCopyCompletion completion) override {
        ++state_->copy_count;
        state_->completion = std::move(completion);
        AsyncCopySubmitResult result;
        result.status = EC_OK;
        result.accepted = true;
        result.operation_id = operation_id;
        return result;
    }

private:
    std::shared_ptr<AsyncCopyLifetimeState> state_;
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

TEST_F(DataStorageManagerTest, TestAsyncCopyValidationNeedsOnlyFinalCompletion) {
    DataStorageManager manager(metrics_registry_);
    RequestContext context("async-copy");
    const std::vector<DataStorageUri> source{DataStorageUri("file://missing/src?size=128")};
    const std::vector<DataStorageUri> target{DataStorageUri("file://missing/dst?size=128")};
    bool completed = false;
    auto completion = [&](AsyncCopyBatchResult) { completed = true; };

    auto result = manager.CopyAsync(&context, "missing", {}, target, "op", AsyncCopyOptions{}, completion);
    EXPECT_FALSE(result.accepted);
    EXPECT_EQ(EC_BADARGS, result.status);
    result = manager.CopyAsync(&context, "missing", source, target, "op", AsyncCopyOptions{}, {});
    EXPECT_FALSE(result.accepted);
    EXPECT_EQ(EC_BADARGS, result.status);
    result = manager.CopyAsync(&context, "missing", source, target, "op", AsyncCopyOptions{}, completion);
    EXPECT_FALSE(result.accepted);
    EXPECT_EQ(EC_NOENT, result.status);
    EXPECT_FALSE(completed); // Rejection never transfers callback ownership.
}

TEST_F(DataStorageManagerTest, TestAsyncCompletionKeepsBackendAliveAfterUnregister) {
    DataStorageManager manager(metrics_registry_);
    auto state = std::make_shared<AsyncCopyLifetimeState>();
    auto backend = std::make_shared<LifetimeAsyncBackend>(metrics_registry_, state);
    manager.storage_map_["async_source"] = backend;
    const std::weak_ptr<DataStorageBackend> old_backend = backend;
    RequestContext context("async_lifetime");
    const std::vector<DataStorageUri> sources{DataStorageUri("dummy://async_source/source?size=128")};
    const std::vector<DataStorageUri> targets{DataStorageUri("dummy://target/destination?size=128")};
    int completion_count = 0;
    bool alive_during_completion = false;
    const auto submitted = manager.CopyAsync(
        &context, "async_source", sources, targets, "original-operation", {}, [&](AsyncCopyBatchResult result) {
            ++completion_count;
            alive_during_completion = !old_backend.expired();
            EXPECT_EQ(EC_UNKNOWN, result.status);
        });
    ASSERT_TRUE(submitted.accepted);
    EXPECT_EQ(1, state->copy_count);
    ASSERT_TRUE(state->completion);
    auto completion = std::move(state->completion);
    EXPECT_EQ(0, completion_count);

    // Completion is deliberately withheld. Close/UnRegister must finish after
    // local handoff without waiting for the remote operation.
    auto unregister = std::async(std::launch::async, [&] { return manager.UnRegisterStorage("async_source"); });
    const auto unregister_status = unregister.wait_for(std::chrono::seconds(1));
    if (unregister_status != std::future_status::ready) {
        // Release a regressed completion-held lock before joining the test thread.
        completion(AsyncCopyBatchResult{});
        completion = {};
    }
    ASSERT_EQ(EC_OK, unregister.get());
    ASSERT_EQ(std::future_status::ready, unregister_status);
    EXPECT_EQ(1, state->close_count);
    EXPECT_EQ(0, completion_count);

    // Remove every owner except the manager's callback lease. This checks
    // object lifetime, not whether Close proves remote allocation drain.
    backend.reset();
    EXPECT_FALSE(old_backend.expired());
    EXPECT_EQ(0, state->destroy_count);
    completion(AsyncCopyBatchResult{});
    EXPECT_EQ(1, completion_count);
    EXPECT_TRUE(alive_during_completion);
    completion = {};
    EXPECT_TRUE(old_backend.expired());
    EXPECT_EQ(1, state->destroy_count);
}

TEST_F(DataStorageManagerTest, TestAsyncCopyUnknownCannotBePublishedAsSuccess) {
    AsyncCopyBatchResult result;
    result.status = EC_OK; // Status alone is not authoritative.
    result.items.resize(1);
    result.items[0].outcome = AsyncCopyOutcome::kSuccess;
    EXPECT_FALSE(result.AllSucceeded());
    result.items[0].terminal = true;
    EXPECT_FALSE(result.AllSucceeded());
    result.items[0].safe_to_reuse_dst = true;
    EXPECT_TRUE(result.AllSucceeded());
    result.items[0].outcome = AsyncCopyOutcome::kUnknown;
    EXPECT_FALSE(result.AllSucceeded());
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
