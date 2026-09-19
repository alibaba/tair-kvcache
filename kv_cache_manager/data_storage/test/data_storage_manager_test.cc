#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/nfs_backend.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;

class DataStorageManagerTest : public TESTBASE {
public:
    void SetUp() override { metrics_registry_ = std::make_shared<MetricsRegistry>(); }
    void TearDown() override {}
    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

class DedicatedKvMetaCreateBackend final : public DataStorageBackend,
                                           public KvMetaDataStorageBackendExtension {
public:
    explicit DedicatedKvMetaCreateBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : DataStorageBackend(std::move(metrics_registry)) {}

    DataStorageType GetType() override { return DataStorageType::DATA_STORAGE_TYPE_DUMMY; }
    bool Available() override { return IsAvailable(); }
    double GetStorageUsageRatio(const std::string &) const override { return 0.0; }
    ErrorCode DoOpen(const StorageConfig &, const std::string &) override {
        SetAvailable(true);
        return EC_OK;
    }
    ErrorCode Close() override {
        SetAvailable(false);
        return EC_OK;
    }

    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &,
                                                             std::function<void()> cb) override {
        ++legacy_create_calls;
        if (cb) cb();
        return MakeResult(keys.size(), size_per_key, "/legacy");
    }
    bool HasDedicatedKvMetaCreate() const noexcept override { return true; }
    bool RequiresKvMetaCreateCommit() const noexcept override { return true; }
    std::vector<std::pair<ErrorCode, DataStorageUri>> CreateForKvMeta(
        const std::vector<std::string> &keys,
        size_t size_per_key,
        const std::string &,
        std::function<void()> cb) override {
        ++kv_meta_create_calls;
        if (cb) cb();
        return MakeResult(keys.size(), size_per_key, "/kvmeta");
    }
    std::vector<ErrorCode> CommitKvMetaCreate(const std::vector<std::string> &allocation_keys,
                                               const std::string &) override {
        committed_keys = allocation_keys;
        return std::vector<ErrorCode>(allocation_keys.size(), EC_OK);
    }

    std::vector<ErrorCode> Delete(const std::vector<DataStorageUri> &uris,
                                  const std::string &,
                                  std::function<void()> cb) override {
        if (cb) cb();
        return std::vector<ErrorCode>(uris.size(), EC_OK);
    }
    std::vector<ErrorCode> DeleteAndConfirmAbsent(const std::vector<DataStorageUri> &uris,
                                                  const std::string &,
                                                  std::function<void()> cb) override {
        return Delete(uris, {}, std::move(cb));
    }
    std::int64_t GetFailedWriteCleanupGraceSeconds() const noexcept override { return 0; }
    std::vector<bool> Exist(const std::vector<DataStorageUri> &uris) override {
        return std::vector<bool>(uris.size(), true);
    }
    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &uris) override {
        return std::vector<ErrorCode>(uris.size(), EC_OK);
    }
    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &uris) override {
        return std::vector<ErrorCode>(uris.size(), EC_OK);
    }

    std::size_t legacy_create_calls = 0;
    std::size_t kv_meta_create_calls = 0;
    std::vector<std::string> committed_keys;

private:
    static std::vector<std::pair<ErrorCode, DataStorageUri>>
    MakeResult(std::size_t count, std::size_t size, const std::string &path) {
        DataStorageUri uri;
        uri.SetProtocol("file");
        uri.SetPath(path);
        uri.SetParam("size", std::to_string(size));
        return std::vector<std::pair<ErrorCode, DataStorageUri>>(count, {EC_OK, uri});
    }
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

TEST_F(DataStorageManagerTest, DedicatedKvMetaCreateDoesNotChangeLegacyCreateRouting) {
    DataStorageManager data_storage_manager(metrics_registry_);
    RequestContext request_context("test");
    auto backend = std::make_shared<DedicatedKvMetaCreateBackend>(metrics_registry_);
    auto spec = std::make_shared<DummyStorageSpec>();
    spec->set_root_path("memory://");
    StorageConfig config(DataStorageType::DATA_STORAGE_TYPE_DUMMY, "isolated", spec);
    ASSERT_EQ(EC_OK, backend->Open(config, request_context.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(data_storage_manager.rw_lock_);
        data_storage_manager.storage_map_["isolated"] = backend;
    }

    bool legacy_callback = false;
    const auto legacy = data_storage_manager.Create(
        &request_context, "isolated", {"key"}, 17, [&]() { legacy_callback = true; });
    ASSERT_EQ(legacy.size(), 1u);
    EXPECT_EQ(legacy[0].first, EC_OK);
    EXPECT_EQ(legacy[0].second.GetPath(), "/legacy");
    EXPECT_EQ(legacy[0].second.GetHostName(), "isolated");
    EXPECT_TRUE(legacy_callback);
    EXPECT_EQ(backend->legacy_create_calls, 1u);
    EXPECT_EQ(backend->kv_meta_create_calls, 0u);

    bool kv_meta_callback = false;
    const auto kv_meta = data_storage_manager.CreateForKvMeta(
        &request_context, "isolated", {"key"}, 29, [&]() { kv_meta_callback = true; });
    ASSERT_EQ(kv_meta.size(), 1u);
    EXPECT_EQ(kv_meta[0].first, EC_OK);
    EXPECT_EQ(kv_meta[0].second.GetPath(), "/kvmeta");
    EXPECT_EQ(kv_meta[0].second.GetHostName(), "isolated");
    EXPECT_TRUE(kv_meta_callback);
    EXPECT_EQ(backend->legacy_create_calls, 1u);
    EXPECT_EQ(backend->kv_meta_create_calls, 1u);

    const auto commit = data_storage_manager.CommitKvMetaCreate(
        &request_context, "isolated", {"allocation-a", "allocation-b"});
    EXPECT_EQ(commit, (std::vector<ErrorCode>{EC_OK, EC_OK}));
    EXPECT_EQ(backend->committed_keys,
              (std::vector<std::string>{"allocation-a", "allocation-b"}));
}

TEST_F(DataStorageManagerTest, KvMetaCommitFailsClosedForMissingOrOnePhaseBackend) {
    DataStorageManager data_storage_manager(metrics_registry_);
    RequestContext request_context("test");
    EXPECT_EQ(data_storage_manager.CommitKvMetaCreate(
                  &request_context, "missing", {"allocation"}),
              (std::vector<ErrorCode>{EC_NOENT}));

    auto backend = std::make_shared<NfsBackend>(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_root_path("/tmp/");
    spec->set_key_count_per_file(1);
    StorageConfig config(DataStorageType::DATA_STORAGE_TYPE_NFS, "one_phase", spec);
    ASSERT_EQ(backend->Open(config, request_context.trace_id()), EC_OK);
    {
        std::unique_lock<std::shared_mutex> lock(data_storage_manager.rw_lock_);
        data_storage_manager.storage_map_["one_phase"] = backend;
    }
    EXPECT_EQ(data_storage_manager.CommitKvMetaCreate(
                  &request_context, "one_phase", {"allocation"}),
              (std::vector<ErrorCode>{EC_UNIMPLEMENTED}));
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
