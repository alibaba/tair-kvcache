#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <utility>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/kv_meta_uri.h"
#include "kv_cache_manager/data_storage/nfs_backend.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;

namespace {

class FaultInjectingNfsBackend final : public NfsBackend {
public:
    enum class Mode {
        kError,
        kShortResult
    };

    FaultInjectingNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry, Mode mode)
        : NfsBackend(std::move(metrics_registry)), mode_(mode) {}

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        if (cb) {
            cb();
        }
        if (mode_ == Mode::kShortResult) {
            return std::vector<ErrorCode>(storage_uris.empty() ? 0 : storage_uris.size() - 1, EC_OK);
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_IO_ERROR);
    }

private:
    Mode mode_;
};

class DirectorySyncFailingNfsBackend final : public NfsBackend {
public:
    explicit DirectorySyncFailingNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::size_t sync_attempts() const noexcept { return sync_attempts_; }

protected:
    bool SyncKvMetaDeleteDirectory(const std::string &) const noexcept override {
        ++sync_attempts_;
        return false;
    }

private:
    mutable std::size_t sync_attempts_{0};
};

} // namespace

class NfsBackendTest : public TESTBASE {
public:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        test_root_ = GetPrivateTestRuntimeDataPath() + "nfs_root/";
        outside_root_ = GetPrivateTestRuntimeDataPath() + "nfs_outside/";
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(test_root_, ec);
        ec.clear();
        std::filesystem::remove_all(outside_root_, ec);
    }
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::string test_root_;
    std::string outside_root_;
};

// TestSimple参考版，使用Open(StorageConfig)初始化，其他测试用例都调整如下：
TEST_F(NfsBackendTest, TestSimple) {
    // 一个key是一个文件的形式
    {
        NfsBackend backend(metrics_registry_);
        std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
        spec->set_key_count_per_file(1);
        spec->set_root_path("/data/");
        StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
        ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
        std::vector<std::string> keys = {"key1", "key2", "key3", "key4", "key5"};
        auto results = backend.Create(keys, 128, "fake_trace_id_2", []() {});
        ASSERT_EQ(results.size(), keys.size());
        ASSERT_EQ("file:///data/key1?size=128", results[0].second.ToUriString());
        ASSERT_EQ("file:///data/key2?size=128", results[1].second.ToUriString());
        ASSERT_EQ("file:///data/key3?size=128", results[2].second.ToUriString());
        ASSERT_EQ("file:///data/key4?size=128", results[3].second.ToUriString());
        ASSERT_EQ("file:///data/key5?size=128", results[4].second.ToUriString());
    }
    // 多个key一个文件的形式
    {
        NfsBackend backend(metrics_registry_);
        std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
        spec->set_key_count_per_file(2);
        spec->set_root_path("/data/");
        StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
        ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_3"));
        std::vector<std::string> keys = {"key1", "key2", "key3", "key4", "key5"};
        auto results = backend.Create(keys, 128, "fake_trace_id_4", []() {});
        ASSERT_EQ(results.size(), keys.size());
        EXPECT_EQ("file:///data/key1_5a560a3d977cc6f2?blkid=0&size=128", results[0].second.ToUriString());
        EXPECT_EQ("file:///data/key1_5a560a3d977cc6f2?blkid=1&size=128", results[1].second.ToUriString());
        EXPECT_EQ("file:///data/key3_1184f2d3fc112241?blkid=0&size=128", results[2].second.ToUriString());
        EXPECT_EQ("file:///data/key3_1184f2d3fc112241?blkid=1&size=128", results[3].second.ToUriString());
        EXPECT_EQ("file:///data/key5?blkid=0&size=128", results[4].second.ToUriString());
    }
}

TEST_F(NfsBackendTest, TestGetTypeAndAvailableStatus) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(2);
    spec->set_root_path("/data/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_FALSE(backend.Available());
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    ASSERT_TRUE(backend.Available());
    ASSERT_EQ(backend.Close(), EC_OK);
    ASSERT_FALSE(backend.Available());
}

TEST_F(NfsBackendTest, TestCreateWithBatchingAndCallbackInvocation) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(2);
    spec->set_root_path("/data/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    std::vector<std::string> keys = {"key1", "key2", "key3", "key4", "key5"};
    bool callback_called = false;
    auto callback = [&callback_called]() { callback_called = true; };
    auto results = backend.Create(keys, 100, "fake_trace_id_2", callback);
    ASSERT_TRUE(callback_called);
    ASSERT_EQ(results.size(), keys.size());
    EXPECT_EQ(results[0].second.ToUriString(), "file:///data/key1_5a560a3d977cc6f2?blkid=0&size=100");
    EXPECT_EQ(results[1].second.ToUriString(), "file:///data/key1_5a560a3d977cc6f2?blkid=1&size=100");
    EXPECT_EQ(results[2].second.ToUriString(), "file:///data/key3_1184f2d3fc112241?blkid=0&size=100");
    EXPECT_EQ(results[3].second.ToUriString(), "file:///data/key3_1184f2d3fc112241?blkid=1&size=100");
    EXPECT_EQ(results[4].second.ToUriString(), "file:///data/key5?blkid=0&size=100");
    for (size_t i = 0; i < results.size(); ++i) {
        ASSERT_EQ(results[i].first, EC_OK);
    }
}

TEST_F(NfsBackendTest, TestCreateWithBatchSizeOneAndEmptyKeys) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(1);
    spec->set_root_path("/data/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    bool callback_called = false;
    auto cb = [&callback_called]() { callback_called = true; };
    auto results_empty = backend.Create({}, 100, "fake_trace_id_2", cb);
    ASSERT_TRUE(callback_called);
    ASSERT_TRUE(results_empty.empty());

    std::vector<std::string> keys = {"a", "b"};
    callback_called = false;
    auto results = backend.Create(keys, 100, "fake_trace_id_3", [&callback_called]() { callback_called = true; });
    ASSERT_TRUE(callback_called);
    ASSERT_EQ(results.size(), keys.size());
    ASSERT_EQ(results[0].second.ToUriString(), "file:///data/a?size=100");
    ASSERT_EQ(results[1].second.ToUriString(), "file:///data/b?size=100");
}

TEST_F(NfsBackendTest, TestDeleteReturnsOkAndSameSize) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(1);
    spec->set_root_path("/data/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    std::vector<DataStorageUri> uris(3);
    auto res = backend.Delete(uris, "fake_trace_id_2", []() {});
    ASSERT_EQ(res.size(), uris.size());
    for (auto code : res) {
        ASSERT_EQ(code, EC_OK);
    }
}

TEST_F(NfsBackendTest, TestKvMetaDeleteRemovesAndConfirmsExactFile) {
    NfsBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'a');
    const std::filesystem::path object_path = std::filesystem::path(test_root_) / object_key;
    ASSERT_TRUE(std::filesystem::create_directories(object_path.parent_path()));
    {
        std::ofstream output(object_path);
        ASSERT_TRUE(output.good());
        output << "embedding";
    }

    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath(object_path.string());
    uri.SetParam("size", "9");

    // Keep the historical fixed-block path unchanged: only the KVMeta side
    // capability may remove singleton objects.
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK}), backend.Delete({uri}, "legacy_delete", nullptr));
    EXPECT_TRUE(std::filesystem::exists(object_path));

    bool callback_called = false;
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK}),
              backend.DeleteForKvMeta({uri}, "delete", [&]() { callback_called = true; }));
    EXPECT_TRUE(callback_called);
    EXPECT_FALSE(std::filesystem::exists(object_path));
    EXPECT_FALSE(std::filesystem::exists(object_path.parent_path()));
    EXPECT_FALSE(std::filesystem::exists(object_path.parent_path().parent_path()));
    EXPECT_TRUE(std::filesystem::exists(object_path.parent_path().parent_path().parent_path()));

    // Exact delete is idempotent: an already-absent generation is terminal.
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK}), backend.DeleteForKvMeta({uri}, "delete_again", nullptr));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteRejectsFileOutsideConfiguredNamespace) {
    NfsBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'b');
    const std::filesystem::path outside_path = std::filesystem::path(outside_root_) / object_key;
    ASSERT_TRUE(std::filesystem::create_directories(outside_path.parent_path()));
    {
        std::ofstream output(outside_path);
        ASSERT_TRUE(output.good());
        output << "do-not-delete";
    }

    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath(outside_path.string());
    uri.SetParam("size", "13");
    bool callback_called = false;
    EXPECT_EQ((std::vector<ErrorCode>{EC_CORRUPTION}),
              backend.DeleteForKvMeta({uri}, "reject", [&]() { callback_called = true; }));
    EXPECT_TRUE(callback_called);
    EXPECT_TRUE(std::filesystem::exists(outside_path));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteDoesNotRemoveFileWhenLegacyHookFailsOrReturnsShortResult) {
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'c');
    const std::filesystem::path object_path = std::filesystem::path(test_root_) / object_key;
    ASSERT_TRUE(std::filesystem::create_directories(object_path.parent_path()));
    {
        std::ofstream output(object_path);
        ASSERT_TRUE(output.good());
        output << "preserve-on-uncertainty";
    }

    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath(object_path.string());
    uri.SetParam("size", "23");

    FaultInjectingNfsBackend failing(metrics_registry_, FaultInjectingNfsBackend::Mode::kError);
    ASSERT_EQ(EC_OK, failing.Open(storage_config, "open_failing"));
    EXPECT_EQ((std::vector<ErrorCode>{EC_IO_ERROR}), failing.DeleteForKvMeta({uri}, "failed_delete", nullptr));
    EXPECT_TRUE(std::filesystem::exists(object_path));

    FaultInjectingNfsBackend short_result(metrics_registry_, FaultInjectingNfsBackend::Mode::kShortResult);
    ASSERT_EQ(EC_OK, short_result.Open(storage_config, "open_short"));
    EXPECT_EQ((std::vector<ErrorCode>{EC_MISMATCH}),
              short_result.DeleteForKvMeta({uri}, "short_delete", nullptr));
    EXPECT_TRUE(std::filesystem::exists(object_path));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteRetainsLedgerWhenDirectoryDurabilityIsUncertain) {
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'f');
    const std::filesystem::path object_path = std::filesystem::path(test_root_) / object_key;
    ASSERT_TRUE(std::filesystem::create_directories(object_path.parent_path()));
    {
        std::ofstream output(object_path);
        ASSERT_TRUE(output.good());
        output << "deleted-but-not-yet-durable";
    }

    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath(object_path.string());
    uri.SetParam("size", "27");

    DirectorySyncFailingNfsBackend failing(metrics_registry_);
    ASSERT_EQ(EC_OK, failing.Open(storage_config, "open_failing"));
    bool callback_called = false;
    EXPECT_EQ((std::vector<ErrorCode>{EC_IO_ERROR}),
              failing.DeleteForKvMeta({uri}, "uncertain_sync", [&]() { callback_called = true; }));
    EXPECT_TRUE(callback_called);
    EXPECT_EQ(1u, failing.sync_attempts());
    EXPECT_FALSE(std::filesystem::exists(object_path));
    EXPECT_TRUE(std::filesystem::exists(object_path.parent_path()));

    // The durable tombstone makes retrying an already-unlinked generation
    // safe. A later successful barrier may confirm absence and prune parents.
    NfsBackend retry(metrics_registry_);
    ASSERT_EQ(EC_OK, retry.Open(storage_config, "open_retry"));
    EXPECT_EQ((std::vector<ErrorCode>{EC_OK}), retry.DeleteForKvMeta({uri}, "retry", nullptr));
    EXPECT_FALSE(std::filesystem::exists(object_path.parent_path()));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteConfirmsAllocationThatWasNeverMaterialized) {
    NfsBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));
    ASSERT_TRUE(std::filesystem::create_directories(test_root_));

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, '0');
    const std::filesystem::path object_path = std::filesystem::path(test_root_) / object_key;
    ASSERT_FALSE(std::filesystem::exists(object_path.parent_path().parent_path().parent_path()));

    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath(object_path.string());
    uri.SetParam("size", "64");

    EXPECT_EQ((std::vector<ErrorCode>{EC_OK}), backend.DeleteForKvMeta({uri}, "never_materialized", nullptr));
    EXPECT_FALSE(std::filesystem::exists(object_path));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteDoesNotTreatAMissingConfiguredRootAsRemoteAbsence) {
    NfsBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    const StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));
    ASSERT_FALSE(std::filesystem::exists(test_root_));

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'm');
    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath((std::filesystem::path(test_root_) / object_key).string());
    uri.SetParam("size", "64");

    // An unmounted NFS root can look exactly like an absent local path. Keep
    // the durable tombstone instead of releasing quota on that observation.
    EXPECT_EQ((std::vector<ErrorCode>{EC_IO_ERROR}),
              backend.DeleteForKvMeta({uri}, "missing_configured_root", nullptr));
    EXPECT_FALSE(std::filesystem::exists(test_root_));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteReportsFilesystemFailureWithoutClaimingAbsence) {
    NfsBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'd');
    const std::filesystem::path object_path = std::filesystem::path(test_root_) / object_key;
    ASSERT_TRUE(std::filesystem::create_directories(object_path / "child"));

    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath(object_path.string());
    uri.SetParam("size", "1");
    EXPECT_EQ((std::vector<ErrorCode>{EC_IO_ERROR}),
              backend.DeleteForKvMeta({uri}, "non_empty_directory", nullptr));
    EXPECT_TRUE(std::filesystem::exists(object_path));
}

TEST_F(NfsBackendTest, TestKvMetaDeleteNeverTreatsAnEmptyDirectoryAsAnObjectGeneration) {
    NfsBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_key_count_per_file(1);
    spec->set_root_path(test_root_);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test_nfs", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "open"));

    const std::string object_key = "kvmeta/1/2/" + std::string(kKvMetaObjectNonceBytes, 'e');
    const std::filesystem::path object_path = std::filesystem::path(test_root_) / object_key;
    ASSERT_TRUE(std::filesystem::create_directories(object_path));

    DataStorageUri uri;
    uri.SetProtocol(ToString(DataStorageType::DATA_STORAGE_TYPE_NFS));
    uri.SetHostName("test_nfs");
    uri.SetPath(object_path.string());
    uri.SetParam("size", "1");
    EXPECT_EQ((std::vector<ErrorCode>{EC_IO_ERROR}), backend.DeleteForKvMeta({uri}, "empty_directory", nullptr));
    EXPECT_TRUE(std::filesystem::is_directory(object_path));
}

TEST_F(NfsBackendTest, TestExistReturnsTrues) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(1);
    spec->set_root_path("/data/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    // TODO(qisa.cb) 没实现
    std::vector<DataStorageUri> uris(5);
    auto res = backend.Exist(uris);
    ASSERT_EQ(res.size(), uris.size());
    for (bool flag : res) {
        ASSERT_TRUE(flag);
    }
}

TEST_F(NfsBackendTest, TestLockAndUnLockReturnOk) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(1);
    spec->set_root_path("/data/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    std::vector<DataStorageUri> uris(4);
    auto lock_res = backend.Lock(uris);
    auto unlock_res = backend.UnLock(uris);
    ASSERT_EQ(lock_res.size(), uris.size());
    ASSERT_EQ(unlock_res.size(), uris.size());
    for (auto code : lock_res) {
        ASSERT_EQ(code, EC_OK);
    }
    for (auto code : unlock_res) {
        ASSERT_EQ(code, EC_OK);
    }
}

TEST_F(NfsBackendTest, TestCreateHandlesInvalidBatchSize) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(0); // 0 应该被内部处理为 1
    spec->set_root_path("/root/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    std::vector<std::string> keys = {"k1", "k2"};
    bool cb_called = false;
    auto results = backend.Create(keys, 50, "fake_trace_id_2", [&cb_called]() { cb_called = true; });
    ASSERT_TRUE(cb_called);
    ASSERT_EQ(results.size(), keys.size());
    ASSERT_EQ(results[0].second.ToUriString(), "file:///root/k1?size=50");
    ASSERT_EQ(results[1].second.ToUriString(), "file:///root/k2?size=50");
}

TEST_F(NfsBackendTest, TestCreateSingleKeyBatch) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(10);
    spec->set_root_path("/root/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    std::vector<std::string> keys = {"singlekey"};
    bool cb_called = false;
    auto results = backend.Create(keys, 10, "fake_trace_id_2", [&cb_called]() { cb_called = true; });
    ASSERT_TRUE(cb_called);
    ASSERT_EQ(results.size(), keys.size());
    ASSERT_EQ(results[0].second.ToUriString(), "file:///root/singlekey?blkid=0&size=10");
}
