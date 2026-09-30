#include <gtest/gtest.h>
#include <memory>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/nfs_backend.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;

class NfsBackendTest : public TESTBASE {
public:
    void SetUp() override { metrics_registry_ = std::make_shared<MetricsRegistry>(); }
    void TearDown() override {}
    std::shared_ptr<MetricsRegistry> metrics_registry_;
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
