#include <filesystem>
#include <gtest/gtest.h>
#include <memory>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/hf3fs_backend.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;

class Hf3fsBackendTest : public TESTBASE {
public:
    void SetUp() override {
        auto root_path = GetPrivateTestRuntimeDataPath();
        std::filesystem::path p(root_path);
        auto parent = p.parent_path().parent_path();
        root_dir_ = p.lexically_relative(parent);
        mountpoint_ = parent.string();
        metrics_registry_ = std::make_shared<MetricsRegistry>();
    }
    void TearDown() override {}

private:
    std::string BuildUri(const std::string &suffix) { return "hf3fs://" + mountpoint_ + "/" + root_dir_ + suffix; }

    std::shared_ptr<ThreeFSStorageSpec> GetDefaultStorageSpec() {
        std::shared_ptr<ThreeFSStorageSpec> spec(new ThreeFSStorageSpec);
        spec->set_cluster_name(cluster_name_);
        spec->set_mountpoint(mountpoint_);
        spec->set_root_dir(root_dir_);
        return spec;
    }

private:
    std::string cluster_name_ = "test_cluster";
    std::string mountpoint_;
    std::string root_dir_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

// TestSimple参考版，使用Open(StorageConfig)初始化，其他测试用例都调整如下：
TEST_F(Hf3fsBackendTest, TestSimple) {
    // 一个key是一个文件的形式
    {
        Hf3fsBackend backend(metrics_registry_);
        auto spec = GetDefaultStorageSpec();
        spec->set_key_count_per_file(1);
        StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_HF3FS, "test", spec);
        storage_config.set_check_storage_available_when_open(true);
        ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
        std::vector<std::string> keys = {"key1", "key2", "key3", "key4", "key5"};
        auto results = backend.Create(keys, 128, "fake_trace_id_2", []() {});
        ASSERT_EQ(results.size(), keys.size());
        ASSERT_EQ(BuildUri("key1?size=128"), results[0].second.ToUriString());
        ASSERT_EQ(BuildUri("key2?size=128"), results[1].second.ToUriString());
        ASSERT_EQ(BuildUri("key3?size=128"), results[2].second.ToUriString());
        ASSERT_EQ(BuildUri("key4?size=128"), results[3].second.ToUriString());
        ASSERT_EQ(BuildUri("key5?size=128"), results[4].second.ToUriString());
    }
    // 多个key一个文件的形式
    {
        Hf3fsBackend backend(metrics_registry_);
        auto spec = GetDefaultStorageSpec();
        spec->set_key_count_per_file(2);
        StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_HF3FS, "test", spec);
        storage_config.set_check_storage_available_when_open(true);
        ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_3"));
        std::vector<std::string> keys = {"key1", "key2", "key3", "key4", "key5"};
        auto results = backend.Create(keys, 128, "fake_trace_id_4", []() {});
        ASSERT_EQ(results.size(), keys.size());
        EXPECT_EQ(BuildUri("key1_5a560a3d977cc6f2?blkid=0&size=128"), results[0].second.ToUriString());
        EXPECT_EQ(BuildUri("key1_5a560a3d977cc6f2?blkid=1&size=128"), results[1].second.ToUriString());
        EXPECT_EQ(BuildUri("key3_1184f2d3fc112241?blkid=0&size=128"), results[2].second.ToUriString());
        EXPECT_EQ(BuildUri("key3_1184f2d3fc112241?blkid=1&size=128"), results[3].second.ToUriString());
        EXPECT_EQ(BuildUri("key5?blkid=0&size=128"), results[4].second.ToUriString());
    }
}

TEST_F(Hf3fsBackendTest, TestGetTypeAndAvailableStatus) {
    Hf3fsBackend backend(metrics_registry_);
    auto spec = GetDefaultStorageSpec();
    spec->set_key_count_per_file(2);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_HF3FS, "test", spec);
    ASSERT_FALSE(backend.Available());
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));
    ASSERT_TRUE(backend.Available());
    ASSERT_EQ(backend.Close(), EC_OK);
    ASSERT_FALSE(backend.Available());
}

TEST_F(Hf3fsBackendTest, TestCreateWithBatchingAndCallbackInvocation) {
    Hf3fsBackend backend(metrics_registry_);
    auto spec = GetDefaultStorageSpec();
    spec->set_key_count_per_file(2);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_HF3FS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_2"));
    std::vector<std::string> keys = {"key1", "key2", "key3", "key4", "key5"};
    bool callback_called = false;
    auto callback = [&callback_called]() { callback_called = true; };
    auto results = backend.Create(keys, 100, "fake_trace_id_3", callback);
    ASSERT_TRUE(callback_called);
    ASSERT_EQ(results.size(), keys.size());
    EXPECT_EQ(BuildUri("key1_5a560a3d977cc6f2?blkid=0&size=100"), results[0].second.ToUriString());
    EXPECT_EQ(BuildUri("key1_5a560a3d977cc6f2?blkid=1&size=100"), results[1].second.ToUriString());
    EXPECT_EQ(BuildUri("key3_1184f2d3fc112241?blkid=0&size=100"), results[2].second.ToUriString());
    EXPECT_EQ(BuildUri("key3_1184f2d3fc112241?blkid=1&size=100"), results[3].second.ToUriString());
    EXPECT_EQ(BuildUri("key5?blkid=0&size=100"), results[4].second.ToUriString());
    for (size_t i = 0; i < results.size(); ++i) {
        ASSERT_EQ(results[i].first, EC_OK);
    }
}

TEST_F(Hf3fsBackendTest, TestCreateWithBatchSizeOneAndEmptyKeys) {
    Hf3fsBackend backend(metrics_registry_);
    auto spec = GetDefaultStorageSpec();
    spec->set_key_count_per_file(1);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_HF3FS, "test", spec);
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
    EXPECT_EQ(BuildUri("a?size=100"), results[0].second.ToUriString());
    EXPECT_EQ(BuildUri("b?size=100"), results[1].second.ToUriString());
}

TEST_F(Hf3fsBackendTest, TestDeleteNotExistFile) {
    Hf3fsBackend backend(metrics_registry_);
    auto spec = GetDefaultStorageSpec();
    spec->set_key_count_per_file(1);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_HF3FS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_4"));
    std::vector<DataStorageUri> uris;
    uris.emplace_back(BuildUri("not_exist1"));
    uris.emplace_back(BuildUri("not_exist2"));
    auto res = backend.Delete(uris, "fake_trace_id", []() {});
    ASSERT_EQ(res.size(), uris.size());
    for (auto code : res) {
        ASSERT_EQ(code, EC_OK);
    }
}

TEST_F(Hf3fsBackendTest, TestExistAndDelete) {
    Hf3fsBackend backend(metrics_registry_);
    auto spec = GetDefaultStorageSpec();
    spec->set_key_count_per_file(1);
    spec->set_touch_file_when_create(true);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_HF3FS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_5"));
    std::vector<std::string> keys = {"key1", "key2"};
    auto results = backend.Create(keys, 100, "fake_trace_id_1", []() {});
    ASSERT_EQ(results.size(), keys.size());
    ASSERT_EQ(BuildUri("key1?size=100"), results[0].second.ToUriString());
    ASSERT_EQ(BuildUri("key2?size=100"), results[1].second.ToUriString());
    {
        std::vector<DataStorageUri> uris;
        uris.emplace_back(results[0].second);
        uris.emplace_back(results[1].second);
        uris.emplace_back(BuildUri("not_exist_key"));
        auto res = backend.Exist(uris);
        ASSERT_EQ(res.size(), uris.size());
        ASSERT_TRUE(res[0]);
        ASSERT_TRUE(res[1]);
        ASSERT_FALSE(res[2]);
    }
    {
        std::vector<DataStorageUri> uris;
        uris.emplace_back(results[0].second);
        uris.emplace_back(results[1].second);
        uris.emplace_back(BuildUri("not_exist_key2"));
        auto res = backend.Delete(uris, "fake_trace_id_2", []() {});
        ASSERT_EQ(res.size(), uris.size());
        ASSERT_EQ(EC_OK, res[0]);
        ASSERT_EQ(EC_OK, res[1]);
        ASSERT_EQ(EC_OK, res[2]);
    }
}

TEST_F(Hf3fsBackendTest, TestLockAndUnLockReturnOk) {
    Hf3fsBackend backend(metrics_registry_);
    auto spec = GetDefaultStorageSpec();
    spec->set_key_count_per_file(1);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_HF3FS, "test", spec);
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

TEST_F(Hf3fsBackendTest, TestCreateHandlesInvalidBatchSize) {
    Hf3fsBackend backend(metrics_registry_);
    auto spec = GetDefaultStorageSpec();
    spec->set_key_count_per_file(0);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_HF3FS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_2"));
    std::vector<std::string> keys = {"k1", "k2"};
    bool cb_called = false;
    auto results = backend.Create(keys, 50, "fake_trace_id_3", [&cb_called]() { cb_called = true; });
    ASSERT_TRUE(cb_called);
    ASSERT_EQ(results.size(), keys.size());
    EXPECT_EQ(BuildUri("k1?size=50"), results[0].second.ToUriString());
    EXPECT_EQ(BuildUri("k2?size=50"), results[1].second.ToUriString());
}

TEST_F(Hf3fsBackendTest, TestCreateSingleKeyBatch) {
    Hf3fsBackend backend(metrics_registry_);
    auto spec = GetDefaultStorageSpec();
    spec->set_key_count_per_file(10);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_HF3FS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_4"));
    std::vector<std::string> keys = {"singlekey"};
    bool cb_called = false;
    auto results = backend.Create(keys, 10, "fake_trace_id_5", [&cb_called]() { cb_called = true; });
    ASSERT_TRUE(cb_called);
    ASSERT_EQ(results.size(), keys.size());
    EXPECT_EQ(BuildUri("singlekey?blkid=0&size=10"), results[0].second.ToUriString());
}
