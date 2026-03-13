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

        CreateBlocksRequest request;
        request.instance_id = "test_instance";
        SpecBlockKeys spec_block;
        spec_block.spec_name = "tp0";
        spec_block.spec_size = 128;
        spec_block.block_keys = {0x6b657931, 0x6b657932, 0x6b657933, 0x6b657934, 0x6b657935};
        spec_block.original_key_indices = {0, 1, 2, 3, 4};
        request.spec_block_keys.push_back(std::move(spec_block));

        auto result = backend.Create(request, "fake_trace_id_2", []() {});
        ASSERT_EQ(result.size(), 1);
        ASSERT_EQ(result[0].size(), 5);
        ASSERT_EQ("file:///data/test_instance/tp0/6b657931?size=128", result[0][0].second.ToUriString());
        ASSERT_EQ("file:///data/test_instance/tp0/6b657932?size=128", result[0][1].second.ToUriString());
        ASSERT_EQ("file:///data/test_instance/tp0/6b657933?size=128", result[0][2].second.ToUriString());
        ASSERT_EQ("file:///data/test_instance/tp0/6b657934?size=128", result[0][3].second.ToUriString());
        ASSERT_EQ("file:///data/test_instance/tp0/6b657935?size=128", result[0][4].second.ToUriString());
    }
    // 多个key一个文件的形式
    {
        NfsBackend backend(metrics_registry_);
        std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
        spec->set_key_count_per_file(2);
        spec->set_root_path("/data/");
        StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
        ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_3"));

        CreateBlocksRequest request;
        request.instance_id = "test_instance";
        SpecBlockKeys spec_block;
        spec_block.spec_name = "tp0";
        spec_block.spec_size = 128;
        spec_block.block_keys = {0x6b657931, 0x6b657932, 0x6b657933, 0x6b657934, 0x6b657935};
        spec_block.original_key_indices = {0, 1, 2, 3, 4};
        request.spec_block_keys.push_back(std::move(spec_block));

        auto result = backend.Create(request, "fake_trace_id_4", []() {});
        ASSERT_EQ(result.size(), 1);
        ASSERT_EQ(result[0].size(), 5);
        EXPECT_EQ("file:///data/test_instance/tp0/6b657931_36fc91bdda6e6263?blkid=0&size=128", result[0][0].second.ToUriString());
        EXPECT_EQ("file:///data/test_instance/tp0/6b657931_36fc91bdda6e6263?blkid=1&size=128", result[0][1].second.ToUriString());
        EXPECT_EQ("file:///data/test_instance/tp0/6b657933_d893940787994b17?blkid=0&size=128", result[0][2].second.ToUriString());
        EXPECT_EQ("file:///data/test_instance/tp0/6b657933_d893940787994b17?blkid=1&size=128", result[0][3].second.ToUriString());
        EXPECT_EQ("file:///data/test_instance/tp0/6b657935?blkid=0&size=128", result[0][4].second.ToUriString());
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

    CreateBlocksRequest request;
    request.instance_id = "test_instance";
    SpecBlockKeys spec_block;
    spec_block.spec_name = "tp0";
    spec_block.spec_size = 100;
    spec_block.block_keys = {0x6b657931, 0x6b657932, 0x6b657933, 0x6b657934, 0x6b657935};
    spec_block.original_key_indices = {0, 1, 2, 3, 4};
    request.spec_block_keys.push_back(std::move(spec_block));

    bool callback_called = false;
    auto callback = [&callback_called]() { callback_called = true; };
    auto result = backend.Create(request, "fake_trace_id_2", callback);
    ASSERT_TRUE(callback_called);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0].size(), 5);
    EXPECT_EQ("file:///data/test_instance/tp0/6b657931_36fc91bdda6e6263?blkid=0&size=100", result[0][0].second.ToUriString());
    EXPECT_EQ("file:///data/test_instance/tp0/6b657931_36fc91bdda6e6263?blkid=1&size=100", result[0][1].second.ToUriString());
    EXPECT_EQ("file:///data/test_instance/tp0/6b657933_d893940787994b17?blkid=0&size=100", result[0][2].second.ToUriString());
    EXPECT_EQ("file:///data/test_instance/tp0/6b657933_d893940787994b17?blkid=1&size=100", result[0][3].second.ToUriString());
    EXPECT_EQ("file:///data/test_instance/tp0/6b657935?blkid=0&size=100", result[0][4].second.ToUriString());
    for (size_t i = 0; i < result[0].size(); ++i) {
        ASSERT_EQ(result[0][i].first, EC_OK);
    }
}

TEST_F(NfsBackendTest, TestCreateWithBatchSizeOneAndEmptyKeys) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(1);
    spec->set_root_path("/data/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));

    // Test empty request
    CreateBlocksRequest empty_request;
    empty_request.instance_id = "test_instance";
    // No spec_block_keys added
    bool callback_called = false;
    auto cb = [&callback_called]() { callback_called = true; };
    auto results_empty = backend.Create(empty_request, "fake_trace_id_2", cb);
    ASSERT_TRUE(callback_called);
    ASSERT_TRUE(results_empty.empty());

    // Test with keys
    CreateBlocksRequest request;
    request.instance_id = "test_instance";
    SpecBlockKeys spec_block;
    spec_block.spec_name = "tp0";
    spec_block.spec_size = 100;
    spec_block.block_keys = {0x61, 0x62}; // hex of 'a', 'b'
    spec_block.original_key_indices = {0, 1};
    request.spec_block_keys.push_back(std::move(spec_block));

    callback_called = false;
    auto result = backend.Create(request, "fake_trace_id_3", [&callback_called]() { callback_called = true; });
    ASSERT_TRUE(callback_called);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0].size(), 2);
    ASSERT_EQ("file:///data/test_instance/tp0/61?size=100", result[0][0].second.ToUriString());
    ASSERT_EQ("file:///data/test_instance/tp0/62?size=100", result[0][1].second.ToUriString());
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

    CreateBlocksRequest request;
    request.instance_id = "test_instance";
    SpecBlockKeys spec_block;
    spec_block.spec_name = "tp0";
    spec_block.spec_size = 50;
    spec_block.block_keys = {0x6b31, 0x6b32}; // hex of "k1", "k2"
    spec_block.original_key_indices = {0, 1};
    request.spec_block_keys.push_back(std::move(spec_block));

    bool cb_called = false;
    auto result = backend.Create(request, "fake_trace_id_2", [&cb_called]() { cb_called = true; });
    ASSERT_TRUE(cb_called);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0].size(), 2);
    ASSERT_EQ("file:///root/test_instance/tp0/6b31?size=50", result[0][0].second.ToUriString());
    ASSERT_EQ("file:///root/test_instance/tp0/6b32?size=50", result[0][1].second.ToUriString());
}

TEST_F(NfsBackendTest, TestCreateSingleKeyBatch) {
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(10);
    spec->set_root_path("/root/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));

    CreateBlocksRequest request;
    request.instance_id = "test_instance";
    SpecBlockKeys spec_block;
    spec_block.spec_name = "tp0";
    spec_block.spec_size = 10;
    spec_block.block_keys = {0x12345678abcdef00LL}; // a valid int64 key
    spec_block.original_key_indices = {0};
    request.spec_block_keys.push_back(std::move(spec_block));

    bool cb_called = false;
    auto result = backend.Create(request, "fake_trace_id_2", [&cb_called]() { cb_called = true; });
    ASSERT_TRUE(cb_called);
    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0].size(), 1);
    ASSERT_EQ("file:///root/test_instance/tp0/12345678abcdef00?blkid=0&size=10", result[0][0].second.ToUriString());
}

TEST_F(NfsBackendTest, TestMultipleSpecsNotMixed) {
    // Test that keys from different specs are never mixed in the same batch
    NfsBackend backend(metrics_registry_);
    std::shared_ptr<NfsStorageSpec> spec(new NfsStorageSpec);
    spec->set_key_count_per_file(3); // Large batch size
    spec->set_root_path("/data/");
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "test", spec);
    ASSERT_EQ(EC_OK, backend.Open(storage_config, "fake_trace_id_1"));

    CreateBlocksRequest request;
    request.instance_id = "test_instance";

    // First spec with 2 keys
    SpecBlockKeys spec_block1;
    spec_block1.spec_name = "tp0";
    spec_block1.spec_size = 128;
    spec_block1.block_keys = {0x6b657931, 0x6b657932};
    spec_block1.original_key_indices = {0, 1};
    request.spec_block_keys.push_back(std::move(spec_block1));

    // Second spec with 2 keys
    SpecBlockKeys spec_block2;
    spec_block2.spec_name = "tp1";
    spec_block2.spec_size = 128;
    spec_block2.block_keys = {0x6b657933, 0x6b657934};
    spec_block2.original_key_indices = {2, 3};
    request.spec_block_keys.push_back(std::move(spec_block2));

    auto result = backend.Create(request, "fake_trace_id_2", []() {});
    ASSERT_EQ(result.size(), 2);
    ASSERT_EQ(result[0].size(), 2);
    ASSERT_EQ(result[1].size(), 2);

    EXPECT_EQ("file:///data/test_instance/tp0/6b657931_36fc91bdda6e6263?blkid=0&size=128", result[0][0].second.ToUriString());
    EXPECT_EQ("file:///data/test_instance/tp0/6b657931_36fc91bdda6e6263?blkid=1&size=128", result[0][1].second.ToUriString());
    EXPECT_EQ("file:///data/test_instance/tp1/6b657933_3533532bb1ef35e1?blkid=0&size=128", result[1][0].second.ToUriString());
    EXPECT_EQ("file:///data/test_instance/tp1/6b657933_3533532bb1ef35e1?blkid=1&size=128", result[1][1].second.ToUriString());
}
