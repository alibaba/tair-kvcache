#include <cstring>
#include <filesystem>
#include <fstream>

#include "kv_cache_manager/client/src/transfer_client_impl.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

class TransferClientTest : public TESTBASE {
public:
    void SetUp() override {
        root_path_ = GetPrivateTestRuntimeDataPath();
        client_config_ = R"({
            "instance_group": "test_group",
            "instance_id": "test_instance",
            "block_size": 16,
            "sdk_config": {
                "thread_num": 4,
                "queue_size": 1000,
                "sdk_config": [],
                "timeout_config": {
                    "get_timeout_ms": 10000,
                    "put_timeout_ms": 30000
                }
            },
            "location_spec_infos": {
                "tp0": 1024
            }
        })";

        init_params_.role_type = RoleType::WORKER;
        init_params_.regist_span = new RegistSpan();
        auto buffer = malloc(1024 * 1024);
        init_params_.regist_span->base = buffer;
        init_params_.regist_span->size = 1024 * 1024;
        init_params_.self_location_spec_name = "tp0";
        init_params_.storage_configs = R"([
            {
                "type": "file",
                "global_unique_name": "test_nfs",
                "storage_spec": {
                    "root_path": "/tmp/test/",
                    "key_count_per_file": 5
                }
            }
        ])";
        ;
        InitFile();
    }

    void TearDown() override {
        client_config_.clear();
        free(init_params_.regist_span->base);
        delete init_params_.regist_span;
    }

private:
    void InitFile() {
        std::filesystem::create_directories(root_path_ + "tmp/test");
        std::string file_path = root_path_ + "tmp/test/key1";
        std::ofstream ofs(file_path);
        ASSERT_TRUE(ofs);
        ofs << test_data1_;
        ofs.close();

        file_path = root_path_ + "tmp/test/key2";
        ofs.open(file_path);
        ASSERT_TRUE(ofs);
        ofs << test_data2_;
        ofs.close();
        locations_ = {"file://test_nfs/" + root_path_ + "tmp/test/key1?blkid=0&size=1024",
                      "file://test_nfs/" + root_path_ + "tmp/test/key2?blkid=0&size=1024"};
    }

private:
    const char *test_data1_ = "test key1";
    const char *test_data2_ = "test key2";
    std::string root_path_;
    std::string client_config_;
    InitParams init_params_;
    UriStrVec locations_;
};

TEST_F(TransferClientTest, TestCreate) {
    {
        auto client = TransferClient::Create(client_config_, init_params_);
        EXPECT_NE(client, nullptr);
    }
    {
        std::string invalid_config = R"({})";
        auto client = TransferClient::Create(invalid_config, init_params_);
        EXPECT_EQ(client, nullptr);
    }
}

TEST_F(TransferClientTest, TestCreateWithEmptySelfLocationSpecName) {
    auto init_params = init_params_;
    init_params.self_location_spec_name = "";
    auto client = TransferClient::Create(client_config_, init_params);
    EXPECT_EQ(client, nullptr);
}

TEST_F(TransferClientTest, TestCreateWithEmptyAddress) {
    std::string client_config = R"({
        "instance_group": "group",
        "instance_id": "instance",
        "block_size": 128,
        "sdk_config": {},
        "model_deployment": {
            "model_name": "test_model",
            "dtype": "FP8",
            "use_mla": false,
            "tp_size": 1,
            "dp_size": 1,
            "pp_size": 1
        },
        "location_spec_infos": {
            "tp0": 1024
        }
    })";
    auto client = TransferClient::Create(client_config, init_params_);
    EXPECT_NE(client, nullptr);
}

TEST_F(TransferClientTest, TestLoadKvCaches) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);

    BlockBuffer buffer1, buffer2;
    BlockBuffers block_buffers = {buffer1, buffer2};

    EXPECT_EQ(ER_OK, client->LoadKvCaches(locations_, block_buffers));
}

TEST_F(TransferClientTest, TestSaveKvCaches) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);

    BlockBuffer buffer1, buffer2;
    BlockBuffers block_buffers = {buffer1, buffer2};

    auto result = client->SaveKvCaches(locations_, block_buffers);

    EXPECT_EQ(ER_OK, result.first);
    EXPECT_EQ(result.second.size(), locations_.size());
}

TEST_F(TransferClientTest, TestEmptyLocations) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);

    UriStrVec uri_str_vec = {};
    BlockBuffers block_buffers = {};

    EXPECT_EQ(ER_INVALID_PARAMS, client->LoadKvCaches(uri_str_vec, block_buffers));

    auto save_result = client->SaveKvCaches(uri_str_vec, block_buffers);
    EXPECT_EQ(ER_INVALID_PARAMS, save_result.first);
    EXPECT_TRUE(save_result.second.empty());
}

TEST_F(TransferClientTest, TestManyLocations) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);

    UriStrVec uri_str_vec;
    BlockBuffers block_buffers;

    for (int i = 0; i < 100; i++) {
        uri_str_vec.push_back("file://test_nfs/" + root_path_ + "tmp/test/key_" + std::to_string(i) +
                              "?blkid=0&size=1024");
        block_buffers.push_back(BlockBuffer());
    }

    auto save_result = client->SaveKvCaches(uri_str_vec, block_buffers);
    EXPECT_EQ(ER_OK, save_result.first);
    EXPECT_EQ(save_result.second.size(), uri_str_vec.size());
}

TEST_F(TransferClientTest, TestWrongRoleType) {
    auto init_params = init_params_;
    init_params.role_type = RoleType::SCHEDULER;
    auto client = TransferClient::Create(client_config_, init_params);
    EXPECT_EQ(client, nullptr);
}

TEST_F(TransferClientTest, TestMismatchedLocationsAndBuffers) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);

    BlockBuffer buffer1;
    BlockBuffers block_buffers = {buffer1};

    EXPECT_EQ(ER_INVALID_PARAMS, client->LoadKvCaches(locations_, block_buffers));
}

TEST_F(TransferClientTest, TestBlockBufferUsage) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);
    size_t len1 = strlen(test_data1_);
    size_t len2 = strlen(test_data2_);

    auto get_buffer = malloc(1024 * 1024);
    std::memcpy(get_buffer, test_data1_, len1);
    std::memcpy(static_cast<char *>(get_buffer) + len1, test_data2_, len2);

    BlockBuffer buffer1, buffer2;
    buffer1.iovs.resize(1);
    buffer2.iovs.resize(1);

    buffer1.iovs[0].type = MemoryType::CPU;
    buffer1.iovs[0].base = get_buffer;
    buffer1.iovs[0].size = len1;
    buffer1.iovs[0].ignore = false;

    buffer2.iovs[0].type = MemoryType::CPU;
    buffer2.iovs[0].base = static_cast<char *>(get_buffer) + len1;
    buffer2.iovs[0].size = len2;
    buffer2.iovs[0].ignore = false;

    BlockBuffers block_buffers = {buffer1, buffer2};
    ASSERT_EQ(ER_OK, client->LoadKvCaches(locations_, block_buffers));

    ASSERT_EQ(std::memcmp(buffer1.iovs[0].base, test_data1_, buffer1.iovs[0].size), 0);
    ASSERT_EQ(std::memcmp(buffer2.iovs[0].base, test_data2_, buffer2.iovs[0].size), 0);

    free(get_buffer);
}

// ============================================================================
// Multi-storage TransferClient tests
// ============================================================================
class TransferClientMultiStorageTest : public TESTBASE {
public:
    void SetUp() override {
        root_path_ = GetPrivateTestRuntimeDataPath();
        client_config_ = R"({
            "instance_group": "test_group",
            "instance_id": "test_instance",
            "block_size": 16,
            "sdk_config": {
                "thread_num": 4,
                "queue_size": 1000,
                "sdk_config": [],
                "timeout_config": {
                    "get_timeout_ms": 10000,
                    "put_timeout_ms": 30000
                }
            },
            "location_spec_infos": {
                "tp0": 1024
            }
        })";

        init_params_.role_type = RoleType::WORKER;
        init_params_.regist_span = new RegistSpan();
        auto buffer = malloc(1024 * 1024);
        init_params_.regist_span->base = buffer;
        init_params_.regist_span->size = 1024 * 1024;
        init_params_.self_location_spec_name = "tp0";
        init_params_.storage_configs = R"([
            {
                "type": "file",
                "global_unique_name": "nfs_a",
                "storage_spec": {
                    "root_path": "/tmp/nfs_a/",
                    "key_count_per_file": 5
                }
            },
            {
                "type": "file",
                "global_unique_name": "nfs_b",
                "storage_spec": {
                    "root_path": "/tmp/nfs_b/",
                    "key_count_per_file": 5
                }
            }
        ])";
    }

    void TearDown() override {
        free(init_params_.regist_span->base);
        delete init_params_.regist_span;
    }

protected:
    std::string root_path_;
    std::string client_config_;
    InitParams init_params_;
};

TEST_F(TransferClientMultiStorageTest, TestCreateWithMultiStorage) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);
}

TEST_F(TransferClientMultiStorageTest, TestSaveAndLoadMixedStorage) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);

    // 混合 URI: nfs_a, nfs_b, nfs_a（同 path 不同 blkid 保证 SDK 内部不重排）
    UriStrVec uri_str_vec = {
        "file://nfs_a/" + root_path_ + "tmp/nfs_a/file1?blkid=0&size=1024",
        "file://nfs_b/" + root_path_ + "tmp/nfs_b/file1?blkid=0&size=1024",
        "file://nfs_a/" + root_path_ + "tmp/nfs_a/file1?blkid=1&size=1024",
    };
    BlockBuffers block_buffers = {BlockBuffer(), BlockBuffer(), BlockBuffer()};

    // Save
    auto [save_ec, actual_uris] = client->SaveKvCaches(uri_str_vec, block_buffers);
    ASSERT_EQ(ER_OK, save_ec);
    ASSERT_EQ(3, actual_uris.size());
    // 验证返回 URI 顺序与输入一致
    EXPECT_EQ(uri_str_vec[0], actual_uris[0]);
    EXPECT_EQ(uri_str_vec[1], actual_uris[1]);
    EXPECT_EQ(uri_str_vec[2], actual_uris[2]);

    // Load
    EXPECT_EQ(ER_OK, client->LoadKvCaches(actual_uris, block_buffers));
}

// 回归测试：nfs_a/path1, nfs_b/pathX, nfs_a/path2 —— 同 backend（nfs_a）内多个
// 不同 path 被 nfs_b 交错隔开。修复前 LocalFileSdk::Put 按 path 分组后以
// unordered_map 迭代顺序回填 actual_uris，返回顺序与输入不一致，上层按契约回填
// 时会把 URI 写到错误位置，后续按返回 URI 读取将拿到错块数据。
TEST_F(TransferClientMultiStorageTest, TestSaveLoadInterleavedMultiPathOrdering) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(client, nullptr);

    UriStrVec uri_str_vec = {
        "file://nfs_a/" + root_path_ + "tmp/nfs_a/path1?blkid=0&size=1024",
        "file://nfs_b/" + root_path_ + "tmp/nfs_b/pathX?blkid=0&size=1024",
        "file://nfs_a/" + root_path_ + "tmp/nfs_a/path2?blkid=0&size=1024",
    };

    const char *payload1 = "payload for nfs_a path1";
    const char *payloadX = "payload for nfs_b pathX";
    const char *payload2 = "payload for nfs_a path2";
    size_t len1 = strlen(payload1);
    size_t lenX = strlen(payloadX);
    size_t len2 = strlen(payload2);

    void *mem1 = malloc(1024);
    void *memX = malloc(1024);
    void *mem2 = malloc(1024);
    std::memcpy(mem1, payload1, len1);
    std::memcpy(memX, payloadX, lenX);
    std::memcpy(mem2, payload2, len2);

    auto make_buffer = [](void *base, size_t size) {
        BlockBuffer bb;
        Iov iov;
        iov.type = MemoryType::CPU;
        iov.base = base;
        iov.size = size;
        iov.ignore = false;
        bb.iovs.push_back(iov);
        return bb;
    };
    BlockBuffers block_buffers = {make_buffer(mem1, len1), make_buffer(memX, lenX), make_buffer(mem2, len2)};

    // Save：返回 URI 顺序必须与输入一致
    auto [save_ec, actual_uris] = client->SaveKvCaches(uri_str_vec, block_buffers);
    ASSERT_EQ(ER_OK, save_ec);
    ASSERT_EQ(uri_str_vec.size(), actual_uris.size());
    EXPECT_EQ(uri_str_vec[0], actual_uris[0]);
    EXPECT_EQ(uri_str_vec[1], actual_uris[1]);
    EXPECT_EQ(uri_str_vec[2], actual_uris[2]);

    // Load：用全新 buffer 按返回 URI 读回，验证数据确实落在各自的文件
    void *get1 = malloc(1024);
    void *getX = malloc(1024);
    void *get2 = malloc(1024);
    BlockBuffers get_buffers = {make_buffer(get1, len1), make_buffer(getX, lenX), make_buffer(get2, len2)};
    EXPECT_EQ(ER_OK, client->LoadKvCaches(actual_uris, get_buffers));
    EXPECT_EQ(std::memcmp(get1, payload1, len1), 0);
    EXPECT_EQ(std::memcmp(getX, payloadX, lenX), 0);
    EXPECT_EQ(std::memcmp(get2, payload2, len2), 0);

    free(mem1);
    free(memX);
    free(mem2);
    free(get1);
    free(getX);
    free(get2);
}
