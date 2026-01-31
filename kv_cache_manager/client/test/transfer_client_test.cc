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
                    "get_timeout_ms": 5000,
                    "put_timeout_ms": 10000
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
