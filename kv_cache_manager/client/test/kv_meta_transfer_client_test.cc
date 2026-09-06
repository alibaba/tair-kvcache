#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "kv_cache_manager/client/include/kv_meta_transfer_client.h"
#include "kv_cache_manager/client/include/transfer_client.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {
namespace {

BlockBuffer MakeBuffer(void *base, std::size_t size) {
    BlockBuffer buffer;
    buffer.iovs.push_back({MemoryType::CPU, base, size, false});
    return buffer;
}

class KvMetaTransferClientTest : public TESTBASE {
protected:
    void SetUp() override {
        root_path_ = GetPrivateTestRuntimeDataPath() + "kvmeta_objects/";
        std::filesystem::create_directories(root_path_);
        client_config_ = R"({
            "instance_group": "test_group",
            "instance_id": "test_instance",
            "block_size": 1,
            "sdk_config": {
                "thread_num": 2,
                "queue_size": 32,
                "sdk_config": [],
                "timeout_config": {
                    "get_timeout_ms": 10000,
                    "put_timeout_ms": 10000
                }
            },
            "location_spec_infos": {
                "value": 1
            }
        })";
        init_params_.role_type = RoleType::WORKER;
        init_params_.self_location_spec_name = "value";
        init_params_.storage_configs = R"([
            {
                "type": "file",
                "global_unique_name": "test_nfs",
                "storage_spec": {
                    "root_path": "/tmp/unused/",
                    "key_count_per_file": 1
                }
            }
        ])";
    }

    std::string root_path_;
    std::string client_config_;
    InitParams init_params_;
};

TEST_F(KvMetaTransferClientTest, SavesAndLoadsDifferentObjectSizesInOneBatch) {
    auto client = KvMetaTransferClient::Create(client_config_, init_params_, 1024);
    ASSERT_NE(nullptr, client);

    std::vector<char> first{1, 2, 3, 4, 5};
    std::vector<char> second{9, 8, 7, 6, 5, 4, 3, 2, 1};
    const std::string first_path = root_path_ + "first";
    const std::string second_path = root_path_ + "second";
    const UriStrVec uris = {
        "file://test_nfs/" + first_path + "?blkid=0&size=5",
        "file://test_nfs/" + second_path + "?blkid=0&size=9",
    };
    const std::vector<std::uint64_t> sizes = {first.size(), second.size()};
    const BlockBuffers source = {
        MakeBuffer(first.data(), first.size()),
        MakeBuffer(second.data(), second.size()),
    };

    auto [save_ec, actual_uris] = client->SaveObjects(uris, sizes, source);
    ASSERT_EQ(ER_OK, save_ec);
    EXPECT_EQ(uris, actual_uris);
    ASSERT_TRUE(std::filesystem::exists(first_path));
    ASSERT_TRUE(std::filesystem::exists(second_path));
    EXPECT_EQ(first.size(), std::filesystem::file_size(first_path));
    EXPECT_EQ(second.size(), std::filesystem::file_size(second_path));

    std::vector<char> loaded_first(first.size());
    std::vector<char> loaded_second(second.size());
    const BlockBuffers destination = {
        MakeBuffer(loaded_first.data(), loaded_first.size()),
        MakeBuffer(loaded_second.data(), loaded_second.size()),
    };
    ASSERT_EQ(ER_OK, client->LoadObjects(uris, sizes, destination));
    EXPECT_EQ(first, loaded_first);
    EXPECT_EQ(second, loaded_second);
}

TEST_F(KvMetaTransferClientTest, RejectsUriAndBufferSizeMismatchBeforeIo) {
    auto client = KvMetaTransferClient::Create(client_config_, init_params_, 1024);
    ASSERT_NE(nullptr, client);
    std::vector<char> payload(5, 1);
    const std::string path = root_path_ + "must_not_exist";
    const UriStrVec uris = {"file://test_nfs/" + path + "?blkid=0&size=4"};

    auto [ec, actual_uris] =
        client->SaveObjects(uris, {payload.size()}, {MakeBuffer(payload.data(), payload.size())});
    EXPECT_EQ(ER_INVALID_PARAMS, ec);
    EXPECT_TRUE(actual_uris.empty());
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(KvMetaTransferClientTest, RejectsIgnoredOrOversizedObjects) {
    auto client = KvMetaTransferClient::Create(client_config_, init_params_, 8);
    ASSERT_NE(nullptr, client);
    std::vector<char> payload(9, 1);
    const UriStrVec uris = {"file://test_nfs/" + root_path_ + "oversized?blkid=0&size=9"};
    auto buffer = MakeBuffer(payload.data(), payload.size());
    EXPECT_EQ(ER_INVALID_PARAMS, client->LoadObjects(uris, {payload.size()}, {buffer}));

    payload.resize(5);
    buffer = MakeBuffer(payload.data(), payload.size());
    buffer.iovs[0].ignore = true;
    const UriStrVec ignored_uri = {"file://test_nfs/" + root_path_ + "ignored?blkid=0&size=5"};
    EXPECT_EQ(ER_INVALID_LOCAL_BUFFERS, client->LoadObjects(ignored_uri, {payload.size()}, {buffer}));
}

TEST_F(KvMetaTransferClientTest, RegularTransferClientKeepsFixedSizePolicy) {
    auto client = TransferClient::Create(client_config_, init_params_);
    ASSERT_NE(nullptr, client);
    std::vector<char> payload(5, 1);
    const std::string path = root_path_ + "regular_reject";
    const UriStrVec uris = {"file://test_nfs/" + path + "?blkid=0&size=5"};
    const auto result = client->SaveKvCaches(uris, {MakeBuffer(payload.data(), payload.size())});
    EXPECT_NE(ER_OK, result.first);
}

} // namespace
} // namespace kv_cache_manager
