#include <cstring>
#include <filesystem>
#include <limits>
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
                "queue_size": 64,
                "sdk_backend_configs": [],
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

    auto overflowing_buffer =
        MakeBuffer(reinterpret_cast<void *>(std::numeric_limits<std::uintptr_t>::max() - payload.size() + 1),
                   payload.size());
    EXPECT_EQ(ER_INVALID_LOCAL_BUFFERS,
              client->LoadObjects(ignored_uri, {payload.size()}, {overflowing_buffer}));
}

TEST_F(KvMetaTransferClientTest, RejectsMaxObjectSizeAboveTheServiceContract) {
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(client_config_, init_params_, 0));
    EXPECT_EQ(nullptr,
              KvMetaTransferClient::Create(
                  client_config_, init_params_, 1ULL * 1024 * 1024 * 1024 + 1));
}

TEST_F(KvMetaTransferClientTest, RejectsMalformedConstructionInputsBeforeBackendInitialization) {
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create("", init_params_, 1024));
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create("{invalid-json", init_params_, 1024));

    auto invalid_init_params = init_params_;
    invalid_init_params.role_type = RoleType::SCHEDULER;
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(client_config_, invalid_init_params, 1024));

    invalid_init_params = init_params_;
    invalid_init_params.self_location_spec_name = "tp0";
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(client_config_, invalid_init_params, 1024));

    invalid_init_params = init_params_;
    invalid_init_params.storage_configs.clear();
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(client_config_, invalid_init_params, 1024));
}

TEST_F(KvMetaTransferClientTest, RejectsServiceBatchLimitsBeforeDataPlaneIo) {
    auto client = KvMetaTransferClient::Create(client_config_, init_params_, 1ULL * 1024 * 1024 * 1024);
    ASSERT_NE(nullptr, client);

    char payload = 0;
    UriStrVec too_many_uris;
    std::vector<std::uint64_t> too_many_sizes;
    BlockBuffers too_many_buffers;
    for (std::size_t i = 0; i < 65; ++i) {
        too_many_uris.push_back(
            "file://test_nfs/" + root_path_ + "too-many-" + std::to_string(i) + "?blkid=0&size=1");
        too_many_sizes.push_back(1);
        too_many_buffers.push_back(MakeBuffer(&payload, 1));
    }
    EXPECT_EQ(ER_INVALID_PARAMS,
              client->LoadObjects(too_many_uris, too_many_sizes, too_many_buffers));

    constexpr std::size_t kOneGiB = 1ULL * 1024 * 1024 * 1024;
    UriStrVec oversized_batch_uris;
    std::vector<std::uint64_t> oversized_batch_sizes;
    BlockBuffers oversized_batch_buffers;
    for (std::size_t i = 0; i < 5; ++i) {
        oversized_batch_uris.push_back(
            "file://test_nfs/" + root_path_ + "too-large-" + std::to_string(i) +
            "?blkid=0&size=" + std::to_string(kOneGiB));
        oversized_batch_sizes.push_back(kOneGiB);
        oversized_batch_buffers.push_back(MakeBuffer(&payload, kOneGiB));
    }
    const auto [save_ec, actual_uris] =
        client->SaveObjects(oversized_batch_uris, oversized_batch_sizes, oversized_batch_buffers);
    EXPECT_EQ(ER_INVALID_PARAMS, save_ec);
    EXPECT_TRUE(actual_uris.empty());
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_FALSE(std::filesystem::exists(root_path_ + "too-large-" + std::to_string(i)));
    }
}

TEST_F(KvMetaTransferClientTest, RejectsConfigsThatDoNotUseTheExactKvMetaMarker) {
    const auto replace_once = [](std::string value, const std::string &from, const std::string &to) {
        const auto position = value.find(from);
        EXPECT_NE(std::string::npos, position);
        if (position != std::string::npos) {
            value.replace(position, from.size(), to);
        }
        return value;
    };

    const auto wrong_marker_size = replace_once(client_config_, R"("value": 1)", R"("value": 2)");
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(wrong_marker_size, init_params_, 1024));

    const auto extra_location_spec =
        replace_once(client_config_, R"("value": 1)", R"("value": 1, "other": 1)");
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(extra_location_spec, init_params_, 1024));

    const auto wrong_block_size = replace_once(client_config_, R"("block_size": 1)", R"("block_size": 2)");
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(wrong_block_size, init_params_, 1024));

    const auto grouped_location_spec =
        replace_once(client_config_,
                     R"("value": 1)",
                     R"("value": 1
            },
            "location_spec_groups": {
                "value_group": ["value"])");
    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(grouped_location_spec, init_params_, 1024));
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

TEST_F(KvMetaTransferClientTest, RejectsQueueSmallerThanOneServiceBatchOnlyForKvMeta) {
    const auto position = client_config_.find(R"("queue_size": 64)");
    ASSERT_NE(std::string::npos, position);
    auto undersized_queue = client_config_;
    undersized_queue.replace(position, std::string(R"("queue_size": 64)").size(), R"("queue_size": 63)");

    EXPECT_EQ(nullptr, KvMetaTransferClient::Create(undersized_queue, init_params_, 1024));

    // The queue-capacity contract belongs to the new exact-object API. Keep
    // the established fixed-block TransferClient configuration path intact.
    EXPECT_NE(nullptr, TransferClient::Create(undersized_queue, init_params_));
}

} // namespace
} // namespace kv_cache_manager
