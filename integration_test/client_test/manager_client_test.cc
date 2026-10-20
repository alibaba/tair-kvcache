#include <array>
#include <memory>
#include <vector>

#include "client_test_base.h"
#include "kv_cache_manager/client/include/manager_client.h"
#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/logger.h"

using namespace ::testing;
using namespace kv_cache_manager;

class ManagerClientTest : public CLIENTTESTBASE {
public:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(ManagerClientTest, TestCreateManagerClient) {
    auto prefix = GetCurrentTestName();
    auto manager_client = CreateClient<ManagerClient>(prefix, init_params_);
    ASSERT_NE(nullptr, manager_client.get());
}

TEST_F(ManagerClientTest, TestEmptyMetaClient) {
    auto prefix = GetCurrentTestName();
    kv_cache_manager::InitParams init_params = {RoleType::WORKER, nullptr, "tp0", R"([
            {
                "type": "file",
                "global_unique_name": "test_nfs",
                "storage_spec": {
                    "root_path": "/tmp/test/",
                    "key_count_per_file": 5
                }
            }
        ])"};
    auto manager_client = CreateClient<ManagerClient>(prefix, init_params);
    ASSERT_NE(nullptr, manager_client.get());
    {
        auto [success, write_location] = manager_client->StartWrite(prefix + "_1", {1, 2, 3, 4}, {}, {}, 1000000);
        ASSERT_EQ(ER_CLIENT_NOT_EXISTS, success);
    }
    {
        std::vector<ClientReplicationHint> hints;
        auto [success, locations_map] = manager_client->MatchLocation(
            prefix + "_2", QueryType::QT_PREFIX_MATCH, {1, 2, 3, 4}, {}, static_cast<size_t>(0), 0, {}, hints);
        ASSERT_EQ(ER_CLIENT_NOT_EXISTS, success);
    }
    {
        BlockMask success_block = BlockMaskVector({true, true, false, false});
        ASSERT_EQ(ER_CLIENT_NOT_EXISTS,
                  manager_client->FinishWrite(prefix + "_3", "test_write_session_id", success_block, {}));
    }
    {
        auto [success, metas] = manager_client->MatchMeta(prefix + "_4", {1, 2, 3, 4}, {}, static_cast<size_t>(0), 0);
        ASSERT_EQ(ER_CLIENT_NOT_EXISTS, success);
    }
    {
        ASSERT_EQ(ER_CLIENT_NOT_EXISTS,
                  manager_client->RemoveCache(prefix + "_5", {1, 2, 3, 4}, {}, static_cast<size_t>(0)));
    }
}

TEST_F(ManagerClientTest, TestEmptyTransferClient) {
    auto prefix = GetCurrentTestName();
    kv_cache_manager::InitParams init_params = {RoleType::SCHEDULER, nullptr, "tp0"};
    auto manager_client = CreateClient<ManagerClient>(prefix, init_params);
    ASSERT_NE(nullptr, manager_client.get());
    { ASSERT_EQ(ER_CLIENT_NOT_EXISTS, manager_client->LoadKvCaches({}, {})); }
    {
        auto [success, locations_map] = manager_client->SaveKvCaches({}, {});
        ASSERT_EQ(ER_CLIENT_NOT_EXISTS, success);
    }
}

TEST_F(ManagerClientTest, TestStartWriteCache) {
    auto prefix = GetCurrentTestName();
    auto manager_client = CreateClient<ManagerClient>(prefix, init_params_);
    ASSERT_NE(nullptr, manager_client.get());
    TestStartWrite<ManagerClient>(prefix, manager_client);
}

TEST_F(ManagerClientTest, TestFinishWriteCacheSuccess) {
    auto prefix = GetCurrentTestName();
    auto manager_client = CreateClient<ManagerClient>(prefix, init_params_);
    ASSERT_NE(nullptr, manager_client.get());
    TestFinishWriteSuccess<ManagerClient>(prefix, manager_client);
}

TEST_F(ManagerClientTest, TestFinishWriteCacheFail) {
    auto prefix = GetCurrentTestName();
    auto manager_client = CreateClient<ManagerClient>(prefix, init_params_);
    ASSERT_NE(nullptr, manager_client.get());
    TestFinishWriteFail<ManagerClient>(prefix, manager_client);
}

TEST_F(ManagerClientTest, TestGetCacheLocation) {
    auto prefix = GetCurrentTestName();
    auto manager_client = CreateClient<ManagerClient>(prefix, init_params_);
    ASSERT_NE(nullptr, manager_client.get());
    TestGetCacheLocation<ManagerClient>(prefix, manager_client);
}

TEST_F(ManagerClientTest, TestSaveAndLoad) {
    auto prefix = GetCurrentTestName();
    constexpr size_t kBlocks = 3;
    constexpr size_t kPublishedBlocks = 2;
    constexpr size_t kSpecs = 2;
    constexpr size_t kBytes = 1024;
    std::array<ClientPtr<ManagerClient>, kSpecs> clients;
    for (size_t spec = 0; spec < kSpecs; ++spec) {
        auto params = init_params_;
        params.self_location_spec_name = "tp" + std::to_string(spec);
        clients[spec] = CreateClient<ManagerClient>(prefix, params);
        ASSERT_NE(nullptr, clients[spec]);
    }
    auto &manager = *clients[0];
    auto [write_ec, write] = manager.StartWrite(prefix + "_start", {1, 2, 3}, {}, {}, 60);
    ASSERT_EQ(ER_OK, write_ec);
    ASSERT_FALSE(write.write_session_id.empty());
    ASSERT_EQ(kBlocks, write.locations.size());
    ExpectLocationsValid(write.locations, kBlocks);
    ASSERT_FALSE(HasFailure());
    std::vector<ClientReplicationHint> hints;
    const auto expect_miss = [&](const std::vector<int64_t> &keys) {
        for (const auto key : keys) {
            SCOPED_TRACE("key=" + std::to_string(key));
            auto [ec, locations] = manager.MatchLocation(
                prefix + "_miss", QueryType::QT_PREFIX_MATCH, {key}, {}, BlockMaskOffset{0}, 0, {}, hints);
            ASSERT_EQ(ER_OK, ec);
            EXPECT_TRUE(locations.empty());
            EXPECT_TRUE(hints.empty());
        }
    };
    expect_miss({1, 2, 3});
    ASSERT_FALSE(HasFailure());

    // Each TP worker transfers its own spec; every block/spec has a different byte pattern.
    std::array<std::array<std::vector<char>, kBlocks>, kSpecs> expected;
    Locations published = write.locations;
    for (size_t spec = 0; spec < kSpecs; ++spec) {
        UriStrVec uris;
        BlockBuffers buffers;
        for (size_t block = 0; block < kBlocks; ++block) {
            auto &bytes = expected[spec][block];
            bytes.resize(kBytes);
            for (size_t offset = 0; offset < kBytes; ++offset) {
                bytes[offset] = static_cast<char>((offset * 17 + block * 31 + spec * 67) % 251 + 1);
            }
            uris.push_back(write.locations[block][spec].uri);
            buffers.push_back({{Iov{MemoryType::CPU, bytes.data(), bytes.size(), false}}});
        }
        auto [save_ec, saved_uris] = clients[spec]->SaveKvCaches(uris, buffers);
        ASSERT_EQ(ER_OK, save_ec);
        ASSERT_EQ(kBlocks, saved_uris.size());
        for (size_t block = 0; block < kBlocks; ++block) {
            ASSERT_FALSE(saved_uris[block].empty());
            published[block][spec].uri = saved_uris[block];
        }
    }
    expect_miss({1, 2, 3}); // Saved bytes remain invisible until FinishWrite publishes all specs.
    ASSERT_FALSE(HasFailure());
    ASSERT_EQ(
        ER_OK,
        manager.FinishWrite(prefix + "_finish", write.write_session_id, BlockMaskVector{true, true, false}, published));
    expect_miss({3}); // A failed block must stay unreadable even though both specs were saved.
    ASSERT_FALSE(HasFailure());

    auto [match_ec, matched] = manager.MatchLocation(
        prefix + "_match", QueryType::QT_PREFIX_MATCH, {1, 2, 3}, {}, BlockMaskOffset{0}, 0, {}, hints);
    ASSERT_EQ(ER_OK, match_ec);
    ExpectLocationsEq(Locations(published.begin(), published.begin() + kPublishedBlocks), matched);
    ASSERT_FALSE(HasFailure());
    for (size_t spec = 0; spec < kSpecs; ++spec) {
        UriStrVec uris;
        BlockBuffers buffers;
        std::array<std::vector<char>, kPublishedBlocks> actual;
        for (size_t block = 0; block < kPublishedBlocks; ++block) {
            actual[block].resize(kBytes, 0);
            uris.push_back(matched[block][spec].uri);
            buffers.push_back({{Iov{MemoryType::CPU, actual[block].data(), kBytes, false}}});
        }
        ASSERT_EQ(ER_OK, clients[spec]->LoadKvCaches(uris, buffers));
        for (size_t block = 0; block < kPublishedBlocks; ++block) {
            EXPECT_EQ(expected[spec][block], actual[block]) << "spec=" << spec << " block=" << block;
        }
    }
}

// --- Error detail tests for the latest commit ---

TEST_F(ManagerClientTest, TestRegisterInstanceErrorDetails) {
    auto prefix = GetCurrentTestName();

    // Part 1: duplicate instance with different model_deployment
    {
        std::string instance_id = prefix + "_dup_instance";

        // First registration with model "test_model"
        {
            auto config = createClientConfigWithModel(instance_id, "test_model");
            auto client = ManagerClient::Create(config, init_params_);
            ASSERT_NE(nullptr, client.get());
        }

        // Second registration with a different model name
        {
            auto config = createClientConfigWithModel(instance_id, "different_model");
            auto client = ManagerClient::Create(config, init_params_);
            EXPECT_EQ(nullptr, client.get())
                << "Re-registration with different model should fail during client creation";
        }
    }

    // Part 2: non-existent instance_group
    {
        std::string instance_id = prefix + "_nogroup_instance";
        auto config = createClientConfigWithGroup(instance_id, "nonexistent_group_xyz");
        auto client = ManagerClient::Create(config, init_params_);
        EXPECT_EQ(nullptr, client.get()) << "Registration with non-existent group should fail during client creation";
    }
}
