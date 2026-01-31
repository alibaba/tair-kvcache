#include <memory>

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
        auto [success, locations_map] = manager_client->MatchLocation(
            prefix + "_2", QueryType::QT_PREFIX_MATCH, {1, 2, 3, 4}, {}, static_cast<size_t>(0), 0, {});
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
    auto manager_client = CreateClient<ManagerClient>(prefix, init_params_);
    ASSERT_NE(nullptr, manager_client.get());

    std::string write_session_id;
    {
        auto [write_success, write_location] = manager_client->StartWrite(prefix + "_1", {1, 2}, {}, {}, 1000000);
        ASSERT_EQ(ER_OK, write_success);
        write_session_id = write_location.write_session_id;

        // TODO:wait support set nfs backend root path
        //  BlockBuffer buffer1, buffer2;
        //  BlockBuffers block_buffers = {&buffer1, &buffer2};
        //  Locations locations = write_location.locations_map.at({0,0});
        //  ASSERT_EQ(2, locations.size());

        // auto write_result = manager_client->SaveKvCaches(locations, block_buffers);
        // ASSERT_TRUE(write_result.first);
        // ASSERT_EQ(2, write_result.second.size());
    }
    {
        BlockMask success_block = static_cast<size_t>(2);
        ASSERT_EQ(ER_OK, manager_client->FinishWrite(prefix + "_2", write_session_id, success_block, {}));
    }
    {
        auto [match_success, read_location] = manager_client->MatchLocation(
            prefix + "_3", QueryType::QT_PREFIX_MATCH, {1, 2}, {}, static_cast<size_t>(0), 0, {});
        ASSERT_EQ(ER_OK, match_success);

        // BlockBuffer buffer1, buffer2;
        // BlockBuffers block_buffers = {&buffer1, &buffer2};
        // auto read_result = manager_client->LoadKvCaches(locations, block_buffers);
        // ASSERT_TRUE(read_result.first);
    }
}
