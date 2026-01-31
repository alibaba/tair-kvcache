#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "kv_cache_manager/client/include/manager_client.h"
#include "kv_cache_manager/client/include/meta_client.h"
#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/unittest.h"
#include "service_process_controller.h"

using namespace ::testing;
using namespace kv_cache_manager;

template <typename T>
using ClientPtr = std::shared_ptr<T>;

class CLIENTTESTBASE : public TESTBASE {
public:
    static void SetUpTestSuite() {
        workspace_path_ = std::getenv("TEST_SRCDIR");
        workspace_path_ /= std::getenv("TEST_WORKSPACE");
        controller_.StartServiceInSubProcess(workspace_path_);
    }

    static void TearDownTestSuite() { controller_.StopService(); }
    std::string GetCurrentTestName() const {
        const ::testing::TestInfo *const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        return std::string(test_info->test_case_name()) + "_" + test_info->name();
    }

    static void ExpectLocationsEq(const Locations &a, const Locations &b) {
        EXPECT_EQ(a.size(), b.size());
        size_t location_size = std::min(a.size(), b.size());
        for (size_t i = 0; i < location_size; i++) {
            EXPECT_EQ(a[i].size(), b[i].size());
            size_t spec_size = std::min(a[i].size(), b[i].size());
            for (size_t j = 0; j < spec_size; j++) {
                EXPECT_EQ(a[i][j].spec_name, b[i][j].spec_name);
                EXPECT_EQ(a[i][j].uri, b[i][j].uri);
            }
        }
    }
    static void ExpectLocationsValid(const Locations &locations, size_t location_count) {
        EXPECT_EQ(location_count, locations.size());
        for (auto &location : locations) {
            EXPECT_EQ(2, location.size());
            EXPECT_EQ(location[0].spec_name, "tp0");
            EXPECT_GT(location[0].uri.size(), 0);
            EXPECT_EQ(location[1].spec_name, "tp1");
            EXPECT_GT(location[1].uri.size(), 0);
        }
    }

protected:
    std::string createClientConfig(const std::string &instance_id) const {
        std::array<char, 2048> buffer;
        int n = std::snprintf(buffer.data(),
                              buffer.size(),
                              R"({
"instance_group": "default",
"instance_id": "%s",
"address": [
    "127.0.0.1:%d"
],
"block_size": 128,
"location_spec_infos":{
    "tp0": 1024,
    "tp1": 1024
},
"sdk_config": {},
"model_deployment": {
    "model_name": "test_model",
    "dtype": "FP8",
    "use_mla": false,
    "tp_size": 2,
    "dp_size": 1,
    "pp_size": 1,
    "pp_infos": [
        "layer0"
    ]
}
})",
                              instance_id.c_str(),
                              controller_.rpc_port());
        return std::string(buffer.data(), n);
    }

    template <typename ClientType>
    ClientPtr<ClientType> CreateClient(const std::string &prefix, InitParams &init_params) {
        auto client = ClientType::Create(createClientConfig(prefix + "_instance"), init_params);
        return client;
    }

    template <typename ClientType>
    void TestStartWrite(const std::string &prefix, const ClientPtr<ClientType> &client) {
        {
            auto [success, write_location] = client->StartWrite(prefix + "_1", {1, 2, 3, 4}, {}, {}, 10000000);
            ASSERT_EQ(ER_OK, success);
            ASSERT_EQ(0, std::get<BlockMaskOffset>(write_location.block_mask));
            ExpectLocationsValid(write_location.locations, 4);
            ASSERT_FALSE(HasFailure());
        }
        {
            auto [success, write_location] = client->StartWrite(prefix + "_2", {1, 2, 3, 4}, {}, {}, 1000000);
            ASSERT_EQ(ER_OK, success);
            ASSERT_EQ(4, std::get<BlockMaskOffset>(write_location.block_mask));
            ASSERT_EQ(Locations({}), write_location.locations);
        }
        {
            auto [success, write_location] = client->StartWrite(prefix + "_3", {1, 2, 3, 4, 5, 6}, {}, {}, 1000000);
            ASSERT_EQ(ER_OK, success);
            ASSERT_EQ(4, std::get<BlockMaskOffset>(write_location.block_mask));
            ExpectLocationsValid(write_location.locations, 2);
            ASSERT_FALSE(HasFailure());
        }
        {
            auto [success, write_location] = client->StartWrite(prefix + "_4", {5, 6, 7, 8}, {}, {}, 1000000);
            ASSERT_EQ(ER_OK, success);
            ASSERT_EQ(2, std::get<BlockMaskOffset>(write_location.block_mask));
            ExpectLocationsValid(write_location.locations, 2);
            ASSERT_FALSE(HasFailure());
        }
        {
            auto [success, write_location] = client->StartWrite(prefix + "_5", {5, 55, 66, 6}, {}, {}, 1000000);
            ASSERT_EQ(ER_OK, success);
            ASSERT_EQ(BlockMaskVector({true, false, false, true}),
                      std::get<BlockMaskVector>(write_location.block_mask));
            ExpectLocationsValid(write_location.locations, 2);
            ASSERT_FALSE(HasFailure());
        }
    }

    template <typename ClientType>
    void TestFinishWriteSuccess(const std::string &prefix, const ClientPtr<ClientType> &client) {
        std::string write_session_id;
        {
            auto [success, write_location] = client->StartWrite(prefix + "_1", {1, 2, 3, 4}, {}, {}, 1000000);
            ASSERT_EQ(ER_OK, success);
            write_session_id = write_location.write_session_id;
        }

        {
            BlockMask success_block = static_cast<size_t>(4);
            ASSERT_EQ(ER_OK, client->FinishWrite(prefix + "_2", write_session_id, success_block, {}));
        }
    }

    template <typename ClientType>
    void TestFinishWriteFail(const std::string &prefix, const ClientPtr<ClientType> &client) {
        std::string write_session_id;
        {
            auto [success, write_location] = client->StartWrite(prefix + "_1", {1, 2, 3, 4}, {}, {}, 1000000);
            ASSERT_EQ(ER_OK, success);
            write_session_id = write_location.write_session_id;
        }
        {
            BlockMask success_block = static_cast<size_t>(4);
            EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR,
                      client->FinishWrite(prefix + "_2", write_session_id + "_bug", success_block, {}));
        }
        {
            BlockMask success_block = BlockMaskVector(2, true);
            EXPECT_EQ(ER_SERVICE_INVALID_ARGUMENT,
                      client->FinishWrite(prefix + "_3", write_session_id, success_block, {}));
        }
        {
            BlockMask success_block = static_cast<size_t>(6);
            EXPECT_EQ(ER_SERVICE_INTERNAL_ERROR,
                      client->FinishWrite(prefix + "_4", write_session_id, success_block, {}));
        }
    }

    template <typename ClientType>
    void TestGetCacheLocation(const std::string &prefix, const ClientPtr<ClientType> &client) {
        std::string write_session_id;
        Locations target_locations;
        {
            auto [success, write_location] = client->StartWrite(prefix + "_1", {1, 2, 3, 4}, {}, {}, 1000000);
            ASSERT_EQ(ER_OK, success);
            write_session_id = write_location.write_session_id;
            target_locations = write_location.locations;
        }
        {
            auto [success, locations] = client->MatchLocation(
                prefix + "_2", QueryType::QT_PREFIX_MATCH, {1, 2, 3, 4}, {}, static_cast<size_t>(0), 0, {});
            ASSERT_EQ(ER_OK, success);
            ASSERT_EQ(Locations({}), locations);
        }
        {
            BlockMask success_block = BlockMaskVector({true, true, false, false});
            ASSERT_EQ(ER_OK, client->FinishWrite(prefix + "_2", write_session_id, success_block, {}));
        }
        {
            auto [success, locations] = client->MatchLocation(
                prefix + "_3", QueryType::QT_PREFIX_MATCH, {1, 2, 3, 4}, {}, static_cast<size_t>(0), 0, {});
            ASSERT_EQ(ER_OK, success);
            ExpectLocationsEq(Locations{target_locations[0], target_locations[1]}, locations);
            ASSERT_FALSE(HasFailure());
        }
        {
            auto [success, locations] = client->MatchLocation(
                prefix + "_4", QueryType::QT_PREFIX_MATCH, {1, 2, 3}, {}, static_cast<size_t>(1), 0, {});
            ASSERT_EQ(ER_OK, success);
            ExpectLocationsEq(Locations{target_locations[1]}, locations);
            ASSERT_FALSE(HasFailure());
        }
        {
            auto [success, locations] = client->MatchLocation(
                prefix + "_5", QueryType::QT_PREFIX_MATCH, {1, 2, 3}, {}, static_cast<size_t>(6), 0, {});
            ASSERT_EQ(ER_OK, success);
            ASSERT_EQ(Locations({}), locations);
        }
        {
            BlockMask block_mask = BlockMaskVector({true, false, false, false});
            auto [success, locations] =
                client->MatchLocation(prefix + "_6", QueryType::QT_PREFIX_MATCH, {1, 2, 3}, {}, block_mask, 0, {});
            ASSERT_EQ(ER_OK, success);
            ExpectLocationsEq(Locations{target_locations[1]}, locations);
            ASSERT_FALSE(HasFailure());
        }
    }

    static ServiceProcessController controller_;
    static std::filesystem::path workspace_path_;
    kv_cache_manager::InitParams init_params_ = {RoleType::HYBRID, nullptr, "tp0"};
};