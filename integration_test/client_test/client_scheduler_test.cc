#include <memory>

#include "client_test_base.h"
#include "kv_cache_manager/client/include/meta_client.h"
#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/logger.h"

using namespace ::testing;
using namespace kv_cache_manager;

class ClientSchedulerTest : public CLIENTTESTBASE {
public:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(ClientSchedulerTest, TestCreateWithInvalidRoleType) {
    std::array<char, 2048> buffer;
    kv_cache_manager::InitParams init_params = {RoleType::WORKER, nullptr, "tp0"};
    int n = std::snprintf(buffer.data(),
                          buffer.size(),
                          R"({
"instance_group": "default",
"instance_id": "test_instance",
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
                          controller_.rpc_port());
    std::string client_config = std::string(buffer.data(), n);
    auto client = MetaClient::Create(client_config, init_params);
    ASSERT_EQ(client, nullptr);
}

TEST_F(ClientSchedulerTest, TestCreateWithTpSizeInvalid) {
    std::array<char, 2048> buffer;
    int n = std::snprintf(buffer.data(),
                          buffer.size(),
                          R"({
"instance_group": "default",
"instance_id": "test_instance",
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
    "tp_size": -1,
    "dp_size": 1,
    "pp_size": 1,
    "pp_infos": [
        "layer0"
    ]
}
})",
                          controller_.rpc_port());
    std::string client_config = std::string(buffer.data(), n);
    auto client = MetaClient::Create(client_config, init_params_);
    ASSERT_EQ(client, nullptr);
}

TEST_F(ClientSchedulerTest, TestCreateWithEmptyAddress) {
    std::string client_config = R"({
            "instance_group": "group",
            "instance_id": "instance",
            "sdk_config": {},
            "block_size": 128,
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
    auto client = MetaClient::Create(client_config, init_params_);
    ASSERT_EQ(client, nullptr);
}

TEST_F(ClientSchedulerTest, TestStartWriteCache) {
    auto prefix = GetCurrentTestName();
    auto meta_client = CreateClient<MetaClient>(prefix, init_params_);
    ASSERT_NE(nullptr, meta_client.get());
    TestStartWrite<MetaClient>(prefix, meta_client);
}

TEST_F(ClientSchedulerTest, TestFinishWriteCacheSuccess) {
    auto prefix = GetCurrentTestName();
    auto meta_client = CreateClient<MetaClient>(prefix, init_params_);
    ASSERT_NE(nullptr, meta_client.get());
    TestFinishWriteSuccess<MetaClient>(prefix, meta_client);
}

TEST_F(ClientSchedulerTest, TestFinishWriteCacheFail) {
    auto prefix = GetCurrentTestName();
    auto meta_client = CreateClient<MetaClient>(prefix, init_params_);
    ASSERT_NE(nullptr, meta_client.get());
    TestFinishWriteFail<MetaClient>(prefix, meta_client);
}

TEST_F(ClientSchedulerTest, TestGetCacheLocation) {
    auto prefix = GetCurrentTestName();
    auto meta_client = CreateClient<MetaClient>(prefix, init_params_);
    ASSERT_NE(nullptr, meta_client.get());
    TestGetCacheLocation<MetaClient>(prefix, meta_client);
}