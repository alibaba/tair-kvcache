#include <cstring>
#include <filesystem>
#include <fstream>
#include <gmock/gmock.h>

#include "kv_cache_manager/client/src/internal/stub/stub.h"
#include "kv_cache_manager/client/src/meta_client_impl.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

class MetaClientTest : public TESTBASE {
public:
    void SetUp() override {
        LoggerBroker::InitLoggerForClientOnce();
        root_path_ = GetPrivateTestRuntimeDataPath();
        client_config_ = R"({
            "instance_group": "test_group",
            "instance_id": "test_instance",
            "address": [
                "127.0.0.1:8080",
                "127.0.0.1:9191"
            ],
            "block_size": 16,
            "location_spec_infos": {
                "tp0": 1024
            },
            "model_deployment": {
                "model_name": "test_model",
                "dtype": "FP8",
                "use_mla": false,
                "tp_size": 1,
                "dp_size": 1,
                "pp_size": 1
            },
            "location_spec_groups": {
                "group0": ["tp0"]
            }
        })";
        init_params_.role_type = RoleType::HYBRID;
    }

    void TearDown() override {}

protected:
    std::string root_path_;
    std::string client_config_;
    InitParams init_params_;
};

class MockStub : public Stub {
public:
    using KeyType = Stub::KeyType;
    using KeyVector = Stub::KeyVector;
    using TokenIds = Stub::TokenIds;
    using TokenIdsVector = Stub::TokenIdsVector;
    using LocationSpecInfoMap = Stub::LocationSpecInfoMap;
    using LocationSpecGroups = Stub::LocationSpecGroups;

    MOCK_METHOD(ClientErrorCode, AddConnection, (const std::string &address, uint32_t connection_timeout), (override));

    MOCK_METHOD(void, RemoveAllConnections, (), (override));

    MOCK_METHOD((std::pair<ClientErrorCode, std::string>),
                RegisterInstance,
                (const std::string &trace_id,
                 const std::string instance_group,
                 const std::string &instance_id,
                 int32_t block_size,
                 const LocationSpecInfoMap &location_spec_infos,
                 const ModelDeployment &model_deployment,
                 const LocationSpecGroups &location_spec_groups),
                (override));

    MOCK_METHOD((std::pair<ClientErrorCode, InstanceInfo>),
                GetInstanceInfo,
                (const std::string &trace_id, const std::string &instance_id),
                (override));

    MOCK_METHOD((std::pair<ClientErrorCode, Metas>),
                GetCacheMeta,
                (const std::string &trace_id,
                 const std::string &instance_id,
                 const KeyVector &keys,
                 const TokenIdsVector &tokens,
                 const BlockMask &block_mask,
                 int32_t detail_level),
                (override));

    MOCK_METHOD((std::pair<ClientErrorCode, Locations>),
                GetCacheLocation,
                (const std::string &trace_id,
                 const std::string &instance_id,
                 QueryType query_type,
                 const KeyVector &keys,
                 const TokenIdsVector &tokens,
                 const BlockMask &block_mask,
                 int32_t sw_size,
                 const std::vector<std::string> &location_spec_names),
                (override));

    MOCK_METHOD((std::pair<ClientErrorCode, WriteLocation>),
                StartWriteCache,
                (const std::string &trace_id,
                 const std::string &instance_id,
                 const KeyVector &keys,
                 const TokenIdsVector &tokens,
                 const std::vector<std::string> &location_spec_group_names,
                 int64_t write_timeout_seconds),
                (override));

    MOCK_METHOD(ClientErrorCode,
                FinishWriteCache,
                (const std::string &trace_id,
                 const std::string &instance_id,
                 const std::string write_session_id,
                 const BlockMask &success_block,
                 const Locations &locations),
                (override));

    MOCK_METHOD(ClientErrorCode,
                RemoveCache,
                (const std::string &trace_id,
                 const std::string &instance_id,
                 const KeyVector &keys,
                 const TokenIdsVector &tokens,
                 const BlockMask &block_mask),
                (override));

    MOCK_METHOD(bool, TrimCache, (), (override));
};

TEST_F(MetaClientTest, TestCreateSimple) {
    auto mock_stub = new MockStub();
    auto client = std::make_unique<MetaClientImpl>();
    client->stub_.reset(mock_stub);
    // 期望 AddConnection 被调用一次，地址是 "127.0.0.1:8080"，timeout 任意，返回 ER_OK
    EXPECT_CALL(*mock_stub, AddConnection("127.0.0.1:8080", ::testing::_)).Times(1).WillOnce(::testing::Return(ER_OK));
    // 期望 RegisterInstance 被调用一次，参数不逐个校验（使用 _），返回 (ER_OK, "session_id")
    EXPECT_CALL(*mock_stub,
                RegisterInstance(
                    ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1)
        .WillOnce(::testing::Return(std::make_pair(ER_OK, std::string("{fake_storage_config}"))));
    auto ec = client->Init(client_config_, init_params_);
    ASSERT_EQ(ec, ER_OK);
}

TEST_F(MetaClientTest, TestCreateWithLeaderMode) {
    auto mock_stub = new MockStub();
    auto client = std::make_unique<MetaClientImpl>();
    client->stub_.reset(mock_stub);
    // 127.0.0.1:9191是主，127.0.0.1:8080是从，双方都能正常连接上
    EXPECT_CALL(*mock_stub, AddConnection("127.0.0.1:8080", ::testing::_)).Times(1).WillOnce(::testing::Return(ER_OK));
    EXPECT_CALL(*mock_stub, AddConnection("127.0.0.1:9191", ::testing::_)).Times(1).WillOnce(::testing::Return(ER_OK));
    EXPECT_CALL(*mock_stub, RemoveAllConnections()).Times(1).WillOnce(::testing::Return());
    // 当向从节点RegisterInstance时，会返回非主错误
    ::testing::Sequence call_register_seq;
    EXPECT_CALL(*mock_stub,
                RegisterInstance(
                    ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .InSequence(call_register_seq)
        .WillOnce(::testing::Return(std::make_pair(ER_SERVICE_NOT_LEADER, std::string("{}"))));
    EXPECT_CALL(*mock_stub,
                RegisterInstance(
                    ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .InSequence(call_register_seq)
        .WillOnce(::testing::Return(std::make_pair(ER_OK, std::string("{fake_storage_config}"))));
    auto ec = client->Init(client_config_, init_params_);
    ASSERT_EQ(ec, ER_OK);
}

// 测试所有地址都不是Leader的情况
TEST_F(MetaClientTest, TestCreateWithAllNotLeader) {
    auto mock_stub = new MockStub();
    auto client = std::make_unique<MetaClientImpl>();
    client->stub_.reset(mock_stub);

    // 期望所有地址都被尝试连接
    EXPECT_CALL(*mock_stub, AddConnection("127.0.0.1:8080", ::testing::_)).Times(1).WillOnce(::testing::Return(ER_OK));
    EXPECT_CALL(*mock_stub, AddConnection("127.0.0.1:9191", ::testing::_)).Times(1).WillOnce(::testing::Return(ER_OK));

    // 期望每次RegisterInstance都返回非Leader错误
    EXPECT_CALL(*mock_stub, RemoveAllConnections()).Times(2).WillRepeatedly(::testing::Return());
    EXPECT_CALL(*mock_stub,
                RegisterInstance(
                    ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(2)
        .WillRepeatedly(::testing::Return(std::make_pair(ER_SERVICE_NOT_LEADER, std::string("{}"))));

    auto ec = client->Init(client_config_, init_params_);
    // 根据当前实现，如果所有节点都不是Leader，最终会返回ER_SERVICE_NOT_LEADER
    ASSERT_EQ(ec, ER_SERVICE_NOT_LEADER);
}

// 测试连接失败的情况
TEST_F(MetaClientTest, TestCreateWithConnectFail) {
    auto mock_stub = new MockStub();
    auto client = std::make_unique<MetaClientImpl>();
    client->stub_.reset(mock_stub);

    // 第一个地址连接失败
    EXPECT_CALL(*mock_stub, AddConnection("127.0.0.1:8080", ::testing::_))
        .Times(1)
        .WillOnce(::testing::Return(ER_CONNECT_FAIL));

    auto ec = client->Init(client_config_, init_params_);
    // 期望返回连接失败错误
    ASSERT_EQ(ec, ER_CONNECT_FAIL);
}

// 测试RegisterInstance返回其他错误的情况
TEST_F(MetaClientTest, TestCreateWithRegisterError) {
    auto mock_stub = new MockStub();
    auto client = std::make_unique<MetaClientImpl>();
    client->stub_.reset(mock_stub);

    // 第一个地址连接成功
    EXPECT_CALL(*mock_stub, AddConnection("127.0.0.1:8080", ::testing::_)).Times(1).WillOnce(::testing::Return(ER_OK));

    // RegisterInstance返回内部错误
    EXPECT_CALL(*mock_stub,
                RegisterInstance(
                    ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1)
        .WillOnce(::testing::Return(std::make_pair(ER_SERVICE_INTERNAL_ERROR, std::string("{}"))));

    auto ec = client->Init(client_config_, init_params_);
    // 期望返回内部错误
    ASSERT_EQ(ec, ER_SERVICE_INTERNAL_ERROR);
}

// 测试单个地址的情况（非主从模式）
TEST_F(MetaClientTest, TestCreateSingleAddress) {
    // 修改配置为单个地址
    client_config_ = R"({
        "instance_group": "test_group",
        "instance_id": "test_instance",
        "address": [
            "127.0.0.1:8080"
        ],
        "block_size": 16,
        "location_spec_infos": {
            "tp0": 1024
        },
        "model_deployment": {
            "model_name": "test_model",
            "dtype": "FP8",
            "use_mla": false,
            "tp_size": 1,
            "dp_size": 1,
            "pp_size": 1
        },
        "location_spec_groups": {
            "group0": ["tp0"]
        }
    })";

    auto mock_stub = new MockStub();
    auto client = std::make_unique<MetaClientImpl>();
    client->stub_.reset(mock_stub);

    // 期望只调用一次AddConnection
    EXPECT_CALL(*mock_stub, AddConnection("127.0.0.1:8080", ::testing::_)).Times(1).WillOnce(::testing::Return(ER_OK));

    // 期望只调用一次RegisterInstance并返回成功
    EXPECT_CALL(*mock_stub,
                RegisterInstance(
                    ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(1)
        .WillOnce(::testing::Return(std::make_pair(ER_OK, std::string("{fake_storage_config}"))));

    auto ec = client->Init(client_config_, init_params_);
    ASSERT_EQ(ec, ER_OK);
}