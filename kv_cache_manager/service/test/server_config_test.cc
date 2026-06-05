#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/service/server_config.h"

using namespace kv_cache_manager;

class ServerConfigTest : public TESTBASE {
public:
    void SetUp() override {}

    void TearDown() override {}

public:
};

TEST_F(ServerConfigTest, TestSimple) {
    // empty
    {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ;
        ASSERT_TRUE(config.Parse("", environ));
        ASSERT_FALSE(config.Check());
    }
    // config_file not exist
    {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ;
        ASSERT_FALSE(config.Parse(GetPrivateTestDataPath() + "not_exist_config_file.conf", environ));
    }
    // from config_file
    {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ;
        std::string config_file = GetPrivateTestDataPath() + "server_config_simple.conf";
        ASSERT_TRUE(config.Parse(config_file, environ));
        ASSERT_TRUE(config.Check());
        ASSERT_EQ("redis://127.0.0.1:6379?auth=123456", config.GetRegistryStorageUri());
        ASSERT_EQ(6381, config.GetServiceRpcPort());
        ASSERT_EQ(6382, config.GetServiceHttpPort());
        ASSERT_EQ(2, config.GetServiceIoThreadNum());
        ASSERT_TRUE(config.IsEnableDebugService());
    }
    // from environ
    {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ;
        environ.insert({"kvcm.service.rpc_port", "6381"});
        environ.insert({"kvcm.service.http_port", "6382"});
        ASSERT_TRUE(config.Parse("", environ));
        ASSERT_FALSE(config.Check());
        environ.insert({"kvcm.registry_storage.uri", "redis://127.0.0.1:6379?auth=123456"});
        ASSERT_TRUE(config.Parse("", environ));
        ASSERT_TRUE(config.Check());
        ASSERT_EQ("redis://127.0.0.1:6379?auth=123456", config.GetRegistryStorageUri());
        ASSERT_EQ(6381, config.GetServiceRpcPort());
        ASSERT_EQ(6382, config.GetServiceHttpPort());
        ASSERT_EQ(0, config.GetServiceIoThreadNum());
        ASSERT_FALSE(config.IsEnableDebugService());
    }
    // from config_file + environ
    {
        ServerConfig config;
        std::string config_file = GetPrivateTestDataPath() + "server_config_simple.conf";
        std::unordered_map<std::string, std::string> environ;
        environ.insert({"kvcm.service.rpc_port", "7381"});
        ScopedEnv env("kvcm.service.http_port", "7382");
        environ.insert({"kvcm.service.io_thread_num", "4"});
        environ.insert({"kvcm.logger.log_level", "3"});
        ASSERT_TRUE(config.Parse(config_file, environ));
        ASSERT_TRUE(config.Check());
        ASSERT_EQ("redis://127.0.0.1:6379?auth=123456", config.GetRegistryStorageUri());
        ASSERT_EQ(7381, config.GetServiceRpcPort());
        ASSERT_EQ(7382, config.GetServiceHttpPort());
        ASSERT_EQ(4, config.GetServiceIoThreadNum());
        ASSERT_TRUE(config.IsEnableDebugService());
        ASSERT_EQ(3, config.GetLogLevel());
    }
}

TEST_F(ServerConfigTest, RaftConfigParsing) {
    ServerConfig config;
    std::unordered_map<std::string, std::string> environ;
    environ["kvcm.registry_storage.uri"] = "redis://127.0.0.1:6379";
    environ["kvcm.raft.server_id"] = "3";
    environ["kvcm.raft.host"] = "10.0.0.1";
    environ["kvcm.raft.port"] = "9100";
    environ["kvcm.raft.peers"] = "1:10.0.0.1:9100,2:10.0.0.2:9100,3:10.0.0.3:9100";
    environ["kvcm.raft.data_dir"] = "/data/raft";
    environ["kvcm.raft.snapshot_distance"] = "50000";
    environ["kvcm.raft.election_timeout_lower"] = "500";
    environ["kvcm.raft.election_timeout_upper"] = "1000";
    environ["kvcm.raft.heart_beat_interval"] = "200";
    ASSERT_TRUE(config.Parse("", environ));
    ASSERT_TRUE(config.Check());
    ASSERT_TRUE(config.IsRaftEnabled());
    ASSERT_EQ(3, config.GetRaftServerId());
    ASSERT_EQ("10.0.0.1", config.GetRaftHost());
    ASSERT_EQ(9100, config.GetRaftPort());
    ASSERT_EQ(3u, config.GetRaftPeers().size());
    EXPECT_EQ(1, config.GetRaftPeers()[0].server_id);
    EXPECT_EQ("10.0.0.1", config.GetRaftPeers()[0].host);
    EXPECT_EQ(9100, config.GetRaftPeers()[0].port);
    EXPECT_EQ(2, config.GetRaftPeers()[1].server_id);
    ASSERT_EQ("/data/raft", config.GetRaftDataDir());
    ASSERT_EQ(50000, config.GetRaftSnapshotDistance());
    ASSERT_EQ(500, config.GetRaftElectionTimeoutLower());
    ASSERT_EQ(1000, config.GetRaftElectionTimeoutUpper());
    ASSERT_EQ(200, config.GetRaftHeartBeatInterval());
}

TEST_F(ServerConfigTest, RaftDisabledByDefault) {
    ServerConfig config;
    std::unordered_map<std::string, std::string> environ;
    environ["kvcm.registry_storage.uri"] = "redis://127.0.0.1:6379";
    ASSERT_TRUE(config.Parse("", environ));
    ASSERT_FALSE(config.IsRaftEnabled());
}

TEST_F(ServerConfigTest, RaftCheckFailsWithInvalidServerId) {
    ServerConfig config;
    std::unordered_map<std::string, std::string> environ;
    environ["kvcm.registry_storage.uri"] = "redis://127.0.0.1:6379";
    environ["kvcm.raft.data_dir"] = "/data/raft";
    environ["kvcm.raft.port"] = "9100";
    // server_id defaults to 0 which is invalid
    ASSERT_TRUE(config.Parse("", environ));
    ASSERT_FALSE(config.Check());
}

TEST_F(ServerConfigTest, TestUnderscoreEnvFallback) {
    // 仅设置下划线版本，验证 fallback 生效
    {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ;
        ScopedEnv env1("kvcm_registry_storage_uri", "redis://127.0.0.1:6379?auth=abc");
        ScopedEnv env2("kvcm_service_rpc_port", "7381");
        ASSERT_TRUE(config.Parse("", environ));
        ASSERT_TRUE(config.Check());
        ASSERT_EQ("redis://127.0.0.1:6379?auth=abc", config.GetRegistryStorageUri());
        ASSERT_EQ(7381, config.GetServiceRpcPort());
    }
    // 同时设置 dotted 和 underscore 版本，验证 dotted 优先
    {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ;
        ScopedEnv env1("kvcm.registry_storage.uri", "redis://dotted-wins");
        ScopedEnv env2("kvcm_registry_storage_uri", "redis://underscore-loses");
        ASSERT_TRUE(config.Parse("", environ));
        ASSERT_TRUE(config.Check());
        ASSERT_EQ("redis://dotted-wins", config.GetRegistryStorageUri());
    }
}
