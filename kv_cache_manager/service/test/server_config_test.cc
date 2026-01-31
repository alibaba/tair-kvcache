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
