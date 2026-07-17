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
    // empty config is valid (registry_storage_uri is optional, falls back to local backend)
    {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ;
        ASSERT_TRUE(config.Parse("", environ));
        ASSERT_TRUE(config.Check());
        ASSERT_EQ(60000, config.GetCacheReclaimerInflightDeleteTimeoutMs());
        ASSERT_EQ(100000, config.GetCacheReclaimerPendingLocationLimitPerGroupType());
        ASSERT_EQ(64ULL * 1024 * 1024 * 1024, config.GetCacheReclaimerPendingBytesLimitPerGroupType());
        ASSERT_EQ(1024, config.GetCacheReclaimerPendingDeleteHandlerLimit());
        ASSERT_EQ(256ULL * 1024 * 1024 * 1024, config.GetCacheReclaimerPendingBytesLimit());
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
        ASSERT_TRUE(config.Check()); // registry_storage_uri is optional
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

TEST_F(ServerConfigTest, TestCacheReclaimerAsyncDeleteConfig) {
    ServerConfig config;
    std::unordered_map<std::string, std::string> environ{
        {"kvcm.cache_reclaimer.inflight_delete_timeout_ms", "1234"},
        {"kvcm.cache_reclaimer.pending_location_limit_per_group_type", "11"},
        {"kvcm.cache_reclaimer.pending_bytes_limit_per_group_type", "22"},
        {"kvcm.cache_reclaimer.pending_delete_handler_limit", "33"},
        {"kvcm.cache_reclaimer.pending_bytes_limit", "44"},
    };
    ASSERT_TRUE(config.Parse("", environ));
    ASSERT_TRUE(config.Check());
    EXPECT_EQ(1234, config.GetCacheReclaimerInflightDeleteTimeoutMs());
    EXPECT_EQ(11, config.GetCacheReclaimerPendingLocationLimitPerGroupType());
    EXPECT_EQ(22, config.GetCacheReclaimerPendingBytesLimitPerGroupType());
    EXPECT_EQ(33, config.GetCacheReclaimerPendingDeleteHandlerLimit());
    EXPECT_EQ(44, config.GetCacheReclaimerPendingBytesLimit());

    environ["kvcm.cache_reclaimer.pending_delete_handler_limit"] = "0";
    ASSERT_TRUE(config.Parse("", environ));
    EXPECT_FALSE(config.Check());
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
