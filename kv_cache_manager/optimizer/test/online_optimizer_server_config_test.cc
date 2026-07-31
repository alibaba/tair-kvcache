#include <unordered_map>

#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/service/online_optimizer_server_config.h"

namespace kv_cache_manager {

class OnlineOptimizerServerConfigTest : public TESTBASE {};

TEST_F(OnlineOptimizerServerConfigTest, DefaultValues) {
    OnlineOptimizerServerConfig config;
    EXPECT_EQ(50052, config.rpc_port());
    EXPECT_EQ(8082, config.http_port());
    EXPECT_TRUE(config.registry_storage_uri().empty());
    EXPECT_EQ("local", config.metrics_reporter_type());
    EXPECT_EQ(10000, config.metrics_report_interval_ms());
    EXPECT_TRUE(config.enable_prometheus());
    EXPECT_EQ("kvcm_optimizer", config.prometheus_prefix());
    EXPECT_EQ(4, config.io_thread_num());
}

TEST_F(OnlineOptimizerServerConfigTest, ParseFromJson) {
    std::string json = R"({
        "rpc_port": 50053,
        "http_port": 8083,
        "registry_storage_uri": "file:///tmp/test",
        "metrics_reporter_type": "prometheus",
        "metrics_report_interval_ms": 5000,
        "enable_prometheus": false,
        "prometheus_prefix": "my_prefix",
        "io_thread_num": 8
    })";

    OnlineOptimizerServerConfig config;
    ASSERT_TRUE(config.FromJsonString(json));
    EXPECT_EQ(50053, config.rpc_port());
    EXPECT_EQ(8083, config.http_port());
    EXPECT_EQ("file:///tmp/test", config.registry_storage_uri());
    EXPECT_EQ("prometheus", config.metrics_reporter_type());
    EXPECT_EQ(5000, config.metrics_report_interval_ms());
    EXPECT_FALSE(config.enable_prometheus());
    EXPECT_EQ("my_prefix", config.prometheus_prefix());
    EXPECT_EQ(8, config.io_thread_num());
}

TEST_F(OnlineOptimizerServerConfigTest, PartialJsonUsesDefaults) {
    std::string json = R"({
        "rpc_port": 60000,
        "registry_storage_uri": "redis://localhost"
    })";

    OnlineOptimizerServerConfig config;
    ASSERT_TRUE(config.FromJsonString(json));
    EXPECT_EQ(60000, config.rpc_port());
    EXPECT_EQ(8082, config.http_port());
    EXPECT_EQ("redis://localhost", config.registry_storage_uri());
    EXPECT_EQ("local", config.metrics_reporter_type());
}

TEST_F(OnlineOptimizerServerConfigTest, SerializeAndDeserialize) {
    std::string json = R"({
        "rpc_port": 50099,
        "http_port": 9090,
        "registry_storage_uri": "memory://",
        "metrics_reporter_type": "custom",
        "metrics_report_interval_ms": 3000,
        "enable_prometheus": true,
        "prometheus_prefix": "test",
        "io_thread_num": 16
    })";

    OnlineOptimizerServerConfig config1;
    ASSERT_TRUE(config1.FromJsonString(json));

    std::string serialized = config1.ToJsonString();

    OnlineOptimizerServerConfig config2;
    ASSERT_TRUE(config2.FromJsonString(serialized));
    EXPECT_EQ(config1.rpc_port(), config2.rpc_port());
    EXPECT_EQ(config1.http_port(), config2.http_port());
    EXPECT_EQ(config1.registry_storage_uri(), config2.registry_storage_uri());
    EXPECT_EQ(config1.metrics_reporter_type(), config2.metrics_reporter_type());
    EXPECT_EQ(config1.metrics_report_interval_ms(), config2.metrics_report_interval_ms());
    EXPECT_EQ(config1.enable_prometheus(), config2.enable_prometheus());
    EXPECT_EQ(config1.prometheus_prefix(), config2.prometheus_prefix());
    EXPECT_EQ(config1.io_thread_num(), config2.io_thread_num());
}

TEST_F(OnlineOptimizerServerConfigTest, OverrideFromEnvironMap) {
    OnlineOptimizerServerConfig config;
    ASSERT_TRUE(config.FromJsonString(R"({"rpc_port": 50052, "registry_storage_uri": "file:///tmp"})"));

    std::unordered_map<std::string, std::string> environ = {
        {"kvcm_optimizer.rpc_port", "60000"},
        {"kvcm_optimizer.io_thread_num", "16"},
    };
    ASSERT_TRUE(config.OverrideFromEnviron(environ));
    EXPECT_EQ(60000, config.rpc_port());
    EXPECT_EQ(16, config.io_thread_num());
    EXPECT_EQ("file:///tmp", config.registry_storage_uri());
}

TEST_F(OnlineOptimizerServerConfigTest, OverrideFromSystemEnv) {
    OnlineOptimizerServerConfig config;
    ASSERT_TRUE(config.FromJsonString(R"({"rpc_port": 50052})"));

    ScopedEnv env("kvcm_optimizer.http_port", "9999");
    std::unordered_map<std::string, std::string> empty_environ;
    ASSERT_TRUE(config.OverrideFromEnviron(empty_environ));
    EXPECT_EQ(9999, config.http_port());
    EXPECT_EQ(50052, config.rpc_port());
}

TEST_F(OnlineOptimizerServerConfigTest, SystemEnvOverridesEnvironMap) {
    OnlineOptimizerServerConfig config;
    ASSERT_TRUE(config.FromJsonString(R"({"rpc_port": 50052})"));

    ScopedEnv env("kvcm_optimizer.rpc_port", "70000");
    std::unordered_map<std::string, std::string> environ = {
        {"kvcm_optimizer.rpc_port", "60000"},
    };
    ASSERT_TRUE(config.OverrideFromEnviron(environ));
    EXPECT_EQ(70000, config.rpc_port());
}

TEST_F(OnlineOptimizerServerConfigTest, UnderscoreEnvKeyFallback) {
    OnlineOptimizerServerConfig config;
    ASSERT_TRUE(config.FromJsonString(R"({"rpc_port": 50052})"));

    ScopedEnv env("kvcm_optimizer_http_port", "8888");
    std::unordered_map<std::string, std::string> empty_environ;
    ASSERT_TRUE(config.OverrideFromEnviron(empty_environ));
    EXPECT_EQ(8888, config.http_port());
}

} // namespace kv_cache_manager
