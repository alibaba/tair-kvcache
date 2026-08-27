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
    EXPECT_TRUE(config.kvcm_event_subscriptions().empty());
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
        "io_thread_num": 8,
        "kvcm_event_subscriptions": [
            {
                "service_discovery_url": "static://127.0.0.1:6490",
                "consumer_id": "optimizer-a",
                "discovery_refresh_interval_ms": 1234
            },
            {
                "service_discovery_url": "static://127.0.0.1:6491",
                "consumer_id": "optimizer-b",
                "discovery_refresh_interval_ms": 2345
            }
        ]
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
    ASSERT_EQ(2, config.kvcm_event_subscriptions().size());
    EXPECT_EQ("static://127.0.0.1:6490", config.kvcm_event_subscriptions()[0].service_discovery_url());
    EXPECT_EQ("optimizer-a", config.kvcm_event_subscriptions()[0].consumer_id());
    EXPECT_EQ(1234, config.kvcm_event_subscriptions()[0].discovery_refresh_interval_ms());
    EXPECT_EQ("static://127.0.0.1:6491", config.kvcm_event_subscriptions()[1].service_discovery_url());
    EXPECT_EQ("optimizer-b", config.kvcm_event_subscriptions()[1].consumer_id());
    EXPECT_EQ(2345, config.kvcm_event_subscriptions()[1].discovery_refresh_interval_ms());
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

TEST_F(OnlineOptimizerServerConfigTest, ReparseWithoutSubscriptionRestoresDefaults) {
    OnlineOptimizerServerConfig config;
    ASSERT_TRUE(config.FromJsonString(R"({
        "kvcm_event_subscriptions": [{
            "service_discovery_url": "static://127.0.0.1:6381"
        }]
    })"));
    ASSERT_EQ(1, config.kvcm_event_subscriptions().size());

    ASSERT_TRUE(config.FromJsonString(R"({})"));
    EXPECT_TRUE(config.kvcm_event_subscriptions().empty());
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
        "io_thread_num": 16,
        "kvcm_event_subscriptions": [
            {
                "service_discovery_url": "static://127.0.0.1:6490",
                "consumer_id": "optimizer-a",
                "discovery_refresh_interval_ms": 4321
            },
            {
                "service_discovery_url": "static://127.0.0.1:6491",
                "consumer_id": "optimizer-b",
                "discovery_refresh_interval_ms": 5432
            }
        ]
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
    ASSERT_EQ(config1.kvcm_event_subscriptions().size(), config2.kvcm_event_subscriptions().size());
    for (std::size_t i = 0; i < config1.kvcm_event_subscriptions().size(); ++i) {
        EXPECT_EQ(config1.kvcm_event_subscriptions()[i].service_discovery_url(),
                  config2.kvcm_event_subscriptions()[i].service_discovery_url());
        EXPECT_EQ(config1.kvcm_event_subscriptions()[i].consumer_id(),
                  config2.kvcm_event_subscriptions()[i].consumer_id());
        EXPECT_EQ(config1.kvcm_event_subscriptions()[i].discovery_refresh_interval_ms(),
                  config2.kvcm_event_subscriptions()[i].discovery_refresh_interval_ms());
    }
}

TEST_F(OnlineOptimizerServerConfigTest, OverrideFromEnvironMap) {
    OnlineOptimizerServerConfig config;
    ASSERT_TRUE(config.FromJsonString(R"({"rpc_port": 50052, "registry_storage_uri": "file:///tmp"})"));

    std::unordered_map<std::string, std::string> environ = {
        {"kvcm_optimizer.rpc_port", "60000"},
        {"kvcm_optimizer.io_thread_num", "16"},
        {"kvcm_optimizer.kvcm_event_subscriptions",
         R"([{"service_discovery_url":"static://127.0.0.1:6381","consumer_id":"optimizer-env-a","discovery_refresh_interval_ms":2500},{"service_discovery_url":"static://127.0.0.1:6382","consumer_id":"optimizer-env-b","discovery_refresh_interval_ms":3500}])"},
    };
    ASSERT_TRUE(config.OverrideFromEnviron(environ));
    EXPECT_EQ(60000, config.rpc_port());
    EXPECT_EQ(16, config.io_thread_num());
    EXPECT_EQ("file:///tmp", config.registry_storage_uri());
    ASSERT_EQ(2, config.kvcm_event_subscriptions().size());
    EXPECT_EQ("static://127.0.0.1:6381", config.kvcm_event_subscriptions()[0].service_discovery_url());
    EXPECT_EQ("optimizer-env-a", config.kvcm_event_subscriptions()[0].consumer_id());
    EXPECT_EQ(2500, config.kvcm_event_subscriptions()[0].discovery_refresh_interval_ms());
    EXPECT_EQ("static://127.0.0.1:6382", config.kvcm_event_subscriptions()[1].service_discovery_url());
}

TEST_F(OnlineOptimizerServerConfigTest, OverrideQuotaPlannerFromEnvironMap) {
    OnlineOptimizerServerConfig config;
    ASSERT_TRUE(config.FromJsonString(R"({})"));

    std::unordered_map<std::string, std::string> environ = {
        {"kvcm_optimizer.quota_planner_enable", "true"},
        {"kvcm_optimizer.quota_planner_enable_hard_resize", "false"},
        {"kvcm_optimizer.quota_planner_period_seconds", "60"},
        {"kvcm_optimizer.quota_planner_plan_ttl_seconds", "300"},
        {"kvcm_optimizer.quota_planner_release_timeout_seconds", "120"},
        {"kvcm_optimizer.quota_planner_release_consecutive_samples", "4"},
        {"kvcm_optimizer.quota_planner_pools",
         R"([{"pool_id":"pool-a","quota_scope":"per_replica","allocatable_bytes":2000,"allocatable_source":"test","candidate_step_bytes":100,"members":[{"quota_target_id":"kvcm-a","source_id":"instance-a","instance_group":"default","quota_scope":"per_replica","current_quota_bytes":1000,"min_quota_bytes":500,"configured_max_quota_bytes":1500,"hardware_max_quota_bytes":1600,"configured_max_source":"test-config","hardware_max_source":"test-hardware"}]}])"},
    };

    ASSERT_TRUE(config.OverrideFromEnviron(environ));
    const auto &quota = config.quota_planner_config();
    EXPECT_TRUE(quota.enable);
    EXPECT_FALSE(quota.enable_hard_resize);
    EXPECT_EQ(60, quota.period_seconds);
    EXPECT_EQ(300, quota.plan_ttl_seconds);
    EXPECT_EQ(120, quota.release_timeout_seconds);
    EXPECT_EQ(4, quota.release_consecutive_samples);
    ASSERT_EQ(1, quota.pools.size());
    EXPECT_EQ("pool-a", quota.pools[0].pool_id);
    ASSERT_EQ(1, quota.pools[0].members.size());
    EXPECT_EQ("kvcm-a", quota.pools[0].members[0].quota_target_id);
}

TEST_F(OnlineOptimizerServerConfigTest, RejectsInvalidSubscriptions) {
    OnlineOptimizerServerConfig config;
    EXPECT_FALSE(config.FromJsonString(R"({"kvcm_event_subscriptions":[{}]})"));
    EXPECT_FALSE(config.FromJsonString(
        R"({"kvcm_event_subscriptions":[{"service_discovery_url":"static://127.0.0.1:6381","consumer_id":"","discovery_refresh_interval_ms":5000}]})"));
    EXPECT_FALSE(config.FromJsonString(
        R"({"kvcm_event_subscriptions":[{"service_discovery_url":"static://127.0.0.1:6381","consumer_id":"optimizer","discovery_refresh_interval_ms":0}]})"));
    EXPECT_FALSE(config.FromJsonString(
        R"({"kvcm_event_subscriptions":[{"service_discovery_url":"static://127.0.0.1:6381"},{"service_discovery_url":"static://127.0.0.1:6381"}]})"));
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

TEST_F(OnlineOptimizerServerConfigTest, ParsesHardResizeQuotaPlannerWithAuthoritativeBounds) {
    OnlineOptimizerServerConfig config;
    ASSERT_TRUE(config.FromJsonString(R"({
        "quota_planner_enable": true,
        "quota_planner_enable_hard_resize": true,
        "quota_planner_plan_ttl_seconds": 3600,
        "quota_planner_release_timeout_seconds": 1800,
        "quota_planner_pools": [{
            "pool_id": "pool-a",
            "quota_scope": "per_replica",
            "allocatable_bytes": 2000,
            "allocatable_source": "pool-config",
            "candidate_step_bytes": 100,
            "members": [{
                "quota_target_id": "kvcm-a",
                "source_id": "instance-a",
                "instance_group": "group-a",
                "quota_scope": "per_replica",
                "current_quota_bytes": 1000,
                "min_quota_bytes": 500,
                "configured_max_quota_bytes": 1500,
                "hardware_max_quota_bytes": 1600,
                "configured_max_source": "instance-group",
                "hardware_max_source": "node-inventory"
            }]
        }]
    })"));
    const auto &quota = config.quota_planner_config();
    ASSERT_TRUE(quota.enable);
    ASSERT_TRUE(quota.enable_hard_resize);
    ASSERT_EQ(1, quota.pools.size());
    EXPECT_EQ(100, quota.pools[0].candidate_step_bytes);
    EXPECT_EQ("instance-a", quota.pools[0].members[0].source_id);
}

TEST_F(OnlineOptimizerServerConfigTest, RejectsHardResizeWithoutPlanner) {
    OnlineOptimizerServerConfig config;
    EXPECT_FALSE(config.FromJsonString(R"({"quota_planner_enable_hard_resize": true})"));
}

} // namespace kv_cache_manager
