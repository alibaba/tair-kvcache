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
        ASSERT_EQ(2, config.GetSchedulePlanExecutorThreadCount());
        ASSERT_EQ(1u, config.GetSchedulePlanMigrationWorkerBudget());
        ASSERT_EQ(4u, config.GetMetaQueryWorkerCount());
        ASSERT_EQ(256u, config.GetMetaQueryParallelThreshold());
        ASSERT_EQ(128u, config.GetMetaQueryChunkSize());
        ASSERT_EQ(60000, config.GetCacheReclaimerInflightDeleteTimeoutMs());
        ASSERT_EQ(100000, config.GetCacheReclaimerPendingLocationLimitPerGroupType());
        ASSERT_EQ(64ULL * 1024 * 1024 * 1024, config.GetCacheReclaimerPendingBytesLimitPerGroupType());
        ASSERT_EQ(1024, config.GetCacheReclaimerPendingDeleteHandlerLimit());
        ASSERT_EQ(256ULL * 1024 * 1024 * 1024, config.GetCacheReclaimerPendingBytesLimit());
        ASSERT_FALSE(config.IsCacheGcEnabled());
        ASSERT_EQ(1000, config.GetCacheGcScanIntervalMs());
        ASSERT_EQ(86400000, config.GetCacheGcRoundPauseMs());
        ASSERT_EQ(256, config.GetCacheGcScanBatchSize());
        ASSERT_EQ(86400000, config.GetCacheGcOrphanWritingGracePeriodMs());
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

TEST_F(ServerConfigTest, ParsesQuotaPolicyPollerConfig) {
    ServerConfig config;
    std::unordered_map<std::string, std::string> environ = {
        {"kvcm.quota_policy_poller.enable", "true"},
        {"kvcm.quota_policy_poller.enable_hard_resize", "true"},
        {"kvcm.quota_policy_poller.optimizer_service_discovery_url", "static://127.0.0.1:50052"},
        {"kvcm.quota_policy_poller.pool_id", "pool-a"},
        {"kvcm.quota_policy_poller.quota_target_id", "kvcm-a"},
        {"kvcm.quota_policy_poller.instance_group", "group-a"},
        {"kvcm.quota_policy_poller.state_file", "/tmp/kvbrain-quota-state"},
        {"kvcm.quota_policy_poller.poll_interval_seconds", "15"},
        {"kvcm.quota_policy_poller.rpc_timeout_ms", "2000"},
    };
    ASSERT_TRUE(config.Parse("", environ));
    ASSERT_TRUE(config.Check());
    const auto poller = config.GetQuotaPolicyPollerConfig();
    EXPECT_TRUE(poller.enable);
    EXPECT_TRUE(poller.enable_hard_resize);
    EXPECT_EQ("static://127.0.0.1:50052", poller.optimizer_service_discovery_url);
    EXPECT_EQ("pool-a", poller.pool_id);
    EXPECT_EQ("kvcm-a", poller.quota_target_id);
    EXPECT_EQ(15, poller.poll_interval_seconds);
    EXPECT_EQ(2000, poller.rpc_timeout_ms);
}

TEST_F(ServerConfigTest, RejectsHardResizeWhenQuotaPolicyPollerIsDisabled) {
    ServerConfig config;
    std::unordered_map<std::string, std::string> environ = {
        {"kvcm.quota_policy_poller.enable_hard_resize", "true"},
    };
    ASSERT_TRUE(config.Parse("", environ));
    EXPECT_FALSE(config.Check());
}

TEST_F(ServerConfigTest, TestMetricsReporterType) {
    {
        ServerConfig config;
        EXPECT_EQ("local", config.metrics_reporter_type());
    }
    {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ;
        ASSERT_TRUE(config.Parse("", environ));
        EXPECT_TRUE(config.Check());
        EXPECT_EQ("local", config.metrics_reporter_type());
    }
    for (const auto &type : {"", "local", "logging"}) {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ{{"kvcm.metrics.reporter_type", type}};
        ASSERT_TRUE(config.Parse("", environ));
        EXPECT_TRUE(config.Check()) << "type: " << type;
        EXPECT_EQ(type, config.metrics_reporter_type());
    }
    {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ{{"kvcm.metrics.reporter_type", "logging"}};
        ASSERT_TRUE(config.Parse("", environ));
        EXPECT_EQ("logging", config.metrics_reporter_type());
        ASSERT_TRUE(config.Parse("", {}));
        EXPECT_EQ("local", config.metrics_reporter_type());
    }
    {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ{{"kvcm.metrics.reporter_type", "unknown"}};
        ASSERT_TRUE(config.Parse("", environ));
        EXPECT_FALSE(config.Check());
    }
}

TEST_F(ServerConfigTest, TestSchedulePlanMigrationWorkerBudget) {
    {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ{
            {"kvcm.schedule_plan_executor_thread_count", "8"},
            {"kvcm.schedule_plan_migration_worker_budget", "3"},
        };
        ASSERT_TRUE(config.Parse("", environ));
        EXPECT_TRUE(config.Check());
        EXPECT_EQ(8, config.GetSchedulePlanExecutorThreadCount());
        EXPECT_EQ(3u, config.GetSchedulePlanMigrationWorkerBudget());
    }
    {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ{
            {"kvcm.schedule_plan_executor_thread_count", "8"},
            {"kvcm.schedule_plan_migration_worker_budget", "8"},
        };
        ASSERT_TRUE(config.Parse("", environ));
        EXPECT_FALSE(config.Check());
    }
    {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ{
            {"kvcm.schedule_plan_executor_thread_count", "8"},
            {"kvcm.schedule_plan_migration_worker_budget", "0"},
        };
        ASSERT_TRUE(config.Parse("", environ));
        EXPECT_FALSE(config.Check());
    }
    {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ{
            {"kvcm.schedule_plan_migration_worker_budget", "3x"},
        };
        EXPECT_FALSE(config.Parse("", environ));
    }
    {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ{
            {"kvcm.schedule_plan_migration_worker_budget", "invalid"},
        };
        EXPECT_FALSE(config.Parse("", environ));
    }
}

TEST_F(ServerConfigTest, TestMetaQueryExecutorConfig) {
    {
        ServerConfig config;
        std::unordered_map<std::string, std::string> environ{
            {"kvcm.meta_query.worker_count", "8"},
            {"kvcm.meta_query.parallel_threshold", "512"},
            {"kvcm.meta_query.chunk_size", "64"},
        };
        ASSERT_TRUE(config.Parse("", environ));
        EXPECT_TRUE(config.Check());
        EXPECT_EQ(8u, config.GetMetaQueryWorkerCount());
        EXPECT_EQ(512u, config.GetMetaQueryParallelThreshold());
        EXPECT_EQ(64u, config.GetMetaQueryChunkSize());
    }
    for (const auto &invalid : std::vector<std::pair<std::string, std::string>>{
             {"kvcm.meta_query.worker_count", "0"},
             {"kvcm.meta_query.worker_count", "65"},
             {"kvcm.meta_query.parallel_threshold", "0"},
             {"kvcm.meta_query.chunk_size", "0"},
         }) {
        ServerConfig config;
        ASSERT_TRUE(config.Parse("", {{invalid.first, invalid.second}}));
        EXPECT_FALSE(config.Check()) << invalid.first << "=" << invalid.second;
    }
    {
        ServerConfig config;
        ASSERT_TRUE(
            config.Parse("", {{"kvcm.meta_query.parallel_threshold", "128"}, {"kvcm.meta_query.chunk_size", "129"}}));
        EXPECT_FALSE(config.Check());
    }
    for (const auto &invalid_value : {"invalid", "12x", "-1", "4294967296"}) {
        ServerConfig config;
        EXPECT_FALSE(config.Parse("", {{"kvcm.meta_query.worker_count", invalid_value}})) << invalid_value;
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

TEST_F(ServerConfigTest, TestCacheGcConfig) {
    ServerConfig config;
    std::unordered_map<std::string, std::string> environ{
        {"kvcm.cache_gc.enabled", "true"},
        {"kvcm.cache_gc.scan_interval_ms", "123"},
        {"kvcm.cache_gc.round_pause_ms", "456"},
        {"kvcm.cache_gc.scan_batch_size", "7"},
        {"kvcm.cache_gc.orphan_writing_grace_period_ms", "3600000"},
        {"kvcm.cache_gc.max_inflight_delete_requests", "3"},
    };
    ASSERT_TRUE(config.Parse("", environ));
    ASSERT_TRUE(config.Check());
    EXPECT_TRUE(config.IsCacheGcEnabled());
    EXPECT_EQ(123, config.GetCacheGcScanIntervalMs());
    EXPECT_EQ(456, config.GetCacheGcRoundPauseMs());
    EXPECT_EQ(7, config.GetCacheGcScanBatchSize());
    EXPECT_EQ(3600000, config.GetCacheGcOrphanWritingGracePeriodMs());
    EXPECT_EQ(3, config.GetCacheGcMaxInflightDeleteRequests());

    environ["kvcm.cache_gc.orphan_writing_grace_period_ms"] = "3599999";
    ASSERT_TRUE(config.Parse("", environ));
    EXPECT_FALSE(config.Check());

    environ["kvcm.cache_gc.orphan_writing_grace_period_ms"] = "3600000";
    environ["kvcm.cache_gc.scan_batch_size"] = "0";
    ASSERT_TRUE(config.Parse("", environ));
    EXPECT_FALSE(config.Check());

    environ["kvcm.cache_gc.scan_batch_size"] = "9223372036854775808";
    ASSERT_TRUE(config.Parse("", environ));
    EXPECT_FALSE(config.Check());

    environ["kvcm.cache_gc.scan_batch_size"] = "7";
    environ["kvcm.cache_gc.max_inflight_delete_requests"] = "0";
    ASSERT_TRUE(config.Parse("", environ));
    EXPECT_FALSE(config.Check());

    environ["kvcm.cache_gc.enabled"] = "false";
    ASSERT_TRUE(config.Parse("", environ));
    EXPECT_TRUE(config.Check());
}

TEST_F(ServerConfigTest, TestMalformedNumericEnvironmentValueReturnsParseError) {
    ServerConfig config;
    std::unordered_map<std::string, std::string> environ{
        {"kvcm.cache_gc.scan_interval_ms", "not-a-number"},
    };

    EXPECT_FALSE(config.Parse("", environ));
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
