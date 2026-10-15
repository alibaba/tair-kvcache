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
        ASSERT_EQ(100, config.GetCacheReclaimerKeySamplingSizeTotal());
        ASSERT_EQ(60000, config.GetCacheReclaimerInflightDeleteTimeoutMs());
        ASSERT_EQ(20000, config.GetCacheReclaimerPendingLocationLimitPerGroupType());
        ASSERT_EQ(1ULL * 1024 * 1024 * 1024 * 1024, config.GetCacheReclaimerPendingBytesLimitPerGroupType());
        ASSERT_EQ(1024, config.GetCacheReclaimerPendingDeleteHandlerLimit());
        ASSERT_EQ(4ULL * 1024 * 1024 * 1024 * 1024, config.GetCacheReclaimerPendingBytesLimit());
        ASSERT_TRUE(config.IsCacheGcEnabled());
        ASSERT_EQ(100, config.GetCacheGcScanIntervalMs());
        ASSERT_EQ(300000, config.GetCacheGcRoundPauseMs());
        ASSERT_EQ(256, config.GetCacheGcScanBatchSize());
        ASSERT_EQ(86400000, config.GetCacheGcOrphanWritingGracePeriodMs());
        ASSERT_EQ(64, config.GetCacheGcMaxInflightDeleteRequests());
        ASSERT_TRUE(config.IsCacheGcEventReportCleanupEnabled());
        ASSERT_EQ(256, config.GetCacheGcEventReportActionBatchSize());
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

TEST_F(ServerConfigTest, MetricsEnvTagsResolveAtStartup) {
    ScopedEnv workspace("KVCM_TEST_METRICS_WORKSPACE", " workspace_a ");
    ScopedEnv deployment("KVCM_TEST_METRICS_DEPLOYMENT", "deployment_a");
    ScopedEnv empty("KVCM_TEST_METRICS_EMPTY", "");
    ScopedEnv whitespace("KVCM_TEST_METRICS_WHITESPACE", " \t ");
    const std::string mappings = " workspace = KVCM_TEST_METRICS_WORKSPACE , "
                                 "deployment_name=KVCM_TEST_METRICS_DEPLOYMENT,"
                                 "whitespace=KVCM_TEST_METRICS_WHITESPACE,"
                                 "empty=KVCM_TEST_METRICS_EMPTY,missing=KVCM_TEST_METRICS_UNSET";
    for (const auto &type : {"local", "logging", "dummy", "kmonitor"}) {
        ServerConfig config;
        ASSERT_TRUE(
            config.Parse("", {{"kvcm.metrics.env_tag_mappings", mappings}, {"kvcm.metrics.reporter_type", type}}));
        ASSERT_TRUE(config.Check());
        EXPECT_EQ(
            (MetricsTags{{"workspace", " workspace_a "}, {"deployment_name", "deployment_a"}, {"whitespace", " \t "}}),
            config.metrics_common_tags());
    }

    ServerConfig config;
    ASSERT_TRUE(config.Parse("", {{"kvcm.metrics.env_tag_mappings", mappings}}));
    setenv("KVCM_TEST_METRICS_DEPLOYMENT", "deployment_b", 1);
    ASSERT_TRUE(config.Check());
    EXPECT_EQ("deployment_a", config.metrics_common_tags().at("deployment_name"));
    // Parsing a fresh configuration must not retain tags from an earlier one.
    ASSERT_TRUE(config.Parse("", {}));
    EXPECT_TRUE(config.metrics_common_tags().empty());
    EXPECT_TRUE(config.prometheus_common_tags().empty());
    ASSERT_TRUE(config.Parse("", {{"kvcm.metrics.env_tag_mappings", ""}}));
    EXPECT_TRUE(config.metrics_common_tags().empty());
    EXPECT_TRUE(config.prometheus_common_tags().empty());
}

TEST_F(ServerConfigTest, MetricsEnvTagCollisionsKeepRawTagsAndReportEffectiveLabels) {
    ScopedEnv first("KVCM_TEST_METRICS_FIRST", "first_value");
    ScopedEnv second("KVCM_TEST_METRICS_SECOND", "second_value");
    ScopedEnv empty("KVCM_TEST_METRICS_EMPTY", "");
    ServerConfig config;
    testing::internal::CaptureStderr();
    const auto parsed = config.Parse(
        "",
        {{"kvcm.metrics.env_tag_mappings",
          "host.name=KVCM_TEST_METRICS_FIRST,host-name=KVCM_TEST_METRICS_SECOND,host_name=KVCM_TEST_METRICS_EMPTY"}});
    const auto errors = testing::internal::GetCapturedStderr();
    ASSERT_TRUE(parsed);
    EXPECT_EQ((MetricsTags{{"host.name", "first_value"}, {"host-name", "second_value"}}), config.metrics_common_tags());
    EXPECT_EQ((MetricsTags{{"host_name", "first_value"}}), config.prometheus_common_tags());
    EXPECT_NE(std::string::npos, errors.find("ERROR:"));
    EXPECT_NE(std::string::npos, errors.find("[host.name] normalizes to duplicate label [host_name]"));
    EXPECT_NE(std::string::npos, errors.find("overwriting value \"second_value\" with \"first_value\""));

    testing::internal::CaptureStderr();
    const auto disabled = config.Parse(
        "",
        {{"kvcm.metrics.enable_prometheus", "false"},
         {"kvcm.metrics.env_tag_mappings", "host.name=KVCM_TEST_METRICS_FIRST,host-name=KVCM_TEST_METRICS_SECOND"}});
    const auto disabled_errors = testing::internal::GetCapturedStderr();
    ASSERT_TRUE(disabled);
    EXPECT_EQ(2, config.metrics_common_tags().size());
    EXPECT_TRUE(config.prometheus_common_tags().empty());
    EXPECT_TRUE(disabled_errors.empty());
}

TEST_F(ServerConfigTest, MetricsEnvTagsUseExistingEnvironmentPrecedence) {
    ScopedEnv workspace("KVCM_TEST_METRICS_WORKSPACE", "workspace_a");
    ScopedEnv underscore("kvcm_metrics_env_tag_mappings", "workspace=KVCM_TEST_METRICS_WORKSPACE");
    ServerConfig config;
    ASSERT_TRUE(config.Parse("", {{"kvcm.metrics.env_tag_mappings", "other=KVCM_TEST_METRICS_WORKSPACE"}}));
    EXPECT_EQ((MetricsTags{{"workspace", "workspace_a"}}), config.metrics_common_tags());
    ScopedEnv dotted("kvcm.metrics.env_tag_mappings", "deployment=KVCM_TEST_METRICS_WORKSPACE");
    ASSERT_TRUE(config.Parse("", {}));
    EXPECT_EQ((MetricsTags{{"deployment", "workspace_a"}}), config.metrics_common_tags());
}

TEST_F(ServerConfigTest, MetricsEnvTagsRejectMalformedAndDuplicateMappings) {
    for (const auto &mappings : {"workspace",
                                 "=ENV",
                                 "workspace=",
                                 "workspace=ENV=OTHER",
                                 "workspace=ENV,workspace=OTHER",
                                 "workspace=ENV,",
                                 ",workspace=ENV",
                                 "workspace=ENV,,deployment=OTHER",
                                 "workspace=ENV, =OTHER"}) {
        ServerConfig config;
        EXPECT_FALSE(config.Parse("", {{"kvcm.metrics.env_tag_mappings", mappings}})) << mappings;
        EXPECT_TRUE(config.metrics_common_tags().empty());
    }
}

TEST_F(ServerConfigTest, TestGroupLruLimits) {
    ServerConfig defaults;
    ASSERT_TRUE(defaults.Parse("", {}));
    EXPECT_EQ(65536, defaults.GetCacheReclaimerGroupLruMaxSamplingSize());
    EXPECT_EQ(128, defaults.GetCacheReclaimerGroupLruMaxDeleteRequestsPerRound());
    EXPECT_EQ(10, defaults.GetCacheReclaimerGroupLruMinSamplingRatio());
    // Group LRU's larger sample pool must not change the shared legacy defaults.
    EXPECT_EQ(100, defaults.GetCacheReclaimerKeySamplingSizeTotal());
    EXPECT_EQ(100, defaults.GetCacheReclaimerDelBatchSize());
    ServerConfig configured;
    ASSERT_TRUE(configured.Parse("",
                                 {{"kvcm.cache_reclaimer.group_lru_max_sampling_size", "131072"},
                                  {"kvcm.cache_reclaimer.group_lru_max_delete_requests_per_round", "256"},
                                  {"kvcm.cache_reclaimer.group_lru_min_sampling_ratio", "5"}}));
    ASSERT_TRUE(configured.Check());
    EXPECT_EQ(131072, configured.GetCacheReclaimerGroupLruMaxSamplingSize());
    EXPECT_EQ(256, configured.GetCacheReclaimerGroupLruMaxDeleteRequestsPerRound());
    EXPECT_EQ(5, configured.GetCacheReclaimerGroupLruMinSamplingRatio());
    for (const auto &key : {"kvcm.cache_reclaimer.group_lru_max_sampling_size",
                            "kvcm.cache_reclaimer.group_lru_max_delete_requests_per_round",
                            "kvcm.cache_reclaimer.group_lru_min_sampling_ratio"}) {
        for (const auto &value : {"0", "-1", "", "1.5", "100x", "18446744073709551616"}) {
            ServerConfig invalid;
            EXPECT_FALSE(invalid.Parse("", {{key, value}})) << key << "=" << value;
        }
    }
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
        {"kvcm.cache_gc.event_report_cleanup_enabled", "true"},
        {"kvcm.cache_gc.event_report_action_batch_size", "5"},
    };
    ASSERT_TRUE(config.Parse("", environ));
    ASSERT_TRUE(config.Check());
    EXPECT_TRUE(config.IsCacheGcEnabled());
    EXPECT_EQ(123, config.GetCacheGcScanIntervalMs());
    EXPECT_EQ(456, config.GetCacheGcRoundPauseMs());
    EXPECT_EQ(7, config.GetCacheGcScanBatchSize());
    EXPECT_EQ(3600000, config.GetCacheGcOrphanWritingGracePeriodMs());
    EXPECT_EQ(3, config.GetCacheGcMaxInflightDeleteRequests());
    EXPECT_TRUE(config.IsCacheGcEventReportCleanupEnabled());
    EXPECT_EQ(5, config.GetCacheGcEventReportActionBatchSize());

    environ["kvcm.cache_gc.round_pause_ms"] = "0";
    ASSERT_TRUE(config.Parse("", environ));
    EXPECT_TRUE(config.Check());

    environ["kvcm.cache_gc.round_pause_ms"] = "-1";
    ASSERT_TRUE(config.Parse("", environ));
    EXPECT_FALSE(config.Check());

    environ["kvcm.cache_gc.round_pause_ms"] = "456";

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

    environ["kvcm.cache_gc.max_inflight_delete_requests"] = "3";
    environ["kvcm.cache_gc.event_report_action_batch_size"] = "0";
    ASSERT_TRUE(config.Parse("", environ));
    EXPECT_FALSE(config.Check());

    environ["kvcm.cache_gc.event_report_action_batch_size"] = "8";
    ASSERT_TRUE(config.Parse("", environ));
    EXPECT_TRUE(config.Check());

    environ["kvcm.cache_gc.event_report_action_batch_size"] = "5";
    environ["kvcm.cache_gc.enabled"] = "false";
    ASSERT_TRUE(config.Parse("", environ));
    EXPECT_TRUE(config.Check());
    EXPECT_FALSE(config.IsCacheGcEnabled());
    EXPECT_TRUE(config.IsCacheGcEventReportCleanupEnabled());

    environ["kvcm.cache_gc.event_report_cleanup_enabled"] = "false";
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
