#include <cmath>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/manager/online_runtime/online_optimizer_manager.h"
#include "kv_cache_manager/optimizer/metrics/optimizer_metrics_collector.h"
#include "kv_cache_manager/optimizer/metrics/optimizer_metrics_reporter.h"

namespace kv_cache_manager {

class OptimizerMetricsReporterTest : public TESTBASE {
protected:
    void SetUp() override {
        TESTBASE::SetUp();
        opt_registry_ = std::make_shared<OptimizerRegistryManager>("");
        opt_registry_->Init();
        manager_ = std::make_shared<OnlineOptimizerManager>(opt_registry_);
        registry_ = std::make_shared<MetricsRegistry>();
        reporter_ = std::make_shared<OptimizerMetricsReporter>(manager_, registry_);
    }

    ErrorCode RegisterTestInstance(const std::string &instance_id,
                                   std::vector<double> caps = {1.0},
                                   int64_t ttl_seconds = 0,
                                   bool enable_theoretical_max_cache = false,
                                   int32_t linear_step = 0) {
        OptimizerInstanceGroup group;
        group.set_name("grp1");
        group.set_capacity_gb(caps);
        group.set_enable_theoretical_max_cache(enable_theoretical_max_cache);
        group.set_ttl_seconds(ttl_seconds);

        // Register/update group in registry so RegisterInstance can look it up.
        if (opt_registry_->GetInstanceGroup("grp1")) {
            opt_registry_->UpdateInstanceGroup(group);
        } else {
            opt_registry_->CreateInstanceGroup(group);
        }

        std::vector<LocationSpecInfo> specs = {LocationSpecInfo("full", 1024)};
        std::vector<LocationSpecGroup> groups = {LocationSpecGroup("full_group", {"full"})};
        OptimizerInstanceInfo info(
            "grp1", instance_id, 1024, specs, groups, linear_step, OptimizerStateInfo("full_group", ""));
        RegisterInstanceResult result;
        return manager_->RegisterInstance(info, result);
    }

    std::shared_ptr<OptimizerRegistryManager> opt_registry_;
    std::shared_ptr<OnlineOptimizerManager> manager_;
    std::shared_ptr<MetricsRegistry> registry_;
    std::shared_ptr<OptimizerMetricsReporter> reporter_;
};

TEST_F(OptimizerMetricsReporterTest, ReportEmptyState) { reporter_->ReportInterval(); }

TEST_F(OptimizerMetricsReporterTest, ReportAfterRegistration) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));
    reporter_->ReportInterval();

    MetricsTags tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}};
    Gauge query_total = registry_->GetGauge("trace_query_total", tags);
    EXPECT_EQ(0.0, query_total.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportAfterQueries) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2, 3}, 0, 0, result);
    manager_->TraceQuery("inst1", {1, 2, 4}, 0, 0, result);

    reporter_->ReportInterval();

    MetricsTags tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}};
    Gauge query_total = registry_->GetGauge("trace_query_total", tags);
    EXPECT_EQ(2.0, query_total.Get());

    Gauge blocks_total = registry_->GetGauge("trace_query_blocks_total", tags);
    EXPECT_EQ(6.0, blocks_total.Get());

    Gauge unique_keys = registry_->GetGauge("trace_query_unique_keys", tags);
    EXPECT_EQ(4.0, unique_keys.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportMultipleInstances) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst2"));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2}, 0, 0, result);

    reporter_->ReportInterval();

    MetricsTags tags1 = {{"instance_group", "grp1"}, {"instance_id", "inst1"}};
    Gauge qt1 = registry_->GetGauge("trace_query_total", tags1);
    EXPECT_EQ(1.0, qt1.Get());

    MetricsTags tags2 = {{"instance_group", "grp1"}, {"instance_id", "inst2"}};
    Gauge qt2 = registry_->GetGauge("trace_query_total", tags2);
    EXPECT_EQ(0.0, qt2.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportIntervalPerCapacityHitRate) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1", {1.0, 5.0}));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2, 3}, 0, 0, result);
    manager_->TraceQuery("inst1", {1, 2, 3}, 0, 0, result);

    reporter_->ReportInterval();

    MetricsTags tags_cap1 = {
        {"instance_group", "grp1"}, {"instance_id", "inst1"}, {"capacity_gb", std::to_string(1.0)}};
    Gauge rate1 = registry_->GetGauge("trace_query_hit_rate", tags_cap1);
    EXPECT_GE(rate1.Get(), 0.0);
    EXPECT_LE(rate1.Get(), 1.0);

    MetricsTags tags_cap5 = {
        {"instance_group", "grp1"}, {"instance_id", "inst1"}, {"capacity_gb", std::to_string(5.0)}};
    Gauge rate5 = registry_->GetGauge("trace_query_hit_rate", tags_cap5);
    EXPECT_GE(rate5.Get(), 0.0);
    EXPECT_LE(rate5.Get(), 1.0);

    EXPECT_GE(rate5.Get(), rate1.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportPerQueryWritesToRegistry) {
    auto collector = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
    ASSERT_TRUE(collector->Init());
    collector->set_instance_id("inst1");
    collector->set_client_ip("10.0.0.1");
    collector->set_total_blocks(10);
    collector->set_input_token_len(2048);
    collector->set_per_capacity_hits({
        {1.0, 7},
        {0.5, 3},
    });

    reporter_->ReportPerQuery(collector.get());

    MetricsTags tags1 = {{"instance_group", ""},
                         {"instance_id", "inst1"},
                         {"client_ip", "10.0.0.1"},
                         {"capacity_gb", std::to_string(1.0)}};
    Gauge rate1 = registry_->GetGauge("query_hit_rate", tags1);
    EXPECT_NEAR(0.7, rate1.Get(), 1e-9);

    Gauge hit1 = registry_->GetGauge("query_hit_count", tags1);
    EXPECT_DOUBLE_EQ(7.0, hit1.Get());

    MetricsTags tags2 = {{"instance_group", ""},
                         {"instance_id", "inst1"},
                         {"client_ip", "10.0.0.1"},
                         {"capacity_gb", std::to_string(0.5)}};
    Gauge rate2 = registry_->GetGauge("query_hit_rate", tags2);
    EXPECT_NEAR(0.3, rate2.Get(), 1e-9);

    MetricsTags base_tags = {{"instance_group", ""}, {"instance_id", "inst1"}, {"client_ip", "10.0.0.1"}};
    Gauge blocks = registry_->GetGauge("query_total_blocks", base_tags);
    EXPECT_DOUBLE_EQ(10.0, blocks.Get());

    MetricsTags service_tags = {{"instance_group", ""}, {"instance_id", "inst1"}};
    EXPECT_EQ(2048u, registry_->GetCounter("service.input_tokens_total", service_tags).Get());
    EXPECT_EQ(2048u, registry_->GetCounter("service.input_tokens_total").Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportPerQueryMaxHitMetrics) {
    auto collector = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
    ASSERT_TRUE(collector->Init());
    collector->set_instance_id("inst1");
    collector->set_client_ip("10.0.0.2");
    collector->set_total_blocks(10);
    collector->set_per_capacity_hits({{1.0, 7}});
    collector->set_max_hit_count(8);
    collector->set_max_hit_rate(0.8);

    reporter_->ReportPerQuery(collector.get());

    MetricsTags base_tags = {{"instance_group", ""}, {"instance_id", "inst1"}, {"client_ip", "10.0.0.2"}};
    Gauge max_hit = registry_->GetGauge("query_max_hit_count", base_tags);
    EXPECT_DOUBLE_EQ(8.0, max_hit.Get());

    Gauge max_rate = registry_->GetGauge("query_max_hit_rate", base_tags);
    EXPECT_NEAR(0.8, max_rate.Get(), 1e-9);
}

TEST_F(OptimizerMetricsReporterTest, ReportPerQueryCapacityEfficiency) {
    auto collector = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
    ASSERT_TRUE(collector->Init());
    collector->set_instance_id("inst1");
    collector->set_client_ip("10.0.0.3");
    collector->set_total_blocks(10);
    collector->set_per_capacity_hits({{1.0, 6}, {5.0, 8}});
    collector->set_max_hit_count(8);
    collector->set_max_hit_rate(0.8);

    reporter_->ReportPerQuery(collector.get());

    MetricsTags tags1 = {{"instance_group", ""},
                         {"instance_id", "inst1"},
                         {"client_ip", "10.0.0.3"},
                         {"capacity_gb", std::to_string(1.0)}};
    MetricsTags tags5 = {{"instance_group", ""},
                         {"instance_id", "inst1"},
                         {"client_ip", "10.0.0.3"},
                         {"capacity_gb", std::to_string(5.0)}};
    EXPECT_NEAR(0.75, registry_->GetGauge("query_capacity_efficiency", tags1).Get(), 1e-9);
    EXPECT_NEAR(1.0, registry_->GetGauge("query_capacity_efficiency", tags5).Get(), 1e-9);

    // A query with no theoretical hits makes the ratio undefined. Overwrite the
    // previous values instead of leaving stale efficiency metrics behind.
    auto zero_hit_collector = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
    ASSERT_TRUE(zero_hit_collector->Init());
    zero_hit_collector->set_instance_id("inst1");
    zero_hit_collector->set_client_ip("10.0.0.3");
    zero_hit_collector->set_total_blocks(10);
    zero_hit_collector->set_per_capacity_hits({{1.0, 0}, {5.0, 0}});
    zero_hit_collector->set_max_hit_count(0);
    zero_hit_collector->set_max_hit_rate(0.0);
    reporter_->ReportPerQuery(zero_hit_collector.get());

    EXPECT_TRUE(std::isnan(registry_->GetGauge("query_capacity_efficiency", tags1).Get()));
    EXPECT_TRUE(std::isnan(registry_->GetGauge("query_capacity_efficiency", tags5).Get()));
}

TEST_F(OptimizerMetricsReporterTest, ReportPerQueryMaxHitNotApplicable) {
    auto collector = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
    ASSERT_TRUE(collector->Init());
    collector->set_instance_id("inst1");
    collector->set_total_blocks(10);
    collector->set_per_capacity_hits({{1.0, 7}});

    reporter_->ReportPerQuery(collector.get());

    MetricsTags base_tags = {{"instance_group", ""}, {"instance_id", "inst1"}, {"client_ip", ""}};
    Gauge blocks = registry_->GetGauge("query_total_blocks", base_tags);
    EXPECT_DOUBLE_EQ(10.0, blocks.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportPerQueryZeroBlocksSkips) {
    auto collector = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
    ASSERT_TRUE(collector->Init());
    collector->set_instance_id("inst1");

    reporter_->ReportPerQuery(collector.get());
}

TEST_F(OptimizerMetricsReporterTest, ReportPerQueryNullCollectorSafe) { reporter_->ReportPerQuery(nullptr); }

TEST_F(OptimizerMetricsReporterTest, ReportPerQueryUsesRequestLocalServiceSamples) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));
    const MetricsTags instance_tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}};

    auto first = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
    ASSERT_TRUE(first->Init());
    first->set_instance_id("inst1");
    first->set_service_query_rt_us_metrics(123.0);
    first->set_service_error_code_metrics(static_cast<double>(EC_ERROR));
    reporter_->ReportPerQuery(first.get());

    EXPECT_DOUBLE_EQ(123.0, registry_->GetGauge("service.query_rt_us", instance_tags).Get());
    EXPECT_DOUBLE_EQ(static_cast<double>(EC_ERROR), registry_->GetGauge("service.error_code", instance_tags).Get());
    EXPECT_EQ(1u, registry_->GetCounter("service.error_counter", instance_tags).Get());

    auto second = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
    ASSERT_TRUE(second->Init());
    second->set_instance_id("inst1");
    second->set_service_query_rt_us_metrics(456.0);
    second->set_service_error_code_metrics(static_cast<double>(EC_OK));
    reporter_->ReportPerQuery(second.get());

    EXPECT_DOUBLE_EQ(456.0, registry_->GetGauge("service.query_rt_us", instance_tags).Get());
    EXPECT_EQ(579u, registry_->GetCounter("service.query_rt_us_sum", instance_tags).Get());
    EXPECT_DOUBLE_EQ(0.0, registry_->GetGauge("service.error_code", instance_tags).Get());
    EXPECT_EQ(1u, registry_->GetCounter("service.error_counter", instance_tags).Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportPerQueryTagsServiceMetricsByGroupAndInstance) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));
    for (int i = 0; i < 2; ++i) {
        auto collector = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
        ASSERT_TRUE(collector->Init());
        collector->set_instance_id("inst1");
        collector->set_service_query_rt_us_metrics(123.0);
        collector->set_service_error_code_metrics(static_cast<double>(EC_ERROR));
        reporter_->ReportPerQuery(collector.get());
    }

    const MetricsTags instance_tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}};
    EXPECT_EQ(2u, registry_->GetCounter("service.query_counter", instance_tags).Get());
    EXPECT_DOUBLE_EQ(123.0, registry_->GetGauge("service.query_rt_us", instance_tags).Get());
    EXPECT_EQ(246u, registry_->GetCounter("service.query_rt_us_sum", instance_tags).Get());
    EXPECT_DOUBLE_EQ(static_cast<double>(EC_ERROR), registry_->GetGauge("service.error_code", instance_tags).Get());
    EXPECT_EQ(2u, registry_->GetCounter("service.error_counter", instance_tags).Get());
    EXPECT_EQ(2u, registry_->GetCounter("service.query_counter").Get());
    EXPECT_DOUBLE_EQ(123.0, registry_->GetGauge("service.query_rt_us").Get());
    EXPECT_EQ(246u, registry_->GetCounter("service.query_rt_us_sum").Get());
    EXPECT_EQ(2u, registry_->GetCounter("service.error_counter").Get());

    auto query_counter_data = registry_->GetMetricsData("service.query_counter");
    ASSERT_NE(nullptr, query_counter_data);
    const auto query_counter_values = query_counter_data->GetMetricsValues();
    ASSERT_EQ(2u, query_counter_values.size());
}

TEST_F(OptimizerMetricsReporterTest, ReportIntervalStaticMetrics) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));
    reporter_->ReportInterval();

    MetricsTags tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}};

    Gauge bytes_per_block_gauge = registry_->GetGauge("trace_query_bytes_per_block", tags);
    EXPECT_DOUBLE_EQ(1024.0, bytes_per_block_gauge.Get());

    Gauge linear_step = registry_->GetGauge("trace_query_linear_step", tags);
    EXPECT_DOUBLE_EQ(0.0, linear_step.Get());

    Gauge eviction = registry_->GetGauge("trace_query_eviction_count", tags);
    EXPECT_DOUBLE_EQ(0.0, eviction.Get());

    Gauge memory = registry_->GetGauge("trace_query_memory_usage_bytes", tags);
    EXPECT_GE(memory.Get(), 0.0);

    Gauge kv_cache = registry_->GetGauge("trace_query_kv_cache_usage_bytes", tags);
    EXPECT_GE(kv_cache.Get(), 0.0);

    Gauge ttl_eviction = registry_->GetGauge("trace_query_ttl_eviction_count", tags);
    EXPECT_DOUBLE_EQ(0.0, ttl_eviction.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportIntervalMaxHitRate) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1", {1.0}, 0, true));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2, 3}, 0, 0, result);
    manager_->TraceQuery("inst1", {1, 2, 3}, 0, 0, result);

    reporter_->ReportInterval();

    MetricsTags tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}};
    Gauge max_rate = registry_->GetGauge("trace_query_max_hit_rate", tags);
    EXPECT_GT(max_rate.Get(), 0.0);
    EXPECT_LE(max_rate.Get(), 1.0);
}

TEST_F(OptimizerMetricsReporterTest, ReportIntervalMrc) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1", {1.0}, 0, true));

    std::vector<int64_t> keys;
    for (int64_t key = 1; key <= 20; ++key) {
        keys.push_back(key);
    }
    TraceQueryResult result;
    manager_->TraceQuery("inst1", keys, 0, 0, result);
    manager_->TraceQuery("inst1", keys, 0, 0, result);

    reporter_->ReportInterval();

    const std::vector<std::pair<std::string, double>> expected = {
        {"60", 12.0}, {"80", 16.0}, {"90", 18.0}, {"95", 19.0}, {"99", 20.0}, {"99.5", 20.0}};
    for (const auto &[target, blocks] : expected) {
        MetricsTags tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"target_hit_rate_percent", target}};
        EXPECT_DOUBLE_EQ(blocks * 1024.0, registry_->GetGauge("mrc", tags).Get());
    }

    manager_->TraceQuery("inst1", {1}, 0, 0, result);
    reporter_->ReportInterval();
    for (const auto &entry : expected) {
        MetricsTags tags = {
            {"instance_group", "grp1"}, {"instance_id", "inst1"}, {"target_hit_rate_percent", entry.first}};
        EXPECT_DOUBLE_EQ(1.0 * 1024.0, registry_->GetGauge("mrc", tags).Get());
    }

    reporter_->ReportInterval();
    for (const auto &entry : expected) {
        MetricsTags tags = {
            {"instance_group", "grp1"}, {"instance_id", "inst1"}, {"target_hit_rate_percent", entry.first}};
        EXPECT_DOUBLE_EQ(0.0, registry_->GetGauge("mrc", tags).Get());
    }
}

TEST_F(OptimizerMetricsReporterTest, ReportIntervalCapacityEfficiency) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1", {1.0, 5.0}, 0, true));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2, 3}, 0, 0, result);
    manager_->TraceQuery("inst1", {1, 2, 3}, 0, 0, result);

    reporter_->ReportInterval();

    // With theoretical max cache enabled, max_hit_rate should be > 0 after repeated queries.
    MetricsTags tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}};
    Gauge max_rate = registry_->GetGauge("trace_query_max_hit_rate", tags);
    ASSERT_GT(max_rate.Get(), 0.0);

    // capacity_efficiency = cap_hit_rate / max_hit_rate, should be in [0, 1]
    std::string cap_str = std::to_string(1.0);
    MetricsTags cap_tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"capacity_gb", cap_str}};
    Gauge efficiency = registry_->GetGauge("trace_query_capacity_efficiency", cap_tags);
    EXPECT_GE(efficiency.Get(), 0.0);
    EXPECT_LE(efficiency.Get(), 1.0);

    // The largest tier should have efficiency close to 1.0
    std::string cap_str5 = std::to_string(5.0);
    MetricsTags cap_tags5 = {{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"capacity_gb", cap_str5}};
    Gauge efficiency5 = registry_->GetGauge("trace_query_capacity_efficiency", cap_tags5);
    EXPECT_GE(efficiency5.Get(), efficiency.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportIntervalCapacityEfficiencyUndefinedWhenNoQueries) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));

    reporter_->ReportInterval();

    // No queries → max_hit_rate=0 → capacity_efficiency is undefined.
    MetricsTags tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}};
    Gauge max_rate = registry_->GetGauge("trace_query_max_hit_rate", tags);
    EXPECT_DOUBLE_EQ(0.0, max_rate.Get());

    std::string cap_str = std::to_string(1.0);
    MetricsTags cap_tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"capacity_gb", cap_str}};
    Gauge hit_rate = registry_->GetGauge("trace_query_hit_rate", cap_tags);
    EXPECT_DOUBLE_EQ(0.0, hit_rate.Get());
    Gauge efficiency = registry_->GetGauge("trace_query_capacity_efficiency", cap_tags);
    EXPECT_TRUE(std::isnan(efficiency.Get()));
}

TEST_F(OptimizerMetricsReporterTest, ReportIntervalMetrics) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1", {1.0, 5.0}, 0, true));

    TraceQueryResult result;
    ASSERT_EQ(EC_OK, manager_->TraceQuery("inst1", {1, 2, 3}, 0, 0, result));
    ASSERT_EQ(EC_OK, manager_->TraceQuery("inst1", {1, 2, 3}, 0, 0, result));
    reporter_->ReportInterval();

    MetricsTags instance_tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}};
    EXPECT_NEAR(0.5, registry_->GetGauge("interval.query_max_hit_rate", instance_tags).Get(), 1e-9);

    for (double capacity_gb : {1.0, 5.0}) {
        MetricsTags capacity_tags = instance_tags;
        capacity_tags["capacity_gb"] = std::to_string(capacity_gb);
        EXPECT_NEAR(0.5, registry_->GetGauge("interval.query_hit_rate", capacity_tags).Get(), 1e-9);
        EXPECT_NEAR(1.0, registry_->GetGauge("interval.query_capacity_efficiency", capacity_tags).Get(), 1e-9);
    }

    reporter_->ReportInterval();
    EXPECT_TRUE(std::isnan(registry_->GetGauge("interval.query_max_hit_rate", instance_tags).Get()));
    MetricsTags capacity_tags = instance_tags;
    capacity_tags["capacity_gb"] = std::to_string(1.0);
    EXPECT_TRUE(std::isnan(registry_->GetGauge("interval.query_hit_rate", capacity_tags).Get()));
    EXPECT_TRUE(std::isnan(registry_->GetGauge("interval.query_capacity_efficiency", capacity_tags).Get()));
}

TEST_F(OptimizerMetricsReporterTest, ReportIntervalHitAgeBucketRatio) {
    // TTL > 0 triggers TtlCacheIndexerWrapper, enabling age-bucket tracking
    ASSERT_EQ(EC_OK,
              RegisterTestInstance("inst1",
                                   {1.0},
                                   /*ttl_seconds=*/3600,
                                   /*enable_theoretical_max_cache=*/false,
                                   /*linear_step=*/1));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2, 3}, 0, 0, result);
    manager_->TraceQuery("inst1", {1, 2, 3}, 0, 0, result); // all 3 keys hit

    reporter_->ReportInterval();

    // With near-zero age, all hits should fall in the first bucket (threshold=5s)
    MetricsTags bucket_tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"age_bucket", "5s"}};
    Gauge bucket_ratio = registry_->GetGauge("trace_query_hit_age_bucket_ratio", bucket_tags);
    EXPECT_GT(bucket_ratio.Get(), 0.0);

    // The "inf" bucket should have ratio = 0 (no hits that old)
    MetricsTags inf_tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"age_bucket", "inf"}};
    Gauge inf_ratio = registry_->GetGauge("trace_query_hit_age_bucket_ratio", inf_tags);
    EXPECT_DOUBLE_EQ(0.0, inf_ratio.Get());
}

TEST_F(OptimizerMetricsReporterTest, RemoveInstanceMetricsCleansUp) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2, 3}, 0, 0, result);
    manager_->TraceQuery("inst1", {1, 2, 3}, 0, 0, result);

    reporter_->ReportInterval();

    // Verify metrics were written
    MetricsTags tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}};
    Gauge qt = registry_->GetGauge("trace_query_total", tags);
    EXPECT_EQ(2.0, qt.Get());

    // Remove instance metrics
    reporter_->RemoveInstanceMetrics("inst1");

    // After removal, getting the gauge creates a fresh entry with default 0.0
    Gauge qt_after = registry_->GetGauge("trace_query_total", tags);
    EXPECT_DOUBLE_EQ(0.0, qt_after.Get());
}

} // namespace kv_cache_manager
