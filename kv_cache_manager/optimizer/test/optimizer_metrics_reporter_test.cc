#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/online_runtime/online_optimizer_manager.h"
#include "kv_cache_manager/optimizer/service/metrics/optimizer_metrics_collector.h"
#include "kv_cache_manager/optimizer/service/metrics/optimizer_metrics_reporter.h"

namespace kv_cache_manager {

class OptimizerMetricsReporterTest : public TESTBASE {
protected:
    void SetUp() override {
        TESTBASE::SetUp();
        opt_registry_ = std::make_shared<OptimizerRegistryManager>("");
        opt_registry_->Init();
        manager_ = std::make_shared<OnlineOptimizerManager>(opt_registry_);
        registry_ = std::make_shared<MetricsRegistry>();
        reporter_ = std::make_shared<OptimizerMetricsReporter>(manager_, registry_, "test_opt");
    }

    ErrorCode RegisterTestInstance(const std::string &instance_id,
                                   std::vector<double> caps = {1.0},
                                   int64_t ttl_seconds = 0,
                                   bool enable_theoretical_max_cache = false) {
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
        OptimizerInstanceInfo info("grp1", instance_id, 1024, specs, groups, 0, OptimizerStateInfo("full_group", ""));
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

    MetricsTags tags = {{"instance_id", "inst1"}};
    Gauge query_total = registry_->GetGauge("trace_query_total", tags);
    EXPECT_EQ(0.0, query_total.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportAfterQueries) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2, 3}, result);
    manager_->TraceQuery("inst1", {1, 2, 4}, result);

    reporter_->ReportInterval();

    MetricsTags tags = {{"instance_id", "inst1"}};
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
    manager_->TraceQuery("inst1", {1, 2}, result);

    reporter_->ReportInterval();

    MetricsTags tags1 = {{"instance_id", "inst1"}};
    Gauge qt1 = registry_->GetGauge("trace_query_total", tags1);
    EXPECT_EQ(1.0, qt1.Get());

    MetricsTags tags2 = {{"instance_id", "inst2"}};
    Gauge qt2 = registry_->GetGauge("trace_query_total", tags2);
    EXPECT_EQ(0.0, qt2.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportIntervalPerCapacityHitRate) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1", {1.0, 5.0}));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2, 3}, result);
    manager_->TraceQuery("inst1", {1, 2, 3}, result);

    reporter_->ReportInterval();

    MetricsTags tags_cap1 = {{"instance_id", "inst1"}, {"capacity_gb", std::to_string(1.0)}};
    Gauge rate1 = registry_->GetGauge("trace_query_hit_rate", tags_cap1);
    EXPECT_GE(rate1.Get(), 0.0);
    EXPECT_LE(rate1.Get(), 1.0);

    MetricsTags tags_cap5 = {{"instance_id", "inst1"}, {"capacity_gb", std::to_string(5.0)}};
    Gauge rate5 = registry_->GetGauge("trace_query_hit_rate", tags_cap5);
    EXPECT_GE(rate5.Get(), 0.0);
    EXPECT_LE(rate5.Get(), 1.0);

    EXPECT_GE(rate5.Get(), rate1.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportPerQueryWritesToRegistry) {
    auto collector = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
    collector->Init();

    collector->set_instance_id("inst1");
    collector->set_client_ip("10.0.0.1");
    collector->set_total_blocks(10);
    collector->set_cache_hit_count(7);
    collector->set_per_capacity_hits({
        {1.0, 7},
        {0.5, 3},
    });

    reporter_->ReportPerQuery(collector.get());

    MetricsTags tags1 = {{"instance_id", "inst1"}, {"client_ip", "10.0.0.1"}, {"capacity_gb", std::to_string(1.0)}};
    Gauge rate1 = registry_->GetGauge("query_hit_rate", tags1);
    EXPECT_NEAR(0.7, rate1.Get(), 1e-9);

    Gauge hit1 = registry_->GetGauge("query_hit_count", tags1);
    EXPECT_DOUBLE_EQ(7.0, hit1.Get());

    MetricsTags tags2 = {{"instance_id", "inst1"}, {"client_ip", "10.0.0.1"}, {"capacity_gb", std::to_string(0.5)}};
    Gauge rate2 = registry_->GetGauge("query_hit_rate", tags2);
    EXPECT_NEAR(0.3, rate2.Get(), 1e-9);

    MetricsTags base_tags = {{"instance_id", "inst1"}, {"client_ip", "10.0.0.1"}};
    Gauge blocks = registry_->GetGauge("query_total_blocks", base_tags);
    EXPECT_DOUBLE_EQ(10.0, blocks.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportPerQueryMaxHitMetrics) {
    auto collector = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
    collector->Init();

    collector->set_instance_id("inst1");
    collector->set_client_ip("10.0.0.2");
    collector->set_total_blocks(10);
    collector->set_cache_hit_count(7);
    collector->set_per_capacity_hits({{1.0, 7}});
    collector->set_max_hit_count(8);
    collector->set_max_hit_rate(0.8);

    reporter_->ReportPerQuery(collector.get());

    MetricsTags base_tags = {{"instance_id", "inst1"}, {"client_ip", "10.0.0.2"}};
    Gauge max_hit = registry_->GetGauge("query_max_hit_count", base_tags);
    EXPECT_DOUBLE_EQ(8.0, max_hit.Get());

    Gauge max_rate = registry_->GetGauge("query_max_hit_rate", base_tags);
    EXPECT_NEAR(0.8, max_rate.Get(), 1e-9);
}

TEST_F(OptimizerMetricsReporterTest, ReportPerQueryCapacityEfficiency) {
    // Case 1: max_hit_rate > 0 — efficiency is computed
    {
        auto collector = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
        collector->Init();
        collector->set_instance_id("inst1");
        collector->set_client_ip("10.0.0.3");
        collector->set_total_blocks(10);
        collector->set_cache_hit_count(6);
        collector->set_per_capacity_hits({{1.0, 6}, {5.0, 8}});
        collector->set_max_hit_count(8);
        collector->set_max_hit_rate(0.8);

        reporter_->ReportPerQuery(collector.get());

        // capacity 1.0: hit_rate=0.6, efficiency=0.6/0.8=0.75
        MetricsTags tags1 = {{"instance_id", "inst1"}, {"client_ip", "10.0.0.3"}, {"capacity_gb", std::to_string(1.0)}};
        EXPECT_NEAR(0.75, registry_->GetGauge("query_capacity_efficiency", tags1).Get(), 1e-9);

        // capacity 5.0: hit_rate=0.8, efficiency=0.8/0.8=1.0
        MetricsTags tags5 = {{"instance_id", "inst1"}, {"client_ip", "10.0.0.3"}, {"capacity_gb", std::to_string(5.0)}};
        EXPECT_NEAR(1.0, registry_->GetGauge("query_capacity_efficiency", tags5).Get(), 1e-9);
    }

    // Case 2: max_hit_rate = 0 — efficiency is NOT written
    {
        auto collector = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
        collector->Init();
        collector->set_instance_id("inst2");
        collector->set_client_ip("10.0.0.4");
        collector->set_total_blocks(10);
        collector->set_cache_hit_count(5);
        collector->set_per_capacity_hits({{1.0, 5}});
        // max_hit_rate defaults to 0.0

        reporter_->ReportPerQuery(collector.get());

        MetricsTags tags = {{"instance_id", "inst2"}, {"client_ip", "10.0.0.4"}, {"capacity_gb", std::to_string(1.0)}};
        EXPECT_NEAR(0.5, registry_->GetGauge("query_hit_rate", tags).Get(), 1e-9);
        EXPECT_DOUBLE_EQ(0.0, registry_->GetGauge("query_capacity_efficiency", tags).Get());
    }
}

TEST_F(OptimizerMetricsReporterTest, ReportPerQueryMaxHitNotApplicable) {
    auto collector = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
    collector->Init();

    collector->set_instance_id("inst1");
    collector->set_total_blocks(10);
    collector->set_cache_hit_count(7);
    collector->set_per_capacity_hits({{1.0, 7}});

    reporter_->ReportPerQuery(collector.get());

    MetricsTags base_tags = {{"instance_id", "inst1"}, {"client_ip", ""}};
    Gauge blocks = registry_->GetGauge("query_total_blocks", base_tags);
    EXPECT_DOUBLE_EQ(10.0, blocks.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportPerQueryZeroBlocksSkips) {
    auto collector = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
    collector->Init();

    collector->set_instance_id("inst1");
    collector->set_total_blocks(0);
    collector->set_cache_hit_count(0);

    reporter_->ReportPerQuery(collector.get());
}

TEST_F(OptimizerMetricsReporterTest, ReportPerQueryNullCollectorSafe) { reporter_->ReportPerQuery(nullptr); }

TEST_F(OptimizerMetricsReporterTest, ReportIntervalStaticMetrics) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));
    reporter_->ReportInterval();

    MetricsTags tags = {{"instance_id", "inst1"}};

    Gauge avg_bytes = registry_->GetGauge("trace_query_avg_bytes_per_block", tags);
    EXPECT_DOUBLE_EQ(1024.0, avg_bytes.Get());

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
    manager_->TraceQuery("inst1", {1, 2, 3}, result);
    manager_->TraceQuery("inst1", {1, 2, 3}, result);

    reporter_->ReportInterval();

    MetricsTags tags = {{"instance_id", "inst1"}};
    Gauge max_rate = registry_->GetGauge("trace_query_max_hit_rate", tags);
    EXPECT_GT(max_rate.Get(), 0.0);
    EXPECT_LE(max_rate.Get(), 1.0);
}

TEST_F(OptimizerMetricsReporterTest, ReportIntervalCapacityEfficiency) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1", {1.0, 5.0}, 0, true));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2, 3}, result);
    manager_->TraceQuery("inst1", {1, 2, 3}, result);

    reporter_->ReportInterval();

    // With theoretical max cache enabled, max_hit_rate should be > 0 after repeated queries.
    MetricsTags tags = {{"instance_id", "inst1"}};
    Gauge max_rate = registry_->GetGauge("trace_query_max_hit_rate", tags);
    ASSERT_GT(max_rate.Get(), 0.0);

    // capacity_efficiency = cap_hit_rate / max_hit_rate, should be in [0, 1]
    std::string cap_str = std::to_string(1.0);
    MetricsTags cap_tags = {{"instance_id", "inst1"}, {"capacity_gb", cap_str}};
    Gauge efficiency = registry_->GetGauge("trace_query_capacity_efficiency", cap_tags);
    EXPECT_GE(efficiency.Get(), 0.0);
    EXPECT_LE(efficiency.Get(), 1.0);

    // The largest tier should have efficiency close to 1.0
    std::string cap_str5 = std::to_string(5.0);
    MetricsTags cap_tags5 = {{"instance_id", "inst1"}, {"capacity_gb", cap_str5}};
    Gauge efficiency5 = registry_->GetGauge("trace_query_capacity_efficiency", cap_tags5);
    EXPECT_GE(efficiency5.Get(), efficiency.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportIntervalCapacityEfficiencySkippedWhenNoQueries) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));

    reporter_->ReportInterval();

    // No queries → max_hit_rate=0 → capacity_efficiency is not reported.
    // The hit_rate gauge is still reported (0.0), but efficiency gauge is untouched.
    MetricsTags tags = {{"instance_id", "inst1"}};
    Gauge max_rate = registry_->GetGauge("trace_query_max_hit_rate", tags);
    EXPECT_DOUBLE_EQ(0.0, max_rate.Get());

    std::string cap_str = std::to_string(1.0);
    MetricsTags cap_tags = {{"instance_id", "inst1"}, {"capacity_gb", cap_str}};
    Gauge hit_rate = registry_->GetGauge("trace_query_hit_rate", cap_tags);
    EXPECT_DOUBLE_EQ(0.0, hit_rate.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportIntervalHitAgeBucketRatio) {
    // TTL > 0 triggers TtlCacheIndexerWrapper, enabling age-bucket tracking
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1", {1.0}, /*ttl_seconds=*/3600));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2, 3}, result);
    manager_->TraceQuery("inst1", {1, 2, 3}, result); // all 3 keys hit

    reporter_->ReportInterval();

    // With near-zero age, all hits should fall in the first bucket (threshold=5s)
    MetricsTags bucket_tags = {{"instance_id", "inst1"}, {"age_bucket", "5s"}};
    Gauge bucket_ratio = registry_->GetGauge("trace_query_hit_age_bucket_ratio", bucket_tags);
    EXPECT_GT(bucket_ratio.Get(), 0.0);

    // The "inf" bucket should have ratio = 0 (no hits that old)
    MetricsTags inf_tags = {{"instance_id", "inst1"}, {"age_bucket", "inf"}};
    Gauge inf_ratio = registry_->GetGauge("trace_query_hit_age_bucket_ratio", inf_tags);
    EXPECT_DOUBLE_EQ(0.0, inf_ratio.Get());
}

TEST_F(OptimizerMetricsReporterTest, RemoveInstanceMetricsCleansUp) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2, 3}, result);
    manager_->TraceQuery("inst1", {1, 2, 3}, result);

    reporter_->ReportInterval();

    // Verify metrics were written
    MetricsTags tags = {{"instance_id", "inst1"}};
    Gauge qt = registry_->GetGauge("trace_query_total", tags);
    EXPECT_EQ(2.0, qt.Get());

    // Remove instance metrics
    reporter_->RemoveInstanceMetrics("inst1");

    // After removal, getting the gauge creates a fresh entry with default 0.0
    Gauge qt_after = registry_->GetGauge("trace_query_total", tags);
    EXPECT_DOUBLE_EQ(0.0, qt_after.Get());
}

} // namespace kv_cache_manager
