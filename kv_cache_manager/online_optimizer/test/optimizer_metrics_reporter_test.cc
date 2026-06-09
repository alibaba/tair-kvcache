#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/online_optimizer/metrics/optimizer_metrics_reporter.h"

#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/online_optimizer/manager/online_optimizer_manager.h"
#include "kv_cache_manager/online_optimizer/metrics/optimizer_metrics_collector.h"

namespace kv_cache_manager {

class OptimizerMetricsReporterTest : public TESTBASE {
protected:
    void SetUp() override {
        TESTBASE::SetUp();
        manager_ = std::make_shared<OnlineOptimizerManager>(nullptr);
        registry_ = std::make_shared<MetricsRegistry>();
        reporter_ = std::make_shared<OptimizerMetricsReporter>(manager_, registry_, "test_opt");
    }

    ErrorCode RegisterTestInstance(const std::string &instance_id,
                                   std::vector<double> caps = {1.0}) {
        OptimizerInstanceGroup group;
        group.set_name("grp1");
        group.set_enabled(true);
        group.set_capacity_gb(caps);
        group.set_indexer_type("fenwick_lru");

        std::vector<LocationSpecInfo> specs = {LocationSpecInfo("full", 1024)};
        OptimizerInstanceInfo info("grp1", instance_id, 1024, specs, {});
        RegisterInstanceResult result;
        return manager_->RegisterInstance(info, group, result);
    }

    std::shared_ptr<OnlineOptimizerManager> manager_;
    std::shared_ptr<MetricsRegistry> registry_;
    std::shared_ptr<OptimizerMetricsReporter> reporter_;
};

TEST_F(OptimizerMetricsReporterTest, ReportEmptyState) {
    reporter_->ReportInterval();
}

TEST_F(OptimizerMetricsReporterTest, ReportAfterRegistration) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));
    reporter_->ReportInterval();

    MetricsTags tags = {{"instance_id", "inst1"}};
    Gauge query_total = registry_->GetGauge("test_opt.trace_query_total", tags);
    EXPECT_EQ(0.0, query_total.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportAfterQueries) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2, 3}, result);
    manager_->TraceQuery("inst1", {1, 2, 4}, result);

    reporter_->ReportInterval();

    MetricsTags tags = {{"instance_id", "inst1"}};
    Gauge query_total = registry_->GetGauge("test_opt.trace_query_total", tags);
    EXPECT_EQ(2.0, query_total.Get());

    Gauge blocks_total = registry_->GetGauge("test_opt.trace_query_blocks_total", tags);
    EXPECT_EQ(6.0, blocks_total.Get());

    Gauge unique_keys = registry_->GetGauge("test_opt.trace_query_unique_keys", tags);
    EXPECT_EQ(4.0, unique_keys.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportMultipleInstances) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst2"));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2}, result);

    reporter_->ReportInterval();

    MetricsTags tags1 = {{"instance_id", "inst1"}};
    Gauge qt1 = registry_->GetGauge("test_opt.trace_query_total", tags1);
    EXPECT_EQ(1.0, qt1.Get());

    MetricsTags tags2 = {{"instance_id", "inst2"}};
    Gauge qt2 = registry_->GetGauge("test_opt.trace_query_total", tags2);
    EXPECT_EQ(0.0, qt2.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportIntervalPerCapacityHitRate) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1", {1.0, 5.0}));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2, 3}, result);
    manager_->TraceQuery("inst1", {1, 2, 3}, result);

    reporter_->ReportInterval();

    MetricsTags tags_cap1 = {{"instance_id", "inst1"}, {"capacity_gb", std::to_string(1.0)}};
    Gauge rate1 = registry_->GetGauge("test_opt.trace_query_hit_rate", tags_cap1);
    EXPECT_GE(rate1.Get(), 0.0);
    EXPECT_LE(rate1.Get(), 1.0);

    MetricsTags tags_cap5 = {{"instance_id", "inst1"}, {"capacity_gb", std::to_string(5.0)}};
    Gauge rate5 = registry_->GetGauge("test_opt.trace_query_hit_rate", tags_cap5);
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
    Gauge rate1 = registry_->GetGauge("test_opt.query_hit_rate", tags1);
    EXPECT_NEAR(0.7, rate1.Get(), 1e-9);

    Gauge hit1 = registry_->GetGauge("test_opt.query_hit_count", tags1);
    EXPECT_DOUBLE_EQ(7.0, hit1.Get());

    MetricsTags tags2 = {{"instance_id", "inst1"}, {"client_ip", "10.0.0.1"}, {"capacity_gb", std::to_string(0.5)}};
    Gauge rate2 = registry_->GetGauge("test_opt.query_hit_rate", tags2);
    EXPECT_NEAR(0.3, rate2.Get(), 1e-9);

    MetricsTags base_tags = {{"instance_id", "inst1"}, {"client_ip", "10.0.0.1"}};
    Gauge blocks = registry_->GetGauge("test_opt.query_total_blocks", base_tags);
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
    Gauge max_hit = registry_->GetGauge("test_opt.query_max_hit_count", base_tags);
    EXPECT_DOUBLE_EQ(8.0, max_hit.Get());

    Gauge max_rate = registry_->GetGauge("test_opt.query_max_hit_rate", base_tags);
    EXPECT_NEAR(0.8, max_rate.Get(), 1e-9);
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
    Gauge blocks = registry_->GetGauge("test_opt.query_total_blocks", base_tags);
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

TEST_F(OptimizerMetricsReporterTest, ReportPerQueryNullCollectorSafe) {
    reporter_->ReportPerQuery(nullptr);
}

} // namespace kv_cache_manager
