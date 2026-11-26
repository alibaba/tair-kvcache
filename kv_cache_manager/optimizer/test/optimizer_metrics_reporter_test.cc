#include <cmath>
#include <map>
#include <vector>

#include "kmonitor/client/core/MutableMetric.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/manager/online_runtime/online_optimizer_manager.h"
#include "kv_cache_manager/optimizer/metrics/optimizer_kmonitor_metrics_reporter.h"
#include "kv_cache_manager/optimizer/metrics/optimizer_metrics_collector.h"
#include "kv_cache_manager/optimizer/metrics/optimizer_metrics_reporter.h"
#include "stub.h"

namespace kv_cache_manager {

namespace {
using KmonSamples = std::map<std::pair<std::string, MetricsTags>, std::vector<double>>;
thread_local KmonSamples kmon_samples;

void CaptureOptimizerMetric(kmonitor::MutableMetric *metric, const kmonitor::MetricsTags *tags, double value) {
    for (const auto *prefix : {"service.", "query.", "trace.", "interval.", "mrc"}) {
        const auto pos = metric->metric_name_.find(std::string(".") + prefix);
        if (pos != std::string::npos) {
            kmon_samples[{metric->metric_name_.substr(pos + 1), tags->GetTagsMap()}].push_back(value);
            return;
        }
    }
}
} // namespace

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
        group.set_ttl_seconds_list({0});
        group.set_name("grp1");
        group.set_capacity_gb(caps);
        group.set_enable_theoretical_max_cache(enable_theoretical_max_cache);
        group.set_ttl_seconds_list({ttl_seconds});

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

TEST_F(OptimizerMetricsReporterTest, KmonitorAndPrometheusShareMultiTtlSnapshots) {
    auto kmon = std::make_shared<OptimizerKmonitorMetricsReporter>("test_optimizer_parity");
    ASSERT_TRUE(kmon->Init());
    reporter_ = std::make_shared<OptimizerMetricsReporter>(manager_, registry_, kmon);
    Stub stub;
    stub.set(static_cast<void (kmonitor::MutableMetric::*)(const kmonitor::MetricsTags *, double)>(
                 &kmonitor::MutableMetric::Report),
             CaptureOptimizerMetric);

    OptimizerInstanceGroup group;
    group.set_name("grp1");
    group.set_ttl_seconds_list({2, 30});
    group.set_capacity_gb({1.0});
    group.set_enable_theoretical_max_cache(true);
    ASSERT_EQ(EC_OK, manager_->CreateInstanceGroup(group));
    OptimizerInstanceInfo info("grp1",
                               "inst1",
                               16,
                               {LocationSpecInfo("full", 1024)},
                               {LocationSpecGroup("full_group", {"full"})},
                               0,
                               OptimizerStateInfo("full_group", ""));
    RegisterInstanceResult registration;
    ASSERT_EQ(EC_OK, manager_->RegisterInstance(info, registration));
    TraceQueryResult result;
    constexpr int64_t second = 1000000000LL;
    // Two full blocks plus one trailing token: input denominator must be 33,
    // not 32. Only complete blocks have keys.
    ASSERT_EQ(EC_OK, manager_->TraceQuery("inst1", {1, 2}, 33, 100 * second, result));
    ASSERT_EQ(EC_OK, manager_->TraceQuery("inst1", {1, 2}, 33, 104 * second, result));
    kmon_samples.clear();
    reporter_->ReportInterval();

    for (int64_t ttl : {2, 30}) {
        const MetricsTags tags{{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", std::to_string(ttl)}};
        EXPECT_DOUBLE_EQ(66., registry_->GetGauge("trace_query_input_tokens_total", tags).Get());
        EXPECT_DOUBLE_EQ(ttl == 2 ? 0. : 32., registry_->GetGauge("trace_query_max_hit_tokens_total", tags).Get());
        const auto key = std::make_pair(std::string("trace.query_input_tokens_total"), tags);
        ASSERT_EQ(1u, kmon_samples.count(key)) << ::testing::PrintToString(kmon_samples);
        ASSERT_EQ(1u, kmon_samples.at(key).size());
        EXPECT_DOUBLE_EQ(66., kmon_samples.at(key).front());
    }
    // Compare every emitted interval metric, including all MRC targets and
    // reuse quantiles. Both backends must consume the same (non-empty) window.
    for (const auto &[key, values] : kmon_samples) {
        auto name = key.first;
        EXPECT_EQ("inst1", key.second.at("instance_id"));
        EXPECT_EQ(1u, key.second.count("ttl"));
        if (name.rfind("trace.", 0) == 0) {
            name.replace(0, 6, "trace_");
        }
        ASSERT_NE(nullptr, registry_->GetMetricsData(name)) << name;
        ASSERT_EQ(1u, values.size()) << name;
        EXPECT_DOUBLE_EQ(registry_->GetGauge(name, key.second).Get(), values.front()) << name;
        EXPECT_EQ(std::string::npos, name.find("capacity_history"));
    }
    const MetricsTags long_tags{{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "30"}};
    for (const auto *suffix : {"p995", "p999"}) {
        const std::string name = std::string("trace.query_reuse_interval_") + suffix + "_seconds";
        ASSERT_EQ(1u, kmon_samples.count({name, long_tags}));
        EXPECT_DOUBLE_EQ(4., kmon_samples.at({name, long_tags}).front());
    }
    kmon_samples.clear();
    reporter_->ReportInterval();
    for (const auto &[key, values] : kmon_samples) {
        EXPECT_EQ(std::string::npos, key.first.find("reuse_interval"));
    }
    EXPECT_TRUE(std::isnan(registry_->GetGauge("trace_query_reuse_interval_p999_seconds", long_tags).Get()));
    // Cumulative token snapshots do not reset when the reporting window resets.
    EXPECT_DOUBLE_EQ(66., registry_->GetGauge("trace_query_input_tokens_total", long_tags).Get());
    // A real zero interval is not an empty window: both backends report zero.
    ASSERT_EQ(EC_OK, manager_->TraceQuery("inst1", {1, 2}, 33, 104 * second, result));
    kmon_samples.clear();
    reporter_->ReportInterval();
    for (const auto *suffix : {"p995", "p999"}) {
        const std::string kmon_name = std::string("trace.query_reuse_interval_") + suffix + "_seconds";
        ASSERT_EQ(1u, kmon_samples.count({kmon_name, long_tags}));
        EXPECT_DOUBLE_EQ(0., kmon_samples.at({kmon_name, long_tags}).front());
        EXPECT_DOUBLE_EQ(
            0., registry_->GetGauge(std::string("trace_query_reuse_interval_") + suffix + "_seconds", long_tags).Get());
    }
    ASSERT_EQ(EC_OK, manager_->ResetStats("inst1"));
    reporter_->ReportInterval();
    EXPECT_DOUBLE_EQ(0., registry_->GetGauge("trace_query_input_tokens_total", long_tags).Get());
    EXPECT_DOUBLE_EQ(0., registry_->GetGauge("trace_query_max_hit_tokens_total", long_tags).Get());
    reporter_->RemoveInstanceMetrics("inst1");
    EXPECT_TRUE(registry_->GetMetricsData("trace_query_input_tokens_total")->GetMetricsValues().empty());
}

TEST_F(OptimizerMetricsReporterTest, KmonitorServiceRatesCountTokensAndQueriesOnce) {
    auto kmon = std::make_shared<OptimizerKmonitorMetricsReporter>("test_optimizer_service");
    ASSERT_TRUE(kmon->Init());
    reporter_ = std::make_shared<OptimizerMetricsReporter>(manager_, registry_, kmon);
    Stub stub;
    stub.set(static_cast<void (kmonitor::MutableMetric::*)(const kmonitor::MetricsTags *, double)>(
                 &kmonitor::MutableMetric::Report),
             CaptureOptimizerMetric);
    kmon_samples.clear();
    OptimizerServiceMetricsCollector collector(registry_);
    ASSERT_TRUE(collector.Init());
    collector.set_instance_id("inst1");
    collector.set_input_token_len(33);
    collector.set_total_blocks(2);
    collector.set_ttl_results({{2, {}, 0, 0.}, {30, {}, 2, 32. / 33.}});
    collector.set_service_query_rt_us_metrics(123.);
    collector.set_service_error_code_metrics(static_cast<double>(EC_ERROR));
    reporter_->ReportPerQuery(&collector);
    const MetricsTags tags{{"instance_group", ""}, {"instance_id", "inst1"}};
    for (const auto &[name, value] :
         std::map<std::string, double>{{"service.qps", 1.},
                                       {"service.error_qps", 1.},
                                       {"service.input_tokens_qps", 33.},
                                       {"service.query_rt_us", 123.},
                                       {"service.error_code", static_cast<double>(EC_ERROR)}}) {
        ASSERT_EQ(1u, kmon_samples.count({name, tags})) << name << ::testing::PrintToString(kmon_samples);
        ASSERT_EQ(1u, kmon_samples.at({name, tags}).size()) << name;
        EXPECT_DOUBLE_EQ(value, kmon_samples.at({name, tags}).front()) << name;
    }
    EXPECT_EQ(33u, registry_->GetCounter("service.input_tokens_total", tags).Get());
    EXPECT_EQ(1u, registry_->GetCounter("service.query_counter", tags).Get());
    for (const auto &[key, values] : kmon_samples) {
        if (key.first.rfind("service.", 0) == 0) {
            EXPECT_EQ(0u, key.second.count("ttl"));
        } else {
            EXPECT_EQ("inst1", key.second.at("instance_id"));
            EXPECT_EQ(1u, key.second.count("ttl"));
        }
    }
    for (const auto ttl : {2, 30}) {
        const MetricsTags query_tags{
            {"instance_group", ""}, {"instance_id", "inst1"}, {"client_ip", ""}, {"ttl", std::to_string(ttl)}};
        const auto key = std::make_pair(std::string("query.max_hit_rate"), query_tags);
        ASSERT_EQ(1u, kmon_samples.count(key));
        EXPECT_DOUBLE_EQ(ttl == 2 ? 0. : 32. / 33., kmon_samples.at(key).front());
    }
}

TEST_F(OptimizerMetricsReporterTest, TokenSnapshotsOmitDisabledTheory) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1", {1.0}, 30, false));
    TraceQueryResult result;
    ASSERT_EQ(EC_OK, manager_->TraceQuery("inst1", {1, 2}, 0, 0, result));
    reporter_->ReportInterval();
    const MetricsTags tags{{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "30"}};
    EXPECT_DOUBLE_EQ(2048., registry_->GetGauge("trace_query_input_tokens_total", tags).Get());
    EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_max_hit_tokens_total"));
    EXPECT_NE(nullptr, registry_->GetMetricsData("trace_query_hit_tokens_total"));
}

TEST_F(OptimizerMetricsReporterTest, TokenSnapshotsOmitUnconfiguredCapacity) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1", {}, 30, true));
    reporter_->ReportInterval();
    EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_hit_tokens_total"));
    EXPECT_NE(nullptr, registry_->GetMetricsData("trace_query_max_hit_tokens_total"));
}

TEST_F(OptimizerMetricsReporterTest, ReportEmptyState) { reporter_->ReportInterval(); }

TEST_F(OptimizerMetricsReporterTest, ReportAfterRegistration) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));
    reporter_->ReportInterval();

    MetricsTags tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "0"}};
    Gauge query_total = registry_->GetGauge("trace_query_total", tags);
    EXPECT_EQ(0.0, query_total.Get());
    EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_reuse_interval_count"));
    EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_hit_age_bucket_ratio"));
}

TEST_F(OptimizerMetricsReporterTest, ReportAfterQueries) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2, 3}, 0, 0, result);
    manager_->TraceQuery("inst1", {1, 2, 4}, 0, 0, result);

    reporter_->ReportInterval();

    MetricsTags tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "0"}};
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

    MetricsTags tags1 = {{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "0"}};
    Gauge qt1 = registry_->GetGauge("trace_query_total", tags1);
    EXPECT_EQ(1.0, qt1.Get());

    MetricsTags tags2 = {{"instance_group", "grp1"}, {"instance_id", "inst2"}, {"ttl", "0"}};
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
        {"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "0"}, {"capacity_gb", std::to_string(1.0)}};
    Gauge rate1 = registry_->GetGauge("trace_query_hit_rate", tags_cap1);
    EXPECT_GE(rate1.Get(), 0.0);
    EXPECT_LE(rate1.Get(), 1.0);

    MetricsTags tags_cap5 = {
        {"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "0"}, {"capacity_gb", std::to_string(5.0)}};
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
    collector->set_ttl_results({{0,
                                 {
                                     {1.0, 7},
                                     {0.5, 3},
                                 },
                                 -1,
                                 -1}});
    reporter_->ReportPerQuery(collector.get());

    MetricsTags tags1 = {{"instance_group", ""},
                         {"instance_id", "inst1"},
                         {"ttl", "0"},
                         {"client_ip", "10.0.0.1"},
                         {"capacity_gb", std::to_string(1.0)}};
    Gauge rate1 = registry_->GetGauge("query_hit_rate", tags1);
    EXPECT_NEAR(0.7, rate1.Get(), 1e-9);

    Gauge hit1 = registry_->GetGauge("query_hit_count", tags1);
    EXPECT_DOUBLE_EQ(7.0, hit1.Get());

    MetricsTags tags2 = {{"instance_group", ""},
                         {"instance_id", "inst1"},
                         {"ttl", "0"},
                         {"client_ip", "10.0.0.1"},
                         {"capacity_gb", std::to_string(0.5)}};
    Gauge rate2 = registry_->GetGauge("query_hit_rate", tags2);
    EXPECT_NEAR(0.3, rate2.Get(), 1e-9);

    MetricsTags base_tags = {{"instance_group", ""}, {"instance_id", "inst1"}, {"ttl", "0"}, {"client_ip", "10.0.0.1"}};
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
    collector->set_ttl_results({{0, {{1.0, 7}}, 8, 0.8}});

    reporter_->ReportPerQuery(collector.get());

    MetricsTags base_tags = {{"instance_group", ""}, {"instance_id", "inst1"}, {"ttl", "0"}, {"client_ip", "10.0.0.2"}};
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
    collector->set_ttl_results({{0, {{1.0, 6}, {5.0, 8}}, 8, 0.8}});

    reporter_->ReportPerQuery(collector.get());

    MetricsTags tags1 = {{"instance_group", ""},
                         {"instance_id", "inst1"},
                         {"ttl", "0"},
                         {"client_ip", "10.0.0.3"},
                         {"capacity_gb", std::to_string(1.0)}};
    MetricsTags tags5 = {{"instance_group", ""},
                         {"instance_id", "inst1"},
                         {"ttl", "0"},
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
    zero_hit_collector->set_ttl_results({{0, {{1.0, 0}, {5.0, 0}}, 0, 0.0}});

    reporter_->ReportPerQuery(zero_hit_collector.get());

    EXPECT_TRUE(std::isnan(registry_->GetGauge("query_capacity_efficiency", tags1).Get()));
    EXPECT_TRUE(std::isnan(registry_->GetGauge("query_capacity_efficiency", tags5).Get()));
}

TEST_F(OptimizerMetricsReporterTest, ReportPerQueryMaxHitNotApplicable) {
    auto collector = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
    ASSERT_TRUE(collector->Init());
    collector->set_instance_id("inst1");
    collector->set_total_blocks(10);
    collector->set_ttl_results({{0, {{1.0, 7}}, -1, -1}});
    reporter_->ReportPerQuery(collector.get());

    MetricsTags base_tags = {{"instance_group", ""}, {"instance_id", "inst1"}, {"ttl", "0"}, {"client_ip", ""}};
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

    MetricsTags tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "0"}};

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

    MetricsTags tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "0"}};
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
        MetricsTags tags = {
            {"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "0"}, {"target_hit_rate_percent", target}};
        EXPECT_DOUBLE_EQ(blocks * 1024.0, registry_->GetGauge("mrc", tags).Get());
    }

    manager_->TraceQuery("inst1", {1}, 0, 0, result);
    reporter_->ReportInterval();
    for (const auto &entry : expected) {
        MetricsTags tags = {{"instance_group", "grp1"},
                            {"instance_id", "inst1"},
                            {"ttl", "0"},
                            {"target_hit_rate_percent", entry.first}};
        EXPECT_DOUBLE_EQ(1.0 * 1024.0, registry_->GetGauge("mrc", tags).Get());
    }

    reporter_->ReportInterval();
    for (const auto &entry : expected) {
        MetricsTags tags = {{"instance_group", "grp1"},
                            {"instance_id", "inst1"},
                            {"ttl", "0"},
                            {"target_hit_rate_percent", entry.first}};
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
    MetricsTags tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "0"}};
    Gauge max_rate = registry_->GetGauge("trace_query_max_hit_rate", tags);
    ASSERT_GT(max_rate.Get(), 0.0);

    // capacity_efficiency = cap_hit_rate / max_hit_rate, should be in [0, 1]
    std::string cap_str = std::to_string(1.0);
    MetricsTags cap_tags = {
        {"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "0"}, {"capacity_gb", cap_str}};
    Gauge efficiency = registry_->GetGauge("trace_query_capacity_efficiency", cap_tags);
    EXPECT_GE(efficiency.Get(), 0.0);
    EXPECT_LE(efficiency.Get(), 1.0);

    // The largest tier should have efficiency close to 1.0
    std::string cap_str5 = std::to_string(5.0);
    MetricsTags cap_tags5 = {
        {"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "0"}, {"capacity_gb", cap_str5}};
    Gauge efficiency5 = registry_->GetGauge("trace_query_capacity_efficiency", cap_tags5);
    EXPECT_GE(efficiency5.Get(), efficiency.Get());
}

TEST_F(OptimizerMetricsReporterTest, ReportIntervalCapacityEfficiencyUndefinedWhenNoQueries) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));

    reporter_->ReportInterval();

    // No queries → max_hit_rate=0 → capacity_efficiency is undefined.
    MetricsTags tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "0"}};
    Gauge max_rate = registry_->GetGauge("trace_query_max_hit_rate", tags);
    EXPECT_DOUBLE_EQ(0.0, max_rate.Get());

    std::string cap_str = std::to_string(1.0);
    MetricsTags cap_tags = {
        {"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "0"}, {"capacity_gb", cap_str}};
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

    MetricsTags instance_tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "0"}};
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

TEST_F(OptimizerMetricsReporterTest, LegacyIndexerDoesNotReportReuseInterval) {
    for (int64_t ttl_seconds : {0, 3600}) {
        registry_ = std::make_shared<MetricsRegistry>();
        reporter_ = std::make_shared<OptimizerMetricsReporter>(manager_, registry_);
        const std::string instance_id = "legacy_" + std::to_string(ttl_seconds);
        ASSERT_EQ(EC_OK, RegisterTestInstance(instance_id, {1.0}, ttl_seconds, false, 1));
        TraceQueryResult result;
        ASSERT_EQ(EC_OK, manager_->TraceQuery(instance_id, {1, 2, 3}, 0, 0, result));
        ASSERT_EQ(EC_OK, manager_->TraceQuery(instance_id, {1, 2, 3}, 0, 0, result));
        reporter_->ReportInterval();

        EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_reuse_interval_count"));
        EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_reuse_interval_bucket_ratio"));
        EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_reuse_interval_min_seconds"));
        EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_reuse_interval_max_seconds"));
        EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_reuse_interval_avg_seconds"));
        EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_reuse_interval_p95_seconds"));
        EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_reuse_interval_p99_seconds"));
        EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_reuse_interval_p995_seconds"));
        EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_reuse_interval_p999_seconds"));
        EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_input_tokens_total"));
        EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_max_hit_tokens_total"));
        EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_hit_tokens_total"));
        EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_hit_age_bucket_ratio"));
        MetricsTags tags = {
            {"instance_group", "grp1"}, {"instance_id", instance_id}, {"ttl", std::to_string(ttl_seconds)}};
        EXPECT_DOUBLE_EQ(2.0, registry_->GetGauge("trace_query_total", tags).Get());
    }
}

TEST_F(OptimizerMetricsReporterTest, ReportFullAttentionReuseIntervalWindow) {
    constexpr int64_t kSecond = 1000000000LL;
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));
    TraceQueryResult result;
    ASSERT_EQ(EC_OK, manager_->TraceQuery("inst1", {1, 2}, 0, 100 * kSecond, result));
    ASSERT_EQ(EC_OK, manager_->TraceQuery("inst1", {1, 2}, 0, 104 * kSecond, result));
    ASSERT_EQ(EC_OK, manager_->TraceQuery("inst1", {1}, 0, 140 * kSecond, result));
    reporter_->ReportInterval();

    MetricsTags tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "0"}};
    EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_reuse_interval_count"));
    EXPECT_DOUBLE_EQ(4.0, registry_->GetGauge("trace_query_reuse_interval_min_seconds", tags).Get());
    EXPECT_DOUBLE_EQ(36.0, registry_->GetGauge("trace_query_reuse_interval_max_seconds", tags).Get());
    // Weighted by block samples (4, 4, 36), not by request averages.
    EXPECT_NEAR(44.0 / 3.0, registry_->GetGauge("trace_query_reuse_interval_avg_seconds", tags).Get(), 1e-9);
    EXPECT_DOUBLE_EQ(36.0, registry_->GetGauge("trace_query_reuse_interval_p95_seconds", tags).Get());
    EXPECT_DOUBLE_EQ(36.0, registry_->GetGauge("trace_query_reuse_interval_p99_seconds", tags).Get());
    EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_reuse_interval_bucket_ratio"));

    reporter_->ReportInterval();
    for (const auto *name : {"trace_query_reuse_interval_min_seconds",
                             "trace_query_reuse_interval_max_seconds",
                             "trace_query_reuse_interval_avg_seconds",
                             "trace_query_reuse_interval_p95_seconds",
                             "trace_query_reuse_interval_p99_seconds"}) {
        EXPECT_TRUE(std::isnan(registry_->GetGauge(name, tags).Get()));
    }
    EXPECT_EQ(nullptr, registry_->GetMetricsData("trace_query_reuse_interval_count"));

    // The next window is independent, but last-access history survives.
    ASSERT_EQ(EC_OK, manager_->TraceQuery("inst1", {1}, 0, 150 * kSecond, result));
    reporter_->ReportInterval();
    EXPECT_DOUBLE_EQ(10.0, registry_->GetGauge("trace_query_reuse_interval_min_seconds", tags).Get());
    EXPECT_DOUBLE_EQ(10.0, registry_->GetGauge("trace_query_reuse_interval_max_seconds", tags).Get());
    EXPECT_DOUBLE_EQ(10.0, registry_->GetGauge("trace_query_reuse_interval_avg_seconds", tags).Get());
    EXPECT_DOUBLE_EQ(10.0, registry_->GetGauge("trace_query_reuse_interval_p95_seconds", tags).Get());
    EXPECT_DOUBLE_EQ(10.0, registry_->GetGauge("trace_query_reuse_interval_p99_seconds", tags).Get());

    // A zero interval is valid and is different from an empty window.
    ASSERT_EQ(EC_OK, manager_->TraceQuery("inst1", {1}, 0, 150 * kSecond, result));
    reporter_->ReportInterval();
    EXPECT_DOUBLE_EQ(0.0, registry_->GetGauge("trace_query_reuse_interval_min_seconds", tags).Get());
    EXPECT_DOUBLE_EQ(0.0, registry_->GetGauge("trace_query_reuse_interval_max_seconds", tags).Get());
    EXPECT_DOUBLE_EQ(0.0, registry_->GetGauge("trace_query_reuse_interval_avg_seconds", tags).Get());
    EXPECT_DOUBLE_EQ(0.0, registry_->GetGauge("trace_query_reuse_interval_p95_seconds", tags).Get());
    EXPECT_DOUBLE_EQ(0.0, registry_->GetGauge("trace_query_reuse_interval_p99_seconds", tags).Get());
}

TEST_F(OptimizerMetricsReporterTest, RemoveInstanceMetricsCleansUp) {
    ASSERT_EQ(EC_OK, RegisterTestInstance("inst1"));

    TraceQueryResult result;
    manager_->TraceQuery("inst1", {1, 2, 3}, 0, 0, result);
    manager_->TraceQuery("inst1", {1, 2, 3}, 0, 0, result);

    reporter_->ReportInterval();

    // Verify metrics were written
    MetricsTags tags = {{"instance_group", "grp1"}, {"instance_id", "inst1"}, {"ttl", "0"}};
    Gauge qt = registry_->GetGauge("trace_query_total", tags);
    EXPECT_EQ(2.0, qt.Get());

    ASSERT_EQ(EC_OK, manager_->RemoveInstance("inst1"));
    reporter_->RemoveInstanceMetrics("inst1");
    reporter_->ReportInterval();
    EXPECT_TRUE(registry_->GetMetricsData("trace_query_total")->GetMetricsValues().empty());

    // After removal, getting the gauge creates a fresh entry with default 0.0
    Gauge qt_after = registry_->GetGauge("trace_query_total", tags);
    EXPECT_DOUBLE_EQ(0.0, qt_after.Get());
}

TEST_F(OptimizerMetricsReporterTest, MultiTtlMetricsCountServiceRequestOnlyOnce) {
    auto collector = std::make_shared<OptimizerServiceMetricsCollector>(registry_);
    ASSERT_TRUE(collector->Init());
    collector->set_instance_id("real");
    collector->set_total_blocks(2);
    collector->set_input_token_len(8);
    collector->set_ttl_results({{10, {}, 0, 0.0}, {30, {}, 2, 1.0}});
    reporter_->ReportPerQuery(collector.get());
    EXPECT_EQ(1u, registry_->GetCounter("service.query_counter").Get());
    EXPECT_EQ(8u, registry_->GetCounter("service.input_tokens_total").Get());
    const MetricsTags service_tags = {{"instance_group", ""}, {"instance_id", "real"}};
    EXPECT_EQ(1u, registry_->GetCounter("service.query_counter", service_tags).Get());
    for (int64_t ttl : {10, 30}) {
        const MetricsTags tags = {
            {"instance_group", ""}, {"instance_id", "real"}, {"ttl", std::to_string(ttl)}, {"client_ip", ""}};
        EXPECT_DOUBLE_EQ(ttl == 10 ? 0.0 : 1.0, registry_->GetGauge("query_max_hit_rate", tags).Get());
    }
    const MetricsTags other_tags{{"instance_id", "other"}, {"ttl", "30"}};
    registry_->GetGauge("query_max_hit_rate", other_tags) = 0.5;
    reporter_->RemoveInstanceMetrics("real");
    EXPECT_EQ(1u, registry_->GetMetricsData("query_max_hit_rate")->GetMetricsValues().size());
    EXPECT_DOUBLE_EQ(0.5, registry_->GetGauge("query_max_hit_rate", other_tags).Get());
}

} // namespace kv_cache_manager
