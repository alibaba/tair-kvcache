#include <cstdint>
#include <memory>
#include <string>
#include <variant>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/prometheus_exporter.h"

using namespace kv_cache_manager;

class PrometheusExporterTest : public TESTBASE {
public:
    void SetUp() override { registry_ = std::make_shared<MetricsRegistry>(); }
    void TearDown() override {}

    std::shared_ptr<MetricsRegistry> registry_;
};

TEST_F(PrometheusExporterTest, EmptyRegistry) {
    std::string output = PrometheusExporter::Expose(*registry_);
    ASSERT_TRUE(output.empty());
}

TEST_F(PrometheusExporterTest, SingleCounter) {
    Counter c = registry_->GetCounter("service.query_counter");
    ++c;
    ++c;
    ++c;

    std::string output = PrometheusExporter::Expose(*registry_);
    EXPECT_NE(output.find("# HELP kvcm_service_query_counter"), std::string::npos);
    EXPECT_NE(output.find("# TYPE kvcm_service_query_counter counter"), std::string::npos);
    EXPECT_NE(output.find("kvcm_service_query_counter 3"), std::string::npos);
}

TEST_F(PrometheusExporterTest, SingleGauge) {
    Gauge g = registry_->GetGauge("meta_indexer.total_key_count");
    g = 42000.0;

    std::string output = PrometheusExporter::Expose(*registry_);
    EXPECT_NE(output.find("# HELP kvcm_meta_indexer_total_key_count"), std::string::npos);
    EXPECT_NE(output.find("# TYPE kvcm_meta_indexer_total_key_count gauge"), std::string::npos);
    EXPECT_NE(output.find("kvcm_meta_indexer_total_key_count 42000"), std::string::npos);
}

TEST_F(PrometheusExporterTest, MetricsWithTags) {
    MetricsTags tags = {{"instance_group", "group_a"}, {"type", "hf3fs"}};
    Gauge g = registry_->GetGauge("data_storage.storage_usage_ratio", tags);
    g = 0.75;

    std::string output = PrometheusExporter::Expose(*registry_);
    // Labels should appear in sorted order (std::map order)
    EXPECT_NE(output.find("kvcm_data_storage_storage_usage_ratio"
                          "{instance_group=\"group_a\",type=\"hf3fs\"} 0.75"),
              std::string::npos)
        << "Actual output:\n"
        << output;
}

TEST_F(PrometheusExporterTest, MultipleTagSets) {
    MetricsTags tags_a = {{"instance_group", "group_a"}};
    MetricsTags tags_b = {{"instance_group", "group_b"}};

    Gauge ga = registry_->GetGauge("cache_manager_group.usage_ratio", tags_a);
    Gauge gb = registry_->GetGauge("cache_manager_group.usage_ratio", tags_b);
    ga = 0.5;
    gb = 0.8;

    std::string output = PrometheusExporter::Expose(*registry_);

    // TYPE line should appear only once
    size_t first = output.find("# TYPE kvcm_cache_manager_group_usage_ratio");
    ASSERT_NE(first, std::string::npos);
    size_t second = output.find("# TYPE kvcm_cache_manager_group_usage_ratio", first + 1);
    EXPECT_EQ(second, std::string::npos) << "TYPE line appears more than once";

    // Both label sets should be present
    EXPECT_NE(output.find("{instance_group=\"group_a\"}"), std::string::npos);
    EXPECT_NE(output.find("{instance_group=\"group_b\"}"), std::string::npos);
}

TEST_F(PrometheusExporterTest, CustomPrefix) {
    Counter c = registry_->GetCounter("service.query_counter");
    ++c;

    std::string output = PrometheusExporter::Expose(*registry_, "myapp");
    EXPECT_NE(output.find("myapp_service_query_counter"), std::string::npos);
    EXPECT_EQ(output.find("kvcm_"), std::string::npos);
}

TEST_F(PrometheusExporterTest, PrefixSanitization) {
    Gauge g = registry_->GetGauge("service.qps");
    g = 1.0;

    std::string output = PrometheusExporter::Expose(*registry_, "my-app.v2");
    EXPECT_NE(output.find("my_app_v2_service_qps"), std::string::npos) << "Actual output:\n" << output;
    EXPECT_EQ(output.find("my-app"), std::string::npos);
}

TEST_F(PrometheusExporterTest, LabelValueEscaping) {
    MetricsTags tags = {{"path", "a\\b\"c\nd"}};
    Gauge g = registry_->GetGauge("test.escape", tags);
    g = 1.0;

    std::string output = PrometheusExporter::Expose(*registry_);
    EXPECT_NE(output.find("path=\"a\\\\b\\\"c\\nd\""), std::string::npos) << "Actual output:\n" << output;
}

TEST_F(PrometheusExporterTest, DotAndDashInNames) {
    Gauge g = registry_->GetGauge("my-group.some-metric");
    g = 1.0;

    std::string output = PrometheusExporter::Expose(*registry_);
    EXPECT_NE(output.find("kvcm_my_group_some_metric"), std::string::npos);
}

TEST_F(PrometheusExporterTest, LabelKeySanitization) {
    MetricsTags tags = {{"storage.type", "hf3fs"}, {"host-name", "node01"}};
    Gauge g = registry_->GetGauge("test.label_keys", tags);
    g = 1.0;

    std::string output = PrometheusExporter::Expose(*registry_);
    // dots and dashes in label keys should be replaced with underscores
    EXPECT_NE(output.find("storage_type=\"hf3fs\""), std::string::npos) << "Actual output:\n" << output;
    EXPECT_NE(output.find("host_name=\"node01\""), std::string::npos) << "Actual output:\n" << output;
    // originals should not appear as label keys
    EXPECT_EQ(output.find("storage.type="), std::string::npos);
    EXPECT_EQ(output.find("host-name="), std::string::npos);
}

TEST_F(PrometheusExporterTest, HelpLineContainsOriginalName) {
    Gauge g = registry_->GetGauge("meta_searcher.indexer_get_time_us");
    g = 99.5;

    std::string output = PrometheusExporter::Expose(*registry_);
    EXPECT_NE(output.find("# HELP kvcm_meta_searcher_indexer_get_time_us "
                          "meta_searcher.indexer_get_time_us"),
              std::string::npos);
}

TEST_F(PrometheusExporterTest, MultipleMetricFamilies) {
    Counter c = registry_->GetCounter("service.query_counter");
    Gauge g = registry_->GetGauge("meta_indexer.total_key_count");
    ++c;
    g = 100.0;

    std::string output = PrometheusExporter::Expose(*registry_);
    EXPECT_NE(output.find("# TYPE kvcm_meta_indexer_total_key_count gauge"), std::string::npos);
    EXPECT_NE(output.find("# TYPE kvcm_service_query_counter counter"), std::string::npos);
}

// -- SanitizeIdentifier specification tests (exercised through Expose) --

TEST_F(PrometheusExporterTest, SpecialCharsInMetricName) {
    // characters beyond '.' and '-': @, space, colon, slash
    Gauge g = registry_->GetGauge("ns:group/metric name@v2");
    g = 1.0;

    std::string output = PrometheusExporter::Expose(*registry_);
    EXPECT_NE(output.find("kvcm_ns_group_metric_name_v2"), std::string::npos) << "Actual output:\n" << output;
    // HELP line must preserve the original raw name (pre-sanitization)
    EXPECT_NE(output.find("# HELP kvcm_ns_group_metric_name_v2 ns:group/metric name@v2"), std::string::npos)
        << "Actual output:\n"
        << output;
}

TEST_F(PrometheusExporterTest, AllValidInputUnchanged) {
    // a name with only [a-zA-Z0-9_] should pass through unchanged
    Gauge g = registry_->GetGauge("abc_XYZ_012");
    g = 1.0;

    std::string output = PrometheusExporter::Expose(*registry_);
    EXPECT_NE(output.find("kvcm_abc_XYZ_012"), std::string::npos);
}

TEST_F(PrometheusExporterTest, ConsecutiveSpecialChars) {
    Gauge g = registry_->GetGauge("a..b--c");
    g = 1.0;

    std::string output = PrometheusExporter::Expose(*registry_);
    EXPECT_NE(output.find("kvcm_a__b__c"), std::string::npos) << "Actual output:\n" << output;
}

TEST_F(PrometheusExporterTest, LeadingDigitInMetricName) {
    Gauge g = registry_->GetGauge("3abc");
    g = 1.0;

    std::string output = PrometheusExporter::Expose(*registry_);
    // leading digit gets a '_' prefix: "3abc" -> "_3abc"
    EXPECT_NE(output.find("kvcm__3abc"), std::string::npos) << "Actual output:\n" << output;
}

TEST_F(PrometheusExporterTest, LeadingDigitInLabelKey) {
    MetricsTags tags = {{"1abc", "val"}};
    Gauge g = registry_->GetGauge("test.leading_digit_label", tags);
    g = 1.0;

    std::string output = PrometheusExporter::Expose(*registry_);
    EXPECT_NE(output.find("_1abc=\"val\""), std::string::npos) << "Actual output:\n" << output;
    // raw key must not appear
    EXPECT_EQ(output.find("{1abc="), std::string::npos);
}

TEST_F(PrometheusExporterTest, LeadingDigitInPrefix) {
    Gauge g = registry_->GetGauge("service.qps");
    g = 1.0;

    std::string output = PrometheusExporter::Expose(*registry_, "9app");
    EXPECT_NE(output.find("_9app_service_qps"), std::string::npos) << "Actual output:\n" << output;
}

// Untouched series (registered but never written) must not appear in
// the output, and a metric family with no touched series must not
// emit its # HELP / # TYPE header either.
TEST_F(PrometheusExporterTest, RegisteredCounterNeverWrittenIsSkipped) {
    Counter c = registry_->GetCounter("service.query_counter");
    (void)c;

    std::string output = PrometheusExporter::Expose(*registry_);
    EXPECT_TRUE(output.empty()) << "Actual output:\n" << output;
}

TEST_F(PrometheusExporterTest, RegisteredGaugeNeverWrittenIsSkipped) {
    Gauge g = registry_->GetGauge("data_storage.healthy_status");
    (void)g;

    std::string output = PrometheusExporter::Expose(*registry_);
    EXPECT_TRUE(output.empty()) << "Actual output:\n" << output;
}

TEST_F(PrometheusExporterTest, OnlyTouchedSeriesAppearInFamily) {
    MetricsTags tags_a = {{"api_name", "GetCacheLocation"}};
    MetricsTags tags_b = {{"api_name", "StartWriteCache"}};

    Gauge ga = registry_->GetGauge("manager.batch_add_location_time_us", tags_a);
    Gauge gb = registry_->GetGauge("manager.batch_add_location_time_us", tags_b);

    // only the StartWriteCache series is actually written
    gb = 42.0;
    (void)ga;

    std::string output = PrometheusExporter::Expose(*registry_);

    EXPECT_NE(output.find("# TYPE kvcm_manager_batch_add_location_time_us gauge"), std::string::npos);
    EXPECT_NE(output.find("api_name=\"StartWriteCache\"} 42"), std::string::npos);
    EXPECT_EQ(output.find("api_name=\"GetCacheLocation\""), std::string::npos)
        << "Untouched series should not appear:\n"
        << output;
}

// Writing zero is still an explicit observation — the series must
// stay visible.
TEST_F(PrometheusExporterTest, GaugeExplicitlySetToZeroAppears) {
    Gauge g = registry_->GetGauge("data_storage.healthy_status");
    g = 0.0;

    std::string output = PrometheusExporter::Expose(*registry_);
    EXPECT_NE(output.find("kvcm_data_storage_healthy_status 0"), std::string::npos) << "Actual output:\n" << output;
}

TEST_F(PrometheusExporterTest, CounterIncrementedThenResetAppears) {
    Counter c = registry_->GetCounter("service.query_counter");
    ++c;
    c.Reset();

    std::string output = PrometheusExporter::Expose(*registry_);
    EXPECT_NE(output.find("kvcm_service_query_counter 0"), std::string::npos) << "Actual output:\n" << output;
}

// Histogram output tests

TEST_F(PrometheusExporterTest, HistogramBasicOutput) {
    // Register histogram family metadata (as RevisitIntervalHistogram::Init would)
    registry_->RegisterHistogramFamily("revisit_interval_seconds");
    registry_->MapMetricToFamily("revisit_interval_seconds_bucket", "revisit_interval_seconds");
    registry_->MapMetricToFamily("revisit_interval_seconds_sum", "revisit_interval_seconds");
    registry_->MapMetricToFamily("revisit_interval_seconds_count", "revisit_interval_seconds");

    MetricsTags tags_le1 = {{"instance_id", "test"}, {"le", "1"}};
    MetricsTags tags_le5 = {{"instance_id", "test"}, {"le", "5"}};
    MetricsTags tags_leinf = {{"instance_id", "test"}, {"le", "+Inf"}};
    MetricsTags tags_sum = {{"instance_id", "test"}};
    MetricsTags tags_count = {{"instance_id", "test"}};

    Counter c1 = registry_->GetCounter("revisit_interval_seconds_bucket", tags_le1);
    Counter c5 = registry_->GetCounter("revisit_interval_seconds_bucket", tags_le5);
    Counter cinf = registry_->GetCounter("revisit_interval_seconds_bucket", tags_leinf);
    Counter csum = registry_->GetCounter("revisit_interval_seconds_sum", tags_sum);
    Counter ccount = registry_->GetCounter("revisit_interval_seconds_count", tags_count);

    c1 += 2;
    c5 += 5;
    cinf += 5;
    csum += 15000000; // 15 seconds stored in microseconds
    ccount += 5;

    std::string output = PrometheusExporter::Expose(*registry_);

    // Should have histogram TYPE
    EXPECT_NE(output.find("# TYPE kvcm_revisit_interval_seconds histogram"), std::string::npos) << "Actual output:\n"
                                                                                                << output;

    // Should have bucket lines
    EXPECT_NE(output.find("kvcm_revisit_interval_seconds_bucket{instance_id=\"test\",le=\"1\"} 2"), std::string::npos)
        << "Actual output:\n"
        << output;
    EXPECT_NE(output.find("kvcm_revisit_interval_seconds_bucket{instance_id=\"test\",le=\"5\"} 5"), std::string::npos)
        << "Actual output:\n"
        << output;
    EXPECT_NE(output.find("kvcm_revisit_interval_seconds_bucket{instance_id=\"test\",le=\"+Inf\"} 5"),
              std::string::npos)
        << "Actual output:\n"
        << output;

    // Should have sum (converted from microseconds to seconds) and count
    EXPECT_NE(output.find("kvcm_revisit_interval_seconds_sum{instance_id=\"test\"} 15"), std::string::npos)
        << "Actual output:\n"
        << output;
    EXPECT_NE(output.find("kvcm_revisit_interval_seconds_count{instance_id=\"test\"} 5"), std::string::npos)
        << "Actual output:\n"
        << output;
}

TEST_F(PrometheusExporterTest, HistogramTypeAppearsOnlyOnce) {
    registry_->RegisterHistogramFamily("revisit_interval_seconds");
    registry_->MapMetricToFamily("revisit_interval_seconds_bucket", "revisit_interval_seconds");

    MetricsTags tags1 = {{"instance_id", "a"}, {"le", "1"}};
    MetricsTags tags2 = {{"instance_id", "b"}, {"le", "1"}};

    Counter c1 = registry_->GetCounter("revisit_interval_seconds_bucket", tags1);
    Counter c2 = registry_->GetCounter("revisit_interval_seconds_bucket", tags2);
    c1 += 1;
    c2 += 2;

    std::string output = PrometheusExporter::Expose(*registry_);

    // TYPE line should appear only once for histogram family
    size_t first = output.find("# TYPE kvcm_revisit_interval_seconds histogram");
    ASSERT_NE(first, std::string::npos);
    size_t second = output.find("# TYPE kvcm_revisit_interval_seconds histogram", first + 1);
    EXPECT_EQ(second, std::string::npos) << "TYPE histogram line appears more than once";
}

// Verify that an unmapped counter ending with _count is NOT misidentified as histogram
TEST_F(PrometheusExporterTest, UnmappedCountSuffixNotMisidentified) {
    // Register a regular counter with _count suffix — no family mapping
    Counter c = registry_->GetCounter("meta_indexer.total_key_count");
    c += 42;

    std::string output = PrometheusExporter::Expose(*registry_);

    // Should be output as a normal counter, not histogram
    EXPECT_NE(output.find("# TYPE kvcm_meta_indexer_total_key_count counter"), std::string::npos) << "Actual output:\n"
                                                                                                  << output;
    EXPECT_EQ(output.find("histogram"), std::string::npos) << "Should not contain histogram TYPE";
}

// Verify GetMetricFamily returns empty string for unmapped metrics
TEST_F(PrometheusExporterTest, GetMetricFamilyReturnsEmpty) {
    EXPECT_TRUE(registry_->GetMetricFamily("nonexistent_metric").empty());
    EXPECT_TRUE(registry_->GetMetricFamily("").empty());

    registry_->MapMetricToFamily("some_metric", "some_family");
    EXPECT_EQ(registry_->GetMetricFamily("some_metric"), "some_family");
    EXPECT_TRUE(registry_->GetMetricFamily("other_metric").empty());
}

// Verify MapMetricToFamily is idempotent
TEST_F(PrometheusExporterTest, MapMetricToFamilyIdempotent) {
    registry_->MapMetricToFamily("metric_a", "family_x");
    registry_->MapMetricToFamily("metric_a", "family_x"); // duplicate
    registry_->MapMetricToFamily("metric_a", "family_y"); // different family — first wins (emplace semantics)

    EXPECT_EQ(registry_->GetMetricFamily("metric_a"), "family_x");
}
