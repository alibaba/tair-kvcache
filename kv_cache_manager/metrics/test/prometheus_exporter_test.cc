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
