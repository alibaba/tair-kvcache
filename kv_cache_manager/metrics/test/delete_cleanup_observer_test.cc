#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/delete_cleanup_observer.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/prometheus_exporter.h"

using namespace kv_cache_manager;

class DeleteCleanupObserverTest : public TESTBASE {};

TEST_F(DeleteCleanupObserverTest, RecordsOnlyBoundedStageSeriesAndAccumulatesLocationCounts) {
    auto registry = std::make_shared<MetricsRegistry>();
    InitializeDeleteCleanupMetrics(registry);

    RecordDeleteCleanupPermanentFailure(registry, DeleteCleanupFailureStage::kPhysicalDelete, 2);
    RecordDeleteCleanupPermanentFailure(registry, DeleteCleanupFailureStage::kPhysicalDelete, 3);
    RecordDeleteCleanupPermanentFailure(registry, DeleteCleanupFailureStage::kMetadataCad, 4);
    RecordDeleteCleanupPermanentFailure(registry, DeleteCleanupFailureStage::kAuthoritativeFence, 5);

    std::vector<MetricsRegistry::metrics_tuple_t> metrics;
    registry->GetAllMetrics(metrics);
    ASSERT_EQ(4, metrics.size());

    std::map<std::string, std::uint64_t> values_by_stage;
    for (const auto &[name, tags, value] : metrics) {
        EXPECT_EQ(kDeleteCleanupPermanentFailureMetricName, name);
        ASSERT_EQ(1, tags.size());
        ASSERT_EQ(1, tags.count("stage"));
        ASSERT_TRUE(std::holds_alternative<CounterValue>(value->value));
        values_by_stage[tags.at("stage")] = std::get<CounterValue>(value->value).load();
    }

    EXPECT_EQ(5u, values_by_stage["physical_delete"]);
    EXPECT_EQ(4u, values_by_stage["metadata_cad"]);
    EXPECT_EQ(5u, values_by_stage["authoritative_fence"]);
    EXPECT_EQ(0u, values_by_stage["dispatch_or_worker"]);
}

TEST_F(DeleteCleanupObserverTest, InitializesPrometheusZeroBaselineForEveryFixedStage) {
    auto registry = std::make_shared<MetricsRegistry>();

    InitializeDeleteCleanupMetrics(registry);

    const std::string output = PrometheusExporter::Expose(*registry);
    EXPECT_NE(output.find("# TYPE kvcm_cache_cleanup_permanent_failure_location_count counter"), std::string::npos)
        << output;
    for (const std::string stage : {"physical_delete", "metadata_cad", "authoritative_fence", "dispatch_or_worker"}) {
        EXPECT_NE(output.find("{stage=\"" + stage + "\"} 0"), std::string::npos) << output;
    }
}

TEST_F(DeleteCleanupObserverTest, IgnoresZeroAndToleratesMissingRegistry) {
    auto registry = std::make_shared<MetricsRegistry>();
    RecordDeleteCleanupPermanentFailure(registry, DeleteCleanupFailureStage::kPhysicalDelete, 0);
    EXPECT_EQ(0u, registry->GetSize());

    EXPECT_NO_THROW(
        RecordDeleteCleanupPermanentFailure(nullptr, DeleteCleanupFailureStage::kMetadataCad, 1));
    EXPECT_NO_THROW(InitializeDeleteCleanupMetrics(nullptr));
}
