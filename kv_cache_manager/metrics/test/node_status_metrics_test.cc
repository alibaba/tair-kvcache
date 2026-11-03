#include <memory>
#include <string>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/node_status_metrics.h"
#include "kv_cache_manager/metrics/prometheus_exporter.h"

using namespace kv_cache_manager;

class NodeStatusMetricsTest : public TESTBASE {};

TEST_F(NodeStatusMetricsTest, RoleTransitionsPublishOnlyLeaderGauge) {
    auto registry = std::make_shared<MetricsRegistry>();
    for (bool leader : {false, true, false, true}) {
        NodeStatusMetrics::Publish(registry, leader);
        EXPECT_DOUBLE_EQ(leader ? 1 : 0, registry->GetGauge(NodeStatusMetrics::kIsLeaderName).Get());
        const auto output = PrometheusExporter::Expose(*registry);
        EXPECT_NE(std::string::npos, output.find("# TYPE kvcm_is_leader gauge"));
        EXPECT_NE(std::string::npos, output.find(leader ? "kvcm_is_leader 1\n" : "kvcm_is_leader 0\n"));
        EXPECT_EQ(std::string::npos, output.find("kvcm_access_mode"));
        EXPECT_EQ(nullptr, registry->GetMetricsData("access_mode"));
    }
}

TEST_F(NodeStatusMetricsTest, NullRegistryIsSafe) {
    EXPECT_NO_FATAL_FAILURE(NodeStatusMetrics::Publish(nullptr, true));
}
