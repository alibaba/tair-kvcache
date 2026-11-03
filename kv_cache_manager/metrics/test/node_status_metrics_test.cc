#include <array>
#include <memory>
#include <string>
#include <utility>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/leader_elector.h"
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
    EXPECT_NO_FATAL_FAILURE(NodeStatusMetrics::PublishRoleState(nullptr, RoleState::LEADER));
}

TEST_F(NodeStatusMetricsTest, AllPhasesRemainIndependentOfLeaseOwnership) {
    auto registry = std::make_shared<MetricsRegistry>();
    const std::array<std::pair<RoleState, const char *>, 4> phases{{{RoleState::FOLLOWER, "follower"},
                                                                    {RoleState::PROMOTING, "promoting"},
                                                                    {RoleState::LEADER, "leader"},
                                                                    {RoleState::DEMOTING, "demoting"}}};
    for (bool lease_held : {false, true}) {
        for (const auto &[current, label] : phases) {
            SCOPED_TRACE(::testing::Message() << "lease=" << lease_held << " phase=" << label);
            NodeStatusMetrics::Publish(registry, lease_held);
            NodeStatusMetrics::PublishRoleState(registry, current);
            EXPECT_DOUBLE_EQ(lease_held ? 1 : 0, registry->GetGauge(NodeStatusMetrics::kIsLeaderName).Get());
            const auto output = PrometheusExporter::Expose(*registry);
            EXPECT_NE(std::string::npos, output.find("# TYPE kvcm_role_state gauge"));
            for (const auto &[phase, name] : phases) {
                EXPECT_DOUBLE_EQ(current == phase ? 1 : 0,
                                 registry->GetGauge(NodeStatusMetrics::kRoleStateName, {{"state", name}}).Get());
                const std::string line =
                    std::string("kvcm_role_state{state=\"") + name + "\"} " + (current == phase ? "1\n" : "0\n");
                EXPECT_NE(std::string::npos, output.find(line));
            }
            EXPECT_EQ(std::string::npos, output.find("kvcm_access_mode"));
        }
    }
}

TEST_F(NodeStatusMetricsTest, TransitionsClearPreviousPhasesWithoutGrowingSeries) {
    auto registry = std::make_shared<MetricsRegistry>();
    // Includes successful promotion, demotion and a promotion aborted by lease loss.
    for (const auto state : {RoleState::FOLLOWER,
                             RoleState::PROMOTING,
                             RoleState::LEADER,
                             RoleState::DEMOTING,
                             RoleState::FOLLOWER,
                             RoleState::PROMOTING,
                             RoleState::DEMOTING,
                             RoleState::FOLLOWER,
                             RoleState::PROMOTING,
                             RoleState::LEADER,
                             RoleState::LEADER}) {
        NodeStatusMetrics::PublishRoleState(registry, state);
        const auto data = registry->GetMetricsData(NodeStatusMetrics::kRoleStateName);
        ASSERT_NE(nullptr, data);
        const auto values = data->GetMetricsValues();
        ASSERT_EQ(4, values.size());
        double sum = 0;
        for (const auto &[tags, value] : values) {
            sum += Gauge(value).Get();
        }
        EXPECT_DOUBLE_EQ(1, sum);
    }
}
