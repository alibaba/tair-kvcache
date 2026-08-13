#include "kv_cache_manager/mrc/online_mrc_registry.h"

#include <chrono>
#include <gtest/gtest.h>
#include <thread>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/mrc/online_mrc_trace_receiver.h"

namespace kv_cache_manager {
namespace {

OnlineMrcConfig MakeConfig() {
    OnlineMrcConfig config;
    config.enable = true;
    config.max_tracked_blocks = 64;
    config.window_seconds = 0;
    config.capacity_gb_grid = {1};
    config.max_instances = 4;
    config.receiver_queue_max_batches = 4;
    config.idle_expire_seconds = 0;
    return config;
}

OnlineMrcSpan MakeSpan(uint64_t sequence, std::vector<int64_t> keys) {
    OnlineMrcSpan span;
    span.cluster = "cluster_a";
    span.instance_group = "group_a";
    span.instance_id = "instance_a";
    span.source_id = "source_a";
    span.bytes_per_block = 32LL * 1024 * 1024;
    span.event_time_us = sequence;
    span.sequence_number = sequence;
    span.keys = std::move(keys);
    return span;
}

} // namespace

TEST(OnlineMrcRegistryTest, ReportsExactCurveAndSequenceQuality) {
    auto metrics = std::make_shared<MetricsRegistry>();
    OnlineMrcRegistry registry(MakeConfig(), metrics);
    ASSERT_TRUE(registry.Observe(MakeSpan(1, {1, 2, 3})));
    ASSERT_TRUE(registry.Observe(MakeSpan(3, {1, 2, 3})));
    registry.ReportMetrics();

    auto hit_data = metrics->GetMetricsData("online_mrc.theoretical_hit_rate");
    ASSERT_NE(nullptr, hit_data);
    MetricsTags hit_tags{{"cluster", "cluster_a"},
                         {"instance_group", "group_a"},
                         {"instance_id", "instance_a"},
                         {"scope", "cumulative"},
                         {"capacity_gb", "1"}};
    auto hit = hit_data->GetGauge(hit_tags);
    ASSERT_TRUE(hit.has_value());
    EXPECT_DOUBLE_EQ(0.5, hit->Get());

    auto quality_data = metrics->GetMetricsData("online_mrc.stream_complete");
    ASSERT_NE(nullptr, quality_data);
    MetricsTags quality_tags{{"cluster", "cluster_a"},
                             {"instance_group", "group_a"},
                             {"instance_id", "instance_a"}};
    auto quality = quality_data->GetGauge(quality_tags);
    ASSERT_TRUE(quality.has_value());
    EXPECT_DOUBLE_EQ(0.0, quality->Get());
}

TEST(OnlineMrcRegistryTest, ReceiverMovesBatchesOffRpcThread) {
    auto registry = std::make_shared<OnlineMrcRegistry>(MakeConfig(), nullptr);
    OnlineMrcTraceReceiver receiver(registry, 4);
    ASSERT_TRUE(receiver.Start());

    OnlineMrcBatch batch;
    batch.spans.push_back(MakeSpan(1, {1, 2, 3}));
    ASSERT_TRUE(receiver.Enqueue(std::move(batch)));
    for (int i = 0; i < 100 && registry->LaneCount() == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_EQ(1u, registry->LaneCount());
    receiver.Stop();
}

TEST(OnlineMrcRegistryTest, DebugDumpDeclaresBoundedExactEngine) {
    OnlineMrcRegistry registry(MakeConfig(), nullptr);
    ASSERT_TRUE(registry.Observe(MakeSpan(1, {1, 2, 3})));
    ASSERT_TRUE(registry.Observe(MakeSpan(2, {1, 2, 3})));
    const std::string json = registry.DumpCurvesJson();
    EXPECT_NE(std::string::npos, json.find("lite_hit_exact_bounded"));
    EXPECT_NE(std::string::npos, json.find("instance_a"));
    EXPECT_NE(std::string::npos, json.find("tracked_capacity_blocks"));
}

TEST(OnlineMrcRegistryTest, KnownGlobalDropConservativelyMarksStreamIncomplete) {
    auto metrics = std::make_shared<MetricsRegistry>();
    OnlineMrcRegistry registry(MakeConfig(), metrics);
    ASSERT_TRUE(registry.Observe(MakeSpan(1, {1, 2, 3})));
    registry.RecordDropped(1, 3);
    registry.ReportMetrics();
    registry.ReportIngressMetrics(2, 1);

    MetricsTags lane_tags{{"cluster", "cluster_a"},
                          {"instance_group", "group_a"},
                          {"instance_id", "instance_a"}};
    auto quality_data = metrics->GetMetricsData("online_mrc.stream_complete");
    ASSERT_NE(nullptr, quality_data);
    auto quality = quality_data->GetGauge(lane_tags);
    ASSERT_TRUE(quality.has_value());
    EXPECT_DOUBLE_EQ(0.0, quality->Get());

    auto ingress_data = metrics->GetMetricsData("online_mrc.receiver_queue_size");
    ASSERT_NE(nullptr, ingress_data);
    const MetricsTags empty_tags;
    auto queue_size = ingress_data->GetGauge(empty_tags);
    ASSERT_TRUE(queue_size.has_value());
    EXPECT_DOUBLE_EQ(2.0, queue_size->Get());
}

} // namespace kv_cache_manager
