#include "kv_cache_manager/mrc/mrc_event_consumer.h"

#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <thread>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/event/spec_events/optimizer_event.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

namespace {

std::shared_ptr<CacheGetEvent> MakeCacheGetEvent(const std::string &instance_id, const std::vector<int64_t> &keys) {
    auto event = std::make_shared<CacheGetEvent>(instance_id);
    event->SetEventTriggerTime();
    event->SetAddtionalArgs("prefix_match", keys, /*tokens=*/{}, BlockMask(), /*sw_size=*/0, {});
    return event;
}

OnlineMrcConfig MakeConfig() {
    OnlineMrcConfig config;
    config.enable = true;
    config.max_tracked_blocks = 1024;
    config.window_seconds = 3600;
    config.report_interval_seconds = 3600; // reporting triggered manually in tests
    config.capacity_gb_grid = {1, 2};
    config.max_instances = 4;
    config.queue_max_size = 1024;
    return config;
}

// Wait until the worker thread drains the queue.
void WaitForDrain(OnlineMrcEventConsumer &consumer) {
    for (int i = 0; i < 200 && consumer.BasicQueueSize() > 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    // One extra tick: the last event may still be inside Observe().
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

} // namespace

TEST(OnlineMrcEventConsumerTest, ConsumesCacheGetEventsAndReportsGauges) {
    auto metrics_registry = std::make_shared<MetricsRegistry>();
    auto consumer = std::make_shared<OnlineMrcEventConsumer>(
        MakeConfig(), metrics_registry, [](const std::string &) -> int64_t { return 1024 * 1024; });
    ASSERT_TRUE(consumer->Init(""));

    consumer->Publish(MakeCacheGetEvent("instance_a", {1, 2, 3}));
    consumer->Publish(MakeCacheGetEvent("instance_a", {1, 2, 3}));
    consumer->Publish(MakeCacheGetEvent("instance_b", {7, 8}));
    WaitForDrain(*consumer);

    consumer->ReportMetrics();

    // Cumulative hit rate for instance_a at 1GB (1024 blocks of 1MB):
    // 6 accesses, 3 hits.
    auto data = metrics_registry->GetMetricsData("online_mrc.theoretical_hit_rate");
    ASSERT_NE(nullptr, data);
    MetricsTags tags{{"instance_id", "instance_a"}, {"scope", "cumulative"}, {"capacity_gb", "1"}};
    auto gauge = data->GetGauge(tags);
    ASSERT_TRUE(gauge.has_value());
    EXPECT_NEAR(0.5, gauge->Get(), 1e-9);

    // Quality gauges present.
    EXPECT_NE(nullptr, metrics_registry->GetMetricsData("online_mrc.tracked_blocks"));
    EXPECT_NE(nullptr, metrics_registry->GetMetricsData("online_mrc.tracked_instances"));

    EXPECT_TRUE(consumer->Stop());
}

TEST(OnlineMrcEventConsumerTest, IgnoresNonCacheGetEvents) {
    auto metrics_registry = std::make_shared<MetricsRegistry>();
    auto consumer = std::make_shared<OnlineMrcEventConsumer>(
        MakeConfig(), metrics_registry, [](const std::string &) -> int64_t { return 0; });
    ASSERT_TRUE(consumer->Init(""));

    auto write_event = std::make_shared<FinishWriteCacheEvent>("instance_a");
    EXPECT_TRUE(consumer->Publish(write_event)); // accepted but not enqueued
    EXPECT_EQ(0u, consumer->BasicQueueSize());

    EXPECT_TRUE(consumer->Stop());
}

TEST(OnlineMrcEventConsumerTest, RespectsInstanceLaneLimit) {
    auto config = MakeConfig();
    config.max_instances = 1;
    auto metrics_registry = std::make_shared<MetricsRegistry>();
    auto consumer = std::make_shared<OnlineMrcEventConsumer>(
        config, metrics_registry, [](const std::string &) -> int64_t { return 0; });
    ASSERT_TRUE(consumer->Init(""));

    consumer->Publish(MakeCacheGetEvent("instance_a", {1}));
    consumer->Publish(MakeCacheGetEvent("instance_b", {2}));
    WaitForDrain(*consumer);

    {
        std::lock_guard<std::mutex> guard(consumer->lanes_mutex_);
        EXPECT_EQ(1u, consumer->lanes_.size());
        EXPECT_EQ(1u, consumer->lanes_.count("instance_a"));
    }
    EXPECT_TRUE(consumer->Stop());
}

TEST(OnlineMrcEventConsumerTest, DumpCurvesJsonContainsInstances) {
    auto metrics_registry = std::make_shared<MetricsRegistry>();
    auto consumer = std::make_shared<OnlineMrcEventConsumer>(
        MakeConfig(), metrics_registry, [](const std::string &) -> int64_t { return 4096; });
    ASSERT_TRUE(consumer->Init(""));

    consumer->Publish(MakeCacheGetEvent("instance_a", {1, 2, 3, 1, 2, 3}));
    WaitForDrain(*consumer);
    consumer->ReportMetrics(); // resolves bytes_per_block

    const std::string json = consumer->DumpCurvesJson();
    EXPECT_NE(std::string::npos, json.find("\"instance_a\""));
    EXPECT_NE(std::string::npos, json.find("\"cumulative\""));
    EXPECT_NE(std::string::npos, json.find("\"curve\""));

    EXPECT_TRUE(consumer->Stop());
}

} // namespace kv_cache_manager
