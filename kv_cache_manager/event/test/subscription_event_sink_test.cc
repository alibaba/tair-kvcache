#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/event/optimizer_stream/subscription_event_sink.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.pb.h"

using namespace ::testing;

namespace kv_cache_manager {

namespace {

proto::optimizer::TraceQueryRequest MakeRequest(const std::string &trace_id) {
    proto::optimizer::TraceQueryRequest request;
    request.set_trace_id(trace_id);
    request.set_instance_id("instance-a");
    request.add_block_keys(11);
    return request;
}

OptimizerEventPublisherConfig MakeSinkConfig(std::size_t max_subscribers, std::size_t subscriber_queue_size) {
    EventPublishersConfig configs;
    EXPECT_TRUE(configs.FromJsonString(std::string(R"({"optimizer":{"max_subscribers":)") +
                                       std::to_string(max_subscribers) + R"(,"subscriber_queue_size":)" +
                                       std::to_string(subscriber_queue_size) + "}}"));
    return configs.optimizer_event_publisher_config();
}

} // namespace

class SubscriptionEventSinkTest : public TESTBASE {};

TEST_F(SubscriptionEventSinkTest, CaptureEpochTracksSubscribersAndRejectsOldEvents) {
    SubscriptionEventSink sink(MakeSinkConfig(2, 4));
    EXPECT_EQ(0u, sink.CaptureEpoch());
    auto first = sink.Subscribe("first");
    const auto epoch = sink.CaptureEpoch();
    ASSERT_NE(0u, epoch);
    auto second = sink.Subscribe("second");
    sink.Unsubscribe(first);
    EXPECT_EQ(epoch, sink.CaptureEpoch());
    ASSERT_TRUE(sink.Send(MakeRequest("queued"), epoch));
    sink.Unsubscribe(second);
    EXPECT_EQ(0u, sink.CaptureEpoch());
    proto::optimizer::TraceQueryRequest event;
    EXPECT_EQ(SubscriptionEventSink::Subscription::WaitResult::kClosed,
              second->WaitNext(&event, std::chrono::milliseconds(10)));
    auto fresh = sink.Subscribe("fresh");
    ASSERT_NE(epoch, sink.CaptureEpoch());
    EXPECT_FALSE(sink.Send(MakeRequest("old"), epoch));
    EXPECT_EQ(SubscriptionEventSink::Subscription::WaitResult::kTimeout,
              fresh->WaitNext(&event, std::chrono::milliseconds(10)));
    sink.DisableSubscriptions();
    EXPECT_EQ(0u, sink.CaptureEpoch());
}

TEST_F(SubscriptionEventSinkTest, TestBroadcastsToEverySubscriberInOrder) {
    SubscriptionEventSink sink(MakeSinkConfig(2, 4));
    auto first = sink.Subscribe("optimizer-1");
    auto second = sink.Subscribe("optimizer-2");
    ASSERT_NE(nullptr, first);
    ASSERT_NE(nullptr, second);

    ASSERT_TRUE(sink.Send(MakeRequest("one"), sink.CaptureEpoch()));
    ASSERT_TRUE(sink.Send(MakeRequest("two"), sink.CaptureEpoch()));

    for (const auto &subscription : {first, second}) {
        proto::optimizer::TraceQueryRequest event;
        EXPECT_EQ(SubscriptionEventSink::Subscription::WaitResult::kEvent,
                  subscription->WaitNext(&event, std::chrono::milliseconds(10)));
        EXPECT_EQ("one", event.trace_id());
        EXPECT_EQ(SubscriptionEventSink::Subscription::WaitResult::kEvent,
                  subscription->WaitNext(&event, std::chrono::milliseconds(10)));
        EXPECT_EQ("two", event.trace_id());
    }
}

TEST_F(SubscriptionEventSinkTest, TestDropsWithoutSubscriberAndAtQueueLimit) {
    SubscriptionEventSink sink(MakeSinkConfig(1, 1));
    EXPECT_FALSE(sink.Send(MakeRequest("no-subscriber"), sink.CaptureEpoch()));
    EXPECT_EQ(1u, sink.DroppedCount());

    auto subscription = sink.Subscribe("slow");
    ASSERT_NE(nullptr, subscription);
    ASSERT_TRUE(sink.Send(MakeRequest("queued"), sink.CaptureEpoch()));
    EXPECT_FALSE(sink.Send(MakeRequest("dropped"), sink.CaptureEpoch()));
    EXPECT_EQ(2u, sink.DroppedCount());

    proto::optimizer::TraceQueryRequest event;
    ASSERT_EQ(SubscriptionEventSink::Subscription::WaitResult::kEvent,
              subscription->WaitNext(&event, std::chrono::milliseconds(10)));
    EXPECT_EQ("queued", event.trace_id());
}

TEST_F(SubscriptionEventSinkTest, TestRejectsExcessSubscribers) {
    SubscriptionEventSink sink(MakeSinkConfig(1, 1));
    ASSERT_NE(nullptr, sink.Subscribe("first"));
    EXPECT_EQ(nullptr, sink.Subscribe("second"));
}

TEST_F(SubscriptionEventSinkTest, TestDisableClosesSubscribersAndEnableAcceptsNewOnes) {
    SubscriptionEventSink sink(MakeSinkConfig(1, 1));
    auto subscription = sink.Subscribe("optimizer");
    ASSERT_NE(nullptr, subscription);

    sink.DisableSubscriptions();
    proto::optimizer::TraceQueryRequest event;
    EXPECT_EQ(SubscriptionEventSink::Subscription::WaitResult::kClosed,
              subscription->WaitNext(&event, std::chrono::milliseconds(10)));
    EXPECT_EQ(nullptr, sink.Subscribe("disabled"));
    EXPECT_FALSE(sink.Send(MakeRequest("disabled"), sink.CaptureEpoch()));

    sink.EnableSubscriptions();
    auto resumed = sink.Subscribe("resumed");
    ASSERT_NE(nullptr, resumed);
    EXPECT_TRUE(sink.Send(MakeRequest("resumed"), sink.CaptureEpoch()));
    EXPECT_EQ(SubscriptionEventSink::Subscription::WaitResult::kEvent,
              resumed->WaitNext(&event, std::chrono::milliseconds(10)));
    EXPECT_EQ("resumed", event.trace_id());
}

TEST_F(SubscriptionEventSinkTest, TestStopWakesSubscriberAndIsIdempotent) {
    SubscriptionEventSink sink(MakeSinkConfig(1, 1));
    auto subscription = sink.Subscribe("optimizer");
    ASSERT_NE(nullptr, subscription);

    sink.Stop();
    sink.Stop();
    proto::optimizer::TraceQueryRequest event;
    EXPECT_EQ(SubscriptionEventSink::Subscription::WaitResult::kClosed,
              subscription->WaitNext(&event, std::chrono::milliseconds(10)));
    EXPECT_EQ(nullptr, sink.Subscribe("late"));
    EXPECT_FALSE(sink.Send(MakeRequest("late"), sink.CaptureEpoch()));
}

} // namespace kv_cache_manager
