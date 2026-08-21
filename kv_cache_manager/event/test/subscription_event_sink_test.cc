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

TEST_F(SubscriptionEventSinkTest, TestBroadcastsToEverySubscriberInOrder) {
    SubscriptionEventSink sink(MakeSinkConfig(2, 4));
    auto first = sink.Subscribe("optimizer-1");
    auto second = sink.Subscribe("optimizer-2");
    ASSERT_NE(nullptr, first);
    ASSERT_NE(nullptr, second);

    ASSERT_TRUE(sink.Send(MakeRequest("one")));
    ASSERT_TRUE(sink.Send(MakeRequest("two")));

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
    EXPECT_FALSE(sink.Send(MakeRequest("no-subscriber")));
    EXPECT_EQ(1u, sink.DroppedCount());

    auto subscription = sink.Subscribe("slow");
    ASSERT_NE(nullptr, subscription);
    ASSERT_TRUE(sink.Send(MakeRequest("queued")));
    EXPECT_FALSE(sink.Send(MakeRequest("dropped")));
    EXPECT_EQ(2u, sink.DroppedCount());

    proto::optimizer::TraceQueryRequest event;
    ASSERT_EQ(SubscriptionEventSink::Subscription::WaitResult::kEvent,
              subscription->WaitNext(&event, std::chrono::milliseconds(10)));
    EXPECT_EQ("queued", event.trace_id());
}

TEST_F(SubscriptionEventSinkTest, TestBatchAdmissionIsAtomicPerSubscriber) {
    SubscriptionEventSink sink(MakeSinkConfig(1, 2));
    auto subscription = sink.Subscribe("optimizer");
    ASSERT_NE(nullptr, subscription);

    ASSERT_TRUE(sink.Send(MakeRequest("queued")));
    EXPECT_FALSE(sink.SendBatch({MakeRequest("batch-1"), MakeRequest("batch-2")}));
    EXPECT_EQ(2u, sink.DroppedCount());
    EXPECT_EQ(1u, sink.QueuedCount());

    proto::optimizer::TraceQueryRequest event;
    ASSERT_EQ(SubscriptionEventSink::Subscription::WaitResult::kEvent,
              subscription->WaitNext(&event, std::chrono::milliseconds(10)));
    EXPECT_EQ("queued", event.trace_id());
    EXPECT_EQ(SubscriptionEventSink::Subscription::WaitResult::kTimeout,
              subscription->WaitNext(&event, std::chrono::milliseconds(10)));
}

TEST_F(SubscriptionEventSinkTest, TestRejectsExcessSubscribers) {
    SubscriptionEventSink sink(MakeSinkConfig(1, 1));
    ASSERT_NE(nullptr, sink.Subscribe("first"));
    EXPECT_EQ(nullptr, sink.Subscribe("second"));
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
    EXPECT_FALSE(sink.Send(MakeRequest("late")));
}

} // namespace kv_cache_manager
