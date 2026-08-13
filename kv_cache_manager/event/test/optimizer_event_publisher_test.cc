#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/event/optimizer_event_publisher.h"
#include "kv_cache_manager/event/spec_events/optimizer_event.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.pb.h"

using namespace ::testing;

namespace kv_cache_manager {

namespace {

// Captures what the publisher hands to the subscription queue.
class RecordingEventSink : public EventSink {
public:
    bool Send(const proto::optimizer::TraceQueryRequest &event) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!accept_) {
            ++dropped_;
            return false;
        }
        events_.push_back(event);
        return true;
    }

    void Stop() override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stop_calls_;
    }

    std::size_t DroppedCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }

    void set_accept(bool accept) {
        std::lock_guard<std::mutex> lock(mutex_);
        accept_ = accept;
    }

    std::size_t EventCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_.size();
    }

    std::size_t StopCalls() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stop_calls_;
    }

    std::vector<proto::optimizer::TraceQueryRequest> Events() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<proto::optimizer::TraceQueryRequest> events_;
    std::size_t dropped_ = 0;
    std::size_t stop_calls_ = 0;
    bool accept_ = true;
};

std::shared_ptr<CacheGetEvent> MakeGetEvent(const std::string &instance_id,
                                            const std::vector<std::int64_t> &keys,
                                            const std::vector<std::int64_t> &tokens,
                                            const std::string &query_type = "prefix_match",
                                            const std::string &trace_id = "trace-1") {
    auto event = std::make_shared<CacheGetEvent>(instance_id);
    event->SetEventTriggerTime();
    event->set_trace_id(trace_id);
    event->SetAddtionalArgs(query_type, keys, tokens, BlockMask(), 0, {});
    return event;
}

bool WaitForEvents(const RecordingEventSink &sink, std::size_t expected, int timeout_ms = 2000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (sink.EventCount() >= expected) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return sink.EventCount() >= expected;
}

OptimizerEventPublisherConfig MakePublisherConfig(std::size_t queue_size) {
    EventPublishersConfig configs;
    EXPECT_TRUE(
        configs.FromJsonString(std::string(R"({"optimizer":{"queue_size":)") + std::to_string(queue_size) + "}}"));
    return configs.optimizer_event_publisher_config();
}

} // namespace

class OptimizerEventPublisherTest : public TESTBASE {
protected:
    void SetUp() override {
        sink_ = std::make_shared<RecordingEventSink>();
        publisher_ = std::make_unique<OptimizerEventPublisher>(sink_, MakePublisherConfig(64));
    }

    void TearDown() override {
        if (publisher_) {
            publisher_->Stop();
            publisher_.reset();
        }
        sink_.reset();
    }

    std::shared_ptr<RecordingEventSink> sink_;
    std::unique_ptr<OptimizerEventPublisher> publisher_;
};

TEST_F(OptimizerEventPublisherTest, TestInitRequiresSink) {
    OptimizerEventPublisher no_sink(nullptr, OptimizerEventPublisherConfig{});
    EXPECT_FALSE(no_sink.Init(""));
}

TEST_F(OptimizerEventPublisherTest, TestForwardsCacheGetEvent) {
    ASSERT_TRUE(publisher_->Init(""));
    ASSERT_TRUE(publisher_->Publish(MakeGetEvent("instance-a", {11, 22, 33}, std::vector<std::int64_t>(700, 5))));
    ASSERT_TRUE(WaitForEvents(*sink_, 1));

    const auto events = sink_->Events();
    ASSERT_EQ(1u, events.size());
    const auto &request = events[0];
    EXPECT_EQ("instance-a", request.instance_id());
    ASSERT_EQ(3, request.block_keys_size());
    EXPECT_EQ(11, request.block_keys(0));
    EXPECT_EQ(22, request.block_keys(1));
    EXPECT_EQ(33, request.block_keys(2));
    EXPECT_EQ(700, request.input_token_len());
    EXPECT_GT(request.timestamp_ns(), 0);
    // token_ids stay out of the wire format: only their count matters.
    EXPECT_EQ(0, request.token_ids_size());
    EXPECT_EQ(1u, publisher_->ForwardedCount());
}

// Microseconds on the event, nanoseconds on the wire.
TEST_F(OptimizerEventPublisherTest, TestTimestampConvertedToNanos) {
    ASSERT_TRUE(publisher_->Init(""));
    auto event = MakeGetEvent("instance-a", {1}, {1, 2});
    const std::int64_t trigger_us = event->event_trigger_time_us();
    ASSERT_TRUE(publisher_->Publish(event));
    ASSERT_TRUE(WaitForEvents(*sink_, 1));

    const auto request = sink_->Events()[0];
    EXPECT_EQ(trigger_us * 1000, request.timestamp_ns());
}

// Every publisher on EventManager sees every event, so non-read events must be
// skipped rather than forwarded as garbage.
TEST_F(OptimizerEventPublisherTest, TestSkipsNonCacheGetEvents) {
    ASSERT_TRUE(publisher_->Init(""));

    auto write_event = std::make_shared<StartWriteCacheEvent>("instance-a");
    write_event->SetEventTriggerTime();
    write_event->SetAddtionalArgs("session-1", {1, 2}, {}, BlockMask(), {}, 0);
    ASSERT_TRUE(publisher_->Publish(write_event));

    ASSERT_TRUE(publisher_->Publish(MakeGetEvent("instance-a", {7}, {1, 2})));
    ASSERT_TRUE(WaitForEvents(*sink_, 1));

    EXPECT_EQ(1u, sink_->EventCount());
    EXPECT_EQ(1u, publisher_->ForwardedCount());
    EXPECT_EQ(1u, publisher_->SkippedCount());
}

TEST_F(OptimizerEventPublisherTest, TestPublishBeforeInitIsRejected) {
    EXPECT_FALSE(publisher_->Publish(MakeGetEvent("instance-a", {1}, {1})));
}

TEST_F(OptimizerEventPublisherTest, TestPublishAfterStopIsRejected) {
    ASSERT_TRUE(publisher_->Init(""));
    ASSERT_TRUE(publisher_->Stop());
    EXPECT_FALSE(publisher_->Publish(MakeGetEvent("instance-a", {1}, {1})));
}

// A sink that refuses everything must not stall or crash the worker.
TEST_F(OptimizerEventPublisherTest, TestSinkRefusalIsNotCountedAsForwarded) {
    ASSERT_TRUE(publisher_->Init(""));
    sink_->set_accept(false);

    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(publisher_->Publish(MakeGetEvent("instance-a", {i}, {1, 2})));
    }
    // Nothing to wait for on the sink, so wait for the queue to be consumed.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (sink_->DroppedCount() < 5 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(5u, sink_->DroppedCount());
    EXPECT_EQ(0u, publisher_->ForwardedCount());
}

// The queue is the boundary that protects serving threads: it must refuse, not
// grow and not block.
TEST_F(OptimizerEventPublisherTest, TestFullQueueDropsInsteadOfBlocking) {
    sink_->set_accept(false);
    const auto config = MakePublisherConfig(4);
    auto publisher = std::make_unique<OptimizerEventPublisher>(sink_, config);
    // No Init(): no worker drains the queue, so it fills deterministically.
    publisher->InitBasicQueue(config.queue_size());
    publisher->running_ = true;

    int accepted = 0;
    for (int i = 0; i < 50; ++i) {
        if (publisher->Publish(MakeGetEvent("instance-a", {i}, {1}))) {
            ++accepted;
        }
    }
    EXPECT_EQ(4, accepted);
    EXPECT_EQ(46u, publisher->DroppedCount());
    publisher->running_ = false;
}

TEST_F(OptimizerEventPublisherTest, TestStopIsIdempotentAndStopsSink) {
    ASSERT_TRUE(publisher_->Init(""));
    EXPECT_TRUE(publisher_->Stop());
    EXPECT_TRUE(publisher_->Stop());
    EXPECT_EQ(1u, sink_->StopCalls());
}

TEST_F(OptimizerEventPublisherTest, TestStopWithoutInitIsSafe) { EXPECT_TRUE(publisher_->Stop()); }

// Trace id is what lets a consumer-side anomaly be joined back to the request
// in kvcm's own logs; the random event id cannot do that.
TEST_F(OptimizerEventPublisherTest, TestTraceIdIsCarriedThrough) {
    ASSERT_TRUE(publisher_->Init(""));
    ASSERT_TRUE(publisher_->Publish(MakeGetEvent("instance-a", {1, 2}, {1, 2, 3}, "prefix_match", "trace-abc")));
    ASSERT_TRUE(WaitForEvents(*sink_, 1));

    const auto request = sink_->Events()[0];
    EXPECT_EQ("trace-abc", request.trace_id());
}

TEST_F(OptimizerEventPublisherTest, TestDoesNotFilterQueryType) {
    ASSERT_TRUE(publisher_->Init(""));
    ASSERT_TRUE(publisher_->Publish(MakeGetEvent("instance-a", {1, 2}, {1, 2}, "reverse_roll_sw_match")));
    ASSERT_TRUE(publisher_->Publish(MakeGetEvent("instance-a", {3, 4}, {1, 2}, "prefix_match")));
    ASSERT_TRUE(publisher_->Publish(MakeGetEvent("instance-a", {5, 6}, {1, 2}, "prefix_match_with_mamba")));
    ASSERT_TRUE(WaitForEvents(*sink_, 3));

    EXPECT_EQ(3u, sink_->EventCount());
    EXPECT_EQ(3u, publisher_->ForwardedCount());
    EXPECT_EQ(0u, publisher_->SkippedCount());

    const auto events = sink_->Events();
    EXPECT_EQ(1, events[0].block_keys(0));
    EXPECT_EQ(3, events[1].block_keys(0));
    EXPECT_EQ(5, events[2].block_keys(0));
}

// A prompt shorter than one block has no complete block key. It is a real
// request whose tokens belong in the hit-rate denominator, so dropping it
// would silently inflate the reported hit rate.
TEST_F(OptimizerEventPublisherTest, TestForwardsEventWithoutBlockKeys) {
    ASSERT_TRUE(publisher_->Init(""));
    ASSERT_TRUE(publisher_->Publish(MakeGetEvent("instance-a", {}, std::vector<std::int64_t>(100, 7))));
    ASSERT_TRUE(WaitForEvents(*sink_, 1));

    const auto request = sink_->Events()[0];
    EXPECT_EQ(0, request.block_keys_size());
    EXPECT_EQ(100, request.input_token_len());
    EXPECT_EQ(1u, publisher_->ForwardedCount());
}

// No tokens means the caller supplied pre-computed keys, so the exact input
// length is unknown here. 0 signals that; the consumer infers a length and
// accepts the upward bias rather than kvcm guessing.
TEST_F(OptimizerEventPublisherTest, TestUnknownInputLenIsZero) {
    ASSERT_TRUE(publisher_->Init(""));
    ASSERT_TRUE(publisher_->Publish(MakeGetEvent("instance-a", {1, 2, 3}, {})));
    ASSERT_TRUE(WaitForEvents(*sink_, 1));

    const auto request = sink_->Events()[0];
    EXPECT_EQ(3, request.block_keys_size());
    EXPECT_EQ(0, request.input_token_len());
}

} // namespace kv_cache_manager
